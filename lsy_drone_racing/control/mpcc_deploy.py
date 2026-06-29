"""Model Predictive Contouring Control (MPCC) for drone racing.

Unlike the reference-tracking MPC (:mod:`mpc_planner_controller`), this controller tracks
the **geometric path** produced by the planner and optimises **progress** along it. The
acados model is augmented with a progress state ``theta`` (arc length) and its speed
``v_theta``; the cost penalises the *contouring* error (perpendicular distance to the path)
and the *lag* error (longitudinal), and rewards advancing ``theta`` at a target speed.

Because there is no time-parameterised reference, the reference can never "run away" from
the drone — so the reference governor / nearest-tick machinery of the tracking MPC is not
needed here. The path is **embedded in the model as a function of the progress state**
``theta`` (formulation A, as in MPCC++ eq. 5): for each shooting node we pass the local
cubic coefficients of the path around that node's predicted ``theta``, and acados evaluates
the path point ``p_d(theta)`` and tangent symbolically from the state — so moving ``theta``
moves the reference and the contouring/lag errors genuinely depend on progress. The cubic
comes from an internal arc-length spline built from the warm-started :class:`SimplePlanner`.
"""

from __future__ import annotations

import glob
import hashlib
import inspect
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import casadi as ca
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from crazyflow.sim.visualize import draw_line, draw_points
from drone_models.core import load_params
from drone_models.so_rpy import symbolic_dynamics_euler
from drone_models.utils.rotation import ang_vel2rpy_rates
from scipy.interpolate import CubicSpline, splev, splprep
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control import mpcc_weight_policy as wp
from lsy_drone_racing.control.mpc_planner_controller import _Cylinder, _GateFrame
from lsy_drone_racing.control.mpcc_weight_policy import WeightPolicy

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ── MPCC cost weights (the controller's single source of truth) ─────────────────────────────
#: Baseline diagonal cost weights of the MPCC NONLINEAR_LS cost. These ARE the controller's
#: weights: the solver is built from them, and with the RL weight planner OFF
#: (``MPCCController.USE_RL_WEIGHTS = False``) they are used verbatim — exactly the no-RL
#: behaviour. With the RL ON, the policy only *scales* these (mpcc_weight_policy.weight_diagonals),
#: so editing them here is the one place that changes the controller's weighting either way.
BASELINE_WEIGHTS = {
    "q_c": 50.0,  # contouring (perpendicular path error)
    "q_l": 150.0,  # lag (longitudinal path error) — keeps theta glued to the drone
    "q_att": 1.0,  # attitude (rpy) state tracking
    "q_dr": 5.0,  # body-rate (drpy) state tracking
    "q_v": 5.0,  # progress-speed (cruise) reward
    "r_rpy": 1.0,  # rpy-command effort
    "r_T": 50.0,  # collective-thrust effort
    "r_at": 0.5,  # progress-acceleration effort
}


# ── Augmented model: physical drone + progress double-integrator ────────────────


def create_mpcc_model(parameters: dict) -> AcadosModel:
    """Build the progress-augmented drone model.

    The drone model (12 states / 4 inputs) is augmented with progress states ``[theta,
    v_theta]`` and a progress-acceleration input ``a_theta``: ``theta_dot = v_theta``,
    ``v_theta_dot = a_theta``.
    """
    x_dot, x, u, _ = symbolic_dynamics_euler(
        mass=parameters["mass"],
        gravity_vec=parameters["gravity_vec"],
        J=parameters["J"],
        J_inv=parameters["J_inv"],
        acc_coef=parameters["acc_coef"],
        cmd_f_coef=parameters["cmd_f_coef"],
        rpy_coef=parameters["rpy_coef"],
        rpy_rates_coef=parameters["rpy_rates_coef"],
        cmd_rpy_coef=parameters["cmd_rpy_coef"],
    )
    theta = ca.MX.sym("theta")
    v_theta = ca.MX.sym("v_theta")
    a_theta = ca.MX.sym("a_theta")

    model = AcadosModel()
    model.name = "mpcc_planner"  # distinct generated-code name (no clash with the MPC)
    model.x = ca.vertcat(x, theta, v_theta)
    model.u = ca.vertcat(u, a_theta)
    model.f_expl_expr = ca.vertcat(x_dot, v_theta, a_theta)
    return model


def _capsule_barrier(p_caps: ca.MX, pos: ca.MX, n_caps: int) -> ca.MX:
    """Smooth soft-barrier residual for each avoidance capsule.

    ``p_caps`` packs ``n_caps`` capsules as ``[p1(3), p2(3), r(1)]`` each. For every capsule
    the residual is ``max(0, 1 - d^2/r^2)^2`` with ``d`` the distance from ``pos`` to the
    closest point on the capsule segment: 0 outside the radius, rising smoothly to 1 on the
    axis. C1-continuous, so it suits the Gauss-Newton SQP. Returns an empty vector when
    ``n_caps == 0``.
    """
    out = []
    for i in range(n_caps):
        p1 = p_caps[i * 7 + 0 : i * 7 + 3]
        p2 = p_caps[i * 7 + 3 : i * 7 + 6]
        r = p_caps[i * 7 + 6]
        v = p2 - p1
        w = pos - p1
        vv = ca.dot(v, v)
        vv = ca.if_else(vv > 1e-6, vv, 1e-6)
        s = ca.fmax(0.0, ca.fmin(1.0, ca.dot(w, v) / vv))  # closest-point param on the segment
        diff = pos - (p1 + s * v)
        d2 = ca.dot(diff, diff)
        out.append(ca.fmax(0.0, 1.0 - d2 / (r**2 + 1e-6)) ** 2)
    return ca.vertcat(*out)


def _solver_signature(payload: dict) -> str:
    """Hash everything that determines the generated acados C code.

    Two parts: (1) the structural/numeric arguments that get baked into the OCP (horizon,
    capsule count, the bounds/penalties, the time grid, and the drone params that enter the
    dynamics symbolically); (2) the *source* of the model/cost/solver builders, so any edit to
    them invalidates the cache automatically. Runtime quantities (gate/obstacle poses, path
    cubics, x0) flow in as solver parameters and are deliberately excluded — they never change
    the C code.
    """

    def _jsonable(v: object) -> object:
        if isinstance(v, np.ndarray):
            return np.round(v, 9).tolist()
        if isinstance(v, (np.floating, np.integer)):
            return float(v)
        if isinstance(v, dict):
            return {k: _jsonable(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_jsonable(x) for x in v]
        return v

    src = "".join(
        inspect.getsource(fn)
        for fn in (create_mpcc_model, _capsule_barrier, create_mpcc_ocp_solver)
    )
    blob = json.dumps(
        {"args": _jsonable(payload), "src": hashlib.sha256(src.encode()).hexdigest()},
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def create_mpcc_ocp_solver(
    Tf: float,
    N: int,
    parameters: dict,
    z_min: float = 0.0,
    z_max: float = 2.5,
    v_max: float = 2.0,
    vtheta_max: float = 2.5,
    atheta_max: float = 6.0,
    n_caps: int = 0,
    capsule_penalty: float = 5000.0,
    ground_soft_z: float = 0.0,
    ground_penalty: float = 0.0,
    time_steps: np.ndarray | None = None,
    verbose: bool = False,
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Build the acados OCP/solver for MPCC.

    NONLINEAR_LS cost on contouring/lag/progress; the per-stage path cubic + target speed are
    model parameters set per solve. When ``n_caps`` > 0, obstacle/gate avoidance is added as a
    smooth soft-barrier in the COST (not a hard constraint): each keep-out is a capsule
    (segment ``p1->p2`` with radius ``r``) and the barrier ``max(0, 1 - d^2/r^2)^2`` (``d`` =
    distance from the drone to the capsule axis) is penalised with weight ``capsule_penalty``.
    A cost barrier never makes the QP infeasible and is cheap (no extra inequality rows /
    slacks), and the capsule shape lets the drone fly through a gate opening while avoiding the
    frame bars — so no per-gate enable/disable is needed.

    Parameter layout per stage: ``p = [theta_i(1), c_x(4), c_y(4), c_z(4), v_target(1),
    <p1(3), p2(3), r(1)> * n_caps]``. The cubic head defines ``p_d(theta) = c*(theta-theta_i)``
    so the path point/tangent are functions of the progress state (re-linearised per node).
    """
    ocp = AcadosOcp()
    ocp.model = create_mpcc_model(parameters)
    nx = ocp.model.x.rows()  # 14
    ocp.solver_options.N_horizon = N

    hover_thrust = parameters["mass"] * -parameters["gravity_vec"][-1]

    # Per-stage parameters: contouring head [theta_i(1), c_x(4), c_y(4), c_z(4), v_target(1)]
    # = the LOCAL CUBIC of the path around this node's predicted theta, followed by n_caps
    # avoidance capsules [p1(3), p2(3), r(1)] (poles + gate stand/bars).
    n_head = 14  # 1 (theta_i) + 12 (cubic coeffs) + 1 (v_target)
    n_p = n_head + 7 * n_caps
    p = ca.MX.sym("p", n_p)
    ocp.model.p = p
    ocp.parameter_values = np.zeros(n_p)
    theta_i = p[0]
    c_x, c_y, c_z = p[1:5], p[5:9], p[9:13]
    v_target = p[13]

    pos = ocp.model.x[0:3]
    rpy = ocp.model.x[3:6]
    drpy = ocp.model.x[9:12]
    theta = ocp.model.x[12]
    v_theta = ocp.model.x[13]
    rpy_cmd = ocp.model.u[0:3]
    thrust = ocp.model.u[3]
    a_theta = ocp.model.u[4]

    # Embedded path: p_d(theta) and its tangent are evaluated symbolically from the progress
    # STATE theta via the local cubic — so e_c/e_l genuinely depend on theta (formulation A).
    t = theta - theta_i
    p_d = ca.vertcat(
        c_x[0] * t**3 + c_x[1] * t**2 + c_x[2] * t + c_x[3],
        c_y[0] * t**3 + c_y[1] * t**2 + c_y[2] * t + c_y[3],
        c_z[0] * t**3 + c_z[1] * t**2 + c_z[2] * t + c_z[3],
    )
    t_vec = ca.vertcat(
        3 * c_x[0] * t**2 + 2 * c_x[1] * t + c_x[2],
        3 * c_y[0] * t**2 + 2 * c_y[1] * t + c_y[2],
        3 * c_z[0] * t**2 + 2 * c_z[1] * t + c_z[2],
    )
    d = pos - p_d
    t_norm = ca.norm_2(t_vec) + 1e-6
    e_l = ca.dot(t_vec, d) / t_norm  # lag (longitudinal, signed) — depends on theta
    e_c = d - e_l * (t_vec / t_norm)  # contouring (perpendicular, 3-vector) — depends on theta
    e_v = v_theta - v_target  # progress-speed tracking (the "reward")

    # Avoidance soft-barrier: one smooth residual per capsule (poles + gate stand/bars),
    # appended to the least-squares cost. Penalises proximity to each capsule axis instead of
    # adding hard constraints, so the QP never goes infeasible and the gate opening stays free.
    y_obs = _capsule_barrier(p[n_head:], pos, n_caps)

    # Ground-clearance soft penalty: one-sided residual max(0, ground_soft_z - z), weighted
    # ground_penalty. Penalises altitude below ground_soft_z (zero above), so the optimiser climbs
    # off the floor while accelerating toward the first gate instead of sinking into it. Appended
    # LAST so the weight layout is [weight_diagonals(...), ground_penalty].
    r_ground = ca.fmax(0.0, ground_soft_z - pos[2])

    # Nonlinear least-squares residual (nonlinear in theta now; acados' Gauss-Newton SQP
    # re-linearises the path each iteration).
    y = ca.vertcat(
        e_c, e_l, rpy, drpy, e_v, rpy_cmd, thrust - hover_thrust, a_theta, y_obs, r_ground
    )
    y_e = ca.vertcat(e_c, e_l, rpy, drpy, e_v, y_obs, r_ground)
    ocp.model.cost_y_expr = y
    ocp.model.cost_y_expr_e = y_e
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"

    # Contouring (q_c, hold the path) and lag (q_l, keep the reference point p_d(theta) tied
    # to the drone longitudinally). Because theta is now a free inherited STATE (#2), q_l is
    # what stops it running away: a strong lag penalty forces the optimiser to slow v_theta
    # whenever the drone falls behind p_d(theta), so theta stays glued to the drone (and the
    # drone slows into corners instead of letting the reference sprint ahead). The progress
    # penalty q_v is kept light so it sets a cruise speed without overpowering the lag — lag
    # must dominate progress (≈30:1), or theta marches at the target speed regardless of the
    # drone and the reference leads it off the line.
    # Baseline cost weights are the controller's own BASELINE_WEIGHTS; weight_diagonals just lays
    # them out (multipliers=1 → exactly these weights). The controller rebuilds W per tick with
    # the policy's multipliers via the same helper, so the built and the runtime weights can never
    # diverge. The n_caps capsule-barrier residuals are weighted capsule_penalty each (appended
    # last to match the y / y_e ordering above).
    w_stage, w_term = wp.weight_diagonals(
        np.ones(wp.N_ACTIONS), BASELINE_WEIGHTS, n_caps, capsule_penalty
    )
    # Append the ground-clearance residual weight last (matches the y / y_e ordering above).
    w_stage = np.append(w_stage, ground_penalty)
    w_term = np.append(w_term, ground_penalty)
    W = np.diag(w_stage)
    W_e = np.diag(w_term)
    ocp.cost.W = W
    ocp.cost.W_e = W_e
    ocp.cost.yref = np.zeros(y.rows())
    ocp.cost.yref_e = np.zeros(y_e.rows())

    # State bounds: z floor/ceiling (2), rpy (3,4,5), vel (6,7,8), v_theta (13: 0..vtheta).
    ocp.constraints.lbx = np.array([z_min, -0.7, -0.7, -0.7, -v_max, -v_max, -v_max, 0.0])
    ocp.constraints.ubx = np.array([z_max, 0.7, 0.7, 0.7, v_max, v_max, v_max, vtheta_max])
    ocp.constraints.idxbx = np.array([2, 3, 4, 5, 6, 7, 8, 13])
    # Soften the velocity and v_theta bounds so a transient overspeed never makes the QP
    # infeasible (positions 4,5,6,7 within idxbx → vel x/y/z and v_theta).
    ocp.constraints.idxsbx = np.array([4, 5, 6, 7])
    # Slack costs for the softened velocity / v_theta bounds. Obstacle/gate avoidance is now a
    # cost barrier (above), not a constraint, so there are no keep-out slacks here any more.
    ocp.cost.zl = 1e3 * np.ones(4)
    ocp.cost.zu = 1e3 * np.ones(4)
    ocp.cost.Zl = 1e3 * np.ones(4)
    ocp.cost.Zu = 1e3 * np.ones(4)

    # Input bounds: rpy commands, collective thrust, progress acceleration.
    ocp.constraints.lbu = np.array([-0.7, -0.7, -0.7, parameters["thrust_min"] * 4, -atheta_max])
    ocp.constraints.ubu = np.array([0.7, 0.7, 0.7, parameters["thrust_max"] * 4, atheta_max])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])

    ocp.constraints.x0 = np.zeros(nx)

    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    # Real-time iteration: ONE SQP step per tick (prepare + feedback), instead of iterating a
    # full SQP to 1e-6 (up to 50 re-linearisations/tick) which blew the ~20 ms control budget.
    # The shifted warm start lands each tick near the optimum, so a single iteration tracks
    # well; correctness was already verified with full SQP (#1/#2) before this speed change.
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.tol = 1e-3
    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1
    # HPIPM QP iteration cap. At 10 the QP was hitting the limit near gates (status 3 at "QP
    # iteration 9" → solve fails → braking to hover → crash), so give it room to converge. This
    # is baked into the generated code, but _solver_signature hashes this function's source, so
    # editing the value here auto-invalidates the cache and rebuilds the solver.
    ocp.solver_options.qp_solver_iter_max = 30
    ocp.solver_options.nlp_solver_max_iter = 1
    if time_steps is not None:
        # Non-uniform shooting grid: same N nodes, but the intervals grow toward the end so
        # the horizon reaches much further ahead at ~no extra cost. tf must equal their sum.
        ts = np.asarray(time_steps, dtype=float)
        ocp.solver_options.time_steps = ts
        ocp.solver_options.tf = float(ts.sum())
        # A couple of ERK substeps keeps the integration accurate over the larger far steps.
        ocp.solver_options.sim_method_num_steps = 2
        # Cost scaling: acados' default is [time_steps, 1.0] — each stage scaled by its
        # shooting interval and the *terminal* node by 1.0. With a growing grid that terminal
        # 1.0 is ~50x any single stage (dt=0.02) and ~2x all stages combined, so the cost is
        # dominated by the last node — which sits ~1.1 s / ~2 m ahead (the blue marker). The
        # drone then optimises mostly to land that far node on the path and chases it, cutting
        # corners. Fix: weight *every* node, terminal included, equally by the real control
        # period dt (= ts[0]). No node dominates → the whole horizon is tracked uniformly and
        # the near (executed) nodes get their fair share. (Equals the uniform default when
        # HORIZON_GROWTH == 1.0, except the terminal is now dt-weighted too rather than 1.0.)
        ocp.solver_options.cost_scaling = ts[0] * np.ones(N + 1)
    else:
        ocp.solver_options.tf = Tf

    # Reuse the already-generated/compiled solver when nothing that affects the C code has
    # changed. Generating + compiling the OCP costs many seconds at every start; it only needs
    # redoing when the OCP structure changes (N, n_caps, the drone model, the time grid, the
    # bounds/penalties) or the builder source is edited — all captured by the signature below.
    gen_dir = "c_generated_code"
    json_file = os.path.join(gen_dir, "mpcc_planner.json")
    sig_file = os.path.join(gen_dir, "mpcc_planner.sig")
    signature = _solver_signature(
        {
            "N": N,
            "z_min": z_min,
            "z_max": z_max,
            "v_max": v_max,
            "vtheta_max": vtheta_max,
            "atheta_max": atheta_max,
            "n_caps": n_caps,
            "capsule_penalty": capsule_penalty,
            "ground_soft_z": ground_soft_z,
            "ground_penalty": ground_penalty,
            "time_steps": None if time_steps is None else np.asarray(time_steps, float),
            "parameters": parameters,
            # Baseline weights are a module constant, not in this function's source, so the source
            # hash would miss a baseline edit; include them here to keep the cache honest.
            "weight_baseline": BASELINE_WEIGHTS,
        }
    )
    lib_exists = bool(glob.glob(os.path.join(gen_dir, "libacados_ocp_solver_mpcc_planner.*")))
    cached_sig = None
    if os.path.exists(sig_file):
        with open(sig_file) as f:
            cached_sig = f.read().strip()
    fresh = lib_exists and cached_sig == signature  # reuse the compiled code as-is

    solver = AcadosOcpSolver(
        ocp, json_file=json_file, verbose=verbose, build=not fresh, generate=not fresh
    )
    if not fresh:  # record the signature of the code we just generated/built
        os.makedirs(gen_dir, exist_ok=True)
        with open(sig_file, "w") as f:
            f.write(signature)
    return solver, ocp


# ── B-spline path planner ────────────────────────────────────────────────────────


class SimplePlanner:
    """Minimal B-spline trajectory planner — exactly the presentation design, no extras.

    * **3 fixed waypoints per gate** (before / center / after), placed along the gate
      normal so the spline crosses each gate straight on.
    * **4 freely movable waypoints per inter-gate segment**, shifted by ``scipy.minimize``
      to minimise a cost of three terms: obstacle penalty, gate-frame penalty, and
      straight-line deviation penalty.
    * a **weighted cubic B-spline** (gate points high weight, free points low weight),
    * a **trapezoidal speed profile** (cruise ``TARGET_SPEED``) resampled at uniform time,
    * **re-planned** whenever a measured gate/obstacle pose changes.

    Every plan covers the *full* track from a fixed start position through all gates (also
    on a replan), so a replan refines the complete path rather than truncating it to the
    remainder from the drone's current position; the previous solution warm-starts the
    optimiser (identical waypoint structure across replans ⇒ direct reuse).
    """

    TARGET_SPEED = 3  # m/s — cruise speed of the velocity profile
    V_EDGE = 0.6  # m/s — speed at trajectory start/end
    ACCEL_DIST = 0.8  # m — ramp-up arc length
    DECEL_DIST = 0.8  # m — ramp-down arc length
    APPROACH_DIST = 0.35  # m — before-gate waypoint offset along the gate normal
    DEPART_DIST = 0.4  # m — after-gate waypoint offset along the gate normal
    #: Obstacle-aware entry/exit waypoints. A pole sitting on a gate's approach/departure line
    #: would otherwise put the fixed entry/exit waypoint inside the pole's keep-out, forcing a
    #: contorted swerve (the Gate-2/Obstacle-2 overshoot and the Gate-3/Obstacle-3 swing). Fix:
    #: SHORTEN the offset toward the gate centre until the waypoint clears every obstacle (stays
    #: aligned with the opening, no frame risk); only if even the minimum offset can't clear it (a
    #: pole right at the gate mouth) add a small perpendicular nudge, capped at GATE_WP_MAX_SHIFT
    #: so it stays well inside the opening (frame bars sit at GATE_BAR_DIST ≈ 0.28 m).
    #: The ENTRY may be shortened all the way to GATE_WP_MIN_DIST, but the EXIT has its own,
    #: LARGER floor GATE_EXIT_MIN_DIST: it must stay far enough PAST the gate plane that the drone
    #: actually crosses it (and the env registers the pass) before the line curves toward the next
    #: gate. Shortening the exit too aggressively made the drone turn back before completing the
    #: pass at reversal gates; if that pulls it nearer an obstacle, the MPCC capsule avoidance
    #: handles the clearance.
    GATE_WP_MIN_DIST = 0.15  # m — floor when shortening the ENTRY offset to clear an obstacle
    GATE_EXIT_MIN_DIST = 0.30  # m — floor for the EXIT offset (keep the gate pass completing)
    GATE_WP_MAX_SHIFT = 0.12  # m — max perpendicular nudge (fallback), kept inside the opening
    #: Upward bias (m) applied to the gate entry/center/exit waypoints. The drone tends to track
    #: the reference slightly LOW through a gate (it pitches/rolls to follow the path, trading
    #: vertical thrust, so it sags a few cm below the planned height). Aiming the gate waypoints
    #: a touch high cancels that steady offset so it crosses nearer the gate centre. Tune up if
    #: it still passes low, down toward 0 if it now passes high.
    GATE_Z_BIAS = 0.04
    DRONE_RADIUS = 0.09  # m — drone half-extent (gate inflation & obstacle clearance)
    FRAME_MARGIN = 0.08  # m — extra gate-frame inflation
    OBSTACLE_RADIUS = 0.015  # m — physical pole radius
    OBSTACLE_BUFFER = 0.20  # m — extra safety gap to the pole surface
    PLAN_CLEARANCE = OBSTACLE_RADIUS + DRONE_RADIUS + OBSTACLE_BUFFER  # center-to-center
    GATE_FRAME_WEIGHT = 40.0  # cost weight: entering gate-frame material
    CYL_WEIGHT = 20.0  # cost weight: violating obstacle clearance
    DEVIATION_WEIGHT = 2.0  # cost weight: straying from the straight-line path
    N_INTERMEDIATE = 4  # free waypoints per inter-gate segment
    N_SAMPLE = 100  # samples along the curve for the cost
    OPT_MAXITER = 200  # max L-BFGS-B iterations
    MAX_OPT_TIME = 0.20  # s — wall-clock budget for the optimiser
    BSPLINE_DEGREE = 3  # cubic
    SMOOTHING = 0.5  # splprep smoothing factor
    W_ANCHOR = 30.0  # fit weight: start waypoint
    W_GATE = 4.0  # fit weight: gate waypoints
    W_FREE = 1.0  # fit weight: free intermediates
    MIN_REF_Z = 0.1  # m — hard floor for the reference

    _WP_LO = np.array([-2.4, -1.4, 0.1])
    _WP_HI = np.array([2.4, 1.4, 1.45])

    def __init__(
        self, obs: dict, config: object, N: int = 25, target_speed: float | None = None
    ) -> None:
        """Build the initial full-track path from the current observation.

        ``target_speed`` overrides the class default cruise speed on this instance, so the
        controller can keep its ``V_TARGET`` as the single source of truth and the planner's
        speed profile can never silently diverge from the MPCC progress target.
        """
        self.freq = config.env.freq
        self.N = N
        if target_speed is not None:
            self.TARGET_SPEED = float(target_speed)
        self._start_pos = obs["pos"].copy()  # fixed full-trajectory anchor (theta = 0)
        self._prev_visited = obs["gates_visited"].copy()
        self._prev_obs_visited = obs["obstacles_visited"].copy()
        self._warm_intermediates = None  # previous optimiser solution (warm start)
        self._warm_target_gate = None
        self.build(obs)

    # ── Public API (same interface as BSplinePlanner) ─────────────────────────

    def get_reference(self, tick: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the (pos, vel) reference window of ``N + 1`` samples from ``tick``."""
        i = min(tick, self.tick_max)
        return self.pos[i : i + self.N + 1], self.vel[i : i + self.N + 1]

    def update(self, obs: dict) -> bool:
        """Re-plan when a gate/obstacle becomes newly visited; return whether it did."""
        new_gates, new_obs = obs["gates_visited"], obs["obstacles_visited"]
        replanned = bool(
            np.any(new_gates & ~self._prev_visited) or np.any(new_obs & ~self._prev_obs_visited)
        )
        if replanned:
            self.build(obs)
        self._prev_visited = new_gates.copy()
        self._prev_obs_visited = new_obs.copy()
        return replanned

    def build(self, obs: dict, v_start: float | None = None) -> None:
        """Compute and install a fresh full-track trajectory from ``obs``."""
        self._apply_trajectory(self._compute_trajectory(obs, v_start))

    def _apply_trajectory(self, result: dict) -> None:
        self.pos = result["pos"]
        self.vel = result["vel"]
        self.tick_max = result["tick_max"]
        self.gate_thetas = result.get("gate_thetas")  # arc length of each gate centre (or None)
        self._raw_waypoints = result["raw_waypoints"]
        if result.get("intermediates") is not None:  # remember as warm start for next replan
            self._warm_intermediates = result["intermediates"]
            self._warm_target_gate = result["target_gate"]
        if result.get("tck") is not None:  # geometric path for the MPCC controller
            self._tck = result["tck"]
            self._s_lut = result["s_lut"]
            self._u_lut = result["u_lut"]
            self.length = result["length"]
            # Cache dense samples + unit tangents so project_to_theta (called every control
            # tick) is a cheap argmin instead of a 2000-point splev each time.
            P = np.stack(splev(self._u_lut, self._tck), axis=-1)
            T = np.gradient(P, axis=0)
            self._path_dense = P
            self._path_dense_tan = T / np.clip(np.linalg.norm(T, axis=1, keepdims=True), 1e-9, None)

    # ── Build ─────────────────────────────────────────────────────────────────

    def _compute_trajectory(self, obs: dict, v_start: float | None = None) -> dict:
        if v_start is None:
            v_start = self.V_EDGE
        target_gate = int(obs["target_gate"])
        if target_gate == -1:  # all gates passed → hover in place
            n = self.N + 2
            return {
                "pos": np.tile(obs["pos"], (n, 1)),
                "vel": np.zeros((n, 3)),
                "tick_max": 1,
                "raw_waypoints": [(obs["pos"].copy(), None)],
            }

        # Always plan the *full* trajectory: from the fixed start position through every
        # gate (not just the remainder from the drone's current position), so a replan only
        # refines the complete path instead of truncating it. The previous solution warm-
        # starts the optimiser — the waypoint structure is identical across replans.
        start_pos = self._start_pos.copy()
        cylinders = [_Cylinder(p, self.PLAN_CLEARANCE) for p in obs["obstacles_pos"]]
        frames = [
            _GateFrame(gp, gq, drone_r=self.DRONE_RADIUS + self.FRAME_MARGIN)
            for gp, gq in zip(obs["gates_pos"], obs["gates_quat"])
        ]
        cyl_tuples = [(float(c.xy[0]), float(c.xy[1]), self.PLAN_CLEARANCE) for c in cylinders]
        gate_data = self._gate_data(obs)

        x0 = self._warm_start(gate_data)
        intermediates = self._optimize(start_pos, gate_data, cyl_tuples, frames, x0=x0)
        raw = self._build_waypoints(start_pos, gate_data, intermediates)
        result = self._finalize(raw, v_start)
        result["raw_waypoints"] = raw
        result["intermediates"] = intermediates  # kept as the warm start for the next replan
        result["target_gate"] = target_gate
        # Arc length θ of each gate centre on this path, so the controller can stop progress at
        # the gate it's flying at until the env confirms the pass (gate-progress barrier).
        centers = np.array([gd[0] for gd in gate_data])
        result["gate_thetas"] = self._project_centers_to_arclength(result, centers)
        return result

    def _project_centers_to_arclength(self, result: dict, centers: np.ndarray) -> np.ndarray:
        """Arc length θ of each gate centre on the installed path, in track order (monotonic).

        Each centre maps to the nearest sample of the dense path; the search start only moves
        forward gate by gate, so the mapping is monotonic and a self-crossing path can't snap an
        earlier gate onto a later branch.
        """
        P = np.stack(splev(result["u_lut"], result["tck"]), axis=-1)
        s_lut = result["s_lut"]
        thetas = np.empty(len(centers))
        lo = 0
        for i, c in enumerate(centers):
            j = lo + int(np.argmin(np.linalg.norm(P[lo:] - c, axis=1)))
            thetas[i] = s_lut[j]
            lo = j
        return thetas

    def _warm_start(self, gate_data: list[tuple]) -> np.ndarray | None:
        """Return the previous optimiser solution as the warm start, else ``None``.

        Falls back to ``None`` (→ straight-line guess) when there is none yet or the gate
        count changed.

        Because every plan now covers the full track from the same fixed start, the
        waypoint structure is identical across replans, so the previous solution can be
        reused directly (no per-gate shifting): it seeds both the optimiser's starting
        point and the deviation anchor.
        """
        prev = self._warm_intermediates
        if prev is None or len(prev) != len(gate_data) * self.N_INTERMEDIATE:
            return None
        return prev.copy()

    def _gate_data(self, obs: dict) -> list[tuple]:
        """Per gate: (center, x_axis, entry, exit) — 3 points along the gate normal.

        Covers *all* gates, oriented from the fixed start position, so the planned path is
        always the complete track (the segment behind the drone is kept on a replan).
        """
        gates_pos, gates_quat = obs["gates_pos"], obs["gates_quat"]
        obstacles_xy = [(float(p[0]), float(p[1])) for p in obs["obstacles_pos"]]
        bias = np.array([0.0, 0.0, self.GATE_Z_BIAS])  # aim a touch high to cancel the sag
        data, prev = [], self._start_pos.copy()
        for i in range(len(gates_pos)):
            x_axis = R.from_quat(gates_quat[i]).apply([1.0, 0.0, 0.0])
            if np.dot(gates_pos[i] - prev, x_axis) < 0:  # orient so entry is on the near side
                x_axis = -x_axis
            center = gates_pos[i] + bias
            # Keep the entry/exit waypoints out of any obstacle's keep-out (see GATE_WP_MIN_DIST):
            # a pole on the approach/departure line otherwise forces a contorted swerve. The exit
            # uses a LARGER floor so the drone still crosses the gate plane (completes the pass).
            entry = self._clear_waypoint(
                center,
                -x_axis,
                self.APPROACH_DIST,
                obstacles_xy,
                self.GATE_WP_MIN_DIST,
                check_approach=True,
            )
            exit_ = self._clear_waypoint(
                center, x_axis, self.DEPART_DIST, obstacles_xy, self.GATE_EXIT_MIN_DIST
            )
            data.append((center, x_axis.copy(), entry, exit_))
            prev = exit_
        return data

    def _clear_waypoint(
        self,
        center: np.ndarray,
        axis: np.ndarray,
        base_dist: float,
        obstacles_xy: list,
        min_dist: float,
        check_approach: bool = False,
    ) -> np.ndarray:
        """Entry/exit waypoint at ``center + axis*base_dist``, kept clear of every obstacle pole.

        ``axis`` is the outward gate-normal unit vector (−normal for entry, +normal for exit).
        First SHORTEN the offset toward the gate centre (the waypoint stays on the gate axis, i.e.
        centred in the opening — no frame risk) until it clears ``PLAN_CLEARANCE`` in xy from all
        poles, but never below ``min_dist`` (the exit floor is larger so the gate pass still
        completes). Only if it still can't clear at ``min_dist`` (a pole right at the gate mouth)
        add a small perpendicular nudge, capped at ``GATE_WP_MAX_SHIFT`` so the waypoint stays well
        inside the opening. Falls back to the shortened point if nothing clears (the MPCC capsule
        avoidance is the backstop).

        With ``check_approach`` (entry only), clearance is tested against the whole committed
        approach LINE ``waypoint → center`` instead of just the waypoint point: a pole sitting on
        the approach line but slightly outside the entry point's own keep-out previously left the
        entry unmoved (the path still grazed the pole — only MPCC caught it). Checking the segment
        lets the shorten/nudge actually push the entry aside. The free intermediates handle the
        path BEFORE the entry, so only the entry→center stretch needs this.
        """

        def clear(p: np.ndarray) -> bool:
            if check_approach:
                return self._segment_clear(p, center, obstacles_xy)
            return all(
                np.hypot(p[0] - ox, p[1] - oy) >= self.PLAN_CLEARANCE for ox, oy in obstacles_xy
            )

        steps = 12
        for k in range(steps + 1):  # shorten base_dist -> min_dist
            d = base_dist - (base_dist - min_dist) * k / steps
            p = center + axis * d
            if clear(p):
                return p
        # Still blocked at the minimum offset: small bounded perpendicular nudge (in the gate
        # plane), kept inside the opening so it never approaches the frame bars.
        p = center + axis * min_dist
        right = np.array([-axis[1], axis[0], 0.0])
        right = right / (np.linalg.norm(right) + 1e-9)
        for s in np.linspace(0.0, self.GATE_WP_MAX_SHIFT, steps + 1):
            for q in (p + right * s, p - right * s):
                if clear(q):
                    return q
        return p

    def _segment_clear(self, a: np.ndarray, b: np.ndarray, obstacles_xy: list) -> bool:
        """True if every pole stays ≥ ``PLAN_CLEARANCE`` from the xy segment ``a``–``b``."""
        ax, ay, bx, by = a[0], a[1], b[0], b[1]
        dx, dy = bx - ax, by - ay
        denom = dx * dx + dy * dy
        for ox, oy in obstacles_xy:
            if denom < 1e-12:
                dist = np.hypot(ox - ax, oy - ay)
            else:
                t = min(1.0, max(0.0, ((ox - ax) * dx + (oy - ay) * dy) / denom))
                dist = np.hypot(ox - (ax + t * dx), oy - (ay + t * dy))
            if dist < self.PLAN_CLEARANCE:
                return False
        return True

    def _segment_starts(self, drone_pos: np.ndarray, gate_data: list[tuple]) -> list:
        return [drone_pos] + [gd[3] for gd in gate_data[:-1]]

    def _straight_intermediates(self, drone_pos: np.ndarray, gate_data: list[tuple]) -> np.ndarray:
        """Initial guess: linear interpolation from each segment start to the gate entry."""
        n_wps = len(gate_data) * self.N_INTERMEDIATE
        pts = np.zeros((n_wps, 3))
        for si, ((_, _, entry, _), start) in enumerate(
            zip(gate_data, self._segment_starts(drone_pos, gate_data))
        ):
            for k in range(self.N_INTERMEDIATE):
                frac = (k + 1) / (self.N_INTERMEDIATE + 1)
                pts[si * self.N_INTERMEDIATE + k] = start + frac * (entry - start)
        return pts

    def _optimize(
        self,
        drone_pos: np.ndarray,
        gate_data: list[tuple],
        cyl_tuples: list,
        frames: list,
        x0: np.ndarray | None = None,
    ) -> np.ndarray:
        """Shift the free waypoints to minimise obstacle + frame + deviation cost.

        ``x0`` seeds the optimiser (and the deviation anchor). On a replan it is the
        previous solution (warm start): the optimiser starts near the last good trajectory
        and the deviation term pulls toward *it* rather than the straight line, so the new
        plan only nudges the old one to react to the moved object — similar trajectory,
        fewer iterations. Defaults to the straight-line guess (the initial plan).
        """
        n_wps = len(gate_data) * self.N_INTERMEDIATE
        if x0 is None:
            x0 = self._straight_intermediates(drone_pos, gate_data)
        anchor = x0.flatten()
        bounds = [
            (float(lo), float(hi)) for _ in range(n_wps) for lo, hi in zip(self._WP_LO, self._WP_HI)
        ]

        def cost(x: np.ndarray) -> float:
            raw = self._build_waypoints(drone_pos, gate_data, x.reshape(n_wps, 3))
            c = self._trajectory_cost(raw, cyl_tuples, frames)
            c += self.DEVIATION_WEIGHT * float(np.sum((x - anchor) ** 2))  # straight-line term
            return c

        t0 = time.perf_counter()

        def _time_cb(_xk: np.ndarray) -> None:
            if time.perf_counter() - t0 > self.MAX_OPT_TIME:
                raise StopIteration

        result = minimize(
            cost,
            anchor,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.OPT_MAXITER, "ftol": 1e-10, "gtol": 1e-7},
            callback=_time_cb,
        )
        return result.x.reshape(n_wps, 3)

    def _build_waypoints(
        self, drone_pos: np.ndarray, gate_data: list[tuple], intermediates: np.ndarray
    ) -> list[tuple]:
        """Interleave waypoints: drone, then per gate [4 free, entry, center, exit].

        Gate points carry a non-None tag (high fit weight); free points carry ``None``
        (low weight).
        """
        pts = [(drone_pos, None)]
        for si, (center, _x, entry, exit_) in enumerate(gate_data):
            for k in range(self.N_INTERMEDIATE):
                pts.append((intermediates[si * self.N_INTERMEDIATE + k].copy(), None))
            pts.append((entry, "gate"))
            pts.append((center.copy(), "gate"))
            pts.append((exit_, "gate"))
        last_x = gate_data[-1][1]
        pts.append((gate_data[-1][3] + last_x * 0.3, "gate"))  # short run-out past the last gate
        return pts

    def _trajectory_cost(self, raw: list[tuple], cyl_tuples: list, frames: list) -> float:
        """Obstacle (cylinder) + gate-frame penalty along the sampled spline."""
        try:
            tck = self._fit(raw)
            x, y, z = splev(np.linspace(0.0, 1.0, self.N_SAMPLE), tck)
            traj = np.stack([x, y, z], axis=1)
        except Exception:
            return 1e6
        cost = 0.0
        for cx, cy, r in cyl_tuples:
            viol = np.maximum(0.0, r - np.linalg.norm(traj[:, :2] - np.array([cx, cy]), axis=1))
            cost += self.CYL_WEIGHT * float(np.sum(viol**2))
        for fr in frames:
            cost += self.GATE_FRAME_WEIGHT * float(np.sum(fr.penetration(traj) ** 2))
        return cost

    # ── Spline fit + speed profile ────────────────────────────────────────────

    def _fit(self, raw: list[tuple]) -> tuple:
        P = np.array([np.asarray(p) for p, _ in raw], dtype=float)
        w = np.array(
            [
                self.W_ANCHOR
                if i == 0
                else 0.5 * self.W_ANCHOR
                if i == len(raw) - 1
                else self.W_GATE
                if tag is not None
                else self.W_FREE
                for i, (_p, tag) in enumerate(raw)
            ]
        )
        keep = np.concatenate(([True], np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-6))
        P, w = P[keep], w[keep]
        k = min(self.BSPLINE_DEGREE, len(P) - 1)
        tck, _ = splprep([P[:, 0], P[:, 1], P[:, 2]], w=w, k=k, s=self.SMOOTHING)
        return tck

    def _speed_profile(self, s: np.ndarray, length: float, v_start: float) -> np.ndarray:
        v = np.full_like(s, self.TARGET_SPEED)
        a, d = min(self.ACCEL_DIST, 0.5 * length), min(self.DECEL_DIST, 0.5 * length)
        if a > 1e-6:
            m = s < a
            f = s[m] / a
            v[m] = v_start + (self.TARGET_SPEED - v_start) * (3 * f**2 - 2 * f**3)
        if d > 1e-6:
            m = s > length - d
            f = (length - s[m]) / d
            v[m] = np.minimum(
                v[m], self.V_EDGE + (self.TARGET_SPEED - self.V_EDGE) * (3 * f**2 - 2 * f**3)
            )
        return np.clip(v, self.V_EDGE, None)

    def _finalize(self, raw: list[tuple], v_start: float, n_dense: int = 2000) -> dict:
        tck = self._fit(raw)
        u = np.linspace(0.0, 1.0, n_dense)
        x, y, z = splev(u, tck)
        P = np.stack([x, y, z], axis=1)
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
        length = float(s[-1])

        v = self._speed_profile(s, length, v_start)
        dt_seg = np.diff(s) / np.clip(0.5 * (v[:-1] + v[1:]), 1e-3, None)
        t = np.concatenate([[0.0], np.cumsum(dt_seg)])
        t_total = float(t[-1])

        n_samp = max(int(self.freq * t_total), self.N + 2)
        t_q = np.linspace(0.0, t_total, n_samp)
        s_q = np.interp(t_q, t, s)
        u_q = np.interp(s_q, s, u)

        xq, yq, zq = splev(u_q, tck)
        dxq, dyq, dzq = splev(u_q, tck, der=1)
        tangent = np.stack([dxq, dyq, dzq], axis=1)
        tangent /= np.clip(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9, None)
        v_q = np.interp(s_q, s, v)

        pos = np.stack([xq, yq, zq], axis=1)
        vel = v_q[:, None] * tangent
        below = pos[:, 2] < self.MIN_REF_Z
        pos[below, 2] = self.MIN_REF_Z
        vel[below, 2] = np.maximum(vel[below, 2], 0.0)
        # Also keep the raw geometric path (B-spline + arc-length LUT) so a contouring
        # controller (MPCC) can query the path by arc length θ; the reference-tracking MPC
        # ignores these and just uses pos/vel.
        return {
            "pos": pos,
            "vel": vel,
            "tick_max": n_samp - 1 - self.N,
            "tck": tck,
            "s_lut": s,
            "u_lut": u,
            "length": length,
        }

    # ── Geometric-path API (used by the MPCC controller) ──────────────────────

    def path_point_tangent(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map arc length θ → (position(3), unit tangent(3)). Accepts a scalar or array."""
        th = np.atleast_1d(np.clip(theta, 0.0, self.length))
        u = np.interp(th, self._s_lut, self._u_lut)  # arc length → spline parameter
        x, y, z = splev(u, self._tck)
        dx, dy, dz = splev(u, self._tck, der=1)
        pos = np.stack([x, y, z], axis=-1)
        tan = np.stack([dx, dy, dz], axis=-1)
        tan /= np.clip(np.linalg.norm(tan, axis=-1, keepdims=True), 1e-9, None)
        if np.isscalar(theta) or np.ndim(theta) == 0:
            return pos[0], tan[0]
        return pos, tan

    def project_to_theta(
        self,
        pos: np.ndarray,
        vel: np.ndarray | None = None,
        theta_prev: float | None = None,
        back: float = 0.30,
        fwd: float = 1.0,
    ) -> float:
        """Drone position → nearest arc length θ on the path.

        The projection must never *teleport* along θ: where the path passes close to an
        earlier/later section (flying through a gate and back, a U-turn near a frame), a
        global nearest-point search can snap to the wrong branch and make the reference jump
        — the drone would then "shortcut" the path. So when ``theta_prev`` (last tick's θ) is
        given, the search is restricted to a *local, forward-biased* arc-length window
        ``[theta_prev - back, theta_prev + fwd]``. Because the window is in arc length, the
        two branches of a self-crossing (spatially close but far apart in θ) are cleanly
        separated, so θ advances continuously and can never snap backwards onto an old branch.
        ``back`` allows small projection wobble / disturbance recovery; ``fwd`` bounds the
        progress per tick (≫ the real V·dt, so honest fast progress is never clipped).

        With ``vel`` given, ties inside the window break toward the branch whose tangent
        aligns with the velocity. Uses the dense path samples + tangents cached at path-
        install time (no per-call ``splev`` over 2000 points — this runs every control tick).
        """
        d = np.linalg.norm(self._path_dense - pos, axis=1)
        speed = 0.0 if vel is None else float(np.linalg.norm(vel))
        if speed >= 0.2:
            d = d - 0.15 * (self._path_dense_tan @ (np.asarray(vel, float) / speed))
        if theta_prev is not None:
            mask = (self._s_lut >= theta_prev - back) & (self._s_lut <= theta_prev + fwd)
            if np.any(mask):
                idx = np.flatnonzero(mask)
                return float(self._s_lut[idx[int(np.argmin(d[idx]))]])
        return float(self._s_lut[int(np.argmin(d))])


# ── Controller ──────────────────────────────────────────────────────────────────


class MPCCController(Controller):
    """MPCC controller: contouring control along the warm-started SimplePlanner path."""

    GROUND_Z = 0.0
    #: Ground-clearance soft penalty (added to the MPCC cost): the drone starts on the ground
    #: (z ≈ 0.01) and, if nothing discourages it, tilts to chase the mostly-horizontal first path
    #: segment, trading away vertical thrust → it sinks into the floor before climbing (the
    #: "ground" crashes at the start). A one-sided residual max(0, GROUND_SOFT_Z − z) penalises any
    #: altitude below GROUND_SOFT_Z (zero above), so the optimiser climbs off the ground WHILE
    #: accelerating toward gate 0 — no separate takeoff phase, no wasted seconds. Also keeps a floor
    #: margin everywhere. Gates sit at 0.7/1.2 m (above GROUND_SOFT_Z), so it never fights a gate.
    GROUND_SOFT_Z = 0.35  # m — altitude below which the penalty engages
    GROUND_PENALTY = 400.0  # weight of the ground-clearance residual (higher = climbs harder)
    V_TARGET = 4  # m/s — target progress speed (cruise); matched to SimplePlanner.TARGET_SPEED
    VTHETA_MAX = 6  # m/s — hard-ish cap on progress speed
    #: Curvature speed limit: cap the progress target by the path curvature so the drone slows
    #: through sharp turns (e.g. the ~180° reversal at a gate exit) and cruises on straights,
    #: instead of carrying full V_TARGET into a corner and overshooting into the frame. The cap is
    #: a lateral-acceleration limit, v_cap = sqrt(MAX_LAT_ACC / kappa), evaluated over a short
    #: arc-length LOOKAHEAD ahead of the drone so braking starts BEFORE the curve. The MPCC tracks
    #: its own v_target (not the planner's speed profile), so this must cap v_target here to bite.
    USE_CURVATURE_SPEED = True
    MAX_LAT_ACC = 20.0  # m/s² — lateral-accel budget in turns (lower = slower/safer corners)
    CURVE_MIN_SPEED = 0.8  # m/s — floor so a very tight turn never stalls progress
    CURVE_LOOKAHEAD = 1.5  # m — arc length ahead scanned for the tightest upcoming curvature
    #: Arc-length window (m) over which the path curvature is smoothed before forming v_cap. The
    #: cubic is fit through SAMPLED path points, so tiny sampling wiggles on an essentially
    #: straight stretch produce a large spurious |r''| → a phantom curvature that needlessly
    #: throttles the drone on safe straights (the MPCC contouring would just average those out).
    #: Averaging kappa over a window the size of a REAL corner (~0.4 m) kills that high-frequency
    #: noise while keeping genuine turns. 0 disables smoothing. Raise if straights still crawl.
    CURVE_SMOOTH_M = 0.7
    #: Near-gate contouring boost. The drone tends to drift off the path LATERALLY right at a gate
    #: (a contouring error) and clip the frame. For shooting nodes whose progress θ sits within
    #: GATE_TRACK_RADIUS arc length of a gate centre, the per-stage contouring weight q_c is
    #: multiplied by GATE_TRACK_BOOST, pulling the prediction tightly onto the gate-centred line
    #: exactly where precision matters — while straights keep the loose, fast baseline weighting.
    #: Applied every tick on top of whatever weights are active (baseline or RL-scaled q_c).
    USE_GATE_TRACK_BOOST = False  # A/B: ×4 q_c near gates stayed within eval noise, hurt gate 0
    GATE_TRACK_BOOST = 4.0  # × q_c near a gate centre
    GATE_TRACK_RADIUS = 0.5  # m — arc-length half-window around each gate centre
    #: Distance (m) from the path end within which the controller switches to end-hover:
    #: it freezes every stage's progress at theta=length and sets the target progress speed
    #: to 0, so contouring+lag pull the drone onto the final path point and v_theta decays to
    #: 0 — the drone station-keeps on the last trajectory point instead of overrunning it.
    END_HOVER_TOL = 0.1
    #: Caution mode: fly slower while a nearby gate/obstacle pose is still UNCERTAIN — i.e. the
    #: object is not yet "visited"/measured, so the observation only holds its rough nominal
    #: pose. This matters most for the gate currently being approached: until the drone is close
    #: enough to measure it, the planned path through it may be off, so the controller creeps in
    #: and only resumes cruise once the true pose is known (which also triggers a replan onto the
    #: corrected path). Implemented by scaling the progress target v_target toward
    #: CAUTION_SPEED_FACTOR as the drone closes within CAUTION_RADIUS of such an object.
    USE_CAUTION = True
    CAUTION_SPEED_FACTOR = 0.5  # fraction of V_TARGET when right at an unmeasured object
    #: xy-distance (m) within which an unmeasured object starts slowing us. MUST be larger than
    #: the env's sensor_range (the xy-distance at which the exact pose is revealed and the object
    #: becomes "visited" — 0.7 m in the level configs), otherwise the pose snaps to exact before
    #: caution ever engages and this has no effect. Set a bit above sensor_range so the approach
    #: is already slow when the true pose is revealed and the replan onto it kicks in.
    CAUTION_RADIUS = 1.3  # m
    #: Gate-progress barrier: tie the MPCC progress state θ to the env's authoritative gate-pass
    #: detection. The env only increments target_gate once the drone has actually flown THROUGH
    #: the current gate's opening; until then θ is not allowed past (gate centre + GATE_PASS_LEAD).
    #: So if the drone is pushed off the line it keeps working through the current gate instead of
    #: letting its progress run on to the next one (and skipping a gate it never passed). Off →
    #: pure path progress with no gate gating.
    USE_GATE_BARRIER = True
    GATE_PASS_LEAD = 0.4  # m past the gate centre θ may reach before the env confirms the pass
    #: Gate-retry recovery. The progress θ is monotonic (v_theta >= 0) and clamped at the gate
    #: barrier, so if the drone MISSES a gate (overshoots / flies past the opening without the env
    #: confirming the pass) it can't reverse on its own — v_target brakes to 0 and it dead-hovers
    #: at the barrier until timeout. This detects that stall and flies the drone back to a point
    #: BEFORE the gate with a simple position controller (bypassing the MPCC), then hands back so
    #: the MPCC re-approaches and re-threads. The back-and-forth also raises the chance of finally
    #: sensing the real gate pose (→ replan onto the correct line). Off → old dead-hover behaviour.
    USE_GATE_RETRY = True
    RETRY_STUCK_SPEED = 0.3  # m/s — below this, at/past the gate, counts as stalled
    RETRY_STUCK_TICKS = 40  # consecutive stalled ticks before a retry triggers (~0.8 s @ 50 Hz)
    RETRY_BACK_DIST = 0.8  # m (arc length) before the gate centre the recovery retreats to
    RETRY_DONE_DIST = 0.25  # m — within this of the retreat point, hand back to the MPCC
    RETRY_MAX = 3  # give up after this many retries on one gate (then let it hover → timeout)
    REPLAN_TOL = 1e-3  # object move (m) beyond which we replan
    MAX_HOLD_TICKS = 3  # consecutive failed solves held before braking to hover
    #: SQP_RTI iterations to run on a "fresh path" tick (replan just installed / first tick /
    #: episode reset), where the warm start is far from the new optimum. Normal ticks run 1.
    RTI_BUMP_ITERS = 5

    #: Per-node growth of the shooting interval (non-uniform time grid). The first interval
    #: is the real control period dt; each later one is this much longer, so the same N
    #: nodes look much further ahead for ~no extra compute. 1.0 → classic uniform dt grid.
    #: WARNING: leave at 1.0. >1.0 is currently broken — the receding-horizon warm start
    #: (_shift_warm_start) and the per-stage theta anchors (_stage_thetas) shift by ONE NODE per
    #: tick, which only equals one control period dt on a UNIFORM grid. With growth>1 the later
    #: nodes are spaced wider than dt, so the shifted warm start lands time-misaligned; with the
    #: single SQP_RTI iteration the solve then starts from a bad iterate and the trajectory
    #: diverges (empirically ~40% out-of-bounds). Fixing it needs a time-aware re-interpolation
    #: of the warm start onto the shifted grid — not just bumping this number.
    HORIZON_GROWTH = 1.0

    #: Add the obstacle/gate-frame avoidance soft-barrier (capsule cost) to the MPCC cost.
    #: Set False to fly with no avoidance (pure path tracking) for debugging.
    USE_AVOIDANCE = True
    #: Weight on each capsule barrier residual — how hard avoidance bites.
    CAPSULE_PENALTY = 5000.0
    #: Extra safety margin (m) added to every capsule radius.
    AVOID_MARGIN = 0.15
    #: Gate-frame geometry (m), matched to the simulator's gate: outer size, bar-centre
    #: distance from the gate centre, bar radius, stand (leg) radius. Each gate becomes a
    #: stand capsule + 4 frame-bar capsules.
    GATE_OUTER = 0.72
    GATE_BAR_DIST = 0.28
    GATE_BAR_RADIUS = 0.08
    GATE_STAND_RADIUS = 0.05
    #: Obstacle pole geometry (m): radius and full height (vertical capsule from the ground).
    POLE_RADIUS = 0.015
    POLE_HEIGHT = 1.52

    #: Print a rolling per-tick timing breakdown (project / set-params / solve) every 50
    #: ticks, to find what eats the 20 ms control budget. Off by default.
    PROFILE = False
    #: Debug the "suddenly too fast" spike: print a one-line snapshot whenever the drone's actual
    #: speed exceeds DEBUG_SPIKE_FACTOR × v_target. Shows v_target, the progress speed v_theta, the
    #: lag (drone ahead of its reference point), target_gate and ticks since the last re-localise —
    #: so we can see whether the over-speed is a lag-driven sprint after a relocalise. Off by
    #: default.
    DEBUG_SPIKE = False
    DEBUG_SPIKE_FACTOR = 1.3

    #: RL weight planner (third layer): adapt the MPCC cost weights per tick from the drone
    #: state via a trained policy (mpcc_weight_policy). Off by default → exact baseline weights.
    #: When on, a missing checkpoint falls back to an identity policy (still the baseline).
    USE_RL_WEIGHTS = False
    #: Checkpoint for the weight policy (produced by train_mpcc_weights.py).
    RL_WEIGHT_CKPT = "lsy_drone_racing/control/mpcc_weights.ckpt"

    #: MPC prediction-horizon length (number of shooting nodes). THIS is the one knob for the
    #: horizon: deployment uses 25. Training (train_mpcc_weights.py --mpc_horizon) temporarily
    #: lowers it for speed by overriding this class attribute in its worker processes — it does
    #: NOT edit this value, so deployment keeps 25. The weight policy is horizon-agnostic (its
    #: features/actions don't depend on N), so a policy trained at a lower N deploys fine at 25.
    #: >>> To change the deployment horizon, edit THIS number. <<<
    N_HORIZON = 25

    #: Arc-length sampling step (m) for the internal cubic spline that feeds the embedded
    #: path coefficients to acados. Finer = closer to the planner B-spline; 5 cm is ample.
    _CUBIC_DS = 0.05

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Build the MPCC solver and the initial path."""
        super().__init__(obs, info, config)
        self._config = config
        # xy-distance at which the env reveals an object's true pose (and marks it "visited").
        # The caution slowdown bottoms out here, so the drone is already at its slowest the
        # moment a blind-approached gate's real position is revealed and its replan kicks in.
        self._sensor_range = float(getattr(config.env, "sensor_range", 0.7))
        self._N = self.N_HORIZON
        self._dt = 1 / config.env.freq
        # Non-uniform shooting grid: first interval = the real control period dt, each later
        # interval grown by HORIZON_GROWTH, so the same N nodes reach much further ahead.
        # node_t[k] is the predicted time of node k (used to propagate the per-stage theta).
        self._time_steps = self._dt * self.HORIZON_GROWTH ** np.arange(self._N)
        self._node_t = np.concatenate(([0.0], np.cumsum(self._time_steps)))  # (N+1,)
        self._T_HORIZON = float(self._node_t[-1])
        self.drone_params = load_params("so_rpy", config.sim.drone_model)
        self._hover_thrust = self.drone_params["mass"] * -self.drone_params["gravity_vec"][-1]

        self._n_obstacles = int(len(obs["obstacles_pos"]))
        self._n_gates = int(len(obs["gates_pos"]))
        # Avoidance capsules: one per pole + 5 per gate (stand + 4 frame bars). The stand slot
        # is always allocated and set to a far-away dummy when the gate sits on the ground, so
        # the capsule count stays fixed for the life of the solver.
        self._n_caps = (self._n_obstacles + 5 * self._n_gates) if self.USE_AVOIDANCE else 0
        self._solver, self._ocp = create_mpcc_ocp_solver(
            self._T_HORIZON,
            self._N,
            self.drone_params,
            z_min=self.GROUND_Z,
            vtheta_max=self.VTHETA_MAX,
            n_caps=self._n_caps,
            capsule_penalty=self.CAPSULE_PENALTY,
            ground_soft_z=self.GROUND_SOFT_Z,
            ground_penalty=self.GROUND_PENALTY,
            time_steps=self._time_steps if self.HORIZON_GROWTH != 1.0 else None,
        )
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()

        # Baseline cost diagonals the solver was built with (multipliers = 1, ground weight
        # appended last). Used as the static base for the near-gate q_c boost on the no-RL path.
        w_s, w_t = wp.weight_diagonals(
            np.ones(wp.N_ACTIONS), BASELINE_WEIGHTS, self._n_caps, self.CAPSULE_PENALTY
        )
        self._w_stage_base = np.append(w_s, self.GROUND_PENALTY)
        self._w_term_base = np.append(w_t, self.GROUND_PENALTY)

        # Background replanning (never stalls the 50 Hz loop).
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._prof = {"proj": 0.0, "setp": 0.0, "solve": 0.0, "n": 0}

        # RL weight planner (third layer). The deployment policy runs internally; the trainer
        # injects multipliers via set_external_multipliers (then _weight_policy is unused).
        self._weight_policy = WeightPolicy(self.RL_WEIGHT_CKPT) if self.USE_RL_WEIGHTS else None
        self._external_mult = None  # set by the trainer to override the internal policy
        self._last_features = None  # last weight-feature vector (logging / inspection)
        self._last_mult = None  # last weight multipliers actually applied (None ⇒ baseline/RL off)
        self.last_solve_ok = False  # whether the most recent solve succeeded (training reward)

        # Per-episode state (planner, embedded path, warm start). Factored out so the trainer
        # can reuse the (expensive) acados solver across episodes / randomized tracks.
        self.reset_for_new_episode(obs)

    def reset_for_new_episode(self, obs: dict) -> None:
        """Rebuild the planner/embedded path and clear all warm-start state for a fresh episode.

        Reuses the already-built acados solver (only the path + warm start are episode-specific),
        so the training loop can run many episodes without paying the solver-build cost again.
        """
        # Path planner (warm-started B-spline planner; we use its geometric-path API). Its cruise
        # speed is pinned to V_TARGET so planner profile and MPCC progress target stay in sync.
        self.planner = SimplePlanner(obs, self._config, N=self._N, target_speed=self.V_TARGET)
        # Internal arc-length cubic spline of the planner path; supplies the local cubic
        # coefficients the model needs to evaluate p_d(theta) symbolically. Rebuilt on replan.
        self._cubic = None
        self._cubic_smax = 0.0
        self._curv_s = None  # arc-length grid of the curvature speed cap (set by _build_cubic)
        self._vcap = None  # curvature-limited speed cap v_cap(s) along the path
        self._build_cubic()
        self._snapshot_objects(obs)
        self._replan_future = None

        # Near-gate contouring boost: live per-node q_c factor currently pushed to the solver.
        # The reused solver may still carry the previous episode's boosts, so re-base every stage
        # and clear the tracker (skipped under RL, which rebuilds every node's W each tick anyway).
        self._boost_applied: dict[int, float] = {}
        if self.USE_GATE_TRACK_BOOST and not self.USE_RL_WEIGHTS:
            W_s, W_t = np.diag(self._w_stage_base), np.diag(self._w_term_base)
            for k in range(self._N):
                self._solver.cost_set(k, "W", W_s)
            self._solver.cost_set(self._N, "W", W_t)

        self._theta_pred = None  # last solve's per-stage theta (warm start for relinearisation)
        self._pos_pred = None  # last solve's per-stage positions (keep-out linearisation point)
        self._x_pred = None  # last solve's full state trajectory (N+1, nx) — primal warm start
        self._u_pred = None  # last solve's full input trajectory (N, nu)   — primal warm start
        self._last_u = None
        self._consec_fail = 0
        self._finished = False
        self._last_obs = obs
        self._theta_est = None  # last tick's progress θ; constrains the projection (no jumps)
        self._progress_point = None  # drone's current projection on the path (render marker)
        self._external_mult = None
        self._ticks_since_reloc = 0  # ticks since the last re-localise (DEBUG_SPIKE context)

        # Gate-retry recovery state (see USE_GATE_RETRY).
        self._stuck_ticks = 0  # consecutive stalled ticks at the current (unpassed) gate
        self._retrying = False  # currently flying back to re-attempt a missed gate
        self._retry_target = None  # world point (before the gate) the recovery flies to
        self._retry_theta = 0.0  # arc length of that retreat point (MPCC re-localises here)
        self._retry_gate = -1  # the gate index we are retrying
        self._retry_count = 0  # retries already spent on the current gate

    def set_external_multipliers(self, mult: np.ndarray | None) -> None:
        """Inject weight multipliers from an outside policy (training); None → internal policy."""
        self._external_mult = None if mult is None else np.asarray(mult, dtype=float)

    # ── Re-planning (background) ──────────────────────────────────────────────

    def _snapshot_objects(self, obs: dict):
        self._last_gates_pos = obs["gates_pos"].copy()
        self._last_gates_quat = obs["gates_quat"].copy()
        self._last_obstacles_pos = obs["obstacles_pos"].copy()

    def _objects_changed(self, obs: dict) -> bool:
        return bool(
            np.any(np.abs(obs["gates_pos"] - self._last_gates_pos) > self.REPLAN_TOL)
            or np.any(np.abs(obs["gates_quat"] - self._last_gates_quat) > self.REPLAN_TOL)
            or np.any(np.abs(obs["obstacles_pos"] - self._last_obstacles_pos) > self.REPLAN_TOL)
        )

    @staticmethod
    def _copy_obs(obs: dict) -> dict:
        return {k: (v.copy() if hasattr(v, "copy") else v) for k, v in obs.items()}

    def _maybe_replan(self, obs: dict):
        # Install a finished replan (just swaps the path; theta re-projects next).
        if self._replan_future is not None and self._replan_future.done():
            try:
                self.planner._apply_trajectory(self._replan_future.result())
                self._build_cubic()  # refresh the embedded path coefficients onto the new path

                # New path ⇒ the old progress state no longer means the same arc length, so
                # the inherited θ/positions are stale. Clear them; compute_control then
                # re-localises by projecting onto the fresh path on the next tick (#2).
                self._theta_pred = None
                self._pos_pred = None

            except Exception as exc:
                print(f"[MPCC] background replan failed: {exc!r}")
            self._replan_future = None
        # Kick off a new replan if the track changed and none is in flight.
        if self._replan_future is None and self._objects_changed(obs):
            self._replan_future = self._executor.submit(
                self.planner._compute_trajectory, self._copy_obs(obs), self.V_TARGET
            )
            self._snapshot_objects(obs)

    # ── Embedded path: local cubic coefficients ───────────────────────────────

    def _build_cubic(self) -> None:
        """(Re)build the internal arc-length cubic spline of the planner path.

        The MPCC cost evaluates p_d(theta) as a function of the progress STATE theta
        (formulation A / MPCC++ eq. 5). acados does that from per-node local cubic
        coefficients, which this spline supplies via ``_path_segment_coeffs``. Built from the
        planner's public ``path_point_tangent`` so it needs no planner internals.
        """
        length = float(getattr(self.planner, "length", 0.0))
        if length <= 1e-6:
            self._cubic = None
            self._curv_s = None
            self._vcap = None
            return
        n = max(self._N + 2, int(length / self._CUBIC_DS) + 1)
        s = np.linspace(0.0, length, n)
        pos, _ = self.planner.path_point_tangent(s)
        self._cubic = CubicSpline(s, np.asarray(pos, dtype=float))
        self._cubic_smax = length
        self._build_curvature_cap(s)

    def _build_curvature_cap(self, s: np.ndarray) -> None:
        """Precompute the curvature-limited speed cap v_cap(s) along the path.

        The internal cubic is parameterised by arc length, so its second derivative is the
        curvature vector and ``kappa = |r''(s)|``. The cap is the lateral-accel limit
        ``sqrt(MAX_LAT_ACC / kappa)``, clipped to ``[CURVE_MIN_SPEED, V_TARGET]`` (full speed on
        straights, floored on the sharpest turns). Looked up per tick by ``_curvature_speed_cap``.
        """
        if not self.USE_CURVATURE_SPEED or self._cubic is None:
            self._curv_s = None
            self._vcap = None
            return
        kappa = np.linalg.norm(self._cubic(s, 2), axis=1)
        # Smooth out sampling-induced curvature noise on straight stretches (see CURVE_SMOOTH_M):
        # average kappa over a window the size of a real corner so phantom curvature on safe
        # straights doesn't throttle the drone, while genuine turns survive.
        ds = (s[-1] - s[0]) / max(len(s) - 1, 1)
        win = int(round(self.CURVE_SMOOTH_M / ds)) if ds > 0 else 0
        if win > 1:
            kernel = np.ones(win) / win
            kappa = np.convolve(np.pad(kappa, win // 2, mode="edge"), kernel, mode="valid")[
                : len(s)
            ]
        vcap = np.sqrt(self.MAX_LAT_ACC / np.maximum(kappa, 1e-6))
        self._curv_s = s
        self._vcap = np.clip(vcap, self.CURVE_MIN_SPEED, self.V_TARGET)

    def _curvature_speed_cap(self, theta0: float) -> float:
        """Tightest curvature speed cap over the next ``CURVE_LOOKAHEAD`` metres of path.

        Scanning ahead (not just at theta0) makes the drone start braking BEFORE it enters a
        curve, so it is already slow at the apex. Returns V_TARGET when the cap is unavailable.
        """
        if not self.USE_CURVATURE_SPEED or self._vcap is None:
            return float(self.V_TARGET)
        mask = (self._curv_s >= theta0) & (self._curv_s <= theta0 + self.CURVE_LOOKAHEAD)
        if not np.any(mask):
            return float(self.V_TARGET)
        return float(np.min(self._vcap[mask]))

    def _path_segment_coeffs(
        self, theta: float
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Local cubic coefficients (theta_i, c_x, c_y, c_z) of the segment containing theta.

        Matches the symbolic model: ``p_d(theta) = c[0]*t^3 + c[1]*t^2 + c[2]*t + c[3]`` with
        ``t = theta - theta_i``. scipy stores these in ``CubicSpline.c[:, seg, dim]``.
        """
        cs = self._cubic
        th = float(np.clip(theta, 0.0, self._cubic_smax))
        seg = int(np.searchsorted(cs.x, th, side="right") - 1)
        seg = int(np.clip(seg, 0, len(cs.x) - 2))
        return float(cs.x[seg]), cs.c[:, seg, 0], cs.c[:, seg, 1], cs.c[:, seg, 2]

    # ── Control ───────────────────────────────────────────────────────────────

    def _build_capsule_params(self, obs: dict) -> np.ndarray:
        """Flat avoidance-capsule params ``[p1(3), p2(3), r(1)] * n_caps`` from the live poses.

        Poles become a vertical capsule (ground -> POLE_HEIGHT). Each gate becomes a stand
        (leg) plus 4 frame bars, built from the gate position and its up/right axes; radii get
        AVOID_MARGIN. The stand slot is set to a far-away dummy when the gate is airborne (no
        leg), so the capsule count matches the fixed ``n_caps`` the solver was built with.
        """
        caps = np.empty(self._n_caps * 7)
        i = 0

        def put(p1: np.ndarray, p2: np.ndarray, r: float) -> None:
            nonlocal i
            caps[i * 7 + 0 : i * 7 + 3] = p1
            caps[i * 7 + 3 : i * 7 + 6] = p2
            caps[i * 7 + 6] = r
            i += 1

        r_pole = self.POLE_RADIUS + self.AVOID_MARGIN
        for op in obs["obstacles_pos"]:
            put([op[0], op[1], 0.0], [op[0], op[1], self.POLE_HEIGHT], r_pole)

        half = self.GATE_OUTER / 2.0
        bar_dist = self.GATE_BAR_DIST
        r_bar = self.GATE_BAR_RADIUS + self.AVOID_MARGIN
        r_stand = self.GATE_STAND_RADIUS + self.AVOID_MARGIN
        far = np.full(3, 1e3)  # dummy capsule, never near the drone
        for gp, gq in zip(obs["gates_pos"], obs["gates_quat"]):
            rot = R.from_quat(gq)
            up = rot.apply([0.0, 0.0, 1.0])  # gate vertical axis
            right = rot.apply([0.0, 1.0, 0.0])  # gate lateral axis
            # Stand: leg from the gate's bottom down to the ground (dummy if airborne).
            stand_h = float(gp[2]) - half
            if stand_h > 0:
                put(gp - up * half, gp - up * (half + stand_h), r_stand)
            else:
                put(far, far, r_stand)
            # 4 frame bars: top, bottom, left, right (each spans the outer width).
            put(gp + up * bar_dist - right * half, gp + up * bar_dist + right * half, r_bar)
            put(gp - up * bar_dist - right * half, gp - up * bar_dist + right * half, r_bar)
            put(gp - right * bar_dist + up * half, gp - right * bar_dist - up * half, r_bar)
            put(gp + right * bar_dist + up * half, gp + right * bar_dist - up * half, r_bar)
        return caps

    def _stage_thetas(self, theta0: float) -> np.ndarray:
        """Predict theta at each shooting node (warm start for the path relinearisation).

        The previous solution is shifted one step and re-anchored at the current projection.
        """
        if self._theta_pred is None:
            # First guess: constant progress speed over the (non-uniform) node times.
            return theta0 + self._node_t * self.V_TARGET
        th = np.empty(self._N + 1)
        th[:-1] = self._theta_pred[1:]
        th[-1] = self._theta_pred[-1] + self._time_steps[-1] * self.V_TARGET
        th[0] = theta0  # re-anchor to where the drone actually is
        return np.maximum.accumulate(th)  # keep monotonic

    def _gate_track_boosts(self, thetas: np.ndarray) -> np.ndarray:
        """Per-stage contouring (q_c) multiplier: GATE_TRACK_BOOST near a gate centre, else 1.0."""
        gt = getattr(self.planner, "gate_thetas", None)
        if not self.USE_GATE_TRACK_BOOST or gt is None:
            return np.ones(self._N + 1)
        gt = np.asarray(gt, dtype=float)
        dist = np.abs(thetas[:, None] - gt[None, :]).min(axis=1)
        return np.where(dist <= self.GATE_TRACK_RADIUS, self.GATE_TRACK_BOOST, 1.0)

    @staticmethod
    def _boosted_diag(w: np.ndarray, factor: float) -> np.ndarray:
        """Diagonal cost matrix from ``w`` with the contouring entries (q_c, diag 0..2) scaled."""
        if factor == 1.0:
            return np.diag(w)
        w = w.copy()
        w[:3] *= factor
        return np.diag(w)

    def _apply_stage_weights(
        self, thetas: np.ndarray, w_stage: np.ndarray, w_term: np.ndarray, rl_active: bool
    ) -> None:
        """Push the per-stage cost weights to the solver, with the near-gate contouring boost.

        ``w_stage``/``w_term`` are the active baseline (or RL-scaled) diagonals. Nodes whose
        progress θ sits within ``GATE_TRACK_RADIUS`` of a gate centre get their q_c entries scaled
        by ``GATE_TRACK_BOOST`` (see :data:`USE_GATE_TRACK_BOOST`). When RL is on, W is rebuilt for
        every node each tick anyway, so all nodes are set. With RL off the base W is static in the
        solver, so only nodes whose boost CHANGED since last tick are touched (straights stay
        free); ``self._boost_applied`` tracks the live per-node factor.
        """
        boosts = self._gate_track_boosts(thetas)
        if rl_active:
            for k in range(self._N):
                self._solver.cost_set(k, "W", self._boosted_diag(w_stage, boosts[k]))
            self._solver.cost_set(self._N, "W", self._boosted_diag(w_term, boosts[self._N]))
            return
        for k in range(self._N + 1):
            if boosts[k] == self._boost_applied.get(k, 1.0):
                continue
            base = w_stage if k < self._N else w_term
            self._solver.cost_set(k, "W", self._boosted_diag(base, boosts[k]))
            if boosts[k] == 1.0:
                self._boost_applied.pop(k, None)
            else:
                self._boost_applied[k] = float(boosts[k])

    def _shift_warm_start(self) -> None:
        """Receding-horizon primal warm start.

        Seed the solver with the *previous* solution shifted one node forward (node k ← old
        node k+1, last node/input repeated). Because the horizon advances by one control step
        each tick, old node k+1 is the best guess for new node k. acados otherwise keeps the
        previous solution *unshifted*; the explicit shift lands the SQP much closer to the
        optimum, so it converges in a couple of iterations instead of rebuilding the
        trajectory — the bulk of the compute saving. Node 0's state guess is irrelevant
        (pinned by lbx=ubx=x0 below).
        """
        if self._x_pred is None or self._u_pred is None:
            return
        xg, ug = self._x_pred, self._u_pred
        for k in range(self._N):
            self._solver.set(k, "x", xg[k + 1])
            self._solver.set(k, "u", ug[min(k + 1, self._N - 1)])
        self._solver.set(self._N, "x", xg[self._N])

    def _caution_factor(self, obs: dict) -> float:
        """Speed scale in [CAUTION_SPEED_FACTOR, 1] based on nearby UNMEASURED objects.

        An object whose pose is still uncertain (not yet ``*_visited``) and that lies within
        ``CAUTION_RADIUS`` of the drone pulls the factor down — fully to ``CAUTION_SPEED_FACTOR``
        when the drone is right on top of it, linearly back to 1 at the radius. The gate
        currently being approached counts as well as any unmeasured obstacle, so the drone slows
        into objects whose true position it doesn't know yet and resumes cruise once they are
        measured. Returns 1 (no slowdown) when caution is off or everything nearby is measured.
        """
        if not self.USE_CAUTION:
            return 1.0
        pos = obs["pos"]
        uncertain = []
        tg = int(obs["target_gate"])
        # The gate we're flying at stays a caution source until its true pose is measured AND the
        # replan onto that corrected pose has been installed (no replan still in flight). Keying
        # only on "not visited" let the drone snap back to full speed the instant the pose was
        # revealed (at the sensor range, ~0.7 m) while the corrected path was still being computed
        # in the background — so it committed at cruise speed to the stale nominal line and clipped
        # the frame. Staying slow until the replan lands gives it room to react to the new path.
        if tg >= 0:  # the gate we're flying at
            gate_unknown = not bool(obs["gates_visited"][tg]) or self._replan_future is not None
            if gate_unknown:
                uncertain.append(obs["gates_pos"][tg])
        for i, op in enumerate(obs["obstacles_pos"]):
            if not bool(obs["obstacles_visited"][i]):
                uncertain.append(op)
        if not uncertain:
            return 1.0
        # Measure distance in the xy-plane only, matching how the env decides "visited"
        # (race_core: ||drone_xy - object_xy|| < sensor_range). So CAUTION_RADIUS is directly
        # comparable to sensor_range and MUST exceed it, or the pose snaps to exact before the
        # slowdown ever engages.
        dmin = min(float(np.linalg.norm(np.asarray(u)[:2] - pos[:2])) for u in uncertain)
        # Bottom the ramp out at the sensor range, not at distance 0 (which a gate is never
        # reached — the drone flies through it). So the slowdown is already FULL by the time the
        # true pose is revealed and the replan onto it kicks in: frac = 1 at CAUTION_RADIUS,
        # falling to 0 (full caution) once dmin <= sensor_range. This is what gives a blind
        # approach room to react to the corrected path instead of committing to it at speed.
        floor = self._sensor_range
        denom = max(self.CAUTION_RADIUS - floor, 1e-3)
        frac = np.clip((dmin - floor) / denom, 0.0, 1.0)
        return float(self.CAUTION_SPEED_FACTOR + (1.0 - self.CAUTION_SPEED_FACTOR) * frac)

    def _gate_progress_cap(self, obs: dict) -> float:
        """Upper bound on progress θ imposed by the env's gate-pass state (gate barrier).

        θ may not advance past ``gate_thetas[target_gate] + GATE_PASS_LEAD`` until the env
        confirms the current gate was flown through (target_gate increments). Returns the full
        path length (no cap) when the barrier is off, all gates are passed (target_gate == -1),
        or the gate arc-lengths aren't available yet.
        """
        gt = getattr(self.planner, "gate_thetas", None)
        if not self.USE_GATE_BARRIER or gt is None:
            return self.planner.length
        tg = int(obs["target_gate"])
        if tg < 0 or tg >= len(gt):
            return self.planner.length
        return float(min(gt[tg] + self.GATE_PASS_LEAD, self.planner.length))

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Project onto the path, set per-stage contouring params, solve MPCC."""
        self._last_obs = obs
        self._maybe_replan(obs)

        # Missed-gate recovery: if the drone has stalled at an unpassed gate, fly it back for
        # another attempt (the monotonic MPCC progress can't reverse on its own). Returns a
        # command while recovering, else None → fall through to the normal MPCC solve.
        retry_cmd = self._handle_gate_retry(obs)
        if retry_cmd is not None:
            self._last_u = retry_cmd[:4]
            return retry_cmd

        tic = time.perf_counter()

        # Progress state θ / v_theta. θ is a genuine progress STATE that evolves inside the
        # solve (MPCC++ eq. 6) — it must NOT be re-pinned to the geometric projection every
        # tick, or its lag dynamics collapse and the cost degenerates back to point-tracking.
        # So on a NORMAL tick we INHERIT θ / v_theta from the previous solution at node 1
        # (the node that becomes "now" once the horizon advances one control step). We
        # re-localise by projecting ONLY when there is no valid prediction on the current
        # path: the first tick, an episode reset, or right after a replan installs a new path
        # — all signalled by _theta_pred having been cleared. The projection is window-
        # constrained around the last θ so it can never snap onto an earlier/later branch.
        relocalize = self._theta_pred is None or self._x_pred is None
        self._ticks_since_reloc = 0 if relocalize else self._ticks_since_reloc + 1
        if relocalize:
            theta0 = self.planner.project_to_theta(
                obs["pos"], obs["vel"], theta_prev=self._theta_est
            )
            p0, t0 = self.planner.path_point_tangent(theta0)
            vtheta0 = float(np.clip(np.dot(obs["vel"], t0), 0.0, self.VTHETA_MAX))
        else:
            theta0 = float(self._theta_pred[1])
            p0, _ = self.planner.path_point_tangent(theta0)
            vtheta0 = float(np.clip(self._x_pred[1, 13], 0.0, self.VTHETA_MAX))
        self._theta_est = theta0  # window centre for the next re-localisation projection
        self._progress_point = p0  # the single "where is the drone along the path" marker

        # Initial state (augmented).
        rpy = R.from_quat(obs["quat"]).as_euler("xyz")
        drpy = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((obs["pos"], rpy, obs["vel"], drpy, [theta0, vtheta0]))
        # Receding-horizon warm start: reuse the previous solve's trajectory, shifted one
        # node forward, as the initial guess (so SQP converges in a few iterations).
        self._shift_warm_start()
        self._solver.set(0, "lbx", x0)
        self._solver.set(0, "ubx", x0)
        t_proj = time.perf_counter()

        # Per-stage path: the LOCAL CUBIC of the path around each node's predicted theta, so
        # acados can evaluate p_d(theta) symbolically. Plus the avoidance capsules (same for
        # every node), built once from the live gate/obstacle poses.
        # End-hover: once the drone's progress reaches the end of the path, freeze every
        # stage's reference at theta=length and drop the target progress speed to 0. The
        # contouring+lag cost then pulls the drone onto the final path point p_d(length) and
        # the progress reward drives v_theta -> 0, so it station-keeps on the last trajectory
        # point instead of overrunning the path end.
        # Gate barrier with a smooth approach brake. The per-stage θ anchors are clamped at the
        # barrier so the horizon never references a gate we haven't passed, and v_target is ramped
        # linearly to 0 over the last GATE_PASS_LEAD metres before it. The brake is what keeps the
        # predicted progress from overshooting the clamped anchors: without it the path cubic gets
        # evaluated far outside its segment and the terminal prediction (the blue marker) shoots
        # off the trajectory. On a clean pass target_gate increments around the gate centre, so the
        # barrier jumps ahead before the brake really bites and cruise speed is barely affected; it
        # only slows things when a pass is actually failing. End-of-path hover always wins.
        finishing = theta0 >= self.planner.length - self.END_HOVER_TOL
        theta_cap = self._gate_progress_cap(obs)
        if finishing:
            v_target = 0.0
            thetas = np.full(self._N + 1, self.planner.length)
        else:
            brake = float(np.clip((theta_cap - theta0) / self.GATE_PASS_LEAD, 0.0, 1.0))
            v_cruise = self._curvature_speed_cap(theta0)  # slow into sharp turns, full on straights
            v_target = v_cruise * self._caution_factor(obs) * brake
            thetas = np.minimum(self._stage_thetas(theta0), theta_cap)
        caps = self._build_capsule_params(obs) if self._n_caps else None
        for k in range(self._N + 1):
            theta_i, cx, cy, cz = self._path_segment_coeffs(thetas[k])
            head = np.concatenate(([theta_i], cx, cy, cz, [v_target]))
            p_k = head if caps is None else np.concatenate((head, caps))
            self._solver.set(k, "p", p_k)

        # Per-tick cost weights = baseline × multipliers. Multipliers come from the RL weight
        # planner (trainer override or internal policy) when it's on, else identity (an exact
        # no-op: weight_diagonals rebuilds the same W layout the solver was built with). The
        # near-gate q_c boost (_apply_stage_weights) then tightens contouring around each gate; on
        # the no-RL path it only touches the few nodes near a gate, so straights stay as cheap as
        # before. acados updates W in place.
        if self.USE_RL_WEIGHTS:
            if self._external_mult is not None:
                mult = self._external_mult.copy()
            else:
                self._last_features = wp.build_features(obs, self.planner, self._theta_est)
                mult = self._weight_policy.multipliers(self._last_features).copy()
            self._last_mult = mult  # remember the applied multipliers (render overlay / logging)
            w_stage, w_term = wp.weight_diagonals(
                mult, BASELINE_WEIGHTS, self._n_caps, self.CAPSULE_PENALTY
            )
            # Append the ground-clearance weight — same layout the solver was built with.
            w_stage = np.append(w_stage, self.GROUND_PENALTY)
            w_term = np.append(w_term, self.GROUND_PENALTY)
        else:
            w_stage, w_term = self._w_stage_base, self._w_term_base
        self._apply_stage_weights(thetas, w_stage, w_term, self.USE_RL_WEIGHTS)
        t_setp = time.perf_counter()

        # SQP_RTI solve: phase 1 prepares (linearise + condense the QP at the warm-started
        # iterate), phase 2 solves the QP and applies the feedback step. Normally one iteration
        # per tick; on a fresh path (replan just installed / first tick / reset) the warm start
        # is far from the new optimum, so run a short burst to re-converge within this tick
        # instead of lagging the jump over the next several ticks.
        n_iters = self.RTI_BUMP_ITERS if relocalize else 1
        for _ in range(n_iters):
            self._solver.options_set("rti_phase", 1)
            self._solver.solve()
            self._solver.options_set("rti_phase", 2)
            status = self._solver.solve()
        if self.PROFILE:
            self._profile_tick(t_proj - tic, t_setp - t_proj, time.perf_counter() - t_setp)
        if self.DEBUG_SPIKE:
            speed = float(np.linalg.norm(obs["vel"]))
            if speed > self.DEBUG_SPIKE_FACTOR * max(v_target, 1e-3):
                print(
                    f"[MPCC spike] speed={speed:.2f} v_target={v_target:.2f} "
                    f"vtheta0={vtheta0:.2f} theta0={theta0:.2f} gate={int(obs['target_gate'])} "
                    f"reloc={relocalize} dt_reloc={self._ticks_since_reloc} status={status}"
                )
        self.last_solve_ok = status == 0
        if status == 0:
            xs = np.array([self._solver.get(k, "x") for k in range(self._N + 1)])
            us = np.array([self._solver.get(k, "u") for k in range(self._N)])
            self._theta_pred = xs[:, 12]
            self._pos_pred = xs[:, 0:3]  # warm start for the next tick's keep-out linearisation
            self._x_pred = xs  # full primal trajectory → shifted warm start next tick
            self._u_pred = us
            self._last_u = self._solver.get(0, "u")[:4]  # drop a_theta
            self._consec_fail = 0
            return self._last_u
        self._consec_fail += 1
        if self._last_u is not None and self._consec_fail <= self.MAX_HOLD_TICKS:
            return self._last_u
        print(f"[MPCC] solve failed (status={status}) at theta={theta0:.2f}; braking to hover")
        return self._safe_fallback_cmd(obs)

    def _profile_tick(self, proj: float, setp: float, solve: float) -> None:
        """Accumulate per-tick timings and print a rolling average every 50 ticks."""
        p = self._prof
        p["proj"] += proj
        p["setp"] += setp
        p["solve"] += solve
        p["n"] += 1
        if p["n"] >= 50:
            n = p["n"]
            tot = (p["proj"] + p["setp"] + p["solve"]) / n * 1e3
            print(
                f"[MPCC profile] avg/tick: project {p['proj'] / n * 1e3:.2f} ms | "
                f"set-params {p['setp'] / n * 1e3:.2f} ms | solve {p['solve'] / n * 1e3:.2f} ms"
                f" | total {tot:.2f} ms (budget {self._dt * 1e3:.0f} ms)"
            )
            self._prof = {"proj": 0.0, "setp": 0.0, "solve": 0.0, "n": 0}

    def _safe_fallback_cmd(self, obs: dict) -> NDArray[np.floating]:
        """Stabilising attitude command for when the solver can't be trusted (Befund 2).

        The old fallback was a zero-attitude hover: level, so it could not arrest the horizontal
        momentum the drone carried in at cruise speed — it coasted ballistically into a gate
        frame or out of the arena (the out_of_bounds / frame-streifer cases). Instead, tilt to
        brake: build a desired specific-force vector that damps the measured velocity and
        compensates gravity, turn it into a roll/pitch/yaw attitude (geometric reconstruction,
        round-tripped through the same scipy 'xyz' Euler convention used to read the state) plus a
        matching collective thrust. With zero velocity this reduces exactly to a heading-aligned
        hover; with velocity it decelerates and holds altitude.
        """
        g = float(-self.drone_params["gravity_vec"][-1])
        vel = np.asarray(obs["vel"], dtype=float)
        k_xy, k_z = 2.0, 1.5  # velocity-damping gains (horizontal / vertical)
        f = np.array([-k_xy * vel[0], -k_xy * vel[1], g - k_z * vel[2]])
        psi = float(R.from_quat(obs["quat"]).as_euler("xyz")[2])  # hold current heading
        return self._force_to_cmd(f, psi)

    def _force_to_cmd(self, f: np.ndarray, psi: float) -> NDArray[np.floating]:
        """Turn a desired world-frame specific force ``f`` into a ``[roll, pitch, yaw, thrust]``.

        Geometric attitude reconstruction (desired body-z = f/|f|, heading ``psi``), round-
        tripped through the same scipy 'xyz' Euler convention used to read the state, with
        thrust = mass*|f|. Shared by the brake fallback and the gate-retry recovery. ``f[2]`` is
        floored so the command never tilts past vertical or asks for near-zero thrust.
        """
        g = float(-self.drone_params["gravity_vec"][-1])
        mass = float(self.drone_params["mass"])
        f = np.asarray(f, dtype=float).copy()
        f[2] = max(f[2], 0.5 * g)
        zb = f / np.linalg.norm(f)
        x_c = np.array([np.cos(psi), np.sin(psi), 0.0])
        y_b = np.cross(zb, x_c)
        y_b /= np.linalg.norm(y_b)
        x_b = np.cross(y_b, zb)
        roll, pitch, _ = R.from_matrix(np.column_stack((x_b, y_b, zb))).as_euler("xyz")
        thrust = float(
            np.clip(
                mass * float(np.linalg.norm(f)),
                self.drone_params["thrust_min"] * 4,
                self.drone_params["thrust_max"] * 4,
            )
        )
        roll = float(np.clip(roll, -0.7, 0.7))
        pitch = float(np.clip(pitch, -0.7, 0.7))
        return np.array([roll, pitch, psi, thrust])

    def _recovery_cmd(self, obs: dict) -> NDArray[np.floating]:
        """PD position command flying the drone back to ``self._retry_target`` (gate retry)."""
        g = float(-self.drone_params["gravity_vec"][-1])
        pos = np.asarray(obs["pos"], dtype=float)
        vel = np.asarray(obs["vel"], dtype=float)
        kp, kd = 4.0, 3.0  # position / velocity-damping gains
        a = np.clip(kp * (self._retry_target - pos) - kd * vel, -8.0, 8.0)  # desired accel
        f = np.array([a[0], a[1], g + a[2]])
        psi = float(R.from_quat(obs["quat"]).as_euler("xyz")[2])
        return self._force_to_cmd(f, psi)

    def _handle_gate_retry(self, obs: dict) -> NDArray[np.floating] | None:
        """Detect a missed-gate stall and run the retreat-and-retry recovery.

        Returns a command while recovering (and on the tick a retry is triggered), or ``None`` to
        let the normal MPCC solve run. See USE_GATE_RETRY for the rationale.
        """
        tg = int(obs["target_gate"])
        gt = getattr(self.planner, "gate_thetas", None)
        if (
            not self.USE_GATE_RETRY
            or gt is None
            or tg < 0
            or tg >= len(gt)
            or self._theta_est is None
        ):
            self._stuck_ticks = 0
            self._retrying = False
            return None
        if tg != self._retry_gate:  # new (or advanced) target gate → reset the per-gate retry state
            self._retry_gate = tg
            self._retry_count = 0
            self._stuck_ticks = 0
            self._retrying = False

        if self._retrying:
            if float(np.linalg.norm(self._retry_target - np.asarray(obs["pos"], float))) < (
                self.RETRY_DONE_DIST
            ):
                # Back before the gate: hand control to the MPCC, re-localised at the retreat
                # point so it re-approaches and re-threads the gate.
                self._retrying = False
                self._stuck_ticks = 0
                self._theta_pred = None
                self._pos_pred = None
                self._x_pred = None
                self._u_pred = None
                self._theta_est = self._retry_theta
                return None
            return self._recovery_cmd(obs)

        # Not yet recovering: count consecutive stalled ticks at/past the unpassed gate.
        speed = float(np.linalg.norm(np.asarray(obs["vel"], float)))
        stalled = self._theta_est >= float(gt[tg]) - 0.1 and speed < self.RETRY_STUCK_SPEED
        self._stuck_ticks = self._stuck_ticks + 1 if stalled else 0
        if self._stuck_ticks >= self.RETRY_STUCK_TICKS and self._retry_count < self.RETRY_MAX:
            self._retry_count += 1
            self._retry_theta = max(0.0, float(gt[tg]) - self.RETRY_BACK_DIST)
            self._retry_target = self.planner.path_point_tangent(self._retry_theta)[0]
            self._retrying = True
            print(
                f"[MPCC] gate {tg} missed (try {self._retry_count}/{self.RETRY_MAX}) — retreating"
            )
            return self._recovery_cmd(obs)
        return None

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Mark the run finished once all gates are passed (``target_gate == -1``)."""
        if int(obs["target_gate"]) == -1:
            self._finished = True
        return self._finished

    # ── Visualisation (mirrors the MPC controller's overlay) ──────────────────

    def render_callback(self, sim: object) -> None:
        """Draw the planned path, gates, obstacles and progress markers.

        Gates (yellow + cyan/white frames), obstacles (orange poles), the planned path (green
        line) with the gate waypoints (magenta), a red marker at the drone's current progress
        along the path (its projection point θ), and a single blue marker at the far end of
        the MPCC horizon (how far ahead the controller plans).
        """
        obs = self._last_obs
        if obs is not None:
            draw_points(
                sim, np.atleast_2d(obs["gates_pos"]), rgba=np.array([1.0, 1.0, 0.0, 1.0]), size=0.08
            )
            for gpos, gquat in zip(obs["gates_pos"], obs["gates_quat"]):
                self._draw_square(
                    sim, gpos, gquat, _GateFrame.OUTER / 2, np.array([0.0, 1.0, 1.0, 1.0])
                )
                self._draw_square(
                    sim, gpos, gquat, _GateFrame.OPENING / 2, np.array([1.0, 1.0, 1.0, 1.0])
                )
            for opos in np.atleast_2d(obs["obstacles_pos"]):
                pole = np.array([[opos[0], opos[1], 0.0], [opos[0], opos[1], opos[2]]])
                draw_line(
                    sim, pole, rgba=np.array([1.0, 0.5, 0.0, 1.0]), start_size=6.0, end_size=6.0
                )

        # Single red marker: the drone's current progress θ projected onto the path.
        if self._progress_point is not None:
            draw_points(
                sim,
                np.atleast_2d(self._progress_point),
                rgba=np.array([1.0, 0.0, 0.0, 1.0]),
                size=0.07,
            )

        # Single blue marker: the far end of the MPC horizon (last predicted stage position)
        # — i.e. how far ahead the controller is currently planning.
        if self._pos_pred is not None:
            draw_points(
                sim,
                np.atleast_2d(self._pos_pred[-1]),
                rgba=np.array([0.0, 0.0, 1.0, 1.0]),
                size=0.07,
            )

        # Fixed gate waypoints (entry / center / exit), magenta.
        raw = getattr(self.planner, "_raw_waypoints", None)
        if raw:
            fixed = np.array([np.asarray(p) for p, v in raw if v is not None])
            if len(fixed):
                draw_points(sim, fixed, rgba=np.array([1.0, 0.0, 1.0, 1.0]), size=0.05)

        # Planned path (green line). Downsample to ~200 segments — enough to render the
        # B-spline smoothly (20 made it look visibly faceted) while staying light.
        traj = getattr(self.planner, "pos", None)
        if traj is not None:
            step = max(1, len(traj) // 200)
            draw_line(sim, traj[::step], rgba=np.array([0.0, 1.0, 0.0, 1.0]))

        # Live MPCC cost weights, top-right text box. Without RL these stay at the baseline every
        # tick; with RL they change as the policy rescales them.
        self._draw_weight_overlay(sim)

    def _current_weights(self) -> list[tuple[str, float]]:
        """Current MPCC cost scalars = baseline × the last applied multipliers (1 ⇒ baseline)."""
        mult = self._last_mult if self._last_mult is not None else np.ones(wp.N_ACTIONS)
        m = dict(zip(wp.WEIGHT_GROUPS, np.asarray(mult, dtype=float)))
        b = BASELINE_WEIGHTS
        return [
            ("q_c", b["q_c"] * m["q_c"]),
            ("q_l", b["q_l"] * m["q_l"]),
            ("q_att", b["q_att"] * m["q_track"]),
            ("q_dr", b["q_dr"] * m["q_track"]),
            ("q_v", b["q_v"] * m["q_v"]),
            ("r_rpy", b["r_rpy"] * m["r_ctrl"]),
            ("r_T", b["r_T"] * m["r_thrust"]),
            ("r_at", b["r_at"] * m["r_ctrl"]),
        ]

    def _draw_weight_overlay(self, sim: object) -> None:
        """Draw the current cost weights as a 2D text box in the viewer's top-right corner.

        Uses the MuJoCo viewer's ``add_overlay`` (the same per-frame contract as the markers:
        it must be re-added before every ``render``). No-ops cleanly when there is no on-screen
        viewer (e.g. headless / rgb_array without an overlay-capable viewer).
        """
        viewer = getattr(getattr(sim, "viewer", None), "viewer", None)
        if viewer is None or not hasattr(viewer, "add_overlay"):
            return
        try:
            import mujoco

            pos = mujoco.mjtGridPos.mjGRID_TOPRIGHT
            header = "RL weights ON" if self.USE_RL_WEIGHTS else "baseline (RL off)"
            viewer.add_overlay(pos, "MPCC cost weights", header)
            for name, val in self._current_weights():
                viewer.add_overlay(pos, name, f"{val:.2f}")
        except Exception as exc:  # never let an overlay quirk break the sim
            print(f"[MPCC] weight overlay skipped: {exc!r}")

    @staticmethod
    def _draw_square(
        sim: object, center: np.ndarray, quat: np.ndarray, half: float, rgba: np.ndarray
    ) -> None:
        """Draw an oriented square outline in the gate plane (local y-z plane)."""
        rot = R.from_quat(quat)
        local = np.array(
            [
                [0.0, half, half],
                [0.0, -half, half],
                [0.0, -half, -half],
                [0.0, half, -half],
                [0.0, half, half],
            ]
        )
        draw_line(sim, center + rot.apply(local), rgba=rgba)

    def episode_callback(self) -> None:
        """Clear the warm-start caches and cancel any in-flight replan between episodes."""
        self._theta_pred = None
        self._pos_pred = None
        self._x_pred = None
        self._u_pred = None
        self._theta_est = None  # re-project globally at the next episode's start
        if self._replan_future is not None:
            self._replan_future.cancel()
            self._replan_future = None

    def episode_reset(self) -> None:
        """Reset the finished flag and clear per-episode state."""
        self._finished = False
        self.episode_callback()
