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
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.mpcc import config as cfg
from lsy_drone_racing.control.mpcc import weight_policy as wp
from lsy_drone_racing.control.mpcc.config import BASELINE_WEIGHTS
from lsy_drone_racing.control.mpcc.planner import SimplePlanner, _GateFrame
from lsy_drone_racing.control.mpcc.weight_policy import WeightPolicy

if TYPE_CHECKING:
    from numpy.typing import NDArray

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
        for fn in (
            create_mpcc_model,
            _capsule_barrier,
            _build_mpcc_cost,
            _build_mpcc_constraints,
            _set_mpcc_solver_options,
            _maybe_reuse_solver,
            create_mpcc_ocp_solver,
        )
    )
    blob = json.dumps(
        {"args": _jsonable(payload), "src": hashlib.sha256(src.encode()).hexdigest()},
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def _build_mpcc_cost(
    ocp: AcadosOcp,
    parameters: dict,
    n_caps: int,
    capsule_penalty: float,
    ground_soft_z: float,
    ground_penalty: float,
) -> None:
    """Set up the NONLINEAR_LS contouring/lag/progress cost on ``ocp`` (mutates it in place).

    Declares the per-stage parameter vector ``p`` (path cubic head + avoidance capsules),
    builds the embedded-path residuals (contouring ``e_c``, lag ``e_l``, progress ``e_v``), the
    capsule soft-barrier and the ground-clearance residual, and installs the diagonal weight
    matrices ``W`` / ``W_e`` from :data:`BASELINE_WEIGHTS`.
    """
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


def _build_mpcc_constraints(
    ocp: AcadosOcp,
    parameters: dict,
    z_min: float,
    z_max: float,
    v_max: float,
    vtheta_max: float,
    atheta_max: float,
) -> None:
    """Install the state/input box constraints and the softened velocity slacks on ``ocp``."""
    nx = ocp.model.x.rows()  # 14

    # State bounds: z floor/ceiling (2), rpy (3,4,5), vel (6,7,8), v_theta (13: 0..vtheta).
    ocp.constraints.lbx = np.array([z_min, -0.6, -0.6, -0.6, -v_max, -v_max, -v_max, 0.0])
    ocp.constraints.ubx = np.array([z_max, 0.6, 0.6, 0.6, v_max, v_max, v_max, vtheta_max])
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
    ocp.constraints.lbu = np.array([-0.6, -0.6, -0.6, parameters["thrust_min"] * 4, -atheta_max])
    ocp.constraints.ubu = np.array([0.6, 0.6, 0.6, parameters["thrust_max"] * 4, atheta_max])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])

    ocp.constraints.x0 = np.zeros(nx)


def _set_mpcc_solver_options(
    ocp: AcadosOcp, N: int, Tf: float, time_steps: np.ndarray | None
) -> None:
    """Configure the SQP_RTI / HPIPM solver options and the (optional) non-uniform time grid."""
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


def _maybe_reuse_solver(ocp: AcadosOcp, signature: str, verbose: bool) -> AcadosOcpSolver:
    """Build the acados solver, reusing the cached C code when its signature is unchanged.

    Generating + compiling the OCP costs many seconds at every start; it only needs redoing when
    the OCP structure changes (N, n_caps, the drone model, the time grid, the bounds/penalties)
    or the builder source is edited — all captured by ``signature`` (see :func:`_solver_signature`).
    """
    gen_dir = "c_generated_code"
    json_file = os.path.join(gen_dir, "mpcc_planner.json")
    sig_file = os.path.join(gen_dir, "mpcc_planner.sig")
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
    return solver


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

    The OCP is assembled by the dedicated builders :func:`_build_mpcc_cost`,
    :func:`_build_mpcc_constraints` and :func:`_set_mpcc_solver_options`; the compiled solver is
    cached and reused by :func:`_maybe_reuse_solver`.
    """
    ocp = AcadosOcp()
    ocp.model = create_mpcc_model(parameters)
    ocp.solver_options.N_horizon = N

    _build_mpcc_cost(ocp, parameters, n_caps, capsule_penalty, ground_soft_z, ground_penalty)
    _build_mpcc_constraints(ocp, parameters, z_min, z_max, v_max, vtheta_max, atheta_max)
    _set_mpcc_solver_options(ocp, N, Tf, time_steps)

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
            # Baseline weights are a module constant, not in any builder's source, so the source
            # hash would miss a baseline edit; include them here to keep the cache honest.
            "weight_baseline": BASELINE_WEIGHTS,
        }
    )
    solver = _maybe_reuse_solver(ocp, signature, verbose)
    return solver, ocp


# ── Controller ──────────────────────────────────────────────────────────────────


class MPCCController(Controller):
    """MPCC controller: contouring control along the warm-started SimplePlanner path."""

    # Tuning constants — values defined in mpcc/config.py (single source of truth).
    GROUND_Z = cfg.GROUND_Z
    GROUND_SOFT_Z = cfg.GROUND_SOFT_Z
    GROUND_PENALTY = cfg.GROUND_PENALTY
    V_TARGET = cfg.V_TARGET
    VTHETA_MAX = cfg.VTHETA_MAX
    USE_CURVATURE_SPEED = cfg.USE_CURVATURE_SPEED
    MAX_LAT_ACC = cfg.MAX_LAT_ACC
    CURVE_MIN_SPEED = cfg.CURVE_MIN_SPEED
    CURVE_LOOKAHEAD = cfg.CURVE_LOOKAHEAD
    CURVE_SMOOTH_M = cfg.CURVE_SMOOTH_M
    USE_GATE_TRACK_BOOST = cfg.USE_GATE_TRACK_BOOST
    GATE_TRACK_BOOST = cfg.GATE_TRACK_BOOST
    GATE_TRACK_RADIUS = cfg.GATE_TRACK_RADIUS
    END_HOVER_TOL = cfg.END_HOVER_TOL
    USE_CAUTION = cfg.USE_CAUTION
    CAUTION_SPEED_FACTOR = cfg.CAUTION_SPEED_FACTOR
    CAUTION_RADIUS = cfg.CAUTION_RADIUS
    USE_GATE_BARRIER = cfg.USE_GATE_BARRIER
    GATE_PASS_LEAD = cfg.GATE_PASS_LEAD
    GATE_BRAKE_FLOOR = cfg.GATE_BRAKE_FLOOR
    USE_GATE_RETRY = cfg.USE_GATE_RETRY
    RETRY_STUCK_SPEED = cfg.RETRY_STUCK_SPEED
    RETRY_STUCK_TICKS = cfg.RETRY_STUCK_TICKS
    RETRY_BACK_DIST = cfg.RETRY_BACK_DIST
    RETRY_DONE_DIST = cfg.RETRY_DONE_DIST
    RETRY_MAX = cfg.RETRY_MAX
    REPLAN_TOL = cfg.REPLAN_TOL
    MAX_HOLD_TICKS = cfg.MAX_HOLD_TICKS
    USE_PRED_SAFETY = cfg.USE_PRED_SAFETY
    MAX_PRED_DEV = cfg.MAX_PRED_DEV
    RTI_BUMP_ITERS = cfg.RTI_BUMP_ITERS
    HORIZON_GROWTH = cfg.HORIZON_GROWTH
    USE_AVOIDANCE = cfg.USE_AVOIDANCE
    CAPSULE_PENALTY = cfg.CAPSULE_PENALTY
    AVOID_MARGIN = cfg.AVOID_MARGIN
    GATE_OUTER = cfg.GATE_OUTER
    GATE_BAR_DIST = cfg.GATE_BAR_DIST
    GATE_BAR_RADIUS = cfg.GATE_BAR_RADIUS
    GATE_STAND_RADIUS = cfg.GATE_STAND_RADIUS
    POLE_RADIUS = cfg.POLE_RADIUS
    POLE_HEIGHT = cfg.POLE_HEIGHT
    PROFILE = cfg.PROFILE
    DEBUG_SPIKE = cfg.DEBUG_SPIKE
    DEBUG_SPIKE_FACTOR = cfg.DEBUG_SPIKE_FACTOR
    USE_RL_WEIGHTS = cfg.USE_RL_WEIGHTS
    RL_WEIGHT_CKPT = cfg.RL_WEIGHT_CKPT
    N_HORIZON = cfg.N_HORIZON

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

        # Near-gate contouring boost: re-base every stage's weight and clear the boost tracker so a
        # reused solver doesn't carry the previous episode's boosts (no-op under RL).
        self._rebase_stage_weights()

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
                # The primal warm start carries the OLD path's progress state θ in column 12 of
                # every node; on the new path those θ values are meaningless. Seeding the solver
                # with them (via _shift_warm_start) starts the SQP from an iterate whose θ-state and
                # the freshly set per-node path cubic disagree, so the prediction (the blue marker)
                # shoots off the line right after a replan — most visibly where the new path jumps
                # (e.g. a sudden climb). Clear them too so the next tick re-localises from a clean
                # warm start and the RTI_BUMP_ITERS burst re-converges within the tick.
                self._x_pred = None
                self._u_pred = None

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

    def _rebase_stage_weights(self) -> None:
        """Write the static baseline cost weight W back onto every node and clear the boost tracker.

        Used after a fresh start of the solver (episode reset, or ``solver.reset()`` on a
        re-localise), where the per-node W in the solver may be stale or cleared. The no-RL
        near-gate q_c boost path (:meth:`_apply_stage_weights`) updates W *incrementally* — it
        assumes the baseline already sits on every non-boosted node and tracks the live boosts in
        ``_boost_applied``. So after a reset we must (1) rewrite the baseline W onto every node and
        (2) clear ``_boost_applied``, or the tracker desyncs and non-boosted nodes are left without
        weights. No-op under RL, which rebuilds every node's W from scratch each tick anyway.
        """
        self._boost_applied: dict[int, float] = {}
        if self.USE_RL_WEIGHTS:
            return
        W_s, W_t = np.diag(self._w_stage_base), np.diag(self._w_term_base)
        for k in range(self._N):
            self._solver.cost_set(k, "W", W_s)
        self._solver.cost_set(self._N, "W", W_t)

    def _seed_warm_start(self, x0: np.ndarray) -> None:
        """Consistent cold-start primal guess: every node = the current augmented state ``x0``.

        Used on a re-localise (first tick / episode reset / right after a replan installs a new
        path), where there is no valid previous trajectory on the *current* path. After a replan
        the receding-horizon shift is skipped (``_x_pred`` was cleared), so acados would otherwise
        keep its internal iterate from the OLD path — whose progress state ``theta`` no longer
        matches the freshly swapped per-node path cubics. That primal/parameter mismatch is what
        drove HPIPM to NaN (QP status 3) right after a replan and sent the terminal prediction (the
        blue marker) off the line. Overwriting every node with the self-consistent ``x0`` keeps the
        QP well-conditioned; the ``RTI_BUMP_ITERS`` burst then rolls it out onto the new path.
        """
        u0 = np.array([0.0, 0.0, 0.0, self._hover_thrust, 0.0])
        for k in range(self._N):
            self._solver.set(k, "x", x0)
            self._solver.set(k, "u", u0)
        self._solver.set(self._N, "x", x0)

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
        """Project onto the path, set per-stage contouring params, solve MPCC.

        Runs one control tick as an ordered sequence of phases, each a dedicated helper so this
        method stays a readable outline: re-plan check → missed-gate retry → progress state →
        warm start → per-stage parameters → cost weights → SQP solve → commit-or-recover.
        """
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
        theta0, vtheta0, relocalize = self._progress_state(obs)

        # Initial state (augmented).
        rpy = R.from_quat(obs["quat"]).as_euler("xyz")
        drpy = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((obs["pos"], rpy, obs["vel"], drpy, [theta0, vtheta0]))
        self._apply_warm_start(x0, relocalize)
        t_proj = time.perf_counter()

        v_target, thetas = self._set_stage_parameters(obs, theta0)
        self._apply_tick_weights(obs, thetas)
        t_setp = time.perf_counter()

        status = self._run_sqp(relocalize)
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
        return self._commit_or_recover(status, obs, theta0)

    def _progress_state(self, obs: dict) -> tuple[float, float, bool]:
        """Initial progress state ``(theta0, v_theta0)`` for this tick, plus the re-localise flag.

        θ is a genuine progress STATE that evolves inside the solve (MPCC++ eq. 6) — it must NOT
        be re-pinned to the geometric projection every tick, or its lag dynamics collapse and the
        cost degenerates back to point-tracking. So on a NORMAL tick we INHERIT θ / v_theta from
        the previous solution at node 1 (the node that becomes "now" once the horizon advances one
        control step). We re-localise by projecting ONLY when there is no valid prediction on the
        current path: the first tick, an episode reset, or right after a replan installs a new path
        — all signalled by _theta_pred having been cleared. The projection is window-constrained
        around the last θ so it can never snap onto an earlier/later branch.
        """
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
        return theta0, vtheta0, relocalize

    def _apply_warm_start(self, x0: np.ndarray, relocalize: bool) -> None:
        """Seed the solver for this tick and pin node 0 to the measured state ``x0``.

        On a normal tick: reuse the previous solve's trajectory shifted one node forward (SQP
        converges in a few iterations). On a re-localise (first tick / reset / right after a
        replan): there is no valid previous trajectory on the current path, so seed a cold start
        (every node = x0) instead of inheriting acados' stale old-path iterate — the latter
        mismatches the freshly set path cubics and drove HPIPM to NaN.
        """
        if relocalize:
            # Wipe acados' internal iterate (primal AND dual / HPIPM QP memory) before seeding a
            # fresh cold start. _seed_warm_start only overwrites the primal node states, but the
            # stale dual variables from the old-path solve can still drive HPIPM to NaN after a
            # replan (the persistent solve failures → coast-into-wall). reset() also drops the
            # per-node cost weights, so _rebase_stage_weights writes the baseline W back and resets
            # the boost tracker (no-op under RL) — keeping it compatible with USE_GATE_TRACK_BOOST.
            self._solver.reset()
            self._rebase_stage_weights()
            self._seed_warm_start(x0)
        else:
            self._shift_warm_start()
        self._solver.set(0, "lbx", x0)
        self._solver.set(0, "ubx", x0)

    def _set_stage_parameters(self, obs: dict, theta0: float) -> tuple[float, np.ndarray]:
        """Write each node's path-cubic + avoidance params; return ``(v_target, thetas)``.

        Per-stage path: the LOCAL CUBIC of the path around each node's predicted theta, so acados
        can evaluate p_d(theta) symbolically. Plus the avoidance capsules (same for every node),
        built once from the live gate/obstacle poses.

        End-hover: once the drone's progress reaches the end of the path, freeze every stage's
        reference at theta=length and drop the target progress speed to 0. The contouring+lag cost
        then pulls the drone onto the final path point p_d(length) and the progress reward drives
        v_theta -> 0, so it station-keeps on the last trajectory point instead of overrunning the
        path end.

        Gate barrier with a smooth approach brake. The per-stage θ anchors are clamped at the
        barrier so the horizon never references a gate we haven't passed, and v_target is ramped
        linearly to 0 over the last GATE_PASS_LEAD metres before it. The brake is what keeps the
        predicted progress from overshooting the clamped anchors: without it the path cubic gets
        evaluated far outside its segment and the terminal prediction (the blue marker) shoots
        off the trajectory. On a clean pass target_gate increments around the gate centre, so the
        barrier jumps ahead before the brake really bites and cruise speed is barely affected; it
        only slows things when a pass is actually failing. End-of-path hover always wins.
        """
        finishing = theta0 >= self.planner.length - self.END_HOVER_TOL
        theta_cap = self._gate_progress_cap(obs)
        if finishing:
            v_target = 0.0
            thetas = np.full(self._N + 1, self.planner.length)
        else:
            # Brake on the gate barrier using the drone's TRUE progress (geometric projection of the
            # measured position), NOT the inherited progress state theta0. theta0 can lead the real
            # position (lag), so braking on it slowed the drone to a stop while it was still
            # physically SHORT of the gate plane — and since the env only registers a pass on a real
            # x<0 → x>0 crossing, it never confirmed and the barrier never lifted (the drone waited
            # at the gate). Gating the brake on where the drone actually is makes it bite only once
            # the body itself is past the gate centre (a genuine missed pass), not merely when the
            # progress state is. The floor (GATE_BRAKE_FLOOR) then keeps a small forward push so the
            # drone always creeps the last centimetres through the plane instead of parking short.
            theta_phys = self.planner.project_to_theta(
                obs["pos"], obs["vel"], theta_prev=self._theta_est
            )
            brake = float(np.clip((theta_cap - theta_phys) / self.GATE_PASS_LEAD, 0.0, 1.0))
            brake = max(brake, self.GATE_BRAKE_FLOOR)
            v_cruise = self._curvature_speed_cap(theta0)  # slow into sharp turns, full on straights
            v_target = v_cruise * self._caution_factor(obs) * brake
            thetas = np.minimum(self._stage_thetas(theta0), theta_cap)
        caps = self._build_capsule_params(obs) if self._n_caps else None
        for k in range(self._N + 1):
            theta_i, cx, cy, cz = self._path_segment_coeffs(thetas[k])
            head = np.concatenate(([theta_i], cx, cy, cz, [v_target]))
            p_k = head if caps is None else np.concatenate((head, caps))
            self._solver.set(k, "p", p_k)
        return v_target, thetas

    def _apply_tick_weights(self, obs: dict, thetas: np.ndarray) -> None:
        """Push this tick's cost weights (RL-scaled or baseline) + near-gate q_c boost.

        Per-tick cost weights = baseline × multipliers. Multipliers come from the RL weight planner
        (trainer override or internal policy) when it's on, else identity (an exact no-op:
        weight_diagonals rebuilds the same W layout the solver was built with). The near-gate q_c
        boost (_apply_stage_weights) then tightens contouring around each gate; on the no-RL path it
        only touches the few nodes near a gate, so straights stay as cheap as before. acados updates
        W in place.
        """
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

    def _run_sqp(self, relocalize: bool) -> int:
        """Run the SQP_RTI solve(s) for this tick and return the acados solver status.

        SQP_RTI solve: phase 1 prepares (linearise + condense the QP at the warm-started iterate),
        phase 2 solves the QP and applies the feedback step. Normally one iteration per tick; on a
        fresh path (replan just installed / first tick / reset) the warm start is far from the new
        optimum, so run a short burst to re-converge within this tick instead of lagging the jump
        over the next several ticks.
        """
        n_iters = self.RTI_BUMP_ITERS if relocalize else 1
        for _ in range(n_iters):
            self._solver.options_set("rti_phase", 1)
            self._solver.solve()
            self._solver.options_set("rti_phase", 2)
            status = self._solver.solve()
        return status

    def _commit_or_recover(self, status: int, obs: dict, theta0: float) -> NDArray[np.floating]:
        """Commit a trusted solve, else replay the last plan / brake. Returns the executed command.

        On a successful, non-divergent solve the trajectory is committed (warm-start caches updated)
        and node 0's input returned. A divergent or failed solve is not committed; the controller
        replays the next input from the last TRUSTED prediction, falling back to a braking command.
        """
        self.last_solve_ok = status == 0
        if status == 0:
            xs = np.array([self._solver.get(k, "x") for k in range(self._N + 1)])
            us = np.array([self._solver.get(k, "u") for k in range(self._N)])
            # Prediction-divergence safety net (real-hardware guard): if the predicted trajectory
            # has run far off the planned path (the blue marker flying away), DON'T commit it and
            # DON'T warm-start from it — clear the warm-start caches so the next tick re-localises
            # (solver.reset + reseed) and re-converges, and fall through to replay/brake. This stops
            # the controller ever flying a divergent solution into a gate frame. _u_pred is kept
            # as-is (the last TRUSTED plan) so the replay below continues that, not this solve.
            if self.USE_PRED_SAFETY and self._pred_deviation(xs) > self.MAX_PRED_DEV:
                self._theta_pred = None
                self._pos_pred = None
                self._x_pred = None
                self.last_solve_ok = False
                print(
                    f"[MPCC] prediction diverged (>{self.MAX_PRED_DEV:.2f} m) at "
                    f"theta={theta0:.2f}; rejecting solve and re-localising"
                )
            else:
                self._theta_pred = xs[:, 12]
                self._pos_pred = xs[:, 0:3]  # warm start for next tick's keep-out linearisation
                self._x_pred = xs  # full primal trajectory → shifted warm start next tick
                self._u_pred = us
                self._last_u = self._solver.get(0, "u")[:4]  # drop a_theta
                self._consec_fail = 0
                return self._last_u
        # Solve failed OR was rejected as divergent: instead of dead-holding the single last
        # command (which lets the drone coast off the line), replay the NEXT input from the last
        # TRUSTED MPCC prediction — node _consec_fail of that optimal trajectory, i.e. the step the
        # controller had already planned to take next. This keeps executing the intended path for
        # the few ticks the solver needs to recover, instead of hovering. Falls back to the braking
        # command when there is no usable prediction (e.g. a failure on the first tick after a
        # replan cleared _u_pred).
        self._consec_fail += 1
        if self._u_pred is not None and self._consec_fail <= self.MAX_HOLD_TICKS:
            idx = min(self._consec_fail, self._N - 1)
            self._last_u = self._u_pred[idx][:4]
            return self._last_u
        print(f"[MPCC] solve failed (status={status}) at theta={theta0:.2f}; braking to hover")
        return self._safe_fallback_cmd(obs)

    def _pred_deviation(self, xs: np.ndarray) -> float:
        """How far the predicted trajectory strays from the path BEYOND the drone's current offset.

        Per node, the distance to the path point at that node's own progress state θ (column 12).
        Node 0 is the drone's current (lbx-pinned) offset from the path, which may legitimately be
        large after a disturbance — a correct recovery solve STARTS there and converges back. So we
        return the *growth* beyond that start offset (``dev.max() - dev[0]``): ~0 (or negative) for
        a solve that tracks or flies home, large only when the prediction genuinely shoots off the
        path (the blue marker flying away). Returning absolute deviation instead would reject every
        solve whenever the drone is merely off-path — blocking the very recovery it needs (the
        reject loop that left it braking/sinking off-track). Vectorised (one path_point_tangent).
        """
        p_path, _ = self.planner.path_point_tangent(xs[:, 12])
        dev = np.linalg.norm(xs[:, 0:3] - p_path, axis=1)
        return float(dev.max() - dev[0])

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
        along the path (its projection point θ), and a blue line tracing the MPCC predicted
        horizon (with a dot at its far end — how far ahead the controller plans).
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

        # Blue line: the full MPC predicted horizon (all N+1 stage positions) — the trajectory
        # the controller is planning over the next ~0.5 s. The line shows the whole prediction
        # so its divergence from the green planned path is visible at a glance; the dot at the
        # far end marks how far ahead the controller is currently planning.
        if self._pos_pred is not None and len(self._pos_pred) > 1:
            draw_line(
                sim,
                self._pos_pred,
                rgba=np.array([0.0, 0.0, 1.0, 1.0]),
                start_size=0.008,
                end_size=0.008,
            )
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
