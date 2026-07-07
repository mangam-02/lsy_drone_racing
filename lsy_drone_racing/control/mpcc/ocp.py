"""acados OCP construction for the MPCC controller.

Pure construction layer, no runtime state: the progress-augmented drone model, the
NONLINEAR_LS contouring/lag/progress cost (with capsule soft-barriers and the
ground-clearance penalty), the softened state/input box constraints, the SQP_RTI/HPIPM
solver options (with the optional non-uniform time grid), and a source-hash cache that
reuses the compiled solver across runs. The controller (:mod:`.controller`) calls
:func:`create_mpcc_ocp_solver` once at startup and interacts with the returned solver only
through per-stage parameters, cost weights and warm starts.

"MPCC++" citations refer to: Krinner et al., "MPCC++: Model Predictive Contouring Control
for Time-Optimal Flight with Safety Constraints", RSS 2024.
"""

from __future__ import annotations

import glob
import hashlib
import inspect
import json
import os

import casadi as ca
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from drone_models.so_rpy import symbolic_dynamics_euler

from lsy_drone_racing.control.mpcc import weight_policy as wp
from lsy_drone_racing.control.mpcc.config import BASELINE_WEIGHTS

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
    # STATE theta via the local cubic — so e_c/e_l genuinely depend on theta (MPCC++ eq. (5)).
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

    # Nonlinear least-squares residual (nonlinear in theta; acados' Gauss-Newton SQP
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
    # to the drone longitudinally). Because theta is a free inherited STATE, q_l is
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
    # Slack costs for the softened velocity / v_theta bounds. Obstacle/gate avoidance is a
    # cost barrier (see _build_mpcc_cost), not a constraint, so no keep-out slacks are needed.
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
    # Real-time iteration: ONE SQP step per tick (prepare + feedback). Iterating a full SQP
    # to 1e-6 takes up to ~50 re-linearisations per tick, far over the ~20 ms control budget;
    # the shifted warm start lands each tick near the optimum, so a single iteration tracks
    # well.
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.tol = 1e-3
    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1
    # HPIPM QP iteration cap. Near gates the QP needs the most iterations; a tight cap makes it
    # hit the limit there (solve fails → brake to hover), so give it headroom to converge.
    # Note: solver options are compiled into the generated C code, not read at runtime. The
    # cached solver is still safe to edit against: _solver_signature hashes this function's
    # source, so changing the value forces a rebuild with the new setting.
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
        # 1.0 dwarfs any single dt-scaled stage and rivals all stages combined, so the cost is
        # dominated by the last node — the farthest point of the horizon. The drone then
        # optimises mostly to land that far node on the path and chases it, cutting corners.
        # Instead, weight *every* node, terminal included, equally by the real control period
        # dt (= ts[0]). No node dominates → the whole horizon is tracked uniformly and the
        # near (executed) nodes get their fair share. (Equals the uniform default when
        # HORIZON_GROWTH == 1.0, except the terminal is dt-weighted too rather than 1.0.)
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
