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
from lsy_drone_racing.control.mpc_planner_controller import (
    _CON_SLACK_LIN,
    _CON_SLACK_QUAD,
    _Cylinder,
    _GateFrame,
)

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


def create_mpcc_ocp_solver(
    Tf: float,
    N: int,
    parameters: dict,
    z_min: float = 0.0,
    z_max: float = 2.5,
    v_max: float = 2.0,
    vtheta_max: float = 2.5,
    atheta_max: float = 6.0,
    n_obstacles: int = 0,
    n_gates: int = 0,
    obs_clearance: float = 0.15,
    gate_drone_r: float = 0.09,
    time_steps: np.ndarray | None = None,
    verbose: bool = False,
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Build the acados OCP/solver for MPCC.

    NONLINEAR_LS cost on contouring/lag/progress; per-stage
    path point + tangent + target speed are model parameters set per solve. When
    ``n_obstacles``/``n_gates`` > 0 the same soft obstacle/gate keep-out as the tracking MPC
    is added (smooth ``con_h`` with slack-penalty cost), parametrised by the live poses.

    Parameter layout per stage: ``p = [theta_i(1), c_x(4), c_y(4), c_z(4), v_target(1),
    <one convex half-plane [a_x, a_y, b] per keep-out>]``. The 13-number cubic head defines
    ``p_d(theta) = c*(theta - theta_i)`` so the path point/tangent are functions of the
    progress state (re-linearised per node from the warm-started theta).
    """
    ocp = AcadosOcp()
    ocp.model = create_mpcc_model(parameters)
    nx = ocp.model.x.rows()  # 14
    ocp.solver_options.N_horizon = N

    hover_thrust = parameters["mass"] * -parameters["gravity_vec"][-1]

    # Per-stage parameters: contouring head [theta_i(1), c_x(4), c_y(4), c_z(4), v_target(1)]
    # = the LOCAL CUBIC of the path around this node's predicted theta, + one convex
    # half-plane [a_x, a_y, b] per keep-out (obstacle or non-target gate).
    n_keepout = n_obstacles + n_gates
    n_head = 14  # 1 (theta_i) + 12 (cubic coeffs) + 1 (v_target)
    n_p = n_head + 3 * n_keepout
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

    # Nonlinear least-squares residual (nonlinear in theta now; acados' Gauss-Newton SQP
    # re-linearises the path each iteration).
    y = ca.vertcat(e_c, e_l, rpy, drpy, e_v, rpy_cmd, thrust - hover_thrust, a_theta)
    y_e = ca.vertcat(e_c, e_l, rpy, drpy, e_v)
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
    q_c, q_l, q_att, q_dr, q_v = 50.0, 150.0, 1.0, 5.0, 5.0
    r_rpy, r_T, r_at = 1.0, 50.0, 0.5
    W = np.diag(
        [
            q_c,
            q_c,
            q_c,
            q_l,
            q_att,
            q_att,
            q_att,
            q_dr,
            q_dr,
            q_dr,
            q_v,
            r_rpy,
            r_rpy,
            r_rpy,
            r_T,
            r_at,
        ]
    )
    W_e = np.diag([q_c, q_c, q_c, q_l, q_att, q_att, q_att, q_dr, q_dr, q_dr, q_v])
    ocp.cost.W = W
    ocp.cost.W_e = W_e
    ocp.cost.yref = np.zeros(y.rows())
    ocp.cost.yref_e = np.zeros(y_e.rows())

    # State bounds: z floor/ceiling (2), rpy (3,4,5), vel (6,7,8), v_theta (13: 0..vtheta).
    ocp.constraints.lbx = np.array([z_min, -0.5, -0.5, -0.5, -v_max, -v_max, -v_max, 0.0])
    ocp.constraints.ubx = np.array([z_max, 0.5, 0.5, 0.5, v_max, v_max, v_max, vtheta_max])
    ocp.constraints.idxbx = np.array([2, 3, 4, 5, 6, 7, 8, 13])
    # Soften the velocity and v_theta bounds so a transient overspeed never makes the QP
    # infeasible (positions 4,5,6,7 within idxbx → vel x/y/z and v_theta).
    ocp.constraints.idxsbx = np.array([4, 5, 6, 7])
    # Slack costs accumulate across soft constraints (velocity bounds first, then the
    # optional obstacle/gate keep-out below) in acados' [sbx, sh] order.
    zl_parts = [1e3 * np.ones(4)]
    zu_parts = [1e3 * np.ones(4)]
    Zl_parts = [1e3 * np.ones(4)]
    Zu_parts = [1e3 * np.ones(4)]

    # Convex per-stage keep-out: one half-plane ``a_x·px + a_y·py - b >= 0`` per obstacle
    # and per non-target gate. The normal ``a`` points from the keep-out centre to the
    # *predicted* drone position and ``b`` puts the line tangent to the keep-out circle, so
    # the feasible side is convex — re-linearised every tick (SCP). Linear ⇒ the QP stays
    # convex and solves fast/reliably, and *every* gate is kept out (no "just-passed gate"
    # exemption, so a U-turn back toward a frame is still caught). The target gate's plane
    # is disabled at run time (``a=0, b=-1e9``) since we fly through it on the path.
    if n_keepout > 0:
        px, py = ocp.model.x[0], ocp.model.x[1]
        h_list = []
        off = n_head
        for _ in range(n_keepout):
            h_list.append(p[off] * px + p[off + 1] * py - p[off + 2])
            off += 3
        ocp.model.con_h_expr = ca.vertcat(*h_list)
        ocp.constraints.lh = np.zeros(n_keepout)
        ocp.constraints.uh = 1e9 * np.ones(n_keepout)
        ocp.constraints.idxsh = np.arange(n_keepout)
        zl_parts.append(_CON_SLACK_LIN * np.ones(n_keepout))
        zu_parts.append(_CON_SLACK_LIN * np.ones(n_keepout))
        Zl_parts.append(_CON_SLACK_QUAD * np.ones(n_keepout))
        Zu_parts.append(_CON_SLACK_QUAD * np.ones(n_keepout))

    ocp.cost.zl = np.concatenate(zl_parts)
    ocp.cost.zu = np.concatenate(zu_parts)
    ocp.cost.Zl = np.concatenate(Zl_parts)
    ocp.cost.Zu = np.concatenate(Zu_parts)

    # Input bounds: rpy commands, collective thrust, progress acceleration.
    ocp.constraints.lbu = np.array([-0.5, -0.5, -0.5, parameters["thrust_min"] * 4, -atheta_max])
    ocp.constraints.ubu = np.array([0.5, 0.5, 0.5, parameters["thrust_max"] * 4, atheta_max])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])

    ocp.constraints.x0 = np.zeros(nx)

    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.tol = 1e-6
    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1
    ocp.solver_options.qp_solver_iter_max = 50
    ocp.solver_options.nlp_solver_max_iter = 50
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

    solver = AcadosOcpSolver(
        ocp,
        json_file="c_generated_code/mpcc_planner.json",
        verbose=verbose,
        build=True,
        generate=True,
    )
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

    TARGET_SPEED = 1.5  # m/s — cruise speed of the velocity profile
    V_EDGE = 0.6  # m/s — speed at trajectory start/end
    ACCEL_DIST = 0.8  # m — ramp-up arc length
    DECEL_DIST = 0.8  # m — ramp-down arc length
    APPROACH_DIST = 0.35  # m — before-gate waypoint offset along the gate normal
    DEPART_DIST = 0.35  # m — after-gate waypoint offset along the gate normal
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
    SMOOTHING = 0.1  # splprep smoothing factor
    W_ANCHOR = 30.0  # fit weight: start waypoint
    W_GATE = 4.0  # fit weight: gate waypoints
    W_FREE = 1.0  # fit weight: free intermediates
    MIN_REF_Z = 0.1  # m — hard floor for the reference

    _WP_LO = np.array([-2.4, -1.4, 0.1])
    _WP_HI = np.array([2.4, 1.4, 1.45])

    def __init__(self, obs: dict, config: object, N: int = 25) -> None:
        """Build the initial full-track path from the current observation."""
        self.freq = config.env.freq
        self.N = N
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
        return result

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
        data, prev = [], self._start_pos.copy()
        for i in range(len(gates_pos)):
            x_axis = R.from_quat(gates_quat[i]).apply([1.0, 0.0, 0.0])
            if np.dot(gates_pos[i] - prev, x_axis) < 0:  # orient so entry is on the near side
                x_axis = -x_axis
            entry = gates_pos[i] - x_axis * self.APPROACH_DIST
            exit_ = gates_pos[i] + x_axis * self.DEPART_DIST
            data.append((gates_pos[i].copy(), x_axis.copy(), entry, exit_))
            prev = exit_
        return data

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
    V_TARGET = 1.5  # m/s — target progress speed (cruise); matched to SimplePlanner.TARGET_SPEED
    VTHETA_MAX = 5  # m/s — hard-ish cap on progress speed
    REPLAN_TOL = 1e-3  # object move (m) beyond which we replan
    MAX_HOLD_TICKS = 3  # consecutive failed solves held before braking to hover

    #: Per-node growth of the shooting interval (non-uniform time grid). The first interval
    #: is the real control period dt; each later one is this much longer, so the same N
    #: nodes look much further ahead for ~no extra compute. 1.0 → classic uniform dt grid.
    #: e.g. N=25, dt=0.02, growth=1.06 → horizon ≈ 1.1 s instead of 0.5 s.
    HORIZON_GROWTH = 1.0

    #: Add the MPC's soft obstacle/gate-frame keep-out (penalty cost on violation) to MPCC.
    USE_OBSTACLE_CONSTRAINTS = True
    #: Add the per-gate keep-out half-planes. Set False to temporarily switch the gate costs
    #: off (the solver is then rebuilt without any gate constraint); obstacle keep-out is
    #: unaffected. Has no effect when USE_OBSTACLE_CONSTRAINTS is False (everything off).
    USE_GATE_CONSTRAINTS = False
    #: MPC soft-obstacle clearance (m), kept below the planner's PLAN_CLEARANCE so the path
    #: stays feasible and the constraint only bites on real drift toward a pole.
    MPC_OBS_CLEARANCE = 0.15

    #: Print a rolling per-tick timing breakdown (project / set-params / solve) every 50
    #: ticks, to find what eats the 20 ms control budget. Off by default.
    PROFILE = False

    #: Arc-length sampling step (m) for the internal cubic spline that feeds the embedded
    #: path coefficients to acados. Finer = closer to the planner B-spline; 5 cm is ample.
    _CUBIC_DS = 0.05

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Build the MPCC solver and the initial path."""
        super().__init__(obs, info, config)
        self._config = config
        self._N = 25
        self._dt = 1 / config.env.freq
        # Non-uniform shooting grid: first interval = the real control period dt, each later
        # interval grown by HORIZON_GROWTH, so the same N nodes reach much further ahead.
        # node_t[k] is the predicted time of node k (used to propagate the per-stage theta).
        self._time_steps = self._dt * self.HORIZON_GROWTH ** np.arange(self._N)
        self._node_t = np.concatenate(([0.0], np.cumsum(self._time_steps)))  # (N+1,)
        self._T_HORIZON = float(self._node_t[-1])
        self.drone_params = load_params("so_rpy", config.sim.drone_model)
        self._hover_thrust = self.drone_params["mass"] * -self.drone_params["gravity_vec"][-1]

        self._n_obstacles = int(len(obs["obstacles_pos"])) if self.USE_OBSTACLE_CONSTRAINTS else 0
        use_gates = self.USE_OBSTACLE_CONSTRAINTS and self.USE_GATE_CONSTRAINTS
        self._n_gates = int(len(obs["gates_pos"])) if use_gates else 0
        self._n_keepout = self._n_obstacles + self._n_gates
        self._solver, self._ocp = create_mpcc_ocp_solver(
            self._T_HORIZON,
            self._N,
            self.drone_params,
            z_min=self.GROUND_Z,
            vtheta_max=self.VTHETA_MAX,
            n_obstacles=self._n_obstacles,
            n_gates=self._n_gates,
            obs_clearance=self.MPC_OBS_CLEARANCE,
            gate_drone_r=SimplePlanner.DRONE_RADIUS,
            time_steps=self._time_steps if self.HORIZON_GROWTH != 1.0 else None,
        )
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()

        # Path planner (warm-started B-spline planner; we use its geometric-path API).
        self.planner = SimplePlanner(obs, config, N=self._N)
        # Internal arc-length cubic spline of the planner path; supplies the local cubic
        # coefficients the model needs to evaluate p_d(theta) symbolically. Rebuilt on replan.
        self._cubic = None
        self._cubic_smax = 0.0
        self._build_cubic()
        self._snapshot_objects(obs)

        # Background replanning (never stalls the 50 Hz loop).
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._replan_future = None

        self._theta_pred = None  # last solve's per-stage theta (warm start for relinearisation)
        self._pos_pred = None  # last solve's per-stage positions (keep-out linearisation point)
        self._x_pred = None  # last solve's full state trajectory (N+1, nx) — primal warm start
        self._u_pred = None  # last solve's full input trajectory (N, nu)   — primal warm start
        self._last_u = None
        self._hover_cmd = np.array([0.0, 0.0, 0.0, self._hover_thrust])
        self._consec_fail = 0
        self._finished = False
        self._last_obs = obs
        self._theta_est = None  # last tick's progress θ; constrains the projection (no jumps)
        self._progress_point = None  # drone's current projection on the path (render marker)
        self._prof = {"proj": 0.0, "setp": 0.0, "solve": 0.0, "n": 0}

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
            return
        n = max(self._N + 2, int(length / self._CUBIC_DS) + 1)
        s = np.linspace(0.0, length, n)
        pos, _ = self.planner.path_point_tangent(s)
        self._cubic = CubicSpline(s, np.asarray(pos, dtype=float))
        self._cubic_smax = length

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

    def _keepout_halfplanes(self, obs: dict, pos_pred: np.ndarray) -> np.ndarray:
        """Build per-stage convex keep-out half-planes for each obstacle and non-target gate.

        Each half-plane ``[a_x, a_y, b]`` is linearised at the predicted per-stage positions
        ``pos_pred`` ((N+1, 3)). The normal ``a`` points from the keep-out centre to the
        predicted drone position; ``b = a·c + r`` makes the line tangent to the keep-out
        circle of radius ``r`` (obstacle clearance, or the gate's outer frame half-width).
        The target gate is
        disabled (``a=0, b=-1e9`` → constraint always satisfied: we fly through it).

        Returns an array of shape (N+1, n_keepout, 3).
        """
        P = pos_pred[:, :2]  # (N+1, 2) predicted xy
        # (centre xy, radius, disabled) for every keep-out: obstacles then gates.
        items = [
            (np.asarray(op[:2], float), self.MPC_OBS_CLEARANCE, False)
            for op in obs["obstacles_pos"]
        ]
        target = int(obs["target_gate"])
        gate_r = _GateFrame.OUTER / 2 + SimplePlanner.DRONE_RADIUS  # clear the frame
        for j in range(self._n_gates):
            items.append((np.asarray(obs["gates_pos"][j][:2], float), gate_r, j == target))

        out = np.zeros((self._N + 1, len(items), 3))
        for m, (c, r, disabled) in enumerate(items):
            if disabled:
                out[:, m, 2] = -1e9  # a=0, b=-1e9 → a·p - b = 1e9 >= 0 always
                continue
            d = P - c  # (N+1, 2)
            n = d / np.clip(np.linalg.norm(d, axis=1, keepdims=True), 1e-6, None)
            out[:, m, 0:2] = n
            out[:, m, 2] = (n * c).sum(axis=1) + r  # b = a·c + r
        return out

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

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Project onto the path, set per-stage contouring params, solve MPCC."""
        self._last_obs = obs
        self._maybe_replan(obs)
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
        if self._theta_pred is None or self._x_pred is None:
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
        # acados can evaluate p_d(theta) symbolically. Plus the convex keep-out half-planes
        # linearised at the predicted positions.
        thetas = self._stage_thetas(theta0)
        pts, _tans = self.planner.path_point_tangent(thetas)  # only for the keep-out fallback
        pos_pred = self._pos_pred if self._pos_pred is not None else pts
        hp = self._keepout_halfplanes(obs, pos_pred) if self._n_keepout else None
        for k in range(self._N + 1):
            theta_i, cx, cy, cz = self._path_segment_coeffs(thetas[k])
            head = np.concatenate(([theta_i], cx, cy, cz, [self.V_TARGET]))
            p_k = head if hp is None else np.concatenate((head, hp[k].reshape(-1)))
            self._solver.set(k, "p", p_k)
        t_setp = time.perf_counter()

        status = self._solver.solve()
        if self.PROFILE:
            self._profile_tick(t_proj - tic, t_setp - t_proj, time.perf_counter() - t_setp)
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
        return self._hover_cmd

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