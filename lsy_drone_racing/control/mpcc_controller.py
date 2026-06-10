"""Model Predictive Contouring Control (MPCC) for drone racing.

Unlike the reference-tracking MPC (:mod:`mpc_planner_controller`), this controller tracks
the **geometric path** produced by the planner and optimises **progress** along it. The
acados model is augmented with a progress state ``theta`` (arc length) and its speed
``v_theta``; the cost penalises the *contouring* error (perpendicular distance to the path)
and the *lag* error (longitudinal), and rewards advancing ``theta`` at a target speed.

Because there is no time-parameterised reference, the reference can never "run away" from
the drone — so the reference governor / nearest-tick machinery of the tracking MPC is not
needed here. The path enters acados as per-stage parameters: for each shooting node we pass
the path point ``p_ref`` and unit tangent ``t_ref`` at that node's predicted ``theta``
(re-linearised every tick — standard practical MPCC). The path comes from the same
warm-started :class:`SimplePlanner`, queried by arc length via ``path_point_tangent``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import casadi as ca
import numpy as np
import scipy.linalg
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from crazyflow.sim.visualize import draw_line, draw_points
from drone_models.core import load_params
from drone_models.so_rpy import symbolic_dynamics_euler
from drone_models.utils.rotation import ang_vel2rpy_rates
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.mpc_planner_controller import SimplePlanner, _GateFrame

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ── Augmented model: physical drone + progress double-integrator ────────────────


def create_mpcc_model(parameters: dict) -> AcadosModel:
    """Drone model (12 states / 4 inputs) augmented with progress states ``[theta, v_theta]``
    and a progress-acceleration input ``a_theta``: ``theta_dot = v_theta``, ``v_theta_dot =
    a_theta``."""
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
    Tf: float, N: int, parameters: dict, z_min: float = 0.0, z_max: float = 2.5,
    v_max: float = 2.0, vtheta_max: float = 2.5, atheta_max: float = 6.0,
    verbose: bool = False,
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Acados OCP/solver for MPCC. NONLINEAR_LS cost on contouring/lag/progress; per-stage
    path point + tangent + target speed are model parameters ``p = [p_ref(3), t_ref(3),
    v_target(1)]`` set per solve."""
    ocp = AcadosOcp()
    ocp.model = create_mpcc_model(parameters)
    nx = ocp.model.x.rows()  # 14
    nu = ocp.model.u.rows()  # 5
    ocp.solver_options.N_horizon = N

    hover_thrust = parameters["mass"] * -parameters["gravity_vec"][-1]

    # Per-stage parameters: path reference point, unit tangent, target progress speed.
    p = ca.MX.sym("p", 7)
    ocp.model.p = p
    ocp.parameter_values = np.zeros(7)
    p_ref, t_ref, v_target = p[0:3], p[3:6], p[6]

    pos = ocp.model.x[0:3]
    rpy = ocp.model.x[3:6]
    drpy = ocp.model.x[9:12]
    v_theta = ocp.model.x[13]
    rpy_cmd = ocp.model.u[0:3]
    thrust = ocp.model.u[3]
    a_theta = ocp.model.u[4]

    d = pos - p_ref
    e_l = ca.dot(t_ref, d)  # lag (longitudinal, signed)
    e_c = d - e_l * t_ref  # contouring (perpendicular, 3-vector)
    e_v = v_theta - v_target  # progress-speed tracking (the "reward")

    # Smooth least-squares residual (Gauss-Newton friendly: linear in the optimisation
    # variables given the frozen per-stage path params).
    y = ca.vertcat(e_c, e_l, rpy, drpy, e_v, rpy_cmd, thrust - hover_thrust, a_theta)
    y_e = ca.vertcat(e_c, e_l, rpy, drpy, e_v)
    ocp.model.cost_y_expr = y
    ocp.model.cost_y_expr_e = y_e
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"

    q_c, q_l, q_att, q_dr, q_v = 50.0, 1.0, 1.0, 5.0, 12.0
    r_rpy, r_T, r_at = 1.0, 50.0, 0.5
    W = np.diag([
        q_c, q_c, q_c, q_l, q_att, q_att, q_att, q_dr, q_dr, q_dr, q_v,
        r_rpy, r_rpy, r_rpy, r_T, r_at,
    ])
    W_e = np.diag([q_c, q_c, q_c, q_l, q_att, q_att, q_att, q_dr, q_dr, q_dr, q_v])
    ocp.cost.W = W
    ocp.cost.W_e = W_e
    ocp.cost.yref = np.zeros(y.rows())
    ocp.cost.yref_e = np.zeros(y_e.rows())

    # State bounds: z floor/ceiling (2), rpy (3,4,5), vel (6,7,8), v_theta (13: 0..vtheta).
    ocp.constraints.lbx = np.array(
        [z_min, -0.5, -0.5, -0.5, -v_max, -v_max, -v_max, 0.0]
    )
    ocp.constraints.ubx = np.array(
        [z_max, 0.5, 0.5, 0.5, v_max, v_max, v_max, vtheta_max]
    )
    ocp.constraints.idxbx = np.array([2, 3, 4, 5, 6, 7, 8, 13])
    # Soften the velocity and v_theta bounds so a transient overspeed never makes the QP
    # infeasible (positions 4,5,6,7 within idxbx → vel x/y/z and v_theta).
    ocp.constraints.idxsbx = np.array([4, 5, 6, 7])
    ns = 4
    ocp.cost.zl = 1e3 * np.ones(ns)
    ocp.cost.zu = 1e3 * np.ones(ns)
    ocp.cost.Zl = 1e3 * np.ones(ns)
    ocp.cost.Zu = 1e3 * np.ones(ns)

    # Input bounds: rpy commands, collective thrust, progress acceleration.
    ocp.constraints.lbu = np.array(
        [-0.5, -0.5, -0.5, parameters["thrust_min"] * 4, -atheta_max]
    )
    ocp.constraints.ubu = np.array(
        [0.5, 0.5, 0.5, parameters["thrust_max"] * 4, atheta_max]
    )
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
    ocp.solver_options.tf = Tf

    solver = AcadosOcpSolver(
        ocp, json_file="c_generated_code/mpcc_planner.json",
        verbose=verbose, build=True, generate=True,
    )
    return solver, ocp


# ── Controller ──────────────────────────────────────────────────────────────────


class MPCCController(Controller):
    """MPCC controller: contouring control along the warm-started SimplePlanner path."""

    GROUND_Z = 0.0
    V_TARGET = 1.2  # m/s — target progress speed (cruise)
    VTHETA_MAX = 2.5  # m/s — hard-ish cap on progress speed
    REPLAN_TOL = 1e-3  # object move (m) beyond which we replan
    MAX_HOLD_TICKS = 3  # consecutive failed solves held before braking to hover

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Build the MPCC solver and the initial path."""
        super().__init__(obs, info, config)
        self._config = config
        self._N = 25
        self._dt = 1 / config.env.freq
        self._T_HORIZON = self._N * self._dt
        self.drone_params = load_params("so_rpy", config.sim.drone_model)
        self._hover_thrust = self.drone_params["mass"] * -self.drone_params["gravity_vec"][-1]

        self._solver, self._ocp = create_mpcc_ocp_solver(
            self._T_HORIZON, self._N, self.drone_params,
            z_min=self.GROUND_Z, vtheta_max=self.VTHETA_MAX,
        )
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()

        # Path planner (warm-started B-spline planner; we use its geometric-path API).
        self.planner = SimplePlanner(obs, config, N=self._N)
        self._snapshot_objects(obs)

        # Background replanning (never stalls the 50 Hz loop).
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._replan_future = None

        self._theta_pred = None  # last solve's per-stage theta (warm start for relinearisation)
        self._last_u = None
        self._hover_cmd = np.array([0.0, 0.0, 0.0, self._hover_thrust])
        self._consec_fail = 0
        self._finished = False
        self._last_obs = obs
        self._progress_point = None  # drone's current projection on the path (render marker)

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
            except Exception as exc:
                print(f"[MPCC] background replan failed: {exc!r}")
            self._replan_future = None
        # Kick off a new replan if the track changed and none is in flight.
        if self._replan_future is None and self._objects_changed(obs):
            self._replan_future = self._executor.submit(
                self.planner._compute_trajectory, self._copy_obs(obs), self.V_TARGET
            )
            self._snapshot_objects(obs)

    # ── Control ───────────────────────────────────────────────────────────────

    def _stage_thetas(self, theta0: float) -> np.ndarray:
        """Predicted theta at each shooting node (warm start for the path relinearisation):
        the previous solution shifted one step, re-anchored at the current projection."""
        if self._theta_pred is None:
            return theta0 + np.arange(self._N + 1) * self._dt * self.V_TARGET
        th = np.empty(self._N + 1)
        th[:-1] = self._theta_pred[1:]
        th[-1] = self._theta_pred[-1] + self._dt * self.V_TARGET
        th[0] = theta0  # re-anchor to where the drone actually is
        return np.maximum.accumulate(th)  # keep monotonic

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Project onto the path, set per-stage contouring params, solve MPCC."""
        self._last_obs = obs
        self._maybe_replan(obs)

        # Project the drone onto the path → current progress theta0 and progress speed.
        theta0 = self.planner.project_to_theta(obs["pos"], obs["vel"])
        p0, t0 = self.planner.path_point_tangent(theta0)
        self._progress_point = p0  # the single "where is the drone along the path" marker
        vtheta0 = float(np.clip(np.dot(obs["vel"], t0), 0.0, self.VTHETA_MAX))

        # Initial state (augmented).
        rpy = R.from_quat(obs["quat"]).as_euler("xyz")
        drpy = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((obs["pos"], rpy, obs["vel"], drpy, [theta0, vtheta0]))
        self._solver.set(0, "lbx", x0)
        self._solver.set(0, "ubx", x0)

        # Per-stage path reference (point + unit tangent) at the predicted theta of each node.
        thetas = self._stage_thetas(theta0)
        pts, tans = self.planner.path_point_tangent(thetas)
        for k in range(self._N + 1):
            self._solver.set(k, "p", np.concatenate((pts[k], tans[k], [self.V_TARGET])))

        status = self._solver.solve()
        if status == 0:
            self._theta_pred = np.array([self._solver.get(k, "x")[12] for k in range(self._N + 1)])
            self._last_u = self._solver.get(0, "u")[:4]  # drop a_theta
            self._consec_fail = 0
            return self._last_u
        self._consec_fail += 1
        if self._last_u is not None and self._consec_fail <= self.MAX_HOLD_TICKS:
            return self._last_u
        print(f"[MPCC] solve failed (status={status}) at theta={theta0:.2f}; braking to hover")
        return self._hover_cmd

    def step_callback(self, action, obs, reward, terminated, truncated, info) -> bool:
        if int(obs["target_gate"]) == -1:
            self._finished = True
        return self._finished

    # ── Visualisation (mirrors the MPC controller's overlay) ──────────────────

    def render_callback(self, sim: object) -> None:
        """Draw gates (yellow + cyan/white frames), obstacles (orange poles), the planned
        path (green line) with the gate waypoints (magenta), and a single red marker at the
        drone's current progress along the path (its projection point θ)."""
        obs = self._last_obs
        if obs is not None:
            draw_points(sim, np.atleast_2d(obs["gates_pos"]),
                        rgba=np.array([1.0, 1.0, 0.0, 1.0]), size=0.08)
            for gpos, gquat in zip(obs["gates_pos"], obs["gates_quat"]):
                self._draw_square(sim, gpos, gquat, _GateFrame.OUTER / 2,
                                  np.array([0.0, 1.0, 1.0, 1.0]))
                self._draw_square(sim, gpos, gquat, _GateFrame.OPENING / 2,
                                  np.array([1.0, 1.0, 1.0, 1.0]))
            for opos in np.atleast_2d(obs["obstacles_pos"]):
                pole = np.array([[opos[0], opos[1], 0.0], [opos[0], opos[1], opos[2]]])
                draw_line(sim, pole, rgba=np.array([1.0, 0.5, 0.0, 1.0]),
                          start_size=6.0, end_size=6.0)

        # Single red marker: the drone's current progress θ projected onto the path.
        if self._progress_point is not None:
            draw_points(sim, np.atleast_2d(self._progress_point),
                        rgba=np.array([1.0, 0.0, 0.0, 1.0]), size=0.07)

        # Fixed gate waypoints (entry / center / exit), magenta.
        raw = getattr(self.planner, "_raw_waypoints", None)
        if raw:
            fixed = np.array([np.asarray(p) for p, v in raw if v is not None])
            if len(fixed):
                draw_points(sim, fixed, rgba=np.array([1.0, 0.0, 1.0, 1.0]), size=0.05)

        # Planned path (green line, downsampled).
        traj = getattr(self.planner, "pos", None)
        if traj is not None:
            step = max(1, len(traj) // 20)
            draw_line(sim, traj[::step], rgba=np.array([0.0, 1.0, 0.0, 1.0]))

    @staticmethod
    def _draw_square(sim: object, center: np.ndarray, quat: np.ndarray, half: float,
                     rgba: np.ndarray) -> None:
        """Draw an oriented square outline in the gate plane (local y-z plane)."""
        rot = R.from_quat(quat)
        local = np.array([
            [0.0, half, half], [0.0, -half, half],
            [0.0, -half, -half], [0.0, half, -half], [0.0, half, half],
        ])
        draw_line(sim, center + rot.apply(local), rgba=rgba)

    def episode_callback(self):
        self._theta_pred = None
        if self._replan_future is not None:
            self._replan_future.cancel()
            self._replan_future = None

    def episode_reset(self):
        self._theta_pred = None
        self._finished = False
        self.episode_callback()
