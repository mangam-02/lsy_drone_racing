"""Path-planning + MPC controller for drone racing.

This controller combines two stages:

1. **Path planning** — a self-contained :class:`BSplinePlanner` (defined in this
   file) builds a B-spline reference trajectory through all remaining gates while
   avoiding the obstacles. It places fixed orthogonal waypoints a fixed distance
   before and after each gate (along the gate normal) so the spline crosses each
   gate straight on.
2. **Tracking MPC** — an acados nonlinear MPC (collective-thrust + attitude
   interface) tracks the planned position/velocity reference over a receding
   horizon.

Re-planning policy
------------------
The planner is rebuilt whenever the *measured* position (or orientation) of any
gate or obstacle changes. In the higher difficulty levels the true object poses
are only revealed once the drone gets within ``sensor_range`` — before that the
observation reports the nominal pose. As soon as a real pose differs from what we
last planned with (beyond a small tolerance), we replan from the drone's current
state so the trajectory always reflects the latest knowledge of the track.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from crazyflow.sim.visualize import draw_line, draw_points
from drone_models.core import load_params
from drone_models.utils.rotation import ang_vel2rpy_rates
from scipy.interpolate import make_interp_spline
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.attitude_mpc import create_ocp_solver

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ── Obstacle primitives ─────────────────────────────────────────────────────────


class _Cylinder:
    """Vertical cylinder obstacle (pole)."""

    def __init__(self, pos: np.ndarray, radius: float, height: float = 1.52):
        self.xy = pos[:2].copy()
        self.radius = radius
        self.height = height

    def contains(self, p: np.ndarray) -> bool:
        return np.linalg.norm(p[:2] - self.xy) < self.radius and 0.0 <= p[2] <= self.height


class _GateFrame:
    """Square gate frame — free inside the opening, solid frame material.

    Gate local frame: x = approach axis, y = horizontal, z = vertical.
    Outer: 0.72 m. Opening: 0.40 m.
    """

    OUTER = 0.72
    OPENING = 0.40
    DEPTH = 0.10

    def __init__(self, center: np.ndarray, quat: np.ndarray, drone_r: float = 0.07):
        self.center = center.copy()
        self.rot = R.from_quat(quat)
        self.rot_inv = self.rot.inv()
        self.ho = self.OUTER / 2 + drone_r
        self.hi = self.OPENING / 2 - drone_r
        self.hd = self.DEPTH / 2 + drone_r

    def contains(self, p: np.ndarray) -> bool:
        local = self.rot_inv.apply(p - self.center)
        if abs(local[0]) > self.hd:
            return False
        if abs(local[1]) > self.ho or abs(local[2]) > self.ho:
            return False
        return not (abs(local[1]) <= self.hi and abs(local[2]) <= self.hi)


def _free_3d(p1: np.ndarray, p2: np.ndarray, obstacles: list, n: int = 10) -> bool:
    """True if segment p1→p2 clears all 3D obstacle objects."""
    for t in np.linspace(0.0, 1.0, n):
        p = p1 + t * (p2 - p1)
        for obs in obstacles:
            if obs.contains(p):
                return False
    return True


# ── B-spline trajectory planner ──────────────────────────────────────────────────


class BSplinePlanner:
    """Offline-optimized B-spline trajectory planner.

    Before takeoff, scipy.minimize finds intermediate waypoint positions that
    minimise obstacle violations along the full B-spline curve. Fixed orthogonal
    waypoints are placed a fixed distance before/after each gate (along the gate
    normal) so the spline crosses each gate straight on. Online, the pre-computed
    pos/vel arrays are just indexed by tick.
    """

    TARGET_SPEED = 1.5  # m/s — inter-gate speed for time assignment
    GATE_SPEED = 1.2  # m/s — crossing speed tag at gate waypoints
    APPROACH_DIST = 0.35  # m — fixed orthogonal waypoint before each gate (gate normal)
    DEPART_DIST = 0.35  # m — fixed orthogonal waypoint after each gate (gate normal)
    DRONE_RADIUS = 0.07  # m — gate frame inflation
    PLAN_CLEARANCE = 0.145  # m — physical drone-to-obstacle clearance
    N_INTERMEDIATE = 2  # optimisable waypoints per inter-gate segment
    N_SAMPLE = 100  # trajectory points sampled for obstacle check
    OPT_MAXITER = 300  # max L-BFGS-B iterations
    REFINE_TIME_BUDGET = 5.0  # seconds — optional time budget when maxiter is set
    BSPLINE_DEGREE = 3  # cubic B-spline (C2 continuous)

    _WP_LO = np.array([-2.4, -1.4, 0.1])
    _WP_HI = np.array([2.4, 1.4, 1.45])

    def __init__(self, obs: dict, config: object, N: int = 25) -> None:
        """Initialize planner and build the trajectory for the current track state."""
        self.freq = config.env.freq
        self.N = N
        self._prev_visited = obs["gates_visited"].copy()
        self._prev_obs_visited = obs["obstacles_visited"].copy()
        self.build(obs)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_reference(self, tick: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (pos, vel) horizon of shape (N+1, 3) for the MPC."""
        i = min(tick, self.tick_max)
        return self.pos[i : i + self.N + 1], self.vel[i : i + self.N + 1]

    def update(self, obs: dict) -> bool:
        """Replan if new gate/obstacle positions revealed. Returns True if replanned."""
        new_gates = obs["gates_visited"]
        new_obs = obs["obstacles_visited"]
        replanned = bool(
            np.any(new_gates & ~self._prev_visited) or np.any(new_obs & ~self._prev_obs_visited)
        )
        if replanned:
            self.build(obs)
        self._prev_visited = new_gates.copy()
        self._prev_obs_visited = new_obs.copy()
        return replanned

    # ── Build / optimise ──────────────────────────────────────────────────────

    def build(self, obs: dict):
        """Run offline optimiser and store pre-computed trajectory."""
        target_gate = int(obs["target_gate"])

        if target_gate == -1:
            n = self.N + 2
            self.pos = np.tile(obs["pos"], (n, 1))
            self.vel = np.zeros((n, 3))
            self.tick_max = 1
            self._raw_waypoints = [(obs["pos"].copy(), np.zeros(3))]
            return

        cylinders, gate_frames = self._build_obstacles(obs)
        gate_data = self._compute_gate_data(obs, target_gate, cylinders, gate_frames)
        cyl_tuples = [(float(c.xy[0]), float(c.xy[1]), self.PLAN_CLEARANCE) for c in cylinders]

        t0 = time.perf_counter()
        opt_intermediates = self._optimize(
            obs["pos"].copy(), obs["vel"].copy(), gate_data, cyl_tuples
        )
        print(f"[BSplinePlanner] optimization done in {(time.perf_counter() - t0) * 1e3:.3f} ms")

        raw = self._build_waypoint_list(
            obs["pos"].copy(), obs["vel"].copy(), gate_data, opt_intermediates
        )
        self._raw_waypoints = raw
        self._finalize(raw)

    def _build_obstacles(self, obs: dict) -> tuple[list, list]:
        cylinders = [
            _Cylinder(pos=opos, radius=self.PLAN_CLEARANCE, height=1.52)
            for opos in obs["obstacles_pos"]
        ]
        gate_frames = [
            _GateFrame(gpos, gquat, drone_r=self.DRONE_RADIUS)
            for gpos, gquat in zip(obs["gates_pos"], obs["gates_quat"])
        ]
        return cylinders, gate_frames

    def _compute_gate_data(
        self, obs: dict, target_gate: int, cylinders: list, gate_frames: list
    ) -> list[tuple]:
        """Return [(gate_center, x_axis, entry_pt, exit_pt), ...] for gates from target_gate."""
        gates_pos = obs["gates_pos"]
        gates_quat = obs["gates_quat"]
        n_gates = len(gates_pos)

        gate_data = []
        prev_pos = obs["pos"].copy()
        for i in range(target_gate, n_gates):
            rot = R.from_quat(gates_quat[i])
            x_axis = rot.apply([1.0, 0.0, 0.0])
            if np.dot(gates_pos[i] - prev_pos, x_axis) < 0:
                x_axis = -x_axis
            seg_obs = cylinders + gate_frames[target_gate : i + 1]
            entry_pt = self._safe_entry(gates_pos[i], x_axis, cylinders)
            exit_pt = self._safe_exit(gates_pos[i], x_axis, seg_obs)
            gate_data.append((gates_pos[i].copy(), x_axis.copy(), entry_pt.copy(), exit_pt.copy()))
            prev_pos = exit_pt
        return gate_data

    def _safe_entry(self, gate_pos: np.ndarray, x_axis: np.ndarray, obstacles: list) -> np.ndarray:
        """Fixed orthogonal waypoint before the gate (falls back if blocked)."""
        for d in [self.APPROACH_DIST, 0.4, 0.5, 0.2]:
            pt = gate_pos - x_axis * d
            if _free_3d(pt, gate_pos, obstacles):
                return pt
        return gate_pos.copy()

    def _safe_exit(self, gate_pos: np.ndarray, x_axis: np.ndarray, obstacles: list) -> np.ndarray:
        """Fixed orthogonal waypoint after the gate (falls back if blocked)."""
        for d in [self.DEPART_DIST, 0.5, 0.3, 0.2]:
            pt = gate_pos + x_axis * d
            if _free_3d(gate_pos, pt, obstacles):
                return pt
        return gate_pos.copy()

    # ── Offline optimiser ─────────────────────────────────────────────────────

    def _optimize(
        self,
        drone_pos: np.ndarray,
        drone_vel: np.ndarray,
        gate_data: list[tuple],
        cyl_tuples: list[tuple],
        x0_override: np.ndarray | None = None,
        maxiter: int | None = None,
    ) -> np.ndarray:
        n_wps = len(gate_data) * self.N_INTERMEDIATE
        x0 = (
            x0_override
            if x0_override is not None
            else self._initial_intermediates(drone_pos, gate_data)
        )

        bounds = [
            (float(lo), float(hi)) for _ in range(n_wps) for lo, hi in zip(self._WP_LO, self._WP_HI)
        ]

        def cost(x: np.ndarray) -> float:
            intermediates = x.reshape(n_wps, 3)
            raw = self._build_waypoint_list(drone_pos, drone_vel, gate_data, intermediates)
            return self._trajectory_cost(raw, cyl_tuples)

        t_start = time.perf_counter()
        budget = self.REFINE_TIME_BUDGET if maxiter else None

        def _time_cb(xk: np.ndarray) -> None:
            if budget is not None and (time.perf_counter() - t_start) > budget:
                raise StopIteration

        result = minimize(
            cost,
            x0.flatten(),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter or self.OPT_MAXITER, "ftol": 1e-10, "gtol": 1e-7},
            callback=_time_cb,
        )
        return result.x.reshape(n_wps, 3)

    def _initial_intermediates(self, drone_pos: np.ndarray, gate_data: list[tuple]) -> np.ndarray:
        """Initial waypoints between segment endpoints.

        For U-turn segments (exit direction nearly opposite to direction to next entry),
        seeds a bypass waypoint perpendicular to the gate to route clear of the frame.
        """
        n_wps = len(gate_data) * self.N_INTERMEDIATE
        pts = np.zeros((n_wps, 3))
        seg_starts = [drone_pos] + [gd[3] for gd in gate_data[:-1]]

        for seg_i, ((gate_center, _, entry_pt, exit_pt), seg_start) in enumerate(
            zip(gate_data, seg_starts)
        ):
            # Detect U-turn: previous gate exit direction vs direction to next entry
            is_uturn = False
            bypass = None
            if seg_i > 0:
                prev_x = gate_data[seg_i - 1][1]  # previous gate's x_axis
                to_entry = entry_pt - seg_start
                d = np.linalg.norm(to_entry)
                if d > 1e-6 and np.dot(to_entry / d, prev_x) < -0.5:
                    is_uturn = True
                    # Perpendicular to exit axis, toward the entry side
                    perp = np.cross(prev_x, [0.0, 0.0, 1.0])
                    pn = np.linalg.norm(perp)
                    if pn > 1e-6:
                        perp /= pn
                    sign = np.sign(np.dot(entry_pt - seg_start, perp)) or 1.0
                    bypass = np.clip(
                        seg_start + perp * sign * 0.7 + np.array([0.0, 0.0, 0.1]),
                        self._WP_LO,
                        self._WP_HI,
                    )

            if is_uturn and bypass is not None:
                pts[seg_i * self.N_INTERMEDIATE] = bypass
                for k in range(1, self.N_INTERMEDIATE):
                    frac = k / self.N_INTERMEDIATE
                    pts[seg_i * self.N_INTERMEDIATE + k] = np.clip(
                        bypass + frac * (entry_pt - bypass), self._WP_LO, self._WP_HI
                    )
            else:
                for k in range(self.N_INTERMEDIATE):
                    frac = (k + 1) / (self.N_INTERMEDIATE + 1)
                    pts[seg_i * self.N_INTERMEDIATE + k] = seg_start + frac * (entry_pt - seg_start)
        return pts

    def _trajectory_cost(self, raw: list[tuple], cyl_tuples: list[tuple]) -> float:
        """Evaluate obstacle violation along the sampled B-spline trajectory."""
        try:
            traj_pos, _, _ = self._sample(raw, self.N_SAMPLE)
        except Exception:
            return 1e6

        cost = 0.0
        for cx, cy, r in cyl_tuples:
            dist = np.linalg.norm(traj_pos[:, :2] - np.array([cx, cy]), axis=1)
            violation = np.maximum(0.0, r - dist)
            cost += float(np.sum(violation**2))
        return cost

    # ── Waypoint assembly ─────────────────────────────────────────────────────

    def _build_waypoint_list(
        self,
        drone_pos: np.ndarray,
        drone_vel: np.ndarray,
        gate_data: list[tuple],
        intermediates: np.ndarray,
    ) -> list[tuple]:
        """Interleave optimised intermediates with fixed gate/entry/exit waypoints.

        Gate-related waypoints carry a velocity tag (along the gate normal); the
        optimised intermediates carry ``None`` so callers can tell them apart.
        """
        points = [(drone_pos, drone_vel)]

        for seg_i, (gate_center, x_axis, entry_pt, exit_pt) in enumerate(gate_data):
            for k in range(self.N_INTERMEDIATE):
                wp = intermediates[seg_i * self.N_INTERMEDIATE + k]
                points.append((wp.copy(), None))
            points.append((entry_pt, x_axis * self.GATE_SPEED))
            points.append((gate_center, x_axis * self.GATE_SPEED))
            points.append((exit_pt, x_axis * self.GATE_SPEED))

        last_x = gate_data[-1][1]
        last_exit = gate_data[-1][3]
        points.append((last_exit + last_x * 0.3, np.zeros(3)))
        return points

    # ── Finalise trajectory (B-spline) ────────────────────────────────────────

    def _finalize(self, raw: list[tuple]):
        times, _ = self._times_and_positions(raw)
        t_total = float(times[-1])
        n_samp = max(int(self.freq * t_total), self.N + 2)
        self.pos, self.vel, _ = self._sample(raw, n_samp)
        self.tick_max = n_samp - 1 - self.N

    def _times_and_positions(self, raw: list[tuple]) -> tuple[np.ndarray, np.ndarray]:
        """Cumulative arrival times and positions (strictly increasing times)."""
        times, P, t = [], [], 0.0
        prev = np.asarray(raw[0][0])
        for i, (pos, _vel) in enumerate(raw):
            pos = np.asarray(pos)
            if i > 0:
                t += max(np.linalg.norm(pos - prev) / self.TARGET_SPEED, 0.1)
            times.append(t)
            P.append(pos)
            prev = pos
        times, P = np.asarray(times), np.asarray(P)
        keep = np.concatenate(([True], np.diff(times) > 1e-9))
        return times[keep], P[keep]

    def _sample(self, raw: list[tuple], n_samples: int) -> tuple[np.ndarray, np.ndarray, float]:
        """Sample (pos, vel, t_total) along the interpolating B-spline."""
        times, P = self._times_and_positions(raw)
        k = min(self.BSPLINE_DEGREE, len(P) - 1)
        spline = make_interp_spline(times, P, k=k)
        t_s = np.linspace(0.0, times[-1], n_samples)
        return spline(t_s), spline.derivative()(t_s), float(times[-1])


# ── Controller ────────────────────────────────────────────────────────────────


class MPCPlanner(Controller):
    """Replanning B-spline path-planner combined with a tracking attitude-MPC."""

    #: Move (m) / rotation (quat component) beyond which we consider an object to
    #: have "changed" and trigger a replan.
    REPLAN_TOL = 1e-3

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Build the MPC solver and the initial reference trajectory.

        Args:
            obs: Initial observation of the environment's state.
            info: Additional environment information from the reset.
            config: The race configuration.
        """
        super().__init__(obs, info, config)
        self._config = config

        # ── MPC setup ──────────────────────────────────────────────────────────
        self._N = 25
        self._dt = 1 / config.env.freq
        self._T_HORIZON = self._N * self._dt

        self.drone_params = load_params("so_rpy", config.sim.drone_model)
        self._acados_ocp_solver, self._ocp = create_ocp_solver(
            self._T_HORIZON, self._N, self.drone_params
        )
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()
        self._ny = self._nx + self._nu
        self._ny_e = self._nx
        self._hover_thrust = self.drone_params["mass"] * -self.drone_params["gravity_vec"][-1]

        # ── Path planner ───────────────────────────────────────────────────────
        self.planner = BSplinePlanner(obs, config, N=self._N)
        self._snapshot_objects(obs)  # remember poses we just planned with

        self._tick = 0
        self._finished = False
        self._last_obs = obs  # latest observation, for rendering registered poses

    # ── Re-planning ────────────────────────────────────────────────────────────

    def _snapshot_objects(self, obs: dict):
        """Store the gate/obstacle poses the current plan was built from."""
        self._last_gates_pos = obs["gates_pos"].copy()
        self._last_gates_quat = obs["gates_quat"].copy()
        self._last_obstacles_pos = obs["obstacles_pos"].copy()

    def _objects_changed(self, obs: dict) -> bool:
        """True if any gate/obstacle pose moved beyond the replan tolerance."""
        return bool(
            np.any(np.abs(obs["gates_pos"] - self._last_gates_pos) > self.REPLAN_TOL)
            or np.any(np.abs(obs["gates_quat"] - self._last_gates_quat) > self.REPLAN_TOL)
            or np.any(np.abs(obs["obstacles_pos"] - self._last_obstacles_pos) > self.REPLAN_TOL)
        )

    def _maybe_replan(self, obs: dict):
        """Rebuild the trajectory from the current state if the track changed."""
        if self._objects_changed(obs):
            self.planner.build(obs)  # plans from obs["pos"] -> restart tracking at 0
            self._tick = 0
            self._snapshot_objects(obs)

    # ── Control ─────────────────────────────────────────────────────────────────

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Replan if needed, then solve the MPC tracking problem.

        Args:
            obs: The current observation of the environment.
            info: Optional additional information.

        Returns:
            The attitude command [roll, pitch, yaw, collective_thrust].
        """
        self._last_obs = obs
        self._maybe_replan(obs)

        # Reference horizon (N+1 points) from the planned trajectory.
        pos_ref, vel_ref = self.planner.get_reference(self._tick)

        # ── Initial state x0 = [pos, rpy, vel, drpy] ───────────────────────────
        rpy = R.from_quat(obs["quat"]).as_euler("xyz")
        drpy = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((obs["pos"], rpy, obs["vel"], drpy))
        self._acados_ocp_solver.set(0, "lbx", x0)
        self._acados_ocp_solver.set(0, "ubx", x0)

        # ── Stage references ───────────────────────────────────────────────────
        yref = np.zeros((self._N, self._ny))
        yref[:, 0:3] = pos_ref[: self._N]  # position
        yref[:, 6:9] = vel_ref[: self._N]  # velocity
        yref[:, 15] = self._hover_thrust  # input ref: hover thrust
        for j in range(self._N):
            self._acados_ocp_solver.set(j, "yref", yref[j])

        # ── Terminal reference ─────────────────────────────────────────────────
        yref_e = np.zeros((self._ny_e,))
        yref_e[0:3] = pos_ref[self._N]
        yref_e[6:9] = vel_ref[self._N]
        self._acados_ocp_solver.set(self._N, "yref", yref_e)

        # ── Solve ──────────────────────────────────────────────────────────────
        self._acados_ocp_solver.solve()
        return self._acados_ocp_solver.get(0, "u")

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Advance the trajectory index and check for completion."""
        self._tick += 1
        if int(obs["target_gate"]) == -1:  # all gates passed
            self._finished = True
        return self._finished

    def render_callback(self, sim: object) -> None:
        """Draw a lightweight plan overlay into the live simulation.

        Kept lightweight to avoid GUI lag: the currently *registered* gate (yellow)
        and obstacle (orange pole) positions from the observation, the current MPC
        target point (red), the fixed gate entry/center/exit waypoints (magenta),
        and the planned trajectory (green line, heavily downsampled). The
        intermediate optimization waypoints are not drawn.
        """
        # Registered gate / obstacle positions (snap to true pose on reveal).
        obs = self._last_obs
        if obs is not None:
            draw_points(
                sim, np.atleast_2d(obs["gates_pos"]),
                rgba=np.array([1.0, 1.0, 0.0, 1.0]), size=0.08,
            )
            for opos in np.atleast_2d(obs["obstacles_pos"]):
                pole = np.array([[opos[0], opos[1], 0.0], [opos[0], opos[1], opos[2]]])
                orange = np.array([1.0, 0.5, 0.0, 1.0])
                draw_line(sim, pole, rgba=orange, start_size=6.0, end_size=6.0)

        # Current MPC target point.
        pos_ref, _ = self.planner.get_reference(self._tick)
        draw_points(sim, pos_ref[:1], rgba=np.array([1.0, 0.0, 0.0, 1.0]), size=0.06)

        # Fixed gate waypoints (entry / center / exit).
        raw = getattr(self.planner, "_raw_waypoints", None)
        if raw:
            fixed = np.array([np.asarray(p) for p, v in raw if v is not None])
            if len(fixed):
                draw_points(sim, fixed, rgba=np.array([1.0, 0.0, 1.0, 1.0]), size=0.05)

        # Planned trajectory (downsampled to ~30 segments to stay light).
        traj = self.planner.pos
        step = max(1, len(traj) // 30)
        draw_line(sim, traj[::step], rgba=np.array([0.0, 1.0, 0.0, 1.0]))

    def episode_callback(self):
        """Reset the trajectory index after an episode."""
        self._tick = 0

    def episode_reset(self):
        """Reset internal state for a new episode."""
        self._tick = 0
        self._finished = False
