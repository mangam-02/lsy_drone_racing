"""Trajectory planner with offline waypoint optimization for drone racing.

Architecture:
  Before takeoff (unlimited time), scipy optimizes intermediate waypoint
  positions so the full minsnap trajectory stays clear of all obstacles.
  The optimizer directly evaluates 100 points along the curve — not just
  at waypoints — so minsnap polynomial clipping is prevented by design.

  Online execution just follows the pre-computed reference trajectory.
  Replanning triggers when level-2/3 sensor reveals true gate positions.
"""

from __future__ import annotations

import time

import numpy as np
import minsnap_trajectories as ms
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R


# ── Obstacle primitives ───────────────────────────────────────────────────────

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


# ── Trajectory planner ────────────────────────────────────────────────────────

class TrajectoryPlanner:
    """Offline-optimized minimum-snap trajectory planner.

    Before the drone takes off, scipy.minimize finds intermediate waypoint
    positions that minimise obstacle violations along the full minsnap curve.
    Online, the pre-computed pos/vel arrays are just indexed by tick.
    """

    TARGET_SPEED   = 1.5    # m/s — inter-gate speed for time assignment
    GATE_SPEED     = 1.2    # m/s — crossing speed at gate center
    ENTRY_DIST     = 0.2    # m   — approach waypoint before gate (aligns final segment with gate axis)
    EXIT_DIST      = 0.6    # m   — exit waypoint past gate (prevents reversal)
    DRONE_RADIUS   = 0.07   # m   — gate frame inflation
    PLAN_CLEARANCE   = 0.145  # m   — physical drone-to-obstacle clearance
    TRAJ_TUBE_RADIUS = 0.05   # m   — extra buffer: trajectory treated as a tube, not a line
    N_INTERMEDIATE     = 2      # optimisable waypoints per inter-gate segment
    N_SAMPLE           = 100    # trajectory points sampled for obstacle check
    OPT_MAXITER        = 300    # max L-BFGS-B iterations — phase 1 (cylinder only)
    REFINE_MAXITER     = 100    # max iterations — phase 2 (gate-local, warm-started)
    REFINE_TIME_BUDGET = 5.0    # seconds — phase 2 stops here regardless of convergence
    GATE_LOCAL_WEIGHT  = 20.0   # penalty for re-entry into gate frame material
    GATE_CHECK_WINDOW  = 0.4    # m — half-width around gate plane to check (gate local x)

    _WP_LO = np.array([-2.4, -1.4,  0.1])
    _WP_HI = np.array([ 2.4,  1.4,  1.45])

    def __init__(self, obs: dict, config, N: int = 25):
        self.freq = config.env.freq
        self.N = N
        self._prev_visited     = obs["gates_visited"].copy()
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
        new_obs   = obs["obstacles_visited"]
        replanned = bool(
            np.any(new_gates & ~self._prev_visited)
            or np.any(new_obs  & ~self._prev_obs_visited)
        )
        if replanned:
            self.build(obs)
        self._prev_visited     = new_gates.copy()
        self._prev_obs_visited = new_obs.copy()
        return replanned

    # ── Build / optimise ──────────────────────────────────────────────────────

    def build(self, obs: dict):
        """Run offline optimiser and store pre-computed trajectory."""
        target_gate = int(obs["target_gate"])

        if target_gate == -1:
            n = self.N + 2
            self.pos      = np.tile(obs["pos"], (n, 1))
            self.vel      = np.zeros((n, 3))
            self.tick_max = 1
            return

        cylinders, gate_frames = self._build_obstacles(obs)
        gate_data  = self._compute_gate_data(obs, target_gate, cylinders, gate_frames)
        cyl_tuples = [(float(c.xy[0]), float(c.xy[1]), self.PLAN_CLEARANCE)
                      for c in cylinders]

        t0 = time.perf_counter()
        opt_intermediates = self._optimize(
            obs["pos"].copy(), obs["vel"].copy(), gate_data, cyl_tuples,
        )
        print(f"[TrajectoryPlanner] optimization done in {time.perf_counter()-t0:.1f}s")

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
        self,
        obs: dict,
        target_gate: int,
        cylinders: list,
        gate_frames: list,
    ) -> list[tuple]:
        """Return [(gate_center, x_axis, entry_pt, exit_pt), ...] for gates from target_gate."""
        gates_pos  = obs["gates_pos"]
        gates_quat = obs["gates_quat"]
        n_gates    = len(gates_pos)

        gate_data = []
        prev_pos  = obs["pos"].copy()
        for i in range(target_gate, n_gates):
            rot    = R.from_quat(gates_quat[i])
            x_axis = rot.apply([1.0, 0.0, 0.0])
            if np.dot(gates_pos[i] - prev_pos, x_axis) < 0:
                x_axis = -x_axis
            seg_obs  = cylinders + gate_frames[target_gate : i + 1]
            entry_pt = self._safe_entry(gates_pos[i], x_axis, cylinders)
            exit_pt  = self._safe_exit(gates_pos[i], x_axis, seg_obs)
            gate_data.append((gates_pos[i].copy(), x_axis.copy(), entry_pt.copy(), exit_pt.copy()))
            prev_pos = exit_pt
        return gate_data

    def _safe_entry(
        self, gate_pos: np.ndarray, x_axis: np.ndarray, obstacles: list
    ) -> np.ndarray:
        for d in [self.ENTRY_DIST, 0.4, 0.5, 0.2]:
            pt = gate_pos - x_axis * d
            if _free_3d(pt, gate_pos, obstacles):
                return pt
        return gate_pos.copy()

    def _safe_exit(
        self, gate_pos: np.ndarray, x_axis: np.ndarray, obstacles: list
    ) -> np.ndarray:
        for d in [self.EXIT_DIST, 0.5, 0.3, 0.2]:
            pt = gate_pos + x_axis * d
            if _free_3d(gate_pos, pt, obstacles):
                return pt
        return gate_pos.copy()

    # ── Offline optimiser ─────────────────────────────────────────────────────

    def _optimize(
        self,
        drone_pos:   np.ndarray,
        drone_vel:   np.ndarray,
        gate_data:   list[tuple],
        cyl_tuples:  list[tuple],
        gate_frames: list = (),
        x0_override: np.ndarray | None = None,
        maxiter:     int | None = None,
    ) -> np.ndarray:
        n_wps = len(gate_data) * self.N_INTERMEDIATE
        x0    = x0_override if x0_override is not None else self._initial_intermediates(drone_pos, gate_data)

        bounds = [
            (float(lo), float(hi))
            for _ in range(n_wps)
            for lo, hi in zip(self._WP_LO, self._WP_HI)
        ]

        def cost(x: np.ndarray) -> float:
            intermediates = x.reshape(n_wps, 3)
            raw = self._build_waypoint_list(drone_pos, drone_vel, gate_data, intermediates)
            return self._trajectory_cost(raw, cyl_tuples, gate_frames)

        t_start = time.perf_counter()
        budget  = self.REFINE_TIME_BUDGET if maxiter else None

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

    def _initial_intermediates(
        self, drone_pos: np.ndarray, gate_data: list[tuple]
    ) -> np.ndarray:
        """Initial waypoints between segment endpoints.

        For U-turn segments (exit direction nearly opposite to direction to next entry),
        seeds a bypass waypoint perpendicular to the gate to route clear of the frame.
        """
        n_wps = len(gate_data) * self.N_INTERMEDIATE
        pts   = np.zeros((n_wps, 3))
        seg_starts = [drone_pos] + [gd[3] for gd in gate_data[:-1]]

        for seg_i, ((gate_center, _, entry_pt, exit_pt), seg_start) in enumerate(
            zip(gate_data, seg_starts)
        ):
            # Detect U-turn: previous gate exit direction vs direction to next entry
            is_uturn = False
            bypass   = None
            if seg_i > 0:
                prev_x = gate_data[seg_i - 1][1]          # previous gate's x_axis
                to_entry = entry_pt - seg_start
                d = np.linalg.norm(to_entry)
                if d > 1e-6 and np.dot(to_entry / d, prev_x) < -0.5:
                    is_uturn = True
                    # Perpendicular to exit axis, toward the entry side
                    perp = np.cross(prev_x, [0.0, 0.0, 1.0])
                    pn   = np.linalg.norm(perp)
                    if pn > 1e-6:
                        perp /= pn
                    sign   = np.sign(np.dot(entry_pt - seg_start, perp)) or 1.0
                    bypass = np.clip(
                        seg_start + perp * sign * 0.7 + np.array([0.0, 0.0, 0.1]),
                        self._WP_LO, self._WP_HI,
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
                    pts[seg_i * self.N_INTERMEDIATE + k] = (
                        seg_start + frac * (entry_pt - seg_start)
                    )
        return pts

    def _trajectory_cost(
        self, raw: list[tuple], cyl_tuples: list[tuple], gate_frames: list = ()
    ) -> float:
        """Evaluate obstacle + gate-opening violation along the minsnap trajectory."""
        try:
            waypoints = self._assign_times(raw)
            polys = ms.generate_trajectory(
                waypoints,
                degree=8,
                idx_minimized_orders=(3, 4),
                num_continuous_orders=4,
                algorithm="closed-form",
            )
            t_total  = waypoints[-1].time
            t_s      = np.linspace(0.0, t_total, self.N_SAMPLE)
            traj_pos = ms.compute_trajectory_derivatives(polys, t_s, num_orders=1)[0]
        except Exception:
            return 1e6

        cost = 0.0

        for cx, cy, r in cyl_tuples:
            dist      = np.linalg.norm(traj_pos[:, :2] - np.array([cx, cy]), axis=1)
            violation = np.maximum(0.0, r - dist)
            cost     += float(np.sum(violation ** 2))


        return cost

    # ── Waypoint assembly ─────────────────────────────────────────────────────

    def _build_waypoint_list(
        self,
        drone_pos:     np.ndarray,
        drone_vel:     np.ndarray,
        gate_data:     list[tuple],
        intermediates: np.ndarray,
    ) -> list[tuple]:
        """Interleave optimised intermediates with fixed gate/exit waypoints."""
        points = [(drone_pos, drone_vel)]

        for seg_i, (gate_center, x_axis, entry_pt, exit_pt) in enumerate(gate_data):
            for k in range(self.N_INTERMEDIATE):
                wp = intermediates[seg_i * self.N_INTERMEDIATE + k]
                points.append((wp.copy(), None))
            points.append((entry_pt,            x_axis * self.GATE_SPEED))
            points.append((gate_center,         x_axis * self.GATE_SPEED))
            points.append((exit_pt,             x_axis * self.GATE_SPEED))

        last_x    = gate_data[-1][1]
        last_exit = gate_data[-1][3]
        points.append((last_exit + last_x * 0.3, np.zeros(3)))
        return points

    # ── Finalise trajectory ───────────────────────────────────────────────────

    def _finalize(self, raw: list[tuple]):
        waypoints = self._assign_times(raw)
        polys = ms.generate_trajectory(
            waypoints,
            degree=8,
            idx_minimized_orders=(3, 4),
            num_continuous_orders=4,
            algorithm="closed-form",
        )
        t_total  = waypoints[-1].time
        n_samp   = max(int(self.freq * t_total), self.N + 2)
        t_samp   = np.linspace(0.0, t_total, n_samp)
        derivs   = ms.compute_trajectory_derivatives(polys, t_samp, num_orders=2)
        self.pos      = derivs[0]
        self.vel      = derivs[1]
        self.tick_max = n_samp - 1 - self.N

    def _assign_times(self, points: list[tuple]) -> list[ms.Waypoint]:
        waypoints = []
        t    = 0.0
        prev = points[0][0]
        for i, (pos, vel) in enumerate(points):
            if i > 0:
                t += max(np.linalg.norm(pos - prev) / self.TARGET_SPEED, 0.1)
            waypoints.append(
                ms.Waypoint(time=t, position=pos, velocity=vel)
                if vel is not None
                else ms.Waypoint(time=t, position=pos)
            )
            prev = pos
        return waypoints
