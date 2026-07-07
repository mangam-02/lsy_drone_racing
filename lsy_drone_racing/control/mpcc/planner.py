"""B-spline path planner and obstacle/gate geometry for the MPCC controller.

Contains the :class:`SimplePlanner` (waypoint placement, obstacle-aware optimisation,
weighted cubic B-spline fit, trapezoidal speed profile) and the small geometry helpers
:class:`_Cylinder` / :class:`_GateFrame` used by the planner cost, the controller's capsule
avoidance and the :mod:`.viz` overlay. All tunable constants live in :mod:`mpcc.config`.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.interpolate import splev, splprep
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.mpcc import config as cfg

# ── Obstacle / gate geometry helpers ─────────────────────────────────────────────


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

    def penetration(self, points: np.ndarray) -> np.ndarray:
        """Per-point penetration depth (m) into the solid frame material.

        Returns 0 where a point is safe — i.e. far from the gate plane
        (outside the depth slab), inside the opening (fly-through), or outside the
        outer square (flying around). Inside the square frame ring it returns the
        distance to the nearest safe edge, so the optimizer is pushed either into
        the opening or out past the frame.

        The frame is a square annulus in the gate plane, so the Chebyshev distance
        ``m = max(|y|, |z|)`` cleanly separates opening (m<=hi), frame (hi<m<ho)
        and free space (m>=ho).
        """
        local = self.rot_inv.apply(points - self.center)
        lx, ly, lz = np.abs(local[:, 0]), np.abs(local[:, 1]), np.abs(local[:, 2])
        m = np.maximum(ly, lz)
        in_material = (lx <= self.hd) & (m > self.hi) & (m < self.ho)
        depth = np.minimum(m - self.hi, self.ho - m)
        return np.where(in_material, depth, 0.0)


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

    # Tuning constants — values defined in mpcc/config.py (single source of truth).
    TARGET_SPEED = cfg.TARGET_SPEED
    V_EDGE = cfg.V_EDGE
    ACCEL_DIST = cfg.ACCEL_DIST
    DECEL_DIST = cfg.DECEL_DIST
    APPROACH_DIST = cfg.APPROACH_DIST
    DEPART_DIST = cfg.DEPART_DIST
    GATE_WP_MIN_DIST = cfg.GATE_WP_MIN_DIST
    GATE_EXIT_MIN_DIST = cfg.GATE_EXIT_MIN_DIST
    GATE_WP_MAX_SHIFT = cfg.GATE_WP_MAX_SHIFT
    GATE_Z_BIAS = cfg.GATE_Z_BIAS
    DRONE_RADIUS = cfg.DRONE_RADIUS
    FRAME_MARGIN = cfg.FRAME_MARGIN
    OBSTACLE_RADIUS = cfg.OBSTACLE_RADIUS
    OBSTACLE_BUFFER = cfg.OBSTACLE_BUFFER
    PLAN_CLEARANCE = cfg.PLAN_CLEARANCE
    GATE_FRAME_WEIGHT = cfg.GATE_FRAME_WEIGHT
    CYL_WEIGHT = cfg.CYL_WEIGHT
    DEVIATION_WEIGHT = cfg.DEVIATION_WEIGHT
    N_INTERMEDIATE = cfg.N_INTERMEDIATE
    N_SAMPLE = cfg.N_SAMPLE
    OPT_MAXITER = cfg.OPT_MAXITER
    MAX_OPT_TIME = cfg.MAX_OPT_TIME
    BSPLINE_DEGREE = cfg.BSPLINE_DEGREE
    SMOOTHING = cfg.SMOOTHING
    W_ANCHOR = cfg.W_ANCHOR
    W_GATE = cfg.W_GATE
    W_FREE = cfg.W_FREE
    MIN_REF_Z = cfg.MIN_REF_Z

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
        data = []
        for i in range(len(gates_pos)):
            x_axis = R.from_quat(gates_quat[i]).apply([1.0, 0.0, 0.0])
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
        the approach line but slightly outside the entry point's own keep-out would otherwise
        leave the entry unmoved while the path still grazes the pole. Checking the segment lets
        the shorten/nudge actually push the entry aside. The free intermediates handle the path
        BEFORE the entry, so only the entry→center stretch needs this.
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
        # Also keep the raw geometric path (B-spline + arc-length LUT) so the contouring
        # controller (MPCC) can query the path by arc length θ; pos/vel feed the viz path
        # overlay and the analysis/figure scripts.
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
