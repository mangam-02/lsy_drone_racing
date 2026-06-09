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
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import casadi as ca
import numpy as np
import scipy.linalg
from acados_template import AcadosOcp, AcadosOcpSolver
from crazyflow.sim.visualize import draw_line, draw_points
from drone_models.core import load_params
from drone_models.utils.rotation import ang_vel2rpy_rates
from scipy.interpolate import splev, splprep
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.attitude_mpc import create_acados_model

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ── MPC solver with a ground (z) hard constraint ────────────────────────────────


# Slack penalty weights for the soft obstacle/gate keep-out constraints. 1e4 enforces
# the clearance closely while staying solvable (1e5 makes the QP too stiff → failures).
_CON_SLACK_LIN = 1e4
_CON_SLACK_QUAD = 1e4

# Per-axis speed cap (m/s), enforced as a SOFT state bound (so a transient overspeed never
# makes the QP infeasible). Stops the drone ever accelerating away fast when it is far off
# the reference — essential for hardware safety. Set above cruise (1.2) so it only bites on
# a runaway, not during normal tracking.
_V_MAX = 2.0
_VEL_SLACK_LIN = 1e3
_VEL_SLACK_QUAD = 1e3

# Smoothing scale (m) for the gate keep-out constraint (option B). The original
# constraint used ``fmax``/``fabs``, whose kinks gave the Gauss-Newton/HPIPM solver
# jumping gradients → it failed to converge (status=2) and ran to max iterations every
# tick. We replace them by C¹ surrogates so the QP converges. 0.02 m is small enough to
# track the true geometry yet large enough to round off the corners for the solver.
_CON_SMOOTH_EPS = 0.02


def _smooth_abs(x, eps: float = _CON_SMOOTH_EPS):
    """C¹ approximation of |x| (=sqrt(x²+ε²)); avoids the kink at 0."""
    return ca.sqrt(x * x + eps * eps)


def _smooth_max(a, b, eps: float = _CON_SMOOTH_EPS):
    """C¹ approximation of max(a, b)."""
    return 0.5 * (a + b + ca.sqrt((a - b) ** 2 + eps * eps))


def _smooth_relu(z, eps: float = _CON_SMOOTH_EPS):
    """C¹ approximation of max(0, z): ~0 for z ≪ 0, ~z for z ≫ 0."""
    return 0.5 * (z + ca.sqrt(z * z + eps * eps))


def _create_ocp_solver(
    Tf: float, N: int, parameters: dict, z_min: float = 0.0, z_max: float = 2.5,
    verbose: bool = False, n_obstacles: int = 0, n_gates: int = 0,
    obs_clearance: float = 0.15, gate_drone_r: float = 0.07,
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Acados OCP/solver that tracks a reference and keeps the drone above ground.

    Same formulation as ``attitude_mpc.create_ocp_solver`` but adds a hard state
    bound on the z position (index 2): ``z_min <= z <= z_max`` on every shooting
    node. With ``z_min = 0`` the MPC can never plan a path through the floor.

    When ``n_obstacles``/``n_gates`` > 0 it also adds *soft* (slacked) nonlinear
    path constraints that make the MPC itself steer clear of poles and gate frames,
    even when tracking error pushes the drone off the planned path:

    * Obstacles: keep the xy position outside a cylinder of radius ``obs_clearance``.
    * Gate frames: keep out of the solid frame material while leaving the opening and
      the fly-around region free (``fmin`` of the in-ring and near-plane depths is
      positive only inside the material).

    The obstacle/gate centers and gate axes are model parameters (``model.p``) set per
    solve from the live observation. Everything is soft so the QP stays feasible.
    """
    ocp = AcadosOcp()
    ocp.model = create_acados_model(parameters)
    ocp.model.name = "mpc_planner_ground"  # distinct generated-code name

    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    ny = nx + nu
    ny_e = nx
    ocp.solver_options.N_horizon = N

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    # Position weight raised (xy 50→250, z 400→450) and velocity 10→25 so the drone
    # tracks the reference far more tightly — much less inward drift in gate corners,
    # which was a main cause of clipping frames/poles.
    Q = np.diag([250.0, 250.0, 450.0, 1.0, 1.0, 1.0, 25.0, 25.0, 25.0, 5.0, 5.0, 5.0])
    Rmat = np.diag([1.0, 1.0, 1.0, 50.0])
    ocp.cost.W = scipy.linalg.block_diag(Q, Rmat)
    ocp.cost.W_e = Q.copy()

    Vx = np.zeros((ny, nx))
    Vx[0:nx, 0:nx] = np.eye(nx)
    ocp.cost.Vx = Vx
    Vu = np.zeros((ny, nu))
    Vu[nx : nx + nu, :] = np.eye(nu)
    ocp.cost.Vu = Vu
    Vx_e = np.zeros((ny_e, nx))
    Vx_e[0:nx, 0:nx] = np.eye(nx)
    ocp.cost.Vx_e = Vx_e
    ocp.cost.yref, ocp.cost.yref_e = np.zeros((ny,)), np.zeros((ny_e,))

    # State constraints: z floor/ceiling (2), rpy < ~30° (3,4,5), and a per-axis speed cap
    # (6,7,8). The speed cap is softened (idxsbx) so a transient overspeed never makes the
    # QP infeasible; it stops the drone ever accelerating away fast when it is far off the
    # reference (hardware safety). Set just above cruise so normal tracking is unaffected.
    ocp.constraints.lbx = np.array([z_min, -0.5, -0.5, -0.5, -_V_MAX, -_V_MAX, -_V_MAX])
    ocp.constraints.ubx = np.array([z_max, 0.5, 0.5, 0.5, _V_MAX, _V_MAX, _V_MAX])
    ocp.constraints.idxbx = np.array([2, 3, 4, 5, 6, 7, 8])
    ocp.constraints.idxsbx = np.array([4, 5, 6])  # soften the three velocity bounds

    # Input constraints (rpy + collective thrust).
    ocp.constraints.lbu = np.array([-0.5, -0.5, -0.5, parameters["thrust_min"] * 4])
    ocp.constraints.ubu = np.array([0.5, 0.5, 0.5, parameters["thrust_max"] * 4])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    # Slack costs are accumulated across all soft constraints (soft velocity bounds first,
    # then the optional obstacle/gate keep-out below), in acados' [sbx, sh] order.
    zl_parts = [_VEL_SLACK_LIN * np.ones(3)]
    zu_parts = [_VEL_SLACK_LIN * np.ones(3)]
    Zl_parts = [_VEL_SLACK_QUAD * np.ones(3)]
    Zu_parts = [_VEL_SLACK_QUAD * np.ones(3)]

    # Soft obstacle/gate-frame avoidance (one constraint each), parametrised by the
    # live obstacle/gate poses set per solve. Obstacles first, then gates.
    #   * Obstacle: squared xy distance >= obs_clearance**2 (lower-bounded).
    #   * Gate: while near the gate plane, stay inside the opening (lateral offset
    #     m <= hi). Enforced only for the *target* gate (per-gate flag), since other
    #     gates are flown around, not through. A smooth plane-proximity weight keeps a
    #     lateral gradient so the drone is pushed into the opening rather than into the
    #     frame bars. Gate params: [is_target, center(3), x/y/z axis(9)] = 13 each.
    n_con = n_obstacles + n_gates
    if n_con > 0:
        n_p = 2 * n_obstacles + 13 * n_gates
        p = ca.MX.sym("p", n_p)  # match the model's MX symbol type
        ocp.model.p = p
        ocp.parameter_values = np.zeros(n_p)
        px, py, pz = ocp.model.x[0], ocp.model.x[1], ocp.model.x[2]

        h_list = []
        for i in range(n_obstacles):
            h_list.append((px - p[2 * i]) ** 2 + (py - p[2 * i + 1]) ** 2)

        hi = _GateFrame.OPENING / 2 - gate_drone_r  # opening half-width (drone-inflated)
        hd = _GateFrame.DEPTH / 2 + gate_drone_r  # plane-proximity length scale
        base = 2 * n_obstacles
        for j in range(n_gates):
            o = base + 13 * j
            flag = p[o]
            c, xax, yax, zax = p[o + 1 : o + 4], p[o + 4 : o + 7], p[o + 7 : o + 10], p[o + 10 : o + 13]
            dxyz = ca.vertcat(px - c[0], py - c[1], pz - c[2])
            lx, ly, lz = ca.dot(dxyz, xax), ca.dot(dxyz, yax), ca.dot(dxyz, zax)
            m = _smooth_max(_smooth_abs(ly), _smooth_abs(lz))  # smoothed Chebyshev offset
            plane_w = ca.exp(-(lx**2) / hd**2)  # ~1 at the plane, decays away from it
            h_list.append(flag * plane_w * _smooth_relu(m - hi))  # > 0 only off-opening near plane

        ocp.model.con_h_expr = ca.vertcat(*h_list)
        ocp.constraints.lh = np.concatenate(
            [np.full(n_obstacles, obs_clearance**2), np.full(n_gates, -1e9)]
        )
        ocp.constraints.uh = np.concatenate(
            [np.full(n_obstacles, 1e9), np.full(n_gates, 0.0)]
        )
        # Slack all of them so an infeasible spot is penalised, never rejected.
        ocp.constraints.idxsh = np.arange(n_con)
        zl_parts.append(_CON_SLACK_LIN * np.ones(n_con))
        zu_parts.append(_CON_SLACK_LIN * np.ones(n_con))
        Zl_parts.append(_CON_SLACK_QUAD * np.ones(n_con))
        Zu_parts.append(_CON_SLACK_QUAD * np.ones(n_con))

    ocp.cost.zl = np.concatenate(zl_parts)
    ocp.cost.zu = np.concatenate(zu_parts)
    ocp.cost.Zl = np.concatenate(Zl_parts)
    ocp.cost.Zu = np.concatenate(Zu_parts)

    ocp.constraints.x0 = np.zeros((nx))

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
        ocp, json_file="c_generated_code/mpc_planner_ground.json",
        verbose=verbose, build=True, generate=True,
    )
    return solver, ocp


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

    TARGET_SPEED = 1.0  # m/s — cruise speed (lowered from 1.5: less overshoot off the
    #                          reference in tight gate corners → fewer frame/pole clips)
    V_EDGE = 0.6  # m/s — speed at trajectory start/end (ramp endpoints)
    ACCEL_DIST = 0.8  # m — arc length over which to ramp up to cruise speed
    DECEL_DIST = 0.8  # m — arc length over which to ramp down at the end
    GATE_SPEED = 0.85  # m/s — crossing speed tag at gate waypoints (marks them fixed)
    APPROACH_DIST = 0.35  # m — fixed orthogonal waypoint before each gate (gate normal)
    DEPART_DIST = 0.35  # m — fixed orthogonal waypoint after each gate (gate normal)
    MIN_GATE_OFFSET = 0.10  # m — smallest entry/exit offset; the waypoint is pulled in to
    #                           at most this (just past the frame depth) but never onto the
    #                           gate center, so the spline always crosses straight through
    # On an in-flight replan, lead the new path out along the current velocity before it
    # curves toward the gate, so a moving drone is never asked to reverse at its current
    # position (the cause of the "fly backwards / zigzag" on replan near a gate). The
    # lead-in length scales with speed (faster ⇒ commit further), and it carries a fit
    # weight (like a gate waypoint) so the spline actually follows it instead of smoothing
    # it away — a weak/short lead-in let the path still form a hairpin at the drone.
    MOMENTUM_TIME = 0.30  # s — lead-in length = speed × this
    MOMENTUM_LEAD_MIN = 0.15  # m — lower clamp on the lead-in length
    MOMENTUM_LEAD_MAX = 0.50  # m — upper clamp on the lead-in length
    MOMENTUM_MIN_SPEED = 0.4  # m/s — below this (e.g. takeoff) no lead-in waypoint is added
    DRONE_RADIUS = 0.09  # m — drone half-extent incl. props (was 0.07: the drone is wide,
    #                         so plan further from poles AND gate frames). Feeds both the
    #                         obstacle clearance and the gate-frame inflation.
    FRAME_MARGIN = 0.08  # m — extra gate-frame inflation on top of DRONE_RADIUS; keeps the
    #                         0.40 m opening threadable as the drone radius grew (opening
    #                         half-width hi = 0.20 − 0.09 − 0.08 = 0.03, i.e. central crossing)
    OBSTACLE_RADIUS = 0.015  # m — physical pole radius (0.03 m diameter)
    OBSTACLE_BUFFER = 0.22  # m — extra gap drone↔obstacle surface; with DRONE_RADIUS this
    #                            keeps the reference well clear of poles (planner was still
    #                            routing too close on level 2)
    GATE_FRAME_WEIGHT = 60.0  # penalty for entering gate frame material (raised from 10:
    #                            on U-turns after a gate the smoothed spline used to clip
    #                            the gate's own frame; now strongly routed around it)
    CYL_WEIGHT = 30.0  # penalty weight for violating obstacle clearance
    # Min center-to-center distance the trajectory must keep from an obstacle:
    # obstacle radius + drone radius + safety buffer.
    PLAN_CLEARANCE = OBSTACLE_RADIUS + DRONE_RADIUS + OBSTACLE_BUFFER
    # Extra clearance (m) for objects whose true pose is NOT yet revealed. Matches the
    # track randomization range (gate/obstacle ±0.15 m): until the drone senses an
    # object within sensor_range it only knows the nominal pose, so the path keeps this
    # much further away to tolerate the shift. A reveal triggers a replan that tightens.
    UNCERTAINTY_MARGIN = 0.0
    N_INTERMEDIATE = 4  # optimisable waypoints per inter-gate segment
    N_SAMPLE = 100  # trajectory points sampled for obstacle check
    OPT_MAXITER = 300  # max L-BFGS-B iterations
    MAX_OPT_TIME = 0.25  # seconds — wall-clock budget for the optimizer (online replan)
    BSPLINE_DEGREE = 3  # cubic B-spline (C2 continuous)
    FIT_OVERSHOOT_MARGIN = 0.5  # m — if the fitted spline leaves the waypoint hull by more
    #                             than this, it's a splprep overshoot artifact; _fit retries
    #                             with stronger smoothing (prevents "wild" replanned curves)
    SMOOTHING = 0.15  # splprep smoothing factor s (0 = interpolate, larger = smoother).
    #                   Kept low: higher s let the spline cut the gate corners on tight
    #                   U-turns and clip frames. The replan jaggedness was actually the
    #                   optimiser flinging waypoints (fixed via DEVIATION_WEIGHT), not the
    #                   spline fit.
    W_ANCHOR = 30.0  # fit weight for the start waypoint (must hold)
    W_GATE = 4.0  # fit weight for gate entry/center/exit (loose enough to round corners)
    W_FREE = 1.0  # fit weight for optimized intermediates (free to smooth)
    DEVIATION_WEIGHT = 3.0  # penalty pulling intermediates toward the straight-line path.
    #                         Raised from 2.0 (but kept below 6, which over-anchored and
    #                         blocked the necessary gate U-turn detours): enough to stop the
    #                         optimiser flinging free waypoints to the arena corners (wild
    #                         zigzag on replan) while still allowing real obstacle detours.
    WP_MARGIN = 0.5  # m — how far optimised waypoints may leave the gate-to-gate bbox
    #                      (tightened from 0.6 to bound stray detours, but not so tight it
    #                      blocks valid avoidance)
    MIN_REF_Z = 0.1  # m — hard floor for the reference trajectory (never plan below this)

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

    def build(self, obs: dict, v_start: float | None = None):
        """Run offline optimiser and store pre-computed trajectory (synchronous).

        Thin wrapper that computes a trajectory and applies it to ``self`` in one
        step. Used for the initial (takeoff) build; online replanning instead calls
        :meth:`_compute_trajectory` in a worker thread and :meth:`_apply_trajectory`
        on the main thread (see :class:`MPCPlanner`).
        """
        self._apply_trajectory(self._compute_trajectory(obs, v_start))

    def _compute_trajectory(self, obs: dict, v_start: float | None = None) -> dict:
        """Compute a trajectory without mutating ``self`` (thread-safe worker body).

        Returns a result dict with keys ``pos``/``vel``/``tick_max``/``raw_waypoints``
        that :meth:`_apply_trajectory` can install. ``v_start`` seeds the speed
        profile so a replanned trajectory continues at the drone's current speed
        instead of dropping back to ``V_EDGE`` (defaults to ``V_EDGE``).
        """
        if v_start is None:
            v_start = self.V_EDGE
        target_gate = int(obs["target_gate"])

        if target_gate == -1:
            n = self.N + 2
            return {
                "pos": np.tile(obs["pos"], (n, 1)),
                "vel": np.zeros((n, 3)),
                "tick_max": 1,
                "raw_waypoints": [(obs["pos"].copy(), np.zeros(3))],
            }

        cylinders, gate_frames = self._build_obstacles(obs)
        gate_data = self._compute_gate_data(obs, target_gate, cylinders, gate_frames)
        cyl_tuples = [(float(c.xy[0]), float(c.xy[1]), c.radius) for c in cylinders]

        # Constrain the optimiser to a box around the actual gate-to-gate corridor and
        # anchor it to the straight-line path, so it can't wander off into long detours.
        wp_lo, wp_hi = self._waypoint_bounds(obs["pos"].copy(), gate_data)
        straight = self._straight_intermediates(obs["pos"].copy(), gate_data).flatten()

        t0 = time.perf_counter()
        opt_intermediates = self._optimize(
            obs["pos"].copy(), obs["vel"].copy(), gate_data, cyl_tuples, gate_frames,
            wp_lo=wp_lo, wp_hi=wp_hi, reg_anchor=straight,
        )
        print(f"[BSplinePlanner] optimization done in {(time.perf_counter() - t0) * 1e3:.3f} ms")

        raw = self._build_waypoint_list(
            obs["pos"].copy(), obs["vel"].copy(), gate_data, opt_intermediates
        )
        result = self._finalize(raw, v_start)
        result["raw_waypoints"] = raw
        return result

    def _apply_trajectory(self, result: dict) -> None:
        """Install a trajectory computed by :meth:`_compute_trajectory` (main thread)."""
        self.pos = result["pos"]
        self.vel = result["vel"]
        self.tick_max = result["tick_max"]
        self._raw_waypoints = result["raw_waypoints"]

    def _build_obstacles(self, obs: dict) -> tuple[list, list]:
        # Obstacles outside sensor_range report their NOMINAL pose, which the track
        # randomization can shift by up to UNCERTAINTY_MARGIN. Until a pole is actually
        # seen (obstacles_visited), inflate its keep-out radius by that margin so the
        # path tolerates the shift; a reveal then triggers a replan with the true pose.
        seen = np.asarray(obs["obstacles_visited"], bool)
        cylinders = [
            _Cylinder(
                pos=opos,
                radius=self.PLAN_CLEARANCE + (0.0 if s else self.UNCERTAINTY_MARGIN),
                height=1.52,
            )
            for opos, s in zip(obs["obstacles_pos"], seen)
        ]
        gate_frames = [
            _GateFrame(gpos, gquat, drone_r=self.DRONE_RADIUS + self.FRAME_MARGIN)
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
        """Before-gate waypoint along the gate normal, pulled toward the gate if blocked.

        Starting at APPROACH_DIST we shrink the offset until the segment to the gate is
        clear. Because a blocking obstacle is usually *beyond* the waypoint (further from
        the gate), pulling the point in moves it away from the obstacle. We never collapse
        onto the gate center: a missing entry point removes the straight-in guidance and
        lets the spline cut the corner into the frame.
        """
        return self._pull_in_waypoint(gate_pos, -x_axis, self.APPROACH_DIST, obstacles)

    def _safe_exit(self, gate_pos: np.ndarray, x_axis: np.ndarray, obstacles: list) -> np.ndarray:
        """After-gate waypoint along the gate normal, pulled toward the gate if blocked.

        Same idea as :meth:`_safe_entry`: shrink the offset until clear instead of
        collapsing onto the gate center (which caused the spline to turn inside the
        frame right at the gate on tight U-turns).
        """
        return self._pull_in_waypoint(gate_pos, x_axis, self.DEPART_DIST, obstacles)

    def _pull_in_waypoint(
        self, gate_pos: np.ndarray, direction: np.ndarray, d_max: float, obstacles: list
    ) -> np.ndarray:
        """Point at ``gate_pos + direction * d``, shrinking d from d_max until the segment
        to the gate is obstacle-free. Falls back to the minimum offset (never the center).
        """
        d = d_max
        while d >= self.MIN_GATE_OFFSET:
            pt = gate_pos + direction * d
            if _free_3d(gate_pos, pt, obstacles):
                return pt
            d -= 0.05
        return gate_pos + direction * self.MIN_GATE_OFFSET

    # ── Offline optimiser ─────────────────────────────────────────────────────

    def _optimize(
        self,
        drone_pos: np.ndarray,
        drone_vel: np.ndarray,
        gate_data: list[tuple],
        cyl_tuples: list[tuple],
        gate_frames: list = (),
        x0_override: np.ndarray | None = None,
        maxiter: int | None = None,
        wp_lo: np.ndarray | None = None,
        wp_hi: np.ndarray | None = None,
        reg_anchor: np.ndarray | None = None,
    ) -> np.ndarray:
        n_wps = len(gate_data) * self.N_INTERMEDIATE
        x0 = (
            x0_override
            if x0_override is not None
            else self._initial_intermediates(drone_pos, gate_data)
        )

        lo = self._WP_LO if wp_lo is None else wp_lo
        hi = self._WP_HI if wp_hi is None else wp_hi
        bounds = [
            (float(blo), float(bhi)) for _ in range(n_wps) for blo, bhi in zip(lo, hi)
        ]

        def cost(x: np.ndarray) -> float:
            intermediates = x.reshape(n_wps, 3)
            raw = self._build_waypoint_list(drone_pos, drone_vel, gate_data, intermediates)
            c = self._trajectory_cost(raw, cyl_tuples, gate_frames)
            # Direct-path regularisation: penalise straying from the straight line so
            # the optimiser only detours as far as the obstacles actually require.
            if reg_anchor is not None:
                c += self.DEVIATION_WEIGHT * float(np.sum((x - reg_anchor) ** 2))
            return c

        t_start = time.perf_counter()

        def _time_cb(xk: np.ndarray) -> None:
            if (time.perf_counter() - t_start) > self.MAX_OPT_TIME:
                raise StopIteration

        x0_flat = x0.flatten()
        result = minimize(
            cost,
            x0_flat,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter or self.OPT_MAXITER, "ftol": 1e-10, "gtol": 1e-7},
            callback=_time_cb,
        )
        # Robustness guard: L-BFGS-B can stop on the time budget at a point *worse* than it
        # started, which shows up as a wild, self-crossing replanned path (the drone then
        # jerks/flies off). Never accept a result worse than the straight-line / bypass
        # start — fall back to that sane initial guess instead.
        if not np.all(np.isfinite(result.x)) or cost(result.x) > cost(x0_flat):
            return x0.reshape(n_wps, 3)
        return result.x.reshape(n_wps, 3)

    def _segment_starts(self, drone_pos: np.ndarray, gate_data: list[tuple]) -> list:
        """Start point of each inter-gate segment: drone, then each gate's exit."""
        return [drone_pos] + [gd[3] for gd in gate_data[:-1]]

    def _straight_intermediates(self, drone_pos: np.ndarray, gate_data: list[tuple]) -> np.ndarray:
        """Straight-line waypoints: linear interpolation from each segment start to entry.

        This is the "direct path" baseline used both as the optimiser anchor
        (see :meth:`_optimize`) and as the non-U-turn seed.
        """
        n_wps = len(gate_data) * self.N_INTERMEDIATE
        pts = np.zeros((n_wps, 3))
        seg_starts = self._segment_starts(drone_pos, gate_data)
        for seg_i, ((_, _, entry_pt, _), seg_start) in enumerate(zip(gate_data, seg_starts)):
            for k in range(self.N_INTERMEDIATE):
                frac = (k + 1) / (self.N_INTERMEDIATE + 1)
                pts[seg_i * self.N_INTERMEDIATE + k] = seg_start + frac * (entry_pt - seg_start)
        return pts

    def _waypoint_bounds(
        self, drone_pos: np.ndarray, gate_data: list[tuple]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Box bounds for the optimised waypoints: bbox(path points) ± WP_MARGIN.

        Restricting the optimiser to a margin around the actual gate-to-gate corridor
        prevents it from sending waypoints off into the arena corners (which is what
        produces the occasional extremely long detour, especially on replan). The box
        is intersected with the global arena limits so it never exceeds them.
        """
        pts = [np.asarray(drone_pos)]
        for gate_center, _, entry_pt, exit_pt in gate_data:
            pts.extend([gate_center, entry_pt, exit_pt])
        pts = np.asarray(pts)
        lo = np.maximum(pts.min(axis=0) - self.WP_MARGIN, self._WP_LO)
        hi = np.minimum(pts.max(axis=0) + self.WP_MARGIN, self._WP_HI)
        return lo, hi

    def _initial_intermediates(self, drone_pos: np.ndarray, gate_data: list[tuple]) -> np.ndarray:
        """Initial waypoints between segment endpoints (optimiser starting point).

        Starts from the straight-line baseline; for U-turn segments (exit direction
        nearly opposite to direction to next entry) it overwrites the segment with a
        bypass waypoint perpendicular to the gate to route clear of the frame.
        """
        pts = self._straight_intermediates(drone_pos, gate_data)
        seg_starts = self._segment_starts(drone_pos, gate_data)

        for seg_i, ((gate_center, _, entry_pt, exit_pt), seg_start) in enumerate(
            zip(gate_data, seg_starts)
        ):
            if seg_i == 0:
                continue  # first segment never U-turns (drone already faces the gate)
            prev_x = gate_data[seg_i - 1][1]  # previous gate's x_axis
            to_entry = entry_pt - seg_start
            d = np.linalg.norm(to_entry)
            if d <= 1e-6 or np.dot(to_entry / d, prev_x) >= -0.5:
                continue  # not a U-turn
            # Perpendicular to exit axis, toward the entry side.
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
            pts[seg_i * self.N_INTERMEDIATE] = bypass
            for k in range(1, self.N_INTERMEDIATE):
                frac = k / self.N_INTERMEDIATE
                pts[seg_i * self.N_INTERMEDIATE + k] = np.clip(
                    bypass + frac * (entry_pt - bypass), self._WP_LO, self._WP_HI
                )
        return pts

    def _trajectory_cost(
        self, raw: list[tuple], cyl_tuples: list[tuple], gate_frames: list = ()
    ) -> float:
        """Evaluate obstacle + gate-frame violation along the smoothed B-spline.

        Cylinders use a soft clearance margin; gate frames penalise any point that
        enters the solid frame material. The frame opening is exempt, so the
        targeted gate stays fly-through while every gate is avoided otherwise.
        """
        try:
            tck = self._fit(raw)
            u = np.linspace(0.0, 1.0, self.N_SAMPLE)
            x, y, z = splev(u, tck)
            traj_pos = np.stack([x, y, z], axis=1)
        except Exception:
            return 1e6

        cost = 0.0
        for cx, cy, r in cyl_tuples:
            dist = np.linalg.norm(traj_pos[:, :2] - np.array([cx, cy]), axis=1)
            violation = np.maximum(0.0, r - dist)
            cost += self.CYL_WEIGHT * float(np.sum(violation**2))

        for gate in gate_frames:
            pen = gate.penetration(traj_pos)
            cost += self.GATE_FRAME_WEIGHT * float(np.sum(pen**2))
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

        For the current target gate (the first segment), any leading waypoint that
        the drone has already passed along the gate normal is dropped. This avoids
        a backward/sideways jog when replanning happens close to the gate (the
        entry point would otherwise sit behind the drone).
        """
        points = [(drone_pos, drone_vel)]

        # In-flight replan: nudge the path out along the current heading before it curves to
        # the gate, so a moving drone isn't asked to reverse abruptly. Added as a *free*
        # (None-tagged) waypoint, NOT a gate-weighted one: a hard, forced lead-in made the
        # spline pass exactly through it, which created a hook/zigzag whenever the velocity
        # didn't line up with the path to the gate. As a free point it only biases the
        # start; the optimiser/guard, the direction-aware resume and the governor handle
        # the rest.
        speed = float(np.linalg.norm(drone_vel))
        if speed > self.MOMENTUM_MIN_SPEED:
            lead = float(np.clip(speed * self.MOMENTUM_TIME, self.MOMENTUM_LEAD_MIN, self.MOMENTUM_LEAD_MAX))
            points.append((drone_pos + (drone_vel / speed) * lead, None))

        for seg_i, (gate_center, x_axis, entry_pt, exit_pt) in enumerate(gate_data):
            gate_wps = [entry_pt, gate_center, exit_pt]
            skip_intermediates = False
            if seg_i == 0:
                while len(gate_wps) > 1 and np.dot(drone_pos - gate_wps[0], x_axis) > 0:
                    gate_wps.pop(0)
                skip_intermediates = len(gate_wps) < 3  # entry was dropped

            if not skip_intermediates:
                for k in range(self.N_INTERMEDIATE):
                    wp = intermediates[seg_i * self.N_INTERMEDIATE + k]
                    points.append((wp.copy(), None))
            for wp in gate_wps:
                points.append((wp, x_axis * self.GATE_SPEED))

        last_x = gate_data[-1][1]
        last_exit = gate_data[-1][3]
        points.append((last_exit + last_x * 0.3, np.zeros(3)))
        return points

    # ── Finalise trajectory (smoothed B-spline + arc-length speed profile) ─────

    def _fit(self, raw: list[tuple]) -> tuple:
        """Fit a weighted, smoothing B-spline through the waypoints (option A).

        Gate/start waypoints get high weights so the curve stays tight to them;
        optimized intermediates get low weights so the spline can smooth out the
        sharp kinks that an exact interpolation would create.
        """
        P = np.array([np.asarray(p) for p, _ in raw], dtype=float)
        w = self._fit_weights(raw)
        # splprep requires distinct consecutive points; drop duplicates (they occur
        # when e.g. _safe_entry falls back onto the gate center).
        keep = np.concatenate(([True], np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-6))
        P, w = P[keep], w[keep]
        k = min(self.BSPLINE_DEGREE, len(P) - 1)
        # Fit; if the smoothing spline overshoots the waypoint hull, retry with stronger
        # smoothing. When the first few waypoints are clustered and non-monotonic (drone +
        # lead-in + intermediates right next to a gate), splprep can fling the curve metres
        # outside the arena (length ≫ the waypoint polyline) — the "wild replan" the drone
        # then tries to chase. Stronger smoothing tames that; it only kicks in for these
        # pathological fits, normal ones use SMOOTHING and stay tight to the gates.
        lo = P.min(axis=0) - self.FIT_OVERSHOOT_MARGIN
        hi = P.max(axis=0) + self.FIT_OVERSHOOT_MARGIN
        tck = None
        for s in (self.SMOOTHING, 0.5, 1.0, 3.0, 8.0):
            tck, _ = splprep([P[:, 0], P[:, 1], P[:, 2]], w=w, k=k, s=s)
            xyz = np.asarray(splev(np.linspace(0.0, 1.0, 80), tck))  # shape (3, 80)
            if np.all(xyz.min(axis=1) >= lo) and np.all(xyz.max(axis=1) <= hi):
                break
        return tck

    def _fit_weights(self, raw: list[tuple]) -> np.ndarray:
        """Per-waypoint fit weights: anchor start, hold gates, free intermediates."""
        n = len(raw)
        w = np.empty(n)
        for i, (_pos, vel) in enumerate(raw):
            if i == 0:
                w[i] = self.W_ANCHOR
            elif i == n - 1:
                w[i] = 0.5 * self.W_ANCHOR
            elif vel is not None:  # gate entry / center / exit
                w[i] = self.W_GATE
            else:  # optimized intermediate
                w[i] = self.W_FREE
        return w

    def _speed_profile(
        self, s: np.ndarray, length: float, v_start: float | None = None
    ) -> np.ndarray:
        """Trapezoidal speed profile over arc length (option C).

        Cruises at TARGET_SPEED with smooth (smoothstep) ramps at the start and end
        so the reference speed is continuous and bounded. The start ramp begins at
        ``v_start`` (defaults to ``V_EDGE``); seeding it with the drone's current
        speed avoids a discontinuous jump down to ``V_EDGE`` when replanning in
        flight. The end ramp always settles to ``V_EDGE``.
        """
        if v_start is None:
            v_start = self.V_EDGE
        v = np.full_like(s, self.TARGET_SPEED)
        a = min(self.ACCEL_DIST, 0.5 * length)
        d = min(self.DECEL_DIST, 0.5 * length)
        if a > 1e-6:
            m = s < a
            f = s[m] / a
            v[m] = v_start + (self.TARGET_SPEED - v_start) * (3 * f**2 - 2 * f**3)
        if d > 1e-6:
            m = s > length - d
            f = (length - s[m]) / d
            v[m] = np.minimum(v[m], self.V_EDGE + (self.TARGET_SPEED - self.V_EDGE) * (3 * f**2 - 2 * f**3))
        return np.clip(v, self.V_EDGE, None)

    def _finalize(self, raw: list[tuple], v_start: float | None = None, n_dense: int = 2000) -> dict:
        """Resample the fitted curve at constant time step with the speed profile.

        Returns a result dict (``pos``/``vel``/``tick_max``) instead of mutating
        ``self`` so it can run in a background worker thread.
        """
        tck = self._fit(raw)

        # Dense geometric sampling + arc length.
        u = np.linspace(0.0, 1.0, n_dense)
        x, y, z = splev(u, tck)
        P = np.stack([x, y, z], axis=1)
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        length = float(s[-1])

        # Speed profile -> time of each dense node (t = ∫ ds / v).
        v = self._speed_profile(s, length, v_start)
        dt_seg = np.diff(s) / np.clip(0.5 * (v[:-1] + v[1:]), 1e-3, None)
        t = np.concatenate([[0.0], np.cumsum(dt_seg)])
        t_total = float(t[-1])

        # Resample at uniform time -> arc length -> spline parameter.
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
        # Hard floor: a smoothing spline can dip below the lowest waypoint, so clamp
        # the reference above ground and stop it commanding descent at the floor.
        below = pos[:, 2] < self.MIN_REF_Z
        pos[below, 2] = self.MIN_REF_Z
        vel[below, 2] = np.maximum(vel[below, 2], 0.0)

        return {
            "pos": pos,
            "vel": vel,
            "tick_max": n_samp - 1 - self.N,
        }


# ── Controller ────────────────────────────────────────────────────────────────


class MPCPlanner(Controller):
    """Replanning B-spline path-planner combined with a tracking attitude-MPC."""

    #: Move (m) / rotation (quat component) beyond which we consider an object to
    #: have "changed" and trigger a replan.
    REPLAN_TOL = 1e-3

    #: Distance "bonus" (m) rewarding a resume point whose trajectory tangent aligns with
    #: the drone's velocity, so a replan never resumes on a branch running opposite to the
    #: drone's motion (see :meth:`_nearest_tick`).
    RESUME_DIR_BONUS = 0.15

    #: How many ticks ahead (along the current trajectory) to plan a replan from, to
    #: compensate the background-build lag. Kept below the true lag (~15 ticks) so the new
    #: trajectory starts just *behind* the drone on install — cutting stale-start zigzag
    #: without putting the reference ahead of the drone (which causes a forward jump).
    REPLAN_PRED_TICKS = 8

    #: Max consecutive failed MPC solves for which the last good command is held before
    #: braking to a safe level-hover. Keeps transient glitches smooth but stops the drone
    #: ever flying away on a frozen command (hardware safety).
    MAX_HOLD_TICKS = 3

    #: Reference-governor lookahead (ticks). The reference index may lead the drone's
    #: *actual* progress along the trajectory by at most this much (~0.3 s at 50 Hz). If
    #: the drone falls behind (large tracking error), the reference is held back instead
    #: of marching on to the goal — otherwise the MPC chases a far-ahead point at full
    #: thrust and the drone shoots off. Critical for hardware safety.
    MAX_LOOKAHEAD_TICKS = 15

    #: Hard MPC floor: the drone's z position may never go below this (m). Kept at
    #: ground level so it stays feasible at takeoff (start height ~0.01 m).
    GROUND_Z = 0.0

    #: MPC soft-obstacle clearance (m), kept clearly below the planner's PLAN_CLEARANCE
    #: (0.235) so the nominal path is NOT on the constraint boundary (that makes the QP
    #: near-active everywhere and stiff at takeoff). It only bites when tracking error
    #: drifts the drone inward toward a pole — the safety net for the "controller off the
    #: reference" case. Extra pole margin comes from the planner buffer, not from raising
    #: this toward 0.235.
    MPC_OBS_CLEARANCE = 0.15

    #: Whether the MPC carries its own soft obstacle/gate keep-out constraints (option B).
    #: ON: the constraints now use C¹ smooth surrogates (``_smooth_*``) instead of the
    #: non-smooth ``fmax``/``fabs`` that previously broke convergence (status=2). This
    #: lets the MPC keep clearance from poles even when it can't track the planned path
    #: exactly — the planner alone left the drone too close on inward drift.
    USE_SOFT_CONSTRAINTS = False

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
        # Solver with a hard ground constraint (z >= GROUND_Z) plus soft obstacle and
        # gate-frame keep-out constraints, parametrised by the live object poses.
        self._n_obstacles = int(len(obs["obstacles_pos"])) if self.USE_SOFT_CONSTRAINTS else 0
        self._n_gates = int(len(obs["gates_pos"])) if self.USE_SOFT_CONSTRAINTS else 0
        self._n_con = self._n_obstacles + self._n_gates
        self._acados_ocp_solver, self._ocp = _create_ocp_solver(
            self._T_HORIZON, self._N, self.drone_params, z_min=self.GROUND_Z,
            n_obstacles=self._n_obstacles, n_gates=self._n_gates,
            obs_clearance=self.MPC_OBS_CLEARANCE, gate_drone_r=BSplinePlanner.DRONE_RADIUS,
        )
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()
        self._ny = self._nx + self._nu
        self._ny_e = self._nx
        self._hover_thrust = self.drone_params["mass"] * -self.drone_params["gravity_vec"][-1]

        # ── Path planner ───────────────────────────────────────────────────────
        self.planner = BSplinePlanner(obs, config, N=self._N)
        self._snapshot_objects(obs)  # remember poses we just planned with

        # Background replanning: the heavy planner build runs in a worker thread so
        # it never stalls the 50 Hz control loop. The drone keeps flying the current
        # trajectory until the new one is ready, then we swap it in atomically.
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._replan_future = None

        self._tick = 0
        self._finished = False
        self._last_obs = obs  # latest observation, for rendering registered poses

        # Solver fallback: last command from a successful solve, plus a hover command
        # (level attitude + hover thrust) as the very first fallback.
        self._last_u = None
        self._hover_cmd = np.array([0.0, 0.0, 0.0, self._hover_thrust])
        self._consec_fail = 0  # consecutive failed solves (for the safe-fallback brake)

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

    @staticmethod
    def _copy_obs(obs: dict) -> dict:
        """Snapshot the observation so the worker never reads sim-reused buffers."""
        return {k: (v.copy() if hasattr(v, "copy") else v) for k, v in obs.items()}

    def _progress_tick(self, drone_pos: np.ndarray, drone_vel: np.ndarray) -> int:
        """Forward-aware nearest trajectory index within a window around the current tick.

        Used by the reference governor. Searching only a local window (not the whole
        path) keeps it cheap and stops it snapping to a far part of a self-crossing
        trajectory; the velocity term keeps it on the forward-running branch.
        """
        traj = self.planner.pos
        lo = max(0, self._tick - 5)
        hi = min(len(traj), self._tick + self._N + 10)
        seg = traj[lo:hi]
        d = np.linalg.norm(seg - drone_pos, axis=1)
        speed = float(np.linalg.norm(drone_vel))
        if speed >= 0.2 and len(seg) >= 2:
            vhat = np.asarray(drone_vel, float) / speed
            tang = np.gradient(seg, axis=0)
            tn = tang / np.clip(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9, None)
            d = d - self.RESUME_DIR_BONUS * (tn @ vhat)
        return lo + int(np.argmin(d))

    def _nearest_tick(self, drone_pos: np.ndarray, drone_vel: np.ndarray | None = None) -> int:
        """Trajectory index to resume at after a replan: nearest to the drone, but biased
        toward points whose tangent runs *with* the drone's velocity.

        After a background replan the drone has moved on from the position the plan was
        built from, so we resume at the nearest point. On a self-looping path (e.g. a
        U-turn near a gate) the drone can sit close to *two* branches — the incoming one
        and the returning one. Picking purely by distance can land on the returning
        branch, whose tangent runs opposite to the drone's motion, so the MPC reference
        points backward and the drone is told to reverse. A small alignment reward keeps
        the resume point on the forward-running branch.
        """
        traj = self.planner.pos
        d = np.linalg.norm(traj - drone_pos, axis=1)
        speed = 0.0 if drone_vel is None else float(np.linalg.norm(drone_vel))
        if speed < 0.2:  # too slow for a meaningful heading → nearest by distance
            idx = int(np.argmin(d))
        else:
            vhat = np.asarray(drone_vel, float) / speed
            tang = np.gradient(traj, axis=0)
            tn = tang / np.clip(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9, None)
            align = tn @ vhat  # cos(tangent, velocity) ∈ [-1, 1]
            idx = int(np.argmin(d - self.RESUME_DIR_BONUS * align))
        return int(np.clip(idx, 0, self.planner.tick_max))

    def _maybe_replan(self, obs: dict):
        """Swap in a finished replan and/or kick off a new one — never blocking.

        The heavy build runs in a worker thread. Each tick we (1) install a finished
        trajectory if one is ready, realigning the tick to the drone's current
        position, and (2) submit a new build if the track changed and none is in
        flight. Snapshotting the poses at submit time prevents re-submitting the same
        change every tick.
        """
        # (1) Install a finished replan.
        if self._replan_future is not None and self._replan_future.done():
            try:
                result = self._replan_future.result()
                self.planner._apply_trajectory(result)
                self._tick = self._nearest_tick(obs["pos"], obs["vel"])
            except Exception as exc:  # keep flying the current trajectory on failure
                print(f"[MPCPlanner] background replan failed: {exc!r}")
            self._replan_future = None

        # (2) Kick off a new replan if the track changed and none is in flight.
        if self._replan_future is None and self._objects_changed(obs):
            v_start = max(float(np.linalg.norm(obs["vel"])), BSplinePlanner.V_EDGE)
            # The build runs in a background thread (~15 ticks); by install the drone has
            # moved on, so planning from the *current* position leaves the new trajectory's
            # start ~0.5 m behind the drone — it then has to navigate the stale, slightly
            # zigzaggy first waypoints. We plan instead from a point a few ticks ahead along
            # the *current* trajectory (the drone is tracking it), which cuts the staleness.
            # The offset is kept BELOW the true lag so the start stays just behind the drone
            # — never ahead of it, which would make the reference jump forward on install.
            obs_pred = self._copy_obs(obs)
            pred_idx = min(self._tick + self.REPLAN_PRED_TICKS, self.planner.tick_max)
            obs_pred["pos"] = self.planner.pos[pred_idx].copy()
            self._replan_future = self._executor.submit(
                self.planner._compute_trajectory, obs_pred, v_start
            )
            self._snapshot_objects(obs)  # plan reflects these poses now

    def _obstacle_gate_params(self, obs: dict) -> np.ndarray:
        """Flatten obstacle/gate poses into the solver's parameter vector layout.

        Layout: [obstacle xy ...] then per gate [is_target, center(3), x_axis(3),
        y_axis(3), z_axis(3)] — matching the ``con_h`` indexing in ``_create_ocp_solver``.
        The opening constraint is only active for the current target gate (flag 1.0).
        """
        parts = []
        if self._n_obstacles:
            parts.append(np.asarray(obs["obstacles_pos"])[:, :2].reshape(-1))
        target_gate = int(obs["target_gate"])
        for i in range(self._n_gates):
            parts.append(np.array([1.0 if i == target_gate else 0.0]))
            parts.append(np.asarray(obs["gates_pos"][i], dtype=float))
            axes = R.from_quat(obs["gates_quat"][i]).apply(np.eye(3))  # rows: x,y,z axes
            parts.append(axes.reshape(-1))
        return np.concatenate(parts) if parts else np.zeros(0)

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

<<<<<<< HEAD
        # Without this line, the tick is just a clock (it counts up by 1 each step),
        # so it points to where the drone *should* be by now. If the drone falls
        # behind, that target runs away from it and tracking degrades.
        # With this line, the tick is set from where the drone *actually* is: we
        # snap it to the nearest point on the path to the drone's real position.
        self._tick = self._nearest_tick(obs["pos"])
=======
        # Reference governor: keep the reference anchored near the drone's actual progress
        # so it can never run away. If the drone has fallen behind, hold the reference
        # index back (≤ MAX_LOOKAHEAD_TICKS ahead of the nearest point) instead of letting
        # it march to the goal — otherwise the MPC chases a far point at full thrust and
        # the drone shoots off (unsafe on hardware).
        near = self._progress_tick(obs["pos"], obs["vel"])
        self._tick = int(np.clip(self._tick, near, near + self.MAX_LOOKAHEAD_TICKS))
>>>>>>> 4010458 (mid save)

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

        # ── Obstacle / gate-frame parameters (soft keep-out constraints) ───────
        if self._n_con:
            p_val = self._obstacle_gate_params(obs)
            for stage in range(self._N + 1):
                self._acados_ocp_solver.set(stage, "p", p_val)

        # ── Solve ──────────────────────────────────────────────────────────────
        # On a failed solve (non-zero status) the returned solution may be garbage,
        # so hold the last good command (hover on the very first failure) instead.
        status = self._acados_ocp_solver.solve()
        if status == 0:
            self._last_u = self._acados_ocp_solver.get(0, "u")
            self._consec_fail = 0
            return self._last_u
        # Hold the last good command for a few transient failures (smooth), but if the
        # solver keeps failing, brake to a safe level-hover instead of flying on a stale,
        # possibly-aggressive command — a frozen command can run the drone away, which is
        # unsafe on hardware.
        self._consec_fail += 1
        if self._last_u is not None and self._consec_fail <= self.MAX_HOLD_TICKS:
            return self._last_u
        print(f"[MPCPlanner] MPC solve failed (status={status}) at tick {self._tick}; braking to hover")
        return self._hover_cmd

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
            # Gate frames: outer square (cyan) + opening (white), real dimensions.
            for gpos, gquat in zip(obs["gates_pos"], obs["gates_quat"]):
                self._draw_square(sim, gpos, gquat, _GateFrame.OUTER / 2,
                                   np.array([0.0, 1.0, 1.0, 1.0]))
                self._draw_square(sim, gpos, gquat, _GateFrame.OPENING / 2,
                                   np.array([1.0, 1.0, 1.0, 1.0]))
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

        # Planned trajectory (downsampled to ~20 segments to stay light).
        traj = self.planner.pos
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

    def _cancel_replan(self):
        """Drop any pending background replan (keeps the executor for reuse)."""
        if self._replan_future is not None:
            self._replan_future.cancel()
            self._replan_future = None

    def episode_callback(self):
        """Reset the trajectory index after an episode."""
        self._tick = 0
        self._cancel_replan()

    def episode_reset(self):
        """Reset internal state for a new episode."""
        self._tick = 0
        self._finished = False
        self._cancel_replan()
