"""Tuning constants for the MPCC controller and its path planner.

Single source of truth for every tunable value. The classes in :mod:`planner` and
:mod:`controller` bind these onto themselves (``TARGET_SPEED = cfg.TARGET_SPEED`` …), so
editing a value here changes the controller's behaviour without touching any logic.

The file is split into three clearly separated sections:

* ``PLANNER``    — :class:`planner.SimplePlanner` waypoint/speed/optimiser constants.
* ``CONTROLLER`` — :class:`controller.MPCCController` MPCC/safety/avoidance constants.
* ``MPC COST WEIGHTS`` — the baseline diagonal cost weights (``BASELINE_WEIGHTS``).
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════════════════════
# PLANNER  (SimplePlanner)
# ═══════════════════════════════════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════════════════════════════════
# CONTROLLER  (MPCCController)
# ═══════════════════════════════════════════════════════════════════════════════════════════

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
V_TARGET = 2.5  # m/s — target progress speed (cruise); matched to SimplePlanner.TARGET_SPEED
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
USE_GATE_TRACK_BOOST = True  # A/B: ×4 q_c near gates stayed within eval noise, hurt gate 0
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
#: corrected path). Implemented by scaling the progress target v_target toward.
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
#: Floor on the gate-barrier brake so v_target never ramps fully to 0 at the barrier. Without
#: it the drone could decelerate to a dead stop just SHORT of the gate plane; the env only
#: registers a pass on a real x<0 → x>0 crossing of the drone's true position, so a parked
#: drone never triggers it and the barrier never lifts (the "waiting after the gate" deadlock).
#: The floor keeps a small forward push so the drone always creeps the final centimetres THROUGH
#: the plane. It can't overshoot a genuinely missed gate: the per-stage θ stays clamped at
#: theta_cap, so the drone settles at the capped reference (≈ gate centre + GATE_PASS_LEAD) and
#: a real miss still ends in the gate-retry recovery.
GATE_BRAKE_FLOOR = 0.25
#: Gate-retry recovery. The progress θ is monotonic (v_theta >= 0) and clamped at the gate
#: barrier, so if the drone MISSES a gate (overshoots / flies past the opening without the env
#: confirming the pass) it can't reverse on its own — v_target brakes to 0 and it dead-hovers
#: at the barrier until timeout. This detects that stall and flies the drone back to a point
#: BEFORE the gate with a simple position controller (bypassing the MPCC), then hands back so
#: the MPCC re-approaches and re-threads. The back-and-forth also raises the chance of finally
#: sensing the real gate pose (→ replan onto the correct line). Off → old dead-hover behaviour.
USE_GATE_RETRY = False
RETRY_STUCK_SPEED = 0.3  # m/s — below this, at/past the gate, counts as stalled
RETRY_STUCK_TICKS = 40  # consecutive stalled ticks before a retry triggers (~0.8 s @ 50 Hz)
RETRY_BACK_DIST = 0.8  # m (arc length) before the gate centre the recovery retreats to
RETRY_DONE_DIST = 0.25  # m — within this of the retreat point, hand back to the MPCC
RETRY_MAX = 3  # give up after this many retries on one gate (then let it hover → timeout)
REPLAN_TOL = 1e-3  # object move (m) beyond which we replan
MAX_HOLD_TICKS = 3  # consecutive failed solves held before braking to hover
#: Prediction-divergence safety net. The MPCC can occasionally return a "successful" (status 0)
#: solve whose predicted trajectory shoots off the planned path — the terminal "blue marker"
#: flies away. Executing such a command on real hardware risks slamming a gate frame. We measure
#: DIVERGENCE = how far the prediction strays from the path BEYOND the drone's current offset
#: (max node deviation − node-0 deviation), NOT absolute deviation. Node 0 is the drone's actual
#: (lbx-pinned) offset, which may legitimately be large after a disturbance — a correct recovery
#: solve STARTS there and converges back, so an absolute-deviation check would reject the very
#: solves that fly it home and leave it stuck braking/sinking off-path. When the divergence
#: exceeds MAX_PRED_DEV the solve is UNTRUSTED: not committed, not stored as a warm start; a
#: clean re-localise (solver.reset + reseed) is forced next tick to re-converge, and meanwhile
#: the controller replays the last TRUSTED plan / brakes. Off → no check.
USE_PRED_SAFETY = True
MAX_PRED_DEV = 1.5  # m — how far the prediction may stray PAST the current offset before reject
#: SQP_RTI iterations to run on a "fresh path" tick (replan just installed / first tick /
#: episode reset), where the warm start is far from the new optimum. Normal ticks run 1.
RTI_BUMP_ITERS = 10

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

# ═══════════════════════════════════════════════════════════════════════════════════════════
# MPC COST WEIGHTS  (the controller's single source of truth)
# ═══════════════════════════════════════════════════════════════════════════════════════════

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
