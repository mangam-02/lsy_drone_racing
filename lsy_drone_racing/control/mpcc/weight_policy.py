"""RL weight planner for the MPCC controller.

The MPCC cost weights ``W`` (:func:`mpcc_deploy.create_mpcc_ocp_solver`) are normally static —
a single compromise that serves neither tight gate approaches nor smooth straights optimally.
This module is the planned *third layer* on top of the planner + MPCC: a small policy that, per
control tick, adapts the MPCC cost weights from the drone's state (aggressive near gates, smooth
on straights) to push racing performance.

Design (maximum compatibility with the controller):

* The **baseline weights are owned by the controller** — ``mpcc_deploy.BASELINE_WEIGHTS`` is the
  single source of truth. This module never hard-codes weight values; :func:`weight_diagonals`
  takes the controller's baseline dict and *scales* it. The solver build calls it with
  ``multipliers = 1`` (→ exactly the controller's weights) and the controller calls it per tick
  with the policy's multipliers — so the built and the runtime ``W`` can never diverge, and
  changing the weights in ``mpcc_deploy`` automatically flows into both.
* The **action** is a small vector of *bounded multipliers* on grouped baseline weights (not
  absolute 16-dim weights, which are fragile and slow to learn). ``action = 0`` maps to
  ``multiplier = 1`` (identity), so an untrained / dummy policy reproduces the controller's
  weights exactly. Turning the RL off entirely (``USE_RL_WEIGHTS = False``) keeps the controller
  bit-identical to the no-RL version (the weight code path is skipped).
* The **observation** is a compact, solver-free feature vector (:func:`build_features`) built
  from the live obs + the planner's geometric path projection.

Train the policy with ``train_mpcc_weights.py`` (MPCC kept in the training loop); deploy it via
``MPCCController.USE_RL_WEIGHTS`` in ``mpcc_deploy.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation as R

try:  # torch is only needed for the RL weight policy (the optional 'rl' extra). The rest of this
    import torch  # module (features, weight scaling, constants) is torch-free, so the MPCC
    import torch.nn as nn  # controller can be deployed in a clean env that has no torch installed.
    from torch.distributions.normal import Normal

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - only hit in torch-free deploy envs
    _HAS_TORCH = False

if TYPE_CHECKING:
    from torch import Tensor

# ── Weight layout (the baseline VALUES live in the controller; this is only the layout) ─────

#: Baseline cost-scalar keys the controller must provide (``mpcc_deploy.BASELINE_WEIGHTS``).
#: This module owns only the *layout*, never the values — so the controller stays the single
#: source of truth for the weights.
BASELINE_KEYS = ("q_c", "q_l", "q_att", "q_dr", "q_v", "r_rpy", "r_T", "r_at")

#: Policy action groups: each is one bounded multiplier knob. Few, grouped scalars learn far
#: better than 16 independent weights. ``q_track`` scales both attitude and rate tracking;
#: ``r_ctrl`` scales both rpy-command and progress-accel effort.
WEIGHT_GROUPS = ("q_c", "q_l", "q_v", "q_track", "r_thrust", "r_ctrl")
N_ACTIONS = len(WEIGHT_GROUPS)

#: Multiplier bounds: ``mult = exp(MULT_LOG_RANGE * tanh(action)) ∈ [1/F, F]`` with ``F = 4``.
#: Symmetric in log-space and equal to 1 at ``action = 0`` (identity → baseline). Bounding the
#: action space (rather than clamping afterwards) keeps the QP from going steep/infeasible.
MULT_FACTOR = 4.0
MULT_LOG_RANGE = float(np.log(MULT_FACTOR))

#: Observation layout (compact, solver-free). Keep in sync with :func:`build_features`.
FEATURE_NAMES = (
    # ── Path-relative state ──
    "contour_err",  # perpendicular distance to the path (m)
    "lag_err",  # signed longitudinal path error (m)
    "speed",  # |velocity| (m/s)
    "vtheta_proj",  # velocity component along the path tangent (m/s)
    "progress_frac",  # theta / path length ∈ [0, 1]
    # ── Path curvature ahead (corner anticipation) ──
    "curv_near",  # path curvature ~0.5 m ahead (1/m)
    "curv_far",  # path curvature ~1.5 m ahead (1/m)
    # ── Target (next) gate ──
    "gate_dist",  # distance to the current target gate centre (m)
    "gate_rel_x",  # target-gate offset from the drone, world frame (m)
    "gate_rel_y",
    "gate_rel_z",
    "gate_align",  # |cos| between approach direction and gate normal (1 = head-on)
    "gate_known",  # 1 if the target gate's true pose is sensed, 0 if still nominal
    # ── Next-next gate (anticipate the turn after the upcoming gate) ──
    "gate2_dist",  # distance to the gate after the target (m; 10 if none)
    "gate2_rel_x",  # next-next-gate offset from the drone, world frame (m)
    "gate2_rel_y",
    "gate2_rel_z",
    "gate2_known",  # 1 if the next-next gate's true pose is sensed, 0 if nominal/none
    # ── Two nearest obstacles (xy), nearest first ──
    "obs1_dist",  # distance to the nearest obstacle pole, xy (m)
    "obs1_rel_x",  # nearest-obstacle offset, world frame xy (m)
    "obs1_rel_y",
    "obs1_known",  # 1 if the nearest obstacle's true pose is sensed, 0 if nominal
    "obs2_dist",  # distance to the 2nd-nearest obstacle pole, xy (m)
    "obs2_rel_x",  # 2nd-nearest-obstacle offset, world frame xy (m)
    "obs2_rel_y",
    "obs2_known",  # 1 if the 2nd-nearest obstacle's true pose is sensed, 0 if nominal
)
N_FEATURES = len(FEATURE_NAMES)


def multipliers_from_action(action: np.ndarray) -> np.ndarray:
    """Map a raw policy action to bounded weight multipliers ``∈ [1/MULT_FACTOR, MULT_FACTOR]``.

    ``action = 0`` → ``1.0`` (identity), so a zero/untrained policy yields the controller's
    baseline weights unchanged.
    """
    return np.exp(MULT_LOG_RANGE * np.tanh(np.asarray(action, dtype=float)))


def weight_diagonals(
    multipliers: np.ndarray, baseline: dict, n_caps: int, capsule_penalty: float
) -> tuple[np.ndarray, np.ndarray]:
    """Scale the controller's ``baseline`` weights by group multipliers into (stage, terminal) diag.

    ``baseline`` is the controller's weight dict (``mpcc_deploy.BASELINE_WEIGHTS``, keys
    :data:`BASELINE_KEYS`); ``multipliers = 1`` reproduces it exactly. Layout matches the
    NONLINEAR_LS residual ordering in ``create_mpcc_ocp_solver``: ``[e_c(3), e_l, rpy(3), drpy(3),
    e_v, rpy_cmd(3), thrust, a_theta]`` for the stage cost and ``[e_c(3), e_l, rpy(3), drpy(3),
    e_v]`` for the terminal cost, each followed by ``n_caps`` capsule-barrier residuals weighted
    ``capsule_penalty`` (avoidance is safety, not tuned).
    """
    m = dict(zip(WEIGHT_GROUPS, np.asarray(multipliers, dtype=float)))
    q_c = baseline["q_c"] * m["q_c"]
    q_l = baseline["q_l"] * m["q_l"]
    q_att = baseline["q_att"] * m["q_track"]
    q_dr = baseline["q_dr"] * m["q_track"]
    q_v = baseline["q_v"] * m["q_v"]
    r_rpy = baseline["r_rpy"] * m["r_ctrl"]
    r_T = baseline["r_T"] * m["r_thrust"]
    r_at = baseline["r_at"] * m["r_ctrl"]

    term = [q_c, q_c, q_c, q_l, q_att, q_att, q_att, q_dr, q_dr, q_dr, q_v]
    stage = term + [r_rpy, r_rpy, r_rpy, r_T, r_at]
    caps = [capsule_penalty] * n_caps
    return np.array(stage + caps), np.array(term + caps)


# ── Observation features (solver-free, built from obs + planner path) ───────────────


def _path_curvature(planner: object, theta: float, lookahead: float, ds: float = 0.2) -> float:
    """Unsigned path curvature (1/m) a short distance ``lookahead`` ahead of ``theta``.

    Estimated from the change in unit tangent over ``ds`` of arc length (``|dT/ds| ≈ curvature``,
    since the path is arc-length parameterised): ~0 on a straight, large in a tight corner. Lets
    the policy anticipate corners and stiffen/slow before reaching them. Clamped to the path end.
    """
    length = float(getattr(planner, "length", 0.0))
    if length <= 1e-6:
        return 0.0
    s0 = min(theta + lookahead, length)
    s1 = min(s0 + ds, length)
    if s1 - s0 < 1e-6:
        return 0.0
    _, t0 = planner.path_point_tangent(s0)
    _, t1 = planner.path_point_tangent(s1)
    return float(np.linalg.norm(np.asarray(t1) - np.asarray(t0)) / (s1 - s0))


def build_features(obs: dict, planner: object, theta_prev: float | None) -> np.ndarray:
    """Compact feature vector for the weight policy — see :data:`FEATURE_NAMES`.

    Deliberately solver-free: it projects the drone onto the planner's geometric path (the same
    ``project_to_theta`` / ``path_point_tangent`` the controller uses) so the policy can be
    evaluated *before* the MPCC solve, both in deployment and in training. Beyond the immediate
    path-relative state it also exposes the curvature ahead, the next *and* next-next gate, and the
    two nearest obstacles — each gate/obstacle carrying a ``known`` flag (true sensed pose vs. only
    the nominal a-priori pose) so the policy can be cautious while a pose is still uncertain.
    """
    pos = np.asarray(obs["pos"], dtype=float)
    vel = np.asarray(obs["vel"], dtype=float)

    # Path-relative state.
    length = float(getattr(planner, "length", 0.0))
    theta0 = planner.project_to_theta(pos, vel, theta_prev=theta_prev)
    p_d, tan = planner.path_point_tangent(theta0)
    d = pos - p_d
    lag = float(np.dot(tan, d))
    contour = float(np.linalg.norm(d - lag * tan))
    speed = float(np.linalg.norm(vel))
    vtheta_proj = float(np.dot(vel, tan))
    progress_frac = theta0 / length if length > 1e-6 else 0.0

    # Curvature ahead (near + far lookahead) for corner anticipation.
    curv_near = _path_curvature(planner, theta0, 0.5)
    curv_far = _path_curvature(planner, theta0, 1.5)

    # Gates: target (next) + the one after it, each with a "true pose known yet?" flag. Until a
    # gate is sensed the obs only carries its NOMINAL pose, so the flag tells the policy whether to
    # trust the planned line through it (sensed) or stay cautious (still nominal).
    gates_pos = np.atleast_2d(np.asarray(obs["gates_pos"], dtype=float))
    gates_quat = np.atleast_2d(np.asarray(obs["gates_quat"], dtype=float))
    gates_visited = np.atleast_1d(np.asarray(obs["gates_visited"])).astype(float)
    n_gates = len(gates_pos)
    target_gate = int(obs["target_gate"])
    gi = target_gate if 0 <= target_gate < n_gates else n_gates - 1

    gate_rel = gates_pos[gi] - pos
    gate_dist = float(np.linalg.norm(gate_rel))
    gate_normal = R.from_quat(gates_quat[gi]).apply([1.0, 0.0, 0.0])  # gate plane normal
    gate_align = abs(float(np.dot(gate_normal, gate_rel / max(gate_dist, 1e-6))))
    gate_known = float(gates_visited[gi]) if gi < len(gates_visited) else 0.0

    if gi + 1 < n_gates:  # next-next gate: anticipate the turn waiting after the upcoming gate
        gj = gi + 1
        gate2_rel = gates_pos[gj] - pos
        gate2_dist = float(np.linalg.norm(gate2_rel))
        gate2_known = float(gates_visited[gj]) if gj < len(gates_visited) else 0.0
    else:  # already heading through the final gate → nothing meaningful after it
        gate2_rel = np.zeros(3)
        gate2_dist = 10.0
        gate2_known = 0.0

    # Two nearest obstacle poles (xy), nearest first, each with its own known flag.
    obstacles_pos = np.atleast_2d(np.asarray(obs["obstacles_pos"], dtype=float))
    obs_visited = np.atleast_1d(np.asarray(obs["obstacles_visited"])).astype(float)
    if len(obstacles_pos):
        rel_xy = obstacles_pos[:, :2] - pos[:2]
        d_xy = np.linalg.norm(rel_xy, axis=1)
        order = np.argsort(d_xy)
    else:
        rel_xy, d_xy, order = np.zeros((0, 2)), np.zeros(0), np.zeros(0, dtype=int)
    obs_block: list[float] = []
    for k in range(2):
        if k < len(order):
            j = int(order[k])
            known = float(obs_visited[j]) if j < len(obs_visited) else 0.0
            obs_block += [float(d_xy[j]), float(rel_xy[j, 0]), float(rel_xy[j, 1]), known]
        else:  # fewer than 2 obstacles on this track → far-away dummy
            obs_block += [10.0, 0.0, 0.0, 0.0]

    feats = np.array(
        [
            contour,
            lag,
            speed,
            vtheta_proj,
            progress_frac,
            curv_near,
            curv_far,
            gate_dist,
            gate_rel[0],
            gate_rel[1],
            gate_rel[2],
            gate_align,
            gate_known,
            gate2_dist,
            gate2_rel[0],
            gate2_rel[1],
            gate2_rel[2],
            gate2_known,
            *obs_block,
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(feats, nan=0.0, posinf=1e3, neginf=-1e3)


# ── Policy network (PPO actor-critic; shared by training and deployment) ────────────


if _HAS_TORCH:

    def _layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias: float = 0.0) -> nn.Linear:
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias)
        return layer

    class WeightPolicyNet(nn.Module):
        """Small Gaussian-policy actor-critic over weight features.

        The actor outputs *raw* actions (no squashing here); squashing/bounding to multipliers is
        done by :func:`multipliers_from_action`, so the same net serves training (sample) and
        deployment (mean). ``actor_logstd`` starts low so the initial policy stays near identity.
        """

        def __init__(self, n_obs: int = N_FEATURES, n_act: int = N_ACTIONS, hidden: int = 128):
            """Build the actor-critic MLP over ``n_obs`` features and ``n_act`` weight knobs."""
            super().__init__()
            self.critic = nn.Sequential(
                _layer_init(nn.Linear(n_obs, hidden)),
                nn.Tanh(),
                _layer_init(nn.Linear(hidden, hidden)),
                nn.Tanh(),
                _layer_init(nn.Linear(hidden, 1), std=1.0),
            )
            self.actor_mean = nn.Sequential(
                _layer_init(nn.Linear(n_obs, hidden)),
                nn.Tanh(),
                _layer_init(nn.Linear(hidden, hidden)),
                nn.Tanh(),
                _layer_init(nn.Linear(hidden, n_act), std=0.01),
            )
            self.actor_logstd = nn.Parameter(torch.full((1, n_act), -1.0))

        def get_value(self, x: Tensor) -> Tensor:
            """Critic value estimate for the feature batch ``x``."""
            return self.critic(x)

        def get_action_and_value(
            self, x: Tensor, action: Tensor | None = None, deterministic: bool = False
        ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
            """Return (action, log-prob, entropy, value); sample unless ``deterministic``."""
            mean = self.actor_mean(x)
            std = torch.exp(self.actor_logstd.expand_as(mean))
            dist = Normal(mean, std)
            if action is None:
                action = mean if deterministic else dist.sample()
            return action, dist.log_prob(action).sum(1), dist.entropy().sum(1), self.critic(x)


class WeightPolicy:
    """Deployment-side wrapper: feature → bounded weight multipliers.

    Loads a trained :class:`WeightPolicyNet` checkpoint. With ``ckpt_path=None`` (or a missing
    file) it stays a **dummy / identity** policy returning all-ones multipliers, so the MPCC
    runs on the exact baseline weights until a policy is trained.
    """

    def __init__(self, ckpt_path: str | None = None, device: str = "cpu"):
        """Load the policy from ``ckpt_path``; stay an identity policy if it is None/missing."""
        if not _HAS_TORCH:
            raise ImportError(
                "WeightPolicy needs PyTorch (the optional 'rl' extra). Install it with "
                "`pip install -e '.[rl]'`, or keep USE_RL_WEIGHTS off to run without torch."
            )
        self.device = torch.device(device)
        self.net: WeightPolicyNet | None = None
        if ckpt_path is not None:
            import os

            if os.path.exists(ckpt_path):
                net = WeightPolicyNet().to(self.device)
                net.load_state_dict(torch.load(ckpt_path, map_location=self.device))
                net.eval()
                self.net = net
            else:
                print(f"[WeightPolicy] no checkpoint at {ckpt_path!r}; using identity weights")

    def multipliers(self, features: np.ndarray) -> np.ndarray:
        """Return weight multipliers for the given features (all-ones if no policy is loaded)."""
        if self.net is None:
            return np.ones(N_ACTIONS)
        with torch.no_grad():
            x = torch.as_tensor(features, dtype=torch.float32, device=self.device).reshape(1, -1)
            action, _, _, _ = self.net.get_action_and_value(x, deterministic=True)
        return multipliers_from_action(action.cpu().numpy().reshape(-1))
