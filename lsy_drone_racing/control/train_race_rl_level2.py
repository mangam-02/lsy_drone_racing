"""PPO training on the real Level-2 vectorized drone racing environment.

This script trains directly on VecDroneRaceEnv instead of a random spline-following DroneEnv.
It therefore uses the real race observations containing gates, obstacles, target_gate, etc.

Put this file into:
    lsy_drone_racing/control/train_race_rl_level2.py

Run from repo root:
    python lsy_drone_racing/control/train_race_rl_level2.py --train True --eval 3

For a quick smoke test:
    python lsy_drone_racing/control/train_race_rl_level2.py --total_timesteps 20000 --num_envs 64 --eval 1
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import jax

import fire
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.spaces import flatten_space
from gymnasium.vector import VectorEnv, VectorObservationWrapper, VectorRewardWrapper
from gymnasium.wrappers.vector.jax_to_torch import JaxToTorch
from torch import Tensor
from torch.distributions.normal import Normal

from crazyflow.envs.norm_actions_wrapper import NormalizeActions
from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv
from lsy_drone_racing.utils import load_config


@dataclass
class Args:
    seed: int = 42
    cuda: bool = True
    torch_deterministic: bool = True

    config: str = "level2.toml"
    device_env: str = "cpu"

    total_timesteps: int = 2_000_000
    learning_rate: float = 3e-4
    num_envs: int = 256
    num_steps: int = 16
    anneal_lr: bool = True

    gamma: float = 0.98
    gae_lambda: float = 0.97
    num_minibatches: int = 8
    update_epochs: int = 8
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.001
    vf_coef: float = 0.7
    max_grad_norm: float = 1.0
    target_kl: float | None = None

    # reward shaping
    obstacle_safe_dist: float = 0.35
    obstacle_coef: float = 1.5
    gate_half_size: float = 0.225
    gate_safe_margin: float = 0.08
    gate_edge_coef: float = 2.0
    action_smooth_coef: float = 0.03

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

    @staticmethod
    def create(**kwargs: Any) -> "Args":
        args = Args(**kwargs)
        args.batch_size = int(args.num_envs * args.num_steps)
        args.minibatch_size = int(args.batch_size // args.num_minibatches)
        args.num_iterations = max(1, args.total_timesteps // args.batch_size)
        return args


def set_seeds(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def _as_np(x: Any) -> np.ndarray:
    """Convert JAX/Torch/NumPy arrays to NumPy without breaking normal arrays."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class FlattenDictObservation(VectorObservationWrapper):
    """Flatten VecDroneRaceEnv's dict observations into one float32 vector.

    This keeps insertion order from the environment's observation dict. That is okay because
    training and evaluation both use this same wrapper.
    """

    def __init__(self, env: VectorEnv):
        super().__init__(env)
        # Manual flattened dimension, matching observations()
        flat_dim = 0
        for space in env.single_observation_space.spaces.values():
            if isinstance(space, gym.spaces.Discrete):
                flat_dim += 1
            else:
                flat_dim += int(np.prod(space.shape))

        self.single_observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(flat_dim,),
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_envs, flat_dim),
            dtype=np.float32,
        )

    def observations(self, obs: dict[str, Any]) -> np.ndarray:
        parts = []
        n = self.num_envs

        for value in obs.values():
            arr = _as_np(value)

            # VecDroneRaceEnv returns batched arrays with leading shape (num_envs, ...).
            # Some scalar entries may be shape (num_envs,), which reshape handles.
            arr = arr.reshape(n, -1)
            parts.append(arr)

        flat = np.concatenate(parts, axis=-1).astype(np.float32)

        # Avoid occasional inf from Box spaces / casts.
        flat = np.nan_to_num(flat, nan=0.0, posinf=1e6, neginf=-1e6)
        return flat


class RaceRewardShaping(VectorRewardWrapper):
    """Add penalties for obstacles, gate edges and action jumps.

    This wrapper operates on the full dict observation before flattening.
    """

    def __init__(
        self,
        env: VectorEnv,
        obstacle_safe_dist: float = 0.35,
        obstacle_coef: float = 6.0,
        gate_half_size: float = 0.225,
        gate_safe_margin: float = 0.08,
        gate_edge_coef: float = 8.0,
        action_smooth_coef: float = 0.15,
    ):
        super().__init__(env)
        self.obstacle_safe_dist = obstacle_safe_dist
        self.obstacle_coef = obstacle_coef
        self.gate_half_size = gate_half_size
        self.gate_safe_margin = gate_safe_margin
        self.gate_edge_coef = gate_edge_coef
        self.action_smooth_coef = action_smooth_coef
        self._last_action = np.zeros((self.num_envs, self.single_action_space.shape[0]), dtype=np.float32)
        self._last_obs: dict[str, Any] | None = None
        self._current_action: np.ndarray | None = None

        # For progress reward towards the current target gate
        self._last_gate_dist = np.zeros(self.num_envs, dtype=np.float32)

    def reset(self, **kwargs):
        self._last_action[...] = 0.0
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs

        # Initialize distance to target gate for progress reward
        self._last_gate_dist = self._compute_gate_distance(obs)

        return obs, info
    
    def _compute_gate_distance(self, obs: dict[str, Any]) -> np.ndarray:
        """Compute distance from drone to current target gate center."""
        pos = _as_np(obs["pos"]).astype(np.float32)
        gates_pos = _as_np(obs["gates_pos"]).astype(np.float32)
        target_gate = _as_np(obs["target_gate"]).astype(np.int32).reshape(self.num_envs)

        pos = pos.reshape(self.num_envs, -1)
        if pos.shape[-1] > 3:
            pos = pos[:, :3]

        if gates_pos.ndim == 2:
            gates_pos = gates_pos.reshape(self.num_envs, -1, 3)

        n_gates = gates_pos.shape[1]
        safe_idx = np.clip(target_gate, 0, n_gates - 1)

        gate_pos = gates_pos[np.arange(self.num_envs), safe_idx]
        gate_dist = np.linalg.norm(pos - gate_pos, axis=-1)

        return gate_dist.astype(np.float32)

    def step(self, action):
        self._current_action = _as_np(action).astype(np.float32)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_obs = obs
        shaped = self.rewards(reward)
        self._last_action = self._current_action.copy()
        return obs, shaped, terminated, truncated, info

    def rewards(self, reward):
        reward_np = _as_np(reward).astype(np.float32).copy()

        if self._last_obs is None:
            return reward_np

        obs = self._last_obs

        # ------------------------------------------------------------
        # Strong reward for moving towards the current target gate
        # ------------------------------------------------------------
        gate_dist = self._compute_gate_distance(obs)
        gate_progress = self._last_gate_dist - gate_dist

        reward_np += 20.0 * gate_progress
        reward_np -= 1.0 * gate_dist

        self._last_gate_dist = gate_dist

        pos = _as_np(obs["pos"]).astype(np.float32)
        obstacles_pos = _as_np(obs["obstacles_pos"]).astype(np.float32)

        # Shapes should be:
        # pos: (num_envs, 3)
        # obstacles_pos: (num_envs, n_obstacles, 3)
        pos = pos.reshape(self.num_envs, -1)
        if pos.shape[-1] > 3:
            pos = pos[:, :3]

        # ------------------------------------------------------------
        # Keep drone at useful racing altitude
        # ------------------------------------------------------------
        target_height = 0.8
        height_error = pos[:, 2] - target_height
        reward_np -= 4.0 * (height_error ** 2)

        # Strong penalty for being near/on the ground
        ground_penalty = np.where(pos[:, 2] < 0.25, (0.25 - pos[:, 2]) ** 2, 0.0)
        reward_np -= 25.0 * ground_penalty

        if obstacles_pos.ndim == 2:
            obstacles_pos = obstacles_pos.reshape(self.num_envs, -1, 3)

        # obstacle penalty
        d_obs = obstacles_pos - pos[:, None, :]
        dist_obs = np.linalg.norm(d_obs, axis=-1)
        nearest_dist = np.min(dist_obs, axis=1)

        obstacle_penalty = np.where(
            nearest_dist < self.obstacle_safe_dist,
            (self.obstacle_safe_dist - nearest_dist) ** 2,
            0.0,
        )

        reward_np -= self.obstacle_coef * obstacle_penalty.astype(np.float32)

        # approximate gate-edge penalty using world-frame y/z offset to target gate.
        # This is intentionally simple and robust; exact gate-frame penalty can be added later.
        try:
            gates_pos = _as_np(obs["gates_pos"]).astype(np.float32)
            target_gate = _as_np(obs["target_gate"]).astype(np.int32).reshape(self.num_envs)

            if gates_pos.ndim == 2:
                gates_pos = gates_pos.reshape(self.num_envs, -1, 3)

            n_gates = gates_pos.shape[1]
            safe_idx = np.clip(target_gate, 0, n_gates - 1)
            gate_pos = gates_pos[np.arange(self.num_envs), safe_idx]

            gate_rel = pos - gate_pos
            gate_yz_abs = np.abs(gate_rel[:, 1:3])
            dist_to_edge = self.gate_half_size - np.max(gate_yz_abs, axis=-1)

            edge_penalty = np.where(
                dist_to_edge < self.gate_safe_margin,
                (self.gate_safe_margin - dist_to_edge) ** 2,
                0.0,
            )

            # Only apply near the gate, otherwise this can punish normal approach.
            near_gate_plane = np.abs(gate_rel[:, 0]) < 0.5
            reward_np -= self.gate_edge_coef * edge_penalty.astype(np.float32) * near_gate_plane.astype(np.float32)
        except Exception:
            pass

        # action smoothness penalty
        if self._current_action is not None:
            d_action = self._current_action - self._last_action
            smooth_penalty = np.sum(d_action**2, axis=-1)
            reward_np -= self.action_smooth_coef * smooth_penalty.astype(np.float32)

        return reward_np


def make_envs(args: Args, torch_device: torch.device) -> VectorEnv:
    config_path = Path(__file__).parents[2] / "config" / args.config
    config = load_config(config_path)

    max_episode_steps = getattr(config.env, "max_episode_steps", 1500)
    sensor_range = getattr(config.env, "sensor_range", 0.5)
    disturbances = getattr(config.env, "disturbances", None)
    randomizations = getattr(config.env, "randomizations", None)

    env = VecDroneRaceEnv(
        num_envs=args.num_envs,
        freq=config.env.freq,
        sim_config=config.sim,
        track=config.env.track,
        sensor_range=sensor_range,
        control_mode="attitude",
        disturbances=disturbances,
        randomizations=randomizations,
        seed=args.seed,
        max_episode_steps=max_episode_steps,
        device=args.device_env,
    )

    # IMPORTANT:
    # The policy outputs normalized actions in [-1, 1].
    # NormalizeActions converts them to the real attitude/thrust action space.
    # NormalizeActions expects a real JAX device object, not the string "cpu".
    env.device = jax.devices(args.device_env)[0]

    # The policy outputs normalized actions in [-1, 1].
    # NormalizeActions converts them to the real attitude/thrust action space.
    env = NormalizeActions(env)

    env = RaceRewardShaping(
        env,
        obstacle_safe_dist=args.obstacle_safe_dist,
        obstacle_coef=args.obstacle_coef,
        gate_half_size=args.gate_half_size,
        gate_safe_margin=args.gate_safe_margin,
        gate_edge_coef=args.gate_edge_coef,
        action_smooth_coef=args.action_smooth_coef,
    )

    env = FlattenDictObservation(env)
    env = JaxToTorch(env, torch_device)

    return env


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_shape: tuple[int, ...], action_shape: tuple[int, ...]):
        super().__init__()

        obs_dim = int(np.prod(obs_shape))
        action_dim = int(np.prod(action_shape))

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, action_dim), std=0.01),
            nn.Tanh(),
        )
        # Bias thrust output slightly upward at initialization.

        with torch.no_grad():

            self.actor_mean[-2].bias[3] = 0.25

        # roll, pitch, yaw, thrust std in normalized action space
        self.actor_logstd = nn.Parameter(torch.tensor([[-3.0, -3.0, -4.0, -2.0]], dtype=torch.float32))

    def get_value(self, x: Tensor) -> Tensor:
        return self.critic(x)

    def get_action_and_value(
        self,
        x: Tensor,
        action: Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = action_mean if deterministic else probs.sample()

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


def train_ppo(args: Args, model_path: Path, device: torch.device) -> list[float]:
    set_seeds(args.seed, args.torch_deterministic)

    print(f"Training on device: {device} | Environment device: {args.device_env}")
    print(f"Training config: {args.config}")
    print(f"num_envs={args.num_envs}, num_steps={args.num_steps}, total_timesteps={args.total_timesteps}")

    envs = make_envs(args, torch_device=device)

    assert isinstance(envs.single_action_space, gym.spaces.Box), "Only continuous action spaces supported."

    agent = Agent(envs.single_observation_space.shape, envs.single_action_space.shape).to(device)
    optimizer = optim.AdamW(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    global_step = 0
    start_train = time.time()

    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = next_obs.to(device).float() if isinstance(next_obs, torch.Tensor) else torch.tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, device=device)

    episode_rewards = torch.zeros(args.num_envs, device=device)
    episode_reward_hist: list[float] = []

    for iteration in range(1, args.num_iterations + 1):
        iter_start = time.time()

        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs

            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminated, truncated, info = envs.step(action)

            next_obs = next_obs.to(device).float() if isinstance(next_obs, torch.Tensor) else torch.tensor(next_obs, dtype=torch.float32, device=device)
            reward = reward.to(device).float() if isinstance(reward, torch.Tensor) else torch.tensor(reward, dtype=torch.float32, device=device)
            terminated = terminated.to(device) if isinstance(terminated, torch.Tensor) else torch.tensor(terminated, device=device)
            truncated = truncated.to(device) if isinstance(truncated, torch.Tensor) else torch.tensor(truncated, device=device)

            next_done = (terminated | truncated).float()
            rewards[step] = reward

            episode_rewards += reward

            done_mask = next_done.bool()
            if done_mask.any():
                finished_rewards = episode_rewards[done_mask]
                episode_reward_hist.extend([float(x) for x in finished_rewards.detach().cpu().tolist()])
                episode_rewards[done_mask] = 0.0

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0

            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]

                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )

            returns = advantages + values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)

            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds],
                    b_actions[mb_inds],
                )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        with torch.no_grad():
            y_pred = b_values.detach().cpu().numpy()
            y_true = b_returns.detach().cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        recent_reward = np.mean(episode_reward_hist[-50:]) if episode_reward_hist else float("nan")
        sps = int(global_step / max(time.time() - start_train, 1e-6))

        print(
            f"Iter {iteration:04d}/{args.num_iterations} | "
            f"{time.time() - iter_start:.2f}s | "
            f"SPS {sps} | "
            f"recent_ep_reward {recent_reward:.2f} | "
            f"policy_loss {pg_loss.item():.3f} | "
            f"value_loss {v_loss.item():.3f} | "
            f"EV {explained_var:.3f}"
        )

    torch.save(agent.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    envs.close()
    return episode_reward_hist


def evaluate_ppo(args: Args, n_eval: int, model_path: Path) -> tuple[list[float], list[int]]:
    eval_args = Args.create(**{**vars(args), "num_envs": 1, "total_timesteps": args.num_steps})
    device = torch.device("cpu")

    env = make_envs(eval_args, torch_device=device)
    agent = Agent(env.single_observation_space.shape, env.single_action_space.shape).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    rewards_out: list[float] = []
    lengths_out: list[int] = []

    with torch.no_grad():
        for ep in range(n_eval):
            obs, _ = env.reset(seed=args.seed + ep + 123)
            obs = obs.float() if isinstance(obs, torch.Tensor) else torch.tensor(obs, dtype=torch.float32)

            done = torch.zeros(1, dtype=torch.bool)
            ep_reward = 0.0
            ep_len = 0

            while not done.any():
                action, _, _, _ = agent.get_action_and_value(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                obs = obs.float() if isinstance(obs, torch.Tensor) else torch.tensor(obs, dtype=torch.float32)

                done = terminated | truncated
                ep_reward += float(reward[0].item() if isinstance(reward, torch.Tensor) else reward[0])
                ep_len += 1

                try:
                    env.render()
                except Exception:
                    pass

            rewards_out.append(ep_reward)
            lengths_out.append(ep_len)
            print(f"Eval episode {ep + 1}: reward={ep_reward:.2f}, length={ep_len}")

    print(f"Eval mean reward={np.mean(rewards_out):.2f}, mean length={np.mean(lengths_out):.1f}")
    env.close()
    return rewards_out, lengths_out


def main(
    train: bool = True,
    eval: int = 3,
    **kwargs: Any,
) -> None:
    args = Args.create(**kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    model_path = Path(__file__).parent / "ppo_race_level2.ckpt"

    if train:
        train_ppo(args, model_path, device)

    if eval > 0:
        if not model_path.exists:
            raise FileNotFoundError(f"Model not found: {model_path}")
        evaluate_ppo(args, eval, model_path)


if __name__ == "__main__":
    fire.Fire(main)
