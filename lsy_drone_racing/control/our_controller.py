# ruff: noqa
"""RL controller trained with train_race_rl_level2.py."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from drone_models.core import load_params
from torch import Tensor
from torch.distributions.normal import Normal

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


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

        self.actor_logstd = nn.Parameter(
            torch.tensor([[-1.0, -1.0, -2.0, 0.5]], dtype=torch.float32)
        )

    def get_value(self, x: Tensor) -> Tensor:
        return self.critic(x)

    def get_action_and_value(
        self, x: Tensor, action: Tensor | None = None, deterministic: bool = False
    ):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = action_mean if deterministic else probs.sample()

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


class AttitudeRL(Controller):
    """Controller for policies trained with train_race_rl_level2.py."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)

        drone_params = load_params(config.sim.physics, config.sim.drone_model)
        self.drone_mass = drone_params["mass"]
        self.thrust_min = drone_params["thrust_min"] * 4
        self.thrust_max = drone_params["thrust_max"] * 4
        self.hover_thrust = self.drone_mass * 9.81
        self.takeoff_steps = int(0.35 * config.env.freq)

        self._finished = False
        self._tick = 0

        obs_vec = self._flatten_obs(obs)
        obs_dim = obs_vec.shape[0]

        self.agent = Agent((obs_dim,), (4,)).to("cpu")

        model_path = Path(__file__).parent / "ppo_race_level2.ckpt"
        self.agent.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        self.agent.eval()

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:

        # Takeoff guard: first second, go straight up with zero attitude.
        if self._tick < self.takeoff_steps:
            return np.array(
                [
                    0.0,  # roll
                    0.0,  # pitch
                    0.0,  # yaw
                    1.25 * self.hover_thrust,  # thrust
                ],
                dtype=np.float32,
            )

        obs_vec = self._flatten_obs(obs)
        obs_tensor = torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            action, _, _, _ = self.agent.get_action_and_value(obs_tensor, deterministic=True)

        action_np = action.squeeze(0).numpy().astype(np.float32)

        # yaw is not needed
        action_np[2] = 0.0

        action_scaled = self._scale_actions(action_np).astype(np.float32)
        action_scaled[2] = 0.0

        return action_scaled

    def _scale_actions(self, actions: NDArray) -> NDArray:
        """Rescale actions from [-1, 1] to conservative attitude/thrust commands."""
        max_roll_pitch = 0.35  # rad, about 20 degrees
        max_yaw = 0.0

        scale = np.array(
            [max_roll_pitch, max_roll_pitch, max_yaw, 0.35 * self.hover_thrust], dtype=np.float32
        )

        mean = np.array([0.0, 0.0, 0.0, 1.05 * self.hover_thrust], dtype=np.float32)

        action = np.clip(actions, -1.0, 1.0) * scale + mean

        action[3] = np.clip(action[3], self.thrust_min, self.thrust_max)

        return action

    def _flatten_obs(self, obs: dict[str, NDArray[np.floating]]) -> NDArray[np.float32]:
        """Flatten dict observation exactly like train_race_rl_level2.py.

        Important: this relies on the same obs dict insertion order as VecDroneRaceEnv.
        """
        parts = []

        for value in obs.values():
            arr = np.asarray(value)

            # Single env: no batch dimension.
            arr = arr.reshape(-1)
            parts.append(arr)

        flat = np.concatenate(parts, axis=-1).astype(np.float32)
        flat = np.nan_to_num(flat, nan=0.0, posinf=1e6, neginf=-1e6)

        return flat

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        self._tick += 1

        # Stop if race env says all gates are done
        if "target_gate" in obs and int(obs["target_gate"]) == -1:
            self._finished = True

        return self._finished

    def episode_callback(self):
        self._tick = 0
        self._finished = False
