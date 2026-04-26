"""Improved AttitudeRL controller.

Idea:
- Keep the existing RL policy trained with train_rl.py / ppo_drone_racing.ckpt.
- Do NOT change the policy input dimension.
- Improve performance by replacing the fixed hand-tuned spline trajectory with a race-aware trajectory:
    1. Use observed/nominal gate centers as waypoints.
    2. Build a smooth cubic spline through the gates.
    3. Repel trajectory samples away from obstacles.
    4. Keep the old RL policy as a trajectory tracker.

Save this file as:
    lsy_drone_racing/control/attitude_rl_improved.py
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from drone_models.core import load_params
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as scipy_R
from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.train_rl import Agent

if TYPE_CHECKING:
    from numpy.typing import NDArray


class AttitudeRL(Controller):
    """RL trajectory-tracking controller with gate/obstacle-aware reference trajectory."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)

        self.freq = config.env.freq

        drone_params = load_params(config.sim.physics, config.sim.drone_model)
        self.drone_mass = drone_params["mass"]
        self.thrust_min = drone_params["thrust_min"] * 4
        self.thrust_max = drone_params["thrust_max"] * 4
        self.hover_thrust = self.drone_mass * 9.81

        # Policy observation settings. Must match original train_rl.py.
        self.n_obs = 2
        self.n_samples = 10
        self.samples_dt = 0.1
        self.trajectory_time = 15.0
        self.sample_offsets = np.array(
            np.arange(self.n_samples) * self.freq * self.samples_dt,
            dtype=int,
        )

        self._tick = 0
        self._finished = False

        # Rebuild trajectory occasionally because level2 observations can reveal real object poses.
        self.replan_interval_steps = int(1 * self.freq)
        self.safe_obstacle_dist = 0.45
        self.obstacle_repulsion_strength = 0.35

        self.trajectory = self._build_race_trajectory(obs)

        self.agent = Agent((13 + 3 * self.n_samples + self.n_obs * 13 + 4,), (4,)).to("cpu")

        model_path = Path(__file__).parent / "ppo_drone_racing.ckpt"
        self.agent.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        self.agent.eval()

        # Important: keep last_action in normalized RL action space [-1, 1].
        self.last_action = np.zeros(4, dtype=np.float32)

        self.basic_obs_key = ["pos", "quat", "vel", "ang_vel"]
        basic_obs = np.concatenate([obs[k] for k in self.basic_obs_key], axis=-1)
        self.prev_obs = np.tile(basic_obs[None, :], (self.n_obs, 1)).astype(np.float32)

    def compute_control(
        self,
        obs: dict[str, NDArray[np.floating]],
        info: dict | None = None,
    ) -> NDArray[np.floating]:
        """Compute collective thrust and desired roll/pitch/yaw."""

        if self._tick % self.replan_interval_steps == 0:
            self.trajectory = self._build_race_trajectory(obs)

        i = min(self._tick, self.trajectory.shape[0] - 1)
        if i == self.trajectory.shape[0] - 1:
            self._finished = True

        obs_rl = self._obs_rl(obs)
        obs_rl = torch.tensor(obs_rl, dtype=torch.float32).unsqueeze(0).to("cpu")

        with torch.no_grad():
            act, _, _, _ = self.agent.get_action_and_value(obs_rl, deterministic=True)

        act_np = act.squeeze(0).numpy().astype(np.float32)

        # Yaw is not useful here.
        act_np[2] = 0.0

        # Store normalized action, because this is what the policy saw during training.
        self.last_action = act_np.copy()

        action_scaled = self._scale_actions(act_np).astype(np.float32)
        action_scaled[2] = 0.0

        return action_scaled

    def _obs_rl(self, obs: dict[str, NDArray[np.floating]]) -> NDArray[np.floating]:
        """Build original train_rl.py policy observation."""

        obs_rl = {}

        obs_rl["basic_obs"] = np.concatenate([obs[k] for k in self.basic_obs_key], axis=-1)

        idx = np.clip(self._tick + self.sample_offsets, 0, self.trajectory.shape[0] - 1)
        dpos = self.trajectory[idx] - obs["pos"]
        obs_rl["local_samples"] = dpos.reshape(-1)

        obs_rl["prev_obs"] = self.prev_obs.reshape(-1)
        obs_rl["last_action"] = self.last_action

        self.prev_obs = np.concatenate(
            [self.prev_obs[1:, :], obs_rl["basic_obs"][None, :]],
            axis=0,
        )

        return np.concatenate([v for v in obs_rl.values()], axis=-1).astype(np.float32)

    def _scale_actions(self, actions: NDArray) -> NDArray:
        """Rescale actions from [-1, 1] to real attitude/thrust commands.

        This is intentionally more conservative than +/- pi/2 roll/pitch.
        """
        max_roll_pitch = 0.3  # rad, approx 26 deg
        max_yaw = 0.0

        scale = np.array(
            [
                max_roll_pitch,
                max_roll_pitch,
                max_yaw,
                0.3 * self.hover_thrust,
            ],
            dtype=np.float32,
        )

        mean = np.array(
            [
                0.0,
                0.0,
                0.0,
                1.05 * self.hover_thrust,
            ],
            dtype=np.float32,
        )

        action = np.clip(actions, -1.0, 1.0) * scale + mean
        action[3] = np.clip(action[3], self.thrust_min, self.thrust_max)

        return action.astype(np.float32)

    def _build_race_trajectory(self, obs: dict[str, NDArray[np.floating]]) -> NDArray[np.float32]:
        """Build a gate-centered trajectory and push it away from obstacles."""

        current_pos = np.asarray(obs["pos"], dtype=np.float32).copy()

        if "gates_pos" in obs:
            gates_pos = np.asarray(obs["gates_pos"], dtype=np.float32).copy()

            if gates_pos.ndim == 1:
                gates_pos = gates_pos.reshape(-1, 3)

            # If target_gate is available, start from the current target gate.
            target_gate = int(obs.get("target_gate", 0))
            target_gate = max(0, target_gate)

            gates_pos = gates_pos[target_gate:]

            if gates_pos.shape[0] == 0:
                gates_pos = np.asarray(
                    [current_pos + np.array([0.5, 0.0, 0.0], dtype=np.float32)],
                    dtype=np.float32,
                )

            # Gate centers. Keep z in a reasonable flight range.
            gates_pos[:, 2] = np.clip(gates_pos[:, 2], 0.55, 1.15)

            gate_waypoints = []

            gates_quat = None
            if "gates_quat" in obs:
                gates_quat = np.asarray(obs["gates_quat"], dtype=np.float32).copy()
                if gates_quat.ndim == 1:
                    gates_quat = gates_quat.reshape(-1, 4)
                gates_quat = gates_quat[target_gate:]

            for i, gate_pos in enumerate(gates_pos):
                if gates_quat is not None and i < gates_quat.shape[0]:
                    gate_rot = scipy_R.from_quat(gates_quat[i])

                    # In this repo gate passing is checked in gate local x-direction.
                    # Use local x-axis as gate normal.
                    gate_normal = gate_rot.apply(np.array([1.0, 0.0, 0.0], dtype=np.float32))
                    gate_normal = gate_normal / (np.linalg.norm(gate_normal) + 1e-6)
                else:
                    # Fallback: direction from previous point to gate.
                    prev = current_pos if i == 0 else gates_pos[i - 1]
                    gate_normal = gate_pos - prev
                    gate_normal = gate_normal / (np.linalg.norm(gate_normal) + 1e-6)

                approach_dist = 0.35
                exit_dist = 0.25

                gate_waypoints.append(gate_pos - approach_dist * gate_normal)
                gate_waypoints.append(gate_pos)
                gate_waypoints.append(gate_pos + exit_dist * gate_normal)

            gate_waypoints = np.asarray(gate_waypoints, dtype=np.float32)

            waypoints = np.vstack(
                [
                    current_pos,
                    gate_waypoints,
                ]
            )
        else:
            waypoints = self._default_waypoints()

        # Remove duplicate/nearly duplicate consecutive points.
        filtered = [waypoints[0]]
        for p in waypoints[1:]:
            if np.linalg.norm(p - filtered[-1]) > 1e-3:
                filtered.append(p)
        waypoints = np.asarray(filtered, dtype=np.float32)

        if waypoints.shape[0] < 2:
            waypoints = np.vstack(
                [
                    current_pos,
                    current_pos + np.array([0.5, 0.0, 0.3], dtype=np.float32),
                ]
            )

        n_steps = int(self.freq * self.trajectory_time)

        # Time allocation proportional to distance between waypoints.
        distances = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        distances = np.maximum(distances, 1e-3)
        t_waypoints = np.concatenate([[0.0], np.cumsum(distances)])
        t_waypoints = t_waypoints / t_waypoints[-1] * self.trajectory_time

        ts = np.linspace(0, self.trajectory_time, n_steps)

        try:
            spline = CubicSpline(t_waypoints, waypoints, bc_type="clamped")
            trajectory = spline(ts).astype(np.float32)
        except Exception:
            trajectory = np.repeat(waypoints[None, 0, :], n_steps, axis=0).astype(np.float32)

        if "obstacles_pos" in obs:
            obstacles_pos = np.asarray(obs["obstacles_pos"], dtype=np.float32)
            if obstacles_pos.ndim == 1:
                obstacles_pos = obstacles_pos.reshape(-1, 3)

            trajectory = self._repel_trajectory_from_obstacles(
                trajectory,
                obstacles_pos,
                safe_dist=self.safe_obstacle_dist,
                strength=self.obstacle_repulsion_strength,
                gates_pos=gates_pos if "gates_pos" in obs else None,
            )

        trajectory[:, 2] = np.clip(trajectory[:, 2], 0.25, 1.6)

        return trajectory.astype(np.float32)

    def _repel_trajectory_from_obstacles(
        self,
        trajectory: NDArray[np.float32],
        obstacles_pos: NDArray[np.float32],
        safe_dist: float,
        strength: float,
        gates_pos: NDArray[np.float32] | None = None,
    ) -> NDArray[np.float32]:
        """Push trajectory samples away from obstacles, but preserve gate centers."""
        traj = trajectory.copy()

        for obstacle in obstacles_pos:
            vec_xy = traj[:, :2] - obstacle[None, :2]
            dist_xy = np.linalg.norm(vec_xy, axis=-1, keepdims=True) + 1e-6

            mask = dist_xy < safe_dist

            push = strength * (safe_dist - dist_xy) * vec_xy / dist_xy

            # Do not push trajectory points that are very close to a gate center.
            # Otherwise the reference path gets moved toward the gate frame/rim.
            if gates_pos is not None:
                gate_dist = np.min(
                    np.linalg.norm(traj[:, None, :2] - gates_pos[None, :, :2], axis=-1),
                    axis=1,
                    keepdims=True,
                )
                gate_center_mask = gate_dist < 0.30
                mask = mask & (~gate_center_mask)

            traj[:, :2] += mask * push

        return traj.astype(np.float32)

    def _default_waypoints(self) -> NDArray[np.float32]:
        """Fallback trajectory from the original controller."""
        return np.array(
            [
                [-1.5, 0.75, 0.05],
                [-1.0, 0.55, 0.4],
                [0.3, 0.35, 0.7],
                [1.3, -0.15, 0.9],
                [0.85, 0.85, 1.2],
                [-0.5, -0.05, 0.7],
                [-1.2, -0.2, 0.8],
                [-1.2, -0.2, 1.2],
                [-0.0, -0.7, 1.2],
                [0.5, -0.75, 1.2],
            ],
            dtype=np.float32,
        )

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

        if "target_gate" in obs and int(obs["target_gate"]) == -1:
            self._finished = True

        return self._finished

    def episode_callback(self):
        self._tick = 0
        self._finished = False
        self.last_action = np.zeros(4, dtype=np.float32)
