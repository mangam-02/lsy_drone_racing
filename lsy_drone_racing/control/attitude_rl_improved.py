"""Gate- and obstacle-aware RL trajectory-tracking controller.

This controller keeps the existing trained RL policy:
    ppo_drone_racing.ckpt

The neural network is still only used as a low-level trajectory tracker.
The controller builds a better high-level reference trajectory:

    current position
    -> pre-gate point
    -> gate center
    -> post-gate point
    -> ...
    -> obstacle-safe spline

Save as:
    lsy_drone_racing/control/attitude_rl_gate_obstacle.py

Then in your level/config file set:
    [controller]
    file = "attitude_rl_gate_obstacle.py"
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

from crazyflow.sim.visualize import draw_line, draw_points

if TYPE_CHECKING:
    from numpy.typing import NDArray


class AttitudeRL(Controller):
    """RL trajectory tracker with gate-centered obstacle-safe reference planning."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)

        # ============================================================
        # 1. Environment / drone physics
        # ============================================================
        self.freq = config.env.freq

        drone_params = load_params(config.sim.physics, config.sim.drone_model)

        self.drone_mass = drone_params["mass"]
        self.thrust_min = drone_params["thrust_min"] * 4
        self.thrust_max = drone_params["thrust_max"] * 4
        self.hover_thrust = self.drone_mass * 9.81

        # ============================================================
        # 2. RL policy observation settings
        # Must match original train_rl.py
        # ============================================================
        self.n_obs = 2
        self.n_samples = 10
        self.samples_dt = 0.14

        self.basic_obs_key = ["pos", "quat", "vel", "ang_vel"]

        self.sample_offsets = np.array(
            np.arange(self.n_samples) * self.freq * self.samples_dt,
            dtype=int,
        )

        # ============================================================
        # 3. Global trajectory timing
        # Larger trajectory_time = slower planned reference trajectory
        # ============================================================
        self.trajectory_time = 18.0

        # ============================================================
        # 4. Target tracking / blue point behavior
        # ============================================================
        self.lookahead_index = 18
        self.sample_index_spacing = 6
        self.current_traj_index = 0

        self.max_index_advance_per_step = 1
        self.nearest_search_back = 20
        self.nearest_search_forward = 80

        # ============================================================
        # 5. Curve slowdown
        # Smaller lookahead in curves = slower and safer curve tracking
        # ============================================================
        self.min_lookahead_index = 7
        self.max_lookahead_index = 18
        self.curve_slowdown_strength = 35.0

        # ============================================================
        # 6. Replanning behavior
        # Larger value = less frequent replanning, more stable route
        # ============================================================
        self.replan_interval_steps = int(2.0 * self.freq)

        # ============================================================
        # 6b. Replanning with velocity awareness
        # Prevents replanning from suddenly pointing backwards while the drone has momentum
        # ============================================================
        self.replan_velocity_lookahead_time = 0.35  # seconds
        self.replan_velocity_max_shift = 0.35       # meters
        self.replan_velocity_min_speed = 0.15       # m/s


        # ============================================================
        # 7. Gate trajectory planning
        # pre_gate -> gate_center -> post_gate
        # ============================================================
        self.approach_dist = 0.75
        self.exit_dist = 0.80
        self.gate_center_protection_radius = 0.65
        self.skip_pre_gate_dist = 0.70

        self.old_gate_safe_dist = 1.0

        # Slow down before gates to reduce overshoot into gate rims.
        self.gate_slowdown_radius = 1.80
        self.gate_slowdown_min_lookahead = 4

        # Force a tighter artificial center corridor through gates.
        self.gate_center_blend_radius = 45
        self.gate_center_blend_strength = 0.75

        # ============================================================
        # 8. Gate-aware obstacle avoidance
        # Shifts pre/post gate points sideways if obstacle blocks corridor
        # ============================================================
        self.gate_corridor_width = 0.28
        self.gate_lateral_shift = 0.25

        # ============================================================
        # 9. General obstacle avoidance / repulsion
        # Applies to helper waypoints and dense trajectory samples
        # ============================================================
        self.obstacle_min_dist = 0.45
        self.obstacle_safe_dist = 0.35
        self.obstacle_repulsion_strength = 0.20
        self.obstacle_repulsion_iterations = 4

        # ============================================================
        # 10. Vertical flight envelope
        # Prevents planned trajectory from going too low/high
        # ============================================================
        self.min_z = 0.35
        self.max_z = 1.45

        # ============================================================
        # 11. Action scaling / controller aggressiveness
        # Larger values = faster but more overshoot risk
        # ============================================================
        self.max_roll_pitch = 0.28
        self.thrust_delta = 0.18 * self.hover_thrust
        self.thrust_mean = 1.00 * self.hover_thrust

        # ============================================================
        # 12. Internal controller state
        # ============================================================
        self._tick = 0
        self._finished = False
        self.last_action = np.zeros(4, dtype=np.float32)
        self._last_traj = None

        # ============================================================
        # 13. Initial trajectory and rendering state
        # ============================================================
        self.trajectory = self._build_safe_gate_trajectory(obs)
        self.last_target_gate = int(obs.get("target_gate", 0))
        self.last_gates_pos = None
        self.last_obstacles_pos = None
        self.replan_position_threshold = 0.05  # meters
        # Slow-down phase directly after replanning.
        self.replan_slowdown_steps = int(1.2 * self.freq)
        self.replan_slowdown_until_tick = -1

        self.replan_slowdown_max_roll_pitch = 0.15
        self.replan_slowdown_thrust_delta_factor = 0.45
        self.replan_slowdown_lookahead = 4

        # Global speed limiter.
        self.global_max_roll_pitch = 1.28
        self.global_thrust_delta_factor = 2.00
        self.global_max_lookahead = 10
        self.global_sample_index_spacing = 5

        self.debug_render = True
        self.debug_draw_detected_gates = True
        self.current_target_point = self.trajectory[0].copy()

        # ============================================================
        # 14. RL policy loading
        # ============================================================
        self.agent = Agent((13 + 3 * self.n_samples + self.n_obs * 13 + 4,), (4,)).to("cpu")

        model_path = Path(__file__).parent / "ppo_drone_racing.ckpt"
        self.agent.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        self.agent.eval()

        # ============================================================
        # 15. Previous observation buffer
        # ============================================================
        basic_obs = np.concatenate([obs[k] for k in self.basic_obs_key], axis=-1)
        self.prev_obs = np.tile(basic_obs[None, :], (self.n_obs, 1)).astype(np.float32)

    def compute_control(
        self,
        obs: dict[str, NDArray[np.floating]],
        info: dict | None = None,
    ) -> NDArray[np.floating]:
        """Compute desired [roll, pitch, yaw, thrust]."""
        self._last_obs_for_render = obs
        current_gate = int(obs.get("target_gate", 0))

        gate_advanced = current_gate > self.last_target_gate
        environment_changed = self._environment_changed(obs)

        # Replan only if:
        # 1. we passed a gate, or
        # 2. detected gates/obstacles changed position enough
        #should_replan = gate_advanced or environment_changed
        should_replan = environment_changed
        if should_replan:
            self.trajectory = self._build_safe_gate_trajectory(obs)
            self.last_target_gate = current_gate

            # Slow down briefly after replanning so the drone can adapt.
            self.replan_slowdown_until_tick = self._tick + self.replan_slowdown_steps

            # New trajectory starts near the current drone position, so reset index.
            self.current_traj_index = 0
            self.current_target_point = self.trajectory[0].copy()

        # Do not finish when the planned trajectory ends.
        # Keep the progress index valid, but do not jump it to the end because of global time.
        self.current_traj_index = int(
            np.clip(self.current_traj_index, 0, self.trajectory.shape[0] - 2)
        )

        obs_rl = self._obs_rl(obs)
        obs_tensor = torch.tensor(obs_rl, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            action, _, _, _ = self.agent.get_action_and_value(obs_tensor, deterministic=True)

        action_np = action.squeeze(0).numpy().astype(np.float32)

        # No yaw tracking needed.
        action_np[2] = 0.0

        self.last_action = action_np.copy()

        action_scaled = self._scale_actions(action_np)
        action_scaled[2] = 0.0

        return action_scaled.astype(np.float32)

    def _is_close_to_current_gate(self, obs: dict[str, NDArray[np.floating]]) -> bool:
        """Return True if the drone is close enough to the current gate to avoid replanning."""
        if "gates_pos" not in obs:
            return False

        gates_pos = np.asarray(obs["gates_pos"], dtype=np.float32)
        if gates_pos.ndim == 1:
            gates_pos = gates_pos.reshape(-1, 3)

        target_gate = int(obs.get("target_gate", 0))

        if target_gate < 0 or target_gate >= gates_pos.shape[0]:
            return False

        pos = np.asarray(obs["pos"], dtype=np.float32)
        gate_pos = gates_pos[target_gate]

        dist_xy = np.linalg.norm(pos[:2] - gate_pos[:2])

        return dist_xy < self.skip_pre_gate_dist

    def _environment_changed(self, obs: dict[str, NDArray[np.floating]]) -> bool:
        """Return True if detected gate or obstacle positions changed enough."""
        gates_pos = None
        obstacles_pos = None

        if "gates_pos" in obs:
            gates_pos = np.asarray(obs["gates_pos"], dtype=np.float32).copy()
            if gates_pos.ndim == 1:
                gates_pos = gates_pos.reshape(-1, 3)

        if "obstacles_pos" in obs:
            obstacles_pos = np.asarray(obs["obstacles_pos"], dtype=np.float32).copy()
            if obstacles_pos.ndim == 1:
                obstacles_pos = obstacles_pos.reshape(-1, 3)

        changed = False

        if gates_pos is not None:
            if self.last_gates_pos is None:
                changed = True
            elif gates_pos.shape != self.last_gates_pos.shape:
                changed = True
            else:
                max_gate_shift = np.max(np.linalg.norm(gates_pos - self.last_gates_pos, axis=1))
                if max_gate_shift > self.replan_position_threshold:
                    changed = True

        if obstacles_pos is not None:
            if self.last_obstacles_pos is None:
                changed = True
            elif obstacles_pos.shape != self.last_obstacles_pos.shape:
                changed = True
            else:
                max_obstacle_shift = np.max(
                    np.linalg.norm(obstacles_pos - self.last_obstacles_pos, axis=1)
                )
                if max_obstacle_shift > self.replan_position_threshold:
                    changed = True

        self.last_gates_pos = gates_pos
        self.last_obstacles_pos = obstacles_pos

        return changed

    def _obs_rl(self, obs: dict[str, NDArray[np.floating]]) -> NDArray[np.float32]:
        """Build observation exactly like original train_rl.py expects."""
        obs_rl = {}

        obs_rl["basic_obs"] = np.concatenate([obs[k] for k in self.basic_obs_key], axis=-1)

        # Find nearest trajectory point, but only in a small window.
        # This avoids sudden jumps forward in curves.
        pos = np.asarray(obs["pos"], dtype=np.float32)

        search_start = max(0, self.current_traj_index - self.nearest_search_back)
        search_end = min(
            self.trajectory.shape[0],
            self.current_traj_index + self.nearest_search_forward,
        )

        local_traj = self.trajectory[search_start:search_end]
        nearest_local = int(np.argmin(np.linalg.norm(local_traj - pos[None, :], axis=1)))
        nearest_idx = search_start + nearest_local

        # Never go backwards, but also never jump too far forward in one control step.
        desired_index = max(self.current_traj_index, nearest_idx)
        max_allowed_index = self.current_traj_index + self.max_index_advance_per_step
        self.current_traj_index = min(desired_index, max_allowed_index)

        dynamic_lookahead = self._curvature_based_lookahead(self.current_traj_index)
        dynamic_lookahead = self._gate_based_lookahead(obs, dynamic_lookahead)

        # Directly after replanning: use smaller lookahead.
        # This makes the blue target point less aggressive.
        if self._tick < self.replan_slowdown_until_tick:
            dynamic_lookahead = min(dynamic_lookahead, self.replan_slowdown_lookahead)

        dynamic_lookahead = min(dynamic_lookahead, self.global_max_lookahead)

        sample_index_spacing = min(
            self.sample_index_spacing,
            self.global_sample_index_spacing,
        )

        idx = (
            self.current_traj_index
            + dynamic_lookahead
            + np.arange(self.n_samples) * sample_index_spacing
        )

        idx = np.clip(idx, 0, self.trajectory.shape[0] - 1)

        self.current_target_point = self.trajectory[idx[0]].copy()

        dpos = self.trajectory[idx] - pos

        # Avoid huge tracking commands when far from trajectory.
        dpos = np.clip(dpos, -1.0, 1.0)
        dpos[:, 2] *= 0.65

        obs_rl["local_samples"] = dpos.reshape(-1)

        obs_rl["prev_obs"] = self.prev_obs.reshape(-1)
        obs_rl["last_action"] = self.last_action

        self.prev_obs = np.concatenate(
            [self.prev_obs[1:, :], obs_rl["basic_obs"][None, :]],
            axis=0,
        )

        return np.concatenate([v for v in obs_rl.values()], axis=-1).astype(np.float32)

    def _gate_based_lookahead(
        self,
        obs: dict[str, NDArray[np.floating]],
        current_lookahead: int,
    ) -> int:
        """Reduce lookahead near current gate to slow down before passing it."""
        if "gates_pos" not in obs:
            return current_lookahead

        gates_pos = np.asarray(obs["gates_pos"], dtype=np.float32)
        if gates_pos.ndim == 1:
            gates_pos = gates_pos.reshape(-1, 3)

        target_gate = int(obs.get("target_gate", 0))
        if target_gate < 0 or target_gate >= gates_pos.shape[0]:
            return current_lookahead

        pos = np.asarray(obs["pos"], dtype=np.float32)
        gate_pos = gates_pos[target_gate]

        dist_xy = np.linalg.norm(pos[:2] - gate_pos[:2])

        if dist_xy > self.gate_slowdown_radius:
            return current_lookahead

        alpha = dist_xy / self.gate_slowdown_radius

        lookahead = (
            self.gate_slowdown_min_lookahead
            + alpha * (current_lookahead - self.gate_slowdown_min_lookahead)
        )

        return int(np.clip(lookahead, self.gate_slowdown_min_lookahead, current_lookahead))

    def _curvature_based_lookahead(self, base_index: int) -> int:
        """Return smaller lookahead in curves and larger lookahead on straights."""
        i0 = int(np.clip(base_index, 0, self.trajectory.shape[0] - 1))
        i1 = int(np.clip(base_index + 10, 0, self.trajectory.shape[0] - 1))
        i2 = int(np.clip(base_index + 25, 0, self.trajectory.shape[0] - 1))

        p0 = self.trajectory[i0, :2]
        p1 = self.trajectory[i1, :2]
        p2 = self.trajectory[i2, :2]

        v1 = p1 - p0
        v2 = p2 - p1

        n1 = np.linalg.norm(v1) + 1e-6
        n2 = np.linalg.norm(v2) + 1e-6

        cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        angle = np.arccos(cos_angle)

        slowdown = self.curve_slowdown_strength * angle

        lookahead = self.max_lookahead_index - slowdown
        lookahead = np.clip(lookahead, self.min_lookahead_index, self.max_lookahead_index)

        return int(lookahead)

    def _scale_actions(self, actions: NDArray[np.floating]) -> NDArray[np.float32]:
        """Scale normalized policy action [-1, 1] to real attitude/thrust command."""
        max_roll_pitch = min(self.max_roll_pitch, self.global_max_roll_pitch)
        thrust_delta = self.thrust_delta * self.global_thrust_delta_factor

        # Directly after replanning: reduce aggressiveness.
        if self._tick < self.replan_slowdown_until_tick:
            max_roll_pitch = self.replan_slowdown_max_roll_pitch
            thrust_delta = self.thrust_delta * self.replan_slowdown_thrust_delta_factor

        scale = np.array(
            [
                max_roll_pitch,
                max_roll_pitch,
                0.0,
                thrust_delta,
            ],
            dtype=np.float32,
        )

        mean = np.array(
            [
                0.0,
                0.0,
                0.0,
                self.thrust_mean,
            ],
            dtype=np.float32,
        )

        action = np.clip(actions, -1.0, 1.0) * scale + mean
        action[3] = np.clip(action[3], self.thrust_min, self.thrust_max)

        return action.astype(np.float32)

    def _velocity_aware_replan_start(
        self,
        obs: dict[str, NDArray[np.floating]],
        current_pos: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Return a replanning start point that respects current drone velocity."""
        start_pos = current_pos.copy()

        if "vel" not in obs:
            return start_pos.astype(np.float32)

        vel = np.asarray(obs["vel"], dtype=np.float32).copy()
        speed_xy = np.linalg.norm(vel[:2])

        # If the drone is almost standing still, do not shift the start point.
        if speed_xy < self.replan_velocity_min_speed:
            return start_pos.astype(np.float32)

        # Predict a short distance ahead using current velocity.
        shift = vel * self.replan_velocity_lookahead_time

        # Limit shift length, otherwise high speed creates too large jumps.
        shift_norm = np.linalg.norm(shift[:2])
        if shift_norm > self.replan_velocity_max_shift:
            shift[:2] *= self.replan_velocity_max_shift / (shift_norm + 1e-6)

        # Mostly use xy momentum. Keep z conservative to avoid vertical jumps.
        shift[2] *= 0.25

        start_pos = current_pos + shift
        start_pos[2] = np.clip(start_pos[2], self.min_z, self.max_z)

        return start_pos.astype(np.float32)

    def _build_safe_gate_trajectory(
        self,
        obs: dict[str, NDArray[np.floating]],
    ) -> NDArray[np.float32]:
        """Build trajectory through dynamically observed gate centers."""

        current_pos = np.asarray(obs["pos"], dtype=np.float32).copy()
        current_pos[2] = np.clip(current_pos[2], self.min_z, self.max_z)

        start_pos = self._velocity_aware_replan_start(obs, current_pos)

        if "gates_pos" not in obs:
            return self._default_trajectory(start_pos)

        all_gates_pos = np.asarray(obs["gates_pos"], dtype=np.float32).copy()
        if all_gates_pos.ndim == 1:
            all_gates_pos = all_gates_pos.reshape(-1, 3)

        target_gate = int(obs.get("target_gate", 0))
        target_gate = max(0, target_gate)

        # These are the actual observed gate centers.
        # Do not modify them. They are the hard constraints.
        gates_pos_raw = all_gates_pos[target_gate:].copy()

        # Planning copy: only helper points may use clipped z if needed.
        gates_pos = gates_pos_raw.copy()

        # Old gates were already passed, but they are still physical objects.
        # They are used as obstacles so the replanned trajectory does not route through them.
        old_gates_pos = all_gates_pos[:target_gate].copy()

        if gates_pos.shape[0] == 0:
            return self._default_trajectory(start_pos)

        #gates_pos[:, 2] = np.clip(gates_pos[:, 2], 0.55, 1.20)

        gates_quat = None
        if "gates_quat" in obs:
            gates_quat = np.asarray(obs["gates_quat"], dtype=np.float32).copy()
            if gates_quat.ndim == 1:
                gates_quat = gates_quat.reshape(-1, 4)
            gates_quat = gates_quat[target_gate:].copy()

        waypoints = [start_pos]

        for i, gate_pos in enumerate(gates_pos):
            gate_normal = self._gate_normal(
                gate_index=i,
                gates_pos=gates_pos,
                gates_quat=gates_quat,
                current_pos=current_pos,
            )

            pre_gate = gate_pos - self.approach_dist * gate_normal
            post_gate = gate_pos + self.exit_dist * gate_normal

            pre_gate[2] = np.clip(pre_gate[2], self.min_z, self.max_z)
            post_gate[2] = np.clip(post_gate[2], self.min_z, self.max_z)

            dist_to_gate = np.linalg.norm(current_pos[:2] - gate_pos[:2])

            if i == 0 and dist_to_gate < self.skip_pre_gate_dist:
                waypoints.append(gate_pos.astype(np.float32))
                waypoints.append(post_gate.astype(np.float32))
            else:
                waypoints.append(pre_gate.astype(np.float32))
                waypoints.append(gate_pos.astype(np.float32))
                waypoints.append(post_gate.astype(np.float32))

        waypoints = np.asarray(waypoints, dtype=np.float32)
        waypoints = self._remove_duplicate_waypoints(waypoints)

        # Use only real obstacles from the current observation.
        # Do NOT add old gates as obstacles.
        obstacles_pos = None

        # Real obstacles from current observation.
        real_obstacles_pos = None
        if "obstacles_pos" in obs:
            real_obstacles_pos = np.asarray(obs["obstacles_pos"], dtype=np.float32).copy()
            if real_obstacles_pos.ndim == 1:
                real_obstacles_pos = real_obstacles_pos.reshape(-1, 3)

        # Old gates are already passed, but they are still physical objects.
        # They must be avoided, otherwise the drone can crash into them after replanning.
        old_gate_obstacles_pos = None
        if old_gates_pos.shape[0] > 0:
            old_gate_obstacles_pos = old_gates_pos.copy()
            old_gate_obstacles_pos[:, 2] = np.clip(old_gate_obstacles_pos[:, 2], 0.55, 1.20)

        # Combined obstacle list:
        # - real obstacles
        # - old gates
        # Future gates are NOT included here because they are hard constraints.
        if real_obstacles_pos is not None and old_gate_obstacles_pos is not None:
            obstacles_pos = np.vstack([real_obstacles_pos, old_gate_obstacles_pos])
        elif real_obstacles_pos is not None:
            obstacles_pos = real_obstacles_pos
        elif old_gate_obstacles_pos is not None:
            obstacles_pos = old_gate_obstacles_pos
        else:
            obstacles_pos = None

        if obstacles_pos is not None:
            waypoints = self._avoid_obstacles_in_waypoints(
                waypoints=waypoints,
                obstacles_pos=obstacles_pos,
                gate_centers=gates_pos,
            )

        trajectory = self._spline_from_waypoints(waypoints)

        # Then push the dense trajectory away from real obstacles.
        if obstacles_pos is not None:
            trajectory = self._repel_trajectory_from_obstacles(
                trajectory=trajectory,
                obstacles_pos=obstacles_pos,
                gate_centers=gates_pos,
            )

            trajectory = self._enforce_min_obstacle_distance(
                trajectory=trajectory,
                obstacles_pos=obstacles_pos,
                gate_centers=gates_pos,
            )

        trajectory = self._smooth_xy(trajectory, window=31)

        # Keep spacing uniform.
        trajectory = self._resample_trajectory_by_arclength(trajectory)

        trajectory = self._enforce_gate_center_hard_constraints(
            trajectory=trajectory,
            gates_pos=gates_pos_raw,
        )

        #trajectory[:, 2] = np.clip(trajectory[:, 2], self.min_z, self.max_z)

        self._last_traj = trajectory.copy()
        return trajectory.astype(np.float32)

    def _enforce_gate_center_hard_constraints(
        self,
        trajectory: NDArray[np.float32],
        gates_pos: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Force trajectory to pass through gate centers in correct gate order."""

        traj = trajectory.copy()

        if gates_pos.shape[0] == 0 or traj.shape[0] < 2:
            return traj.astype(np.float32)

        n = traj.shape[0]
        last_gate_idx = 0

        for gate_pos in gates_pos:
            # Search only after the previous gate.
            search_start = last_gate_idx + 1
            search_end = n - 1

            if search_start >= search_end:
                break

            local_traj = traj[search_start:search_end]
            local_dist = np.linalg.norm(local_traj - gate_pos[None, :], axis=1)

            gate_idx = search_start + int(np.argmin(local_dist))

            # Set exact gate center.
            traj[gate_idx] = gate_pos.astype(np.float32)

            # Pull a wider local neighborhood toward the gate center.
            # This artificially makes the usable gate opening smaller.
            blend_radius = self.gate_center_blend_radius
            start = max(0, gate_idx - blend_radius)
            end = min(n - 1, gate_idx + blend_radius)

            for i in range(start, end + 1):
                if i == gate_idx:
                    continue

                alpha = 1.0 - abs(i - gate_idx) / blend_radius
                alpha = np.clip(alpha, 0.0, 1.0)

                # Stronger pull => trajectory stays closer to gate center.
                alpha *= self.gate_center_blend_strength

                traj[i] = (1.0 - alpha) * traj[i] + alpha * gate_pos

            # Re-enforce exact gate center.
            traj[gate_idx] = gate_pos.astype(np.float32)

            last_gate_idx = gate_idx

        return traj.astype(np.float32)

    def _gate_normal(
        self,
        gate_index: int,
        gates_pos: NDArray[np.float32],
        gates_quat: NDArray[np.float32] | None,
        current_pos: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Return gate normal direction used for pre/post gate points."""

        if gates_quat is not None and gate_index < gates_quat.shape[0]:
            rot = scipy_R.from_quat(gates_quat[gate_index])
            normal = rot.apply(np.array([1.0, 0.0, 0.0], dtype=np.float32))
            normal = normal.astype(np.float32)
        else:
            prev = current_pos if gate_index == 0 else gates_pos[gate_index - 1]
            normal = gates_pos[gate_index] - prev

        normal[2] = 0.0

        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            if gate_index + 1 < gates_pos.shape[0]:
                normal = gates_pos[gate_index + 1] - gates_pos[gate_index]
                normal[2] = 0.0
                norm = np.linalg.norm(normal)

        if norm < 1e-6:
            normal = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            norm = 1.0

        return (normal / norm).astype(np.float32)

    def _shift_gate_helper_if_blocked(
        self,
        helper_point: NDArray[np.float32],
        gate_pos: NDArray[np.float32],
        gate_normal: NDArray[np.float32],
        obstacles_pos: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Shift pre/post-gate helper point sideways if obstacle blocks gate corridor.

        The actual gate center is not moved. Only the approach/exit helper point is shifted.
        """
        helper = helper_point.copy()

        # Gate tangent in xy-plane: perpendicular to gate normal.
        tangent = np.array([-gate_normal[1], gate_normal[0], 0.0], dtype=np.float32)
        tangent_norm = np.linalg.norm(tangent[:2])

        if tangent_norm < 1e-6:
            return helper.astype(np.float32)

        tangent = tangent / tangent_norm

        segment_vec = helper[:2] - gate_pos[:2]
        segment_len = np.linalg.norm(segment_vec)

        if segment_len < 1e-6:
            return helper.astype(np.float32)

        segment_dir = segment_vec / segment_len

        blocked = False

        for obstacle in obstacles_pos:
            # Check distance from obstacle to segment gate_pos -> helper_point.
            dist = self._distance_point_to_segment_xy(
                point=obstacle[:2],
                a=gate_pos[:2],
                b=helper[:2],
            )

            if dist < self.gate_corridor_width:
                blocked = True
                break

        if not blocked:
            return helper.astype(np.float32)

        # Try both lateral sides and choose the one with larger obstacle clearance.
        candidate_left = helper + self.gate_lateral_shift * tangent
        candidate_right = helper - self.gate_lateral_shift * tangent

        score_left = np.min(np.linalg.norm(obstacles_pos[:, :2] - candidate_left[None, :2], axis=-1))
        score_right = np.min(np.linalg.norm(obstacles_pos[:, :2] - candidate_right[None, :2], axis=-1))

        helper = candidate_left if score_left > score_right else candidate_right
        helper[2] = np.clip(helper[2], self.min_z, self.max_z)

        return helper.astype(np.float32)


    def _distance_point_to_segment_xy(
        self,
        point: NDArray[np.float32],
        a: NDArray[np.float32],
        b: NDArray[np.float32],
    ) -> float:
        """Distance from xy point to xy line segment."""
        ab = b - a
        denom = float(np.dot(ab, ab))

        if denom < 1e-8:
            return float(np.linalg.norm(point - a))

        t = float(np.dot(point - a, ab) / denom)
        t = np.clip(t, 0.0, 1.0)

        closest = a + t * ab
        return float(np.linalg.norm(point - closest))

    def _avoid_obstacles_in_waypoints(
        self,
        waypoints: NDArray[np.float32],
        obstacles_pos: NDArray[np.float32],
        gate_centers: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Move helper waypoints away from obstacles while preserving gate centers."""

        adjusted = waypoints.copy()

        for _ in range(self.obstacle_repulsion_iterations):
            for i in range(adjusted.shape[0]):
                p = adjusted[i]

                # Do not move current position.
                if i == 0:
                    continue

                # Protect actual gate centers.
                dist_to_gate = np.min(np.linalg.norm(gate_centers[:, :2] - p[None, :2], axis=-1))
                if dist_to_gate < self.gate_center_protection_radius:
                    continue

                for obs_pos in obstacles_pos:
                    vec = p[:2] - obs_pos[:2]
                    dist = np.linalg.norm(vec) + 1e-6

                    if dist < self.obstacle_safe_dist:
                        push = (
                            self.obstacle_repulsion_strength
                            * (self.obstacle_safe_dist - dist)
                            * vec
                            / dist
                        )
                        adjusted[i, :2] += push.astype(np.float32)

        adjusted[:, 2] = np.clip(adjusted[:, 2], self.min_z, self.max_z)
        return adjusted.astype(np.float32)

    def _enforce_min_obstacle_distance(
        self,
        trajectory: NDArray[np.float32],
        obstacles_pos: NDArray[np.float32],
        gate_centers: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Project trajectory points outward so obstacle_min_dist is not violated."""
        traj = trajectory.copy()

        for obstacle in obstacles_pos:
            vec = traj[:, :2] - obstacle[None, :2]
            dist = np.linalg.norm(vec, axis=-1, keepdims=True)

            too_close = dist < self.obstacle_min_dist

            # Avoid division by zero.
            direction = vec / (dist + 1e-6)

            projected_xy = obstacle[None, :2] + direction * self.obstacle_min_dist

            # Do not move exact gate-center points too strongly.
            gate_dist = np.min(
                np.linalg.norm(traj[:, None, :2] - gate_centers[None, :, :2], axis=-1),
                axis=1,
                keepdims=True,
            )

            # Near gate center, only project if really dangerously close to obstacle.
            gate_center_zone = gate_dist < self.gate_center_protection_radius
            critical = dist < 0.22

            mask = too_close & ((~gate_center_zone) | critical)

            traj[:, :2] = np.where(mask, projected_xy, traj[:, :2])

        return traj.astype(np.float32)



    def _repel_trajectory_from_obstacles(
        self,
        trajectory: NDArray[np.float32],
        obstacles_pos: NDArray[np.float32],
        gate_centers: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Repel dense trajectory from obstacle centers while protecting gate centers."""

        traj = trajectory.copy()

        for _ in range(self.obstacle_repulsion_iterations):
            for obstacle in obstacles_pos:
                vec_xy = traj[:, :2] - obstacle[None, :2]
                dist_xy = np.linalg.norm(vec_xy, axis=-1, keepdims=True) + 1e-6

                push = (
                    self.obstacle_repulsion_strength
                    * (self.obstacle_safe_dist - dist_xy)
                    * vec_xy
                    / dist_xy
                )

                obstacle_mask = dist_xy < self.obstacle_safe_dist

                gate_dist = np.min(
                    np.linalg.norm(traj[:, None, :2] - gate_centers[None, :, :2], axis=-1),
                    axis=1,
                    keepdims=True,
                )

                # Near gates, only allow obstacle repulsion if the obstacle is really close.
                # Otherwise keep the trajectory close to the gate center.
                gate_near_mask = gate_dist < 0.75
                obstacle_critical_mask = dist_xy < 0.22

                mask = obstacle_mask & ((~gate_near_mask) | obstacle_critical_mask)
                traj[:, :2] += mask * push

        return traj.astype(np.float32)

    def _spline_from_waypoints(self, waypoints: NDArray[np.float32]) -> NDArray[np.float32]:
        """Create smooth trajectory from waypoints."""

        path_length = float(np.sum(np.linalg.norm(np.diff(waypoints, axis=0), axis=1)))

        reference_speed = 0.4  # m/s, higher = faster trajectory
        min_trajectory_time = 5.0
        max_trajectory_time = self.trajectory_time

        trajectory_time = np.clip(
            path_length / reference_speed,
            min_trajectory_time,
            max_trajectory_time,
        )

        n_steps = int(self.freq * trajectory_time)

        if waypoints.shape[0] < 2:
            return np.repeat(waypoints[None, 0, :], n_steps, axis=0).astype(np.float32)

        distances = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        distances = np.maximum(distances, 1e-3)

        t_waypoints = np.concatenate([[0.0], np.cumsum(distances)])
        
        t_waypoints = t_waypoints / t_waypoints[-1] * trajectory_time

        ts = np.linspace(0.0, trajectory_time, n_steps)

        try:
            spline = CubicSpline(t_waypoints, waypoints, bc_type="clamped")
            traj = spline(ts).astype(np.float32)

            # Smooth vertical motion: avoid steep z jumps.
            z = traj[:, 2].copy()
            window = 151
            kernel = np.ones(window, dtype=np.float32) / window
            z_smooth = np.convolve(z, kernel, mode="same")

            # Keep start/end stable after convolution.
            z_smooth[: window // 2] = z[: window // 2]
            z_smooth[-window // 2 :] = z[-window // 2 :]

            traj[:, 2] = z_smooth
        except Exception:
            traj = np.zeros((n_steps, 3), dtype=np.float32)
            for dim in range(3):
                traj[:, dim] = np.interp(ts, t_waypoints, waypoints[:, dim])

        return traj.astype(np.float32)

    def _smooth_xy(
        self,
        trajectory: NDArray[np.float32],
        window: int = 31,
    ) -> NDArray[np.float32]:
        """Smooth x/y trajectory after obstacle projection."""
        if window < 3 or trajectory.shape[0] < window:
            return trajectory.astype(np.float32)

        traj = trajectory.copy()
        kernel = np.ones(window, dtype=np.float32) / window
        half = window // 2

        for dim in [0, 1]:
            smoothed = np.convolve(traj[:, dim], kernel, mode="same")
            smoothed[:half] = traj[:half, dim]
            smoothed[-half:] = traj[-half:, dim]
            traj[:, dim] = smoothed

        return traj.astype(np.float32)

    def _resample_trajectory_by_arclength(
        self,
        trajectory: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Resample trajectory so consecutive points have approximately equal spatial distance."""
        if trajectory.shape[0] < 2:
            return trajectory.astype(np.float32)

        segment_lengths = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])

        total_length = cumulative[-1]
        if total_length < 1e-6:
            return trajectory.astype(np.float32)

        n_steps = trajectory.shape[0]
        uniform_s = np.linspace(0.0, total_length, n_steps)

        resampled = np.zeros_like(trajectory, dtype=np.float32)
        for dim in range(3):
            resampled[:, dim] = np.interp(uniform_s, cumulative, trajectory[:, dim])

        return resampled.astype(np.float32)


    def _remove_duplicate_waypoints(
        self,
        waypoints: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        filtered = [waypoints[0]]

        for p in waypoints[1:]:
            if np.linalg.norm(p - filtered[-1]) > 1e-3:
                filtered.append(p)

        return np.asarray(filtered, dtype=np.float32)

    def _default_trajectory(self, current_pos: NDArray[np.float32]) -> NDArray[np.float32]:
        """Fallback trajectory if gate observations are unavailable."""

        waypoints = np.array(
            [
                current_pos,
                current_pos + np.array([0.3, 0.0, 0.3], dtype=np.float32),
                current_pos + np.array([0.8, 0.0, 0.6], dtype=np.float32),
            ],
            dtype=np.float32,
        )

        return self._spline_from_waypoints(waypoints)

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

    def render_callback(self, sim):
        """Render planned trajectory, current target point, and detected gate centers."""
        if not self.debug_render:
            return

        try:
            if self.trajectory is not None and len(self.trajectory) > 1:
                draw_line(
                    sim,
                    self.trajectory[::10],
                    rgba=np.array([0.0, 1.0, 0.0, 0.8]),
                    start_size=2.0,
                    end_size=2.0,
                )

                draw_points(
                    sim,
                    self.trajectory[::25],
                    rgba=np.array([1.0, 0.0, 0.0, 1.0]),
                    size=0.02,
                )

            if self.current_target_point is not None:
                draw_points(
                    sim,
                    self.current_target_point[None, :],
                    rgba=np.array([0.0, 0.0, 1.0, 1.0]),
                    size=0.05,
                )

            # ============================================================
            # Debug: draw detected gate centers from the latest observation
            # ============================================================
            if self.debug_draw_detected_gates and hasattr(self, "_last_obs_for_render"):
                obs = self._last_obs_for_render

                if "gates_pos" in obs:
                    gates_pos = np.asarray(obs["gates_pos"], dtype=np.float32)
                    if gates_pos.ndim == 1:
                        gates_pos = gates_pos.reshape(-1, 3)

                    target_gate = int(obs.get("target_gate", 0))

                    # Draw all detected gates in cyan
                    if gates_pos.shape[0] > 0:
                        draw_points(
                            sim,
                            gates_pos,
                            rgba=np.array([0.0, 1.0, 1.0, 1.0]),
                            size=0.06,
                        )

                    # Draw current target gate larger in yellow
                    if 0 <= target_gate < gates_pos.shape[0]:
                        draw_points(
                            sim,
                            gates_pos[target_gate][None, :],
                            rgba=np.array([1.0, 1.0, 0.0, 1.0]),
                            size=0.12,
                        )
            # ============================================================
            # Debug: draw detected obstacle centers from the latest observation
            # ============================================================
            if self.debug_draw_detected_gates and hasattr(self, "_last_obs_for_render"):
                obs = self._last_obs_for_render

                if "obstacles_pos" in obs:
                    obstacles_pos = np.asarray(obs["obstacles_pos"], dtype=np.float32)
                    if obstacles_pos.ndim == 1:
                        obstacles_pos = obstacles_pos.reshape(-1, 3)

                    if obstacles_pos.shape[0] > 0:
                        draw_points(
                            sim,
                            obstacles_pos,
                            rgba=np.array([1.0, 0.0, 1.0, 1.0]),
                            size=0.10,
                        )
        except Exception:
            pass

    def episode_callback(self):
        self._tick = 0
        self._finished = False
        self.last_action = np.zeros(4, dtype=np.float32)
        self.current_traj_index = 0
