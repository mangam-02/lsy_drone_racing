"""Robust attitude controller for level-2 drone racing.."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from crazyflow.sim.visualize import draw_line, draw_points
from drone_models.core import load_params
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


class NewController(Controller):
    """Self-contained slow and stable attitude controller."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the controller."""
        super().__init__(obs, info, config)

        self.freq = float(config.env.freq)
        self.gravity_const = 9.81
        self.max_episode_time = 25.0

        drone_params = load_params(config.sim.physics, config.sim.drone_model)
        self.drone_mass = float(drone_params["mass"])
        self.thrust_min = float(drone_params["thrust_min"] * 4.0)
        self.thrust_max = float(drone_params["thrust_max"] * 4.0)

        self.pos_gain = np.array([0.55, 0.55, 1.25], dtype=np.float64)
        self.vel_gain = np.array([0.95, 0.95, 0.85], dtype=np.float64)

        self.tilt_limit_rad = np.deg2rad(30.0)
        self.ref_acc_limit = 4.5

        self.fixed_obstacle_pos = np.array(
            [[0.08, 0.72, 1.60], [0.95, 0.32, 1.60], [-1.42, -0.18, 1.60], [-0.58, -0.70, 1.60]],
            dtype=np.float64,
        )
        self._initial_gate_positions: NDArray[np.floating] | None = None
        self._latest_gate_positions: NDArray[np.floating] | None = None

        self._initial_obstacle_positions = self.fixed_obstacle_pos.copy()
        self._latest_obstacle_positions = self.fixed_obstacle_pos.copy()

        self._debug_path_points: NDArray[np.floating] | None = None
        self._debug_sampled_path: NDArray[np.floating] | None = None
        self._debug_enabled = True

        self.segment_durations = np.array([2.0, 2.5, 2.0, 2.5], dtype=np.float64)

        self._tick = 0
        self._finished = False
        self._current_target_idx = -1
        self._segment_start_times = np.zeros(4, dtype=np.float64)

        self._path_spline: CubicHermiteSpline | None = None
        self._path_end_time = self.max_episode_time
        self._cached_gate_positions: NDArray[np.floating] | None = None
        self._cached_obstacle_positions: NDArray[np.floating] | None = None
        self._previous_command = np.array(
            [0.0, 0.0, 0.0, self.drone_mass * self.gravity_const], dtype=np.float32
        )

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next control command."""
        del info

        sim_time = min(self._tick / self.freq, self.max_episode_time)

        gate_idx = self._read_target_gate(obs)
        if gate_idx < 0 or gate_idx >= 4:
            self._finished = True
            return self._previous_command

        if sim_time >= self.max_episode_time:
            self._finished = True
            return self._previous_command

        drone_pos = np.asarray(obs["pos"], dtype=np.float64)
        drone_vel = np.asarray(obs["vel"], dtype=np.float64)
        drone_quat = np.asarray(obs["quat"], dtype=np.float64)

        measured_gate_pos = np.asarray(obs["gates_pos"], dtype=np.float64)
        measured_gate_quat = np.asarray(obs["gates_quat"], dtype=np.float64)
        measured_gate_angles = R.from_quat(measured_gate_quat).as_euler("xyz", degrees=False)

        self._store_gate_positions(measured_gate_pos)

        if "obstacles_pos" in obs:
            measured_obstacle_pos = np.asarray(obs["obstacles_pos"], dtype=np.float64)
        else:
            measured_obstacle_pos = self.fixed_obstacle_pos.copy()

        self._store_obstacle_positions(measured_obstacle_pos)

        gate_pos_changed = (
            self._cached_gate_positions is None
            or np.max(np.linalg.norm(measured_gate_pos - self._cached_gate_positions, axis=1))
            > 0.01
        )

        obstacle_pos_changed = (
            self._cached_obstacle_positions is None
            or np.max(
                np.linalg.norm(measured_obstacle_pos - self._cached_obstacle_positions, axis=1)
            )
            > 0.01
        )

        should_rebuild_path = (
            self._path_spline is None
            or self._current_target_idx != gate_idx
            or gate_pos_changed
            or obstacle_pos_changed
        )

        if should_rebuild_path:
            if self._current_target_idx != gate_idx:
                self._segment_start_times[gate_idx] = sim_time

            self._current_target_idx = gate_idx
            self._cached_gate_positions = measured_gate_pos.copy()
            self._cached_obstacle_positions = measured_obstacle_pos.copy()

            self._path_spline, self._path_end_time = self._make_path_spline(
                gate_idx=gate_idx,
                gate_pos=measured_gate_pos,
                gate_angles=measured_gate_angles,
                start_time=float(self._segment_start_times[gate_idx]),
                travel_time=float(self.segment_durations[gate_idx]),
            )

        spline_time = min(sim_time, self._path_end_time)
        command = self._make_attitude_command(
            spline=self._get_path_spline(),
            drone_pos=drone_pos,
            drone_vel=drone_vel,
            drone_quat=drone_quat,
            spline_time=spline_time,
        )

        self._previous_command = command
        return command

    def _make_attitude_command(
        self,
        spline: CubicHermiteSpline,
        drone_pos: NDArray[np.floating],
        drone_vel: NDArray[np.floating],
        drone_quat: NDArray[np.floating],
        spline_time: float,
    ) -> NDArray[np.floating]:
        wanted_pos = spline(spline_time)
        wanted_vel = spline.derivative(1)(spline_time)
        wanted_acc = spline.derivative(2)(spline_time)

        wanted_acc_norm = np.linalg.norm(wanted_acc)
        if wanted_acc_norm > self.ref_acc_limit:
            wanted_acc = wanted_acc * self.ref_acc_limit / (wanted_acc_norm + 1e-9)

        pos_error = wanted_pos - drone_pos
        vel_error = wanted_vel - drone_vel

        force_cmd = (
            self.pos_gain * pos_error
            + self.vel_gain * vel_error
            + 0.25 * self.drone_mass * wanted_acc
        )
        force_cmd[2] += self.drone_mass * self.gravity_const

        # Convert desired force to desired acceleration.
        acc_cmd = force_cmd / self.drone_mass

        # Direct roll/pitch computation from desired acceleration.
        yaw_cmd = 0.0

        roll_cmd = np.arcsin(
            np.clip(
                (acc_cmd[0] * np.sin(yaw_cmd) - acc_cmd[1] * np.cos(yaw_cmd)) / self.gravity_const,
                -1.0,
                1.0,
            )
        )

        pitch_cmd = np.arctan2(
            acc_cmd[0] * np.cos(yaw_cmd) + acc_cmd[1] * np.sin(yaw_cmd),
            self.gravity_const + acc_cmd[2],
        )

        roll_cmd = np.clip(roll_cmd, -self.tilt_limit_rad, self.tilt_limit_rad)
        pitch_cmd = np.clip(pitch_cmd, -self.tilt_limit_rad, self.tilt_limit_rad)

        actual_z_axis = R.from_quat(drone_quat).as_matrix()[:, 2]
        thrust_cmd = float(force_cmd.dot(actual_z_axis))
        thrust_cmd = np.clip(thrust_cmd, self.thrust_min, self.thrust_max)

        return np.array([roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd], dtype=np.float32)

    def _limit_body_z_tilt(self, desired_z_axis: NDArray[np.floating]) -> NDArray[np.floating]:
        world_z_axis = np.array([0.0, 0.0, 1.0])
        tilt_angle = np.arccos(np.clip(np.dot(desired_z_axis, world_z_axis), -1.0, 1.0))

        if tilt_angle <= self.tilt_limit_rad:
            return desired_z_axis

        horizontal_part = desired_z_axis.copy()
        horizontal_part[2] = 0.0
        horizontal_norm = np.linalg.norm(horizontal_part)

        if horizontal_norm <= 1e-9:
            return desired_z_axis

        horizontal_part = horizontal_part / horizontal_norm * np.sin(self.tilt_limit_rad)
        limited_z_axis = np.array(
            [horizontal_part[0], horizontal_part[1], np.cos(self.tilt_limit_rad)]
        )
        return limited_z_axis / np.linalg.norm(limited_z_axis)

    def _make_path_spline(
        self,
        gate_idx: int,
        gate_pos: NDArray[np.floating],
        gate_angles: NDArray[np.floating],
        start_time: float,
        travel_time: float,
    ) -> tuple[CubicHermiteSpline, float]:
        path_points = self._make_checkpoint_list(gate_idx, gate_pos, gate_angles)
        path_points = self._push_points_away_from_obstacles(path_points)

        self._store_path_points(path_points)

        end_time = start_time + travel_time

        segment_lengths = np.linalg.norm(np.diff(path_points, axis=0), axis=1)
        total_length = float(np.sum(segment_lengths))

        if total_length < 1e-9:
            knot_times = np.linspace(start_time, end_time, len(path_points))
        else:
            relative_distances = np.concatenate(([0.0], np.cumsum(segment_lengths) / total_length))
            knot_times = start_time + relative_distances * travel_time

        velocities = self._make_spline_tangents(path_points, knot_times)
        spline = CubicHermiteSpline(knot_times, path_points, velocities)

        self._store_sampled_path(spline, start_time, end_time)

        return spline, end_time

    def _make_spline_tangents(
        self, path_points: NDArray[np.floating], knot_times: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        tangents = np.zeros_like(path_points)

        for i in range(len(path_points)):
            if i == 0:
                dt = knot_times[1] - knot_times[0]
                tangents[i] = (path_points[1] - path_points[0]) / max(dt, 1e-6)

            elif i == len(path_points) - 1:
                dt = knot_times[-1] - knot_times[-2]
                tangents[i] = (path_points[-1] - path_points[-2]) / max(dt, 1e-6)

            else:
                dt = knot_times[i + 1] - knot_times[i - 1]
                tangents[i] = (path_points[i + 1] - path_points[i - 1]) / max(dt, 1e-6)

        return 0.65 * tangents

    def _make_checkpoint_list(
        self, gate_idx: int, gate_pos: NDArray[np.floating], gate_angles: NDArray[np.floating]
    ) -> NDArray[np.floating]:

        # tmp_gate = None
        if gate_idx == 0:
            before_gate, after_gate = self._gate_direction_points(
                gate_pos[0], gate_angles[0], dist_before=0.25
            )
            checkpoints = [
                np.array([-1.5, 0.8, 0.1]),
                np.array([-1, 0.6, 0.45]),
                before_gate,
                gate_pos[0],
                # after_gate,
            ]

        elif gate_idx == 1:
            before_gate, after_gate = self._gate_direction_points(gate_pos[1], gate_angles[1])
            checkpoints = [
                gate_pos[0],
                np.array([1, -0.4, 1]),
                np.array([1.5, -0.1, 1]),
                before_gate,
                gate_pos[1],
                # after_gate,
            ]

        elif gate_idx == 2:
            before_gate, after_gate = self._gate_direction_points(gate_pos[2], gate_angles[2])
            # tmp_gate = before_gate
            checkpoints = [
                gate_pos[1],
                np.array([0.7, 0.88, 1]),
                np.array([0.2, 0.4, 1]),
                before_gate,
                gate_pos[2],
                # after_gate,
            ]

        elif gate_idx == 3:
            before_gate, after_gate = self._gate_direction_points(gate_pos[3], gate_angles[3])
            checkpoints = [
                gate_pos[2],
                # tmp_gate,
                np.array([-0.4, -0.25, 0.8]),
                np.array([-0.4, -0.4, 1.1]),
                # before_gate,
                gate_pos[3],
                # after_gate,
            ]

        else:
            raise ValueError(f"Invalid gate index: {gate_idx}")

        return np.asarray(checkpoints, dtype=np.float64)

    def _push_points_away_from_obstacles(
        self, path_points: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        pushed_points = path_points.copy()

        safety_radius = 0.28
        max_push = 0.22

        obstacle_positions = (
            self._latest_obstacle_positions
            if self._latest_obstacle_positions is not None
            else self.fixed_obstacle_pos
        )

        for idx in range(len(pushed_points)):
            total_push = np.zeros(2, dtype=np.float64)

            for obstacle_pos in obstacle_positions:
                xy_offset = pushed_points[idx, :2] - obstacle_pos[:2]
                xy_dist = np.linalg.norm(xy_offset)

                if xy_dist < 1e-9:
                    xy_offset = np.array([1.0, 0.0], dtype=np.float64)
                    xy_dist = 1e-9

                if xy_dist < safety_radius:
                    direction = xy_offset / xy_dist
                    push_strength = max_push * (1.0 - xy_dist / safety_radius)
                    total_push += push_strength * direction

            pushed_points[idx, :2] += total_push

        return pushed_points

    @staticmethod
    def _gate_direction_points(
        gate_pos: NDArray[np.floating],
        gate_angles: NDArray[np.floating],
        dist_before: float = 0.25,
        dist_after: float = 0.32,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        yaw = float(gate_angles[2])

        forward_xy = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64)

        forward_xy /= np.linalg.norm(forward_xy) + 1e-9

        before_gate = gate_pos.copy()
        after_gate = gate_pos.copy()

        before_gate[:2] -= dist_before * forward_xy
        after_gate[:2] += dist_after * forward_xy

        return before_gate, after_gate

    @staticmethod
    def _read_target_gate(obs: dict[str, NDArray[np.floating]]) -> int:
        gate_obs = np.asarray(obs["target_gate"])
        return int(gate_obs.reshape(-1)[0])

    def _get_path_spline(self) -> CubicHermiteSpline:
        if self._path_spline is None:
            raise RuntimeError("Path spline requested before initialization.")
        return self._path_spline

    def _store_gate_positions(self, gate_pos: NDArray[np.floating]) -> None:
        if self._initial_gate_positions is None:
            self._initial_gate_positions = gate_pos.copy()

        self._latest_gate_positions = gate_pos.copy()

    def _store_obstacle_positions(self, obstacle_pos: NDArray[np.floating]) -> None:
        self._latest_obstacle_positions = obstacle_pos.copy()

    def _store_path_points(self, path_points: NDArray[np.floating]) -> None:
        self._debug_path_points = path_points.copy()

    def _store_sampled_path(
        self, spline: CubicHermiteSpline, start_time: float, end_time: float
    ) -> None:
        sample_times = np.linspace(start_time, end_time, 80)
        self._debug_sampled_path = spline(sample_times)

    def render_callback(self, sim: object) -> None:
        """Render debug trajectory information."""
        if not self._debug_enabled:
            return

        # Initial gate positions: small blue points
        if self._initial_gate_positions is not None:
            draw_points(sim, self._initial_gate_positions, rgba=(0.0, 0.2, 1.0, 1.0), size=0.035)

        # Current/randomized gate positions: larger cyan points
        if self._latest_gate_positions is not None:
            draw_points(sim, self._latest_gate_positions, rgba=(0.0, 1.0, 1.0, 1.0), size=0.055)

        # Initial obstacle positions: orange points
        if self._initial_obstacle_positions is not None:
            draw_points(sim, self._initial_obstacle_positions, rgba=(1.0, 0.5, 0.0, 1.0), size=0.04)

        # Current obstacle positions: red points
        if self._latest_obstacle_positions is not None:
            draw_points(sim, self._latest_obstacle_positions, rgba=(1.0, 0.0, 0.0, 1.0), size=0.06)

        # Planned checkpoint points: magenta points
        if self._debug_path_points is not None:
            draw_points(sim, self._debug_path_points, rgba=(1.0, 0.0, 1.0, 1.0), size=0.035)

        # Planned spline trajectory: green line
        if self._debug_sampled_path is not None:
            draw_line(sim, self._debug_sampled_path, rgba=(0.0, 1.0, 0.0, 1.0))

        # Current setpoint on trajectory: yellow point
        if self._path_spline is not None:
            t_now = min(self._tick / self.freq, self._path_end_time)
            current_setpoint = self._path_spline(t_now).reshape(1, 3)
            draw_points(sim, current_setpoint, rgba=(1.0, 1.0, 0.0, 1.0), size=0.05)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Update controller state after each simulation step."""
        del action, obs, reward, info
        self._tick += 1
        if terminated or truncated:
            self._finished = True
        return self._finished

    def episode_callback(self) -> None:
        """Reset controller state at the start of an episode."""
        self._tick = 0
        self._finished = False
        self._current_target_idx = -1
        self._path_spline = None
        self._path_end_time = self.max_episode_time
        self._cached_gate_positions = None
        self._cached_obstacle_positions = None
        self._previous_command = np.array(
            [0.0, 0.0, 0.0, self.drone_mass * self.gravity_const], dtype=np.float32
        )
