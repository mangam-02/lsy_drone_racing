"""Hover controller for visualizing the minimum-snap planned trajectory in sim.

Uses MPC hovering (same as hover_mpc.py) while the trajectory planner runs in the
background. Use this to visually verify the planned path through the gates before
enabling full MPC tracking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from crazyflow.sim.visualize import draw_line, draw_points
from drone_models.core import load_params
from drone_models.utils.rotation import ang_vel2rpy_rates
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.hover_mpc import create_ocp_solver
from lsy_drone_racing.control.trajectory_planner import TrajectoryPlanner

if TYPE_CHECKING:
    from numpy.typing import NDArray


class MPCControllerHover(Controller):
    """MPC hover while showing the planned trajectory via render_callback."""

    HOVER_HEIGHT = 0.5  # m
    HOVER_DURATION = 30  # s — long enough to inspect the trajectory

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize hover MPC and pre-plan the full trajectory for visualization."""
        super().__init__(obs, info, config)
        self._N = 25
        self._dt = 1 / config.env.freq
        self._T_HORIZON = self._N * self._dt

        hover_pos = np.array([obs["pos"][0], obs["pos"][1], self.HOVER_HEIGHT])
        n_steps = int(config.env.freq * self.HOVER_DURATION)
        self._waypoints_pos = np.tile(hover_pos, (n_steps, 1))
        self._waypoints_vel = np.zeros((n_steps, 3))
        self._waypoints_yaw = np.zeros(n_steps)
        self._tick_max = n_steps - 1 - self._N

        self.drone_params = load_params("so_rpy", config.sim.drone_model)
        self._acados_ocp_solver, self._ocp = create_ocp_solver(
            self._T_HORIZON, self._N, self.drone_params
        )
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()
        self._ny = self._nx + self._nu
        self._ny_e = self._nx

        self._planner = TrajectoryPlanner(obs, config, N=self._N)
        self._tick = 0

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """MPC hover at fixed point above start position."""
        i = min(self._tick, self._tick_max)

        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((obs["pos"], obs["rpy"], obs["vel"], obs["drpy"]))
        self._acados_ocp_solver.set(0, "lbx", x0)
        self._acados_ocp_solver.set(0, "ubx", x0)

        yref = np.zeros((self._N, self._ny))
        yref[:, 0:3] = self._waypoints_pos[i : i + self._N]
        yref[:, 5] = self._waypoints_yaw[i : i + self._N]
        yref[:, 6:9] = self._waypoints_vel[i : i + self._N]
        yref[:, 15] = self.drone_params["mass"] * -self.drone_params["gravity_vec"][-1]
        for j in range(self._N):
            self._acados_ocp_solver.set(j, "yref", yref[j])

        yref_e = np.zeros((self._ny_e,))
        yref_e[0:3] = self._waypoints_pos[i + self._N]
        yref_e[5] = self._waypoints_yaw[i + self._N]
        yref_e[6:9] = self._waypoints_vel[i + self._N]
        self._acados_ocp_solver.set(self._N, "y_ref", yref_e)

        self._acados_ocp_solver.solve()
        return self._acados_ocp_solver.get(0, "u")

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Advance tick; replan if new gate/obstacle positions revealed."""
        self._tick += 1
        if self._planner.update(obs):
            self._tick = 0
        return False

    def render_callback(self, sim: object) -> None:
        """Draw planned trajectory (green) and current target point (red)."""
        draw_line(sim, self._planner.pos, rgba=(0.0, 1.0, 0.0, 1.0))
        pos_ref, _ = self._planner.get_reference(self._tick)
        draw_points(sim, pos_ref[:1], rgba=(1.0, 0.0, 0.0, 1.0), size=0.05)

    def episode_callback(self) -> None:
        """Reset tick counter at episode start."""
        self._tick = 0
