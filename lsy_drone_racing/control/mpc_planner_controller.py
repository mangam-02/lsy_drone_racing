"""Path-planning + MPC controller for drone racing.

This controller combines two stages:

1. **Path planning** — the :class:`TrajectoryPlanner` builds a minimum-snap
   reference trajectory through all remaining gates while avoiding the obstacles.
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

from typing import TYPE_CHECKING

import numpy as np
from drone_models.core import load_params
from drone_models.utils.rotation import ang_vel2rpy_rates
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.attitude_mpc import create_ocp_solver
from lsy_drone_racing.control.trajectory_planner import TrajectoryPlanner

if TYPE_CHECKING:
    from numpy.typing import NDArray


class MPCPlanner(Controller):
    """Replanning path-planner combined with a tracking attitude-MPC."""

    #: Move (m) / rotation (quat component) beyond which we consider an object to
    #: have "changed" and trigger a replan.
    REPLAN_TOL = 1e-3

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
        self._acados_ocp_solver, self._ocp = create_ocp_solver(
            self._T_HORIZON, self._N, self.drone_params
        )
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()
        self._ny = self._nx + self._nu
        self._ny_e = self._nx
        self._hover_thrust = self.drone_params["mass"] * -self.drone_params["gravity_vec"][-1]

        # ── Path planner ───────────────────────────────────────────────────────
        self.planner = TrajectoryPlanner(obs, config, N=self._N)
        self._snapshot_objects(obs)  # remember poses we just planned with

        self._tick = 0
        self._finished = False

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

    def _maybe_replan(self, obs: dict):
        """Rebuild the trajectory from the current state if the track changed."""
        if self._objects_changed(obs):
            self.planner.build(obs)  # plans from obs["pos"] -> restart tracking at 0
            self._tick = 0
            self._snapshot_objects(obs)

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
        self._maybe_replan(obs)

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

        # ── Solve ──────────────────────────────────────────────────────────────
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
        """Advance the trajectory index and check for completion."""
        self._tick += 1
        if int(obs["target_gate"]) == -1:  # all gates passed
            self._finished = True
        return self._finished

    def episode_callback(self):
        """Reset the trajectory index after an episode."""
        self._tick = 0

    def episode_reset(self):
        """Reset internal state for a new episode."""
        self._tick = 0
        self._finished = False
