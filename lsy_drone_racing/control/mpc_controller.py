"""MPC controller for drone racing using minimum-snap trajectory planning.

Uses a minimum-snap trajectory planner (TrajectoryPlanner) to generate smooth
gate-to-gate references, and an acados NMPC to track them in attitude mode.
Replans online when gate positions are updated by the sensor model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from crazyflow.sim.visualize import draw_line, draw_points
from drone_models.core import load_params
from drone_models.so_rpy import symbolic_dynamics_euler
from drone_models.utils.rotation import ang_vel2rpy_rates
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.trajectory_planner import TrajectoryPlanner

if TYPE_CHECKING:
    from numpy.typing import NDArray


def create_acados_model(parameters: dict) -> AcadosModel:
    """Creates an acados model from the so_rpy symbolic drone model."""
    X_dot, X, U, _ = symbolic_dynamics_euler(
        mass=parameters["mass"],
        gravity_vec=parameters["gravity_vec"],
        J=parameters["J"],
        J_inv=parameters["J_inv"],
        acc_coef=parameters["acc_coef"],
        cmd_f_coef=parameters["cmd_f_coef"],
        rpy_coef=parameters["rpy_coef"],
        rpy_rates_coef=parameters["rpy_rates_coef"],
        cmd_rpy_coef=parameters["cmd_rpy_coef"],
    )
    model = AcadosModel()
    model.name = "mpc_controller"
    model.f_expl_expr = X_dot
    model.f_impl_expr = None
    model.x = X
    model.u = U
    return model


def create_ocp_solver(
    Tf: float, N: int, parameters: dict, verbose: bool = False
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Creates an acados OCP and solver for attitude-mode drone control."""
    ocp = AcadosOcp()
    ocp.model = create_acados_model(parameters)

    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    ny = nx + nu
    ny_e = nx

    ocp.solver_options.N_horizon = N

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    Q = np.diag([50.0, 50.0, 400.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0])
    R_mat = np.diag([1.0, 1.0, 1.0, 50.0])
    Q_e = Q.copy()
    ocp.cost.W = scipy.linalg.block_diag(Q, R_mat)
    ocp.cost.W_e = Q_e

    Vx = np.zeros((ny, nx))
    Vx[0:nx, 0:nx] = np.eye(nx)
    ocp.cost.Vx = Vx

    Vu = np.zeros((ny, nu))
    Vu[nx : nx + nu, :] = np.eye(nu)
    ocp.cost.Vu = Vu

    Vx_e = np.zeros((ny_e, nx))
    Vx_e[0:nx, 0:nx] = np.eye(nx)
    ocp.cost.Vx_e = Vx_e

    ocp.cost.yref = np.zeros((ny,))
    ocp.cost.yref_e = np.zeros((ny_e,))

    ocp.constraints.lbx = np.array([-0.5, -0.5, -0.5])
    ocp.constraints.ubx = np.array([0.5, 0.5, 0.5])
    ocp.constraints.idxbx = np.array([3, 4, 5])

    ocp.constraints.lbu = np.array([-0.5, -0.5, -0.5, parameters["thrust_min"] * 4])
    ocp.constraints.ubu = np.array([0.5, 0.5, 0.5, parameters["thrust_max"] * 4])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    ocp.constraints.x0 = np.zeros((nx,))

    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.tol = 1e-6
    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1
    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 50
    ocp.solver_options.tf = Tf

    acados_ocp_solver = AcadosOcpSolver(
        ocp,
        json_file="c_generated_code/mpc_controller.json",
        verbose=verbose,
        build=True,
        generate=True,
    )
    return acados_ocp_solver, ocp


class MPCController(Controller):
    """Attitude-mode MPC with minimum-snap gate-to-gate trajectory planning."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize attitude-mode MPC racing controller and pre-plan the trajectory."""
        super().__init__(obs, info, config)
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

        self._planner = TrajectoryPlanner(obs, config, N=self._N)
        self._tick = 0
        self._warm_started = False

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute attitude command [roll, pitch, yaw, thrust] to track the planned trajectory."""
        pos_ref, vel_ref = self._planner.get_reference(self._tick)

        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((obs["pos"], obs["rpy"], obs["vel"], obs["drpy"]))

        if not self._warm_started:
            hover_thrust = self.drone_params["mass"] * -self.drone_params["gravity_vec"][-1]
            for j in range(self._N + 1):
                x_init = np.zeros(self._nx)
                x_init[:3] = pos_ref[j]
                x_init[6:9] = vel_ref[j]
                self._acados_ocp_solver.set(j, "x", x_init)
            for j in range(self._N):
                self._acados_ocp_solver.set(j, "u", np.array([0.0, 0.0, 0.0, hover_thrust]))
            self._warm_started = True

        self._acados_ocp_solver.set(0, "lbx", x0)
        self._acados_ocp_solver.set(0, "ubx", x0)

        hover_thrust = self.drone_params["mass"] * -self.drone_params["gravity_vec"][-1]
        yref = np.zeros((self._N, self._ny))
        yref[:, 0:3] = pos_ref[: self._N]
        yref[:, 6:9] = vel_ref[: self._N]
        yref[:, 15] = hover_thrust
        for j in range(self._N):
            self._acados_ocp_solver.set(j, "yref", yref[j])

        yref_e = np.zeros((self._ny_e,))
        yref_e[0:3] = pos_ref[self._N]
        yref_e[6:9] = vel_ref[self._N]
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
        """Advance tick; replan if new gate/obstacle positions are revealed."""
        self._tick += 1
        if self._planner.update(obs):
            self._tick = 0
            self._warm_started = False
        return self._tick >= self._planner.tick_max

    def render_callback(self, sim: object) -> None:
        """Draw the planned trajectory as a green line and current target as a red dot."""
        draw_line(sim, self._planner.pos, rgba=(0.0, 1.0, 0.0, 1.0))
        pos_ref, _ = self._planner.get_reference(self._tick)
        draw_points(sim, pos_ref[:1], rgba=(1.0, 0.0, 0.0, 1.0), size=0.05)

    def episode_callback(self) -> None:
        """Reset tick counter at episode start."""
        self._tick = 0
