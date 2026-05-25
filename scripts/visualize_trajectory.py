"""Visualize the minimum-snap planned trajectory through the race gates."""

import os

os.environ["SCIPY_ARRAY_API"] = "1"

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.trajectory_planner import TrajectoryPlanner
from lsy_drone_racing.utils import load_config


def make_obs_from_config(config) -> dict:
    """Build a full obs dict from the loaded config (gates, obstacles, drone start)."""
    gates = config.env.track.gates
    obstacles = config.env.track.obstacles
    drone = config.env.track.drones[0]

    gates_pos = np.array([g["pos"] for g in gates])
    gates_quat = np.array([R.from_euler("xyz", g["rpy"]).as_quat() for g in gates])
    obstacles_pos = np.array([o["pos"] for o in obstacles])

    return {
        "pos": np.array(drone["pos"]),
        "vel": np.zeros(3),
        "quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "ang_vel": np.zeros(3),
        "target_gate": np.int32(0),
        "gates_visited": np.zeros(len(gates), dtype=bool),
        "gates_pos": gates_pos,
        "gates_quat": gates_quat,
        "obstacles_pos": obstacles_pos,
        "obstacles_visited": np.zeros(len(obstacles), dtype=bool),
    }


def main():
    config = load_config(Path(__file__).parents[1] / "config" / "level0.toml")
    obs = make_obs_from_config(config)
    gates_pos = obs["gates_pos"]
    gates_quat = obs["gates_quat"]

    planner = TrajectoryPlanner(obs, config, N=25)
    traj = planner.pos  # (T, 3)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Trajectory
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], "b-", linewidth=2, label="Planned trajectory")

    # Start position
    ax.scatter(*obs["pos"], color="green", s=100, zorder=5, label="Start")

    # Gates — draw a square opening for each
    GATE_SIZE = 0.2  # half-width for visualization
    for i, (gpos, gquat) in enumerate(zip(gates_pos, gates_quat)):
        rot = R.from_quat(gquat)
        up = rot.apply([0, 0, 1])
        side = rot.apply([0, 1, 0])
        corners = np.array(
            [
                gpos + GATE_SIZE * (up + side),
                gpos + GATE_SIZE * (up - side),
                gpos + GATE_SIZE * (-up - side),
                gpos + GATE_SIZE * (-up + side),
                gpos + GATE_SIZE * (up + side),
            ]
        )
        ax.plot(corners[:, 0], corners[:, 1], corners[:, 2], "r-", linewidth=2)
        ax.text(gpos[0], gpos[1], gpos[2] + 0.15, f"G{i}", color="red", fontsize=9)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Minimum-snap trajectory through gates (level 0)")
    ax.legend()
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(0, 2.0)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
