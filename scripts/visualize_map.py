"""Visualize the 3D obstacle map: poles, gate frames, and planned trajectory."""

import os

os.environ["SCIPY_ARRAY_API"] = "1"

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.trajectory_planner import TrajectoryPlanner
from lsy_drone_racing.utils import load_config


def draw_cylinder(ax, center_xy, radius, height, color, alpha=0.3, n=20):
    """Draw a vertical cylinder."""
    theta = np.linspace(0, 2 * np.pi, n)
    z = np.array([0.0, height])
    theta_grid, z_grid = np.meshgrid(theta, z)
    x = center_xy[0] + radius * np.cos(theta_grid)
    y = center_xy[1] + radius * np.sin(theta_grid)
    ax.plot_surface(x, y, z_grid, color=color, alpha=alpha)


def draw_gate_frame(ax, center, quat, outer=0.72, opening=0.4, color="red", alpha=0.4):
    """Draw a gate frame as 4 rectangles."""
    rot = R.from_quat(quat)
    # Gate local axes: x=approach, y=horizontal, z=vertical
    y_ax = rot.apply([0.0, 1.0, 0.0])
    z_ax = rot.apply([0.0, 0.0, 1.0])

    ho = outer / 2
    hi = opening / 2
    # 4 frame segments: top, bottom, left, right
    # Each is a flat quad in the gate plane
    segments = [
        # (y_center, z_center, y_half, z_half)
        (0, ho - (ho - hi) / 2, ho, (ho - hi) / 2),  # top
        (0, -(ho - (ho - hi) / 2), ho, (ho - hi) / 2),  # bottom
        (-(ho - (ho - hi) / 2), 0, (ho - hi) / 2, hi),  # left
        ((ho - (ho - hi) / 2), 0, (ho - hi) / 2, hi),  # right
    ]

    for yc, zc, yh, zh in segments:
        corners = np.array(
            [
                center + (yc - yh) * y_ax + (zc - zh) * z_ax,
                center + (yc + yh) * y_ax + (zc - zh) * z_ax,
                center + (yc + yh) * y_ax + (zc + zh) * z_ax,
                center + (yc - yh) * y_ax + (zc + zh) * z_ax,
            ]
        )
        poly = Poly3DCollection([corners], alpha=alpha, color=color)
        ax.add_collection3d(poly)

    # Draw gate opening border
    corners_outer = np.array(
        [
            center + (-ho) * y_ax + (-ho) * z_ax,
            center + (ho) * y_ax + (-ho) * z_ax,
            center + (ho) * y_ax + (ho) * z_ax,
            center + (-ho) * y_ax + (ho) * z_ax,
            center + (-ho) * y_ax + (-ho) * z_ax,
        ]
    )
    ax.plot(corners_outer[:, 0], corners_outer[:, 1], corners_outer[:, 2], color=color, linewidth=2)

    corners_inner = np.array(
        [
            center + (-hi) * y_ax + (-hi) * z_ax,
            center + (hi) * y_ax + (-hi) * z_ax,
            center + (hi) * y_ax + (hi) * z_ax,
            center + (-hi) * y_ax + (hi) * z_ax,
            center + (-hi) * y_ax + (-hi) * z_ax,
        ]
    )
    ax.plot(
        corners_inner[:, 0], corners_inner[:, 1], corners_inner[:, 2], color="green", linewidth=2
    )


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
    obstacles_pos = obs["obstacles_pos"]

    planner = TrajectoryPlanner(obs, config, N=25)

    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Planned trajectory
    traj = planner.pos
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], "b-", linewidth=2, label="Planned trajectory")

    # Start position
    ax.scatter(*obs["pos"], color="green", s=150, zorder=5, label="Start")

    # Gate frames
    for i, (gpos, gquat) in enumerate(zip(gates_pos, gates_quat)):
        draw_gate_frame(ax, gpos, gquat, color="red", alpha=0.35)
        ax.text(
            gpos[0],
            gpos[1],
            gpos[2] + 0.25,
            f"Gate {i}",
            color="red",
            fontsize=9,
            fontweight="bold",
        )

    # Pole cylinders (physical size + inflated safety radius)
    for i, opos in enumerate(obstacles_pos):
        draw_cylinder(
            ax, opos[:2], radius=0.015, height=1.52, color="brown", alpha=0.8
        )  # actual pole
        draw_cylinder(
            ax, opos[:2], radius=0.015 + 0.07, height=1.52, color="orange", alpha=0.15
        )  # safety margin
        ax.text(opos[0], opos[1], 1.6, f"Pole {i}", color="brown", fontsize=8)

    # Ground plane reference
    xx, yy = np.meshgrid([-2.5, 2.5], [-1.5, 1.5])
    ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.05, color="gray")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("3D obstacle map — gates (red), poles (brown), safety margin (orange)")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(0, 2.0)
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
