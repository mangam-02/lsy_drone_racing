"""2D top-down (bird's-eye) view of the race track and planned trajectory."""

import os

os.environ["SCIPY_ARRAY_API"] = "1"

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.trajectory_planner import TrajectoryPlanner
from lsy_drone_racing.utils import load_config


def make_obs_from_config(config) -> dict:
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


def draw_gate_topdown(ax, center, quat, label):
    """Draw gate as a thin rectangle from above: wide along y-axis, thin along approach x-axis."""
    rot = R.from_quat(quat)
    x_axis = rot.apply([1.0, 0.0, 0.0])[:2]  # approach direction (thin dimension)
    y_axis = rot.apply([0.0, 1.0, 0.0])[:2]  # gate width direction (wide dimension)

    ho = 0.36  # outer half-width along y (0.72m / 2)
    hi = 0.20  # inner half-width along y (0.40m / 2) — the opening
    hd = 0.05  # half-depth along x (DEPTH 0.10m / 2)
    cx, cy = center[0], center[1]

    def rect(hy, hx):
        return np.array(
            [
                [cx - hy * y_axis[0] - hx * x_axis[0], cy - hy * y_axis[1] - hx * x_axis[1]],
                [cx + hy * y_axis[0] - hx * x_axis[0], cy + hy * y_axis[1] - hx * x_axis[1]],
                [cx + hy * y_axis[0] + hx * x_axis[0], cy + hy * y_axis[1] + hx * x_axis[1]],
                [cx - hy * y_axis[0] + hx * x_axis[0], cy - hy * y_axis[1] + hx * x_axis[1]],
            ]
        )

    # Outer frame (thin red rectangle)
    outer = rect(ho, hd)
    ax.fill(outer[:, 0], outer[:, 1], color="red", alpha=0.4, zorder=2)
    outer_closed = np.vstack([outer, outer[0]])
    ax.plot(outer_closed[:, 0], outer_closed[:, 1], "r-", lw=2, zorder=3)

    # Opening (green strip inside)
    inner = rect(hi, hd)
    ax.fill(inner[:, 0], inner[:, 1], color="green", alpha=0.5, zorder=3)

    # Approach arrow
    ax.annotate(
        "",
        xy=(cx + x_axis[0] * 0.5, cy + x_axis[1] * 0.5),
        xytext=(cx - x_axis[0] * 0.5, cy - x_axis[1] * 0.5),
        arrowprops=dict(arrowstyle="->", color="darkred", lw=1.5),
        zorder=4,
    )

    # Label
    ax.text(cx + 0.08, cy + 0.08, label, color="red", fontsize=10, fontweight="bold", zorder=5)


def main():
    config = load_config(Path(__file__).parents[1] / "config" / "level0.toml")
    obs = make_obs_from_config(config)

    planner = TrajectoryPlanner(obs, config, N=25)
    traj = planner.pos  # (T, 3)

    # Extract raw waypoints for overlay
    wp_pos = [pos for pos, _ in planner._raw_waypoints]
    wp_vel = [vel for _, vel in planner._raw_waypoints]

    fig, ax = plt.subplots(figsize=(12, 7))

    # Arena boundary
    boundary = plt.Polygon(
        [[-2.5, -1.5], [2.5, -1.5], [2.5, 1.5], [-2.5, 1.5]],
        fill=False,
        edgecolor="gray",
        lw=1.5,
        linestyle="--",
    )
    ax.add_patch(boundary)

    # Trajectory
    ax.plot(traj[:, 0], traj[:, 1], "b-", lw=2, label="Trajectory (top view)", zorder=1)

    # Waypoints: line connecting them, dots, numbered labels
    wx = [p[0] for p in wp_pos]
    wy = [p[1] for p in wp_pos]
    ax.plot(wx, wy, "--", color="purple", lw=1.2, alpha=0.7, zorder=4, label="Waypoints")
    for i, (p, v) in enumerate(zip(wp_pos, wp_vel)):
        color = "magenta" if v is not None else "purple"
        ax.scatter(p[0], p[1], color=color, s=60, zorder=7)
        ax.text(
            p[0] + 0.04, p[1] + 0.04, str(i), color=color, fontsize=8, fontweight="bold", zorder=8
        )

    # Mark start
    ax.scatter(*obs["pos"][:2], color="green", s=120, zorder=6, label="Start")

    # Obstacles
    for i, opos in enumerate(obs["obstacles_pos"]):
        ax.add_patch(plt.Circle(opos[:2], radius=0.015, color="brown", zorder=4))
        ax.add_patch(plt.Circle(opos[:2], radius=0.085, color="orange", alpha=0.3, zorder=3))
        ax.text(opos[0] + 0.05, opos[1] + 0.05, f"P{i}", color="brown", fontsize=8)

    # Gates
    for i, (gpos, gquat) in enumerate(zip(obs["gates_pos"], obs["gates_quat"])):
        draw_gate_topdown(ax, gpos, gquat, f"G{i}")

    ax.set_aspect("equal")
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Left-click to sketch  |  Backspace = undo  |  Enter = print  |  'c' = clear")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # --- Interactive sketch ---
    sketch_pts = []  # list of (x, y)
    (sketch_dots,) = ax.plot([], [], "o", color="orange", ms=8, zorder=7)
    (sketch_line,) = ax.plot([], [], "-", color="orange", lw=2, alpha=0.8, zorder=6)
    ax.text(-2.5, -1.45, "orange = your sketch", color="orange", fontsize=8)

    def redraw():
        if sketch_pts:
            xs, ys = zip(*sketch_pts)
            sketch_dots.set_data(xs, ys)
            sketch_line.set_data(xs, ys)
            for i, (x, y) in enumerate(sketch_pts):
                ax.text(x + 0.04, y + 0.04, str(i), color="darkorange", fontsize=8, zorder=8)
        else:
            sketch_dots.set_data([], [])
            sketch_line.set_data([], [])
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax or event.button != 1:
            return
        sketch_pts.append((event.xdata, event.ydata))
        redraw()

    def on_key(event):
        if event.key == "backspace" and sketch_pts:
            sketch_pts.pop()
            # remove last label — redraw clears and rewrites, just refresh
            for txt in ax.texts[-1:]:
                txt.remove()
            redraw()
        elif event.key == "enter":
            print("\n=== Sketch waypoints (x, y) ===")
            for i, (x, y) in enumerate(sketch_pts):
                print(f"  {i}: ({x:.3f}, {y:.3f})")
        elif event.key == "c":
            sketch_pts.clear()
            for txt in ax.texts[:]:
                pass  # leave gate labels, only user dots are redrawn
            redraw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


if __name__ == "__main__":
    main()
