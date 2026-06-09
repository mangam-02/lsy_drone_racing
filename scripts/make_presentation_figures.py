"""Generate presentation figures (PNG + PDF) for the slides.

Produces four figures into ``figures/``:

1. ``planning_topdown``  — bird's-eye view of the level-3 track + the
   ``BSplinePlanner`` trajectory, waypoints, gates and obstacles (slide 1).
2. ``speed_profile``     — the trapezoidal speed profile along arc length
   (slide 1).
3. ``mpc_blockdiagram``  — Planner -> MPC -> Drone control loop (slide 2).
4. ``rl_blockdiagram``   — the future RL-tuning layer on top (slide 3).

The track is taken from a *seeded* env reset so the figure is reproducible.

Run:  ./venv_drone/bin/python scripts/make_presentation_figures.py
"""

from __future__ import annotations

import os

os.environ["SCIPY_ARRAY_API"] = "1"

from pathlib import Path

import gymnasium
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.mpc_planner_controller import BSplinePlanner, _GateFrame
from lsy_drone_racing.utils import load_config

SEED = 7
OUT = Path(__file__).parents[1] / "figures"


def save(fig, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    print(f"  wrote figures/{name}.png  +  .pdf")


# ── Track from a seeded reset ────────────────────────────────────────────────


def get_track_obs(config) -> dict:
    config.sim.render = False
    env = gymnasium.make(
        config.env.id, freq=config.env.freq, sim_config=config.sim,
        sensor_range=config.env.sensor_range, control_mode=config.env.control_mode,
        track=config.env.track, disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"), seed=SEED,
    )
    env = JaxToNumpy(env)
    obs, _ = env.reset(seed=SEED)
    env.close()
    return {k: np.asarray(v) for k, v in obs.items()}


# ── Figure 1: top-down trajectory ────────────────────────────────────────────


def draw_gate_topdown(ax, center, quat, label):
    rot = R.from_quat(quat)
    x_axis = rot.apply([1.0, 0.0, 0.0])[:2]
    y_axis = rot.apply([0.0, 1.0, 0.0])[:2]
    ho, hi, hd = _GateFrame.OUTER / 2, _GateFrame.OPENING / 2, _GateFrame.DEPTH / 2
    cx, cy = center[0], center[1]

    def rect(hy, hx):
        return np.array([
            [cx - hy * y_axis[0] - hx * x_axis[0], cy - hy * y_axis[1] - hx * x_axis[1]],
            [cx + hy * y_axis[0] - hx * x_axis[0], cy + hy * y_axis[1] - hx * x_axis[1]],
            [cx + hy * y_axis[0] + hx * x_axis[0], cy + hy * y_axis[1] + hx * x_axis[1]],
            [cx - hy * y_axis[0] + hx * x_axis[0], cy - hy * y_axis[1] + hx * x_axis[1]],
        ])

    outer = rect(ho, hd)
    ax.fill(outer[:, 0], outer[:, 1], color="#c0392b", alpha=0.85, zorder=3)
    inner = rect(hi, hd)
    ax.fill(inner[:, 0], inner[:, 1], color="white", zorder=4)
    ax.text(cx + 0.10, cy + 0.10, label, color="#c0392b", fontsize=11,
            fontweight="bold", zorder=6)


def fig_topdown(planner: BSplinePlanner, obs: dict):
    traj = planner.pos
    raw = planner._raw_waypoints

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.add_patch(plt.Polygon([[-2.5, -1.5], [2.5, -1.5], [2.5, 1.5], [-2.5, 1.5]],
                             fill=False, edgecolor="gray", lw=1.5, ls="--"))

    # Planned trajectory.
    ax.plot(traj[:, 0], traj[:, 1], "-", color="#1f77b4", lw=2.5,
            label="Planned B-spline trajectory", zorder=2)

    # Waypoints: fixed gate waypoints (magenta) vs free intermediates (purple).
    for p, v in raw:
        c = "magenta" if v is not None else "#7f5fbf"
        ax.scatter(p[0], p[1], color=c, s=45, zorder=5,
                   edgecolors="white", linewidths=0.5)
    ax.scatter([], [], color="magenta", s=45, label="Fixed gate waypoints")
    ax.scatter([], [], color="#7f5fbf", s=45, label="Optimized free waypoints")

    # Start.
    ax.scatter(*obs["pos"][:2], color="#2ca02c", s=160, marker="*",
               zorder=6, label="Start", edgecolors="black", linewidths=0.6)

    # Obstacles: physical pole + clearance ring.
    for opos in obs["obstacles_pos"]:
        ax.add_patch(plt.Circle(opos[:2], BSplinePlanner.PLAN_CLEARANCE,
                                color="orange", alpha=0.25, zorder=1))
        ax.add_patch(plt.Circle(opos[:2], 0.04, color="#8B4513", zorder=5))
    ax.scatter([], [], marker="o", s=140, color="orange", alpha=0.25,
               label="Obstacle clearance")

    for i, (gpos, gquat) in enumerate(zip(obs["gates_pos"], obs["gates_quat"])):
        draw_gate_topdown(ax, gpos, gquat, f"G{i+1}")

    ax.set_aspect("equal")
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    # Inside the dashed arena box, bottom-right corner.
    ax.text(2.4, -1.4, f"level 3, seed {SEED}",
            ha="right", va="bottom", fontsize=9, color="#555",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbb", alpha=0.8))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, "planning_topdown")


# ── Figure 2: speed profile ──────────────────────────────────────────────────


def fig_speed(planner: BSplinePlanner):
    pos, vel = planner.pos, planner.vel
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))])
    speed = np.linalg.norm(vel, axis=1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(s, speed, "-", color="#1f77b4", lw=2.5, label="Reference speed")
    ax.axhline(BSplinePlanner.TARGET_SPEED, ls="--", color="#2ca02c", lw=1.5,
               label=f"Cruise speed ({BSplinePlanner.TARGET_SPEED} m/s)")

    # Shade ramp regions.
    a = BSplinePlanner.ACCEL_DIST
    ax.axvspan(0, a, color="orange", alpha=0.15)
    ax.axvspan(s[-1] - BSplinePlanner.DECEL_DIST, s[-1], color="orange", alpha=0.15)
    ax.text(a / 2, 0.2, "accel", ha="center", color="darkorange", fontsize=9)
    ax.text(s[-1] - BSplinePlanner.DECEL_DIST / 2, 0.2, "decel", ha="center",
            color="darkorange", fontsize=9)

    ax.set_xlabel("Arc length along trajectory [m]")
    ax.set_ylabel("Speed [m/s]")
    ax.set_ylim(0, BSplinePlanner.TARGET_SPEED + 0.4)
    ax.set_xlim(0, s[-1])
    ax.legend(loc="lower center", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, "speed_profile")


# ── Block-diagram helpers ────────────────────────────────────────────────────


def box(ax, xy, w, h, text, fc, ec="black", fs=11, fontweight="bold"):
    cx, cy = xy
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        fc=fc, ec=ec, lw=1.8, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            fontweight=fontweight, zorder=3, wrap=True)


def arrow(ax, p1, p2, text="", color="black", rad=0.0, fs=9, ls="-"):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=18, lw=1.8,
        color=color, connectionstyle=f"arc3,rad={rad}", zorder=1, linestyle=ls))
    if text:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + 0.06 + abs(rad) * 0.8, text, ha="center", va="bottom",
                fontsize=fs, color=color, style="italic")


# ── Figure 3: MPC block diagram ──────────────────────────────────────────────


def fig_mpc_diagram():
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")

    box(ax, (2.0, 2.6), 2.6, 1.2, "Planner\n(B-spline path)", "#aed6f1")
    box(ax, (5.7, 2.6), 2.6, 1.2, "MPC\n(acados, N=25)", "#f9e79f")
    box(ax, (9.6, 2.6), 2.6, 1.2, "Drone", "#a9dfbf")

    arrow(ax, (3.3, 2.6), (4.4, 2.6), "reference\nhorizon")
    arrow(ax, (7.0, 2.6), (8.3, 2.6), "1st command\n[r, p, y, T]")
    # Feedback loop: drone state back into the MPC.
    arrow(ax, (9.6, 2.0), (5.4, 2.0), "", color="#888", rad=-0.3)
    ax.text(7.5, 0.7, "state feedback (pos, vel, attitude)", ha="center",
            fontsize=9, color="#555", style="italic")
    fig.tight_layout()
    save(fig, "mpc_blockdiagram")


# ── Figure 4: RL-tuning block diagram ────────────────────────────────────────


def fig_rl_diagram():
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    box(ax, (2.0, 2.2), 2.6, 1.2, "Planner\n(B-spline path)", "#aed6f1")
    box(ax, (5.7, 2.2), 2.6, 1.2, "MPC\n(acados, N=25)", "#f9e79f")
    box(ax, (9.6, 2.2), 2.6, 1.2, "Drone", "#a9dfbf")
    # RL layer on top.
    box(ax, (5.7, 4.2), 3.6, 1.0, "RL policy", "#f5b7b1")

    arrow(ax, (3.3, 2.2), (4.4, 2.2), "reference")
    arrow(ax, (7.0, 2.2), (8.3, 2.2), "1st command")
    arrow(ax, (9.6, 1.6), (5.4, 1.6), "", color="#888", rad=-0.3)
    ax.text(7.5, 0.55, "state feedback", ha="center", fontsize=9, color="#555", style="italic")

    # RL wiring: state in (from drone), weight multipliers out (into MPC).
    arrow(ax, (9.2, 2.8), (7.5, 3.9), "", color="#c0392b", rad=-0.25)
    ax.text(8.9, 3.35, "state\n(error, speed,\ngate/obstacle)", ha="left", va="center",
            fontsize=8.5, color="#c0392b", style="italic")
    arrow(ax, (4.9, 3.7), (5.5, 2.85), "", color="#c0392b", rad=0.1)
    ax.text(4.05, 3.25, "weight\nmultipliers", ha="center", va="center",
            fontsize=8.5, color="#c0392b", style="italic")
    fig.tight_layout()
    save(fig, "rl_blockdiagram")


def main():
    config = load_config(Path(__file__).parents[1] / "config" / "level3.toml")
    print(f"Building level-3 track (seed {SEED}) ...")
    obs = get_track_obs(config)
    obs["target_gate"] = np.int32(0)
    planner = BSplinePlanner(obs, config, N=25)

    print("Rendering figures ...")
    fig_topdown(planner, obs)
    fig_speed(planner)
    fig_mpc_diagram()
    fig_rl_diagram()
    print(f"\nDone. All figures in: {OUT}")


if __name__ == "__main__":
    main()
