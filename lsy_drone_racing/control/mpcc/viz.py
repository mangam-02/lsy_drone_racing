"""Rendering and trace-dumping for the MPCC controller.

Read-only presentation layer (mirrors the MPC controller's overlay): the race scene (planned
path, gates, obstacles, progress markers), the live cost-weight text box in the MuJoCo
viewer, and the per-tick speed-target trace dump. Nothing here mutates controller state —
:class:`~lsy_drone_racing.control.mpcc.controller.MPCCController` only delegates to these
helpers from its ``render_callback`` / ``episode_callback``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from crazyflow.sim.visualize import draw_line, draw_points
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.mpcc import weight_policy as wp
from lsy_drone_racing.control.mpcc.config import BASELINE_WEIGHTS
from lsy_drone_racing.control.mpcc.planner import _GateFrame

if TYPE_CHECKING:
    from lsy_drone_racing.control.mpcc.planner import SimplePlanner


def draw_square(
    sim: object, center: np.ndarray, quat: np.ndarray, half: float, rgba: np.ndarray
) -> None:
    """Draw an oriented square outline in the gate plane (local y-z plane)."""
    rot = R.from_quat(quat)
    local = np.array(
        [
            [0.0, half, half],
            [0.0, -half, half],
            [0.0, -half, -half],
            [0.0, half, -half],
            [0.0, half, half],
        ]
    )
    draw_line(sim, center + rot.apply(local), rgba=rgba)


def draw_scene(
    sim: object,
    obs: dict | None,
    planner: SimplePlanner,
    progress_point: np.ndarray | None,
    pos_pred: np.ndarray | None,
) -> None:
    """Draw the planned path, gates, obstacles and progress markers.

    Gates (yellow + cyan/white frames), obstacles (orange poles), the planned path (green
    line) with the gate waypoints (magenta), a red marker at the drone's current progress
    along the path (its projection point θ), and a blue line tracing the MPCC predicted
    horizon (with a dot at its far end — how far ahead the controller plans).
    """
    if obs is not None:
        draw_points(
            sim, np.atleast_2d(obs["gates_pos"]), rgba=np.array([1.0, 1.0, 0.0, 1.0]), size=0.08
        )
        for gpos, gquat in zip(obs["gates_pos"], obs["gates_quat"]):
            draw_square(sim, gpos, gquat, _GateFrame.OUTER / 2, np.array([0.0, 1.0, 1.0, 1.0]))
            draw_square(sim, gpos, gquat, _GateFrame.OPENING / 2, np.array([1.0, 1.0, 1.0, 1.0]))
        for opos in np.atleast_2d(obs["obstacles_pos"]):
            pole = np.array([[opos[0], opos[1], 0.0], [opos[0], opos[1], opos[2]]])
            draw_line(sim, pole, rgba=np.array([1.0, 0.5, 0.0, 1.0]), start_size=6.0, end_size=6.0)

    # Single red marker: the drone's current progress θ projected onto the path.
    if progress_point is not None:
        draw_points(
            sim, np.atleast_2d(progress_point), rgba=np.array([1.0, 0.0, 0.0, 1.0]), size=0.07
        )

    # Blue line: the full MPC predicted horizon (all N+1 stage positions) — the trajectory
    # the controller is planning over the next ~0.5 s. The line shows the whole prediction
    # so its divergence from the green planned path is visible at a glance; the dot at the
    # far end marks how far ahead the controller is currently planning.
    if pos_pred is not None and len(pos_pred) > 1:
        draw_line(
            sim, pos_pred, rgba=np.array([0.0, 0.0, 1.0, 1.0]), start_size=0.008, end_size=0.008
        )
        draw_points(
            sim, np.atleast_2d(pos_pred[-1]), rgba=np.array([0.0, 0.0, 1.0, 1.0]), size=0.07
        )

    # Fixed gate waypoints (entry / center / exit), magenta.
    raw = getattr(planner, "_raw_waypoints", None)
    if raw:
        fixed = np.array([np.asarray(p) for p, v in raw if v is not None])
        if len(fixed):
            draw_points(sim, fixed, rgba=np.array([1.0, 0.0, 1.0, 1.0]), size=0.05)

    # Planned path (green line). Downsample to ~200 segments — enough to render the
    # B-spline smoothly (20 made it look visibly faceted) while staying light.
    traj = getattr(planner, "pos", None)
    if traj is not None:
        step = max(1, len(traj) // 200)
        draw_line(sim, traj[::step], rgba=np.array([0.0, 1.0, 0.0, 1.0]))


def weight_table(last_mult: np.ndarray | None) -> list[tuple[str, float]]:
    """Current MPCC cost scalars = baseline × the last applied multipliers (None ⇒ baseline)."""
    mult = last_mult if last_mult is not None else np.ones(wp.N_ACTIONS)
    m = dict(zip(wp.WEIGHT_GROUPS, np.asarray(mult, dtype=float)))
    b = BASELINE_WEIGHTS
    return [
        ("q_c", b["q_c"] * m["q_c"]),
        ("q_l", b["q_l"] * m["q_l"]),
        ("q_att", b["q_att"] * m["q_track"]),
        ("q_dr", b["q_dr"] * m["q_track"]),
        ("q_v", b["q_v"] * m["q_v"]),
        ("r_rpy", b["r_rpy"] * m["r_ctrl"]),
        ("r_T", b["r_T"] * m["r_thrust"]),
        ("r_at", b["r_at"] * m["r_ctrl"]),
    ]


def draw_weight_overlay(sim: object, use_rl_weights: bool, last_mult: np.ndarray | None) -> None:
    """Draw the current cost weights as a 2D text box in the viewer's top-right corner.

    The weight table (baseline × ``last_mult``) is computed only after the viewer check and
    inside the guard, so headless runs pay nothing and a weight-math error can never break
    the sim. Uses the MuJoCo viewer's ``add_overlay`` (the same per-frame contract as the
    markers: it must be re-added before every ``render``). No-ops cleanly when there is no
    on-screen viewer (e.g. headless / rgb_array without an overlay-capable viewer).
    """
    viewer = getattr(getattr(sim, "viewer", None), "viewer", None)
    if viewer is None or not hasattr(viewer, "add_overlay"):
        return
    try:
        import mujoco

        pos = mujoco.mjtGridPos.mjGRID_TOPRIGHT
        header = "RL weights ON" if use_rl_weights else "baseline (RL off)"
        viewer.add_overlay(pos, "MPCC cost weights", header)
        for name, val in weight_table(last_mult):
            viewer.add_overlay(pos, name, f"{val:.2f}")
    except Exception as exc:  # never let an overlay quirk break the sim
        print(f"[MPCC] weight overlay skipped: {exc!r}")


def save_speed_trace(
    path: str,
    trace: list[tuple[float, ...]],
    planner: SimplePlanner,
    obs: dict,
    dt: float,
    caution_factor: float,
    caution_radius: float,
) -> None:
    """Dump a finished episode's per-tick speed-target trace to ``path`` (npz).

    Columns let a plot attribute each drop separately: ``v_tgt`` is the flat cruise ceiling,
    ``v_cruise`` = ceiling after the curvature cap, ``caution``/``brake`` are the two remaining
    multipliers, and ``v_target`` = v_cruise*caution*brake is what the MPCC actually used.
    """
    tr = np.asarray(trace, dtype=float)  # (T, 7)
    gate_thetas = getattr(planner, "gate_thetas", None)
    gate_thetas = np.asarray(gate_thetas, float) if gate_thetas is not None else np.array([])

    # Project each obstacle onto the path: its nearest arc length (the point where the pole sits
    # tangential to the racing line) and the in-plane clearance there, so the plot can mark it.
    dense = np.asarray(planner._path_dense, float)
    s_dense = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))))
    obst = np.asarray(obs.get("obstacles_pos", []), float).reshape(-1, 3)
    obst_thetas = np.array(
        [s_dense[np.argmin(np.linalg.norm(dense[:, :2] - o[:2], axis=1))] for o in obst]
    )
    obst_clear = np.array([np.min(np.linalg.norm(dense[:, :2] - o[:2], axis=1)) for o in obst])
    np.savez(
        path,
        theta=tr[:, 0],
        v_tgt=tr[:, 1],
        v_cruise=tr[:, 2],
        caution=tr[:, 3],
        brake=tr[:, 4],
        v_target=tr[:, 5],
        speed=tr[:, 6],
        gate_thetas=gate_thetas,
        obstacle_thetas=obst_thetas,
        obstacle_clear=obst_clear,
        path_length=float(planner.length),
        freq=float(1.0 / dt),
        caution_factor=float(caution_factor),
        caution_radius=float(caution_radius),
    )
    print(f"[MPCC] saved speed trace ({len(tr)} ticks) -> {path}")
