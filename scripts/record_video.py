"""Record an MP4 video of a single simulation run (for the results presentation).

Works like ``scripts/sim.py`` (same env + controller loading), but instead of opening a
live window it renders each frame off-screen (``mode="rgb_array"``) and pipes the frames
straight into the system ``ffmpeg`` binary — so it needs no extra Python packages.

Run as:

    $ python scripts/record_video.py --config level2.toml                    # chase cam (default)
    $ python scripts/record_video.py --config level2.toml --camera fpv_cam:0  # drone's-eye view
    $ python scripts/record_video.py --config level2.toml --controller mpcc/controller.py \
          --out report/presentation/mpcc_level2.mp4 --seed 1 --fps 30

Films from the drone's chase camera (``track_cam:0``) by default; ``--camera fpv_cam:0`` gives
the first-person view. The controller's path/gate overlays are drawn in, but the live MPCC
weight text box is always suppressed. Pass ``--overlays False`` for a clean camera shot.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import gymnasium
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.utils import load_config, load_controller

if TYPE_CHECKING:
    from lsy_drone_racing.control.controller import Controller
    from lsy_drone_racing.envs.drone_race import DroneRaceEnv

logger = logging.getLogger(__name__)

#: Ablation presets → the controller class flag each one turns off (matches ablation_search.py).
ABLATIONS: dict[str, str | None] = {
    "all_on": None,
    "no_curvature": "USE_CURVATURE_SPEED",
    "no_gate_boost": "USE_GATE_TRACK_BOOST",
    "no_caution": "USE_CAUTION",
}


def _open_ffmpeg(out: Path, width: int, height: int, fps: int) -> subprocess.Popen:
    """Start an ffmpeg process that reads raw RGB frames from stdin and writes an MP4."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH — install it (e.g. `brew install ffmpeg`).")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-",  # read frames from stdin
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        str(out),
    ]  # fmt: skip
    return subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def record(
    config: str = "level2.toml",
    controller: str | None = None,
    out: str = "report/presentation/simulation.mp4",
    seed: int | None = None,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
    camera: int | str | None = "track_cam:0",
    overlays: bool = True,
    ablate: str | None = None,
    max_seconds: float | None = None,
) -> str:
    """Fly one episode and save it as an MP4 video.

    Args:
        config: Config file name in ``config/``.
        controller: Controller file in ``lsy_drone_racing/control/`` or None (use config).
        out: Output video path (relative to the repo root). Parent dirs are created.
        seed: Reset seed for a reproducible run (same track/positions). None = random.
        fps: Video frame rate. Frames are subsampled from the sim to hit this rate.
        width: Frame width in pixels (even number for H.264).
        height: Frame height in pixels (even number for H.264).
        camera: Which camera to film from. Default ``"track_cam:0"`` is a chase cam behind the
            drone. Use ``"fpv_cam:0"`` for the drone's first-person view, None for the config's
            world view (``camera=-1``), or an int camera id / camera name.
        overlays: Draw the controller's overlays (planned path, gate/obstacle markers). The
            live MPCC weight text box is always suppressed. Set False for a clean camera shot.
        ablate: Ablation preset for the ablation videos. One of ``"all_on"`` (default behaviour),
            ``"no_curvature"``, ``"no_gate_boost"``, ``"no_caution"``. Turns the matching MPCC
            feature off for this run (class-flag override; the controller file is untouched).
        max_seconds: Optional cap on recorded flight time; None runs until the episode ends.

    Returns:
        The path to the written video file.
    """
    repo_root = Path(__file__).parents[1]
    cfg = load_config(repo_root / "config" / config)
    cfg.sim.render = False  # we render off-screen ourselves; no live window
    if seed is not None:
        cfg.env.seed = int(seed)

    control_path = repo_root / "lsy_drone_racing/control"
    controller_cls = load_controller(control_path / (controller or cfg.controller.file))
    if ablate is not None and ablate != "all_on":
        if ablate not in ABLATIONS:
            raise ValueError(f"--ablate must be one of {list(ABLATIONS)}, got {ablate!r}")
        setattr(controller_cls, ABLATIONS[ablate], False)  # turn the feature off for this run

    env: DroneRaceEnv = gymnasium.make(
        cfg.env.id,
        freq=cfg.env.freq,
        sim_config=cfg.sim,
        sensor_range=cfg.env.sensor_range,
        control_mode=cfg.env.control_mode,
        track=cfg.env.track,
        disturbances=cfg.env.get("disturbances"),
        randomizations=cfg.env.get("randomizations"),
        seed=cfg.env.seed,
    )
    env = JaxToNumpy(env)
    sim = env.unwrapped

    # Camera + its cam_config: a named/fixed camera ignores cam_config (world-view only), so a
    # None picks the config's world view; anything explicit overrides it.
    cam = sim.settings.camera if camera is None else camera
    cam_config = sim.settings.cam_config if cam == -1 else None

    obs, info = env.reset() if seed is None else env.reset(seed=int(seed))
    ctrl: Controller = controller_cls(obs, info, cfg)
    # Never film the MPCC weight text box: the off-screen viewer honours add_overlay, so blank
    # the controller's weight overlay on this instance (leaves the path/gate markers intact).
    if hasattr(ctrl, "_draw_weight_overlay"):
        ctrl._draw_weight_overlay = lambda _sim: None

    out_path = repo_root / out
    proc = _open_ffmpeg(out_path, width, height, fps)
    n_frames = 0
    i = 0
    curr_time = 0.0
    logger.info(
        f"recording {config} [{ablate or 'all_on'}] seed={seed} → {out} "
        f"({width}x{height} @ {fps} fps)..."
    )
    try:
        while True:
            curr_time = i / cfg.env.freq
            action = ctrl.compute_control(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            finished = ctrl.step_callback(action, obs, reward, terminated, truncated, info)
            if terminated or truncated or finished:
                break
            if max_seconds is not None and curr_time >= max_seconds:
                break
            # Subsample sim steps down to the target video fps.
            if ((i * fps) % cfg.env.freq) < fps:
                if overlays:
                    ctrl.render_callback(sim.sim)  # planned path, gate/obstacle markers
                if not sim.data.sim_data.core.mjx_synced:
                    sim.data, sim.sim.mjx_data = sim._render_sync(sim.data, sim.sim.mjx_data)
                frame = sim.sim.render(
                    mode="rgb_array", camera=cam, cam_config=cam_config, width=width, height=height
                )
                proc.stdin.write(frame.tobytes())
                n_frames += 1
            i += 1
    finally:
        ctrl.episode_callback()
        proc.stdin.close()
        proc.wait()
        env.close()

    # Classify how the run ended so callers/loops can react (e.g. "keep the crash-at-gate-0 take").
    target_gate = int(obs["target_gate"])  # -1 once the final gate is passed
    if target_gate == -1:
        outcome, crash_gate = "success", None
    elif truncated and not terminated:
        outcome, crash_gate = "timeout", target_gate
    else:
        outcome, crash_gate = "crash", target_gate
    logger.info(f"done: {n_frames} frames, {curr_time:.2f}s flight → {out_path}")
    # Parseable one-liner for shell loops (grep 'RESULT: ... gate=0').
    print(
        f"RESULT: outcome={outcome} gate={crash_gate} time={curr_time:.2f} out={out_path}",
        flush=True,
    )
    return str(out_path)


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    fire.Fire(record, serialize=lambda _: None)
