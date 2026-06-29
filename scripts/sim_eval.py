"""Batch evaluation of a controller with an end-of-run summary.

Works exactly like ``scripts/sim.py`` (same env, same controller loading), but runs
many episodes headless and prints a summary at the end:

  * how often the run succeeded (all gates passed)
  * the percentage distribution of gates passed
  * how the drone crashed (collision / ground / out-of-bounds / timeout) and how often
  * at which gate each crash happened

Run as:

    $ python scripts/sim_eval.py --config level2.toml --n_runs 50

Crash classification note: when a drone is disabled the env warps it to (-1,-1,-1),
so we classify using the last *alive* position (captured before each step).
"""

from __future__ import annotations

import logging
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import gymnasium
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.utils import load_config, load_controller

if TYPE_CHECKING:
    from lsy_drone_racing.control.controller import Controller
    from lsy_drone_racing.envs.drone_race import DroneRaceEnv

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    """Enable INFO logging. Called in __main__ AND in each worker process (which is spawned
    fresh, so it has no handlers and would otherwise print nothing for its per-run lines)."""
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)


def _classify_crash(prev_pos: np.ndarray, config) -> str:
    """Classify a terminated (non-success) episode from the last alive position."""
    try:
        lo = np.asarray(config.env.track.safety_limits.pos_limit_low, dtype=float)
        hi = np.asarray(config.env.track.safety_limits.pos_limit_high, dtype=float)
    except Exception:  # safety limits not in config — fall back to arena defaults
        lo = np.array([-2.5, -1.5, -1e-3])
        hi = np.array([2.5, 1.5, 2.0])
    if prev_pos[2] <= 0.12:
        return "ground"
    if (prev_pos[:2] < lo[:2]).any() or (prev_pos[:2] > hi[:2]).any() or prev_pos[2] > hi[2]:
        return "out_of_bounds"
    return "collision"


# ── Crash diagnostics: which physical object was the drone closest to? ──────────────────────
# Physical geometry of the track objects (no avoidance margin) — matched to the simulator and to
# the controller's capsule model, so the "nearest object at crash" is a real collision check, not
# the inflated keep-out the controller plans against.
_POLE_RADIUS = 0.015  # obstacle pole (0.03 m diameter)
_POLE_HEIGHT = 1.52
_GATE_OUTER = 0.72  # outer frame width
_GATE_BAR_DIST = 0.28  # bar-centre distance from the gate centre
_GATE_BAR_RADIUS = 0.08  # frame-bar half-thickness
_GATE_STAND_RADIUS = 0.05  # stand (leg) radius


def _seg_clearance(p: np.ndarray, a: np.ndarray, b: np.ndarray, r: float) -> float:
    """Signed clearance from point ``p`` to the capsule (segment ``a→b``, radius ``r``).

    Distance from ``p`` to the closest point on the segment minus ``r``: > 0 outside the
    capsule, < 0 inside it (penetrating). Used to find the object the drone hit.
    """
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(float(np.dot(ab, ab)), 1e-9), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)) - r)


def _collision_capsules(obs: dict) -> list[tuple]:
    """Physical collision capsules ``(type, gate/obstacle idx, p1, p2, radius)`` from live poses.

    Poles → one vertical capsule; each gate → a stand (if it has a leg) plus 4 frame bars, built
    from the gate's up/right axes exactly like the controller's avoidance model but with physical
    radii. ``type`` is ``"obstacle"``, ``"gate_frame"`` or ``"gate_stand"``.
    """
    # Imported lazily: importing scipy at module top (before lsy_drone_racing → crazyflow sets
    # SCIPY_ARRAY_API) makes crazyflow raise at startup.
    from scipy.spatial.transform import Rotation as R

    caps: list[tuple] = []
    for i, op in enumerate(obs["obstacles_pos"]):
        a = np.array([op[0], op[1], 0.0])
        b = np.array([op[0], op[1], _POLE_HEIGHT])
        caps.append(("obstacle", i, a, b, _POLE_RADIUS))
    half = _GATE_OUTER / 2.0
    bd = _GATE_BAR_DIST
    for gi, (gp, gq) in enumerate(zip(obs["gates_pos"], obs["gates_quat"])):
        rot = R.from_quat(gq)
        up = rot.apply([0.0, 0.0, 1.0])
        right = rot.apply([0.0, 1.0, 0.0])

        def bar(p1, p2, typ="gate_frame", r=_GATE_BAR_RADIUS, gi=gi):  # noqa: ANN001, ANN202
            caps.append((typ, gi, p1, p2, r))

        stand_h = float(gp[2]) - half
        if stand_h > 0:  # leg down to the ground (airborne gates have no stand)
            bar(gp - up * half, gp - up * (half + stand_h), "gate_stand", _GATE_STAND_RADIUS)
        bar(gp + up * bd - right * half, gp + up * bd + right * half)
        bar(gp - up * bd - right * half, gp - up * bd + right * half)
        bar(gp - right * bd + up * half, gp - right * bd - up * half)
        bar(gp + right * bd + up * half, gp + right * bd - up * half)
    return caps


def _diagnose_crash(prev_pos: np.ndarray, obs: dict) -> dict:
    """Nearest physical object to the last alive position → likely crash cause.

    Returns the closest capsule's type/index and the signed clearance (negative ⇒ the drone
    was inside that object). For a true collision this pins down whether it was a gate frame,
    a pole or a gate stand; for ground/out-of-bounds it's just the nearest object for context.
    """
    caps = _collision_capsules(obs)
    if not caps:
        return {"obj_type": "none", "obj_idx": -1, "clearance": float("inf")}
    typ, idx, a, b, r = min(caps, key=lambda c: _seg_clearance(prev_pos, c[2], c[3], c[4]))
    return {"obj_type": typ, "obj_idx": int(idx), "clearance": _seg_clearance(prev_pos, a, b, r)}


def _build_env(config_name: str, controller_name: str | None, render: bool, seed: int | None):
    """Load config + controller and build the (JaxToNumpy-wrapped) env. Returns the pieces a
    run loop needs: ``(env, controller_cls, config, n_gates)``."""
    config = load_config(Path(__file__).parents[1] / "config" / config_name)
    config.sim.render = render
    if seed is not None:
        config.env.seed = int(seed)
    control_path = Path(__file__).parents[1] / "lsy_drone_racing/control"
    controller_path = control_path / (controller_name or config.controller.file)
    controller_cls = load_controller(controller_path)
    env: DroneRaceEnv = gymnasium.make(
        config.env.id,
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=config.env.seed,
    )
    return JaxToNumpy(env), controller_cls, config, len(config.env.track.gates)


def _run_episode(env, controller_cls, config, n_gates: int, seed: int | None, run: int):
    """Run a single episode; return ``(result_dict, log_body)``.

    A fresh controller is built per run (no state carry-over between episodes). With a seed,
    run ``r`` resets with ``seed + r`` so the batch is reproducible regardless of run order.
    """
    obs, info = env.reset() if seed is None else env.reset(seed=int(seed) + run)
    controller: Controller = controller_cls(obs, info, config)
    i = 0
    prev_pos = np.asarray(obs["pos"], dtype=float).copy()

    while True:
        curr_time = i / config.env.freq
        action = controller.compute_control(obs, info)
        prev_pos = np.asarray(obs["pos"], dtype=float).copy()  # last alive position
        obs, reward, terminated, truncated, info = env.step(action)
        controller_finished = controller.step_callback(
            action, obs, reward, terminated, truncated, info
        )
        if terminated or truncated or controller_finished:
            break
        if config.sim.render and ((i * 60) % config.env.freq) < 60:
            controller.render_callback(env.unwrapped.sim)
            env.render()
        i += 1

    controller.episode_callback()

    target_gate = int(obs["target_gate"])
    success = target_gate == -1
    gates_passed = n_gates if success else target_gate
    if success:
        outcome = "success"
    elif truncated and not terminated:
        outcome = "timeout"
    else:
        outcome = _classify_crash(prev_pos, config)

    diag = None if success or outcome == "timeout" else _diagnose_crash(prev_pos, obs)
    result = {
        "outcome": outcome,
        "gates_passed": gates_passed,
        "gate_at_crash": None if success else target_gate,
        "time": curr_time if success else None,
        "crash_obj": None if diag is None else f"{diag['obj_type']}[{diag['obj_idx']}]",
        "crash_obj_type": None if diag is None else diag["obj_type"],
        "crash_clearance": None if diag is None else diag["clearance"],
    }
    controller.episode_reset()
    diag_str = ""
    if diag is not None:
        px, py, pz = prev_pos
        diag_str = (
            f"  near={diag['obj_type']}[{diag['obj_idx']}]"
            f" clr={diag['clearance']:+.3f}m at ({px:+.2f},{py:+.2f},{pz:.2f})"
        )
    log_body = (
        f"{outcome:13s} gates={gates_passed}/{n_gates}"
        + (f"  time={curr_time:.2f}s" if success else "")
        + diag_str
    )
    return result, log_body


def _run_chunk(config_name, controller_name, seed, n_runs, runs):
    """Worker entry point: build one env/controller and run the assigned run indices.

    Logs each run's outcome live (workers are spawned without logging configured, so this sets
    it up first) and returns the per-episode result dicts. Builds its own env so each process is
    self-contained (no sharing of the JAX env / acados solver across processes).
    """
    _setup_logging()
    env, controller_cls, config, n_gates = _build_env(config_name, controller_name, False, seed)
    out = []
    for run in runs:
        result, log_body = _run_episode(env, controller_cls, config, n_gates, seed, run)
        logger.info(f"run {run + 1}/{n_runs}: {log_body}")  # live, as each run finishes
        out.append(result)
    env.close()
    return out


def _evaluate_parallel(config_name, controller_name, seed, n_runs, workers):
    """Run the batch across ``workers`` processes; return the collected per-episode results."""
    # Warm up acados codegen/compile ONCE in the main process so the parallel workers only LOAD
    # the cached solver — otherwise several workers race to generate the same C code at startup.
    logger.info(f"warming up solver, then running {n_runs} runs across {workers} workers...")
    env, controller_cls, cfg, _ = _build_env(config_name, controller_name, False, seed)
    obs, info = env.reset() if seed is None else env.reset(seed=int(seed))
    controller_cls(obs, info, cfg)  # triggers the build/compile if not already cached
    env.close()

    chunks = [c for w in range(workers) if (c := list(range(w, n_runs, workers)))]  # round-robin
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(_run_chunk, config_name, controller_name, seed, n_runs, c) for c in chunks
        ]
        try:
            for fut in as_completed(futs):
                results.extend(fut.result())  # per-run lines were already logged live by workers
        except KeyboardInterrupt:
            logger.info(f"\nInterrupted — summarizing the {len(results)} completed run(s) so far.")
            ex.shutdown(wait=False, cancel_futures=True)
    return results


def evaluate(
    config: str = "level2.toml",
    controller: str | None = None,
    n_runs: int = 20,
    render: bool = False,
    seed: int | None = None,
    workers: int = 1,
) -> dict:
    """Run ``n_runs`` episodes and print a summary of outcomes.

    Args:
        config: Config file name in ``config/``.
        controller: Controller file in ``lsy_drone_racing/control/`` or None (use config).
        n_runs: Number of episodes to run.
        render: Enable rendering (slow; off by default for batch evaluation).
        seed: Base seed for reproducibility. When given, run ``r`` is reset with ``seed + r``,
            so the same ``--seed`` always produces the exact same sequence of randomized tracks
            (level 3) / object positions (level 2) — letting you compare controller changes on an
            identical scenario set. ``None`` (default) keeps the config's behaviour (random).
        workers: Process count for the batch. ``>1`` runs episodes in parallel across CPU cores
            (the big speed-up; the per-run seeds keep results identical to a serial run). Forced
            to 1 when ``render`` is on. The solver is warmed up once in the main process first so
            workers don't race on the acados code generation.
    """
    n_gates = len(load_config(Path(__file__).parents[1] / "config" / config).env.track.gates)
    workers = int(workers)

    if workers > 1 and not render:
        results = _evaluate_parallel(config, controller, seed, n_runs, workers)
    else:
        env, controller_cls, run_cfg, _ = _build_env(config, controller, render, seed)
        results = []
        # Ctrl-C still summarizes: an interrupt breaks out of the loop and we report on whatever
        # runs have already completed, instead of throwing away the partial batch.
        try:
            for run in range(n_runs):
                result, log_body = _run_episode(env, controller_cls, run_cfg, n_gates, seed, run)
                results.append(result)
                logger.info(f"run {run + 1}/{n_runs}: {log_body}")
        except KeyboardInterrupt:
            logger.info(f"\nInterrupted — summarizing the {len(results)} completed run(s) so far.")
        env.close()

    n_done = len(results)
    if n_done == 0:
        logger.info("No runs completed; nothing to summarize.")
        return {}
    # Percentages are over the runs actually completed (n_done), not the requested n_runs, so an
    # interrupted batch still reports correct rates.
    summary = _summarize(results, n_gates, n_done)
    summary["config"] = config  # so the printed summary states which level/track it was
    summary["seed"] = seed  # so the summary records which seed set the scenario sequence
    _print_summary(summary, n_gates, n_done)
    return summary


def _summarize(results: list[dict], n_gates: int, n_runs: int) -> dict:
    outcomes = Counter(r["outcome"] for r in results)
    gates_hist = Counter(r["gates_passed"] for r in results)
    crash_gate = Counter(r["gate_at_crash"] for r in results if r["gate_at_crash"] is not None)
    crash_obj_type = Counter(r["crash_obj_type"] for r in results if r.get("crash_obj_type"))
    crash_obj = Counter(r["crash_obj"] for r in results if r.get("crash_obj"))
    times = [r["time"] for r in results if r["time"] is not None]
    return {
        "n_runs": n_runs,
        "n_gates": n_gates,
        "success": outcomes.get("success", 0),
        "outcomes": dict(outcomes),
        "gates_hist": dict(gates_hist),
        "crash_by_gate": dict(crash_gate),
        "crash_by_object_type": dict(crash_obj_type),
        "crash_by_object": dict(crash_obj),
        "avg_success_time": float(np.mean(times)) if times else None,
        "avg_gates_passed": float(np.mean([r["gates_passed"] for r in results])),
    }


def _bar(frac: float, width: int = 20) -> str:
    return "█" * int(round(frac * width)) + "·" * (width - int(round(frac * width)))


def _print_summary(s: dict, n_gates: int, n_runs: int):
    pct = lambda c: 100.0 * c / n_runs  # noqa: E731
    print("\n" + "=" * 60)
    cfg = s.get("config")
    seed = s.get("seed")
    head = f" EVALUATION SUMMARY  ({n_runs} runs"
    head += f", {cfg}" if cfg else ""
    head += f", seed={seed}" if seed is not None else ", seed=random"
    print(head + ")")
    print("=" * 60)
    print(f" Success (all {n_gates} gates): {s['success']:3d}  ({pct(s['success']):5.1f}%)")
    print(f" Avg gates passed:            {s['avg_gates_passed']:.2f} / {n_gates}")
    if s["avg_success_time"] is not None:
        print(f" Avg time (successful runs):  {s['avg_success_time']:.2f}s")

    print("\n Gates-passed distribution:")
    for g in range(n_gates + 1):
        c = s["gates_hist"].get(g, 0)
        tag = " (finished)" if g == n_gates else ""
        print(f"   {g} gates: {c:3d}  ({pct(c):5.1f}%) {_bar(c / n_runs)}{tag}")

    print("\n Outcomes (how it ended):")
    order = ["success", "collision", "ground", "out_of_bounds", "timeout"]
    for k in order:
        c = s["outcomes"].get(k, 0)
        if c:
            print(f"   {k:14s}: {c:3d}  ({pct(c):5.1f}%) {_bar(c / n_runs)}")

    if s["crash_by_gate"]:
        print("\n Crashes by gate (which gate it was heading to):")
        for g in sorted(s["crash_by_gate"]):
            c = s["crash_by_gate"][g]
            print(f"   heading to gate {g}: {c:3d}  ({pct(c):5.1f}%)")

    if s.get("crash_by_object_type"):
        print("\n Nearest object at crash (frame vs pole — physical clearance):")
        for k, c in sorted(s["crash_by_object_type"].items(), key=lambda kv: -kv[1]):
            print(f"   {k:12s}: {c:3d}  ({pct(c):5.1f}%) {_bar(c / n_runs)}")
        print("\n   by specific object:")
        for k, c in sorted(s["crash_by_object"].items(), key=lambda kv: -kv[1]):
            print(f"     {k:16s}: {c:3d}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    _setup_logging()
    fire.Fire(evaluate, serialize=lambda _: None)
