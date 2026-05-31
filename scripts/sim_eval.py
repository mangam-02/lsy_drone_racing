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


def evaluate(
    config: str = "level2.toml",
    controller: str | None = None,
    n_runs: int = 20,
    render: bool = False,
) -> dict:
    """Run ``n_runs`` episodes and print a summary of outcomes.

    Args:
        config: Config file name in ``config/``.
        controller: Controller file in ``lsy_drone_racing/control/`` or None (use config).
        n_runs: Number of episodes to run.
        render: Enable rendering (slow; off by default for batch evaluation).

    Returns:
        A dict with the aggregated statistics.
    """
    config = load_config(Path(__file__).parents[1] / "config" / config)
    config.sim.render = render

    control_path = Path(__file__).parents[1] / "lsy_drone_racing/control"
    controller_path = control_path / (controller or config.controller.file)
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
    env = JaxToNumpy(env)

    n_gates = len(config.env.track.gates)
    results = []  # per-episode dicts

    for run in range(n_runs):
        obs, info = env.reset()
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

        results.append(
            {
                "outcome": outcome,
                "gates_passed": gates_passed,
                "gate_at_crash": None if success else target_gate,
                "time": curr_time if success else None,
            }
        )
        controller.episode_reset()
        logger.info(
            f"run {run + 1}/{n_runs}: {outcome:13s} gates={gates_passed}/{n_gates}"
            + (f"  time={curr_time:.2f}s" if success else "")
        )

    env.close()
    summary = _summarize(results, n_gates, n_runs)
    _print_summary(summary, n_gates, n_runs)
    return summary


def _summarize(results: list[dict], n_gates: int, n_runs: int) -> dict:
    outcomes = Counter(r["outcome"] for r in results)
    gates_hist = Counter(r["gates_passed"] for r in results)
    crash_gate = Counter(r["gate_at_crash"] for r in results if r["gate_at_crash"] is not None)
    times = [r["time"] for r in results if r["time"] is not None]
    return {
        "n_runs": n_runs,
        "n_gates": n_gates,
        "success": outcomes.get("success", 0),
        "outcomes": dict(outcomes),
        "gates_hist": dict(gates_hist),
        "crash_by_gate": dict(crash_gate),
        "avg_success_time": float(np.mean(times)) if times else None,
        "avg_gates_passed": float(np.mean([r["gates_passed"] for r in results])),
    }


def _bar(frac: float, width: int = 20) -> str:
    return "█" * int(round(frac * width)) + "·" * (width - int(round(frac * width)))


def _print_summary(s: dict, n_gates: int, n_runs: int):
    pct = lambda c: 100.0 * c / n_runs  # noqa: E731
    print("\n" + "=" * 60)
    print(f" EVALUATION SUMMARY  ({n_runs} runs)")
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
    print("=" * 60 + "\n")


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    fire.Fire(evaluate, serialize=lambda _: None)
