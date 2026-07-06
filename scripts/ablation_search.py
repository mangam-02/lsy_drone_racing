"""Ablation search: find a seed where the FULL controller succeeds but each ablation fails.

For every seed in a range, the MPCC controller is run in four configurations:

  * ``all_on``       — all features enabled (the baseline)
  * ``no_curvature`` — ``USE_CURVATURE_SPEED = False``  (curvature speed limit off)
  * ``no_gate_boost``— ``USE_GATE_TRACK_BOOST = False`` (near-gate contouring boost off)
  * ``no_caution``   — ``USE_CAUTION = False``          (blind-approach slowdown off)

The flags are class attributes read at runtime, so each variant just toggles them on the
controller class before the run — no solver rebuild, no edits to the controller file.

It then prints:
  * a grid of every seed × variant outcome,
  * the "ideal" seeds where ``all_on`` succeeds AND all three ablations fail (one seed that
    demonstrates every feature at once), and
  * per-ablation coverage: for each feature, the seeds where turning it off breaks a run the
    full controller completes — plus a minimal set of seeds that covers all three cases, and
    ready-to-run ``record_video.py`` commands.

Run as:

    $ python scripts/ablation_search.py --config level2.toml --seeds 0-30
    $ python scripts/ablation_search.py --config level2.toml --seeds 0-200 --workers 8
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

import fire

# Reuse sim_eval's env building + episode/crash logic (single source of truth).
sys.path.insert(0, str(Path(__file__).parent))
from sim_eval import _build_env, _run_episode, _setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

#: The three ablatable features → the class flag each turns off (``None`` = full controller).
ABLATIONS: dict[str, str | None] = {
    "all_on": None,
    "no_curvature": "USE_CURVATURE_SPEED",
    "no_gate_boost": "USE_GATE_TRACK_BOOST",
    "no_caution": "USE_CAUTION",
}
_FLAGS = ("USE_CURVATURE_SPEED", "USE_GATE_TRACK_BOOST", "USE_CAUTION")


def _parse_seeds(seeds: int | str) -> list[int]:
    """Parse ``--seeds``: int ``N`` = ``0..N-1``; ``"a-b"`` = inclusive range; ``"a,b,c"`` list."""
    if isinstance(seeds, int):
        return list(range(seeds))
    s = str(seeds).strip()
    if "-" in s and "," not in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",") if x.strip()]


def _apply_variant(cls: type, off_flag: str | None, base: dict[str, bool]) -> None:
    """Reset all ablation flags to their baseline, then turn ``off_flag`` (if any) off."""
    for k in _FLAGS:
        setattr(cls, k, base[k])
    if off_flag is not None:
        setattr(cls, off_flag, False)


_ABLATION_NAMES = [v for v in ABLATIONS if v != "all_on"]


def _is_ideal(res: dict[str, dict]) -> bool:
    """True when the full controller succeeds but every ablation fails (one seed shows it all)."""
    return res["all_on"]["outcome"] == "success" and all(
        res[a]["outcome"] != "success" for a in _ABLATION_NAMES
    )


def _log_seed(seed: int, res: dict[str, dict]) -> None:
    """Print one seed's result the moment it finishes; flag it loudly if it's an ideal seed.

    Uses ``print(flush=True)`` (not logging) so the line shows immediately in both serial and
    worker processes — worker module loggers aren't raised to INFO and logging can buffer.
    """
    line = "  ".join(f"{v}={_fmt(res[v])}" for v in ABLATIONS)
    mark = (
        "   ★★★ IDEAL (all_on OK, every ablation fails) — Ctrl-C to stop ★★★"
        if _is_ideal(res)
        else ""
    )
    print(f"seed {seed:>4}: {line}{mark}", flush=True)


_LIBC = ctypes.CDLL(None) if sys.platform != "win32" else None


@contextmanager
def _silence_c_output():
    """Redirect C-level stdout+stderr (fds 1 & 2) to /dev/null for the block.

    acados prints QP-solver warnings ("SQP_RTI: QP solver returned error status 3 ...") via C
    ``printf`` — Python logging can't touch them, and ablations make them frequent (expected). We
    only silence during the solves; our progress and the final table are printed *outside* this
    block. libc's own buffers are flushed into /dev/null before the fds are restored, otherwise
    fully-buffered output (when stdout is a pipe, not a TTY) would leak out afterwards.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    saved_out, saved_err = os.dup(1), os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        if _LIBC is not None:
            _LIBC.fflush(None)  # push buffered C stdio into /dev/null before restoring
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)


def _run_one_seed(env, cls, cfg, n_gates: int, base: dict[str, bool], seed: int) -> dict[str, dict]:
    """Run all four variants for a single seed and return ``{variant: result_dict}``."""
    res: dict[str, dict] = {}
    with _silence_c_output():  # hide acados QP-solver spam during the solves
        for variant, off_flag in ABLATIONS.items():
            _apply_variant(cls, off_flag, base)
            result, _ = _run_episode(env, cls, cfg, n_gates, seed=seed, run=0)
            res[variant] = result
    return res


def _run_seed_chunk(config_name: str, controller_name: str | None, seeds: list[int]) -> dict:
    """Worker: run every variant for each assigned seed; return ``{seed: {variant: result}}``.

    Builds its own env/controller (self-contained per process) and captures the class's baseline
    flag values first so ``all_on`` reflects the real defaults, not a hard-coded ``True``. Prints
    each seed live (so parallel runs still stream results) and returns whatever finished if the
    worker is interrupted.
    """
    _setup_logging()
    env, cls, cfg, n_gates = _build_env(config_name, controller_name, False, None)
    base = {k: getattr(cls, k) for k in _FLAGS}
    out: dict[int, dict[str, dict]] = {}
    try:
        for s in seeds:
            out[s] = _run_one_seed(env, cls, cfg, n_gates, base, s)
            _log_seed(s, out[s])
    except KeyboardInterrupt:
        pass  # return the seeds finished so far; the main process reports them
    finally:
        env.close()
    return out


def _fmt(result: dict) -> str:
    """Compact outcome for the grid: ``✓ 3.1s`` on success, else ``✗ coll@g2`` (with the object)."""
    if result["outcome"] == "success":
        return f"✓ {result['time']:.1f}s"
    gate = result["gate_at_crash"]
    where = f"@g{gate}" if gate is not None else ""
    short = {"collision": "coll", "ground": "grnd", "out_of_bounds": "oob", "timeout": "time"}
    tag = short.get(result["outcome"], result["outcome"])
    return f"✗ {tag}{where}"


def search(
    config: str = "level2.toml",
    controller: str | None = None,
    seeds: int | str = "0-30",
    workers: int = 1,
) -> dict:
    """Search seeds for one where the full controller wins but every ablation loses.

    Args:
        config: Config file name in ``config/`` (e.g. ``level2.toml``).
        controller: Controller file in ``lsy_drone_racing/control/`` or None (use config).
        seeds: Seeds to test. Int ``N`` → ``0..N-1``; ``"a-b"`` inclusive range; ``"a,b,c"`` list.
        workers: Process count. ``>1`` runs seed chunks in parallel (the acados solver is warmed
            up once in the main process first so workers only load the cached solver).

    Returns:
        ``{seed: {variant: result_dict}}`` for all tested seeds.
    """
    seed_list = _parse_seeds(seeds)
    workers = int(workers)
    print(
        f"testing {len(seed_list)} seeds × {len(ABLATIONS)} variants on {config} "
        f"(first seed is slow: builds/compiles the solver)...",
        flush=True,
    )
    print("results stream in per seed; Ctrl-C anytime to stop and still get the table.", flush=True)
    results: dict[int, dict] = {}

    if workers > 1:
        # Warm up codegen/compile ONCE so parallel workers only load the cached solver.
        env, cls, cfg, _ = _build_env(config, controller, False, None)
        obs, info = env.reset(seed=seed_list[0])
        cls(obs, info, cfg)
        env.close()
        chunks = [c for w in range(workers) if (c := seed_list[w::workers])]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_run_seed_chunk, config, controller, c) for c in chunks]
            try:
                for fut in as_completed(futs):
                    results.update(fut.result())
            except KeyboardInterrupt:
                msg = f"\nInterrupted — reporting {len(results)} seed(s) from done chunks."
                print(msg, flush=True)
                ex.shutdown(wait=False, cancel_futures=True)
    else:
        env, cls, cfg, n_gates = _build_env(config, controller, False, None)
        base = {k: getattr(cls, k) for k in _FLAGS}
        try:
            for s in seed_list:
                res = _run_one_seed(env, cls, cfg, n_gates, base, s)
                results[s] = res  # store only fully-completed seeds
                _log_seed(s, res)
        except KeyboardInterrupt:
            msg = f"\nInterrupted — reporting the {len(results)} seed(s) completed so far."
            print(msg, flush=True)
        finally:
            env.close()

    _report(results, config)
    return results


def _report(results: dict[int, dict], config: str) -> None:
    """Print the seed × variant grid, the ideal seeds, and per-ablation coverage + commands."""
    if not results:
        print("No seeds completed — nothing to report.", flush=True)
        return
    seeds = sorted(results)
    variants = list(ABLATIONS)
    ablations = [v for v in variants if v != "all_on"]

    def ok(seed: int, variant: str) -> bool:
        return results[seed][variant]["outcome"] == "success"

    # ── Grid ────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f" ABLATION GRID  ({len(seeds)} seeds, {config})   ✓ = finished, ✗ = failed")
    print("=" * 78)
    w = 13
    print(" seed | " + " | ".join(f"{v:<{w}}" for v in variants))
    print("-" * 78)
    for s in seeds:
        row = " | ".join(f"{_fmt(results[s][v]):<{w}}" for v in variants)
        star = "  ←IDEAL" if ok(s, "all_on") and all(not ok(s, a) for a in ablations) else ""
        print(f" {s:>4} | {row}{star}")

    # ── Ideal seeds: all_on wins, every ablation loses ────────────────────────────────
    ideal = [s for s in seeds if ok(s, "all_on") and all(not ok(s, a) for a in ablations)]
    print("\n" + "=" * 78)
    if ideal:
        print(f" IDEAL SEEDS (all_on succeeds, ALL ablations fail): {ideal}")
        print(" → one seed demonstrates every feature. Recommended:", ideal[0])
    else:
        print(" No single seed breaks all three ablations while all_on succeeds.")
        print(" → cover the cases with several seeds (see below).")

    # ── Per-ablation coverage ────────────────────────────────────────────────────────
    print("\n Per-ablation coverage (all_on succeeds, this feature-off run fails):")
    cover: dict[str, list[int]] = {}
    for a in ablations:
        hits = [s for s in seeds if ok(s, "all_on") and not ok(s, a)]
        cover[a] = hits
        preview = ", ".join(f"{s}({_fmt(results[s][a])})" for s in hits[:6]) or "— none found —"
        print(f"   {a:<14}: {len(hits):>3} seed(s)  e.g. {preview}")

    # Minimal seed set covering all three ablations (greedy).
    print("\n" + "=" * 78)
    if all(cover[a] for a in ablations):
        chosen = _min_cover(cover, ablations)
        print(f" Minimal seed set covering all three ablations: {sorted(chosen)}")
        _print_commands(chosen, cover, ablations, config)
    else:
        missing = [a for a in ablations if not cover[a]]
        print(f" Could not find a failing seed for: {missing}  — widen --seeds and retry.")
    print("=" * 78 + "\n")


def _min_cover(cover: dict[str, list[int]], ablations: list[str]) -> set[int]:
    """Greedy set-cover: pick the fewest seeds so every ablation has a failing seed among them."""
    # Prefer an ideal seed (covers everything) if one exists.
    for s in cover[ablations[0]]:
        if all(s in cover[a] for a in ablations):
            return {s}
    chosen: set[int] = set()
    remaining = {a: set(cover[a]) for a in ablations}
    while any(remaining.values()):
        # Pick the seed that newly covers the most still-uncovered ablations.
        counts: dict[int, int] = {}
        for a, ss in remaining.items():
            for s in ss:
                counts[s] = counts.get(s, 0) + 1
        best = max(counts, key=lambda s: counts[s])
        chosen.add(best)
        for a in list(remaining):
            remaining[a].discard(best)
            if best in cover[a]:
                remaining[a].clear()  # this ablation is now covered
    return chosen


def _print_commands(chosen: set[int], cover: dict, ablations: list[str], config: str) -> None:
    """Emit ready-to-run record_video.py commands for the chosen demonstration seeds."""
    print("\n Record the videos (all_on + the ablation(s) each seed demonstrates):")
    tag = config.replace(".toml", "")
    for s in sorted(chosen):
        print(f"\n   # seed {s}")
        print(
            f"   python scripts/record_video.py --config {config} --seed {s}"
            f" --out report/presentation/{tag}_seed{s}_all_on.mp4"
        )
        for a in ablations:
            if s in cover[a]:
                print(
                    f"   python scripts/record_video.py --config {config} --seed {s}"
                    f" --ablate {a} --out report/presentation/{tag}_seed{s}_{a}.mp4"
                )


if __name__ == "__main__":
    _setup_logging()
    logger.setLevel(logging.INFO)
    fire.Fire(search, serialize=lambda _: None)
