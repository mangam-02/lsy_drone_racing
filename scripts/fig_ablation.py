"""Generate the ablation bar charts (report figures).

Success rate (bars, left axis) and mean lap time (markers, right axis) for the
five ablation configurations, one figure per level (n=100 each, seed=0). Numbers
are produced by scripts/sim_eval.py and recorded in report/sim_eval_results.md.

  figures/ablation_bars.pdf      — Level 2 (development track)
  figures/ablation_bars_l3.pdf   — Level 3 (held-out, randomised tracks)

Run:  ./venv_drone/bin/python scripts/fig_ablation.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parents[1] / "figures"

LABELS = ["Full", "$-$Boost", "$-$Curv.", "$-$Caution", "All off"]

# Level 2 (fixed track), n=100, seed=0.
SUCCESS_L2 = [88.0, 83.0, 88.0, 79.0, 62.0]  # %
TIME_L2 = [7.37, 7.46, 7.15, 6.13, 5.97]      # s

# Level 3 (randomised tracks, held-out), n=100, seed=0.
SUCCESS_L3 = [87.0, 91.0, 83.0, 90.0, 81.0]   # %
TIME_L3 = [8.51, 8.32, 7.86, 7.46, 7.30]      # s


def save(fig, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    print(f"  wrote figures/{name}.png  +  .pdf")


def make(success, time, title, name) -> None:
    x = np.arange(len(LABELS))
    fig, ax1 = plt.subplots(figsize=(5.6, 3.4))

    ax1.bar(x, success, width=0.62, color="#4c78a8",
            edgecolor="#274c6e", label="success rate")
    ax1.set_ylabel("success rate [%]", color="#274c6e", fontsize=11)
    ax1.set_ylim(0, 100)
    ax1.tick_params(axis="y", labelcolor="#274c6e")
    ax1.set_xticks(x)
    ax1.set_xticklabels(LABELS, fontsize=10)
    for xi, s in zip(x, success):
        ax1.annotate(f"{s:.0f}", (xi, s + 1.5), ha="center", fontsize=9,
                     color="#274c6e")

    ax2 = ax1.twinx()
    ax2.plot(x, time, "o-", color="#d94801", lw=2.0, ms=7, label="mean lap time")
    ax2.set_ylabel("mean lap time [s]", color="#d94801", fontsize=11)
    lo = np.floor(min(time) * 2) / 2 - 0.5
    hi = np.ceil(max(time) * 2) / 2 + 0.5
    ax2.set_ylim(lo, hi)
    ax2.tick_params(axis="y", labelcolor="#d94801")
    for xi, t in zip(x, time):
        ax2.annotate(f"{t:.2f}", (xi + 0.18, t + 0.04), ha="left", fontsize=9,
                     color="#d94801")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower left", frameon=False, fontsize=9)

    ax1.set_title(title, fontsize=11)
    fig.tight_layout()
    save(fig, name)


def main() -> None:
    make(SUCCESS_L2, TIME_L2, "Leave-one-out ablation (Level 2, $n=100$)",
         "ablation_bars")
    make(SUCCESS_L3, TIME_L3, "Held-out ablation (Level 3, $n=100$)",
         "ablation_bars_l3")


if __name__ == "__main__":
    main()
