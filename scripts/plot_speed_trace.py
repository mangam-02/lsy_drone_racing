"""Plot the *measured* effective speed target of one MPCC run over arc length.

Reads the per-tick trace written by the controller when it is run with the env var
``MPCC_SPEED_TRACE=<out.npz>`` set, and draws the effective target ``v_target`` along the
path, decomposing every drop below the cruise ceiling into the three multiplicative
factors so one can see *what caused it*:

* **Curvature cap** — geometric, always on (V_tgt -> v_cruise).
* **Caution** — slows near a still-unmeasured object (v_cruise -> v_cruise*caution).
* **Gate-pass brake** — barrier ramp before an unpassed gate, ~1 on a clean pass.

Usage::

    MPCC_SPEED_TRACE=/tmp/trace.npz python scripts/sim_eval.py ...  # run once, records episode 0
    python scripts/plot_speed_trace.py /tmp/trace.npz  # -> figures/speed_profile_run.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

C_CAUT = "#ef8a1f"  # caution band (top: straight ramps, information-driven)
C_CURV = "#7e6bb3"  # curvature-cap band (below: adaptive to path geometry)
C_TGT = "#1a1a1a"  # effective target line
C_GATE = "#3a3a3a"  # gate marker
C_OBST = "#8c564b"  # obstacle marker (pole tangential to path)
OBST_NEAR_M = 1.0  # only mark obstacles this close (in-plane) to the racing line
C_SPEED = "#1f77b4"  # actual speed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace", type=Path, help="npz written via MPCC_SPEED_TRACE")
    ap.add_argument("-o", "--out", type=Path, default=Path("figures/speed_profile_run.png"))
    args = ap.parse_args()

    d = np.load(args.trace)
    # Ticks in recorded order. The x-axis is the distance actually flown (integral of speed), which
    # is strictly monotonic — unlike the progress state theta, which jumps backwards on every replan
    # (a revealed pose re-localizes theta) and would otherwise fold the curve back on itself.
    theta = d["theta"]
    v_tgt = d["v_tgt"]
    v_cruise = d["v_cruise"]
    caution = d["caution"]
    v_target = d["v_target"]
    speed = d["speed"]
    brake = d["brake"]
    dt = 1.0 / float(d["freq"])
    s = np.cumsum(speed) * dt  # distance flown [m], monotonic
    vmax = float(d["v_tgt"].max())
    # Attribute the two reductions with CAUTION on top (from the flat ceiling, so its edges are the
    # straight information-driven ramp), CURVATURE below it (adaptive to the path geometry). The
    # order of the multiplicative factors is a display choice; the effective target is unchanged.
    v_after_caution = v_tgt * caution  # ceiling scaled by caution only
    v_after_curv = (
        v_cruise * caution
    )  # then also scaled by the curvature cap (= v_target if brake≈1)

    # Confirm the gate-pass brake never bit on this clean run (so we can omit its band entirely).
    brake_active = int((brake < 0.99).sum())
    print(
        f"gate-brake: max reduction {float((1.0 - brake).max()):.3f}, ticks active {brake_active}"
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(
        s, v_after_caution, v_tgt, color=C_CAUT, alpha=0.6, lw=0, label="Caution reduction"
    )
    ax.fill_between(
        s,
        v_after_curv,
        v_after_caution,
        color=C_CURV,
        alpha=0.6,
        lw=0,
        label="Curvature-cap reduction",
    )

    ax.plot(s, v_target, "-", color=C_TGT, lw=2.2, label="Effective target $v_{\\mathrm{tgt}}$")
    ax.plot(s, speed, "-", color=C_SPEED, lw=1.2, alpha=0.8, label="Actual drone speed")
    ax.axhline(
        vmax,
        ls="--",
        color="#2ca02c",
        lw=1.3,
        label=f"Cruise ceiling $V_{{\\mathrm{{tgt}}}}$ ({vmax:.1f} m/s)",
    )

    def theta_to_x(gt: float) -> float | None:
        hit = np.where(theta >= gt)[0]
        return float(s[hit[0]]) if hit.size else None

    def marker(x: float, color: str, text: str, legend: str | None, dy: float) -> None:
        ax.axvline(x, ls=(0, (5, 2)), color=color, lw=1.3, alpha=0.85, label=legend)
        ax.plot([x], [0.12], marker="^", color=color, ms=8, clip_on=False)
        near_right = x > s.min() + 0.93 * (s.max() - s.min())  # flip label left of the axis edge
        # Label offset to the side of the line (in points) so it never sits on the dashed line;
        # gates and obstacles sit on two tiers (different dy) so nearby ones never collide.
        ax.annotate(
            text,
            xy=(x, 0.12),
            xytext=(-4 if near_right else 4, dy),
            textcoords="offset points",
            ha="right" if near_right else "left",
            va="bottom",
            fontsize=7.5,
            color=color,
        )

    # Gates and obstacles are marked in the same style (dashed line + triangle at their nearest arc
    # length), distinguished only by colour: gates grey, obstacles brown (with in-plane clearance).
    seen_gate = False
    for i, gt in enumerate(np.asarray(d["gate_thetas"], float), start=1):
        xg = theta_to_x(gt) if gt > 0 else None
        if xg is not None:
            marker(xg, C_GATE, f"G{i}", None if seen_gate else "Gate", dy=16)
            seen_gate = True

    o_theta = np.asarray(d["obstacle_thetas"], float) if "obstacle_thetas" in d else np.array([])
    o_clear = np.asarray(d["obstacle_clear"], float) if "obstacle_clear" in d else np.array([])
    seen_obst = False
    for ot, oc in zip(o_theta, o_clear):
        xo = theta_to_x(ot)
        if xo is not None and oc <= OBST_NEAR_M:
            lbl = None if seen_obst else "Obstacle (nearest to path)"
            marker(xo, C_OBST, f"{oc:.2f} m", lbl, dy=4)
            seen_obst = True

    ax.set_xlabel("Distance flown  $s$  [m]")
    ax.set_ylabel("Speed [m/s]")
    ax.set_xlim(s.min(), s.max())
    ax.set_ylim(0, vmax + 0.35)
    # Reorder into 3 columns: [ceiling, target, speed] | [caution, curvature] | [gate, obstacle].
    # matplotlib fills the ncol grid column-major with balanced 3/2/2 columns, so listing the
    # columns' contents in order gives exactly that layout.
    handles, labels = ax.get_legend_handles_labels()
    lut = {lab: h for h, lab in zip(handles, labels)}
    order = [
        "Cruise ceiling",
        "Effective target",
        "Actual drone speed",
        "Caution reduction",
        "Curvature-cap reduction",
        "Gate",
        "Obstacle",
    ]
    picked = [(lut[lab], lab) for key in order for lab in labels if lab.startswith(key)]
    oh = [h for h, _ in picked]
    ol = [lab for _, lab in picked]
    ax.legend(
        oh, ol, loc="lower center", bbox_to_anchor=(0.5, 1.03), ncol=3, fontsize=8.5, framealpha=0.9
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {args.out}  ({len(theta)} ticks)")


if __name__ == "__main__":
    main()
