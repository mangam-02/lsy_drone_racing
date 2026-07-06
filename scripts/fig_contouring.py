"""Generate the MPCC contouring/lag-error schematic (report figure b).

A conceptual diagram: the geometric reference path p_d(theta), the drone at a
predicted position p, its projection onto the path, the unit tangent t(theta),
and the decomposition of the deviation d = p - p_d into the longitudinal (lag)
error e_l along t and the perpendicular (contouring) error e_c.

Run:  ./venv_drone/bin/python scripts/fig_contouring.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

OUT = Path(__file__).parents[1] / "figures"


def save(fig, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    print(f"  wrote figures/{name}.png  +  .pdf")


def arrow(ax, p0, p1, color, lw=2.0, style="-|>", ms=14):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle=style,
            mutation_scale=ms,
            lw=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=5,
        )
    )


def main() -> None:
    # A smooth reference path (a gentle curve) parameterised by arc length.
    s = np.linspace(0, 1, 400)
    px = 3.0 * s
    py = 0.9 * np.sin(2.2 * s) + 0.4 * s
    path = np.column_stack([px, py])

    # Point on the path where we linearise (the projection p_d(theta)).
    i = 250
    pd = path[i]
    # Unit tangent at that point.
    tan = path[i + 1] - path[i - 1]
    tan = tan / np.linalg.norm(tan)
    nrm = np.array([-tan[1], tan[0]])  # left normal

    # Drone predicted position: off the path (some lag + some contouring).
    e_l_true = 0.95  # longitudinal component
    e_c_true = 0.70  # perpendicular component
    p = pd + e_l_true * tan + e_c_true * nrm

    foot = pd + e_l_true * tan  # foot of the perpendicular from p onto the tangent

    fig, ax = plt.subplots(figsize=(5.2, 3.4))

    # Path.
    ax.plot(
        path[:, 0],
        path[:, 1],
        color="#1f4e79",
        lw=2.6,
        zorder=1,
        label=r"reference path $p_d(\theta)$",
    )

    # Tangent line (thin, dashed) through pd.
    tl = np.array([pd - 1.15 * tan, pd + 1.15 * tan])
    ax.plot(tl[:, 0], tl[:, 1], "--", color="#888888", lw=1.2, zorder=1)

    # Deviation vector d = p - p_d.
    arrow(ax, pd, p, "#444444", lw=1.6, style="-|>", ms=11)
    ax.annotate(
        r"$d = p - p_d$",
        (pd + p) / 2 + np.array([-0.55, 0.10]),
        color="#444444",
        fontsize=11,
        ha="right",
    )

    # Lag error e_l along tangent.
    arrow(ax, pd, foot, "#2c7fb8", lw=2.4)
    ax.annotate(
        r"$e_l\,t(\theta)$",
        (pd + foot) / 2 + np.array([-0.05, -0.22]),
        color="#2c7fb8",
        fontsize=12,
    )

    # Contouring error e_c perpendicular.
    arrow(ax, foot, p, "#d94801", lw=2.4)
    ax.annotate(r"$e_c$", (foot + p) / 2 + np.array([0.10, -0.02]), color="#d94801", fontsize=13)

    # Tangent direction arrow.
    arrow(ax, pd, pd + 0.9 * tan, "#666666", lw=1.4, ms=10)
    ax.annotate(
        r"$t(\theta)$", pd + 0.95 * tan + np.array([0.0, -0.2]), color="#666666", fontsize=11
    )

    # Right-angle marker at the foot.
    d1 = -tan * 0.09
    d2 = nrm * 0.09
    corner = np.array([foot + d1, foot + d1 + d2, foot + d2])
    ax.plot(corner[:, 0], corner[:, 1], color="#d94801", lw=1.0, zorder=6)

    # Points.
    ax.scatter(*pd, s=45, color="#1f4e79", zorder=7)
    ax.annotate(
        r"$p_d(\theta)$", pd + np.array([-0.02, 0.16]), color="#1f4e79", fontsize=11, ha="right"
    )
    ax.scatter(*p, s=60, color="black", zorder=7)
    ax.annotate(r"drone $p$", p + np.array([0.14, -0.02]), color="black", fontsize=11, ha="left")

    ax.set_aspect("equal")
    ax.set_xlim(px.min() - 0.3, px.max() + 0.3)
    ax.set_ylim(py.min() - 0.5, py.max() + 0.9)
    ax.axis("off")
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    fig.tight_layout()
    save(fig, "contouring_schematic")


if __name__ == "__main__":
    main()
