"""Regenerate the wave-equation notebook's four-device halo diagram."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as DrawingPath

plt.switch_backend("Agg")

output = Path(__file__).resolve().parent
output.mkdir(exist_ok=True)
orange, green, ink = "#ac5418", "#16746a", "#25323a"
fig, axes = plt.subplots(
    3, 1, figsize=(13.5, 8.8), gridspec_kw={"height_ratios": [1, 1, 0.78]}
)
centres = [1.6, 5.1, 8.6, 12.1]
width = 2.4

for ax, direction, colour, row, halo in (
    (axes[0], 1, orange, "last", "left_halo"),
    (axes[1], -1, green, "first", "right_halo"),
):
    ax.set(xlim=(0, 13.7), ylim=(-0.25, 3.15))
    ax.axis("off")
    source = "u_local[-1]" if direction == 1 else "u_local[0]"
    perm_name = "send_right" if direction == 1 else "send_left"
    title = (
        "Send last rows right; receive left halos"
        if direction == 1
        else "Send first rows left; receive right halos"
    )
    ax.text(0.1, 2.95, title, fontsize=14, weight="bold", color=ink)
    ax.text(
        0.1,
        2.48,
        f"{halo} = lax.ppermute({source}, AXIS_NAME, {perm_name})",
        fontsize=11,
        family="monospace",
        color=ink,
    )
    for p, x in enumerate(centres):
        received_from = (p - direction) % 4
        box = FancyBboxPatch(
            (x - width / 2, 0.8),
            width,
            1.25,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor="#f6f7f8",
            edgecolor="#85919a",
            lw=1,
        )
        ax.add_patch(box)
        ax.text(
            x, 1.79, f"Device {p}", ha="center", fontsize=12, weight="bold", color=ink
        )
        ax.text(x, 1.41, f"sends {row}[{p}]", ha="center", fontsize=11, color=colour)
        ax.text(
            x,
            1.07,
            f"receives {row}[{received_from}]",
            ha="center",
            fontsize=11,
            color=ink,
        )
    for p in range(4):
        destination = (p + direction) % 4
        if abs(destination - p) == 1:
            start = (centres[p] + direction * width / 2, 1.41)
            end = (centres[destination] - direction * width / 2, 1.41)
            ax.add_patch(
                FancyArrowPatch(
                    start,
                    end,
                    arrowstyle="-|>",
                    mutation_scale=15,
                    lw=1.7,
                    color=colour,
                )
            )
            ax.text(
                (start[0] + end[0]) / 2,
                1.71,
                f"({p}, {destination})",
                ha="center",
                fontsize=10,
                color=ink,
            )
        else:
            vertices = [
                (centres[p], 0.8),
                (centres[p], 0.23),
                (centres[destination], 0.23),
                (centres[destination], 0.8),
            ]
            path = DrawingPath(
                vertices, [DrawingPath.MOVETO] + [DrawingPath.LINETO] * 3
            )
            ax.add_patch(
                FancyArrowPatch(
                    path=path, arrowstyle="-|>", mutation_scale=15, lw=1.7, color=colour
                )
            )
            ax.text(
                6.85,
                -0.04,
                f"Periodic wrap: ({p}, {destination})",
                ha="center",
                fontsize=11,
                color=ink,
            )

ax = axes[2]
ax.set(xlim=(0, 13.7), ylim=(0, 2.3))
ax.axis("off")
ax.text(
    0.1, 2.03, "Device 1 after both exchanges", fontsize=14, weight="bold", color=ink
)
ax.text(
    0.1,
    1.62,
    "padded = concatenate([left_halo[None, :], u_local, right_halo[None, :]])",
    fontsize=10.5,
    family="monospace",
    color=ink,
)
for x, w, face, edge, label in (
    (0.1, 3.75, "#faeee4", orange, "left_halo\nlast row from device 0"),
    (3.85, 5.85, "#f0f2f4", "#85919a", "u_local\nrows owned by device 1"),
    (9.7, 3.8, "#e6f3ef", green, "right_halo\nfirst row from device 2"),
):
    ax.add_patch(
        FancyBboxPatch(
            (x, 0.3),
            w,
            0.9,
            boxstyle="square,pad=0",
            facecolor=face,
            edgecolor=edge,
            lw=1.2,
        )
    )
    ax.text(
        x + w / 2,
        0.75,
        label,
        ha="center",
        va="center",
        fontsize=12,
        color=ink,
        linespacing=1.6,
    )
fig.subplots_adjust(left=0.03, right=0.985, top=0.975, bottom=0.03, hspace=0.12)
fig.savefig(output / "wave_halo_exchange.png", dpi=150, facecolor="white")
print(output / "wave_halo_exchange.png")
