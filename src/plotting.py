"""House style ported verbatim from Phase 1 Cell 2, plus Phase 2 figure helpers."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from .config import FIGDIR

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
       "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ_CMAP = mpl.colors.LinearSegmentedColormap.from_list("seq_blue", SEQ)

CRITICAL = "#d03b3b"

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK_2,
    "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlepad": 10, "axes.labelsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "text.color": INK, "font.size": 9.5,
    "font.family": "sans-serif", "legend.frameon": False, "legend.fontsize": 9,
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
})


def save_fig(fig, name):
    """Write a figure to figures/<name>.png at 150 dpi for the report document."""
    path = FIGDIR / f"{name}.png"
    fig.savefig(path)
    print(f"saved -> {path}")
    return path


def thousands(x, _pos=None):
    return f"{x:,.0f}"


KFMT = FuncFormatter(thousands)


def rate_bars(ax, rates, los, his, counts, labels, title, *,
              colours=None, overall=None, xlabel=None, rotate=0, pct_fmt="{:.2f}%"):
    """Standard default-rate bar panel: Wilson CIs, counts in the tick labels."""
    rates, los, his = np.asarray(rates), np.asarray(los), np.asarray(his)
    x = np.arange(len(rates))
    ax.bar(x, rates, color=(colours if colours is not None else CAT[0]), width=0.62, zorder=3)
    ax.errorbar(x, rates, yerr=[rates - los, his - rates], fmt="none",
                ecolor=INK_2, elinewidth=1.1, capsize=4, zorder=4)
    top = float(his.max())
    ax.set_ylim(0, top * 1.24)
    for xi, r, h in zip(x, rates, his):
        ax.text(xi, h + top * 0.045, pct_fmt.format(r), ha="center", va="bottom",
                fontsize=8.5, color=INK, zorder=5)
    if overall is not None:
        ax.axhline(overall, color=AXIS, lw=1.1, zorder=2, label=f"overall {overall:.2f}%")
    ax.set_xticks(x, [f"{labels[i]}\nn={c:,}" for i, c in enumerate(counts)], rotation=rotate)
    if rotate:
        for lb in ax.get_xticklabels():
            lb.set_ha("right")
    ax.set_ylabel("Default rate (%)")
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)


def errorbar_ladder(ax, labels, means, sds, title, *, xlabel="ROC-AUC", ref=None):
    """Horizontal mean +/- SD panel, used for the complexity ladder and ablations."""
    y = np.arange(len(labels))[::-1]
    ax.errorbar(means, y, xerr=sds, fmt="o", color=CAT[0], ecolor=INK_2,
                elinewidth=1.1, capsize=4, markersize=5, zorder=4)
    if ref is not None:
        ax.axvline(ref, color=AXIS, lw=1.1, zorder=2)
    ax.set_yticks(y, labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
