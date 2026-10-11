#!/usr/bin/env python3
"""
The manuscript's four figures, drawn at their printed width for one journal.

WHY THIS FILE EXISTS. The analysis scripts draw their figures for the screen,
at sizes chosen for reading the results, and the manuscript used to place
those files shrunk to fit the page. Shrinking is what broke them in print. The
label cost curve was drawn 16 inches wide and placed at 6, so its text printed
at about 3.4 pt, and the control matrix was placed at 151 dpi. A journal
figure has to be drawn at the size it will be printed, with its text at 8 to
9 pt at that size and never below 7.

So the builder hands this module the numbers, read from the same released
files the tables are read from, and this module only draws. Nothing here reads
a result or computes one, apart from the pooling of reliability bins, which is
imported from grid/arms/reliability_control_figure.py so that there is one
copy of it.

Every figure is drawn at its final width with constrained layout and saved
without a tight bounding box, because a tight box changes the size the figure
was drawn at. Each is written three times: a vector PDF and an EPS, the files
a journal asks for, and a 1000 dpi PNG that the Word file embeds at exactly
its drawn width. Fonts are embedded as TrueType, never as Type 3. There are
no titles inside a figure. The caption carries the title, and panels carry a
letter.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "grid"))
sys.path.insert(0, str(ROOT / "grid" / "arms"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.text import Text  # noqa: E402

import figstyle as FS  # noqa: E402
import reliability_control_figure as RC  # noqa: E402

MM = 1 / 25.4
DPI = 1000
MIN_PT = 7.0
# In order of preference. Arial and Helvetica are what Elsevier and Springer
# name; Liberation Sans is metrically identical to Arial and is what this
# laptop has, and Nimbus Sans is the Helvetica clone.
FONTS = ("Arial", "Helvetica", "Liberation Sans", "Nimbus Sans")
# EPS and PDF carry no creation date, so redrawing unchanged content gives
# identical bytes. The PostScript backend reads the date from here.
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")


def font() -> str:
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in FONTS:
        if name in have:
            return name
    raise SystemExit(f"\nFIGURE CHECK FAILED: none of {', '.join(FONTS)} is "
                     f"installed, so the figures cannot be drawn in a font the "
                     f"journals accept.")


def style() -> str:
    FS.apply_style()
    name = font()
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [name],
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5, "figure.titlesize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "standard", "savefig.dpi": DPI,
        "axes.linewidth": 0.6, "lines.linewidth": 1.0,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "grid.linewidth": 0.4, "grid.alpha": 1.0,
        # Transparency is not allowed: EPS cannot carry it.
        "legend.framealpha": 1.0,
    })
    return name


def smallest_font(fig) -> float:
    """The smallest size, in points at the drawn size, of any visible text."""
    sizes = [t.get_fontsize() for t in fig.findobj(Text)
             if t.get_visible() and t.get_text().strip()]
    return min(sizes)


def save(fig, stem: str, outdir: Path, embed_dir: Path) -> dict:
    """Write the PDF, the EPS and the PNG, and describe what was written."""
    w_in, h_in = fig.get_size_inches()
    small = smallest_font(fig)
    if small < MIN_PT:
        raise SystemExit(f"\nFIGURE CHECK FAILED: {stem} has text at "
                         f"{small:.1f} pt at its printed size. The floor is "
                         f"{MIN_PT:.0f} pt.")
    outdir.mkdir(parents=True, exist_ok=True)
    embed_dir.mkdir(parents=True, exist_ok=True)
    FS.save(fig, outdir / f"{stem}.pdf")
    fig.savefig(outdir / f"{stem}.eps", format="eps")
    png = embed_dir / f"{stem}.png"
    FS.save(fig, png, dpi=DPI, metadata={"Software": None})
    plt.close(fig)
    return {"stem": stem, "png": png, "pdf": outdir / f"{stem}.pdf",
            "eps": outdir / f"{stem}.eps", "width_mm": w_in / MM,
            "height_mm": h_in / MM, "min_font_pt": small}


def _mesh(ax, grid, vmax):
    """Heatmap cells as vector patches. An image would be resampled to the
    raster resolution in EPS, which made a two by two matrix 38 MB."""
    n, m = grid.shape
    im = ax.pcolormesh(np.arange(m + 1) - 0.5, np.arange(n + 1) - 0.5, grid,
                       cmap=FS.SEQUENTIAL, vmin=0, vmax=vmax,
                       edgecolors="face", linewidth=0.2, snap=True)
    ax.set_xlim(-0.5, m - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_aspect("equal")
    return im


def _heatmap_cells(ax, grid, fmt, threshold, size):
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            ax.text(j, i, fmt(i, j), ha="center", va="center",
                    fontsize=size,
                    color=FS.SURFACE if v > threshold else FS.INK)


def control_matrix(width_mm: float, cells: dict, acq: tuple,
                   vmax: float = 0.75):
    """The internal acquisition control. cells[(s, t)] = (mean, sd)."""
    fig, ax = plt.subplots(figsize=(width_mm * MM, 0.82 * width_mm * MM),
                           layout="constrained")
    grid = np.array([[cells[(s, t)][0] for t in acq] for s in acq])
    im = _mesh(ax, grid, vmax)
    names = [a.replace("lohi_", "") for a in acq]
    ax.set_xticks(range(len(acq)), names)
    ax.set_yticks(range(len(acq)), names)
    ax.set_xlabel("tested on")
    ax.set_ylabel("trained on")
    ax.grid(False)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    _heatmap_cells(ax, grid, lambda i, j: (
        f"{cells[(acq[i], acq[j])][0]:.3f}\n"
        f"±{cells[(acq[i], acq[j])][1]:.3f}"), 0.4, 8)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("defect-class mAP50")
    cb.outline.set_visible(False)
    return fig


def box_heatmap(width_mm: float, cells: dict, sources: tuple, targets: tuple,
                vmax: float = 0.75):
    """Box-level cross-dataset transfer. cells[(s, t)] = mean mAP50."""
    fig, ax = plt.subplots(figsize=(width_mm * MM, 0.42 * width_mm * MM),
                           layout="constrained")
    grid = np.array([[cells[(s, t)] for t in targets] for s in sources])
    im = _mesh(ax, grid, vmax)
    ax.set_xticks(range(len(targets)), targets)
    ax.set_yticks(range(len(sources)), sources)
    ax.set_xlabel("evaluated on")
    ax.set_ylabel("trained on")
    ax.grid(False)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    _heatmap_cells(ax, grid, lambda i, j: f"{grid[i, j]:.3f}", 0.4, 8)
    cb = fig.colorbar(im, ax=ax, shrink=0.9)
    cb.set_label("defect-class mAP50")
    cb.outline.set_visible(False)
    return fig


def label_cost_curve(width_mm: float, points: list, pairs: list,
                     families: list):
    """Recovery against realised scenes, six panels lettered (a) to (f).

    points: dicts with source, target, family, scenes, recovery, exhausted.
    pairs: the six (source, target) pairs in panel order.
    """
    nrow, ncol = 2, 3
    if len(pairs) != nrow * ncol:
        raise SystemExit(f"\nFIGURE CHECK FAILED: the label cost curve has "
                         f"{len(pairs)} pairs for {nrow * ncol} panels.")
    fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey=True,
                             figsize=(width_mm * MM, 0.62 * width_mm * MM),
                             layout="constrained")
    ys = [p["recovery"] for p in points]
    lo, hi = min(-0.15, min(ys) - 0.05), max(1.15, max(ys) + 0.05)
    for n, (ax, (src, tgt)) in enumerate(zip(axes.flat, pairs)):
        ax.axhline(0, color=FS.INK_MUTED, lw=0.7, ls=":", gid=FS.REFERENCE,
                   zorder=1)
        ax.axhline(1, color=FS.INK_MUTED, lw=0.7, ls="--", gid=FS.REFERENCE,
                   zorder=1)
        for fam in families:
            sel = [p for p in points if p["source"] == src
                   and p["target"] == tgt and p["family"] == fam]
            for ex in (False, True):
                pts = [p for p in sel if p["exhausted"] == ex]
                if not pts:
                    continue
                colour = FS.CONFIG_COLOR[fam]
                ax.scatter([p["scenes"] for p in pts],
                           [p["recovery"] for p in pts],
                           marker=FS.CONFIG_MARKER[fam], s=14 if not ex else 26,
                           facecolors=FS.SURFACE if ex else colour,
                           edgecolors=colour, linewidths=0.8, zorder=3)
        ax.set_xscale("log")
        ax.set_ylim(lo, hi)
        ax.set_title(f"({chr(97 + n)}) {src} to {tgt}", loc="left")
        ax.grid(True, which="major")
    for ax in axes[-1]:
        ax.set_xlabel("realised scenes annotated")
    for ax in axes[:, 0]:
        ax.set_ylabel("recovery R")
    handles = [Line2D([0], [0], linestyle="none",
                      marker=FS.CONFIG_MARKER[f], markersize=4,
                      markerfacecolor=FS.CONFIG_COLOR[f],
                      markeredgecolor=FS.CONFIG_COLOR[f],
                      label=FS.label_for(f)) for f in families]
    handles += [
        Line2D([0], [0], linestyle="none", marker="o", markersize=5,
               markerfacecolor=FS.SURFACE, markeredgecolor=FS.INK,
               label="draw exhausted the train split"),
        Line2D([0], [0], color=FS.INK_MUTED, lw=0.7, ls=":",
               label="zero shot, R = 0"),
        Line2D([0], [0], color=FS.INK_MUTED, lw=0.7, ls="--",
               label="in domain, R = 1")]
    fig.legend(handles=handles, loc="outside lower center", ncol=5,
               frameon=False)
    return fig


def reliability(width_mm: float, rows: list, order: tuple):
    """Reliability inside the internal control, four panels (a) to (d), one
    per configuration in the given order."""
    base = RC.positive_share(rows)
    series = {(f, g): RC.pooled(rows, f, g)
              for f in order for g, *_ in RC.REGIMES}
    n_max = max(n for pts in series.values() for _, _, n in pts)
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True,
                             figsize=(width_mm * MM, 1.04 * width_mm * MM),
                             layout="constrained")
    for n, (ax, fam) in enumerate(zip(axes.flat, order)):
        colour, marker = FS.CONFIG_COLOR[fam], FS.CONFIG_MARKER[fam]
        ax.plot([0, 1], [0, 1], color=FS.INK_MUTED, lw=0.7, ls="--",
                zorder=1, gid=FS.REFERENCE)
        ax.axhline(base, color=FS.INK_MUTED, lw=0.7, ls=":", zorder=1,
                   gid=FS.REFERENCE)
        for regime, _, ls, filled in RC.REGIMES:
            pts = series[(fam, regime)]
            line = [p for p in pts if p[2] >= RC.MIN_LINE_COUNT]
            ax.plot([p[0] for p in line], [p[1] for p in line],
                    color=colour, ls=ls, lw=0.9, zorder=2)
            ax.scatter([p[0] for p in pts], [p[1] for p in pts],
                       s=[6 + 90 * (p[2] / n_max) ** 0.5 for p in pts],
                       marker=marker, zorder=3, clip_on=False,
                       facecolors=colour if filled else FS.SURFACE,
                       edgecolors=colour, linewidths=0.8)
        ax.set_title(f"({chr(97 + n)}) {FS.label_for(fam)}", loc="left")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.06)
        ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_aspect("equal")
    handles = [Line2D([0], [0], color=FS.INK, ls=ls, marker="o",
                      markersize=4,
                      markerfacecolor=FS.INK if filled else FS.SURFACE,
                      markeredgecolor=FS.INK, label=label)
               for _, label, ls, filled in RC.REGIMES]
    handles += [
        Line2D([0], [0], color=FS.INK_MUTED, ls="--", lw=0.7,
               label="perfect calibration"),
        Line2D([0], [0], color=FS.INK_MUTED, ls=":", lw=0.7,
               label=f"share of images with a defect, {base:.3f}")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2,
               frameon=False)
    for ax in axes[1]:
        ax.set_xlabel("mean image-level score in bin")
    for ax in axes[:, 0]:
        ax.set_ylabel("observed share with a defect")
    return fig


def reliability_rows() -> list:
    return RC.load(ROOT)
