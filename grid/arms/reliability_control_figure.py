#!/usr/bin/env python3
"""
Reliability diagrams inside the internal acquisition control, one panel per
configuration. This is Figure 4.4 of the manuscript.

WHY THIS FILE EXISTS. The manuscript's calibration claims are made inside the
LoHi-WELD internal control, and Table 4.13 reports them from
calibration_arms.csv. Its reliability diagram used to be the main grid's
figures/reliability.png, and that figure was BLANK. grid/analysis.py read the
calibration bins with the keys lower, upper, confidence and accuracy, while
metrics.expected_calibration_error writes them as lo, hi, mean_confidence and
observed_positive_rate. Only count matched, so every point was NaN, nothing
but the diagonal was drawn, and four columns of
grid_results/analysis/tables/reliability_bins.csv were empty for the same
reason. Found 24 September 2026, when the built manuscript was opened and
Figure 4.4 held no data. The reader now takes the evaluator's keys, and the
pooled main-grid figure is retired, because a curve pooled over detector
families and over targets with different defect rates cannot be read.

This script reads the RELEASED reliability_bins_arms.csv and nothing else.
analysis_arms.py wrote that file with the evaluator's own keys. No model, no
GPU and no torch are needed.

WHAT IT DRAWS. Within one configuration, the six in-domain cells (two diagonal
cells by three seeds) are pooled bin by bin, weighted by image count, and so
are the six cross-acquisition cells. That is the same six cells per entry that
Table 4.13 averages. Colour follows the configuration, as in every figure of
the manuscript. The regime is carried by fill and line style, so identity is
never carried by colour alone. Marker area grows with the number of images in
the bin, on a square-root scale because one bin can hold over a thousand
images and another one, and where the mass sits is the finding. Lines join
only bins holding at least MIN_LINE_COUNT images. A bin of one to three images
is shown, as a small marker, but a line through it would draw the eye to
noise. A dotted line marks the share of target images that carry a defect,
which is where a score with no information would put every point.

Configurations are never pooled with one another. Their confidence scores come
from different heads and different normalisations, and Section 4.8 reads
calibration within a configuration and across the shift, never between them.

Usage: python3 grid/arms/reliability_control_figure.py [project root]
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import figstyle as F  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ARM = "lohictl"
MIN_LINE_COUNT = 10
REGIMES = (
    # key, legend label, line style, filled marker
    ("in", "in domain", "-", True),
    ("cross", "across the acquisition boundary", "--", False),
)


def load(root: Path) -> list[dict]:
    src = root / "grid_results_arms" / "analysis" / "reliability_bins_arms.csv"
    return [r for r in csv.DictReader(src.open()) if r["arm"] == ARM]


def pooled(rows: list[dict], family: str, regime: str):
    """Count-weighted pooling of one configuration's cells, bin by bin.
    Returns (mean confidence, observed defect share, images) per non-empty
    bin, in confidence order."""
    acc = defaultdict(lambda: [0.0, 0.0, 0])
    for r in rows:
        if r["family"] != family:
            continue
        if (r["source"] == r["target"]) != (regime == "in"):
            continue
        n = int(r["count"])
        if n == 0:
            continue
        b = acc[round(float(r["lo"]), 6)]
        b[0] += float(r["mean_confidence"]) * n
        b[1] += float(r["observed_positive_rate"]) * n
        b[2] += n
    return [(c / n, o / n, n) for _, (c, o, n) in sorted(acc.items())]


def positive_share(rows: list[dict]) -> float:
    """Share of target images carrying a defect, over the in-domain cells.
    Every regime scores the same two test splits, so any one gives it."""
    num = den = 0.0
    for r in rows:
        n = int(r["count"])
        if n and r["source"] == r["target"]:
            num += float(r["observed_positive_rate"]) * n
            den += n
    return num / den


def main(root: Path) -> None:
    F.apply_style()
    rows = load(root)
    base = positive_share(rows)
    series = {(f, g): pooled(rows, f, g)
              for f in F.CONFIG_ORDER for g, *_ in REGIMES}
    n_max = max(n for pts in series.values() for _, _, n in pts)

    fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.6), sharex=True,
                             sharey=True)
    for ax, fam in zip(axes.flat, F.CONFIG_ORDER):
        colour = F.CONFIG_COLOR[fam]
        marker = F.CONFIG_MARKER[fam]
        # Reference marks, not data. The empty-figure guard skips them, which
        # is what makes a panel holding only these two lines refuse to save.
        ax.plot([0, 1], [0, 1], color=F.INK_MUTED, linewidth=0.8,
                linestyle="--", zorder=1, gid=F.REFERENCE)
        ax.axhline(base, color=F.INK_MUTED, linewidth=0.8, linestyle=":",
                   zorder=1, gid=F.REFERENCE)
        for regime, _, style, filled in REGIMES:
            pts = series[(fam, regime)]
            line = [p for p in pts if p[2] >= MIN_LINE_COUNT]
            ax.plot([p[0] for p in line], [p[1] for p in line],
                    color=colour, linestyle=style, linewidth=1.0, zorder=2)
            ax.scatter([p[0] for p in pts], [p[1] for p in pts],
                       s=[10 + 150 * (p[2] / n_max) ** 0.5 for p in pts],
                       marker=marker, zorder=3, clip_on=False,
                       facecolors=colour if filled else F.SURFACE,
                       edgecolors=colour, linewidths=1.0)
        ax.set_title(F.label_for(fam))
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.06)
        ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_aspect("equal")

    # Both reference lines are named in the legend rather than labelled in
    # the panels, because every panel's upper band is where the data sits.
    handles = [Line2D([0], [0], color=F.INK, linestyle=style, marker="o",
                      markerfacecolor=F.INK if filled else F.SURFACE,
                      markeredgecolor=F.INK, label=label)
               for _, label, style, filled in REGIMES]
    handles += [
        Line2D([0], [0], color=F.INK_MUTED, linestyle="--", linewidth=0.8,
               label="perfect calibration"),
        Line2D([0], [0], color=F.INK_MUTED, linestyle=":", linewidth=0.8,
               label=f"share of images with a defect, {base:.3f}"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.02))
    for ax in axes[1]:
        ax.set_xlabel("mean image-level score in bin")
    for ax in axes[:, 0]:
        ax.set_ylabel("observed share with a defect")
    fig.suptitle("Reliability inside the internal acquisition control")
    fig.tight_layout(rect=(0, 0.07, 1, 1))

    out = root / "grid_results_arms" / "analysis" / "reliability_control"
    F.save(fig, out.with_suffix(".png"))
    F.save(fig, out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out.with_suffix('.png')} and .pdf")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve() if len(sys.argv) > 1
         else HERE.parent.parent)
