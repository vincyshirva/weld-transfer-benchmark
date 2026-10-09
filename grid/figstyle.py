#!/usr/bin/env python3
"""
Figure style for the manuscript. One place, so every figure agrees.

COLOUR WAS COMPUTED, NOT CHOSEN. The categorical palette below was selected by
searching the Okabe-Ito set for the four-colour subset whose worst pair
separation survives colour vision deficiency simulation. Measured, in OKLab
distance times 100:

  worst pair, normal vision      15.6   (floor 15)
  worst pair, deuteranopia       11.2   (floor 8)
  worst pair, protanopia         11.2   (floor 8)

The first choice, blue/orange/green/pink, failed: green against pink scores
7.9 under deuteranopia, below the floor. Vermillion replaces pink.

Minimum contrast against a white page is 2.25, which is under the 3.0 wanted
for a thin mark. That is why every categorical figure also carries a distinct
marker shape per series AND a companion table written beside it. Identity is
never carried by colour alone.

Sequential ramps are single hue, light to dark, for magnitude. Diverging ramps
are two hues with a neutral grey midpoint, for signed quantities such as the
generalization gap. Never a rainbow, and never a hue at a diverging midpoint.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")           # no display on the cluster or in CI
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Fixed order. A configuration keeps its colour in every figure, and colour
# follows the configuration, never its rank.
CONFIG_ORDER = ["yolo11n", "yolo11s", "rtdetr_l", "faster_rcnn"]
CONFIG_COLOR = {
    "yolo11n":     "#0072B2",   # blue
    "yolo11s":     "#E69F00",   # orange
    "rtdetr_l":    "#009E73",   # green
    "faster_rcnn": "#D55E00",   # vermillion
}
# Secondary encoding, required because minimum contrast is under 3.0.
CONFIG_MARKER = {"yolo11n": "o", "yolo11s": "s",
                 "rtdetr_l": "^", "faster_rcnn": "D"}

CONFIG_LABEL = {"yolo11n": "YOLO11n", "yolo11s": "YOLO11s",
                "rtdetr_l": "RT-DETR-l", "faster_rcnn": "Faster R-CNN"}

# Ink. Text never wears a series colour.
INK = "#1A1A1A"
INK_MUTED = "#6B6B6B"
GRID = "#DDDDDD"
SURFACE = "#FFFFFF"

# Sequential, one hue light to dark, for magnitudes on [0, 1].
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "weld_seq", ["#F2F7FB", "#CFE2F0", "#8EC0DE", "#3D8DC0", "#0072B2", "#004E7A"])

# Diverging, two hues with a neutral grey midpoint, for signed quantities.
DIVERGING = LinearSegmentedColormap.from_list(
    "weld_div", ["#004E7A", "#3D8DC0", "#CFE2F0",
                 "#EDEDED",
                 "#F6C9AC", "#E2884A", "#8A3B00"])


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.edgecolor": INK_MUTED,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.9,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "text.color": INK,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
    })


def label_for(config: str) -> str:
    return CONFIG_LABEL.get(config, config)


# ---------------------------------------------------------------- saving
# THE EMPTY-FIGURE GUARD. On 24 September 2026 the manuscript's Figure 4.4 was
# found to hold axes, a title and a diagonal, and no data, because the reader
# asked for calibration bin keys the evaluator never writes. It had passed
# every check this project has, because every check reads a number and a
# blank figure contains none. So nothing is saved any more unless every
# visible axes holds at least one data artist with a finite point.
#
# A reference mark is not data: a diagonal of perfect calibration, a base-rate
# line, a zero or in-domain line. Draw each one with gid=REFERENCE and the
# guard skips it. WITHOUT THAT EXCLUSION THE OLD BLANK FIGURE WOULD PASS,
# because its diagonal is a line with finite points. An axes that is meant to
# be empty, such as a spare panel in a grid, must be turned off, not left
# blank. Colourbars are not data axes and are skipped.
REFERENCE = "reference"


def _has_finite_point(artist) -> bool:
    import numpy as np
    from matplotlib.collections import Collection, PathCollection, QuadMesh
    from matplotlib.image import AxesImage
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    def any_finite_row(xy) -> bool:
        xy = np.asarray(xy, dtype=float)
        return bool(xy.size) and bool(
            np.isfinite(xy.reshape(len(xy), -1)).all(axis=1).any())

    if isinstance(artist, Line2D):
        return any_finite_row(artist.get_xydata())
    if isinstance(artist, (AxesImage, QuadMesh)):
        data = artist.get_array()
        if data is None:
            return False
        return bool(np.ma.masked_invalid(
            np.ma.asarray(data, dtype=float)).count())
    if isinstance(artist, PathCollection):
        # A scatter's points are its offsets.
        return any_finite_row(artist.get_offsets())
    if isinstance(artist, Collection):
        # Any other collection, a fill_between band for instance, carries its
        # data in its paths. Its offsets default to a finite (0, 0), so they
        # must not be read as data.
        return any(any_finite_row(p.vertices) for p in artist.get_paths())
    if isinstance(artist, Patch):
        return any_finite_row(artist.get_verts())
    return False


def empty_axes(fig) -> list:
    """Every visible data axes that holds no data artist with a finite point.
    Reference marks, invisible artists, axes that are turned off and
    colourbars are skipped."""
    empty = []
    for ax in fig.get_axes():
        if not ax.get_visible() or not ax.axison:
            continue
        if getattr(ax, "_colorbar", None) is not None \
                or ax.get_label() == "<colorbar>":
            continue
        artists = [*ax.lines, *ax.collections, *ax.images, *ax.patches]
        data = [a for a in artists
                if a.get_visible() and a.get_gid() != REFERENCE]
        if not any(_has_finite_point(a) for a in data):
            empty.append(ax)
    return empty


def save(fig, path, **kwargs) -> None:
    """fig.savefig, refusing an empty figure, with reproducible PDF bytes.

    Raises ValueError, naming the axes, if any visible data axes holds no
    data. PDFs are written without a CreationDate, so saving unchanged
    content again produces identical bytes. PNGs carry no date already.
    """
    from pathlib import Path

    empty = empty_axes(fig)
    if empty:
        names = ", ".join(repr(ax.get_title() or "untitled") for ax in empty)
        raise ValueError(
            f"refusing to save {path}: {len(empty)} axes hold no data artist "
            f"with a finite point ({names}). A reference mark alone is not "
            f"data. Turn an axes off if it is meant to be empty.")
    if Path(path).suffix.lower() == ".pdf":
        kwargs["metadata"] = {"CreationDate": None, **kwargs.get("metadata", {})}
    fig.savefig(path, **kwargs)
