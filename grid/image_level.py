#!/usr/bin/env python3
"""Image-level results measured against chance.

WHY THIS FILE EXISTS. Found on 6 October 2026 in a review of the
released results, and reproduced here the same day. The manuscript read image-level average precision minus the
target's positive rate, the "lift", as if zero were the level of a scorer that
knows nothing. It is not. `metrics.binary_metrics` interpolates average
precision at 101 recall points under a monotone precision envelope, and on a
small test split that estimator sits ABOVE the positive rate for uninformative
scores: +0.052 on kaggle_v2's 72 positive and 69 negative images, above zero in
99 percent of runs. The transferred detectors' lifts on that target are +0.038
to +0.054, so the "13 to 21 percent of the in-domain margin" the abstract
quoted was the estimator's own floor.

A constant score ties every image, and then the lift depends on tie order:
`np.argsort` keeps archive order, archive order is filename order, and on
kaggle_v2 the filename carries the label, so a constant gets +0.184 there. Real
cells hold few ties, so released values barely move. The defect is in a pinned
file and is recorded rather than fixed.

WHAT THIS MODULE RELEASES. AUROC has a known chance level of 0.5, and ties are
averaged, so a constant scores exactly 0.5 whatever the order. For every
image-level evaluation of the study it writes one row: the sample as the split
blocks see it, the released AP, AUROC for every target, the evaluator's
negatives rule as a FLAG rather than a suppression, the AP a shuffled label
vector gets against the cell's real scores, and a split-block interval on
AUROC. Beside it, the image-level analogue of the transfer ratio:

    retained margin = (AUROC_ij - 0.5) / (AUROC_jj - 0.5)

with numerator and denominator on the same resampled blocks, as
transfer_bootstrap.py pairs P_ij and P_jj. Six summaries follow as markdown.

WHAT `chance_ap` IS, AND IS NOT. It estimates the estimator's floor and
nothing else. Shuffling images treats them as independent, which they are not:
the split block is this study's unit of independence. So no p-value is
published from it. On weld_surface_v2 a shuffle test would call 11 of 12
lohi_weld cells significant where the block interval admits 4.

THE BOOTSTRAP. Blocks are resampled with replacement, every image of a drawn
block travelling with it, exactly as transfer_bootstrap.block_draws does. A
draw is held as a multiplicity per block, so AUROC over the drawn sample is the
weighted Mann-Whitney statistic, ties counted half. That equals rank-based
AUROC with averaged ranks over the concatenated sample, which test_analysis.py
checks against scipy.stats.rankdata. Seeds come from sha256, never hash().

Every evaluation on one test split shares one set of draws, so any two of them
are paired. A shipped (downloaded) test split has no split blocks, and the
archive gives every image its own block. That is stated wherever it is used.

Reads only the raw archives and the evaluation JSONs. Imports no torch:
transfer_bootstrap.py cannot be imported for that reason, because
evaluate_cell imports yolo_dataset, which imports torch. Called by
grid/analysis.py and grid/arms/analysis_arms.py, so the README's two commands
still produce everything.

    python3 grid/image_level.py --kaggle-groups   # once, needs the label tree
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C  # noqa: E402
import metrics as M  # noqa: E402

RESAMPLES = 10_000
SHUFFLES = 2_000
ALPHA = 0.05
SEED_SALT = "image_level"
CHANCE = 0.5

KAGGLE = "kaggle_v2"
KAGGLE_GROUPS = C.CANDIDATE / "kaggle_v2_test_image_groups.json"
# One block of 43 consecutive video frames holds 31 of kaggle_v2's 72 test
# positives. Its effective weight is why the "video block left out" variant
# exists. Found by family, then asserted to be exactly one block.
VIDEO_FAMILY = "SampleV1_2_mp4"
NO_VIDEO = "without the SampleV1_2_mp4 block"

# Display order, and the names the manuscript uses.
TARGET_ORDER = ["kaggle_v2", "weld_surface_v2", "lohi_weld", "visual",
                "twoclass"]
FAMILY_LABEL = {"yolo11n": "YOLO11n", "yolo11s": "YOLO11s",
                "rtdetr_l": "RT-DETR-l", "faster_rcnn": "Faster R-CNN"}


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def stable_seed(*parts) -> int:
    """A seed from sha256. Python's hash() changes per process, the project log 5m."""
    text = ":".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def finite(v) -> bool:
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


def band(values: np.ndarray) -> tuple:
    """Percentile interval over the defined draws, and how many were not."""
    values = np.asarray(values, dtype=float)
    ok = values[np.isfinite(values)]
    n_undef = int(len(values) - len(ok))
    if len(ok) < 2:
        return None, None, n_undef
    lo, hi = np.percentile(ok, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return float(lo), float(hi), n_undef


def write_csv(path: Path, rows: list[dict]) -> None:
    """Same convention as analysis.write_csv: sorted columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r}) if rows else []
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def f3(v, signed=False) -> str:
    if not finite(v):
        return "n/a"
    return f"{v:+.3f}" if signed else f"{v:.3f}"


def interval(lo, hi, signed=False) -> str:
    if not (finite(lo) and finite(hi)):
        return "[n/a]"
    return f"[{f3(lo, signed)}, {f3(hi, signed)}]"


# --------------------------------------------------------------------------
# the sample as blocks see it
# --------------------------------------------------------------------------

def effective_blocks(counts) -> float:
    """(sum c)^2 / sum c^2 over the blocks a class occupies. A class spread
    evenly over n blocks gives n; one block holding everything gives 1."""
    c = np.asarray([x for x in counts if x > 0], dtype=float)
    if len(c) == 0:
        return 0.0
    return float(c.sum() ** 2 / (c ** 2).sum())


def sample_structure(labels: np.ndarray, blocks: np.ndarray) -> dict:
    labels = labels.astype(bool)
    keys = sorted(set(map(str, blocks)))
    pos = defaultdict(int)
    neg = defaultdict(int)
    for y, b in zip(labels, map(str, blocks)):
        (pos if y else neg)[b] += 1
    return {"n_images": int(len(labels)), "n_blocks": len(keys),
            "n_pos": int(labels.sum()), "n_neg": int((~labels).sum()),
            "pos_blocks": len(pos), "neg_blocks": len(neg),
            "pos_effective_blocks": effective_blocks(pos.values()),
            "neg_effective_blocks": effective_blocks(neg.values()),
            "largest_block_images": max(
                (pos[k] + neg[k] for k in keys), default=0)}


# --------------------------------------------------------------------------
# AUROC, chance AP and the block bootstrap
# --------------------------------------------------------------------------

def auroc(scores: np.ndarray, labels: np.ndarray):
    """The evaluator's own AUROC, ranks with ties averaged, for EVERY target.

    binary_metrics suppresses AUROC under MIN_NEGATIVES_FOR_FPR negatives.
    Passing 0 asks it for the number it would have printed. Undefined only
    when a class is empty."""
    labels = np.asarray(labels).astype(bool)
    if labels.all() or not labels.any():
        return None
    return float(M.binary_metrics(np.asarray(scores), labels, 0)["auroc"])


def weighted_auroc(scores: np.ndarray, labels: np.ndarray,
                   weights: np.ndarray) -> np.ndarray:
    """AUROC for every row of weights, a draws by images multiplicity matrix.

    Images are sorted once and tied scores grouped. Within a draw, a positive
    image of multiplicity w beats every negative copy below its score and
    half of every negative copy level with it. Summed and normalised, that is
    the Mann-Whitney statistic of the drawn sample, which is rank-based AUROC
    with ties averaged. NaN where a draw holds no positive or no negative.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = labels.astype(bool)
    order = np.argsort(scores, kind="stable")
    s = scores[order]
    starts = np.r_[0, np.flatnonzero(np.diff(s)) + 1]
    w = np.atleast_2d(weights)[:, order]
    y = labels[order]
    p = np.add.reduceat(w * y, starts, axis=1)
    n = np.add.reduceat(w * ~y, starts, axis=1)
    below = np.cumsum(n, axis=1) - n
    num = (p * (below + 0.5 * n)).sum(axis=1)
    den = p.sum(axis=1) * n.sum(axis=1)
    out = np.full(len(w), np.nan)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def chance_ap(scores: np.ndarray, labels: np.ndarray, shuffles: int,
              seed: int) -> tuple:
    """Mean and 95th percentile of the evaluator's AP over label shuffles.

    binary_metrics sorts by `np.argsort(-scores)`, which depends on the
    scores alone, so a shuffle permutes the labels under one fixed order.
    Each shuffle's AP is then computed from the positions of the positives:
    precision at the k-th positive is k over its position plus one, the
    envelope is the running maximum from the right, and the 101 recall points
    pick the first positive at or above each. That is binary_metrics's AP,
    line for line, which test_analysis.py checks shuffle by shuffle.
    """
    labels = np.asarray(labels).astype(bool)
    n_pos = int(labels.sum())
    if n_pos == 0 or n_pos == len(labels):
        return None, None
    aps = shuffled_aps(scores, labels, shuffle_perms(len(labels), shuffles,
                                                     seed))
    return float(aps.mean()), float(np.percentile(aps, 95))


def shuffle_perms(n: int, shuffles: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.permuted(np.tile(np.arange(n), (shuffles, 1)), axis=1)


def shuffled_aps(scores, labels, perms: np.ndarray) -> np.ndarray:
    """The evaluator's AP for labels[perm], one value per row of perms."""
    labels = np.asarray(labels).astype(bool)
    n_pos = int(labels.sum())
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    y = labels[perms[:, order]]
    pos = np.nonzero(y)[1].reshape(len(perms), n_pos)
    k = np.arange(1, n_pos + 1)
    precision = k / (pos + 1)
    envelope = np.maximum.accumulate(precision[:, ::-1], axis=1)[:, ::-1]
    pick = np.searchsorted(k / n_pos, M.RECALL_POINTS, side="left")
    return envelope[:, pick].mean(axis=1)


def tie_envelope(scores: np.ndarray, labels: np.ndarray) -> tuple:
    """The evaluator's AP with tied scores broken against the positives and
    for them. Every tie order gives an AP inside this interval.

    WHY IT IS NEEDED. binary_metrics sorts with np.argsort(-scores), whose
    default kind is not stable, and numpy picks a vectorised sort by CPU. The
    laptop runs the same numpy 2.5.2 as the cluster and still orders ties
    differently, so a cell with tied scores does not reproduce its released
    AP to the last bit off the cluster. The released value must lie inside
    this envelope. Outside it, the predictions themselves would differ.
    The project log 6a-iii found the same thing and used the same test."""
    s = np.asarray(scores)
    y = np.asarray(labels).astype(bool)
    out = []
    for key in (y, ~y):           # negatives first, then positives first
        order = np.lexsort((key, -s))
        yy = y[order]
        tp, fp = np.cumsum(yy), np.cumsum(~yy)
        precision = tp / np.maximum(tp + fp, 1e-9)
        precision = np.maximum.accumulate(precision[::-1])[::-1]
        out.append(float(M._interp_precision(tp / y.sum(), precision)))
    return out[0], out[1]


def chance_ap_slow(scores, labels, perm) -> float:
    """binary_metrics itself on one shuffle, for the test."""
    return float(M.binary_metrics(np.asarray(scores, dtype=np.float64),
                                  np.asarray(labels).astype(bool)[perm],
                                  0)["ap"])


class Split:
    """One test split, its block structure and its shared bootstrap draws."""

    def __init__(self, sid: str, names, labels, blocks, seed_name: str,
                 resamples: int = RESAMPLES, shipped: bool = False):
        self.sid = sid
        self.names = np.asarray(names)
        self.labels = np.asarray(labels).astype(bool)
        self.blocks = np.asarray(blocks).astype(str)
        self.shipped = shipped
        self.structure = sample_structure(self.labels, self.blocks)
        keys = sorted(set(self.blocks))
        index = {k: i for i, k in enumerate(keys)}
        self.block_of = np.array([index[b] for b in self.blocks])
        # The same sampling call as transfer_bootstrap.block_draws: one row
        # of block picks per draw, with replacement, keys sorted.
        rng = np.random.default_rng(stable_seed(SEED_SALT, seed_name))
        picks = rng.integers(0, len(keys), size=(resamples, len(keys)))
        flat = (np.arange(resamples)[:, None] * len(keys) + picks).ravel()
        mult = np.bincount(flat, minlength=resamples * len(keys))
        self.weights = mult.reshape(resamples, len(keys))[:, self.block_of]
        self.resamples = resamples

    def subset(self, mask: np.ndarray, sid: str) -> "Split":
        return Split(sid, self.names[mask], self.labels[mask],
                     self.blocks[mask], sid, self.resamples, self.shipped)

    def same_images(self, names, labels, blocks) -> bool:
        return (np.array_equal(self.names, np.asarray(names))
                and np.array_equal(self.labels, np.asarray(labels).astype(bool))
                and np.array_equal(self.blocks, np.asarray(blocks).astype(str)))


# --------------------------------------------------------------------------
# reading evaluations
# --------------------------------------------------------------------------

def evaluations(results_dir: Path, patterns: list[tuple[str, str, str]]):
    """Every image-level evaluation under results_dir.

    patterns: (eval prefix, raw prefix, endpoint label). Returns a list of
    dicts and whether every record read was fabricated."""
    out, fabricated = [], []
    for eval_prefix, raw_prefix, endpoint in patterns:
        for path in sorted(results_dir.glob(f"{eval_prefix}*.json")):
            rec = json.loads(path.read_text())
            name = rec["name"]
            fabricated.append(rec.get("FABRICATED") is True)
            cell = rec["cell"]
            raw = results_dir / f"{raw_prefix}{name}.npz"
            for key, entry in rec["targets"].items():
                target, _ = C.split_result_key(key)
                out.append({
                    "name": name, "result_key": key, "target": target,
                    "test_split": entry.get("test_split", "audited"),
                    "arm": cell.get("arm", cell.get("kind", "main")),
                    "source": cell["source"], "family": cell["family"],
                    "seed": cell.get("seed"), "k": cell.get("k"),
                    "draw": cell.get("draw"),
                    "endpoint": rec.get("endpoint", endpoint),
                    "raw": raw, "released": entry.get("image_level", {}),
                    "fabricated": rec.get("FABRICATED") is True})
    return out, bool(fabricated) and all(fabricated)


def split_id(ev: dict) -> str:
    return (ev["target"] if ev["test_split"] != "downloaded"
            else f"{ev['target']}@downloaded")


def load_arrays(ev: dict) -> dict:
    z = np.load(ev["raw"], allow_pickle=True)
    key = ev["result_key"]
    arrays = {f: z[f"{key}__{f}"] for f in (
        "image_score", "image_label", "image_name", "split_block",
        "pred_image", "pred_score", "pred_class")}
    z.close()
    return arrays


def score_row(ev: dict, arrays: dict, split: Split) -> tuple[dict, np.ndarray]:
    """One released row, and the evaluation's AUROC on every draw."""
    # The evaluator scored the float32 array it stored. numpy's sort can break
    # ties differently by dtype, and tie order moves AP, so the point estimate
    # is computed on the stored array itself.
    stored = arrays["image_score"]
    scores = stored.astype(np.float64)
    labels = split.labels
    rel = ev["released"]
    if int(labels.sum()) != rel.get("n_pos") or \
            int((~labels).sum()) != rel.get("n_neg"):
        raise SystemExit(f"{ev['name']} {ev['result_key']}: the raw archive "
                         f"and the record disagree on the label counts.")
    ap = float(M.binary_metrics(stored, labels, C.MIN_NEGATIVES_FOR_FPR,
                                C.ECE_BINS)["ap"])
    retained = rel.get("n_pred_retained")
    tie_lo, tie_hi = tie_envelope(stored, labels)
    if rel.get("ap") is None:
        # evaluate_arm.guard_degenerate_image_ap: nothing retained, every
        # image tied at zero, AP decided by sort order. Kept null as released.
        ap_out = None
    else:
        # The released value is the one reported. Recomputed here it must be
        # identical, or differ only by tie order, inside the envelope.
        if not (tie_lo - 1e-12 <= rel["ap"] <= tie_hi + 1e-12) or \
                not (tie_lo - 1e-12 <= ap <= tie_hi + 1e-12):
            raise SystemExit(f"{ev['name']} {ev['result_key']}: released AP "
                             f"{rel['ap']!r} or recomputed {ap!r} lies "
                             f"outside the tie envelope [{tie_lo!r}, "
                             f"{tie_hi!r}]. The predictions differ.")
        ap_out = float(rel["ap"])
    auc = auroc(stored, labels)
    if rel.get("auroc") is not None and auc != rel["auroc"]:
        raise SystemExit(f"{ev['name']} {ev['result_key']}: AUROC {auc!r} "
                         f"does not reproduce the released {rel['auroc']!r}.")
    draws = weighted_auroc(scores, labels, split.weights)
    if auc is not None:
        whole = weighted_auroc(scores, labels, np.ones((1, len(scores))))[0]
        if abs(whole - auc) > 1e-12:
            raise SystemExit(f"{ev['name']}: weighted AUROC {whole!r} on the "
                             f"full sample is not the evaluator's {auc!r}.")
    lo, hi, n_undef = band(draws)
    cmean, c95 = chance_ap(scores, labels, SHUFFLES, stable_seed(
        "chance", ev["name"], ev["result_key"], ev["endpoint"]))
    rate = float(labels.mean())
    _, tie_counts = np.unique(scores, return_counts=True)
    row = {
        "name": ev["name"], "arm": ev["arm"], "source": ev["source"],
        "family": ev["family"], "seed": ev["seed"], "k": ev["k"],
        "draw": ev["draw"], "endpoint": ev["endpoint"],
        "target": ev["target"], "result_key": ev["result_key"],
        "test_split": ev["test_split"],
        "blocks_are_images": split.shipped,
        **split.structure,
        "positive_rate": rate,
        "image_ap": ap_out,
        "image_ap_above_base": None if ap_out is None else ap_out - rate,
        "image_ap_recomputed_here": ap,
        "image_ap_reproduced_exactly": ap_out is not None and ap == ap_out,
        "image_ap_tie_lo": tie_lo, "image_ap_tie_hi": tie_hi,
        "n_pred_retained": retained,
        "retains_no_prediction": retained == 0,
        "n_images_in_tied_scores": int(tie_counts[tie_counts > 1].sum()),
        "auroc": auc,
        "auroc_released": rel.get("auroc"),
        "meets_min_negatives": int((~labels).sum()) >= C.MIN_NEGATIVES_FOR_FPR,
        "min_negatives_rule": C.MIN_NEGATIVES_FOR_FPR,
        "chance_ap": cmean, "chance_ap_p95": c95,
        "chance_ap_above_base": None if cmean is None else cmean - rate,
        "chance_shuffles": SHUFFLES,
        "auroc_lo": lo, "auroc_hi": hi,
        "auroc_resamples": split.resamples,
        "auroc_undefined_draws": n_undef,
        "auroc_interval_wholly_above_half": finite(lo) and lo > CHANCE,
        "auroc_interval_wholly_below_half": finite(hi) and hi < CHANCE,
    }
    return row, draws


def build_splits(evs: list[dict], arrays_of) -> dict:
    """One Split per test split, asserting every evaluation agrees on it."""
    splits = {}
    for ev in evs:
        sid = split_id(ev)
        a = arrays_of(ev)
        if sid not in splits:
            splits[sid] = Split(sid, a["image_name"], a["image_label"],
                                a["split_block"], sid,
                                shipped=ev["test_split"] == "downloaded")
        elif not splits[sid].same_images(a["image_name"], a["image_label"],
                                         a["split_block"]):
            raise SystemExit(f"{ev['name']} sees a different {sid} test split "
                             f"from the other evaluations. Refusing to pair.")
    return splits


def score_all(evs: list[dict], label: str) -> tuple[list, dict, dict]:
    cache: dict = {}

    def arrays_of(ev):
        k = (ev["name"], ev["result_key"], ev["endpoint"])
        if k not in cache:
            cache[k] = load_arrays(ev)
        return cache[k]

    splits = build_splits(evs, arrays_of)
    rows, draws = [], {}
    for i, ev in enumerate(evs, 1):
        row, d = score_row(ev, arrays_of(ev), splits[split_id(ev)])
        rows.append(row)
        draws[(ev["name"], ev["result_key"], ev["endpoint"])] = d
        if i % 100 == 0 or i == len(evs):
            print(f"  image level, {label}: {i}/{len(evs)} evaluations",
                  flush=True)
    return rows, draws, splits


# --------------------------------------------------------------------------
# the retained margin, AUROC's analogue of the transfer ratio
# --------------------------------------------------------------------------

def margin(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """(num - 0.5) / (den - 0.5), undefined where the denominator is not
    above chance or either side is undefined."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    out = np.full(np.broadcast(num, den).shape, np.nan)
    ok = np.isfinite(num) & np.isfinite(den) & (den - CHANCE > 0)
    out[ok] = (num[ok] - CHANCE) / (den[ok] - CHANCE)
    return out


def margin_rows(point: dict, draws: dict, sources, targets, families, seeds,
                denominator, subset: str = "full") -> list[dict]:
    """Retained margin at three levels.

    point and draws are keyed (source, family, seed, target). denominator
    maps (target, family, seed) to the key of the target's own model.
    seed:          one cell over its own in-domain cell.
    configuration: the three seeds averaged inside every draw, both sides.
    all:           all twelve averaged inside every draw, both sides.
    The diagonal must come back exactly 1 with zero width."""
    out = []
    for t in targets:
        for s in sources:
            groups = {"all": [(f, sd) for f in families for sd in seeds]}
            for f in families:
                groups[f] = [(f, sd) for sd in seeds]
                for sd in seeds:
                    groups[(f, sd)] = [(f, sd)]
            for g, members in groups.items():
                nk = [(s, f, sd, t) for f, sd in members]
                dk = [denominator(t, f, sd) for f, sd in members]
                if not all(k in draws for k in nk + dk):
                    continue
                pn = np.mean([point[k] if finite(point[k]) else np.nan
                              for k in nk])
                pd = np.mean([point[k] if finite(point[k]) else np.nan
                              for k in dk])
                dn = np.mean([draws[k] for k in nk], axis=0)
                dd = np.mean([draws[k] for k in dk], axis=0)
                r = margin(dn, dd)
                p = float(margin(pn, pd)) if np.isfinite(margin(pn, pd)) \
                    else None
                lo, hi, n_undef = band(r)
                level = ("all" if g == "all" else
                         "configuration" if isinstance(g, str) else "seed")
                row = {"subset": subset, "level": level, "source": s,
                       "target": t,
                       "family": None if g == "all" else members[0][0],
                       "seed": members[0][1] if level == "seed" else None,
                       "n_cells": len(members),
                       "auroc_ij": None if not np.isfinite(pn) else float(pn),
                       "auroc_jj": None if not np.isfinite(pd) else float(pd),
                       "retained_margin": p,
                       "retained_margin_lo": lo, "retained_margin_hi": hi,
                       "resamples": len(r),
                       "undefined_draws": n_undef,
                       "denominator_not_above_chance_draws": int(
                           np.sum(np.isfinite(dd) & (dd - CHANCE <= 0))),
                       "interval_excludes_zero": finite(lo) and (
                           lo > 0 or hi < 0)}
                if s == t:
                    ok = np.isfinite(r)
                    if p != 1.0 or not np.all(r[ok] == 1.0):
                        raise SystemExit(
                            f"diagonal retained margin for {t} at level "
                            f"{level} is not exactly 1. Numerator and "
                            f"denominator are not paired.")
                out.append(row)
    return out


def mean_rows(point: dict, draws: dict, sources, targets, families, seeds,
              ap_lift: dict, chance_lift: dict, own: dict) -> list[dict]:
    """Mean AUROC per configuration and over all four, averaged inside each
    draw, with the count of member evaluations wholly above or below 0.5."""
    out = []
    for t in targets:
        for s in sources:
            groups = {"all": [(f, sd) for f in families for sd in seeds]}
            for f in families:
                groups[f] = [(f, sd) for sd in seeds]
            for g, members in groups.items():
                keys = [(s, f, sd, t) for f, sd in members]
                if not all(k in draws for k in keys):
                    continue
                d = np.mean([draws[k] for k in keys], axis=0)
                pts = [point[k] for k in keys]
                lo, hi, n_undef = band(d)
                out.append({
                    "source": s, "target": t,
                    "family": None if g == "all" else g,
                    "in_domain": s == t, "n_cells": len(keys),
                    "mean_auroc": (float(np.mean(pts))
                                   if all(finite(p) for p in pts) else None),
                    "mean_auroc_lo": lo, "mean_auroc_hi": hi,
                    "resamples": len(d), "undefined_draws": n_undef,
                    "mean_ap_lift": float(np.mean([ap_lift[k] for k in keys]))
                    if all(finite(ap_lift[k]) for k in keys) else None,
                    "mean_chance_lift": float(np.mean(
                        [chance_lift[k] for k in keys])),
                    "cells_wholly_above_half": sum(
                        own[k][0] for k in keys),
                    "cells_wholly_below_half": sum(
                        own[k][1] for k in keys)})
    return out


# --------------------------------------------------------------------------
# kaggle_v2: the filename families and the label groups
# --------------------------------------------------------------------------

def family_of(name: str) -> str:
    """The collection an aggregated kaggle_v2 image came from, read off its
    file name: the Roboflow suffix and the export extensions removed, then a
    trailing frame or index number. `bad_weld_vid177_jpeg.rf.<hash>.jpg` is
    `bad_weld_vid`, `SampleV1_2_mp4-12_jpg.rf.<hash>.jpg` is
    `SampleV1_2_mp4`."""
    stem = name.split(".rf.")[0]
    stem = re.sub(r"(_(jpe?g|png))+$", "", stem, flags=re.I)
    m = re.match(r"^(.*?)(?:[-_][\d_\-x]*\d|\d+)$", stem)
    return m.group(1) if m and m.group(1) else stem


def write_kaggle_groups(root: Path = C.PROJECT) -> Path:
    """Write the three label groups of every kaggle_v2 test image, keyed by
    file name, so the label-rule checks run without the image tree.

    defect                    a `Defect` box is present: positive under the
                              frozen rule
    bad_weld_without_defect   a `Bad Weld` box and no `Defect` box: the 11
                              images judgement call 2 in config.py moves
    good_weld_only            neither
    """
    classes = C.DATASETS[KAGGLE]["classes"]
    cid = {c: i for i, c in enumerate(classes)}
    lst = C.CANDIDATE / "splits" / f"{KAGGLE}_test.txt"
    rows = {}
    for entry in lst.read_text().split():
        label = (C.CANDIDATE / KAGGLE / entry.replace("images/", "labels/", 1)
                 ).with_suffix(".txt")
        present = {int(line.split()[0]) for line in label.read_text().splitlines()
                   if line.strip()} if label.exists() else set()
        if cid["Defect"] in present:
            group = "defect"
        elif cid["Bad Weld"] in present:
            group = "bad_weld_without_defect"
        else:
            group = "good_weld_only"
        name = Path(entry).name
        rows[name] = {"group": group, "family": family_of(name),
                      "classes_present": sorted(classes[i] for i in present)}
    out = {"generated_by": "grid/image_level.py --kaggle-groups",
           "source": f"clean_candidate/{KAGGLE}/test/labels",
           "groups": {g: sum(r["group"] == g for r in rows.values())
                      for g in ("defect", "bad_weld_without_defect",
                                "good_weld_only")},
           "images": dict(sorted(rows.items()))}
    KAGGLE_GROUPS.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {KAGGLE_GROUPS}: {out['groups']}")
    return KAGGLE_GROUPS


def family_baseline(split: Split) -> np.ndarray:
    """A score that looks at no pixel: the positive share of the image's
    filename family among the OTHER blocks of the split. An image whose
    family appears in no other block gets the positive share of all the
    other blocks. Scores are computed once, on the full split."""
    fam = np.array([family_of(n) for n in split.names])
    out = np.empty(len(fam))
    for i in range(len(fam)):
        other = split.blocks != split.blocks[i]
        same = other & (fam == fam[i])
        pool = same if same.any() else other
        out[i] = split.labels[pool].mean()
    return out


# --------------------------------------------------------------------------
# main grid
# --------------------------------------------------------------------------

def main_key(row: dict) -> tuple:
    return (row["source"], row["family"], row["seed"], row["target"])


def run_main(results_dir: Path, tabs: Path) -> dict | None:
    """Everything the main, reference and downloaded-partition cells carry."""
    evs, fabricated = evaluations(results_dir, [("eval__", "raw__", "best")])
    missing = [e for e in evs if not e["raw"].is_file()]
    if fabricated and missing:
        print("  image level: the fixtures ship no raw archives, skipped")
        return None
    if missing:
        names = sorted({e["raw"].name for e in missing})
        raise SystemExit(f"image level: {len(names)} raw archives missing "
                         f"under {results_dir}, e.g. {names[:3]}. The "
                         f"image-level tables were not written.")
    rows, draws, splits = score_all(evs, "main grid")
    write_csv(tabs / "image_level.csv", rows)

    grid = [r for r in rows if r["arm"] in ("main", "reference")
            and r["test_split"] == "audited"]
    pt = {main_key(r): r["auroc"] for r in grid}
    dr = {main_key(r): draws[(r["name"], r["result_key"], "best")]
          for r in grid}
    ap_lift = {main_key(r): r["image_ap_above_base"] for r in grid}
    chance_lift = {main_key(r): r["chance_ap_above_base"] for r in grid}
    own = {main_key(r): (r["auroc_interval_wholly_above_half"],
                         r["auroc_interval_wholly_below_half"]) for r in grid}
    sources = C.TRAIN_SOURCES + [t for t in C.ALL_TARGETS
                                 if t not in C.TRAIN_SOURCES]
    families, seeds = list(C.FAMILIES), C.SEEDS

    def denominator(t, f, sd):
        return (t, f, sd, t)

    margins = margin_rows(pt, dr, C.TRAIN_SOURCES + [
        t for t in C.ALL_TARGETS if t not in C.TRAIN_SOURCES],
        C.ALL_TARGETS, families, seeds, denominator)
    margins = [m for m in margins
               if m["source"] in C.TRAIN_SOURCES or m["source"] == m["target"]]

    # The video block left out, on kaggle_v2 only.
    ks = splits[KAGGLE]
    fam = np.array([family_of(n) for n in ks.names])
    video_blocks = set(ks.blocks[fam == VIDEO_FAMILY])
    if len(video_blocks) != 1:
        raise SystemExit(f"{VIDEO_FAMILY} spans {len(video_blocks)} blocks, "
                         f"expected exactly one.")
    keep = ~np.isin(ks.blocks, list(video_blocks))
    nov = ks.subset(keep, f"{KAGGLE} {NO_VIDEO}")
    pt_nv, dr_nv = {}, {}
    for r in grid:
        if r["target"] != KAGGLE:
            continue
        a = load_arrays(next(e for e in evs if e["name"] == r["name"]
                             and e["result_key"] == r["result_key"]))
        s = a["image_score"].astype(np.float64)[keep]
        pt_nv[main_key(r)] = auroc(s, nov.labels)
        dr_nv[main_key(r)] = weighted_auroc(s, nov.labels, nov.weights)
    margins += margin_rows(pt_nv, dr_nv, C.TRAIN_SOURCES, [KAGGLE], families,
                           seeds, denominator, subset=NO_VIDEO)
    write_csv(tabs / "image_level_retained_margin.csv", margins)

    means = mean_rows(pt, dr, sources, C.ALL_TARGETS, families, seeds,
                      ap_lift, chance_lift, own)
    means = [m for m in means
             if m["source"] in C.TRAIN_SOURCES or m["in_domain"]]
    write_csv(tabs / "image_level_means.csv", means)

    samples = sample_table(rows, splits)
    write_csv(tabs / "image_level_samples.csv", samples)
    samples_markdown(tabs / "image_level_samples.md", samples,
                     "Every test split the main grid scores, as the split "
                     "blocks see it. The internal control's two splits are "
                     "in grid_results_arms/analysis/image_level_samples_"
                     "arms.md.")
    rule = {r["target"]: r["meets_min_negatives"] for r in grid}
    matrix_markdown(tabs / "image_level_matrix.md", means, margins, rule)
    audit = audit_rows(rows, splits)
    write_csv(tabs / "image_level_audit.csv", audit)
    audit_markdown(tabs / "image_level_audit.md", audit)
    checks = kaggle_checks(evs, rows, splits, nov, keep)
    write_csv(tabs / "image_level_kaggle_checks.csv", checks["rows"])
    kaggle_markdown(tabs / "image_level_kaggle_checks.md", checks)
    print(f"  image level: {len(rows)} evaluations, {len(margins)} retained "
          f"margins, written to {tabs}")
    return {"rows": rows, "margins": margins, "means": means, "audit": audit,
            "checks": checks, "samples": samples}


def constant_score_lift(labels: np.ndarray):
    """AP minus the positive rate that binary_metrics gives a score holding
    one value for every image, in archive order. Every image is tied, so the
    value is set by the sort's tie order alone, and on kaggle_v2 archive order
    is filename order, which carries the label. Computed on a float32 array,
    as the evaluator stores scores."""
    labels = np.asarray(labels).astype(bool)
    if labels.all() or not labels.any():
        return None
    flat = np.zeros(len(labels), dtype=np.float32)
    return float(M.binary_metrics(flat, labels, 0)["ap"] - labels.mean())


def sample_table(rows: list[dict], splits: dict) -> list[dict]:
    out = []
    for sid, sp in splits.items():
        r = next(x for x in rows if (x["target"] if x["test_split"] !=
                                     "downloaded" else
                                     f"{x['target']}@downloaded") == sid)
        out.append({"split": sid, "target": r["target"],
                    "test_split": r["test_split"],
                    "blocks_are_images": sp.shipped, **sp.structure,
                    "positive_rate": sp.structure["n_pos"]
                    / sp.structure["n_images"],
                    "constant_score_ap_above_base":
                        constant_score_lift(sp.labels)})
    return out


def samples_markdown(path: Path, samples: list[dict], note: str) -> None:
    lines = [
        "# Image-level test splits, counted in split blocks",
        "",
        "The split block is the unit of independence in this study. "
        "\"Effective\" is the squared sum of the per-block counts over the "
        "sum of their squares, so a class spread evenly over n blocks counts "
        "n and a class in one block counts 1. The evaluator's rule of "
        f"{C.MIN_NEGATIVES_FOR_FPR} negative images counts images and does "
        "not see this.",
        "",
        note,
        "",
        "| Test split | Images, blocks | Positive images, in blocks, effective "
        "| Negative images, in blocks, effective | Largest block "
        "| Constant score, AP above the positive rate |",
        "|---|---|---|---|---|---|"]
    for s in samples:
        label = (f"{s['target']}, shipped test split" if s["test_split"] ==
                 "downloaded" else s["split"])
        lines.append(
            f"| {label} | {s['n_images']}, {s['n_blocks']} "
            f"| {s['n_pos']}, in {s['pos_blocks']}, effective "
            f"{s['pos_effective_blocks']:.1f} "
            f"| {s['n_neg']}, in {s['neg_blocks']}, effective "
            f"{s['neg_effective_blocks']:.1f} "
            f"| {s['largest_block_images']} "
            f"| {f3(s['constant_score_ap_above_base'], True)} |")
    lines += ["", "The last column is what binary_metrics gives a score that "
              "holds one value for every image. Every image is tied, so the "
              "value is set by tie order alone, and on kaggle_v2 archive "
              "order is filename order, which carries the label. AUROC "
              "averages ties and gives such a score exactly 0.5."]
    if any(s["blocks_are_images"] for s in samples):
        lines += ["", "A shipped test split carries no split blocks. The raw "
                  "archive gives every image its own block, so its counts "
                  "here are image counts and its intervals treat images as "
                  "independent. Adjacent video frames on those splits are "
                  "not independent, the project log audit finding 2."]
    path.write_text("\n".join(lines) + "\n")


def matrix_markdown(path: Path, means: list[dict], margins: list[dict],
                    rule: dict) -> None:
    fams = list(C.FAMILIES)
    lines = [
        "# Image-level AUROC, source by target",
        "",
        "AUROC has a chance level of 0.5, and ties are averaged, so a "
        "constant score gets exactly 0.5. It is computed here for every "
        "target, including the three where the evaluator suppresses it under "
        f"{C.MIN_NEGATIVES_FOR_FPR} negative images. Those rows are flagged "
        "in the last column. The interval is a 95 percent split-block "
        "percentile interval on the twelve-cell mean, "
        f"{RESAMPLES:,} draws, with the twelve cells averaged inside each "
        "draw. \"AP lift\" is image-level AP minus the positive rate. "
        "\"Chance lift\" is the mean lift when the labels are shuffled "
        f"{SHUFFLES:,} times against each cell's real scores. It is the "
        "level an uninformative scorer reaches on this estimator, not zero. "
        "The last two counts are evaluations whose OWN interval lies wholly "
        "above or wholly below 0.5.",
        "",
        "| Target | Trained on | " + " | ".join(FAMILY_LABEL[f] for f in fams)
        + " | Mean AUROC, with interval | AP lift | Chance lift "
          "| Wholly above 0.5 | Wholly below 0.5 | Negatives under the rule |",
        "|---|---|" + "---|" * len(fams) + "---|---|---|---|---|---|"]
    by = {(m["source"], m["target"], m["family"]): m for m in means}
    for t in TARGET_ORDER:
        order = [s for s in TARGET_ORDER if s in C.TRAIN_SOURCES]
        if t not in C.TRAIN_SOURCES:
            order.append(t)
        for s in order:
            a = by.get((s, t, None))
            if a is None:
                continue
            per = " | ".join(f3(by[(s, t, f)]["mean_auroc"])
                             if (s, t, f) in by else "n/a" for f in fams)
            who = f"{s}, in domain" if s == t else s
            lines.append(
                f"| {t} | {who} | {per} | {f3(a['mean_auroc'])} "
                f"{interval(a['mean_auroc_lo'], a['mean_auroc_hi'])} "
                f"| {f3(a['mean_ap_lift'], True)} "
                f"| {f3(a['mean_chance_lift'], True)} "
                f"| {a['cells_wholly_above_half']} of {a['n_cells']} "
                f"| {a['cells_wholly_below_half']} of {a['n_cells']} "
                f"| {'no' if rule[t] else 'yes'} |")
    lines += [
        "",
        "## Retained share of the in-domain margin above chance",
        "",
        "(AUROC_ij - 0.5) over (AUROC_jj - 0.5), where j's own model is the "
        "same configuration and seed. Numerator and denominator are scored "
        "on the same resampled blocks, and over the four configurations all "
        "twelve cells are averaged inside each draw on both sides. A draw "
        "whose denominator is not above 0.5 is undefined and counted. The "
        "diagonal is exactly 1 by construction and is checked.",
        "",
        "| Target | Trained on | Subset | Retained margin, with interval "
        "| Undefined draws | Interval excludes zero |",
        "|---|---|---|---|---|---|"]
    for m in margins:
        if m["level"] != "all" or m["source"] == m["target"]:
            continue
        lines.append(
            f"| {m['target']} | {m['source']} | {m['subset']} "
            f"| {f3(m['retained_margin'], True)} "
            f"{interval(m['retained_margin_lo'], m['retained_margin_hi'], True)} "
            f"| {m['undefined_draws']} of {m['resamples']:,} "
            f"| {'yes' if m['interval_excludes_zero'] else 'no'} |")
    path.write_text("\n".join(lines) + "\n")


def audit_rows(rows: list[dict], splits: dict) -> list[dict]:
    """A, B and C of the downloaded-against-audited comparison, in AUROC and
    in AP, per seed and as the mean of the three."""
    idx = {(r["arm"], r["source"], r["family"], r["seed"], r["result_key"]): r
           for r in rows}
    out = []
    for r in rows:
        if r["arm"] != "dirty" or r["test_split"] != "downloaded":
            continue
        src, fam, sd = r["source"], r["family"], r["seed"]
        b = idx[("dirty", src, fam, sd, src + C.CLEAN_TEST_SUFFIX)]
        c = idx[("main", src, fam, sd, src)]
        row = {"source": src, "family": fam, "seed": sd}
        for tag, x in (("A", r), ("B", b), ("C", c)):
            row[f"{tag}_auroc"] = x["auroc"]
            row[f"{tag}_auroc_lo"] = x["auroc_lo"]
            row[f"{tag}_auroc_hi"] = x["auroc_hi"]
            row[f"{tag}_image_ap"] = x["image_ap"]
            row[f"{tag}_positive_rate"] = x["positive_rate"]
            row[f"{tag}_n_pos"] = x["n_pos"]
            row[f"{tag}_n_neg"] = x["n_neg"]
            # Pairs a positive and a negative image out of order, ties half.
            row[f"{tag}_pairs_out_of_order"] = (
                (1 - x["auroc"]) * x["n_pos"] * x["n_neg"]
                if finite(x["auroc"]) else None)
            row[f"{tag}_pairs"] = x["n_pos"] * x["n_neg"]
        row["A_minus_C_auroc"] = row["A_auroc"] - row["C_auroc"]
        row["A_minus_C_image_ap"] = row["A_image_ap"] - row["C_image_ap"]
        row["A_blocks_are_images"] = r["blocks_are_images"]
        out.append(row)
    return out


def audit_markdown(path: Path, audit: list[dict]) -> None:
    lines = [
        "# The leakage audit in AUROC",
        "",
        "A is the downloaded-partition model on its shipped test split, B the "
        "same weights on the audited test split, C the audited model on the "
        "audited test split, YOLO11s, mean of three seeds. A and C are scored "
        "on test splits with different positive rates. Average precision "
        "cannot be compared across that, because its floor moves with the "
        "positive rate, and AUROC can. The AP columns are those of "
        "downloaded_vs_audited.csv, recomputed here and identical.",
        "",
        "| Source | A AUROC | B AUROC | C AUROC | A minus C, AUROC "
        "| A AP | B AP | C AP | A minus C, AP | Shipped test, positive "
        "and negative | Audited test, positive and negative "
        "| A, pairs out of order, worst seed |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for src in sorted({a["source"] for a in audit}):
        g = [a for a in audit if a["source"] == src]
        mean = {k: float(np.mean([a[k] for a in g])) for k in (
            "A_auroc", "B_auroc", "C_auroc", "A_image_ap", "B_image_ap",
            "C_image_ap", "A_minus_C_auroc", "A_minus_C_image_ap")}
        worst = max(g, key=lambda a: a["A_pairs_out_of_order"])
        lines.append(
            f"| {src} | {mean['A_auroc']:.3f} | {mean['B_auroc']:.3f} "
            f"| {mean['C_auroc']:.3f} | {mean['A_minus_C_auroc']:+.3f} "
            f"| {mean['A_image_ap']:.3f} | {mean['B_image_ap']:.3f} "
            f"| {mean['C_image_ap']:.3f} | {mean['A_minus_C_image_ap']:+.3f} "
            f"| {g[0]['A_n_pos']} and {g[0]['A_n_neg']} "
            f"| {g[0]['C_n_pos']} and {g[0]['C_n_neg']} "
            f"| {worst['A_pairs_out_of_order']:g} of {worst['A_pairs']:,} |")
    lines += ["", "Per seed, with each evaluation's own interval, in "
              "image_level_audit.csv. A's intervals treat images as "
              "independent, because a shipped split carries no blocks."]
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# kaggle_v2 checks, exploratory
# --------------------------------------------------------------------------

def per_image(arrays: dict, n: int, keep_ids, rule: str) -> np.ndarray:
    """An image score from the raw predictions, before any NMS, which is what
    the frozen image score uses. Classes outside keep_ids are ignored."""
    m = np.isin(arrays["pred_class"], list(keep_ids))
    img = arrays["pred_image"][m]
    s = arrays["pred_score"][m].astype(np.float64)
    out = np.zeros(n)
    if rule == "max":
        np.maximum.at(out, img, s)
    elif rule == "sum":
        np.add.at(out, img, s)
    elif rule.startswith("count>="):
        thr = float(rule.split(">=")[1])
        np.add.at(out, img[s >= thr], 1.0)
    elif rule == "mean_top3":
        for i in range(n):
            top = np.sort(s[img == i])[::-1][:3]
            out[i] = top.sum() / 3.0
    else:
        raise ValueError(rule)
    return out


def kaggle_checks(evs, rows, splits, nov: Split, keep: np.ndarray) -> dict:
    ks = splits[KAGGLE]
    groups = json.loads(KAGGLE_GROUPS.read_text())["images"] \
        if KAGGLE_GROUPS.is_file() else None
    cells = [e for e in evs if e["target"] == KAGGLE and e["arm"] == "main"
             and e["test_split"] == "audited"]
    arrays = {e["name"]: load_arrays(e) for e in cells}
    n = len(ks.labels)
    # The frozen score must come back from the raw predictions exactly.
    for e in cells:
        a = arrays[e["name"]]
        mine = per_image(a, n, C.defect_ids(e["source"]), "max")
        if not np.array_equal(mine.astype(np.float32), a["image_score"]):
            raise SystemExit(f"{e['name']}: the frozen image score does not "
                             f"reproduce from the raw predictions.")

    out_rows = []

    def summarise(check, variant, labels, split, score_of, sources,
                  per_config=True):
        for src in sources:
            mine = [e for e in cells if e["source"] == src]
            groups_ = {"all": mine}
            if per_config:
                for f in C.FAMILIES:
                    groups_[f] = [e for e in mine if e["family"] == f]
            for g, members in groups_.items():
                scores = [score_of(e) for e in members]
                d = np.mean([weighted_auroc(s, labels, split.weights)
                             for s in scores], axis=0)
                pts = [auroc(s, labels) for s in scores]
                lo, hi, nu = band(d)
                out_rows.append({
                    "check": check, "variant": variant, "source": src,
                    "family": None if g == "all" else g,
                    "n_cells": len(members),
                    "n_pos": int(labels.sum()), "n_neg": int((~labels).sum()),
                    "mean_auroc": float(np.mean(pts)),
                    "mean_auroc_lo": lo, "mean_auroc_hi": hi,
                    "resamples": split.resamples, "undefined_draws": nu})

    frozen = lambda e: arrays[e["name"]]["image_score"].astype(np.float64)  # noqa: E731
    summarise("label rule", "frozen: a Defect box is present", ks.labels, ks,
              frozen, C.TRAIN_SOURCES)
    summarise("label rule", NO_VIDEO, nov.labels, nov,
              lambda e: frozen(e)[keep], C.TRAIN_SOURCES)

    baseline = family_baseline(ks)
    lo, hi, nu = band(weighted_auroc(baseline, ks.labels, ks.weights))
    out_rows.append({
        "check": "filename family", "variant": "positive share of the "
        "image's family among the other blocks", "source": None,
        "family": None, "n_cells": 0, "n_pos": int(ks.labels.sum()),
        "n_neg": int((~ks.labels).sum()),
        "mean_auroc": auroc(baseline, ks.labels),
        "mean_auroc_lo": lo, "mean_auroc_hi": hi,
        "resamples": ks.resamples, "undefined_draws": nu})
    fam = np.array([family_of(x) for x in ks.names])
    families = []
    for f in sorted(set(fam)):
        m = fam == f
        families.append({"family": f, "images": int(m.sum()),
                         "positives": int(ks.labels[m].sum()),
                         "blocks": len(set(ks.blocks[m]))})

    group_counts = None
    if groups is not None:
        g = np.array([groups[x]["group"] for x in ks.names])
        if not np.array_equal(g == "defect", ks.labels):
            raise SystemExit(f"{KAGGLE_GROUPS.name} disagrees with the frozen "
                             f"image labels.")
        group_counts = {k: int((g == k).sum()) for k in (
            "defect", "bad_weld_without_defect", "good_weld_only")}
        broad = (g == "defect") | (g == "bad_weld_without_defect")
        bw = C.DATASETS[KAGGLE]["classes"].index("Bad Weld")

        def broad_score(e):
            if e["source"] == KAGGLE:
                return per_image(arrays[e["name"]], n,
                                 C.defect_ids(KAGGLE) | {bw}, "max")
            return frozen(e)
        summarise("label rule", "Bad Weld or Defect present", broad, ks,
                  broad_score, C.TRAIN_SOURCES)
        no_bw = g != "bad_weld_without_defect"
        sub = ks.subset(no_bw, f"{KAGGLE} without bad_weld_without_defect")
        summarise("label rule", "Defect against Good Weld only", sub.labels,
                  sub, lambda e: frozen(e)[no_bw], C.TRAIN_SOURCES)

    for rule, label in (("sum", "sum of the scores"),
                        ("count>=0.25", "count of predictions at or above 0.25"),
                        ("count>=0.10", "count of predictions at or above 0.10"),
                        ("mean_top3", "mean of the top three, zero where fewer")):
        summarise("score rule", label, ks.labels, ks,
                  lambda e, r=rule: per_image(
                      arrays[e["name"]], n, C.defect_ids(e["source"]), r),
                  C.TRAIN_SOURCES)

    lohi_classes = C.DATASETS["lohi_weld"]["classes"]
    for ci, cname in enumerate(lohi_classes):
        summarise("one class at a time", f"lohi_weld model, {cname} alone",
                  ks.labels, ks,
                  lambda e, c=ci: per_image(arrays[e["name"]], n, {c}, "max"),
                  ["lohi_weld"])
    # Which class supplies the strongest prediction on the defect-free images.
    lead = defaultdict(int)
    n_neg_scored = 0
    for e in cells:
        if e["source"] != "lohi_weld":
            continue
        a = arrays[e["name"]]
        m = np.isin(a["pred_class"], list(C.defect_ids("lohi_weld")))
        img, s, c = a["pred_image"][m], a["pred_score"][m], a["pred_class"][m]
        for i in np.flatnonzero(~ks.labels):
            sel = img == i
            if not sel.any():
                continue
            n_neg_scored += 1
            lead[lohi_classes[int(c[sel][np.argmax(s[sel])])]] += 1
    return {"rows": out_rows, "families": families, "groups": group_counts,
            "lead_class_on_negatives": dict(lead),
            "negatives_scored": n_neg_scored,
            "video_block": sorted(set(ks.blocks[~keep])),
            "video_split": nov.structure}


def kaggle_markdown(path: Path, checks: dict) -> None:
    rows = checks["rows"]

    def get(check, variant, source, family=None):
        return next((r for r in rows if r["check"] == check and
                     r["variant"] == variant and r["source"] == source and
                     r["family"] == family), None)

    def cell(r):
        if r is None:
            return "n/a"
        return f"{f3(r['mean_auroc'])} {interval(r['mean_auroc_lo'], r['mean_auroc_hi'])}"

    base = next(r for r in rows if r["check"] == "filename family")
    lines = [
        "# kaggle_v2 image-level checks. Exploratory, not pre-registered",
        "",
        "Every figure is a twelve-cell mean AUROC unless it says otherwise, "
        "with a 95 percent split-block interval, "
        f"{RESAMPLES:,} draws, the twelve cells averaged inside each draw. "
        "These checks were chosen after the data had been seen. They are "
        "leads, not results.",
        "",
        "## The filename carries the label",
        "",
        "kaggle_v2 aggregates several collections, and the collection "
        "predicts the label. A score that looks at no pixel, the positive "
        "share of the image's filename family among the other blocks, "
        f"reaches an AUROC of {cell(base)} on {base['n_pos']} positive and "
        f"{base['n_neg']} negative test images. The scores are computed once "
        "on the full split. The interval resamples blocks under them.",
        "",
        "| Filename family | Test images | Positive | Blocks |",
        "|---|---|---|---|"]
    for f in sorted(checks["families"], key=lambda x: (-x["images"],
                                                       x["family"])):
        if f["images"] >= 3:
            lines.append(f"| `{f['family']}` | {f['images']} "
                         f"| {f['positives']} | {f['blocks']} |")
    lines += ["", "Families with fewer than three test images are left out "
              "of this list and kept in the score.", ""]
    g = checks["groups"]
    v = checks["video_split"]
    lines += [
        "## The label rule, and the video block left out",
        "",
        (f"The split holds {g['defect']} images with a `Defect` box, "
         f"{g['bad_weld_without_defect']} with a `Bad Weld` box and no "
         f"`Defect` box, and {g['good_weld_only']} with neither. "
         if g else "The label groups file is missing, so only the frozen "
         "rule is shown. Run `python3 grid/image_level.py --kaggle-groups`. ")
        + "For \"Bad Weld or Defect present\" the in-domain score is the "
          "strongest `Bad Weld` or `Defect` prediction, which is what "
          "`KAGGLE_BAD_WELD_IS_DEFECT = True` would give. Foreign models keep "
          f"their frozen score. The video block, {checks['video_block'][0]}, "
          f"holds the `{VIDEO_FAMILY}` frames. Without it the split holds "
          f"{v['n_pos']} positive and {v['n_neg']} negative images in "
          f"{v['n_blocks']} blocks.",
        "",
        "| Trained on | Frozen rule | Bad Weld or Defect present "
        "| Defect against Good Weld only | Frozen rule, video block left out |",
        "|---|---|---|---|---|"]
    for src in C.TRAIN_SOURCES:
        who = f"{src}, in domain" if src == KAGGLE else src
        lines.append(
            f"| {who} | {cell(get('label rule', 'frozen: a Defect box is present', src))} "
            f"| {cell(get('label rule', 'Bad Weld or Defect present', src))} "
            f"| {cell(get('label rule', 'Defect against Good Weld only', src))} "
            f"| {cell(get('label rule', NO_VIDEO, src))} |")
    lines += ["", "## The score rule", "",
              "The frozen score is the strongest defect-class prediction. "
              "The alternatives below use the same predictions, before NMS, "
              "restricted to the model's own defect classes.", "",
              "| Score rule | Trained on | Over four | " + " | ".join(
                  C.FAMILIES) + " |",
              "|---|---|---|" + "---|" * len(C.FAMILIES)]
    for variant in ("sum of the scores", "count of predictions at or above 0.25",
                    "count of predictions at or above 0.10",
                    "mean of the top three, zero where fewer"):
        for src in C.TRAIN_SOURCES:
            lines.append(
                f"| {variant} | {src} | {cell(get('score rule', variant, src))} | "
                + " | ".join(cell(get("score rule", variant, src, f))
                             for f in C.FAMILIES) + " |")
    lead = checks["lead_class_on_negatives"]
    tot = sum(lead.values())
    lines += ["", "## One lohi_weld class at a time", "",
              "| Class | Over four | " + " | ".join(C.FAMILIES) + " |",
              "|---|---|" + "---|" * len(C.FAMILIES)]
    for cname in C.DATASETS["lohi_weld"]["classes"]:
        var = f"lohi_weld model, {cname} alone"
        lines.append(f"| {cname} | {cell(get('one class at a time', var, 'lohi_weld'))} | "
                     + " | ".join(cell(get("one class at a time", var,
                                           "lohi_weld", f))
                                  for f in C.FAMILIES) + " |")
    lines += ["", "Over the twelve lohi_weld cells, the class supplying the "
              "strongest prediction on a defect-free kaggle_v2 image: "
              + ", ".join(f"{k} {v} of {tot} ({100 * v / tot:.0f} percent)"
                          for k, v in sorted(lead.items(),
                                             key=lambda x: -x[1])) + "."]
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# arms: control, label cost, pooling
# --------------------------------------------------------------------------

ARM_PATTERNS = [("eval__lohictl__", "raw__lohictl__", "best"),
                ("eval__armd__", "raw__armd__", "best"),
                ("eval_last__armb__", "raw_last__armb__", "last"),
                ("eval__armb__", "raw__armb__", "best")]


def arm_evaluations(arm_dir: Path) -> list[dict]:
    """The arm records, named as analysis_arms reads them. The prefixes carry
    the arm name, so it is put back into the cell name here."""
    out = []
    for eval_prefix, raw_prefix, endpoint in ARM_PATTERNS:
        arm = eval_prefix.split("__")[1]
        for path in sorted(arm_dir.glob(f"{eval_prefix}*.json")):
            rec = json.loads(path.read_text())
            name = rec["name"]
            cell = rec["cell"]
            raw = arm_dir / f"{raw_prefix.split('__')[0]}__{name}.npz"
            for key, entry in rec["targets"].items():
                out.append({
                    "name": name, "result_key": key, "target": key,
                    "test_split": "audited", "arm": cell["arm"],
                    "source": cell["source"], "family": cell["family"],
                    "seed": cell.get("seed"), "k": cell.get("k"),
                    "draw": cell.get("draw"),
                    "endpoint": rec.get("endpoint", endpoint),
                    "raw": raw, "released": entry.get("image_level", {}),
                    "fabricated": False})
            assert arm in name
    return out


def run_arms(arm_dir: Path, main_dir: Path, out: Path) -> dict:
    evs = arm_evaluations(arm_dir)
    missing = sorted({e["raw"].name for e in evs if not e["raw"].is_file()})
    if missing:
        raise SystemExit(f"image level, arms: {len(missing)} raw archives "
                         f"missing, e.g. {missing[:3]}.")
    # The main-grid cells the arms are read against: zero-shot starts, the
    # targets' own models and the pool members. Scored on the same splits,
    # so on the same draws.
    main_evs, _ = evaluations(main_dir, [("eval__", "raw__", "best")])
    main_evs = [e for e in main_evs if e["arm"] in ("main", "reference")
                and e["test_split"] == "audited"]
    rows, draws, splits = score_all(evs + main_evs, "arms")
    arm_rows = [r for r in rows if r["arm"] in ("lohi_control", "armb", "armd")]
    main_rows = {main_key(r): r for r in rows
                 if r["arm"] in ("main", "reference")}
    main_draws = {main_key(r): draws[(r["name"], r["result_key"], "best")]
                  for r in main_rows.values()}
    # The arm splits that are main targets must be the main grid's splits.
    write_csv(out / "image_level_arms.csv", arm_rows)

    ctl = [r for r in arm_rows if r["arm"] == "lohi_control"]
    ctl_samples = [s for s in sample_table(ctl, {
        k: v for k, v in splits.items() if k in ("lohi_high", "lohi_low")})]
    write_csv(out / "image_level_samples_arms.csv", ctl_samples)
    samples_markdown(out / "image_level_samples_arms.md", ctl_samples,
                     "The internal control's two test splits. lohi_high's has "
                     "no negative image, so AUROC is undefined on it and no "
                     "score can be overconfident there. Every other split the "
                     "arms score is a main-grid test split, in "
                     "grid_results/analysis/tables/image_level_samples.md.")
    acq = ["lohi_high", "lohi_low"]
    pt = {main_key(r): r["auroc"] for r in ctl}
    dr = {main_key(r): draws[(r["name"], r["result_key"], r["endpoint"])]
          for r in ctl}
    ctl_margin = margin_rows(pt, dr, acq, ["lohi_low"], list(C.FAMILIES),
                             C.SEEDS, lambda t, f, sd: (t, f, sd, t))
    write_csv(out / "image_level_control_margin.csv", ctl_margin)

    pool = pooling_rows(arm_rows, main_rows)
    write_csv(out / "image_level_pooling.csv", pool)
    pool_seed = pooling_seed_rows(pool)
    write_csv(out / "image_level_pooling_seed_averaged.csv", pool_seed)
    pooling_markdown(out / "image_level_pooling.md", pool, pool_seed)

    cost = label_cost_rows(arm_rows, draws, main_rows, main_draws)
    write_csv(out / "image_level_label_cost.csv", cost)
    label_cost_markdown(out / "image_level_label_cost.md", cost, ctl_margin)
    print(f"  image level, arms: {len(arm_rows)} evaluations written to {out}")
    return {"rows": arm_rows, "pool": pool, "cost": cost,
            "control_margin": ctl_margin}


def pool_members(pool: str) -> list[str]:
    held = pool[len("pool_no_"):]
    return [s for s in C.TRAIN_SOURCES if s != held]


def pooling_rows(arm_rows, main_rows) -> list[dict]:
    out = []
    for r in arm_rows:
        if r["arm"] != "armd":
            continue
        mem = [main_rows[(m, r["family"], r["seed"], r["target"])]
               for m in pool_members(r["source"])]
        row = {"pool": r["source"], "held_out": r["source"][len("pool_no_"):],
               "target": r["target"], "family": r["family"],
               "seed": r["seed"], "meets_min_negatives": r["meets_min_negatives"],
               "members": " and ".join(pool_members(r["source"]))}
        for metric, key in (("auroc", "auroc"), ("image_ap", "image_ap")):
            p = r[key]
            ms = [m[key] for m in mem]
            row[f"pooled_{metric}"] = p
            row[f"member_a_{metric}"], row[f"member_b_{metric}"] = ms
            row[f"better_member_{metric}"] = max(ms)
            row[f"worse_member_{metric}"] = min(ms)
            row[f"members_mean_{metric}"] = float(np.mean(ms))
            row[f"beats_better_{metric}"] = p > max(ms)
            row[f"beats_mean_{metric}"] = p > float(np.mean(ms))
            row[f"beats_worse_{metric}"] = p > min(ms)
        out.append(row)
    return out


def pooling_seed_rows(pool: list[dict]) -> list[dict]:
    """One row per pool, target and configuration, every AUROC and AP first
    averaged over the three seeds, the pooled model's and each member's
    alike. The better and worse member are then chosen on those means."""
    groups = defaultdict(list)
    for r in pool:
        groups[(r["pool"], r["target"], r["family"])].append(r)
    out = []
    for (pk, t, fam), g in sorted(groups.items()):
        if sorted(r["seed"] for r in g) != list(C.SEEDS):
            raise SystemExit(f"pooling {pk} {t} {fam}: seeds "
                             f"{sorted(r['seed'] for r in g)}.")
        row = {"pool": pk, "held_out": g[0]["held_out"], "target": t,
               "family": fam, "n_seeds": len(g),
               "meets_min_negatives": g[0]["meets_min_negatives"]}
        for metric in ("auroc", "image_ap"):
            p = float(np.mean([r[f"pooled_{metric}"] for r in g]))
            a = float(np.mean([r[f"member_a_{metric}"] for r in g]))
            b = float(np.mean([r[f"member_b_{metric}"] for r in g]))
            row.update({f"pooled_{metric}": p,
                        f"better_member_{metric}": max(a, b),
                        f"worse_member_{metric}": min(a, b),
                        f"members_mean_{metric}": (a + b) / 2,
                        f"beats_better_{metric}": p > max(a, b),
                        f"beats_mean_{metric}": p > (a + b) / 2,
                        f"beats_worse_{metric}": p > min(a, b)})
        out.append(row)
    return out


def pooling_markdown(path: Path, pool: list[dict],
                     pool_seed: list[dict]) -> None:
    n = len(pool)
    lines = [
        "# Pooling, arm D, at image level",
        "",
        "Each pooled model against its two members, the main-grid models "
        "trained on each of the pool's sources alone, at the same "
        "configuration and seed, on the same test images. The better member "
        "is chosen per cell on the target's own test split, so it is an "
        "optimistic comparator. A pool also trains on more images than either "
        "member, so the training budget is not held fixed.",
        "",
        "| | In AUROC | In image-level AP |",
        "|---|---|---|"]
    for label, k in (("Mean over the cells, pooled", "pooled"),
                     ("Mean, better member", "better_member"),
                     ("Mean, members' mean", "members_mean"),
                     ("Mean, worse member", "worse_member")):
        lines.append(f"| {label} | {np.mean([p[f'{k}_auroc'] for p in pool]):.3f} "
                     f"| {np.mean([p[f'{k}_image_ap'] for p in pool]):.3f} |")
    for label, k in (("Pooled beats the better member", "beats_better"),
                     ("Pooled beats the members' mean", "beats_mean"),
                     ("Pooled beats the worse member", "beats_worse")):
        lines.append(f"| {label} | {sum(p[f'{k}_auroc'] for p in pool)} of {n} "
                     f"| {sum(p[f'{k}_image_ap'] for p in pool)} of {n} |")
    ns = len(pool_seed)
    for label, k in (("Seeds averaged first, beats the better member",
                      "beats_better"),
                     ("Seeds averaged first, beats the members' mean",
                      "beats_mean"),
                     ("Seeds averaged first, beats the worse member",
                      "beats_worse")):
        lines.append(f"| {label} | {sum(p[f'{k}_auroc'] for p in pool_seed)} "
                     f"of {ns} | {sum(p[f'{k}_image_ap'] for p in pool_seed)} "
                     f"of {ns} |")
    lines += ["", "Seeds averaged first: the pooled model's and each "
              "member's value are averaged over the three seeds, and the "
              "better member is chosen on those means. armd.csv instead "
              "takes the better member seed by seed, which is why its AP "
              "count of seed-averaged wins can differ by one or two."]
    lines += ["", "## By fold and target, mean over twelve cells", "",
              "| Pool | Target | Pooled AUROC | Better member | Worse member "
              "| Beats the better member | Negatives under the rule |",
              "|---|---|---|---|---|---|---|"]
    keys = sorted({(p["pool"], p["target"]) for p in pool})
    for pk, t in keys:
        g = [p for p in pool if p["pool"] == pk and p["target"] == t]
        lines.append(
            f"| {pk} | {t}{' (held out)' if t == g[0]['held_out'] else ''} "
            f"| {np.mean([p['pooled_auroc'] for p in g]):.3f} "
            f"| {np.mean([p['better_member_auroc'] for p in g]):.3f} "
            f"| {np.mean([p['worse_member_auroc'] for p in g]):.3f} "
            f"| {sum(p['beats_better_auroc'] for p in g)} of {len(g)} "
            f"| {'no' if g[0]['meets_min_negatives'] else 'yes'} |")
    path.write_text("\n".join(lines) + "\n")


def label_cost_rows(arm_rows, draws, main_rows, main_draws) -> list[dict]:
    """Arm B at image level. Draw d pairs with seed d, as arm B's recovery
    ratio does, the project log 6a. Primary endpoint, last.pt, 6a-ii."""
    b = [r for r in arm_rows if r["arm"] == "armb" and r["endpoint"] == "last"]
    out = []
    for (src, tgt, fam) in sorted({(r["source"], r["target"], r["family"])
                                   for r in b}):
        zero = [main_rows[(src, fam, sd, tgt)] for sd in C.SEEDS]
        own = [main_rows[(tgt, fam, sd, tgt)] for sd in C.SEEDS]
        dz = np.mean([main_draws[main_key(z)] for z in zero], axis=0)
        do = np.mean([main_draws[main_key(o)] for o in own], axis=0)
        pz = float(np.mean([z["auroc"] for z in zero]))
        po = float(np.mean([o["auroc"] for o in own]))
        base = {"source": src, "target": tgt, "family": fam,
                "meets_min_negatives": zero[0]["meets_min_negatives"],
                "zero_shot_auroc": pz, "in_domain_auroc": po}
        for tag, d in (("zero_shot_auroc", dz), ("in_domain_auroc", do)):
            base[f"{tag}_lo"], base[f"{tag}_hi"], _ = band(d)
        rz = margin(dz, do)
        base["zero_shot_margin"] = float(margin(pz, po)) \
            if np.isfinite(margin(pz, po)) else None
        base["zero_shot_margin_lo"], base["zero_shot_margin_hi"], _ = band(rz)
        for k in sorted({r["k"] for r in b}):
            cells = sorted((r for r in b if (r["source"], r["target"],
                                             r["family"], r["k"]) ==
                            (src, tgt, fam, k)), key=lambda r: r["draw"])
            if [c["draw"] for c in cells] != list(C.SEEDS):
                raise SystemExit(f"arm B {src} to {tgt} {fam} k={k}: draws "
                                 f"{[c['draw'] for c in cells]}.")
            dk = np.mean([draws[(c["name"], c["result_key"], "last")]
                          for c in cells], axis=0)
            pk = float(np.mean([c["auroc"] for c in cells]))
            lo, hi, nu = band(dk)
            rk = margin(dk, do)
            mlo, mhi, mnu = band(rk)
            pm = margin(pk, po)
            out.append({**base, "k_requested": k,
                        "draw_aurocs": " ".join(f"{c['auroc']:.4f}"
                                                for c in cells),
                        "draws_retaining_nothing": sum(
                            c["retains_no_prediction"] for c in cells),
                        "auroc": pk, "auroc_lo": lo, "auroc_hi": hi,
                        "auroc_undefined_draws": nu,
                        "retained_margin": float(pm) if np.isfinite(pm)
                        else None,
                        "retained_margin_lo": mlo, "retained_margin_hi": mhi,
                        "retained_margin_undefined_draws": mnu,
                        "resamples": len(dk),
                        "draw_intervals_wholly_above_half": sum(
                            c["auroc_interval_wholly_above_half"]
                            for c in cells)})
    return out


def label_cost_markdown(path: Path, cost: list[dict], ctl: list[dict]) -> None:
    ks = sorted({c["k_requested"] for c in cost})
    lines = [
        "# The label cost arm at image level",
        "",
        "AUROC on the target by requested budget k, primary endpoint "
        "`last.pt`, the three draws averaged inside each block draw. Draw d "
        "pairs with seed d of the zero-shot start and of the target's own "
        "model, as arm B's recovery ratio does. Requested k is a sampling "
        "instruction, the project log 6a-i: realised draws differ. Intervals are "
        f"95 percent split-block percentile intervals, {RESAMPLES:,} draws. "
        "The retained margin is (AUROC - 0.5) over (in-domain AUROC - 0.5). "
        "A pair whose target has fewer than "
        f"{C.MIN_NEGATIVES_FOR_FPR} negative images is marked, and its "
        "intervals are wide for that reason.",
        ""]
    for tgt in sorted({c["target"] for c in cost}):
        lines += [f"## Target {tgt}", "",
                  "| Pair | Configuration | Zero shot | " + " | ".join(
                      f"k = {k}" for k in ks) + " | In domain |",
                  "|---|---|---|" + "---|" * len(ks) + "---|"]
        for (src, fam) in sorted({(c["source"], c["family"]) for c in cost
                                  if c["target"] == tgt}):
            g = {c["k_requested"]: c for c in cost if c["target"] == tgt and
                 c["source"] == src and c["family"] == fam}
            z = next(iter(g.values()))
            flag = "" if z["meets_min_negatives"] else ", few negatives"
            lines.append(
                f"| {src} to {tgt}{flag} | {fam} "
                f"| {f3(z['zero_shot_auroc'])} "
                f"{interval(z['zero_shot_auroc_lo'], z['zero_shot_auroc_hi'])} | "
                + " | ".join(f"{f3(g[k]['auroc'])} "
                             f"{interval(g[k]['auroc_lo'], g[k]['auroc_hi'])}"
                             for k in ks)
                + f" | {f3(z['in_domain_auroc'])} "
                  f"{interval(z['in_domain_auroc_lo'], z['in_domain_auroc_hi'])} |")
        lines += ["", "Retained margin:", "",
                  "| Pair | Configuration | Zero shot | " + " | ".join(
                      f"k = {k}" for k in ks) + " |",
                  "|---|---|---|" + "---|" * len(ks)]
        for (src, fam) in sorted({(c["source"], c["family"]) for c in cost
                                  if c["target"] == tgt}):
            g = {c["k_requested"]: c for c in cost if c["target"] == tgt and
                 c["source"] == src and c["family"] == fam}
            z = next(iter(g.values()))
            lines.append(
                f"| {src} to {tgt} | {fam} "
                f"| {f3(z['zero_shot_margin'], True)} "
                f"{interval(z['zero_shot_margin_lo'], z['zero_shot_margin_hi'], True)} | "
                + " | ".join(
                    f"{f3(g[k]['retained_margin'], True)} "
                    f"{interval(g[k]['retained_margin_lo'], g[k]['retained_margin_hi'], True)}"
                    for k in ks) + " |")
        lines.append("")
    lines += ["## The internal control on lohi_low", "",
              "lohi_low's test split holds 11 negative images, so these are "
              "wide. lohi_high's holds none, and AUROC is undefined there.",
              "", "| Trained on | Level | AUROC on lohi_low, mean "
              "| Retained margin, with interval |", "|---|---|---|---|"]
    for m in ctl:
        if m["level"] == "seed":
            continue
        lines.append(f"| {m['source']} | {m['family'] or 'all four'} "
                     f"| {f3(m['auroc_ij'])} | {f3(m['retained_margin'], True)} "
                     f"{interval(m['retained_margin_lo'], m['retained_margin_hi'], True)} |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kaggle-groups", action="store_true",
                    help="write clean_candidate/kaggle_v2_test_image_groups."
                         "json from the audited label files, then stop")
    ap.add_argument("--main", action="store_true", help="main grid only")
    ap.add_argument("--arms", action="store_true", help="arms only")
    args = ap.parse_args()
    if args.kaggle_groups:
        write_kaggle_groups()
        return
    if not args.arms:
        run_main(C.PROJECT / "grid_results" / "results",
                 C.PROJECT / "grid_results" / "analysis" / "tables")
    if not args.main:
        run_arms(C.PROJECT / "grid_results_arms" / "results",
                 C.PROJECT / "grid_results" / "results",
                 C.PROJECT / "grid_results_arms" / "analysis")


if __name__ == "__main__":
    main()
