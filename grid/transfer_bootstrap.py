#!/usr/bin/env python3
"""Split-block bootstrap confidence intervals for the transfer ratio T_ij.

WHY THIS EXISTS, AND WHY FRIEDMAN DOES NOT COVER IT.

The pre-registered Friedman test asks one question: do the four configurations
differ from each other. It answered no. That is a statement about
configurations, and it is silent on the claim the paper actually makes, which
is that transfer collapses. "Collapses" is a claim about the magnitude of
T_ij, so it needs its own interval, estimated on the only source of
uncertainty that matters for it: the finite, correlated test sample.

THE PAIRING IS THE WHOLE POINT.

T_ij = P_ij / P_jj. Both terms are measured on the SAME test images of target
j. A bootstrap that resampled them independently would ignore that and give a
much wider interval than the truth. Every draw here samples split blocks of
target j once, then scores the foreign model and the in-domain reference on
exactly those images. The diagonal is included as a self-check: when i == j
the two models are the same, so every draw must return exactly 1.0.

THE UNIT IS THE SPLIT BLOCK, NOT THE IMAGE.

Images from one weld scene are correlated, and adjacent video frames are
nearly the same photograph. Resampling images independently would treat those
as independent evidence and understate the interval. The frozen split block is
the coarsest leakage-safe unit and is what the released splits are built on.

BOTH SCALES ARE REPORTED.

Raw AP inherits the target base rate in numerator and denominator alike, which
compresses the ratio towards 1 on the four unbalanced columns. The base-rate
corrected ratio divides lift by lift. The base rate is RECOMPUTED INSIDE EVERY
DRAW, because resampling blocks changes the positive share of the sample and
holding it fixed at the full-sample value would understate the spread.

Usage:
    python3 grid/transfer_bootstrap.py                 # both levels, cached
    python3 grid/transfer_bootstrap.py --level image
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C  # noqa: E402
import metrics as M  # noqa: E402
from evaluate_cell import load_ground_truth  # noqa: E402

ALPHA = 0.05


# --------------------------------------------------------------------------
# point metrics, matching the evaluator exactly
# --------------------------------------------------------------------------

def image_ap(scores: np.ndarray, labels: np.ndarray) -> float:
    """Positive-class AP, identical to metrics.binary_metrics's `ap` branch.

    Reimplemented rather than reused because binary_metrics also computes
    AUROC, ECE and reliability bins, and paying for those 2,000 times per cell
    would dominate the runtime. The recall and precision construction below is
    copied line for line so the point estimate reproduces the evaluator.
    """
    n_pos = int(labels.sum())
    if n_pos == 0 or len(labels) == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    y = labels[order].astype(np.float64)
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    recall = tp / n_pos
    precision = tp / np.maximum(tp + fp, 1e-9)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    return float(M._interp_precision(recall, precision))


def harmonized_boxes(raw, prefix: str, source: str):
    """Predictions as the box-level metric sees them: defect classes only,
    then class-agnostic NMS. The raw sidecar stores the backend's output
    before harmonization, so this reproduces evaluate_cell.flatten(collapse).
    """
    keep = list(C.defect_ids(source))
    pi = raw[prefix + "pred_image"]
    pb = raw[prefix + "pred_box"]
    ps = raw[prefix + "pred_score"]
    pc = raw[prefix + "pred_class"]
    take = np.isin(pc, keep)
    pi, pb, ps = pi[take], pb[take], ps[take]

    by_image: dict[int, tuple] = {}
    for image_id in np.unique(pi):
        m = pi == image_id
        b, s = pb[m], ps[m]
        k = M.class_agnostic_nms(b, s)
        by_image[int(image_id)] = (b[k], s[k])
    return by_image


def precompute_box_tp(by_image: dict, gt_defect: dict,
                      iou_thr: float = 0.5) -> dict:
    """Per image: descending scores, their true-positive flags, and n_gt.

    THIS IS EXACT, NOT AN APPROXIMATION, and it is what makes the box-level
    bootstrap affordable. metrics._match assigns greedily in descending score
    order, but a prediction can only match ground truth in its OWN image and
    the `used` mask is per image. So whether a prediction is a true positive
    depends on the score order WITHIN its image and on nothing outside it.
    Matching can therefore be done once, and a resample becomes a
    concatenate-and-sort instead of a full rematch. Measured on lohi_weld this
    turns 1.24 s per draw into about 1 ms, which is the difference between 16
    hours and 2 minutes.

    reproduces_average_precision() below asserts the equivalence on real data
    rather than trusting the argument.
    """
    empty_b = np.zeros((0, 4), dtype=np.float32)
    empty_s = np.zeros(0, dtype=np.float32)
    per = {}
    for image_id, gts in gt_defect.items():
        b, s = by_image.get(int(image_id), (empty_b, empty_s))
        if len(s) == 0:
            per[int(image_id)] = (empty_s, np.zeros(0, bool), len(gts))
            continue
        order = np.argsort(-s)
        b, s = b[order], s[order]
        tp = np.zeros(len(s), dtype=bool)
        if len(gts):
            used = np.zeros(len(gts), dtype=bool)
            for k in range(len(s)):
                ious = M.iou_matrix(b[k][None, :], gts)[0]
                ious[used] = -1
                best = int(ious.argmax())
                if ious[best] >= iou_thr:
                    tp[k] = True
                    used[best] = True
        per[int(image_id)] = (s, tp, len(gts))
    return per


def box_ap_from_tp(per: dict, index: np.ndarray) -> float:
    """Class-agnostic AP50 over a resampled multiset of original image ids."""
    scores, flags, n_gt = [], [], 0
    for old_id in index:
        s, tp, g = per[int(old_id)]
        n_gt += g
        if len(s):
            scores.append(s)
            flags.append(tp)
    if n_gt == 0:
        return float("nan")
    if not scores:
        return 0.0
    s = np.concatenate(scores)
    tp = np.concatenate(flags)
    order = np.argsort(-s, kind="stable")
    tp = tp[order]
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(~tp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    return float(M._interp_precision(recall, precision))


def fast_cluster_bootstrap(pred_img, pred_box, pred_score, gt_by_image,
                           cluster_ids, resamples: int, seed: int = 0,
                           alpha: float = ALPHA) -> dict:
    """Drop-in replacement for metrics.bootstrap_detection_ap_by_cluster.

    WHY IT IS NOT IN metrics.py. Every cell.json pins the sha256 of config.py,
    evaluate_cell.py, metrics.py and yolo_dataset.py, and evaluate() refuses to
    score a checkpoint whose code has moved. Editing metrics.py would make all
    69 completed cells un-re-evaluable, which is the guard working as designed.
    The project log 5h records what that costs. So the fast path lives here and
    metrics.py stays byte identical.

    Same result, same draws, same seed. The speedup is the per-image TP
    precompute described in precompute_box_tp, applied at all ten COCO IoU
    thresholds. equivalent_to_metrics() checks it against the real function on
    real predictions instead of asserting it.
    """
    cluster_ids = np.asarray(cluster_ids, dtype=object)
    n_images = len(cluster_ids)
    if n_images == 0:
        return {"unit": "split_block", "n_clusters": 0, "resamples": resamples,
                "map50": None, "map50_95": None}

    groups: dict[str, list[int]] = {}
    for image_id, cluster in enumerate(cluster_ids):
        groups.setdefault(str(cluster), []).append(image_id)
    keys = sorted(groups)

    by_image = {}
    for image_id in range(n_images):
        take = np.nonzero(pred_img == image_id)[0]
        by_image[image_id] = (pred_box[take], pred_score[take])
    gt_full = {i: gt_by_image.get(i, np.zeros((0, 4), np.float32))
               for i in range(n_images)}

    # One TP table per IoU threshold, computed once.
    tables = [precompute_box_tp(by_image, gt_full, float(t))
              for t in M.COCO_IOUS]
    full = np.arange(n_images)
    point50 = box_ap_from_tp(tables[0], full)
    point5095 = float(np.nanmean([box_ap_from_tp(t, full) for t in tables]))

    # The draw sequence must match metrics.py exactly, or the intervals are
    # from a different bootstrap and the comparison below is meaningless.
    rng = np.random.default_rng(seed)
    members = {k: np.array(groups[k], dtype=int) for k in keys}
    draws50, draws5095 = [], []
    for _ in range(resamples):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        index = np.concatenate([members[str(c)] for c in sampled])
        draws50.append(box_ap_from_tp(tables[0], index))
        draws5095.append(float(np.nanmean(
            [box_ap_from_tp(t, index) for t in tables])))

    def interval(values, point_value):
        values = np.asarray(values, dtype=np.float64)
        return {"point": float(point_value),
                "lo": float(np.nanpercentile(values, 100 * alpha / 2)),
                "hi": float(np.nanpercentile(values, 100 * (1 - alpha / 2)))}

    return {"unit": "split_block", "n_clusters": len(keys),
            "resamples": resamples,
            "map50": interval(draws50, point50),
            "map50_95": interval(draws5095, point5095)}


def equivalent_to_metrics(pred_img, pred_box, pred_score, gt_by_image,
                          cluster_ids, resamples: int = 25, seed: int = 0,
                          tol: float = 1e-9) -> bool:
    """Run both implementations on the same input and require equality."""
    fast = fast_cluster_bootstrap(pred_img, pred_box, pred_score, gt_by_image,
                                  cluster_ids, resamples, seed)
    slow = M.bootstrap_detection_ap_by_cluster(
        pred_img, pred_box, pred_score, gt_by_image, cluster_ids,
        resamples, seed)
    for key in ("map50", "map50_95"):
        for field in ("point", "lo", "hi"):
            a, b = fast[key][field], slow[key][field]
            if not (np.isnan(a) and np.isnan(b)) and abs(a - b) > tol:
                raise SystemExit(
                    f"fast cluster bootstrap disagrees on {key}.{field}: "
                    f"{a!r} against {b!r}")
    return True


# Largest disagreement seen between the fast and slow AP paths, across every
# cell checked. Printed at the end of a run so drift would be visible rather
# than silently absorbed by the tolerance.
_MAX_AP_DISCREPANCY = 0.0


def reproduces_average_precision(per: dict, by_image: dict, gt_defect: dict,
                                 tol: float = 1e-4) -> bool:
    """Assert the fast path equals metrics.average_precision on the full set.

    WHY THE TOLERANCE IS NOT EXACT, AND WHY IT IS 1e-4.

    AP is not uniquely defined when two predictions carry the same score,
    because their order on the precision-recall curve is then arbitrary.
    metrics.average_precision sorts all predictions globally with an unstable
    quicksort. This path sorts within each image first and merges stably, so
    tied scores from different images can land in a different order. Measured
    on real predictions, 1.5 percent of scores are tied, in groups of up to
    three. Both orderings are legitimate readings of the same metric.

    The disagreement grows with the number of predictions, because more
    predictions means more ties. Observed: about 1e-8 on kaggle_v2 with 3,881
    predictions, and 1.1e-6 on lohi_weld with Faster R-CNN, where both models
    sit at the 300 detections per image cap across 453 images and so carry
    135,900 predictions.

    **The threshold is set by reporting precision, not by float precision.**
    Nothing in this study is quoted beyond four decimal places, so a
    discrepancy that cannot reach the fourth decimal cannot change a reported
    number. 1e-4 is that bound, and it still sits two orders of magnitude
    above the worst case actually observed, so real drift would trip it. An
    earlier 1e-6 was picked from float intuition rather than from what tie
    ambiguity can produce, and it failed a full run twice.
    """
    global _MAX_AP_DISCREPANCY
    index = np.array(sorted(gt_defect), dtype=int)
    fast = box_ap_from_tp(per, index)
    img, box, score, gt = [], [], [], {}
    empty = np.zeros((0, 4), dtype=np.float32)
    for new_id, old_id in enumerate(index):
        b, s = by_image.get(int(old_id), (empty, np.zeros(0, np.float32)))
        if len(s):
            img.append(np.full(len(s), new_id, dtype=int))
            box.append(b)
            score.append(s)
        gt[new_id] = gt_defect.get(int(old_id), empty)
    slow = float(M.average_precision(
        np.concatenate(img) if img else np.zeros(0, int),
        np.concatenate(box) if box else empty,
        np.concatenate(score) if score else np.zeros(0, np.float32),
        gt)["ap"]) if gt else float("nan")
    if not (np.isnan(fast) or np.isnan(slow)):
        gap = abs(fast - slow)
        _MAX_AP_DISCREPANCY = max(_MAX_AP_DISCREPANCY, gap)
        if gap > tol:
            raise SystemExit(
                f"fast box AP path disagrees with metrics.average_precision "
                f"by {gap:.3e}, over the {tol:.0e} tolerance: "
                f"{fast!r} against {slow!r}. This is larger than tie ordering "
                f"can explain. Do not raise the tolerance, find the cause.")
    return True


# --------------------------------------------------------------------------
# cell addressing
# --------------------------------------------------------------------------

def numerator_name(source: str, family: str, seed: int) -> str:
    return C.cell_name({"source": source, "family": family, "seed": seed,
                        "kind": "main"})


def denominator_name(target: str, family: str, seed: int) -> str:
    """P_jj comes from the main diagonal for a training source, and from the
    auxiliary reference cell for a test-only target. Same configuration and
    same seed in both cases, which is why the references run all four
    configurations rather than one.
    """
    kind = "main" if target in C.TRAIN_SOURCES else "reference"
    return C.cell_name({"source": target, "family": family, "seed": seed,
                        "kind": kind})


def open_raw(results_dir: Path, name: str):
    path = results_dir / f"raw__{name}.npz"
    if not path.is_file():
        return None
    return np.load(path, allow_pickle=True)


# --------------------------------------------------------------------------
# the bootstrap
# --------------------------------------------------------------------------

def target_seed(seed: int, target: str, salt: int = 0) -> int:
    """Deterministic per-target seed offset.

    NEVER USE THE BUILT-IN hash() HERE. Python randomises string hashing per
    process unless PYTHONHASHSEED is fixed, so an earlier version of this file
    produced a DIFFERENT bootstrap on every run: the same cell gave
    [0.575, 0.912] once and [0.576, 0.899] the next time. Confidence intervals
    that move between runs cannot be released, and C5 of this project is
    precisely a released, reproducible artefact. sha256 of the target name is
    stable across processes, machines and Python versions.
    """
    digest = hashlib.sha256(f"{salt}:{target}".encode()).digest()
    return seed + int.from_bytes(digest[:4], "big") % 100000


def block_draws(blocks: np.ndarray, resamples: int, seed: int):
    """Resampled image-index arrays, one per draw, sampling blocks with
    replacement. Every image of a sampled block travels with it.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for image_id, block in enumerate(blocks):
        groups[str(block)].append(image_id)
    keys = sorted(groups)
    members = [np.array(groups[k], dtype=int) for k in keys]
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(keys), size=(resamples, len(keys)))
    for row in picks:
        yield np.concatenate([members[i] for i in row])


def percentile_band(values: list[float], point: float) -> dict:
    finite = np.array([v for v in values if np.isfinite(v)], dtype=float)
    n_undef = len(values) - len(finite)
    if len(finite) < 2:
        return {"point": point, "lo": None, "hi": None,
                "n_undefined_draws": n_undef}
    return {"point": point,
            "lo": float(np.percentile(finite, 100 * ALPHA / 2)),
            "hi": float(np.percentile(finite, 100 * (1 - ALPHA / 2))),
            "n_undefined_draws": n_undef}


def image_level_rows(results_dir: Path, resamples: int, seed: int) -> list[dict]:
    out = []
    for target in C.ALL_TARGETS:
        # The block axis is a property of the target's frozen test split, so it
        # is read once and shared by every numerator and denominator on it.
        probe = open_raw(results_dir, numerator_name(
            C.TRAIN_SOURCES[0], "yolo11s", C.SEEDS[0]))
        if probe is None:
            continue
        blocks = probe[target + "__split_block"]
        labels_ref = probe[target + "__image_label"].astype(bool)
        names_ref = probe[target + "__image_name"]
        probe.close()

        draws = list(block_draws(blocks, resamples, target_seed(seed, target)))

        for family in C.FAMILIES:
            for source in C.TRAIN_SOURCES:
                num_s, den_s = [], []
                ok = True
                for s in C.SEEDS:
                    rn = open_raw(results_dir, numerator_name(source, family, s))
                    rd = open_raw(results_dir, denominator_name(target, family, s))
                    if rn is None or rd is None:
                        ok = False
                        break
                    # The two archives must describe the same test images in
                    # the same order, or the pairing is meaningless.
                    if not (np.array_equal(rn[target + "__image_name"], names_ref)
                            and np.array_equal(rd[target + "__image_name"], names_ref)):
                        raise SystemExit(
                            f"test image axis disagrees for {target}, "
                            f"{source}->{family} seed {s}. Refusing to pair.")
                    num_s.append(rn[target + "__image_score"].astype(np.float64))
                    den_s.append(rd[target + "__image_score"].astype(np.float64))
                    rn.close()
                    rd.close()
                if not ok:
                    continue

                full = np.arange(len(labels_ref))
                p_ij = float(np.mean([image_ap(x[full], labels_ref) for x in num_s]))
                p_jj = float(np.mean([image_ap(x[full], labels_ref) for x in den_s]))
                base = float(labels_ref.mean())
                t_raw = p_ij / p_jj if p_jj > 0 else float("nan")
                t_lift = ((p_ij - base) / (p_jj - base)
                          if (p_jj - base) > 0 else float("nan"))

                raw_draws, lift_draws, lift_num_draws = [], [], []
                for idx in draws:
                    y = labels_ref[idx]
                    br = float(y.mean())
                    a = np.mean([image_ap(x[idx], y) for x in num_s])
                    b = np.mean([image_ap(x[idx], y) for x in den_s])
                    raw_draws.append(a / b if b > 0 else float("nan"))
                    lift_draws.append((a - br) / (b - br)
                                      if (b - br) > 0 else float("nan"))
                    lift_num_draws.append(a - br)

                row = {
                    "level": "image", "source": source, "target": target,
                    "family": family,
                    "P_ij": p_ij, "P_jj": p_jj, "target_base_rate": base,
                    "P_ij_above_base": p_ij - base,
                    "P_jj_above_base": p_jj - base,
                    "resamples": resamples, "n_blocks": len(set(map(str, blocks))),
                    "n_test_images": len(labels_ref),
                }
                for label, band in (
                        ("transfer_ratio", percentile_band(raw_draws, t_raw)),
                        ("transfer_ratio_above_base",
                         percentile_band(lift_draws, t_lift)),
                        ("P_ij_above_base_ci",
                         percentile_band(lift_num_draws, p_ij - base))):
                    row[label] = band["point"]
                    row[label + "_lo"] = band["lo"]
                    row[label + "_hi"] = band["hi"]
                    row[label + "_undefined_draws"] = band["n_undefined_draws"]
                row["denominator_is_weak_by_construction"] = target == "twoclass"
                out.append(row)
                print(f"  image  {source:>16s} -> {target:<16s} {family:<12s} "
                      f"T={row['transfer_ratio']:.3f} "
                      f"[{row['transfer_ratio_lo']:.3f}, {row['transfer_ratio_hi']:.3f}]  "
                      f"T_lift={row['transfer_ratio_above_base']:.3f}",
                      flush=True)
    return out


def box_level_rows(results_dir: Path, resamples: int, seed: int) -> list[dict]:
    out = []
    for target in C.BOX_TARGETS:
        gt = load_ground_truth(target)
        blocks = gt["split_blocks"]
        draws = list(block_draws(blocks, resamples,
                                 target_seed(seed, target, salt=7919)))
        full = np.arange(len(gt["images"]))

        for family in C.FAMILIES:
            for source in C.BOX_SOURCES:
                num_s, den_s = [], []
                ok = True
                for s in C.SEEDS:
                    rn = open_raw(results_dir, numerator_name(source, family, s))
                    rd = open_raw(results_dir, denominator_name(target, family, s))
                    if rn is None or rd is None:
                        ok = False
                        break
                    hn = harmonized_boxes(rn, target + "__", source)
                    hd = harmonized_boxes(rd, target + "__", target)
                    gtd = gt["gt_defect"]
                    pn = precompute_box_tp(hn, gtd)
                    pd = precompute_box_tp(hd, gtd)
                    # Checked once per cell against the real metric, on real
                    # predictions. An equivalence argument in a docstring is
                    # not evidence.
                    reproduces_average_precision(pn, hn, gtd)
                    reproduces_average_precision(pd, hd, gtd)
                    num_s.append(pn)
                    den_s.append(pd)
                    rn.close()
                    rd.close()
                if not ok:
                    continue

                p_ij = float(np.mean([box_ap_from_tp(x, full) for x in num_s]))
                p_jj = float(np.mean([box_ap_from_tp(x, full) for x in den_s]))
                t_raw = p_ij / p_jj if p_jj > 0 else float("nan")

                draws_t = []
                for idx in draws:
                    a = np.mean([box_ap_from_tp(x, idx) for x in num_s])
                    b = np.mean([box_ap_from_tp(x, idx) for x in den_s])
                    draws_t.append(a / b if b > 0 else float("nan"))
                band = percentile_band(draws_t, t_raw)

                out.append({
                    "level": "box", "source": source, "target": target,
                    "family": family, "P_ij": p_ij, "P_jj": p_jj,
                    "transfer_ratio": band["point"],
                    "transfer_ratio_lo": band["lo"],
                    "transfer_ratio_hi": band["hi"],
                    "transfer_ratio_undefined_draws": band["n_undefined_draws"],
                    "resamples": resamples,
                    "n_blocks": len(set(map(str, blocks))),
                    "n_test_images": len(full),
                    "denominator_is_weak_by_construction": target == "twoclass",
                    # Box AP has no prevalence floor, so there is no base-rate
                    # correction to make here. The image level is the only one
                    # that needs it.
                })
                print(f"  box    {source:>16s} -> {target:<16s} {family:<12s} "
                      f"T={band['point']:.4f} "
                      f"[{band['lo']:.4f}, {band['hi']:.4f}]", flush=True)
    return out


# THE TWO LEVELS USE DIFFERENT RESAMPLE COUNTS, AND THIS IS A COST DECISION
# THAT MUST BE REPORTED, NOT HIDDEN.
#
# An image-level draw scores 450 numbers and costs microseconds. A box-level
# draw on lohi_weld scores up to 135,900 predictions, because both models sit
# at the 300 detections per image cap across 453 images, and the in-domain
# LoHi cells were measured at roughly 15 minutes each for 2,000 draws. Eight
# such cells is two hours for an interval whose width barely moves.
#
# So the image level, which carries the collapse claim, keeps the full 2,000
# draws from config.py. The box level uses 500. Monte Carlo error on a 2.5th
# percentile at 500 draws is small next to the sampling spread being measured,
# and every row records its own `resamples` value so a reader can see which
# number came from which.
BOX_RESAMPLES = 500


def compute(results_dir: Path, resamples: int, seed: int,
            level: str = "both", box_resamples: int | None = None) -> list[dict]:
    box_n = BOX_RESAMPLES if box_resamples is None else box_resamples
    rows = []
    if level in ("image", "both"):
        print(f"image-level transfer ratios, {resamples} draws")
        rows += image_level_rows(results_dir, resamples, seed)
    if level in ("box", "both"):
        print(f"box-level transfer ratios, {box_n} draws")
        rows += box_level_rows(results_dir, box_n, seed)
        print(f"largest fast-against-slow AP discrepancy over every cell "
              f"checked: {_MAX_AP_DISCREPANCY:.3e}")
    return rows


def cached(results_dir: Path, resamples: int, seed: int,
           level: str = "both", box_resamples: int | None = None) -> list[dict]:
    """Same caching contract as the thin-class bootstrap: keyed by resample
    counts and seed, so changing any of them invalidates it.
    """
    box_n = BOX_RESAMPLES if box_resamples is None else box_resamples
    path = results_dir.parent / "transfer_bootstrap_cache.json"
    key = f"{resamples}:{box_n}:{seed}:{level}"
    if path.is_file():
        blob = json.loads(path.read_text())
        if blob.get("key") == key:
            return blob["rows"]
    t0 = time.time()
    rows = compute(results_dir, resamples, seed, level, box_n)
    path.write_text(json.dumps(
        {"key": key, "elapsed_seconds": round(time.time() - t0, 1),
         "image_resamples": resamples, "box_resamples": box_n,
         "rows": rows}, indent=1))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path,
                    default=C.PROJECT / "grid_results" / "results")
    ap.add_argument("--resamples", type=int, default=C.BOOTSTRAP_RESAMPLES)
    ap.add_argument("--seed", type=int, default=C.BOOTSTRAP_SEED)
    ap.add_argument("--level", choices=["image", "box", "both"], default="both")
    ap.add_argument("--box-resamples", type=int, default=BOX_RESAMPLES,
                    help="box level is far more expensive per draw, see "
                         "BOX_RESAMPLES")
    args = ap.parse_args()
    rows = cached(args.results_dir, args.resamples, args.seed, args.level,
                  args.box_resamples)
    print(f"\n{len(rows)} transfer ratios with split-block intervals")


if __name__ == "__main__":
    main()
