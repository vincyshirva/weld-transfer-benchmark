#!/usr/bin/env python3
"""
Detection and image-level metrics for the transfer matrix.

Written out rather than pulled from a library because the box level matrix is
class agnostic, which is not what an off-the-shelf mAP implementation assumes.
Getting the collapse wrong would silently change every transfer number, so the
semantics are explicit here and unit tested in test_metrics.py.

Conventions:
  boxes    absolute xyxy, in whatever frame the caller used, consistently for
           predictions and ground truth
  scores   confidence, higher is better
  image_id groups predictions and ground truth. Matching never crosses images.
"""

from __future__ import annotations

import numpy as np

COCO_IOUS = np.arange(0.5, 1.0, 0.05)     # 0.50:0.05:0.95
RECALL_POINTS = np.linspace(0, 1, 101)    # COCO 101-point interpolation


def _interp_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """Mean precision at the 101 COCO recall points.

    Deliberately not np.interp. Recall repeats whenever a false positive
    follows the last true positive, and np.interp resolves a duplicate x by
    taking the LAST matching y, which is the smallest precision on that
    plateau. That silently understated every AP. searchsorted with side="left"
    takes the first index whose recall reaches the point, which is the COCO
    definition: precision is the maximum over all recalls at or above r.
    """
    if len(recall) == 0:
        return 0.0
    idx = np.searchsorted(recall, RECALL_POINTS, side="left")
    inside = idx < len(precision)
    out = np.zeros_like(RECALL_POINTS, dtype=np.float64)
    out[inside] = precision[idx[inside]]
    return float(out.mean())


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU. a is (N,4), b is (M,4), result is (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.prod(np.clip(a[:, 2:] - a[:, :2], 0, None), axis=1)
    area_b = np.prod(np.clip(b[:, 2:] - b[:, :2], 0, None), axis=1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0).astype(np.float32)


def class_agnostic_nms(boxes: np.ndarray, scores: np.ndarray,
                       iou_thr: float = 0.7) -> np.ndarray:
    """Indices to keep, ignoring class.

    Needed after collapsing classes. A model trained on lohi_weld may place a
    `pore` box and a `stain` box on one object. Per-class NMS keeps both, and
    once the classes collapse to a single `defect` the second becomes a false
    positive the model never actually made.

    Where this bites is narrower than it first looks, and the tests pin it
    down. Under the COCO precision envelope a duplicate ranked below the true
    positive does not change AP at all. It does change the false positive
    count directly, so it moves recall at a fixed FP budget and raw precision.
    Since the FP budget metric is the one an inspection engineer cares about,
    the collapse still has to be done properly.
    """
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.int64)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = iou_matrix(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thr]
    return np.array(keep, dtype=np.int64)


def _match(pred_img, pred_box, pred_score, gt_by_image, iou_thr):
    """Greedy highest-score-first matching. Returns tp flags and gt total."""
    order = np.argsort(-pred_score)
    tp = np.zeros(len(order), dtype=bool)
    used = {k: np.zeros(len(v), dtype=bool) for k, v in gt_by_image.items()}
    for rank, idx in enumerate(order):
        img = pred_img[idx]
        gts = gt_by_image.get(img)
        if gts is None or len(gts) == 0:
            continue
        ious = iou_matrix(pred_box[idx][None, :], gts)[0]
        ious[used[img]] = -1              # a ground truth box matches once
        best = int(ious.argmax())
        if ious[best] >= iou_thr:
            tp[rank] = True
            used[img][best] = True
    return tp, sum(len(v) for v in gt_by_image.values())


def average_precision(pred_img, pred_box, pred_score, gt_by_image,
                      iou_thr: float = 0.5) -> dict:
    """COCO-style AP at one IoU threshold, with 101-point interpolation."""
    n_gt = sum(len(v) for v in gt_by_image.values())
    if n_gt == 0:
        return {"ap": float("nan"), "n_gt": 0, "n_pred": len(pred_score),
                "precision": float("nan"), "recall": float("nan")}
    if len(pred_score) == 0:
        return {"ap": 0.0, "n_gt": n_gt, "n_pred": 0,
                "precision": float("nan"), "recall": 0.0}

    tp, _ = _match(pred_img, pred_box, pred_score, gt_by_image, iou_thr)
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(~tp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    # Make precision monotonically decreasing, then sample at 101 recalls.
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    ap = _interp_precision(recall, precision)
    return {"ap": ap, "n_gt": int(n_gt), "n_pred": int(len(pred_score)),
            "precision": float(precision[-1]), "recall": float(recall[-1])}


def average_precision_range(pred_img, pred_box, pred_score, gt_by_image) -> dict:
    """AP averaged over IoU 0.50:0.05:0.95, plus the 0.50 value."""
    aps = [average_precision(pred_img, pred_box, pred_score, gt_by_image, t)["ap"]
           for t in COCO_IOUS]
    at50 = average_precision(pred_img, pred_box, pred_score, gt_by_image, 0.5)
    return {"map50": at50["ap"], "map50_95": float(np.nanmean(aps)),
            "n_gt": at50["n_gt"], "n_pred": at50["n_pred"]}


def bootstrap_detection_ap_by_cluster(pred_img, pred_box, pred_score,
                                      gt_by_image, cluster_ids, resamples: int,
                                      seed: int = 0,
                                      alpha: float = 0.05) -> dict:
    """Cluster bootstrap for AP, preserving all images from one split block.

    Images from one weld scene are correlated. Resampling boxes or images as
    if they were independent would make the interval too narrow. Each draw
    therefore samples split blocks with replacement, then recreates every
    image and prediction belonging to the sampled block under fresh image
    identifiers.
    """
    cluster_ids = np.asarray(cluster_ids, dtype=object)
    n_images = len(cluster_ids)
    if n_images == 0:
        return {"unit": "split_block", "n_clusters": 0,
                "resamples": resamples, "map50": None, "map50_95": None}

    groups = {}
    for image_id, cluster in enumerate(cluster_ids):
        groups.setdefault(str(cluster), []).append(image_id)
    keys = sorted(groups)

    pred_by_image = {}
    for image_id in range(n_images):
        take = np.nonzero(pred_img == image_id)[0]
        pred_by_image[image_id] = (pred_box[take], pred_score[take])

    point = average_precision_range(pred_img, pred_box, pred_score, gt_by_image)
    rng = np.random.default_rng(seed)
    draws50, draws5095 = [], []
    empty_box = np.zeros((0, 4), dtype=np.float32)
    empty_score = np.zeros(0, dtype=np.float32)

    for _ in range(resamples):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        out_img, out_box, out_score, out_gt = [], [], [], {}
        new_image_id = 0
        for cluster in sampled:
            for old_image_id in groups[str(cluster)]:
                boxes, scores = pred_by_image[old_image_id]
                if len(scores):
                    out_img.append(np.full(len(scores), new_image_id, dtype=int))
                    out_box.append(boxes)
                    out_score.append(scores)
                out_gt[new_image_id] = gt_by_image.get(old_image_id, empty_box)
                new_image_id += 1

        pi = np.concatenate(out_img) if out_img else np.zeros(0, dtype=int)
        pb = np.concatenate(out_box) if out_box else empty_box
        ps = np.concatenate(out_score) if out_score else empty_score
        value = average_precision_range(pi, pb, ps, out_gt)
        draws50.append(value["map50"])
        draws5095.append(value["map50_95"])

    def interval(values, point_value):
        values = np.asarray(values, dtype=np.float64)
        return {
            "point": float(point_value),
            "lo": float(np.nanpercentile(values, 100 * alpha / 2)),
            "hi": float(np.nanpercentile(values, 100 * (1 - alpha / 2))),
        }

    return {
        "unit": "split_block",
        "n_clusters": len(keys),
        "resamples": resamples,
        "map50": interval(draws50, point["map50"]),
        "map50_95": interval(draws5095, point["map50_95"]),
    }


def recall_at_fp_budget(pred_img, pred_box, pred_score, gt_by_image,
                        n_images: int, budgets, iou_thr: float = 0.5) -> dict:
    """Recall when allowed at most B false positive boxes per image.

    Inspection requirements are written this way and no weld detection paper
    reports it. Unlike an image-level false alarm rate, this stays measurable
    when a target has almost no negative images, which is the situation on
    lohi_weld, visual and twoclass.
    """
    n_gt = sum(len(v) for v in gt_by_image.values())
    out = {}
    if n_gt == 0 or n_images == 0:
        return {f"recall_at_{b}_fp_per_image": float("nan") for b in budgets}
    if len(pred_score) == 0:
        return {f"recall_at_{b}_fp_per_image": 0.0 for b in budgets}
    tp, _ = _match(pred_img, pred_box, pred_score, gt_by_image, iou_thr)
    tp_cum, fp_cum = np.cumsum(tp), np.cumsum(~tp)
    for b in budgets:
        allowed = b * n_images
        ok = np.nonzero(fp_cum <= allowed)[0]
        out[f"recall_at_{b}_fp_per_image"] = (
            float(tp_cum[ok[-1]] / n_gt) if len(ok) else 0.0)
    return out


def expected_calibration_error(scores: np.ndarray, labels: np.ndarray,
                               n_bins: int = 15) -> dict:
    """Equal-width ECE and reliability bins for a binary endpoint."""
    scores = np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)
    labels = np.asarray(labels, dtype=bool)
    if len(scores) == 0:
        return {"ece": None, "brier": None, "n_bins": n_bins, "bins": []}
    which = np.minimum((scores * n_bins).astype(int), n_bins - 1)
    bins, ece = [], 0.0
    for i in range(n_bins):
        take = which == i
        count = int(take.sum())
        mean_conf = float(scores[take].mean()) if count else None
        observed = float(labels[take].mean()) if count else None
        if count:
            ece += count / len(scores) * abs(mean_conf - observed)
        bins.append({
            "lo": i / n_bins,
            "hi": (i + 1) / n_bins,
            "count": count,
            "mean_confidence": mean_conf,
            "observed_positive_rate": observed,
        })
    return {"ece": float(ece),
            "brier": float(np.mean((scores - labels.astype(float)) ** 2)),
            "n_bins": n_bins, "bins": bins}


def binary_metrics(scores: np.ndarray, labels: np.ndarray,
                   min_negatives: int, ece_bins: int = 15) -> dict:
    """Image-level annotated-target-present metrics.

    AUROC needs negatives. Three of the five targets have fewer than 12
    negative test images, so it is reported as None there rather than as a
    number that looks meaningful. Average precision on the positive class is
    reported everywhere, since it does not depend on the negative count in the
    same way.
    """
    labels = labels.astype(bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    result = {"n_pos": n_pos, "n_neg": n_neg,
              "positive_rate": float(n_pos / max(1, len(labels)))}
    result["calibration"] = expected_calibration_error(scores, labels, ece_bins)

    if n_pos == 0 or len(labels) == 0:
        return {**result, "auroc": None, "ap": None, "auroc_reportable": False}

    order = np.argsort(-scores)
    s, y = scores[order], labels[order]
    tp_cum, fp_cum = np.cumsum(y), np.cumsum(~y)
    recall = tp_cum / n_pos
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    result["ap"] = _interp_precision(recall, precision)

    if n_neg >= min_negatives:
        # Rank-based AUROC, ties averaged.
        ranks = np.empty(len(s), dtype=np.float64)
        order2 = np.argsort(scores)
        sorted_scores = scores[order2]
        i = 0
        r = np.empty(len(s))
        while i < len(sorted_scores):
            j = i
            while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
                j += 1
            r[i:j + 1] = (i + j) / 2 + 1
            i = j + 1
        ranks[order2] = r
        auroc = (ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        result["auroc"] = float(auroc)
        result["auroc_reportable"] = True
    else:
        result["auroc"] = None
        result["auroc_reportable"] = False
        result["auroc_suppressed_reason"] = (
            f"{n_neg} negative images, under the {min_negatives} needed for a "
            f"calibrated false alarm rate")
    return result
