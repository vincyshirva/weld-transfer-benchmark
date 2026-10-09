#!/usr/bin/env python3
"""Checks for metrics.py, against cases where the answer is known by hand."""

from __future__ import annotations

import numpy as np

from metrics import (average_precision, binary_metrics,
                     bootstrap_detection_ap_by_cluster,
                     class_agnostic_nms, expected_calibration_error,
                     iou_matrix, recall_at_fp_budget)

FAILURES = []


def check(name: str, got, want, tol=1e-6):
    ok = abs(got - want) < tol if isinstance(want, float) else got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got}, want {want}")
    if not ok:
        FAILURES.append(name)


B = lambda *v: np.array(v, dtype=np.float32).reshape(-1, 4)  # noqa: E731

print("iou_matrix")
check("identical boxes", float(iou_matrix(B(0, 0, 10, 10), B(0, 0, 10, 10))[0, 0]), 1.0)
check("disjoint boxes", float(iou_matrix(B(0, 0, 10, 10), B(20, 20, 30, 30))[0, 0]), 0.0)
# half overlap: intersection 50, union 150
check("half overlap", float(iou_matrix(B(0, 0, 10, 10), B(5, 0, 15, 10))[0, 0]), 1 / 3, 1e-5)
check("empty prediction", iou_matrix(np.zeros((0, 4)), B(0, 0, 1, 1)).shape, (0, 1))

print("\nclass_agnostic_nms")
# Two boxes on the same object, different classes. The collapse must keep one.
boxes = B(0, 0, 10, 10, 1, 1, 11, 11)
keep = class_agnostic_nms(boxes, np.array([0.9, 0.8]), iou_thr=0.5)
check("suppresses a duplicate across classes", len(keep), 1)
check("keeps the higher score", int(keep[0]), 0)
far = B(0, 0, 10, 10, 100, 100, 110, 110)
check("keeps genuinely separate boxes",
      len(class_agnostic_nms(far, np.array([0.9, 0.8]), 0.5)), 2)

print("\naverage_precision")
# One image, one ground truth, one perfect prediction.
gt = {0: B(0, 0, 10, 10)}
r = average_precision(np.array([0]), B(0, 0, 10, 10), np.array([0.9]), gt)
check("perfect single detection", r["ap"], 1.0)
# A prediction that misses entirely.
r = average_precision(np.array([0]), B(50, 50, 60, 60), np.array([0.9]), gt)
check("complete miss", r["ap"], 0.0)
# No predictions at all.
r = average_precision(np.array([], dtype=int), np.zeros((0, 4)), np.array([]), gt)
check("no predictions", r["ap"], 0.0)
# No ground truth: AP is undefined, not zero. A zero here would silently drag
# down any average that includes an absent class.
r = average_precision(np.array([0]), B(0, 0, 1, 1), np.array([0.9]), {0: np.zeros((0, 4))})
check("no ground truth is nan", bool(np.isnan(r["ap"])), True)
# Two ground truths, one found. AP must be 0.5, not 1.0.
gt2 = {0: B(0, 0, 10, 10, 100, 100, 110, 110)}
r = average_precision(np.array([0]), B(0, 0, 10, 10), np.array([0.9]), gt2)
check("recall ceiling caps AP", round(r["ap"], 2), 0.5)
# One ground truth, a hit then a duplicate at a LOWER score. Under the COCO
# precision envelope this does not reduce AP, because precision at a given
# recall is the maximum over all recalls at or above it. This is correct, not
# a bug, and it is why class-agnostic NMS matters for the false-positive
# budget metric rather than for AP.
r = average_precision(np.array([0, 0]), B(0, 0, 10, 10, 0, 0, 10, 10),
                      np.array([0.9, 0.8]), gt)
check("duplicate below the hit leaves AP at 1", r["ap"], 1.0)
# A false positive ranked ABOVE the true positives does reduce AP.
gt3 = {0: B(0, 0, 10, 10, 100, 100, 110, 110)}
r = average_precision(np.array([0, 0, 0]),
                      B(500, 500, 510, 510, 0, 0, 10, 10, 100, 100, 110, 110),
                      np.array([0.95, 0.9, 0.8]), gt3)
check("false positive above the hits reduces AP", round(r["ap"], 3), 0.667, 1e-3)

print("\nrecall_at_fp_budget")
# 2 images, 2 ground truths, 1 hit then 3 false positives.
gtb = {0: B(0, 0, 10, 10), 1: B(0, 0, 10, 10)}
pi = np.array([0, 0, 0, 1])
pb = B(0, 0, 10, 10, 50, 50, 60, 60, 70, 70, 80, 80, 90, 90, 95, 95)
ps = np.array([0.9, 0.8, 0.7, 0.6])
r = recall_at_fp_budget(pi, pb, ps, gtb, n_images=2, budgets=[0.5, 1.0])
check("tight budget finds the one hit", r["recall_at_0.5_fp_per_image"], 0.5)
check("loose budget still 0.5", r["recall_at_1.0_fp_per_image"], 0.5)
# A duplicate ranked between two true positives consumes the zero-FP budget
# before the second object can be reached. Class-agnostic NMS restores it.
gt_dup = {0: B(0, 0, 10, 10), 1: B(50, 50, 60, 60)}
dup_b = B(0, 0, 10, 10, 1, 1, 11, 11, 50, 50, 60, 60)
dup_i = np.array([0, 0, 1])
dup_s = np.array([0.9, 0.8, 0.7])
r1 = recall_at_fp_budget(dup_i, dup_b, dup_s, gt_dup,
                         n_images=2, budgets=[0.0])
keep = class_agnostic_nms(dup_b, dup_s, 0.5)
r2 = recall_at_fp_budget(dup_i[keep], dup_b[keep], dup_s[keep], gt_dup,
                         n_images=2, budgets=[0.0])
check("duplicate consumes the zero FP budget",
      r1["recall_at_0.0_fp_per_image"], 0.5)
check("class-agnostic NMS restores the second hit",
      r2["recall_at_0.0_fp_per_image"], 1.0)

print("\nbinary_metrics")
s = np.array([0.9, 0.8, 0.2, 0.1])
y = np.array([1, 1, 0, 0])
r = binary_metrics(s, y, min_negatives=2)
check("perfect separation AUROC", r["auroc"], 1.0)
check("perfect separation AP", r["ap"], 1.0)
r = binary_metrics(s, np.array([0, 0, 1, 1]), min_negatives=2)
check("inverted AUROC", r["auroc"], 0.0)
# The case that matters: too few negatives to quote a false alarm rate.
r = binary_metrics(np.array([0.9, 0.8, 0.1]), np.array([1, 1, 0]), min_negatives=30)
check("AUROC suppressed on 1 negative", r["auroc"], None)
check("AP still reported", r["ap"] is not None, True)
check("suppression is flagged", r["auroc_reportable"], False)

print("\ncalibration")
c = expected_calibration_error(np.array([1.0, 1.0, 0.0, 0.0]),
                               np.array([1, 1, 0, 0]), n_bins=5)
check("perfect calibration ECE", c["ece"], 0.0)
check("perfect calibration Brier", c["brier"], 0.0)

print("\ncluster bootstrap")
boot_gt = {0: B(0, 0, 10, 10), 1: B(20, 20, 30, 30)}
boot = bootstrap_detection_ap_by_cluster(
    np.array([0, 1]), B(0, 0, 10, 10, 20, 20, 30, 30),
    np.array([0.9, 0.8]), boot_gt, np.array(["a", "b"]),
    resamples=20, seed=1)
check("perfect bootstrap point", boot["map50"]["point"], 1.0)
check("perfect bootstrap lower bound", boot["map50"]["lo"], 1.0)
check("bootstrap uses two clusters", boot["n_clusters"], 2)

print()
if FAILURES:
    raise SystemExit(f"{len(FAILURES)} FAILED: {FAILURES}")
print("all metric checks passed")
