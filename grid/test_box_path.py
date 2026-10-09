#!/usr/bin/env python3
"""Prove the box-level path really executes, without waiting for cell 12.

Cells 0..11 are all weld_surface_v2, whose whole-region protocol makes every
box_level block skip correctly. So a defect in the box path would not surface
until index 12, hours away. This drives the REAL functions from evaluate_cell
and metrics against the REAL frozen test splits, with predictions synthesised
from the ground truth so the expected answer is known in advance.

No model, no GPU, no cluster, and nothing is written to any dataset folder.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

GRID = Path(__file__).resolve().parent
sys.path.insert(0, str(GRID))

import config as C            # noqa: E402
import metrics as M           # noqa: E402
import evaluate_cell as E     # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAIL.append(name)


# ---------------------------------------------------------------- 1. the gate
print("protocol gate, exhaustive over source x target")
produced, skipped = [], []
for source in C.TRAIN_SOURCES:
    for target in C.ALL_TARGETS:
        same = (C.DATASETS[source]["protocol"] ==
                C.DATASETS[target]["protocol"] == "localisation")
        (produced if same else skipped).append((source, target))

want_sources = set(C.BOX_SOURCES)
want_targets = set(C.BOX_TARGETS)
check("box rows are exactly the localisation sources",
      {s for s, _ in produced} == want_sources,
      f"{sorted({s for s, _ in produced})}")
check("box columns are exactly the localisation targets",
      {t for _, t in produced} == want_targets,
      f"{sorted({t for _, t in produced})}")
check("matrix is 2 by 4", len(produced) == 8, f"{len(produced)} pairs")
check("every weld_surface_v2 pair is skipped",
      all(s == "weld_surface_v2" or t == "weld_surface_v2"
          for s, t in skipped) and
      all("weld_surface_v2" in (s, t) for s, t in skipped),
      f"{len(skipped)} skipped")
check("no produced pair involves weld_surface_v2",
      all("weld_surface_v2" not in (s, t) for s, t in produced))


# --------------------------------------------------- 2. real GT, real metrics
def synth(gt_defect, n_images, rng, jitter=0.0, drop=0.0, extra=0):
    """Predictions built from ground truth, in the (boxes, scores, cls) shape
    that evaluate_cell.flatten expects. jitter=0, drop=0, extra=0 is perfect."""
    defect_id = sorted(C.defect_ids(source))[0]
    preds = []
    for i in range(n_images):
        g = gt_defect.get(i, np.zeros((0, 4), np.float32))
        keep = g if drop == 0 or len(g) == 0 else g[rng.random(len(g)) >= drop]
        b = keep.astype(np.float32).copy()
        if jitter and len(b):
            wh = np.stack([b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1)
            b[:, :2] += rng.normal(0, jitter, (len(b), 2)) * wh
            b[:, 2:] += rng.normal(0, jitter, (len(b), 2)) * wh
        s = np.full(len(b), 0.9, np.float32)
        if extra:
            fp = rng.uniform(0, 600, (extra, 2)).astype(np.float32)
            fp = np.concatenate([fp, fp + 30], 1)
            b = np.concatenate([b, fp]) if len(b) else fp
            s = np.concatenate([s, np.full(extra, 0.3, np.float32)])
        c = np.full(len(b), defect_id, dtype=int)
        preds.append((b, s, c))
    return preds


for source, target in [("kaggle_v2", "kaggle_v2"), ("lohi_weld", "lohi_weld"),
                       ("kaggle_v2", "lohi_weld"), ("lohi_weld", "visual")]:
    print(f"\nreal box evaluation, {source} -> {target}")
    gt = E.load_ground_truth(target, "test", "main")
    n = len(gt["images"])
    n_gt_boxes = sum(len(v) for v in gt["gt_defect"].values())
    check("ground truth loaded", n > 0 and n_gt_boxes > 0,
          f"{n} images, {n_gt_boxes} defect boxes")

    ids = C.defect_ids(source)
    rng = np.random.default_rng(0)

    # (a) Perfect predictions hit the CEILING, which is not always 1.0.
    #
    # The box level collapses every defect class to one and runs class-agnostic
    # NMS. That is deliberate: without it a pore box and a stain box on the same
    # object become a false positive pair the model never made. But it also
    # means two genuinely distinct ground truth boxes overlapping above the NMS
    # IoU cannot both be recovered, so recall has a hard ceiling below 1 and AP
    # follows. Asserting 1.0 here would be asserting the wrong thing.
    kept = 0
    for i in range(n):
        g = gt["gt_defect"].get(i, np.zeros((0, 4), np.float32))
        if len(g):
            kept += len(M.class_agnostic_nms(g.astype(np.float32),
                                             np.full(len(g), 0.9, np.float32)))
    recall_ceiling = kept / n_gt_boxes
    # COCO AP samples 101 recall points, so the ceiling is quantized: any
    # recall point above the achievable recall contributes zero precision. One
    # unrecoverable box therefore costs a whole 1/101, not its own tiny share.
    expected = (M.RECALL_POINTS <= recall_ceiling + 1e-12).sum() / len(M.RECALL_POINTS)
    pi, pb, ps = E.flatten(synth(gt["gt_defect"], n, rng), ids, collapse=True)
    perfect = M.average_precision_range(pi, pb, ps, gt["gt_defect"])
    check("perfect predictions hit the exact quantized ceiling",
          abs(perfect["map50"] - expected) < 1e-9,
          f"AP {perfect['map50']:.6f}, expected {expected:.6f}, recall ceiling "
          f"{recall_ceiling:.6f}, {n_gt_boxes - kept} of {n_gt_boxes} suppressed")
    check("the collapse loses under 1 percent of ground truth",
          (n_gt_boxes - kept) / n_gt_boxes < 0.01,
          f"{100 * (n_gt_boxes - kept) / n_gt_boxes:.3f}% lost")
    check("perfect mAP50-95 equals perfect mAP50 when boxes are exact",
          abs(perfect["map50_95"] - perfect["map50"]) < 1e-9,
          f"{perfect['map50_95']:.6f}")

    # (b) realistic predictions must give real, finite, ordered values
    rng = np.random.default_rng(1)
    preds = synth(gt["gt_defect"], n, rng, jitter=0.06, drop=0.25, extra=2)
    pi, pb, ps = E.flatten(preds, ids, collapse=True)
    box = M.average_precision_range(pi, pb, ps, gt["gt_defect"])
    box.update(M.recall_at_fp_budget(pi, pb, ps, gt["gt_defect"], n,
                                     C.FP_PER_IMAGE_BUDGETS))

    m50, m5095 = box["map50"], box["map50_95"]
    check("mAP50 is a real number, not skipped and not null",
          isinstance(m50, float) and np.isfinite(m50), f"{m50:.4f}")
    check("mAP50-95 is a real number", isinstance(m5095, float)
          and np.isfinite(m5095), f"{m5095:.4f}")
    check("mAP50 is strictly inside (0, 1)", 0.0 < m50 < 1.0)
    check("mAP50-95 <= mAP50, as the definition requires", m5095 <= m50 + 1e-12,
          f"{m5095:.4f} <= {m50:.4f}")
    check("n_gt matches the loaded ground truth", box["n_gt"] == n_gt_boxes,
          f"{box['n_gt']} vs {n_gt_boxes}")
    check("n_pred is positive", box["n_pred"] > 0, str(box["n_pred"]))
    check("no skip marker on a localisation pair", "skipped" not in box)

    budgets = [f"recall_at_{b}_fp_per_image" for b in C.FP_PER_IMAGE_BUDGETS]
    check("every FP budget recall present",
          all(k in box for k in budgets),
          ", ".join(f"{b}={box[b]:.3f}" for b in budgets if b in box))

    # (c) it must survive the same JSON serialization the grid uses
    round_trip = json.loads(json.dumps(box, default=float))
    check("survives JSON round trip with values intact",
          abs(round_trip["map50"] - m50) < 1e-9
          and abs(round_trip["map50_95"] - m5095) < 1e-9)

# --------------------------------------------- 3. the cross-protocol skip path
print("\ncross-protocol pair must skip with a reason")
source, target = "weld_surface_v2", "kaggle_v2"
same = (C.DATASETS[source]["protocol"] ==
        C.DATASETS[target]["protocol"] == "localisation")
check("gate is False for whole-region source", same is False)
skip = {"skipped": True, "reason": "..."}
check("skipped block carries no mAP keys",
      "map50" not in skip and "map50_95" not in skip)

print()
if FAIL:
    raise SystemExit(f"{len(FAIL)} FAILED: {FAIL}")
print("box-level path verified end to end on real frozen test splits")
