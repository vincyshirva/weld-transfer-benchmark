#!/usr/bin/env python3
"""Why is kaggle_v2's own box-level mAP50 only 0.080?

REPORT ONLY. This script writes nothing into clean_candidate/, grid_results/
or any original dataset folder. It extracts a checkpoint into a scratch
directory, reads the frozen test split, and prints. Nothing it produces is an
experiment and nothing it produces changes a frozen artefact.

Four questions, in the order a fault would be found:

  1. Per ORIGINAL class localisation AP on the kaggle diagonal. If `Defect`
     alone is low while the two verdict classes are fine, the pipeline works
     on these images and the defect class is the hard part.
  2. Are `Good Weld` and `Bad Weld` excluded from the box-level positive set?
     A verdict box is not a defect box. If either leaked into the ground truth
     or into the kept predictions, the number is a mapping fault.
  3. Kaggle `Defect` box sizes in pixels, in the letterboxed 640 frame the
     model actually sees, and the fraction below 16 x 16. Small objects are
     the standard explanation for a low mAP and it has to be measured, not
     assumed.
  4. mAP50 at 1024 input for one YOLO11s seed. If the score climbs sharply,
     640 was destroying the objects and the finding is resolution loss rather
     than dataset difficulty.

Usage:  python3 grid/diagnose_kaggle_box.py [--scratch DIR]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C  # noqa: E402
import metrics as M  # noqa: E402

TAG = "kaggle_v2"
RESULTS = C.PROJECT / "grid_results" / "results"


def rule(title: str) -> None:
    print(f"\n{'=' * 76}\n{title}\n{'=' * 76}")


# --------------------------------------------------------------------------
# 1. per original class localisation AP on the diagonal
# --------------------------------------------------------------------------

def per_class_diagonal() -> None:
    rule("1. Per-original-class localisation AP, kaggle_v2 diagonal")
    print("Read from the frozen evaluation records. These are the source's own\n"
          "classes, which is the only place per-class AP is well posed.\n")

    acc = defaultdict(lambda: defaultdict(list))
    counts: dict[str, int] = {}
    for family in C.FAMILIES:
        for seed in C.SEEDS:
            name = C.cell_name({"source": TAG, "family": family, "seed": seed,
                                "kind": "main"})
            path = RESULTS / f"eval__{name}.json"
            if not path.is_file():
                continue
            rec = json.loads(path.read_text())
            pc = rec["targets"][TAG].get("per_class_in_domain") or {}
            for cls, info in pc.items():
                if info.get("map50") is not None:
                    acc[family][cls].append(info["map50"])
                counts[cls] = info.get("test_instances")

    header = f"{'class':<12s}" + "".join(f"{f:>14s}" for f in C.FAMILIES) + \
             f"{'test boxes':>12s}"
    print(header)
    print("-" * len(header))
    for cls in C.DATASETS[TAG]["classes"]:
        line = f"{cls:<12s}"
        for family in C.FAMILIES:
            vals = acc[family].get(cls, [])
            line += (f"{np.mean(vals):>10.3f}+-{np.std(vals):<3.2f}"
                     if vals else f"{'n/a':>14s}")
        line += f"{counts.get(cls, 0):>12d}"
        print(line)

    print("\nNOTE. These per-class numbers are class-AWARE and are computed on\n"
          "the source's own labels. The 0.080 headline is the class-AGNOSTIC\n"
          "box matrix value, which collapses every defect class to one and is\n"
          "therefore the `Defect` column alone for kaggle_v2.")


# --------------------------------------------------------------------------
# 2. is the verdict class excluded from the box-level positive set?
# --------------------------------------------------------------------------

def exclusion_check() -> None:
    rule("2. Are Good Weld and Bad Weld excluded from the box-level positives?")
    from evaluate_cell import flatten, load_ground_truth

    names = C.DATASETS[TAG]["classes"]
    keep = C.defect_ids(TAG)
    print(f"KAGGLE_BAD_WELD_IS_DEFECT = {C.KAGGLE_BAD_WELD_IS_DEFECT}")
    print(f"classes            {names}")
    print(f"non_defect         {sorted(C.DATASETS[TAG]['non_defect'])}")
    print(f"excluded           {sorted(C.DATASETS[TAG]['excluded'])}")
    print(f"defect_classes()   {C.defect_classes(TAG)}")
    print(f"defect_ids()       {sorted(keep)}  "
          f"-> {[names[i] for i in sorted(keep)]}")

    gt = load_ground_truth(TAG)

    # GROUND TRUTH SIDE. gt_defect is what the box metric scores against.
    n_defect_gt = sum(len(v) for v in gt["gt_defect"].values())
    per_class_gt = {names[c]: sum(len(v) for v in d.values())
                    for c, d in gt["gt_by_class"].items()}
    print(f"\nground truth boxes by original class: {per_class_gt}")
    print(f"boxes in gt_defect, the box-level positive set: {n_defect_gt}")
    verdict_total = sum(v for k, v in per_class_gt.items()
                        if k in {"Bad Weld", "Good Weld"})
    print(f"  Bad Weld + Good Weld boxes present in the dataset: {verdict_total}")
    print(f"  of which inside gt_defect: "
          f"{n_defect_gt - per_class_gt.get('Defect', 0)}  (must be 0)")

    # PREDICTION SIDE. Verdict predictions must be dropped before NMS, so they
    # cannot become false positives against a Defect ground truth box.
    name = C.cell_name({"source": TAG, "family": "yolo11s", "seed": 0,
                        "kind": "main"})
    with np.load(RESULTS / f"raw__{name}.npz", allow_pickle=True) as raw:
        pc = raw[f"{TAG}__pred_class"]
        n_by_class = {names[c]: int((pc == c).sum()) for c in np.unique(pc)}
        preds = []
        pi = raw[f"{TAG}__pred_image"]
        pb = raw[f"{TAG}__pred_box"]
        ps = raw[f"{TAG}__pred_score"]
        for i in range(len(gt["images"])):
            m = pi == i
            preds.append((pb[m], ps[m], pc[m]))
    kept_i, kept_b, kept_s = flatten(preds, keep, collapse=True)
    print(f"\nraw predictions by class (YOLO11s seed 0): {n_by_class}")
    print(f"predictions surviving the defect filter and class-agnostic NMS: "
          f"{len(kept_s)}")

    verdict_pred = sum(v for k, v in n_by_class.items()
                       if k in {"Bad Weld", "Good Weld"})
    print(f"  verdict-class predictions discarded before scoring: {verdict_pred}")

    ap = M.average_precision(kept_i, kept_b, kept_s, gt["gt_defect"])
    print(f"\nrecomputed class-agnostic mAP50 from the raw sidecar: {ap['ap']:.4f}"
          f"   (n_gt={ap['n_gt']}, n_pred={ap['n_pred']})")

    # ORACLE. Feed the ground truth back in as perfect predictions. Anything
    # other than 1.0 means the coordinate frame or the matcher is broken.
    o_i, o_b, o_s = [], [], []
    for i, boxes in gt["gt_defect"].items():
        if len(boxes):
            o_i.append(np.full(len(boxes), i, dtype=int))
            o_b.append(boxes)
            o_s.append(np.ones(len(boxes), np.float32))
    oracle = M.average_precision(np.concatenate(o_i), np.concatenate(o_b),
                                 np.concatenate(o_s), gt["gt_defect"])
    print(f"oracle mAP50, ground truth fed back as predictions: {oracle['ap']:.4f}"
          f"   (must be 1.0000)")

    # Could a verdict box be silently standing in for a Defect box? If they
    # overlapped, excluding them would be throwing away true positives.
    best = []
    for i in range(len(gt["images"])):
        d = gt["gt_defect"].get(i, np.zeros((0, 4), np.float32))
        if not len(d):
            continue
        for c, cname in enumerate(names):
            if cname == "Defect":
                continue
            v = gt["gt_by_class"].get(c, {}).get(i, np.zeros((0, 4), np.float32))
            if len(v):
                best.append((cname, float(M.iou_matrix(d, v).max())))
    for cname in ("Bad Weld", "Good Weld"):
        vals = [b for n, b in best if n == cname]
        if vals:
            print(f"median best IoU, Defect box against {cname:<10s}: "
                  f"{np.median(vals):.3f}   max {max(vals):.3f}   n={len(vals)}")


# --------------------------------------------------------------------------
# 3. defect box size distribution at 640
# --------------------------------------------------------------------------

def box_sizes() -> None:
    rule("3. kaggle_v2 Defect box sizes in the letterboxed 640 frame")
    from evaluate_cell import load_ground_truth

    rows = {}
    for tag in ("kaggle_v2", "lohi_weld", "visual", "twoclass"):
        gt = load_ground_truth(tag)
        boxes = np.concatenate([v for v in gt["gt_defect"].values() if len(v)]) \
            if any(len(v) for v in gt["gt_defect"].values()) else np.zeros((0, 4))
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        side = np.sqrt(w * h)
        rows[tag] = {
            "n": len(boxes),
            "median_w": float(np.median(w)), "median_h": float(np.median(h)),
            "median_side": float(np.median(side)),
            "p10_side": float(np.percentile(side, 10)),
            "frac_under_16": float((side < 16).mean()),
            "frac_under_32": float((side < 32).mean()),
            "frac_area_under_0.1pct": float(((w * h) / (640 * 640) < 0.001).mean()),
        }

    head = (f"{'dataset':<16s}{'boxes':>7s}{'med w':>8s}{'med h':>8s}"
            f"{'med sqrt(wh)':>14s}{'p10 side':>10s}{'<16x16':>9s}{'<32x32':>9s}")
    print(head)
    print("-" * len(head))
    for tag, r in rows.items():
        print(f"{tag:<16s}{r['n']:>7d}{r['median_w']:>8.1f}{r['median_h']:>8.1f}"
              f"{r['median_side']:>14.1f}{r['p10_side']:>10.1f}"
              f"{r['frac_under_16']:>8.1%}{r['frac_under_32']:>8.1%}")

    print("\nCOCO calls an object small below 32 x 32 in a 640-ish frame.\n"
          "The comparison that matters is kaggle against lohi_weld, because\n"
          "lohi reaches 0.666 in domain on the same metric and the same\n"
          "pipeline.")


# --------------------------------------------------------------------------
# 4. scale check at 1024
# --------------------------------------------------------------------------

def scale_check(scratch: Path, family: str = "yolo11s", seed: int = 0) -> None:
    rule(f"4. Scale check: class-agnostic mAP50 at 640 against 1024, "
         f"{family} seed {seed}")
    print("The SAME frozen checkpoint, re-run at a larger input. Only inference\n"
          "resolution changes. No retraining, no threshold change, and nothing\n"
          "is written back into the frozen tree.\n")

    name = C.cell_name({"source": TAG, "family": family, "seed": seed,
                        "kind": "main"})
    archive = RESULTS / f"{name}.tar.gz"
    scratch.mkdir(parents=True, exist_ok=True)
    weights = scratch / name / "weights" / "best.pt"
    if not weights.is_file():
        subprocess.run(["tar", "-xzf", str(archive), "-C", str(scratch)],
                       check=True)
    print(f"checkpoint: {weights}")

    import evaluate_cell as E

    original = C.IMGSZ
    out = {}
    for imgsz in (640, 1024):
        # In-process override only. This process writes no artefact, so the
        # frozen configuration fingerprint is untouched on disk. Both the
        # ground truth letterbox and the model input use C.IMGSZ, so setting
        # it here keeps prediction and ground truth in ONE coordinate frame,
        # which is the only way the two rows below are comparable.
        C.IMGSZ = imgsz
        E.C.IMGSZ = imgsz
        gt = E.load_ground_truth(TAG)
        preds = E.predict_ultralytics(weights, gt["images"], TAG)
        pi, pb, ps = E.flatten(preds, C.defect_ids(TAG), collapse=True)
        res = M.average_precision_range(pi, pb, ps, gt["gt_defect"])
        rec = M.recall_at_fp_budget(pi, pb, ps, gt["gt_defect"],
                                    len(gt["images"]), C.FP_PER_IMAGE_BUDGETS)
        out[imgsz] = {**res, **rec}
        print(f"  imgsz {imgsz:<5d} mAP50 {res['map50']:.4f}   "
              f"mAP50-95 {res['map50_95']:.4f}   "
              f"recall@1.0FP {rec.get('recall_at_1.0_fp_per_image', float('nan')):.4f}   "
              f"n_pred {res['n_pred']}")
    C.IMGSZ = original
    E.C.IMGSZ = original

    d50 = out[1024]["map50"] - out[640]["map50"]
    print(f"\nchange from 640 to 1024: mAP50 {d50:+.4f} "
          f"({d50 / max(out[640]['map50'], 1e-9):+.1%} relative)")
    print("\nThe 640 row must reproduce the frozen grid value for this cell.\n"
          "If it does not, the diagnosis is invalid and the discrepancy is\n"
          "the finding.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scratch", type=Path,
                    default=Path("/tmp/weld_scratch/kaggle_diag"))
    ap.add_argument("--skip-gpu", action="store_true",
                    help="omit the 1024 scale check")
    args = ap.parse_args()

    print(__doc__.split("Usage:")[0])
    per_class_diagonal()
    exclusion_check()
    box_sizes()
    if not args.skip_gpu:
        scale_check(args.scratch)


if __name__ == "__main__":
    main()
