#!/usr/bin/env python3
"""Evaluate one arm cell, reusing the frozen metric and prediction code.

WHY THIS IS NOT `evaluate_cell.evaluate()`

That function guards on `C.config_fingerprint()` matching what the cell
recorded. An arm process has arm datasets registered, so its live fingerprint
is deliberately NOT f5567a1938278e88 and the guard would fire. The guard is
correct and is not weakened. This module reproduces it one level up: it
requires the ARM fingerprint to match, and the arm fingerprint embeds the
frozen main one, so both are pinned.

Everything that touches a number is imported from the frozen modules:
`load_ground_truth`, `predict_ultralytics`, `predict_torchvision`, `flatten`,
`flatten_raw` and all of `metrics`. There is no second copy of the metric code.

THE SPLIT BLOCK MAP HAS TO BE TAUGHT ABOUT ARM PATHS.
`evaluate_cell.split_block_map()` maps clean-relative paths to split blocks
from the frozen manifest. An arm image lives at `lohi_high/test/images/x.jpg`,
which is not in the manifest, but it is a SYMLINK to the lohi_weld image that
is. Resolving the link recovers the original path and therefore the original
block id. That is the correct answer, not a workaround: an arm image is the
same photograph in the same scene, so it belongs to the same independence unit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import config as C  # noqa: E402
import config_arms as A  # noqa: E402

A.assert_main_unmoved()
A.register()

import evaluate_cell as E  # noqa: E402
import metrics as M  # noqa: E402


def teach_split_blocks() -> int:
    """Add arm image paths to the frozen split block map, in this process.

    Every arm image is a symlink into clean_candidate. Resolving it gives the
    frozen path, whose block id is authoritative. Arm D pools are unions of
    frozen datasets, so the same resolution works for them too.
    """
    base = E.split_block_map()
    added = 0
    for tag in A.ARM_DATASETS:
        root = C.CANDIDATE / tag
        if not root.is_dir():
            continue
        for split in ("train", "valid", "test"):
            images = root / split / "images"
            if not images.is_dir():
                continue
            for path in images.iterdir():
                rel = path.relative_to(C.CANDIDATE).as_posix()
                if rel in base:
                    continue
                try:
                    original = path.resolve().relative_to(C.CANDIDATE).as_posix()
                except (OSError, ValueError):
                    continue
                block = base.get(original)
                if block is not None:
                    base[rel] = block
                    added += 1
    return added


def prediction_class_space(cell: dict) -> str:
    """The dataset whose class list the MODEL predicts in. Not always `source`.

    THE DEFECT THIS FIXES, found 15 September 2026 and recorded in the project log
    6a-iii. Every arm used to filter predictions by the SOURCE's defect ids.
    That is right for two of the three arms and wrong for the third.

    lohi_control, armd   The model trains on its own source and predicts in
                         that source's class space. Source ids are correct.
    armb                 `kshot.materialise` writes the TARGET's class list
                         into the k-shot data.yaml, so ultralytics discards the
                         source detection head and builds a new one at the
                         target's nc. The fine-tuned model therefore predicts
                         in the TARGET's class space, and the source ids index
                         a class list the model no longer has.

    The consequence was silent rather than loud. Asking for a class index the
    head cannot emit retains nothing, so kaggle_v2 to twoclass returned
    `n_pred: 0` with `map50: 0.0`, which reads as a measurement. Where the
    index did exist it named a different class: kaggle_v2 to visual kept
    `workpiece`, which config.py marks excluded, and scored scene furniture
    against defect ground truth.
    """
    if cell["arm"] == "armb":
        return cell["target"]
    return cell["source"]


def prediction_filter_ids(cell: dict) -> set[int]:
    """Defect class ids in the space the model actually predicts in."""
    return A.arm_defect_ids(prediction_class_space(cell))


def assert_ids_within_head(ids: set[int], nc: int, label: str) -> None:
    """Every retained id must be a class the head can actually emit.

    SILENT EMPTINESS IS WHAT LET THE DEFECT RUN FOR 240 CELLS. An id at or
    above the head width is not an error to numpy, it simply matches nothing,
    so the evaluation completes and writes a zero. Raising here converts that
    into a failure at the first cell instead of a plausible looking number in
    a figure.
    """
    bad = sorted(i for i in ids if i >= nc or i < 0)
    if bad:
        sys.exit(f"{label}: defect ids {bad} are outside the model's output "
                 f"range [0, {nc}). The filter and the detection head "
                 f"disagree about the class space. Refusing to score.")


def guard_degenerate_image_ap(image_level: dict, n_retained: int) -> dict:
    """An image level AP computed from an all-tied score vector is not a
    measurement, so report it as null with a reason. The project log 6a-iii.

    WHAT GOES WRONG. When no prediction survives the defect filter, every
    image scores exactly 0.0. `metrics.binary_metrics` then sorts a vector in
    which every element is tied, and AP becomes a function of the tie order
    alone. The order comes from `np.argsort(-scores)`, whose default is
    introsort and is NOT stable, so the value is arbitrary rather than merely
    unlucky. The kaggle_v2 to twoclass cells returned 0.9070 against a base
    rate of 0.8000 with `n_pred` 0, which reads as a model beating its base
    rate while actually describing nothing.

    The same ambiguity exists in weaker form whenever a large block of images
    ties at zero. That case is left as a number, because the non tied part of
    the ranking still carries signal, but it is why the recomputation proof
    compares APs against a tie envelope rather than to the last decimal.
    """
    if n_retained > 0:
        return image_level
    out = dict(image_level)
    out["ap"] = None
    out["ap_suppressed_reason"] = (
        "no prediction survived the defect class filter, so every image "
        "scores exactly 0.0 and the ranking is entirely tied. Average "
        "precision over an all tied vector is determined by the sort order, "
        "not by the model, and numpy's default sort is not stable. Reported "
        "as null rather than as a number. See the project log 6a-iii.")
    out["auroc"] = None
    out["auroc_reportable"] = False
    return out


ENDPOINT_NOTE = {
    "best": ("checkpoint with the best validation mAP. For arm B the "
             "validation set is the TARGET's frozen valid split, so target "
             "labels beyond the k shots were used to select it. Secondary "
             "endpoint, see the project log 6a-ii."),
    "last": ("checkpoint at the final epoch of the fixed schedule. No target "
             "label outside the draw influenced selection, which matches how "
             "P_ij(0) was produced. Primary arm B endpoint, see the project log "
             "6a-ii."),
}


def evaluate(cell_dir: Path, which: str = "best") -> dict:
    record = json.loads((cell_dir / "cell.json").read_text())
    cell = record["cell"]
    name = record["name"]
    family = cell["family"]

    # RESOLVE THE CHECKPOINT RELATIVE TO THE CELL DIRECTORY, not from the
    # absolute path in cell.json. That path points into the cluster home
    # (/home/USERNAME/...), so an archive pulled to the laptop can
    # never satisfy it. Both are recorded so the provenance stays visible.
    recorded = record.get("weights")
    weights = cell_dir / "weights" / f"{which}.pt"
    if not weights.is_file():
        sys.exit(f"missing {which}.pt checkpoint: {weights}")

    if record.get("arm_sha256_16") != A.arm_fingerprint():
        sys.exit("arm configuration changed after training. Refusing to "
                 "evaluate a checkpoint under a different protocol")
    if record.get("frozen_main_fingerprint") != A.FROZEN_MAIN_FINGERPRINT:
        sys.exit("the frozen main fingerprint recorded by this cell is not "
                 "the one the 69 main cells used. Refusing to mix protocols")
    code_root = HERE.parent
    for filename in ("config.py", "evaluate_cell.py", "metrics.py",
                     "yolo_dataset.py"):
        if record.get("code_sha256", {}).get(filename) != E.file_sha256(
                code_root / filename):
            sys.exit(f"{filename} changed after training. Refusing to mix "
                     "code versions within an evaluation")

    current = json.loads((C.CANDIDATE / "split_hashes.json").read_text())
    if record.get("split_hashes") != current:
        sys.exit("frozen split hashes changed after training. Refusing "
                 "mismatched evaluation")

    # WHICH CLASS SPACE THE MODEL PREDICTS IN decides which ids count. That is
    # the source for lohi_control and armd, and the TARGET for arm B, because
    # arm B's head is rebuilt at the target nc. See prediction_class_space.
    # For an arm D pool it is the pooled index space, which is why the pool's
    # class list is part of the arm fingerprint.
    source = cell["source"]
    id_space = prediction_class_space(cell)
    src_defect_ids = prediction_filter_ids(cell)
    head_nc = len(A.dataset_entry(id_space)["classes"])
    assert_ids_within_head(src_defect_ids, head_nc, name)
    predict = (E.predict_torchvision
               if C.FAMILIES[family]["backend"] == "torchvision"
               else E.predict_ultralytics)

    results = {
        "cell": cell, "name": name, "arm": cell["arm"],
        "arm_sha256_16": record["arm_sha256_16"],
        "frozen_main_fingerprint": record["frozen_main_fingerprint"],
        "endpoint": which,
        "endpoint_selected_by": ENDPOINT_NOTE[which],
        # RECORDED SO A FUTURE READER SEES THE CLASS SPACE FROM THE RESULT
        # ALONE, rather than deducing it from the arm and the cell name. The
        # defect in the project log 6a-iii was invisible in the output json, which is
        # part of why it survived 240 cells.
        "defect_id_space": id_space,
        "defect_ids_used": sorted(src_defect_ids),
        "defect_class_names_used": [
            A.dataset_entry(id_space)["classes"][i]
            for i in sorted(src_defect_ids)],
        "head_nc": head_nc,
        "weights_recorded_at_training": recorded,
        "weights_resolved": str(weights),
        "image_endpoint": ("at least one retained annotation target is "
                           "present, not a universal part verdict"),
        "postprocessing": {
            "confidence_floor": C.CONF_FLOOR,
            "nms_iou": C.NMS_IOU,
            "max_detections_per_image": C.MAX_DET,
        },
        "targets": {},
    }
    raw = {}

    for result_key, target in A.arm_eval_plan(cell):
        gt = E.load_ground_truth(target)
        preds = predict(weights, gt["images"], source)
        n_images = len(gt["images"])

        scores = np.zeros(n_images, dtype=np.float32)
        n_retained = 0
        for i, (b, s, c) in enumerate(preds):
            if len(s):
                m = np.isin(c, list(src_defect_ids))
                n_retained += int(m.sum())
                scores[i] = float(s[m].max()) if m.any() else 0.0
        image_level = M.binary_metrics(scores, gt["image_labels"],
                                       C.MIN_NEGATIVES_FOR_FPR, C.ECE_BINS)
        image_level["n_pred_retained"] = n_retained
        image_level = guard_degenerate_image_ap(image_level, n_retained)

        prefix = result_key + "__"
        ri, rb, rs, rc = E.flatten_raw(preds)
        raw[prefix + "pred_image"] = ri
        raw[prefix + "pred_box"] = rb
        raw[prefix + "pred_score"] = rs
        raw[prefix + "pred_class"] = rc
        raw[prefix + "image_score"] = scores
        raw[prefix + "image_label"] = gt["image_labels"].astype(np.uint8)
        raw[prefix + "image_name"] = np.array([p.name for p in gt["images"]])
        raw[prefix + "split_block"] = gt["split_blocks"]

        # the project log 6b. A pool containing weld_surface_v2 is protocol `mixed`,
        # so its box level number would measure annotation mismatch rather
        # than domain shift. It is skipped with the reason recorded, exactly
        # as the main grid skips a cross-protocol pair.
        if A.box_level_defined(source, target):
            pi, pb, ps = E.flatten(preds, src_defect_ids, collapse=True)
            box_level = M.average_precision_range(pi, pb, ps, gt["gt_defect"])
            box_level.update(M.recall_at_fp_budget(
                pi, pb, ps, gt["gt_defect"], n_images, C.FP_PER_IMAGE_BUDGETS))
        else:
            box_level = {
                "skipped": True,
                "reason": (f"source protocol "
                           f"{A.dataset_entry(source)['protocol']}, target "
                           f"protocol {A.dataset_entry(target)['protocol']}. "
                           "A box level number across protocols measures "
                           "annotation mismatch, not domain shift.")}

        results["targets"][result_key] = {
            "target": target, "test_split": "audited",
            "n_test_images": n_images,
            "image_level": image_level,
            "box_level": box_level,
        }
        ap = image_level.get("ap")
        m50 = box_level.get("map50")
        ap_txt = f"{ap:.4f}" if ap is not None else "n/a"
        box_txt = f"{m50:.4f}" if m50 is not None else "skipped"
        print(f"    {result_key:<26s} n={n_images:<5d} "
              f"image AP {ap_txt:<8s} box mAP50 {box_txt}")

    return results, raw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--runs-dir", type=Path, default=A.ARM_RUNS)
    ap.add_argument("--out-dir", type=Path, default=A.ARM_RESULTS)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--weights", choices=["best", "last"], default="best",
                    help="which checkpoint to score. `best` is the default so "
                         "the running cluster chain is unaffected. `last` is "
                         "the primary arm B endpoint, the project log 6a-ii, and "
                         "writes eval_last__/raw_last__ instead.")
    ap.add_argument("--cell-dir", type=Path,
                    help="explicit cell directory, for scoring an archive "
                         "extracted somewhere other than --runs-dir")
    args = ap.parse_args()

    cells = A.all_arm_cells()
    cell = cells[args.index]
    name = A.arm_cell_name(cell)
    cell_dir = args.cell_dir or (
        args.runs_dir / (("smoke_" + name) if args.smoke else name))

    added = teach_split_blocks()
    print(f"=== evaluating arm cell {args.index}: {name} "
          f"[endpoint {args.weights}] ===")
    print(f"    arm path aliases added to the split block map: {added}")

    results, raw = evaluate(cell_dir, args.weights)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = ("smoke_" + name) if args.smoke else name
    # The `best` names are left exactly as they were, so nothing the cluster
    # already wrote is shadowed or overwritten.
    prefix = "eval__" if args.weights == "best" else "eval_last__"
    raw_prefix = "raw__" if args.weights == "best" else "raw_last__"
    out = args.out_dir / f"{prefix}{stem}.json"
    out.write_text(json.dumps(E.json_safe(results), indent=2))
    np.savez_compressed(args.out_dir / f"{raw_prefix}{stem}.npz", **raw)
    print(f"    wrote {out}")


if __name__ == "__main__":
    main()
