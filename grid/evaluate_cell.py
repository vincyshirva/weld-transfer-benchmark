#!/usr/bin/env python3
"""
Evaluate one trained cell against every target dataset.

This is where harmonization happens. Models trained on their source's native
labels are mapped into a shared space at evaluation time, which is why 36
trainings serve both matrices and no model is ever retrained to be measured
again.

For each (cell, target) pair it produces:

  image level   annotated target feature present or not. Comparable across
                all five datasets because it does not depend on how boxes
                were drawn. It is not a universal part acceptance label.
  box level     class agnostic localisation, but only when source and target
                share the localisation protocol. Otherwise it is recorded as
                null with a reason, rather than as a low number a reader would
                mistake for domain shift.
  in domain     per class AP with test instance counts, only when source and
                target are the same dataset. Cross dataset per class AP is not
                defined, since the class lists do not correspond.

Usage:
    python3 evaluate_cell.py --index 7
    python3 evaluate_cell.py --name kaggle_v2__yolo11s__seed0
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C  # noqa: E402
import metrics as M  # noqa: E402
from yolo_dataset import IMG_EXT, letterbox  # noqa: E402


_SPLIT_BLOCKS = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_block_map() -> dict[str, str]:
    """Map each retained clean image to its conservative independence unit."""
    global _SPLIT_BLOCKS
    if _SPLIT_BLOCKS is None:
        _SPLIT_BLOCKS = {}
        with (C.CANDIDATE / "manifest.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["kept"].lower() == "true":
                    _SPLIT_BLOCKS[row["representative_clean_path"]] = row["split_block"]
    return _SPLIT_BLOCKS


def resolve_test_images(target: str, kind: str, split: str = "test"):
    """The images a cell is scored on.

    A dirty cell must be scored on the staged copy of the original shipped
    test split, not the clean one. The two values use different test sets, so
    their difference is an estimate shift and not a causal leakage effect.
    """
    if kind != "dirty":
        return sorted(p for p in (C.CANDIDATE / target / split / "images").iterdir()
                      if p.suffix.lower() in IMG_EXT)
    yaml_path = C.data_yaml({"kind": "dirty", "source": target})
    cfg = yaml.safe_load(yaml_path.read_text())
    entry = Path(cfg[{"test": "test", "valid": "val", "train": "train"}[split]])
    # ANCHOR TO THE YAML, NEVER TO THE WORKING DIRECTORY. The dirty yamls hold
    # relative entries like `test/images`, exactly as the clean ones do. Left
    # relative, they resolve against wherever the caller happens to be, so
    # is_dir() is False, the directory branch is skipped, and the list-file
    # branch then tries to read a directory name as a text file. This is the
    # same failure as CLAUDE.md 4c, one layer further in: there the fix was to
    # drop `path:` from the yaml, here it is to resolve against the yaml's own
    # parent.
    if not entry.is_absolute():
        entry = yaml_path.parent / entry
    if entry.is_dir():
        return sorted(p for p in entry.iterdir() if p.suffix.lower() in IMG_EXT)
    return [Path(line) for line in entry.read_text().split() if line.strip()]


def label_for(image: Path) -> Path:
    """YOLO convention: .../images/x.jpg -> .../labels/x.txt."""
    parts = list(image.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def load_ground_truth(target: str, split: str = "test", kind: str = "main"):
    """Ground truth in letterboxed 640 space, with exclusions applied.

    Both backends are evaluated in the same letterboxed frame, so predictions
    and ground truth are always in one coordinate system. visual's
    `Welding Line` and `workpiece` are dropped here, from the ground truth
    itself, so they cannot be matched by anything downstream.
    """
    from PIL import Image

    excluded = C.excluded_ids(target)
    defects = C.defect_ids(target)

    images, gt_defect, gt_by_class, labels_img, split_blocks = [], {}, {}, [], []
    for i, path in enumerate(resolve_test_images(target, kind, split)):
        images.append(path)
        if kind == "dirty":
            split_blocks.append(f"dirty:{i}")
        else:
            rel = path.relative_to(C.CANDIDATE).as_posix()
            block = split_block_map().get(rel)
            if block is None:
                raise RuntimeError(f"no split block recorded for {rel}")
            split_blocks.append(block)
        lf = label_for(path)
        rows = []
        if lf.is_file():
            for line in lf.read_text().splitlines():
                p = line.split()
                if len(p) >= 5:
                    rows.append([float(v) for v in p[:5]])
        arr = np.array(rows, dtype=np.float32) if rows else np.zeros((0, 5), np.float32)

        with Image.open(path) as im:
            w0, h0 = im.size
        scale = min(C.IMGSZ / w0, C.IMGSZ / h0)
        px, py = (C.IMGSZ - round(w0 * scale)) // 2, (C.IMGSZ - round(h0 * scale)) // 2

        keep = np.array([int(c) not in excluded for c in arr[:, 0]], dtype=bool) \
            if len(arr) else np.zeros(0, dtype=bool)
        arr = arr[keep]

        if len(arr):
            cx, cy = arr[:, 1] * w0, arr[:, 2] * h0
            bw, bh = arr[:, 3] * w0, arr[:, 4] * h0
            box = np.stack([(cx - bw / 2) * scale + px, (cy - bh / 2) * scale + py,
                            (cx + bw / 2) * scale + px, (cy + bh / 2) * scale + py], 1)
            cls = arr[:, 0].astype(int)
        else:
            box, cls = np.zeros((0, 4), np.float32), np.zeros(0, dtype=int)

        gt_defect[i] = box[np.isin(cls, list(defects))] if len(cls) else box
        for c in np.unique(cls):
            gt_by_class.setdefault(int(c), {})[i] = box[cls == c]
        labels_img.append(bool(len(gt_defect[i])))

    # every image must appear for every class, so absent images count as
    # negatives rather than being skipped
    for c in gt_by_class:
        for i in range(len(images)):
            gt_by_class[c].setdefault(i, np.zeros((0, 4), np.float32))

    return {"images": images, "gt_defect": gt_defect, "gt_by_class": gt_by_class,
            "image_labels": np.array(labels_img, dtype=bool),
            "split_blocks": np.array(split_blocks, dtype=str)}


def predict_ultralytics(weights: Path, images, source: str):
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path.home() / ".ultralytics_cfg"))
    from ultralytics import RTDETR, YOLO
    cls = RTDETR if "rtdetr" in str(weights) else YOLO
    model = cls(str(weights))
    out = []
    for start in range(0, len(images), 32):
        chunk = [str(p) for p in images[start:start + 32]]
        for j, res in enumerate(model.predict(chunk, imgsz=C.IMGSZ, conf=C.CONF_FLOOR,
                                              iou=C.NMS_IOU, max_det=C.MAX_DET, verbose=False,
                                              device=0)):
            b = res.boxes
            if b is None or len(b) == 0:
                out.append((np.zeros((0, 4), np.float32), np.zeros(0, np.float32),
                            np.zeros(0, int)))
                continue
            xyxy = b.xyxy.cpu().numpy()
            h0, w0 = res.orig_shape
            scale = min(C.IMGSZ / w0, C.IMGSZ / h0)
            px, py = (C.IMGSZ - round(w0 * scale)) // 2, (C.IMGSZ - round(h0 * scale)) // 2
            xyxy = xyxy * scale + np.array([px, py, px, py], dtype=np.float32)
            out.append((xyxy.astype(np.float32), b.conf.cpu().numpy(),
                        b.cls.cpu().numpy().astype(int)))
    return out


def predict_torchvision(weights: Path, images, source: str):
    import torch
    from PIL import Image
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    nc = ckpt["nc"]
    model = fasterrcnn_resnet50_fpn_v2(
        weights=None, weights_backbone=None,
        min_size=C.IMGSZ, max_size=C.IMGSZ,
        box_score_thresh=C.CONF_FLOOR,
        box_nms_thresh=C.NMS_IOU,
        box_detections_per_img=C.MAX_DET,
    )
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, nc + 1)
    model.load_state_dict(ckpt["model"])
    model.cuda().eval()
    model.roi_heads.score_thresh = C.CONF_FLOOR
    model.roi_heads.nms_thresh = C.NMS_IOU
    model.roi_heads.detections_per_img = C.MAX_DET

    out = []
    with torch.no_grad():
        for start in range(0, len(images), 16):
            batch = []
            for p in images[start:start + 16]:
                with Image.open(p) as im:
                    canvas, _, _, _ = letterbox(im.convert("RGB"), C.IMGSZ)
                batch.append(torch.from_numpy(
                    np.asarray(canvas, np.float32).transpose(2, 0, 1) / 255.0).cuda())
            for res in model(batch):
                # subtract the background offset added at training time
                out.append((res["boxes"].cpu().numpy().astype(np.float32),
                            res["scores"].cpu().numpy(),
                            res["labels"].cpu().numpy().astype(int) - 1))
    return out


def flatten(preds, keep_ids=None, collapse: bool = False):
    """Stack per-image predictions into flat arrays for the metric functions.

    keep_ids filters to the source's defect classes. collapse then merges them
    into one class and runs class-agnostic NMS, which is the box level
    harmonization.
    """
    img, box, score = [], [], []
    for i, (b, s, c) in enumerate(preds):
        if len(b) == 0:
            continue
        if keep_ids is not None:
            m = np.isin(c, list(keep_ids))
            b, s = b[m], s[m]
        if len(b) == 0:
            continue
        if collapse:
            k = M.class_agnostic_nms(b, s)
            b, s = b[k], s[k]
        img.append(np.full(len(b), i, dtype=int))
        box.append(b)
        score.append(s)
    if not img:
        return np.zeros(0, int), np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    return np.concatenate(img), np.concatenate(box), np.concatenate(score)


def flatten_raw(preds):
    """All backend predictions, before label harmonization."""
    img, box, score, cls = [], [], [], []
    for i, (b, s, c) in enumerate(preds):
        if not len(s):
            continue
        img.append(np.full(len(s), i, dtype=np.int32))
        box.append(b.astype(np.float32))
        score.append(s.astype(np.float32))
        cls.append(c.astype(np.int16))
    if not img:
        return (np.zeros(0, np.int32), np.zeros((0, 4), np.float32),
                np.zeros(0, np.float32), np.zeros(0, np.int16))
    return (np.concatenate(img), np.concatenate(box), np.concatenate(score),
            np.concatenate(cls))


def evaluate(cell_dir: Path, targets=None) -> dict:
    record = json.loads((cell_dir / "cell.json").read_text())
    cell = record["cell"]
    source, family = cell["source"], cell["family"]
    weights = Path(record["weights"])
    if not weights.is_file():
        sys.exit(f"missing checkpoint: {weights}")
    if record.get("config_sha256_16") != C.config_fingerprint():
        sys.exit("configuration changed after training. Refusing to evaluate "
                 "a checkpoint under a different protocol")
    code_root = Path(__file__).resolve().parent
    for filename in ("config.py", "evaluate_cell.py", "metrics.py",
                     "yolo_dataset.py"):
        if record.get("code_sha256", {}).get(filename) != file_sha256(
                code_root / filename):
            sys.exit(f"{filename} changed after training. Refusing to mix "
                     "code versions within an evaluation")

    if cell.get("kind") == "dirty":
        current_splits = json.loads(
            (C.PROJECT / "grid" / "dirty" / "split_hashes.json").read_text()
        )["datasets"][cell["source"]]
    else:
        current_splits = json.loads((C.CANDIDATE / "split_hashes.json").read_text())
    if record.get("split_hashes") != current_splits:
        sys.exit("split hashes changed after training. Refusing mismatched evaluation")

    src_defect_ids = C.defect_ids(source)
    src_classes = C.DATASETS[source]["classes"]
    predict = (predict_torchvision if C.FAMILIES[family]["backend"] == "torchvision"
               else predict_ultralytics)

    results = {"cell": cell, "name": record["name"],
               "config_sha256_16": record.get("config_sha256_16"),
               "image_endpoint": ("at least one retained annotation target "
                                  "is present, not a universal part verdict"),
               "postprocessing": {
                   "confidence_floor": C.CONF_FLOOR,
                   "nms_iou": C.NMS_IOU,
                   "max_detections_per_image": C.MAX_DET,
               },
               "targets": {}}
    raw = {}

    plan = C.eval_plan(cell)
    if targets:
        plan = [item for item in plan if item[1] in targets]
    for result_key, target, test_split in plan:
        # A downloaded-data cell appears twice here, on the same target: once
        # on the split it shipped with, once on the audited split. Only the
        # test set changes between them, so the pair isolates the grading
        # partition from the training data.
        gt_kind = "dirty" if test_split == "dirty" else "main"
        gt = load_ground_truth(target, kind=gt_kind)
        preds = predict(weights, gt["images"], source)
        n_images = len(gt["images"])

        # ---- image level: annotated target feature present -------------
        # Score is the strongest defect-class prediction anywhere in the
        # image. Invariant to box protocol, which is the whole point.
        scores = np.zeros(n_images, dtype=np.float32)
        for i, (b, s, c) in enumerate(preds):
            if len(s):
                m = np.isin(c, list(src_defect_ids))
                scores[i] = float(s[m].max()) if m.any() else 0.0
        image_level = M.binary_metrics(scores, gt["image_labels"],
                                       C.MIN_NEGATIVES_FOR_FPR, C.ECE_BINS)

        # Keyed by result_key, not target. A downloaded-data cell evaluates the
        # same target twice, so keying by target would have the audited pass
        # silently overwrite the downloaded one in the sidecar.
        prefix = result_key + "__"
        ri, rb, rs, rc = flatten_raw(preds)
        raw[prefix + "pred_image"] = ri
        raw[prefix + "pred_box"] = rb
        raw[prefix + "pred_score"] = rs
        raw[prefix + "pred_class"] = rc
        raw[prefix + "image_score"] = scores
        raw[prefix + "image_label"] = gt["image_labels"].astype(np.uint8)
        raw[prefix + "image_name"] = np.array([p.name for p in gt["images"]])
        raw[prefix + "split_block"] = gt["split_blocks"]

        # ---- box level: class agnostic ---------------------------------
        same_protocol = (C.DATASETS[source]["protocol"] ==
                         C.DATASETS[target]["protocol"] == "localisation")
        if same_protocol:
            pi, pb, ps = flatten(preds, src_defect_ids, collapse=True)
            box_level = M.average_precision_range(pi, pb, ps, gt["gt_defect"])
            box_level.update(M.recall_at_fp_budget(
                pi, pb, ps, gt["gt_defect"], n_images, C.FP_PER_IMAGE_BUDGETS))
        else:
            box_level = {"skipped": True, "reason":
                         f"source protocol {C.DATASETS[source]['protocol']}, "
                         f"target protocol {C.DATASETS[target]['protocol']}. "
                         "A box level number across protocols measures "
                         "annotation mismatch, not domain shift."}

        entry = {"image_level": image_level, "box_level": box_level,
                 "n_test_images": n_images}

        # ---- LoHi only: the line's own pass/fail verdict -----------------
        # A second image level label, on the low resolution subset where the
        # filename carries it. Box presence makes LoHi 97.6 percent positive
        # and leaves 11 negatives, too few for a false alarm rate. The verdict
        # is 136 negatives against 165 positives in the test split, so AUROC
        # is reportable here and nowhere else on LoHi.
        #
        # OK and NG parts are annotated almost identically, so expect a score
        # near chance. That is the result, not a failure of the pipeline:
        # annotation presence is not a proxy for part acceptance.
        if target == "lohi_weld":
            import re as _re
            rx = _re.compile(C.LOHI_VERDICT_PATTERN)
            idx, verdict = [], []
            for i, path in enumerate(gt["images"]):
                m = rx.match(path.stem)
                if m:
                    idx.append(i)
                    verdict.append(m.group(2) == "NG")
            if idx:
                sub = M.binary_metrics(scores[np.array(idx)],
                                       np.array(verdict, dtype=bool),
                                       C.MIN_NEGATIVES_FOR_FPR, C.ECE_BINS)
                sub["subset"] = C.LOHI_VERDICT_SUBSET
                sub["definition"] = ("positive means the production line "
                                     "rejected the part (NG), from the cycle "
                                     "filename, not from box presence")
                sub["n_highres_excluded"] = n_images - len(idx)
                entry["verdict_level"] = sub

        # ---- in domain only: per class AP with instance counts ----------
        if target == source:
            per_class = {}
            for cid, name in enumerate(src_classes):
                if name in C.DATASETS[source]["excluded"]:
                    continue
                gts = gt["gt_by_class"].get(cid, {})
                n_inst = sum(len(v) for v in gts.values())
                sel = flatten([(b[c == cid], s[c == cid], c[c == cid])
                               for b, s, c in preds])
                ap = M.average_precision_range(*sel, gts)
                per_class[name] = {
                    "map50": ap["map50"], "map50_95": ap["map50_95"],
                    "test_instances": int(n_inst),
                    "thin": bool(n_inst < C.MIN_TEST_INSTANCES),
                }
                if (n_inst < C.MIN_TEST_INSTANCES and
                        cell.get("kind", "main") != "dirty"):
                    # Run after retrieval on the laptop. Holding the H200 while
                    # 2,000 CPU bootstrap draws execute would waste the slice.
                    # The raw sidecar contains every prediction needed.
                    per_class[name]["cluster_bootstrap_required"] = True
            entry["per_class_in_domain"] = per_class

        entry["target"] = target
        entry["test_split"] = ("downloaded" if test_split == "dirty"
                               else "audited")
        results["targets"][result_key] = entry
        il = image_level
        bl = box_level.get("map50")
        print(f"    {result_key:<32} [{entry['test_split']:<10}] "
              f"image AP {il['ap']:.3f}"
              f"  AUROC {il['auroc'] if il['auroc'] is None else round(il['auroc'], 3)}"
              f"  box mAP50 {'skipped' if bl is None else round(bl, 3)}")

    return results, raw


def json_safe(value):
    """Replace nonfinite floats with null so the output is strict JSON."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=int)
    ap.add_argument("--name")
    ap.add_argument("--runs-dir", type=Path, default=C.RUNS)
    ap.add_argument("--out-dir", type=Path, default=C.RESULTS)
    ap.add_argument("--targets", nargs="+", choices=C.ALL_TARGETS)
    args = ap.parse_args()

    if args.index is not None:
        name = C.cell_name(C.all_cells()[args.index])
    elif args.name:
        name = args.name
    else:
        sys.exit("give --index or --name")

    cell_dir = args.runs_dir / name
    if not (cell_dir / "cell.json").is_file():
        sys.exit(f"no trained cell at {cell_dir}")

    print(f"=== evaluating {name} ===")
    results, raw = evaluate(cell_dir, args.targets)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"eval__{name}.json"
    out.write_text(json.dumps(json_safe(results), indent=2))
    raw_out = args.out_dir / f"raw__{name}.npz"
    np.savez_compressed(raw_out, **raw)
    print(f"    wrote {out}")
    print(f"    wrote {raw_out}")


if __name__ == "__main__":
    main()
