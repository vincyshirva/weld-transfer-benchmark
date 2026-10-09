#!/usr/bin/env python3
"""
Train one cell of the generalization grid.

A cell is one (source, family, seed). It trains on that source's own label
set, using the frozen hyperparameters in config.py, and writes a checkpoint
plus a small provenance record. It computes no transfer metric. Evaluation is
a separate pass so that a model is never retrained to be measured again.

Usage:
    python3 train_cell.py --index 7                  # by position in the grid
    python3 train_cell.py --source kaggle_v2 --family yolo11s --seed 0
    python3 train_cell.py --index 7 --smoke          # 2 epochs, tiny, for CI

Outputs, under runs/<cell_name>/:
    weights/best.pt      the checkpoint evaluation will load
    cell.json            provenance: config hash, split hashes, timings
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C  # noqa: E402

# Faster R-CNN needs 26.4 GB at the frozen batch of 32, so a smoke test at
# that batch can only run on the cluster. Smoke mode exists to prove the code
# path anywhere, so it drops the batch. It is never a result either way.
SMOKE_BATCH = 2


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_fingerprint(cell: dict) -> dict:
    """The split hashes this cell trained against.

    Reproducibility rests on the frozen split files, not on re-running the
    solver, so every checkpoint records which splits it saw. If a future
    rebuild changes them, results computed here become non-comparable and this
    record is how that gets caught.
    """
    if cell.get("kind") == "dirty":
        path = Path(__file__).resolve().parent / "dirty" / "split_hashes.json"
        if not path.is_file():
            return {}
        return json.loads(path.read_text())["datasets"][cell["source"]]
    path = C.CANDIDATE / "split_hashes.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_pretrained(cell: dict, smoke: bool) -> dict | None:
    """Require the released checkpoint and verify it before fitting."""
    if not C.PRETRAINED or smoke:
        return None
    path = C.pretrained_path(cell["family"])
    record = C.pretrained_records().get(cell["family"])
    if record is None or not path.is_file():
        sys.exit(f"missing pinned pretrained checkpoint or manifest entry: {path}")
    got = file_sha256(path)
    if got != record.get("sha256"):
        sys.exit(f"pretrained checkpoint hash mismatch: {path}")
    return {**record, "path": str(path)}


def code_fingerprints() -> dict:
    """Hash every executable grid source used by a cell."""
    root = Path(__file__).resolve().parent
    return {p.name: file_sha256(p) for p in sorted(root.glob("*.py"))}


def train_ultralytics(cell: dict, out: Path, smoke: bool) -> dict:
    """YOLO11 at n and s, and RT-DETR. One code path, three configs."""
    # /tmp/Ultralytics on the H200 node belongs to another user, so without
    # this every import errors and settings never persist. See the project log 4a.
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path.home() / ".ultralytics_cfg"))
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)

    import ultralytics.engine.trainer as _trainer
    # The AMP probe downloads a reference checkpoint mid-run. AMP is set
    # explicitly in the frozen config, so skip the probe rather than depend on
    # the network 36 times.
    _trainer.check_amp = lambda *_a, **_k: True
    from ultralytics import RTDETR, YOLO

    spec = C.FAMILIES[cell["family"]]
    cls = RTDETR if cell["family"].startswith("rtdetr") else YOLO
    weights = (str(C.pretrained_path(cell["family"]))
               if C.PRETRAINED and not smoke else spec["cfg"])
    model = cls(weights)

    hp = dict(C.ULTRALYTICS_HP)
    hp["pretrained"] = C.PRETRAINED and not smoke
    args = dict(
        data=str(C.data_yaml(cell)),
        imgsz=C.IMGSZ, batch=SMOKE_BATCH if smoke else C.BATCH,
        epochs=2 if smoke else C.EPOCHS,
        workers=2 if smoke else C.WORKERS,
        patience=C.PATIENCE,
        seed=cell["seed"], device=0,
        project=str(out.parent), name=out.name, exist_ok=True,
        plots=False, save=True, val=True, verbose=False,
        **{k: v for k, v in hp.items() if k != "pretrained"},
    )
    if smoke:
        args["fraction"] = 0.05

    t0 = time.perf_counter()
    model.train(**args)
    return {"train_seconds": round(time.perf_counter() - t0, 1),
            "weights": str(out / "weights" / "best.pt")}


def train_torchvision(cell: dict, out: Path, smoke: bool) -> dict:
    """Faster R-CNN R50-FPN v2, from torchvision.

    Ultralytics has no Faster R-CNN, so this family needs its own loop. It
    reads the same YOLO-format tree, so the splits and labels are byte
    identical to what the other three families see. Only the model and the
    optimizer differ, which is the point of having a third family.
    """
    import torch
    from torch.utils.data import DataLoader
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    import metrics as M
    from yolo_dataset import YoloDetection, collate  # local module

    hp = C.TORCHVISION_HP
    nc = len(C.DATASETS[cell["source"]]["classes"])

    model = fasterrcnn_resnet50_fpn_v2(
        weights=None, weights_backbone=None,
        trainable_backbone_layers=hp["trainable_backbone_layers"],
        min_size=C.IMGSZ, max_size=C.IMGSZ,
        box_score_thresh=C.CONF_FLOOR,
        box_nms_thresh=C.NMS_IOU,
        box_detections_per_img=C.MAX_DET,
    )
    if C.PRETRAINED and not smoke:
        state = torch.load(C.pretrained_path(cell["family"]),
                           map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    # +1 for background, which torchvision reserves as class 0.
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, nc + 1)
    model.cuda()

    # Faster R-CNN appears in the main and reference blocks. Both use the
    # audited clean tree. Dirty baselines alone are restricted to AUX_FAMILY.
    root = C.CANDIDATE / cell["source"]
    train_ds = YoloDetection(root / "train", C.IMGSZ, train=True)
    val_ds = YoloDetection(root / "valid", C.IMGSZ, train=False)
    if smoke:
        train_ds = torch.utils.data.Subset(train_ds, range(min(64, len(train_ds))))
        val_ds = torch.utils.data.Subset(val_ds, range(min(32, len(val_ds))))

    workers = 2 if smoke else C.WORKERS
    batch = SMOKE_BATCH if smoke else C.BATCH
    train_ld = DataLoader(train_ds, batch_size=batch, shuffle=True,
                          num_workers=workers, collate_fn=collate, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=batch, shuffle=False,
                        num_workers=workers, collate_fn=collate)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=hp["lr"], momentum=hp["momentum"],
                          weight_decay=hp["weight_decay"])
    epochs = 2 if smoke else C.EPOCHS
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=hp["amp"])

    best = float("-inf")
    best_path = out / "weights" / "best.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    history, bad_epochs = [], 0

    t0 = time.perf_counter()
    step = 0
    for epoch in range(epochs):
        model.train()
        run = 0.0
        for images, targets in train_ld:
            images = [i.cuda(non_blocking=True) for i in images]
            targets = [{k: v.cuda(non_blocking=True) for k, v in t.items()}
                       for t in targets]
            # Linear warmup, matching the ultralytics recipe in spirit.
            if step < hp["warmup_iters"]:
                for g in opt.param_groups:
                    g["lr"] = hp["lr"] * (step + 1) / hp["warmup_iters"]
            with torch.amp.autocast("cuda", enabled=hp["amp"]):
                loss = sum(model(images, targets).values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += loss.item()
            step += 1
        sched.step()

        # Select checkpoints by native-class validation mAP50:95, matching the
        # detection-quality criterion used by the Ultralytics families. A
        # validation-loss checkpoint would introduce a family-specific model
        # selection confound.
        model.eval()
        pred_img, pred_box, pred_score, pred_class = [], [], [], []
        gt_by_class = {c: {} for c in range(1, nc + 1)}
        image_id = 0
        with torch.no_grad():
            for images, targets in val_ld:
                images = [i.cuda(non_blocking=True) for i in images]
                with torch.amp.autocast("cuda", enabled=hp["amp"]):
                    outputs = model(images)
                for target, output in zip(targets, outputs):
                    gt_boxes = target["boxes"].numpy()
                    gt_labels = target["labels"].numpy()
                    for cid in range(1, nc + 1):
                        gt_by_class[cid][image_id] = gt_boxes[gt_labels == cid]
                    scores = output["scores"].cpu().numpy()
                    if len(scores):
                        pred_img.append(np.full(len(scores), image_id, dtype=int))
                        pred_box.append(output["boxes"].cpu().numpy())
                        pred_score.append(scores)
                        pred_class.append(output["labels"].cpu().numpy())
                    image_id += 1
        pi = np.concatenate(pred_img) if pred_img else np.zeros(0, dtype=int)
        pb = (np.concatenate(pred_box) if pred_box
              else np.zeros((0, 4), dtype=np.float32))
        ps = (np.concatenate(pred_score) if pred_score
              else np.zeros(0, dtype=np.float32))
        pc = (np.concatenate(pred_class) if pred_class
              else np.zeros(0, dtype=np.int64))
        class_maps = []
        for cid in range(1, nc + 1):
            metric = M.average_precision_range(
                pi[pc == cid], pb[pc == cid], ps[pc == cid],
                gt_by_class[cid])
            if metric["n_gt"]:
                class_maps.append(metric["map50_95"])
        val = float(np.mean(class_maps)) if class_maps else float("-inf")
        history.append({"epoch": epoch, "train_loss": run / max(1, len(train_ld)),
                        "val_map50_95": val})

        if val > best:
            best, bad_epochs = val, 0
            torch.save({"model": model.state_dict(), "nc": nc,
                        "source": cell["source"], "epoch": epoch}, best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= C.PATIENCE:
                break

    (out / "history.json").write_text(json.dumps(history, indent=2))
    return {"train_seconds": round(time.perf_counter() - t0, 1),
            "weights": str(best_path), "best_val_map50_95": best,
            "epochs_run": len(history)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=int, help="position in the grid, 0 based")
    ap.add_argument("--source", choices=C.ALL_TARGETS)
    ap.add_argument("--kind", default="main",
                    choices=["main", "reference", "dirty"])
    ap.add_argument("--family", choices=list(C.FAMILIES))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--smoke", action="store_true",
                    help="2 epochs on a slice of the data, at SMOKE_BATCH, to "
                         "prove the path runs on any GPU. Never a result.")
    ap.add_argument("--runs-dir", type=Path, default=C.RUNS)
    args = ap.parse_args()

    all_cells = C.all_cells()
    if args.index is not None:
        if not 0 <= args.index < len(all_cells):
            sys.exit(f"index must be 0..{len(all_cells) - 1}")
        cell = all_cells[args.index]
    elif args.source and args.family and args.seed is not None:
        cell = {"source": args.source, "family": args.family,
                "seed": args.seed, "kind": args.kind}
    else:
        sys.exit("give --index, or all of --source --family --seed")

    name = C.cell_name(cell)
    out = args.runs_dir / ("smoke_" + name if args.smoke else name)
    out.mkdir(parents=True, exist_ok=True)

    done = out / "cell.json"
    if done.is_file() and not args.smoke:
        # Idempotent by design. A 69 cell array can be resubmitted after a
        # node fault or a time limit, and finished cells must not be redone.
        print(f"already complete, skipping: {name}")
        return

    print(f"=== cell {name} ===")
    print(f"    kind {cell.get('kind', 'main')}, source {cell['source']}, "
          f"family {cell['family']}, seed {cell['seed']}")
    print(f"    data {C.data_yaml(cell)}")
    set_all_seeds(cell["seed"])
    pretrained = verified_pretrained(cell, args.smoke)

    backend = C.FAMILIES[cell["family"]]["backend"]
    fn = train_ultralytics if backend == "ultralytics" else train_torchvision

    # On the laptop, this snapshot is built as a symlink farm from the
    # originals. It is transferred to the server with links dereferenced. The
    # content snapshot is always checked. The original stat fingerprint is
    # also checked when the originals happen to exist on the compute host.
    dirty = cell.get("kind") == "dirty"
    if dirty:
        from make_dirty_yaml import (fingerprint, originals_present,
                                     verify_snapshot)
        snapshot_before = verify_snapshot(cell["source"])
        original_before = fingerprint() if originals_present() else None

    result = fn(cell, out, args.smoke)

    if dirty:
        snapshot_after = verify_snapshot(cell["source"])
        if snapshot_before != snapshot_after:
            sys.exit("FATAL: the portable dirty snapshot changed during training")
        if original_before is not None and original_before != fingerprint():
            sys.exit("FATAL: an original dataset folder changed during a dirty "
                     "cell. Training must never write there. Check for a "
                     "labels.cache under the original tree.")
        result["dirty_snapshot_verified"] = True
        result["originals_verified_unchanged"] = original_before is not None
        result["originals_present_on_compute_host"] = original_before is not None

    record = {
        "cell": cell,
        "name": name,
        "backend": backend,
        "smoke": args.smoke,
        "config_sha256_16": C.config_fingerprint(),
        "split_hashes": split_fingerprint(cell),
        "code_sha256": code_fingerprints(),
        "pretrained_checkpoint": pretrained,
        "hyperparameters": {
            "common": C.COMMON, "pretrained": C.PRETRAINED,
            "family": C.ULTRALYTICS_HP if backend == "ultralytics"
                      else C.TORCHVISION_HP,
        },
        "environment": {
            "python": platform.python_version(),
            "host": platform.node(),
            "slurm_job": os.environ.get("SLURM_JOB_ID"),
        },
        **result,
    }
    try:
        import torch
        record["environment"]["torch"] = torch.__version__
        record["environment"]["torch_cuda"] = torch.version.cuda
        if backend == "ultralytics":
            import ultralytics
            record["environment"]["ultralytics"] = ultralytics.__version__
        try:
            import torchvision
            record["environment"]["torchvision"] = torchvision.__version__
        except ImportError:
            pass
        record["environment"]["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass

    done.write_text(json.dumps(record, indent=2))
    print(f"    trained in {result['train_seconds']} s")
    print(f"    wrote {done}")


if __name__ == "__main__":
    main()
