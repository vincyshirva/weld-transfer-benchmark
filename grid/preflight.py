#!/usr/bin/env python3
"""Read-only launch gate for the weld generalization grid."""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C  # noqa: E402
from make_dirty_yaml import verify_snapshot  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_clean_splits() -> None:
    expected = json.loads((C.CANDIDATE / "split_hashes.json").read_text())
    for name, wanted in expected.items():
        path = C.CANDIDATE / "splits" / name
        if sha256(path) != wanted:
            raise RuntimeError(f"clean split hash mismatch: {path}")
    print(f"PASS clean split hashes: {len(expected)}")


def check_yaml(path: Path) -> None:
    data = yaml.safe_load(path.read_text())
    if "path" in data:
        raise RuntimeError(f"unsafe path key in {path}")
    for key in ("train", "val", "test"):
        entry = Path(data[key])
        resolved = entry if entry.is_absolute() else path.parent / entry
        if not resolved.exists():
            raise RuntimeError(f"missing {key} path from {path}: {resolved}")


def check_data() -> None:
    for tag in C.ALL_TARGETS:
        check_yaml(C.CANDIDATE / tag / "data.yaml")
    for tag in C.DIRTY_SOURCES:
        check_yaml(Path(__file__).resolve().parent / "dirty" / tag / "data.yaml")
    verify_snapshot()
    print("PASS clean and portable dirty dataset paths")


def check_test_resolution() -> None:
    """Every cell's test images must resolve from ANY working directory.

    This is the check that was missing on 5 September 2026. The dirty yamls
    hold relative entries, `resolve_test_images` did not anchor them to the
    yaml's parent, and they silently resolved against the caller's working
    directory. Every main and reference cell passed because the clean branch
    uses absolute paths, so the fault only surfaced 20 hours in, at cell 60.
    Resolving from a foreign directory here reproduces it in a second.
    """
    import os
    import tempfile

    from evaluate_cell import resolve_test_images

    here = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            for tag in C.ALL_TARGETS:
                n = len(resolve_test_images(tag, "main", "test"))
                if n == 0:
                    raise RuntimeError(f"clean test split resolved empty: {tag}")
            for tag in C.DIRTY_SOURCES:
                for split in ("train", "valid", "test"):
                    n = len(resolve_test_images(tag, "dirty", split))
                    if n == 0:
                        raise RuntimeError(
                            f"dirty {split} split resolved empty: {tag}. A "
                            "relative yaml entry is being resolved against the "
                            "working directory.")
    finally:
        os.chdir(here)
    print("PASS test splits resolve from any working directory")


def check_weights() -> None:
    records = C.pretrained_records()
    if set(records) != set(C.FAMILIES):
        raise RuntimeError("pretrained manifest does not cover every family")
    for family, record in records.items():
        path = C.pretrained_path(family)
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            raise RuntimeError(f"missing or wrong-sized checkpoint: {path}")
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {path}")
    print(f"PASS pinned pretrained checkpoints: {len(records)}")


def check_model_loading() -> None:
    """Deserialize every checkpoint without fitting or allocating a GPU."""
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from ultralytics import RTDETR, YOLO

    YOLO(str(C.pretrained_path("yolo11n")))
    YOLO(str(C.pretrained_path("yolo11s")))
    RTDETR(str(C.pretrained_path("rtdetr_l")))

    model = fasterrcnn_resnet50_fpn_v2(
        weights=None, weights_backbone=None,
        min_size=C.IMGSZ, max_size=C.IMGSZ,
        box_score_thresh=C.CONF_FLOOR,
        box_nms_thresh=C.NMS_IOU,
        box_detections_per_img=C.MAX_DET,
    )
    state = torch.load(C.pretrained_path("faster_rcnn"),
                       map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    if tuple(model.transform.min_size) != (C.IMGSZ,):
        raise RuntimeError("Faster R-CNN minimum image size is not frozen")
    if model.transform.max_size != C.IMGSZ:
        raise RuntimeError("Faster R-CNN maximum image size is not frozen")
    if model.roi_heads.nms_thresh != C.NMS_IOU:
        raise RuntimeError("Faster R-CNN NMS differs from the common protocol")
    print("PASS checkpoint deserialization and Faster R-CNN transform")


def check_code() -> None:
    root = Path(__file__).resolve().parent
    files = sorted(root.glob("*.py"))
    for path in files:
        ast.parse(path.read_text(), filename=str(path))
    slurm = (root / "run_grid.slurm").read_text()
    expected = len(C.all_cells()) - 1
    if f"#SBATCH --array=0-{expected}%1" not in slurm:
        raise RuntimeError("Slurm array range does not match configured cells")
    if f"#SBATCH --cpus-per-task={C.WORKERS}" not in slurm:
        raise RuntimeError("Slurm CPU request does not match data loader workers")
    print(f"PASS code syntax and array shape: {len(C.all_cells())} cells")


def check_chain() -> None:
    """The chain script is what actually gets submitted.

    faculty_qos caps submitted jobs at 2 and counts each array task as one, so
    the 69 task array in run_grid.slurm is rejected by the scheduler. The chain
    runs the same cells sequentially and queues one successor. These checks
    guard the properties that make that safe to leave unattended.
    """
    root = Path(__file__).resolve().parent
    path = root / "run_grid_chain.slurm"
    if not path.is_file():
        raise RuntimeError("missing run_grid_chain.slurm, the submitted script")
    text = path.read_text()

    if f"#SBATCH --cpus-per-task={C.WORKERS}" not in text:
        raise RuntimeError("chain CPU request does not match data loader workers")
    # Only real directives count. The prose above explains why the array was
    # abandoned and legitimately contains the word.
    directives = [ln.strip() for ln in text.splitlines()
                  if ln.strip().startswith("#SBATCH")]
    if any("--array" in ln for ln in directives):
        raise RuntimeError("chain script must not declare an array, the QOS "
                           "counts every task against a limit of 2")
    # The successor must be queued with afterany, otherwise a wall clock kill
    # or a node fault silently ends the grid partway through.
    if "--dependency=afterany:${SLURM_JOB_ID}" not in text:
        raise RuntimeError("chain does not queue its successor with afterany")
    if "run_grid_chain.slurm" not in text.split("SUCCESSOR=", 1)[-1][:400]:
        raise RuntimeError("chain successor does not resubmit the chain script")
    # It must derive the cell count rather than hardcode it, or the chain and
    # the configuration can silently disagree.
    if "len(C.all_cells())" not in text:
        raise RuntimeError("chain does not derive its cell count from config")
    # A cell counts as done only with all three artefacts, matching the
    # per-cell contract in run_grid.slurm.
    for artefact in ("cell.json", "eval__${CELL}.json",
                     "raw__${CELL}.npz", "${CELL}.tar.gz"):
        if artefact not in text:
            raise RuntimeError(f"chain does not check for {artefact}")
    print("PASS chain script: no array, afterany successor, 3 artefacts per cell")


def main() -> None:
    check_code()
    check_chain()
    check_clean_splits()
    check_data()
    check_test_resolution()
    check_weights()
    check_model_loading()
    print(f"PASS frozen configuration: {C.config_fingerprint()}")
    print("PREFLIGHT PASS")


if __name__ == "__main__":
    main()
