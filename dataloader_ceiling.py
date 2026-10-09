#!/usr/bin/env python3
"""
How fast can the CPU feed a detector, with no GPU in the loop?

The H200 cluster caps a job at --cpus-per-task=2 and its guide states that
num_workers=2 is the verified maximum. YOLO mosaic augmentation composes four
images per sample on the CPU, so that cap is a hard ceiling on images per
second regardless of which MIG slice is granted.

Measuring it on the training laptop answers a question about the cluster,
because it is a property of the CPU pipeline and not of the GPU: it is the
rate above which a faster GPU buys nothing. Compare it against the measured
end to end training rate to see how much headroom a bigger slice would find.

This iterates the real ultralytics training dataloader over the frozen splits
and touches nothing else. No model, no CUDA, no gradients.

Usage:   python3 dataloader_ceiling.py [--workers 2 4 8] [--batches 60]
Outputs: dataloader_ceiling.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).parent
CANDIDATE = ROOT / "clean_candidate"


def set_root(root: Path) -> None:
    """Point the probe at a project root other than this file's folder."""
    global ROOT, CANDIDATE
    ROOT = root.resolve()
    CANDIDATE = ROOT / "clean_candidate"


def measure(tag: str, workers: int, batch: int, imgsz: int,
            n_batches: int, warmup: int) -> dict:
    """Iterate the real augmentation pipeline through a plain DataLoader.

    Ultralytics' own build_dataloader returns an InfiniteDataLoader whose
    sampler never raises StopIteration. Driving that by hand deadlocked here,
    so this uses a stock torch DataLoader over the same dataset object. The
    per-sample work, mosaic included, is identical, which is the thing being
    timed.
    """
    import copy

    from torch.utils.data import DataLoader
    from ultralytics.data import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils import DEFAULT_CFG

    data = check_det_dataset(str(CANDIDATE / tag / "data.yaml"))
    cfg = copy.copy(DEFAULT_CFG)      # DEFAULT_CFG is shared global state
    cfg.imgsz, cfg.mosaic, cfg.cache = imgsz, 1.0, False
    dataset = build_yolo_dataset(cfg, data["train"], batch, data, mode="train")
    loader = DataLoader(
        dataset, batch_size=batch, shuffle=True, num_workers=workers,
        collate_fn=getattr(dataset, "collate_fn", None),
        drop_last=True, persistent_workers=False,
    )

    seen = 0
    dt = 0.0
    t0 = None
    for i, b in enumerate(loader):
        if i == warmup:                # start the clock after warmup
            t0 = time.perf_counter()
        if t0 is not None:
            seen += b["img"].shape[0]
        if i >= warmup + n_batches - 1:
            break
    dt = time.perf_counter() - t0 if t0 else 0.0
    return {
        "dataset": tag, "workers": workers, "batch": batch, "imgsz": imgsz,
        "batches": n_batches, "images": seen,
        "seconds": round(dt, 2),
        "images_per_s": round(seen / dt, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+",
                    default=["lohi_weld", "weld_surface_v2"])
    ap.add_argument("--workers", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batches", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--root", type=Path,
                    help="project root holding clean_candidate/")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    if args.root:
        set_root(args.root)

    out = {"note": "CPU only. No model and no CUDA. This is the rate above "
                   "which a faster GPU cannot help.",
           "root": str(ROOT), "results": []}
    for tag in args.datasets:
        for w in args.workers:
            r = measure(tag, w, args.batch, args.imgsz, args.batches, args.warmup)
            out["results"].append(r)
            print(f"  {tag:<18} workers {w:>2}  {r['images_per_s']:>7.1f} img/s")

    path = args.out or (ROOT / "dataloader_ceiling.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
