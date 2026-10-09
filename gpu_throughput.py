#!/usr/bin/env python3
"""
Confirm GPU throughput for the cross-dataset training grid.

Step 4 of the plan asks one question: can this machine run roughly 60 to 90
training runs in acceptable wall clock time, and does every detector family
fit in memory. This script answers it by measurement, not by estimate.

Two stages, because they measure different things.

STAGE 1, synthetic forward and backward. Random tensors, no disk, no
augmentation. This isolates the compute cost of one image through one model,
and it is the number that fixes the batch size. It sweeps batch size upward
until the allocator runs out, so the reported maximum is the real one on this
card with the desktop already resident.

STAGE 2, real epochs. Ultralytics adds mosaic augmentation, letterboxing,
label loading and a validation pass every epoch. That overhead is often larger
than the model itself on small datasets, so extrapolating from stage 1 alone
would understate the grid. Stage 2 trains for a few epochs on a real frozen
split and times the steady-state ones.

Nothing here is an experiment. Weights are randomly initialised from YAML, so
no pretrained file is downloaded, and every result is discarded. Throughput
does not depend on initialisation.

Usage:
    python3 gpu_throughput.py [--stage 1|2|all] [--imgsz 640] [--epochs 4]

Outputs: gpu_throughput.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).parent
CANDIDATE = ROOT / "clean_candidate"
SCRATCH = ROOT / "_throughput_scratch"


def set_root(root: Path) -> None:
    """Point the probe at a project root other than this file's folder.

    On the cluster the scripts live in code/ while the data sits one level up,
    so the default of __file__.parent is wrong there.
    """
    global ROOT, CANDIDATE, SCRATCH
    ROOT = root.resolve()
    CANDIDATE = ROOT / "clean_candidate"
    SCRATCH = ROOT / "_throughput_scratch"

# The four configurations named in the plan: YOLO11 at n and s, RT-DETR,
# Faster R-CNN. Names are the ultralytics YAMLs, which build from scratch.
YOLO_CONFIGS = ["yolo11n.yaml", "yolo11s.yaml", "rtdetr-l.yaml"]
BATCHES = [4, 8, 16, 32, 64]

# Largest training source in the frozen splits, so stage 2 measures the worst
# case rather than the easiest one.
STAGE2_SOURCE = "lohi_weld"


def gpu_info() -> dict:
    if not torch.cuda.is_available():
        sys.exit("No CUDA device visible. Cannot confirm GPU throughput.")
    p = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info(0)
    return {
        "name": p.name,
        "compute_capability": f"{p.major}.{p.minor}",
        "multi_processors": p.multi_processor_count,
        "total_vram_gib": round(p.total_memory / 2**30, 2),
        "free_vram_gib_at_start": round(free / 2**30, 2),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "arch_list": torch.cuda.get_arch_list(),
        "driver_cuda": _driver_cuda(),
    }


def _driver_cuda() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20)
        return out.stdout.strip()
    except Exception:
        return "?"


def build_model(name: str, nc: int = 7):
    """Random-init model on the GPU. No pretrained weights are fetched."""
    if name == "fasterrcnn_resnet50_fpn":
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
        return fasterrcnn_resnet50_fpn_v2(
            weights=None, weights_backbone=None, num_classes=nc + 1).cuda()
    from ultralytics import YOLO, RTDETR
    cls = RTDETR if name.startswith("rtdetr") else YOLO
    return cls(name).model.cuda()


def _reduce(out) -> torch.Tensor:
    """Sum every tensor an ultralytics head returns, so backward is defined."""
    if torch.is_tensor(out):
        return out.float().sum()
    if isinstance(out, dict):
        out = list(out.values())
    if isinstance(out, (list, tuple)):
        parts = [_reduce(o) for o in out if torch.is_tensor(o)
                 or isinstance(o, (list, tuple, dict))]
        return torch.stack(parts).sum() if parts else torch.zeros((), device="cuda")
    return torch.zeros((), device="cuda")


def synthetic_step(name: str, model, batch: int, imgsz: int, amp: bool):
    """One forward and backward on random data. Raises on OOM.

    The ultralytics loss needs a populated hyperparameter namespace that only
    the trainer builds, so this drives the network directly with a tensor and
    reduces the raw head outputs instead. Backward then traverses the whole
    backbone, neck and head, which is what the batch size and VRAM ceiling
    actually depend on. The loss itself is a rounding error in compute, and
    stage 2 measures the real loss path anyway.
    """
    x = torch.rand(batch, 3, imgsz, imgsz, device="cuda")
    if name == "fasterrcnn_resnet50_fpn":
        targets = [{
            "boxes": torch.tensor([[10.0, 10.0, 100.0, 100.0]], device="cuda"),
            "labels": torch.ones(1, dtype=torch.int64, device="cuda"),
        } for _ in range(batch)]
        with torch.autocast("cuda", enabled=amp):
            loss = sum(model(list(x), targets).values())
    else:
        with torch.autocast("cuda", enabled=amp):
            loss = _reduce(model(x))
    loss.backward()
    model.zero_grad(set_to_none=True)


def stage1(imgsz: int, amp: bool, batches=None,
           warmup: int = 3, iters: int = 12) -> dict:
    results = {}
    batches = batches or BATCHES
    names = YOLO_CONFIGS + ["fasterrcnn_resnet50_fpn"]
    for name in names:
        entry = {"batches": {}, "max_batch": None, "best_ips": 0.0}
        for batch in batches:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                model = build_model(name)
                model.train()
                for _ in range(warmup):
                    synthetic_step(name, model, batch, imgsz, amp)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(iters):
                    synthetic_step(name, model, batch, imgsz, amp)
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                ips = iters * batch / dt
                peak = torch.cuda.max_memory_allocated() / 2**30
                entry["batches"][batch] = {
                    "images_per_s": round(ips, 1),
                    "ms_per_step": round(1000 * dt / iters, 1),
                    "peak_vram_gib": round(peak, 2),
                }
                entry["max_batch"] = batch
                entry["best_ips"] = max(entry["best_ips"], round(ips, 1))
                print(f"  {name:<28} batch {batch:>3}  "
                      f"{ips:>7.1f} img/s  peak {peak:>5.2f} GiB")
            except torch.OutOfMemoryError:
                entry["batches"][batch] = {"oom": True}
                print(f"  {name:<28} batch {batch:>3}  OOM")
                del model
                torch.cuda.empty_cache()
                break
            except Exception as exc:  # noqa: BLE001
                entry["batches"][batch] = {"error": f"{type(exc).__name__}: {exc}"}
                print(f"  {name:<28} batch {batch:>3}  "
                      f"ERROR {type(exc).__name__}: {str(exc)[:90]}")
                break
            finally:
                if "model" in dir():
                    del model
                torch.cuda.empty_cache()
        results[name] = entry
    return results


def stage2(imgsz: int, epochs: int, batch: int, workers: int = 8) -> dict:
    """Time real epochs on a frozen split, including dataloading and val.

    `workers` matters more than it looks. The H200 cluster caps a job at
    --cpus-per-task=2 and tells users to set num_workers=2. Mosaic augmentation
    composes four images per sample on the CPU, so two workers can starve a
    fast GPU. Running this at 2 and at 8 on the same machine separates the
    model cost from the dataloader cost, and that difference is what predicts
    whether a bigger MIG slice would actually be used.
    """
    # Ultralytics probes AMP correctness by downloading a small reference
    # checkpoint. That is a network fetch this benchmark does not need and
    # must not depend on, and it decides a setting we want to control
    # explicitly. Assert AMP on and skip the probe.
    import ultralytics.engine.trainer as _trainer
    _trainer.check_amp = lambda *_a, **_k: True
    from ultralytics import YOLO

    data = CANDIDATE / STAGE2_SOURCE / "data.yaml"
    if not data.is_file():
        return {"skipped": f"missing {data}"}
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)

    # Time each epoch directly with callbacks. The results.csv `time` column
    # is cumulative and its schema moves between releases, and parsing it gave
    # negative deltas here. A callback pair cannot drift.
    marks: list[float] = []
    epoch_s: list[float] = []

    def _start(_trainer_obj):
        marks.append(time.perf_counter())

    def _end(_trainer_obj):
        if marks:
            epoch_s.append(time.perf_counter() - marks[-1])

    t0 = time.perf_counter()
    model = YOLO("yolo11s.yaml")
    model.add_callback("on_train_epoch_start", _start)
    model.add_callback("on_fit_epoch_end", _end)
    model.train(
        data=str(data), epochs=epochs, imgsz=imgsz, batch=batch,
        device=0, workers=workers, amp=True, pretrained=False, val=True,
        project=str(SCRATCH), name="probe", exist_ok=True,
        plots=False, save=False, verbose=False, seed=0,
    )
    total = time.perf_counter() - t0

    per_epoch = None
    if epoch_s:
        # Epoch 1 carries cudnn autotune and warmup, so it is never the rate
        # a long run sustains. Drop it when there is anything left.
        steady = epoch_s[1:] or epoch_s
        per_epoch = {
            "all_epochs_s": [round(d, 1) for d in epoch_s],
            "steady_state_s": round(sum(steady) / len(steady), 1),
            "steady_min_s": round(min(steady), 1),
            "steady_max_s": round(max(steady), 1),
        }
    n_train = len((CANDIDATE / "splits" / f"{STAGE2_SOURCE}_train.txt")
                  .read_text().strip().splitlines())
    out = {
        "source": STAGE2_SOURCE,
        "train_images": n_train,
        "epochs_run": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "workers": workers,
        "total_wall_s": round(total, 1),
        "per_epoch": per_epoch,
    }
    if per_epoch:
        out["images_per_s_end_to_end"] = round(
            n_train / per_epoch["steady_state_s"], 1)
    shutil.rmtree(SCRATCH, ignore_errors=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="all", choices=["1", "2", "all"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, nargs="+", default=[8, 2],
                    help="dataloader worker counts to compare. 2 is the H200 "
                         "cluster cap, 8 is what this laptop can give.")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--root", type=Path,
                    help="project root holding clean_candidate/. Defaults to "
                         "this script's folder.")
    ap.add_argument("--batches", type=int, nargs="+", default=BATCHES,
                    help="batch sizes to sweep in stage 1. Go higher on a "
                         "large slice, the sweep stops at the first OOM.")
    ap.add_argument("--out", type=Path, help="where to write the json report")
    args = ap.parse_args()

    if args.root:
        set_root(args.root)

    info = gpu_info()
    print(json.dumps(info, indent=2))
    report = {"device": info, "imgsz": args.imgsz, "amp": not args.no_amp,
              "root": str(ROOT)}

    if args.stage in ("1", "all"):
        print(f"\nSTAGE 1  synthetic forward and backward at {args.imgsz} px")
        report["stage1_synthetic"] = stage1(args.imgsz, not args.no_amp,
                                            args.batches)

    if args.stage in ("2", "all"):
        report["stage2_real"] = {}
        for w in args.workers:
            print(f"\nSTAGE 2  real epochs, {STAGE2_SOURCE}, "
                  f"batch {args.batch}, workers {w}")
            r = stage2(args.imgsz, args.epochs, args.batch, w)
            report["stage2_real"][f"workers_{w}"] = r
            print(json.dumps(r, indent=2))

    out = args.out or (ROOT / "gpu_throughput.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
