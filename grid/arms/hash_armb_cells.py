#!/usr/bin/env python3
"""THE STANDING POST-RETRAIN INTEGRITY CHECK FOR ARM B. The project log 6a-viii.

    python3 grid/arms/hash_armb_cells.py
    python3 grid/arms/hash_armb_cells.py --endpoint best

Hashes the TENSORS of every arm B checkpoint and reports any group of cells
sharing a hash. Report only: it writes a JSON summary and changes nothing.

WHY THIS IS THE BEST CHECK AVAILABLE, AND WHY IT IS NOT OPTIONAL.

When `train_arm` passed `pretrained=False`, ultralytics discarded the start
checkpoint and trained from random initialisation. Two arm B cells that share a
target, k and draw differ ONLY in their start checkpoint, and YOLO11s training
is deterministic under a fixed seed. So the defect produced **45 groups of
exactly 2 bit-identical YOLO11s cells**, 195 distinct hashes over 240.

After a correct retrain those groups must be gone and the count must be 240.
That tests the exact property that failed, which no assertion about flags or
arguments can do. The preflight's relative-L2 check is prospective and this is
retrospective; keep both.

**Hash the TENSORS, not the file.** Two ultralytics checkpoints with identical
weights still differ as files, because `date` and the recorded
`train_args.model` path go into the pickle. Hashing files reports 240 distinct
and hides the defect completely. That mistake was made once during the original
diagnosis and caught only because an array-level comparison disagreed with it.

RT-DETR-l cells were EQUALLY broken and never collided, because RT-DETR
training is not deterministic under a fixed seed. **A family that fails to
collide is not a family that is correct.** Do not read a clean RT-DETR result
as evidence of anything on its own.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import config as C  # noqa: E402


def tensor_hash(fobj) -> str:
    """sha256 over the model's tensors, in sorted key order."""
    import torch
    ck = torch.load(fobj, map_location="cpu", weights_only=False)
    model = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    sd = model.state_dict() if hasattr(model, "state_dict") else model
    digest = hashlib.sha256()
    for key in sorted(sd):
        digest.update(key.encode())
        digest.update(sd[key].detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path,
                    default=C.PROJECT / "grid_results_arms" / "results")
    ap.add_argument("--endpoint", choices=["last", "best", "both"],
                    default="both")
    ap.add_argument("--out", type=Path,
                    default=C.PROJECT / "grid_results_arms"
                    / "armb_tensor_hashes.json")
    args = ap.parse_args()

    archives = sorted(args.results_dir.glob("armb__*.tar.gz"))
    if not archives:
        sys.exit(f"no arm B archives under {args.results_dir}")
    which = (["last", "best"] if args.endpoint == "both" else [args.endpoint])
    print(f"{len(archives)} arm B archives, endpoints {which}")

    rows, groups = [], defaultdict(list)
    for i, archive in enumerate(archives, 1):
        name = archive.name[: -len(".tar.gz")]
        with tarfile.open(archive) as tar:
            for endpoint in which:
                member = f"{name}/weights/{endpoint}.pt"
                try:
                    data = io.BytesIO(tar.extractfile(member).read())
                except Exception as exc:                      # noqa: BLE001
                    print(f"  {name} {endpoint}: {exc}")
                    continue
                h = tensor_hash(data)
                rows.append({"cell": name, "endpoint": endpoint,
                             "tensor_sha256_16": h})
                groups[(endpoint, h)].append(name)
        if i % 40 == 0:
            print(f"  {i}/{len(archives)}", flush=True)

    summary, failed = {}, False
    for endpoint in which:
        g = {h: v for (e, h), v in groups.items() if e == endpoint}
        n_cells = sum(len(v) for v in g.values())
        shared = {h: sorted(v) for h, v in g.items() if len(v) > 1}
        by_family = defaultdict(int)
        for members in shared.values():
            fam = "yolo11s" if "yolo11s" in members[0] else "rtdetr_l"
            by_family[fam] += len(members)
        print(f"\n=== {endpoint}.pt: {len(g)} DISTINCT tensor hashes over "
              f"{n_cells} cells")
        if shared:
            failed = True
            print(f"    {len(shared)} hashes shared by more than one cell, "
                  f"{dict(by_family)}")
            for h, members in sorted(shared.items())[:5]:
                print(f"      {h}  {members}")
            if len(shared) > 5:
                print(f"      ... and {len(shared) - 5} more")
        else:
            print("    every cell is distinct, which is the expected result")
        summary[endpoint] = {
            "n_cells": n_cells, "n_distinct": len(g),
            "n_shared_hashes": len(shared),
            "cells_in_a_shared_hash": sum(len(v) for v in shared.values()),
            "shared": shared, "by_family": dict(by_family)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"summary": summary, "rows": rows,
         "reference_broken_run": {
             "n_distinct": 195, "n_cells": 240, "n_shared_hashes": 45,
             "note": "the project log 6a-viii, all 45 groups YOLO11s pairs"}},
        indent=1))
    print(f"\nwrote {args.out}")

    if failed:
        sys.exit(
            "\nCELLS SHARE WEIGHTS. Two arm B cells differing only in their "
            "start checkpoint produced identical tensors, which is the exact "
            "signature of the project log 6a-viii. Do not analyse these results.")
    print("\nPASS: no two arm B cells share weights.")


if __name__ == "__main__":
    main()
