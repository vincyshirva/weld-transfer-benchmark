#!/usr/bin/env python3
"""
Is LoHi-WELD low_resolution_welds a separate acquisition, or the same welds
re-exported at a smaller size?

The two folders share no filenames and use different naming schemes. High res
is <plate>_<index>. Low res is <date>_<time>_<cycle>_CAM1_<NG|OK>_<n>. Their
counts, 1,022 and 2,000, sum to the 3,022 the IEEE Access description reports.
That is consistent with either reading, so it has to be decided on content.

Method is the accepted two-stage one, unchanged: standardized grayscale NCC
maximized over the 8 dihedral transforms, 32 pixels to generate candidates and
128 pixels to confirm. Every high res image is compared against every low res
image, which is more than the 100-image sample the question needs and costs
little, so the answer does not depend on which sample was drawn.

Decision rule:
  any pair >= T_COPY   the low res set re-encodes high res welds, exclusion is
                       correct and the paper must say why
  nothing >= T_COPY    separate acquisitions, the low res set is usable data

Usage:   python3 lohi_lowres_probe.py [root]
Outputs: lohi_lowres_probe.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from weld_rebuild_clean import (  # noqa: E402
    SCENE_CONFIRM_THUMB, THUMB, T_COPY, T_SCENE,
    descriptor_at, dihedral_stack, pair_ncc,
)

ROOT = Path(__file__).resolve().parent
BASE = "LoHi-WELD weld-dataset/weld-dataset"
TOP_K = 40          # strongest coarse pairs to re-check at 128 pixels


def load(folder: Path):
    paths, rows = [], []
    for p in sorted(folder.glob("*.jpg")):
        d = descriptor_at(p, THUMB)
        if d is None:
            print(f"  !! unreadable: {p.name}")
            continue
        paths.append(p)
        rows.append(d)
    return paths, np.vstack(rows).astype(np.float32)


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT
    hi_dir = root / BASE / "high_resolution_welds"
    lo_dir = root / BASE / "low_resolution_welds"
    for d in (hi_dir, lo_dir):
        if not d.is_dir():
            sys.exit(f"Not found: {d}")

    print("Loading descriptors")
    hi_paths, HI = load(hi_dir)
    lo_paths, LO = load(lo_dir)
    print(f"  high res {len(hi_paths)}, low res {len(lo_paths)}")
    print(f"  comparing all {len(hi_paths) * len(lo_paths):,} cross pairs "
          f"at {THUMB} pixels\n")

    # best-of-8 similarity, blocked so the full matrix is never held at once
    variants = dihedral_stack(LO)
    norm = float(THUMB * THUMB)
    best_per_hi = np.full(len(hi_paths), -2.0, dtype=np.float32)
    best_arg = np.zeros(len(hi_paths), dtype=np.int64)
    step = 256
    for a in range(0, len(hi_paths), step):
        blk = HI[a:a + step]
        acc = None
        for V in variants:
            s = (blk @ V.T) / norm
            acc = s if acc is None else np.maximum(acc, s)
        j = acc.argmax(axis=1)
        best_per_hi[a:a + len(blk)] = acc[np.arange(len(blk)), j]
        best_arg[a:a + len(blk)] = j

    order = np.argsort(-best_per_hi)
    coarse_max = float(best_per_hi.max())
    print(f"coarse NCC{THUMB} over every cross pair")
    print(f"  maximum        {coarse_max:.4f}")
    print(f"  99th pct       {float(np.percentile(best_per_hi, 99)):.4f}")
    print(f"  median         {float(np.median(best_per_hi)):.4f}")
    print(f"  pairs >= T_copy {T_COPY}:  "
          f"{int((best_per_hi >= T_COPY).sum())}")
    print(f"  pairs >= T_scene {T_SCENE}: "
          f"{int((best_per_hi >= T_SCENE).sum())}\n")

    print(f"confirming the {TOP_K} strongest at {SCENE_CONFIRM_THUMB} pixels")
    confirmed = []
    for rank, i in enumerate(order[:TOP_K], 1):
        hp, lp = hi_paths[i], lo_paths[best_arg[i]]
        a = descriptor_at(hp, SCENE_CONFIRM_THUMB)
        b = descriptor_at(lp, SCENE_CONFIRM_THUMB)
        fine = pair_ncc(a, b, SCENE_CONFIRM_THUMB) if a is not None and b is not None else None
        confirmed.append({
            "rank": rank,
            "high_res": hp.name,
            "low_res": lp.name,
            "ncc32": round(float(best_per_hi[i]), 4),
            "ncc128": None if fine is None else round(float(fine), 4),
        })
        if rank <= 10:
            print(f"  {rank:>2}. NCC32 {best_per_hi[i]:.4f}  "
                  f"NCC128 {fine:.4f}   {hp.name}  vs  {lp.name}")

    fine_scores = [c["ncc128"] for c in confirmed if c["ncc128"] is not None]
    fine_max = max(fine_scores) if fine_scores else None
    print(f"\n  maximum NCC{SCENE_CONFIRM_THUMB} among the top {TOP_K}: "
          f"{fine_max:.4f}")

    separate = coarse_max < T_COPY
    verdict = ("SEPARATE ACQUISITIONS" if separate
               else "LOW RES RE-ENCODES HIGH RES WELDS")
    print(f"\nVERDICT: {verdict}")
    print(f"  nothing reaches T_copy {T_COPY}. The strongest cross pair scores "
          f"{coarse_max:.4f} coarse and {fine_max:.4f} confirmed."
          if separate else
          f"  at least one cross pair reaches T_copy {T_COPY}.")

    out = root / "lohi_lowres_probe.json"
    out.write_text(json.dumps({
        "method": {
            "metric": "standardized grayscale NCC, best of 8 dihedral transforms",
            "candidate_thumb": THUMB,
            "confirmation_thumb": SCENE_CONFIRM_THUMB,
            "t_copy": T_COPY, "t_scene": T_SCENE,
        },
        "counts": {"high_res": len(hi_paths), "low_res": len(lo_paths),
                   "cross_pairs": len(hi_paths) * len(lo_paths),
                   "shared_filenames": 0},
        "coarse": {
            "max": coarse_max,
            "p99": float(np.percentile(best_per_hi, 99)),
            "median": float(np.median(best_per_hi)),
            "n_at_or_above_t_copy": int((best_per_hi >= T_COPY).sum()),
            "n_at_or_above_t_scene": int((best_per_hi >= T_SCENE).sum()),
        },
        "confirmed_top": confirmed,
        "max_ncc128_among_top": fine_max,
        "separate_acquisitions": bool(separate),
        "verdict": verdict,
    }, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
