#!/usr/bin/env python3
"""Materialise the arm datasets from the FROZEN splits. Nothing is re-solved.

WHAT THIS DOES NOT DO. It does not create a split, run the partitioner, or
touch any existing file under clean_candidate/. Every arm dataset is a view of
the frozen split lists. The solver is not reproducible (see the project log "Split
reproducibility"), so re-solving anything here would silently produce a
different partition and make arm numbers incomparable with main grid numbers.

TWO KINDS OF ARM DATASET

  acquisition subset  lohi_high and lohi_low. The frozen lohi_weld split lists
                      filtered by C.LOHI_VERDICT_PATTERN. Images are symlinks
                      and labels are symlinks, because the labels are unchanged.
  pool                arm D's three leave-one-out pools. Images are symlinks
                      but labels are REWRITTEN, because a pooled dataset needs
                      one class index space and each member's indices have to
                      be shifted into it.

WHY SYMLINKS. Copying 430 MB three times is wasteful, and more importantly a
copy could drift from the frozen tree without anything noticing. A symlink
cannot. Ultralytics writes labels.cache beside the labels directory on first
scan, so the cache lands in the arm folder rather than in the frozen tree.
The project log 5b records the smoke test that dropped caches into the kaggle
originals, which is the failure this layout avoids.

THE CHECK THAT MAKES THE ACQUISITION SPLIT LEGITIMATE is re-run here every
time rather than trusted: no split block may contain images from both
acquisitions. If one ever did, filtering by acquisition would cut a block in
half and put correlated images on both sides of a train/test boundary.

Usage:
    python3 grid/arms/build_arm_datasets.py
    python3 grid/arms/build_arm_datasets.py --check-only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import config as C  # noqa: E402
import config_arms as A  # noqa: E402

SPLITS = ["train", "valid", "test"]
YAML_SPLIT_KEY = {"train": "train", "valid": "val", "test": "test"}


def frozen_split_list(tag: str, split: str) -> list[str]:
    path = C.CANDIDATE / "splits" / f"{tag}_{split}.txt"
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def split_blocks() -> dict[str, str]:
    """clean-relative image path -> split block, from the frozen manifest."""
    out = {}
    with (C.CANDIDATE / "manifest.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["kept"].lower() == "true":
                out[row["representative_clean_path"]] = row["split_block"]
    return out


# --------------------------------------------------------------------------
# the check that licenses the acquisition split
# --------------------------------------------------------------------------

def check_acquisition_blocks(verbose: bool = True) -> dict:
    """No split block may straddle the two LoHi acquisitions."""
    blocks = split_blocks()
    by_block = defaultdict(set)
    counts = defaultdict(lambda: defaultdict(int))
    for split in SPLITS:
        for rel in frozen_split_list("lohi_weld", split):
            acq = A.acquisition_of(Path(rel).stem)
            counts[split][acq] += 1
            key = f"lohi_weld/{rel}"
            block = blocks.get(key)
            if block is None:
                raise SystemExit(f"no split block recorded for {key}")
            by_block[block].add(acq)

    straddling = sorted(b for b, s in by_block.items() if len(s) > 1)
    if verbose:
        print("LoHi acquisition partition, from the frozen split lists")
        for split in SPLITS:
            hi, lo = counts[split]["lohi_high"], counts[split]["lohi_low"]
            print(f"  {split:<6s} high {hi:5d}   low {lo:5d}   total {hi + lo:5d}")
        print(f"  split blocks {len(by_block)}, straddling both "
              f"acquisitions: {len(straddling)}")
    if straddling:
        raise SystemExit(
            f"{len(straddling)} split blocks contain BOTH acquisitions, for "
            f"example {straddling[:3]}. Filtering by acquisition would cut a "
            f"block in half and put correlated images on both sides of a "
            f"train/test boundary. Refusing to build.")
    return {"n_blocks": len(by_block), "straddling": 0,
            "counts": {s: dict(counts[s]) for s in SPLITS}}


# --------------------------------------------------------------------------
# materialisation
# --------------------------------------------------------------------------

def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)


def label_path(image_rel: str) -> str:
    return image_rel.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"


def write_yaml(root: Path, classes: list[str]) -> None:
    """A data.yaml with NO `path` key.

    The project log 4c: ultralytics resolves the dataset root as
    `data.get("path") or Path(yaml_file).parent`, so a `path: .` entry always
    exists and every split therefore resolves against the CALLER's working
    directory. Omitting the key anchors the file to its own folder.
    """
    lines = [f"{YAML_SPLIT_KEY[s]}: {s}/images" for s in SPLITS]
    lines.append(f"nc: {len(classes)}")
    lines.append("names:")
    lines += [f"- {c}" for c in classes]
    (root / "data.yaml").write_text("\n".join(lines) + "\n")


def build_acquisition(tag: str, out_root: Path) -> dict:
    """lohi_high or lohi_low: filter the frozen lohi_weld lists in place."""
    parent = "lohi_weld"
    root = out_root / tag
    if root.exists():
        shutil.rmtree(root)
    counts = {}
    listing = {}
    for split in SPLITS:
        kept = [rel for rel in frozen_split_list(parent, split)
                if A.acquisition_of(Path(rel).stem) == tag]
        for rel in kept:
            link((C.CANDIDATE / parent / rel).resolve(), root / rel)
            lab = label_path(rel)
            src_lab = C.CANDIDATE / parent / lab
            if src_lab.is_file():
                link(src_lab.resolve(), root / lab)
        counts[split] = len(kept)
        listing[split] = kept
    write_yaml(root, A.ARM_DATASETS[tag]["classes"])
    return {"counts": counts, "listing": listing}


def build_pool(pool: str, out_root: Path) -> dict:
    """An arm D pool: union the members and shift class indices.

    Images are symlinked under a per-member prefix so two datasets cannot
    collide on a filename. Labels are rewritten because the pooled class index
    space is the concatenation of the members' spaces, in the order recorded in
    config_arms.pool_classes.
    """
    members = A.ARMD_POOLS[pool]
    classes = A.ARM_DATASETS[pool]["classes"]
    index_of = {name: i for i, name in enumerate(classes)}
    root = out_root / pool
    if root.exists():
        shutil.rmtree(root)

    counts = {s: 0 for s in SPLITS}
    listing = {s: [] for s in SPLITS}
    for tag in members:
        offset_names = C.DATASETS[tag]["classes"]
        for split in SPLITS:
            for rel in frozen_split_list(tag, split):
                name = Path(rel).name
                dst_rel = f"{split}/images/{tag}__{name}"
                link((C.CANDIDATE / tag / rel).resolve(), root / dst_rel)

                src_lab = C.CANDIDATE / tag / label_path(rel)
                dst_lab = root / dst_rel.replace("/images/", "/labels/")
                dst_lab = dst_lab.with_suffix(".txt")
                dst_lab.parent.mkdir(parents=True, exist_ok=True)
                out_lines = []
                if src_lab.is_file():
                    for line in src_lab.read_text().splitlines():
                        parts = line.split()
                        if len(parts) < 5:
                            continue
                        old = int(float(parts[0]))
                        qualified = f"{tag}:{offset_names[old]}"
                        parts[0] = str(index_of[qualified])
                        out_lines.append(" ".join(parts))
                dst_lab.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))

                counts[split] += 1
                listing[split].append(dst_rel)
    write_yaml(root, classes)
    return {"counts": counts, "listing": listing, "members": members}


def hash_listing(listing: dict) -> dict:
    """sha256 over each split's sorted file list.

    The arm datasets are views of frozen splits, so this is the arm equivalent
    of clean_candidate/split_hashes.json: it is what a reader verifies rather
    than regenerating.
    """
    out = {}
    for split, entries in listing.items():
        blob = "\n".join(sorted(entries))
        out[split] = hashlib.sha256(blob.encode()).hexdigest()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check-only", action="store_true",
                    help="run the straddling-block check and stop")
    ap.add_argument("--out-root", type=Path, default=C.CANDIDATE)
    args = ap.parse_args()

    A.assert_main_unmoved()
    A.register()

    info = check_acquisition_blocks()
    if args.check_only:
        print("\ncheck only, nothing written")
        return

    manifest = {"arm_fingerprint": A.arm_fingerprint(),
                "frozen_main_fingerprint": A.FROZEN_MAIN_FINGERPRINT,
                "acquisition_block_check": info,
                "datasets": {}}

    print("\nbuilding acquisition subsets")
    for tag in A.LOHI_ACQUISITIONS:
        built = build_acquisition(tag, args.out_root)
        manifest["datasets"][tag] = {
            "kind": "acquisition_subset", "parent": "lohi_weld",
            "counts": built["counts"], "hashes": hash_listing(built["listing"])}
        print(f"  {tag:<12s} " + "  ".join(
            f"{s} {built['counts'][s]}" for s in SPLITS))

    print("\nbuilding arm D pools")
    for pool in A.ARMD_POOLS:
        built = build_pool(pool, args.out_root)
        manifest["datasets"][pool] = {
            "kind": "pool", "members": built["members"],
            "protocol": A.ARM_DATASETS[pool]["protocol"],
            "n_classes": len(A.ARM_DATASETS[pool]["classes"]),
            "counts": built["counts"], "hashes": hash_listing(built["listing"])}
        print(f"  {pool:<26s} " + "  ".join(
            f"{s} {built['counts'][s]}" for s in SPLITS)
            + f"   nc={len(A.ARM_DATASETS[pool]['classes'])}"
              f"   protocol={A.ARM_DATASETS[pool]['protocol']}")

    out = A.ARMS_ROOT / "arm_split_hashes.json"
    out.write_text(json.dumps(manifest, indent=1))
    print(f"\nwrote {out}")
    print("clean_candidate/splits/ and split_hashes.json were not touched")


if __name__ == "__main__":
    main()
