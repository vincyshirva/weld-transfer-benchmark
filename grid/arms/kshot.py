#!/usr/bin/env python3
"""Arm B k-shot draws. Scene aware, by split block. PRE-REGISTERED, 6a.

THE SAMPLING UNIT IS THE SPLIT BLOCK, NOT THE IMAGE, AND THAT IS THE POINT.

Sampling images independently would let two frames of one weld count as two
shots. Weld Surface is video derived and adjacent frames are nearly the same
photograph, so a "50 shot" draw built that way could contain a dozen distinct
scenes and would badly overstate how little annotation a practitioner needs.
The label cost curve is the most citable object in this paper, so the units on
its x axis have to be honest.

A draw therefore accumulates WHOLE blocks until it reaches k images. The
realised image count overshoots k slightly, because a block is indivisible.
Both numbers are recorded, and both must be reported:

    n_images   what was annotated
    n_scenes   how many distinct welds that was

DISJOINTNESS IS ASSERTED, NOT ASSUMED. Shots come from the target's TRAIN
split only and scoring stays on the frozen TEST split. Those are different
blocks by construction, but `assert_disjoint` re-checks it against the frozen
split lists on every draw, because "by construction" is exactly the kind of
claim that stops being true after someone edits a loader.

NOTHING HERE IS TUNED ON A TARGET TEST SPLIT. The draw depends only on the
target, k, and the draw index.
"""

from __future__ import annotations

import csv
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import config as C  # noqa: E402
import config_arms as A  # noqa: E402


def _blocks_for(tag: str, split: str) -> dict[str, list[str]]:
    """split block -> image paths, relative to clean_candidate/<tag>/."""
    blocks = {}
    with (C.CANDIDATE / "manifest.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["kept"].lower() == "true":
                blocks[row["representative_clean_path"]] = row["split_block"]

    listing = [line.strip() for line
               in (C.CANDIDATE / "splits" / f"{tag}_{split}.txt").read_text().splitlines()
               if line.strip()]
    out = defaultdict(list)
    for rel in listing:
        block = blocks.get(f"{tag}/{rel}")
        if block is None:
            raise SystemExit(f"no split block recorded for {tag}/{rel}")
        out[block].append(rel)
    return dict(out)


def draw_seed(target: str, k: int, draw: int) -> int:
    """Deterministic and independent of process hashing.

    Never seed from the built-in hash(): Python randomises string hashing per
    process, so the same draw would select different images on different runs.
    That defect was found in transfer_bootstrap and is recorded in the project log
    5m. The k-shot draws are a released artefact, so they must be stable.
    """
    digest = hashlib.sha256(f"armb:{target}:{k}:{draw}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def sample(target: str, k: int, draw: int) -> dict:
    """One k-shot draw from the target's train split, by whole split blocks."""
    blocks = _blocks_for(target, "train")
    keys = sorted(blocks)
    rng = np.random.default_rng(draw_seed(target, k, draw))
    order = rng.permutation(len(keys))

    chosen_blocks, images = [], []
    for idx in order:
        if len(images) >= k:
            break
        block = keys[idx]
        chosen_blocks.append(block)
        images.extend(blocks[block])

    return {
        "target": target, "k_requested": k, "draw": draw,
        "n_images": len(images),
        "n_scenes": len(chosen_blocks),
        "images": sorted(images),
        "blocks": sorted(chosen_blocks),
        "exhausted_train_split": len(images) < k,
        "n_train_images_available": sum(len(v) for v in blocks.values()),
        "n_train_blocks_available": len(keys),
        "seed": draw_seed(target, k, draw),
    }


def assert_disjoint(drawn: dict) -> None:
    """Adaptation blocks must not touch evaluation blocks."""
    test_blocks = set(_blocks_for(drawn["target"], "test"))
    overlap = test_blocks & set(drawn["blocks"])
    if overlap:
        raise SystemExit(
            f"arm B draw for {drawn['target']} k={drawn['k_requested']} "
            f"draw={drawn['draw']} shares {len(overlap)} split blocks with the "
            f"TEST split, for example {sorted(overlap)[:3]}. Adaptation and "
            f"evaluation blocks must be disjoint.")

    valid_blocks = set(_blocks_for(drawn["target"], "valid"))
    if valid_blocks & set(drawn["blocks"]):
        raise SystemExit(
            f"arm B draw for {drawn['target']} overlaps the VALID split. "
            f"Shots must come from the train split only.")


def materialise(drawn: dict, root: Path) -> Path:
    """Write the draw as a YOLO dataset that ultralytics can train from.

    The k-shot images are the TRAIN split of the little dataset. Its VALID
    split is the target's own frozen valid split, unchanged, because
    ultralytics needs a validation set to select a checkpoint and the frozen
    valid split is the only choice that is not the test set. Using any part of
    the test split here would be tuning on the target, which 6a forbids.
    """
    target = drawn["target"]
    if root.exists():
        import shutil
        shutil.rmtree(root)

    def link(rel: str, split: str) -> None:
        src = (C.CANDIDATE / target / rel).resolve()
        dst = root / split / "images" / Path(rel).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src)
        lab_src = C.CANDIDATE / target / rel.replace("/images/", "/labels/")
        lab_src = lab_src.with_suffix(".txt")
        lab_dst = root / split / "labels" / Path(rel).with_suffix(".txt").name
        lab_dst.parent.mkdir(parents=True, exist_ok=True)
        if lab_src.is_file():
            lab_dst.symlink_to(lab_src.resolve())
        else:
            lab_dst.write_text("")

    for rel in drawn["images"]:
        link(rel, "train")
    for rel in [line.strip() for line
                in (C.CANDIDATE / "splits" / f"{target}_valid.txt").read_text().splitlines()
                if line.strip()]:
        link(rel, "valid")

    classes = A.dataset_entry(target)["classes"]
    lines = ["train: train/images", "val: valid/images",
             f"nc: {len(classes)}", "names:"]
    lines += [f"- {c}" for c in classes]
    # No `path` key. The project log 4c.
    (root / "data.yaml").write_text("\n".join(lines) + "\n")
    return root / "data.yaml"


if __name__ == "__main__":
    A.assert_main_unmoved()
    A.register()
    print(f"{'target':<18s}{'k':>5s}{'draw':>6s}{'images':>8s}{'scenes':>8s}"
          f"{'exhausted':>11s}")
    for target in C.ALL_TARGETS:
        for k in A.ARMB_K:
            for draw in A.ARMB_DRAWS:
                d = sample(target, k, draw)
                assert_disjoint(d)
                print(f"{target:<18s}{k:>5d}{draw:>6d}{d['n_images']:>8d}"
                      f"{d['n_scenes']:>8d}"
                      f"{'YES' if d['exhausted_train_split'] else '':>11s}")
