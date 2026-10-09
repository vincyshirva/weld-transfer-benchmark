#!/usr/bin/env python3
"""
Build a portable snapshot of the ORIGINAL shipped splits.

The pre-audit versus post-audit comparison needs models trained the way a
practitioner would train them: download the dataset, use the split it ships
with, and report the in-domain number. The audit found augmented duplicates
and adjacent video frames straddling the train and test boundary. The
downloaded and clean values use different tests, so their difference is an
estimate shift rather than a causal measure of leakage inflation.

NOTHING UNDER AN ORIGINAL DATASET FOLDER IS WRITTEN, and getting that right
took two attempts. Pointing a data.yaml straight at an original folder is NOT
enough: ultralytics writes a `labels.cache` beside the labels directory the
first time it scans a split, which lands inside the original tree. That
happened once here and was caught by the fingerprint check below.

Each dirty split is first built as a symlink farm under grid/dirty/. The
directories are real, so every cache is written there. Transfer to the server
dereferences those links. The server receives a standalone snapshot and does
not need the originals. Content hashes verify both forms. The local build also
checks the original stat fingerprint before and after generation.

The three sources ship their splits differently:

  weld_surface_v2  Roboflow train/valid/test folders, 15045/1435/715
  kaggle_v2        Roboflow train/valid/test folders, 1619/283/126, nested one
                   level deeper than the folder name suggests
  lohi_weld        no detection split at all, and its labels are `.yolo`
                   files sitting beside the images rather than a labels/
                   folder, which ultralytics cannot read. So the dirty LoHi
                   set is the full 3,022 image conversion in lohi_weld_yolo/,
                   which is pre-dedup and therefore genuinely dirty, split by
                   the shipped kfold lists. test.txt becomes the test split
                   and every fifth remaining image the validation split.

Usage:   python3 make_dirty_yaml.py [--check]
Outputs: grid/dirty/<tag>/data.yaml for each tag, a symlink farm under
         <tag>/{train,valid,test}/{images,labels}, and the content manifest
         grid/dirty/split_hashes.json.

ALL THREE TAGS HAVE THE SAME SHAPE. Every data.yaml addresses directories,
never a list file. lohi_weld additionally carries train.txt, val.txt and
test.txt left over from an earlier list-based build. Nothing reads them, they
agree exactly with the directories beside them, and they are kept only because
they record how the LoHi split was derived from the authors' kfold lists.
Their absence from kaggle_v2 and weld_surface_v2 is not a missing artefact.
Completeness is defined by split_hashes.json, which pins the byte count, file
count and sha256 of every split directory, and preflight verifies all three.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

DIRTY = Path(__file__).resolve().parent / "dirty"

ORIGINALS = {
    "weld_surface_v2": C.PROJECT / "Weld Surface Defect Detection.v2i.yolov8",
    "kaggle_v2": (C.PROJECT / "Welding Defect, Object Detection_kaggle"
                  / "The Welding Defect Dataset - v2"
                  / "The Welding Defect Dataset - v2"),
    "lohi_weld": C.PROJECT / "LoHi-WELD weld-dataset" / "weld-dataset",
}


def originals_present() -> bool:
    return all(path.is_dir() for path in ORIGINALS.values())


def fingerprint() -> str:
    """Stat fingerprint of every original folder, to prove we did not write."""
    if not originals_present():
        missing = [str(path) for path in ORIGINALS.values() if not path.is_dir()]
        raise FileNotFoundError(f"original folders are not present: {missing}")
    digest = hashlib.sha256()
    for tag in sorted(ORIGINALS):
        for path in sorted(p for p in ORIGINALS[tag].rglob("*") if p.is_file()):
            st = path.stat()
            digest.update(f"{path}|{st.st_size}|{st.st_mtime_ns}".encode())
    return digest.hexdigest()


def content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_split(path: Path) -> dict:
    """Content fingerprint for a portable dirty split.

    Only images and text labels are included. Ultralytics may create a
    labels.cache beside them, and that derived cache must not change the
    released snapshot identity.
    """
    digest = hashlib.sha256()
    files, total_bytes = 0, 0
    for folder in ("images", "labels"):
        base = path / folder
        for item in sorted(p for p in base.iterdir() if p.is_file()):
            size = item.stat().st_size
            sha = content_sha256(item)
            digest.update(f"{folder}/{item.name}|{size}|{sha}\n".encode())
            files += 1
            total_bytes += size
    return {"files": files, "bytes": total_bytes,
            "sha256": digest.hexdigest()}


def snapshot_dataset(tag: str) -> dict:
    root = DIRTY / tag
    return {split: snapshot_split(root / split)
            for split in ("train", "valid", "test")}


def verify_snapshot(tag: str | None = None) -> dict:
    """Verify the generated snapshot and return the expected record."""
    path = DIRTY / "split_hashes.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing dirty snapshot manifest: {path}")
    expected = json.loads(path.read_text())["datasets"]
    tags = [tag] if tag else sorted(expected)
    for name in tags:
        got = snapshot_dataset(name)
        if got != expected[name]:
            raise RuntimeError(f"dirty snapshot hash mismatch for {name}")
    return expected[tag] if tag else expected


def link_split(src_images: Path, dst_split: Path, stems=None) -> int:
    """Mirror one split as symlinks. Real dirs, linked files.

    The real directory is what keeps ultralytics' labels.cache out of the
    original tree. Symlinking the directory itself would not: a write into a
    symlinked directory lands in its target.
    """
    src_labels = Path(str(src_images).replace("/images", "/labels"))
    di, dl = dst_split / "images", dst_split / "labels"
    for d in (di, dl):
        if d.is_dir():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    n = 0
    for img in sorted(src_images.iterdir()):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        if stems is not None and img.stem not in stems:
            continue
        (di / img.name).symlink_to(img)
        lab = src_labels / f"{img.stem}.txt"
        if lab.is_file():
            (dl / lab.name).symlink_to(lab)
        n += 1
    return n


def roboflow(tag: str, out: Path) -> dict:
    """Mirror a Roboflow export's shipped splits as symlinks."""
    root = ORIGINALS[tag]
    counts = {}
    for split in ("train", "valid", "test"):
        d = root / split / "images"
        if not d.is_dir():
            sys.exit(f"{tag}: missing {d}")
        counts[split] = link_split(d, out / split)
    return {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(C.DATASETS[tag]["classes"]),
        "names": C.DATASETS[tag]["classes"],
        "_counts": counts,
    }


def lohi(out: Path) -> dict:
    """Rebuild the shipped k-fold as train, val and test image lists.

    Both acquisitions are pooled, matching the clean tree, so the dirty and
    clean numbers differ only by the split and not by which images exist. The
    shipped lists name files relative to the dataset root, so they are
    rewritten here as absolute paths into a list file under grid/dirty/.
    """
    root = ORIGINALS["lohi_weld"]
    # Images come from the conversion staging folder, which holds all 3,022
    # with no deduplication applied. That is what makes it the dirty set. The
    # split comes from the shipped kfold lists, which are the authors' own.
    staging = C.PROJECT / "lohi_weld_yolo" / "train" / "images"
    if not staging.is_dir():
        sys.exit(f"lohi_weld: missing {staging}. Run lohi_to_yolo.py first.")

    test_stems = set()
    for folder in ("kfold_high", "kfold_low"):
        base = root / folder
        if not base.is_dir():
            sys.exit(f"lohi_weld: missing {base}")
        test_stems |= {Path(x).stem for x in (base / "test.txt").read_text().split()}

    test, trainval = [], []
    for img in sorted(staging.iterdir()):
        if img.suffix.lower() != ".jpg":
            continue
        (test if img.stem in test_stems else trainval).append(img.stem)

    # fold0 of the remainder becomes validation, matching a 5-fold protocol
    val = set(trainval[::5])
    train = [s for s in trainval if s not in val]

    counts = {}
    for name, split, stems in (("train", "train", set(train)),
                               ("valid", "val", val),
                               ("test", "test", set(test))):
        counts[name] = link_split(staging, out / name, stems)
    return {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(C.DATASETS["lohi_weld"]["classes"]),
        "names": C.DATASETS["lohi_weld"]["classes"],
        "_counts": counts,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="only verify the originals are untouched")
    ap.add_argument("--verify-snapshot", action="store_true",
                    help="verify the portable dirty split content hashes")
    args = ap.parse_args()

    if args.verify_snapshot:
        checked = verify_snapshot()
        print(f"portable dirty snapshots verified: {len(checked)} datasets")
        return

    before = fingerprint()
    if args.check:
        print(f"original tree fingerprint: {before[:16]}")
        return

    print(f"original tree fingerprint before: {before[:16]}")
    summary = {}
    snapshots = {}
    for tag in ORIGINALS:
        out = DIRTY / tag
        out.mkdir(parents=True, exist_ok=True)
        cfg = lohi(out) if tag == "lohi_weld" else roboflow(tag, out)
        counts = cfg.pop("_counts")
        # No `path` key, for the same reason the clean tree has none: it would
        # resolve against the working directory. See the project log 4c.
        (out / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        clean = {s: len((C.CANDIDATE / "splits" /
                         f"{tag}_{s}.txt").read_text().strip().splitlines())
                 for s in ("train", "valid", "test")}
        summary[tag] = {"dirty": counts, "clean": clean}
        snapshots[tag] = snapshot_dataset(tag)
        print(f"  {tag:<18} dirty {counts}")
        print(f"  {'':<18} clean {clean}")

    after = fingerprint()
    print(f"original tree fingerprint after:  {after[:16]}")
    if before != after:
        sys.exit("FAIL: an original dataset folder changed. Nothing was "
                 "supposed to write there.")
    print("originals verified unchanged")
    manifest = {"schema_version": 1, "datasets": snapshots,
                "original_stat_fingerprint": before}
    (DIRTY / "split_hashes.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True))
    verify_snapshot()
    print(f"wrote and verified {DIRTY / 'split_hashes.json'}")


if __name__ == "__main__":
    main()
