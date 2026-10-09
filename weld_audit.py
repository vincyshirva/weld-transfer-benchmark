#!/usr/bin/env python3
"""Authoritative NCC audit for a rebuilt weld dataset tree.

This verifier does not use perceptual hash distance to decide duplication.
It measures direct normalized cross correlation on standardized 32 by 32
grayscale thumbnails and maximizes over the eight dihedral transforms.

The audit fails when it finds any of the following:

1. A retained same-photograph pair at NCC 0.96 or above.
2. A 32 pixel scene candidate, or a 128 pixel confirmed scene pair, across
   two splits.
3. A documented frame run or specimen crossing a split boundary.
4. Invalid image-label pairing, malformed YOLO labels, or nonportable YAML.
5. A manifest row that does not resolve to one retained representative.
6. A release split list that disagrees with the emitted image tree.

Usage:
    python3 weld_audit.py clean_candidate
    python3 weld_audit.py clean_candidate --out clean_candidate/audit
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

T_COPY = 0.96
T_SCENE = 0.90
THUMB = 32
SCENE_CONFIRM_THUMB = 128
BLOCK = 512
SPLITS = ("train", "valid", "test")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
RE_RF = re.compile(r"\.rf\.[0-9a-f]{32}\.\w+$")


def descriptor_at(path: Path, thumb: int) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(
            image.convert("L").resize((thumb, thumb), Image.Resampling.LANCZOS),
            dtype=np.float32,
        )
    array -= array.mean()
    std = array.std()
    return array.ravel() if std < 1e-6 else (array / std).ravel()


def descriptor(path: Path) -> np.ndarray:
    return descriptor_at(path, THUMB)


def pair_ncc(left: np.ndarray, right: np.ndarray, thumb: int) -> float:
    a = left.reshape(thumb, thumb)
    b = right.reshape(thumb, thumb)
    return max(
        float(np.mean(a * variant))
        for rotation in range(4)
        for variant in (
            np.rot90(b, rotation),
            np.rot90(b, rotation)[:, ::-1],
        )
    )


def scene_semantically_compatible(left, right):
    """Apply an independent known-identity guard to Visual candidates."""
    if left["dataset"] != right["dataset"]:
        return True
    if left["dataset"] != "visual":
        return True
    pattern = re.compile(r"^(\d+)_")
    match_left = pattern.match(RE_RF.sub("", left["path"].name))
    match_right = pattern.match(RE_RF.sub("", right["path"].name))
    key_left = match_left.group(1) if match_left else None
    key_right = match_right.group(1) if match_right else None
    if key_left is None or key_right is None:
        return key_left is None and key_right is None
    return key_left == key_right


def dihedral_stack(D: np.ndarray) -> list[np.ndarray]:
    square = D.reshape(len(D), THUMB, THUMB)
    variants = []
    for k in range(4):
        rotated = np.rot90(square, k, axes=(1, 2))
        variants.append(np.ascontiguousarray(rotated.reshape(len(D), -1)))
        variants.append(np.ascontiguousarray(rotated[:, :, ::-1].reshape(len(D), -1)))
    return variants


def pairs_above(D: np.ndarray, threshold: float):
    """Yield every pair at or above threshold without holding a full matrix."""
    variants = dihedral_stack(D)
    norm = float(THUMB * THUMB)
    for start in range(0, len(D), BLOCK):
        block = D[start:start + BLOCK]
        best = None
        for variant in variants:
            scores = block @ variant.T / norm
            best = scores if best is None else np.maximum(best, scores)
        rows, cols = np.where(best >= threshold)
        for row, col in zip(rows, cols):
            left = start + int(row)
            right = int(col)
            if left < right:
                yield left, right, float(best[row, col])


def load_names(data_yaml: Path) -> list[str]:
    config = yaml.safe_load(data_yaml.read_text()) or {}
    names = config.get("names", [])
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names, key=lambda x: int(x))]
    return [str(name) for name in names]


def discover_datasets(root: Path):
    found = []
    for data_yaml in sorted(root.rglob("data.yaml")):
        dataset = data_yaml.parent
        if all((dataset / split / "images").is_dir() for split in SPLITS):
            found.append((str(dataset.relative_to(root)), dataset, data_yaml))
    return found


def parse_label(path: Path, n_classes: int):
    errors = []
    boxes = []
    if not path.exists():
        return boxes, ["missing label file"]
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"line {line_number}: expected 5 fields")
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, width, height = (float(value) for value in parts[1:])
        except ValueError:
            errors.append(f"line {line_number}: nonnumeric field")
            continue
        if not 0 <= cls < n_classes:
            errors.append(f"line {line_number}: class {cls} outside 0..{n_classes - 1}")
        if not all(np.isfinite([cx, cy, width, height])):
            errors.append(f"line {line_number}: nonfinite coordinate")
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
            errors.append(f"line {line_number}: invalid normalized box")
        boxes.append((cls, cx, cy, width, height))
    return boxes, errors


def independent_run_key(dataset: str, name: str):
    """Parse only filename structures independently documented as video runs."""
    stem = RE_RF.sub("", name)
    if dataset == "weld_surface_v2":
        match = re.match(
            r"^([A-Za-z]+)[-_]?9?[-_](\d+)[-_]?_(jpg|jpeg|png)$", stem, re.I)
        if match:
            return f"{match.group(1).lower()}:{match.group(3).lower()}", int(match.group(2))
    elif dataset == "twoclass":
        match = re.match(r"^(\d+)-(\d+)-", stem)
        if match:
            return match.group(1), int(match.group(2))
    elif dataset == "kaggle_v2":
        stem = re.sub(r"(?:_(?:jpg|jpeg|png))+$", "", stem, flags=re.I)
        for pattern in (r"^(.*_mp\d+)-(\d+)$", r"^(.*_vid)(\d+)$"):
            match = re.match(pattern, stem)
            if match:
                return match.group(1), int(match.group(2))
    return None


def structural_leaks(records):
    errors = []
    by_dataset = defaultdict(list)
    for record in records:
        by_dataset[record["dataset"]].append(record)

    for dataset, dataset_records in by_dataset.items():
        runs = defaultdict(list)
        for record in dataset_records:
            key = independent_run_key(dataset, record["path"].name)
            if key is not None:
                runs[key[0]].append((key[1], record["split"], str(record["path"])))
        for run, sequence in runs.items():
            sequence.sort()
            component = [sequence[0]] if sequence else []
            for current in sequence[1:]:
                if current[0] - component[-1][0] <= 3:
                    component.append(current)
                else:
                    splits = {item[1] for item in component}
                    if len(splits) > 1:
                        errors.append({"dataset": dataset, "kind": "frame_run",
                                       "key": run, "splits": sorted(splits)})
                    component = [current]
            splits = {item[1] for item in component}
            if len(splits) > 1:
                errors.append({"dataset": dataset, "kind": "frame_run",
                               "key": run, "splits": sorted(splits)})

        if dataset == "visual":
            specimens = defaultdict(set)
            for record in dataset_records:
                match = re.match(r"^(\d+)_", RE_RF.sub("", record["path"].name))
                if match:
                    specimens[match.group(1)].add(record["split"])
            for specimen, splits in specimens.items():
                if len(splits) > 1:
                    errors.append({"dataset": dataset, "kind": "specimen",
                                   "key": specimen, "splits": sorted(splits)})
    return errors


def audit_manifest(root: Path, manifest_path: Path):
    errors = []
    if not manifest_path.exists():
        return {"rows": 0, "errors": ["manifest missing"]}
    rows = list(csv.DictReader(manifest_path.open(newline="")))
    required = {
        "dataset", "original_path", "kept", "copy_group", "representative_path",
        "representative_clean_path", "content_scene_group", "scene_group",
        "split_block", "new_split",
    }
    missing = sorted(required - set(rows[0] if rows else []))
    if missing:
        return {"rows": len(rows), "errors": [f"missing columns: {missing}"]}

    kept_originals = {row["original_path"] for row in rows if row["kept"] == "True"}
    copy_groups = defaultdict(list)
    scene_splits = defaultdict(set)
    block_splits = defaultdict(set)
    workspace = root.parent
    for row_number, row in enumerate(rows, 2):
        copy_groups[row["copy_group"]].append(row)
        scene_splits[row["scene_group"]].add(row["new_split"])
        block_splits[row["split_block"]].add(row["new_split"])
        if row["representative_path"] not in kept_originals:
            errors.append(f"row {row_number}: representative is not retained")
        if not (workspace / row["original_path"]).is_file():
            errors.append(f"row {row_number}: original path missing")
        if not (root / row["representative_clean_path"]).is_file():
            errors.append(f"row {row_number}: clean representative missing")
        if row["kept"] == "True" and row["original_path"] != row["representative_path"]:
            errors.append(f"row {row_number}: retained row points elsewhere")
    for group, members in copy_groups.items():
        retained = sum(row["kept"] == "True" for row in members)
        if retained != 1:
            errors.append(f"{group}: expected one retained representative, found {retained}")
    for group, splits in scene_splits.items():
        if len(splits) != 1:
            errors.append(f"{group}: spans splits {sorted(splits)}")
    for group, splits in block_splits.items():
        if len(splits) != 1:
            errors.append(f"{group}: spans splits {sorted(splits)}")
    return {"rows": len(rows), "copy_groups": len(copy_groups),
            "scene_groups": len(scene_splits), "split_blocks": len(block_splits),
            "errors": errors}


def audit_split_lists(root: Path, records):
    errors = []
    expected = defaultdict(set)
    for record in records:
        expected[(record["dataset"], record["split"])].add(
            f"{record['split']}/images/{record['path'].name}")
    for key, paths in expected.items():
        dataset, split = key
        split_file = root / "splits" / f"{dataset}_{split}.txt"
        if not split_file.exists():
            errors.append(f"missing split file {split_file.name}")
            continue
        lines = [line for line in split_file.read_text().splitlines() if line]
        if len(lines) != len(set(lines)):
            errors.append(f"duplicate lines in {split_file.name}")
        actual = set(lines)
        if actual != paths:
            errors.append(
                f"{split_file.name}: {len(paths - actual)} missing, "
                f"{len(actual - paths)} extra")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path("clean"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        sys.exit(f"Not a directory: {root}")
    out = args.out.resolve() if args.out else root / "audit"
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or root / "manifest.csv"

    records = []
    inventory = {}
    integrity_errors = []
    print(f"Root: {root}")
    for dataset, directory, data_yaml in discover_datasets(root):
        names = load_names(data_yaml)
        config = yaml.safe_load(data_yaml.read_text()) or {}
        path_field = str(config.get("path", ""))
        if Path(path_field).is_absolute():
            integrity_errors.append(f"{dataset}: data.yaml path is absolute")
        dataset_record = {"classes": names, "splits": {}, "total": {}}
        total_boxes = 0
        total_images = 0
        total_background = 0
        per_class = Counter()
        for split in SPLITS:
            image_dir = directory / split / "images"
            label_dir = directory / split / "labels"
            images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXT)
            labels = {p.stem for p in label_dir.glob("*.txt")}
            image_stems = {p.stem for p in images}
            for orphan in sorted(labels - image_stems):
                integrity_errors.append(f"{dataset}/{split}: orphan label {orphan}.txt")
            split_boxes = 0
            split_background = 0
            split_classes = Counter()
            sizes = Counter()
            areas = []
            for image_path in images:
                label_path = label_dir / f"{image_path.stem}.txt"
                boxes, errors = parse_label(label_path, len(names))
                for error in errors:
                    integrity_errors.append(f"{dataset}/{split}/{image_path.name}: {error}")
                try:
                    with Image.open(image_path) as image:
                        sizes[f"{image.width}x{image.height}"] += 1
                    vector = descriptor(image_path)
                except Exception as error:
                    integrity_errors.append(
                        f"{dataset}/{split}/{image_path.name}: unreadable image: {error}")
                    continue
                for cls, _, _, width, height in boxes:
                    split_classes[cls] += 1
                    areas.append(width * height)
                if not boxes:
                    split_background += 1
                split_boxes += len(boxes)
                records.append({"dataset": dataset, "split": split,
                                "path": image_path, "descriptor": vector})
            dataset_record["splits"][split] = {
                "images": len(images), "boxes": split_boxes,
                "background_images": split_background,
                "per_class": {names[c]: split_classes[c] for c in range(len(names))},
                "resolutions": dict(sizes),
                "median_box_area_fraction": (
                    round(float(np.median(areas)), 6) if areas else None),
            }
            total_images += len(images)
            total_boxes += split_boxes
            total_background += split_background
            per_class.update(split_classes)
        dataset_record["total"] = {
            "images": total_images, "boxes": total_boxes,
            "background_images": total_background,
            "per_class": {names[c]: per_class[c] for c in range(len(names))},
        }
        inventory[dataset] = dataset_record
        print(f"  {dataset}: {total_images} images, {total_boxes} boxes")

    if not records:
        sys.exit("No rebuilt datasets found")

    D = np.vstack([record["descriptor"] for record in records]).astype(np.float32)
    copy_counts = Counter()
    scene_candidate_counts = Counter()
    scene_confirmed_counts = Counter()
    examples = defaultdict(list)
    max_scores = defaultdict(float)
    print(f"Comparing {len(records)} retained images by NCC")
    confirmation_cache = {}
    for left, right, score in pairs_above(D, T_SCENE):
        a, b = records[left], records[right]
        same_dataset = a["dataset"] == b["dataset"]
        same_split = same_dataset and a["split"] == b["split"]
        kind = "within_split" if same_split else (
            "cross_split" if same_dataset else "cross_dataset")
        family = "copy" if score >= T_COPY else "scene_candidate_32"
        counts = copy_counts if family == "copy" else scene_candidate_counts
        counts[kind] += 1
        max_scores[f"{family}_{kind}"] = max(
            max_scores[f"{family}_{kind}"], score)
        key = f"{family}_{kind}"
        if len(examples[key]) < 20:
            examples[key].append({
                "score": round(score, 6),
                "image_a": str(a["path"].relative_to(root)),
                "image_b": str(b["path"].relative_to(root)),
            })
        if family == "copy" or not scene_semantically_compatible(a, b):
            continue
        for index in (left, right):
            if index not in confirmation_cache:
                confirmation_cache[index] = descriptor_at(
                    records[index]["path"], SCENE_CONFIRM_THUMB)
        confirmation_score = pair_ncc(
            confirmation_cache[left], confirmation_cache[right],
            SCENE_CONFIRM_THUMB)
        if confirmation_score < T_SCENE:
            continue
        scene_confirmed_counts[kind] += 1
        max_scores[f"scene_confirmed_128_{kind}"] = max(
            max_scores[f"scene_confirmed_128_{kind}"], confirmation_score)
        confirmed_key = f"scene_confirmed_128_{kind}"
        if len(examples[confirmed_key]) < 20:
            examples[confirmed_key].append({
                "score": round(confirmation_score, 6),
                "candidate_score_32": round(score, 6),
                "image_a": str(a["path"].relative_to(root)),
                "image_b": str(b["path"].relative_to(root)),
            })

    structural_errors = structural_leaks(records)
    manifest = audit_manifest(root, manifest_path)
    split_list_errors = audit_split_lists(root, records)

    failures = []
    if integrity_errors:
        failures.append(f"{len(integrity_errors)} integrity errors")
    if sum(copy_counts.values()):
        failures.append(f"{sum(copy_counts.values())} retained copy pairs")
    if scene_candidate_counts["cross_split"]:
        failures.append(
            f"{scene_candidate_counts['cross_split']} cross-split scene candidates")
    if scene_confirmed_counts["cross_split"]:
        failures.append(
            f"{scene_confirmed_counts['cross_split']} cross-split confirmed scenes")
    if structural_errors:
        failures.append(f"{len(structural_errors)} structural split leaks")
    if manifest["errors"]:
        failures.append(f"{len(manifest['errors'])} manifest errors")
    if split_list_errors:
        failures.append(f"{len(split_list_errors)} split-list errors")

    report = {
        "schema_version": 3,
        "root": str(root),
        "thresholds": {"copy_ncc": T_COPY, "scene_ncc": T_SCENE,
                       "candidate_thumb": THUMB,
                       "confirmation_thumb": SCENE_CONFIRM_THUMB},
        "status": "FAIL" if failures else "PASS",
        "failures": failures,
        "inventory": inventory,
        "ncc_pairs": {
            "copy": dict(copy_counts),
            "scene_candidate_32": dict(scene_candidate_counts),
            "scene_confirmed_128": dict(scene_confirmed_counts),
            "max_scores": {key: round(value, 6) for key, value in max_scores.items()},
            "examples": dict(examples),
        },
        "structural_split_leaks": structural_errors,
        "integrity_errors": integrity_errors,
        "manifest": manifest,
        "split_list_errors": split_list_errors,
    }
    (out / "audit_report.json").write_text(json.dumps(report, indent=2))

    with (out / "inventory.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "split", "images", "boxes", "background_images"])
        for dataset, record in inventory.items():
            for split, values in record["splits"].items():
                writer.writerow([dataset, split, values["images"], values["boxes"],
                                 values["background_images"]])

    print(f"Copy pairs: {dict(copy_counts)}")
    print(f"Scene candidates at 32: {dict(scene_candidate_counts)}")
    print(f"Confirmed scenes at 128: {dict(scene_confirmed_counts)}")
    print(f"Structural leaks: {len(structural_errors)}")
    print(f"Manifest errors: {len(manifest['errors'])}")
    print(f"Split-list errors: {len(split_list_errors)}")
    print(f"Result: {report['status']}")
    print(f"Wrote {out / 'audit_report.json'}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
