#!/usr/bin/env python3
"""Acceptance checks for a candidate clean weld dataset rebuild.

Run this after weld_audit.py. It verifies the audit result, rebuild method,
multilabel split balance, release lists, lineage, portable paths, and source
tree stat fingerprint. It writes acceptance_report.json and exits nonzero on
any failure.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import yaml

from weld_rebuild_clean import (
    DATASETS, IMG_EXT, SCENE_CONFIRM_THUMB, T_COPY, T_SCENE, run_key,
    source_id, specimen_key,
)

SPLITS = ("train", "valid", "test")


def tree_stat_fingerprint(root: Path):
    digest = hashlib.sha256()
    count = 0
    size = 0
    for tag in DATASETS:
        dataset = root / DATASETS[tag]
        for path in sorted(p for p in dataset.rglob("*") if p.is_file()):
            stat = path.stat()
            digest.update(str(path.relative_to(root)).encode())
            digest.update(str(stat.st_size).encode())
            digest.update(str(stat.st_mtime_ns).encode())
            count += 1
            size += stat.st_size
    return {"files": count, "bytes": size, "stat_sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--source-stat-sha256")
    args = parser.parse_args()

    candidate = args.candidate.resolve()
    workspace = candidate.parent
    failures = []
    checks = {}

    for script in ("weld_rebuild_clean.py", "weld_audit.py",
                   "weld_diagnostics.py", "weld_acceptance.py"):
        try:
            ast.parse((workspace / script).read_text())
        except Exception as error:
            failures.append(f"{script}: syntax error: {error}")
    checks["python_syntax"] = "PASS" if not failures else "FAIL"

    audit_path = candidate / "audit" / "audit_report.json"
    if not audit_path.exists():
        failures.append("authoritative audit report missing")
        audit = {}
    else:
        audit = json.loads(audit_path.read_text())
        if audit.get("status") != "PASS":
            failures.append("authoritative NCC audit did not pass")
        thresholds = audit.get("thresholds", {})
        if thresholds.get("confirmation_thumb") != SCENE_CONFIRM_THUMB:
            failures.append("authoritative audit did not use scene confirmation")
    checks["authoritative_audit"] = audit.get("status", "MISSING")

    rebuild_path = candidate / "rebuild_report.json"
    if not rebuild_path.exists():
        failures.append("rebuild report missing")
        rebuild = {}
    else:
        rebuild = json.loads(rebuild_path.read_text())
        expected_methods = {
            "copy_selection": "maximum_independent_set_at_threshold",
            "copy_chain_diagnostic": "complete_linkage",
            "scene_linkage": "complete",
            "scene_confirmation": "ncc_at_128_after_ncc_at_32",
            "split_method": "mixed_integer_multilabel_group_balancing",
        }
        for key, expected in expected_methods.items():
            if rebuild.get(key) != expected:
                failures.append(f"rebuild {key} is {rebuild.get(key)!r}, expected {expected!r}")
        if rebuild.get("t_copy") != T_COPY:
            failures.append("copy threshold changed")
        if rebuild.get("t_scene") != T_SCENE:
            failures.append("scene threshold changed")
        if rebuild.get("scene_confirmation_thumb") != SCENE_CONFIRM_THUMB:
            failures.append("scene confirmation size changed")
        expected_scopes = {
            "weld_surface_v2": "scene_and_split",
            "twoclass": "scene_and_split",
            "kaggle_v2": "split_only",
            "visual": "unused",
            "lohi_weld": "unused",
        }
        for dataset, scope in expected_scopes.items():
            actual = rebuild.get("datasets", {}).get(dataset, {}).get("frame_run_scope")
            if actual != scope:
                failures.append(
                    f"{dataset} frame run scope is {actual!r}, expected {scope!r}")
    checks["rebuild_methods"] = {
        key: rebuild.get(key) for key in
        ("copy_selection", "copy_chain_diagnostic", "scene_linkage",
         "scene_confirmation", "split_method", "t_copy", "t_scene",
         "scene_candidate_thumb", "scene_confirmation_thumb")
    }

    balance_failures = []
    for dataset, record in rebuild.get("datasets", {}).items():
        balance = record.get("split_balance", {})
        for feature, values in balance.get("features", {}).items():
            granularity = float(values.get("granularity_pp", 0.0))
            # Scene count is secondary to image and label balance. With only
            # 34 scenes, moving one large group can improve scene count while
            # severely damaging every class total. Allow two scene-count
            # quanta, while all data-bearing features retain one-quantum limits.
            tolerance = (max(6.0, 2.0 * granularity + 0.5)
                         if feature == "groups"
                         else max(2.0, granularity + 0.5))
            for split, split_values in values.get("splits", {}).items():
                deviation = abs(float(split_values["deviation_pp"]))
                if deviation > tolerance + 1e-9:
                    balance_failures.append(
                        f"{dataset}/{feature}/{split}: {deviation:.3f} pp exceeds "
                        f"{tolerance:.3f} pp")
    failures.extend(balance_failures)
    checks["multilabel_balance"] = {
        "status": "PASS" if not balance_failures else "FAIL",
        "errors": balance_failures,
    }

    # LoHi independence. This used to assert one scene per image, which was
    # right while only the high resolution acquisition was converted. The low
    # resolution acquisition ships exactly two images per production cycle, so
    # the invariant is now stated in terms of the cycle key: an image with no
    # recoverable cycle id is its own scene, and images sharing a cycle id are
    # one scene. Anything else means the cycle key stopped parsing.
    lohi = rebuild.get("datasets", {}).get("lohi_weld", {})
    lohi_images = lohi.get("after", {}).get("images")
    lohi_scenes = lohi.get("scene_groups")
    staging = workspace / DATASETS["lohi_weld"] / "train" / "images"
    expected_scenes = None
    if staging.is_dir():
        keyed, unkeyed = set(), 0
        for image in staging.iterdir():
            if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            key = specimen_key("lohi_weld", image.name)
            if key is None:
                unkeyed += 1
            else:
                keyed.add(key)
        expected_scenes = unkeyed + len(keyed)
    if expected_scenes is not None and lohi_scenes != expected_scenes:
        failures.append(
            f"LoHi scene count {lohi_scenes} does not equal "
            f"{expected_scenes}, the number of distinct production cycles")
    checks["lohi_independence"] = {
        "images": lohi_images,
        "scene_groups": lohi_scenes,
        "expected_scene_groups": expected_scenes,
        "rule": "one scene per production cycle, not per image",
        "status": ("PASS" if expected_scenes is not None
                   and lohi_scenes == expected_scenes else "FAIL"),
    }

    parsing_errors = []
    kaggle_zero = "SampleV2_1_mp4-0_jpg.rf.32dc225d9b5d11d01260f75c918a9961.jpg"
    kaggle_ninety = "SampleV2_1_mp4-90_jpg.rf.19ad4979865d6aba1cfaaa23f1dfb1f9.jpg"
    weld_jpeg = "LoF-9-51-_jpeg.rf.4ceca8458447417a864703039d66aa38.jpg"
    weld_jpg = "LoF-9-51-_jpg.rf.8240f1b2d02a19347dc8bd3eb8d3e1bb.jpg"
    if source_id("kaggle_v2", kaggle_zero) == source_id("kaggle_v2", kaggle_ninety):
        parsing_errors.append("Kaggle frame 0 and frame 90 still collide")
    if source_id("weld_surface_v2", weld_jpeg) == source_id("weld_surface_v2", weld_jpg):
        parsing_errors.append("Weld Surface jpg and jpeg sources still collide")
    if run_key("lohi_weld", "25_100.jpg") is not None:
        parsing_errors.append("LoHi is still parsed as a frame run")
    failures.extend(parsing_errors)
    checks["dataset_specific_parsing"] = {
        "status": "PASS" if not parsing_errors else "FAIL", "errors": parsing_errors}

    manifest_path = candidate / "manifest.csv"
    manifest_errors = []
    if manifest_path.exists():
        rows = list(csv.DictReader(manifest_path.open(newline="")))
        kept = {row["original_path"] for row in rows if row["kept"] == "True"}
        for row_number, row in enumerate(rows, 2):
            if row["representative_path"] not in kept:
                manifest_errors.append(f"row {row_number}: unresolved representative")
            if row["copy_match_method"] in {"ncc", "filename+ncc"}:
                if float(row["copy_ncc_to_representative"]) + 1e-6 < T_COPY:
                    manifest_errors.append(f"row {row_number}: NCC below copy threshold")
        if len(kept) != sum(
                item["after"]["images"] for item in rebuild.get("datasets", {}).values()):
            manifest_errors.append("retained manifest count disagrees with rebuild report")
        visual_specimens = {}
        pattern = re.compile(r"^(\d+)_")
        for row in rows:
            if row["kept"] != "True" or row["dataset"] != "visual":
                continue
            match = pattern.match(Path(row["original_path"]).name)
            if match:
                visual_specimens.setdefault(row["scene_group"], set()).add(match.group(1))
        mixed = {group: ids for group, ids in visual_specimens.items() if len(ids) > 1}
        if mixed:
            manifest_errors.append(
                f"{len(mixed)} Visual scene groups mix known specimen ids")
    else:
        manifest_errors.append("manifest missing")
        rows = []
    failures.extend(manifest_errors)
    checks["manifest_lineage"] = {
        "rows": len(rows), "status": "PASS" if not manifest_errors else "FAIL",
        "errors": manifest_errors,
    }

    # A config is portable when it resolves the same way from any working
    # directory. An earlier version of this check asserted `path == "."`, which
    # enforced the opposite: ultralytics keeps a relative `path` that exists,
    # and "." always exists, so every split resolved against the caller's cwd.
    # The check now does what a trainer does. It resolves each split from the
    # yaml's own parent, confirms the directory is there, and confirms the
    # image count matches the frozen split list.
    yaml_errors = []
    skipped_arm = []
    for data_yaml in sorted(candidate.glob("*/data.yaml")):
        tag = data_yaml.parent.name
        # ARM DATASETS ARE SIBLINGS IN THIS TREE AND ARE NOT OURS TO CHECK.
        # `grid/arms/build_arm_datasets.py` materialises lohi_high, lohi_low
        # and the three pools under the same candidate directory, from the
        # frozen split lists, and registers their hashes in
        # grid/arms/arm_split_hashes.json rather than in this tree's
        # splits/ directory. Globbing for data.yaml finds them, and this
        # check then looked for a split list that was never meant to be
        # here and died on a FileNotFoundError. Skip them BY NAME and
        # report the skip, so an arm dataset that goes missing is visible
        # rather than silently excused.
        if tag not in DATASETS:
            skipped_arm.append(tag)
            continue
        config = yaml.safe_load(data_yaml.read_text()) or {}
        if "path" in config:
            yaml_errors.append(
                f"{tag}: has a 'path' key, which ultralytics resolves against "
                f"the working directory. Omit it so the yaml anchors to its "
                f"own folder.")
        for key, split in (("train", "train"), ("val", "valid"), ("test", "test")):
            rel = config.get(key)
            if not rel:
                yaml_errors.append(f"{tag}: missing '{key}'")
                continue
            resolved = (data_yaml.parent / rel).resolve()
            if not resolved.is_dir():
                yaml_errors.append(f"{tag}: '{key}' does not resolve to a "
                                   f"directory ({resolved})")
                continue
            listed = len((candidate / "splits" / f"{tag}_{split}.txt")
                         .read_text().strip().splitlines())
            found = sum(1 for p in resolved.iterdir() if p.suffix.lower() in IMG_EXT)
            if found != listed:
                yaml_errors.append(
                    f"{tag}/{key}: {found} images on disk but {listed} in "
                    f"{tag}_{split}.txt")
    failures.extend(yaml_errors)
    checks["portable_yaml"] = {
        "status": "PASS" if not yaml_errors else "FAIL", "errors": yaml_errors,
        "skipped_arm_datasets": sorted(skipped_arm)}

    split_hashes = {}
    for split_file in sorted((candidate / "splits").glob("*.txt")):
        split_hashes[split_file.name] = hashlib.sha256(split_file.read_bytes()).hexdigest()
    (candidate / "split_hashes.json").write_text(json.dumps(split_hashes, indent=2))
    checks["split_files"] = {"count": len(split_hashes), "sha256_file": "split_hashes.json"}
    if len(split_hashes) != len(DATASETS) * len(SPLITS):
        failures.append(f"expected {len(DATASETS) * len(SPLITS)} split files")

    source_fingerprint = tree_stat_fingerprint(workspace)
    if (args.source_stat_sha256 and
            source_fingerprint["stat_sha256"] != args.source_stat_sha256):
        failures.append("source dataset stat fingerprint changed")
    checks["source_tree_fingerprint"] = source_fingerprint

    report = {
        "schema_version": 1,
        "candidate": str(candidate),
        "status": "FAIL" if failures else "PASS",
        "failures": failures,
        "checks": checks,
    }
    (candidate / "acceptance_report.json").write_text(json.dumps(report, indent=2))
    print(f"Acceptance result: {report['status']}")
    for failure in failures:
        print(f"  FAIL: {failure}")
    print(f"Wrote {candidate / 'acceptance_report.json'}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
