#!/usr/bin/env python3
"""Build reproducible tables and uncertainty estimates after grid retrieval.

This runs on the laptop, not while an H200 slice is held. It enriches thin
in-domain classes with a split-block cluster bootstrap, writes cell and
aggregate tables, constructs same-family transfer ratios, records dirty versus
clean estimate shifts, exports reliability bins, and performs Holm-corrected
paired family comparisons over source-target cells.

Usage:
    python3 grid/postprocess_results.py
    python3 grid/postprocess_results.py --results-dir grid_results/results
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C  # noqa: E402
import metrics as M  # noqa: E402
from evaluate_cell import json_safe, load_ground_truth  # noqa: E402


METRICS = [
    "image_ap", "image_auroc", "image_ece", "image_brier",
    "box_map50", "box_map50_95",
    "recall_at_0.1_fp_per_image", "recall_at_0.5_fp_per_image",
    "recall_at_1.0_fp_per_image",
]


def finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def row_from_target(record: dict, result_key: str, entry: dict) -> dict:
    cell = record["cell"]
    target, key_split = C.split_result_key(result_key)
    image = entry["image_level"]
    box = entry["box_level"]
    row = {
        "name": record["name"],
        "kind": cell.get("kind", "main"),
        "source": cell["source"],
        "family": cell["family"],
        "seed": cell["seed"],
        "target": target,
        "result_key": result_key,
        # "downloaded" or "audited". A downloaded-data cell contributes one
        # row of each, on the same target.
        "test_split": entry.get(
            "test_split", "downloaded" if key_split == "native"
            and cell.get("kind") == "dirty" else "audited"),
        "n_test_images": entry["n_test_images"],
        "n_positive_images": image["n_pos"],
        "n_negative_images": image["n_neg"],
        "image_ap": image.get("ap"),
        "image_auroc": image.get("auroc"),
        "image_ece": image.get("calibration", {}).get("ece"),
        "image_brier": image.get("calibration", {}).get("brier"),
        "box_map50": box.get("map50"),
        "box_map50_95": box.get("map50_95"),
    }
    for budget in C.FP_PER_IMAGE_BUDGETS:
        key = f"recall_at_{budget}_fp_per_image"
        row[key] = box.get(key)
    return row


def bootstrap_thin_classes(record: dict, raw, enriched: dict) -> None:
    cell = record["cell"]
    if cell.get("kind", "main") == "dirty":
        return
    source = cell["source"]
    target_entry = enriched["targets"].get(source)
    if not target_entry or "per_class_in_domain" not in target_entry:
        return

    gt = load_ground_truth(source, kind=cell.get("kind", "main"))
    prefix = source + "__"
    pred_img = raw[prefix + "pred_image"]
    pred_box = raw[prefix + "pred_box"]
    pred_score = raw[prefix + "pred_score"]
    pred_class = raw[prefix + "pred_class"]

    for class_name, class_result in target_entry["per_class_in_domain"].items():
        if not class_result.pop("cluster_bootstrap_required", False):
            continue
        class_id = C.DATASETS[source]["classes"].index(class_name)
        take = pred_class == class_id
        gts = gt["gt_by_class"].get(class_id, {})
        seed_offset = sum(ord(ch) for ch in
                          f"{record['name']}:{source}:{class_id}")
        # transfer_bootstrap.fast_cluster_bootstrap is a drop-in for
        # M.bootstrap_detection_ap_by_cluster: same draws, same seed, same
        # numbers, verified by equivalent_to_metrics on real predictions. The
        # difference is that it matches boxes once per IoU threshold instead of
        # once per draw. Measured on a real thin class, 2,000 draws fell from
        # roughly 12 minutes to 1.3 seconds. At 48 thin class cells the slow
        # path made `python3 grid/analysis.py` a ten hour command, which is
        # why the one-command promise had quietly stopped being true.
        from transfer_bootstrap import fast_cluster_bootstrap
        class_result["cluster_bootstrap_95ci"] = fast_cluster_bootstrap(
            pred_img[take], pred_box[take], pred_score[take], gts,
            gt["split_blocks"], C.BOOTSTRAP_RESAMPLES,
            seed=C.BOOTSTRAP_SEED + seed_offset)


def aggregate_rows(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        key = (row["kind"], row["source"], row["family"], row["target"])
        groups[key].append(row)
    out = []
    for key, members in sorted(groups.items()):
        item = dict(zip(("kind", "source", "family", "target"), key))
        item["seeds"] = len(members)
        for metric in METRICS:
            values = [float(r[metric]) for r in members if finite(r.get(metric))]
            item[metric + "_mean"] = float(np.mean(values)) if values else None
            item[metric + "_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else None)
        out.append(item)
    return out


def transfer_rows(rows: list[dict]) -> list[dict]:
    index = {(r["kind"], r["source"], r["family"], r["seed"], r["target"]): r
             for r in rows}
    out = []
    for row in rows:
        if row["kind"] != "main":
            continue
        target = row["target"]
        denominator_kind = "main" if target in C.TRAIN_SOURCES else "reference"
        denominator = index.get((denominator_kind, target, row["family"],
                                 row["seed"], target))
        if denominator is None:
            continue
        item = {k: row[k] for k in
                ("source", "family", "seed", "target")}
        item["denominator_kind"] = denominator_kind
        for metric in ("image_ap", "box_map50", "box_map50_95"):
            num, den = row.get(metric), denominator.get(metric)
            item[metric + "_numerator"] = num
            item[metric + "_denominator"] = den
            item[metric + "_ratio"] = (
                float(num) / float(den)
                if finite(num) and finite(den) and float(den) > 0 else None)
        out.append(item)
    return out


def gap_rows(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        if row["kind"] == "main":
            groups[(row["source"], row["family"], row["seed"])].append(row)
    out = []
    for (source, family, seed), members in sorted(groups.items()):
        own = next((r["image_ap"] for r in members if r["target"] == source), None)
        cross = [r["image_ap"] for r in members
                 if r["target"] != source and finite(r.get("image_ap"))]
        out.append({"source": source, "family": family, "seed": seed,
                    "in_domain_image_ap": own,
                    "mean_cross_domain_image_ap": float(np.mean(cross)),
                    "generalization_gap": float(own) - float(np.mean(cross))
                    if finite(own) and cross else None})
    return out


def dirty_shift_rows(rows: list[dict]) -> list[dict]:
    """The three-number downloaded-versus-audited comparison.

      A  downloaded train -> downloaded test   what a practitioner publishes
      B  downloaded train -> audited test      the same weights, honest test
      C  audited train    -> audited test      the main grid diagonal

    Two differences are reported and they mean different things.

      A - B  only the test set changes. The weights are identical. This
             isolates how much of the published-style score depends on being
             graded on a partition that shares content with training.
      A - C  the whole pipeline changes, training data and test set together.
             This is the practical gap between what a downloaded dataset
             appears to give and what an audited one does.

    Neither is a causal estimate of leakage. Both are estimate shifts under a
    change of protocol, and the caption must say so. A and B share weights but
    not a test set; A and C share neither.
    """
    index = {(r["kind"], r["source"], r["family"], r["seed"], r["result_key"]): r
             for r in rows}
    out = []
    for a in rows:
        if a["kind"] != "dirty" or a["test_split"] != "downloaded":
            continue
        src, fam, seed = a["source"], a["family"], a["seed"]
        b = index.get(("dirty", src, fam, seed, src + C.CLEAN_TEST_SUFFIX))
        # The audited diagonal exists once per seed for this family.
        c = index.get(("main", src, fam, seed, src))
        item = {"source": src, "family": fam, "seed": seed,
                "comparison": "estimate shift under a change of protocol, "
                              "not a causal effect of leakage",
                "A_downloaded_train_downloaded_test": True,
                "B_downloaded_train_audited_test": b is not None,
                "C_audited_train_audited_test": c is not None}
        for metric in ("image_ap", "box_map50", "box_map50_95"):
            va = a.get(metric)
            vb = b.get(metric) if b else None
            vc = c.get(metric) if c else None
            item["A_" + metric] = va
            item["B_" + metric] = vb
            item["C_" + metric] = vc
            item["shift_test_set_only_" + metric] = (
                float(va) - float(vb) if finite(va) and finite(vb) else None)
            item["shift_full_pipeline_" + metric] = (
                float(va) - float(vc) if finite(va) and finite(vc) else None)
        out.append(item)
    return out


def reliability_rows(records: list[dict]) -> list[dict]:
    out = []
    for record in records:
        cell = record["cell"]
        for target, entry in record["targets"].items():
            cal = entry["image_level"].get("calibration", {})
            for i, bin_row in enumerate(cal.get("bins", [])):
                out.append({"name": record["name"],
                            "kind": cell.get("kind", "main"),
                            "source": cell["source"],
                            "family": cell["family"],
                            "seed": cell["seed"], "target": target,
                            "bin": i, **bin_row})
    return out


def kaggle_sensitivity_rows(record: dict, raw) -> list[dict]:
    """Re-score the grid with Bad Weld treated as a positive image label.

    Training never changes. The raw sidecar contains all native-class
    predictions, so both semantic readings can be reported from one model.
    """
    cell = record["cell"]
    source = cell["source"]
    out = []
    for target, entry in record["targets"].items():
        if source != "kaggle_v2" and target != "kaggle_v2":
            continue
        prefix = target + "__"
        pred_img = raw[prefix + "pred_image"]
        pred_score = raw[prefix + "pred_score"]
        pred_class = raw[prefix + "pred_class"]
        n_images = len(raw[prefix + "image_label"])
        keep = ({0, 2} if source == "kaggle_v2" else C.defect_ids(source))
        scores = np.zeros(n_images, dtype=np.float32)
        for image_id in range(n_images):
            take = (pred_img == image_id) & np.isin(pred_class, list(keep))
            if take.any():
                scores[image_id] = float(pred_score[take].max())

        if target == "kaggle_v2":
            gt = load_ground_truth(target, kind=cell.get("kind", "main"))
            labels = np.array([
                bool(len(gt["gt_by_class"].get(0, {}).get(i, [])) or
                     len(gt["gt_by_class"].get(2, {}).get(i, [])))
                for i in range(n_images)
            ], dtype=bool)
        else:
            labels = raw[prefix + "image_label"].astype(bool)

        alt = M.binary_metrics(scores, labels, C.MIN_NEGATIVES_FOR_FPR,
                               C.ECE_BINS)
        current = entry["image_level"]
        out.append({
            "name": record["name"], "kind": cell.get("kind", "main"),
            "source": source, "family": cell["family"],
            "seed": cell["seed"], "target": target,
            "primary_rule": "only Defect is positive",
            "sensitivity_rule": "Bad Weld or Defect is positive",
            "primary_ap": current.get("ap"),
            "sensitivity_ap": alt.get("ap"),
            "primary_auroc": current.get("auroc"),
            "sensitivity_auroc": alt.get("auroc"),
            "primary_positive_images": current.get("n_pos"),
            "sensitivity_positive_images": alt.get("n_pos"),
            "sensitivity_negative_images": alt.get("n_neg"),
            "sensitivity_ece": alt.get("calibration", {}).get("ece"),
        })
    return out


def family_tests(rows: list[dict]) -> list[dict]:
    main = [r for r in rows if r["kind"] == "main"]
    by_cell = defaultdict(list)
    for row in main:
        by_cell[(row["source"], row["target"], row["family"])].append(row)

    tests = []
    for metric in ("image_ap", "box_map50", "box_map50_95"):
        means = {}
        for key, members in by_cell.items():
            values = [r[metric] for r in members if finite(r.get(metric))]
            if values:
                means[key] = float(np.mean(values))
        for a, b in itertools.combinations(C.FAMILIES, 2):
            domain_keys = sorted({(s, t) for s, t, f in means if f == a} &
                                 {(s, t) for s, t, f in means if f == b})
            va = [means[(s, t, a)] for s, t in domain_keys]
            vb = [means[(s, t, b)] for s, t in domain_keys]
            if len(va) < 3 or np.allclose(va, vb):
                p = None
            else:
                p = float(wilcoxon(va, vb, zero_method="wilcox").pvalue)
            tests.append({"metric": metric, "family_a": a, "family_b": b,
                          "paired_domain_cells": len(va),
                          "mean_difference_a_minus_b":
                              float(np.mean(np.asarray(va) - np.asarray(vb)))
                              if va else None,
                          "p_raw": p, "p_holm": None})

    valid = sorted((row["p_raw"], i) for i, row in enumerate(tests)
                   if row["p_raw"] is not None)
    running = 0.0
    m = len(valid)
    for rank, (p, index) in enumerate(valid):
        adjusted = min(1.0, (m - rank) * p)
        running = max(running, adjusted)
        tests[index]["p_holm"] = running
    return tests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path,
                        default=C.PROJECT / "grid_results" / "results")
    parser.add_argument("--out-dir", type=Path,
                        default=C.PROJECT / "grid_results" / "summary")
    args = parser.parse_args()

    eval_files = sorted(args.results_dir.glob("eval__*.json"))
    if not eval_files:
        raise SystemExit(f"no evaluation files under {args.results_dir}")

    expected = {C.cell_name(cell): cell for cell in C.all_cells()}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    enriched_dir = args.out_dir / "cells"
    enriched_dir.mkdir(exist_ok=True)
    records, rows, kaggle_sensitivity, seen = [], [], [], set()
    for path in eval_files:
        record = json.loads(path.read_text())
        name = record.get("name")
        if name not in expected:
            raise SystemExit(f"unexpected or stale evaluation file: {path}")
        if name in seen:
            raise SystemExit(f"duplicate evaluation record for {name}")
        if record.get("cell") != expected[name]:
            raise SystemExit(f"cell metadata mismatch in {path}")
        if record.get("config_sha256_16") != C.config_fingerprint():
            raise SystemExit(f"configuration hash mismatch in {path}")
        wanted_targets = set(C.eval_targets(expected[name]))
        if set(record.get("targets", {})) != wanted_targets:
            raise SystemExit(f"target set mismatch in {path}")
        raw_path = args.results_dir / path.name.replace("eval__", "raw__").replace(".json", ".npz")
        if not raw_path.is_file():
            raise SystemExit(f"missing raw prediction archive: {raw_path}")
        with np.load(raw_path, allow_pickle=False) as raw:
            bootstrap_thin_classes(record, raw, record)
            kaggle_sensitivity.extend(kaggle_sensitivity_rows(record, raw))
        (enriched_dir / path.name).write_text(
            json.dumps(json_safe(record), indent=2))
        records.append(record)
        seen.add(name)
        rows.extend(row_from_target(record, target, entry)
                    for target, entry in record["targets"].items())

    write_rows(args.out_dir / "cells.csv", rows)
    write_rows(args.out_dir / "aggregate.csv", aggregate_rows(rows))
    write_rows(args.out_dir / "transfer_ratios.csv", transfer_rows(rows))
    write_rows(args.out_dir / "generalization_gaps.csv", gap_rows(rows))
    write_rows(args.out_dir / "dirty_estimate_shifts.csv", dirty_shift_rows(rows))
    write_rows(args.out_dir / "reliability_bins.csv", reliability_rows(records))
    write_rows(args.out_dir / "kaggle_bad_weld_sensitivity.csv",
               kaggle_sensitivity)
    write_rows(args.out_dir / "family_wilcoxon_holm.csv", family_tests(rows))
    missing = sorted(set(expected) - seen)
    manifest = {"schema_version": 1, "evaluations": len(eval_files),
                "expected_cells": len(C.all_cells()),
                "complete": not missing,
                "missing_cells": missing,
                "config_sha256_16": C.config_fingerprint(),
                "bootstrap_unit": "split_block",
                "bootstrap_resamples": C.BOOTSTRAP_RESAMPLES}
    (args.out_dir / "summary_manifest.json").write_text(
        json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
