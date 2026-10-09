#!/usr/bin/env python3
"""
Fabricate evaluation records with the real schema, to exercise the analysis.

EVERY NUMBER THESE FILES CONTAIN IS INVENTED. They exist so the figure and
table code is finished and tested before the grid runs, which is the step that
usually blocks people afterwards. They must never be mistaken for results, so:

  * they are written to grid/fixtures/, never to results/ or grid_results/
  * every record carries "FABRICATED": true and a "warning" string
  * the directory gets a FABRICATED_DATA_DO_NOT_CITE.txt marker
  * analysis.py refuses nothing, but the manifest it writes records the flag,
    so a figure built from fixtures is identifiable after the fact

Values are plausible rather than random: in-domain scores are high, transfer
scores fall away with a source-specific penalty, whole-region against
localisation pairs are skipped exactly as the real evaluator skips them, and
LoHi carries the near-degenerate positive rate that its box-presence label
actually produces.

The output is reproducible. Each cell's random stream is seeded from a sha256
of the cell's identity, so every run writes byte identical files. See
cell_seed().

Usage:  python3 grid/make_fixtures.py [--out grid/fixtures/results]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

MARKER = "FABRICATED_DATA_DO_NOT_CITE.txt"

# Plausible in-domain image-level AP per dataset, and how hard it is to
# transfer INTO that dataset. Invented, but ordered the way the audit suggests.
IN_DOMAIN = {"weld_surface_v2": 0.88, "kaggle_v2": 0.74, "lohi_weld": 0.97,
             "visual": 0.83, "twoclass": 0.66}
TRANSFER_PENALTY = {"weld_surface_v2": 0.55, "kaggle_v2": 0.45,
                    "lohi_weld": 0.30, "visual": 0.50, "twoclass": 0.60}
FAMILY_SKILL = {"yolo11n": -0.04, "yolo11s": 0.0,
                "rtdetr_l": 0.03, "faster_rcnn": 0.01}

# Test-split negatives, from the real audit. Drives AUROC availability.
NEGATIVES = {"kaggle_v2": 69, "weld_surface_v2": 62, "visual": 9,
             "twoclass": 9, "lohi_weld": 11}
POSITIVES = {"kaggle_v2": 72, "weld_surface_v2": 383, "visual": 75,
             "twoclass": 36, "lohi_weld": 442}
TEST_IMAGES = {"weld_surface_v2": 445, "kaggle_v2": 141, "lohi_weld": 453,
               "visual": 84, "twoclass": 45}
# Real counts, so the thin-class flag exercises the path it will really take.
CLASS_INSTANCES = {
    "weld_surface_v2": {"Burn Through-BT": 42, "Contamination-CN": 88,
                        "Lack of Fusion-LOF": 71, "Normal Weld-NW": 62,
                        "Spatter-SR": 80, "Surface Crack-SC": 38,
                        "Undercut-UC": 70},
    "kaggle_v2": {"Bad Weld": 110, "Good Weld": 182, "Defect": 159},
    "lohi_weld": {"pore": 589, "deposit": 440, "discontinuity": 1082,
                  "stain": 1246},
    "visual": {"Defect": 259},
    "twoclass": {"Geometric defect": 43, "Non-fusion defect": 30},
}


def image_level(rng, target, score):
    n_pos, n_neg = POSITIVES[target], NEGATIVES[target]
    reportable = n_neg >= C.MIN_NEGATIVES_FOR_FPR
    out = {
        "n_pos": n_pos, "n_neg": n_neg,
        "positive_rate": n_pos / (n_pos + n_neg),
        "ap": round(float(score), 4),
        "auroc": round(float(np.clip(0.5 + (score - 0.5) * 0.9, 0, 1)), 4)
                 if reportable else None,
        "auroc_reportable": reportable,
        "calibration": {
            "ece": round(float(abs(rng.normal(0.08, 0.03))), 4),
            "brier": round(float(abs(rng.normal(0.15, 0.04))), 4),
            "bins": [],
        },
    }
    if not reportable:
        out["auroc_suppressed_reason"] = (
            f"{n_neg} negative images, under the "
            f"{C.MIN_NEGATIVES_FOR_FPR} needed for a calibrated false alarm rate")
    # Reliability bins, under the keys metrics.expected_calibration_error
    # writes. Until 24 September 2026 these were `lower`, `upper`,
    # `confidence` and `accuracy`, copied from analysis.py's reader rather
    # than from the producer, so a reader asking for keys the evaluator never
    # writes passed every fixture test. test_analysis.py now checks these
    # keys against the producer itself. Cross-domain is made deliberately
    # overconfident.
    edges = np.linspace(0, 1, C.ECE_BINS + 1) if hasattr(C, "ECE_BINS") \
        else np.linspace(0, 1, 11)
    for lo, hi in zip(edges[:-1], edges[1:]):
        conf = float((lo + hi) / 2)
        acc = float(np.clip(conf * (0.6 + 0.4 * score) + rng.normal(0, 0.02), 0, 1))
        count = int(rng.integers(5, 60))
        out["calibration"]["bins"].append(
            {"lo": round(float(lo), 3), "hi": round(float(hi), 3),
             "count": count, "mean_confidence": round(conf, 3),
             "observed_positive_rate": round(acc, 3)})
    return out


def box_level(rng, source, target, score):
    same = (C.DATASETS[source]["protocol"] == C.DATASETS[target]["protocol"]
            == "localisation")
    if not same:
        return {"skipped": True,
                "reason": (f"source protocol {C.DATASETS[source]['protocol']}, "
                           f"target protocol {C.DATASETS[target]['protocol']}. "
                           "A box level number across protocols measures "
                           "annotation mismatch, not domain shift.")}
    m50 = float(np.clip(score * 0.75 + rng.normal(0, 0.02), 0, 1))
    out = {"map50": round(m50, 4),
           "map50_95": round(float(m50 * 0.62), 4),
           "n_gt": int(rng.integers(70, 3400)),
           "n_pred": int(rng.integers(200, 9000))}
    for b in C.FP_PER_IMAGE_BUDGETS:
        out[f"recall_at_{b}_fp_per_image"] = round(
            float(np.clip(m50 * (0.5 + 0.4 * b) + rng.normal(0, 0.02), 0, 1)), 4)
    return out


def per_class(rng, target, score):
    """Fabricated per class entries in EVALUATE_CELL.PY's schema, not the
    reader's.

    This function previously emitted `ap50` and `bootstrap`, while
    evaluate_cell.py writes `map50` and postprocess_results.py writes
    `cluster_bootstrap_95ci`. analysis.py read the fixture names, so the
    fixture test passed while every per class AP on real data came out `n/a`.
    A fixture that agrees with the consumer instead of the producer cannot
    catch a schema drift, which is the whole reason the fixtures exist.
    """
    out = {}
    for name, n in CLASS_INSTANCES[target].items():
        if name in C.DATASETS[target]["excluded"]:
            continue
        ap = float(np.clip(score + rng.normal(0, 0.06), 0, 1))
        entry = {"map50": round(ap, 4),
                 "map50_95": round(float(ap * 0.62), 4),
                 "test_instances": n,
                 "thin": n < C.MIN_TEST_INSTANCES}
        if entry["thin"]:
            half = 0.18                       # wide, as a thin class should be
            band = {"point": round(ap, 4),
                    "lo": round(max(0.0, ap - half), 4),
                    "hi": round(min(1.0, ap + half), 4)}
            entry["cluster_bootstrap_95ci"] = {
                "unit": "split_block", "n_clusters": n,
                "resamples": C.BOOTSTRAP_RESAMPLES,
                "map50": band,
                "map50_95": {k: (round(v * 0.62, 4) if k != "point"
                                 else round(ap * 0.62, 4))
                             for k, v in band.items()}}
        out[name] = entry
    return out


def score_for(rng, source, target, family):
    base = IN_DOMAIN[target] if source == target else (
        IN_DOMAIN[target] * (1 - TRANSFER_PENALTY[target]))
    return float(np.clip(base + FAMILY_SKILL[family] + rng.normal(0, 0.02), 0, 1))


def cell_seed(cell: dict) -> int:
    """The seed of one cell's fabricated stream, stable across processes.

    NEVER USE THE BUILT-IN hash() HERE. Python randomises string hashing per
    process unless PYTHONHASHSEED is fixed. Until 24 September 2026 this file
    seeded from hash() of the cell's identity, so every run wrote different
    values and no run could reproduce the committed fixtures. The project log 5m
    records the same defect in the transfer bootstrap. A sha256 of the
    identity is stable across processes, machines and Python versions, which
    is how transfer_bootstrap.target_seed() derives its offsets.
    """
    key = f"{cell['source']}:{cell['family']}:{cell['seed']}:{cell['kind']}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")


def build(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / MARKER).write_text(
        "Every number in this directory is FABRICATED.\n\n"
        "These files exist only to test grid/analysis.py before the real grid\n"
        "runs. They are not measurements. Do not cite them, do not copy them\n"
        "into results/ or grid_results/, and do not let a figure built from\n"
        "them reach the manuscript.\n\n"
        "Regenerate with: python3 grid/make_fixtures.py\n")

    written = 0
    for cell in C.all_cells():
        rng = np.random.default_rng(cell_seed(cell))
        name = C.cell_name(cell)
        record = {
            "FABRICATED": True,
            "warning": "fabricated fixture, not a measurement, do not cite",
            "cell": cell,
            "name": name,
            "config_sha256_16": C.config_fingerprint(),
            "targets": {},
        }
        for key, target, test_split in C.eval_plan(cell):
            score = score_for(rng, cell["source"], target, cell["family"])
            if test_split == "dirty":
                # A downloaded split shares content across its boundary, so the
                # published-style number sits above the audited one.
                score = float(np.clip(score + 0.09, 0, 1))
            entry = {
                "target": target,
                "test_split": "downloaded" if test_split == "dirty" else "audited",
                "n_test_images": TEST_IMAGES[target],
                "image_level": image_level(rng, target, score),
                "box_level": box_level(rng, cell["source"], target, score),
            }
            if target == cell["source"]:
                entry["per_class_in_domain"] = per_class(rng, target, score)
            record["targets"][key] = entry
        (out_dir / f"eval__{name}.json").write_text(json.dumps(record, indent=2))
        written += 1
    return written


def kshot_fixture(path: Path) -> int:
    """Arm B has not run. This is the schema the recovery code consumes."""
    rng = np.random.default_rng(7)
    records = []
    for source in C.TRAIN_SOURCES:
        for target in C.ALL_TARGETS:
            if target == source:
                continue
            for family in C.FAMILIES:
                p0 = score_for(rng, source, target, family)
                pjj = IN_DOMAIN[target]
                by_k = {}
                for k in (10, 25, 50, 100, 250):
                    frac = 1 - np.exp(-k / 90)          # saturating, plausible
                    by_k[str(k)] = [
                        float(np.clip(p0 + (pjj - p0) * frac + rng.normal(0, 0.03),
                                      0, 1)) for _ in range(3)]
                records.append({"FABRICATED": True, "source": source,
                                "target": target, "family": family,
                                "P_ij_zero_shot": p0, "P_jj_reference": pjj,
                                "by_k": by_k})
    path.write_text(json.dumps(records, indent=2))
    return len(records)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "fixtures" / "results")
    args = ap.parse_args()
    n = build(args.out)
    k = kshot_fixture(args.out.parent / "kshot_fixture.json")
    print(f"wrote {n} fabricated evaluation records to {args.out}")
    print(f"wrote {k} fabricated arm B records")
    print(f"marker: {args.out / MARKER}")


if __name__ == "__main__":
    main()
