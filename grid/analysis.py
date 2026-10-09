#!/usr/bin/env python3
"""
Every figure and table the manuscript needs, from the evaluation JSONs.

One command:

    python3 grid/analysis.py                       # real results
    python3 grid/analysis.py --results-dir grid/fixtures/results --out-dir /tmp/x

It reads `eval__*.json`, so it works on whatever subset of the grid has landed
and says plainly what is missing. It needs no GPU. It does need the raw
prediction sidecar beside every main and reference record. The per-class
table is corrected from them, see per_class_nms.py, and main() stops rather
than write it uncorrected. Nothing here trains or evaluates a model.

OUTPUTS

  figures/transfer_image_<config>.png    transfer matrix heatmap, per config
  figures/transfer_box_<config>.png      box level, localisation pairs only
  figures/transfer_image_mean.png        averaged over configurations
  figures/generalization_gap.png         signed, diverging ramp
  figures/recovery_kshot.png             arm B curves with variability bands
  tables/reliability_bins.csv            calibration bins, the evaluator's keys
  tables/*.csv and tables/*.md           every figure has a companion table

  No main-grid reliability figure is drawn. See the comment where
  reliability_rows is called. Figure 4.4 comes from the internal control.

WHY EVERY FIGURE SHIPS A TABLE. The validated palette's minimum contrast on a
white page is 2.25, under the 3.0 wanted for a thin mark. Identity is
therefore carried by marker shape and direct labels as well as colour, and a
reader who cannot separate two series reads the number instead.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C  # noqa: E402
import figstyle as F  # noqa: E402
import per_class_nms as PCN  # noqa: E402
import stats_tests as S  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

IMAGE_METRIC = "image_ap"
BOX_METRIC = "box_map50"
LIFT_METRIC = "image_ap_above_base"

# WHY EVERY IMAGE-LEVEL AP NEEDS ITS BASE RATE PRINTED BESIDE IT.
#
# Image-level AP is positive-class average precision. A predictor that calls
# every image positive, or that ranks images at random, scores the positive
# rate. So AP is bounded BELOW by the base rate, not by zero, and the floor is
# different in every column of the transfer matrix.
#
# lohi_weld's test split is 442 positive against 11 negative, a base rate of
# 0.976. A foreign model scoring 0.986 there has produced almost nothing: the
# lift over a constant-positive predictor is 0.010. Read without the base rate
# it looks like near-perfect transfer. kaggle_v2 at 72 positive against 69
# negative has a base rate of 0.511, so the same 0.986 would be a real result.
#
# The two numbers are therefore never reported apart. `image_ap_above_base` is
# AP minus the base rate, which is the honest quantity to compare across
# columns, and it is negative when a model is worse than trivial.
BALANCED_BAND = (0.35, 0.65)


def is_balanced(positive_rate) -> bool:
    """True when a column's AP is not dominated by its own base rate."""
    return (positive_rate is not None and finite(positive_rate)
            and BALANCED_BAND[0] <= positive_rate <= BALANCED_BAND[1])


BASE_RATE_NOTE = (
    "Base rate is the share of positive test images and is printed under each "
    "column. AP cannot fall below it for a sensible ranker, so columns are not "
    "comparable on AP alone. Only a column marked (balanced) is near 0.5.")


def base_rate_ticks(targets, rates) -> list:
    """Column labels carrying the base rate, so the matrix cannot be misread."""
    out = []
    for t in targets:
        pr = rates.get(t)
        if pr is None:
            out.append(t)
        elif is_balanced(pr):
            out.append(f"{t}\nbase {pr:.2f} (balanced)")
        else:
            out.append(f"{t}\nbase {pr:.2f}")
    return out


def base_rates(rows) -> dict:
    """target -> positive rate on its audited test split.

    Taken from the evaluation records rather than hardcoded, so it cannot
    drift away from what was actually measured.
    """
    out = {}
    for r in rows:
        if r["test_split"] == "audited" and finite(r.get("positive_rate")):
            out.setdefault(r["target"], r["positive_rate"])
    return out


def finite(v) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(float(v))


def mean(values):
    vals = [float(v) for v in values if finite(v)]
    return float(np.mean(vals)) if vals else None


def std(values):
    vals = [float(v) for v in values if finite(v)]
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------

def load_rows(results_dir: Path):
    """Flatten every evaluation record into one row per (cell, result key)."""
    rows, records, missing = [], [], []
    expected = {C.cell_name(c): c for c in C.all_cells()}
    seen = set()
    for path in sorted(results_dir.glob("eval__*.json")):
        record = json.loads(path.read_text())
        name = record.get("name")
        seen.add(name)
        records.append(record)
        cell = record["cell"]
        for key, entry in record["targets"].items():
            target, _ = C.split_result_key(key)
            image = entry.get("image_level", {})
            box = entry.get("box_level", {})
            # The base rate travels with the AP from the moment it is loaded,
            # so no downstream table or figure can present one without the
            # other. AP minus base rate is what a constant-positive predictor
            # would not have got for free.
            n_pos, n_neg = image.get("n_pos"), image.get("n_neg")
            pos_rate = image.get("positive_rate")
            if pos_rate is None and finite(n_pos) and finite(n_neg) \
                    and (n_pos + n_neg):
                pos_rate = n_pos / (n_pos + n_neg)
            ap = image.get("ap")
            lift = (ap - pos_rate) if finite(ap) and finite(pos_rate) else None
            rows.append({
                "name": name,
                "kind": cell.get("kind", "main"),
                "source": cell["source"],
                "family": cell["family"],
                "seed": cell["seed"],
                "target": target,
                "result_key": key,
                "test_split": entry.get("test_split", "audited"),
                "n_test_images": entry.get("n_test_images"),
                "image_ap": ap,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "positive_rate": pos_rate,
                "image_ap_above_base": lift,
                "target_balanced": is_balanced(pos_rate),
                "image_auroc": image.get("auroc"),
                "image_auroc_reportable": image.get("auroc_reportable"),
                "box_map50": None if box.get("skipped") else box.get("map50"),
                "box_map50_95": None if box.get("skipped") else box.get("map50_95"),
                "box_skipped_reason": box.get("reason"),
            })
    missing = sorted(set(expected) - seen)
    return rows, records, missing


# --------------------------------------------------------------------------
# Derived quantities
# --------------------------------------------------------------------------

def matrix_for(rows, family, metric, kind="main"):
    """(source, target) -> mean over seeds, for one configuration."""
    acc = defaultdict(list)
    for r in rows:
        if r["kind"] != kind or r["family"] != family:
            continue
        if r["test_split"] != "audited":
            continue
        if finite(r.get(metric)):
            acc[(r["source"], r["target"])].append(r[metric])
    return {k: mean(v) for k, v in acc.items()}


def reference_scores(rows, metric):
    """P_jj for every (target, configuration).

    The diagonal of the main grid supplies it for a training source. A
    reference cell supplies it for a test-only target. Both are same
    configuration and same seed set, which is the point of running references
    across all four configurations rather than one.
    """
    acc = defaultdict(list)
    for r in rows:
        if not finite(r.get(metric)) or r["test_split"] != "audited":
            continue
        if r["kind"] == "main" and r["source"] == r["target"]:
            acc[(r["target"], r["family"])].append(r[metric])
        elif r["kind"] == "reference" and r["source"] == r["target"]:
            acc[(r["target"], r["family"])].append(r[metric])
    return {k: mean(v) for k, v in acc.items()}


def transfer_ratio_rows(rows, metric):
    """T_ij = P_ij / P_jj, same configuration, same seed set.

    The ratio is the PRE-REGISTERED definition and is not changed here. But on
    an image-level metric it inherits the base rate in both numerator and
    denominator, and a near-degenerate column makes it flatter than the truth.
    On lohi_weld, base rate 0.976, a foreign model on 0.986 against an
    in-domain 0.990 gives T = 0.996, which reads as almost perfect transfer.
    The same pair in lift terms is 0.010 against 0.014, a ratio of 0.71.
    So the base-rate-corrected ratio is reported ALONGSIDE, never instead.
    """
    refs = reference_scores(rows, metric)
    rates = base_rates(rows)
    lift_refs = reference_scores(rows, LIFT_METRIC) \
        if metric == IMAGE_METRIC else {}
    lift_all = matrix_all(rows, LIFT_METRIC) if metric == IMAGE_METRIC else {}
    out = []
    for (source, target), value in sorted(matrix_all(rows, metric).items()):
        for family, score in value.items():
            denom = refs.get((target, family))
            weak = (target == "twoclass")
            pr = rates.get(target)
            row = {
                "metric": metric, "source": source, "target": target,
                "family": family,
                "P_ij": score, "P_jj": denom,
                "transfer_ratio": (score / denom
                                   if finite(score) and finite(denom) and denom > 0
                                   else None),
                "target_base_rate": pr,
                "target_balanced": is_balanced(pr),
                "denominator_is_weak_by_construction": weak,
                "note": ("twoclass reference trains on 213 images in 23 scene "
                         "blocks from 7 joint configurations. A small "
                         "denominator inflates the ratio, so never quote this "
                         "ratio without P_jj beside it." if weak else ""),
            }
            if metric == IMAGE_METRIC:
                ln = lift_all.get((source, target), {}).get(family)
                ld = lift_refs.get((target, family))
                row["P_ij_above_base"] = ln
                row["P_jj_above_base"] = ld
                row["transfer_ratio_above_base"] = (
                    ln / ld if finite(ln) and finite(ld) and ld > 0 else None)
                row["base_rate_warning"] = (
                    "" if is_balanced(pr) else
                    f"target base rate {pr:.3f}. The raw ratio is compressed "
                    f"towards 1 because both terms sit on that floor. Prefer "
                    f"transfer_ratio_above_base." if finite(pr) else "")
            out.append(row)
    return out


def matrix_all(rows, metric, kind="main"):
    """(source, target) -> {family: mean over seeds}."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["kind"] != kind or r["test_split"] != "audited":
            continue
        if finite(r.get(metric)):
            acc[(r["source"], r["target"])][r["family"]].append(r[metric])
    return {k: {f: mean(v) for f, v in fam.items()} for k, fam in acc.items()}


def generalization_gap_rows(rows, metric):
    """delta_i = P_ii - mean over j != i of P_ij."""
    cells = matrix_all(rows, metric)
    out = []
    families = sorted({f for v in cells.values() for f in v})
    for source in C.TRAIN_SOURCES:
        for family in families:
            in_domain = cells.get((source, source), {}).get(family)
            cross = [v.get(family) for (s, t), v in cells.items()
                     if s == source and t != source]
            cross_mean = mean(cross)
            out.append({
                "metric": metric, "source": source, "family": family,
                "in_domain": in_domain, "cross_domain_mean": cross_mean,
                "generalization_gap": (in_domain - cross_mean
                                       if finite(in_domain) and finite(cross_mean)
                                       else None),
            })
    return out


def recovery_rows(kshot_records):
    """R_ij(k) = (P_ij(k) - P_ij(0)) / (P_jj - P_ij(0)).

    Arm B has not run yet. This consumes the schema arm B will emit, so the
    curve code is ready rather than written under time pressure afterwards.
    """
    out = []
    for rec in kshot_records:
        p0, pjj = rec.get("P_ij_zero_shot"), rec.get("P_jj_reference")
        for k, draws in sorted(rec.get("by_k", {}).items(), key=lambda x: int(x[0])):
            vals = [v for v in draws if finite(v)]
            if not vals or not finite(p0) or not finite(pjj):
                continue
            denom = pjj - p0
            recov = [((v - p0) / denom) if abs(denom) > 1e-12 else None for v in vals]
            recov = [r for r in recov if r is not None]
            out.append({
                "source": rec["source"], "target": rec["target"],
                "family": rec["family"], "k": int(k),
                "n_draws": len(vals),
                "P_ij_k_mean": mean(vals), "P_ij_k_std": std(vals),
                "recovery_mean": mean(recov), "recovery_std": std(recov),
                "P_ij_zero_shot": p0, "P_jj_reference": pjj,
            })
    return out


def dirty_three_number_rows(rows):
    """A, B, C and the two differences. See config.eval_plan for the meaning."""
    idx = {(r["kind"], r["source"], r["family"], r["seed"], r["result_key"]): r
           for r in rows}
    out = []
    for a in rows:
        if a["kind"] != "dirty" or a["test_split"] != "downloaded":
            continue
        src, fam, seed = a["source"], a["family"], a["seed"]
        b = idx.get(("dirty", src, fam, seed, src + C.CLEAN_TEST_SUFFIX))
        c = idx.get(("main", src, fam, seed, src))
        row = {"source": src, "family": fam, "seed": seed,
               "A_downloaded_train_downloaded_test": a.get(IMAGE_METRIC),
               "B_downloaded_train_audited_test": b.get(IMAGE_METRIC) if b else None,
               "C_audited_train_audited_test": c.get(IMAGE_METRIC) if c else None}
        row["shift_test_set_only"] = (
            row["A_downloaded_train_downloaded_test"] - row["B_downloaded_train_audited_test"]
            if finite(row["A_downloaded_train_downloaded_test"])
            and finite(row["B_downloaded_train_audited_test"]) else None)
        row["shift_full_pipeline"] = (
            row["A_downloaded_train_downloaded_test"] - row["C_audited_train_audited_test"]
            if finite(row["A_downloaded_train_downloaded_test"])
            and finite(row["C_audited_train_audited_test"]) else None)
        row["interpretation"] = ("estimate shift under a change of protocol, "
                                 "not a causal effect of leakage")
        out.append(row)
    return out


# The evaluator's own key names for a per class entry. analysis.py MUST read
# these, and the fixtures MUST emit these. On 9 September 2026 they disagreed:
# evaluate_cell.py wrote `map50`, analysis.py read `ap50`, and the fixtures had
# been written to match the reader instead of the producer. So every per class
# AP in the real table was silently `n/a`, and the fixture test passed anyway.
# test_analysis.py now asserts the fixtures use these names.
PER_CLASS_AP_KEY = "map50"
PER_CLASS_AP_RANGE_KEY = "map50_95"
PER_CLASS_BOOTSTRAP_KEY = "cluster_bootstrap_95ci"


def attach_thin_class_bootstraps(records, results_dir: Path) -> int:
    """Compute the split-block bootstrap for every thin in-domain class.

    This has to happen here rather than on the cluster: 2,000 resamples per
    thin class is CPU work, and holding an H200 slice while it runs would waste
    the allocation. evaluate_cell.py therefore marks the class with
    `cluster_bootstrap_required` and leaves the raw predictions in the sidecar.
    Running it from analysis.py is what keeps this a single command.
    """
    from postprocess_results import bootstrap_thin_classes

    # 2,000 resamples per thin class over a few hundred images is minutes of
    # CPU, and it is deterministic given the seed in config.py. Cache it beside
    # the results so a re-run of the pipeline, which happens constantly while
    # figures are being tuned, does not pay for it again. The cache key carries
    # the resample count and seed, so changing either invalidates it.
    cache_path = results_dir.parent / "bootstrap_cache.json"
    key = f"{C.BOOTSTRAP_RESAMPLES}:{C.BOOTSTRAP_SEED}"
    cache = {}
    if cache_path.is_file():
        blob = json.loads(cache_path.read_text())
        if blob.get("key") == key:
            cache = blob.get("entries", {})

    done, computed = 0, 0
    for record in records:
        wanted = any(
            info.get("cluster_bootstrap_required")
            for entry in record["targets"].values()
            for info in (entry.get("per_class_in_domain") or {}).values())
        if not wanted:
            continue
        hit = cache.get(record["name"])
        if hit is not None:
            for target, classes in hit.items():
                entry = record["targets"].get(target, {})
                for cls, band in classes.items():
                    info = (entry.get("per_class_in_domain") or {}).get(cls)
                    if info is not None:
                        info.pop("cluster_bootstrap_required", None)
                        info[PER_CLASS_BOOTSTRAP_KEY] = band
            done += 1
            continue
        raw_path = results_dir / f"raw__{record['name']}.npz"
        if not raw_path.is_file():
            continue
        with np.load(raw_path, allow_pickle=True) as raw:
            bootstrap_thin_classes(record, raw, record)
        cache[record["name"]] = {
            target: {cls: info[PER_CLASS_BOOTSTRAP_KEY]
                     for cls, info in (entry.get("per_class_in_domain") or {}).items()
                     if PER_CLASS_BOOTSTRAP_KEY in info}
            for target, entry in record["targets"].items()}
        done += 1
        computed += 1
        print(f"  bootstrap {computed}: {record['name']}")

    cache_path.write_text(json.dumps({"key": key, "entries": cache}, indent=1))
    return done


def per_class_rows(records, corrected=None):
    """In-domain per class AP, with the test instance count beside every value.

    A per class AP without its sample size is not reportable in this project.
    The count travels in the same row so a table cannot be built without it.
    """
    # THE VALUES COME FROM `per_class_nms`, NOT FROM THE RECORD. The record's
    # per-class AP was computed without the class-agnostic NMS that the
    # box-level path applies, which charged RT-DETR for duplicate queries the
    # other three families' backends had already suppressed. See the module
    # docstring. `evaluate_cell.py` is sha256 pinned, so the correction lives
    # outside it, as `transfer_bootstrap.py` does for `metrics.py`.
    # PASSED IN, NOT LOADED HERE. Loading the real cache inside this
    # function contaminated the fixture run: fixture cell names collide with
    # real ones, so fabricated rows silently picked up real corrected values
    # and real thin flags. A fixture set that can reach real data is not a
    # fixture set. The caller supplies it, and the tests supply nothing.
    if corrected is None:
        corrected = {}

    def cid_in_space(target, cname):
        classes = C.DATASETS[target]["classes"]
        return (cname in classes
                and classes.index(cname) in C.defect_ids(target))

    def cname_excluded(target, cname):
        return cname in C.DATASETS[target]["excluded"]

    out = []
    for record in records:
        cell = record["cell"]
        for key, entry in record["targets"].items():
            per_class = entry.get("per_class_in_domain")
            if not per_class:
                continue
            target, _ = C.split_result_key(key)
            # Iterate the dataset's full class list rather than the record's
            # keys, so classes excluded from the defect space appear too. A
            # reader cannot see the harmonised taxonomy without them.
            names = C.DATASETS[target]["classes"]
            for name in names:
                fixed = corrected.get(f"{record['name']}|{target}|{name}")
                if fixed is None:
                    # No correction for this cell. The fabricated fixtures
                    # ship no raw archive, and per_class_nms.py skips
                    # downloaded-data cells. Fall back to the record's own
                    # value, and say so in the row rather than silently
                    # mixing two conventions. For any other cell main()
                    # refuses the result, see uncorrected_rows(). A class the
                    # record never wrote is genuinely absent and is skipped.
                    if name not in per_class:
                        continue
                    info = dict(per_class[name])
                    in_space = cid_in_space(target, name)
                    nms_ok = False
                else:
                    info = dict(per_class.get(name, {}))
                    info.update({k: fixed[k] for k in
                                 ("map50", "map50_95", "test_instances",
                                  "thin")})
                    in_space = fixed["in_defect_space"]
                    nms_ok = True
                boot = (info.get(PER_CLASS_BOOTSTRAP_KEY) or {}).get("map50") or {}
                out.append({
                    "source": cell["source"], "family": cell["family"],
                    "seed": cell["seed"], "target": target,
                    # The cell kind is load bearing for the markdown table, and
                    # its absence is what let downloaded-data cells be averaged
                    # into the in-domain per class AP. Keep it in the CSV too,
                    # so the row says which pipeline produced it.
                    "kind": cell.get("kind", "main"),
                    "test_split": entry.get("test_split", "audited"),
                    "class": name,
                    # C1 is the harmonised taxonomy and this table is where a
                    # reader can see it. Without these two columns the table
                    # lists Normal Weld among defect classes unflagged, and
                    # shows visual with one class and no sign of the other
                    # two.
                    "in_defect_space": in_space,
                    # Two different exclusions, and the table must not merge
                    # them. `excluded` classes are dropped from the ground
                    # truth itself, so they have no instances and cannot be
                    # scored. A verdict class is kept and scored, and simply
                    # is not a defect.
                    "excluded_from_gt": (fixed["excluded"] if fixed
                                         else cname_excluded(target, name)),
                    "nms_corrected": nms_ok,
                    "ap50": info.get(PER_CLASS_AP_KEY),
                    "ap50_95": info.get(PER_CLASS_AP_RANGE_KEY),
                    "test_instances": info.get("test_instances"),
                    "thin": info.get("thin"),
                    "ci_lo": boot.get("lo"),
                    "ci_hi": boot.get("hi"),
                })
    return out


def uncorrected_rows(rows, records):
    """Per-class rows that fell back where no fallback is allowed.

    A ROW WITHOUT THE NMS CORRECTION IS THE the project log 5p DEFECT, NOT A
    FALLBACK. The evaluator's own per-class AP skips NMS, so RT-DETR-l is
    charged for its duplicate queries, in 17 of 68 class groups. Until
    25 September 2026 the only trace was `nms_corrected` False in the CSV.
    build_results_section.py compared a per-class value with its box diagonal,
    but only on kaggle_v2 and visual, the two sources whose defect space holds
    one class. On the other three the defect was invisible. Now main() stops
    before writing such a row, and the builder checks the released CSV too.

    Two kinds of row fall back by design, and only those two are allowed.
    A downloaded-data cell is skipped by per_class_nms.py and kept out of the
    markdown table, the project log 5o. A fabricated fixture set ships no raw
    archive, so it has nothing to correct. The fixture exemption covers a
    results set only when every record in it is fabricated. A set that mixes
    fabricated and real records is itself the contamination the project log 5l
    forbids, so its fabricated rows are refused along with the rest.
    """
    if all(r.get("FABRICATED") is True for r in records):
        return []
    return [r for r in rows if not r["nms_corrected"] and r["kind"] != "dirty"]


# The calibration bin keys exactly as metrics.expected_calibration_error writes
# them. reliability_rows reads them and writes them out under the same names,
# so reliability_bins.csv and the arms' reliability_bins_arms.csv share one
# schema.
#
# Until 24 September 2026 the reader asked for `lower`, `upper`, `confidence`
# and `accuracy`, which the evaluator never writes. Every value came out empty,
# four columns of reliability_bins.csv were blank in all 3,330 rows, and
# reliability.png was drawn with no data. The fixture tests passed, because
# make_fixtures.py had been written to match this reader rather than the
# producer, which is the project log 5l's lesson again. test_analysis.py now takes
# the keys from the producer itself and checks the reader and the fixtures
# against them.
CALIBRATION_BIN_KEYS = ("lo", "hi", "count", "mean_confidence",
                        "observed_positive_rate")


def reliability_rows(records):
    """Calibration bins, one row per bin, split into in domain and cross domain.

    Columns: source, family, seed, kind, test_split, target and regime, then the
    evaluator's own bin keys unchanged: lo, hi, count, mean_confidence and
    observed_positive_rate. An empty bin keeps its count of zero and carries
    no confidence and no observed rate, as the evaluator writes it.

    `kind` and `test_split` are there because a downloaded-data cell is
    evaluated in domain twice, on the downloaded test split and on the audited
    one. Without them the two rows would be indistinguishable, and averaging
    them would mix the memorisation figure of the project log 5j into an in-domain
    mean, which is exactly the 5o defect.

    A bin missing any of the keys raises. A lenient read is what let the old
    key names go unnoticed.
    """
    out = []
    for record in records:
        cell = record["cell"]
        for key, entry in record["targets"].items():
            target, _ = C.split_result_key(key)
            bins = (entry.get("image_level", {}).get("calibration", {}) or {}).get("bins")
            if not bins:
                continue
            regime = "in domain" if target == cell["source"] else "cross domain"
            for b in bins:
                missing = [k for k in CALIBRATION_BIN_KEYS if k not in b]
                if missing:
                    raise KeyError(
                        f"{record.get('name', cell)} {key}: calibration bin "
                        f"lacks {missing}. metrics.expected_calibration_error "
                        f"writes {list(CALIBRATION_BIN_KEYS)}.")
                row = {"source": cell["source"], "family": cell["family"],
                       "seed": cell["seed"], "kind": cell.get("kind", "main"),
                       "test_split": entry.get("test_split"),
                       "target": target, "regime": regime}
                row.update({k: b[k] for k in CALIBRATION_BIN_KEYS})
                out.append(row)
    return out


# --------------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------------

def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r}) if rows else []
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_markdown_matrix(path: Path, cells, sources, targets, title, note="",
                          rates=None, lift=None):
    """Matrix table. For image-level metrics `rates` puts the base rate in the
    header, so a column can never be read without the floor it sits on."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    if note:
        lines += [note, ""]
    if rates:
        lines += [
            "**Read every column against its base rate.** Image-level AP is "
            "positive-class average precision, so a constant-positive "
            "predictor already scores the positive rate. The floor differs by "
            "column, and only a column marked balanced is near 0.5.", ""]
    lines.append("| train \\ test | " + " | ".join(targets) + " |")
    lines.append("|---" * (len(targets) + 1) + "|")
    if rates:
        cells_hdr = []
        for t in targets:
            pr = rates.get(t)
            if pr is None:
                cells_hdr.append("n/a")
            else:
                tag = " **balanced**" if is_balanced(pr) else ""
                cells_hdr.append(f"{pr:.3f}{tag}")
        lines.append("| *base rate (positive class)* | "
                     + " | ".join(cells_hdr) + " |")
    for s in sources:
        vals = []
        for t in targets:
            v = cells.get((s, t))
            vals.append("n/a" if v is None else f"{v:.3f}")
        lines.append(f"| **{s}** | " + " | ".join(vals) + " |")
    if lift:
        lines += ["", "## Same matrix, AP minus base rate", "",
                  "The lift over a constant-positive predictor. This is the "
                  "quantity that is comparable across columns. Negative means "
                  "worse than trivial.", "",
                  "| train \\ test | " + " | ".join(targets) + " |",
                  "|---" * (len(targets) + 1) + "|"]
        for s in sources:
            vals = []
            for t in targets:
                v = lift.get((s, t))
                vals.append("n/a" if v is None else f"{v:+.3f}")
            lines.append(f"| **{s}** | " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n")


def heatmap(path: Path, cells, sources, targets, title, cbar_label,
            diverging=False, xlabel="evaluated on", ylabel="trained on",
            xticklabels=None, note=None):
    if not sources or not targets:
        return
    data = np.full((len(sources), len(targets)), np.nan)
    for i, s in enumerate(sources):
        for j, t in enumerate(targets):
            v = cells.get((s, t))
            if finite(v):
                data[i, j] = v

    fig, ax = plt.subplots(figsize=(1.35 * len(targets) + 2.2,
                                    0.85 * len(sources) + 2.0))
    if diverging:
        lim = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0
        im = ax.imshow(data, cmap=F.DIVERGING, vmin=-lim, vmax=lim)
    else:
        im = ax.imshow(data, cmap=F.SEQUENTIAL, vmin=0, vmax=1)
    ax.set_xticks(range(len(targets)), xticklabels or targets,
                  rotation=30, ha="right")
    ax.set_yticks(range(len(sources)), sources)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    ax.grid(False)

    # Direct labels in every cell. The figure is readable without the ramp.
    mid = 0 if diverging else 0.55
    for i in range(len(sources)):
        for j in range(len(targets)):
            v = data[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "n/a", ha="center", va="center",
                        color=F.INK_MUTED, fontsize=8)
                continue
            dark = (abs(v) > (np.nanmax(np.abs(data)) * 0.6)) if diverging else v > mid
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=F.SURFACE if dark else F.INK, fontsize=8)
    cb = fig.colorbar(im, ax=ax, shrink=0.82)
    cb.set_label(cbar_label)
    cb.outline.set_visible(False)
    # A scope note belongs ON the figure, not only in the methods. A reader
    # who meets the box-level matrix as a 2 by 4 grid must be able to see why
    # weld_surface_v2 is absent without hunting for the caption.
    if note:
        fig.text(0.5, -0.02, note, ha="center", va="top",
                 fontsize=7.5, color=F.INK_MUTED, wrap=True)
    F.save(fig, path)
    plt.close(fig)


def recovery_figure(path: Path, rows):
    """k-shot recovery, one line per configuration, with variability bands.

    Rows arrive one per (source, target, configuration, k). They MUST be
    aggregated over source-target pairs before plotting: without that,
    matplotlib draws every pair at the same k and the line becomes a column of
    markers rather than a curve.

    The band is the spread across source-target pairs at that k, which is the
    quantity a reader cares about. It is dispersion, not a confidence
    interval, and the caption says so.
    """
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    agg = defaultdict(list)
    for r in rows:
        if finite(r.get("recovery_mean")):
            agg[(r["family"], int(r["k"]))].append(float(r["recovery_mean"]))

    plotted, all_k = 0, sorted({k for _, k in agg})
    for family in F.CONFIG_ORDER:
        ks = [k for k in all_k if (family, k) in agg]
        if not ks:
            continue
        m = np.array([float(np.mean(agg[(family, k)])) for k in ks])
        sd = np.array([float(np.std(agg[(family, k)], ddof=1))
                       if len(agg[(family, k)]) > 1 else 0.0 for k in ks])
        colour = F.CONFIG_COLOR[family]
        ax.fill_between(ks, m - sd, m + sd, color=colour, alpha=0.13, linewidth=0)
        ax.plot(ks, m, color=colour, marker=F.CONFIG_MARKER[family],
                label=F.label_for(family))
        plotted += 1

    ax.axhline(1.0, color=F.INK_MUTED, linewidth=0.8, linestyle="--",
               gid=F.REFERENCE)
    ax.text(all_k[0] if all_k else 0, 1.005, " in-domain reference", fontsize=8,
            color=F.INK_MUTED, va="bottom")
    ax.set_xscale("log")
    # Label the k values actually used, not decade ticks.
    if all_k:
        ax.set_xticks(all_k, [str(k) for k in all_k])
        ax.minorticks_off()
    ax.set_xlabel("target images annotated, k")
    ax.set_ylabel("recovery  R(k)")
    ax.set_title("Recovery towards the in-domain reference, arm B\n"
                 "band is spread across source-target pairs, not a confidence interval",
                 fontsize=9)
    # Legend only. End-of-line labels collided at k=250 where the curves
    # converge, which is worse than a legend round-trip. Identity is still
    # double encoded by colour and marker shape.
    if plotted >= 2:
        ax.legend(loc="lower right", ncol=2)
    F.save(fig, path)
    plt.close(fig)


def per_class_markdown(path: Path, rows):
    """Per class AP. Every value carries its test instance count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    by = defaultdict(list)
    dropped = 0
    for r in rows:
        # DOWNLOADED-DATA CELLS MUST NOT ENTER THIS TABLE. They contribute two
        # rows each, one graded on the downloaded test split and one on the
        # audited split, and the second is the memorisation figure the project log 5j
        # says must never be compared against the main grid number. Averaging
        # all three pipelines together produced kaggle_v2 YOLO11s `Defect` at
        # 0.468 +- 0.304 where the main grid alone gives 0.112 +- 0.013.
        # Reference cells DO belong here: for visual and twoclass they are the
        # only in-domain cells there are.
        if r.get("kind") == "dirty":
            dropped += 1
            continue
        by[(r["source"], r["family"])].append(r)
    lines = ["# In-domain per-class AP50", "",
             "**Every class of every source is listed, and the `in defect "
             "space` column says which of them the harmonised taxonomy "
             "retains.** A class marked no is scene furniture or a "
             "whole-region quality verdict, and it contributes to no "
             "cross-dataset figure anywhere in this study. Printing only the "
             "retained classes would hide the taxonomy this table exists to "
             "demonstrate, and printing all of them unflagged would show "
             "Normal Weld sitting among defect classes.", "",
             f"**Values apply non-maximum suppression at IoU {C.NMS_IOU}, "
             "WITHIN each class, recomputed by `per_class_nms.py`.** The "
             "evaluator wrote these with no suppression at all, which charged "
             "RT-DETR for duplicate queries that the other three backends "
             "remove before a prediction is stored. On a source whose defect "
             "space holds one class this value and the class-agnostic box "
             "mAP50 are the same quantity, and they now agree exactly.", "",
             "**Within-class suppression is a decision, not the function that "
             "was to hand.** The alternative is to suppress across the whole "
             "defect space first, which is what the box-level matrix does "
             "because there every defect class collapses to one label. That "
             "rule is wrong for a per-class table: it makes a class's value "
             "depend on which other classes sit in the taxonomy, so removing "
             "one class would move another's AP without anything about that "
             "class changing. The difference is not negligible, reaching "
             "0.0999 on twoclass Non-fusion defect with RT-DETR-l. Run "
             "`per_class_nms.py --compare-rules` to reproduce.", "",
             "Main-grid and reference cells only. "
             f"{dropped} downloaded-data rows are excluded, because a "
             "downloaded-data cell is graded on a partition that overlaps its "
             "own training split and averaging it in would inflate the "
             "in-domain value. Those rows are kept in `per_class_ap.csv`, "
             "tagged with `kind`.", "",
             "Every value carries its test instance count. A class marked "
             "**thin** has fewer than "
             f"{C.MIN_TEST_INSTANCES} test instances.", "",
             "**The two uncertainty columns answer different questions and a "
             "thin class needs both.** Seed SD is dispersion over the three "
             "training seeds, so it measures training variance. It says "
             "nothing about a small correlated test sample: a class with 38 "
             "test boxes can look stable across seeds and still be highly "
             "uncertain. The bootstrap column is a "
             f"{C.BOOTSTRAP_RESAMPLES} draw percentile interval resampling "
             "whole split blocks, which is the unit that keeps correlated "
             "images together. It is reported for thin classes, averaged over "
             "seeds. Quote the bootstrap interval for a thin class, never the "
             "seed SD alone.", ""]
    for (source, family), items in sorted(by.items()):
        lines += [f"## {source}, {F.label_for(family)}", "",
                  "| class | in defect space | AP50 (mean over seeds) | "
                  "seed SD | split-block bootstrap | test instances | "
                  "status |",
                  "|---|---|---|---|---|---|---|"]
        agg = defaultdict(list)
        los, his = defaultdict(list), defaultdict(list)
        counts, thin, space, gone = {}, {}, {}, {}
        for r in items:
            agg[r["class"]].append(r["ap50"])
            if r.get("ci_lo") is not None:
                los[r["class"]].append(r["ci_lo"])
            if r.get("ci_hi") is not None:
                his[r["class"]].append(r["ci_hi"])
            # The count is taken from whichever row comes last, which is only
            # safe while every row in the group is graded on the same split.
            # Assert that rather than trust it: a mixed group would silently
            # print one split's count beside another split's AP.
            prev = counts.get(r["class"])
            if prev is not None and prev != r["test_instances"]:
                raise SystemExit(
                    f"per-class test instance count disagrees within "
                    f"{source}/{family}/{r['class']}: {prev} against "
                    f"{r['test_instances']}. A row graded on a different test "
                    f"split has entered the in-domain table.")
            counts[r["class"]] = r["test_instances"]
            thin[r["class"]] = r["thin"]
            space[r["class"]] = r.get("in_defect_space")
            gone[r["class"]] = r.get("excluded_from_gt")
        for cls in sorted(agg):
            m, s = mean(agg[cls]), std(agg[cls])
            value = "n/a" if m is None else f"{m:.3f}"
            sd = "n/a" if m is None else f"± {s:.3f}"
            lo, hi = mean(los.get(cls, [])), mean(his.get(cls, []))
            if lo is not None and hi is not None:
                boot = f"[{lo:.3f}, {hi:.3f}]"
            else:
                # Not thin, so no bootstrap was computed. Say so rather than
                # leaving a blank cell that reads as a missing measurement.
                boot = "not needed"
            if space.get(cls):
                mark, status = "yes", "**thin**" if thin[cls] else "ok"
            elif gone.get(cls):
                # Dropped from the ground truth itself, so it has no
                # instances and no AP. Marking it thin would read as a small
                # sample rather than as a class that is not scored at all.
                mark, status = "**no, not scored**", "excluded from ground truth"
            else:
                mark, status = "**no, verdict**", "scored, not a defect class"
            lines.append(f"| {cls} | {mark} | {value} | {sd} | {boot} | "
                         f"{counts[cls]} | {status} |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def seen_in_downloaded_train() -> dict:
    """How much of each audited test split sits inside the downloaded train split.

    Measured, not assumed. The downloaded splits were partitioned by their
    original authors and the audited splits were partitioned here, so the two
    partitions are unrelated and overlap heavily. Column B is scored with those
    images in it, which is why B must never be described as a held-out test.
    """
    from evaluate_cell import resolve_test_images

    out = {}
    for tag in C.DIRTY_SOURCES:
        train = {p.name for p in resolve_test_images(tag, "dirty", "train")}
        test = [p.name for p in resolve_test_images(tag, "main", "test")]
        n_seen = sum(1 for n in test if n in train)
        out[tag] = {"n_test": len(test), "n_seen": n_seen,
                    "n_held_out": len(test) - n_seen,
                    "share_seen": n_seen / len(test) if test else float("nan")}
    return out


def dirty_markdown(path: Path, rows, overlap: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Downloaded against audited, three numbers per source", "",
             "A. downloaded train, downloaded test: what a practitioner publishes.",
             "B. downloaded train, audited test: the same weights, regraded on "
             "the audited test split.",
             "C. audited train, audited test: the main grid diagonal.", "",
             "**Column B is not a held-out test, and must never be called one.** "
             "The downloaded and audited partitions were drawn independently, so "
             "most of the audited test split is inside the downloaded training "
             "split. The share is measured per source in the last column below. "
             "B therefore scores a model largely on images it was trained on, and "
             "is reported only to hold the weights fixed while the grading "
             "partition changes.", "",
             "A minus B changes only the test set. A minus C changes the whole "
             "pipeline. **Neither is a causal estimate of leakage.** Both are "
             "estimate shifts under a change of protocol, measured on different "
             "test sets.", "",
             "| source | A | B | C | A-B, test set only | A-C, full pipeline | "
             "audited test images already in downloaded train |",
             "|---|---|---|---|---|---|---|"]
    by = defaultdict(list)
    for r in rows:
        by[r["source"]].append(r)
    fmt = lambda v: "n/a" if not finite(v) else f"{v:.3f}"  # noqa: E731
    for source in sorted(by):
        items = by[source]
        cells = [mean([i["A_downloaded_train_downloaded_test"] for i in items]),
                 mean([i["B_downloaded_train_audited_test"] for i in items]),
                 mean([i["C_audited_train_audited_test"] for i in items]),
                 mean([i["shift_test_set_only"] for i in items]),
                 mean([i["shift_full_pipeline"] for i in items])]
        ov = overlap.get(source)
        seen = ("n/a" if ov is None else
                f"{ov['n_seen']} of {ov['n_test']} ({100 * ov['share_seen']:.0f}%)")
        lines.append(f"| {source} | " + " | ".join(fmt(v) for v in cells)
                     + f" | {seen} |")
    lines += ["",
              "Restricting B to the genuinely unseen remainder was considered and "
              "rejected. The remainders are "
              + ", ".join(f"{t} {overlap[t]['n_held_out']}"
                          for t in sorted(overlap))
              + " images, too few and too heavily one-class to carry a number."]
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------

def _draw_counts(rows: list[dict]) -> list[int]:
    """The resample counts these rows actually used, read rather than typed.

    This file's header once stated 2,000 draws for the whole table while its
    own box-level rows recorded 500. A released table that misstates its own
    method is a defect in the release, the same class as the project log 5p, so the
    count is derived from the rows it describes.
    """
    return sorted({int(r["resamples"]) for r in rows
                   if r.get("resamples") not in (None, "")})


def _draws_phrase(counts: list[int]) -> str:
    if not counts:
        return "an unrecorded number of"
    return " or ".join(f"{c:,}" for c in counts)


def transfer_interval_markdown(path: Path, boot: list[dict]) -> None:
    """T_ij with a 95 percent split-block interval, image level then box."""
    lines = [
        "# Transfer ratios with split-block confidence intervals", "",
        "`T_ij = P_ij / P_jj`, same configuration, same seed, and a 95 percent "
        "split-block percentile interval. **The number of bootstrap draws "
        "differs by level and is stated at the head of each section.** Every "
        "row also records its own count in the `resamples` column of "
        "`transfer_ratio_intervals.csv`.", "",
        "**The unit is the split block, not the image.** Images from one weld "
        "scene are correlated and adjacent video frames are nearly the same "
        "photograph, so resampling images independently would treat those as "
        "independent evidence and make every interval too narrow.", "",
        "**Numerator and denominator are resampled together.** Both are scored "
        "on the same test images of the target, so one draw selects the blocks "
        "and both models are then scored on exactly those images. The diagonal "
        "rows are the self-check: when the source is the target the two models "
        "are the same model, so T must be exactly 1.000 with zero width.", "",
        "**This interval answers a different question from the Friedman "
        "test.** Friedman asks whether the four configurations differ from "
        "each other, and it found no detectable difference. These intervals "
        "ask how large the collapse is, which is the claim the paper actually "
        "makes.", "",
    ]
    for level, title, note in (
            ("image", "Image level",
             "Two scales. The raw ratio inherits the target base rate in both "
             "terms, which compresses it towards 1 on the four unbalanced "
             "columns. The corrected ratio divides lift by lift, and the base "
             "rate is recomputed inside every draw."),
            ("box", "Box level",
             "Class-agnostic mAP50, localisation-protocol pairs only. Box AP "
             "has no prevalence floor, so no correction applies.")):
        sel = [r for r in boot if r["level"] == level]
        if not sel:
            continue
        draws = (f"**{_draws_phrase(_draw_counts(sel))} bootstrap draws per "
                 f"interval at this level.**")
        if level == "box":
            draws += (" Fewer than at image level, for cost: a box-level draw "
                      "on lohi_weld rescores up to 135,900 predictions, "
                      "because both models sit at the 300 detections per "
                      "image cap across 453 test images.")
        lines += [f"## {title}", "", note, "", draws, ""]
        if level == "image":
            lines += ["| train | test | configuration | T raw [95% CI] | "
                      "T corrected [95% CI] | lift P_ij [95% CI] |",
                      "|---|---|---|---|---|---|"]
        else:
            lines += ["| train | test | configuration | P_ij | P_jj | "
                      "T [95% CI] |", "|---|---|---|---|---|---|"]
        for r in sorted(sel, key=lambda x: (x["target"], x["source"],
                                            x["family"])):
            same = r["source"] == r["target"]
            mark = " *(diagonal)*" if same else ""
            weak = (" **weak reference**"
                    if r.get("denominator_is_weak_by_construction") and not same
                    else "")

            def band(prefix):
                p, lo, hi = (r.get(prefix), r.get(prefix + "_lo"),
                             r.get(prefix + "_hi"))
                if p is None or lo is None:
                    return "n/a"
                return f"{p:.3f} [{lo:.3f}, {hi:.3f}]"

            if level == "image":
                lines.append(
                    f"| {r['source']} | {r['target']}{mark} | {r['family']} | "
                    f"{band('transfer_ratio')} | "
                    f"{band('transfer_ratio_above_base')}{weak} | "
                    f"{band('P_ij_above_base_ci')} |")
            else:
                lines.append(
                    f"| {r['source']} | {r['target']}{mark} | {r['family']} | "
                    f"{r['P_ij']:.4f} | {r['P_jj']:.4f} | "
                    f"{band('transfer_ratio')}{weak} |")
        lines.append("")
    path.write_text("\n".join(lines))


def seed_block_scores(rows, metric):
    """(source->target, seed) -> {configuration: score}, seeds NOT averaged.

    The pre-registration commits to a sensitivity analysis that repeats the
    test with (cell, seed) as the blocking unit, tripling the block count. The
    primary test uses the seed mean as the cell value, which is correct but
    throws away training variance, so the sensitivity check is what shows
    whether the conclusion depended on that choice.
    """
    acc = defaultdict(dict)
    for r in rows:
        if r["kind"] != "main" or r["test_split"] != "audited":
            continue
        if not finite(r.get(metric)):
            continue
        acc[f"{r['source']}->{r['target']}|seed{r['seed']}"][r["family"]] = r[metric]
    return dict(acc)


def run_stats(rows, out_dir: Path) -> dict:
    """The pre-registered comparison, on image level and box level."""
    results = {}
    for metric in (IMAGE_METRIC, BOX_METRIC):
        cells = matrix_all(rows, metric)
        block_scores = {f"{s}->{t}": fam for (s, t), fam in cells.items()}
        configs = [c for c in F.CONFIG_ORDER
                   if any(c in v for v in block_scores.values())]
        results[metric] = S.compare_configurations(block_scores, configs, metric)

        # PRE-REGISTERED SENSITIVITY ANALYSIS. Seeds are replicates inside a
        # cell, not blocks, so the primary test uses the seed mean. This
        # repeats it with (cell, seed) as the block. If the two disagree,
        # BOTH are reported and the disagreement is the finding.
        per_seed = seed_block_scores(rows, metric)
        if per_seed:
            sens = S.compare_configurations(per_seed, configs, metric)
            primary_sig = results[metric]["primary"]["significant"]
            results[metric]["sensitivity_cell_by_seed"] = {
                "blocking_unit": "(evaluation cell, seed)",
                "preregistered": True,
                "note": ("Seeds treated as blocks rather than as replicates. "
                         "Reported because the pre-registration commits to it, "
                         "not because the primary result needed support."),
                "agrees_with_primary":
                    sens["primary"]["significant"] == primary_sig,
                **sens,
            }
    (out_dir / "statistical_tests.json").write_text(
        json.dumps(results, indent=2, default=float))
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path,
                    default=C.PROJECT / "grid_results" / "results")
    ap.add_argument("--out-dir", type=Path,
                    default=C.PROJECT / "grid_results" / "analysis")
    ap.add_argument("--kshot", type=Path,
                    help="arm B records, JSON list. Optional until arm B runs.")
    args = ap.parse_args()

    F.apply_style()
    rows, records, missing = load_rows(args.results_dir)
    if not rows:
        raise SystemExit(f"no eval__*.json under {args.results_dir}")

    figs, tabs = args.out_dir / "figures", args.out_dir / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)

    families = [f for f in F.CONFIG_ORDER
                if any(r["family"] == f and r["kind"] == "main" for r in rows)]

    # Measured from the records, before anything is plotted, so every
    # image-level artefact below carries the floor its numbers sit on.
    rates = base_rates(rows)
    write_csv(tabs / "target_base_rates.csv", [
        {"target": t,
         "n_pos": next((r["n_pos"] for r in rows if r["target"] == t
                        and r["test_split"] == "audited"), None),
         "n_neg": next((r["n_neg"] for r in rows if r["target"] == t
                        and r["test_split"] == "audited"), None),
         "positive_rate": pr,
         "balanced": is_balanced(pr),
         "note": ("near 0.5, so AP is informative on its own" if is_balanced(pr)
                  else "AP is dominated by the base rate, quote the lift")}
        for t, pr in sorted(rates.items())])

    # --- transfer matrices, per configuration and averaged -----------------
    for family in families:
        img = matrix_for(rows, family, IMAGE_METRIC)
        lift = matrix_for(rows, family, LIFT_METRIC)
        heatmap(figs / f"transfer_image_{family}.png", img,
                C.TRAIN_SOURCES, C.ALL_TARGETS,
                f"Image-level transfer, {F.label_for(family)}",
                "annotated-feature AP",
                xticklabels=base_rate_ticks(C.ALL_TARGETS, rates),
                note=BASE_RATE_NOTE)
        heatmap(figs / f"transfer_lift_{family}.png", lift,
                C.TRAIN_SOURCES, C.ALL_TARGETS,
                f"Image-level lift over base rate, {F.label_for(family)}",
                "AP minus base rate", diverging=True,
                xticklabels=base_rate_ticks(C.ALL_TARGETS, rates),
                note=("Zero means the model matched a constant-positive "
                      "predictor. This is the image-level quantity that is "
                      "comparable across columns."))
        write_markdown_matrix(tabs / f"transfer_image_{family}.md", img,
                              C.TRAIN_SOURCES, C.ALL_TARGETS,
                              f"Image-level transfer, {F.label_for(family)}",
                              rates=rates, lift=lift)
        box = matrix_for(rows, family, BOX_METRIC)
        box_note = ("Only localisation-protocol pairs appear. weld_surface_v2 "
                    "is whole-region, so its row and column are undefined "
                    "here, and a number there would measure annotation "
                    "mismatch rather than domain shift.")
        heatmap(figs / f"transfer_box_{family}.png", box,
                C.BOX_SOURCES, C.BOX_TARGETS,
                f"Box-level transfer, {F.label_for(family)}",
                "class-agnostic mAP50", note=box_note)
        write_markdown_matrix(
            tabs / f"transfer_box_{family}.md", box, C.BOX_SOURCES, C.BOX_TARGETS,
            f"Box-level transfer, {F.label_for(family)}", note=box_note)

    pooled = {k: mean(list(v.values())) for k, v in matrix_all(rows, IMAGE_METRIC).items()}
    pooled_lift = {k: mean(list(v.values()))
                   for k, v in matrix_all(rows, LIFT_METRIC).items()}
    heatmap(figs / "transfer_image_mean.png", pooled,
            C.TRAIN_SOURCES, C.ALL_TARGETS,
            "Image-level transfer, mean over configurations",
            "annotated-feature AP",
            xticklabels=base_rate_ticks(C.ALL_TARGETS, rates),
            note=BASE_RATE_NOTE)
    heatmap(figs / "transfer_lift_mean.png", pooled_lift,
            C.TRAIN_SOURCES, C.ALL_TARGETS,
            "Image-level lift over base rate, mean over configurations",
            "AP minus base rate", diverging=True,
            xticklabels=base_rate_ticks(C.ALL_TARGETS, rates),
            note=("Zero means the model matched a constant-positive "
                  "predictor. Compare columns here, not on raw AP."))
    write_markdown_matrix(tabs / "transfer_image_mean.md", pooled,
                          C.TRAIN_SOURCES, C.ALL_TARGETS,
                          "Image-level transfer, mean over configurations",
                          rates=rates, lift=pooled_lift)

    # --- derived quantities ------------------------------------------------
    # The gap is computed on BOTH the raw AP and the base-rate corrected
    # margin. On raw AP the gap is not comparable across sources, because each
    # target column sits on its own prevalence floor: lohi at 0.976, visual at
    # 0.893, twoclass at 0.800, kaggle at 0.511. A source whose cross-domain
    # targets happen to be high-prevalence gets a flattering gap for free, and
    # kaggle_v2 gets a NEGATIVE gap purely because its own diagonal is the only
    # one measured on a balanced column. Subtracting the base rate from every
    # term removes that, so the corrected gap is the one to read.
    gaps = (generalization_gap_rows(rows, IMAGE_METRIC)
            + generalization_gap_rows(rows, LIFT_METRIC))
    write_csv(tabs / "generalization_gaps.csv", gaps)
    for metric, stem, title in (
            (LIFT_METRIC, "generalization_gap",
             "Generalization gap, base-rate corrected (AP minus base rate)"),
            (IMAGE_METRIC, "generalization_gap_raw_ap",
             "Generalization gap, raw AP (prevalence-confounded, see corrected)")):
        cells = {(g["source"], g["family"]): g["generalization_gap"]
                 for g in gaps if g["metric"] == metric}
        # The columns here are configurations, not datasets, so the default
        # "evaluated on" label would be wrong.
        heatmap(figs / f"{stem}.png", cells,
                C.TRAIN_SOURCES, families, title,
                "gap", diverging=True,
                xlabel="configuration",
                xticklabels=[F.label_for(f) for f in families],
                note=("Corrected gap. Every term is AP minus that target's "
                      "positive rate, so columns with different prevalence "
                      "are comparable."
                      if metric == LIFT_METRIC else
                      "Raw AP. Each target sits on its own prevalence floor, "
                      "so these values are NOT comparable across sources. "
                      "Read generalization_gap.png instead."))

    write_csv(tabs / "transfer_ratios.csv", transfer_ratio_rows(rows, IMAGE_METRIC))
    write_csv(tabs / "cells.csv", rows)

    # --- split-block intervals on every transfer ratio ----------------------
    # The pre-registered Friedman test asks whether the four configurations
    # differ, and it answered no. That is not the paper's claim. The claim is
    # that transfer collapses, which is a statement about the SIZE of T_ij, so
    # it needs an interval of its own from the finite correlated test sample.
    # Numerator and denominator are scored on the same test images, so one
    # block resample drives both. The diagonal is a self-check: T_ii must come
    # back exactly 1.0 with zero width.
    import transfer_bootstrap as TB
    boot = TB.cached(args.results_dir, C.BOOTSTRAP_RESAMPLES, C.BOOTSTRAP_SEED)
    write_csv(tabs / "transfer_ratio_intervals.csv", boot)
    transfer_interval_markdown(tabs / "transfer_ratio_intervals.md", boot)
    bad = [r for r in boot
           if r["source"] == r["target"]
           and abs((r["transfer_ratio"] or 0) - 1.0) > 1e-9]
    if bad:
        raise SystemExit(f"diagonal transfer ratio is not 1.0 for {len(bad)} "
                         f"cells. The numerator and denominator are not paired "
                         f"on the same test images.")
    n_boot = attach_thin_class_bootstraps(records, args.results_dir)
    if n_boot:
        print(f"split-block bootstrap computed for {n_boot} cells")
    # Pass the results dir in use. Defaulting it here is what let a fixture
    # run load the real correction cache.
    per_class = per_class_rows(
        records, PCN.load(args.results_dir, verbose=True))
    stale = uncorrected_rows(per_class, records)
    if stale:
        cells = sorted({C.cell_name(r) for r in stale})
        raise SystemExit(
            f"{len(stale)} per-class rows in {len(cells)} cells carry the "
            f"evaluator's own AP, which skips NMS. That is the the project log 5p "
            f"defect, so per_class_ap.csv and .md were not written. "
            f"per_class_nms.py corrects a cell only from its raw archive. "
            f"Check that these exist under {args.results_dir}: "
            + ", ".join(f"raw__{c}.npz" for c in cells))
    write_csv(tabs / "per_class_ap.csv", per_class)
    per_class_markdown(tabs / "per_class_ap.md", per_class)
    dirty = dirty_three_number_rows(rows)
    overlap = seen_in_downloaded_train()
    for row in dirty:
        row.update({f"overlap_{k}": v
                    for k, v in overlap.get(row["source"], {}).items()})
    write_csv(tabs / "downloaded_vs_audited.csv", dirty)
    dirty_markdown(tabs / "downloaded_vs_audited.md", dirty, overlap)
    # Image-level results against chance, since 6 October 2026. AP minus the
    # positive rate has a floor above zero under this estimator, so the
    # image-level claims are read in AUROC, with split-block intervals and
    # the retained margin above 0.5. The fixtures ship no raw archives, so a
    # fixture run skips it. See image_level.py.
    import image_level as IL
    IL.run_main(args.results_dir, tabs)
    rel = reliability_rows(records)
    write_csv(tabs / "reliability_bins.csv", rel)
    # NO MAIN-GRID RELIABILITY FIGURE IS DRAWN, deliberately, since 24 September
    # 2026. The one this used to draw pooled every configuration and every
    # target into one curve. The paper reads calibration within a configuration
    # and never across families, whose scores come from different heads, and
    # the five targets have defect rates from about 0.51 to 0.98, so a pooled
    # curve has no interpretation. It was also blank, because the bins were
    # read under the wrong keys. Figure 4.4 is drawn from the internal control
    # by grid/arms/reliability_control_figure.py. If a main-grid reliability
    # figure is ever wanted, it must be one panel per configuration and one
    # curve per target, never a pool.

    # --- arm B, if it has run ----------------------------------------------
    kshot = []
    if args.kshot and args.kshot.is_file():
        kshot = recovery_rows(json.loads(args.kshot.read_text()))
        write_csv(tabs / "recovery_kshot.csv", kshot)
        recovery_figure(figs / "recovery_kshot.png", kshot)

    stats = run_stats(rows, args.out_dir)

    manifest = {
        "evaluation_records": len(records),
        "rows": len(rows),
        "expected_cells": len(C.all_cells()),
        "missing_cells": missing,
        "complete": not missing,
        "configurations_present": families,
        "kshot_points": len(kshot),
        "statistical_test": {
            m: {"applicable": v.get("applicable"),
                "primary_p": (v.get("primary") or {}).get("p_value"),
                "significant": (v.get("primary") or {}).get("significant")}
            for m, v in stats.items()},
        "figures": sorted(p.name for p in figs.glob("*.png")),
        "tables": sorted(p.name for p in tabs.glob("*")),
    }
    (args.out_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=float))
    print(json.dumps(manifest, indent=2, default=float))
    if missing:
        print(f"\nINCOMPLETE: {len(missing)} cells have no evaluation yet.")


if __name__ == "__main__":
    main()
