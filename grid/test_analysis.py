#!/usr/bin/env python3
"""
End-to-end check that analysis.py produces every manuscript artefact.

Runs against the fabricated fixtures, so it can run with no GPU, no cluster
and no results. Its job is to guarantee that when the real evaluations land,
`python3 grid/analysis.py` is genuinely one command.

    python3 grid/test_analysis.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analysis as A  # noqa: E402
import config as C  # noqa: E402
import stats_tests as S  # noqa: E402

FIXTURES = HERE / "fixtures" / "results"
FAILURES = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


print("fixtures")
check("fixtures exist", FIXTURES.is_dir(),
      "run make_fixtures.py first" if not FIXTURES.is_dir() else "")
if not FIXTURES.is_dir():
    raise SystemExit("no fixtures")
check("marker file present", (FIXTURES / "FABRICATED_DATA_DO_NOT_CITE.txt").is_file())
check("fixtures are outside the results tree",
      "fixtures" in FIXTURES.parts and "grid_results" not in FIXTURES.parts)
sample = json.loads(next(FIXTURES.glob("eval__*.json")).read_text())
check("every record is flagged fabricated", sample.get("FABRICATED") is True)
check("one record per cell",
      len(list(FIXTURES.glob("eval__*.json"))) == len(C.all_cells()),
      f"{len(list(FIXTURES.glob('eval__*.json')))} of {len(C.all_cells())}")

print("\nanalysis end to end")
# The fixtures ship no raw archives, so there is nothing for per_class_nms to
# correct, and since 25 September 2026 load() writes no cache for them. The
# file may still exist in an older checkout, so compare before and after.
FIXTURE_NMS_CACHE = HERE / "fixtures" / "per_class_nms_cache.json"


def stat_or_none(path: Path):
    return path.stat().st_mtime_ns if path.exists() else None


nms_cache_before = stat_or_none(FIXTURE_NMS_CACHE)
with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp) / "analysis"
    proc = subprocess.run(
        [sys.executable, str(HERE / "analysis.py"),
         "--results-dir", str(FIXTURES),
         "--kshot", str(HERE / "fixtures" / "kshot_fixture.json"),
         "--out-dir", str(out)],
        capture_output=True, text=True)
    check("analysis.py exits cleanly", proc.returncode == 0,
          proc.stderr.strip().splitlines()[-1] if proc.returncode else "")
    if proc.returncode:
        raise SystemExit(proc.stderr)

    manifest = json.loads((out / "analysis_manifest.json").read_text())
    check("no missing cells on a complete fixture set",
          manifest["complete"] is True, str(manifest["missing_cells"])[:60])
    check("all four configurations present",
          len(manifest["configurations_present"]) == 4)

    figures = set(manifest["figures"])
    for want in ("transfer_image_mean.png", "generalization_gap.png",
                 "recovery_kshot.png"):
        check(f"figure {want}", want in figures)
    # Retired on 24 September 2026. It pooled every configuration and every
    # target into one curve, which cannot be read, and it was blank besides.
    check("no pooled main-grid reliability figure is drawn",
          "reliability.png" not in figures)
    for family in C.FAMILIES:
        check(f"per-configuration heatmap {family}",
              f"transfer_image_{family}.png" in figures)

    tables = set(manifest["tables"])
    for want in ("per_class_ap.md", "downloaded_vs_audited.md",
                 "transfer_ratios.csv", "generalization_gaps.csv",
                 "recovery_kshot.csv", "reliability_bins.csv", "cells.csv"):
        check(f"table {want}", want in tables)

    # Every per-class AP must carry its instance count. This is a project rule,
    # not a nicety, so it is asserted rather than trusted.
    import csv
    with (out / "tables" / "per_class_ap.csv").open() as fh:
        pc = list(csv.DictReader(fh))
    check("per-class rows exist", len(pc) > 0, f"{len(pc)} rows")
    check("every per-class AP carries a test instance count",
          all(r["test_instances"] not in ("", None) for r in pc))
    # The count alone is not enough. On 9 September 2026 every AP value was
    # empty because analysis.py read `ap50` while evaluate_cell.py wrote
    # `map50`, and this suite passed regardless because it only checked the
    # count. Assert the value itself.
    check("every per-class AP carries a value",
          all(r["ap50"] not in ("", None) for r in pc),
          f"{sum(1 for r in pc if r['ap50'] in ('', None))} empty of {len(pc)}")
    thin = [r for r in pc if r["thin"] == "True"]
    check("thin classes carry a bootstrap interval",
          all(r["ci_lo"] not in ("", None) for r in thin),
          f"{len(thin)} thin rows")

    # Same failure as the per-class one above, in the calibration bins. Until
    # 24 September 2026 four columns of reliability_bins.csv were empty in
    # every row because the reader asked for keys the evaluator never writes,
    # and nothing here looked at the values. Assert them.
    with (out / "tables" / "reliability_bins.csv").open() as fh:
        rb = list(csv.DictReader(fh))
    live = [r for r in rb if int(r["count"]) > 0]
    check("reliability bins exist", len(live) > 0, f"{len(live)} non-empty")
    check("every non-empty reliability bin carries all five values",
          all(r[k] not in ("", None) for r in live
              for k in A.CALIBRATION_BIN_KEYS),
          f"{sum(1 for r in live for k in A.CALIBRATION_BIN_KEYS if r[k] in ('', None))} empty cells")

    # The CSV carrying the interval is not enough. The rendered table is what
    # a reader sees, and an earlier version printed only seed dispersion for
    # thin classes while its own header promised a bootstrap. Seed SD and the
    # bootstrap answer different questions, so the table must show both.
    md = (out / "tables" / "per_class_ap.md").read_text()
    check("per-class table has a seed SD column", "seed SD" in md)
    check("per-class table has a bootstrap column", "split-block bootstrap" in md)
    # Table rows only. The prose above also contains the word "thin".
    thin_lines = [ln for ln in md.splitlines()
                  if ln.startswith("| ") and "**thin**" in ln]
    check("thin rows exist in the rendered table", len(thin_lines) > 0,
          f"{len(thin_lines)} rows")
    check("every rendered thin row shows a bootstrap interval",
          all("[" in ln and "]" in ln for ln in thin_lines))
    check("non-thin rows do not claim an interval they never computed",
          all("not needed" in ln
              for ln in md.splitlines()
              if ln.startswith("| ") and "| ok |" in ln))

    # The three-number comparison must be complete for every source.
    with (out / "tables" / "downloaded_vs_audited.csv").open() as fh:
        dv = list(csv.DictReader(fh))
    check("downloaded comparison covers every source",
          {r["source"] for r in dv} == set(C.DIRTY_SOURCES))
    check("all three numbers present",
          all(r["A_downloaded_train_downloaded_test"] and
              r["B_downloaded_train_audited_test"] and
              r["C_audited_train_audited_test"] for r in dv))
    check("comparison is labelled as an estimate shift",
          all("not a causal" in r["interpretation"] for r in dv))

    stats = json.loads((out / "statistical_tests.json").read_text())
    check("pre-registered test ran on image level",
          stats["image_ap"]["applicable"] is True)
    check("primary test is Friedman",
          "Friedman" in stats["image_ap"]["primary"]["test"])
    check("Nemenyi post-hoc present",
          "critical_difference" in stats["image_ap"]["primary"]["posthoc"])
    check("secondary ANOVA reported for Lema comparability",
          stats["image_ap"]["secondary"]["anova"]["applicable"] is True)
    check("marked pre-registered", stats["image_ap"]["preregistered"] is True)
check("a fixture render writes no per-class NMS cache",
      stat_or_none(FIXTURE_NMS_CACHE) == nms_cache_before)

print("\nper-class NMS cache, the two ways it returned nothing for real results")
# the project log 5p. Until 25 September 2026 per_class_nms.load() returned {} for
# real results in two ways, and analysis.py then fell back to per-class
# values that skip NMS. A cache written empty before the results arrived was
# trusted with no notice. A cache keyed to a project folder that had since
# moved returned {} after one printed line. These run load() in a scratch
# tree with recompute() replaced, because a real recompute needs the raw
# archives and the ground truth. Only the caching rule is under test.
import per_class_nms as PCN

rebuilds = []


def fake_recompute(results_dir, verbose=True):
    rebuilds.append(Path(results_dir))
    return {k: {"map50": 0.5} for k in PCN.expected_keys(results_dir)}


def write_records(results: Path):
    """Two main cells and one downloaded-data cell, with no raw archives."""
    results.mkdir(parents=True, exist_ok=True)
    for name, kind, seed in (("kaggle_v2__yolo11s__seed0", "main", 0),
                             ("kaggle_v2__yolo11s__seed1", "main", 1),
                             ("dirty_kaggle_v2__yolo11s__seed0", "dirty", 0)):
        cell = {"source": "kaggle_v2", "family": "yolo11s", "seed": seed,
                "kind": kind}
        record = {"name": name, "cell": cell, "targets": {"kaggle_v2": {
            "per_class_in_domain": {"Defect": {"map50": 0.1}}}}}
        (results / f"eval__{name}.json").write_text(json.dumps(record))


def add_raw_archives(results: Path):
    # Existence is all load() checks. recompute() would open them.
    for path in results.glob("eval__*.json"):
        name = path.name[len("eval__"):-len(".json")]
        (results / f"raw__{name}.npz").write_bytes(b"")


n_classes = len(C.DATASETS["kaggle_v2"]["classes"])
real_recompute = PCN.recompute
PCN.recompute = fake_recompute
try:
    with tempfile.TemporaryDirectory() as tmp:
        results = Path(tmp) / "project" / "grid_results" / "results"
        cache = PCN.cache_path(results)
        check("a results directory that does not exist yet returns nothing",
              PCN.load(results) == {} and not rebuilds)
        check("and leaves no cache behind to be trusted later",
              not cache.exists())
        write_records(results)
        check("records with no raw archive, like the fixtures, return nothing",
              PCN.load(results) == {} and not cache.exists() and not rebuilds)

        add_raw_archives(results)
        got = PCN.load(results)
        check("results that arrive later are corrected, not skipped",
              len(rebuilds) == 1 and len(got) == 2 * n_classes,
              f"{len(rebuilds)} rebuilds, {len(got)} values")
        check("the downloaded-data cell is not corrected",
              not any(k.startswith("dirty_") for k in got))
        check("the cache records the directory it was built for",
              json.loads(cache.read_text())["results_dir"]
              == str(results.resolve()))
        check("a cache that fits is read without a rebuild",
              PCN.load(results) == got and len(rebuilds) == 1)

        cache.write_text(json.dumps({"results_dir": str(results.resolve()),
                                     "resamples": None, "values": {}}))
        check("an empty cache under the right key is rebuilt, not trusted",
              PCN.load(results) == got and len(rebuilds) == 2)

        moved = Path(tmp) / "moved"
        (Path(tmp) / "project").rename(moved)
        results = moved / "grid_results" / "results"
        check("a cache keyed to a folder that has moved is rebuilt",
              PCN.load(results) == got and len(rebuilds) == 3)
        check("and the rebuilt cache records the new folder",
              json.loads(PCN.cache_path(results).read_text())["results_dir"]
              == str(results.resolve()))

        (results / "raw__kaggle_v2__yolo11s__seed1.npz").unlink()
        fewer = PCN.load(results)
        check("a cell whose raw archive is gone is rebuilt out of the cache",
              len(rebuilds) == 4 and len(fewer) == n_classes
              and not any("seed1" in k for k in fewer))
finally:
    PCN.recompute = real_recompute

print("\nanalysis.py refuses a per-class row without the NMS correction")


def per_class_row(kind, corrected):
    return {"kind": kind, "nms_corrected": corrected, "source": "kaggle_v2",
            "family": "rtdetr_l", "seed": 0}


real_records = [{"name": "kaggle_v2__rtdetr_l__seed0"}]
fabricated_records = [{"name": "kaggle_v2__rtdetr_l__seed0", "FABRICATED": True}]
check("a real main row without the correction is refused",
      len(A.uncorrected_rows([per_class_row("main", False)], real_records)) == 1)
check("a real reference row without it is refused",
      len(A.uncorrected_rows([per_class_row("reference", False)],
                             real_records)) == 1)
check("a corrected row passes",
      A.uncorrected_rows([per_class_row("main", True)], real_records) == [])
check("a downloaded-data row passes, since it is never corrected",
      A.uncorrected_rows([per_class_row("dirty", False)], real_records) == [])
check("a wholly fabricated set passes, since it has nothing to correct",
      A.uncorrected_rows([per_class_row("main", False)],
                         fabricated_records) == [])
check("a set mixing fabricated and real records does not",
      len(A.uncorrected_rows([per_class_row("main", False)],
                             fabricated_records + real_records)) == 1)

# End to end. The fixtures with their FABRICATED flag removed stand for real
# records whose raw archives are missing. analysis.py must stop before it
# writes a per-class table, and say which archives to look for. The two
# bootstrap caches go beside the copy, as they sit beside the fixtures. Both
# are keyed by resample count and seed, so they apply unchanged, and without
# them the transfer bootstrap would try to load ground truth.
import shutil

with tempfile.TemporaryDirectory() as tmp:
    unflagged = Path(tmp) / "results"
    unflagged.mkdir()
    for path in FIXTURES.glob("eval__*.json"):
        record = json.loads(path.read_text())
        record.pop("FABRICATED", None)
        (unflagged / path.name).write_text(json.dumps(record))
    for name in ("bootstrap_cache.json", "transfer_bootstrap_cache.json"):
        shutil.copy(FIXTURES.parent / name, Path(tmp) / name)
    out = Path(tmp) / "analysis"
    proc = subprocess.run(
        [sys.executable, str(HERE / "analysis.py"),
         "--results-dir", str(unflagged), "--out-dir", str(out)],
        capture_output=True, text=True)
    last = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else ""
    check("analysis.py stops on real records it cannot correct",
          proc.returncode != 0 and "the project log 5p" in last, last[:80])
    check("and names the raw archives to look for",
          "raw__kaggle_v2__rtdetr_l__seed0.npz" in last)
    check("and writes no per-class table",
          not (out / "tables" / "per_class_ap.csv").exists()
          and not (out / "tables" / "per_class_ap.md").exists())

print("\nstatistical test behaviour")
cfgs = ["a", "b", "c", "d"]
sep = {f"c{i}": {c: 0.2 + 0.15 * k for k, c in enumerate(cfgs)} for i in range(12)}
r = S.compare_configurations(sep, cfgs)
check("detects a real separation", r["primary"]["significant"] is True,
      f"p={r['primary']['p_value']:.2e}")
flat = {f"c{i}": {c: 0.5 for c in cfgs} for i in range(12)}
r2 = S.compare_configurations(flat, cfgs)
check("does not fire on identical scores", not r2["primary"]["significant"])
# distinct key: reusing "c0" would be overwritten by sep
part = {**sep, "incomplete": {"a": 0.5, "b": None, "c": 0.4, "d": 0.3}}
r3 = S.compare_configurations(part, cfgs)
check("incomplete blocks are dropped, not imputed",
      r3["n_blocks_dropped_incomplete"] == 1)

print("\nschema conformance, fixtures against the real evaluator")
# A fixture that agrees with the CONSUMER instead of the PRODUCER cannot catch
# a schema drift, which is the only reason the fixtures exist. This reads the
# key names evaluate_cell.py actually assigns into per_class[name], straight
# from its syntax tree, and requires the fixtures and analysis.py to match.
import ast

src = ast.parse((Path(__file__).resolve().parent / "evaluate_cell.py").read_text())
produced = set()
for node in ast.walk(src):
    if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)
            and any(isinstance(t, ast.Subscript)
                    and isinstance(t.value, ast.Name)
                    and t.value.id == "per_class" for t in node.targets)):
        produced = {k.value for k in node.value.keys
                    if isinstance(k, ast.Constant)}
check("evaluator per-class keys were located", bool(produced),
      ", ".join(sorted(produced)))
check("analysis reads the AP key the evaluator writes",
      A.PER_CLASS_AP_KEY in produced,
      f"reads {A.PER_CLASS_AP_KEY!r}, evaluator writes {sorted(produced)}")
check("analysis reads the AP range key the evaluator writes",
      A.PER_CLASS_AP_RANGE_KEY in produced,
      f"reads {A.PER_CLASS_AP_RANGE_KEY!r}")

fixture_entry = next(
    info
    for path in sorted(FIXTURES.glob("eval__*.json"))
    for entry in json.loads(path.read_text())["targets"].values()
    for info in (entry.get("per_class_in_domain") or {}).values())
check("fixtures emit the evaluator's AP key, not the reader's",
      A.PER_CLASS_AP_KEY in fixture_entry,
      f"fixture keys {sorted(fixture_entry)}")
check("fixtures carry no key the evaluator never writes",
      set(fixture_entry) <= produced | {A.PER_CLASS_BOOTSTRAP_KEY,
                                        "cluster_bootstrap_required"},
      f"extra {sorted(set(fixture_entry) - produced)}")

print("\ncalibration bins, the reader and the fixtures against the producer")
# The bin keys are taken from the producer by calling it, not typed here, so a
# rename in metrics.py and a stale reader cannot agree with each other by
# accident. The synthetic record below is built from those bins, so this
# check would have caught the reader asking for `lower`, `upper`,
# `confidence` and `accuracy`, which is what left reliability_bins.csv empty.
import math

import numpy as np

import metrics as M

rng = np.random.default_rng(20260924)
scores = np.concatenate([rng.uniform(0.0, 0.25, 40), rng.uniform(0.6, 1.0, 60)])
labels = rng.uniform(size=scores.size) < scores
cal = M.expected_calibration_error(scores, labels)
producer_keys = tuple(cal["bins"][0])
check("the producer's bin keys were read by calling it",
      len(producer_keys) > 0, ", ".join(producer_keys))
check("analysis reads exactly the keys the producer writes",
      tuple(A.CALIBRATION_BIN_KEYS) == producer_keys,
      f"reads {list(A.CALIBRATION_BIN_KEYS)}, producer {list(producer_keys)}")
empty_bins = sum(1 for b in cal["bins"] if b["count"] == 0)
synthetic = {"name": "synthetic", "cell": {"source": "kaggle_v2",
                                           "family": "yolo11s", "seed": 0,
                                           "kind": "main"},
             "targets": {"kaggle_v2": {"test_split": "audited",
                                       "image_level": {"calibration": cal}},
                         "lohi_weld": {"test_split": "audited",
                                       "image_level": {"calibration": cal}}}}
rows = A.reliability_rows([synthetic])
live = [r for r in rows if r["count"] > 0]
check("synthetic bins include empty ones, so both branches are exercised",
      empty_bins > 0 and len(live) > 0, f"{empty_bins} empty, {len(live)} live")
check("every column is finite in every bin whose count is above zero",
      all(isinstance(r[k], (int, float)) and math.isfinite(r[k])
          for r in live for k in A.CALIBRATION_BIN_KEYS))
check("the reader keeps both regimes apart",
      {r["regime"] for r in rows} == {"in domain", "cross domain"})
try:
    A.reliability_rows([{"name": "stale", "cell": synthetic["cell"],
                         "targets": {"kaggle_v2": {"image_level": {
                             "calibration": {"bins": [{"lower": 0.0}]}}}}}])
    raised = False
except KeyError:
    raised = True
check("a bin under the old key names raises instead of reading as empty",
      raised)
fixture_bin = next(
    b
    for path in sorted(FIXTURES.glob("eval__*.json"))
    for entry in json.loads(path.read_text())["targets"].values()
    for b in entry["image_level"]["calibration"]["bins"])
check("fixtures emit the producer's calibration keys, not the reader's",
      tuple(fixture_bin) == producer_keys, f"fixture keys {list(fixture_bin)}")

print("\nthe empty-figure guard, figstyle.save")
# A blank Figure 4.4 passed every check this project had, because every check
# reads a number. figstyle.save refuses any visible data axes with no data.
# Every real figure passing is shown by the end-to-end run above, which saves
# every figure analysis.py draws through it.
import figstyle as F
import matplotlib.pyplot as plt


def refuses(fig) -> bool:
    with tempfile.TemporaryDirectory() as d:
        try:
            F.save(fig, Path(d) / "x.png")
        except ValueError:
            return True
        finally:
            plt.close(fig)
    return False


fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], gid=F.REFERENCE)
check("a figure holding only a reference line refuses to save", refuses(fig))

fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], gid=F.REFERENCE)
ax.axhline(0.976, gid=F.REFERENCE)
ax.plot([np.nan, np.nan], [0.5, 0.7])
ax.scatter([], [])
ax.fill_between([0.1, 0.2], [np.nan, np.nan], [np.nan, np.nan])
check("the old blank Figure 4.4, reference lines and NaN data, refuses",
      refuses(fig))

fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], gid=F.REFERENCE)
ax.scatter([0.3], [0.9])
check("one finite data point is enough to save", not refuses(fig))

fig, axes = plt.subplots(1, 2)
axes[0].plot([0.1, 0.2], [0.3, 0.4])
axes[1].axis("off")
check("an axes turned off is not held to the rule", not refuses(fig))

fig, axes = plt.subplots(1, 2)
axes[0].plot([0.1, 0.2], [0.3, 0.4])
check("a blank panel left switched on refuses to save",
      refuses(fig))

fig, ax = plt.subplots()
im = ax.imshow(np.full((2, 3), np.nan))
fig.colorbar(im, ax=ax)
check("an all n/a heatmap refuses, and its colourbar does not rescue it",
      refuses(fig))

fig, ax = plt.subplots()
ax.plot([0.1, 0.2], [0.3, 0.4])
with tempfile.TemporaryDirectory() as d:
    F.save(fig, Path(d) / "a.pdf")
    F.save(fig, Path(d) / "b.pdf")
    same = (Path(d) / "a.pdf").read_bytes() == (Path(d) / "b.pdf").read_bytes()
plt.close(fig)
check("a PDF saved twice is byte identical, with no CreationDate", same)

print("\nimage level against chance, image_level.py")
# 6 October 2026. Image-level AP minus the positive rate was read as if zero
# meant no skill. Under the evaluator's interpolated estimator an
# uninformative scorer sits above zero, so AUROC, whose chance level is 0.5,
# carries the image-level claims now. These pin the properties that matter.
import image_level as IL
from scipy.stats import rankdata

rng = np.random.default_rng(20261006)
lab = np.r_[np.ones(72, bool), np.zeros(69, bool)]
aucs, lifts = [], []
for _ in range(400):
    s = rng.random(len(lab))
    aucs.append(IL.auroc(s, lab))
    lifts.append(IL.chance_ap(s, lab, 200, 1)[0] - lab.mean())
check("uninformative scores give AUROC near 0.5",
      abs(np.mean(aucs) - 0.5) < 0.01, f"{np.mean(aucs):.4f}")
check("and a chance AP above the positive rate, on kaggle_v2's sizes",
      np.mean(lifts) > 0.03, f"lift {np.mean(lifts):+.4f}")

blocks = np.repeat(np.arange(47), 3)[:len(lab)].astype(str)
sp = IL.Split("test", np.arange(len(lab)).astype(str), lab, blocks, "test",
              resamples=300)
perfect = lab.astype(float)
d = IL.weighted_auroc(perfect, lab, sp.weights)
check("a perfect scorer gives AUROC 1.0, on the full sample and every draw",
      IL.auroc(perfect, lab) == 1.0 and np.all(d[np.isfinite(d)] == 1.0))

flat = np.zeros(len(lab))
flipped = lab[::-1].copy()
check("all-tied scores give AUROC 0.5 whatever the archive order",
      IL.auroc(flat, lab) == 0.5 and IL.auroc(flat, flipped) == 0.5
      and np.allclose(IL.weighted_auroc(flat, lab, sp.weights), 0.5))
lo, hi = IL.tie_envelope(flat, lab)
check("and the AP of all-tied scores spans the positive rate to 1.0",
      abs(lo - lab.mean()) < 1e-12 and hi == 1.0, f"[{lo:.3f}, {hi:.3f}]")

s = np.round(rng.random(len(lab)), 1)            # deliberately tied
w = sp.weights[:25]
ok = True
for row, auc in zip(w, IL.weighted_auroc(s, lab, w)):
    idx = np.repeat(np.arange(len(lab)), row)
    y, r = lab[idx], rankdata(s[idx])
    if y.all() or not y.any():
        ok &= not np.isfinite(auc)
        continue
    ref = (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())
    ok &= abs(ref - auc) < 1e-12
check("weighted AUROC equals rank AUROC over the concatenated draw", ok)

perms = IL.shuffle_perms(len(lab), 40, 3)
fast = IL.shuffled_aps(s, lab, perms)
slow = np.array([IL.chance_ap_slow(s, lab, p) for p in perms])
check("chance AP's fast path is binary_metrics, shuffle by shuffle",
      np.max(np.abs(fast - slow)) < 1e-12, f"{np.max(np.abs(fast - slow)):.1e}")

even = IL.sample_structure(np.ones(12, bool), np.repeat(np.arange(4), 3))
check("a class spread evenly over n blocks has n effective blocks",
      even["pos_effective_blocks"] == 4.0 and
      IL.effective_blocks([5]) == 1.0, f"{even['pos_effective_blocks']}")

key = ("t", "yolo11s", 0, "t")
pt = {key: 0.8}
dr = {key: IL.weighted_auroc(rng.random(len(lab)), lab, sp.weights)}
m = IL.margin_rows(pt, dr, ["t"], ["t"], ["yolo11s"], [0],
                   lambda t, f, sd: (t, f, sd, t))
check("the diagonal retained margin is exactly 1 with zero width",
      all(r["retained_margin"] == 1.0 and r["retained_margin_lo"] == 1.0
          and r["retained_margin_hi"] == 1.0 for r in m), f"{len(m)} rows")
check("a denominator at chance leaves the margin undefined",
      np.isnan(IL.margin(0.7, 0.5)) and np.isnan(IL.margin(0.7, 0.4)))
check("the kaggle_v2 filename family strips frame numbers, not names",
      IL.family_of("SampleV1_2_mp4-12_jpg.rf.aa.jpg") == "SampleV1_2_mp4"
      and IL.family_of("bad_weld_vid177_jpeg.rf.5b.jpg") == "bad_weld_vid"
      and IL.family_of("Screenshot-2022-12-05-223445_jpeg_jpg.rf.7e.jpg")
      == "Screenshot")

print()
if FAILURES:
    raise SystemExit(f"{len(FAILURES)} FAILED: {FAILURES}")
print("analysis pipeline verified end to end on fabricated fixtures")
