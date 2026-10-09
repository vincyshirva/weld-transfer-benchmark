#!/usr/bin/env python3
"""Confirm the arm B class space defect. REPORT ONLY, writes nothing.

THE DEFECT, in one sentence: `evaluate_arm.py` filters predictions by the
SOURCE dataset's defect ids, but an arm B model predicts in the TARGET's class
space, because `kshot.materialise` writes the target's class list into the
k-shot data.yaml and ultralytics rebuilds the detection head at the target nc.

For `lohi_control` and arm D the source filter is correct, since those models
train on their own source and predict in its class space. Arm B is the only
arm where the training class space is not the source's.

This script confirms four facts before anything is changed:

  1. the head really is rebuilt, read off the predicted class indices
  2. kaggle to twoclass retains nothing, so every box number is a floor of 0
  3. kaggle to visual retains `workpiece`, an EXCLUDED class, and scores it
     against defect ground truth
  4. the source against target filter table, reproduced from config.py

Usage:
    python3 grid/arms/confirm_class_space.py
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import config as C  # noqa: E402
import config_arms as A  # noqa: E402

RESULTS = C.PROJECT / "grid_results_arms" / "results"
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILED.append(name)


def head_nc(target: str) -> int:
    """Output class count of an arm B head, which is the target's nc.

    `kshot.materialise` writes `nc: len(classes)` for the TARGET, so
    ultralytics discards the source head and builds a new one this wide.
    """
    return len(A.dataset_entry(target)["classes"])


# --------------------------------------------------------------------------
# Fact 1
# --------------------------------------------------------------------------

def fact_1_head_is_rebuilt() -> None:
    print("\nFACT 1. the detection head is rebuilt at the TARGET class count")
    cases = [
        ("armb__kaggle_v2_to_lohi_weld__yolo11s__k250__draw0", "kaggle_v2",
         "lohi_weld"),
        ("armb__lohi_weld_to_kaggle_v2__yolo11s__k250__draw0", "lohi_weld",
         "kaggle_v2"),
    ]
    for cell, source, target in cases:
        path = RESULTS / f"raw_last__{cell}.npz"
        if not path.is_file():
            check(f"{cell} raw archive present", False, str(path))
            continue
        z = np.load(path, allow_pickle=True)
        ids = z[f"{target}__pred_class"].astype(int)
        seen = sorted(set(ids.tolist()))
        counts = dict(sorted(collections.Counter(ids.tolist()).items()))
        n_src, n_tgt = head_nc(source), head_nc(target)
        print(f"    {cell}")
        print(f"      source {source} nc={n_src}, target {target} nc={n_tgt}")
        print(f"      predicted indices {seen}")
        print(f"      counts {counts}")
        check(f"{source} to {target}: indices span the TARGET nc, not the "
              f"source nc", max(seen) + 1 == n_tgt and max(seen) + 1 != n_src,
              f"max index {max(seen)}, target nc {n_tgt}, source nc {n_src}")


# --------------------------------------------------------------------------
# Fact 2
# --------------------------------------------------------------------------

def fact_2_twoclass_retains_nothing() -> None:
    print("\nFACT 2. kaggle_v2 to twoclass retains no prediction at all")
    print("    the filter asks for source index 2, the head has 2 outputs")
    rows, bad = [], []
    for endpoint in ("eval__", "eval_last__"):
        for path in sorted(RESULTS.glob(f"{endpoint}armb__kaggle_v2_to_twoclass__*.json")):
            rec = json.loads(path.read_text())
            box = rec["targets"]["twoclass"]["box_level"]
            rows.append((path.name, box.get("n_pred"), box.get("map50")))
            if box.get("n_pred") != 0 or box.get("map50") != 0.0:
                bad.append(rows[-1])
    print(f"    records inspected: {len(rows)}  "
          f"(30 cells at two endpoints)")
    check("every kaggle to twoclass record has n_pred 0 and map50 0.0",
          not bad and len(rows) == 60,
          f"{len(bad)} exceptions, {len(rows)} records")
    for name, n, m in rows[:3]:
        print(f"      {name}  n_pred={n}  map50={m}")


# --------------------------------------------------------------------------
# Fact 3
# --------------------------------------------------------------------------

def fact_3_visual_scores_workpiece() -> None:
    print("\nFACT 3. kaggle_v2 to visual scores an EXCLUDED class as a defect")
    classes = A.dataset_entry("visual")["classes"]
    excluded = A.dataset_entry("visual")["excluded"]
    print(f"    visual classes {classes}")
    print(f"    index 2 is {classes[2]!r}, excluded={classes[2] in excluded}")
    check("visual index 2 is an excluded class", classes[2] in excluded,
          f"{classes[2]!r}")
    check("the arm B filter used was the SOURCE kaggle_v2 ids",
          A.arm_defect_ids("kaggle_v2") == {2},
          f"{sorted(A.arm_defect_ids('kaggle_v2'))}")
    check("the correct filter is visual's own defect ids",
          A.arm_defect_ids("visual") == {0},
          f"{sorted(A.arm_defect_ids('visual'))}")

    path = RESULTS / "eval_last__armb__kaggle_v2_to_visual__yolo11s__k250__draw0.json"
    rec = json.loads(path.read_text())
    box = rec["targets"]["visual"]["box_level"]
    print(f"    {path.name}")
    print(f"      n_pred={box.get('n_pred')}  n_gt={box.get('n_gt')}  "
          f"map50={box.get('map50')}")
    check("predictions were retained and scored to exactly zero",
          box.get("n_pred", 0) > 0 and box.get("map50") == 0.0,
          "scene furniture matched against defect ground truth")


# --------------------------------------------------------------------------
# Fact 4
# --------------------------------------------------------------------------

def pair_table() -> list[dict]:
    """Reproduce the source against target filter table FROM config.py.

    Built rather than copied, so it cannot drift away from the configuration
    it describes. `test_class_space.py` asserts this against the table in
    the 15 September plan.
    """
    rows = []
    for source in A.ARMB_START_SOURCES:
        for target in A.armb_targets(source):
            used = A.arm_defect_ids(source)
            correct = A.arm_defect_ids(target)
            nc = head_nc(target)
            # WHAT ACTUALLY APPLIED is the requested set intersected with the
            # head's output range. An id at or above nc can never be predicted,
            # so asking for it is a silent no-op rather than an error. This is
            # why lohi_weld to twoclass comes out correct: it asks for
            # {0,1,2,3} against a 2 output head, and {0,1} is exactly right.
            # Comparing the REQUESTED sets instead would wrongly call that pair
            # broken, and would also hide that kaggle to twoclass retains
            # nothing at all.
            effective = {i for i in used if i < nc}
            rows.append({
                "source": source,
                "target": target,
                "filter_used": sorted(used),
                "correct_filter": sorted(correct),
                "head_nc": nc,
                "box_level": ("scored" if A.box_level_defined(source, target)
                              else "skipped"),
                "effective_filter": sorted(effective),
                "image_level_wrong": effective != correct,
                "retainable": sorted(effective),
                "n_retainable": len(effective),
            })
    return rows


def fact_4_pair_table() -> None:
    print("\nFACT 4. the filter table, reproduced from config.py")
    rows = pair_table()
    print(f"    {'pair':<34s}{'used':<15s}{'correct':<17s}{'nc':>3s}"
          f"{'box':>9s}  verdict")
    for r in rows:
        pair = f"{r['source']} to {r['target']}"
        tgt = A.dataset_entry(r["target"])
        if not r["image_level_wrong"]:
            verdict = "correct by coincidence"
        elif r["n_retainable"] == 0:
            verdict = "nothing retained, n_pred 0"
        else:
            kept = [tgt["classes"][i] for i in r["retainable"]]
            wrongly = [c for c in kept
                       if c in tgt["excluded"] or c in tgt["non_defect"]]
            verdict = f"keeps {', '.join(kept)}"
            if wrongly:
                verdict += f"; {', '.join(wrongly)} is NOT a defect class"
        print(f"    {pair:<34s}{str(r['filter_used']):<15s}"
              f"{str(r['correct_filter']):<17s}{r['head_nc']:>3d}"
              f"{r['box_level']:>9s}  {verdict}")

    wrong_img = [r for r in rows if r["image_level_wrong"]]
    wrong_box = [r for r in rows
                 if r["image_level_wrong"] and r["box_level"] == "scored"]
    check("7 of 8 pairs use the wrong filter at image level",
          len(wrong_img) == 7, f"{len(wrong_img)} of {len(rows)}")
    box_defined = [r for r in rows if r["box_level"] == "scored"]
    check("5 of 6 box defined pairs are wrong at box level",
          len(wrong_box) == 5 and len(box_defined) == 6,
          f"{len(wrong_box)} of {len(box_defined)}")

    # Cell counts. 30 cells per pair, two families by five k by three draws.
    per_pair = len(A.ARMB_FAMILIES) * len(A.ARMB_K) * len(A.ARMB_DRAWS)
    print(f"\n    cells per pair: {per_pair}")
    check("210 of 240 arm B cells are wrong at image level",
          len(wrong_img) * per_pair == 210,
          f"{len(wrong_img)} pairs x {per_pair}")
    check("150 arm B cells are wrong at box level",
          len(wrong_box) * per_pair == 150,
          f"{len(wrong_box)} pairs x {per_pair}")


def main() -> None:
    A.assert_main_unmoved()
    A.register()
    print("confirming the arm B class space defect. Report only, writes "
          "nothing.")
    print(f"  arm fingerprint {A.arm_fingerprint()}, frozen main "
          f"{A.FROZEN_MAIN_FINGERPRINT}")
    fact_1_head_is_rebuilt()
    fact_2_twoclass_retains_nothing()
    fact_3_visual_scores_workpiece()
    fact_4_pair_table()
    print()
    if FAILED:
        raise SystemExit(f"{len(FAILED)} CHECK(S) FAILED: {FAILED}\n"
                         "Do not fix what has not been confirmed.")
    print("all four facts confirmed. The defect is real and is confined to "
          "arm B.")


if __name__ == "__main__":
    main()
