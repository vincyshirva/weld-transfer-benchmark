#!/usr/bin/env python3
"""Known-answer checks for result aggregation and denominator matching."""

from __future__ import annotations

import config as C
from postprocess_results import dirty_shift_rows, gap_rows, transfer_rows


def row(kind, source, family, seed, target, image_ap, test_split="audited",
        result_key=None):
    return {"kind": kind, "source": source, "family": family, "seed": seed,
            "target": target, "result_key": result_key or target,
            "test_split": test_split, "image_ap": image_ap,
            "box_map50": None, "box_map50_95": None}


assert len(list(C.cells())) == 36
assert len(list(C.reference_cells())) == 24
assert len(list(C.dirty_cells())) == 9
assert len(C.all_cells()) == 69
assert sum(len(C.eval_targets(c)) for c in C.all_cells()) == 222

# Each downloaded-data cell is scored twice on the same target: the split it
# shipped with, and the audited split. Same weights, different test set.
_dirty = next(c for c in C.all_cells() if c["kind"] == "dirty")
_plan = C.eval_plan(_dirty)
assert len(_plan) == 2, _plan
assert [s for _, _, s in _plan] == ["dirty", "clean"], _plan
assert {t for _, t, _ in _plan} == {_dirty["source"]}, _plan
assert C.split_result_key(_dirty["source"] + C.CLEAN_TEST_SUFFIX) == (
    _dirty["source"], "clean")

rows = [
    row("main", "kaggle_v2", "yolo11n", 0, "kaggle_v2", 0.9),
    row("main", "kaggle_v2", "yolo11n", 0, "visual", 0.5),
    row("main", "kaggle_v2", "yolo11n", 0, "twoclass", 0.4),
    row("reference", "visual", "yolo11n", 0, "visual", 0.8),
    row("reference", "visual", "yolo11s", 0, "visual", 0.2),
    # A: downloaded weights on the downloaded test split
    row("dirty", "kaggle_v2", "yolo11n", 0, "kaggle_v2", 0.95,
        test_split="downloaded"),
    # B: the SAME weights on the audited test split
    row("dirty", "kaggle_v2", "yolo11n", 0, "kaggle_v2", 0.72,
        result_key="kaggle_v2" + C.CLEAN_TEST_SUFFIX),
]

ratios = transfer_rows(rows)
visual = next(r for r in ratios if r["target"] == "visual")
assert visual["image_ap_denominator"] == 0.8
assert visual["image_ap_ratio"] == 0.625

gaps = gap_rows(rows)
assert len(gaps) == 1
assert abs(gaps[0]["generalization_gap"] - 0.45) < 1e-12




# --- the three-number downloaded comparison -------------------------------
shift = dirty_shift_rows(rows)
assert len(shift) == 1, shift
s = shift[0]
assert s["A_image_ap"] == 0.95          # downloaded train, downloaded test
assert s["B_image_ap"] == 0.72          # same weights, audited test
assert s["C_image_ap"] == 0.9           # audited train, audited test
# Only the test set changes between A and B, so this isolates the grading
# partition from the training data.
assert abs(s["shift_test_set_only_image_ap"] - 0.23) < 1e-9
# A and C change both the training data and the test set.
assert abs(s["shift_full_pipeline_image_ap"] - 0.05) < 1e-9
assert "not a causal effect" in s["comparison"]

print("postprocess checks passed")
