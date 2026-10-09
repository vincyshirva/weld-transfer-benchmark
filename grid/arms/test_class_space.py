#!/usr/bin/env python3
"""Pin the arm B class space rule. Run with `python3 grid/arms/test_class_space.py`.

WHY THIS TEST EXISTS. The defect it guards against ran for 240 cells and two
endpoints without raising anything, because asking for a class index the model
cannot emit is a silent no-op. Nothing crashed, nothing warned, and one pair
returned a clean `map50: 0.0` that looked like a measurement. So the rule is
pinned here rather than left as a comment.

The expected table is transcribed from the 15 September 2026 review plan.
`confirm_class_space.pair_table()` derives the same table from `config.py`. If
the two ever disagree, either the configuration moved or the rule did, and
both deserve to fail loudly.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import config_arms as A  # noqa: E402

# IMPORT ORDER IS LOAD BEARING. `assert_main_unmoved` must run BEFORE
# `register`, because registering the arm datasets changes what
# `C.config_fingerprint()` returns inside this process. `evaluate_arm` does
# both, in that order, at import time. So import it first and do not call
# register() here. Doing it the other way round makes the guard fire on a
# fingerprint the test itself moved, which is a confusing way to learn that
# the guard works.
import evaluate_arm as EA  # noqa: E402
import confirm_class_space as CS  # noqa: E402

# (source, target) -> (filter used, correct filter, head nc, box level)
EXPECTED = {
    ("kaggle_v2", "weld_surface_v2"): ([2], [0, 1, 2, 4, 5, 6], 7, "skipped"),
    ("kaggle_v2", "lohi_weld"):       ([2], [0, 1, 2, 3],       4, "scored"),
    ("kaggle_v2", "visual"):          ([2], [0],                3, "scored"),
    ("kaggle_v2", "twoclass"):        ([2], [0, 1],             2, "scored"),
    ("lohi_weld", "weld_surface_v2"): ([0, 1, 2, 3], [0, 1, 2, 4, 5, 6], 7, "skipped"),
    ("lohi_weld", "kaggle_v2"):       ([0, 1, 2, 3], [2],       3, "scored"),
    ("lohi_weld", "visual"):          ([0, 1, 2, 3], [0],       3, "scored"),
    ("lohi_weld", "twoclass"):        ([0, 1, 2, 3], [0, 1],    2, "scored"),
}

# The one pair where the old source filter happened to be right. It asks for
# {0,1,2,3} against a two output head, and {0,1} is exactly twoclass's defect
# set. Kept explicit so nobody "fixes" it into the wrong list.
CORRECT_BY_COINCIDENCE = {("lohi_weld", "twoclass")}


def test_pair_table_matches_the_review() -> None:
    rows = {(r["source"], r["target"]): r for r in CS.pair_table()}
    assert set(rows) == set(EXPECTED), (
        f"pair set moved: {sorted(set(rows) ^ set(EXPECTED))}")
    for key, (used, correct, nc, box) in EXPECTED.items():
        r = rows[key]
        assert r["filter_used"] == used, (key, r["filter_used"], used)
        assert r["correct_filter"] == correct, (key, r["correct_filter"], correct)
        assert r["head_nc"] == nc, (key, r["head_nc"], nc)
        assert r["box_level"] == box, (key, r["box_level"], box)
    print(f"  PASS  all {len(EXPECTED)} arm B pairs match the reviewed table")


def test_counts_of_affected_cells() -> None:
    rows = CS.pair_table()
    wrong_img = [r for r in rows if r["image_level_wrong"]]
    wrong_box = [r for r in wrong_img if r["box_level"] == "scored"]
    per_pair = len(A.ARMB_FAMILIES) * len(A.ARMB_K) * len(A.ARMB_DRAWS)
    assert per_pair == 30, per_pair
    assert len(wrong_img) == 7, len(wrong_img)
    assert len(wrong_box) == 5, len(wrong_box)
    assert len(wrong_img) * per_pair == 210
    assert len(wrong_box) * per_pair == 150
    print("  PASS  210 cells wrong at image level, 150 at box level")


def test_coincidence_is_identified_not_assumed() -> None:
    rows = {(r["source"], r["target"]): r for r in CS.pair_table()}
    ok = {k for k, r in rows.items() if not r["image_level_wrong"]}
    assert ok == CORRECT_BY_COINCIDENCE, (
        f"the set of accidentally correct pairs moved: {sorted(ok)}")
    # And it is correct only AFTER intersecting with the head width.
    r = rows[("lohi_weld", "twoclass")]
    assert r["filter_used"] != r["correct_filter"], (
        "this pair must differ on the REQUESTED ids, otherwise the test is "
        "not checking what it claims")
    assert r["effective_filter"] == r["correct_filter"]
    print("  PASS  lohi_weld to twoclass is correct only after the nc clip")


def test_filter_rule_selects_by_arm() -> None:
    """The fix itself: arm B filters by TARGET, the other arms by SOURCE."""
    armb = {"arm": "armb", "source": "kaggle_v2", "target": "lohi_weld"}
    ctl = {"arm": "lohi_control", "source": "lohi_high", "target": None}
    armd = {"arm": "armd", "source": "pool_no_lohi_weld", "target": None}

    assert EA.prediction_class_space(armb) == "lohi_weld"
    assert EA.prediction_filter_ids(armb) == A.arm_defect_ids("lohi_weld")
    assert EA.prediction_filter_ids(armb) != A.arm_defect_ids("kaggle_v2")

    assert EA.prediction_class_space(ctl) == "lohi_high"
    assert EA.prediction_filter_ids(ctl) == A.arm_defect_ids("lohi_high")

    assert EA.prediction_class_space(armd) == "pool_no_lohi_weld"
    assert EA.prediction_filter_ids(armd) == A.arm_defect_ids("pool_no_lohi_weld")
    print("  PASS  the filter is chosen by arm, target for B, source otherwise")


def test_every_armb_pair_now_uses_the_correct_filter() -> None:
    for (source, target), (_, correct, nc, _) in EXPECTED.items():
        cell = {"arm": "armb", "source": source, "target": target}
        got = sorted(EA.prediction_filter_ids(cell))
        assert got == correct, (source, target, got, correct)
        EA.assert_ids_within_head(set(got), nc, f"{source} to {target}")
    print(f"  PASS  all {len(EXPECTED)} pairs now filter in the target space")


def test_out_of_range_ids_are_rejected() -> None:
    """Silent emptiness is what let this run for 240 cells. Now it raises."""
    try:
        EA.assert_ids_within_head({2}, 2, "kaggle_v2 to twoclass, old filter")
    except SystemExit:
        pass
    else:
        raise AssertionError("an id at or above nc must fail loudly")
    EA.assert_ids_within_head({0, 1}, 2, "ok")
    print("  PASS  an id at or above the head width fails loudly")


if __name__ == "__main__":
    print("arm B class space tests")
    test_pair_table_matches_the_review()
    test_counts_of_affected_cells()
    test_coincidence_is_identified_not_assumed()
    test_filter_rule_selects_by_arm()
    test_every_armb_pair_now_uses_the_correct_filter()
    test_out_of_range_ids_are_rejected()
    print("all class space tests passed")
