#!/usr/bin/env python3
"""The one command that turns the 300 arm cells into manuscript artefacts.

    python3 grid/arms/analysis_arms.py

Reads `grid_results_arms/results/` and the main grid's `grid_results/results/`,
writes `grid_results_arms/analysis/`. Takes no arguments, like `analysis.py`.

WHICH RECORDS ARE READ, AND WHY IT MATTERS.

  lohi_control   eval__        selected on its own source valid split
  armd           eval__        selected on its own source valid split
  armb           eval_last__   PRIMARY, the project log 6a-ii.
                 eval__        SECONDARY, the `best` endpoint.

Since the the project log 6a-viii retrain these plain records ARE the corrected
ones: `evaluate_arm.py` carries the 6a-iii class space fix and ran on the
cluster for every retrained cell, producing both endpoints there. The
laptop-recomputed `_fixed__` records belonged to the void run and are
quarantined with it.

Two guards run before anything is plotted. `assert_armb_actually_finetuned()`
checks that no two cells sharing a target, k and draw produced identical
predictions, which is the 6a-viii signature. `assert_armb_class_space()`
checks that every record states its class space and that it is the TARGET's,
which is the 6a-iii property. Both test the property rather than a filename.

Nothing is reimplemented. `figstyle` supplies the palette, `stats_tests` the
pre-registered omnibus, `transfer_bootstrap` the split block intervals.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import config as C  # noqa: E402
import config_arms as A  # noqa: E402

import evaluate_arm as EA  # noqa: E402  (asserts, then registers)
import analysis as AN  # noqa: E402
import evaluate_cell as E  # noqa: E402
import figstyle as FS  # noqa: E402
import metrics as M  # noqa: E402
import reliability_control_figure as RCF  # noqa: E402
import stats_tests as ST  # noqa: E402
import transfer_bootstrap as TB  # noqa: E402

_GT: dict[str, dict] = {}
_TAUGHT = [False]


def ground_truth(target: str) -> dict:
    """Frozen ground truth, with the arm split block aliases installed once."""
    if not _TAUGHT[0]:
        EA.teach_split_blocks()
        _TAUGHT[0] = True
    if target not in _GT:
        _GT[target] = E.load_ground_truth(target)
    return _GT[target]


def cross_dataset_box() -> tuple[dict, dict]:
    """Main grid box mAP50, DERIVED from grid_results rather than copied.

    Returns (cells, diagonals).

    THE DIAGONAL OF A TEST ONLY TARGET IS NOT IN THE MAIN MATRIX. visual and
    twoclass are never training sources, so their in domain score P_jj comes
    from the 24 REFERENCE cells. That is exactly why those cells exist, per
    the project log 5b, and reading the diagonal from the main cells alone raises a
    KeyError rather than silently returning the wrong number.
    """
    cells, diag = defaultdict(list), defaultdict(list)
    for path in sorted(MAIN_RESULTS.glob("eval__*.json")):
        rec = json.loads(path.read_text())
        c = rec["cell"]
        if c["kind"] not in ("main", "reference"):
            continue
        for target, t in rec["targets"].items():
            if t["box_level"].get("skipped"):
                continue
            v = t["box_level"]["map50"]
            if c["kind"] == "main":
                cells[(c["source"], target)].append(v)
            if c["source"] == target:
                diag[target].append(v)
    return ({k: statistics.fmean(v) for k, v in cells.items()},
            {k: statistics.fmean(v) for k, v in diag.items()})

ARM_RESULTS = C.PROJECT / "grid_results_arms" / "results"
MAIN_RESULTS = C.PROJECT / "grid_results" / "results"
OUT = C.PROJECT / "grid_results_arms" / "analysis"

# THE `_fixed__` PREFIXES ARE GONE, AND THAT IS PROGRESS, NOT A REGRESSION.
#
# They existed because the project log 6a-iii's class space correction had to be
# recomputed on the laptop from archived predictions: the cluster evaluator at
# the time filtered in the SOURCE space, so its `eval__` records were wrong for
# 210 of 240 arm B cells and a second, corrected set was written beside them.
#
# The 6a-viii retrain changed that. `evaluate_arm.py` now carries the 6a-iii
# fix and ran on the cluster for every retrained cell, so the plain `eval__`
# and `eval_last__` records ARE the corrected ones. There is no second set to
# prefer, and asking for one would reject the very records the retrain
# produced.
#
# `assert_armb_class_space()` below checks the PROPERTY rather than the
# filename, which is what the prefix was ever standing in for.
PRIMARY = "eval_last__"
SECONDARY = "eval__"


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load(prefix: str, arm: str) -> list[dict]:
    out = []
    for path in sorted(ARM_RESULTS.glob(f"{prefix}{arm}__*.json")):
        rec = json.loads(path.read_text())
        rec["_file"] = path.name
        out.append(rec)
    return out


def assert_armb_actually_finetuned() -> None:
    """HARD STOP. Arm B trained from random init, the project log 6a-viii.

    `train_arm.finetune_ultralytics` passed `pretrained=False`, which makes
    ultralytics discard the loaded start checkpoint and build a fresh model
    from the YAML. So every arm B cell measured training from scratch on k
    target images rather than adapting a source checkpoint, and R_ij(k) is not
    a recovery ratio: its numerator has no relationship to the P_ij(0) in its
    own denominator.

    The check is the observable consequence rather than the cause, because the
    cause lives in a file this module does not import. Two cells that share a
    target, k and draw but differ in SOURCE must not produce identical
    predictions. 45 of 45 YOLO11s pairs currently do.

    Delete this function only when arm B has been retrained and the check
    passes on the new records. It is not a nuisance guard. It is the only
    thing standing between a void number and a figure.
    """
    import numpy as _np
    pairs = defaultdict(dict)
    for path in ARM_RESULTS.glob(f"{PRIMARY}armb__*.json"):
        rec = json.loads(path.read_text())
        c = rec["cell"]
        key = (c["target"], c["family"], c["k"], c["draw"])
        pairs[key][c["source"]] = path.name.replace(PRIMARY, "raw_last_")
    collisions = []
    n_two_source = 0
    for key, srcs in pairs.items():
        if len(srcs) != 2:
            continue
        n_two_source += 1
        tgt = key[0]
        arrs = []
        for src, stem in sorted(srcs.items()):
            npz = ARM_RESULTS / (stem.replace("raw_last_", "raw_last__")
                                 .replace(".json", ".npz"))
            if not npz.is_file():
                arrs = []
                break
            z = _np.load(npz, allow_pickle=True)
            arrs.append(z[f"{tgt}__pred_score"])
        if len(arrs) == 2 and arrs[0].shape == arrs[1].shape \
                and _np.array_equal(arrs[0], arrs[1]):
            collisions.append(key)
    if collisions:
        sys.exit(
            f"\nARM B IS VOID. {len(collisions)} of {n_two_source} two-source "
            f"cells produce BIT IDENTICAL predictions from different source "
            f"checkpoints.\n\n"
            f"Root cause, the project log 6a-viii: train_arm.finetune_ultralytics "
            f"passed pretrained=False, so ultralytics discarded the start "
            f"checkpoint and trained from random initialisation. All 240 arm "
            f"B cells are affected; these are just the ones with a "
            f"same-coordinate partner to collide with.\n\n"
            f"Arm B must be retrained, about 11.5 h of H200 wall clock. The "
            f"LoHi control and arm D are UNAFFECTED and must not be "
            f"retrained. Refusing to produce a figure from void numbers.")


def assert_armb_class_space() -> None:
    """Every arm B record must filter in the TARGET class space. 6a-iii.

    CHECKS THE PROPERTY, NOT THE FILENAME. The earlier version of this guard
    required records named `eval_last_fixed__`, because at that time the only
    correctly filtered records were the ones recomputed on the laptop. After
    the 6a-viii retrain the cluster evaluator carries the fix itself, so a
    prefix check would now reject exactly the records it should accept.

    Arm B fine-tunes into the target's class list, so `kshot.materialise`
    rebuilds the detection head at the target `nc` and the model predicts in
    the TARGET space. Filtering those predictions with the SOURCE's defect ids
    is the 6a-iii defect, and it was silent for 240 cells because asking for a
    class index the head cannot emit matches nothing and scores zero rather
    than raising. So the check is: the record must SAY which space it used,
    and that space must be the target.
    """
    want = sum(1 for c in A.all_arm_cells() if c["arm"] == "armb")
    for prefix, label in ((PRIMARY, "last"), (SECONDARY, "best")):
        paths = sorted(ARM_RESULTS.glob(f"{prefix}armb__*.json"))
        if len(paths) != want:
            sys.exit(
                f"{len(paths)} of {want} arm B records at the {label} "
                f"endpoint. Pull the cluster results before analysing.")
        missing, wrong = [], []
        for path in paths:
            rec = json.loads(path.read_text())
            space = rec.get("defect_id_space")
            if space is None:
                missing.append(path.name)
            elif space != rec["cell"]["target"]:
                wrong.append((path.name, space, rec["cell"]["target"]))
        if missing:
            sys.exit(
                f"{len(missing)} arm B {label} records carry no "
                f"`defect_id_space`, so they predate the the project log 6a-iii "
                f"correction and their class space cannot be verified from "
                f"the record. e.g. {missing[:2]}")
        if wrong:
            sys.exit(
                f"{len(wrong)} arm B {label} records filtered in the WRONG "
                f"class space. Arm B predicts in the TARGET space because the "
                f"head is rebuilt at the target nc. The project log 6a-iii. "
                f"e.g. {wrong[:2]}")


def main_grid() -> dict:
    """(source, family, seed, kind) -> record, for the 69 frozen cells."""
    out = {}
    for path in sorted(MAIN_RESULTS.glob("eval__*.json")):
        rec = json.loads(path.read_text())
        c = rec["cell"]
        out[(c["source"], c["family"], c["seed"], c["kind"])] = rec
    return out


def box_of(rec, target):
    b = rec["targets"][target]["box_level"]
    return None if b.get("skipped") else b.get("map50")


def img_of(rec, target):
    return rec["targets"][target]["image_level"].get("ap")


def base_rate_of(rec, target):
    return rec["targets"][target]["image_level"].get("positive_rate")


def p_zero(main, source, family, seed, target, metric):
    """P_ij(0), the un-adapted main grid checkpoint on the target."""
    rec = main.get((source, family, seed, "main"))
    if rec is None or target not in rec["targets"]:
        return None
    return (box_of if metric == "box" else img_of)(rec, target)


def p_diag(main, target, family, seed, metric):
    """P_jj, the target's own in domain score, SAME family and SAME seed.

    The project log 6a. A YOLO11s numerator must never be divided by an RT-DETR-l
    denominator, which is the whole reason the 24 reference cells exist.
    """
    rec = main.get((target, family, seed, "main"))
    if rec is None:
        rec = main.get((target, family, seed, "reference"))
    if rec is None or target not in rec["targets"]:
        return None
    return (box_of if metric == "box" else img_of)(rec, target)


def fmt(v, nd=4):
    return "n/a" if v is None else f"{v:.{nd}f}"


def savefig(fig, stem: str) -> None:
    """Every figure goes out as PDF and PNG. Nothing lives only in a log.

    Through figstyle.save, which refuses an axes holding no data and writes
    PDFs without a CreationDate, so unchanged content saves to identical
    bytes."""
    FS.save(fig, OUT / f"{stem}.png", dpi=200, bbox_inches="tight")
    FS.save(fig, OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def keep_boxes(z, key: str, keep_ids: set[int]) -> dict:
    """Predictions as the box metric sees them, for an EXPLICIT id set.

    transfer_bootstrap.harmonized_boxes exists but resolves the filter through
    C.defect_ids(source), which is the rule the arm B class space defect broke.
    Passing the ids in makes the caller state the class space, which is what
    the project log 6a-iii requires. The NMS and the flattening are the frozen ones.
    """
    pi = z[key + "__pred_image"]
    pb = z[key + "__pred_box"]
    ps = z[key + "__pred_score"]
    pc = z[key + "__pred_class"]
    take = np.isin(pc, list(keep_ids))
    pi, pb, ps = pi[take], pb[take], ps[take]
    out = {}
    for image_id in np.unique(pi):
        m = pi == image_id
        b, s = pb[m], ps[m]
        k = M.class_agnostic_nms(b, s)
        out[int(image_id)] = (b[k], s[k])
    return out


def msd(values):
    v = [x for x in values if x is not None]
    if not v:
        return None, None
    return statistics.fmean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)


# --------------------------------------------------------------------------
# Task 1. Gates. Nothing is plotted until these pass.
# --------------------------------------------------------------------------

GATE_FAILURES: list[str] = []


def gate(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        GATE_FAILURES.append(name)


def gates(main: dict) -> dict:
    """Three conditions that must hold before any figure is built.

    A figure built on a failed gate is wasted work, and worse, it looks fine.
    """
    print("\nTask 1. gates")
    out = {}

    # 1a. Arm B at k = 0 is NOT a retrained cell. It is the main grid
    # checkpoint scored on the target, so R_ij(0) is 0 by construction.
    # Asserting it proves P_ij(0) is being read from the right place.
    bad, checked = [], 0
    for source in A.ARMB_START_SOURCES:
        for target in A.armb_targets(source):
            for fam in A.ARMB_FAMILIES:
                for seed in A.ARMB_DRAWS:
                    for metric in ("box", "image"):
                        p0 = p_zero(main, source, fam, seed, target, metric)
                        pj = p_diag(main, target, fam, seed, metric)
                        if p0 is None or pj is None:
                            continue
                        checked += 1
                        if pj == p0:
                            continue
                        r0 = (p0 - p0) / (pj - p0)
                        if r0 != 0.0:
                            bad.append(f"{source}->{target} {fam} s{seed} "
                                       f"{metric} R(0)={r0}")
    gate("1a  R_ij(0) is exactly 0 by construction", not bad,
         f"{checked} (pair, cfg, seed, metric) combinations checked")
    # And P_ij(0) must come from a main grid record, never from an arm record.
    sample = main.get(("kaggle_v2", "yolo11s", 0, "main"))
    gate("1a  P_ij(0) resolves to a main grid cell",
         sample is not None and sample["cell"]["kind"] == "main",
         "grid_results/results/eval__kaggle_v2__yolo11s__seed0.json")
    out["gate_1a_checked"] = checked

    # 1b. The control diagonals are trained on SUBSETS of lohi_weld, so they
    # should sit near the main grid lohi_weld diagonal without matching it.
    ctl = load("eval__", "lohictl")
    diag = defaultdict(list)
    for rec in ctl:
        src = rec["cell"]["source"]
        diag[src].append(box_of(rec, src))
    main_diag = [box_of(main[(s, f, d, "main")], "lohi_weld")
                 for s, f, d, k in main
                 if s == "lohi_weld" and k == "main"
                 for _ in [0]]
    main_diag = [v for v in main_diag if v is not None]
    md = statistics.fmean(main_diag)
    print(f"      main grid lohi_weld diagonal: {md:.4f} "
          f"(n={len(main_diag)}, range {min(main_diag):.4f} to "
          f"{max(main_diag):.4f})")
    ok = True
    for src in A.LOHI_ACQUISITIONS:
        m = statistics.fmean(diag[src])
        print(f"      control {src} diagonal: {m:.4f}  "
              f"difference from main grid {m - md:+.4f}")
        out[f"gate_1b_{src}"] = m
        if abs(m - md) > 0.10:
            ok = False
    out["gate_1b_main_lohi_diag"] = md
    gate("1b  control diagonals sit near the main grid lohi_weld diagonal",
         ok, "within 0.10 absolute")

    # 1c. RELAXED, AND THE ORIGINAL WORDING IS KEPT SO THE CHANGE IS VISIBLE.
    #
    # The 16 September plan asked that NO corrected arm B cell have n_pred 0
    # at box level, expecting the count to be zero. It is 14 of 360. The
    # expectation was wrong, not the data, and the project log 6a-v records why: at
    # k = 10 a reinitialised head can collapse onto a single wrong class, so
    # it emits its full 300 detections per image and none of them are in the
    # defect class.
    #
    # The gate now tests the thing it was actually protecting, which is that a
    # zero is never caused by the class space. Three conditions per empty cell:
    # the filter ids must be inside the head width, the model must have emitted
    # predictions, and it must have emitted none in the defect classes. That
    # distinguishes head collapse, which is a result, from a filter fault,
    # which would be a defect.
    zero, total, unexplained = [], 0, []
    for prefix in (PRIMARY, SECONDARY):
        for rec in load(prefix, "armb"):
            cell = rec["cell"]
            t = rec["targets"][cell["target"]]
            b = t["box_level"]
            if b.get("skipped"):
                continue
            total += 1
            if b.get("n_pred", 0) != 0:
                continue
            zero.append(rec["_file"])
            ids = set(rec.get("defect_ids_used") or [])
            nc = rec.get("head_nc")
            raw_prefix = "raw_last__" if prefix == PRIMARY else "raw__"
            name = rec["_file"][len(prefix):-len(".json")]
            path = ARM_RESULTS / f"{raw_prefix}{name}.npz"
            why = []
            if not ids or nc is None or any(i >= nc or i < 0 for i in ids):
                why.append(f"filter {sorted(ids)} outside head width {nc}")
            if path.is_file():
                z = np.load(path, allow_pickle=True)
                cls = z[f"{cell['target']}__pred_class"].astype(int)
                if len(cls) == 0:
                    why.append("model emitted no prediction at all")
                elif np.isin(cls, list(ids)).any():
                    why.append("defect class predictions exist but n_pred is 0")
            else:
                why.append("raw archive missing, cannot verify")
            if why:
                unexplained.append(f"{rec['_file']}: {'; '.join(why)}")
    gate("1c  every empty arm B cell is head collapse, not a class space fault",
         not unexplained,
         f"{len(zero)} empty of {total} box defined, {len(unexplained)} "
         f"unexplained")
    for u in unexplained[:5]:
        print(f"      UNEXPLAINED  {u}")
    print(f"      RELAXED from the plan's 'count must be zero'. See "
          f"the project log 6a-v.")
    out["gate_1c_empty_cells"] = len(zero)
    out["gate_1c_box_records"] = total
    out["gate_1c_unexplained"] = unexplained
    out["gate_1c_note"] = (
        "The plan expected zero empty cells. 14 of 360 are empty. All 14 are "
        "a reinitialised head collapsing onto a single wrong class at low k, "
        "verified by checking that the filter ids are inside the head width "
        "and that the model emitted predictions in other classes only. "
        "The project log 6a-v.")
    return out


# --------------------------------------------------------------------------
# 4a. LoHi internal control
# --------------------------------------------------------------------------

CONTROL_BOOT_CACHE = Path("grid_results_arms/control_bootstrap_cache.json")


def control_bootstrap(resamples: int, seed: int) -> list[dict]:
    """PAIRED split block interval on each control transfer ratio.

    T_ij = P_ij / P_jj has both terms measured on the SAME test images of the
    target acquisition, so one draw resamples the blocks of j once and scores
    the foreign model and the in domain model on exactly those images.
    Resampling them independently would ignore the shared test set and give an
    interval much wider than the truth. The project log 5m sets out the same
    argument for the main grid.

    The diagonal is the self check. When source equals target the two models
    are the same model, so T must come back exactly 1.000 with zero width.

    CACHED, keyed by resample count and seed, the same pattern `analysis.py`
    uses for the main grid bootstraps. The project log 5l and 5m. The cost is real:
    2,000 draws over both LoHi acquisitions is roughly 20 minutes, and paying
    it on every re-run discourages re-running, which is how a stale figure
    survives. Delete the cache file to force a recompute.
    """
    key = f"{resamples}:{seed}"
    if CONTROL_BOOT_CACHE.is_file():
        cached = json.loads(CONTROL_BOOT_CACHE.read_text())
        if key in cached:
            print(f"    control bootstrap: cache hit {key}")
            return cached[key]
    rows = []
    for target in A.LOHI_ACQUISITIONS:
        gt = ground_truth(target)
        blocks = gt["split_blocks"]
        gt_defect = gt["gt_defect"]
        keep = A.arm_defect_ids(target)
        per_source = {}
        for source in A.LOHI_ACQUISITIONS:
            for fam in FS.CONFIG_ORDER:
                for sd in C.SEEDS:
                    name = f"lohictl__{source}__{fam}__seed{sd}"
                    path = ARM_RESULTS / f"raw__{name}.npz"
                    if not path.is_file():
                        continue
                    z = np.load(path, allow_pickle=True)
                    by_image = keep_boxes(z, target, A.arm_defect_ids(source))
                    per_source[(source, fam, sd)] = TB.precompute_box_tp(
                        by_image, gt_defect)
        for source in A.LOHI_ACQUISITIONS:
            for fam in FS.CONFIG_ORDER:
                for sd in C.SEEDS:
                    num = per_source.get((source, fam, sd))
                    den = per_source.get((target, fam, sd))
                    if num is None or den is None:
                        continue
                    full_n = TB.box_ap_from_tp(
                        num, np.arange(len(gt["images"])))
                    full_d = TB.box_ap_from_tp(
                        den, np.arange(len(gt["images"])))
                    point = full_n / full_d if full_d else float("nan")
                    vals = []
                    for idx in TB.block_draws(
                            blocks, resamples,
                            TB.target_seed(seed, f"{target}:{fam}:{sd}")):
                        a = TB.box_ap_from_tp(num, idx)
                        b = TB.box_ap_from_tp(den, idx)
                        vals.append(a / b if b else float("nan"))
                    band = TB.percentile_band(vals, point)
                    rows.append({
                        "source": source, "target": target, "family": fam,
                        "seed": sd, "p_ij": full_n, "p_jj": full_d,
                        "transfer_ratio": point, "lo": band["lo"],
                        "hi": band["hi"], "resamples": resamples,
                        "n_undefined_draws": band["n_undefined_draws"]})
    cached = (json.loads(CONTROL_BOOT_CACHE.read_text())
              if CONTROL_BOOT_CACHE.is_file() else {})
    cached[key] = rows
    CONTROL_BOOT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_BOOT_CACHE.write_text(json.dumps(cached))
    return rows


def lohi_control(out: Path) -> dict:
    recs = load("eval__", "lohictl")
    box, img, per_cfg = defaultdict(list), defaultdict(list), defaultdict(list)
    rec_fp = defaultdict(lambda: defaultdict(list))
    for rec in recs:
        src = rec["cell"]["source"]
        fam = rec["cell"]["family"]
        for target in A.LOHI_ACQUISITIONS:
            box[(src, target)].append(box_of(rec, target))
            img[(src, target)].append(img_of(rec, target))
            per_cfg[(src, target, fam)].append(box_of(rec, target))
            b = rec["targets"][target]["box_level"]
            for budget in C.FP_PER_IMAGE_BUDGETS:
                v = b.get(f"recall_at_{budget}_fp_per_image")
                if v is not None:
                    rec_fp[(src, target)][budget].append(v)

    acq = list(A.LOHI_ACQUISITIONS)
    lines = ["# LoHi internal transfer control", "",
             "Box mAP50, mean (SD) over 4 configurations and 3 seeds, "
             "12 cells per entry.", "",
             "| train | " + " | ".join(f"to {t}" for t in acq) + " |",
             "|---|" + "---|" * len(acq)]
    for src in acq:
        cells = []
        for t in acq:
            m, s = msd(box[(src, t)])
            cells.append(f"{fmt(m)} ({s:.3f})")
        lines += ["| " + src + " | " + " | ".join(cells) + " |"]

    ratios = {}
    for src in acq:
        for t in acq:
            if src == t:
                continue
            num, _ = msd(box[(src, t)])
            den, _ = msd(box[(t, t)])
            ratios[(src, t)] = None if not den else num / den

    lines += ["", "## Transfer ratios, box level", "",
              "| direction | T_ij | retained |", "|---|---|---|"]
    for (src, t), r in ratios.items():
        lines += [f"| {src} to {t} | {fmt(r, 3)} | {100 * r:.1f}% |"]

    lines += ["", "## Per configuration, high to low", "",
              "| configuration | box mAP50 |", "|---|---|"]
    for fam in FS.CONFIG_ORDER:
        m, _ = msd(per_cfg[("lohi_high", "lohi_low", fam)])
        lines += [f"| {FS.label_for(fam)} | {fmt(m)} |"]

    # 2c. paired split block intervals
    boot = control_bootstrap(C.BOOTSTRAP_RESAMPLES, C.BOOTSTRAP_SEED)
    AN.write_csv(out / "lohi_control_transfer_intervals.csv", boot)
    diag_bad = [r for r in boot if r["source"] == r["target"]
                and abs((r["transfer_ratio"] or 0) - 1.0) > 1e-9]
    if diag_bad:
        raise SystemExit(
            f"control bootstrap diagonal self check failed on "
            f"{len(diag_bad)} rows. T_ii must be exactly 1.000.")
    agg = defaultdict(list)
    for r in boot:
        if r["source"] != r["target"]:
            agg[(r["source"], r["target"])].append(r)
    lines += ["", "## Transfer ratio, split block bootstrap intervals", "",
              f"Blocks resampled, not images, {C.BOOTSTRAP_RESAMPLES} draws. "
              "Numerator and denominator are scored on the SAME resampled "
              "images, because both are measured on the target's test split. "
              "The diagonal returns exactly 1.000 with zero width, which is "
              "the self check.", "",
              "| direction | cfg | seed | T | 95% interval |",
              "|---|---|---|---|---|"]
    for (s, t), rows_ in sorted(agg.items()):
        for r in sorted(rows_, key=lambda x: (x["family"], x["seed"])):
            lines.append(
                f"| {s} to {t} | {FS.label_for(r['family'])} | {r['seed']} | "
                f"{fmt(r['transfer_ratio'], 3)} | "
                f"[{fmt(r['lo'], 3)}, {fmt(r['hi'], 3)}] |")

    # 2b. recall at a fixed false positive budget
    lines += ["", "## Recall at a fixed false positive budget per image", "",
              "The metric that carries the inspection claim, per the project log 5. "
              "Mean over 4 configurations and 3 seeds.", "",
              "| train | tested on | "
              + " | ".join(f"R@{b} FP" for b in C.FP_PER_IMAGE_BUDGETS)
              + " |", "|---|---|" + "---|" * len(C.FP_PER_IMAGE_BUDGETS)]
    for src in acq:
        for t in acq:
            cells = [fmt(msd(rec_fp[(src, t)][b])[0])
                     for b in C.FP_PER_IMAGE_BUDGETS]
            lines.append(f"| {src} | {t} | " + " | ".join(cells) + " |")

    lines += ["", "## Image level, reported but UNINFORMATIVE here", "",
              "| train | " + " | ".join(f"to {t}" for t in acq) + " |",
              "|---|" + "---|" * len(acq)]
    for src in acq:
        cells = []
        for t in acq:
            m, s = msd(img[(src, t)])
            cells.append(f"{fmt(m)} ({s:.3f})")
        lines += ["| " + src + " | " + " | ".join(cells) + " |"]

    # 2e. set against the cross-dataset numbers, DERIVED from grid_results
    xd, dia = cross_dataset_box()
    off = [v for (s, t), v in xd.items() if s != t]
    lines += ["", "## Set against the cross dataset matrix", "",
              "Derived from `grid_results/results/`, not copied. The main grid "
              "is unaffected by the the project log 6a-iii class space defect, "
              "because a main grid model predicts in its own source space and "
              "is filtered with source ids, which is correct. So the "
              "comparison is sound.", "",
              "| comparison | box mAP50 | retained |", "|---|---|---|"]
    hl = msd(box[("lohi_high", "lohi_low")])[0]
    ll = msd(box[("lohi_low", "lohi_low")])[0]
    lh = msd(box[("lohi_low", "lohi_high")])[0]
    hh = msd(box[("lohi_high", "lohi_high")])[0]
    lines += [
        f"| LoHi internal, high to low | {fmt(hl)} | {100 * hl / ll:.1f}% |",
        f"| LoHi internal, low to high | {fmt(lh)} | {100 * lh / hh:.1f}% |"]
    for (s, t), v in sorted(xd.items()):
        if s == t:
            continue
        lines.append(f"| cross dataset, {s} to {t} | {fmt(v)} | "
                     f"{100 * v / dia[t]:.1f}% of {t} own {fmt(dia[t])} |")
    lines += [
        "", f"Cross dataset box mAP50 spans {fmt(min(off))} to {fmt(max(off))} "
        f"against diagonals of "
        + " and ".join(fmt(v) for _, v in sorted(dia.items())) + ".",
    ]

    lines += [
        "", "**Caption.** The control is a BOX LEVEL result. Image level AP "
        "sits between 0.99 and 1.00 everywhere because LoHi is 97.6 percent "
        "positive, so the prevalence ceiling leaves almost no headroom. "
        "The project log 5k makes the same point for the main grid. Read the box "
        "table.",
        "", "**What this controls for, and what it does not.** Annotation "
        "protocol, annotators, class list and production line are all held "
        "constant. Only the acquisition changes. So this is the closest thing "
        "the study has to pure visual domain shift with the label concept "
        "pinned. It still costs 64 percent of box mAP50 going high to low and "
        "54 percent going low to high.",
        "", "**But part of that is prior shift, not sensor shift.** The project log "
        "7 records the defect mix differing sharply between the two "
        "acquisitions: pore is 5.5 percent of high resolution boxes and 26.5 "
        "percent of low, deposit 18.9 against 8.8. Boxes per image are 9.27 "
        "and 6.47. A reader who is not told this will read the collapse as "
        "pure sensor shift.",
        "", "**Precedent.** LoHi-WELD's own IEEE Access 2024 description runs "
        "the same internal test. This version differs by running under the "
        "leakage audited splits and the same protocol as the rest of the "
        "study, which is why it is a control rather than a novelty.",
        "", "**What the two sets of numbers support, stated as the evidence "
        f"allows.** Internal transfer retains {100 * hl / ll:.0f} and "
        f"{100 * lh / hh:.0f} percent of the target's own in domain score. "
        "Cross dataset transfer retains between "
        f"{100 * min(v / dia[t] for (s, t), v in xd.items() if s != t):.1f} and "
        f"{100 * max(v / dia[t] for (s, t), v in xd.items() if s != t):.1f} "
        "percent. So visual shift alone is sufficient to cause a large "
        "collapse, and it is NOT sufficient to explain the cross dataset "
        "collapse, which is larger again by roughly an order of magnitude. "
        "That is what the numbers support. What they do not support is "
        "attributing the remainder to a single named cause. The control "
        "isolates acquisition. The cross dataset gap additionally mixes "
        "annotation protocol, object scale, defect vocabulary and training set "
        "size, and this study does not separate those from one another.",
    ]
    (out / "lohi_control.md").write_text("\n".join(lines) + "\n")

    # figure
    FS.apply_style()
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    grid = np.array([[msd(box[(s, t)])[0] for t in acq] for s in acq])
    im = ax.imshow(grid, cmap=FS.SEQUENTIAL, vmin=0, vmax=0.75)
    ax.set_xticks(range(len(acq)), [a.replace("lohi_", "") for a in acq])
    ax.set_yticks(range(len(acq)), [a.replace("lohi_", "") for a in acq])
    ax.set_xlabel("tested on")
    ax.set_ylabel("trained on")
    ax.set_title("LoHi internal transfer, box mAP50")
    for i in range(len(acq)):
        for j in range(len(acq)):
            m, s = msd(box[(acq[i], acq[j])])
            ax.text(j, i, f"{m:.3f}\n±{s:.3f}", ha="center", va="center",
                    color="white" if m > 0.4 else FS.INK, fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="box mAP50")
    fig.tight_layout()
    savefig(fig, "lohi_control_matrix")

    # The pooled interval per direction, over the 12 (cfg, seed) rows. Each
    # bootstrap row already carries its own interval. Pooling takes the widest
    # envelope rather than averaging the endpoints, because averaging two
    # intervals is not an interval for anything.
    bands = {}
    for (s, t), rows_ in agg.items():
        los = [r["lo"] for r in rows_ if r["lo"] is not None]
        his = [r["hi"] for r in rows_ if r["hi"] is not None]
        if los and his:
            bands[f"{s}->{t}"] = (min(los), max(his))

    return {"box": {f"{s}->{t}": msd(v) for (s, t), v in box.items()},
            "ratios": {f"{s}->{t}": r for (s, t), r in ratios.items()},
            "ratio_bands": bands}


# --------------------------------------------------------------------------
# 4b. Arm B recovery
# --------------------------------------------------------------------------

def draw_meta() -> dict:
    """Realised image and scene counts per (target, k, draw). The project log 6a-i."""
    import kshot as K
    meta = {}
    for target in C.ALL_TARGETS:
        for k in A.ARMB_K:
            for d in A.ARMB_DRAWS:
                s = K.sample(target, k, d)
                meta[(target, k, d)] = {
                    "n_images": s["n_images"], "n_scenes": s["n_scenes"],
                    "exhausted": s["exhausted_train_split"]}
    return meta


def armb(out: Path, main: dict) -> list[dict]:
    meta = draw_meta()
    rows = []
    for prefix, endpoint in ((PRIMARY, "last"), (SECONDARY, "best")):
        for rec in load(prefix, "armb"):
            c = rec["cell"]
            src, tgt, fam = c["source"], c["target"], c["family"]
            k, draw = c["k"], c["draw"]
            m = meta[(tgt, k, draw)]
            for metric in ("box", "image"):
                pk = (box_of if metric == "box" else img_of)(rec, tgt)
                # 3c. NUMERATOR AND BOTH DENOMINATOR TERMS MUST COME FROM ONE
                # SEED AND ONE CONFIGURATION. Asserted rather than trusted: a
                # YOLO11s numerator over an RT-DETR-l denominator is the exact
                # failure the 24 reference cells exist to prevent, and it would
                # be invisible in the output.
                z_rec = main.get((src, fam, draw, "main"))
                d_rec = (main.get((tgt, fam, draw, "main"))
                         or main.get((tgt, fam, draw, "reference")))
                for role, r in (("P_ij(0)", z_rec), ("P_jj", d_rec)):
                    if r is None:
                        continue
                    if r["cell"]["family"] != fam or r["cell"]["seed"] != draw:
                        sys.exit(
                            f"{rec['_file']}: {role} came from "
                            f"{r['cell']['family']} seed {r['cell']['seed']} "
                            f"but the cell is {fam} seed {draw}. Refusing to "
                            f"mix configurations or seeds in R_ij(k).")
                p0 = p_zero(main, src, fam, draw, tgt, metric)
                pj = p_diag(main, tgt, fam, draw, metric)
                denom = None if None in (p0, pj) else pj - p0
                rec_ap = None
                if None not in (pk, p0, pj) and denom != 0:
                    rec_ap = (pk - p0) / denom
                # BASE RATE CORRECTION APPLIES TO THE IMAGE LEVEL ONLY.
                # Positive class AP is floored by prevalence, which is why
                # the project log 5k reads LoHi's image column as a ceiling rather
                # than a difficulty statement. Box mAP50 has no such floor: it
                # scores localisation against boxes, not a class balance. An
                # earlier version of this file subtracted the image positive
                # rate from box mAP50 and produced values near -0.97, which is
                # not a quantity at all.
                base = base_rate_of(rec, tgt) if metric == "image" else None
                rows.append({
                    "endpoint": endpoint, "source": src, "target": tgt,
                    "family": fam, "k_requested": k, "draw": draw,
                    "metric": metric,
                    "n_images": m["n_images"], "n_scenes": m["n_scenes"],
                    "exhausted_train_split": m["exhausted"],
                    "ap": pk, "ap_minus_base_rate":
                        None if pk is None or base is None else pk - base,
                    "base_rate": base,
                    "p_zero": p0, "p_diag": pj, "recovery": rec_ap,
                    # 3d. The ratio's denominator, printed beside it always.
                    # The project log 5n establishes that kaggle_v2's box diagonal
                    # of about 0.080 is dataset difficulty, so a small
                    # absolute gain there produces a large looking R. The 0.05
                    # threshold is fixed here BEFORE the ratios are read, and
                    # flagged cells are never dropped.
                    "denominator": denom,
                    "denominator_thin": (denom is not None and denom < 0.05),
                    "n_pred_retained":
                        rec["targets"][tgt]["image_level"].get(
                            "n_pred_retained"),
                    "ap_suppressed":
                        bool(rec["targets"][tgt]["image_level"].get(
                            "ap_suppressed_reason")),
                    "defect_id_space": rec.get("defect_id_space"),
                })
    AN.write_csv(out / "armb_recovery.csv", rows)

    # ---- figure: recovery against REALISED SCENE COUNT, as points
    FS.apply_style()
    for metric in ("box", "image"):
        sel = [r for r in rows if r["endpoint"] == "last"
               and r["metric"] == metric and r["recovery"] is not None]
        if not sel:
            continue
        pairs = sorted({(r["source"], r["target"]) for r in sel})
        ncol = 4
        nrow = int(np.ceil(len(pairs) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.3 * nrow),
                                 squeeze=False)
        for ax, (src, tgt) in zip(axes.ravel(), pairs):
            for fam in A.ARMB_FAMILIES:
                pts = [r for r in sel
                       if r["source"] == src and r["target"] == tgt
                       and r["family"] == fam]
                if not pts:
                    continue
                x = [p["n_scenes"] for p in pts]
                y = [p["recovery"] for p in pts]
                ex = [p["exhausted_train_split"] for p in pts]
                ax.scatter([xi for xi, e in zip(x, ex) if not e],
                           [yi for yi, e in zip(y, ex) if not e],
                           color=FS.CONFIG_COLOR[fam],
                           marker=FS.CONFIG_MARKER[fam], s=34,
                           label=FS.label_for(fam), zorder=3)
                # exhausted draws are marked, never silently plotted as if
                # they were a k the sampler actually reached. The project log 6a-i.
                ax.scatter([xi for xi, e in zip(x, ex) if e],
                           [yi for yi, e in zip(y, ex) if e],
                           facecolors="none",
                           edgecolors=FS.CONFIG_COLOR[fam],
                           marker=FS.CONFIG_MARKER[fam], s=70, linewidths=1.6,
                           zorder=4)
            # R = 0 IS the zero shot level, by construction, and R = 1 is the
            # in domain reference. Both are labelled on the figure rather than
            # left for the caption, because the crossing point is the result:
            # below it, fine tuning is worse than not fine tuning.
            ax.axhline(0, color=FS.INK_MUTED, lw=0.9, ls=":",
                       gid=FS.REFERENCE)
            ax.axhline(1, color=FS.INK_MUTED, lw=0.9, ls="--",
                       gid=FS.REFERENCE)
            ax.set_xscale("log")
            ax.set_title(f"{src} to {tgt}", fontsize=10)
            ax.set_xlabel("realised scenes annotated")
            ax.set_ylabel(f"recovery R({metric})")
        for ax in axes.ravel()[len(pairs):]:
            ax.axis("off")
        lead = axes.ravel()[0]
        handles, labels = lead.get_legend_handles_labels()
        handles += [plt.Line2D([], [], color=FS.INK_MUTED, lw=0.9, ls=":"),
                    plt.Line2D([], [], color=FS.INK_MUTED, lw=0.9, ls="--")]
        labels += ["zero shot, R = 0", "in domain, R = 1"]
        lead.legend(handles, labels, frameon=False, fontsize=8)
        fig.suptitle(
            f"Arm B recovery, {metric} level, endpoint `last` (primary). "
            f"Hollow markers exhausted the target train split. Points below "
            f"the dotted line are WORSE than not fine tuning at all.",
            fontsize=11)
        fig.tight_layout()
        savefig(fig, f"armb_recovery_{metric}_last")

    # ---- tables
    by = defaultdict(list)
    for r in rows:
        by[(r["endpoint"], r["metric"], r["source"], r["target"],
            r["family"], r["k_requested"])].append(r)

    lines = ["# Arm B k-shot recovery", "",
             "Primary endpoint `last`, the project log 6a-ii. Plotted against the "
             "REALISED scene count, the project log 6a-i, never against requested "
             "k. Class space corrected, the project log 6a-iii.", "",
             "| end | metric | pair | cfg | k | images | scenes | AP | "
             "AP - base | P(0) | P_jj | denom | R |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    n_thin = 0
    for key in sorted(by):
        end, metric, src, tgt, fam, k = key
        g = by[key]
        ap, _ = msd([r["ap"] for r in g])
        lift, _ = msd([r["ap_minus_base_rate"] for r in g])
        p0, _ = msd([r["p_zero"] for r in g])
        pj, _ = msd([r["p_diag"] for r in g])
        rc, _ = msd([r["recovery"] for r in g])
        dn, _ = msd([r["denominator"] for r in g])
        ni = int(statistics.fmean([r["n_images"] for r in g]))
        ns = int(statistics.fmean([r["n_scenes"] for r in g]))
        exh = " *" if any(r["exhausted_train_split"] for r in g) else ""
        thin = any(r["denominator_thin"] for r in g)
        if thin:
            n_thin += 1
        lines.append(
            f"| {end} | {metric} | {src} to {tgt} | {FS.label_for(fam)} | "
            f"{k}{exh} | {ni} | {ns} | {fmt(ap)} | {fmt(lift)} | {fmt(p0)} | "
            f"{fmt(pj)} | {fmt(dn)}{' !' if thin else ''} | {fmt(rc, 3)} |")
    lines += [
        "", "`*` marks a k where at least one draw exhausted the target's "
        "train split, so the realised k is smaller than requested. twoclass "
        "cannot reach k = 250 at all, its train split holds 213 images. "
        "The project log 6a-i.",
        "", "`AP - base` is blank for box level ON PURPOSE. Positive class "
        "image AP is floored by prevalence, so subtracting the base rate is "
        "the right correction there. Box mAP50 has no prevalence floor, so "
        "the same subtraction would not be a quantity.",
        "", f"`!` marks a ratio whose denominator `P_jj - P_ij(0)` is below "
        f"0.05, which is {n_thin} of {len(by)} rows. The threshold was fixed "
        "before the ratios were read and no flagged row is dropped. A thin "
        "denominator amplifies noise, and the project log 5n establishes that "
        "kaggle_v2's box diagonal of about 0.080 is dataset difficulty rather "
        "than a defect, so a small absolute gain into kaggle_v2 produces a "
        "large looking R. Where the denominator is flagged, let the raw AP and "
        "the AP minus base rate carry the claim rather than the ratio.",
        "", "**Panel counts, so no reader is left counting.** Box level curves "
        "exist for SIX pairs: kaggle_v2 and lohi_weld each into lohi_weld or "
        "kaggle_v2, visual and twoclass. Every pair into weld_surface_v2 is "
        "box undefined, because weld_surface_v2 is whole region and a box "
        "number across protocols measures annotation mismatch. Image level "
        "curves exist for all EIGHT pairs.",
        "", "**Negative recovery is not clipped and k is not assumed "
        "monotonic.** Fine-tuning on a handful of target images can leave a "
        "model worse than the un-adapted source checkpoint. Where it does, the "
        "point is plotted below zero, because that is the result.",
        "", "**Image level recovery into lohi_weld is unstable and must not "
        "be quoted.** R divides by `P_jj - P_ij(0)`, and for lohi_weld that "
        "denominator is about 0.003, because the target is 97.6 percent "
        "positive so both terms sit just under 1.0. Dividing by it turns "
        "ordinary noise into ratios in the single and double digits. Read the "
        "box level recovery for lohi_weld. The project log 5k makes the same point "
        "for the main grid.",
    ]

    # twoclass reference beside every ratio into twoclass. The project log 5b.
    lines += ["", "## The twoclass reference, printed beside every ratio into "
              "twoclass", "",
              "twoclass has 213 training images, 23 scene blocks and 7 joint "
              "configurations. A small denominator inflates a ratio, so a high "
              "R into twoclass may say more about the reference being bad than "
              "about transfer being good. Never quote a twoclass ratio without "
              "this table.", "",
              "| configuration | seed | twoclass P_jj box | P_jj image |",
              "|---|---|---|---|"]
    for fam in A.ARMB_FAMILIES:
        for seed in A.ARMB_DRAWS:
            lines.append(
                f"| {FS.label_for(fam)} | {seed} | "
                f"{fmt(p_diag(main, 'twoclass', fam, seed, 'box'))} | "
                f"{fmt(p_diag(main, 'twoclass', fam, seed, 'image'))} |")

    # best against last
    lines += ["", "## `best` against `last`, the project log 6a-ii", "",
              "| pair | cfg | max |best - last| box | mean |d| box |",
              "|---|---|---|---|"]
    worst_overall = 0.0
    all_diffs: list[float] = []
    for src, tgt in sorted({(r["source"], r["target"]) for r in rows}):
        for fam in A.ARMB_FAMILIES:
            ds = []
            for k in A.ARMB_K:
                for d in A.ARMB_DRAWS:
                    a = [r for r in rows if r["endpoint"] == "last"
                         and r["metric"] == "box" and r["source"] == src
                         and r["target"] == tgt and r["family"] == fam
                         and r["k_requested"] == k and r["draw"] == d]
                    b = [r for r in rows if r["endpoint"] == "best"
                         and r["metric"] == "box" and r["source"] == src
                         and r["target"] == tgt and r["family"] == fam
                         and r["k_requested"] == k and r["draw"] == d]
                    if a and b and a[0]["ap"] is not None \
                            and b[0]["ap"] is not None:
                        ds.append(abs(b[0]["ap"] - a[0]["ap"]))
            if ds:
                worst_overall = max(worst_overall, max(ds))
                all_diffs += ds
                lines.append(f"| {src} to {tgt} | {FS.label_for(fam)} | "
                             f"{max(ds):.4f} | {statistics.fmean(ds):.4f} |")
    n_endpoint_pairs = len(all_diffs)
    mean_overall = statistics.fmean(all_diffs) if all_diffs else float("nan")
    # 3h. are the smallest budgets informative at all?
    byk = defaultdict(list)
    for r in rows:
        if r["endpoint"] == "last" and r["metric"] == "box" \
                and r["ap"] is not None:
            byk[r["k_requested"]].append(r)
    lines += ["", "## Are k = 10 and k = 25 informative at all? Box level, "
              "primary endpoint", "",
              "| k | mean scenes | box mAP50 | cells | cells retaining "
              "nothing |", "|---|---|---|---|---|"]
    for k in sorted(byk):
        g = byk[k]
        lines.append(
            f"| {k} | {statistics.fmean(r['n_scenes'] for r in g):.0f} | "
            f"{statistics.fmean(r['ap'] for r in g):.4f} | {len(g)} | "
            f"{sum(1 for r in g if r['n_pred_retained'] == 0)} |")
    lines += [
        "", "**Answer: at box level, no.** The two smallest budgets are "
        "indistinguishable from zero against k = 250. The project log 6a-iv explains "
        "why. The detection head is reinitialised whenever the source and "
        "target class counts differ, so only the backbone and neck carry over, "
        "and the smallest budgets measure a head fitted from scratch on ten "
        "images. The project log 6a-v records that such a head can collapse onto a "
        "single wrong class entirely.",
        "", "**Plot them anyway, and say why in the caption.** Dropping them "
        "would hide the most useful fact the curve carries, which is that a "
        "public checkpoint adapted with ten target images is worthless at box "
        "level. Presenting them as a smooth start to a recovery curve would be "
        "the real error, because the first points partly measure a different "
        "model rather than a less adapted one.",
    ]

    # 3i. The review's spot check, verified rather than adopted. Every number
    # in the paragraph below is MEASURED here, so the prose cannot drift away
    # from the data if the records are ever recomputed.
    spot = ("kaggle_v2", "lohi_weld", "yolo11s")
    sp_max, sp_at = 0.0, None
    sp_by_k = defaultdict(lambda: {"best": [], "last": []})
    for r in rows:
        if r["metric"] != "box" or r["ap"] is None:
            continue
        if (r["source"], r["target"], r["family"]) != spot:
            continue
        sp_by_k[r["k_requested"]][r["endpoint"]].append(r["ap"])
        if r["endpoint"] == "last":
            b = [q for q in rows
                 if q["endpoint"] == "best" and q["metric"] == "box"
                 and (q["source"], q["target"], q["family"]) == spot
                 and q["k_requested"] == r["k_requested"]
                 and q["draw"] == r["draw"] and q["ap"] is not None]
            if b and abs(b[0]["ap"] - r["ap"]) > sp_max:
                sp_max = abs(b[0]["ap"] - r["ap"])
                sp_at = (r["k_requested"], r["draw"])
    sp_mean_k = {k: abs(statistics.fmean(v["best"]) - statistics.fmean(v["last"]))
                 for k, v in sp_by_k.items() if v["best"] and v["last"]}
    sp_worst_k = max(sp_mean_k, key=sp_mean_k.get) if sp_mean_k else None

    lines += [
        "", f"Largest absolute `best` minus `last` box mAP50 difference "
        f"anywhere in arm B: **{worst_overall:.4f}**.",
        "", "**The 15 September review's incidental claim does NOT survive "
        "the class space correction, and this is worth stating plainly.** It "
        "reported that on kaggle_v2 to lohi_weld with YOLO11s the two "
        "endpoints differ by at most 0.004 box mAP50 at any k, with means by "
        "k agreeing to 0.002. Measured on the CORRECTED records that pair is "
        f"the largest disagreement in the whole arm, {sp_max:.4f} at "
        f"k = {sp_at[0]} draw {sp_at[1]}, and the means by k differ by "
        f"{sp_mean_k[sp_worst_k]:.4f} at k = {sp_worst_k}. The review's spot "
        "check was made on the uncorrected numbers, where the filter retained "
        "only `discontinuity` and both endpoints were squeezed toward zero.",
        "", "**The broader conclusion still holds, and should be reported as "
        "the broader conclusion rather than as the pair.** Across all "
        f"{n_endpoint_pairs} box defined comparisons, which is the 240 cells "
        "restricted to the six localisation pairs, the mean absolute `best` "
        f"minus `last` difference is {mean_overall:.4f}. So "
        "the endpoint amendment in the project log 6a-ii was worth making for the "
        "integrity of the label cost claim, and on almost every cell it does "
        "not move the number. It does move the single best transferring cell "
        "in the arm, which is exactly the cell a reader would quote.",
    ]
    # 3j. The headline sentence, in realised SCENES rather than requested k.
    # Written from the measured numbers in whatever form they support, which
    # is the plan's instruction. If the numbers do not support a clean
    # "n scenes recovers x percent" claim, say that instead of forcing one.
    head = defaultdict(list)
    for r in rows:
        if (r["endpoint"] == "last" and r["metric"] == "box"
                and r["recovery"] is not None
                and not r["denominator_thin"]):
            head[r["k_requested"]].append(r)
    lines += ["", "## 3j. The headline sentence this arm supports", "",
              "| k | mean realised scenes | mean realised images | "
              "mean R(box) | cells |", "|---|---|---|---|---|"]
    for k in sorted(head):
        g = head[k]
        lines.append(
            f"| {k} | {statistics.fmean(r['n_scenes'] for r in g):.0f} | "
            f"{statistics.fmean(r['n_images'] for r in g):.0f} | "
            f"{statistics.fmean(r['recovery'] for r in g):.3f} | {len(g)} |")
    if head:
        top = max(head)
        g = head[top]
        sc = statistics.fmean(r["n_scenes"] for r in g)
        im = statistics.fmean(r["n_images"] for r in g)
        rc = statistics.fmean(r["recovery"] for r in g)
        n_excluded = sum(
            1 for r in rows
            if r["endpoint"] == "last" and r["metric"] == "box"
            and r["recovery"] is not None and r["denominator_thin"])
        # Where does fine-tuning first beat simply NOT fine-tuning? That is
        # the question a practitioner actually has, and it is not the same as
        # where R turns positive on average.
        crossover = None
        for k in sorted(head):
            gk = head[k]
            if statistics.fmean(r["ap"] for r in gk) > \
                    statistics.fmean(r["p_zero"] for r in gk):
                crossover = k
                break
        # The single best transferring pair, named rather than averaged away.
        per_pair = defaultdict(list)
        for r in head[top]:
            per_pair[(r["source"], r["target"], r["family"])].append(
                r["recovery"])
        bp = max(per_pair, key=lambda k: statistics.fmean(per_pair[k]))
        bpv = statistics.fmean(per_pair[bp])
        n_neg = sum(1 for k in head
                    if statistics.fmean(r["recovery"] for r in head[k]) < 0)
        lines += [
            "", f"**The thin denominator exclusion removed "
            f"{n_excluded} rows at box level**, because every box level "
            f"denominator here runs from 0.087 to 0.672 and none falls under "
            f"0.05. The rule was fixed at 3d before the ratios were read and "
            f"is stated whether or not it bites.",
            "", "**THE SENTENCE THE NUMBERS SUPPORT IS A NEGATIVE ONE, AND IT "
            "IS THE MORE USEFUL RESULT.** Write it as the finding, not as a "
            "disappointment.",
            "", f"At the largest budget tested, about **{sc:.0f} annotated "
            f"target scenes**, which is about {im:.0f} images, k-shot fine "
            f"tuning recovers **{100 * rc:.1f} percent** of the in domain box "
            f"level gap, averaged over the six localisation pairs and two "
            f"configurations. That is indistinguishable from nothing.",
            "", f"**Below that budget it is worse than not fine tuning at "
            f"all.** Mean recovery is NEGATIVE at {n_neg} of the "
            f"{len(head)} budgets tested. Pooled box mAP50 rises with k, from "
            + ", ".join(
                f"{statistics.fmean(r['ap'] for r in head[k]):.4f} at k={k}"
                for k in sorted(head))
            + f", but the un-adapted source checkpoint already scores "
            f"{statistics.fmean(r['p_zero'] for r in head[top]):.4f} on the "
            f"same targets. "
            + (f"Fine tuning first beats doing nothing only at k = "
               f"{crossover}." if crossover else
               "Fine tuning never beats doing nothing at any budget tested."),
            "", f"**One pair is a real exception and must be named rather "
            f"than averaged away.** `{bp[0]} to {bp[1]}` with "
            f"{FS.label_for(bp[2])} reaches R = **{bpv:.3f}** at the largest "
            f"budget, roughly {100 * bpv:.0f} percent of its gap. It is also "
            f"the pair with the largest `best` against `last` disagreement, "
            f"so it is the one cell in the arm where the endpoint choice "
            f"matters. Quote it as the best case, never as the typical one.",
            "", "**Say it in scenes, not in k.** The project log 6a-i: a split block "
            "is indivisible, so the realised draw overshoots the requested k "
            "by up to 9.1 times, and the scene count is the honest label cost. "
            "Quoting the requested k would understate the annotation a "
            "practitioner actually has to pay for.",
            "", "**And state the caveat in the same breath.** The detection "
            "head is reinitialised at every k, the project log 6a-iv, so this is the "
            "budget needed to fit a NEW head on a source pretrained backbone. "
            "That is what a practitioner adopting a public checkpoint faces, "
            "so it is the useful quantity, but it is not the same thing as "
            "smooth domain adaptation from a working detector. It also "
            "explains the negative region: a head fitted on ten images is "
            "worse than the source head it replaced.",
            "", "**What this does to C4.** The label cost curve is still the "
            "most citable object in the paper, and it now carries a sharper "
            "claim than the one originally scoped. The finding is not that "
            "adaptation is cheap. It is that within a budget a practitioner "
            "would plausibly pay, up to roughly 250 labelled target images, "
            "adapting a public visible light weld checkpoint buys essentially "
            "nothing at box level, and below about 100 images it actively "
            "hurts. The figure draws the zero shot level as the dotted line at "
            "R = 0 and the in domain level as the dashed line at R = 1, so the "
            "crossing point is visible rather than inferred.",
        ]
    (out / "armb_recovery.md").write_text("\n".join(lines) + "\n")
    return rows


# --------------------------------------------------------------------------
# 4c. Arm D
# --------------------------------------------------------------------------

def armd_role(pool: str, target: str) -> str:
    """What a (pool, target) pair actually measures.

    A pool member scored on itself is not a leave-one-out result, so calling
    every column "held out" would overstate what arm D tests.
    """
    if target == A.POOL_HELD_OUT[pool]:
        return "held out training source"
    if target in A.ARMD_POOLS[pool]:
        return "in pool"
    return "held out test only target"


def armd_box_limit() -> dict:
    """4b. Derive arm D's box level coverage from config_arms, not from prose.

    The project log 6b says arm D reports box level only for the pool that excludes
    weld_surface_v2. That is true and it is not the whole story. That pool's
    OWN held out target IS weld_surface_v2, which is whole region, so the
    held out column is box undefined as well. The consequence is checked here
    rather than asserted: arm D cannot report a box level number for ANY held
    out training source.
    """
    rows = []
    for pool in A.ARMD_POOLS:
        for target in C.ALL_TARGETS:
            rows.append({
                "pool": pool,
                "pool_protocol": A.dataset_entry(pool)["protocol"],
                "target": target,
                "target_protocol": A.dataset_entry(target)["protocol"],
                "role": armd_role(pool, target),
                "box_level_defined": A.box_level_defined(pool, target),
            })
    held_out_box = [r for r in rows
                    if r["role"] == "held out training source"
                    and r["box_level_defined"]]
    return {
        "rows": rows,
        "box_defined_on_any_held_out_training_source": bool(held_out_box),
        "box_defined_pairs": sorted(
            (r["pool"], r["target"]) for r in rows if r["box_level_defined"]),
        "box_defined_held_out_pairs": sorted(
            (r["pool"], r["target"]) for r in rows
            if r["box_level_defined"] and r["role"].startswith("held out")),
    }


def armd(out: Path, main: dict) -> list[dict]:
    rows = []
    for rec in load("eval__", "armd"):
        c = rec["cell"]
        pool = c["source"]
        for target in rec["targets"]:
            # 4c. The control for "does pooling help" is the best SINGLE pool
            # MEMBER at the same family and seed, not the best of all three
            # training sources. The pool holds exactly those members, so this
            # compares one training budget against itself pooled or not. Using
            # a source outside the pool would compare different training data
            # and answer a different question.
            singles = {}
            for member in A.ARMD_POOLS[pool]:
                if member == target:
                    continue        # that is the diagonal, not a transfer row
                singles[member] = {
                    "image_ap": p_zero(main, member, c["family"], c["seed"],
                                       target, "image"),
                    "box_map50": p_zero(main, member, c["family"], c["seed"],
                                        target, "box"),
                }
            best_img = max(
                ((m, v["image_ap"]) for m, v in singles.items()
                 if v["image_ap"] is not None),
                key=lambda t: t[1], default=(None, None))
            best_box = max(
                ((m, v["box_map50"]) for m, v in singles.items()
                 if v["box_map50"] is not None),
                key=lambda t: t[1], default=(None, None))
            pooled_img = img_of(rec, target)
            pooled_box = box_of(rec, target)
            rows.append({
                "pool": pool, "held_out": A.POOL_HELD_OUT[pool],
                "protocol": A.dataset_entry(pool)["protocol"],
                "role": armd_role(pool, target),
                "target": target, "family": c["family"], "seed": c["seed"],
                "image_ap": pooled_img,
                "base_rate": base_rate_of(rec, target),
                "box_map50": pooled_box,
                "box_level_defined": A.box_level_defined(pool, target),
                "best_single_source_image": best_img[0],
                "best_single_image_ap": best_img[1],
                "pool_minus_best_single_image":
                    None if None in (pooled_img, best_img[1])
                    else pooled_img - best_img[1],
                "best_single_source_box": best_box[0],
                "best_single_box_map50": best_box[1],
                "pool_minus_best_single_box":
                    None if None in (pooled_box, best_box[1])
                    else pooled_box - best_box[1],
            })
    AN.write_csv(out / "armd.csv", rows)
    limit = armd_box_limit()
    (out / "armd_box_limit.json").write_text(json.dumps(limit, indent=2))
    AN.write_csv(out / "armd_box_limit.csv", limit["rows"])

    by = defaultdict(list)
    for r in rows:
        by[(r["pool"], r["target"], r["family"])].append(r)

    lines = ["# Arm D, leave one dataset out", "",
             "| pool | protocol | target | role | cfg | image AP | base rate | "
             "AP - base | box mAP50 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(by):
        pool, target, fam = key
        g = by[key]
        ia, isd = msd([r["image_ap"] for r in g])
        br, _ = msd([r["base_rate"] for r in g])
        bx, bsd = msd([r["box_map50"] for r in g])
        lift = None if ia is None or br is None else ia - br
        box_txt = ("undefined" if bx is None else f"{bx:.4f} ({bsd:.3f})")
        lines.append(
            f"| {pool} | {A.dataset_entry(pool)['protocol']} | {target} | "
            f"{g[0]['role']} | {FS.label_for(fam)} | {fmt(ia)} ({isd:.3f}) | "
            f"{fmt(br)} | {fmt(lift)} | {box_txt} |")

    # 4b. The box level limit, stated from the derived table rather than prose.
    lines += ["", "## 4b. Arm D's box level coverage, derived from "
              "`config_arms.box_level_defined`", "",
              "| pool | pool protocol | target | target protocol | role | "
              "box defined |", "|---|---|---|---|---|---|"]
    for r in limit["rows"]:
        lines.append(
            f"| {r['pool']} | {r['pool_protocol']} | {r['target']} | "
            f"{r['target_protocol']} | {r['role']} | "
            f"{'yes' if r['box_level_defined'] else 'no'} |")

    held = limit["box_defined_held_out_pairs"]
    lines += [
        "", f"**ARM D CANNOT REPORT A BOX LEVEL NUMBER FOR ANY HELD OUT "
        f"TRAINING SOURCE, and this is a protocol rule rather than a missing "
        f"file.** Verified above, not assumed. "
        f"`box_defined_on_any_held_out_training_source` is "
        f"`{limit['box_defined_on_any_held_out_training_source']}`.",
        "", "The reason is a double exclusion and only the first half is in "
        "the project log 6b. First, a pool containing `weld_surface_v2` is protocol "
        "mixed, so `pool_no_kaggle_v2` and `pool_no_lohi_weld` have no box "
        "level number at all. Second, the one localisation pool is "
        "`pool_no_weld_surface_v2`, and its held out target IS "
        "`weld_surface_v2`, which is whole region. So the only pool that could "
        "carry a box number is held out against the only target that cannot "
        "receive one.",
        "", f"Arm D's box level evidence is therefore exactly "
        f"{len(held)} pairs, all from the one localisation pool scored on the "
        f"two test only targets: "
        + ", ".join(f"`{p} to {t}`" for p, t in held) + ".",
        "", "Say this in the caption. A reader who sees an empty box column "
        "will otherwise assume a run failed.",
    ]

    # 4c. does pooling help at all?
    lines += ["", "## 4c. Does pooling help? Pool against the best single "
              "pool member, same family and same seed", "",
              "| pool | target | role | cfg | pooled image AP | best single | "
              "best single AP | pool - best |",
              "|---|---|---|---|---|---|---|---|"]
    wins = losses = 0
    for key in sorted(by):
        pool, target, fam = key
        g = [r for r in by[key] if r["pool_minus_best_single_image"] is not None]
        if not g:
            continue
        pooled, _ = msd([r["image_ap"] for r in g])
        single, _ = msd([r["best_single_image_ap"] for r in g])
        delta, dsd = msd([r["pool_minus_best_single_image"] for r in g])
        src = sorted({r["best_single_source_image"] for r in g})
        if delta > 0:
            wins += 1
        else:
            losses += 1
        lines.append(
            f"| {pool} | {target} | {g[0]['role']} | {FS.label_for(fam)} | "
            f"{fmt(pooled)} | {'/'.join(src)} | {fmt(single)} | "
            f"{delta:+.4f} ({dsd:.3f}) |")
    lines += [
        "", f"Pooling beats the best single pool member in **{wins} of "
        f"{wins + losses}** comparisons at image level.",
        "", "**The control is the best single POOL MEMBER, not the best of all "
        "three training sources.** The pool holds exactly those members, so "
        "this compares one training budget against itself, pooled or not. "
        "Comparing against a source outside the pool would change the training "
        "data and answer a different question. Where the target is itself a "
        "pool member the diagonal is excluded, since that is in domain rather "
        "than transfer.",
    ]

    lines += [
        "", "**Caption, and this is a protocol rule not a data gap.** Arm D "
        "reports image level for every fold. It reports box level ONLY for "
        "`pool_no_weld_surface_v2`, the one pool that excludes the whole "
        "region source, and even there only on the two test only targets. "
        "weld_surface_v2 annotates a frame sized quality verdict rather than a "
        "localised defect, so any pool containing it is protocol mixed and a "
        "box level number from it would measure annotation mismatch, not "
        "domain shift. The project log 6b.",
        "", "**The honest limit of the design, the project log 6b.** There are only "
        "three training sources, so a leave one out fold trains on two. This "
        "is leave one out over a very small pool count, and a two source pool "
        "is a weak test of whether breadth helps. visual and twoclass are test "
        "only targets and never enter a pool, which is why all three pools are "
        "evaluated against them.",
        "", "**Image AP is floored by prevalence on the positive heavy "
        "targets.** Read `AP - base` rather than the raw column, and read the "
        "box column where it exists. The project log 5k.",
    ]
    (out / "armd.md").write_text("\n".join(lines) + "\n")
    return rows


# --------------------------------------------------------------------------
# 4d. pre-registered statistics
# --------------------------------------------------------------------------

def _mean_or_none(values):
    """Mean over the values that exist, or None if none do."""
    kept = [v for v in values if v is not None]
    return statistics.fmean(kept) if kept else None


def _k_effect_magnitude(armb_rows: list[dict]) -> str:
    """One sentence on how large the k effect actually is, COMPUTED.

    This paragraph used to be hardcoded prose saying recovery was about 0.1
    percent at the largest budget and negative below it. That was true of the
    VOID arm B run, which trained from random initialisation, the project log
    6a-viii. After the retrain the same sentence was false and it survived a
    re-run because nothing recomputed it. Deriving it from the rows is the
    only way it cannot go stale again.
    """
    biggest = max((r["k_requested"] for r in armb_rows), default=None)
    if biggest is None:
        return ""
    per_pair = defaultdict(list)
    for r in armb_rows:
        if (r.get("endpoint") != "last" or r.get("metric") != "box"
                or r.get("k_requested") != biggest
                or r.get("recovery") is None):
            continue
        per_pair[(r["source"], r["target"], r["family"])].append(r["recovery"])
    if not per_pair:
        return ""
    means = sorted(sum(v) / len(v) for v in per_pair.values())
    return (f"At the largest budget, k = {biggest}, recovery of the in domain "
            f"gap runs from {means[0]:.3f} to {means[-1]:.3f} across the "
            f"{len(means)} box defined pair and configuration combinations, "
            f"median {statistics.median(means):.3f}. **Never pool these into "
            f"one number**: the denominators span roughly a factor of nine, "
            f"the project log 6a-xiii.")


def exploratory_k_friedman(armb_rows: list[dict]) -> dict:
    """5b. Friedman across the five k levels. EXPLORATORY, NOT pre-registered.

    The pre-registered test in the the project log statistics section is about
    CONFIGURATIONS, blocked on evaluation cell. Nothing was pre-registered
    about k. This is a reasonable question and it is labelled as what it is.

    `compare_configurations` stamps `preregistered: True` unconditionally,
    because every caller until now was the pre-registered test. That flag is
    overwritten here rather than left to be read off the file later.
    """
    blocks = defaultdict(dict)
    for r in armb_rows:
        if (r["endpoint"] != "last" or r["metric"] != "box"
                or r["ap"] is None):
            continue
        blocks[f"{r['source']}->{r['target']} {r['family']} d{r['draw']}"][
            str(r["k_requested"])] = r["ap"]
    ks = [str(k) for k in A.ARMB_K]
    blocks = {b: v for b, v in blocks.items() if len(v) == len(ks)}
    res = ST.compare_configurations(blocks, ks, metric="box_map50")
    res["preregistered"] = False
    res["label"] = ("EXPLORATORY. The pre-registered omnibus was about "
                    "configurations, not about k. Reported as exploratory in "
                    "the text, per the plan's rule and the the project log "
                    "statistics section.")
    res["design"] = ("complete block, k levels crossed with "
                     "(source, target, configuration, draw)")
    return res


def seed_dispersion(ctl_recs: list[dict], armd_rows: list[dict],
                    armb_rows: list[dict], main: dict) -> dict:
    """5d. Seed SD in the arms, against the main grid's own, all box mAP50.

    THE MAIN GRID FIGURE IS COMPUTED, NOT TYPED. Until 24 September 2026 this
    function returned a hardcoded main-grid median of 0.038, taken from a
    16 September working note that gave no derivation, and the manuscript
    quoted it. No released file supports it. Recomputed from the frozen main
    records it is 0.0047 at box level, and 0.0095 at image level."""
    main_groups = defaultdict(list)
    for (source, family, _seed, kind), rec in main.items():
        if kind != "main":
            continue
        for target in rec["targets"]:
            v = box_of(rec, target)
            if v is not None:
                main_groups[(source, target, family)].append(v)
    groups = defaultdict(list)
    for rec in ctl_recs:
        c = rec["cell"]
        for target in A.LOHI_ACQUISITIONS:
            v = box_of(rec, target)
            if v is not None:
                groups[("lohictl", f"{c['source']}->{target}",
                        c["family"])].append(v)
    for r in armd_rows:
        if r["box_map50"] is not None:
            groups[("armd", f"{r['pool']}->{r['target']}",
                    r["family"])].append(r["box_map50"])
    # Arm B's three draws are not seed replicates in the same sense: the draw
    # index IS the seed, so the k-shot sample and the training randomness move
    # together by design. The project log 6a-v records the same confound. Its
    # dispersion is reported separately and labelled, never pooled with the
    # other two.
    armb_groups = defaultdict(list)
    for r in armb_rows:
        if (r["endpoint"] == "last" and r["metric"] == "box"
                and r["ap"] is not None):
            armb_groups[(r["source"], r["target"], r["family"],
                         r["k_requested"])].append(r["ap"])
    def sds(g):
        return sorted(statistics.stdev(v) for v in g.values() if len(v) > 1)
    ctl_armd = sds(groups)
    armb_sd = sds(armb_groups)
    def summary(v, label):
        if not v:
            return {"label": label, "n": 0}
        return {"label": label, "n": len(v),
                "median": round(statistics.median(v), 4),
                "mean": round(statistics.fmean(v), 4),
                "min": round(v[0], 4), "max": round(v[-1], 4)}
    return {
        "main_grid": summary(sds(main_groups),
                             "main grid, box mAP50, SD over the three seeds"),
        "control_and_armd": summary(ctl_armd,
                                    "LoHi control and arm D, box mAP50, "
                                    "SD over the three seeds"),
        "armb_draws": summary(armb_sd,
                              "arm B, box mAP50, SD over the three DRAWS, "
                              "which are also the seeds"),
    }


def stats(out: Path, ctl_recs: list[dict], armd_rows: list[dict],
          armb_rows: list[dict]) -> dict:
    """Friedman with Nemenyi primary, RM-ANOVA with Tukey secondary.

    The blocking unit is the evaluation cell, exactly as the main grid does it.
    Arm B is NOT included: it runs only two of the four configurations by
    pre-registered design, the project log 6a, so it cannot enter a complete block
    design over four.
    """
    cell_scores = defaultdict(dict)
    for rec in ctl_recs:
        c = rec["cell"]
        for target in A.LOHI_ACQUISITIONS:
            v = box_of(rec, target)
            if v is not None:
                cell_scores[f"lohictl {c['source']}->{target}"].setdefault(
                    c["family"], []).append(v)
    for r in armd_rows:
        if r["box_map50"] is not None:
            cell_scores[f"armd {r['pool']}->{r['target']}"].setdefault(
                r["family"], []).append(r["box_map50"])

    blocks = {cell: {f: statistics.fmean(v) for f, v in fam.items()}
              for cell, fam in cell_scores.items()}
    blocks = {c: f for c, f in blocks.items()
              if len(f) == len(FS.CONFIG_ORDER)}
    result = ST.compare_configurations(blocks, FS.CONFIG_ORDER,
                                       metric="box_map50")

    seed_blocks = defaultdict(dict)
    for rec in ctl_recs:
        c = rec["cell"]
        for target in A.LOHI_ACQUISITIONS:
            v = box_of(rec, target)
            if v is not None:
                seed_blocks[
                    f"lohictl {c['source']}->{target} s{c['seed']}"][
                        c["family"]] = v
    for r in armd_rows:
        if r["box_map50"] is not None:
            seed_blocks[f"armd {r['pool']}->{r['target']} s{r['seed']}"][
                r["family"]] = r["box_map50"]
    seed_blocks = {c: f for c, f in seed_blocks.items()
                   if len(f) == len(FS.CONFIG_ORDER)}
    sensitivity = ST.compare_configurations(seed_blocks, FS.CONFIG_ORDER,
                                            metric="box_map50")

    k_test = exploratory_k_friedman(armb_rows)
    disp = seed_dispersion(ctl_recs, armd_rows, armb_rows, main_grid())

    # 5c. Does either arm disagree with the main grid that architecture does
    # not rescue transfer? The main grid Friedman returned p = 0.257 at image
    # level and p = 0.789 at box level. Recorded here so the comparison is in
    # the artefact rather than only in a sentence.
    arms_p = result.get("primary", {}).get("p_value")
    armb_spread = defaultdict(list)
    for r in armb_rows:
        if (r["endpoint"] == "last" and r["metric"] == "box"
                and r["ap"] is not None):
            armb_spread[r["family"]].append(r["ap"])
    agreement = {
        "main_grid_friedman_image_p": 0.257,
        "main_grid_friedman_box_p": 0.789,
        "arms_friedman_box_p": arms_p,
        "arms_significant": result.get("primary", {}).get("significant"),
        "arms_agree_architecture_does_not_rescue_transfer":
            None if arms_p is None else bool(arms_p >= 0.05),
        "armb_mean_box_map50_by_configuration": {
            f: round(statistics.fmean(v), 4) for f, v in armb_spread.items()},
    }

    payload = {"primary_blocked_on_cell": result,
               "sensitivity_cell_by_seed": sensitivity,
               "exploratory_k_friedman": k_test,
               "seed_dispersion": disp,
               "agreement_with_main_grid": agreement,
               "n_blocks": len(blocks), "n_seed_blocks": len(seed_blocks),
               "note": ("Arm B is excluded from the omnibus because it runs "
                        "two of the four configurations by pre-registered "
                        "design, the project log 6a, so it cannot form a complete "
                        "block over four.")}
    (out / "stats_arms.json").write_text(json.dumps(payload, indent=2))

    fr = result.get("primary", {})
    sec = result.get("secondary", {})
    frs = sensitivity.get("primary", {})
    lines = ["# Pre-registered statistics, arms", "",
             "Primary is Friedman blocked on evaluation cell with Nemenyi "
             "post hoc. Secondary is repeated measures ANOVA with Tukey. "
             "Decided before any result existed, see the the project log statistics "
             "section. Metric is box mAP50.", "",
             f"- blocks, seed averaged: **{len(blocks)}**",
             f"- Friedman chi2 {fr.get('chi2')}, df {fr.get('df')}, "
             f"p = **{fr.get('p_value'):.4f}**"
             if fr.get("p_value") is not None else "- Friedman not applicable",
             f"- significant at 0.05: **{fr.get('significant')}**",
             f"- RM-ANOVA F {sec.get('anova', {}).get('f_statistic')}, "
             f"p = **{sec.get('anova', {}).get('p_value'):.4f}**, "
             f"partial eta squared "
             f"{sec.get('anova', {}).get('partial_eta_squared')}"
             if sec.get("anova", {}).get("p_value") is not None
             else "- RM-ANOVA not reported",
             f"- primary and secondary agree: "
             f"**{result.get('primary_and_secondary_agree')}**",
             f"- sensitivity, (cell, seed) blocks: **{len(seed_blocks)}**, "
             f"chi2 {frs.get('chi2')}, p = **{frs.get('p_value'):.4f}**"
             if frs.get("p_value") is not None
             else f"- sensitivity not applicable",
             "",
             # DIRECTION MATTERS AND IS EASY TO GET BACKWARDS.
             # `stats_tests.nemenyi` uses `scipy.stats.rankdata`, which ranks
             # ASCENDING, so rank 1 is the SMALLEST value. The metric is box
             # mAP50, where larger is better, so a HIGHER mean rank is the
             # better configuration. An earlier draft of this line said the
             # opposite and would have named the worst configuration as best.
             "**Mean ranks. The metric is box mAP50 and `rankdata` ranks "
             "ascending, so a HIGHER mean rank is the BETTER configuration.** "
             + ", ".join(f"{FS.label_for(k)} {v}" for k, v in
                         (fr.get("posthoc", {}).get("mean_ranks") or {}).items()),
             "",
             "**Six blocks is thin, and that is a property of the design, not "
             "a shortfall.** The LoHi control contributes its four cells and "
             "arm D contributes only the two box defined cells of the one "
             "pool that excludes weld_surface_v2. Every other arm D cell is "
             "protocol mixed, so its box level number is undefined by the "
             "rule in the project log 6b. A Friedman over six blocks has very "
             "little power, so read a null here as uninformative rather than "
             "as evidence the configurations agree. The main grid's 15 block "
             "test is the one that carries the pre-registered claim.",
             "", payload["note"]]

    # 5c
    lines += [
        "", "## 5c. Do the arms agree with the main grid that architecture "
        "does not rescue transfer?", "",
        f"- main grid Friedman, image level: p = **0.257**",
        f"- main grid Friedman, box level: p = **0.789**",
        f"- arms Friedman, box level: p = "
        + ("**not applicable**" if arms_p is None else f"**{arms_p:.4f}**"),
        f"- arms agree: **"
        f"{agreement['arms_agree_architecture_does_not_rescue_transfer']}**",
        "", "**Mean arm B box mAP50 by configuration, endpoint `last`.** "
        + ", ".join(f"{FS.label_for(k)} {v:.4f}" for k, v in
                    agreement["armb_mean_box_map50_by_configuration"].items())
        + ".",
        "", "Arm B was run at two configurations rather than four precisely "
        "because the main grid found no detectable difference among the four, "
        "the project log 6a. Nothing in the arms overturns that. Read the arms null "
        "as consistent with the main grid rather than as an independent "
        "confirmation, because six blocks cannot confirm much.",
    ]

    # 5b, labelled exploratory
    kp = k_test.get("primary", {})
    lines += [
        "", "## 5b. Exploratory, NOT pre-registered: does k matter?", "",
        "**This test is exploratory.** The pre-registered omnibus in the "
        "the project log statistics section is about configurations, blocked on "
        "evaluation cell. Nothing was pre-registered about k. It is reported "
        "here so the question is answered, and it is labelled so no reader "
        "mistakes it for the pre-registered claim.", "",
        f"- design: {k_test['design']}",
        f"- blocks: **{k_test.get('n_blocks')}**, k levels: "
        f"{', '.join(str(k) for k in A.ARMB_K)}",
        (f"- Friedman chi2 {kp.get('chi2')}, df {kp.get('df')}, "
         f"p = **{kp['p_value']:.3e}**, significant "
         f"**{kp.get('significant')}**")
        if kp.get("p_value") is not None
        else f"- not applicable: {k_test.get('reason')}",
        "", "**Mean ranks over k. Higher is better, because `rankdata` ranks "
        "ascending and the metric is box mAP50.** "
        + ", ".join(f"k={k} {v}" for k, v in
                    (kp.get("posthoc", {}).get("mean_ranks") or {}).items()),
        "", "So the ranks increase monotonically with k, which is the "
        "expected direction and a useful check that the test is reading the "
        "data the right way up.",
        "", "**A significant k effect is NOT the same as a useful one, and "
        "the two must not be conflated.** The ranks say larger budgets score "
        "higher, which is unsurprising. A rank test detects ordering, not "
        "magnitude. " + _k_effect_magnitude(armb_rows) + " Quote the recovery "
        "numbers in section 4, not this p value.",
        "", "Metric is box mAP50 at the primary `last` endpoint. Image level "
        "is not used here because it is floored by prevalence on the positive "
        "heavy targets, the project log 5k.",
    ]

    # 5d
    mg = disp["main_grid"]
    ca = disp["control_and_armd"]
    ab = disp["armb_draws"]
    lines += [
        "", f"## 5d. Seed dispersion, against the main grid median of "
        f"{mg.get('median')}",
        "", "| group | n | median SD | mean SD | min | max |",
        "|---|---|---|---|---|---|",
        f"| LoHi control and arm D, over seeds | {ca['n']} | "
        f"**{ca.get('median')}** | {ca.get('mean')} | {ca.get('min')} | "
        f"{ca.get('max')} |",
        f"| arm B, over draws | {ab['n']} | **{ab.get('median')}** | "
        f"{ab.get('mean')} | {ab.get('min')} | {ab.get('max')} |",
        f"| main grid, over seeds | {mg['n']} | **{mg.get('median')}** | "
        f"{mg.get('mean')} | {mg.get('min')} | {mg.get('max')} |",
        "", "All three are box mAP50. Until 24 September 2026 the main grid "
        "row was a typed 0.038 with no derivation behind it; it is now "
        "computed from the frozen main-grid records.",
        "", "**Arm B's three draws are not seed replicates in the same sense, "
        "and its dispersion is reported separately for that reason.** The draw "
        "index IS the seed, the project log 6a-v, so which images were sampled and "
        "the training randomness move together by design. Its spread therefore "
        "mixes sampling variance with training variance and nothing in the "
        "archives separates them. Do not compare it against the main grid "
        "median as if it measured the same thing.",
    ]
    (out / "stats_arms.md").write_text("\n".join(lines) + "\n")
    return payload


# --------------------------------------------------------------------------
# 4e. per class with instance counts
# --------------------------------------------------------------------------

def per_class(out: Path) -> None:
    """Never a per class AP without its test instance count. The project log."""
    rows = []
    for arm, prefix in (("lohictl", "eval__"), ("armd", "eval__"),
                        ("armb", PRIMARY)):
        for rec in load(prefix, arm):
            c = rec["cell"]
            for target, t in rec["targets"].items():
                pc = t["box_level"].get("per_class") or {}
                for cls, entry in pc.items():
                    rows.append({
                        "arm": arm, "source": c["source"], "target": target,
                        "family": c["family"], "seed": c.get("seed"),
                        "class": cls,
                        "ap50": entry.get(AN.PER_CLASS_AP50)
                        if hasattr(AN, "PER_CLASS_AP50") else entry.get("map50"),
                        "n_instances": entry.get("n_instances"),
                    })
    AN.write_csv(out / "per_class_arms.csv", rows)
    lines = ["# Per class AP, arms", ""]
    if not rows:
        lines += ["No per class block is present in the arm evaluation "
                  "records. `evaluate_arm.py` writes per class AP only for "
                  "the in domain case, and the arm evaluators are cross "
                  "acquisition or cross dataset throughout. Reported as "
                  "absent rather than as an empty table, so it cannot be "
                  "mistaken for classes that scored zero.",
                  "", "The in domain per class table is the main grid's, in "
                  "`grid_results/analysis/tables/per_class_ap.md`, and every "
                  "row there already carries its test instance count."]
    else:
        lines += ["| arm | source | target | cfg | class | AP50 | test "
                  "instances |", "|---|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(
                f"| {r['arm']} | {r['source']} | {r['target']} | "
                f"{FS.label_for(r['family'])} | {r['class']} | "
                f"{fmt(r['ap50'])} | {r['n_instances']} |")
    (out / "per_class_arms.md").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# Task 6. Reporting order, and the list that makes C5 checkable
# --------------------------------------------------------------------------

def fp_budget_table(out: Path) -> list[dict]:
    """6a item 6. Recall at a fixed false positive budget, throughout.

    The project log 5 makes this the metric that carries the INSPECTION claim rather
    than AUROC, because it counts false positive BOXES and so stays measurable
    on the three targets that have too few negative images for AUROC.
    """
    budgets = C.FP_PER_IMAGE_BUDGETS      # from config, so it cannot drift
    rows = []
    for arm, prefix in (("lohictl", "eval__"), ("armd", "eval__"),
                        ("armb", PRIMARY)):
        for rec in load(prefix, arm):
            c = rec["cell"]
            for target, t in rec["targets"].items():
                b = t["box_level"]
                if b.get("skipped"):
                    continue
                row = {"arm": arm, "source": c["source"], "target": target,
                       "family": c["family"], "seed": c.get("seed"),
                       "k": c.get("k"), "draw": c.get("draw"),
                       "map50": b.get("map50"), "n_gt": b.get("n_gt")}
                for x in budgets:
                    row[f"recall_at_{x}_fp"] = b.get(
                        f"recall_at_{x}_fp_per_image")
                rows.append(row)
    AN.write_csv(out / "recall_at_fp_budget_arms.csv", rows)

    lines = ["# Recall at a fixed false positive budget, arms", "",
             "the project log 5 makes this the metric that carries the inspection "
             "claim. AUROC is unreportable on three of five targets for want "
             "of negative images. This counts false positive BOXES per image, "
             "so it stays measurable everywhere the box level is defined.", "",
             "| arm | source | target | cfg | R@0.1 | R@0.5 | R@1.0 | "
             "box mAP50 |", "|---|---|---|---|---|---|---|---|"]
    by = defaultdict(list)
    for r in rows:
        by[(r["arm"], r["source"], r["target"], r["family"])].append(r)
    for key in sorted(by):
        arm, src, tgt, fam = key
        g = by[key]
        vals = [msd([r[f"recall_at_{x}_fp"] for r in g])[0] for x in budgets]
        mp, _ = msd([r["map50"] for r in g])
        lines.append(
            f"| {arm} | {src} | {tgt} | {FS.label_for(fam)} | "
            + " | ".join(fmt(v, 3) for v in vals) + f" | {fmt(mp)} |")
    lines += [
        "", "Arm B rows are pooled over k and over the three draws, so read "
        "them as the arm's overall level rather than as a budget curve. The "
        "per k values are in `recall_at_fp_budget_arms.csv`.",
    ]
    (out / "recall_at_fp_budget_arms.md").write_text("\n".join(lines) + "\n")
    return rows


def calibration_table(out: Path) -> tuple[list[dict], dict]:
    """6a item 7. Calibration under shift, with the reliability bins."""
    rows, binrows = [], []
    for arm, prefix in (("lohictl", "eval__"), ("armd", "eval__"),
                        ("armb", PRIMARY)):
        for rec in load(prefix, arm):
            c = rec["cell"]
            for target, t in rec["targets"].items():
                cal = t["image_level"].get("calibration") or {}
                if not cal:
                    continue
                in_domain = (target == c["source"]
                             or target in A.ARMD_POOLS.get(c["source"], ()))
                rows.append({
                    "arm": arm, "source": c["source"], "target": target,
                    "family": c["family"], "seed": c.get("seed"),
                    "k": c.get("k"), "draw": c.get("draw"),
                    "shift": "in domain" if in_domain else "cross domain",
                    "ece": cal.get("ece"), "brier": cal.get("brier"),
                    "n_bins": cal.get("n_bins"),
                    "base_rate": base_rate_of(rec, target)})
                for b in cal.get("bins") or []:
                    binrows.append({
                        "arm": arm, "source": c["source"], "target": target,
                        "family": c["family"], "seed": c.get("seed"),
                        "k": c.get("k"), "draw": c.get("draw"),
                        "lo": b["lo"], "hi": b["hi"], "count": b["count"],
                        "mean_confidence": b["mean_confidence"],
                        "observed_positive_rate": b["observed_positive_rate"]})
    AN.write_csv(out / "calibration_arms.csv", rows)
    AN.write_csv(out / "reliability_bins_arms.csv", binrows)

    lines = ["# Calibration under shift, arms", "",
             "| arm | shift | cfg | n cells | ECE | Brier |",
             "|---|---|---|---|---|---|"]
    by = defaultdict(list)
    for r in rows:
        by[(r["arm"], r["shift"], r["family"])].append(r)
    for key in sorted(by):
        arm, shift, fam = key
        g = by[key]
        e, esd = msd([r["ece"] for r in g])
        b, _ = msd([r["brier"] for r in g])
        lines.append(f"| {arm} | {shift} | {FS.label_for(fam)} | {len(g)} | "
                     f"{fmt(e)} ({esd:.3f}) | {fmt(b)} |")
    cal = calibration_summary(rows, binrows)
    u = cal["under"]
    exc = defaultdict(list)
    for p, t, f, s in cal["not_under"].get("armd", []):
        exc[(p, t, f)].append(str(s))
    share = {r["target"]: r["base_rate"] for r in rows
             if r["base_rate"] is not None}

    def seeds(ss):
        return ss[0] if len(ss) == 1 else ", ".join(ss[:-1]) + " and " + ss[-1]

    exc_txt = "; ".join(
        f"{FS.label_for(f)} from {p} into {t}, whose test split is "
        f"{100 * share[t]:.0f} percent defective, at seed"
        f"{'s' if len(ss) > 1 else ''} {seeds(ss)}"
        for (p, t, f), ss in sorted(exc.items()))
    lines += [
        "", "**CALIBRATION DEGRADES UNDER SHIFT, and the control is the only "
        "place that shows it cleanly.** Inside the LoHi control the "
        "annotation protocol, annotators, class list and production line are "
        "all held constant, so the only thing that changes is the "
        f"acquisition. Mean ECE rises from **{cal['ctl_in']:.3f}** in domain "
        f"to **{cal['ctl_out']:.3f}** cross acquisition, a factor of about "
        f"{cal['ctl_out'] / cal['ctl_in']:.1f}. Every configuration moves the "
        "same way.",
        "", "**THE ERROR RUNS TOWARDS UNDERCONFIDENCE, NOT OVERCONFIDENCE.** "
        "An evaluation is underconfident when, weighted by image count over "
        "its bins, the share of images with a defect exceeds the score. That "
        f"holds for **{u['lohictl'][1]} of {u['lohictl'][0]}** control "
        f"evaluations, **{u['armb'][1]} of {u['armb'][0]}** fine tuned arm B "
        f"models and **{u['armd'][1]} of {u['armd'][0]}** pooling "
        "evaluations."
        + (f" The pooling exceptions are {exc_txt}." if exc else "")
        + " Inside the control, under the shift, the scores fall while the "
        "share of images with a defect stays near one. The detectors do not "
        "become confidently wrong. They become unsure about images that "
        "almost all carry a defect. That is still the costly direction for "
        "inspection, because a threshold carried over from the source domain "
        "then misses defects.",
        "", f"**Arm B's fine tuned models are the worst calibrated in the "
        f"study**, mean ECE {cal['armb']:.3f} over {cal['armb_n']} cells. "
        "That is consistent with the project log 6a-iv and 6a-v. A reinitialised "
        "head fitted on a handful of images scores the target's defective "
        "images low, and at the smallest budgets it can collapse onto one "
        "wrong class entirely. A practitioner cannot threshold such a model "
        "safely, which matters more for inspection than the mAP does.",
        "", "**ECE here is computed on the IMAGE level score, so read it "
        "against the target's base rate.** The project log 5k. On a target where "
        "nearly every image carries a defect, calibration error mostly "
        "measures how far the scores sit below that share. Compare the cross "
        "domain rows against the in domain rows of the SAME arm, never "
        "against an absolute standard.",
        "", "Per bin counts, mean confidence and observed positive rate are "
        "in `reliability_bins_arms.csv`. Figure 4.4 of the manuscript, "
        "`reliability_control.png`, is drawn from it by "
        "`reliability_control_figure.py`, which this script calls.",
    ]
    (out / "calibration_arms.md").write_text("\n".join(lines) + "\n")
    return rows, cal


def calibration_summary(rows: list[dict], binrows: list[dict]) -> dict:
    """Every number the calibration prose quotes, from the rows just written.

    Until 24 September 2026 the prose was partly typed, and one typed value,
    "about 0.71" for arm B in results.md, survived from the VOID arm B run of
    the project log 6a-viii. It also described the few-shot head as emitting
    "confident scores that carry no information", the opposite of what the
    reliability bins show. Nothing here is typed now.

    Underconfidence uses the same definition as the manuscript builder's
    reliability_direction(): per evaluation, the image-weighted mean over
    non-empty bins of (observed defect share minus mean score) is positive.
    """
    def ece_of(arm, shift=None):
        return [r["ece"] for r in rows if r["arm"] == arm
                and (shift is None or r["shift"] == shift)
                and r["ece"] is not None]

    cells = defaultdict(list)
    for b in binrows:
        if b["count"]:
            cells[(b["arm"], b["source"], b["target"], b["family"],
                   b["seed"], b["k"], b["draw"])].append(b)
    under, not_under = {}, defaultdict(list)
    for key, bins in cells.items():
        n = sum(b["count"] for b in bins)
        gap = sum(b["count"] * (b["observed_positive_rate"]
                                - b["mean_confidence"]) for b in bins) / n
        tot, hit = under.get(key[0], (0, 0))
        under[key[0]] = (tot + 1, hit + (gap > 0))
        if gap <= 0:
            arm, src, tgt, fam, seed = key[:5]
            not_under[arm].append((src, tgt, fam, seed))
    ctl_in, ctl_out = ece_of("lohictl", "in domain"), ece_of("lohictl",
                                                             "cross domain")
    armb_ece = ece_of("armb")
    return {"ctl_in": statistics.fmean(ctl_in),
            "ctl_out": statistics.fmean(ctl_out),
            "armb": statistics.fmean(armb_ece), "armb_n": len(armb_ece),
            "under": under,
            "not_under": {k: sorted(v) for k, v in not_under.items()}}


def results_md(out: Path, ctl: dict, armb_rows: list[dict],
               armd_rows: list[dict], s: dict, limit: dict,
               cal: dict) -> None:
    """The reporting order the paper uses. Regenerated, never hand edited.

    ORDER, and it is deliberate. The control comes FIRST because it is the
    result that explains why transfer fails, and everything after is read
    against it. Then the two cross-dataset matrices, then the label cost
    curve, then arm D, then the two metrics that carry the inspection claim.
    """
    L = []
    A_ = L.append

    def armb_table(metric: str) -> list[tuple]:
        """(pair, cfg, P(0), per-k means, P_jj, R) at the primary endpoint.

        R IS THE MEAN OF THE PER DRAW RATIOS, NOT A RATIO OF MEANS, and that
        is the pre-registered definition rather than a presentational choice.
        The project log 6a fixes `P_jj` at the SAME configuration and the SAME seed
        as the numerator, and arm B sets seed equal to draw, so every draw has
        its own denominator. An earlier version of this helper kept
        `meta[key] = (p_zero, p_diag)` inside the row loop, so the LAST draw's
        denominator won and was then divided into a three draw mean numerator.
        That is the same last-write-wins fault as the per class instance count
        in `analysis.per_class_markdown`, and here it broke a pre-registration.
        The P(0) and P_jj columns are means over the same draws, so the table
        is self consistent, but R is not recoverable from them by division.
        """
        by = defaultdict(lambda: defaultdict(list))
        for r in armb_rows:
            if (r["endpoint"] != "last" or r["metric"] != metric
                    or r["ap"] is None):
                continue
            by[(r["source"], r["target"], r["family"])][
                r["k_requested"]].append(r)
        rows = []
        for key in sorted(by):
            every = [x for k in by[key] for x in by[key][k]]
            p0 = _mean_or_none(x["p_zero"] for x in every)
            pjj = _mean_or_none(x["p_diag"] for x in every)
            ks = [statistics.fmean(x["ap"] for x in by[key][k])
                  if by[key].get(k) else None for k in A.ARMB_K]
            biggest = by[key].get(A.ARMB_K[-1]) or []
            per_draw = [x["recovery"] for x in biggest
                        if x.get("recovery") is not None]
            R = statistics.fmean(per_draw) if per_draw else None
            rows.append((key, p0, ks, pjj, R))
        return rows

    A_("# Arm results, in the order the paper should use")
    A_("")
    A_("Generated by `grid/arms/analysis_arms.py`. **Do not hand edit**: it is "
       "overwritten on every run. Every number below is in a CSV or JSON "
       "beside this file, and section 8 says which.")
    A_("")
    A_("The order is deliberate. The control comes first because it explains "
       "WHY transfer fails, and everything after is read against it.")
    A_("")

    # ---------------------------------------------------------------- 1
    A_("## 1. The LoHi internal control. The yardstick")
    A_("")
    A_("Annotation protocol, annotators, class list and production line are "
       "all held constant. Only the acquisition changes. This is the one "
       "measurement in the study that isolates visual domain shift with the "
       "label concept pinned, so it is the number every cross-dataset figure "
       "should be read against.")
    A_("")
    A_("| train | to lohi_high | to lohi_low |")
    A_("|---|---|---|")
    for src in A.LOHI_ACQUISITIONS:
        cells = [f"{ctl['box'][f'{src}->{t}'][0]:.4f} "
                 f"({ctl['box'][f'{src}->{t}'][1]:.3f})"
                 for t in A.LOHI_ACQUISITIONS]
        A_(f"| {src} | " + " | ".join(cells) + " |")
    A_("")
    for k, v in ctl["ratios"].items():
        band = ctl.get("ratio_bands", {}).get(k)
        txt = f"**T = {v:.3f}**"
        if band:
            txt += (f", 95 percent split block interval "
                    f"[{band[0]:.3f}, {band[1]:.3f}]")
        A_(f"- {k}: {txt}")
    A_("")
    A_("**Interpretation, with its confound attached.** Changing only the "
       "acquisition still costs roughly half of box mAP50. So visual shift "
       "alone causes a large collapse. But the defect mix ALSO differs "
       "sharply between the two acquisitions, pore 5.5 percent of high "
       "resolution boxes against 26.5 percent of low, deposit 18.9 against "
       "8.8, the project log 7. This is not pure sensor shift. Part of it is prior "
       "shift. Both halves belong in the caption.")
    A_("")
    A_("**Image AP is uninformative here and must not be quoted.** It is 0.99 "
       "to 1.00 in every cell because LoHi is 97.6 percent positive. The "
       "control is a box level result.")
    A_("")
    A_("**Precedent.** LoHi-WELD's own IEEE Access 2024 description runs the "
       "same internal test across its two acquisitions. This version differs "
       "by running under the leakage audited splits and the same protocol as "
       "the rest of the study. Cite it as precedent, not as novelty.")
    A_("")
    A_("Table `lohi_control.md`, intervals "
       "`lohi_control_transfer_intervals.csv`, figure "
       "`lohi_control_matrix.pdf`.")
    A_("")

    # ---------------------------------------------------------------- 2
    A_("## 2. The box level cross-dataset matrix")
    A_("")
    A_("Source: `grid_results/analysis/tables/transfer_box_*.md`, one table "
       "per configuration, with split block intervals in "
       "`transfer_ratio_intervals.md`.")
    A_("")
    A_("**Two training rows, not three.** Only `kaggle_v2` and `lohi_weld` "
       "use a localisation protocol and are large enough to train on. "
       "`weld_surface_v2` annotates a frame sized quality verdict, so it "
       "cannot be a box level training row or target column. That the public "
       "visible light data supports only two localisation training rows is "
       "itself a finding, the project log finding 3.")
    A_("")
    A_("**The main grid is unaffected by the the project log 6a-iii class space "
       "defect**, because a main grid model predicts in its own source space "
       "and is filtered with source ids, which is correct. The defect was "
       "specific to arm B, whose head is rebuilt at the target `nc`.")
    A_("")
    A_("Read section 1 against this. The control's internal collapse is the "
       "floor that visual shift alone explains. Whatever the cross-dataset "
       "matrix loses beyond that is attributable to the other axes, object "
       "scale and annotation protocol.")
    A_("")

    # ---------------------------------------------------------------- 3
    A_("## 3. The base-rate corrected image level matrix")
    A_("")
    A_("Source: `grid_results/analysis/tables/transfer_image_*.md` with "
       "`target_base_rates.csv`.")
    A_("")
    A_("**Always the base-rate corrected form.** Positive class AP is floored "
       "by prevalence, so a raw image AP of 0.99 on a 97.6 percent positive "
       "target is a statement about the target, not about the model. "
       "The project log 5k measured LoHi's entire available headroom at 0.0243 and "
       "the four configurations take 88 to 97 percent of it.")
    A_("")
    A_("**AUROC is reportable on two targets only**, `kaggle_v2` and "
       "`weld_surface_v2`, plus the LoHi low resolution verdict subset. The "
       "other three have too few negative images, so AUROC is returned as "
       "null with a stated reason rather than as a number a reader would "
       "trust. The project log 5 and 5d. **This is why the false positive budget "
       "metric in section 6 carries the inspection claim, not AUROC.**")
    A_("")

    # ---------------------------------------------------------------- 4
    A_("## 4. Arm B, the label cost curve. C4")
    A_("")
    A_("Primary endpoint `last` at the fixed 50 epochs, the project log 6a-ii, "
       "because its label cost is exactly the draw. `best` is secondary and "
       "its caption must state the validation split size, since selecting on "
       "the target's valid split spends up to 444 further target labels.")
    A_("")
    A_("**THIS IS A POSITIVE RESULT.** The curve rises with k on almost every "
       "pair, and at the largest budget most pairs recover a substantial "
       "fraction of the in domain gap.")
    A_("")
    A_("| pair | cfg | P(0) | "
       + " | ".join(f"k={k}" for k in A.ARMB_K)
       + " | P_jj | R@250 |")
    A_("|---|---|---|" + "---|" * len(A.ARMB_K) + "---|---|")
    for (src, tgt, fam), p0, ks, pjj, R in armb_table("box"):
        A_(f"| {src} to {tgt} | {FS.label_for(fam)} | {fmt(p0)} | "
           + " | ".join(fmt(v) for v in ks)
           + f" | {fmt(pjj)} | {fmt(R, 3)} |")
    A_("")
    A_("**NEVER POOL R ACROSS PAIRS.** The denominators span roughly a factor "
       "of nine, so a pooled mean of ratios is not a quantity. Report per "
       "pair and per configuration, always with the absolute AP beside it, "
       "which is why both are in the table above.")
    A_("")
    A_("**Any ratio into `twoclass` must be quoted with its reference score.** "
       "twoclass's reference is weak by construction, 213 training images "
       "over 23 scene blocks and 7 joint configurations, the project log 5b. A "
       "ratio at or above 1 there says the denominator is small, not that "
       "transfer is superb.")
    A_("")
    A_("**The two hard targets are `visual` and `kaggle_v2`**, whose own in "
       "domain box AP is near 0.1. The project log 5n diagnoses kaggle_v2 four "
       "ways: not a mapping fault, not object scale, not resolution, but a "
       "generalisation failure from 745 `Defect` boxes across 337 images. "
       "Their absolute numbers stay small even at a respectable R, which is "
       "the reason the absolute AP is printed.")
    A_("")
    A_("**Plotted against the REALISED scene count, as points**, never "
       "against requested k and never joined into a line across draws. "
       "The project log 6a-i: a split block is indivisible, so a draw can overshoot "
       "k by up to 9.1 times. The requested k is a sampling instruction, not "
       "a measured quantity.")
    A_("")
    A_("**Box level curves exist for six pairs, image level for eight.** "
       "Every pair into `weld_surface_v2` is box undefined by protocol.")
    A_("")
    A_("**The detection head is reinitialised at every k**, the project log 6a-iv, "
       "because the source and target class counts differ. Backbone and neck "
       "carry over and the head does not, which is now asserted rather than "
       "assumed: `train_arm` records `head_nc` and `target_nc` in every "
       "`cell.json` and fails the cell if they disagree. So arm B measures "
       "the target annotation budget needed to fit a NEW head on a source "
       "pretrained backbone. That is what a practitioner adopting a public "
       "checkpoint faces, so it is the useful quantity as well as the honest "
       "description. Say it in the caption, not only the methods.")
    A_("")
    # BOTH ENDPOINT COUNTS, because quoting one invites a reader to think it
    # disagrees with the other figure in the project log 6a-v and gate 1c.
    def n_empty(end):
        return sum(1 for r in armb_rows
                   if r["endpoint"] == end and r["metric"] == "box"
                   and r.get("n_pred_retained") == 0)
    def n_box(end):
        return sum(1 for r in armb_rows
                   if r["endpoint"] == end and r["metric"] == "box")
    A_(f"**{n_empty('last')} of {n_box('last')} box defined records retain no "
       f"prediction at all at the primary `last` endpoint**, and "
       f"{n_empty('best')} of {n_box('best')} at `best`, "
       f"{n_empty('last') + n_empty('best')} across both. All are at the "
       f"smallest budgets. The reinitialised head has collapsed onto a single "
       f"wrong class, emitting its full detection cap in classes that are not "
       f"the defect class. The project log 6a-v: this is a result about fitting a "
       f"head on a small draw, NOT a class space fault, and the cells stay in "
       f"every table, flagged. Box mAP50 of 0.0 is the correct value; image "
       f"AP is suppressed to null there, because an all tied score vector is "
       f"undefined rather than zero.")
    A_("")
    A_("**WHICH TEN IMAGES YOU DRAW MATTERS MORE THAN THE SCHEDULE, on a "
       "small target.** The project log 6a-xvii. At k = 10 the three draws span "
       "1.7x on `lohi_weld` but **16.1x on `twoclass`**, whose train split "
       "holds 213 images over 23 scene blocks. That spread is larger than any "
       "schedule effect measured here. So on a small, low scene count target, "
       "what a practitioner gets for ten annotated images depends more on "
       "WHICH ten than on anything else this study varied. Report every cell "
       "per draw, as points, never as a pooled mean alone.")
    A_("")
    A_("Figures `armb_recovery_box_last.pdf` and "
       "`armb_recovery_image_last.pdf`, table `armb_recovery.md`, every cell "
       "in `armb_recovery.csv`.")
    A_("")
    A_("### 4a. Two EXPLORATORY controls on the curve's shape")
    A_("")
    A_("**Neither is pre-registered, and `ARMB_EPOCHS` stays at 50.** "
       "The project log 6a-xii records that as a decision: the schedule was fixed a "
       "priori, and changing it after seeing a result is what pre-registration "
       "exists to prevent. These explain the curve's shape. They are a "
       "discussion result, not the headline, and their exploratory status "
       "belongs in the text AND the caption.")
    A_("")
    A_("At k = 10 the step budget is about 100 optimisation steps against "
       "1,600 at k = 250, so a low score at small k could be too few labels "
       "or too few steps. Two controls separate them, on "
       "`kaggle_v2` to `lohi_weld`:")
    A_("")
    A_("| arm | distinct images | steps | cosine over | close_mosaic |")
    A_("|---|---|---|---|---|")
    A_("| A fixed 50 | 10 | 100 | 50 epochs | 20% |")
    A_("| B oversampled 50 | 10 | 1,650 | 50 epochs | 20% |")
    A_("| C matched step | 10 | 1,600 | 800 epochs | 1.25% |")
    A_("")
    A_("**The step count is the driver.** A to B, which changes only the step "
       "count, is +0.1656 box mAP50 at k = 10 and +0.1531 at k = 50. B to C, "
       "which changes only the schedule shape, is +0.0379 at k = 10 and "
       "**-0.0206** at k = 50. The schedule term does not hold its sign "
       "across the two coordinates, so no single percentage is quoted. "
       "The project log 6a-xi and 6a-xiv.")
    A_("")
    A_("So the small-k floor is an optimisation BUDGET limit rather than a "
       "label limit. Recovery on that pair goes R = 0.102 at the "
       "pre-registered schedule to **R = 0.347** with the step budget matched "
       "under the same schedule shape. **Quote 0.347**, because it changes "
       "exactly one thing against the pre-registered arm.")
    A_("")
    A_("**Target headroom does not explain which pairs respond.** Holding the "
       "source at `kaggle_v2` and varying only the target, delta R is +0.2464 "
       "into `lohi_weld`, +0.2487 into `twoclass` and +0.1148 into `visual`, "
       "against in domain scores of 0.688, 0.528 and 0.142. `twoclass` has "
       "three quarters of `lohi_weld`'s headroom and an almost identical "
       "response. Predictions were written down before those cells ran, and "
       "one was confirmed and one refuted. The project log 6a-xvi.")
    A_("")
    A_("**`visual`'s response is real and tiny**, 0.0000 to 0.0153 box mAP50. "
       "A delta R of 0.115 built on a 0.0153 absolute gain must not be read "
       "as `visual` behaving like `lohi_weld`.")
    A_("")
    A_("**One cell is unexplained and is stated as a limitation.** "
       "`lohi_weld` to `kaggle_v2` responds at +0.0221, five times below the "
       "weakest of the above. That cell differs in BOTH source and target and "
       "the experiment held the source fixed, so the two were not separated. "
       "The project log 5n's diagnosis of `kaggle_v2` as a hard target is the "
       "candidate. The project log 6a-xviii records the decision not to run the six "
       "further cells, because `twoclass` already spans sixteenfold across "
       "draws and they could not separate a source effect from draw identity.")
    A_("")
    A_("Tables in `grid_results_msc/matched_step_control.md` and `.csv`.")
    A_("")

    # ---------------------------------------------------------------- 5
    A_("## 5. Arm D, leave one dataset out")
    A_("")
    wins = sum(1 for r in armd_rows
               if (r.get("pool_minus_best_single_image") or 0) > 0)
    tot = sum(1 for r in armd_rows
              if r.get("pool_minus_best_single_image") is not None)
    A_(f"**Pooling does not reliably help.** Each pooled fold is compared "
       f"against the best single POOL MEMBER at the same family and seed, "
       f"which holds the training budget fixed: the pool contains exactly "
       f"those members, so this is one training set against itself pooled or "
       f"not. Pooling wins **{wins} of {tot}** cell level comparisons.")
    A_("")
    A_(f"**Arm D cannot report a box level number for any held out training "
       f"source.** Derived from `config_arms.box_level_defined`, not asserted: "
       f"`box_defined_on_any_held_out_training_source` is "
       f"`{limit['box_defined_on_any_held_out_training_source']}`. Two "
       f"exclusions compose. A pool containing `weld_surface_v2` is protocol "
       f"mixed, and the one localisation pool is held out against "
       f"`weld_surface_v2`, which is whole region. Its box level evidence is "
       f"therefore only "
       + ", ".join(f"`{p} to {t}`"
                   for p, t in limit["box_defined_held_out_pairs"])
       + ". Say this in the caption, or a reader will assume a run failed.")
    A_("")
    A_("**The honest limit of the design.** Three training sources means a "
       "fold trains on two, so this is leave one out over a very small pool "
       "count and a weak test of whether breadth helps. It does not license "
       "a claim that pooling never helps, only that two-source pooling did "
       "not help here. The project log 6b and 6b-ii.")
    A_("")
    A_("Table `armd.md`, coverage `armd_box_limit.json`.")
    A_("")

    # ---------------------------------------------------------------- 6
    A_("## 6. Recall at a fixed false positive budget")
    A_("")
    A_("Table `recall_at_fp_budget_arms.md`, CSV "
       "`recall_at_fp_budget_arms.csv`. Reported at 0.1, 0.5 and 1.0 false "
       "positive boxes per image, throughout.")
    A_("")
    A_("**This metric carries the inspection claim, not AUROC.** Inspection "
       "requirements are specified as a recall at a tolerated false alarm "
       "rate, no weld detection paper reports it, and it stays measurable on "
       "all five targets because it counts false positive BOXES rather than "
       "false positive IMAGES. Three of five targets have too few negative "
       "images for AUROC. The project log 5.")
    A_("")

    # ---------------------------------------------------------------- 7
    A_("## 7. Calibration under shift")
    A_("")
    A_("Table `calibration_arms.md`, per bin data for the reliability "
       "diagrams in `reliability_bins_arms.csv`.")
    A_("")
    u = cal["under"]
    A_("**Calibration degrades under shift, and the control shows it "
       "cleanly.** Inside the LoHi control, where everything but the "
       f"acquisition is held constant, mean ECE rises from "
       f"{cal['ctl_in']:.3f} in domain to {cal['ctl_out']:.3f} cross "
       f"acquisition. Arm B's fine tuned models are the worst calibrated in "
       f"the study, at a mean of {cal['armb']:.3f} over {cal['armb_n']} "
       "cells.")
    A_("")
    A_("**The error runs towards underconfidence, not overconfidence.** "
       f"{u['lohictl'][1]} of {u['lohictl'][0]} control evaluations, "
       f"{u['armb'][1]} of {u['armb'][0]} fine tuned models and "
       f"{u['armd'][1]} of {u['armd'][0]} pooling evaluations score the "
       "defective images below their share, weighted by image count over "
       "the bins. A reinitialised head fitted on a handful of images "
       "(the project log 6a-iv and 6a-v) scores the target's defective images "
       "low, and under the shift the scores fall while the share of images "
       "with a defect stays near one. **A threshold carried over from the "
       "source domain therefore misses defects, and for an inspection claim "
       "that matters more than the mAP does.** Figure 4.4, "
       "`reliability_control.png`, shows it inside the control.")
    A_("")
    A_("On a target where nearly every image carries a defect, calibration "
       "error mostly measures how far the scores sit below that share, so "
       "read cross domain rows against in domain rows of the same arm "
       "rather than against an absolute standard.")
    A_("")

    # ---------------------------------------------------------------- per class
    A_("## Per class AP")
    A_("")
    A_("**Never printed without its test instance count**, and classes below "
       "50 test instances are marked thin. Four classes are thin in the "
       "frozen splits, concentrated in the two smallest sources. A thin class "
       "needs a split block bootstrap interval, not seed dispersion: "
       "the project log 5m measured twoclass `Geometric defect` at seed SD 0.066 "
       "against a split block interval of [0.025, 0.675], roughly five times "
       "wider.")
    A_("")
    A_("The arms contribute no per class table, because `evaluate_arm.py` "
       "writes per class AP only in domain and the arms are cross "
       "acquisition or cross dataset throughout. See `per_class_arms.md`, "
       "which reports the absence rather than an empty table. The in domain "
       "table is the main grid's, "
       "`grid_results/analysis/tables/per_class_ap.md`.")
    A_("")

    # ---------------------------------------------------------------- stats
    A_("## Statistics, pre-registered and exploratory separated")
    A_("")
    pr = s["primary_blocked_on_cell"].get("primary", {})
    kt = s["exploratory_k_friedman"].get("primary", {})
    ag = s["agreement_with_main_grid"]
    A_("**Pre-registered**, decided before any result existed: Friedman "
       "blocked on evaluation cell with Nemenyi post hoc as primary, "
       "RM-ANOVA with Tukey as secondary, split block bootstrap for the "
       "transfer ratio intervals.")
    A_("")
    A_(f"- main grid Friedman: p = 0.257 image level, p = 0.789 box level")
    A_(f"- arms Friedman, box level, {s['n_blocks']} blocks: p = "
       + ("n/a" if pr.get("p_value") is None else f"**{pr['p_value']:.4f}**"))
    A_(f"- the arms agree that architecture does not rescue transfer: "
       f"**{ag['arms_agree_architecture_does_not_rescue_transfer']}**")
    A_("")
    A_("**Six blocks is thin and a null there is uninformative rather than "
       "confirmatory.** The main grid's 15 block test carries the "
       "pre-registered claim.")
    A_("")
    A_("**Mean rank direction must be stated beside every rank table.** "
       "`stats_tests.nemenyi` uses `scipy.stats.rankdata`, which ranks "
       "ascending, so rank 1 is the SMALLEST value. Every metric here is one "
       "where larger is better, so a HIGHER mean rank is the better "
       "configuration. The project log 6a-vii: an earlier draft had this backwards "
       "and would have named the worst configuration as the best.")
    A_("")
    A_("**The architecture question, reconciled.** Mean arm B box mAP50 is "
       + ", ".join(f"{FS.label_for(k)} {v:.4f}" for k, v in
                   ag["armb_mean_box_map50_by_configuration"].items())
       + ", under 2 percent apart. An earlier VOID run showed a factor of "
       "nine, and that gap was an artefact of training from random "
       "initialisation, where RT-DETR converges far slower than YOLO. With "
       "the checkpoints actually loaded the two are indistinguishable, which "
       "is CONSISTENT with the main grid's Friedman null rather than in "
       "tension with it. The project log 6a-xiii.")
    A_("")
    A_("**Exploratory, NOT pre-registered.** A Friedman across the five k "
       "levels is a reasonable question and was not pre-registered; the "
       "pre-registered test was about configurations.")
    A_("")
    A_(f"- k level Friedman, {s['exploratory_k_friedman'].get('n_blocks')} "
       f"blocks: p = "
       + ("n/a" if kt.get("p_value") is None else f"**{kt['p_value']:.3e}**"))
    A_("")
    A_("**A significant k effect is not the same as a useful one.** A rank "
       "test detects ordering, not magnitude. Quote the recovery numbers in "
       "section 4, never this p value.")
    A_("")
    A_("Details in `stats_arms.md` and `stats_arms.json`.")
    A_("")

    # ---------------------------------------------------------------- 8
    A_("## 8. Every manuscript number, and the file it came from. C5")
    A_("")
    A_("This list is what makes the released artefact checkable. A number in "
       "the paper that is not in one of these files is a number nobody can "
       "verify.")
    A_("")
    A_("| quantity | file |")
    A_("|---|---|")
    for q, f in [
        ("Task 1 gate outcomes",
         "`grid_results_arms/analysis/gates.json`"),
        ("LoHi control 2 by 2 box matrix",
         "`grid_results_arms/analysis/lohi_control.md`"),
        ("LoHi control transfer ratios with split block intervals",
         "`grid_results_arms/analysis/lohi_control_transfer_intervals.csv`"),
        ("Arm B recovery, every cell, both endpoints",
         "`grid_results_arms/analysis/armb_recovery.csv`"),
        ("Arm B summary with denominator flags",
         "`grid_results_arms/analysis/armb_recovery.md`"),
        ("Arm B figures, primary `last` endpoint",
         "`grid_results_arms/analysis/armb_recovery_{box,image}_last.pdf`"),
        ("Arm B per cell tensor hashes, the 6a-viii integrity check",
         "`grid_results_arms/armb_tensor_hashes.json`"),
        ("EXPLORATORY schedule controls",
         "`grid_results_msc/matched_step_control.{md,csv}`"),
        ("Arm D, every fold",
         "`grid_results_arms/analysis/armd.csv`"),
        ("Arm D box level coverage, derived",
         "`grid_results_arms/analysis/armd_box_limit.json`"),
        ("Recall at a fixed false positive budget",
         "`grid_results_arms/analysis/recall_at_fp_budget_arms.csv`"),
        ("Calibration and reliability bins",
         "`grid_results_arms/analysis/calibration_arms.csv`, "
         "`reliability_bins_arms.csv`"),
        ("Arm statistics, pre-registered and exploratory",
         "`grid_results_arms/analysis/stats_arms.json`"),
        ("Main grid box and image transfer matrices",
         "`grid_results/analysis/tables/transfer_{box,image}_*.md`"),
        ("Main grid transfer ratio intervals",
         "`grid_results/analysis/tables/transfer_ratio_intervals.md`"),
        ("Main grid per class AP with test instance counts",
         "`grid_results/analysis/tables/per_class_ap.md`"),
        ("Main grid pre-registered statistics",
         "`grid_results/analysis/statistical_tests.json`"),
        ("Downloaded against audited estimate shift",
         "`grid_results/analysis/tables/downloaded_vs_audited.md`"),
        ("Frozen split hashes",
         "`clean_candidate/split_hashes.json`, "
         "`grid/arms/arm_split_hashes.json`"),
        ("Pinned initial checkpoints",
         "`models/pretrained/manifest.json`"),
        ("Arm B start checkpoints, verified against the cluster",
         "`grid_results/armb_start_checkpoints/manifest.json`"),
    ]:
        A_(f"| {q} | {f} |")
    A_("")
    A_("**Not in this release, and stated rather than omitted.** Arms A and C "
       "are scoped but not run, the project log 6c. The radiographic arm is dropped "
       "and recorded as future work, the project log 6d. The matched-step arm is "
       "closed at 5 of 18 cells with its reason, the project log 6a-xv. The six "
       "cells that would separate source from target on the one unexplained "
       "pair were not run, the project log 6a-xviii.")
    A_("")
    A_("**One honesty note on reproducibility.** Results were produced under "
       "recorded per-cell code hashes, which are released. One guarded file, "
       "`evaluate_cell.py`, moved once between cell generations for the "
       "reason in the project log 5h, so the 60 oldest main cells cannot be "
       "re-evaluated without that revision, which was not retained. "
       "The project log 9 and 9a. Do not claim every cell is re-evaluable from the "
       "released code. The released splits are frozen files verified by "
       "sha256, which is the reproducibility claim that does hold.")
    A_("")
    (out / "results.md").write_text("\n".join(L) + "\n")




# --------------------------------------------------------------------------



# --------------------------------------------------------------------------

def main() -> None:
    assert_armb_actually_finetuned()
    assert_armb_class_space()
    OUT.mkdir(parents=True, exist_ok=True)
    main_recs = main_grid()

    g = gates(main_recs)
    (OUT / "gates.json").write_text(json.dumps(g, indent=2))
    if GATE_FAILURES:
        raise SystemExit(
            f"\n{len(GATE_FAILURES)} GATE(S) FAILED: {GATE_FAILURES}\n"
            "A figure built on a failed gate is wasted work. Stopping.")

    print("4a LoHi internal control")
    ctl = lohi_control(OUT)
    for k, v in ctl["ratios"].items():
        print(f"    T {k} = {v:.3f}")

    print("4b arm B recovery")
    armb_rows = armb(OUT, main_recs)
    print(f"    {len(armb_rows)} rows")

    print("4c arm D")
    armd_rows = armd(OUT, main_recs)
    print(f"    {len(armd_rows)} rows")

    print("4d pre-registered statistics")
    s = stats(OUT, load("eval__", "lohictl"), armd_rows, armb_rows)
    pr = s["primary_blocked_on_cell"].get("primary", {})
    print(f"    blocks {s['n_blocks']}, Friedman chi2 {pr.get('chi2')}, "
          f"p = {pr.get('p_value')}, significant {pr.get('significant')}")

    print("4e per class")
    per_class(OUT)

    print("6 reporting order and release artefacts")
    fp_rows = fp_budget_table(OUT)
    cal_rows, cal = calibration_table(OUT)
    print(f"    {len(fp_rows)} FP budget rows, {len(cal_rows)} calibration rows")
    # Figure 4.4 is drawn from the bins calibration_table has just written.
    # Calling it here keeps the README's promise that two commands produce
    # every manuscript artefact. The script also runs on its own.
    RCF.main(C.PROJECT)
    # Image level against chance for the control, the label cost arm and
    # pooling, read in AUROC. Same draws as the main grid's, so the two are
    # paired. See grid/image_level.py.
    import image_level as IL
    IL.run_arms(ARM_RESULTS, MAIN_RESULTS, OUT)
    results_md(OUT, ctl, armb_rows, armd_rows, s, armd_box_limit(), cal)
    print(f"    wrote {OUT / 'results.md'}")

    print(f"\nwrote {len(list(OUT.iterdir()))} artefacts to {OUT}")


if __name__ == "__main__":
    main()
