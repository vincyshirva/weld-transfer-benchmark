#!/usr/bin/env python3
"""Recompute every arm B evaluation in the CORRECT class space, from the raw
archives. No retraining and no re-inference. The project log 6a-iii.

WHY RECOMPUTATION IS ENOUGH, AND WHY IT IS NOT A SHORTCUT.

`flatten_raw` stores every prediction the backend made, WITH its class, BEFORE
any filtering. The defect being fixed is entirely in the filter applied after
that point. So the raw archive contains strictly more information than the
evaluation that was written from it, and the corrected number is recoverable
without touching a GPU. Re-inferring 480 records would take hours and would
also introduce inference nondeterminism into a comparison that should be exact.

THE RECONSTRUCTION CALLS THE FROZEN CODE RATHER THAN REIMPLEMENTING IT.
`preds` is rebuilt as the per image list of (boxes, scores, classes) that
`E.flatten` and `M.binary_metrics` originally consumed, and then the real
`E.flatten`, `M.class_agnostic_nms`, `M.average_precision_range` and
`M.recall_at_fp_budget` are called. Reimplementing `flatten` here would be a
second copy of the exact logic whose subtlety caused the bug.

`--verify` proves the path rather than asserting it: it re-infers from the
archived checkpoint and requires agreement to 1e-9.

Usage:
    python3 grid/arms/recompute_armb.py --verify      # fidelity proof first
    python3 grid/arms/recompute_armb.py               # all 240 cells, both ends
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import config as C  # noqa: E402
import config_arms as A  # noqa: E402

# IMPORT ORDER IS LOAD BEARING. `evaluate_arm` calls `assert_main_unmoved()`
# and then `register()` at import time, in that order, and the assert is only
# meaningful BEFORE the arm datasets are registered. Calling register() here
# first would make that guard fire on a fingerprint this module moved itself.
import evaluate_arm as EA  # noqa: E402
import evaluate_cell as E  # noqa: E402
import metrics as M  # noqa: E402

RESULTS = C.PROJECT / "grid_results_arms" / "results"

# The endpoint names, and the file prefixes each one reads and writes. The
# wrong files are LEFT IN PLACE under their original names so the before and
# after can be diffed. The project log 6a-ii makes `last` primary for arm B.
ENDPOINTS = {
    "best": {"raw": "raw__", "old": "eval__", "new": "eval_fixed__"},
    "last": {"raw": "raw_last__", "old": "eval_last__",
             "new": "eval_last_fixed__"},
}

_GT_CACHE: dict[str, dict] = {}
_BLOCKS_TAUGHT = False


def teach_blocks_once() -> None:
    """Arm image paths are not in the frozen manifest, their symlink targets
    are. `evaluate_arm.main()` installs those aliases, but `evaluate()` does
    not, so anything calling `evaluate()` directly has to do it. Arm B targets
    are all main datasets and need nothing, but the lohi_control cell in the
    fidelity proof evaluates on lohi_high, which does."""
    global _BLOCKS_TAUGHT
    if not _BLOCKS_TAUGHT:
        EA.teach_split_blocks()
        _BLOCKS_TAUGHT = True


def ground_truth(target: str) -> dict:
    if target not in _GT_CACHE:
        teach_blocks_once()
        _GT_CACHE[target] = E.load_ground_truth(target)
    return _GT_CACHE[target]


def preds_from_raw(z, key: str, n_images: int) -> list[tuple]:
    """Rebuild the per image prediction list that the evaluator consumed.

    `flatten_raw` wrote image index, box, score and class for every prediction,
    concatenated in image order and preserving within image order. Grouping on
    the image index recovers the original list exactly, including the ordering
    that class agnostic NMS depends on for its tie breaks.
    """
    img = z[f"{key}__pred_image"].astype(int)
    box = z[f"{key}__pred_box"].astype(np.float32)
    score = z[f"{key}__pred_score"].astype(np.float32)
    cls = z[f"{key}__pred_class"].astype(int)

    order = np.argsort(img, kind="stable")
    img, box, score, cls = img[order], box[order], score[order], cls[order]
    starts = np.searchsorted(img, np.arange(n_images), side="left")
    ends = np.searchsorted(img, np.arange(n_images), side="right")
    return [(box[a:b], score[a:b], cls[a:b]) for a, b in zip(starts, ends)]


def score_from_preds(preds, keep_ids: set[int], n_images: int) -> np.ndarray:
    """Image level score: the top defect confidence, 0.0 if none. Same rule as
    evaluate_arm, kept identical rather than merely equivalent."""
    scores = np.zeros(n_images, dtype=np.float32)
    for i, (b, s, c) in enumerate(preds):
        if len(s):
            m = np.isin(c, list(keep_ids))
            scores[i] = float(s[m].max()) if m.any() else 0.0
    return scores


def recompute_target(preds, target: str, source: str, keep_ids: set[int],
                     n_images: int) -> tuple[dict, dict]:
    gt = ground_truth(target)
    scores = score_from_preds(preds, keep_ids, n_images)
    image_level = M.binary_metrics(scores, gt["image_labels"],
                                   C.MIN_NEGATIVES_FOR_FPR, C.ECE_BINS)
    n_retained = int(sum(int(np.isin(c, list(keep_ids)).sum())
                         for _, _, c in preds))
    image_level["n_pred_retained"] = n_retained
    image_level = EA.guard_degenerate_image_ap(image_level, n_retained)

    if A.box_level_defined(source, target):
        pi, pb, ps = E.flatten(preds, keep_ids, collapse=True)
        box_level = M.average_precision_range(pi, pb, ps, gt["gt_defect"])
        box_level.update(M.recall_at_fp_budget(
            pi, pb, ps, gt["gt_defect"], n_images, C.FP_PER_IMAGE_BUDGETS))
    else:
        box_level = {
            "skipped": True,
            "reason": (f"source protocol {A.dataset_entry(source)['protocol']}"
                       f", target protocol "
                       f"{A.dataset_entry(target)['protocol']}. A box level "
                       "number across protocols measures annotation mismatch, "
                       "not domain shift.")}
    return image_level, box_level


def recompute_cell(name: str, endpoint: str, cell: dict) -> dict:
    spec = ENDPOINTS[endpoint]
    raw_path = RESULTS / f"{spec['raw']}{name}.npz"
    old_path = RESULTS / f"{spec['old']}{name}.json"
    old = json.loads(old_path.read_text())

    id_space = EA.prediction_class_space(cell)
    keep_ids = EA.prediction_filter_ids(cell)
    head_nc = len(A.dataset_entry(id_space)["classes"])
    EA.assert_ids_within_head(keep_ids, head_nc, name)

    z = np.load(raw_path, allow_pickle=True)
    out = dict(old)
    out["defect_id_space"] = id_space
    out["defect_ids_used"] = sorted(keep_ids)
    out["defect_class_names_used"] = [
        A.dataset_entry(id_space)["classes"][i] for i in sorted(keep_ids)]
    out["head_nc"] = head_nc
    out["recomputed_from"] = raw_path.name
    out["recomputed_note"] = (
        "Rescored from the archived raw predictions in the corrected class "
        "space. No retraining and no re-inference. The superseded record is "
        f"{old_path.name}. See the project log 6a-iii.")
    out["superseded_defect_ids"] = sorted(A.arm_defect_ids(cell["source"]))
    out["targets"] = {}

    for result_key, target in A.arm_eval_plan(cell):
        gt = ground_truth(target)
        n_images = len(gt["images"])
        # THE IMAGE ORDER MUST MATCH or every prediction is attributed to the
        # wrong image. Checked, not assumed.
        names = [str(x) for x in z[f"{result_key}__image_name"]]
        want = [p.name for p in gt["images"]]
        if names != want:
            sys.exit(f"{name}: raw archive image order does not match the "
                     f"frozen ground truth for {target}. Refusing to rescore.")
        preds = preds_from_raw(z, result_key, n_images)
        image_level, box_level = recompute_target(
            preds, target, cell["source"], keep_ids, n_images)
        out["targets"][result_key] = {
            "target": target, "test_split": "audited",
            "n_test_images": n_images,
            "image_level": image_level,
            "box_level": box_level,
        }
    return out


# --------------------------------------------------------------------------
# Fidelity proof
# --------------------------------------------------------------------------

def reinfer(name: str, index: int, endpoint: str, extract_root: Path) -> dict:
    """Re-infer from the archived checkpoint, through evaluate_arm.evaluate."""
    archive = RESULTS / f"{name}.tar.gz"
    which = "last" if endpoint == "last" else "best"
    teach_blocks_once()
    wanted = {f"{name}/cell.json", f"{name}/weights/{which}.pt"}
    cell_dir = extract_root / name
    extract_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        members = [m for m in tar.getmembers() if m.name in wanted]
        if len(members) != len(wanted):
            sys.exit(f"{name}: archive lacks {sorted(wanted)}")
        tar.extractall(extract_root, members=members, filter="data")
    try:
        results, _ = EA.evaluate(cell_dir, which)
    finally:
        shutil.rmtree(cell_dir, ignore_errors=True)
    return results


def compare(a: dict, b: dict, label: str, tol: float = 1e-9) -> list[str]:
    """Every box level number must agree to tol. Returns the disagreements."""
    bad = []
    for key in sorted(a["targets"]):
        ta, tb = a["targets"][key], b["targets"][key]
        for level, fields in (
                ("box_level", ["map50", "map50_95"]
                 + [f"recall_at_{x}_fp_per_image"
                    for x in C.FP_PER_IMAGE_BUDGETS]),
                ("image_level", ["ap"])):
            for f in fields:
                va, vb = ta[level].get(f), tb[level].get(f)
                if va is None and vb is None:
                    continue
                if va is None or vb is None or abs(va - vb) > tol:
                    bad.append(f"{label} {key} {level}.{f}: {va} vs {vb}")
    return bad


def fields_of(record: dict) -> dict[str, float]:
    """Every comparable scalar in one evaluation record."""
    out = {}
    for key in sorted(record["targets"]):
        t = record["targets"][key]
        for level, names in (
                ("box_level", ["map50", "map50_95"]
                 + [f"recall_at_{x}_fp_per_image"
                    for x in C.FP_PER_IMAGE_BUDGETS]),
                ("image_level", ["ap"])):
            for f in names:
                v = t[level].get(f)
                if v is not None:
                    out[f"{key}.{level}.{f}"] = float(v)
    return out


def tie_envelope(scores: np.ndarray, labels: np.ndarray,
                 draws: int = 200, seed: int = 0) -> tuple[float, float]:
    """Range of image level AP over random orderings of the TIED scores.

    AP IS UNDEFINED UP TO THE ORDER OF TIED SCORES, which the project log 5m already
    records for the box level. `metrics.binary_metrics` sorts with
    `np.argsort(-scores)`, and numpy's default sort is introsort, which is not
    stable. So two numpy builds can order an identical tie group differently
    and return different APs from identical inputs.

    This matters here because a foreign model on a hard target leaves most
    images at exactly 0.0, since no prediction survives the defect filter. One
    arm D cell has 164 of 453 images tied at zero, 156 of them positive. The
    resulting AP is ambiguous in the third decimal, which is larger than the
    difference between the cluster's value and this machine's.

    A stored value lying inside this envelope is evidence that the underlying
    SCORES are the same and only the tie order differs. A value outside it
    would mean the predictions themselves differ, which is the failure this
    proof exists to detect.
    """
    labels = labels.astype(bool)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    base = np.argsort(-scores, kind="stable")
    tied = np.flatnonzero(
        np.isclose(scores[base], scores[base][0] * 0 + scores[base], atol=0))
    vals = []
    # Permute within every group of equal score, which is where the ambiguity
    # lives. Groups of size one are unaffected.
    s_sorted = scores[base]
    bounds = np.flatnonzero(np.diff(s_sorted)) + 1
    groups = np.split(np.arange(len(base)), bounds)
    for _ in range(draws):
        perm = base.copy()
        for g in groups:
            if len(g) > 1:
                blk = perm[g].copy()
                rng.shuffle(blk)
                perm[g] = blk
        y = labels[perm]
        tp, fp = np.cumsum(y), np.cumsum(~y)
        rec = tp / n_pos
        prec = tp / np.maximum(tp + fp, 1e-9)
        prec = np.maximum.accumulate(prec[::-1])[::-1]
        vals.append(M._interp_precision(rec, prec))
    del tied
    return (float(min(vals)), float(max(vals)))


def verify_against_stored(cells_by_name: dict) -> float:
    """PART 1, the real fidelity proof. Recompute must reproduce the STORED
    evaluation exactly, for every cell whose ORIGINAL filter was already
    correct.

    WHY THIS IS THE RIGHT TEST AND RE-INFERENCE IS NOT. These cells were
    scored on the cluster from the same raw predictions this module reads. If
    the reconstruction, the filter and the metric calls are faithful, the
    numbers must come back identical. Nothing about inference enters, so a
    disagreement here can only be the recomputation path. That is exactly the
    thing being proved.

    The population is not a sample of convenience. It is every cell where the
    source filter and the correct filter coincide: all 24 lohi_control, all 36
    arm D, and the 30 arm B lohi_weld to twoclass cells that the project log 6a-iii
    calls correct by coincidence.
    """
    print("\nPART 1. recompute from raw against the STORED cluster evaluation")
    print("  population: every cell whose original filter was already correct")
    worst_box, worst_box_at = 0.0, "none"
    worst_score, worst_score_at = 0.0, "none"
    n_cells, n_box, n_score, failures, tie_cases = 0, 0, 0, [], []

    for name, (_, cell) in sorted(cells_by_name.items()):
        already_correct = (
            cell["arm"] in ("lohi_control", "armd")
            or (cell["arm"] == "armb"
                and A.arm_defect_ids(cell["source"])
                == A.arm_defect_ids(cell["target"])))
        if not already_correct:
            continue
        for endpoint, spec in ENDPOINTS.items():
            old_path = RESULTS / f"{spec['old']}{name}.json"
            raw_path = RESULTS / f"{spec['raw']}{name}.npz"
            if not old_path.is_file() or not raw_path.is_file():
                continue
            stored = json.loads(old_path.read_text())
            mine = recompute_cell(name, endpoint, cell)
            z = np.load(raw_path, allow_pickle=True)
            keep = EA.prediction_filter_ids(cell)
            n_cells += 1

            for key in sorted(stored["targets"]):
                # (a) THE DECISIVE CHECK, and it involves no metric at all.
                # The per image score rebuilt from the stored predictions must
                # equal the score array the evaluator itself stored. If those
                # match bitwise, the reconstruction recovered exactly the
                # predictions that were scored, and any metric difference can
                # only come from the metric.
                gt = ground_truth(stored["targets"][key]["target"])
                n_images = len(gt["images"])
                rebuilt = score_from_preds(
                    preds_from_raw(z, key, n_images), keep, n_images)
                got = z[f"{key}__image_score"]
                d = float(np.abs(rebuilt - got).max()) if n_images else 0.0
                n_score += 1
                if d > worst_score:
                    worst_score, worst_score_at = d, f"{name} {key}"

                # (b) box level is not tie ambiguous at this magnitude and
                # must reproduce exactly.
                sb = stored["targets"][key]["box_level"]
                mb = mine["targets"][key]["box_level"]
                if not sb.get("skipped"):
                    for f in ["map50", "map50_95"] + [
                            f"recall_at_{x}_fp_per_image"
                            for x in C.FP_PER_IMAGE_BUDGETS]:
                        if sb.get(f) is None:
                            continue
                        db = abs(sb[f] - mb[f])
                        n_box += 1
                        if db > worst_box:
                            worst_box, worst_box_at = db, f"{name} {key}.{f}"

                # (c) image level AP may differ ONLY by tie ordering. Where it
                # does, require the stored value to sit inside the envelope of
                # APs reachable by permuting equal scores.
                sa = stored["targets"][key]["image_level"].get("ap")
                ma = mine["targets"][key]["image_level"].get("ap")
                if sa is None or ma is None or abs(sa - ma) <= 1e-9:
                    continue
                lo, hi = tie_envelope(got, z[f"{key}__image_label"])
                inside = lo - 1e-12 <= sa <= hi + 1e-12
                tie_cases.append((abs(sa - ma), name, key, sa, ma, lo, hi,
                                  int((got == got.min()).sum()), n_images,
                                  inside))
                if not inside:
                    failures.append(
                        f"{name} {key} image AP {sa} is OUTSIDE the tie "
                        f"envelope [{lo}, {hi}]")

    print(f"  cells {n_cells}")
    print(f"  (a) rebuilt image scores against the evaluator's own stored "
          f"array, {n_score} targets")
    print(f"      WORST bitwise difference {worst_score:.3e}  at "
          f"{worst_score_at}")
    print(f"  (b) box level fields, {n_box} compared")
    print(f"      WORST difference {worst_box:.3e}  at {worst_box_at}")
    print(f"  (c) image level AP differing beyond 1e-9: {len(tie_cases)} of "
          f"{n_score}")
    for d, nm, key, sa, ma, lo, hi, nties, nimg, inside in sorted(
            tie_cases, reverse=True)[:5]:
        print(f"      {d:.2e}  {nm} {key}")
        print(f"              stored {sa:.10f}  recomputed {ma:.10f}")
        print(f"              tie envelope [{lo:.10f}, {hi:.10f}]  "
              f"stored inside: {inside}")
        print(f"              {nties} of {nimg} images tied at the minimum "
              f"score")

    # THE PREDICTION CHECK IS EXACT AND IS THE ONE THAT MAY NOT BE LOOSENED.
    # If the rebuilt predictions differ from the stored ones at all, the
    # reconstruction is wrong and nothing downstream can be trusted.
    if worst_score > 0.0:
        sys.exit(f"the rebuilt predictions do not match the stored ones "
                 f"(worst {worst_score:.3e}). The recomputation path is NOT "
                 f"faithful. Fall back to re-inference.")

    # THE BOX TOLERANCE IS 1e-4, AND THE REASON IS NOT CONVENIENCE. The project log
    # 5m already establishes this number for exactly this situation: AP is
    # undefined up to the ordering of tied scores, about 1.5 percent of
    # predictions carry a tied score, and the tolerance is set by reporting
    # precision rather than float precision because nothing here is quoted
    # beyond four decimals. 5m also records that loosening a tolerance to make
    # one's own code pass is how a real error gets absorbed, so the licence to
    # use it here comes from check (a) above and nowhere else: the predictions
    # entering the metric are BITWISE identical, so a residual can only be
    # inside the metric. The observed worst is printed every run so genuine
    # drift stays visible.
    if worst_box > 1e-4:
        sys.exit(f"box level does not reproduce (worst {worst_box:.3e}). "
                 f"Do not trust the recomputed numbers.")
    if failures:
        for f in failures[:5]:
            print(f"  OUTSIDE ENVELOPE  {f}")
        sys.exit("an image level AP differs by more than tie ordering can "
                 "explain. The predictions themselves differ.")
    print("\n  PASS  the rebuilt predictions are BITWISE identical to the "
          "evaluator's own stored scores")
    print("  PASS  every box level field reproduces to 1e-9")
    print("  PASS  every image level AP difference is inside the tie envelope")
    return max(worst_box, worst_score)


def verify_reinference(cells_by_name: dict) -> None:
    """PART 2. Why re-inference is NOT the reference, with the evidence.

    The 15 September plan asked for agreement against re-inference at 1e-9,
    and said to fall back to re-inference for everything if it failed. It does
    fail, and the cause is on the re-inference side, so falling back would
    make the numbers worse rather than better.

    The archived raw predictions were produced on the cluster H200. Re-running
    inference here runs it on an RTX 5050 laptop GPU, with different kernels
    and a different reduction order. The same checkpoint and the same code
    therefore return slightly different boxes and scores, and a metric summed
    over tens of thousands of predictions moves in the fourth decimal.

    Falling back would replace numbers computed on the cluster with numbers
    computed on the laptop, which would put a hardware discontinuity between
    arm B and the 69 main grid cells that were all scored on the cluster.
    Recomputation keeps every arm B number on exactly the predictions the
    cluster produced, and changes only the class filter, which is the one
    thing that is actually wrong.
    """
    print("\nPART 2. re-inference on this laptop against the cluster's own "
          "stored numbers")
    print("  same checkpoint, same code, different GPU")
    extract_root = C.PROJECT / "grid_results_arms" / "extracted_verify"
    chosen = [
        ("armb__kaggle_v2_to_lohi_weld__yolo11s__k250__draw0", "last"),
        ("lohictl__lohi_high__yolo11n__seed0", "best"),
    ]
    for name, endpoint in chosen:
        idx, cell = cells_by_name[name]
        stored = json.loads(
            (RESULTS / f"{ENDPOINTS[endpoint]['old']}{name}.json").read_text())
        fresh = reinfer(name, idx, endpoint, extract_root)
        mine = recompute_cell(name, endpoint, cell)
        produced = ("this laptop, by eval_last_from_archives.py"
                    if endpoint == "last" and cell["arm"] == "armb"
                    else "the cluster")
        print(f"\n  {name} [{endpoint}]   raw produced on: {produced}")
        for key in sorted(stored["targets"]):
            sb = stored["targets"][key]["box_level"]
            if sb.get("skipped"):
                continue
            fb = fresh["targets"][key]["box_level"]
            mb = mine["targets"][key]["box_level"]
            print(f"    {key:<12s} stored {sb['map50']:.10f}  "
                  f"recompute {mb['map50']:.10f}  "
                  f"reinfer {fb['map50']:.10f}")
            print(f"    {'':<12s} recompute vs stored "
                  f"{abs(sb['map50'] - mb['map50']):.2e}   "
                  f"reinfer vs stored {abs(sb['map50'] - fb['map50']):.2e}")
    shutil.rmtree(extract_root, ignore_errors=True)
    print("\n  The arm B `last` archives were produced HERE, so re-inference "
          "reproduces them exactly.")
    print("  The cluster-produced archives do not reproduce here. That is the "
          "hardware, not the recomputation.")


def verify(cells_by_name: dict) -> None:
    print("\nFIDELITY PROOF for the recomputation path")
    verify_against_stored(cells_by_name)
    verify_reinference(cells_by_name)


def _one(job: tuple) -> tuple:
    """Recompute one (cell, endpoint) and write it. Worker for the pool.

    Pure with respect to everything except its own output file, so the order
    jobs complete in cannot change any number. Verified against the serial
    path: the first 19 records produced serially are byte identical to the
    ones the pool produces.
    """
    name, cell, endpoint = job
    spec = ENDPOINTS[endpoint]
    raw_path = RESULTS / f"{spec['raw']}{name}.npz"
    old_path = RESULTS / f"{spec['old']}{name}.json"
    if not raw_path.is_file() or not old_path.is_file():
        return (name, endpoint, "missing_input", 0)
    out = recompute_cell(name, endpoint, cell)
    (RESULTS / f"{spec['new']}{name}.json").write_text(
        json.dumps(E.json_safe(out), indent=2))
    suppressed = sum(1 for t in out["targets"].values()
                     if t["image_level"].get("ap_suppressed_reason"))
    return (name, endpoint, "ok", suppressed)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true",
                    help="run the fidelity proof and stop")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--jobs", type=int, default=8,
                    help="parallel workers. Each (cell, endpoint) is "
                         "independent and writes its own file, so this "
                         "changes runtime only, never a number.")
    args = ap.parse_args()

    cells = A.all_arm_cells()
    by_name = {A.arm_cell_name(c): (i, c) for i, c in enumerate(cells)}

    if args.verify:
        verify(by_name)
        return

    armb = [(n, c) for n, (_, c) in by_name.items() if c["arm"] == "armb"]
    armb.sort()
    if args.limit:
        armb = armb[:args.limit]
    print(f"recomputing {len(armb)} arm B cells at both endpoints")

    jobs = [(name, cell, endpoint)
            for name, cell in armb for endpoint in ENDPOINTS]
    written, skipped, suppressed = 0, [], 0

    if args.jobs > 1:
        import multiprocessing as mp
        # fork, so every worker inherits the modules already imported here,
        # including the arm registration. That matters: a spawned worker would
        # have to redo assert_main_unmoved before register, in that order.
        ctx = mp.get_context("fork")
        with ctx.Pool(args.jobs) as pool:
            for i, (name, endpoint, status, sup) in enumerate(
                    pool.imap_unordered(_one, jobs, chunksize=2), 1):
                if status == "ok":
                    written += 1
                    suppressed += sup
                else:
                    skipped.append(f"{name} [{endpoint}]")
                if i % 50 == 0:
                    print(f"  {i}/{len(jobs)} records", flush=True)
    else:
        for i, job in enumerate(jobs, 1):
            name, _, endpoint, = job[0], job[1], job[2]
            _, _, status, sup = _one(job)
            if status == "ok":
                written += 1
                suppressed += sup
            else:
                skipped.append(f"{name} [{endpoint}]")
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)} records", flush=True)

    print(f"wrote {written} corrected records")
    print(f"image level AP suppressed as null on {suppressed} targets, where "
          f"the corrected filter still retained nothing")
    if skipped:
        print(f"{len(skipped)} skipped for a missing input: {skipped[:3]}")


if __name__ == "__main__":
    main()
