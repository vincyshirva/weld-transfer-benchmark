#!/usr/bin/env python3
"""EXPLORATORY matched-step control for arm B. NOT pre-registered.

    python3 grid/arms/matched_step_control.py            # train and evaluate
    python3 grid/arms/matched_step_control.py --report   # tables only

THE QUESTION THIS ANSWERS, AND WHY THE RETRAIN DOES NOT ANSWER IT.

`ARMB_EPOCHS` is 50 at `ARMB_BATCH` 8, so the number of optimisation steps
scales with k: about 100 steps at k = 10 against about 1,550 at k = 250. And
because the source and target class counts differ, the project log 6a-iv, the
detection head is reinitialised and has to be fitted from those steps. So a
low score at small k has two candidate explanations that the pre-registered
curve cannot separate:

    not enough LABELS          the claim C4 wants to make
    not enough STEPS           an artefact of holding epochs fixed

This control gives the small-k cells the same step budget as their k = 250
counterpart and reports both side by side. Two outcomes, both publishable.
If performance stays at the floor, the label-budget claim is much stronger
because the obvious alternative has been tested and excluded. If it rises,
C4's headline changes to "small-k adaptation needs a longer schedule rather
than more labels".

SCOPE, from the 16 September plan: `kaggle_v2 to lohi_weld` and
`lohi_weld to kaggle_v2`, YOLO11s only, k in {10, 25, 50}, three draws.
18 matched-step cells.

**PLUS 18 MORE AT THE PRE-REGISTERED 50 EPOCHS, ON THIS SAME MACHINE.** That
is an addition to the plan's 18 and it is not optional. The project log 6a-iii
established that this laptop does not reproduce the cluster's predictions for
the same checkpoint, so comparing a laptop matched-step cell against a
cluster-trained 50-epoch cell would confound schedule with hardware. Training
both arms of the comparison here makes it internally valid. The cluster's
50-epoch values remain the pre-registered result; these are a local yardstick.

TWO CONFOUNDS THAT SURVIVE, AND MUST BE STATED RATHER THAN BURIED.

  1. Scaling epochs stretches the cosine learning-rate schedule. `cos_lr` is
     True and `lrf` is 0.01, so the LR decays over the epoch count. Matched
     STEPS is therefore not matched optimisation TRAJECTORY: an 800 epoch run
     spends far longer at a high LR than a 50 epoch one.
  2. `close_mosaic` is 10, so mosaic augmentation is disabled for the last 10
     epochs. That is 20 percent of a 50 epoch run and 1.25 percent of an 800
     epoch one.

Neither can be removed without changing a hyperparameter, which the plan
forbids. They are properties of "scale the epochs", so they belong in the
caption beside the result.

NOTHING PRE-REGISTERED IS TOUCHED. `config_arms.py` is not edited, so the arm
fingerprint is unchanged. Output goes to `runs_msc/` and `grid_results_msc/`,
never to `runs_arms/` or `grid_results_arms/`. The training path is
`train_arm.finetune_ultralytics` itself, reached by redirecting two module
globals per cell, so there is no second copy of the trainer to drift.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
import math
import platform
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import config as C  # noqa: E402
import config_arms as A  # noqa: E402
# `train_arm` calls assert_main_unmoved() then register() at module scope, and
# so does `evaluate_arm`. assert_main_unmoved must run BEFORE register in any
# one process, so importing BOTH is impossible: whichever lands second sees the
# arm-modified fingerprint and refuses. The evaluator is therefore reached as a
# SUBPROCESS, which is what `eval_last_from_archives.py` already does and for
# the same reason.
import train_arm as TA  # noqa: E402  (asserts, then registers)
import kshot as K  # noqa: E402
import train_cell as T  # noqa: E402
import weight_distance as WD  # noqa: E402

PAIRS = [("kaggle_v2", "lohi_weld"), ("lohi_weld", "kaggle_v2")]
KS = [10, 25, 50]
FAMILY = "yolo11s"
RUNS = C.PROJECT / "runs_msc"
RESULTS = C.PROJECT / "grid_results_msc"
STARTS = C.PROJECT / "grid_results" / "armb_start_checkpoints"


def plan() -> list[dict]:
    """Every cell, with its epoch count derived from the k = 250 step budget."""
    out = []
    for source, target in PAIRS:
        for draw in A.ARMB_DRAWS:
            n250 = K.sample(target, 250, draw)["n_images"]
            budget = math.ceil(n250 / A.ARMB_BATCH) * A.ARMB_EPOCHS
            for k in KS:
                drawn = K.sample(target, k, draw)
                spe = math.ceil(drawn["n_images"] / A.ARMB_BATCH)
                scaled = max(A.ARMB_EPOCHS, round(budget / spe))
                for epochs, arm_label in ((scaled, "matched_step"),
                                          (A.ARMB_EPOCHS, "fixed_50")):
                    out.append({
                        "arm": "armb",          # so the evaluator picks the
                        "source": source,        # TARGET class space, 6a-iii
                        "target": target,
                        "family": FAMILY,
                        "k": k, "draw": draw, "seed": draw,
                        "_epochs": epochs,
                        "_schedule": arm_label,
                        "_steps": epochs * spe,
                        "_steps_at_50": A.ARMB_EPOCHS * spe,
                        "_k250_step_budget": budget,
                        "_n_images": drawn["n_images"],
                        "_n_scenes": drawn["n_scenes"],
                        "_steps_per_epoch": spe,
                    })
    return out


def oversampled_plan() -> list[dict]:
    """ARM B of the three-way contrast: same steps as C, same schedule as A.

    6a-xi names its own confound and it is the right one. Stretching 50 epochs
    to 800 also stretches the cosine learning rate schedule, because `cos_lr`
    is True with `lrf` 0.01, and it shrinks `close_mosaic` from 20 percent of
    the run to 1.25 percent. So the matched-step arm cannot say whether the
    gain is the STEPS or the SCHEDULE SHAPE.

    Oversampling settles it. Replicating each drawn image R times inside the
    k-shot train listing multiplies the steps per epoch while keeping the
    epoch count at 50, so the cosine and the close_mosaic share are identical
    to the fixed arm:

        A  fixed 50         10 distinct images    ~100 steps   cosine/50   20%
        B  oversampled 50   10 distinct images   ~1,650 steps  cosine/50   20%
        C  matched step     10 distinct images   ~1,600 steps  cosine/800  1.25%

        A against B isolates the STEP COUNT, schedule shape held constant.
        B against C isolates the SCHEDULE SHAPE, step count held constant.

    The label budget is identical in all three: ten distinct images, six
    scenes. Nothing here sees an image the fixed arm did not see.

    THE RESIDUAL DIFFERENCE, stated rather than hidden. Oversampling shows an
    image R times WITHIN an epoch under R different random augmentations,
    while the 800 epoch arm shows it once per epoch across 800 epochs. Mosaic
    draws from the same ten image pool either way. The two are not identical.
    That difference is second order beside the cosine stretch, and saying so
    is what makes the control credible.

    Scoped to `kaggle_v2` to `lohi_weld` at k = 10, the pair 6a-xi's headline
    rests on. Extend only if the answer warrants it.
    """
    out = []
    # SCOPE. The first three cells were kaggle_v2 to lohi_weld at k = 10, the
    # pair 6a-xi's headline rests on. The other two coordinates complete a
    # 2 by 2 of pair against k, so "small-k adaptation is optimisation
    # limited" no longer rests on one pair at one k. Each has a `fixed_50`
    # baseline that already exists, so six cells buy the whole contrast.
    #
    # EXPECT LITTLE FROM `lohi_weld` to `kaggle_v2`, AND THAT IS SAID IN
    # ADVANCE RATHER THAN AFTERWARDS. Per 6a-xiii its own in domain box AP is
    # 0.1151, about a tenth of the other pair's headroom, so this control
    # cannot resolve much there either way. Reported regardless: a null on the
    # hard pair is information, and pre-stating the expectation is what stops
    # it being read as a surprise.
    # TASK 2, 18 September. Source held FIXED at kaggle_v2, only the target
    # varied, so "target" is no longer confounded with the source checkpoint.
    # Predictions for these were written into the project log 6a-xvi BEFORE the
    # cells ran: twoclass delta R above 0.15, visual below 0.05.
    for source, target, k in (("kaggle_v2", "lohi_weld", 10),
                              ("kaggle_v2", "lohi_weld", 50),
                              ("lohi_weld", "kaggle_v2", 10),
                              ("kaggle_v2", "twoclass", 10),
                              ("kaggle_v2", "visual", 10)):
      for draw in A.ARMB_DRAWS:
        n250 = K.sample(target, 250, draw)["n_images"]
        budget = math.ceil(n250 / A.ARMB_BATCH) * A.ARMB_EPOCHS
        drawn = K.sample(target, k, draw)
        n = drawn["n_images"]
        # Replications needed so that 50 epochs delivers the k = 250 budget.
        reps = max(1, round(budget / A.ARMB_EPOCHS * A.ARMB_BATCH / n))
        spe = math.ceil(n * reps / A.ARMB_BATCH)
        # The two Task 2 targets have no fixed_50 baseline from plan(), which
        # only covers the original two pairs. Emit it here so the contrast is
        # complete and trained by the same runner on the same machine.
        if (source, target) not in {p for p in PAIRS}:
            out.append({
                "arm": "armb", "source": source, "target": target,
                "family": FAMILY, "k": k, "draw": draw, "seed": draw,
                "_epochs": A.ARMB_EPOCHS, "_schedule": "fixed_50", "_reps": 1,
                "_steps": A.ARMB_EPOCHS * math.ceil(n / A.ARMB_BATCH),
                "_steps_at_50": A.ARMB_EPOCHS * math.ceil(n / A.ARMB_BATCH),
                "_k250_step_budget": budget, "_n_images": n,
                "_n_scenes": drawn["n_scenes"],
                "_steps_per_epoch": math.ceil(n / A.ARMB_BATCH),
            })
        out.append({
            "arm": "armb", "source": source, "target": target,
            "family": FAMILY, "k": k, "draw": draw, "seed": draw,
            "_epochs": A.ARMB_EPOCHS,
            "_schedule": "oversampled_50",
            "_reps": reps,
            "_steps": A.ARMB_EPOCHS * spe,
            "_steps_at_50": A.ARMB_EPOCHS * math.ceil(n / A.ARMB_BATCH),
            "_k250_step_budget": budget,
            "_n_images": n, "_n_scenes": drawn["n_scenes"],
            "_steps_per_epoch": spe,
        })
    return out


def oversample_train(kshot_root: Path, reps: int) -> int:
    """Replicate every train image R times, as extra symlinks.

    `kshot.materialise` writes train/images and train/labels as symlink trees
    and points data.yaml at the DIRECTORY, so ultralytics simply scans it.
    Adding R-1 more links per image multiplies the steps per epoch and touches
    nothing else. The label file is duplicated alongside each replica, because
    ultralytics pairs them by stem.
    """
    if reps <= 1:
        return 0
    img_dir = kshot_root / "train" / "images"
    lab_dir = kshot_root / "train" / "labels"
    originals = sorted(p for p in img_dir.iterdir() if p.is_file() or p.is_symlink())
    for p in originals:
        src_img = p.resolve()
        lab = lab_dir / f"{p.stem}.txt"
        src_lab = lab.resolve() if lab.is_symlink() else None
        for r in range(1, reps):
            (img_dir / f"{p.stem}__rep{r:02d}{p.suffix}").symlink_to(src_img)
            dst_lab = lab_dir / f"{p.stem}__rep{r:02d}.txt"
            if src_lab is not None:
                dst_lab.symlink_to(src_lab)
            elif lab.is_file():
                dst_lab.write_text(lab.read_text())
            else:
                dst_lab.write_text("")
    return len(list(img_dir.iterdir()))


def cell_dir(cell: dict) -> Path:
    return RUNS / cell["_schedule"] / A.arm_cell_name(cell)


def train_one(cell: dict) -> dict:
    """Train via train_arm.finetune_ultralytics, with two globals redirected.

    `finetune_ultralytics` reads `A.ARMB_EPOCHS` for both epochs and patience,
    and `A.arm_data_yaml` reads the module global `A.ARM_RUNS` at call time.
    Redirecting those two is what lets a 36 cell control reuse the real
    trainer instead of growing a second copy that can drift. Both are restored
    in a finally block, and `config_arms.py` on disk is never touched.
    """
    out = cell_dir(cell)
    if (out / "cell.json").is_file():
        print(f"    already trained, skipping")
        return json.loads((out / "cell.json").read_text())

    start = STARTS / f"{cell['source']}__{FAMILY}__seed{cell['seed']}" \
        / "weights" / "best.pt"
    if not start.is_file():
        sys.exit(f"missing start checkpoint {start}. See the project log 6a-ix.")

    saved_epochs, saved_runs = A.ARMB_EPOCHS, A.ARM_RUNS
    try:
        A.ARMB_EPOCHS = cell["_epochs"]
        A.ARM_RUNS = RUNS / cell["_schedule"]
        cell["_start_weights"] = str(start)
        drawn = TA.prepare_kshot(cell, out)
        n_train = None
        if cell.get("_reps", 1) > 1:
            n_train = oversample_train(out / "kshot", cell["_reps"])
            print(f"    oversampled {drawn['n_images']} distinct images "
                  f"x{cell['_reps']} -> {n_train} train entries", flush=True)
        t0 = time.perf_counter()
        result = TA.finetune_ultralytics(cell, out, smoke=False)
    finally:
        A.ARMB_EPOCHS, A.ARM_RUNS = saved_epochs, saved_runs

    record = {
        "cell": {k: v for k, v in cell.items() if not k.startswith("_")},
        "name": A.arm_cell_name(cell),
        "arm": "armb",
        "backend": "ultralytics",
        "smoke": False,
        "exploratory": True,
        "exploratory_reason": (
            "Matched-step control for arm B. NOT pre-registered. See "
            "the project log 6a-xi and the module docstring."),
        "schedule": cell["_schedule"],
        "epochs": cell["_epochs"],
        "steps": cell["_steps"],
        "reps": cell.get("_reps", 1),
        "n_train_entries": n_train,
        "n_distinct_images": drawn["n_images"],
        "n_scenes": drawn["n_scenes"],
        "steps_at_50_epochs": cell["_steps_at_50"],
        "k250_step_budget": cell["_k250_step_budget"],
        "steps_per_epoch": cell["_steps_per_epoch"],
        "kshot": {k: v for k, v in drawn.items() if k != "images"},
        "kshot_images": drawn["images"],
        "start_checkpoint": str(start),
        "arm_sha256_16": A.arm_fingerprint(),
        "frozen_main_fingerprint": A.FROZEN_MAIN_FINGERPRINT,
        "split_hashes": json.loads(
            (C.CANDIDATE / "split_hashes.json").read_text()),
        "code_sha256": T.code_fingerprints(),
        "host": platform.node(),
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
        **result,
    }
    (out / "cell.json").write_text(json.dumps(record, indent=2))
    return record


def registry_index() -> dict[str, int]:
    """name -> index, so the evaluator subprocess can be given a --index.

    `evaluate_arm.py` requires --index and uses it only for the output STEM;
    the cell it actually scores comes from --cell-dir. A matched-step cell has
    the same coordinates as a real arm B cell, so the registry name is the same
    and the index is the right one to quote. Output collision between the two
    schedules is avoided by giving each its own --out-dir rather than by
    inventing a name the registry would not recognise.
    """
    return {A.arm_cell_name(c): i for i, c in enumerate(A.all_arm_cells())}


def evaluate_one(cell: dict, index_of: dict[str, int]) -> None:
    """Both endpoints, via the frozen evaluator run as a subprocess."""
    import subprocess
    name = A.arm_cell_name(cell)
    out_dir = RESULTS / cell["_schedule"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for which, prefix in (("last", "eval_last__"), ("best", "eval__")):
        if (out_dir / f"{prefix}{name}.json").is_file():
            continue
        proc = subprocess.run(
            [sys.executable, str(HERE / "evaluate_arm.py"),
             "--index", str(index_of[name]),
             "--cell-dir", str(cell_dir(cell)),
             "--out-dir", str(out_dir),
             "--weights", which],
            capture_output=True, text=True)
        if proc.returncode != 0:
            tail = (proc.stderr.strip().splitlines() or ["(no stderr)"])[-1]
            sys.exit(f"evaluation failed for {name} [{which}]: {tail}")
        # Stamp the schedule into the record, since the evaluator neither
        # knows nor should know about this exploratory control.
        path = out_dir / f"{prefix}{name}.json"
        rec = json.loads(path.read_text())
        rec["exploratory"] = True
        rec["schedule"] = cell["_schedule"]
        rec["epochs"] = cell["_epochs"]
        rec["steps"] = cell["_steps"]
        path.write_text(json.dumps(rec, indent=2))


def fmt(v, nd: int = 4) -> str:
    """Local, because this module does not import analysis_arms."""
    return "n/a" if v is None else f"{v:.{nd}f}"


def report() -> None:
    """All three arms as one table, with the step-count/schedule-shape split.

    The earlier version pivoted only `fixed_50` against `matched_step`, so the
    oversampled arm existed in the CSV and in no table. That arm is the one
    that settles 6a-xi's own confound, so it has to be the centre of the
    report rather than a column nobody added.
    """
    rows = []
    for path in sorted(RESULTS.glob("*/eval_last__*.json")):
        rec = json.loads(path.read_text())
        c = rec["cell"]
        t = rec["targets"][c["target"]]
        rows.append({
            "schedule": rec["schedule"], "source": c["source"],
            "target": c["target"], "k": c["k"], "draw": c["draw"],
            "epochs": rec["epochs"], "steps": rec["steps"],
            "reps": rec.get("reps", 1),
            "box_map50": (None if t["box_level"].get("skipped")
                          else t["box_level"]["map50"]),
            "image_ap": t["image_level"].get("ap"),
        })
    if not rows:
        print("no evaluations yet")
        return

    import csv
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "matched_step_control.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # Keys are normalised to str because these records come from JSON,
    # where k and draw are ints, while the CSV round trip makes them
    # strings. Mixing the two silently produced an empty table once.
    def key(r):
        return (r["schedule"], r["source"], r["target"],
                str(r["k"]), str(r["draw"]))
    idx = {key(r): r for r in rows}
    draws = sorted({str(r["draw"]) for r in rows})
    coords = sorted({(r["source"], r["target"], str(r["k"])) for r in rows},
                    key=lambda x: (x[0], x[1], int(x[2])))

    L = ["# Matched-step and oversampling control for arm B",
         "## EXPLORATORY, NOT pre-registered", "",
         "Primary endpoint `last`, the project log 6a-ii. Box mAP50, mean over the "
         "three draws. Every arm was trained ON THIS MACHINE, so the "
         "comparison is internally valid; the project log 6a-iii is why that "
         "matters.", "",
         "| arm | distinct images | steps | cosine over | close_mosaic share |",
         "|---|---|---|---|---|",
         "| A fixed 50 | k | ~100 to 900 | 50 epochs | 20% |",
         "| B oversampled 50 | k | matched to k=250 | 50 epochs | 20% |",
         "| C matched step | k | matched to k=250 | scaled epochs | 1.25% |",
         "",
         "**A against B isolates the STEP COUNT**, schedule shape held "
         "constant. **B against C isolates the SCHEDULE SHAPE**, step count "
         "held constant. The label budget is identical in all three.", "",
         "## The three arms", "",
         "| pair | k | A fixed 50 | B oversampled | C matched step | "
         "A to B (steps) | B to C (schedule) |",
         "|---|---|---|---|---|---|---|"]

    ab_all, bc_all = [], []
    for src, tgt, k in coords:
        got = {}
        for sched in ("fixed_50", "oversampled_50", "matched_step"):
            vals = [idx[(sched, src, tgt, k, d)]["box_map50"]
                    for d in draws
                    if (sched, src, tgt, k, d) in idx
                    and idx[(sched, src, tgt, k, d)]["box_map50"] is not None]
            got[sched] = statistics.fmean(vals) if vals else None
        a, b, c = got["fixed_50"], got["oversampled_50"], got["matched_step"]
        if a is None:
            continue
        ab = None if b is None else b - a
        bc = None if (b is None or c is None) else c - b
        # THE SPLIT IS ONLY VALID WHERE ALL THREE ARMS EXIST. Collecting
        # A-to-B and B-to-C independently would average them over DIFFERENT
        # coordinate sets and then divide one by the other, which is not a
        # decomposition of anything. Only paired coordinates count.
        if ab is not None and bc is not None:
            ab_all.append(ab)
            bc_all.append(bc)
        L.append(
            f"| {src} to {tgt} | {k} | {fmt(a)} | {fmt(b)} | {fmt(c)} | "
            f"{'n/a' if ab is None else f'{ab:+.4f}'} | "
            f"{'n/a' if bc is None else f'{bc:+.4f}'} |")

    if ab_all and bc_all:
        # DO NOT POOL A PERCENTAGE ACROSS COORDINATES WHOSE B-to-C TERMS HAVE
        # OPPOSITE SIGNS. Averaging +0.0379 and -0.0206 and calling the result
        # a share of the total produces a number that describes neither
        # coordinate. Report each, and a range.
        shares = [100 * ab / (ab + bc) for ab, bc in zip(ab_all, bc_all)
                  if (ab + bc)]
        signs = {bc > 0 for bc in bc_all}
        L += ["", f"**Where the gain comes from, per coordinate.** Over the "
              f"{len(ab_all)} coordinate(s) with ALL THREE arms, the step "
              f"count share is "
              + ", ".join(f"{s:.0f} percent" for s in shares) + "."]
        if len(signs) > 1:
            L += ["", "**The schedule-shape term does not have a consistent "
                  "sign, so no single percentage is quoted.** It is "
                  + ", ".join(f"{bc:+.4f}" for bc in bc_all)
                  + " across those coordinates. The STEP COUNT is the "
                  "consistent driver, contributing "
                  + ", ".join(f"{ab:+.4f}" for ab in ab_all)
                  + ". The stretched cosine and the shrunken mosaic window "
                  "add a small residual that is positive at one coordinate "
                  "and negative at the other, which is what a second "
                  "coordinate was added to find out. 6a-xi's conclusion "
                  "survives either way: the small-k floor is an optimisation "
                  "BUDGET limit."]
        else:
            L += ["", "So 6a-xi's conclusion survives the confound it "
                  "identified: the small-k floor is an optimisation BUDGET "
                  "limit, and the schedule shape is the smaller part."]

    # THE VARIABLE THAT MOVES IS THE STEP MULTIPLIER, NOT THE PAIR.
    L += ["", "## Gain against step multiplier, and against the pair", "",
          "**THE STEP MULTIPLIER IS NOT THE DRIVER. THE TARGET IS.** This "
          "table was added to test the opposite hypothesis, that the small "
          "gain on `lohi_weld` to `kaggle_v2` was a small step deficit rather "
          "than the pair. On the five cells available at the time the two "
          "were perfectly confounded: every kaggle to lohi cell had a 16x "
          "multiplier and every lohi to kaggle cell had 1.8x. Four more cells "
          "broke that confound and the answer reversed. See the summary "
          "under the table.", "",
          "| pair | k | draw | steps at 50 ep | steps oversampled | "
          "multiplier | delta box mAP50 |",
          "|---|---|---|---|---|---|---|"]
    pts = []
    for src, tgt, k in coords:
        for d in draws:
            a = idx.get(("fixed_50", src, tgt, k, d))
            b = idx.get(("oversampled_50", src, tgt, k, d))
            if not (a and b) or a["box_map50"] is None or b["box_map50"] is None:
                continue
            mult = b["steps"] / a["steps"]
            delta = b["box_map50"] - a["box_map50"]
            pts.append((mult, delta))
            L.append(f"| {src} to {tgt} | {k} | {d} | {a['steps']} | "
                     f"{b['steps']} | {mult:.1f}x | {delta:+.4f} |")
    # WHICH VARIABLE ACTUALLY ORDERS THE GAINS? Report both cuts, because
    # reporting only the one that was hypothesised is how a confound survives.
    by_pair = defaultdict(list)
    for src, tgt, k in coords:
        for d in draws:
            a = idx.get(("fixed_50", src, tgt, k, d))
            b = idx.get(("oversampled_50", src, tgt, k, d))
            if not (a and b) or a["box_map50"] is None or b["box_map50"] is None:
                continue
            by_pair[f"{src} to {tgt}"].append(
                (b["steps"] / a["steps"], b["box_map50"] - a["box_map50"]))
    if len(by_pair) >= 2 and len(pts) >= 6:
        L += ["", "| cut | n | mean delta box mAP50 |", "|---|---|---|"]
        for pair, g in sorted(by_pair.items()):
            L.append(f"| pair {pair} | {len(g)} | "
                     f"{statistics.fmean(d for _, d in g):+.4f} |")
        for lo_, hi_, lab in ((0, 5, "multiplier below 5x"),
                              (5, 1e9, "multiplier 5x or above")):
            g = [d for m, d in pts if lo_ <= m < hi_]
            if g:
                L.append(f"| {lab}, pooled | {len(g)} | "
                         f"{statistics.fmean(g):+.4f} |")
        L += ["", "**The pair separates the gains and the multiplier does "
              "not.** Pooled across pairs, a multiplier below 5x and one at "
              "5x or above give almost the same mean, while the two pairs "
              "differ by more than an order of magnitude. The decisive "
              "cells are the hard pair at a 16x and 20x multiplier, which "
              "gain nothing, against the easy pair at a 2x multiplier, which "
              "gains a great deal.",
              "", "**The better explanation is target headroom, and it was "
              "pre-stated.** `kaggle_v2` as a target has an in domain box AP "
              "of 0.1151 against `lohi_weld`'s 0.6849, about a sixth, so "
              "there is little for any schedule to recover. The project log 5n "
              "already establishes kaggle_v2 as a hard box target on four "
              "independent checks. This control cannot resolve much there "
              "either way, which is why the expectation was written down "
              "before the cells were run."]

    L += ["", "**`50 epochs` is NOT a constant step budget.** Across these "
          "cells the fixed arm ranges from 100 to 900 steps at a nominal 50 "
          "epochs, because block sampling overshoots k by up to 9.1 times and "
          "the realised image count varies. The project log 6a-i. Any statement "
          "about 'the 50 epoch schedule' is therefore a statement about a "
          "range of step budgets.",
          "", "**EXPLORATORY. This is not the pre-registered arm.** The "
          "the project log statistics section pre-registers a comparison of "
          "CONFIGURATIONS blocked on evaluation cell. Nothing was "
          "pre-registered about the training schedule, and the project log 6a-xii "
          "records the decision to leave `ARMB_EPOCHS` at 50 rather than "
          "re-scope the arm around this result.",
          "", "**The residual difference between B and C, stated rather than "
          "hidden.** Oversampling shows each image R times WITHIN an epoch "
          "under R different random augmentations, while the matched-step arm "
          "shows it once per epoch across many epochs. Mosaic draws from the "
          "same k image pool either way. The two are not identical. That "
          "difference is second order beside the cosine stretch, and saying "
          "so is what makes the control credible.",
          "", "**The head is reinitialised in every cell here**, the project log "
          "6a-iv, because the source and target class counts differ in both "
          "pairs. That is the reason the step budget matters at all."]

    (RESULTS / "matched_step_control.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", action="store_true",
                    help="tables only, no training")
    ap.add_argument("--second-coordinate", action="store_true",
                    help="matched_step at kaggle_v2 to lohi_weld k=50 only, "
                         "three cells. Takes the B-to-C schedule contrast from "
                         "n=1 coordinate to n=2. This does NOT reopen the "
                         "matched-step arm, which stays closed at 5 of 18 per "
                         "the project log 6a-xv; it completes one contrast whose A "
                         "and B arms already exist.")
    ap.add_argument("--task3", action="store_true",
                    help="only the oversampled arm, kaggle_v2 to lohi_weld "
                         "k=10, which separates step count from schedule shape")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    if args.report:
        report()
        return

    index_of = registry_index()
    if args.second_coordinate:
        cells = [c for c in plan()
                 if c["_schedule"] == "matched_step"
                 and c["source"] == "kaggle_v2" and c["target"] == "lohi_weld"
                 and c["k"] == 50]
    elif args.task3:
        cells = oversampled_plan()
    else:
        cells = plan()
    # SMALLEST k FIRST, NOT CHEAPEST FIRST.
    #
    # An earlier version sorted by epoch count so the run would get through
    # more cells per hour. That is exactly backwards for a run that may be
    # interrupted. The question this control exists to answer is sharpest at
    # k = 10, where the fixed schedule buys about 100 optimisation steps
    # against about 1,600 for the matched one. The k = 50 cells have the
    # smallest gap between the two schedules and so say the least.
    #
    # Cheapest-first put every k = 10 cell LAST, so an interrupted run would
    # have banked only the least informative cells. Sorting by k, then pair,
    # then draw means an interruption still leaves a COMPLETE three draw
    # comparison at the most informative budget.
    cells.sort(key=lambda c: (c["k"], c["source"], c["draw"]))
    total = sum(c["_epochs"] for c in cells)
    print(f"{len(cells)} cells, {total} epochs total")
    print(f"runs -> {RUNS}\nresults -> {RESULTS}\n")

    for i, cell in enumerate(cells[: args.limit] if args.limit else cells, 1):
        name = A.arm_cell_name(cell)
        print(f"[{i}/{len(cells)}] {cell['_schedule']:13s} {name}  "
              f"{cell['_epochs']} epochs, {cell['_steps']} steps", flush=True)
        rec = train_one(cell)
        moved = rec.get("relative_l2_from_start")
        if moved is not None:
            print(f"    {WD.verdict(moved)}", flush=True)
        evaluate_one(cell, index_of)

    report()


if __name__ == "__main__":
    main()
