#!/usr/bin/env python3
"""Frozen configuration for the arms, layered on top of the main grid.

WHAT IS FROZEN HERE AND WHAT IS INHERITED

Everything that decides how a model is fitted or scored is INHERITED from
`grid/config.py` unchanged: image size, batch, epochs, workers, the per-family
hyperparameters, the pretrained checkpoints, the confidence floor, the NMS IoU,
the detection cap, the ECE bins and the bootstrap settings. The arms add
datasets, cells and an evaluation plan. They add no hyperparameter and they
retune nothing. That is what makes an arm number comparable with a main grid
number.

THE ARM FINGERPRINT EMBEDS THE MAIN ONE

`arm_fingerprint()` is a sha256 over the frozen main fingerprint plus the arm
payload. So an arm cell records both. If the main configuration ever moves, the
arm fingerprint moves with it and every arm cell becomes un-re-evaluable, which
is the same guard the main grid has, extended rather than weakened.

THE THREE ARMS

  lohi_control  the project log 7.  2 acquisitions x 4 configurations x 3 seeds = 24
                cells, each evaluated on BOTH acquisitions, so the 2 by 2
                matrix including diagonals costs no extra training.
  armb          the project log 6a. k-shot fine-tuning. 2 configurations x 2 starting
                checkpoints x 4 targets x 5 values of k x 3 draws = 240 cells.
  armd          the project log 6b. Leave one dataset out. 3 pools x 4 configurations
                x 3 seeds = 36 cells, each evaluated on the source it excluded
                plus the two test-only targets.

Those scopes are PRE-REGISTERED. Do not edit them here to match an
implementation convenience. If something cannot be built as written, that is a
finding to record in the project log, not a scope to quietly change.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

GRID = Path(__file__).resolve().parent.parent
if str(GRID) not in sys.path:
    sys.path.insert(0, str(GRID))

import config as C  # noqa: E402

# The value every one of the 69 main cells recorded. If this ever fails, the
# main grid has moved and no arm result is comparable with a main grid result.
FROZEN_MAIN_FINGERPRINT = "f5567a1938278e88"

ARMS_ROOT = Path(__file__).resolve().parent
ARM_RUNS = C.PROJECT / "runs_arms"
ARM_RESULTS = C.PROJECT / "results_arms"

# --------------------------------------------------------------------------
# Arm datasets
# --------------------------------------------------------------------------
# lohi_high and lohi_low are the two LoHi-WELD acquisitions, separated by
# filename. They are NOT new data and they are NOT a new split. They are the
# frozen lohi_weld split lists filtered in place, and the project log 7 records the
# measured fact that makes that legitimate: zero of the 1,977 lohi_weld split
# blocks contain images from both acquisitions, so the filter cuts along
# boundaries that already exist. build_arm_datasets.py re-checks this every
# time it runs rather than trusting the note.
#
# Class list, protocol and defect classes are inherited from lohi_weld exactly,
# because they ARE lohi_weld images. That is the entire point of the control:
# annotation protocol, annotators, class list and production line are held
# constant so the only thing varying is the acquisition.

LOHI_ACQUISITIONS = ["lohi_high", "lohi_low"]

# Arm D pools. Each is named for the training source it EXCLUDES, because that
# is the held-out target and the thing a reader will look for.
ARMD_POOLS = {
    f"pool_no_{held_out}": [s for s in C.TRAIN_SOURCES if s != held_out]
    for held_out in C.TRAIN_SOURCES
}

# A pooled dataset inherits the union of its members' class lists. Order is
# fixed by the order in C.TRAIN_SOURCES then by each source's own class order,
# so the mapping is deterministic and reproducible from this file alone.
def pool_classes(members: list[str]) -> list[str]:
    names = []
    for tag in members:
        for cls in C.DATASETS[tag]["classes"]:
            qualified = f"{tag}:{cls}"
            if qualified not in names:
                names.append(qualified)
    return names


def pool_non_defect(members: list[str]) -> set[str]:
    out = set()
    for tag in members:
        for cls in C.DATASETS[tag]["non_defect"]:
            out.add(f"{tag}:{cls}")
    return out


def pool_excluded(members: list[str]) -> set[str]:
    out = set()
    for tag in members:
        for cls in C.DATASETS[tag]["excluded"]:
            out.add(f"{tag}:{cls}")
    return out


def _build_arm_datasets() -> dict:
    entries = {}
    for tag in LOHI_ACQUISITIONS:
        parent = C.DATASETS["lohi_weld"]
        entries[tag] = {
            "classes": list(parent["classes"]),
            "non_defect": set(parent["non_defect"]),
            "excluded": set(parent["excluded"]),
            "protocol": parent["protocol"],
            "role": "arm",
            "arm_parent": "lohi_weld",
            "arm_kind": "acquisition_subset",
        }
    for pool, members in ARMD_POOLS.items():
        entries[pool] = {
            "classes": pool_classes(members),
            "non_defect": pool_non_defect(members),
            "excluded": pool_excluded(members),
            # A pool containing weld_surface_v2 mixes whole-region verdict
            # boxes with localised defects. The project log 6b: arm D reports image
            # level for every fold and box level ONLY for the pool that
            # excludes weld_surface_v2. The protocol recorded here is what
            # enforces that downstream.
            "protocol": ("localisation"
                         if all(C.DATASETS[m]["protocol"] == "localisation"
                                for m in members)
                         else "mixed"),
            "role": "arm",
            "arm_parent": None,
            "arm_kind": "pool",
            "arm_members": list(members),
        }
    return entries


ARM_DATASETS = _build_arm_datasets()

# The held-out target for each pool, derived from the name so the two can never
# disagree.
POOL_HELD_OUT = {pool: pool[len("pool_no_"):] for pool in ARMD_POOLS}


def register() -> None:
    """Add the arm datasets to config.DATASETS IN THIS PROCESS ONLY.

    config.py on disk is untouched. A main-grid evaluation runs in a separate
    process which never imports this package, so its DATASETS still holds
    exactly five entries and `config_fingerprint()` still returns
    f5567a1938278e88. preflight_arms asserts that in a subprocess.

    Idempotent, so importing the package twice is harmless.
    """
    for tag, entry in ARM_DATASETS.items():
        C.DATASETS.setdefault(tag, entry)


def dataset_entry(tag: str) -> dict:
    if tag in ARM_DATASETS:
        return ARM_DATASETS[tag]
    return C.DATASETS[tag]


def arm_defect_ids(tag: str) -> set[int]:
    """Defect class indices for an arm or main dataset."""
    entry = dataset_entry(tag)
    keep = {c for c in entry["classes"]
            if c not in entry["non_defect"] and c not in entry["excluded"]}
    return {i for i, c in enumerate(entry["classes"]) if c in keep}


# --------------------------------------------------------------------------
# Arm B, k-shot fine-tuning. PRE-REGISTERED, the project log 6a.
# --------------------------------------------------------------------------
# Two configurations, not four. The Friedman test over the main matrix found
# no detectable difference between the four, so spending four times the compute
# to re-ask a question already answered no would be the wrong trade. RT-DETR-l
# is present only to show whether the recovery curve's SHAPE is an artefact of
# one architecture.
ARMB_FAMILIES = ["yolo11s", "rtdetr_l"]

# Both localisation training sources. These are the two box-level training
# rows, so they are the two checkpoints for which a box-level recovery curve is
# even defined.
ARMB_START_SOURCES = [t for t in C.TRAIN_SOURCES
                      if C.DATASETS[t]["protocol"] == "localisation"]

ARMB_K = [10, 25, 50, 100, 250]
ARMB_DRAWS = [0, 1, 2]

# THE DRAW INDEX IS ALSO THE SEED, AND THAT IS DELIBERATE.
# R_ij(k) = (P_ij(k) - P_ij(0)) / (P_jj - P_ij(0)) must divide by the SAME
# configuration and the SAME seed on the target diagonal. Tying the draw to a
# seed makes all three terms come from one seed by construction:
#   P_ij(k)  this cell, fine-tuned from the main cell (start, family, seed)
#   P_ij(0)  that same main cell's zero-shot score on the target
#   P_jj     the target's own diagonal or reference at (family, seed)
# A YOLO11s numerator can therefore never be divided by an RT-DETR-l
# denominator, which is the rule the reference cells exist to make possible.
ARMB_SEED_IS_DRAW = True

# The fine-tuning recipe is FIXED A PRIORI, across every k and every target.
# Nothing here may be selected using any target test split. Epochs are held
# constant rather than scaled with k, so that the only thing varying along the
# curve is how much target annotation the model saw.
ARMB_EPOCHS = 50
ARMB_BATCH = 8          # k can be as small as 10 images, so the frozen 32 would
                        # give a single partial batch per epoch
ARMB_LR0 = 0.0001       # one tenth of the frozen lr0, standard for fine-tuning
ARMB_WARMUP_EPOCHS = 1.0


def armb_targets(start_source: str) -> list[str]:
    """Every target with a reference model, excluding the checkpoint's own
    source. All five datasets have a same-family same-seed diagonal, from the
    main grid for a training source and from a reference cell otherwise.
    """
    return [t for t in C.ALL_TARGETS if t != start_source]


# --------------------------------------------------------------------------
# Cell registries
# --------------------------------------------------------------------------

def lohi_control_cells():
    """the project log 7. Each cell is evaluated on BOTH acquisitions."""
    for source in LOHI_ACQUISITIONS:
        for family in C.FAMILIES:
            for seed in C.SEEDS:
                yield {"arm": "lohi_control", "source": source,
                       "family": family, "seed": seed}


def armb_cells():
    """the project log 6a. Fine-tune a main checkpoint on k target images."""
    for start in ARMB_START_SOURCES:
        for target in armb_targets(start):
            for family in ARMB_FAMILIES:
                for k in ARMB_K:
                    for draw in ARMB_DRAWS:
                        yield {"arm": "armb", "source": start,
                               "target": target, "family": family,
                               "k": k, "draw": draw, "seed": draw}


def armd_cells():
    """the project log 6b. Train on the pool, test on what it excluded."""
    for pool in ARMD_POOLS:
        for family in C.FAMILIES:
            for seed in C.SEEDS:
                yield {"arm": "armd", "source": pool,
                       "family": family, "seed": seed}


def all_arm_cells() -> list[dict]:
    """Every arm cell, in launch order.

    LoHi control first, because it is the cheapest and it is the control that
    every other number is read against. Arm B second, because it carries the
    headline. Arm D last, because it is the most expensive per cell and the
    least load bearing. A chain that runs out of wall clock therefore still
    yields the control and the recovery curve.
    """
    return list(lohi_control_cells()) + list(armb_cells()) + list(armd_cells())


def arm_cell_name(cell: dict) -> str:
    if cell["arm"] == "armb":
        return (f"armb__{cell['source']}_to_{cell['target']}__"
                f"{cell['family']}__k{cell['k']}__draw{cell['draw']}")
    prefix = {"lohi_control": "lohictl", "armd": "armd"}[cell["arm"]]
    return f"{prefix}__{cell['source']}__{cell['family']}__seed{cell['seed']}"


def arm_data_yaml(cell: dict) -> Path:
    """The dataset config an arm cell trains from.

    Arm datasets are materialised UNDER clean_candidate/ as sibling folders, so
    this resolves exactly like the main grid's `C.data_yaml`, and so the frozen
    trainers in train_cell.py work on an arm cell without modification. No
    frozen file inside clean_candidate/ is touched: the arm folders are new
    siblings holding symlinks, and split_hashes.json is unchanged.

    Arm B is the exception. It trains on a k-shot draw, which is materialised
    per cell under runs_arms/, because a draw is not a dataset.
    """
    if cell["arm"] == "armb":
        return ARM_RUNS / arm_cell_name(cell) / "kshot" / "data.yaml"
    return C.CANDIDATE / cell["source"] / "data.yaml"


def arm_eval_plan(cell: dict) -> list[tuple[str, str]]:
    """(result key, target dataset) for one arm cell. Always the audited split.

    lohi_control  BOTH acquisitions, so one training run yields a row of the
                  2 by 2 matrix including its own diagonal.
    armb          the adaptation target only. P_ij(0) and P_jj come from main
                  and reference cells that already exist, so nothing is
                  retrained to supply them.
    armd          the source the pool EXCLUDED, plus the two test-only targets.
                  Across the three pools that covers all five targets, which is
                  what the project log 6b means by five held-out targets.
    """
    arm = cell["arm"]
    if arm == "lohi_control":
        return [(t, t) for t in LOHI_ACQUISITIONS]
    if arm == "armb":
        return [(cell["target"], cell["target"])]
    held_out = POOL_HELD_OUT[cell["source"]]
    targets = [held_out] + [t for t in C.ALL_TARGETS
                            if C.DATASETS[t]["role"] == "target_only"]
    return [(t, t) for t in targets]


def box_level_defined(source: str, target: str) -> bool:
    """Whether a box-level number between these two means domain shift.

    A pool containing weld_surface_v2 is protocol `mixed`, so it never yields
    an interpretable box-level number. This is the project log 6b's rule expressed
    once, in code, rather than remembered per table.
    """
    return (dataset_entry(source)["protocol"] == "localisation"
            and dataset_entry(target)["protocol"] == "localisation")


# --------------------------------------------------------------------------
# Fingerprint
# --------------------------------------------------------------------------

def arm_payload() -> dict:
    return {
        "frozen_main_fingerprint": FROZEN_MAIN_FINGERPRINT,
        "arm_datasets": {
            k: {kk: (sorted(vv) if isinstance(vv, set) else vv)
                for kk, vv in v.items()}
            for k, v in ARM_DATASETS.items()
        },
        "armd_pools": ARMD_POOLS,
        "armb": {
            "families": ARMB_FAMILIES,
            "start_sources": ARMB_START_SOURCES,
            "k": ARMB_K,
            "draws": ARMB_DRAWS,
            "seed_is_draw": ARMB_SEED_IS_DRAW,
            "epochs": ARMB_EPOCHS,
            "batch": ARMB_BATCH,
            "lr0": ARMB_LR0,
            "warmup_epochs": ARMB_WARMUP_EPOCHS,
        },
        "lohi_acquisitions": LOHI_ACQUISITIONS,
        "cell_counts": {
            "lohi_control": len(list(lohi_control_cells())),
            "armb": len(list(armb_cells())),
            "armd": len(list(armd_cells())),
        },
    }


def arm_fingerprint() -> str:
    """sha256 over the arm payload, which embeds the frozen main fingerprint.

    Deliberately NOT a function of the live `C.config_fingerprint()`, because
    this module registers datasets into C.DATASETS at runtime and that would
    make the arm fingerprint depend on import order. The frozen constant is
    checked separately by `assert_main_unmoved()`.
    """
    blob = json.dumps(arm_payload(), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def assert_main_unmoved() -> None:
    """The main configuration must still be exactly what the 69 cells saw.

    Must be called BEFORE register(), because registering arm datasets changes
    what C.config_fingerprint() returns inside this process.
    """
    live = C.config_fingerprint()
    if live != FROZEN_MAIN_FINGERPRINT:
        raise SystemExit(
            f"main configuration fingerprint is {live}, expected "
            f"{FROZEN_MAIN_FINGERPRINT}. The frozen grid has moved, so no arm "
            f"result would be comparable with a main grid result. Refusing to "
            f"continue.")


LOHI_VERDICT_RX = re.compile(C.LOHI_VERDICT_PATTERN)


def acquisition_of(stem: str) -> str:
    """Which LoHi acquisition an image belongs to, from its filename alone.

    The low resolution acquisition writes the production cycle into the name as
    <date>_<time>_<cycle>_CAM1_<NG|OK>_<n>. Everything else is high resolution.
    This is C.LOHI_VERDICT_PATTERN, reused rather than restated.
    """
    return "lohi_low" if LOHI_VERDICT_RX.match(stem) else "lohi_high"


if __name__ == "__main__":
    assert_main_unmoved()
    register()
    payload = arm_payload()
    print(f"frozen main fingerprint : {FROZEN_MAIN_FINGERPRINT}")
    print(f"arm fingerprint         : {arm_fingerprint()}")
    print(f"arm datasets            : {sorted(ARM_DATASETS)}")
    for pool, members in ARMD_POOLS.items():
        print(f"  {pool:<28s} trains on {members}, "
              f"protocol {ARM_DATASETS[pool]['protocol']}")
    counts = payload["cell_counts"]
    print(f"cells: lohi_control {counts['lohi_control']}, "
          f"armb {counts['armb']}, armd {counts['armd']}, "
          f"total {sum(counts.values())}")
