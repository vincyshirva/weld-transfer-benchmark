#!/usr/bin/env python3
"""
Frozen configuration for the cross-dataset generalization grid.

Everything that a reviewer could accuse us of tuning per dataset lives here,
in one file, set once. Nothing downstream may override it.

THE GRID

  training sources  weld_surface_v2, kaggle_v2, lohi_weld
  configurations    yolo11n, yolo11s, rtdetr-l, faster_rcnn. FOUR MODEL
                    CONFIGURATIONS DRAWN FROM THREE DETECTOR FAMILIES:
                    YOLO11 contributes two scales, n and s. Never write
                    "four detector families" anywhere.
  seeds             3
  cells             3 x 4 x 3 = 36 training runs

Each trained cell is then evaluated against all five datasets, so no model is
retrained for evaluation. 36 trainings, 180 evaluations.

Supporting cells add 24 same-family references and 9 downloaded-data
baselines. The complete Slurm array therefore has 69 cells. It produces 222
evaluation records, not 213, because each downloaded-data cell is scored
twice: on the split it shipped with and on the audited split.

TWO MATRICES, ONE SET OF MODELS

Models train on their source's native label set. Harmonization happens at
evaluation time, which is why a single training grid serves both matrices.

  image level   annotated target feature present or not. Invariant to how
                boxes were drawn, so every source can be tested against every
                target. This is not a universal accept or reject verdict. On
                LoHi in particular the boxes mark surface features and are not
                a pass or fail verdict on the part, which is why 97.6 percent
                of its images are positive under this label.
  box level     class agnostic localisation. Only defined between datasets
                that share the localisation protocol.

WHY CLASS AGNOSTIC AT THE BOX LEVEL

kaggle_v2 has one generic `Defect` class. lohi_weld has four specific types,
pore, deposit, discontinuity and stain. There is no correspondence between
them, and inventing one would be the paper's weakest claim. So the box level
matrix collapses every defect class to a single `defect` and measures
localisation quality alone. Per-class AP is reported in domain only, where the
classes are the source's own and the question is well posed.

Collapsing requires class agnostic NMS afterwards, since a model may put a
`pore` box and a `stain` box on the same object. Without it those become a
false positive pair. That is handled in evaluate_cell.py.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

# --------------------------------------------------------------------------
# Paths. Absolute everywhere, because relative paths fail silently under Slurm
# on the H200 node, and because a `path` key in a data.yaml resolves against
# the working directory. See CLAUDE.md 4c.
# --------------------------------------------------------------------------

PROJECT = Path(__file__).resolve().parent.parent
CANDIDATE = PROJECT / "clean_candidate"
RUNS = PROJECT / "runs"
RESULTS = PROJECT / "results"
PRETRAINED_ROOT = PROJECT / "models" / "pretrained"
PRETRAINED_MANIFEST = PRETRAINED_ROOT / "manifest.json"

# --------------------------------------------------------------------------
# Dataset registry
# --------------------------------------------------------------------------
# defect_classes  what counts as an annotated target feature at image level
# excluded        dropped from every transfer figure, in both ground truth and
#                 predictions, before any metric is computed
# protocol        localisation, or whole-region verdict. Decides which pairs
#                 may appear in the box level matrix at all.
#
# The three judgement calls, all inherited from CLAUDE.md:
#
#  1. weld_surface_v2 `Normal Weld-NW` is not a defect. Confirmed in the data:
#     every image carries exactly one class, so the image level label is that
#     class and nothing else.
#  2. kaggle_v2 `Bad Weld` and `Good Weld` are region quality verdicts, not
#     defect types. Only `Defect` is a defect. This is load bearing: 103 of
#     938 images carry a Bad Weld box and no Defect box, and under this rule
#     they are image level negatives. Flip KAGGLE_BAD_WELD_IS_DEFECT to change
#     it, and say which rule was used in the paper.
#  3. visual `Welding Line` and `workpiece` are scene furniture. They are
#     excluded, not merely ignored, so they cannot inflate mAP on the one
#     target where they exist.

KAGGLE_BAD_WELD_IS_DEFECT = False

DATASETS: dict[str, dict] = {
    "weld_surface_v2": {
        "classes": ["Burn Through-BT", "Contamination-CN", "Lack of Fusion-LOF",
                    "Normal Weld-NW", "Spatter-SR", "Surface Crack-SC",
                    "Undercut-UC"],
        "non_defect": {"Normal Weld-NW"},
        "excluded": set(),
        "protocol": "whole_region",
        "role": "source",
    },
    "kaggle_v2": {
        "classes": ["Bad Weld", "Good Weld", "Defect"],
        "non_defect": set() if KAGGLE_BAD_WELD_IS_DEFECT else {"Bad Weld", "Good Weld"},
        "excluded": {"Good Weld"} if KAGGLE_BAD_WELD_IS_DEFECT else set(),
        "protocol": "localisation",
        "role": "source",
    },
    "lohi_weld": {
        "classes": ["pore", "deposit", "discontinuity", "stain"],
        "non_defect": set(),
        "excluded": set(),
        "protocol": "localisation",
        "role": "source",
    },
    "visual": {
        "classes": ["Defect", "Welding Line", "workpiece"],
        "non_defect": set(),
        "excluded": {"Welding Line", "workpiece"},
        # CORRECTED 4 September 2026. CLAUDE.md finding 3 lists visual as a
        # whole-region verdict at 5.12 boxes per image and 16.7% median area.
        # That was measured over ALL THREE classes, before the decision to
        # exclude the scene furniture. `workpiece` covers most of the frame,
        # so it dominated both statistics. Recomputed over the `Defect` class
        # alone, visual is 3.09 boxes per image at 0.18% median area, which is
        # localisation by a wide margin and closer to lohi_weld than to
        # weld_surface_v2. Visual therefore CAN be a box level target.
        "protocol": "localisation",
        "role": "target_only",
    },
    "twoclass": {
        "classes": ["Geometric defect", "Non-fusion defect"],
        "non_defect": set(),
        "excluded": set(),
        "protocol": "localisation",
        "role": "target_only",
    },
}

TRAIN_SOURCES = [t for t, d in DATASETS.items() if d["role"] == "source"]
ALL_TARGETS = list(DATASETS)

# The box level matrix cannot use a whole-region source as a training row. A
# model trained on frame-sized verdict boxes would score near zero against
# localised defects for reasons that are not domain shift. Only two public
# visible-light weld sets are annotated for localisation and large enough to
# train on, which is itself a finding. Say it in the caption.
BOX_SOURCES = [t for t in TRAIN_SOURCES if DATASETS[t]["protocol"] == "localisation"]

# Targets must share the protocol too, otherwise the number measures
# annotation mismatch. weld_surface_v2 is the only whole-region set left after
# the visual correction, so it is the only excluded column. Its whole box
# level column is undefined, numerator as well as denominator.
BOX_TARGETS = [t for t in ALL_TARGETS
               if DATASETS[t]["protocol"] == "localisation"]


def defect_classes(tag: str) -> list[str]:
    """Class names that mean a defect is present, after exclusions."""
    d = DATASETS[tag]
    return [c for c in d["classes"]
            if c not in d["non_defect"] and c not in d["excluded"]]


def defect_ids(tag: str) -> set[int]:
    """Class indices, as they appear in the YOLO label files."""
    keep = set(defect_classes(tag))
    return {i for i, c in enumerate(DATASETS[tag]["classes"]) if c in keep}


def excluded_ids(tag: str) -> set[int]:
    return {i for i, c in enumerate(DATASETS[tag]["classes"])
            if c in DATASETS[tag]["excluded"]}


# --------------------------------------------------------------------------
# Families and seeds
# --------------------------------------------------------------------------

FAMILIES = {
    "yolo11n":     {"backend": "ultralytics", "cfg": "yolo11n.yaml",
                    "weights_file": "yolo11n.pt"},
    "yolo11s":     {"backend": "ultralytics", "cfg": "yolo11s.yaml",
                    "weights_file": "yolo11s.pt"},
    "rtdetr_l":    {"backend": "ultralytics", "cfg": "rtdetr-l.yaml",
                    "weights_file": "rtdetr-l.pt"},
    "faster_rcnn": {"backend": "torchvision",
                    "cfg": "fasterrcnn_resnet50_fpn_v2",
                    "weights_file": "fasterrcnn_resnet50_fpn_v2_coco.pth"},
}

SEEDS = [0, 1, 2]

# The family used for dirty baselines. Reference models use every family so a
# transfer ratio never divides by a denominator from a different detector.
AUX_FAMILY = "yolo11s"

# --- reference models, so P_jj exists for every target ---------------------
# The transfer ratio T_ij = P_ij / P_jj and the recovery curve R_ij(k) both
# divide by the target's own in-domain score. For a test-only target there is
# no diagonal cell in the main grid, so P_jj does not exist and neither
# quantity is computable. These cells supply it by training on the target's
# own train split and evaluating in domain only.
#
# twoclass's reference is WEAK BY CONSTRUCTION and must be labelled as such
# wherever it is used as a denominator: 213 training images in 23 scene
# blocks, from 7 joint configurations. A small denominator inflates T_ij, so a
# high transfer ratio into twoclass may say more about the reference being bad
# than about transfer being good. Report T_ij into twoclass with the reference
# score beside it, never alone.
REFERENCE_TARGETS = ["visual", "twoclass"]
REFERENCE_FAMILIES = list(FAMILIES)

# weld_surface_v2 needs no auxiliary reference. Its image level P_jj comes
# from the main grid diagonal. Its box level column is undefined because it
# uses whole-region annotations.

# --- downloaded-data baseline, the pre-audit versus post-audit comparison --
# Step 2 of the plan lists this as outstanding and it cannot come from the 36
# clean cells. These train on portable snapshots of the original shipped
# splits and are evaluated in domain on the shipped test split.
#
# The downloaded and clean values use different test sets. Their difference
# is therefore an estimate shift, not a causal measure of leakage inflation.
# Say this explicitly in the caption.
DIRTY_SOURCES = ["weld_surface_v2", "kaggle_v2", "lohi_weld"]
DIRTY_SEEDS = [0, 1, 2]

# --------------------------------------------------------------------------
# Frozen hyperparameters
# --------------------------------------------------------------------------
# Recipes are fixed a priori from documented family defaults and frozen
# everywhere. There is no dataset-specific tuning. Per-family differences are
# unavoidable, since a one-stage anchor-free detector and a two-stage region
# proposal network do not share an optimizer schedule. They are stated rather
# than hidden.
#
# BATCH 32 EVERYWHERE. This is a real gain from moving to the H200. On 8 GB,
# RT-DETR and Faster R-CNN were forced to batch 8 while YOLO ran at 32, so
# batch would have differed by family. On 141 GB all four fit at 32, the
# largest being Faster R-CNN at 26.4 GB, so one batch size holds across the
# grid. Measured: larger batches buy almost nothing, YOLO11s gains 8 percent
# from 32 to 256.

IMGSZ = 640
BATCH = 32
EPOCHS = 150
WORKERS = 2          # server guide compliant. Raise only with admin approval.
PATIENCE = EPOCHS    # fixed training length. Best validation checkpoint wins.
CHECKPOINT_SELECTION = "native_validation_map50_95"

COMMON = {
    "imgsz": IMGSZ,
    "batch": BATCH,
    "epochs": EPOCHS,
    "workers": WORKERS,
    "patience": PATIENCE,
}

ULTRALYTICS_HP = {
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "cos_lr": True,
    "amp": True,
    "pretrained": False,   # see PRETRAINED below
    "deterministic": True,
    # Augmentation is part of the frozen recipe. Arm A varies it on purpose,
    # so these are the arm-zero values and arm A overrides them explicitly.
    "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
    "degrees": 0.0, "translate": 0.1, "scale": 0.5,
    "fliplr": 0.5, "flipud": 0.0,
    "mosaic": 1.0, "close_mosaic": 10,
    "erasing": 0.0,
}

TORCHVISION_HP = {
    "optimizer": "SGD",
    "lr": 0.01,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "warmup_iters": 500,
    "lr_schedule": "cosine",
    "amp": True,
    "trainable_backbone_layers": 5,
}

# Whether to start from COCO pretrained weights. This is a real decision, not
# a detail. Pretrained initialisation is the norm in the literature we are
# benchmarking against, so using it keeps the comparison fair. The checkpoints
# are family-specific COCO initializations, not a common bias that a transfer
# ratio can cancel. They are pinned by SHA256 and applied consistently within
# each family.
PRETRAINED = True

# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
# Thin classes get a split-block cluster bootstrap after results are retrieved.
# Seed dispersion measures training variance. It does not measure uncertainty
# from a small or correlated test sample.
MIN_TEST_INSTANCES = 50
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260904

# Shared postprocessing and calibration settings. These are part of the
# frozen evaluation protocol and therefore part of the configuration hash.
CONF_FLOOR = 0.001
MAX_DET = 300
NMS_IOU = 0.7
ECE_BINS = 15

# False positive budgets per image, for the inspection-style recall metric.
FP_PER_IMAGE_BUDGETS = [0.1, 0.5, 1.0]

# Datasets where the image-level false alarm rate is not measurable, because
# there are too few negative test images to estimate it. Measured, not
# assumed: visual has 9 negatives of 84, twoclass 9 of 45, lohi_weld 11 of
# 453. Report positive-side metrics and box-level FP per image for these, and
# do not quote an AUROC or a specificity.
MIN_NEGATIVES_FOR_FPR = 30


def pretrained_path(family: str) -> Path:
    """Pinned local checkpoint. Training must never download one mid-grid."""
    return PRETRAINED_ROOT / FAMILIES[family]["weights_file"]


def pretrained_records() -> dict:
    """The released checkpoint manifest, including source URLs and hashes."""
    if not PRETRAINED_MANIFEST.is_file():
        return {}
    return json.loads(PRETRAINED_MANIFEST.read_text()).get("files", {})


def frozen_config_payload() -> dict:
    """Everything that changes the fitted model or reported metrics."""
    return {
        "common": COMMON,
        "ultralytics": ULTRALYTICS_HP,
        "torchvision": TORCHVISION_HP,
        "pretrained": PRETRAINED,
        "checkpoint_selection": CHECKPOINT_SELECTION,
        "pretrained_manifest": pretrained_records(),
        "families": FAMILIES,
        "datasets": {
            k: {kk: sorted(vv) if isinstance(vv, set) else vv
                for kk, vv in v.items()}
            for k, v in DATASETS.items()
        },
        "evaluation": {
            "conf_floor": CONF_FLOOR,
            "max_det": MAX_DET,
            "nms_iou": NMS_IOU,
            "ece_bins": ECE_BINS,
            "min_test_instances": MIN_TEST_INSTANCES,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "fp_per_image_budgets": FP_PER_IMAGE_BUDGETS,
            "min_negatives_for_fpr": MIN_NEGATIVES_FOR_FPR,
        },
    }


def config_fingerprint() -> str:
    payload = json.dumps(frozen_config_payload(), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]

# --------------------------------------------------------------------------
# LoHi-WELD's production verdict, a second image-level label
# --------------------------------------------------------------------------
# Audited 4 September 2026, and it changes what LoHi's image level number
# means.
#
# The low resolution filenames carry the line's pass or fail verdict:
#   <date>_<time>_<cycle>_CAM1_<NG|OK>_<n>
# There are 570 NG cycles and 430 OK cycles, 1140 and 860 images. So 43
# percent of that acquisition is a part the line ACCEPTED.
#
# But OK images are annotated too. 820 of 860 carry boxes, 5,429 in total,
# and their class mix is almost identical to NG:
#
#            NG                     OK
#   stain    30.5%                  33.8%
#   disc     32.8%                  32.8%
#   pore     28.0%                  24.4%
#   deposit   8.7%                   9.0%
#   boxes/img 6.59                   6.31
#
# So LoHi's boxes mark SURFACE FEATURES, not rejectable defects. Presence of a
# box is not the same question as whether the part passed. That is why the
# box-presence image level label makes LoHi 97.6 percent positive and leaves
# only 11 negative test images.
#
# Nothing was discarded during conversion. All 860 OK images are in the tree,
# 136 of them in the test split. The single dropped image
# (200905_161511_0000000009_CAM1_NG_1) was an NG duplicate removed by NCC.
#
# THE FIX is not to relabel LoHi. Box presence stays the primary definition,
# because it is the one definition that is identical across all five datasets,
# and changing it for one source would break the comparison the paper exists
# to make. Instead the verdict is reported as a SEPARATE, NAMED secondary
# label on the low resolution subset, where it is well defined and balanced.
#
# Expect the detector to score near chance on it. OK and NG are annotated
# almost identically, so a model trained on those boxes has nothing to
# separate them with. That is a finding worth stating plainly: in a real
# industrial line, annotation presence is not a proxy for part acceptance.
LOHI_VERDICT_SUBSET = "lohi_weld_lowres_verdict"
LOHI_VERDICT_PATTERN = r"^(\d{6}_\d{6}_\d+_CAM\d+)_(NG|OK)_(\d+)$"


def cells():
    """The main transfer grid: 3 sources x 4 families x 3 seeds."""
    for source in TRAIN_SOURCES:
        for family in FAMILIES:
            for seed in SEEDS:
                yield {"source": source, "family": family, "seed": seed,
                       "kind": "main"}


def reference_cells():
    """In-domain reference models for the test-only targets, giving P_jj."""
    for target in REFERENCE_TARGETS:
        for family in REFERENCE_FAMILIES:
            for seed in SEEDS:
                yield {"source": target, "family": family, "seed": seed,
                       "kind": "reference"}


def dirty_cells():
    """Baselines trained on portable snapshots of the shipped splits."""
    for source in DIRTY_SOURCES:
        for seed in DIRTY_SEEDS:
            yield {"source": source, "family": AUX_FAMILY, "seed": seed,
                   "kind": "dirty"}


def all_cells():
    """Everything the Slurm array runs, in a stable order.

    Main cells come first so that a partially finished array still yields the
    headline matrix. References and dirty baselines are supporting evidence.
    """
    return list(cells()) + list(reference_cells()) + list(dirty_cells())


def cell_name(cell: dict) -> str:
    prefix = {"main": "", "reference": "ref_", "dirty": "dirty_"}[cell.get("kind", "main")]
    return f"{prefix}{cell['source']}__{cell['family']}__seed{cell['seed']}"


def data_yaml(cell: dict) -> Path:
    """The dataset config a cell trains from.

    Main and reference cells use the audited clean tree. Dirty cells use
    portable staged copies of the original shipped splits.
    """
    if cell.get("kind") == "dirty":
        return Path(__file__).resolve().parent / "dirty" / cell["source"] / "data.yaml"
    return CANDIDATE / cell["source"] / "data.yaml"


# Suffix marking a downloaded-data cell scored on the AUDITED test split.
CLEAN_TEST_SUFFIX = "@clean_test"


def eval_plan(cell: dict) -> list[tuple[str, str, str]]:
    """(result key, target dataset, which test split) for one cell.

    Main cells cross the whole matrix. Reference cells produce one number,
    P_jj, so they are in domain only.

    A downloaded-data cell is scored TWICE, on the same target, because the
    comparison needs three values per source and training only supplies two:

      downloaded train -> downloaded test   what a practitioner publishes
      downloaded train -> audited test      the same model, honestly scored
      audited train    -> audited test      from the main grid diagonal

    The middle one is the addition. It costs one extra evaluation per cell and
    no extra training, and it separates two things the first and third
    conflate. Between the first and the third, both the training data and the
    test set change. Between the first and the second, only the test set
    changes, so it isolates how much of the apparent score depends on being
    graded on a partition that shares content with training.

    None of the three licenses a causal claim. Report the differences as
    estimate shifts under a change of protocol, not as the effect of leakage.
    """
    kind = cell.get("kind", "main")
    source = cell["source"]
    if kind == "main":
        return [(t, t, "clean") for t in ALL_TARGETS]
    if kind == "reference":
        return [(source, source, "clean")]
    return [(source, source, "dirty"),
            (source + CLEAN_TEST_SUFFIX, source, "clean")]


def split_result_key(key: str) -> tuple[str, str]:
    """Result key -> (target dataset, test split)."""
    if key.endswith(CLEAN_TEST_SUFFIX):
        return key[:-len(CLEAN_TEST_SUFFIX)], "clean"
    return key, "native"


def eval_targets(cell: dict) -> list[str]:
    """Result keys a cell is evaluated under. Used to validate a record."""
    return [key for key, _, _ in eval_plan(cell)]


if __name__ == "__main__":
    main_cells = list(cells())
    grid_cells = all_cells()
    print(f"training sources : {TRAIN_SOURCES}")
    print(f"target datasets  : {ALL_TARGETS}")
    print(f"families         : {list(FAMILIES)}")
    print(f"seeds            : {SEEDS}")
    print(f"main cells       : {len(main_cells)}")
    print(f"total cells      : {len(grid_cells)}")
    print(f"evaluations      : {sum(len(eval_targets(c)) for c in grid_cells)}")
    print()
    print(f"box level matrix : sources {BOX_SOURCES}")
    print(f"                   targets {BOX_TARGETS}")
    print()
    for tag in ALL_TARGETS:
        d = DATASETS[tag]
        print(f"{tag:<18} defect={defect_classes(tag)}")
        if d["excluded"]:
            print(f"{'':<18} excluded={sorted(d['excluded'])}")
    print()
    for i, c in enumerate(grid_cells[:3]):
        print(f"  cell {i}: {cell_name(c)}")
    print(f"  ... {len(grid_cells) - 3} more")
