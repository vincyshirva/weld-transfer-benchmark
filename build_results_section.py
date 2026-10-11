#!/usr/bin/env python3
"""Build the whole manuscript as a .docx: title page, abstract, Sections 1
to 6, appendices, the statements and declarations, and the reference list.

Author notes are kept visually separate and clearly marked as not
manuscript text. No dashes as punctuation, and short sentences,
throughout.

    python3 build_results_section.py            # working draft
    python3 build_results_section.py --clean    # submission version

THE --clean FLAG EXISTS SO THE SUBMISSION VERSION IS GENERATED RATHER THAN
HAND STRIPPED. It drops every author note and every "Draft." prefix and
promotes the title to a real heading.

CONSISTENCY CHECKS. Every range quoted in the prose is derived from the table
it sits beside rather than typed independently, and `check_span` fails the
build if the two disagree. This exists because the 19 September audit found
two spans in the prose that contradicted the table three lines above them,
which is the same class of defect as the project log 6a-xix: prose and table
generated from different code paths, with nothing forcing them to agree.

SINCE 24 SEPTEMBER EVERY RESULT IS READ FROM A RELEASED FILE AT BUILD TIME,
at full precision, and rounded once where it is printed. No table and no
sentence restating one is typed from a markdown file any more, because a value
rounded to four decimals there and then to three here can land one unit off.
The checks still read the files for what a derivation cannot guarantee: each
caption's bootstrap draw count against the `resamples` its rows record, the
internal control's intervals against their CSV, the class lists against
grid/config.py, and two released files against each other wherever both
report the same quantity. Since 25 September they also refuse any in-domain
row of per_class_ap.csv that lacks the NMS correction of the project log 5p.

CITATIONS. Every in-text citation and every reference-list entry is rendered
from literature_review/references.json, which resolve_references.py builds
from DOIs. A key cited but not listed, or listed but never cited, fails the
build, and an in-text year cannot disagree with its reference.
"""
import csv
import html
import json
import re
import shutil
import statistics
import sys
import unicodedata
import urllib.error
import urllib.request
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Emu, Inches, Mm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent
MAIN_FIG = ROOT / "grid_results" / "analysis" / "figures"
ARM_FIG = ROOT / "grid_results_arms" / "analysis"

NOTE_COLOUR = RGBColor(0x7A, 0x3E, 0x00)
CFGS = ("Faster R-CNN", "RT-DETR-l", "YOLO11n", "YOLO11s")

def q(x, places: int) -> str:
    """Round half up on the decimal string, not the binary float, so 0.735
    becomes 0.74 as a reader would expect and not 0.73."""
    return str(Decimal(str(x)).quantize(Decimal(1).scaleb(-places),
                                        rounding=ROUND_HALF_UP))


def q_signed(x, places: int) -> str:
    """q() with a plus sign on a value that is not negative."""
    s = q(x, places)
    return s if s.startswith("-") else "+" + s


def q_stored(x, places: int) -> str:
    """q() for a value that a released file stores already rounded.

    Rounding a stored value again gives what rounding the unrounded value
    once would give, unless the stored value ends in a 5 exactly one place
    beyond the printed precision. That case fails the build rather than print
    a number that may be one unit off."""
    digits = str(x).partition(".")[2]
    if len(digits) == places + 1 and digits.endswith("5"):
        raise SystemExit(f"\nCONSISTENCY CHECK FAILED: {x} is stored rounded "
                         f"to {places + 1} decimals and ends in 5, so it "
                         f"cannot be rounded again to {places}. Read the "
                         f"unrounded value instead.")
    return q(x, places)


def pct(x, places: int) -> str:
    return q(Decimal(str(x)) * 100, places)


def span(values, fmt, places: int) -> str:
    """'lo to hi', or the one value when both ends print the same."""
    lo, hi = fmt(min(values), places), fmt(max(values), places)
    return lo if lo == hi else f"{lo} to {hi}"


def shipped_to_vocab(name: str) -> str:
    """'Burn Through-BT' is 'Burn Through': the suffix is a label code."""
    head, _, tail = name.rpartition("-")
    return head if head and tail.isupper() and len(tail) <= 3 else name


# ------------------------------------------------------------ released data
# EVERY RESULT IN THIS FILE IS READ FROM A RELEASED FILE AT FULL PRECISION,
# AND IT IS ROUNDED EXACTLY ONCE, WITH q(), WHERE IT IS PRINTED.
#
# Until 24 September 2026 most of these tables were typed, some from
# four-decimal markdown, and then printed at three decimals. A value rounded
# twice can land one unit off: Table 4.13 printed 0.287 where its CSV gives
# 0.286. A sweep that recomputed every cell of every table from its CSV found
# the same defect in Table 4.10, which printed +0.113 for 0.11248, and in
# three prose figures. It also found two ratios taken between rounded values,
# a factor of 30 that is 31 at full precision and a factor of 107 that is 105.
# Reading the files removes the transcription step, and the defect class with
# it. The names and shapes below are the ones the typed tables had, so every
# use site reads as it did.
TABLES = ROOT / "grid_results" / "analysis" / "tables"
ARMS = ROOT / "grid_results_arms" / "analysis"
RATIO_CSV = TABLES / "transfer_ratio_intervals.csv"
CELLS_CSV = TABLES / "cells.csv"
BASE_CSV = TABLES / "target_base_rates.csv"
AUDIT_CSV = TABLES / "downloaded_vs_audited.csv"
PER_CLASS_CSV = TABLES / "per_class_ap.csv"
STATS_JSON = ROOT / "grid_results" / "analysis" / "statistical_tests.json"
CONTROL_CSV = ARMS / "lohi_control_transfer_intervals.csv"
CONTROL_MD = ARMS / "lohi_control.md"
ARMB_CSV = ARMS / "armb_recovery.csv"
ARMD_CSV = ARMS / "armd.csv"
RECALL_CSV = ARMS / "recall_at_fp_budget_arms.csv"
STATS_ARMS_JSON = ARMS / "stats_arms.json"
SCHEDULE_CSV = ROOT / "grid_results_msc" / "matched_step_control.csv"
GEOMETRY_JSON = ROOT / "clean_candidate" / "box_geometry.json"
THIN_CACHE = ROOT / "grid_results" / "bootstrap_cache.json"
FAMILY_CODE = {"faster_rcnn": "Faster R-CNN", "rtdetr_l": "RT-DETR-l",
               "yolo11n": "YOLO11n", "yolo11s": "YOLO11s"}
CFG_CODE = {v: k for k, v in FAMILY_CODE.items()}
N_SEEDS = 3
TRAIN_SOURCES = ("weld_surface_v2", "kaggle_v2", "lohi_weld")


def read_csv_rows(path: Path) -> list:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def mean_of(values, expect: int) -> float:
    """Full-precision mean that refuses a group of the wrong size, because a
    filter matching too few rows would still print a plausible number."""
    v = [float(x) for x in values]
    if len(v) != expect:
        raise SystemExit(f"\nCONSISTENCY CHECK FAILED: a group read from a "
                         f"released file holds {len(v)} values where "
                         f"{expect} are expected.")
    return statistics.fmean(v)


CELLS = read_csv_rows(CELLS_CSV)


def kind_of(source: str) -> str:
    """A training source's in-domain value is a main-grid diagonal. A
    test-only target's comes from its reference cells."""
    return "main" if source in TRAIN_SOURCES else "reference"


def cell_mean(key: str, source: str, target: str, cfg=None,
              kind: str = "main", split: str = "audited") -> float:
    """One cells.csv column averaged over the three seeds, and over the four
    configurations as well when cfg is None."""
    return mean_of([r[key] for r in CELLS
                    if r["kind"] == kind and r["test_split"] == split
                    and r["source"] == source and r["target"] == target
                    and (cfg is None or r["family"] == CFG_CODE[cfg])],
                   N_SEEDS * (1 if cfg else len(CFGS)))


# Off-diagonal transfer. Pair order is the order the tables print.
BOX_PAIRS = (("lohi_weld", "kaggle_v2"), ("kaggle_v2", "lohi_weld"),
             ("kaggle_v2", "twoclass"), ("lohi_weld", "twoclass"),
             ("kaggle_v2", "visual"), ("lohi_weld", "visual"))
IMG_PAIRS = (("kaggle_v2", "weld_surface_v2"),
             ("lohi_weld", "weld_surface_v2"), ("lohi_weld", "kaggle_v2"),
             ("weld_surface_v2", "kaggle_v2"), ("kaggle_v2", "lohi_weld"),
             ("weld_surface_v2", "lohi_weld"), ("kaggle_v2", "visual"),
             ("lohi_weld", "visual"), ("weld_surface_v2", "visual"),
             ("kaggle_v2", "twoclass"), ("lohi_weld", "twoclass"),
             ("weld_surface_v2", "twoclass"))


def ratio_tables():
    """From transfer_ratio_intervals.csv, whose rows are per configuration
    with the seeds combined:
      box: (source, target): {config: (P_ij, T, lo, hi, weak reference)}
      img: (source, target): {config: (T, lo, hi, weak reference)}
    The image-level ratio is the base-rate corrected one. Every pair must
    carry all four configurations, and no other pair may appear."""
    box = {p: {} for p in BOX_PAIRS}
    img = {p: {} for p in IMG_PAIRS}
    for r in read_csv_rows(RATIO_CSV):
        if r["source"] == r["target"]:
            continue
        key, cfg = (r["source"], r["target"]), FAMILY_CODE[r["family"]]
        weak = r["denominator_is_weak_by_construction"] == "True"
        table = box if r["level"] == "box" else img
        if key not in table:
            raise SystemExit(f"\nCONSISTENCY CHECK FAILED: {RATIO_CSV.name} "
                             f"carries an unexpected {r['level']} pair {key}.")
        if r["level"] == "box":
            box[key][cfg] = (float(r["P_ij"]), float(r["transfer_ratio"]),
                             float(r["transfer_ratio_lo"]),
                             float(r["transfer_ratio_hi"]), weak)
        else:
            img[key][cfg] = (float(r["transfer_ratio_above_base"]),
                             float(r["transfer_ratio_above_base_lo"]),
                             float(r["transfer_ratio_above_base_hi"]), weak)
    for level, table in (("box", box), ("image", img)):
        for key, per_cfg in table.items():
            if sorted(per_cfg) != sorted(CFGS):
                raise SystemExit(f"\nCONSISTENCY CHECK FAILED: "
                                 f"{RATIO_CSV.name} carries {sorted(per_cfg)} "
                                 f"at {level} level for {key}.")
    return box, img


# The image-level half is loaded and checked against its file, and no longer
# quoted: it is a ratio of lifts, withdrawn on 8 October 2026, the project log 5y.
BOX_OFF, _IMG_OFF = ratio_tables()

# Box mAP at 0.5 and at 0.5:0.95 on the two localisation diagonals, and the
# stricter one off the diagonal, each a mean over the three seeds.
BOX_DIAG_STRICT = {d: {c: (cell_mean("box_map50", d, d, c),
                           cell_mean("box_map50_95", d, d, c)) for c in CFGS}
                   for d in ("kaggle_v2", "lohi_weld")}
BOX_OFFDIAG_STRICT = [cell_mean("box_map50_95", s, t, c)
                      for s, t in BOX_PAIRS for c in CFGS]
BOX_OFFDIAG_STRICT_SPAN = (min(BOX_OFFDIAG_STRICT), max(BOX_OFFDIAG_STRICT))

# Positive rates, and the positive and negative test-image counts behind them.
# Section 4.4's base-rate row and its AUROC availability paragraph are built
# from these, and n_pos + n_neg must be the frozen test-split size.
BASE = {r["target"]: r for r in read_csv_rows(BASE_CSV)}
BASE_RATE = {t: float(r["positive_rate"]) for t, r in BASE.items()}
BASE_RATE_COUNTS = {t: (int(r["n_pos"]), int(r["n_neg"]))
                    for t, r in BASE.items()}

# Image-level raw diagonals. The four-configuration mean is what Table 4.5's
# lift is built from; the YOLO11s column is what the downloaded-data
# comparison uses, because that baseline was run at YOLO11s only.
IMG_DIAG = {s: {"mean": cell_mean("image_ap", s, s),
                "YOLO11s": cell_mean("image_ap", s, s, "YOLO11s"),
                "base": BASE_RATE[s],
                "lift": cell_mean("image_ap_above_base", s, s)}
            for s in TRAIN_SOURCES}

# Per class in domain, YOLO11s, from per_class_ap.csv: main-grid rows for the
# three training sources and reference rows for the two test-only targets.
# Downloaded-data rows never enter, the project log 5o. A class's state comes from
# the CSV's own flags, and check_config_classes holds it against
# grid/config.py. Each row:
#   (source, class, AP50, seed SD, split-block interval or None,
#    test instances, thin, state)
# state is "defect" for a class the harmonised taxonomy retains, "verdict" for
# one kept in the ground truth but not a defect, and "excluded" for one
# dropped from the ground truth entirely. AP50 and its SD are over the three
# seeds, and the interval is the mean of the three seeds' intervals, which is
# what per_class_ap.md prints.
PER_CLASS_ROWS = read_csv_rows(PER_CLASS_CSV)
PER_CLASS_ORDER = ("kaggle_v2", "lohi_weld", "weld_surface_v2", "visual",
                   "twoclass")


def per_class_table(cfg: str = "YOLO11s") -> list:
    out = []
    for src in PER_CLASS_ORDER:
        by_class = {}
        for r in PER_CLASS_ROWS:
            if (r["kind"] == kind_of(src) and r["source"] == src
                    and r["target"] == src and r["family"] == CFG_CODE[cfg]):
                by_class.setdefault(shipped_to_vocab(r["class"]),
                                    []).append(r)
        for cls in sorted(by_class):
            rr = by_class[cls]
            if (len(rr) != N_SEEDS
                    or len({r["test_instances"] for r in rr}) != 1):
                raise SystemExit(f"\nCONSISTENCY CHECK FAILED: {src} {cls} "
                                 f"has {len(rr)} per-class rows, or rows that "
                                 f"disagree on the test instance count.")
            first = rr[0]
            state = ("excluded" if first["excluded_from_gt"] == "True" else
                     "defect" if first["in_defect_space"] == "True" else
                     "verdict")
            if state == "excluded":
                out.append((src, cls, None, None, None, 0, False, state))
                continue
            ap = [float(r["ap50"]) for r in rr]
            boot = None
            if all(r["ci_lo"] and r["ci_hi"] for r in rr):
                boot = (mean_of([r["ci_lo"] for r in rr], N_SEEDS),
                        mean_of([r["ci_hi"] for r in rr], N_SEEDS))
            out.append((src, cls, statistics.fmean(ap), statistics.stdev(ap),
                        boot, int(first["test_instances"]),
                        first["thin"] == "True", state))
    return out


PER_CLASS = per_class_table()

# Pooling, from armd.csv. DEFINED ONCE because Sections 4.6 and 5.4 both quote
# it and an earlier draft of 5.4 attached the count to the wrong outcome,
# reporting that pooling failed in 13 of 36 when 13 is the number of WINS.
# check_span guards a value against its table; nothing guards a value against
# the direction of the sentence around it, so the least a build can do is make
# both sections read from one place. A comparison is one pool, held-out target
# and configuration, seed-averaged, against the best single member of the same
# pool at the same configuration and seed.
ARMD = read_csv_rows(ARMD_CSV)


def pool_table_rows() -> list:
    """Table 4.10: per pool and held-out target, the range over the four
    configurations of the seed-averaged AUROC of the pooled model and of its
    better member, from image_level_pooling_seed_averaged.csv, and of the
    seed-averaged box mAP50 where one is defined, from armd.csv. The held-out
    training source comes first in each pool."""
    seed_avg = {(r["pool"], r["target"], r["family"]): r
                for r in read_csv_rows(IL_POOL_SEED_CSV)}
    groups = {}
    for r in ARMD:
        groups.setdefault((r["pool"], r["target"]), []).append(r)
    order = sorted(groups, key=lambda k: (
        k[0], groups[k][0]["role"] != "held out training source", k[1]))
    out = []
    for pool, tgt in order:
        rr = groups[(pool, tgt)]
        by_cfg = {}
        for r in rr:
            by_cfg.setdefault(r["family"], []).append(r)
        if len(by_cfg) != len(CFGS):
            raise SystemExit(f"\nCONSISTENCY CHECK FAILED: {pool} into {tgt} "
                             f"has {len(by_cfg)} configurations.")
        pooled = [fnum(seed_avg[(pool, tgt, f)]["pooled_auroc"])
                  for f in by_cfg]
        better = [fnum(seed_avg[(pool, tgt, f)]["better_member_auroc"])
                  for f in by_cfg]
        role = ("training source"
                if rr[0]["role"] == "held out training source"
                else "test-only target")
        if all(r["box_level_defined"] == "True" for r in rr):
            box = span([mean_of([r["box_map50"] for r in v], N_SEEDS)
                        for v in by_cfg.values()], q, 3)
        elif rr[0]["protocol"] == "mixed":
            box = "undefined, protocol mixed"
        else:
            box = "undefined, whole-region target"
        out.append([pool.replace("pool_no_", "no "), tgt, role,
                    f"{q(min(pooled), 3)} to {q(max(pooled), 3)}",
                    f"{q(min(better), 3)} to {q(max(better), 3)}", box])
    return out


# Sources whose harmonised defect space holds exactly one class. For these
# the per-class AP and the class-agnostic box mAP50 are the same measurement,
# so the two released tables must agree.
SINGLE_DEFECT_CLASS = ("kaggle_v2", "visual")
BOX_DIAG_YOLO11S = {s: cell_mean("box_map50", s, s, "YOLO11s", kind_of(s))
                    for s in SINGLE_DEFECT_CLASS}

# In-domain box diagonals, four-configuration means, from cells.csv. Quoted in
# 4.3 and 4.5. Both halves of those sentences must come from the same file.
BOX_DIAG_MEAN = {s: cell_mean("box_map50", s, s, kind=kind_of(s))
                 for s in ("kaggle_v2", "lohi_weld", "visual", "twoclass")}

# Mendieta et al. 2025, DOI 10.3389/frobt.2025.1632417. The project log's literature
# position. Quoted in the Introduction AND in 5.1, so both read from here: an
# earlier draft typed them twice and two sections that disagree about the
# motivating number is the worst possible drift.
MENDIETA = {
    "in_domain_map50": 94,        # percent, YOLOv7 on 3,000 SMAW images
    "in_domain_precision": 97,    # percent
    "ext_map50": 26.3,            # percent, on their external Kaggle set
    "ext_precision": 65.2,        # percent
    "ext_recall": 25.2,           # percent
    "ext_images": 3,              # curated and cropped
    "ext_images_word": "three",   # for prose, where a bare digit reads badly
    "train_images": 3000,
}

# Our side of that comparison, from the box matrix. Same constants in both
# sections for the same reason.
OUR_KAGGLE_TEST_IMAGES = 141
OUR_KAGGLE_TEST_BOXES = 159

# Scopus sweep recorded in the project log. The denominator matters: the claim is
# about a subfield and a window, not about the world.
SCOPUS_YOLO_WELD_TOTAL = 52
SCOPUS_YOLO_WELD_RECENT = 44

# ----------------------------------------------------------- Section 3 data
# The inventory, from the project log's dataset table and the frozen split lists.
# Section 3.2 and Section 4 both state some of these, so they are defined once
# and `run_checks` asserts the two sections agree.
#   (shipped, after dedup, scene groups, split blocks, licence)
SOURCES = {
    "weld_surface_v2": (17195, 2963, 604, 515, "CC BY 4.0"),
    "kaggle_v2": (2028, 938, 927, 359, "CC0 1.0"),
    "lohi_weld": (3022, 3021, 2022, 1977, "free with citation"),
    "visual": (1288, 558, 313, 285, "CC BY 4.0"),
    "twoclass": (1682, 304, 35, 34, "CC BY 4.0"),
}
CLEAN_TREE_TOTAL = 7784

# Class lists, transcribed from grid/config.py DATASETS. Table 3.1 counts them
# rather than carrying a separate integer.
DATASET_CLASSES = {
    "weld_surface_v2": ("Burn Through", "Contamination", "Lack of Fusion",
                        "Normal Weld", "Spatter", "Surface Crack", "Undercut"),
    "kaggle_v2": ("Bad Weld", "Good Weld", "Defect"),
    "lohi_weld": ("pore", "deposit", "discontinuity", "stain"),
    "visual": ("Defect", "Welding Line", "workpiece"),
    "twoclass": ("Geometric defect", "Non-fusion defect"),
}

# Frozen split sizes, read off clean_candidate/splits/.
SPLITS = {
    "weld_surface_v2": (2074, 444, 445),
    "kaggle_v2": (656, 141, 141),
    "lohi_weld": (2115, 453, 453),
    "visual": (390, 84, 84),
    "twoclass": (213, 46, 45),
}

# Post-audit box geometry over the DEFECT classes alone, read from
# clean_candidate/box_geometry.json, which box_geometry.py writes from the
# frozen splits and the audited labels. The pre-audit all-class figures of
# audit finding 3 are superseded and must not reappear: they understate the
# protocol difference. The protocol is a classification rather than a
# measurement, so it stays typed.
#   (boxes per image, median box area as a percent of the frame, protocol)
GEOMETRY = json.loads(GEOMETRY_JSON.read_text())
PROTOCOL = {"weld_surface_v2": "whole region", "kaggle_v2": "localisation",
            "twoclass": "localisation", "visual": "localisation",
            "lohi_weld": "localisation"}
BOX_GEOMETRY = {s: (GEOMETRY["datasets"][s]["boxes_per_image"],
                    GEOMETRY["datasets"][s]["median_box_area_percent"], p)
                for s, p in PROTOCOL.items()}

# How many times larger the whole-region boxes are than the localisation
# boxes, against the largest localisation median and against the smallest.
# The first is quoted in 3.4 and 5.2, and both in Table 3.2's caption.
# Computed from the unrounded medians: taken from the printed ones, the
# second came out 107 where it is 105.
_WHOLE_AREA = BOX_GEOMETRY["weld_surface_v2"][1]
_LOC_AREAS = [v[1] for v in BOX_GEOMETRY.values() if v[2] == "localisation"]
PROTOCOL_AREA_FOLD = int(q(_WHOLE_AREA / max(_LOC_AREAS), 0))
PROTOCOL_AREA_FOLD_SMALLEST = int(q(_WHOLE_AREA / min(_LOC_AREAS), 0))

# In-domain mean defect-class AP50: per cell the mean over that source's
# classes with in_defect_space true, averaged over the three seeds and then
# over the four configurations. From per_class_ap.csv, main rows for the three
# training sources and reference rows for the two test-only ones.
#
# Quoted in 5.2. An earlier draft of that paragraph argued the protocol point
# from "an AP50 of 0.94 under the whole-region protocol and 0.11 under the
# localisation protocol". Neither figure is in any released table. 0.94
# appears nowhere in the study, and 0.11 is kaggle_v2's YOLO11s value quoted
# as though it stood for the localisation protocol as a whole, when lohi_weld
# reaches 0.567 under the same protocol. The real table does not show the
# protocols separating on in-domain accuracy at all, and the argument does not
# need it to: the protocol difference is geometric, and 5.2 now says so.
def indomain_defect_ap(source: str) -> float:
    per_cfg = []
    for cfg in CFGS:
        per_seed = []
        for seed in range(N_SEEDS):
            ap = [float(r["ap50"]) for r in PER_CLASS_ROWS
                  if r["kind"] == kind_of(source) and r["source"] == source
                  and r["target"] == source and r["seed"] == str(seed)
                  and r["family"] == CFG_CODE[cfg]
                  and r["in_defect_space"] == "True"]
            if not ap:
                raise SystemExit(f"\nCONSISTENCY CHECK FAILED: no in-domain "
                                 f"defect class for {source} {cfg} seed "
                                 f"{seed} in {PER_CLASS_CSV.name}.")
            per_seed.append(statistics.fmean(ap))
        per_cfg.append(statistics.fmean(per_seed))
    return statistics.fmean(per_cfg)


INDOMAIN_DEFECT_AP = {s: indomain_defect_ap(s) for s in PROTOCOL}

# Training protocol, from grid/config.py.
EPOCHS, BATCH, IMGSZ, WORKERS = 150, 32, 640, 2
NMS_IOU, CONF_FLOOR, MAX_DET = 0.70, 0.001, 300
THIN_THRESHOLD = 50
BOX_BOOTSTRAP_DRAWS = 500
FP_BUDGETS = (0.1, 0.5, 1.0)

# Split solver, from weld_rebuild_clean.py.
SPLIT_FRACTIONS = "70/15/15"
SPLIT_SEED = 20260903
SPLIT_TIME_LIMIT_S = 120
T_COPY, T_SCENE = 0.96, 0.90
NCC_SIZES = (32, 128)
N_DIHEDRAL = 8

# Cell counts. Section 4.1 states the same totals.
MAIN_CELLS, REFERENCE_CELLS, DOWNLOADED_CELLS = 36, 24, 9
GRID_CELLS = MAIN_CELLS + REFERENCE_CELLS + DOWNLOADED_CELLS
CONTROL_CELLS, ARMB_CELLS, ARMD_CELLS = 24, 240, 36
ARM_CELLS = CONTROL_CELLS + ARMB_CELLS + ARMD_CELLS

# Arm B design.
ARMB_K = (10, 25, 50, 100, 250)
ARMB_DRAWS = 3
ARMB_EPOCHS, ARMB_BATCH = 50, 8
ARMB_WORST_OVERSHOOT = 9.1

# Compute environment, from the project log 4a and 4b.
GPU = "one NVIDIA H200 NVL MIG 7g.141gb slice"
TORCH_VERSION, ULTRALYTICS_VERSION, CUDA_VERSION = "2.5.1", "8.4.138", "12.1"
GRID_HOURS, ARM_HOURS = 57, 84

# THE REPOSITORY URL LIVES HERE ONLY. The repository was created public on
# 9 October 2026 under the author's GitHub account, `vincyshirva`. An earlier
# address, github.com/vincentldsouza/weld-transfer-benchmark, named an account
# that does not exist, which is why REPO_LIVE exists at all.
#
# While REPO_LIVE is False, every sentence that would carry the address says
# the code will be released on acceptance and prints no address, because
# printing one that nobody has claimed invites someone else to claim it. With
# REPO_LIVE True the build refuses to finish unless the address resolves.
REPO_URL = "https://github.com/vincyshirva/weld-transfer-benchmark"
REPO_LIVE = True
REPO_CHECKED = "9 October 2026"
SPLIT_ENTRIES_VERIFIED = 7784
SPLIT_VERIFY_DATE = "21 September 2026"   # the README states the same date

# Sun et al. 2026, arXiv 2607.04675, read from the paper 20 September 2026.
ICME = {"registered": 86, "submissions": 130, "final_teams": 21,
        "fact_sheets": 12, "screen_weight": 0.4}

# Leakage audit, image-level AP, YOLO11s at three seeds, from
# downloaded_vs_audited.csv. A is the downloaded partition trained and tested
# on itself, C is the audited partition. Quoted in Section 1 and again in
# 4.4.1, so both read from here. The overlap counts the audited test images
# that sit inside the downloaded training split, measured by filename and
# confirmed by sha256, and those that do not.
AUDIT_SOURCES = ("kaggle_v2", "weld_surface_v2", "lohi_weld")


def audit_table() -> dict:
    out = {}
    for s in AUDIT_SOURCES:
        rr = [r for r in read_csv_rows(AUDIT_CSV) if r["source"] == s]
        overlap = {(r["overlap_n_seen"], r["overlap_n_test"],
                    r["overlap_n_held_out"]) for r in rr}
        if len(overlap) != 1:
            raise SystemExit(f"\nCONSISTENCY CHECK FAILED: {AUDIT_CSV.name} "
                             f"gives {s} more than one overlap count.")
        a = mean_of([r["A_downloaded_train_downloaded_test"] for r in rr],
                    N_SEEDS)
        c = mean_of([r["C_audited_train_audited_test"] for r in rr], N_SEEDS)
        # The file's own per-seed differences must average to A minus C.
        shift = mean_of([r["shift_full_pipeline"] for r in rr], N_SEEDS)
        if abs(shift - (a - c)) > 1e-12:
            raise SystemExit(f"\nCONSISTENCY CHECK FAILED: {s} A minus C is "
                             f"{a - c} from the means and {shift} from the "
                             f"file's own difference column.")
        out[s] = (a, c) + tuple(int(x) for x in overlap.pop())
    return out


AUDIT = audit_table()
AUDIT_A_C = {s: (v[0], v[1]) for s, v in AUDIT.items()}
AUDIT_A_MINUS_C = {s: v[0] - v[1] for s, v in AUDIT.items()}
# (in the downloaded training split, audited test images, held out)
AUDIT_OVERLAP = {s: v[2:] for s, v in AUDIT.items()}
# The downloaded-trained kaggle_v2 model scored on the audited test split,
# box level, YOLO11s. Mostly its own training images, the project log 5j.
DOWNLOADED_KAGGLE_BOX = cell_mean("box_map50", "kaggle_v2", "kaggle_v2",
                                  "YOLO11s", "dirty", "audited")

# Recall at a fixed false positive budget per image, from
# recall_at_fp_budget_arms.csv, one column per budget in FP_BUDGETS.
RECALL = read_csv_rows(RECALL_CSV)
FP_KEYS = tuple(f"recall_at_{b}_fp" for b in FP_BUDGETS)


def recall_means(arm: str, source: str, target: str, cfg=None,
                 k=None) -> tuple:
    """Mean recall at each budget over the seeds, or over the draws, and
    over the four configurations as well when cfg is None."""
    rr = [r for r in RECALL
          if r["arm"] == arm and r["source"] == source
          and r["target"] == target
          and (cfg is None or r["family"] == CFG_CODE[cfg])
          and (k is None or r["k"] == str(k))]
    n = N_SEEDS * (1 if cfg else len(CFGS))
    return tuple(mean_of([r[key] for r in rr], n) for key in FP_KEYS)


# Arm B at the LARGEST budget only. The released markdown pools arm B over
# the whole k grid, and a recall pooled across annotation budgets is not a
# quotable quantity.
#   (source, target, config, recall at each budget)
FP_ARMB_K250 = [(s, t, cfg) + recall_means("armb", s, t, cfg, max(ARMB_K))
                for s, t, cfg in sorted({(r["source"], r["target"],
                                          FAMILY_CODE[r["family"]])
                                         for r in RECALL
                                         if r["arm"] == "armb"})]

# Which family each configuration belongs to. The abstract and Section 3.5
# say "four configurations from three families", and that count is read here
# rather than typed, because the project log forbids "four detector families".
CFG_FAMILY = {"Faster R-CNN": "Faster R-CNN", "RT-DETR-l": "RT-DETR",
              "YOLO11n": "YOLO11", "YOLO11s": "YOLO11"}
N_FAMILIES = len(set(CFG_FAMILY.values()))

# Bootstrap draw counts, one constant per released file, because the counts
# are NOT uniform and a caption once quoted the wrong one. The main matrix
# uses 2,000 draws at image level and 500 at box level; the internal control
# is box level at 2,000; the thin-class per-class intervals use 2,000.
# `check_draw_counts` reads each from the rows it describes.
CONTROL_BOOTSTRAP_DRAWS = 2000
THIN_BOOTSTRAP_DRAWS = 2000

# ------------------------------------------------- internal acquisition control
# Box mAP50 of the 24 control models on both acquisitions, read from the map50
# column of recall_at_fp_budget_arms.csv. That column carries the evaluator's
# own values, the ones lohi_control.md is written from. The interval CSV's
# p_ij is the bootstrap's own point estimate and differs from them by up to
# 1e-6 in one cell, which is tie ordering, the project log 5m. Mean and SD are over
# four configurations and three seeds, 12 cells per entry.
ACQ = ("lohi_high", "lohi_low")


def control_box(source: str, target: str, cfg=None) -> list:
    v = [float(r["map50"]) for r in RECALL
         if r["arm"] == "lohictl" and r["source"] == source
         and r["target"] == target
         and (cfg is None or r["family"] == CFG_CODE[cfg])]
    if len(v) != N_SEEDS * (1 if cfg else len(CFGS)):
        raise SystemExit(f"\nCONSISTENCY CHECK FAILED: {len(v)} control "
                         f"cells for {source} to {target}.")
    return v


CONTROL_BOX = {(s, t): (statistics.fmean(control_box(s, t)),
                        statistics.stdev(control_box(s, t)))
               for s in ACQ for t in ACQ}
# The pooled transfer ratio is the cross cell over the TARGET's diagonal,
# taken between the unrounded means.
CONTROL_T = {(s, t): CONTROL_BOX[(s, t)][0] / CONTROL_BOX[(t, t)][0]
             for s, t in (("lohi_high", "lohi_low"),
                          ("lohi_low", "lohi_high"))}
# High to low per configuration, quoted in 4.2 as a span.
CONTROL_HIGH_TO_LOW = {c: statistics.fmean(control_box("lohi_high",
                                                       "lohi_low", c))
                       for c in CFGS}
# Table 4.11, and the in-domain and cross-acquisition recall per budget.
RECALL_CONTROL = {(s, t): recall_means("lohictl", s, t)
                  for s in ACQ for t in ACQ}
RECALL_IN = [[RECALL_CONTROL[(a, a)][i] for a in ACQ]
             for i in range(len(FP_BUDGETS))]
RECALL_CROSS = [[RECALL_CONTROL[st][i] for st in CONTROL_T]
                for i in range(len(FP_BUDGETS))]


def control_image_ap() -> list:
    """The control's image-level AP, one value per train and test
    acquisition, as lohi_control.md prints it at four decimals. No CSV
    carries it. q_stored refuses a stored value that could round differently
    a second time, so what 4.2 prints at three decimals is what rounding the
    unrounded value once would give."""
    section = CONTROL_MD.read_text().split("## Image level", 1)[1]
    section = section.split("\n## ", 1)[0]
    vals = [cell.split("(")[0].strip()
            for line in section.splitlines() if line.startswith("| lohi_")
            for cell in line.split("|")[2:4]]
    if len(vals) != len(ACQ) ** 2:
        raise SystemExit(f"\nCONSISTENCY CHECK FAILED: {CONTROL_MD.name} "
                         f"gives {len(vals)} image-level control values.")
    for v in vals:
        q_stored(v, 3)
    return [Decimal(v) for v in vals]


def control_intervals() -> dict:
    """The control's split-block intervals, READ FROM THE RELEASED CSV.

    The intervals are released per configuration and seed, twelve per
    direction, and there is no interval on the pooled ratio. An earlier draft
    of Section 4.2 quoted the envelope of the twelve as if it were one 95
    percent interval on each pooled ratio. The envelope is returned here under
    that name, together with the matched comparison that the asymmetry
    paragraph in 5.2 needs, so neither has to be typed.
    """
    rows = [r for r in csv.DictReader(CONTROL_CSV.open())
            if r["source"] != r["target"]]
    per = {}
    for r in rows:
        per[(r["source"], r["target"], FAMILY_CODE[r["family"]],
             int(r["seed"]))] = (float(r["transfer_ratio"]), float(r["lo"]),
                                 float(r["hi"]))
    out = {"envelope": {}, "n_cells": {}}
    for d in CONTROL_T:
        cells = [v for k, v in per.items() if (k[0], k[1]) == d]
        out["n_cells"][d] = len(cells)
        out["envelope"][d] = (min(v[1] for v in cells),
                              max(v[2] for v in cells))
    hl, lh = ("lohi_high", "lohi_low"), ("lohi_low", "lohi_high")
    matched = [(per[hl + (c, s)], per[lh + (c, s)])
               for (a, b, c, s) in per if (a, b) == hl]
    out["matched"] = len(matched)
    out["low_to_high_larger"] = sum(y[0] > x[0] for x, y in matched)
    out["disjoint"] = sum(x[2] < y[1] or y[2] < x[1] for x, y in matched)
    return out


# ------------------------------------------------------- label cost curve
# Table 4.8, one row per pair and configuration, primary endpoint, from the
# per-draw rows of armb_recovery.csv. P(0) and P_jj are means over the three
# draws at the largest budget. R is the MEAN OF THE PER-DRAW RATIOS, each
# normalised by its own seed's denominator, the project log 6a-xix, so it is not
# recoverable from the two denominator columns by division.
#   (source, target, config, P(0), k=10, k=50, k=250, P_jj, R@250, weak)
def armb_table() -> list:
    acc = {}
    for r in read_csv_rows(ARMB_CSV):
        if (r["metric"] != "box" or r["endpoint"] != "last"
                or r["recovery"] == ""):
            continue
        key = (r["source"], r["target"], FAMILY_CODE[r["family"]])
        acc.setdefault(key, {}).setdefault(int(r["k_requested"]),
                                           []).append(r)
    top = max(ARMB_K)
    out = []
    for (src, tgt, cfg), by_k in sorted(acc.items()):
        exhausted = {x["exhausted_train_split"] == "True" for x in by_k[top]}
        if exhausted != {tgt == "twoclass"}:
            raise SystemExit(f"\nCONSISTENCY CHECK FAILED: {src} to {tgt} "
                             f"{cfg} exhaustion flag {exhausted}.")

        def m(key: str, k: int) -> float:
            return mean_of([x[key] for x in by_k[k]], ARMB_DRAWS)

        out.append((src, tgt, cfg, m("p_zero", top), m("ap", 10),
                    m("ap", 50), m("ap", top), m("p_diag", top),
                    m("recovery", top), BOX_OFF[(src, tgt)][cfg][4]))
    # Six localisation pairs, two configurations: YOLO11s and RT-DETR-l.
    if len(out) != 2 * len(BOX_PAIRS):
        raise SystemExit(f"\nCONSISTENCY CHECK FAILED: {T('label_cost')} expects "
                         f"{2 * len(BOX_PAIRS)} rows and {ARMB_CSV.name} "
                         f"gives {len(out)}.")
    return out


ARMB_ROWS = armb_table()
# The band quoted for "most" pairs. `armb_summary` counts how many of the
# twelve fall inside it at two decimals, so the count cannot be typed wrong.
ARMB_R250_BAND = (0.29, 0.74)
TWOCLASS_TRAIN = SPLITS["twoclass"][0]      # the budget a twoclass draw hits

# ------------------------------------------------------- schedule controls
# The exploratory controls of 4.5.1 and 4.5.2, from matched_step_control.csv.
# These are laptop cells, compared with one another and never with the
# cluster's, the project log 6a-xi. All YOLO11s, at ten labelled images unless a
# budget is given.
SCHEDULE = read_csv_rows(SCHEDULE_CSV)


def schedule_draws(schedule: str, source: str, target: str,
                   k: int = 10) -> list:
    rr = sorted((r for r in SCHEDULE if r["schedule"] == schedule
                 and r["source"] == source and r["target"] == target
                 and r["k"] == str(k)), key=lambda r: int(r["draw"]))
    if [int(r["draw"]) for r in rr] != list(range(ARMB_DRAWS)):
        raise SystemExit(f"\nCONSISTENCY CHECK FAILED: {SCHEDULE_CSV.name} "
                         f"lacks a full set of draws for {schedule} {source} "
                         f"to {target} at k = {k}.")
    return [float(r["box_map50"]) for r in rr]


def schedule_mean(schedule: str, source: str, target: str,
                  k: int = 10) -> float:
    return statistics.fmean(schedule_draws(schedule, source, target, k))


def schedule_delta_r(source: str, target: str) -> float:
    """The oversampled arm's gain over the fixed arm, as a change in
    recovery. The denominator is P_jj minus P(0) from main-grid YOLO11s means
    over the three seeds, the project log 6a-xvi."""
    p0 = cell_mean("box_map50", source, target, "YOLO11s")
    pjj = cell_mean("box_map50", target, target, "YOLO11s", kind_of(target))
    gain = (schedule_mean("oversampled_50", source, target)
            - schedule_mean("fixed_50", source, target))
    return gain / (pjj - p0)


# Table 4.9, kaggle_v2 to lohi_weld: the step count alone, fixed to
# oversampled, then the schedule shape alone, oversampled to matched step.
# Each is a difference of three-draw means.
SCHEDULE_TABLE = {
    k: (schedule_mean("oversampled_50", "kaggle_v2", "lohi_weld", k)
        - schedule_mean("fixed_50", "kaggle_v2", "lohi_weld", k),
        schedule_mean("matched_step", "kaggle_v2", "lohi_weld", k)
        - schedule_mean("oversampled_50", "kaggle_v2", "lohi_weld", k))
    for k in (10, 50)}



# --------------------------------------------- the image level, in AUROC
# REWRITTEN 8 OCTOBER 2026. Image-level AP minus the positive rate, the lift,
# was read as if zero meant no skill. Under the evaluator's interpolated
# estimator an uninformative scorer sits above zero, +0.052 on kaggle_v2's
# split, the project log 5y. So the image level is reported in AUROC, whose chance
# level is 0.5, from the tables grid/image_level.py releases. Nothing here is
# computed beyond a mean, a minimum or a count over released rows, and every
# claim the text makes in words is asserted in image_level_facts().
IL_MEANS_CSV = TABLES / "image_level_means.csv"
IL_MARGIN_CSV = TABLES / "image_level_retained_margin.csv"
IL_SAMPLES_CSV = TABLES / "image_level_samples.csv"
IL_AUDIT_CSV = TABLES / "image_level_audit.csv"
IL_KAGGLE_CSV = TABLES / "image_level_kaggle_checks.csv"
IL_ROWS_CSV = TABLES / "image_level.csv"
IL_ARMS_CSV = ARMS / "image_level_arms.csv"
IL_ARMS_SAMPLES_CSV = ARMS / "image_level_samples_arms.csv"
IL_CONTROL_CSV = ARMS / "image_level_control_margin.csv"
IL_COST_CSV = ARMS / "image_level_label_cost.csv"
IL_POOL_CSV = ARMS / "image_level_pooling.csv"
IL_POOL_SEED_CSV = ARMS / "image_level_pooling_seed_averaged.csv"
IMAGE_LEVEL_DRAWS = 10000       # every AUROC interval, checked against rows
IMAGE_LEVEL_SHUFFLES = 2000     # the chance AP, checked against rows
NO_VIDEO = "without the SampleV1_2_mp4 block"
IL_TARGET_ORDER = ("kaggle_v2", "weld_surface_v2", "lohi_weld", "visual",
                   "twoclass")
MSC_SCRIPT = ROOT / "grid" / "arms" / "matched_step_control.py"
GPU_JSON = ROOT / "gpu_throughput.json"
H200_PINS = ROOT / "h200_requirements.txt"
LAPTOP_PINS = ROOT / "laptop_requirements.txt"
# The evaluator's rule for a false alarm rate, as the released rows record it.
MIN_NEGATIVES = int(read_csv_rows(IL_ROWS_CSV)[0]["min_negatives_rule"])


def source_constant(path: Path, needle: str, label: str) -> str:
    """Fail the build unless a line of training code still reads exactly as
    the manuscript describes it. For recipe values held in code rather than
    in a configuration dictionary."""
    text = path.read_text()
    if needle not in text:
        raise SystemExit(f"\nCONSISTENCY CHECK FAILED: the text states "
                         f"{label}, read from {path.relative_to(ROOT)} as "
                         f"{needle!r}, and that text is no longer there.")
    return needle


def fnum(v):
    return None if v in ("", None) else float(v)


def contains(lo, hi, x) -> bool:
    return fnum(lo) <= x <= fnum(hi)


def pins(path: Path) -> dict:
    """Package versions pinned in a requirements file, local tags dropped."""
    out = {}
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()
        if "==" in line:
            name, ver = line.split("==", 1)
            out[name.strip().lower()] = ver.split("+")[0].strip()
    return out


def image_level_facts() -> dict:
    """Every image-level number the rewritten text quotes, and every claim it
    makes in words, asserted against the released rows."""
    def fail(msg):
        raise SystemExit(f"\nCONSISTENCY CHECK FAILED: {msg}")

    means = {(r["source"], r["target"], r["family"] or None): r
             for r in read_csv_rows(IL_MEANS_CSV)}
    allr = {k: v for k, v in means.items() if k[2] is None}
    ind = {k[1]: v for k, v in allr.items() if v["in_domain"] == "True"}
    cross = [v for v in allr.values() if v["in_domain"] != "True"]
    if sorted(ind) != sorted(IL_TARGET_ORDER) or len(cross) != 12:
        fail(f"{IL_MEANS_CSV.name} holds {len(ind)} in-domain and "
             f"{len(cross)} cross-dataset rows, expected 5 and 12.")
    a_in = {t: fnum(v["mean_auroc"]) for t, v in ind.items()}
    in_above = sum(int(v["cells_wholly_above_half"]) for v in ind.values())
    in_n = sum(int(v["n_cells"]) for v in ind.values())
    if in_above != in_n:
        fail(f"4.4 says all {in_n} in-domain evaluations sit wholly above "
             f"0.5, and {in_above} do.")
    cm = [fnum(v["mean_auroc"]) for v in cross]
    c_above = sum(int(v["cells_wholly_above_half"]) for v in cross)
    c_below = sum(int(v["cells_wholly_below_half"]) for v in cross)
    c_n = sum(int(v["n_cells"]) for v in cross)
    kag = [v for v in cross if v["target"] == "kaggle_v2"]
    for v in kag:
        if int(v["cells_wholly_above_half"]) or \
                int(v["cells_wholly_below_half"]) or \
                not contains(v["mean_auroc_lo"], v["mean_auroc_hi"], 0.5):
            fail(f"the abstract, 1, 4.4, 5.3 and 6 say no foreign detector's "
                 f"decision on kaggle_v2 is distinguishable from chance, and "
                 f"{v['source']} gives {v['mean_auroc']} "
                 f"[{v['mean_auroc_lo']}, {v['mean_auroc_hi']}].")
    worst = min(cross, key=lambda v: fnum(v["mean_auroc"]))
    best = max(cross, key=lambda v: fnum(v["mean_auroc"]))
    if (worst["source"], worst["target"]) != ("weld_surface_v2",
                                               "lohi_weld") \
            or not fnum(worst["mean_auroc_hi"]) < 0.5:
        fail(f"4.4 names weld_surface_v2 to lohi_weld as inverted, and the "
             f"lowest cross-dataset mean is {worst['source']} to "
             f"{worst['target']} [{worst['mean_auroc_lo']}, "
             f"{worst['mean_auroc_hi']}].")
    if not fnum(best["mean_auroc_lo"]) > 0.5:
        fail("Section 1 says the decision is useful on some foreign target, "
             "and no cross-dataset interval lies wholly above 0.5.")
    chance = {t: fnum(v["mean_chance_lift"]) for t, v in ind.items()}
    if not all(fnum(v["mean_chance_lift"]) > 0 for v in allr.values()):
        fail("the abstract says the lift exceeds zero for uninformative "
             "scores, and a released chance lift is not above zero.")
    if min(chance, key=chance.get) != "lohi_weld":
        fail(f"4.4 says the chance lift falls to its lowest on lohi_weld, "
             f"and the in-domain chance lifts are {chance}.")

    samples = {r["split"]: r for r in read_csv_rows(IL_SAMPLES_CSV)}
    arm_samples = {r["split"]: r for r in read_csv_rows(IL_ARMS_SAMPLES_CSV)}
    for t in IL_TARGET_ORDER:
        n_pos, n_neg = BASE_RATE_COUNTS[t]
        if (int(samples[t]["n_pos"]), int(samples[t]["n_neg"])) != \
                (n_pos, n_neg):
            fail(f"{IL_SAMPLES_CSV.name} and target_base_rates.csv disagree "
                 f"on {t}'s test split.")
    under = [t for t in IL_TARGET_ORDER
             if int(samples[t]["n_neg"]) < MIN_NEGATIVES]

    def checkrow(check, variant, source, family=None):
        rr = [r for r in read_csv_rows(IL_KAGGLE_CSV) if r["check"] == check
              and r["variant"] == variant and r["source"] == (source or "")
              and r["family"] == (family or "")]
        if len(rr) != 1:
            fail(f"{IL_KAGGLE_CSV.name} has {len(rr)} rows for {check}, "
                 f"{variant}, {source}, {family}.")
        return rr[0]

    nv = checkrow("label rule", NO_VIDEO, "lohi_weld")
    k_pos, k_neg = BASE_RATE_COUNTS["kaggle_v2"]
    video = {"images": int(samples["kaggle_v2"]["largest_block_images"]),
             "pos": k_pos - int(nv["n_pos"]), "neg": k_neg - int(nv["n_neg"])}
    if video["images"] != video["pos"] + video["neg"]:
        fail("the largest kaggle_v2 block is not the video block that the "
             "left-out check removes.")
    family = checkrow("filename family", "positive share of the image's "
                      "family among the other blocks", None)
    if not (fnum(family["mean_auroc"]) > a_in["kaggle_v2"]
            and fnum(family["mean_auroc_lo"]) > 0.5):
        fail("4.4 says the filename score beats the in-domain detectors and "
             "is itself well above chance.")
    variants = [("label rule", "Bad Weld or Defect present"),
                ("label rule", "Defect against Good Weld only"),
                ("score rule", "sum of the scores"),
                ("score rule", "count of predictions at or above 0.25"),
                ("score rule", "count of predictions at or above 0.10"),
                ("score rule", "mean of the top three, zero where fewer")]
    for check, variant in variants:
        for src in ("weld_surface_v2", "lohi_weld"):
            r = checkrow(check, variant, src)
            if not contains(r["mean_auroc_lo"], r["mean_auroc_hi"], 0.5):
                fail(f"4.4 says {variant} does not change the reading, and "
                     f"{src} gives [{r['mean_auroc_lo']}, "
                     f"{r['mean_auroc_hi']}].")
    exception = checkrow("score rule", "sum of the scores", "lohi_weld",
                         "faster_rcnn")
    deposit = checkrow("one class at a time",
                       "lohi_weld model, deposit alone", "lohi_weld")

    margins = {(r["subset"], r["level"], r["source"], r["target"]): r
               for r in read_csv_rows(IL_MARGIN_CSV)
               if r["level"] == "all"}
    lk = {s: margins[(s, "all", "lohi_weld", "kaggle_v2")]
          for s in ("full", NO_VIDEO)}
    for s, r in lk.items():
        lo, hi = fnum(r["retained_margin_lo"]), fnum(r["retained_margin_hi"])
        if not (lo < 0 < hi and lo < 0.13 and hi > 0.21):
            fail(f"4.4 says neither lohi_weld to kaggle_v2 interval excludes "
                 f"zero or 13 to 21 percent, and the {s} one is "
                 f"[{lo}, {hi}].")

    audit = {}
    for r in read_csv_rows(IL_AUDIT_CSV):
        audit.setdefault(r["source"], []).append(r)
    au = {}
    for s, rr in audit.items():
        if len(rr) != N_SEEDS:
            fail(f"{IL_AUDIT_CSV.name} holds {len(rr)} seeds for {s}.")
        m = lambda k: statistics.fmean(fnum(r[k]) for r in rr)  # noqa: E731
        worst_pairs = max(rr, key=lambda r: fnum(r["A_pairs_out_of_order"]))
        au[s] = {"A": m("A_auroc"), "C": m("C_auroc"),
                 "A_C": m("A_minus_C_auroc"), "A_C_ap": m("A_minus_C_image_ap"),
                 "A_ap": m("A_image_ap"), "C_ap": m("C_image_ap"),
                 "A_neg": int(rr[0]["A_n_neg"]), "C_neg": int(rr[0]["C_n_neg"]),
                 "A_pos": int(rr[0]["A_n_pos"]),
                 "C_rate": fnum(rr[0]["C_positive_rate"]),
                 # (1 - AUROC) times the pair count, rounded to the half
                 # pair a tie can contribute, so float error cannot move it.
                 "pairs_bad": round(2 * fnum(
                     worst_pairs["A_pairs_out_of_order"])) / 2,
                 "pairs": int(worst_pairs["A_pairs"])}
        check_equal(f"4.4.1 {s} C AP against downloaded_vs_audited.csv",
                    au[s]["C_ap"], AUDIT_A_C[s][1], 9)
    moved = [s for s in AUDIT_SOURCES if au[s]["A_C"] > 0.05]
    if sorted(moved) != ["kaggle_v2", "weld_surface_v2"] \
            or not au["lohi_weld"]["A_C"] <= 0:
        fail(f"1, 4.4.1 and 6 say auditing lowers in-domain AUROC on two "
             f"public sources and not on lohi_weld. A minus C is "
             f"{ {s: v['A_C'] for s, v in au.items()} }.")
    if not au["weld_surface_v2"]["pairs_bad"] <= 2:
        fail("4.4.1 says at most 2 pairs are misordered on weld_surface_v2.")

    ctl = {r["source"]: r for r in read_csv_rows(IL_CONTROL_CSV)
           if r["level"] == "all"}["lohi_high"]
    hi_s, lo_s = arm_samples["lohi_high"], arm_samples["lohi_low"]
    if int(hi_s["n_neg"]) != 0 or int(lo_s["neg_blocks"]) != int(
            lo_s["n_neg"]):
        fail("4.2 says lohi_high's test split holds no defect-free image and "
             "lohi_low's each sit in their own block.")
    arms_rows = read_csv_rows(IL_ARMS_CSV)
    n_high = sum(1 for r in arms_rows if r["arm"] == "lohi_control"
                 and r["target"] == "lohi_high")

    cost = [r for r in read_csv_rows(IL_COST_CSV)]
    meets = sorted({r["target"] for r in cost
                    if r["meets_min_negatives"] == "True"})
    if meets != ["kaggle_v2", "weld_surface_v2"]:
        fail(f"4.5 says kaggle_v2 and weld_surface_v2 are the targets that "
             f"clear the negatives rule, and the label cost file says "
             f"{meets}.")
    lc = {}
    for cfg in ("rtdetr_l", "yolo11s"):
        rr = sorted((r for r in cost if r["target"] == "kaggle_v2"
                     and r["source"] == "lohi_weld" and r["family"] == cfg),
                    key=lambda r: int(r["k_requested"]))
        if [int(r["k_requested"]) for r in rr] != list(ARMB_K):
            fail(f"the label cost file lacks a budget for {cfg}.")
        lc[cfg] = rr
    small = []
    for rr in lc.values():
        small.append((rr[0]["zero_shot_margin_lo"],
                      rr[0]["zero_shot_margin_hi"]))
        small += [(r["retained_margin_lo"], r["retained_margin_hi"])
                  for r in rr if int(r["k_requested"]) <= 100]
    small_excl = sum(1 for lo, hi in small if not contains(lo, hi, 0.0))

    pool = read_csv_rows(IL_POOL_CSV)
    pool_seed = read_csv_rows(IL_POOL_SEED_CSV)

    def wins(rows, k):
        return sum(r[f"beats_{k}_auroc"] == "True" for r in rows)

    for r in read_csv_rows(IL_ROWS_CSV) + arms_rows:
        if int(r["auroc_resamples"]) != IMAGE_LEVEL_DRAWS or \
                int(r["chance_shuffles"]) != IMAGE_LEVEL_SHUFFLES:
            fail(f"an image-level row records {r['auroc_resamples']} draws "
                 f"and {r['chance_shuffles']} shuffles, and the text quotes "
                 f"{IMAGE_LEVEL_DRAWS:,} and {IMAGE_LEVEL_SHUFFLES:,}.")
    for r in read_csv_rows(IL_MEANS_CSV) + read_csv_rows(IL_MARGIN_CSV):
        if int(r["resamples"]) != IMAGE_LEVEL_DRAWS:
            fail(f"an image-level summary row records {r['resamples']} draws.")

    return {
        "in": a_in, "in_range": (min(a_in.values()), max(a_in.values())),
        "in_n": in_n, "cross_range": (min(cm), max(cm)),
        "cross_above": c_above, "cross_below": c_below, "cross_n": c_n,
        "kaggle_n": sum(int(v["n_cells"]) for v in kag),
        "inverted": worst, "means": means, "chance": chance,
        "constant_kaggle": fnum(samples["kaggle_v2"][
            "constant_score_ap_above_base"]),
        "samples": samples, "arm_samples": arm_samples, "under": under,
        "video": video, "family": family, "exception": exception,
        "deposit": deposit, "lk": lk, "audit": au, "control": ctl,
        "n_high": n_high, "cost": lc, "small_n": len(small),
        "small_excl": small_excl,
        "pool": {"n": len(pool),
                 "pooled": statistics.fmean(fnum(r["pooled_auroc"])
                                            for r in pool),
                 "better": statistics.fmean(fnum(r["better_member_auroc"])
                                            for r in pool),
                 "worse": statistics.fmean(fnum(r["worse_member_auroc"])
                                           for r in pool),
                 "better_wins": wins(pool, "better"),
                 "mean_wins": wins(pool, "mean"),
                 "worse_wins": wins(pool, "worse"),
                 "seed_n": len(pool_seed),
                 "seed_better": wins(pool_seed, "better"),
                 "seed_mean": wins(pool_seed, "mean"),
                 "seed_worse": wins(pool_seed, "worse")},
        "pool_seed": pool_seed,
    }


# 4.5.1 and 5.3: the label cost arm's draws into twoclass, at the primary
# endpoint, box level, from armb_recovery.csv. Ratios of max over min are
# quoted only into twoclass, where no score sits at the floor.
def twoclass_draws() -> dict:
    rows = [r for r in read_csv_rows(ARMB_CSV) if r["metric"] == "box"
            and r["endpoint"] == "last" and r["target"] == "twoclass"]
    by = {}
    for r in rows:
        by.setdefault((r["source"], r["family"], int(r["k_requested"])),
                      []).append(r)
    combos = sorted({(s, f) for s, f, _ in by})
    out = {"combos": len(combos)}

    def draws(s, f, k):
        rr = sorted(by[(s, f, k)], key=lambda r: int(r["draw"]))
        if [int(r["draw"]) for r in rr] != list(range(ARMB_DRAWS)):
            raise SystemExit(f"\nCONSISTENCY CHECK FAILED: twoclass draws "
                             f"for {s} {f} at k = {k}.")
        return rr

    small = {c: draws(*c, 10) for c in combos}
    first = next(iter(small.values()))
    out["k10_images"] = [int(r["n_images"]) for r in first]
    out["k10_scenes"] = [int(r["n_scenes"]) for r in first]
    for rr in small.values():
        if [int(r["n_images"]) for r in rr] != out["k10_images"]:
            raise SystemExit("\nCONSISTENCY CHECK FAILED: 4.5.1 says the "
                             "twoclass draws at ten images are shared.")
    spread = [max(float(r["ap"]) for r in rr) / min(float(r["ap"])
                                                   for r in rr)
              for rr in small.values()]
    out["k10_spread"] = (min(spread), max(spread))
    top = {c: draws(*c, max(ARMB_K)) for c in combos}
    for rr in top.values():
        if {r["exhausted_train_split"] for r in rr} != {"True"} or \
                {int(r["n_images"]) for r in rr} != {TWOCLASS_TRAIN}:
            raise SystemExit("\nCONSISTENCY CHECK FAILED: 4.5.1 says every "
                             "twoclass draw at the largest budget holds the "
                             "same 213 training images.")
    out["top_spread"] = max(max(float(r["ap"]) for r in rr)
                            / min(float(r["ap"]) for r in rr)
                            for rr in top.values())
    most = fewest = n_mid = 0
    for k in (50, 100):
        for c in combos:
            rr = draws(*c, k)
            sc = [int(r["n_scenes"]) for r in rr]
            ap = [float(r["ap"]) for r in rr]
            if sc.count(max(sc)) != 1 or sc.count(min(sc)) != 1:
                raise SystemExit("\nCONSISTENCY CHECK FAILED: 4.5.1 needs "
                                 "one draw with the most and one with the "
                                 f"fewest scenes at k = {k}.")
            most += ap.index(max(ap)) == sc.index(max(sc))
            fewest += ap.index(min(ap)) == sc.index(min(sc))
            n_mid += 1
    out.update({"most_best": most, "fewest_worst": fewest,
                "mid_n": n_mid})
    return out


TWOCLASS_DRAWS = twoclass_draws()


def schedule_recovery(schedule: str, source: str, target: str,
                      k: int = 10) -> float:
    """Recovery of a schedule-control mean, normalised as Table 4.8 is, by
    the main-grid YOLO11s three-seed means of P(0) and P_jj."""
    p0 = cell_mean("box_map50", source, target, "YOLO11s")
    pjj = cell_mean("box_map50", target, target, "YOLO11s", kind_of(target))
    return (schedule_mean(schedule, source, target, k) - p0) / (pjj - p0)


def schedule_counts() -> dict:
    """The exploratory runs, counted by schedule. matched_step_control.csv
    does not record the configuration, so it is read from the runner."""
    fam = source_constant(MSC_SCRIPT, 'FAMILY = "yolo11s"',
                          "that every schedule control is YOLO11s")
    c = {}
    for r in SCHEDULE:
        c[r["schedule"]] = c.get(r["schedule"], 0) + 1
    return {"n": len(SCHEDULE), "by": c,
            "family": FAMILY_CODE[fam.split('"')[1]]}


SCHEDULE_COUNTS = schedule_counts()

# The pre-registered tests, as statistical_tests.json and stats_arms.json
# record them. Those files store some statistics already rounded, a
# chi-squared to two decimals and an F to four, so those are printed through
# q_stored.
STATS = json.loads(STATS_JSON.read_text())
STATS_ARMS = json.loads(STATS_ARMS_JSON.read_text())



# ----------------------------------------------------------- calibration
# Everything in 4.8 is read from two released files, and Figure 4.4 is drawn
# from the second.
#
# FIGURE 4.4 WAS BLANK until 24 September 2026. It used to be the main grid's
# figures/reliability.png, which grid/analysis.py drew after reading the
# calibration bins under the keys lower, upper, confidence and accuracy. The
# evaluator writes them as lo, hi, mean_confidence and observed_positive_rate,
# so only count matched, every point was NaN, and the figure held a diagonal
# and nothing else. Every check in this file passed, because each checks a
# number and a blank figure contains none. The reader now takes the
# evaluator's keys, and the pooled figure is retired altogether: see the
# comment where grid/analysis.py calls reliability_rows.
#
# Redrawing it showed that 4.8 read the result backwards. The text said the
# hazard was a detector that becomes wrong while staying confident, and that a
# few-shot head "emits confident scores". Every control evaluation and every
# fine-tuned model is UNDERCONFIDENT: the scores sit below the share of images
# that carry a defect. reliability_direction() computes that from the bins and
# check_calibration() fails the build if it stops being true.
#
# Table 4.13 also carried 0.287 and 0.289 for YOLO11n and YOLO11s in domain.
# Those are calibration_arms.md's four-decimal 0.2865 and 0.2885 rounded a
# second time. The CSV's 0.28647 and 0.28847 round to 0.286 and 0.288. The
# table is now read from the CSV and rounded once.
CALIB_CSV = ARMS / "calibration_arms.csv"
RELIABILITY_CSV = ARMS / "reliability_bins_arms.csv"
RELIABILITY_FIG = ARMS / "reliability_control.png"
RELIABILITY_SCRIPT = ROOT / "grid" / "arms" / "reliability_control_figure.py"
# Lines in Figure 4.4 join bins holding at least this many images. The caption
# prints it, so it must equal MIN_LINE_COUNT in the figure script.
RELIABILITY_MIN_LINE = 10
# 4.8 says that at every one of these thresholds, and for every
# configuration, a smaller share of the defective images clears the threshold
# across the acquisition boundary than in domain. Bin edges 6/15 to 12/15.
THRESHOLD_EDGES = tuple(i / 15 for i in range(6, 13))


def control_ece() -> dict:
    """Table 4.13 from calibration_arms.csv: per configuration, the mean ECE
    over the six in-domain and the six cross-acquisition control
    evaluations, unrounded. Printing rounds it once."""
    acc = {}
    for r in csv.DictReader(CALIB_CSV.open()):
        if r["arm"] == "lohictl":
            acc.setdefault((FAMILY_CODE[r["family"]], r["shift"]),
                           []).append(float(r["ece"]))
    out = {}
    for cfg in CFGS:
        a = acc.get((cfg, "in domain"), [])
        b = acc.get((cfg, "cross domain"), [])
        if len(a) != 6 or len(b) != 6:
            raise SystemExit(f"\nCONSISTENCY CHECK FAILED: {T('ece')} says six "
                             f"cells per entry and {CALIB_CSV.name} holds "
                             f"{len(a)} and {len(b)} for {cfg}.")
        out[cfg] = (sum(a) / len(a), sum(b) / len(b))
    return out


# Mean ECE inside the control, in domain and across the acquisition boundary,
# and over the fine-tuned models: the means that 4.8 and 5.3 quote.
ECE_CONTROL = tuple(statistics.fmean(v[i] for v in control_ece().values())
                    for i in (0, 1))
ECE_ARMB = mean_of([r["ece"] for r in read_csv_rows(CALIB_CSV)
                    if r["arm"] == "armb"], ARMB_CELLS)


def reliability_direction() -> dict:
    """What Figure 4.4 shows, computed from the rows it is drawn from.

    `under`   per arm, (evaluations, evaluations that are underconfident).
              An evaluation is underconfident when, weighted by image count
              over its bins, the share of images with a defect exceeds the
              score.
    `ece`     each control evaluation's ECE recomputed from its bins,
              averaged per configuration and regime, to tie the figure's data
              to Table 4.13.
    `clears`  control only, per configuration and regime: the share of the
              defective images whose score clears each threshold in
              THRESHOLD_EDGES.
    `defect_share`  the share of control test images with a defect.
    """
    cells = {}
    for r in csv.DictReader(RELIABILITY_CSV.open()):
        key = (r["arm"], r["family"], r["source"], r["target"], r["seed"],
               r["k"], r["draw"])
        bins = cells.setdefault(key, [])
        n = int(r["count"])
        if n:
            bins.append((float(r["lo"]), n, float(r["mean_confidence"]),
                         float(r["observed_positive_rate"])))
    under, ece, pos, img = {}, {}, {}, [0.0, 0]
    for key, bins in cells.items():
        arm, fam, src, tgt = key[:4]
        n_img = sum(b[1] for b in bins)
        signed = sum(b[1] * (b[3] - b[2]) for b in bins) / n_img
        tot, hit = under.get(arm, (0, 0))
        under[arm] = (tot + 1, hit + (signed > 0))
        if arm != "lohictl":
            continue
        regime = "in domain" if src == tgt else "cross domain"
        ece.setdefault((FAMILY_CODE[fam], regime), []).append(
            sum(b[1] * abs(b[3] - b[2]) for b in bins) / n_img)
        acc = pos.setdefault((FAMILY_CODE[fam], regime), {})
        for lo, n, _, o in bins:
            acc[lo] = acc.get(lo, 0.0) + n * o
        if regime == "in domain":
            img[0] += sum(b[1] * b[3] for b in bins)
            img[1] += n_img
    clears = {k: {e: sum(v for lo, v in acc.items() if lo >= e - 1e-9)
                  / sum(acc.values()) for e in THRESHOLD_EDGES}
              for k, acc in pos.items()}
    return {"under": under,
            "ece": {k: sum(v) / len(v) for k, v in ece.items()},
            "clears": clears, "defect_share": img[0] / img[1]}

# kaggle_v2's verdict-only images, the project log 5 and grid/config.py: images
# with a Bad Weld box and no Defect box, which the rule makes negatives.
KAGGLE_BADWELD_ONLY = 103

# Where twoclass comes from. Found 24 September 2026 on Roboflow Universe.
# The upstream publishes NO dataset version, so it cannot be exported as it
# stands, and the copy this study used is a fork in the author's workspace.
TWOCLASS_FORK_URL = ("https://universe.roboflow.com/"
                     "vincent-linish-d-souza-nmamit-mech/"
                     "welding-defect-2class-dadbu")
LOHI_URL = "https://github.com/SylvioBlock/LoHi-Weld"

# ------------------------------------------------------------ front matter
# Chosen by Vincent on 9 October 2026. It replaced "How far does a
# visible-light weld defect detector travel? A cross-dataset generalisation
# benchmark for surface weld inspection". Sentence case, no colon.
TITLE = ("Transferability of deep object detectors for visible-light weld "
         "inspection under camera and dataset shift")
AUTHOR = "Vincent Linish Dsouza"
# As Vincent gave it on 11 October 2026, for both journals.
AFFILIATION = ("Department of Robotics and Artificial Intelligence, "
               "NMAM Institute of Technology, NITTE (Deemed to be "
               "University), Nitte 574110, Karnataka, India")
EMAIL = "vincentldsouza@nitte.edu.in"
# Given by Vincent on 9 October 2026. The last character is an ISO 7064
# MOD 11-2 check digit, verified at import, so a mistyped digit fails the build.
ORCID = "0000-0003-0533-2146"


def _orcid_ok(orcid: str) -> bool:
    digits = orcid.replace("-", "")
    total = 0
    for ch in digits[:-1]:
        total = (total + int(ch)) * 2
    check = (12 - total % 11) % 11
    return digits[-1] == ("X" if check == 10 else str(check))


if not _orcid_ok(ORCID):
    raise SystemExit(f"\nCONSISTENCY CHECK FAILED: ORCID {ORCID} fails its "
                     f"check digit.")

KEYWORDS = ("Weld defect detection", "Object detection",
            "Cross-dataset generalisation", "Domain shift", "Data leakage",
            "Visual inspection")

# --------------------------------------------------------------- the journal
# SUBMITTED FIRST TO ADVANCED ENGINEERING INFORMATICS, decided by Vincent on
# 11 October 2026. The Journal of Intelligent Manufacturing stays the fallback
# and must still build, with --journal jim. A profile holds only what differs
# between the two; everything else, the numbering, the table rules and the
# figure sizing, applies to both. The AEI rules come from its Guide for
# Authors, saved by Vincent on 11 October 2026.
JOURNAL = "aei"
if "--journal" in sys.argv:
    JOURNAL = sys.argv[sys.argv.index("--journal") + 1]
PROFILES = {
    "aei": {
        "name": "Advanced Engineering Informatics",
        "abstract_words": (1, 250), "keywords": (1, 7),
        "funding": ("This research did not receive any specific grant from "
                    "funding agencies in the public, commercial, or "
                    "not-for-profit sectors."),
        "back_matter": "elsevier", "captions": "elsevier",
        # Printed widths in mm: Elsevier's full 190, one and a half columns
        # 140 and single column 90. Each figure takes the narrowest it fits.
        "fig_mm": {"control_matrix": 90, "box_heatmap": 140,
                   "label_cost_curve": 190, "reliability": 140},
        "highlights": True, "dataset_marker": True,
    },
    "jim": {
        "name": "Journal of Intelligent Manufacturing",
        "abstract_words": (150, 250), "keywords": (4, 6),
        "funding": "No funding was received for conducting this study.",
        "back_matter": "springer", "captions": "springer",
        # Springer's double column 174 and single column 84.
        "fig_mm": {"control_matrix": 84, "box_heatmap": 174,
                   "label_cost_curve": 174, "reliability": 174},
        "highlights": False, "dataset_marker": False,
    },
}
if JOURNAL not in PROFILES:
    raise SystemExit(f"\n--journal must be one of {', '.join(PROFILES)}, "
                     f"not {JOURNAL!r}.")
PROFILE = PROFILES[JOURNAL]
ABSTRACT_WORDS = PROFILE["abstract_words"]
KEYWORD_COUNT = PROFILE["keywords"]
SUBMISSION = Path(__file__).resolve().parent / "submission" / JOURNAL
# The journal whose builds keep the project-root file names. Changing the
# target journal means changing this too.
ROOT_JOURNAL = "aei"

# Confirmed by Vincent on 11 October 2026. CRediT role names are fixed terms
# and keep their own spelling and punctuation, so the dash and British
# spelling checks skip this one paragraph.
CREDIT = ("Vincent Linish Dsouza: Conceptualization, Methodology, Software, "
          "Validation, Formal analysis, Investigation, Data curation, "
          "Visualization, Writing – original draft, Writing – review "
          "& editing.")
COMPETING_ELSEVIER = ("The author declares that he has no known competing "
                      "financial interests or personal relationships that "
                      "could have appeared to influence the work reported in "
                      "this paper.")
# Set when Vincent has archived the GitHub repository on Zenodo. The data
# availability statement then gives the DOI, and the reference list gains a
# dataset entry for the deposit. Until then the paper cites the GitHub address.
ZENODO_DOI = None
ZENODO_TITLE = None             # the deposit's title as Zenodo shows it
ZENODO_YEAR = 2026
# What the repository does not hold, as README.md states it.
CHECKPOINTS_GB = 28

# ------------------------------------------- numbering of tables and figures
# Tables and figures are numbered 1 to N in order of first citation through
# the whole paper, as both journals require, and every cross-reference is
# generated from a label so none can go stale. The order below is the order
# of first citation in the body text, and check_numbering() asserts it on the
# built document: a table cited for the first time out of this order, or
# placed out of it, or never cited in the body, stops the build. Appendix
# tables are numbered within their appendix.
TABLE_ORDER = (
    "sources", "geometry", "recipe",
    "control", "box_yolo11s", "box_ratio", "diagonals", "auroc", "blocks",
    "audit", "label_cost", "schedule", "pooling", "recall_control",
    "recall_cross", "ece", "per_class", "mendieta")
APPENDIX_TABLES = {"class_map": "A.1", "provenance": "B.1"}
FIGURE_ORDER = ("control_matrix", "box_heatmap", "label_cost_curve",
                "reliability")


def table_no(label: str) -> str:
    if label in APPENDIX_TABLES:
        return APPENDIX_TABLES[label]
    return str(TABLE_ORDER.index(label) + 1)


def T(label: str) -> str:
    """An in-text reference to a table."""
    return f"Table {table_no(label)}"


def F(label: str) -> str:
    """An in-text reference to a figure."""
    return f"Figure {FIGURE_ORDER.index(label) + 1}"

# --------------------------------------------------- the tuning question
# HOW THE RECIPES WERE CHOSEN IS THE AUTHOR'S TO STATE, AND ONE SWITCH STATES
# IT. Earlier drafts of Sections 3.5, 4.4 and 5.5 said the hyperparameters
# were tuned once on kaggle_v2's validation split. grid/config.py, whose
# sha256 every trained cell pins, says the recipes were fixed a priori and
# that there was no tuning pass. One of the two is false. The evidence
# gathered on 25 September 2026 is summarised in the project log. It bears on the
# question and does not settle it, and neither does this build.
#   None         undecided. The text keeps the tuned-once account, the working
#                draft marks each passage, and the build refuses to call the
#                manuscript submittable.
#   "kaggle_v2"  tuned once on kaggle_v2's validation split. 3.5, 4.4 and 5.5
#                keep that account, and 4.4 keeps the confound paragraph.
#   "none"       not tuned. 3.5 and 5.5 say each family's recipe was fixed
#                before any model was trained, Table 3.3 lists it, and the 4.4
#                confound paragraph is deleted, since there is no confound to
#                state.
# The CONFIRM entry below follows this switch, so there is one place to set.
# SET 8 OCTOBER 2026 BY VINCENT: no tuning pass was run. The cluster's own
# accounting, read on 6 October, shows no tuning job before the launch,
# the project log 5x. "fixed from its documented defaults" was dropped from the
# "none" texts on the same day, because the frozen recipe departs from the
# defaults: cosine decay, AdamW, batch 32 and 150 epochs.
TUNED_ON = "none"
if TUNED_ON not in (None, "kaggle_v2", "none"):
    raise SystemExit(f"\nTUNED_ON must be None, 'kaggle_v2' or 'none', "
                     f"not {TUNED_ON!r}.")

# -------------------------------------------------- author confirmations
# Facts only the author can supply, or decisions only the author can take.
# The working draft carries a note wherever one is used and every build
# prints the open list, so a drafted sentence cannot pass for a settled one.
# Flip an item to True once it is confirmed or done. The tuning item is the
# exception: it is closed by setting TUNED_ON, never by editing it here.
CONFIRM = {
    # Closed 8 October 2026: every passage that rested on the lift is
    # rewritten from the image_level tables, and check_withdrawn_wording()
    # keeps the withdrawn claims out. The project log 5z.
    "image_level": (True,
                    "The image-level lift has a chance level above zero. "
                    "Uninformative scores reach about +0.05 on kaggle_v2 "
                    "under this estimator. So the claims in the abstract and "
                    "in Sections 1, 4.4, 4.4.1, 4.6, 5.2, 5.3, 5.4 and 6 that "
                    "rest on the lift are not supported. Rewrite them from "
                    "image_level.csv."),
    "tuning": (TUNED_ON is not None,
               "Sections 3.5, 4.4 and 5.5 say hyperparameters were tuned once "
               "on kaggle_v2's validation split. grid/config.py says the "
               "recipes were fixed a priori with no tuning pass, and nothing "
               "in the repository can settle which is true. Set TUNED_ON to "
               "'kaggle_v2' or 'none'; the text of all three follows it."),
    # Closed 9 October 2026 by Vincent: public at REPO_URL, which the build
    # checks resolves.
    "repository": (REPO_LIVE,
                   "The repository does not exist. The address the manuscript "
                   "carried named a GitHub account that does not exist either. "
                   "Create it, set REPO_URL and REPO_LIVE, and rebuild; until "
                   "then the text says the code will be released on "
                   "acceptance."),
    # Closed 9 October 2026: the fork page now credits kaid Al2 and links
    # the upstream project, CC BY 4.0 and this study's repository. Vincent
    # confirmed that no label was changed in the fork.
    "twoclass_fork": (True,
                      "The twoclass copy used here is a public fork in your "
                      "Roboflow workspace whose page credits no upstream. CC "
                      "BY 4.0 requires crediting kaid Al2. Add the credit to "
                      "the fork page, because the data availability statement "
                      "sends readers there."),
    # Closed 9 October 2026 by Vincent: funding none, no competing interests,
    # sole author, no acknowledgement, keywords as drafted, ORCID given.
    "funding": (True, "Funding statement drafted as none received."),
    "competing": (True, "Competing interests drafted as none."),
    "contributions": (True, "Author contributions drafted for a single "
                            "author."),
    "acknowledgements": (True, "No acknowledgement, by the author's "
                               "decision."),
    "keywords": (True, "Six keywords drafted, within the four to six "
                       "allowed."),
    "orcid": (True, "The corresponding author's ORCID, on the title page and "
                    "for the submission system."),
    # Confirmed 11 October 2026 by Vincent, as proposed.
    "credit": (True, "The CRediT roles of the AEI statement."),
}

# ----------------------------------------------------------- number helpers
_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
          7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven",
          12: "twelve"}


def word(n: int) -> str:
    return _WORDS.get(n, f"{n:,}")


def protocol_geometry():
    """Box geometry split by annotation protocol. 5.2 argues from the
    difference and must not restate the figures Table 3.2 already carries."""
    wr = [v for v in BOX_GEOMETRY.values() if v[2] == "whole region"]
    loc = [v for v in BOX_GEOMETRY.values() if v[2] == "localisation"]
    if len(wr) != 1:
        raise SystemExit("\nCONSISTENCY CHECK FAILED: 5.2 assumes exactly one "
                         "whole-region source and BOX_GEOMETRY holds "
                         f"{len(wr)}.")
    return wr[0], {"n": len(loc),
                   "per_image": (min(v[0] for v in loc),
                                 max(v[0] for v in loc)),
                   "area": (min(v[1] for v in loc), max(v[1] for v in loc))}


def protocol_indomain_ap():
    """In-domain mean defect-class AP50, whole-region source against the span
    over the localisation sources. 5.2 quotes both halves and the point it
    makes is that they OVERLAP in difficulty, so the two must come from one
    table and one aggregation."""
    wr = [INDOMAIN_DEFECT_AP[s] for s, v in BOX_GEOMETRY.items()
          if v[2] == "whole region"]
    loc = [INDOMAIN_DEFECT_AP[s] for s, v in BOX_GEOMETRY.items()
           if v[2] == "localisation"]
    if len(loc) + len(wr) != len(BOX_GEOMETRY):
        raise SystemExit("\nCONSISTENCY CHECK FAILED: INDOMAIN_DEFECT_AP does "
                         "not cover every source in BOX_GEOMETRY.")
    lo, hi = min(loc), max(loc)
    if not hi - lo > wr[0] - hi:
        raise SystemExit(
            "\nCONSISTENCY CHECK FAILED: 5.2 states that the in-domain spread "
            "WITHIN the localisation protocol is wider than the gap between "
            "its strongest source and the whole-region source, which is why "
            "in-domain accuracy does not separate the protocols. The "
            f"localisation range is {lo:.3f} to {hi:.3f}, a spread of "
            f"{hi - lo:.3f}, and the gap to the whole-region value {wr[0]:.3f} "
            f"is {wr[0] - hi:.3f}. If the numbers have moved, the sentence in "
            "5.2 must move with them.")
    return wr[0], (lo, hi)


def headline() -> dict:
    """Every number the abstract, the Introduction's headline paragraph, 5.2,
    5.3 and the conclusion quote, derived from the tables in one place."""
    t_box = [v[1] for cfg in BOX_OFF.values() for v in cfg.values()]
    box = [BOX_OFF[("lohi_weld", "kaggle_v2")][c][1] for c in CFGS]
    ctl = list(CONTROL_T.values())
    band = armb_summary()
    il = image_level_facts()
    au = il["audit"]
    # The realised image counts behind the largest budget, over the twelve
    # combinations whose recovery the abstract quotes. 250 is requested.
    k_top = [int(r["n_images"]) for r in read_csv_rows(ARMB_CSV)
             if r["metric"] == "box" and r["endpoint"] == "last"
             and r["recovery"] != "" and int(r["k_requested"]) == max(ARMB_K)]
    return {
        "control": (pct(min(ctl), 0), pct(max(ctl), 0)),
        "control_cost": (pct(1 - max(ctl), 0), pct(1 - min(ctl), 0)),
        "box": (pct(min(t_box), 1), pct(max(t_box), 1)),
        "locate": (pct(min(box), 1), pct(max(box), 1)),
        "auroc_in": (q(il["in_range"][0], 2), q(il["in_range"][1], 2)),
        "audit_auroc": (q(au["kaggle_v2"]["A_C"], 2),
                        q(au["weld_surface_v2"]["A_C"], 2)),
        "cells": f"{GRID_CELLS + ARM_CELLS:,}",
        "band": (pct(ARMB_R250_BAND[0], 0), pct(ARMB_R250_BAND[1], 0)),
        "band_n": word(band["in_band"]), "armb_n": word(band["n"]),
        "k_real": (f"{min(k_top):,}", f"{max(k_top):,}"),
        # For the conclusion: the camera alone costs MORE THAN HALF, the
        # dataset MORE THAN NINE TENTHS. Both asserted in run_checks.
        "min_control_cost": 1 - max(ctl), "min_dataset_cost": 1 - max(t_box),
    }


def armb_summary() -> dict:
    r = [row[8] for row in ARMB_ROWS]
    lo, hi = ARMB_R250_BAND
    in_band = [x for x in r if Decimal(q(x, 2)) >= Decimal(str(lo))
               and Decimal(q(x, 2)) <= Decimal(str(hi))]
    s = sorted(r)
    med = (s[len(s) // 2 - 1] + s[len(s) // 2]) / 2 if len(s) % 2 == 0 \
        else s[len(s) // 2]
    return {"n": len(r), "min": min(r), "max": max(r), "median": med,
            "in_band": len(in_band),
            "band_edges": (min(in_band), max(in_band))}


def require_phrases(label: str, text: str, phrases) -> None:
    """Fail the build if a paragraph no longer carries a derived phrase.

    The number in the phrase comes from `headline()`. If someone edits the
    paragraph under deadline and types a number over the derived one, the
    derived phrase disappears and the build stops. That is the failure an
    abstract is most exposed to, because it is revised last and fastest.
    """
    missing = [ph for ph in phrases if ph not in text]
    if missing:
        raise SystemExit(
            f"\nCONSISTENCY CHECK FAILED: {label}\n"
            f"  no longer contains {missing}\n"
            f"  Quote the derived value from headline(). Do not type it.")


# -------------------------------------------------------------- bibliography
REFS_PATH = ROOT / "literature_review" / "references.json"
_PARTICLES = {"da", "de", "do", "dos", "das", "del", "della", "di", "du",
              "van", "von", "der", "den", "la", "le"}
REFS = {e["key"]: e for e in json.loads(REFS_PATH.read_text())["entries"]}
# The Zenodo deposit, once Vincent has made it. It is this study's own
# dataset entry, so it is added here rather than resolved from a DOI that does
# not exist yet. Set ZENODO_TITLE with the DOI, to the deposit's own title.
if ZENODO_DOI:
    REFS["deposit"] = {"key": "deposit", "year": ZENODO_YEAR, "csl": {
        "type": "dataset",
        "author": [{"family": "Dsouza", "given": "Vincent Linish"}],
        "title": ZENODO_TITLE or REPO_URL.rsplit("/", 1)[-1],
        "publisher": "Zenodo", "DOI": ZENODO_DOI}}
CITED: set = set()


def _fold(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c)).casefold()


def _clean(s):
    return html.unescape(s).replace(" ", " ") if isinstance(s, str) else s


def _surname(a: dict) -> str:
    return a.get("family") or a.get("literal") or ""


def _initials(given: str) -> str:
    """'Jia-Xin' gives 'J.-X.' and 'João Manuel R. S.' gives 'J. M. R. S.'."""
    out = []
    for w in given.replace(".", ". ").split():
        if w.lower() in _PARTICLES:
            continue
        out.append("-".join(p[0].upper() + "." for p in w.split("-") if p))
    return " ".join(out)


def _particle(given: str) -> str:
    return " ".join(w for w in given.split() if w.lower() in _PARTICLES)


def _first_initials_needed() -> set:
    """APA 7: first authors who share a surname but not initials are cited
    with their initials in the text, even when the years differ."""
    seen = {}
    for e in REFS.values():
        a = e["csl"]["author"][0]
        if "family" in a:
            seen.setdefault(_fold(a["family"]), set()).add(
                _initials(a.get("given", "")))
    return {s for s, ini in seen.items() if len(ini) > 1}


INITIALS_NEEDED = _first_initials_needed()


def _lead(key: str) -> str:
    a = REFS[key]["csl"]["author"][0]
    name = _surname(a)
    if "family" in a and _fold(a["family"]) in INITIALS_NEEDED:
        name = f"{_initials(a.get('given', ''))} {name}"
    return name


def _names(key: str, parenthetical: bool) -> str:
    au = REFS[key]["csl"]["author"]
    if len(au) == 1:
        return _lead(key)
    if len(au) == 2:
        return _lead(key) + (" & " if parenthetical else " and ") + \
            _surname(au[1])
    return _lead(key) + " et al."


def _sort_key(key: str):
    """APA 7 order: surname, then initials, then each later author in turn,
    then year. So S. Zhao precedes X. Zhao precedes Y. Zhao whatever their
    years, and Lin, Dollár (2017) precedes Lin, Maire (2014)."""
    e = REFS[key]
    authors = tuple((_fold(_surname(a)), _fold(_initials(a.get("given", ""))))
                    for a in e["csl"]["author"])
    return (authors, e["year"], _fold(_clean(e["csl"].get("title", ""))))


def _year_labels() -> dict:
    """a and b suffixes for identical in-text forms in the same year."""
    groups = {}
    for k in REFS:
        groups.setdefault((_names(k, False), REFS[k]["year"]), []).append(k)
    out = {}
    for (name, year), keys in groups.items():
        keys.sort(key=_sort_key)
        for i, k in enumerate(keys):
            out[k] = f"{year}{'abcdefgh'[i]}" if len(keys) > 1 else str(year)
    return out


YEAR = _year_labels()


def _use(key: str) -> None:
    if key not in REFS:
        raise SystemExit(
            f"\nCITATION CHECK FAILED: '{key}' is cited but is not in "
            f"{REFS_PATH.name}.\n  Resolve it by identifier in "
            f"resolve_references.py and re-run that script.")
    CITED.add(key)


def ct(key: str) -> str:
    """Narrative citation, for example Ma et al. (2024)."""
    _use(key)
    return f"{_names(key, False)} ({YEAR[key]})"


def cp(*keys: str) -> str:
    """Parenthetical citation, keys ordered as the reference list orders
    them, for example (Barz & Denzler, 2020; Laroca et al., 2023)."""
    for k in keys:
        _use(k)
    return "(" + "; ".join(f"{_names(k, True)}, {YEAR[k]}"
                           for k in sorted(keys, key=_sort_key)) + ")"


def _person(a: dict, initials_first: bool = False) -> str:
    if "literal" in a:
        return a["literal"]
    ini, part = _initials(a.get("given", "")), _particle(a.get("given", ""))
    if initials_first:
        return " ".join(x for x in (ini, part, a["family"]) if x)
    return f"{a['family']}, {ini}" + (f" {part}" if part else "")


def _author_list(au: list) -> str:
    names = [_person(a) for a in au]
    if len(names) > 20:     # APA 7: the first 19, an ellipsis, the last
        out = ", ".join(names[:19]) + ", . . . " + names[-1]
    elif len(names) == 1:
        out = names[0]
    else:
        out = ", ".join(names[:-1]) + ", & " + names[-1]
    # A group author such as "Faza" still closes the author element with a
    # full stop, as APA does for "World Health Organization. (2018)."
    return out if out.endswith(".") else out + "."


def _editor_list(eds: list) -> str:
    """APA 7: 'A. B. Name (Ed.)', 'A & B (Eds.)', 'A, B, & C (Eds.)'."""
    names = [_person(e, True) for e in eds]
    if len(names) == 1:
        return f"{names[0]} (Ed.)"
    if len(names) == 2:
        return f"{names[0]} & {names[1]} (Eds.)"
    return ", ".join(names[:-1]) + f", & {names[-1]} (Eds.)"


def _arxiv_id(doi: str):
    """'10.48550/ARXIV.2304.02296' gives '2304.02296', otherwise None."""
    low = doi.lower() if doi else ""
    return low.split("arxiv.", 1)[1] if low.startswith("10.48550/arxiv.") \
        else None


def _pages(p: str) -> str:
    return _clean(p).replace("-", "–")


def reference_runs(key: str) -> list:
    """One APA 7 reference as (text, italic) runs."""
    e = REFS[key]
    c = {k: _clean(v) for k, v in e["csl"].items()}
    typ = c.get("type", "")
    runs = [(f"{_author_list(e['csl']['author'])} ({YEAR[key]}). ", False)]
    title = c.get("title", "")
    doi = c.get("DOI")
    arxiv = _arxiv_id(doi)
    if arxiv:           # DataCite returns ARXIV in capitals; show the usual form
        doi = f"10.48550/arXiv.{arxiv}"
    link = f"https://doi.org/{doi}" if doi else c.get("URL", "")
    if typ in ("dataset", "software"):
        # Elsevier marks a data reference with a leading [dataset], which
        # reference_paragraph() writes, so the APA tag would say it twice.
        tag = "Data set" if typ == "dataset" else "Computer software"
        if typ == "dataset" and PROFILE["dataset_marker"]:
            tag = None
        runs.append((title, True))
        ver = f" (Version {c['version']})" if c.get("version") else ""
        runs.append((f"{ver}{f' [{tag}]' if tag else ''}. "
                     f"{c.get('publisher', '')}. ", False))
    elif typ == "book-chapter":
        eds = e["csl"].get("editor", [])
        ed_txt = f"{_editor_list(eds)}, " if eds else ""
        runs.append((f"{title}. In {ed_txt}", False))
        runs.append((c.get("book-title", ""), True))
        pg = f" (pp. {_pages(c['page'])})" if c.get("page") else ""
        runs.append((f"{pg}. {c.get('publisher', '')}. ", False))
    elif arxiv and typ != "paper-conference":
        runs.append((title, True))
        runs.append((f" (arXiv:{arxiv}). arXiv. ", False))
    elif typ in ("paper-conference", "proceedings-article"):
        runs.append((f"{title}. In ", False))
        runs.append((c.get("container-title", ""), True))
        tail = []
        if c.get("volume") and typ == "paper-conference":
            tail.append(f"Vol. {c['volume']}")
        if c.get("page"):
            tail.append(f"pp. {_pages(c['page'])}")
        runs.append(((f" ({', '.join(tail)})" if tail else "") + ". " +
                     (f"{c['publisher']}. " if c.get("publisher") and
                      c.get("publisher") != "arXiv" else ""), False))
    else:   # journal article, including online-first with no volume yet
        runs.append((f"{title}. ", False))
        runs.append((c.get("container-title", ""), True))
        if c.get("volume"):
            runs.append((", ", False))
            runs.append((c["volume"], True))
            if c.get("issue"):
                runs.append((f"({c['issue']})", False))
        # An online-first article has no volume yet, and its pages are the
        # provisional 1-16 of the online file, which APA omits.
        loc = (c.get("page") or c.get("article-number")) \
            if c.get("volume") else None
        if loc:
            one = "-" not in str(loc)
            runs.append((f", {'Article ' if one else ''}{_pages(str(loc))}",
                         False))
        runs.append((". ", False))
    runs.append((link, False))
    return runs


def check_references() -> None:
    """Every cited key has an entry and every entry is cited."""
    unused = sorted(set(REFS) - CITED)
    if unused:
        raise SystemExit(
            f"\nCITATION CHECK FAILED: in {REFS_PATH.name} but never cited: "
            f"{unused}\n  Cite them or remove them from resolve_references.py."
            f" The journal lists only works cited in the text.")
    forms = {}
    for k in REFS:
        forms.setdefault(f"{_names(k, True)}, {YEAR[k]}", []).append(k)
    clash = {f: ks for f, ks in forms.items() if len(ks) > 1}
    if clash:
        raise SystemExit(f"\nCITATION CHECK FAILED: ambiguous in-text forms "
                         f"{clash}")
    print(f"  citations: {len(CITED)} cited, {len(REFS)} listed, all matched")


# ------------------------------------------------------- consistency checks
def check_span(label, quoted, values, places=4):
    """Fail the build if a quoted prose range is not its table's true span.

    The 19 September audit found two ranges in the prose that contradicted
    the table directly above them, because prose and table were written
    independently. Deriving the range from the table and asserting equality
    removes the possibility rather than relying on proofreading.
    """
    lo, hi = min(values), max(values)
    ql, qh = quoted
    if round(ql, places) != round(lo, places) or round(qh, places) != round(hi, places):
        raise SystemExit(
            f"\nCONSISTENCY CHECK FAILED: {label}\n"
            f"  prose quotes [{ql}, {qh}]\n"
            f"  table spans  [{round(lo, places)}, {round(hi, places)}]\n"
            f"  Fix the prose or the table. Do not relax this check.")


def check_equal(label, a, b, places=3):
    if round(a, places) != round(b, places):
        raise SystemExit(
            f"\nCONSISTENCY CHECK FAILED: {label}\n"
            f"  {a} against {b}\n"
            f"  Fix the source of one of them. Do not relax this check.")


def run_checks():
    """Every range quoted in the prose, checked against its own table."""
    pij = [v[0] for cfg in BOX_OFF.values() for v in cfg.values()]
    t = [v[1] for cfg in BOX_OFF.values() for v in cfg.values()]
    check_span("4.3 cross-dataset box mAP50 span", (0.0004, 0.0514), pij)
    check_span("4.3 cross-dataset box transfer ratio span", (0.004, 0.079), t,
               places=3)

    # B2. Box-level retention into the one balanced column, which 4.4 and 5.3
    # quote beside the image-level reading. The image-level retention into
    # the same column, 0.133 to 0.211, was withdrawn on 8 October 2026: it is
    # the estimator's floor, the project log 5y. image_level_facts() asserts what
    # replaced it.
    box = [v[1] for v in BOX_OFF[("lohi_weld", "kaggle_v2")].values()]
    check_span("4.4 box-level retention into kaggle_v2", (0.005, 0.040), box,
               places=3)
    image_level_facts()

    # CROSS-TABLE CHECK. Where a source's defect space holds exactly one
    # class, its per-class AP and its class-agnostic box diagonal are the same
    # quantity computed twice, in two different released files. They must
    # agree. Two released tables disagreeing is a defect in the release even
    # when the manuscript happens to quote the correct one, and this is the
    # check that would have caught it. `check_equal` applied across files
    # rather than within one.
    for src, cls, ap, sd, boot, n, thin, state in PER_CLASS:
        if ap is None or src not in SINGLE_DEFECT_CLASS or state != "defect":
            continue
        check_equal(f"{src} per-class {cls} against its box diagonal, YOLO11s",
                    ap, BOX_DIAG_YOLO11S[src], places=3)

    # C1. The strict-threshold diagonal range.
    strict = [v[1] for d in BOX_DIAG_STRICT.values() for v in d.values()]
    check_span("4.3 mAP50-95 diagonal span", (0.0116, 0.2838), strict)

    # SECTION 1 QUOTES FOUR OF SECTION 4's HEADLINE SPANS, as percentages.
    # The Introduction and the Results must not be able to disagree about the
    # numbers a reader will remember, and an Introduction is written months
    # before submission and revised separately.
    check_span("1. cross-dataset box retention, percent", (0.4, 7.9),
               [x * 100 for x in t], places=1)
    check_span("1. box-level retention into kaggle_v2, percent", (0.5, 4.0),
               [x * 100 for x in box], places=1)

    # SECTION 3 AND SECTION 4 NOW STATE SOME OF THE SAME QUANTITIES FROM
    # DIFFERENT CODE PATHS, which is the exact shape of defect that the project log
    # 5o, 6a-xix and the 19 September span audit all had. A methods section
    # and a results section drifting apart on a cell count or a split size is
    # the most embarrassing kind of inconsistency, because both are the
    # author's own bookkeeping.
    check_equal("3.7 and 4.1 agree on the grid cell count",
                GRID_CELLS, MAIN_CELLS + REFERENCE_CELLS + DOWNLOADED_CELLS, 0)
    check_equal("3.7 and 4.1 agree on the arm cell count", ARM_CELLS, 300, 0)
    check_equal("3.5 and 4.1 agree on the main matrix count",
                MAIN_CELLS, 3 * len(CFGS) * N_SEEDS, 0)

    # 3.2's audited test-split sizes against the base rates Section 4.4
    # prints. n_pos + n_neg must be the split size.
    for src, (n_pos, n_neg) in BASE_RATE_COUNTS.items():
        check_equal(f"3.2 and 4.4 agree on the {src} test split size",
                    SPLITS[src][2], n_pos + n_neg, 0)

    # 3.2's box geometry against the pair Table 4.14's caption quotes.
    check_equal("3.2 and 4.14 agree on weld_surface_v2 median box area",
                BOX_GEOMETRY["weld_surface_v2"][1], 19.25, 2)
    check_equal("3.2 and 4.14 agree on lohi_weld median box area",
                BOX_GEOMETRY["lohi_weld"][1], 1.16, 2)
    check_equal("3.2 and 4.14 agree on weld_surface_v2 boxes per image",
                BOX_GEOMETRY["weld_surface_v2"][0], 0.88, 2)

    # 5.2's in-domain means against the box diagonals, on the two sources
    # whose harmonised defect space holds exactly one class. There the
    # per-class AP and the class-agnostic box mAP50 are the same measurement
    # read out of two different released tables, so they must agree. This is
    # the check that would have caught 5.2's earlier 0.94, which came from
    # neither table.
    for src in SINGLE_DEFECT_CLASS:
        check_equal(f"5.2 and 4.3 agree on the {src} in-domain diagonal",
                    INDOMAIN_DEFECT_AP[src], BOX_DIAG_MEAN[src], 4)

    # 3.2's dedup total against the sum of its own per-source rows.
    check_equal("3.2 clean tree total against the sum of its sources",
                CLEAN_TREE_TOTAL, sum(v[1] for v in SOURCES.values()), 0)
    # And each source's dedup count against the sum of its frozen splits.
    for src, sizes in SPLITS.items():
        check_equal(f"3.2 {src} dedup count against its frozen splits",
                    SOURCES[src][1], sum(sizes), 0)

    # Column C of the audit table and the YOLO11s image-level diagonal are
    # the same cells read from two released files.
    for src, (a, c) in AUDIT_A_C.items():
        check_equal(f"4.4.1 {src} C against cells.csv", c,
                    IMG_DIAG[src]["YOLO11s"], 9)

    check_draw_counts()
    check_control()
    control_image_ap()
    check_wording()
    check_config_classes()
    check_calibration()
    check_per_class_nms()

    # The conclusion says a change of camera costs MORE THAN HALF of
    # localisation performance and a change of dataset MORE THAN NINE TENTHS.
    # Words, not numbers, so they need their own guard.
    H = headline()
    if not H["min_control_cost"] > 0.5:
        raise SystemExit("\nCONSISTENCY CHECK FAILED: the conclusion says "
                         "the camera alone costs more than half.")
    if not H["min_dataset_cost"] > 0.9:
        raise SystemExit("\nCONSISTENCY CHECK FAILED: the conclusion says "
                         "a change of dataset costs more than nine tenths.")
    # The pooled control ratios, taken between the unrounded means, against
    # the ratio table lohi_control.md prints from the same cells. Dividing
    # Table 4.1's four-decimal means cannot reproduce the third decimal
    # (0.3031 / 0.6539 gives 0.4635 where the unrounded ratio is 0.463).
    md = CONTROL_MD.read_text()
    for (s, t), v in CONTROL_T.items():
        line = [ln for ln in md.splitlines()
                if ln.startswith(f"| {s} to {t} | ")][0]
        check_equal(f"4.2 control ratio {s} to {t} against lohi_control.md",
                    float(line.split("|")[2]), v, 3)
    # 3.8 quotes the cluster's software versions from these constants, and
    # torchvision and the workstation's stack from the requirements files.
    cluster = pins(H200_PINS)
    if (cluster["torch"], cluster["ultralytics"]) != (TORCH_VERSION,
                                                      ULTRALYTICS_VERSION):
        raise SystemExit(f"\nCONSISTENCY CHECK FAILED: 3.8 quotes PyTorch "
                         f"{TORCH_VERSION} and Ultralytics "
                         f"{ULTRALYTICS_VERSION}, and {H200_PINS.name} pins "
                         f"{cluster['torch']} and {cluster['ultralytics']}.")
    # 4.9 says the image-level test ranks configurations within each source
    # and target cell, so a floor shared by one target cannot reorder them.
    n_cells = len({(r["source"], r["target"]) for r in CELLS
                   if r["kind"] == "main" and r["test_split"] == "audited"})
    if STATS["image_ap"]["n_blocks"] != n_cells:
        raise SystemExit(f"\nCONSISTENCY CHECK FAILED: 4.9 says the "
                         f"image-level test blocks on source and target cell, "
                         f"{n_cells} of them, and it has "
                         f"{STATS['image_ap']['n_blocks']} blocks.")
    # The README and Section 3.8 state the same verification date.
    if SPLIT_VERIFY_DATE not in (ROOT / "README.md").read_text():
        raise SystemExit(f"\nCONSISTENCY CHECK FAILED: README.md does not "
                         f"state the split verification date "
                         f"{SPLIT_VERIFY_DATE} that Section 3.8 quotes.")
    if f"{CHECKPOINTS_GB} GB" not in (ROOT / "README.md").read_text():
        raise SystemExit(f"\nCONSISTENCY CHECK FAILED: the data availability "
                         f"statement says the checkpoints and raw predictions "
                         f"are {CHECKPOINTS_GB} GB, and README.md does not.")
    if REPO_LIVE:
        check_repo_live()

    print("  consistency checks: all ranges agree with their tables")


def check_wording():
    """Claims made in words, or by naming a cell, rather than by a number.

    A derived number cannot drift from its file, but the sentence around it
    can: "the largest in-domain margin", "the maximum, for Faster R-CNN", "fewer
    than one defect in twenty". Each is asserted here against the values the
    sentence describes."""
    def fail(msg):
        raise SystemExit(f"\nCONSISTENCY CHECK FAILED: {msg}")

    # 4.3 names the cells at both extremes of the box-level transfer ratio.
    t = {(s, tt, c): v[1] for (s, tt), d in BOX_OFF.items()
         for c, v in d.items()}
    if max(t, key=t.get) != ("lohi_weld", "twoclass", "Faster R-CNN"):
        fail(f"4.3 names Faster R-CNN from lohi_weld into twoclass as the "
             f"largest box transfer ratio, and it is {max(t, key=t.get)}.")
    if min(t, key=t.get) != ("lohi_weld", "visual", "YOLO11s"):
        fail(f"4.3 names YOLO11s from lohi_weld into visual as the smallest "
             f"box transfer ratio, and it is {min(t, key=t.get)}.")
    # 4.4, only while the tuned-once account is printed: kaggle_v2's
    # diagonal lift is the largest in-domain margin.
    if TUNED_ON != "none":
        lifts = {s: cell_mean("image_ap_above_base", s, s, kind=kind_of(s))
                 for s in PROTOCOL}
        if max(lifts, key=lifts.get) != "kaggle_v2":
            fail(f"4.4 calls kaggle_v2's diagonal lift the largest in-domain "
                 f"margin, and the lifts are {lifts}.")
    # 4.7 and 5.3: in-domain recall at one false positive per image is about
    # half, and cross-acquisition recall at the strictest budget finds fewer
    # than one defect in twenty.
    if not all(0.45 <= v <= 0.55 for v in RECALL_IN[-1]):
        fail(f"5.3 says about half, and in-domain recall at "
             f"{FP_BUDGETS[-1]} false positives is {RECALL_IN[-1]}.")
    if not max(RECALL_CROSS[0]) < 0.05:
        fail(f"4.7 says fewer than one defect in twenty, and cross-"
             f"acquisition recall at {FP_BUDGETS[0]} is {RECALL_CROSS[0]}.")
    # 4.8: every configuration's calibration error rises under the shift.
    if not all(b > a for a, b in control_ece().values()):
        fail("4.8 says every configuration moves the same way under the "
             "shift, and at least one does not.")
    # 4.9: neither pre-registered test is significant, the sensitivity
    # analysis agrees, and it triples the block count.
    for level in ("image_ap", "box_map50"):
        s = STATS[level]
        if s["primary"]["significant"] or s["secondary"]["anova"]["significant"]:
            fail(f"4.9 says neither test is significant at {level}.")
        sens = s["sensitivity_cell_by_seed"]
        if not sens["agrees_with_primary"] \
                or sens["n_blocks"] != N_SEEDS * s["n_blocks"]:
            fail(f"4.9 says the sensitivity analysis triples the block count "
                 f"and agrees, at {level}.")
    # 5.1: their external mAP@0.5 is more than double our YOLO11s diagonal.
    if not MENDIETA["ext_map50"] / 100 > 2 * BOX_DIAG_YOLO11S["kaggle_v2"]:
        fail("5.1 says their external mAP@0.5 is more than double ours.")


# WORDING WITHDRAWN ON 8 OCTOBER 2026, and the reason each went. The build
# fails if any returns. The project log 5y and 5z.
WITHDRAWN = {
    "can still screen": "the image-level screening claim, 5y",
    "survives transfer": "the image-level screening claim, 5y",
    "5 to 31 times": "a ratio between the two granularities, 5y",
    "a factor of 16": "a spread of oversampling gains, not of draws",
    "factor of sixteen": "a spread of oversampling gains, not of draws",
    "sixteen-fold": "a spread of oversampling gains, not of draws",
    "at most 250 labelled": "250 is requested; the draws hold 213 to 316",
    "documented defaults": "the frozen recipe departs from the defaults",
    "roughly five times wider": "undefined; the widths are given instead",
    "largest of the four localisation sets": "twoclass's boxes are larger",
    "holds the training budget fixed": "a pool trains on more images",
}


def check_withdrawn_wording(paragraphs: list) -> None:
    """Fail the build if withdrawn wording returns, in either build mode.

    Two phrases are allowed in one form only. "screening filter" may appear
    only in a negated sentence, since the advice now is not to treat a public
    checkpoint as one. "13 to 21 percent" may appear only in the one 4.4
    sentence that names it as an earlier summary's figure."""
    bad = []
    for text in paragraphs:
        for phrase, why in WITHDRAWN.items():
            if phrase in text:
                bad.append(f"{phrase!r} ({why}): {text[:90]}")
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            if "screening filter" in sentence and not re.search(
                    r"\b(not|neither|nor)\b", sentence, re.I):
                bad.append(f"'screening filter' outside a negated sentence: "
                           f"{sentence[:90]}")
            if "13 to 21 percent" in sentence and "earlier" not in sentence:
                bad.append(f"'13 to 21 percent' outside the sentence naming "
                           f"it as withdrawn: {sentence[:90]}")
    named = sum(text.count("13 to 21 percent") for text in paragraphs)
    if named > 1:
        bad.append(f"'13 to 21 percent' appears {named} times, and only the "
                   f"one sentence in 4.4 may name it")
    if bad:
        raise SystemExit("\nWITHDRAWN WORDING RETURNED:\n" +
                         "\n".join(f"  {b}" for b in bad))
    print("  wording: no withdrawn claim has returned")


def check_calibration():
    """Section 4.8, Table 4.13 and Figure 4.4 against the released files.

    Four things, each of which was wrong or unguarded before 24 September
    2026. The figure is drawn from the right file and that file exists. The
    figure's bins reproduce the table's ECE, so the picture and the numbers
    describe the same evaluations. The direction 4.8 now states, that every
    control evaluation and every fine-tuned model is underconfident and that a
    carried-over threshold clears fewer defective images, still holds. And the
    two ECE means that 4.8 and 5.3 quote agree with the CSV.
    """
    if not RELIABILITY_FIG.exists():
        raise SystemExit(f"\nCONSISTENCY CHECK FAILED: {F('reliability')} needs "
                         f"{RELIABILITY_FIG.name}. Run python3 "
                         f"grid/arms/reliability_control_figure.py.")
    script = RELIABILITY_SCRIPT.read_text()
    if f"MIN_LINE_COUNT = {RELIABILITY_MIN_LINE}\n" not in script:
        raise SystemExit(f"\nCONSISTENCY CHECK FAILED: {F('reliability')}'s caption "
                         f"says lines join bins of at least "
                         f"{RELIABILITY_MIN_LINE} images, and "
                         f"{RELIABILITY_SCRIPT.name} draws something else.")
    table, rd = control_ece(), reliability_direction()
    for cfg, (e_in, e_cross) in table.items():
        for regime, v in (("in domain", e_in), ("cross domain", e_cross)):
            check_equal(f"{F('reliability')} bins reproduce {T('ece')}, {cfg} "
                        f"{regime}", rd["ece"][(cfg, regime)], v, 4)
    ctl_n, ctl_under = rd["under"]["lohictl"]
    armb_n, armb_under = rd["under"]["armb"]
    check_equal("4.8 control evaluations, two per trained model",
                ctl_n, 2 * CONTROL_CELLS, 0)
    check_equal("4.8 fine-tuned models in the reliability bins",
                armb_n, ARMB_CELLS, 0)
    if ctl_under != ctl_n or armb_under != armb_n:
        raise SystemExit(
            f"\nCONSISTENCY CHECK FAILED: 4.8 says every control evaluation "
            f"and every fine-tuned model is underconfident. The bins give "
            f"{ctl_under} of {ctl_n} and {armb_under} of {armb_n}.")
    for cfg in CFGS:
        for e in THRESHOLD_EDGES:
            a = rd["clears"][(cfg, "in domain")][e]
            b = rd["clears"][(cfg, "cross domain")][e]
            if not b < a:
                raise SystemExit(
                    f"\nCONSISTENCY CHECK FAILED: 4.8 says a smaller share of "
                    f"the defective images clears every threshold from "
                    f"{THRESHOLD_EDGES[0]:.2f} to {THRESHOLD_EDGES[-1]:.2f} "
                    f"across the boundary. For {cfg} at {e:.2f} the shares "
                    f"are {a:.3f} in domain and {b:.3f} across it.")
    n_pos, n_neg = BASE_RATE_COUNTS["lohi_weld"]
    check_equal("4.8 defect share of the control's test images against the "
                "lohi_weld base rate", rd["defect_share"],
                n_pos / (n_pos + n_neg), 4)
    means = [sum(v[i] for v in table.values()) / len(table) for i in (0, 1)]
    check_equal("4.8 mean ECE in domain against the CSV", ECE_CONTROL[0],
                means[0], 3)
    check_equal("4.8 mean ECE cross acquisition against the CSV",
                ECE_CONTROL[1], means[1], 3)
    armb = [float(r["ece"]) for r in csv.DictReader(CALIB_CSV.open())
            if r["arm"] == "armb"]
    check_equal("4.8 and 5.3 arm B mean ECE against the CSV", ECE_ARMB,
                sum(armb) / len(armb), 3)


def check_draw_counts():
    """Each caption's draw count against the rows it describes.

    Table 4.3's caption said 2,000 draws while every one of its 32 box-level
    rows records 500, and the released markdown's own header said the same.
    A draw count is metadata rather than a quantity, so no span check could
    see it. This reads the `resamples` the rows record, per level and per
    file, and compares it with the constant the caption prints.
    """
    rows = list(csv.DictReader(RATIO_CSV.open()))
    # The image level is quoted in AUROC since 8 October 2026, and its draw
    # count is checked in image_level_facts(). Only the box level is quoted
    # from this file.
    for level, const, where in (("box", BOX_BOOTSTRAP_DRAWS, T("box_ratio")),):
        got = {int(r["resamples"]) for r in rows if r["level"] == level}
        if got != {const}:
            raise SystemExit(
                f"\nCONSISTENCY CHECK FAILED: {where} caption quotes {const:,} "
                f"draws\n  {RATIO_CSV.name} records {sorted(got)} at "
                f"{level} level. Fix the constant the caption reads.")
    got = {int(r["resamples"]) for r in csv.DictReader(CONTROL_CSV.open())}
    if got != {CONTROL_BOOTSTRAP_DRAWS}:
        raise SystemExit(f"\nCONSISTENCY CHECK FAILED: the control's intervals "
                         f"are quoted at {CONTROL_BOOTSTRAP_DRAWS:,} draws and "
                         f"{CONTROL_CSV.name} records {sorted(got)}.")
    key = json.loads(THIN_CACHE.read_text())["key"]
    if int(key.split(":")[0]) != THIN_BOOTSTRAP_DRAWS:
        raise SystemExit(f"\nCONSISTENCY CHECK FAILED: thin-class intervals "
                         f"are quoted at {THIN_BOOTSTRAP_DRAWS:,} draws and "
                         f"{THIN_CACHE.name} is keyed {key}.")


def check_per_class_nms():
    """Every in-domain row of per_class_ap.csv carries the NMS correction.

    The cross-table check in run_checks() sees a missing correction only on
    kaggle_v2 and visual, where a per-class value must equal its box
    diagonal. On the other three sources the only sign was `nms_corrected`
    False, and their RT-DETR-l values were charged for duplicate queries,
    the project log 5p. Downloaded-data rows are uncorrected by design. The project log
    5o keeps them out of the markdown table.
    """
    bad = [r for r in PER_CLASS_ROWS
           if r["kind"] != "dirty" and r["nms_corrected"] != "True"]
    if not PER_CLASS_ROWS or bad:
        cells = sorted({f"{r['kind']} {r['source']} {r['family']} "
                        f"seed {r['seed']}" for r in bad})
        raise SystemExit(
            f"\nCONSISTENCY CHECK FAILED: {len(bad)} of "
            f"{len(PER_CLASS_ROWS)} rows of {PER_CLASS_CSV.name} carry no NMS "
            f"correction\n  {', '.join(cells) or 'the file has no rows'}\n"
            f"  Re-run grid/analysis.py, which refuses to write such a row.")


def check_control():
    """4.2's quoted spans are the envelope of the per-cell intervals."""
    ci = control_intervals()
    check_span("4.2 control intervals, high to low",
               (0.236, 0.498), list(ci["envelope"][("lohi_high",
                                                    "lohi_low")]), 3)
    check_span("4.2 control intervals, low to high",
               (0.371, 0.659), list(ci["envelope"][("lohi_low",
                                                    "lohi_high")]), 3)
    if ci["matched"] != N_SEEDS * len(CFGS):
        raise SystemExit(f"\nCONSISTENCY CHECK FAILED: {ci['matched']} "
                         f"matched control cells, expected "
                         f"{N_SEEDS * len(CFGS)}.")


def check_config_classes():
    """DATASET_CLASSES was transcribed from grid/config.py. Import the frozen
    configuration read only and compare, suffixes stripped."""
    C = frozen_config()
    for tag, names in DATASET_CLASSES.items():
        shipped = [shipped_to_vocab(c) for c in C.DATASETS[tag]["classes"]]
        if sorted(shipped) != sorted(names):
            raise SystemExit(f"\nCONSISTENCY CHECK FAILED: {tag} classes "
                             f"{sorted(names)} against grid/config.py "
                             f"{sorted(shipped)}")
    for src, cls, ap, sd, boot, n, thin, state in PER_CLASS:
        d = C.DATASETS[src]
        raw = [c for c in d["classes"] if shipped_to_vocab(c) == cls][0]
        cfg_state = ("excluded" if raw in d["excluded"] else
                     "verdict" if raw in d["non_defect"] else "defect")
        if cfg_state != state:
            raise SystemExit(f"\nCONSISTENCY CHECK FAILED: {T('per_class')} marks "
                             f"{src} {cls} as {state}, grid/config.py makes "
                             f"it {cfg_state}.")


def frozen_config():
    """grid/config.py, imported read only. Importing does not change its
    hash, and bytecode is not written, so no pinned file moves."""
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT / "grid"))
    import config as C  # noqa: E402
    return C


def frozen_arms_config():
    """grid/arms/config_arms.py, imported read only after the frozen
    configuration. It imports nothing but the standard library and config,
    and registers no arm dataset unless register() is called, which this
    never does."""
    frozen_config()
    sys.path.insert(0, str(ROOT / "grid" / "arms"))
    import config_arms as A  # noqa: E402
    return A


def recipe_table() -> dict:
    """Table 3.3, the frozen recipe. Every value is read from the code that
    trained the cells: the two hyperparameter dictionaries and the common
    settings in grid/config.py, the label cost arm's departures in
    grid/arms/config_arms.py, and three values that live in code rather than
    in a dictionary. Faster R-CNN's augmentation is in grid/yolo_dataset.py,
    and its cosine schedule and fixed internal resize in grid/train_cell.py.
    Those are checked against the source text, so a change there fails the
    build."""
    C, A = frozen_config(), frozen_arms_config()
    u, t = C.ULTRALYTICS_HP, C.TORCHVISION_HP
    yd, tc = ROOT / "grid" / "yolo_dataset.py", ROOT / "grid" / "train_cell.py"
    flip = source_constant(yd, "if self.train and random.random() < 0.5:",
                           "a horizontal flip at 0.5 for Faster R-CNN")
    bright = source_constant(yd, "random.uniform(0.6, 1.4)",
                             "a brightness factor drawn from 0.6 to 1.4")
    source_constant(tc, "CosineAnnealingLR(opt, T_max=epochs)",
                    "a cosine decay to zero for Faster R-CNN")
    source_constant(tc, "min_size=C.IMGSZ, max_size=C.IMGSZ",
                    "Faster R-CNN's internal resize fixed at the input size")
    p_flip = float(flip.split("<")[1].rstrip(":").strip())
    lo, hi = (float(x) for x in bright.split("(")[1].rstrip(")").split(","))
    if t.get("lr_schedule") != "cosine" or not u.get("cos_lr"):
        raise SystemExit(f"\nCONSISTENCY CHECK FAILED: {T('recipe')} says both "
                         "code paths decay the learning rate on a cosine.")
    if not (C.PRETRAINED and A.ARMB_SEED_IS_DRAW):
        raise SystemExit(f"\nCONSISTENCY CHECK FAILED: {T('recipe')} says COCO "
                         "initialisation and a seed equal to the draw.")
    same = "as the first column"
    aug_u = (f"colour jitter of hue {u['hsv_h']}, saturation {u['hsv_s']} "
             f"and value {u['hsv_v']}, translation {u['translate']}, scale "
             f"{u['scale']}, horizontal flip at {u['fliplr']}, mosaic at "
             f"{u['mosaic']} switched off for the last {u['close_mosaic']} "
             f"epochs; no rotation, vertical flip or erasing")
    if (u["degrees"], u["flipud"], u["erasing"]) != (0.0, 0.0, 0.0):
        raise SystemExit(f"\nCONSISTENCY CHECK FAILED: {T('recipe')} says no "
                         "rotation, vertical flip or erasing.")
    rows = [
        ["Optimiser", u["optimizer"], t["optimizer"], same],
        ["Initial learning rate", f"{u['lr0']:g}", f"{t['lr']:g}",
         f"{A.ARMB_LR0:g}"],
        ["Final learning rate factor", f"{u['lrf']:g}", "0, cosine to zero",
         same],
        ["Momentum", f"{u['momentum']:g}", f"{t['momentum']:g}", same],
        ["Weight decay", f"{u['weight_decay']:g}", f"{t['weight_decay']:g}",
         same],
        ["Warm-up", f"{u['warmup_epochs']:g} epochs",
         f"{t['warmup_iters']} iterations",
         f"{A.ARMB_WARMUP_EPOCHS:g} epoch"],
        ["Schedule", "cosine decay", "cosine decay", same],
        ["Mixed precision", "on" if u["amp"] else "off",
         "on" if t["amp"] else "off", same],
        ["Epochs", str(C.EPOCHS), str(C.EPOCHS), str(A.ARMB_EPOCHS)],
        ["Batch", str(C.BATCH), str(C.BATCH), str(A.ARMB_BATCH)],
        ["Input size", f"{C.IMGSZ} pixels",
         f"{C.IMGSZ} pixels, internal resize fixed at {C.IMGSZ}", same],
        ["Initial weights", "COCO, pinned by sha256",
         "COCO, pinned by sha256",
         "the source's main-grid checkpoint at the same seed"],
        ["Seeds", ", ".join(str(s) for s in C.SEEDS),
         ", ".join(str(s) for s in C.SEEDS), "equal to the draw, 0, 1, 2"],
        ["Checkpoint kept", "best validation mAP at IoU 0.5 to 0.95",
         "best validation mAP at IoU 0.5 to 0.95",
         "final epoch, primary; best validation, secondary"],
        ["Augmentation", aug_u,
         f"horizontal flip at {p_flip:g}, brightness factor drawn uniformly "
         f"from {lo:g} to {hi:g}", same],
    ]
    if (C.EPOCHS, C.BATCH, C.IMGSZ) != (EPOCHS, BATCH, IMGSZ):
        raise SystemExit("\nCONSISTENCY CHECK FAILED: 3.5's epochs, batch "
                         "and input size against grid/config.py.")
    if (A.ARMB_EPOCHS, A.ARMB_BATCH) != (ARMB_EPOCHS, ARMB_BATCH):
        raise SystemExit("\nCONSISTENCY CHECK FAILED: 3.7's label cost "
                         "epochs and batch against grid/arms/config_arms.py.")
    return {"rows": rows, "sources": [
        "grid/config.py", "grid/arms/config_arms.py",
        "grid/yolo_dataset.py and grid/train_cell.py"]}


# ------------------------------------------------- tables, figures, layout
def horizontal_rules(t) -> None:
    """Horizontal rules only: one above the header row, one below it and one
    below the last row. No vertical rule and no shading, as both journals
    ask. Replaces the "Light Grid Accent 1" style, which drew a full grid."""
    tbl_pr = t._tbl.tblPr
    for old in tbl_pr.findall(qn("w:tblBorders")):
        tbl_pr.remove(old)
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "nil")
        borders.append(el)
    look = tbl_pr.find(qn("w:tblLook"))
    if look is not None:
        look.addprevious(borders)       # the schema puts borders before it
    else:
        tbl_pr.append(borders)

    def rule(cell, edges):
        tc_pr = cell._tc.get_or_add_tcPr()
        b = tc_pr.find(qn("w:tcBorders"))
        if b is None:
            b = OxmlElement("w:tcBorders")
            tc_pr.append(b)
        for edge in ("top", "bottom"):
            if edge in edges and b.find(qn(f"w:{edge}")) is None:
                el = OxmlElement(f"w:{edge}")
                for k, v in (("val", "single"), ("sz", "6"), ("space", "0"),
                             ("color", "000000")):
                    el.set(qn(f"w:{k}"), v)
                b.append(el)

    for c in t.rows[0].cells:
        rule(c, ("top", "bottom"))
    for c in t.rows[-1].cells:
        rule(c, ("bottom",))


# The panel order of the reliability figure, figstyle's CONFIG_ORDER.
RELIABILITY_PANELS = ("yolo11n", "yolo11s", "rtdetr_l", "faster_rcnn")
BOX_FIG_SOURCES = ("kaggle_v2", "lohi_weld")
BOX_FIG_TARGETS = ("kaggle_v2", "lohi_weld", "visual", "twoclass")
LABEL_COST_POINTS = [
    {"source": r["source"], "target": r["target"], "family": r["family"],
     "scenes": int(r["n_scenes"]), "recovery": float(r["recovery"]),
     "exhausted": r["exhausted_train_split"] == "True"}
    for r in read_csv_rows(ARMB_CSV)
    if r["endpoint"] == "last" and r["metric"] == "box"
    and r["recovery"] != ""]
LABEL_COST_PAIRS = sorted({(r["source"], r["target"])
                           for r in LABEL_COST_POINTS})
LABEL_COST_FAMILIES = [c for c in ("yolo11n", "yolo11s", "rtdetr_l",
                                   "faster_rcnn")
                       if any(r["family"] == c for r in LABEL_COST_POINTS)]


def draw_figure(label: str) -> dict:
    """Draw one figure at the active journal's printed width, from the same
    released values the tables use. See manuscript_figures.py."""
    import manuscript_figures as MF
    MF.style()
    width = PROFILE["fig_mm"][label]
    stem = f"Figure_{FIGURE_ORDER.index(label) + 1}"
    if label == "control_matrix":
        fig = MF.control_matrix(width, CONTROL_BOX, ACQ)
    elif label == "box_heatmap":
        cells = {(s, t): cell_mean("box_map50", s, t, "YOLO11s")
                 for s in BOX_FIG_SOURCES for t in BOX_FIG_TARGETS}
        fig = MF.box_heatmap(width, cells, BOX_FIG_SOURCES, BOX_FIG_TARGETS)
    elif label == "label_cost_curve":
        fig = MF.label_cost_curve(width, LABEL_COST_POINTS, LABEL_COST_PAIRS,
                                  LABEL_COST_FAMILIES)
    elif label == "reliability":
        fig = MF.reliability(width, MF.reliability_rows(),
                             RELIABILITY_PANELS)
    else:
        raise SystemExit(f"\nunknown figure {label!r}")
    return MF.save(fig, stem, SUBMISSION, SUBMISSION / "embed")


def check_figures(figures: dict) -> None:
    """Each embedded figure is placed at exactly the width it was drawn at,
    and none prints text below 7 pt. Placed width is read from the document,
    drawn width from the PNG itself."""
    import manuscript_figures as MF
    from PIL import Image
    if sorted(figures) != sorted(FIGURE_ORDER):
        raise SystemExit(f"\nFIGURE CHECK FAILED: placed {sorted(figures)}, "
                         f"expected {sorted(FIGURE_ORDER)}.")
    for label, rec in figures.items():
        with Image.open(rec["png"]) as im:
            px = im.size[0]
        drawn_mm = px / MF.DPI * 25.4
        placed_mm = Emu(rec["shape"].width).mm
        if abs(drawn_mm - placed_mm) > 25.4 / MF.DPI \
                or abs(drawn_mm - rec["width_mm"]) > 25.4 / MF.DPI:
            raise SystemExit(f"\nFIGURE CHECK FAILED: {F(label)} is drawn "
                             f"{drawn_mm:.2f} mm wide and placed at "
                             f"{placed_mm:.2f} mm.")
        if rec["min_font_pt"] < MF.MIN_PT:
            raise SystemExit(f"\nFIGURE CHECK FAILED: {F(label)} prints text "
                             f"at {rec['min_font_pt']:.1f} pt.")
    print("  figures: " + "; ".join(
        f"{F(k)} {figures[k]['width_mm']:.0f} x "
        f"{figures[k]['height_mm']:.0f} mm, smallest text "
        f"{figures[k]['min_font_pt']:.1f} pt" for k in FIGURE_ORDER))


def check_numbering(d) -> None:
    """Tables and figures are cited in the body in numerical order, each
    before it is placed, and placed in numerical order."""
    paras = d.paragraphs
    end = next((i for i, p in enumerate(paras) if p.text == "References"),
               len(paras))
    cites = {"Table": {}, "Figure": {}}
    placed = {"Table": [], "Figure": []}
    pats = {"Table": r"\bTable (\d+|[A-Z]\.\d+)\b",
            "Figure": r"\b(?:Figure|Fig\.) (\d+)\b"}
    for i, par in enumerate(paras[:end]):
        if par.text.startswith("NOTE, not manuscript text"):
            continue
        if par.style.name == "Caption":
            m = re.match(r"(Table|Fig\.) (\d+|[A-Z]\.\d+)", par.text)
            if m:
                placed["Table" if m.group(1) == "Table" else "Figure"].append(
                    (m.group(2), i))
                continue
        for kind, pat in pats.items():
            for m in re.finditer(pat, par.text):
                cites[kind].setdefault(m.group(1), i)
    want = {"Table": [str(n) for n in range(1, len(TABLE_ORDER) + 1)]
            + list(APPENDIX_TABLES.values()),
            "Figure": [str(n) for n in range(1, len(FIGURE_ORDER) + 1)]}
    bad = []
    for kind in ("Table", "Figure"):
        got = [n for n, _ in placed[kind]]
        if got != want[kind]:
            bad.append(f"{kind}s placed in the order {got}, expected "
                       f"{want[kind]}")
        where = dict(placed[kind])
        for n in want[kind]:
            if n not in cites[kind]:
                bad.append(f"{kind} {n} is never cited in the body text")
            elif n in where and cites[kind][n] > where[n]:
                bad.append(f"{kind} {n} is first cited after it is placed")
        stray = sorted(set(cites[kind]) - set(want[kind]))
        if stray:
            bad.append(f"{kind} numbers cited that do not exist: {stray}")
        main = [n for n in want[kind] if n.isdigit() and n in cites[kind]]
        order = sorted(main, key=lambda n: cites[kind][n])
        if order != main:
            bad.append(f"{kind}s first cited in the order {order}")
    if bad:
        raise SystemExit("\nNUMBERING CHECK FAILED:\n" + "\n".join(
            f"  {b}" for b in bad))
    print(f"  numbering: {len(want['Table'])} tables and "
          f"{len(want['Figure'])} figures, cited and placed in order")


def check_tables(d) -> None:
    """No vertical rule and no shading anywhere in a table."""
    bad = []
    for n, t in enumerate(d.tables, 1):
        for el in t._tbl.iter():
            if not isinstance(el.tag, str):
                continue
            tag = el.tag.split("}")[-1]
            parent = el.getparent()
            if tag in ("left", "right", "start", "end", "insideV") \
                    and parent is not None \
                    and parent.tag in (qn("w:tblBorders"), qn("w:tcBorders")) \
                    and el.get(qn("w:val")) not in ("nil", "none"):
                bad.append(f"table {n} has a vertical rule, {tag}")
            if tag == "shd" and (el.get(qn("w:fill")) or "auto").upper() \
                    not in ("AUTO", "FFFFFF"):
                bad.append(f"table {n} has cell shading")
        if t.style is not None and t.style.name != "Normal Table":
            bad.append(f"table {n} uses the style {t.style.name}")
    if bad:
        raise SystemExit("\nTABLE CHECK FAILED:\n" + "\n".join(
            f"  {b}" for b in sorted(set(bad))))
    print(f"  tables: {len(d.tables)}, horizontal rules only, no shading")


# American forms the body must not use. Reference titles keep their own
# spelling, and the CRediT role names are fixed terms, so both are skipped.
AMERICAN = re.compile(
    r"\b\w+(?:ization|izations|izing|ized|izes|ize|yzed|yzes|yzing|yze)\b|"
    r"\b(?:color|colors|colored|behavior|behaviors|center|centers|centered|"
    r"labeled|labeling|modeled|modeling|gray|favor|favorable|honor|"
    r"artifact|artifacts|catalog|analog|defense|fiber|toward|aluminum|"
    r"skeptical|acknowledgment)\b|\blicense plates?\b", re.I)
AMERICAN_OK = {"size", "sizes", "sized", "resize", "resized", "seize",
               "prize", "capsize"}


def body_texts(d) -> list:
    paras = [p.text for p in d.paragraphs]
    end = next((i for i, t in enumerate(paras) if t == "References"),
               len(paras))
    return [t for t in paras[:end] if t != CREDIT] + \
        [c.text for t in d.tables for row in t.rows for c in row.cells]


def check_british(d) -> None:
    hits = sorted({m.group(0) for t in body_texts(d)
                   for m in AMERICAN.finditer(t)
                   if m.group(0).lower() not in AMERICAN_OK})
    if hits:
        raise SystemExit(f"\nSPELLING CHECK FAILED: American forms in the "
                         f"body: {hits}")
    print("  spelling: British throughout the body")


def check_plain_text(d) -> None:
    """No underline anywhere, and URLs as plain text rather than hyperlinks,
    as Elsevier asks of the Word file."""
    body = d.element.body
    under = [u for u in body.iter(qn("w:u"))
             if u.get(qn("w:val")) not in (None, "none")]
    links = list(body.iter(qn("w:hyperlink")))
    for style in d.styles:
        font = getattr(style, "font", None)
        if font is not None and font.underline and style.type == 1:
            used = any(p.style.name == style.name for p in d.paragraphs)
            if used:
                under.append(style.name)
    if under or links:
        raise SystemExit(f"\nLAYOUT CHECK FAILED: {len(under)} underlined "
                         f"runs and {len(links)} hyperlinks.")
    print("  layout: single column, no underline, URLs as plain text")


def check_clean_document(d) -> None:
    """The submission build must carry no drafting residue.

    Earlier passes checked this by hand after each build, and the check only
    looked for double asterisks, so literal backticks in two Section 3
    paragraphs went through. The reference list is exempt from the dash rule:
    APA page ranges take an en dash and some published titles contain one.
    """
    paras = [p.text for p in d.paragraphs]
    refs = max((k for k, t in enumerate(paras) if t == "References"),
               default=len(paras))
    # The CRediT role names are fixed terms and keep their en dash.
    text = [t for t in paras[:refs] if t != CREDIT] + \
        [c.text for t in d.tables for row in t.rows for c in row.cells]
    rules = {"literal markdown backtick": "`", "literal markdown bold": "**",
             "em dash": "—", "en dash": "–",
             "dash used as punctuation": " - ",
             "author note": "NOTE, not manuscript", "missing figure":
             "FIGURE MISSING", "fill-in marker": "[TO "}
    bad = [(name, t[:120]) for name, mark in rules.items() for t in text
           if mark in t]
    bad += [("Draft. prefix", t[:120]) for t in paras if t.startswith("Draft.")]
    deep = [p.text[:60] for p in d.paragraphs
            if p.style.name.startswith("Heading ")
            and p.style.name.split()[-1].isdigit()
            and int(p.style.name.split()[-1]) > 3]
    bad += [("heading deeper than the journal's three levels", t) for t in deep]
    if bad:
        raise SystemExit("\nCLEAN BUILD CHECK FAILED:\n" + "\n".join(
            f"  {name}: {t}" for name, t in bad))
    print("  clean build: no drafting residue, no dashes, three heading "
          "levels at most")


def check_repo_live():
    """REPO_LIVE is True, so the address must actually resolve."""
    try:
        req = urllib.request.Request(REPO_URL, method="HEAD",
                                     headers={"User-Agent": "manuscript-build"})
        with urllib.request.urlopen(req, timeout=20) as r:
            ok = r.status == 200
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        ok, why = False, str(e)
    else:
        why = "status not 200"
    if not ok:
        raise SystemExit(f"\nCONSISTENCY CHECK FAILED: REPO_LIVE is True but "
                         f"{REPO_URL} does not resolve ({why}). The text "
                         f"would state a release that a reader cannot open.")


# ------------------------------------------------------------- submission
HIGHLIGHT_MAX = 85          # characters including spaces, Elsevier's limit


def set_properties(doc, title: str) -> None:
    cp_ = doc.core_properties
    cp_.title, cp_.author, cp_.last_modified_by = title, AUTHOR, AUTHOR
    cp_.comments = cp_.subject = cp_.keywords = cp_.category = ""


def highlights(H: dict, abstract: str) -> list:
    """The five highlights, every number from the values the abstract uses.
    A claim made in words must also stand in the abstract, where
    image_level_facts() asserts it."""
    b = armb_summary()
    claim = "no foreign detector's decision is distinguishable from chance"
    if claim not in abstract:
        raise SystemExit("\nHIGHLIGHTS CHECK FAILED: highlight 3's claim is "
                         "no longer in the abstract.")
    lines = [
        f"{word(len(SOURCES)).capitalize()} public visible-light weld "
        f"datasets were audited for leakage and repartitioned",
        f"A camera change keeps {H['control'][0]} to {H['control'][1]}% of "
        f"localisation; a dataset change keeps {H['box'][0]} to "
        f"{H['box'][1]}%",
        f"On the balanced target, {claim}",
        f"Auditing for leakage lowers in-domain AUROC by "
        f"{H['audit_auroc'][0]} on one public dataset",
        f"{H['k_real'][0]} to {H['k_real'][1]} labelled target images recover "
        f"{H['band'][0]} to {H['band'][1]}% of the gap in {b['in_band']} of "
        f"{b['n']} cases",
    ]
    long_ = [(len(x), x) for x in lines if len(x) > HIGHLIGHT_MAX]
    if long_ or not 3 <= len(lines) <= 5:
        raise SystemExit(f"\nHIGHLIGHTS CHECK FAILED: {len(lines)} lines, "
                         f"over {HIGHLIGHT_MAX} characters: {long_}")
    print("  highlights: " + ", ".join(str(len(x)) for x in lines)
          + f" characters, limit {HIGHLIGHT_MAX}")
    return lines


def write_highlights(lines: list) -> None:
    hd = docx.Document()
    set_properties(hd, "Highlights")
    hd.styles["Normal"].font.name = "Calibri"
    hd.add_heading("Highlights", 1)
    for line in lines:
        hd.add_paragraph(line, style="List Bullet")
    hd.save(SUBMISSION / "Highlights.docx")


def write_checklist(n_words: int, lines: list, figures: dict) -> None:
    """Every upload, and the steps only the author can do."""
    j = PROFILE["name"]
    figs = "\n".join(
        f"- `{figures[k]['stem']}.pdf` and `{figures[k]['stem']}.eps`: "
        f"{F(k)}, {figures[k]['width_mm']:.0f} by "
        f"{figures[k]['height_mm']:.0f} mm, smallest text "
        f"{figures[k]['min_font_pt']:.1f} pt" for k in FIGURE_ORDER)
    hl = "\n".join(f"{n}. {x} ({len(x)} characters)"
                    for n, x in enumerate(lines, 1))
    if JOURNAL == "aei":
        author_steps = f"""\
1. Run Elsevier's declarations tool for competing interests and upload the
   Word file it produces. The manuscript's own declaration reads: "{COMPETING_ELSEVIER}"
2. Enter the CRediT roles in the submission system as the manuscript states
   them: "{CREDIT}"
3. Archive the GitHub repository on Zenodo. Then set `ZENODO_DOI` and
   `ZENODO_TITLE` in `build_results_section.py` and rebuild. The data
   availability statement then gives the DOI and the reference list gains the
   deposit as a dataset entry. Until then the paper cites {REPO_URL}.
4. Choose whether to post the preprint on SSRN when the submission system
   offers it.
"""
        uploads = f"""\
- `Manuscript.docx`: the manuscript, single column, figures embedded at
  their printed width, tables with horizontal rules only.
- `Highlights.docx`: {len(lines)} highlights.
{figs}
- The declaration of competing interest from Elsevier's tool, step 1 below.
"""
    else:
        author_steps = "1. Complete the journal's own declarations.\n"
        uploads = f"- `Manuscript.docx`\n{figs}\n"
    text = f"""\
# {j}: submission checklist

Written by `build_results_section.py --clean{'' if JOURNAL == ROOT_JOURNAL
                                             else ' --journal ' + JOURNAL}`.
Every file below is regenerated by that command. Do not edit them by hand.

## Uploads

{uploads}
## Checks the build ran

- Abstract: {n_words} words, limit {ABSTRACT_WORDS[1]}, no citation.
- Keywords: {len(KEYWORDS)}, allowed {KEYWORD_COUNT[0]} to {KEYWORD_COUNT[1]}.
- Tables and figures numbered in order of first citation, each cited before
  it is placed.
- Tables: horizontal rules only, no vertical rules, no shading, notes below
  the table body.
- Figures drawn at their printed width, placed at exactly that width, text
  at 7 pt or larger, fonts embedded.
- No underline anywhere and URLs as plain text. British spelling in the body.
{('- Highlights, each at most ' + str(HIGHLIGHT_MAX) + ' characters:' + chr(10) + chr(10) + hl + chr(10)) if lines else ''}
## Steps only the author can do

{author_steps}"""
    (SUBMISSION / "CHECKLIST.md").write_text(text)


def build(clean: bool = False):
    run_checks()
    # The target journal's builds keep their names in the project root, and
    # the clean one is also copied into its submission folder. The fallback
    # journal's builds go to its submission folder only.
    if JOURNAL == ROOT_JOURNAL:
        out = ROOT / ("Weld_Transfer_Results_Section_clean.docx" if clean
                      else "Weld_Transfer_Results_Section.docx")
    else:
        out = SUBMISSION / ("Manuscript.docx" if clean
                            else "Manuscript_draft.docx")
    SUBMISSION.mkdir(parents=True, exist_ok=True)
    d = docx.Document()
    # The file's own title property matches the title page, and the
    # template's generator text is cleared.
    set_properties(d, TITLE)
    st = d.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(11)
    # Captions in plain black. The template's Caption style is bold and blue.
    cs = d.styles["Caption"]
    cs.font.size, cs.font.bold, cs.font.italic = Pt(9), False, False
    cs.font.color.rgb = RGBColor(0, 0, 0)

    def h(text, level=1):
        d.add_heading(text, level=level)

    def p(text, bold_lead=None):
        par = d.add_paragraph()
        if bold_lead and not clean:
            r = par.add_run(bold_lead)
            r.bold = True
        par.add_run(text)
        return par

    def note(text):
        if clean:
            return None
        par = d.add_paragraph()
        r = par.add_run("NOTE, not manuscript text. ")
        r.bold = True
        r.font.color.rgb = NOTE_COLOUR
        r.font.size = Pt(9)
        r2 = par.add_run(text)
        r2.italic = True
        r2.font.color.rgb = NOTE_COLOUR
        r2.font.size = Pt(9)
        return par

    def bullet(text):
        """A list item. Used only in 3.6, where the metric suite is a list."""
        par = d.add_paragraph(style="List Bullet")
        par.add_run(text)
        return par

    def labelled(label, text):
        """A bold label that survives --clean, unlike p()'s Draft. lead."""
        par = d.add_paragraph()
        r = par.add_run(label + " ")
        r.bold = True
        par.add_run(text)
        return par

    def confirm_note(key):
        """Where an unconfirmed author fact is used, say so in the draft."""
        done, what = CONFIRM[key]
        if not done:
            note(f"TO CONFIRM BY THE AUTHOR ({key}). {what}")

    def numbered_caption(kind, number, text):
        """Elsevier: 'Table 1. Title. Description.' Springer: 'Table 1 Title.
        Description', the label in bold, no punctuation after the number and
        none at the end."""
        par = d.add_paragraph(style="Caption")
        lead = f"{'Table' if kind == 'table' else 'Fig.'} {number}"
        text = text.strip()
        if PROFILE["captions"] == "elsevier":
            par.add_run(lead + ".").bold = True
            par.add_run(" " + text)
        else:
            par.add_run(lead).bold = True
            par.add_run(" " + text.rstrip("."))
        par.alignment = WD_ALIGN_PARAGRAPH.LEFT
        return par

    def table(headers, rows):
        t = d.add_table(rows=1, cols=len(headers))
        for i, htxt in enumerate(headers):
            cell = t.rows[0].cells[i]
            cell.text = ""
            run = cell.paragraphs[0].add_run(htxt)
            run.bold = True
            run.font.size = Pt(8)
        for row in rows:
            cells = t.add_row().cells
            for i, v in enumerate(row):
                cells[i].text = ""
                run = cells[i].paragraphs[0].add_run(str(v))
                run.font.size = Pt(8)
        horizontal_rules(t)
        return t

    def tcap(label, text):
        """A table's caption, placed ABOVE the table it follows in the code,
        with any closing 'Source: ...' moved to a note below the table body,
        as both journals ask."""
        src = re.search(r"\s*Source: (.+)$", text, re.S)
        if src:
            text = text[:src.start()]
        par = numbered_caption("table", table_no(label), text)
        tbl = par._p.getprevious()
        if tbl is None or tbl.tag != qn("w:tbl"):
            raise SystemExit(f"\nNUMBERING CHECK FAILED: the caption of "
                             f"{T(label)} does not follow a table.")
        tbl.addprevious(par._p)
        if src:
            n = d.add_paragraph()
            r = n.add_run("Source: " + src.group(1).strip())
            r.font.size = Pt(8)
        return par

    figures = {}

    def figure(label, cap):
        rec = draw_figure(label)
        d.add_picture(str(rec["png"]), width=Mm(rec["width_mm"]))
        d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        rec["shape"] = d.inline_shapes[-1]
        figures[label] = rec
        numbered_caption("figure", FIGURE_ORDER.index(label) + 1, cap)

    H = headline()

    # ------------------------------------------------------------ title page
    if not clean:
        t = d.add_paragraph()
        r = t.add_run("MANUSCRIPT, WORKING DRAFT")
        r.bold = True
        r.font.size = Pt(16)
        note("Every range quoted in the prose is derived from the table it "
             "sits beside and asserted equal to it at build time, and the "
             "transcribed tables are compared with the released files they "
             "came from. Two spans in an earlier draft contradicted the table "
             "three lines above them, which is what those checks now prevent.")
        open_items = [k for k, (done, _) in CONFIRM.items() if not done]
        if open_items:
            note(f"NOT YET SUBMITTABLE. {len(open_items)} items need the "
                 f"author: {', '.join(open_items)}. Each is marked where it "
                 f"is used, and the build prints the full list.")
    tp = d.add_paragraph()
    r = tp.add_run(TITLE)
    r.bold = True
    r.font.size = Pt(16)
    tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for line in (AUTHOR, AFFILIATION, f"Corresponding author: {EMAIL}",
                 f"ORCID: https://orcid.org/{ORCID}"):
        par = d.add_paragraph(line)
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # --------------------------------------------------------------- abstract
    d.add_page_break()
    h("Abstract", 1)
    release = ("are released" if REPO_LIVE else
               "will be released on acceptance")
    abstract = (
        f"Deep object detectors report high accuracy in visible-light weld "
        f"inspection, almost always measured on a held-out partition of the "
        f"dataset they were trained on. This study measures how far such a "
        f"detector travels. {word(len(SOURCES)).capitalize()} public "
        f"visible-light weld datasets were audited for duplicates and "
        f"cross-split leakage and repartitioned, and their labels were "
        f"harmonised so that whole-region quality verdicts are kept apart "
        f"from localised defects, two protocols whose median box areas differ "
        f"{PROTOCOL_AREA_FOLD}-fold. We trained {H['cells']} models, spanning "
        f"{word(len(CFGS))} detector configurations from {word(N_FAMILIES)} "
        f"families at {word(N_SEEDS)} seeds, and evaluated them across all "
        f"{word(len(SOURCES))} datasets at image level and at box level. "
        f"Changing only the "
        f"camera within one production line retains {H['control'][0]} to "
        f"{H['control'][1]} percent of in-domain localisation performance, "
        f"while changing the dataset retains {H['box'][0]} to {H['box'][1]} "
        f"percent. Whether a defect is present is learned in domain, with an "
        f"area under the receiver operating characteristic curve (AUROC) of "
        f"{H['auroc_in'][0]} to {H['auroc_in'][1]}. On the one balanced "
        f"target, no foreign detector's decision is distinguishable from "
        f"chance. Its usual summary, average precision above the positive "
        f"rate, exceeds zero even for uninformative scores on splits this "
        f"small. Auditing for leakage lowers in-domain AUROC by "
        f"{H['audit_auroc'][0]} on one public dataset and by "
        f"{H['audit_auroc'][1]} on a second. Fine-tuning on {H['k_real'][0]} "
        f"to {H['k_real'][1]} labelled target images recovers "
        f"{H['band'][0]} to {H['band'][1]} percent of the in-domain "
        f"localisation gap for {H['band_n']} of the {H['armb_n']} source, "
        f"target and detector combinations tested. The audited splits, label "
        f"mappings and code {release} as a leakage-audited baseline for "
        f"further work.")
    p(abstract)
    require_phrases("abstract", abstract, [
        f"{H['control'][0]} to {H['control'][1]} percent",
        f"{H['box'][0]} to {H['box'][1]} percent",
        f"(AUROC) of {H['auroc_in'][0]} to {H['auroc_in'][1]}",
        f"by {H['audit_auroc'][0]} on one", f"by {H['audit_auroc'][1]} on a",
        f"on {H['k_real'][0]} to {H['k_real'][1]} labelled",
        f"We trained {H['cells']} models",
        f"{H['band'][0]} to {H['band'][1]} percent",
        f"{H['band_n']} of the {H['armb_n']}",
        f"differ {PROTOCOL_AREA_FOLD}-fold"])
    n_words = len(abstract.split())
    if re.search(r"\(\D+, \d{4}|et al\.", abstract):
        raise SystemExit("\nCONSISTENCY CHECK FAILED: the abstract carries "
                         "a citation, which both journals forbid.")
    if not ABSTRACT_WORDS[0] <= n_words <= ABSTRACT_WORDS[1]:
        raise SystemExit(f"\nCONSISTENCY CHECK FAILED: the abstract is "
                         f"{n_words} words and the journal allows "
                         f"{ABSTRACT_WORDS[0]} to {ABSTRACT_WORDS[1]}.")
    note(f"Abstract: {n_words} words against the journal's limit of "
         f"{ABSTRACT_WORDS[0]} to {ABSTRACT_WORDS[1]}, unstructured, with one "
         f"abbreviation, AUROC, defined at first use, and no citation, as the "
         f"journal requires. Every number is derived by headline() and the "
         f"build fails if one is typed over. The image-level sentences are "
         f"asserted against the released tables by image_level_facts().")
    if not KEYWORD_COUNT[0] <= len(KEYWORDS) <= KEYWORD_COUNT[1]:
        raise SystemExit(f"\nCONSISTENCY CHECK FAILED: {len(KEYWORDS)} "
                         f"keywords, and {PROFILE['name']} allows "
                         f"{KEYWORD_COUNT[0]} to {KEYWORD_COUNT[1]}.")
    labelled("Keywords", " · ".join(KEYWORDS))
    confirm_note("keywords")

    d.add_page_break()
    h("1. Introduction", 1)

    p("Automated visual inspection of welds is a quality control problem with "
      "a direct economic cost attached to both kinds of error. A missed defect "
      "reaches a customer. A false alarm stops a line. Deep object detectors "
      "are now the standard tool for it, and the published results are strong. "
      "The question a practitioner actually faces is different from the "
      "question those results answer. It is whether a detector trained on one "
      "line, one camera and one annotation protocol will still work on "
      "another.",
      bold_lead="Draft. ")

    p(f"That question is rarely asked of weld detectors. A Scopus sweep of "
      f"work on YOLO-based weld defect detection found "
      f"{SCOPUS_YOLO_WELD_RECENT} of {SCOPUS_YOLO_WELD_TOTAL} papers published "
      f"between 2024 and 2026, so the subfield is active and recent. We found "
      f"no study in it that measures transfer between independently collected "
      f"public visible-light weld datasets. Cross-domain weld evaluation "
      f"itself is not new and we do not claim otherwise. It is published for "
      f"radiographic weld imagery {cp('lopez2025', 'parmar2026', 'shen2026')},"
      f" one visible-light dataset reports transfer between two cameras "
      f"inside its own production line {cp('block2024')}, and one paper "
      f"reports a single external test, which we come to next. What is missing "
      f"is a matrix across sources collected independently of one another.")

    p(f"One published visible-light external test exists and it is the "
      f"strongest evidence for the premise of this study. "
      f"{ct('mendieta2025')} "
      f"trained YOLOv7 on {MENDIETA['train_images']:,} shielded metal arc "
      f"welding images and reported {MENDIETA['in_domain_precision']} percent "
      f"precision and {MENDIETA['in_domain_map50']} percent mean average "
      f"precision at an intersection over union (IoU) threshold of 0.5 "
      f"(mAP@0.5) in domain. "
      f"On an external set drawn from a public Kaggle weld defect project they "
      f"reported {MENDIETA['ext_map50']} percent mAP@0.5. A fall from "
      f"{MENDIETA['in_domain_map50']} to {MENDIETA['ext_map50']} is the shape "
      f"of the problem. It is also an anecdote rather than a measurement: "
      f"their external test used {MENDIETA['ext_images_word']} images, kept after "
      f"filtering to similar plane shots and excluding angled views, other "
      f"welding processes and other joint types, then cropped to the bead. The "
      f"collapse happened under conditions chosen to favour transfer, and "
      f"{MENDIETA['ext_images_word']} curated images cannot measure its size. "
      f"That external set is the Kaggle project this study uses as a source, "
      f"which "
      f"we establish in Section 5.1, so the comparison can be made directly.")

    p(f"Elsewhere in industrial inspection this measurement is becoming "
      f"routine. A 2026 grand challenge at the IEEE International Conference "
      f"on Multimedia and Expo, on semiconductor wafer surfaces, is built on "
      f"it {cp('sun2026')}: its test set deliberately introduces domain "
      f"shift through variations in illumination, wafer texture and defect "
      f"scale that are absent from the training data, and it drew "
      f"{ICME['registered']} registered participants and "
      f"{ICME['submissions']} submissions, with {ICME['final_teams']} teams "
      f"reaching the final testing phase. Its "
      f"scoring weights the screening component at {ICME['screen_weight']}, "
      f"above localisation and above classification, because the industrial "
      f"priority is minimising false alarms. Separately, a 2026 benchmark "
      f"merges four public steel surface datasets into a single taxonomy and "
      f"evaluates zero-shot transfer out of it {cp('zhao2026')}. Neither "
      f"concerns welds. "
      f"Together they establish that cross-scenario evaluation under "
      f"deliberate shift is normal practice in industrial defect inspection, "
      f"which makes its absence in visible-light weld inspection a lag rather "
      f"than a niche interest.")

    p("Two obstacles explain part of that lag, and both are addressed here "
      "rather than assumed away. The first is that public weld datasets do not "
      "annotate the same thing. One of the five used here draws a single "
      "frame-sized box that records a quality verdict for the whole weld. "
      "Others draw small boxes around individual defects. A transfer matrix "
      "spanning both would measure the mismatch between those protocols rather "
      "than domain shift, so we treat annotation protocol as a named axis of "
      "shift, report an image-level decision that is invariant to how a box "
      "was drawn across all five sources, and restrict box-level scoring to "
      "the sources that share a localisation protocol. The second obstacle is "
      "that public datasets ship with duplicates, augmented copies and scenes "
      "split across their own partitions. A transfer number measured on top of "
      "that is not interpretable, so every source is audited before anything "
      "is trained.")

    p("This paper contributes a leakage-audited cross-dataset benchmark for "
      "visible-light weld defect detection. Specifically: a harmonised label "
      "taxonomy that separates whole-region verdicts from localised defects; "
      "an audit of five public sources for near-duplicate and cross-split "
      "leakage, with the duplicate counts and the resulting change in reported "
      "performance both released; a full train-source by test-target matrix "
      "across four detector configurations drawn from three families and three "
      "seeds, at both granularities; a label cost curve measuring how many "
      "target images must be annotated to recover in-domain performance; a "
      "measurement showing that the usual image-level summary, average "
      "precision above the positive rate, sits above zero for uninformative "
      "scores on these splits, together with the count of independent scenes "
      "behind each test split, which is why the image level is reported as "
      "the area under the receiver operating characteristic curve (AUROC); "
      "and the frozen splits, label mappings and code.")

    headline_par = (
        f"The headline results are the following. Changing only the camera, "
        f"inside one production line and with the annotation protocol and the "
        f"class list held constant, retains {H['control'][0]} to "
        f"{H['control'][1]} percent of in-domain localisation performance. "
        f"Changing the dataset retains {H['box'][0]} to {H['box'][1]} "
        f"percent. Visual shift alone is therefore sufficient to cause a large "
        f"collapse and is not sufficient to explain the larger one. The "
        f"image-level decision on whether a defect is present does not rescue "
        f"transfer. It is learned in domain on every target, but into the one "
        f"balanced target no foreign detector's decision is distinguishable "
        f"from chance, and on the other targets it ranges from inverted to "
        f"useful. Its usual summary, average precision above the positive "
        f"rate, sits above zero even for uninformative scores on test splits "
        f"this small, so it cannot carry a transfer claim. Auditing for "
        f"leakage lowers in-domain AUROC by {H['audit_auroc'][0]} on one "
        f"public source and by {H['audit_auroc'][1]} on a second, and does "
        f"not lower it on the industrial one.")
    p(headline_par)
    require_phrases("Section 1 headline paragraph", headline_par, [
        f"{H['control'][0]} to {H['control'][1]} percent",
        f"{H['box'][0]} to {H['box'][1]} percent",
        f"AUROC by {H['audit_auroc'][0]} on one",
        f"by {H['audit_auroc'][1]} on a second"])

    p("Section 2 reviews cross-dataset evaluation in defect inspection and the "
      "treatment of annotation inconsistency. Section 3 describes the datasets, "
      "the audit, the harmonised taxonomy and the evaluation protocol. Section "
      "4 reports the results in the order set out in Section 4.1. Section 5 "
      "discusses which axes of shift the design separates, what the results "
      "support for a practitioner, and the limitations of the study. Section "
      "6 concludes. Appendix A gives the complete class mapping and Appendix "
      "B the released file behind every reported number.")

    # --------------------------------------------------------------- SECTION 2
    d.add_page_break()
    h("2. Related work", 1)

    note("Every citation below is rendered from "
         "literature_review/references.json by ct() and cp(), and the "
         "reference list is generated from the same file, so an in-text year "
         "cannot disagree with its entry and a missing or unused entry fails "
         "the build. Every entry was resolved by DOI on 24 September 2026, "
         "including the eight rows the 20 September ledger marked `agent`. "
         "The ledger records what changed.")

    h("2.1 Deep weld defect detection, and where its numbers come from", 2)

    p(f"Deep object detectors are now the default tool for visible-light weld "
      f"inspection {cp('mohammed2025')}, and the YOLO family dominates: a "
      f"Scopus sweep of this literature returned {SCOPUS_YOLO_WELD_TOTAL} "
      f"papers, of which {SCOPUS_YOLO_WELD_RECENT} appeared between 2024 and "
      f"2026. Reported performance is high and the reported margins between "
      f"methods are small. {ct('antony2026')} compare four YOLO versions with "
      f"Faster R-CNN and a single-shot multibox detector on 1,134 annotated "
      f"weld images and report mAP@0.5 of 0.973 for YOLOv5s. "
      f"{ct('liang2025')} report a mean average precision over IoU thresholds "
      f"from 0.5 to 0.95 of 98.2 percent on an industrial welding dataset they "
      f"constructed, and {ct('sun2025')} likewise evaluate on a self-built "
      f"dataset. {ct('xu2024')} target small defects on pipeline weld "
      f"surfaces and report mAP@0.5 of 72.2 percent, and "
      f"{ct('mustajib2026')} evaluate a detection transformer on a new "
      f"shielded metal arc welding dataset by five-fold cross-validation.",
      bold_lead="Draft. ")

    p(f"Almost all of this is measured on a held-out partition of the dataset "
      f"the model was trained on. That is the right first test of a detector "
      f"and the wrong last one, because it cannot see a detector that has "
      f"learned the acquisition rather than the defect. A recent survey of the "
      f"field names the scarcity of large labelled datasets as a persistent "
      f"challenge and domain adaptation as a direction for future work "
      f"{cp('mohammed2025')}. This study measures the size of the problem "
      f"that direction would have to solve.")

    h("2.2 Cross-domain evaluation in weld inspection", 2)

    p(f"Cross-domain evaluation of weld inspectors is not new, and the "
      f"modality qualifier is what makes the gap precise. In radiography, "
      f"classifiers trained on RIAWELC {cp('totino2023')} have been tested on "
      f"GDXray {cp('mery2015')} to demonstrate generalisation "
      f"{cp('lopez2025', 'parmar2026')}, a semi-supervised detector has been "
      f"evaluated on GDXray under domain shift {cp('shen2026')}, one "
      f"classifier has been reported on three radiographic datasets, including "
      f"a private set of low image quality on which its accuracy was 75.8 "
      f"percent {cp('palma2024')}, and domain-adapted networks have been "
      f"proposed for weld classification {cp('parmarpal2026')}. In ultrasonic "
      f"inspection of pipeline welds, zero-shot learning has been used to "
      f"recognise defect classes absent from training {cp('yu2024')}. That is "
      f"zero-shot in the sense of unseen classes rather than unseen datasets, "
      f"and it is a different problem from the one measured here.",
      bold_lead="Draft. ")

    p(f"In visible light the picture is thinner. Cross-domain few-shot "
      f"classification of welding defects in camera images, under shifts in "
      f"material and measurement conditions, has been studied "
      f"{cp('xie2023')}, but as classification rather than detection. "
      f"LoHi-WELD ships two camera acquisitions of one industrial line and "
      f"reports transfer between them {cp('block2024')}, which is a "
      f"within-dataset acquisition control rather than a cross-dataset "
      f"measurement. {ct('mendieta2025')} report the one external test we are "
      f"aware of, and Section 1 gives its size and Section 5.1 compares "
      f"against it directly.")

    p(f"Dataset quality in weld inspection has been examined before, and "
      f"{ct('provencal2021')} published detection of exact and near duplicates "
      f"in phased-array ultrasound weld scans. That is a different modality "
      f"and a different failure mode from the near-duplicate and cross-split "
      f"leakage audited here, and we cite it as the weld-specific precedent "
      f"rather than as an equivalent.")

    h("2.3 Cross-dataset generalisation in industrial defect inspection", 2)

    p(f"Outside welding, cross-dataset evaluation of defect inspectors is "
      f"normal practice. {ct('ma2022b')} generalise rail surface defect "
      f"segmentation to unseen datasets across five target-domain settings, "
      f"motivated by variations in appearance and scale across rail types and "
      f"production conditions. {ct('silva2026')} benchmark ten architectures "
      f"across five public defect datasets under a unified pipeline and "
      f"conclude that domain-specific benchmarks are needed, though their task "
      f"is classification rather than detection. {ct('krassnig2026')} "
      f"introduce ISP-AD and characterise the field as one in which most "
      f"publicly available datasets are biased towards optimal imaging "
      f"conditions, leading to an overestimation of their applicability in "
      f"real-world industrial scenarios. {ct('zou2026')} pursue causal "
      f"invariant representations for the same problem. {ct('oza2024')} "
      f"survey the unsupervised domain adaptation methods for detectors that "
      f"this study deliberately does not use, because the question here is "
      f"what a practitioner gets from an unmodified detector rather than how "
      f"far an adaptation method can close the gap.",
      bold_lead="Draft. ")

    p(f"{ct('zhao2026')} is the nearest structural analogue to this study "
      "anywhere in surface defect inspection, and it deserves its own "
      "description rather than a citation bracket. SteelDefectX merges 7,778 "
      "images from four public steel sources into a single taxonomy of 25 "
      "defect categories, then measures zero-shot transfer out of the merged "
      "set onto aluminium and onto seamless steel tube, which the authors call "
      "cross-material transfer. That is the same manoeuvre this study performs: "
      "harmonise several public sources into one label space, then measure "
      "transfer across the boundary.")

    p("Two things follow, and neither is a criticism of that work. First, "
      "merging heterogeneous sources into one taxonomy is the default move in "
      "this area, and the annotation protocol question is not usually asked at "
      "all. It is reasonable not to ask it on steel surfaces, where annotation "
      "is far more homogeneous than in welding and nothing in the constituent "
      "sources is a whole-region acceptance verdict sitting beside a localised "
      "defect box. Second, the constituent sources are not audited for "
      "duplicates or cross-split leakage before transfer is measured. We are "
      "not aware of any multi-source defect benchmark, of any material, that "
      "does audit them.")

    h("2.4 Dataset quality, near-duplicates and leakage", 2)

    p(f"Near-duplicate leakage is a documented and quantified failure mode "
      f"across computer vision benchmarks, and the quantities are large enough "
      f"to matter to a benchmark paper. {ct('barz2020')} found that 3.3 and "
      f"10 percent of the CIFAR-10 and CIFAR-100 test images have duplicates "
      f"in the training set, and that removing them drops classification "
      f"accuracy by 9 to 14 percent. {ct('ramos2025')} characterise visual "
      f"leakage by modality, coverage and degree, and report that every "
      f"dataset they analysed leaks in some form. {ct('northcutt2021')} cover "
      f"label errors rather than duplicates and are worth naming for "
      f"completeness, since the two failure modes are often conflated.",
      bold_lead="Draft. ")

    p(f"Datasets cut from video are the ones most exposed, and several of the "
      f"weld sources used here are cut from video. {ct('babu2026')} found "
      f"near-identical images across the default train and test splits of two "
      f"of six automotive datasets, amounting to 25 and 5 percent of their "
      f"test images, and single out video-based collections with strong "
      f"spatio-temporal dependence as the ones that need similarity analysis "
      f"before a split is drawn. Section 3.4 handles the weld case with "
      f"frame-run keys for exactly that reason.")

    p(f"Three results make the case sharper than a general warning. "
      f"{ct('laroca2023')} found that on one of two licence plate recognition "
      f"datasets, the ranking of models changed under duplicate-free splits, "
      f"which is why a leakage audit belongs in the body of a benchmark paper "
      f"rather than in a data appendix. {ct('nogueira2026')} mixed closely "
      f"related original and augmented images across splits in oral cancer "
      f"histopathology and measured the inflation directly: accuracy rose by "
      f"up to 18 percent, with a mean of 14.3 percent across 11 models, and "
      f"they describe the gains as methodological artefacts rather than "
      f"generalisation. That is the same mechanism as an export pipeline that "
      f"augments before partitioning, which Section 3 documents in two of the "
      f"sources used here. {ct('adimoolam2023')} applied perceptual hashing to "
      f"building-footprint datasets and found 93 percent of the AICrowd "
      f"Mapping Challenge's validation split present in its own training "
      f"split, which is methodologically the closest published pipeline to "
      f"the audit run here.")

    p(f"The medical imaging literature reaches the same conclusion from its "
      f"own direction. {ct('tampu2022')} measure the inflation of test "
      f"accuracy due to leakage, and {ct('cassidy2022')} found duplicates "
      f"within and across the ISIC skin lesion datasets and proposed a removal "
      f"strategy. We found no visible-light weld dataset that has been audited "
      f"this way, and the export convention that augments a training split "
      f"before release makes the augmentation variant of the problem "
      f"systematic rather than accidental.")

    h("2.5 Annotation protocol and label space misalignment", 2)

    p("Where prior work combines detection datasets it generally treats "
      "annotation inconsistency as an error to be corrected. Section 5.2 gives "
      "that literature and the reason this study takes the opposite position, "
      "so we note here only the shape of the difference. Label transfer "
      "projects annotations into a common space so a detector can be trained "
      "across datasets. Re-annotation rebuilds the labels. Error detection "
      "finds and repairs them. Each is right when the disagreement is a "
      "mistake. A frame-sized quality verdict and a localised defect box are "
      "not a mistake, and this study measures the difference rather than "
      "repairing it away.",
      bold_lead="Draft. ")

    p(f"{ct('chakraborty2026')} is the closest methodological neighbour and "
      f"must be cited plainly. They group detection benchmarks into "
      f"setting-agnostic and setting-specific, evaluate across all train-test "
      f"pairs, and explicitly separate domain shift from label mismatch by "
      f"comparing a closed-label protocol against an open-label one. That is "
      f"structurally the design used here, carried out on general vision "
      f"datasets. The contribution of this paper is therefore not the design. "
      f"It is the design applied to visible-light weld inspection, under a "
      f"leakage audit, with a physical annotation-protocol taxonomy rather "
      f"than a learned label mapping {cp('zhou2022')}, and with an internal "
      f"acquisition control that pins the label concept.")

    p("Their asymmetry finding is worth stating precisely, because it bounds "
      "what our matrix can say about it. They report that transfer within a "
      "setting type is relatively stable, that transfer across setting types "
      "drops substantially and is often asymmetric, and that the most severe "
      "breakdowns occur when transferring from specific sources to agnostic "
      "targets. Every source and every target in this study is "
      "setting-specific, so the weld matrix sits entirely inside their "
      "specific-to-specific regime and cannot test their specific-to-agnostic "
      "claim. What it can report is whether asymmetry appears within that "
      "regime, and Section 5.2 sets out what it shows.")

    h("2.6 Isolating acquisition shift, and label-efficient adaptation", 2)

    p(f"Isolating a change of sensor while holding the label concept fixed is "
      f"rare and is recognised as rare. {ct('saenko2010')} established the "
      f"lineage with a study built on different imaging conditions across "
      f"cameras. {ct('jensen2025')} state the property directly for BugNIST, "
      f"whose dataset is characterised by having objects with the same "
      f"appearance in the source and target domains, which they note is "
      f"uncommon for other benchmark datasets for domain shift. "
      f"{ct('heckler2026')} provide the modern industrial version in MVTec AD "
      f"2, with test scenarios under lighting-condition changes to assess "
      f"robustness under real-world distribution shifts. The internal control "
      f"reported in Section 4.2 has that same property, and the literature "
      f"saying it is uncommon is the argument for reporting it rather than "
      f"treating it as a routine ablation.",
      bold_lead="Draft. ")

    p(f"On label efficiency the neighbouring literature reports that small "
      f"budgets help, and we state what that evidence is rather than implying "
      f"a consensus. {ct('wang2022')} released FS-ND, which they describe as "
      f"the first publicly available few-shot defect detection dataset. "
      f"{ct('wang2025')} report cross-dataset steel defect few-shot detection "
      f"at 1, 5 and 10 shots, motivated explicitly by variations in imaging "
      f"conditions, resolutions and annotation styles, and find that few-shot "
      f"adaptation substantially narrows the gap to full-data performance. "
      f"{ct('benhafid2026')} report that 45 to 65 percent of labels retains "
      f"more than 97 percent of full-supervision mAP@0.5 on industrial "
      f"detection, at budgets far larger than those explored here and on "
      f"food-processing rather than metal inspection imagery.")

    p(f"Two results speak to the small-budget region specifically, and both "
      f"are used again in Section 4.5. {ct('wang2019')} name the mechanism "
      f"behind an unstable low-budget region: with few target images a model "
      f"suffers from over-adaptation and instability that may lead to degraded "
      f"detection performance in the target domain. {ct('chu2016')} studied "
      f"how much of a pre-trained network to copy as a function of visual "
      f"distance from the source and the amount of target data, which is the "
      f"published basis for the head-reinitialisation decision described in "
      f"Section 3.")

    h("2.7 Evaluation under shift", 2)

    p(f"The metric set used here is assembled from an established evaluation "
      f"philosophy rather than chosen piecemeal. {ct('oksuz2023')} argue that "
      f"a detector in a safety-critical setting must be simultaneously robust "
      f"to domain shift, give reliable uncertainty and give calibrated "
      f"confidence, which is the combination reported in Sections 4.7 and "
      f"4.8. {ct('huseljic2025')} evaluate the calibration of pretrained "
      f"detectors under distributional change systematically, and "
      f"{ct('munir2022')} study calibration under domain shift for detection "
      f"specifically. The detection extension of expected calibration error "
      f"originates with {ct('kuppers2020')} and is cited wherever the "
      f"detection form is used.",
      bold_lead="Draft. ")

    p(f"In welding the closest precedent is in this journal. "
      f"{ct('bauer2026')} evaluate intrinsic confidence calibration methods on "
      f"two friction stir welding datasets that contain realistic data drift, "
      f"and find that well-calibrated models predict their own failures "
      f"better, which leads them to argue for looking beyond classification "
      f"accuracy when a model is selected. Their task is classification under "
      f"process drift rather than detection across datasets, and the argument "
      f"transfers: a detector whose confidence stays high while it fails is "
      f"the hazardous case in inspection.")

    p(f"One caution travels with that literature and we state it rather than "
      f"leave it for a reviewer. {ct('kuzucu2025')} argue that current "
      f"detection calibration evaluation frameworks, the metrics themselves "
      f"and the use of temperature scaling have notable drawbacks that lead to "
      f"incorrect conclusions, and that post-hoc calibrators outperform "
      f"train-time methods once evaluated properly. This study reports "
      f"calibration error under shift as a comparison within a configuration "
      f"and across the shift, never as an absolute standard and never between "
      f"detector families, which Section 4.8 states explicitly. We do not "
      f"claim that this answers their objection in full.")

    p(f"The statistical design has a precedent in this journal too. "
      f"{ct('lema2026')} show that reported gains between surface defect "
      f"detectors do not always survive a significance test, using analysis "
      f"of variance with Tukey's test over repeated training runs on one "
      f"dataset. We "
      f"report the same analysis alongside our primary test so that a reader "
      f"who knows that work can compare directly. Section 3.6 explains why the "
      f"primary test here is a Friedman test instead: the design is crossed "
      f"rather than independent, because the four scores in a cell come from "
      f"models graded on the same test images.")

    h("2.8 What this study adds", 2)

    p("Cross-dataset evaluation of defect inspectors is established in rail, "
      "steel and industrial anomaly detection, and a merged multi-source steel "
      "benchmark with cross-material zero-shot transfer has recently been "
      "released. Near-duplicate leakage is a documented and quantified failure "
      "mode across vision benchmarks. Annotation inconsistency across detection "
      "datasets is recognised and is usually repaired rather than measured. "
      "Cross-dataset detection matrices that separate domain shift from label "
      "mismatch have been reported for general vision.",
      bold_lead="Draft. ")

    p("We found no study that applies these together to visible-light weld "
      "inspection. Specifically: no leakage audit of the public visible-light "
      "weld datasets, no train-source by test-target matrix across them, no "
      "separation of whole-region verdict annotation from localised annotation "
      "before cross-dataset scoring, and no internal acquisition control with "
      "the label concept held constant. We also found no multi-source defect "
      "benchmark of any material that audits its constituent sources for "
      "near-duplicate or cross-split leakage before measuring transfer.")

    p("We state these as what our searches returned rather than as facts about "
      "the literature. Two independent search paths were used and neither is "
      "complete.")

    # --------------------------------------------------------------- SECTION 3
    d.add_page_break()
    h("3. Methodology", 1)

    note("THE ORGANISING PRINCIPLE OF THIS SECTION IS THE REVIEWER OBJECTION "
         "IT CLOSES, and that mapping is why it reads as an argument rather "
         "than a list. 3.2 closes 'your datasets are too small to support a "
         "domain claim'. 3.3 closes 'you are measuring label mismatch and "
         "calling it domain shift'. 3.4 closes 'your in-domain baselines are "
         "inflated by leakage'. 3.5 closes 'you tuned on every dataset "
         "separately'. 3.6 closes 'mAP hides the failure you care about'. 3.7 "
         "closes 'you measured the gap and did nothing about it'. Do not "
         "rearrange without checking which objection goes unanswered.")

    h("3.1 Problem formulation", 2)

    p("We consider K publicly available weld defect datasets, each treated as "
      "a distinct domain characterised by its own acquisition hardware, "
      "illumination, joint geometry and labelling convention. A detector "
      "trained on the training partition of one dataset is evaluated on the "
      "test partition of every dataset, producing a performance matrix in "
      "which the diagonal entries are the in-domain results conventionally "
      "reported in this literature and the off-diagonal entries are the "
      "cross-domain results this study introduces.",
      bold_lead="Draft. ")

    p(f"The matrix is not square at both granularities, and the reason is a "
      f"property of the public data rather than a design choice. At image "
      f"level it is {len(SOURCES)} by {len(SOURCES)}: every source can be "
      f"trained on or tested against, because the image-level decision is "
      f"invariant to how a box was drawn. At box level it is 2 by 4, because "
      f"only two of the five sources use a localisation annotation protocol "
      f"and are large enough to train on, and one further source annotates a "
      f"frame-sized quality verdict rather than a localised defect and can "
      f"therefore be neither a training row nor a target column there. "
      f"Section 3.3 defines that distinction and Section 3.2 gives the "
      f"measurements behind it.")

    p("We summarise the matrix with three quantities. The generalisation gap "
      "of a source is its in-domain score minus its mean score on every other "
      "domain, reported in the units of the metric so that the practical size "
      "of the drop is visible. The transfer ratio from one domain to another "
      "is the foreign detector's score divided by the target's own in-domain "
      "score at the same configuration and the same seed; it is unit-free and "
      "therefore comparable across datasets of different difficulty, which is "
      "why it is the quantity we put in the abstract. Recovery at k shots is "
      "the fraction of the remaining transfer deficit that fine-tuning on k "
      "labelled target images closes, with both terms of the denominator "
      "measured at the same configuration and seed as the numerator.")

    h("3.2 Datasets and their measured properties", 2)

    p(f"Five independently collected visible-light weld datasets form the "
      f"study. {T('sources')} gives their shipped size, their size after the audit "
      f"of Section 3.4, the number of distinct physical scenes they contain, "
      f"the number of leakage-safe split blocks those scenes reduce to, and "
      f"their licence. The audited tree holds {CLEAN_TREE_TOTAL:,} images.",
      bold_lead="Draft. ")

    p(f"We refer to the sources by short names. kaggle_v2 is version 2 of a "
      f"Kaggle weld defect object detection collection "
      f"{cp('wijaya2024')}. weld_surface_v2 is export version 2 of the "
      f"Roboflow Universe project Weld Surface Defect Detection "
      f"{cp('welddefect2025')}. visual is export version 1 of Visual Welding "
      f"Defect {cp('faza2024')}. twoclass is the project welding defect "
      f"2class {cp('kaidal2024')}, which publishes no downloadable version, "
      f"so our copy is a fork of it exported as version 1. lohi_weld is "
      f"LoHi-WELD, captured on an industrial metal active gas welding line "
      f"in two camera acquisitions {cp('block2024')}. The first four are "
      f"exports from annotation platforms; LoHi-WELD is distributed by its "
      f"authors and is the only source released with a peer-reviewed "
      f"description.")

    table(["Source", "Classes", "Shipped", "After audit", "Scenes",
           "Split blocks", "Licence"],
          [[s, str(len(DATASET_CLASSES[s])), f"{v[0]:,}", f"{v[1]:,}",
            f"{v[2]:,}", f"{v[3]:,}", v[4]] for s, v in SOURCES.items()])
    tcap("sources", f"The five visible-light sources. The four Roboflow "
            f"exports together ship "
            f"{sum(v[0] for k, v in SOURCES.items() if k != 'lohi_weld'):,} "
            f"files and hold "
            f"{sum(v[1] for k, v in SOURCES.items() if k != 'lohi_weld'):,} "
            f"unique images, so roughly four fifths of what they appear to "
            f"offer is augmentation and repeated frames. LoHi-WELD loses one "
            f"image of {SOURCES['lohi_weld'][0]:,} and is the only source that "
            f"retains native resolution.")

    p("Effective sample size for a domain claim is not the image count, and we "
      "report three quantities rather than one. The image count is what a "
      "practitioner downloads. The scene count is how many distinct physical "
      "welds were photographed. The split-block count is the number of "
      "independent units available once every image that might share a scene "
      "with another is constrained to one side of a partition. The block count "
      "is the conservative figure and it is the unit every confidence interval "
      "in this paper resamples.")

    p("Two sources are used as test-only targets and never as training "
      "sources. twoclass reduces to 35 physical scenes over 7 joint "
      "configurations, and 213 training images cannot support a domain claim. "
      "visual contributes 558 real images and two of its three classes are "
      "scene furniture rather than defects. Both remain valuable as targets, "
      "and Section 3.7 explains why they nevertheless need in-domain reference "
      "models.")

    p("The annotation protocols are not interchangeable, and this is the "
      f"measurement that shapes the rest of the design. {T('geometry')} gives the box "
      "geometry over each source's defect classes alone, after the audit.")

    table(["Source", "Boxes per image", "Median box area, percent of frame",
           "Protocol"],
          [[s, q(v[0], 2), q(v[1], 2), v[2]]
           for s, v in BOX_GEOMETRY.items()])
    tcap("geometry", f"Box geometry over the defect classes alone. "
            f"weld_surface_v2 carries roughly one retained box per image "
            f"covering a fifth of the frame, which is a whole-region quality "
            f"verdict rather than a localised defect. The other four localise. "
            f"Measured against the largest of their four medians, the "
            f"whole-region boxes are {PROTOCOL_AREA_FOLD} times the area, and "
            f"against the smallest they are "
            f"{PROTOCOL_AREA_FOLD_SMALLEST} "
            f"times. That is not a difference in annotation care.")

    p(f"Radiographic weld datasets exist, and cross-domain work in that "
      f"modality is published. RIAWELC {cp('totino2023')}, GDXray Welds "
      f"{cp('mery2015')} and SWRD {cp('zhao2025swrd')} are named here for "
      f"completeness and are deliberately outside the scope of this study. A "
      f"change of imaging physics is not the deployment shift addressed here, "
      f"and the modality qualifier in the title is load bearing. Section 5.5 "
      f"records the radiographic extension as future work, and no radiographic "
      f"number is mixed into a visible-light figure.")

    h("3.3 Label harmonisation and the two granularities", 2)

    p("The five sources use inconsistent class vocabularies for the same "
      "physical defects and, in two cases, quality verdicts rather than defect "
      "types. We therefore evaluate at two granularities, and the choice of "
      "which is primary is the most consequential decision in this "
      "methodology.", bold_lead="Draft. ")

    p("The primary granularity is an image-level decision: whether an "
      "annotated target is present in the image. It is invariant to how a box "
      "was drawn, so it is the one quantity comparable across all five "
      "sources, including the whole-region source. It is reported as the "
      "area under the receiver operating characteristic curve (AUROC), whose "
      "chance level is 0.5 at any positive rate. Average precision minus the "
      "positive rate is reported beside it, with its chance level, because on "
      "splits this small that summary is not zero for an uninformative "
      "scorer (Section 4.4). We are explicit that this is a "
      "decision about annotated-target presence and not a universal rejectable "
      "defect verdict.")

    p("The secondary granularity is a box-level score over the harmonised "
      "defect space. Every defect class collapses to a single label and "
      "matching is class-agnostic, because one source carries a single generic "
      "defect class while another carries four specific types and no "
      "correspondence between them exists. Inventing one would be the weakest "
      "claim in the paper. The collapse runs class-agnostic non-maximum "
      "suppression, because without it two boxes of different types on one "
      "physical defect would become a false positive pair that the model never "
      "produced. This granularity is restricted to the sources sharing a "
      "localisation protocol.")

    p("Within each source, a class takes one of three states, and the "
      "distinction is needed because the three behave differently in every "
      "table. A defect class is retained in the harmonised defect space and "
      "contributes to every figure. A verdict class is retained in the ground "
      "truth and scored per-class in domain, but is not a defect and "
      "contributes to no cross-dataset figure; Bad Weld and Good Weld in "
      "one source and Normal Weld in another are verdict classes. An "
      "excluded class is dropped from the ground truth itself, so it has no "
      "test instances and no score; visual's Welding Line and workpiece "
      "are scene furniture and are excluded, because retaining them would "
      "inflate performance on the one target where they exist and make that "
      "target look easier to transfer to than it is.")

    p("Predictions are filtered to the defect classes of whichever label space "
      "the model emits in. A model trained on its own source predicts in that "
      "source's class space. A model fine-tuned on a target with a different "
      "class count predicts in the target's space, because the detection head "
      "is rebuilt; Section 3.5 states that rule. The complete class mapping, "
      "with one row per merge decision, its source class, its target class and "
      "one sentence of justification, is given in Appendix A.")

    note("Appendix A is generated at build time from the harmonisation table "
         "in grid/config.py, imported read only, so it cannot drift from the "
         "code that produced every number. The project log 5r records why it stays "
         "in scope.")

    h("3.4 Leakage audit and partitioning", 2)

    p("Several of the public sources were assembled from video or shipped "
      "through an export pipeline that augments before partitioning, so "
      "visually near-identical frames can appear on both sides of a published "
      "split. We therefore audit every source before training and repartition "
      "it ourselves.", bold_lead="Draft. ")

    p(f"Similarity is measured by standardised grayscale normalised "
      f"cross-correlation, maximised over the {N_DIHEDRAL} dihedral transforms "
      f"so that a baked-in rotation or flip cannot hide a duplicate. Two "
      f"images are the same photograph when they score at least "
      f"{T_COPY:.2f} at {NCC_SIZES[0]} pixels. Two images belong to the same "
      f"physical scene when they score at least {T_SCENE:.2f} at BOTH "
      f"{NCC_SIZES[0]} and {NCC_SIZES[1]} pixels; the coarse pass generates "
      f"candidates and the fine pass rejects low-entropy collisions. Copy "
      f"selection uses a maximum independent set, so no retained pair remains "
      f"at the threshold.")

    p("Perceptual hashing was tried first and rejected, and we report that "
      "because it is a result about this domain rather than an implementation "
      "detail. Every weld image reduces to a dark ground with a bright bead, "
      "so unrelated photographs collide under a perceptual hash: at a Hamming "
      "distance of five, 65 of 79 inspected pairs were not the same "
      "photograph, and false positives begin at a distance of one. The two "
      "thresholds above are calibrated against anchors that need no judgement "
      "call. Pairs sharing a source identifier are the same photograph by "
      "construction and bottom out at 0.965; pairs drawn from two genuinely "
      "different datasets cannot be the same photograph and top out at 0.877. "
      "The interval between those anchors is empty.")

    p("Content similarity alone is insufficient, and three structural keys "
      "recover relationships that pixels cannot. Frame runs in video-derived "
      "sources are joined when retained frame indices differ by at most three. "
      "A specimen identifier in a filename identifies one plate photographed "
      "from several sides; two views of one plate can score as low as 0.425 "
      "against each other and would otherwise be free to straddle a "
      "partition. A production cycle identifier groups the two camera views "
      "that one industrial line writes per cycle; those score about 0.32 "
      "against each other, so no content threshold would ever link them.")

    p(f"Images are then grouped into split blocks, where a block is the "
      f"smallest unit that can be assigned to a partition without splitting a "
      f"scene or a frame run. Blocks are partitioned {SPLIT_FRACTIONS} into "
      f"training, validation and test sets by a mixed-integer program that "
      f"balances per-class instance counts across the three partitions, solved "
      f"with HiGHS {cp('huangfu2018')} under a {SPLIT_TIME_LIMIT_S} second "
      f"wall-clock budget at seed {SPLIT_SEED}. The resulting splits are "
      f"frozen and released with their sha256 hashes.")

    p("The released artefact is the frozen split files and their hashes, not "
      "the solver. Four of the five sources reached the time limit rather than "
      "proving optimality, so the returned partition depends on machine timing "
      "and a re-run returns a different but equally valid partition. Longer "
      "solving does not help: fifteen times the budget moved the objective by "
      "about four parts in a billion and the largest per-class deviation "
      "stayed at 0.213 percentage points, which is a property of indivisible "
      "blocks rather than of the search. Readers should verify the released "
      "splits against the published hashes rather than regenerate them, and "
      "numbers measured on two different partitions must not be compared.")

    p("To quantify what the audit changes we train three sets of models per "
      "source and label them A, B and C. A is trained and tested on the "
      "partitions as downloaded, which is what a practitioner following the "
      "shipped pipeline obtains. C is trained and tested on our audited "
      "partitions. B holds A's weights fixed and regrades them on the audited "
      "test split. All three use one configuration at three seeds, so that the "
      "comparison holds the detector constant.")

    p("B is not a held-out test and we never present it as one. The downloaded "
      "and audited partitions were drawn independently by different parties, "
      "so most of each audited test split lies inside the corresponding "
      "downloaded training split. B therefore scores a model largely on images "
      "it was trained on, and exists only to hold the weights fixed while the "
      "grading partition changes. The comparison we report is A against C, "
      "which changes the training data, the test data and the partitioning "
      "procedure at once, and is therefore an estimate shift under a change of "
      "protocol rather than a causal estimate of leakage inflation. Section "
      "4.4.1 reports it with the measured overlap printed beside it.")

    h("3.5 Detectors and training protocol", 2)

    p(f"To separate the effect of domain shift from the effect of architecture "
      f"we evaluate {word(len(CFGS))} model configurations drawn from "
      f"{word(N_FAMILIES)} detector families. A single-stage convolutional "
      f"detector, YOLO11 {cp('jocher2024')} at the n and s scales, represents "
      f"the class of models dominating the recent weld inspection literature. "
      f"A query-based transformer detector, RT-DETR-l {cp('zhao2024')}, "
      f"represents the attention-based alternative. A two-stage detector, "
      f"Faster R-CNN {cp('ren2017')} with a ResNet-50 {cp('he2016')} feature "
      f"pyramid network {cp('lin2017')} backbone, provides a classical "
      f"reference. All are initialised from weights pretrained on COCO "
      f"{cp('lin2014')}, pinned by sha256 so that a changed checkpoint cannot "
      f"enter the grid silently.",
      bold_lead="Draft. ")

    p(f"All models train under one protocol: {IMGSZ} pixel input, batch size "
      f"{BATCH}, {WORKERS} data loader workers, and exactly {EPOCHS} epochs "
      f"with no early stopping. The checkpoint retained is the one with the "
      f"best validation mean average precision averaged over IoU 0.5 to 0.95 "
      f"on the source's own validation split, for every family including the "
      f"two-stage detector, which selects on that metric rather than on "
      f"validation loss. Each configuration is trained with "
      f"{N_SEEDS} random seeds.")

    # How the recipes were chosen follows TUNED_ON, as in 4.4 and 5.5.
    if TUNED_ON == "none":
        p("Hyperparameters were not tuned. Each family's recipe was fixed "
          "before any model was trained, with no search on any split, and "
          f"then frozen for every dataset and every arm. {T('recipe')} lists it. "
          "We state this plainly because it is the choice a sceptical "
          "reviewer looks for first. It means no source's diagonal carries a "
          "tuning advantage. A tuned recipe might raise in-domain scores, and "
          "this study does not measure by how much.")
        recipe = recipe_table()
        table(["Setting", "YOLO11n, YOLO11s, RT-DETR-l", "Faster R-CNN",
               "Label cost arm, YOLO11s and RT-DETR-l"], recipe["rows"])
        tcap("recipe", "The frozen training recipe, one column per "
                "training code path. The label cost arm of Section 3.7 "
                "fine-tunes with the first column's recipe and departs from "
                "it only where its column says so. Exact parity of "
                "augmentation across families is not possible: Faster "
                "R-CNN's brightness factor is a rough stand-in for the "
                "colour jitter of the other three, and we state the "
                "difference rather than pretend it away. Read at build time "
                f"from {', '.join(recipe['sources'])}.")
    else:
        p("Hyperparameters were tuned once, on the validation partition of "
          "one source, and then frozen for every dataset and every arm. No "
          "per-dataset tuning is performed. We state this plainly because it "
          "is the choice a sceptical reviewer looks for first, and because it "
          "costs in-domain performance on four of the five sources in "
          "exchange for the entire argument. Section 4.4 reports which source "
          "that was, together with the fact that it is also the source with "
          "the largest in-domain margin, and explains why that makes the "
          "cross-domain collapse a conservative estimate rather than an "
          "inflated one.")
    confirm_note("tuning")

    p("One protocol fact governs every fine-tuning result and belongs here "
      "rather than with the arm that exposes it. Where a source and a target "
      "have different class counts, the detector's classification head is "
      "reinitialised and only the backbone and neck carry over from the source "
      "checkpoint. For this study that is the only possible design, because "
      "the premise is that the taxonomies differ. It means any fine-tuning "
      "result measures the target annotation budget needed to fit a new "
      "detection head on a source-pretrained backbone, which is what a "
      "practitioner adopting a public checkpoint faces. The head width and the "
      "distance the weights move from the starting checkpoint are both "
      "recorded per cell and asserted, so a run that silently discarded its "
      "starting weights cannot be reported as fine-tuning.")

    h("3.6 Evaluation protocol and metrics", 2)

    p(f"Every trained model is evaluated on the frozen test partition of every "
      f"source. All families share one postprocessing configuration: "
      f"non-maximum suppression at IoU {NMS_IOU:.2f}, a confidence floor of "
      f"{CONF_FLOOR}, and at most {MAX_DET} detections per image. Ground truth "
      f"and predictions are compared in the same letterboxed {IMGSZ} pixel "
      f"frame, so both live in one coordinate system.", bold_lead="Draft. ")

    p("We report the following.")

    bullet("Detection accuracy. Mean average precision at IoU 0.5 and "
           "averaged over 0.5 to 0.95. Both are computed and released for "
           "every cell. The manuscript reports the stricter threshold on the "
           "in-domain localisation diagonals, where it retains a third to a "
           "little over two fifths of the looser value and is informative, and "
           "reports the cross-dataset matrix at 0.5 alone, where the stricter "
           f"threshold falls to between {q(BOX_OFFDIAG_STRICT_SPAN[0], 4)} "
           f"and {q(BOX_OFFDIAG_STRICT_SPAN[1], 4)}. Those values sit at the "
           "resolution floor of the measurement and would imply a precision "
           "it does not have. Section 4.3 gives the diagonals at both "
           "thresholds.")

    bullet("Per-class average precision, in domain, in each source's own "
           "vocabulary, never reported without its test instance count beside "
           f"it. A class with fewer than {THIN_THRESHOLD} test instances is "
           "marked thin. Overall mean average precision is never reported "
           "without the per-class breakdown, because the defect classes are "
           "the minority and their failure is otherwise masked.")

    bullet("Operating-point performance. Recall at a fixed budget of "
           f"{', '.join(str(b) for b in FP_BUDGETS)} false positive boxes per "
           "image, obtained by sweeping the confidence threshold. This is the "
           "form in which an inspection requirement is written. It counts "
           "false positive boxes rather than false positive images, which is "
           "what keeps it measurable on targets whose negative image counts "
           "are too small to support a false alarm rate. It is computed for "
           "the internal acquisition control, the label cost curve and the "
           "pooling arm. It is not computed for the zero-shot cells of the "
           "main matrix, where a swept threshold on a detector that is "
           "already at the floor would report an operating point that does "
           "not exist.")

    bullet(f"Calibration. Expected calibration error {cp('naeini2015')} "
           f"computed on the image-level score, with reliability diagrams. We "
           f"state the level "
           "explicitly because a detection-level calibration error over "
           "matched boxes is a different quantity. It is read within a "
           "configuration and across the shift, never between detector "
           "families, whose confidence scores come from different heads and "
           "different normalisations and are not comparable.")

    il = image_level_facts()
    bullet(f"AUROC, for every target, with a split-block interval of "
           f"{IMAGE_LEVEL_DRAWS:,} draws. {word(len(il['under'])).capitalize()} "
           f"targets hold fewer than {MIN_NEGATIVES} defect-free test images, "
           f"the evaluator's threshold for a false alarm rate. Their AUROC is "
           f"reported and flagged, together with the number of split blocks "
           f"the defect-free images occupy (Section 4.4).")

    p("Two definitional choices need stating once, because both are places "
      "where a reasonable alternative gives a materially different answer.")

    p("Per-class average precision applies suppression WITHIN each class, "
      "which is the conventional definition. The alternative is to suppress "
      "across the whole defect space first, as the box-level matrix does. We "
      "reject that for the per-class table because it makes a per-class number "
      "non-local: a box of one class could suppress a box of another, so "
      "removing a class from the taxonomy would change a different class's "
      "reported score although nothing about that class changed. A column "
      "whose entries move when an unrelated class is added is not a per-class "
      "measurement. The same rule is correct for the box-level matrix for the "
      "opposite reason, since there the classes are deliberately collapsed "
      "into one label and cross-class suppression removes duplicates of one "
      "target.")

    p(f"Uncertainty on a ratio is a split-block bootstrap, not seed "
      f"dispersion. The two answer different questions: seed dispersion "
      f"measures training variance and says nothing about a small correlated "
      f"test sample. We resample split blocks rather than images, as the "
      f"clustered bootstrap requires {cp('field2007')}, because images from "
      f"one weld scene are correlated and adjacent video frames are nearly "
      f"the same photograph, so resampling images independently would treat "
      f"those as independent evidence and make every interval too narrow. "
      f"Numerator and denominator are resampled together, because both are "
      f"scored on the same test images of the target; resampling them "
      f"independently would ignore the shared test set. The diagonal is the "
      f"self-check: when source and target coincide the two models are the "
      f"same model, so the ratio must return exactly 1.000 with zero width, "
      f"and the analysis stops if it does not. The number of draws differs by "
      f"quantity and every released row records its own. Image-level AUROC "
      f"and its retained margin above chance use {IMAGE_LEVEL_DRAWS:,} draws, "
      f"and cross-dataset ratios at box level use {BOX_BOOTSTRAP_DRAWS}, a "
      f"reduction for cost that was checked rather than assumed. The "
      f"internal control's "
      f"ratios, which are box level, use {CONTROL_BOOTSTRAP_DRAWS:,}, and so "
      f"do the thin-class per-class intervals, at "
      f"{THIN_BOOTSTRAP_DRAWS:,}.")

    p(f"Every reported entry is the mean and standard deviation over three "
      f"seeds. The pre-registered comparison across the four configurations is "
      f"a Friedman test {cp('friedman1937')} blocked on evaluation cell with a "
      f"Nemenyi post hoc {cp('demsar2006')}, with a repeated-measures analysis "
      f"of variance and Tukey's honestly significant difference test reported "
      f"alongside for comparability with the precedent this design follows "
      f"{cp('lema2026')}. The design is a "
      "complete block: the four scores in a cell come from models trained on "
      "the same images and graded on the same test images, so they are not "
      "independent replicates, and average precision is bounded on the unit "
      "interval with cross-domain values piling up near zero, so normality is "
      "not assumed. Seeds are replicates within a cell rather than blocks, and "
      "a pre-registered sensitivity analysis repeats the test with cell and "
      "seed as the blocking unit. Where the primary and secondary tests "
      "disagree we claim the Friedman result and report the disagreement. This "
      "choice was fixed before any result existed.")

    h("3.7 Controls and mitigation arms", 2)

    p(f"Having measured the gap we report one control and two mitigation "
      f"arms, {ARM_CELLS} training cells in total. The control is presented "
      f"first, because it calibrates how much of the cross-dataset collapse "
      f"visual shift alone can account for. Each cell is trained under the "
      f"protocol of Section 3.5 and evaluated under that of Section 3.6.",
      bold_lead="Draft. ")

    p(f"The internal acquisition control, {CONTROL_CELLS} cells, exploits one "
      f"source that ships two camera acquisitions of a single industrial "
      f"welding line. The annotation protocol, the annotators, the class list "
      f"and the production line are identical across the two, and only the "
      f"acquisition changes, so training on each and testing on both isolates "
      f"visual domain shift with the label concept pinned. That is the one "
      f"thing the cross-dataset matrix cannot do, because every off-diagonal "
      f"cell there confounds illumination and sensor, object scale, and what "
      f"the annotators decided a box means. The control needs no new "
      f"partition: the two acquisitions separate by filename and zero split "
      f"blocks contain images from both, so filtering the frozen lists cuts "
      f"along boundaries that already exist and leaves the released hashes "
      f"valid. Both diagonals come free, since each model is evaluated on both "
      f"acquisitions.")

    p(f"The label cost arm, {ARMB_CELLS} cells, measures how many labelled "
      f"target images are needed to recover in-domain performance. It uses two "
      f"configurations rather than four, YOLO11s as primary and RT-DETR-l to "
      f"establish whether the curve's shape is an artefact of one "
      f"architecture, because the omnibus test on the main matrix found no "
      f"detectable difference among the four and spending four times the "
      f"compute to re-ask an answered question would be the wrong trade. "
      f"Budgets are k in {{{', '.join(str(k) for k in ARMB_K)}}} with "
      f"{ARMB_DRAWS} independent draws each, fine-tuning for {ARMB_EPOCHS} "
      f"epochs at batch {ARMB_BATCH}. Both localisation training sources "
      f"supply starting checkpoints, and every target that has an in-domain "
      f"reference is adapted to, excluding the case where the target is the "
      f"checkpoint's own source.")

    p(f"Draws sample whole split blocks rather than images. Sampling images "
      f"independently would let two frames of one weld count as two shots and "
      f"would overstate how little annotation a practitioner needs. A draw "
      f"accumulates whole blocks until it reaches the requested budget, so the "
      f"realised count overshoots, and by more than slightly: the worst "
      f"overshoot is {ARMB_WORST_OVERSHOOT} times, at the smallest budget, "
      f"where a single indivisible block can carry a draw from zero past the "
      f"target in one step. We therefore report the realised image count and "
      f"the realised scene count for every draw and plot recovery against the "
      f"realised scene count, because the requested budget is a sampling "
      f"instruction and not a measured quantity. Draws that exhaust a target's "
      f"training split are marked, since one target holds 213 training images "
      f"and cannot reach the largest budget at all. Adaptation blocks are "
      f"drawn from the target's training split only and scoring stays on the "
      f"frozen test split, which is a disjoint set of blocks by construction "
      f"and is asserted on every draw rather than assumed. No confidence "
      f"threshold, epoch count or learning rate is selected using any target "
      f"test split.")

    p("The endpoint reported as primary is the checkpoint at the final epoch "
      "of the fixed schedule, and the reasoning is a label-cost argument "
      "rather than a performance one. Fine-tuning needs a validation set, and "
      "the only available choice that is not the test split is the target's "
      "own frozen validation split, which holds up to 444 labelled images. "
      "Selecting on it would make the true label cost of a budget of ten far "
      "larger than ten, and would also change the selection criterion between "
      "zero shots and more than zero, since the zero-shot point is a model "
      "selected on its source's validation split. The best-validation "
      "checkpoint is reported as a secondary endpoint with the validation set "
      "size stated beside it.")

    p(f"The pooling arm, {ARMD_CELLS} cells, trains on the pool of all "
      f"training sources except one and tests on the held-out one, at all four "
      f"configurations and three seeds, because whether pooling helps at all "
      f"is a claim about detectors in general rather than about the shape of a "
      f"curve. With three training sources each fold trains on two. The two "
      f"test-only targets never enter a pool and are evaluated against all "
      f"three pools. Pooling crosses the annotation protocol boundary, so we "
      f"report image level for every fold and box level only for the one pool "
      f"that excludes the whole-region source; a box-level number from a "
      f"protocol-mixed pool would measure annotation mismatch rather than "
      f"domain shift. Each pooled fold is compared against the better single "
      f"member of its own pool at the same configuration and seed. That member "
      f"is chosen on the target's own test split, so it is an optimistic "
      f"comparator, and a pool trains on more images than either member.")

    p("Two further arms, photometric augmentation and illumination "
      "normalisation, are scoped and were not run. Neither carries a claim "
      "this paper depends on, and the omnibus result means the paper makes no "
      "fine distinctions between configurations in any case. We state that "
      "they were not run rather than omit them, because a scoped arm that "
      "silently disappears is indistinguishable from one that was run and "
      "reported selectively.")

    h("3.8 Implementation and reproducibility", 2)

    cluster, laptop = pins(H200_PINS), pins(LAPTOP_PINS)
    sc = SCHEDULE_COUNTS
    gpu_laptop = json.loads(GPU_JSON.read_text())["device"]["name"]
    p(f"The {GRID_CELLS + ARM_CELLS} cells of the study ran on {GPU}, using "
      f"PyTorch {TORCH_VERSION} {cp('paszke2019')} on CUDA {CUDA_VERSION} "
      f"with Ultralytics {ULTRALYTICS_VERSION} for the single-stage and "
      f"transformer families and torchvision {cluster['torchvision']} for "
      f"the two-stage family. The {GRID_CELLS} cell main grid cost about "
      f"{GRID_HOURS} hours of wall clock and the {ARM_CELLS} arm cells about "
      f"{ARM_HOURS} hours. The {sc['n']} exploratory schedule-control runs of "
      f"Section 4.5.2 were trained on a workstation with an {gpu_laptop}, "
      f"under PyTorch {laptop['torch']}, torchvision {laptop['torchvision']} "
      f"and Ultralytics {laptop['ultralytics']}, and are compared only with "
      f"each other.", bold_lead="Draft. ")

    where = (f"is released at {REPO_URL}" if REPO_LIVE else
             "will be released in a public repository on acceptance")
    p(f"The complete pipeline {where}: the duplicate and leakage audit, the "
      f"harmonisation mapping, the frozen split files with their sha256 "
      f"hashes, the training and evaluation code, and the analysis that "
      f"produces every figure and table in this paper. We do not redistribute "
      f"images from any source, and for the five sources used here that is "
      f"our own choice rather than a licence constraint: one of them is "
      f"dedicated to the public domain by an uploader whose description states "
      f"that the images were collected from unnamed prior collections, so we "
      f"treat its provenance as unverified and prefer to point readers at the "
      f"authoritative copy. Every released split file identifies its images by "
      f"their original filenames, and all {SPLIT_ENTRIES_VERIFIED:,} entries "
      f"across the fifteen files were verified on {SPLIT_VERIFY_DATE} to "
      f"resolve against untouched original downloads, so the partitions can "
      f"be reconstructed exactly and checked against the published hashes.")
    confirm_note("repository")

    p("Every cell records the sha256 of the configuration, the evaluation "
      "code, the metric implementation and the dataset loader at the moment it "
      "was trained, together with its checkpoint identity, software versions, "
      "host and GPU, and the evaluator refuses to score a checkpoint whose "
      "recorded hashes no longer match. Those hashes are released per cell. We "
      "do not claim that every cell can be re-evaluated from the released head "
      "of the code: one evaluator revision between cell generations was not "
      "retained, so the sixty oldest cells pin a version that no longer "
      "exists. Their results are unaffected, because the corrected branch "
      "executes only for a cell kind none of them uses, and the three files "
      "whose movement would change a reported number have never moved across "
      "any generation. Section 4.10 states this again where the numbers are.")


    d.add_page_break()
    h("4. Results", 1)

    # ------------------------------------------------------------------- 4.1
    h("4.1 What was run, and how to read it", 2)

    p("We report 69 training cells for the main study and 300 for the "
      "controls and mitigation arms. The main matrix is 36 cells: three "
      "source datasets, four model configurations drawn from three detector "
      "families, and three random seeds. The four configurations are YOLO11 "
      "at the n and s scales, RT-DETR with an l backbone, and Faster R-CNN "
      "with a ResNet-50 FPN backbone. Every trained model is evaluated on the "
      "frozen test partition of all five datasets, so one set of models "
      "serves both the box-level and the image-level matrix and no model is "
      "retrained in order to be measured again. Label harmonisation happens "
      "at evaluation time.", bold_lead="Draft. ")

    p("The remaining 33 cells of the 69 are load bearing and we name them "
      "rather than leave them implicit. Twenty-four are reference cells that "
      "train all four configurations at three seeds in domain on the two "
      "test-only targets, visual and twoclass. They exist because every "
      "transfer ratio and every recovery ratio divides by the target's own "
      "in-domain score at the same configuration and the same seed, and "
      "without them those two targets would have no denominator. The other "
      "nine train on the partitions as downloaded rather than as audited, and "
      "they supply the comparison in Section 4.4.1. They were run at YOLO11s "
      "only, at three seeds, which is why that comparison is reported at "
      "YOLO11s throughout.")

    sc = SCHEDULE_COUNTS
    p(f"Section 4.5.2 adds {sc['n']} exploratory runs: "
      f"{sc['by']['fixed_50']} at the pre-registered schedule, "
      f"{sc['by']['oversampled_50']} oversampled and "
      f"{sc['by']['matched_step']} with matched steps, all {sc['family']}. "
      f"They were trained on a workstation rather than the cluster, they are "
      f"not part of the {GRID_CELLS + ARM_CELLS:,}, and they are compared "
      f"only with each other.")

    p("The order of what follows is deliberate. The internal acquisition "
      "control is reported first, before any cross-dataset number, because it "
      "is the only measurement here that isolates visual domain shift with "
      "the label concept held constant. It therefore calibrates how much of "
      "the cross-dataset collapse visual shift alone can account for, and "
      "every figure after it should be read against it. We then report the "
      "box-level cross-dataset matrix, the image-level matrix in AUROC, the "
      "effect of the leakage audit, the label cost curve, the "
      "pooling arm, recall at a fixed false positive budget, and calibration "
      "under shift.")

    # ------------------------------------------------------------------- 4.2
    h("4.2 The internal acquisition control", 2)

    p("The internal control measures what a change of camera alone costs, "
      "with the annotation protocol, the annotators, the class list and the "
      "production line all held constant.", bold_lead="Draft. ")

    ci = control_intervals()
    hl, lh = ("lohi_high", "lohi_low"), ("lohi_low", "lohi_high")
    p(f"LoHi-WELD ships two acquisitions from one industrial metal active gas "
      f"welding line. Training on each and testing on both gives a two by two "
      f"matrix at no extra training cost. Defect-class mAP50, average "
      f"precision on the harmonised defect label space, falls from "
      f"{q(CONTROL_BOX[(hl[0], hl[0])][0], 3)} in domain to "
      f"{q(CONTROL_BOX[hl][0], 3)} across the acquisition boundary, and from "
      f"{q(CONTROL_BOX[(lh[0], lh[0])][0], 3)} to {q(CONTROL_BOX[lh][0], 3)} "
      f"in the opposite direction. Measured against the target's own "
      f"in-domain value, the transfer ratios are {q(CONTROL_T[hl], 3)} and "
      f"{q(CONTROL_T[lh], 3)}. Their split-block intervals are computed per "
      f"configuration and seed, {CONTROL_BOOTSTRAP_DRAWS:,} draws each, and "
      f"across the {word(ci['n_cells'][hl])} cells in each direction they "
      f"span {q(ci['envelope'][hl][0], 3)} to {q(ci['envelope'][hl][1], 3)} "
      f"and {q(ci['envelope'][lh][0], 3)} to {q(ci['envelope'][lh][1], 3)}. "
      f"So changing the camera alone costs between {H['control_cost'][0]} and "
      f"{H['control_cost'][1]} percent of localisation performance. The "
      f"effect is consistent across configurations: in the high to low "
      f"direction the {word(len(CFGS))} configurations span "
      f"{q(min(CONTROL_HIGH_TO_LOW.values()), 3)} to "
      f"{q(max(CONTROL_HIGH_TO_LOW.values()), 3)} box mAP50. "
      f"{T('control')} gives the four entries and {F('control_matrix')} "
      f"draws them.")

    table(
        ["Trained on", "Tested on lohi_high", "Tested on lohi_low"],
        [[s] + [f"{q(CONTROL_BOX[(s, t)][0], 4)} "
                f"({q(CONTROL_BOX[(s, t)][1], 3)})"
                for t in ("lohi_high", "lohi_low")]
         for s in ("lohi_high", "lohi_low")])
    tcap("control", "LoHi internal transfer control, defect-class mAP50 "
            "under the harmonised label space, mean and standard deviation "
            "over four configurations and three seeds, 12 cells per entry. "
            "Source: lohi_control.md.")

    hi_acq, lo_acq = (GEOMETRY["lohi_acquisitions"][a] for a in ("high", "low"))
    p(f"Part of this cost is prior shift rather than sensor shift, and we "
      f"report both. The defect mix differs between the two acquisitions. Pore "
      f"is {q(hi_acq['class_share_percent']['pore'], 1)} percent of high "
      f"resolution boxes and {q(lo_acq['class_share_percent']['pore'], 1)} "
      f"percent of low resolution boxes. Deposit runs the other way, "
      f"{q(hi_acq['class_share_percent']['deposit'], 1)} against "
      f"{q(lo_acq['class_share_percent']['deposit'], 1)} percent. Boxes per "
      f"image are {q(hi_acq['boxes_per_image'], 2)} and "
      f"{q(lo_acq['boxes_per_image'], 2)}, and median box area is "
      f"{q(hi_acq['median_box_area_percent'], 2)} and "
      f"{q(lo_acq['median_box_area_percent'], 2)} percent of the frame. A "
      f"reader told only that the line and the annotators are constant would "
      f"otherwise read the whole collapse as sensor shift.")

    ctl_img = control_image_ap()
    il = image_level_facts()
    c_hi, c_lo, cm = (il["arm_samples"]["lohi_high"],
                      il["arm_samples"]["lohi_low"], il["control"])
    ctl_par = (
        f"At image level the control is thin. lohi_high's test split holds "
        f"no defect-free image, so AUROC is undefined there and no score can "
        f"be overconfident on it. lohi_low's holds {c_lo['n_neg']}, each in "
        f"its own block. On those, models trained on lohi_high keep "
        f"{q(fnum(cm['retained_margin']), 2)} "
        f"[{q(fnum(cm['retained_margin_lo']), 2)}, "
        f"{q(fnum(cm['retained_margin_hi']), 2)}] of the in-domain margin "
        f"above chance, at an AUROC of {q(fnum(cm['auroc_ij']), 2)} against "
        f"{q(fnum(cm['auroc_jj']), 2)} in domain. That rests on "
        f"{c_lo['n_neg']} images, so the control remains a box-level result.")
    p(ctl_par)
    require_phrases("4.2 image-level paragraph", ctl_par, [
        f"keep {q(fnum(cm['retained_margin']), 2)} ",
        f"AUROC of {q(fnum(cm['auroc_ij']), 2)} against"])

    p(f"LoHi-WELD's own description runs the same internal test across its "
      f"two acquisitions, training on each and testing on the other, and "
      f"reports poor performance in both directions {cp('block2024')}. We cite "
      f"it as precedent rather than as novelty. Our version differs by "
      f"running under the leakage-audited splits and the same protocol as the "
      f"rest of the study, at four configurations rather than one.")

    figure("control_matrix",
           "The internal acquisition control. Annotation "
           "protocol, annotators, class list and production line are held "
           "constant and only the camera changes. Defect-class mAP50. "
           f"Image-level AP is not shown because test splits that are "
           f"{pct(fnum(c_hi['positive_rate']), 0)} and "
           f"{pct(fnum(c_lo['positive_rate']), 1)} percent positive floor it "
           f"at {q(min(ctl_img), 2)} in every cell. Part of the drop is prior "
           "shift: "
           "the defect mix differs between the acquisitions, pore by a factor "
           "of about five and deposit by about two.")

    # ------------------------------------------------------------------- 4.3
    h("4.3 The box-level cross-dataset matrix", 2)

    p("The box-level matrix has two training rows, not three. Only kaggle_v2 "
      "and lohi_weld use a localisation annotation protocol and are large "
      "enough to train on. weld_surface_v2 annotates a frame-sized quality "
      "verdict rather than a localised defect, so it cannot be a box-level "
      "training row or target column, and a number there would measure "
      "annotation mismatch rather than domain shift. That the public "
      "visible-light data supports only two localisation training rows is "
      "itself a finding about the state of the public data.",
      bold_lead="Draft. ")

    p("Throughout this section the box-level metric is average precision on "
      "the harmonised defect label space, not a mean over each dataset's own "
      "classes. Every defect class is collapsed to a single label and "
      "matching is class agnostic, because kaggle_v2 carries one generic "
      "defect class while lohi_weld carries four specific types and no "
      "correspondence between them exists. Per-class values in each source's "
      "own vocabulary are reported in domain only, in Section 4.9. "
      f"{T('box_yolo11s')} gives the matrix for YOLO11s, which "
      f"{F('box_heatmap')} draws, and {T('box_ratio')} gives the transfer "
      f"ratios for all four configurations.")

    box_targets = ("kaggle_v2", "lohi_weld", "visual", "twoclass")
    table(
        ["Trained on"] + list(box_targets),
        [[s] + [q(cell_mean("box_map50", s, t, "YOLO11s"), 3)
                for t in box_targets]
         for s in ("kaggle_v2", "lohi_weld")])
    tcap("box_yolo11s", "Box-level transfer, YOLO11s, defect-class mAP50, mean "
            "over three seeds. Diagonal entries are in domain. The other "
            f"three configurations are in {T('box_ratio')}. Source: "
            "transfer_box_yolo11s.md.")

    rows = []
    for (src, tgt), cfgd in BOX_OFF.items():
        rows.append([f"{src} to {tgt}"] +
                    [f"{q(cfgd[c][1], 3)} [{q(cfgd[c][2], 3)}, "
                     f"{q(cfgd[c][3], 3)}]"
                     + (" w" if cfgd[c][4] else "") for c in CFGS])
    table(["Pair"] + list(CFGS), rows)
    tcap("box_ratio", f"Box-level transfer ratio with 95 percent split-block "
            f"percentile intervals, {BOX_BOOTSTRAP_DRAWS:,} draws, all four "
            f"configurations. The AUROC intervals of {T('auroc')} use "
            f"{IMAGE_LEVEL_DRAWS:,}; box level uses fewer for cost, and every "
            f"released row records its own count. "
            "T is the foreign detector's defect-class mAP50 divided by the "
            "target's own in-domain value at the same configuration and the "
            "same seed. Numerator and denominator are resampled together, "
            "because both are scored on the same test images. Rows marked w "
            "carry a weak in-domain reference: twoclass trains on 213 images "
            "over 23 scene blocks, so a high ratio there says the denominator "
            "is small. The diagonal is the self-check and returns T exactly "
            "1.000 with zero width, which is how a reader can see the "
            "bootstrap is wired correctly. Source: "
            "transfer_ratio_intervals.md.")

    pij = [v[0] for cfg in BOX_OFF.values() for v in cfg.values()]
    t = [v[1] for cfg in BOX_OFF.values() for v in cfg.values()]
    lohi_diag = [v[0] for v in BOX_DIAG_STRICT["lohi_weld"].values()]
    p(f"Across all four configurations and all six localisation pairs, "
      f"cross-dataset defect-class mAP50 spans {q(min(pij), 4)} to "
      f"{q(max(pij), 4)}, against diagonals of {q(min(lohi_diag), 3)} to "
      f"{q(max(lohi_diag), 3)} on lohi_weld. As "
      f"a share of the target's own in-domain score a foreign detector "
      f"retains between {pct(min(t), 1)} and {pct(max(t), 1)} percent. "
      f"Both extremes need a qualifier. The maximum, {q(max(t), 3)} for "
      f"Faster R-CNN from lohi_weld into twoclass, carries the weak reference "
      f"flag, so it is partly a statement about a small denominator. The "
      f"minimum, {q(min(t), 3)} for YOLO11s from lohi_weld into visual, is "
      f"measured against a target whose own in-domain value is only "
      f"{q(BOX_DIAG_YOLO11S['visual'], 3)}. Set against "
      f"the {H['control'][0]} and {H['control'][1]} percent retained inside "
      f"LoHi in Section 4.2, cross-dataset transfer is worse by roughly an "
      f"order of magnitude. Visual shift alone does not account for it.")

    p("We do not attribute the remainder to a single named cause. The control "
      "isolates acquisition. The cross-dataset gap additionally mixes "
      "annotation protocol, object scale, defect vocabulary and training set "
      "size, and this study does not separate those from one another. "
      f"{T('diagonals')} gives the two in-domain localisation diagonals at "
      f"both IoU thresholds.")

    strict = [v[1] for dd in BOX_DIAG_STRICT.values() for v in dd.values()]
    table(["Diagonal", "Metric"] + list(CFGS),
          [["kaggle_v2", "mAP50"] +
           [q(BOX_DIAG_STRICT['kaggle_v2'][c][0], 4) for c in CFGS],
           ["kaggle_v2", "mAP50-95"] +
           [q(BOX_DIAG_STRICT['kaggle_v2'][c][1], 4) for c in CFGS],
           ["lohi_weld", "mAP50"] +
           [q(BOX_DIAG_STRICT['lohi_weld'][c][0], 4) for c in CFGS],
           ["lohi_weld", "mAP50-95"] +
           [q(BOX_DIAG_STRICT['lohi_weld'][c][1], 4) for c in CFGS]])
    ratio = [v[1] / v[0] for dd in BOX_DIAG_STRICT.values()
             for v in dd.values()]
    tcap("diagonals", f"The in-domain localisation diagonals at both IoU "
            f"thresholds. The stricter threshold retains {q(min(ratio), 2)} to "
            f"{q(max(ratio), 2)} of the mAP50 value, which is the usual ratio "
            f"for detectors of this class and indicates no unusual "
            f"localisation looseness. Off the "
            f"diagonal, mAP50-95 falls to between "
            f"{q(BOX_OFFDIAG_STRICT_SPAN[0], 4)} and "
            f"{q(BOX_OFFDIAG_STRICT_SPAN[1], 4)}, which is at the resolution "
            f"floor of the measurement, so the cross-dataset matrix is "
            f"reported at mAP50 and the stricter threshold is given here on "
            f"the diagonals where it still carries information.")

    p(f"Two diagonals are themselves low and this is a finding about the "
      f"public data rather than a pipeline fault. Averaged over the four "
      f"configurations, kaggle_v2's own defect-class mAP50 is "
      f"{q(BOX_DIAG_MEAN['kaggle_v2'], 3)} and visual's is "
      f"{q(BOX_DIAG_MEAN['visual'], 3)}, against lohi_weld's "
      f"{q(BOX_DIAG_MEAN['lohi_weld'], 3)}. We diagnosed the kaggle_v2 case "
      f"five ways.")

    k_seen, k_test, _ = AUDIT_OVERLAP["kaggle_v2"]
    p(f"It is not a label mapping fault, since feeding the ground truth back "
      f"as predictions gives an oracle mAP50 of 1.0000 and the excluded "
      f"verdict classes reach a maximum IoU of 0.460 against a defect box, "
      f"below the matching threshold. It is not object scale, since in the "
      f"640 pixel frame the model sees, kaggle_v2's defect boxes are larger "
      f"than lohi_weld's and visual's on every statistic, and lohi_weld still "
      f"reaches {q(BOX_DIAG_MEAN['lohi_weld'], 3)}. It is not input "
      f"resolution, "
      f"since 2.56 times the pixels buys 0.014 mAP50. A model trained on the "
      f"downloaded partition, which overlaps {pct(k_seen / k_test, 0)} "
      f"percent of the audited test images, reaches "
      f"{q(DOWNLOADED_KAGGLE_BOX, 3)} on that split where the audited model "
      f"reaches {q(BOX_DIAG_YOLO11S['kaggle_v2'], 3)}, so the annotations are "
      f"self-consistent enough to fit.")

    kc = {c: (ap, n) for s, c, ap, _sd, _b, n, _th, _st in PER_CLASS
          if s == "kaggle_v2"}
    kg = GEOMETRY["datasets"]["kaggle_v2"]
    lg = GEOMETRY["datasets"]["lohi_weld"]
    p(f"The fifth check is the most direct of the five. On the same "
      f"photographs, in the same training run, the same model reaches an AP50 "
      f"of {q(kc['Bad Weld'][0], 3)} on Bad Weld over {kc['Bad Weld'][1]} "
      f"test instances and {q(kc['Good Weld'][0], 3)} on Good Weld over "
      f"{kc['Good Weld'][1]}, while reaching {q(kc['Defect'][0], 3)} on "
      f"Defect over {kc['Defect'][1]}. The two quality verdict classes are "
      f"learned and the defect class is not. Nothing about the images, the "
      f"optimiser or the pipeline can produce that pattern, because all "
      f"three classes pass through every one of them. What differs is the "
      f"annotations. kaggle_v2 supplies {kg['train_defect_boxes']:,} defect "
      f"boxes across {kg['train_images_with_a_defect_box']:,} training "
      f"images, against lohi_weld's {lg['train_defect_boxes']:,}, and what "
      f"fails is generalising from them.")

    figure("box_heatmap",
           "Box-level cross-dataset transfer, YOLO11s, "
           "defect-class mAP50. Two training rows, because only kaggle_v2 and "
           "lohi_weld use a localisation protocol and are large enough to "
           "train on. weld_surface_v2 is absent by protocol, not by omission. "
           "Identity is never carried by colour alone.")

    # ------------------------------------------------------------------- 4.4
    h("4.4 The image-level matrix", 2)

    il = image_level_facts()
    ch, smp, vid = il["chance"], il["samples"], il["video"]
    why = (
        f"The image-level decision is whether an annotated target is present. "
        f"It is invariant to how boxes were drawn, so it is the one "
        f"granularity comparable across all five datasets, including the "
        f"whole-region source. Its usual summary, average precision minus "
        f"the target's positive rate, does not have a chance level of zero. "
        f"Under the evaluator's interpolated average precision, scores that "
        f"carry no information sit above the positive rate on splits this "
        f"small: by {q_signed(ch['kaggle_v2'], 3)} on kaggle_v2, down to "
        f"{q_signed(ch['lohi_weld'], 3)} on lohi_weld, as measured by "
        f"shuffling each cell's labels {IMAGE_LEVEL_SHUFFLES:,} times against "
        f"its own scores. A score that holds one value for every image earns "
        f"{q_signed(il['constant_kaggle'], 3)} on kaggle_v2, through tie "
        f"order. We therefore report the matrix in AUROC, whose chance level "
        f"is 0.5 and which gives a constant score exactly that. The lift and "
        f"its chance level stay as columns, so a reader can see both. "
        f"{T('auroc')} gives the matrix.")
    p(why, bold_lead="Draft. ")
    require_phrases("4.4 why not the lift", why, [
        f"by {q_signed(ch['kaggle_v2'], 3)} on kaggle_v2",
        f"down to {q_signed(ch['lohi_weld'], 3)} on lohi_weld",
        f"earns {q_signed(il['constant_kaggle'], 3)} on kaggle_v2"])

    rows = []
    for t in IL_TARGET_ORDER:
        srcs = [s for s in IL_TARGET_ORDER if s in TRAIN_SOURCES]
        if t not in TRAIN_SOURCES:
            srcs.append(t)
        for s in srcs:
            r = il["means"][(s, t, None)]
            rows.append([
                t, f"{s}, in domain" if s == t else s,
                f"{q(fnum(r['mean_auroc']), 3)} "
                f"[{q(fnum(r['mean_auroc_lo']), 3)}, "
                f"{q(fnum(r['mean_auroc_hi']), 3)}]",
                f"{r['cells_wholly_above_half']} of {r['n_cells']}",
                f"{r['cells_wholly_below_half']} of {r['n_cells']}",
                q_signed(fnum(r["mean_ap_lift"]), 3),
                q_signed(fnum(r["mean_chance_lift"]), 3),
                "yes" if t in il["under"] else "no"])
    table(["Target", "Trained on", "Mean AUROC, 95 percent interval",
           "Wholly above 0.5", "Wholly below 0.5", "AP lift", "Chance lift",
           f"Under {MIN_NEGATIVES} negatives"], rows)
    tcap("auroc", f"Image-level AUROC, the mean over four configurations "
            f"and three seeds, with a 95 percent split-block percentile "
            f"interval of {IMAGE_LEVEL_DRAWS:,} draws, the twelve cells "
            f"averaged inside each draw. The two counts are evaluations whose "
            f"own interval lies wholly above or wholly below chance. AP lift is "
            f"image-level average precision minus the positive rate. Chance "
            f"lift is the lift the same estimator gives when each cell's "
            f"labels are shuffled {IMAGE_LEVEL_SHUFFLES:,} times against its "
            f"scores: the level an uninformative scorer reaches, not zero. The "
            f"last column flags targets with fewer than {MIN_NEGATIVES} "
            f"defect-free test images, the evaluator's threshold for a false "
            f"alarm rate, where AUROC is reported but rests on few images. "
            f"Per-configuration values are in the released table. Source: "
            f"image_level_means.csv and image_level_matrix.md.")

    lo_in, hi_in = il["in_range"]
    lo_x, hi_x = il["cross_range"]
    inv = il["inverted"]
    shows = (
        f"The decision is learned in domain on every target, at "
        f"{q(lo_in, 3)} to {q(hi_in, 3)}, and all {il['in_n']} in-domain "
        f"evaluations sit wholly above 0.5. Across datasets the means run "
        f"from {q(lo_x, 3)} to {q(hi_x, 3)}. Of the {il['cross_n']} "
        f"cross-dataset evaluations, {il['cross_above']} sit wholly above 0.5 "
        f"and {il['cross_below']} wholly below. On kaggle_v2, the one balanced "
        f"target, none of the {il['kaggle_n']} does. {inv['source']} to "
        f"{inv['target']} is inverted, at {q(fnum(inv['mean_auroc']), 3)}, "
        f"with its whole interval below chance. {T('blocks')} counts the "
        f"test splits behind these means in split blocks.")
    p(shows)
    require_phrases("4.4 what the matrix shows", shows, [
        f"{q(lo_in, 3)} to {q(hi_in, 3)}", f"{q(lo_x, 3)} to {q(hi_x, 3)}",
        f"{il['cross_above']} sit wholly above",
        f"none of the {il['kaggle_n']} does"])

    table(["Test split", "Images", "Blocks",
           "Defective images, blocks, effective",
           "Defect-free images, blocks, effective", "Largest block"],
          [[t, smp[t]["n_images"], smp[t]["n_blocks"],
            f"{smp[t]['n_pos']}, {smp[t]['pos_blocks']}, "
            f"{q(fnum(smp[t]['pos_effective_blocks']), 1)}",
            f"{smp[t]['n_neg']}, {smp[t]['neg_blocks']}, "
            f"{q(fnum(smp[t]['neg_effective_blocks']), 1)}",
            smp[t]["largest_block_images"]] for t in IL_TARGET_ORDER])
    tcap("blocks", "The audited image-level test splits counted in split "
            "blocks, the unit of independence in this study. Effective blocks "
            "are defined in the text. Source: image_level_samples.csv.")

    k, w, lo_ = smp["kaggle_v2"], smp["weld_surface_v2"], smp["lohi_weld"]
    sample = (
        f"The sample behind each mean is thinner than its image count "
        f"suggests, because images from one scene are not independent. We "
        f"count it in split blocks, and give each class an effective number "
        f"of blocks: the squared sum of its per-block counts over the sum of "
        f"their squares, so a class spread evenly over n blocks counts n and "
        f"a class in one block counts 1. kaggle_v2's {k['n_pos']} defective "
        f"test images sit in {k['pos_blocks']} blocks, an effective "
        f"{q(fnum(k['pos_effective_blocks']), 1)}, because one block of "
        f"{vid['images']} frames from a single video, SampleV1_2_mp4, holds "
        f"{vid['pos']} of them and {vid['neg']} defect-free images besides. "
        f"weld_surface_v2's {w['n_neg']} defect-free images sit in "
        f"{w['neg_blocks']} blocks, an effective "
        f"{q(fnum(w['neg_effective_blocks']), 1)}, fewer independent "
        f"defect-free scenes than lohi_weld's {lo_['n_neg']} images in "
        f"{lo_['neg_blocks']} blocks. The evaluator's rule of {MIN_NEGATIVES} "
        f"defect-free images counts images and does not see this.")
    p(sample)
    require_phrases("4.4 the sample behind it", sample, [
        f"block of {vid['images']} frames", f"holds {vid['pos']} of them",
        f"{w['n_neg']} defect-free images sit in {w['neg_blocks']} blocks"])

    full, nov = il["lk"]["full"], il["lk"][NO_VIDEO]

    def band(r):
        return (f"{q_signed(fnum(r['retained_margin']), 3)} "
                f"[{q_signed(fnum(r['retained_margin_lo']), 3)}, "
                f"{q_signed(fnum(r['retained_margin_hi']), 3)}]")
    pair = (
        f"An earlier version of this analysis led with one pair, lohi_weld to "
        f"kaggle_v2. Measured against chance, the share of the in-domain "
        f"margin that a foreign detector keeps is its AUROC minus 0.5 over "
        f"the target's own AUROC minus 0.5, at the same configuration and "
        f"seed, with both scored on the same resampled blocks. Over the four "
        f"configurations it is {band(full)} for this pair, and {band(nov)} "
        f"with the video block left out. Neither interval excludes zero, and "
        f"neither excludes the 13 to 21 percent an earlier summary of these "
        f"models suggested. The image-level decision is not shown to survive "
        f"transfer to this target, and it is not shown to fail. At box level "
        f"the same models retain {H['locate'][0]} to {H['locate'][1]} percent "
        f"of in-domain localisation, so no ratio between the two "
        f"granularities is quoted.")
    p(pair)
    require_phrases("4.4 the retained margin", pair, [
        band(full), band(nov), f"{H['locate'][0]} to {H['locate'][1]} percent"])

    fam = il["family"]
    cue = (
        f"kaggle_v2 is itself an aggregate of several collections, and the "
        f"collection predicts the label. A score that looks at no pixel, the "
        f"positive share of the image's filename family among the other "
        f"blocks, reaches an AUROC of {q(fnum(fam['mean_auroc']), 3)} "
        f"[{q(fnum(fam['mean_auroc_lo']), 3)}, "
        f"{q(fnum(fam['mean_auroc_hi']), 3)}], above the in-domain detectors' "
        f"{q(il['in']['kaggle_v2'], 3)}. So the in-domain margin on this "
        f"target may be collection appearance and not defect recognition. It "
        f"qualifies kaggle_v2 as the balanced target in either direction.")
    p(cue)
    require_phrases("4.4 the filename cue", cue, [
        f"AUROC of {q(fnum(fam['mean_auroc']), 3)}",
        f"detectors' {q(il['in']['kaggle_v2'], 3)}"])

    ex, dep = il["exception"], il["deposit"]
    checks = (
        f"Two further checks were chosen after seeing the data and are "
        f"exploratory. Counting a Bad Weld box as a defect, and four other "
        f"ways of turning a model's boxes into an image score, do not change "
        f"the reading: every foreign twelve-cell interval still contains 0.5. "
        f"One configuration stands out, Faster R-CNN from lohi_weld, whose "
        f"sum of scores reaches {q(fnum(ex['mean_auroc']), 3)} "
        f"[{q(fnum(ex['mean_auroc_lo']), 3)}, "
        f"{q(fnum(ex['mean_auroc_hi']), 3)}]. Scoring a lohi_weld model by "
        f"its deposit predictions alone gives {q(fnum(dep['mean_auroc']), 3)} "
        f"[{q(fnum(dep['mean_auroc_lo']), 3)}, "
        f"{q(fnum(dep['mean_auroc_hi']), 3)}], a lead and not a result. All "
        f"of these are in image_level_kaggle_checks.csv.")
    p(checks)
    require_phrases("4.4 the exploratory checks", checks, [
        f"reaches {q(fnum(ex['mean_auroc']), 3)}",
        f"alone gives {q(fnum(dep['mean_auroc']), 3)}"])

    # The confound exists only if the recipes were tuned on kaggle_v2, so the
    # paragraph that states it follows TUNED_ON.
    if TUNED_ON != "none":
        p(f"Hyperparameters were tuned once, on kaggle_v2's validation split, "
          f"and kaggle_v2 is also the source with the largest in-domain margin "
          f"at {q_signed(IMG_DIAG['kaggle_v2']['lift'], 3)} averaged over the "
          f"four configurations. We state both halves "
          "together because separating them would hide the confound. The "
          "honest reading is that kaggle_v2's diagonal is the one number in "
          "the study that may carry an optimistic bias, which makes the "
          "cross-domain collapse away from kaggle_v2 a conservative estimate "
          "rather than an inflated one.")
    confirm_note("tuning")

    # ----------------------------------------------------------------- 4.4.1
    h("4.4.1 What the leakage audit changes", 3)

    p("The audit protocol is described in Section 3.4. Its effect is a result "
      "and we report it here. For each of the three training sources we have "
      "three numbers. A is the model trained and tested on the partitions as "
      "downloaded, which is what a practitioner following the shipped "
      "pipeline obtains. C is the model trained and tested on our audited "
      "partitions. B holds the downloaded weights fixed and regrades them on "
      "the audited test split. All three are YOLO11s at three seeds, because "
      "the downloaded-data baseline was run at that configuration only, and "
      "holding the configuration fixed is what makes A and C comparable. "
      f"{T('audit')} gives A and C.",
      bold_lead="Draft. ")

    au = image_level_facts()["audit"]
    audit_rows = []
    for s in AUDIT_SOURCES:
        a = au[s]
        seen, n_test, _ = AUDIT_OVERLAP[s]
        audit_rows.append([s, q(a["A"], 3), q(a["C"], 3), q_signed(a["A_C"], 3),
                           q_signed(a["A_C_ap"], 3),
                           f"{a['A_neg']} and {a['C_neg']}",
                           f"{seen} of {n_test}, {pct(seen / n_test, 0)} "
                           f"percent"])
    table(["Source", "A, AUROC", "C, AUROC", "A minus C, AUROC",
           "A minus C, AP", "Defect-free test images, shipped and audited",
           "Audited test images inside downloaded train"], audit_rows)
    tcap("audit", "The effect of the leakage audit, YOLO11s at three "
            "seeds, in AUROC with average precision beside it. A is trained "
            "and tested on the partitions as downloaded, C on the audited "
            "partitions. They are measured on different test partitions with "
            "different positive rates, so their difference is an estimate "
            "shift under a change of protocol and not a causal estimate of "
            "leakage inflation, and AUROC rather than average precision is "
            "the scale that compares across them. A shipped split carries no "
            "split blocks, so A's intervals in the released table treat "
            "images as independent. Column C is the YOLO11s diagonal, while "
            f"{T('auroc')} averages four configurations: for kaggle_v2 those are "
            f"{q(au['kaggle_v2']['C'], 3)} and "
            f"{q(image_level_facts()['in']['kaggle_v2'], 3)}. The last column "
            "is measured by filename and confirmed byte identical by sha256. "
            "Source: image_level_audit.csv and downloaded_vs_audited.csv.")

    k_, w_, l_ = au["kaggle_v2"], au["weld_surface_v2"], au["lohi_weld"]
    moved = (
        f"Auditing lowers in-domain AUROC by {q(k_['A_C'], 3)} on kaggle_v2 "
        f"and by {q(w_['A_C'], 3)} on weld_surface_v2. In average precision "
        f"the second looked like nothing, {q(w_['A_C_ap'], 3)}, because "
        f"average precision sits on its ceiling on a split that is "
        f"{pct(w_['C_rate'], 0)} percent defective. On its shipped split, "
        f"weld_surface_v2's downloaded model misorders at most "
        f"{w_['pairs_bad']:g} of {w_['pairs']:,} defective and defect-free "
        f"pairs. lohi_weld moves by {q_signed(l_['A_C'], 3)}, which rests on "
        f"{l_['A_neg']} and {l_['C_neg']} defect-free images and is not "
        f"evidence of a loss. So two of the three sources tested move, not "
        f"one.")
    p(moved)
    require_phrases("4.4.1 what the audit moves", moved, [
        f"AUROC by {q(k_['A_C'], 3)} on kaggle_v2",
        f"by {q(w_['A_C'], 3)} on weld_surface_v2",
        f"nothing, {q(w_['A_C_ap'], 3)}",
        f"{w_['pairs_bad']:g} of {w_['pairs']:,}",
        f"moves by {q_signed(l_['A_C'], 3)}"])

    shares = [seen / n for seen, n, _ in AUDIT_OVERLAP.values()]
    held_out = [h for _, _, h in AUDIT_OVERLAP.values()]
    p(f"We report A minus C and not A minus B. The downloaded and audited "
      f"partitions were drawn independently by different parties, so most of "
      f"each audited test split sits inside the corresponding downloaded "
      f"training split, at {pct(min(shares), 0)} to {pct(max(shares), 0)} "
      f"percent as the table shows. B therefore scores a model largely on "
      f"images it was trained on, and it must not be presented as a held-out "
      f"test. Its only use is to hold the weights fixed while the grading "
      f"partition changes, so {T('audit')} does not print it. Its values are in "
      f"image_level_audit.csv and downloaded_vs_audited.csv. Restricting B to "
      f"the genuinely unseen remainder "
      f"was considered and rejected: the remainders are {held_out[0]}, "
      f"{held_out[1]} and {held_out[2]} images and two of the three are "
      f"almost entirely one class, which is too thin to carry a number. We "
      f"state the overlap instead of pretending a {min(held_out)} image "
      f"subset settles it.")

    p("A minus C changes the whole pipeline at once, the training data, the "
      "test data and the partitioning procedure. It is the difference between "
      "two protocols rather than the isolated effect of removing duplicates. "
      "We report it as an estimate shift and make no causal claim from it.")

    note("This is the most transferable finding in the paper for other "
         "dataset authors. The audit itself found that four fifths of what "
         "the four Roboflow sources appear to offer is augmentation and "
         "repeated frames: 22,193 shipped files reduce to 4,763 unique "
         "images. Consider whether that belongs here or in Section 3.4.")

    # ------------------------------------------------------------------- 4.5
    h("4.5 The label cost curve", 2)

    p("This arm measures how many labelled target images are needed to "
      "recover in-domain performance. Recovery is normalised against the "
      "target's own in-domain reference at the same configuration and the "
      "same seed. We report the checkpoint at the final epoch of the fixed "
      "schedule as the primary endpoint, because its label cost is exactly "
      "the draw. Selecting on the target's validation split would consume up "
      "to 444 further target labels and would change the selection criterion "
      "between zero shots and more than zero, so it is reported as a "
      "secondary endpoint with the validation set size stated beside it.",
      bold_lead="Draft. ")

    A = armb_summary()
    p(f"Recovery rises with the annotation budget on almost every pair. At "
      f"the largest budget tested it runs from {q(A['min'], 3)} to "
      f"{q(A['max'], 3)} of the in-domain gap across the {word(A['n'])} pair "
      f"and configuration combinations, with a median of "
      f"{q(A['median'], 3)}. "
      f"{word(A['in_band']).capitalize()} of the {word(A['n'])} fall between "
      f"{q(ARMB_R250_BAND[0], 2)} and {q(ARMB_R250_BAND[1], 2)}. "
      f"{T('label_cost')} gives every pair and configuration.")

    table(
        ["Pair", "Config", "P(0)", "k=10", "k=50", "k=250", "P_jj", "R@250"],
        [[f"{s} to {t}" + (" (w)" if weak else ""), cfg, q(p0, 4),
          q(k10, 4), q(k50, 4),
          q(k250, 4) + (f" (k={TWOCLASS_TRAIN})" if t == "twoclass"
                        else ""), q(pjj, 4), q(r, 3)]
         for s, t, cfg, p0, k10, k50, k250, pjj, r, weak in ARMB_ROWS])
    tcap("label_cost", "Label cost curve, defect-class mAP50 at the primary "
            "endpoint, mean over three draws. Absolute average precision is "
            "printed beside every ratio because both terms sit near the floor "
            "in places. Recovery is the mean of the per-draw ratios, each "
            "normalised by its own seed's in-domain reference, so it is not "
            "recoverable from the two denominator columns by division. "
            "Recovery is never pooled across pairs: the denominators span a "
            "factor of nine. THE FOUR twoclass CELLS AT THE LARGEST BUDGET "
            "ARE NOT AT k = 250. twoclass holds 213 training images, so the "
            "draw exhausts the split at 213 images over 22 scenes, and those "
            "cells are marked accordingly. Rows marked (w) carry a weak "
            "in-domain reference. Source: armb_recovery.csv.")

    p("Two cautions travel with this table. First, the ratio into twoclass "
      "must always be quoted with its reference score. twoclass's in-domain "
      "reference is weak by construction, at 213 training images distributed "
      "over 23 scene blocks and 7 joint configurations, so the value above 1 "
      "says the denominator is small rather than that transfer is excellent. "
      "That same cell is also the one where the budget is 213 rather than "
      "250. Second, the two hard targets are visual and kaggle_v2, whose own "
      f"in-domain box average precision is {q(BOX_DIAG_MEAN['visual'], 3)} "
      f"and {q(BOX_DIAG_MEAN['kaggle_v2'], 3)} averaged over the "
      "four configurations. Their absolute "
      "numbers stay small even at a respectable ratio, which is why the "
      "absolute value is printed in every row.")

    il = image_level_facts()
    lc, kk = il["cost"], il["samples"]["kaggle_v2"]
    ws = il["samples"]["weld_surface_v2"]
    z = [fnum(lc[c][0]["zero_shot_auroc"]) for c in ("rtdetr_l", "yolo11s")]
    top = [fnum(lc[c][-1]["auroc"]) for c in ("rtdetr_l", "yolo11s")]
    own = [fnum(lc[c][0]["in_domain_auroc"]) for c in ("rtdetr_l", "yolo11s")]
    rm = [lc[c][-1] for c in ("rtdetr_l", "yolo11s")]

    def rband(r):
        return (f"{q_signed(fnum(r['retained_margin']), 2)} "
                f"[{q_signed(fnum(r['retained_margin_lo']), 2)}, "
                f"{q_signed(fnum(r['retained_margin_hi']), 2)}]")
    if not all(0.4 <= fnum(r["retained_margin"]) <= 0.7 for r in rm):
        raise SystemExit("\nCONSISTENCY CHECK FAILED: 4.5 says the largest "
                         "budget restores about half of the decision.")
    img_cost = (
        f"At image level, two targets clear the evaluator's rule of "
        f"{MIN_NEGATIVES} defect-free test images, kaggle_v2 and "
        f"weld_surface_v2, and weld_surface_v2's {ws['n_neg']} sit in "
        f"{ws['neg_blocks']} blocks ({T('blocks')}). We therefore follow the "
        f"decision through fine-tuning on kaggle_v2, whose {kk['n_neg']} "
        f"defect-free test images sit in {kk['neg_blocks']} blocks. From "
        f"lohi_weld, "
        f"AUROC rises from {q(z[0], 2)} and {q(z[1], 2)} at zero shot to "
        f"{q(top[0], 2)} and {q(top[1], 2)} at the largest budget, against "
        f"{q(own[0], 2)} and {q(own[1], 2)} in domain, for RT-DETR-l and "
        f"YOLO11s. That retains {rband(rm[0])} and {rband(rm[1])} of the "
        f"in-domain margin. Of the {word(il['small_n'])} intervals from zero "
        f"shot to {ARMB_K[3]} requested images, {word(il['small_excl'])} "
        f"excludes zero. A few labelled images do not visibly restore the "
        f"decision on this target, and a few hundred restore about half of "
        f"it. These are at the primary endpoint, and every pair, "
        f"weld_surface_v2's included, is in image_level_label_cost.csv.")
    p(img_cost)
    require_phrases("4.5 image level", img_cost, [
        f"from {q(z[0], 2)} and {q(z[1], 2)} at zero shot",
        f"retains {rband(rm[0])} and {rband(rm[1])}",
        f"{word(il['small_excl'])} excludes zero"])

    p("The detection head is reinitialised whenever the source and target "
      "class counts differ, which for this study is the only possible design, "
      "because the premise is that the taxonomies differ. Only the backbone "
      "and neck carry over from the source checkpoint. This arm therefore "
      "measures the target annotation budget needed to fit a new detection "
      "head on a source-pretrained backbone, which is what a practitioner "
      "adopting a public checkpoint actually faces. It also means the "
      "smallest budgets partly measure a head that has not finished training. "
      "In 17 of 360 box-defined records the reinitialised head collapses onto "
      "a single wrong class, emitting its full detection cap in classes that "
      "are not the defect class at confidences at or below 0.185. All are at "
      "the smallest budgets. We retain those cells, flagged, because a head "
      "that collapses on ten images is exactly what a label cost curve exists "
      "to reveal.")

    p(f"Recovery is plotted in {F('label_cost_curve')} against the realised "
      "scene count rather than against the requested budget, and the draws "
      "are shown as points rather than joined into a line. A split block is indivisible, so a draw that "
      "sits below the requested count can overshoot it in a single step, by "
      "up to 9.1 times at the smallest budget. The requested budget is a "
      "sampling instruction and not a measured quantity.")

    figure("label_cost_curve",
           "The label cost curve, defect-class mAP50 at the "
           "primary endpoint, plotted against the realised scene count. "
           "Points are individual draws, not a mean, because on a small "
           "target the outcome varies several-fold between attempts (Section "
           "4.5.1). The detection head is "
           "reinitialised at every budget, so the smallest budgets partly "
           "measure a head that has not finished training rather than "
           "adaptation being hard. Pre-registered schedule throughout. "
           "Panels: "
           + ", ".join(f"({chr(97 + i)}) {s_} to {t_}"
                       for i, (s_, t_) in enumerate(LABEL_COST_PAIRS))
           + ". Hollow markers are draws that exhausted the target's "
           "training split.")

    h("4.5.1 At small budgets the outcome varies between attempts", 3)

    td = TWOCLASS_DRAWS
    if td["most_best"] != td["mid_n"]:
        raise SystemExit("\nCONSISTENCY CHECK FAILED: 4.5.1 says the draw "
                         "spanning the most scenes scored best every time.")
    im, scn = td["k10_images"], td["k10_scenes"]
    vary = (
        f"At {word(ARMB_K[0])} requested labelled images into twoclass, the "
        f"smallest target, the three draws hold {im[0]}, {im[1]} and {im[2]} "
        f"images in {scn[0]}, {scn[1]} and {scn[2]} scenes. Their "
        f"defect-class mAP50 varies between {q(td['k10_spread'][0], 1)} and "
        f"{q(td['k10_spread'][1], 1)} times across the "
        f"{word(td['combos'])} source and configuration combinations. Two "
        f"terms contribute, and this design ties them, because each draw also "
        f"trains with its own seed. The arm itself bounds the training term. "
        f"At the largest budget the draw exhausts twoclass's "
        f"{TWOCLASS_TRAIN} training images, so all three runs train on "
        f"identical images, and they still spread by up to "
        f"{q(td['top_spread'], 1)} times. The sampling term shows where draws "
        f"differ in scene count. At {ARMB_K[2]} and {ARMB_K[3]} requested "
        f"images, the draw spanning the most scenes scored best in all "
        f"{word(td['combos'])} combinations, and the draw spanning the fewest "
        f"scored worst in {word(td['fewest_worst'])} of {word(td['mid_n'])}. "
        f"Neither term is negligible on a target this small.")
    p(vary, bold_lead="Draft. ")
    require_phrases("4.5.1", vary, [
        f"hold {im[0]}, {im[1]} and {im[2]} images",
        f"between {q(td['k10_spread'][0], 1)} and "
        f"{q(td['k10_spread'][1], 1)} times",
        f"up to {q(td['top_spread'], 1)} times",
        f"worst in {word(td['fewest_worst'])} of {word(td['mid_n'])}"])

    p("Whichever term dominates, the practical consequence is the same and "
      "it is a result "
      "rather than a caveat. On a small, low-scene-count target, what a "
      "practitioner obtains for a fixed annotation budget varies several-fold "
      "between one attempt and another. A single "
      "few-shot number is not an annotation budget estimate. This also sets "
      "the floor on how precisely any few-shot curve can be read at small "
      "budgets, which is why every cell in this arm is reported per draw and "
      "never as a pooled mean alone.")

    h("4.5.2 Two exploratory controls on the shape of the curve", 3)

    p("The following two controls are exploratory and were not "
      "pre-registered. The training schedule was fixed a priori at 50 epochs "
      "before any cell was trained and was not changed afterwards. The "
      "pre-registered curve in Section 4.5 is the result. These controls "
      "explain its shape and say what a practitioner willing to train longer "
      f"could expect. Their {SCHEDULE_COUNTS['n']} runs were trained on a "
      "workstation rather than the cluster, so they are compared only with "
      "each other and never with the cells of Section 4.5.",
      bold_lead="Draft. ")

    p("At the smallest budget the optimisation budget is about 100 steps, "
      "against about 1,600 at the largest. A low score at a small budget "
      "could therefore be too few labels or too few steps. Two controls "
      "separate them on kaggle_v2 to lohi_weld. Oversampling replicates each "
      "drawn image within the epoch, which multiplies the steps while leaving "
      "the cosine learning-rate schedule and the mosaic window identical to "
      "the pre-registered arm. Matched-step training instead extends the "
      "epoch count, which stretches both. "
      f"{T('schedule')} gives both contrasts.")

    table(
        ["Contrast", "What it changes", "k=10", "k=50"],
        [["Fixed to oversampled", "step count only"]
         + [q_signed(SCHEDULE_TABLE[k][0], 4) for k in (10, 50)],
         ["Oversampled to matched step", "schedule shape only"]
         + [q_signed(SCHEDULE_TABLE[k][1], 4) for k in (10, 50)]])
    tcap("schedule", "Exploratory schedule controls, change in defect-class "
            "mAP50, kaggle_v2 to lohi_weld, mean over three draws. The step "
            "count is the consistent driver. The schedule-shape term does not "
            "hold its sign across the two budgets, so no single percentage "
            "split is quoted. Exploratory, not pre-registered. Source: "
            "matched_step_control.csv.")

    rec = {s: schedule_recovery(s, "kaggle_v2", "lohi_weld")
           for s in ("fixed_50", "oversampled_50", "matched_step")}
    floor = (
        f"So the floor at small budgets is an optimisation budget limit rather "
        f"than a label limit. Recovery on that pair rises from "
        f"{q(rec['fixed_50'], 3)} at the pre-registered schedule to "
        f"{q(rec['oversampled_50'], 3)} with the step budget matched under the "
        f"same schedule shape, and to {q(rec['matched_step'], 3)} with "
        f"matched-step training. We quote {q(rec['oversampled_50'], 3)}, "
        f"because it changes exactly one thing against the pre-registered "
        f"arm. All three use the main grid's three-seed means for the "
        f"zero-shot and in-domain scores, as {T('label_cost')} does.")
    p(floor)
    require_phrases("4.5.2 recoveries", floor, [
        f"from {q(rec['fixed_50'], 3)} at", f"to {q(rec['matched_step'], 3)} "
        "with"])

    dr = {t: schedule_delta_r("kaggle_v2", t)
          for t in ("lohi_weld", "twoclass", "visual")}
    own_ref = {t: cell_mean("box_map50", t, t, "YOLO11s", kind_of(t))
               for t in dr}
    p(f"Target headroom does not explain which pairs respond. Holding the "
      f"source at kaggle_v2 and varying only the target, the change in "
      f"recovery is {q_signed(dr['lohi_weld'], 4)} into lohi_weld, "
      f"{q_signed(dr['twoclass'], 4)} into twoclass and "
      f"{q_signed(dr['visual'], 4)} into visual, against in-domain references "
      f"of {q(own_ref['lohi_weld'], 3)}, {q(own_ref['twoclass'], 3)} and "
      f"{q(own_ref['visual'], 3)}. "
      "twoclass has three quarters of lohi_weld's headroom and an almost "
      "identical response, and visual has a fifth of it and still returns "
      "half. If headroom were the mechanism these would order with it. They "
      "do not. These predictions were recorded before the cells were run: one "
      "was confirmed and one refuted.")

    vis_fixed = schedule_mean("fixed_50", "kaggle_v2", "visual")
    vis_over = schedule_mean("oversampled_50", "kaggle_v2", "visual")
    p(f"visual's response is real and small. Its fixed arm is "
      f"{q(vis_fixed, 4)} defect-class mAP50 and its oversampled arm "
      f"{q(vis_over, 4)}. A recovery change of {q(dr['visual'], 3)} built on "
      f"a {q(vis_over - vis_fixed, 4)} absolute gain must not be read as "
      f"visual behaving like lohi_weld. It moves off a floor by an amount no "
      f"inspector would notice.")

    lk = schedule_delta_r("lohi_weld", "kaggle_v2")
    p(f"One cell in this control resists explanation and we state that "
      f"rather than resolve it by assertion. Adapting from lohi_weld to "
      f"kaggle_v2 responds at {q_signed(lk, 4)}, "
      f"{word(int(q(min(dr.values()) / lk, 0)))} times below the weakest of "
      f"the three above. "
      "That cell differs from the others in both its source and its target, "
      "and this experiment held the source fixed, so the two were not "
      "separated. The difficulty of kaggle_v2 as a target, diagnosed on five "
      "independent checks in Section 4.3, is a candidate explanation rather "
      "than a demonstrated mechanism, and we present it as such. We did not "
      "run the further cells that would separate source from target, because "
      f"twoclass already varies between "
      f"{q(TWOCLASS_DRAWS['k10_spread'][0], 1)} and "
      f"{q(TWOCLASS_DRAWS['k10_spread'][1], 1)} times across three draws at "
      "ten requested images, and three more draws on a second source could "
      "not have separated a source effect from draw identity on a target "
      "that variable.")

    # ------------------------------------------------------------------- 4.6
    h("4.6 Pooling multiple training sources", 2)

    p("This arm trains on the pool of all training sources except one and "
      "tests on the held-out one. Each pooled fold is compared with the "
      "better of its two members, the models trained on each of the pool's "
      "sources alone, at the same configuration and the same seed. The "
      "better member is chosen on the target's own test split, which makes "
      "it an optimistic comparator, and a pool trains on more images than "
      "either member, so the training budget is not held fixed. Comparing "
      "against a source outside the pool would change the data and answer a "
      "different question.", bold_lead="Draft. ")

    pl = image_level_facts()["pool"]
    if not pl["worse_wins"] - pl["better_wins"] > pl["better_wins"]:
        raise SystemExit("\nCONSISTENCY CHECK FAILED: 4.6 says a pool lands "
                         "between its members more often than it beats the "
                         "better one.")
    pool_par = (
        f"At image level, in AUROC, pooling does not reliably help. Over the "
        f"{pl['n']} cells the pooled model averages {q(pl['pooled'], 3)}, "
        f"against {q(pl['better'], 3)} for the better member and "
        f"{q(pl['worse'], 3)} for the worse. It beats the better member in "
        f"{pl['better_wins']} of {pl['n']} cells, the two members' mean in "
        f"{pl['mean_wins']} and the worse member in {pl['worse_wins']}. With "
        f"seeds averaged first the counts are {pl['seed_better']}, "
        f"{pl['seed_mean']} and {pl['seed_worse']} of {pl['seed_n']}. So a "
        f"pool lands between its two members more often than it beats the "
        f"better one. {T('pooling')} gives every fold and target.")
    p(pool_par)
    require_phrases("4.6 pooling", pool_par, [
        f"averages {q(pl['pooled'], 3)}",
        f"in {pl['better_wins']} of {pl['n']} cells",
        f"are {pl['seed_better']}, {pl['seed_mean']} and {pl['seed_worse']} "
        f"of {pl['seed_n']}"])

    pool_rows = pool_table_rows()
    n_box = sum(1 for r in pool_rows if not r[5].startswith("undefined"))
    table(["Pool", "Held-out target", "Role", "AUROC, pooled",
           "AUROC, better member", "Box mAP50"], pool_rows)
    tcap("pooling", f"Pooling, all {word(len(pool_rows))} fold and target "
            f"combinations, ranges over the {word(len(CFGS))} configurations "
            f"with seeds averaged within each. Image-level values are AUROC, "
            f"for the pooled model and for the better of its two members, "
            f"chosen per configuration on the target's own test split. A "
            f"box-level number is undefined for every held-out training "
            f"source, for the protocol reason given below, and the arm's "
            f"box-level evidence is the {word(n_box)} rows that carry one. The "
            f"{pl['seed_better']} of {pl['seed_n']} seed-averaged comparisons "
            f"quoted above are these {word(len(pool_rows))} rows crossed with "
            f"the {word(len(CFGS))} configurations. Source: "
            f"image_level_pooling_seed_averaged.csv, armd.csv and "
            f"armd_box_limit.json.")

    p("This result must be read with the honest limit of the design attached. "
      "There are only three training sources, so a leave-one-out fold trains "
      "on two. This is leave-one-out over a very small pool count and a weak "
      "test of whether breadth helps. It does not license a claim that "
      "pooling never helps. It says that two-source pooling did not help "
      "here.")

    p("This arm reports a box-level number for neither of its held-out "
      "training sources, and that is a consequence of the annotation protocol "
      "rather than a run that failed. Two exclusions compose. Any pool "
      "containing weld_surface_v2 is protocol-mixed, because that source "
      "annotates a whole-region verdict, so two of the three pools carry no "
      "box-level number at all. The one pool that excludes it is by "
      "construction held out against weld_surface_v2 itself, which cannot "
      "receive a box-level score for the same reason. The box-level evidence "
      "from this arm is therefore exactly two pairs, the localisation pool "
      "scored on visual and on twoclass. We state this so that an empty "
      "column is not read as a missing result.")

    # ------------------------------------------------------------------- 4.7
    h("4.7 Recall at a fixed false positive budget", 2)

    p("Inspection requirements are specified as a recall at a tolerated false "
      "alarm rate, so we report recall at 0.1, 0.5 and 1.0 false positive "
      "boxes per image. This metric carries the inspection claim rather than "
      "AUROC, for two reasons. It counts false positive boxes rather than "
      "false positive images, so it stays measurable on every target "
      "including the three whose negative image counts are too small for a "
      "false alarm rate. And it is the form in which the requirement is "
      "written.", bold_lead="Draft. ")

    p("As stated in Section 3.6, the metric covers the internal control, the "
      "label cost curve and the pooling arm, and not the zero-shot cells of "
      "the main matrix. The figures below therefore describe adapted and "
      "in-domain models rather than the zero-shot transfer of Sections 4.3 "
      f"and 4.4. {T('recall_control')} gives the internal control and "
      f"{T('recall_cross')} the cross-dataset models of Section 4.5.")

    table(
        ["Trained on", "Tested on"] + [f"R at {b} FP" for b in FP_BUDGETS],
        [[s, t] + [q(v, 3) for v in RECALL_CONTROL[(s, t)]]
         for s, t in RECALL_CONTROL])
    tcap("recall_control", "Recall at a fixed false positive budget per image, "
            "internal acquisition control, mean over four configurations and "
            "three seeds. Source: recall_at_fp_budget_arms.csv.")

    p(f"The operating-point view is harsher than the mAP view. Inside the "
      f"control, tolerating one false positive box per image retains "
      f"{span(RECALL_IN[-1], pct, 0)} percent recall in domain and "
      f"{span(RECALL_CROSS[-1], pct, 0)} percent across the acquisition "
      f"boundary. At the stricter budget of {FP_BUDGETS[0]} false positives "
      f"per image, in-domain recall is already only "
      f"{span(RECALL_IN[0], q, 3)}, and cross-acquisition recall falls to "
      f"{span(RECALL_CROSS[0], q, 3)}. A practitioner who specifies a tight "
      f"false alarm budget should expect a detector to find fewer than one "
      f"defect in twenty after a change of camera.")

    table(["Pair", "Config"] + [f"R at {b} FP" for b in FP_BUDGETS],
          [[f"{s} to {t}" + (f" (w, k={TWOCLASS_TRAIN})" if t == "twoclass"
                             else ""), cfg] + [q(v, 3) for v in vals]
           for s, t, cfg, *vals in FP_ARMB_K250])
    tcap("recall_cross", "Recall at a fixed false positive budget for the "
            "cross-dataset models of Section 4.5, AT THE LARGEST ANNOTATION "
            "BUDGET ONLY, mean over three draws. The released summary table "
            "pools this metric over the whole budget grid, and a recall "
            "pooled across annotation budgets is not a quotable quantity, so "
            "the largest budget is reported here and the per-budget values "
            "are in the released CSV. The twoclass rows are marked exactly as "
            f"in {T('label_cost')}: their budget is 213 rather than 250 because the "
            "draw exhausts that split, and (w) marks a weak in-domain "
            "reference. Source: recall_at_fp_budget_arms.csv.")

    fp10 = {(s, t, cfg): vals[-1] for s, t, cfg, *vals in FP_ARMB_K250}
    p(f"The cross-dataset picture is harder than the control, and it is the "
      f"one that matters for adoption. Even after adapting on the largest "
      f"annotation budget in the study, a detector taken from lohi_weld to "
      f"kaggle_v2 recovers a recall of "
      f"{q(fp10[('lohi_weld', 'kaggle_v2', 'RT-DETR-l')], 3)} and "
      f"{q(fp10[('lohi_weld', 'kaggle_v2', 'YOLO11s')], 3)} at one false "
      f"positive box per image, and one taken from kaggle_v2 to visual "
      f"recovers {q(fp10[('kaggle_v2', 'visual', 'RT-DETR-l')], 3)} and "
      f"{q(fp10[('kaggle_v2', 'visual', 'YOLO11s')], 3)}. Only the two "
      "twoclass targets and lohi_weld reach a recall "
      "that an inspection process could use, and twoclass carries a weak "
      "in-domain reference. A practitioner adopting a public visible-light "
      f"weld checkpoint, adapting it on the {H['k_real'][0]} to "
      f"{H['k_real'][1]} labelled target images of the largest budget, and "
      "requiring no more than one false alarm per image should expect to miss "
      "most defects on a hard target.")

    # ------------------------------------------------------------------- 4.8
    h("4.8 Calibration under shift", 2)

    p("An acceptance threshold is set on a detector's confidence. If the "
      "confidence stops meaning what it meant in the source domain, the "
      "threshold stops meaning what it meant too. We therefore report "
      "expected calibration error under shift, with reliability diagrams. "
      "The internal control is the only place this can be seen cleanly, "
      "because everything except the acquisition is held constant.",
      bold_lead="Draft. ")

    p(f"Mean expected calibration error rises from {q(ECE_CONTROL[0], 3)} in "
      f"domain to {q(ECE_CONTROL[1], 3)} across the acquisition boundary, a "
      f"factor of about {q(ECE_CONTROL[1] / ECE_CONTROL[0], 1)}, and every "
      f"configuration moves the same way. The fine-tuned models of Section "
      f"4.5 are the worst calibrated in the study, at a mean of "
      f"{q(ECE_ARMB, 3)} over {ARMB_CELLS} cells. That is consistent with the "
      f"head reinitialisation described there. A head fitted on a handful of "
      f"images scores the target's defective images low, and at the smallest "
      f"budgets it can collapse onto one wrong class entirely. A practitioner "
      f"cannot threshold such a model safely, and for an inspection claim "
      f"that matters more than the mean average precision does. "
      f"{T('ece')} gives the control's values per configuration.")

    ece_tab = control_ece()
    table(
        ["Configuration", "ECE in domain", "ECE cross acquisition"],
        [[cfg, q(v[0], 3), q(v[1], 3)] for cfg, v in ece_tab.items()])
    tcap("ece", "Expected calibration error inside the internal "
            "acquisition control, six evaluations per entry, rounded once "
            "from the unrounded means. Read each cross-acquisition value "
            "against the in-domain value of the same configuration, not "
            "against an absolute standard. On a target where nearly every "
            "image carries a defect, calibration error mostly measures how "
            "far the scores sit below that share. Source: "
            "calibration_arms.csv.")

    fr, y11s = ece_tab["Faster R-CNN"][0], ece_tab["YOLO11s"][0]
    p(f"The comparison that this table supports is within a configuration and "
      f"across the shift, not between configurations. Faster R-CNN's "
      f"in-domain value of {q(fr, 3)} sits against YOLO11s's {q(y11s, 3)}, a "
      f"factor of about {round(y11s / fr)}, and detection confidence scores "
      f"are not comparable across detector families: they come from "
      f"different heads, different losses and different score "
      f"normalisations. A reading across the rows would therefore reflect "
      f"score semantics as much as calibration quality. The claim we make is "
      f"the within-family one, that every configuration degrades under the "
      f"shift, and that is unaffected.")

    rd = reliability_direction()
    ctl_n = rd["under"]["lohictl"][0]
    n_pos, n_neg = BASE_RATE_COUNTS["lohi_weld"]
    il = image_level_facts()
    r_hi = fnum(il["arm_samples"]["lohi_high"]["positive_rate"])
    r_lo = fnum(il["arm_samples"]["lohi_low"]["positive_rate"])
    p(f"The direction of the error is largely forced by the test splits, and "
      f"we say so before reading anything into it. lohi_high's test split is "
      f"{pct(r_hi, 0)} percent defective, so none of the {il['n_high']} "
      f"evaluations on it can be overconfident, and lohi_low's is "
      f"{pct(r_lo, 1)} percent defective. All {ctl_n} control evaluations "
      f"are underconfident, and so are all {ARMB_CELLS} fine-tuned models: "
      f"across the bins of each, the share of images with a defect exceeds "
      f"the score ({F('reliability')}). Faster R-CNN in domain is underconfident "
      f"only marginally. What the bins add is informative. Under the shift "
      f"the scores fall while the share of defective images does not, and "
      f"the calibration error rises with them. For inspection this is the "
      f"costly direction. At every threshold from "
      f"{THRESHOLD_EDGES[0]:.2f} to {THRESHOLD_EDGES[-1]:.2f}, and for every "
      f"configuration, a smaller share of the defective images clears it "
      f"once the camera changes. The error a carried-over threshold produces "
      f"is the missed defect that reaches a customer.")

    figure("reliability",
           f"Reliability inside the internal acquisition "
           f"control, one panel per configuration: "
           + ", ".join(f"({chr(97 + i)}) {FAMILY_CODE[c]}"
                       for i, c in enumerate(RELIABILITY_PANELS))
           + f". Each point pools one "
           f"score bin over the six in-domain evaluations, or over the six "
           f"cross-acquisition evaluations, weighted by image count. Marker "
           f"area grows with the number of images in the bin, and lines join "
           f"bins holding at least {word(RELIABILITY_MIN_LINE)} images. A "
           f"point above the diagonal is underconfident. The dotted line is "
           f"the share of test images with a defect, where a score carrying "
           f"no information would place every point. Under the shift the "
           f"mass moves to lower scores while the share with a defect stays "
           f"near one. Source: reliability_bins_arms.csv.")

    # ------------------------------------------------------------------- 4.9
    h("4.9 Per-class results and statistical tests", 2)

    p("Overall mean average precision is never reported without the per-class "
      "breakdown, because the defect classes are the minority and their "
      "failure is otherwise masked. Section 4.3 is a direct instance: "
      "kaggle_v2's low defect-class value sits beside two verdict classes "
      f"that the same model learns well, and only a per-class table, "
      f"{T('per_class')}, shows it.",
      bold_lead="Draft. ")

    rows = []
    STATE = {"defect": ("yes", ""),
             "verdict": ("no, verdict", "scored, not a defect class"),
             "excluded": ("no, not scored", "excluded from ground truth")}
    for src, cls, ap, sd, boot, n, thin, state in PER_CLASS:
        mark, note_txt = STATE[state]
        rows.append([src, cls, mark,
                     q(ap, 3) if ap is not None else "n/a",
                     q(sd, 3) if sd is not None else "n/a",
                     f"[{q(boot[0], 3)}, {q(boot[1], 3)}]" if boot
                     else "not needed",
                     str(n), "thin" if thin else note_txt])
    table(["Source", "Class", "In defect space", "AP50", "Seed SD",
           "Split-block 95 percent", "Test instances", ""], rows)
    tcap("per_class", "In-domain per-class AP50, YOLO11s, mean over three "
            "seeds, with every value carrying its test instance count. EVERY "
            "CLASS OF EVERY SOURCE IS LISTED, and the third column says which "
            "the harmonised taxonomy retains as a defect. A verdict class is "
            "kept in the ground truth and scored but is not a defect, and an "
            "excluded class is dropped from the ground truth entirely, so it "
            "has no instances and no score. Neither contributes to any "
            "cross-dataset figure. Values apply the protocol's class-agnostic "
            "NMS, so that on a source with one defect class this value and "
            "the box-level mAP50 agree exactly. Cells from the "
            "downloaded-partition models are excluded from these means, 84 "
            "rows in total, because such a model is graded on a partition "
            "that overlaps its own training split. A class is marked thin "
            "below 50 test instances. The other three configurations are in "
            "the released table. Source: per_class_ap.md.")

    acq_medians = [GEOMETRY["lohi_acquisitions"][a]["median_box_area_percent"]
                   for a in ("high", "low")]
    p(f"weld_surface_v2's values in this table are not comparable with the "
      f"others and should not be read as evidence that weld defect detection "
      f"is close to solved. That source annotates a frame-sized quality "
      f"verdict, at {q(BOX_GEOMETRY['weld_surface_v2'][0], 2)} boxes per "
      f"image and a median box area of "
      f"{q(BOX_GEOMETRY['weld_surface_v2'][1], 1)} percent of the frame, "
      f"against lohi_weld's median of {q(min(acq_medians), 2)} to "
      f"{q(max(acq_medians), 2)} percent. A "
      "box covering a fifth of the image is trivially easy to localise, so "
      "its AP50 is close to an image-level decision scored with a loose "
      "spatial threshold rather than a localisation score. That is the same "
      "property that excludes the source from the box-level matrix in Section "
      "4.3, and it is why annotation protocol is treated here as an axis of "
      "domain shift rather than as a nuisance.")

    tc_thin = {c: (n, sd, b) for s, c, _ap, sd, b, n, _th, _st in PER_CLASS
               if s == "twoclass"}
    geo_n, geo_sd, geo_b = tc_thin["Geometric defect"]
    nf_n, nf_sd, nf_b = tc_thin["Non-fusion defect"]
    n_thin = sum(1 for row in PER_CLASS if row[6])
    p(f"A thin class is quoted with a split-block bootstrap interval and never "
      f"with seed dispersion alone. The two answer different questions. Seed "
      f"dispersion measures training variance and says nothing about a small "
      f"correlated test sample. For twoclass Geometric defect at {geo_n} test "
      f"instances the YOLO11s seed standard deviation is {q(geo_sd, 3)} while "
      f"the split-block interval is [{q(geo_b[0], 3)}, {q(geo_b[1], 3)}], "
      f"{q(geo_b[1] - geo_b[0], 3)} wide. For Non-fusion defect at {nf_n} "
      f"instances "
      f"the same configuration gives {q(nf_sd, 3)} and "
      f"[{q(nf_b[0], 3)}, {q(nf_b[1], 3)}]. The {word(n_thin)} thin classes "
      f"are concentrated in the two smallest sources.")

    si, sb = STATS["image_ap"], STATS["box_map50"]
    fi, fb = si["primary"], sb["primary"]
    ai, ab = si["secondary"]["anova"], sb["secondary"]["anova"]
    rank_gap = max(pr["rank_difference"] for pr in fi["posthoc"]["pairs"])
    p(f"The pre-registered omnibus test is a Friedman test across the four "
      f"configurations, blocked on evaluation cell, with a Nemenyi post hoc, "
      f"and a repeated-measures ANOVA with Tukey reported alongside for "
      f"comparability. Neither is significant. The Friedman gives chi-squared "
      f"{q_stored(fi['chi2'], 2)} on {fi['df']} degrees of freedom and "
      f"p = {q(fi['p_value'], 3)} over {si['n_blocks']} blocks at image "
      f"level, and chi-squared {q_stored(fb['chi2'], 2)} and "
      f"p = {q(fb['p_value'], 3)} over {sb['n_blocks']} blocks at box level. "
      f"The ANOVA agrees: F = {q_stored(ai['f_statistic'], 3)} on "
      f"{ai['df_treatment']} and {ai['df_error']} degrees of freedom with "
      f"p = {q(ai['p_value'], 3)} and partial eta squared "
      f"{q_stored(ai['partial_eta_squared'], 3)} at image level, and "
      f"F = {q_stored(ab['f_statistic'], 3)} on {ab['df_treatment']} and "
      f"{ab['df_error']} with p = {q(ab['p_value'], 3)} at box level. The "
      f"Nemenyi post hoc was computed and separates no pair: at image level "
      f"the critical difference is "
      f"{q_stored(fi['posthoc']['critical_difference'], 3)} and the largest "
      f"observed rank gap is {q_stored(rank_gap, 3)}. The image-level test "
      f"ranks the four configurations' average precision within each source "
      f"and target cell. The chance level of that estimator is set by the "
      f"target's test split, which all four share, so it cannot reorder "
      f"them.")

    pa = STATS_ARMS["primary_blocked_on_cell"]
    p(f"The box-level test has {sb['n_blocks']} blocks against the image "
      f"level's {si['n_blocks']}, because only localisation-protocol pairs are "
      f"defined there. It therefore has materially less power, and its null "
      f"is correspondingly less informative. The same caution applies to the "
      f"arms, where a Friedman over {pa['n_blocks']} blocks returns "
      f"p = {q(pa['primary']['p_value'], 3)}. We read these smaller nulls as "
      f"uninformative rather than as confirmation that the configurations "
      f"agree.")

    ps_i = si["sensitivity_cell_by_seed"]["primary"]["p_value"]
    ps_b = sb["sensitivity_cell_by_seed"]["primary"]["p_value"]
    p(f"The pre-registered sensitivity analysis blocks on cell and seed, "
      f"which triples the block count, and agrees at p = {q(ps_i, 4)} and "
      f"p = {q(ps_b, 4)}. We report {q(ps_i, 4)} rather than rounding it "
      f"away. It is the closer of the two and is what one would expect if "
      f"there were a small real configuration effect that {si['n_blocks']} "
      f"blocks cannot resolve. It does not license a claim, and the "
      f"pre-registered primary result stands.")

    p("A null omnibus result does not bound the size of the transfer "
      "collapse. The two answer different questions. The omnibus asks whether "
      "the four configurations differ from one another, and the answer is "
      "that they are not distinguishable on these cells. The claim of this "
      "paper is about the magnitude of the transfer, which is why every "
      f"box-level ratio in {T('box_ratio')} and every AUROC in {T('auroc')} carries a "
      "split-block bootstrap interval.")

    note("If a mean rank table goes into the manuscript it must carry the "
         "direction sentence: rankdata ranks ascending, so rank 1 is the "
         "smallest value, and every metric here is one where larger is "
         "better. A HIGHER mean rank is therefore the better configuration. "
         "Image-level ranks are YOLO11n 3.067, Faster R-CNN 2.400, RT-DETR-l "
         "2.333, YOLO11s 2.200. An earlier draft had the direction backwards "
         "and would have named the worst configuration as the best.")

    # ------------------------------------------------------------------ 4.10
    h("4.10 Reproducibility of the reported numbers", 2)

    p("Every result reported above was produced under recorded per-cell code "
      "hashes, and those hashes are released with the cells. We do not claim "
      "that every cell can be re-evaluated from the released head of the "
      "code. One guarded evaluator file was corrected once between cell "
      "generations, and that intermediate revision was not retained, so the "
      "60 oldest cells pin a version of it that no longer exists. Their "
      "results are unaffected, because the corrected branch executes only for "
      "a cell kind that none of them uses. The three files whose movement "
      "would change a reported number, the configuration, the metric "
      "implementation and the dataset loader, have never moved across any "
      "generation. We prefer this statement to a blanket claim of "
      "reproducibility, because the blanket claim would be false.",
      bold_lead="Draft. ")

    p("The released artefact for the partitioning is the frozen split files "
      "and their sha256 hashes, and not the solver that produced them. Four "
      "of the five datasets reached the solver's wall-clock limit rather than "
      "proving optimality, so the returned partition depends on machine "
      "timing and a re-run returns a different but equally valid partition. "
      "Readers should verify the released splits against the published hashes "
      "rather than regenerate them, and numbers from two different partitions "
      "should not be compared. Each split file identifies its images by their "
      "original filenames, so the partitions can be reconstructed exactly "
      "against a copy obtained from the original authors.")

    # --------------------------------------------------------------- 5 seed
    d.add_page_break()
    h("5. Discussion", 1)

    h("5.1 Comparison with the one published visible-light external test", 2)

    p(f"One prior study reports an external test of a visible-light weld "
      f"detector, and its external target is the same public dataset we use as "
      f"a source. {ct('mendieta2025')} trained YOLOv7 on "
      f"{MENDIETA['train_images']:,} shielded metal arc welding images and "
      f"reported {MENDIETA['in_domain_precision']} percent precision and "
      f"{MENDIETA['in_domain_map50']} percent mAP@0.5 in domain. On an "
      f"external set drawn from a public Kaggle weld defect project they "
      f"reported {MENDIETA['ext_precision']} percent precision, "
      f"{MENDIETA['ext_recall']} percent recall and {MENDIETA['ext_map50']} "
      f"percent mAP@0.5. We establish from their reference list that this "
      f"external set is the Kaggle project that supplies our kaggle_v2, and we "
      f"confirmed the identity of our copy against the live dataset by total "
      f"file count and total byte count, which agree exactly.",
      bold_lead="Draft. ")

    p(f"The comparison must be stated with the sample size on both sides, "
      f"because the two sides differ by a factor of almost fifty. Their "
      f"external test used {MENDIETA['ext_images_word']} images. Those "
      f"{MENDIETA['ext_images_word']} were retained after filtering to similar "
      f"plane shots and excluding angled views, other welding processes and "
      f"other joint types, then cropped to the weld bead and rescaled to the "
      f"training input size. Our figure is measured on the full audited "
      f"kaggle_v2 test split of {OUR_KAGGLE_TEST_IMAGES} images carrying "
      f"{OUR_KAGGLE_TEST_BOXES} defect boxes, with no curation and no "
      f"cropping. {T('mendieta')} sets the two side by side.")

    lk_box = [v[0] for v in BOX_OFF[("lohi_weld", "kaggle_v2")].values()]
    table(
        ["", f"{ct('mendieta2025')}, external", "This study, in domain",
         "This study, cross dataset"],
        [["Test images",
          f"{MENDIETA['ext_images']}, curated and cropped",
          f"{OUR_KAGGLE_TEST_IMAGES}, audited",
          f"{OUR_KAGGLE_TEST_IMAGES}, audited"],
         ["mAP@0.5", f"{MENDIETA['ext_map50'] / 100:.3f}",
          f"{q(BOX_DIAG_YOLO11S['kaggle_v2'], 3)} (YOLO11s), "
          f"{q(BOX_DIAG_MEAN['kaggle_v2'], 3)} mean",
          span(lk_box, q, 3)],
         ["Training source", f"{MENDIETA['train_images']:,} SMAW images",
          "kaggle_v2 itself", "lohi_weld"]])
    tcap("mendieta", "The one available like-for-like comparison on a "
            "visible-light weld dataset. The three columns are not the same "
            "quantity and the table exists to show that. Their number is a "
            "cross-dataset result on three hand-selected images. Our "
            f"cross-dataset result on the same dataset, measured on "
            f"{OUR_KAGGLE_TEST_IMAGES} uncurated images, is "
            f"{q(max(lk_box), 3)} or below.")

    p(f"The comparison cuts in an unexpected direction and we report it as it "
      f"falls. Their external mAP@0.5 of {MENDIETA['ext_map50'] / 100:.3f}, "
      f"obtained by a model that had never seen this dataset, is more than "
      f"double the {q(BOX_DIAG_YOLO11S['kaggle_v2'], 3)} that our best "
      f"configuration achieves on the same dataset after training on it. Two "
      f"readings are available and we prefer the first. Either "
      f"{MENDIETA['ext_images_word']} images cropped to the bead and filtered for "
      f"viewpoint are not a measurement of cross-dataset performance, or "
      f"kaggle_v2 is easier than our audited evaluation makes it look. The "
      f"evidence supports the first. Section 4.3 shows that a model trained on "
      f"the shipped kaggle_v2 partition, which overlaps "
      f"{pct(AUDIT_OVERLAP['kaggle_v2'][0] / AUDIT_OVERLAP['kaggle_v2'][1], 0)} "
      f"percent of our test images, reaches {q(DOWNLOADED_KAGGLE_BOX, 3)} on "
      f"that split, so the annotations are learnable. "
      f"What collapses is generalisation, and a {MENDIETA['ext_images_word']} image "
      f"sample cannot observe a collapse.")

    p(f"This is why we regard their result as motivating this study rather "
      f"than anticipating it, as Section 1 sets out. The one published "
      f"visible-light external test suggests a fall from "
      f"{MENDIETA['in_domain_map50']} to {MENDIETA['ext_map50']} mAP@0.5, and "
      f"it observed that fall under conditions chosen to favour transfer. What "
      f"it could not do, on {MENDIETA['ext_images_word']} images, is measure it. "
      f"Our contribution is the measurement on a proper target split, and the "
      f"measurement is substantially worse than the anecdote suggested.")

    # ------------------------------------------------------------------- 5.2
    h("5.2 Which axes of shift these results separate, and which they do not",
      2)

    p("A cross-dataset transfer matrix confounds several things at once, and "
      "the central limitation of this study is that it separates only one of "
      "them. Two public weld datasets differ in illumination and sensor, in "
      "the physical scale of the defects photographed, in the vocabulary each "
      "project chose for its classes, in how many images each contains, and "
      "in what its annotators decided a box means. When transfer between them "
      "fails, the matrix alone cannot say which of those caused it.",
      bold_lead="Draft. ")

    p(f"The internal acquisition control of Section 4.2 isolates exactly one "
      f"axis. The annotation protocol, the annotators, the class list, the "
      f"production line and the defect vocabulary are all held constant, and "
      f"only the camera changes. It therefore measures visual shift with the "
      f"label concept pinned, which is the one quantity the cross-dataset "
      f"matrix cannot produce. Even that isolation is imperfect and we report "
      f"the imperfection: the defect mix differs between the two "
      f"acquisitions, so part of the {H['control_cost'][0]} to "
      f"{H['control_cost'][1]} percent cost is prior shift rather than sensor "
      f"shift.")

    bound_par = (
        f"Reading the two against each other bounds the problem without "
        f"resolving it. Changing only the camera retains {H['control'][0]} to "
        f"{H['control'][1]} percent of in-domain localisation performance. "
        f"Changing the dataset retains {H['box'][0]} to {H['box'][1]} percent. "
        f"Visual shift alone is therefore sufficient to cause a large collapse "
        f"and is not sufficient to explain the larger one. What remains is "
        f"attributable to the other axes jointly, and this study does not "
        f"apportion it among them. We say so rather than name a cause.")
    p(bound_par)
    require_phrases("5.2 bound paragraph", bound_par, [
        f"{H['control'][0]} to {H['control'][1]} percent",
        f"{H['box'][0]} to {H['box'][1]} percent"])

    p(f"Annotation protocol is treated here as a named axis of that shift "
      f"rather than as a defect to be repaired, and that is a substantive "
      f"choice rather than a convenience. Where prior work combines detection "
      f"datasets it generally treats annotation inconsistency as an error to "
      f"be corrected. Label transfer projects annotations from several source "
      f"datasets into a target label space so that a detector can be trained "
      f"across them {cp('kennerley2025')}. Re-annotation rebuilds the labels "
      f"themselves, and has found that the effect on measured performance "
      f"depends on the category {cp('ma2022a')} and that detectors had been "
      f"penalised by imprecise original masks {cp('singh2025')}. On an "
      f"underwater benchmark, re-annotation showed that up to 38 percent of "
      f"apparent false positives were correct detections scored against "
      f"faulty ground truth {cp('awad2026')}. A recent survey gives a "
      f"standardised taxonomy of annotation error types in detection datasets "
      f"and of the methods that detect them {cp('hussain2026')}. Each of "
      f"those is the right response when the disagreement is a mistake.",
      bold_lead="Draft. ")

    wr_box, loc_box = protocol_geometry()
    wr_ap, loc_ap = protocol_indomain_ap()
    p(f"It is the wrong response here. A frame-sized quality verdict and a "
      f"localised defect box are not two attempts at the same annotation, one "
      f"of them careless. They are two different inspection tasks that happen "
      f"to share a file format. The whole-region source carries "
      f"{q(wr_box[0], 2)} boxes per image, each covering a median "
      f"{q(wr_box[1], 2)} percent of the frame. The {word(loc_box['n'])} "
      f"localisation sources carry {q(loc_box['per_image'][0], 2)} to "
      f"{q(loc_box['per_image'][1], 2)} boxes per image, covering a median "
      f"{q(loc_box['area'][0], 2)} to {q(loc_box['area'][1], 2)} percent. An "
      f"intersection over union is therefore computed on objects that differ "
      f"by at least a factor of {PROTOCOL_AREA_FOLD} in size, so one nominal "
      f"threshold admits a different kind of agreement on each.")

    p(f"In-domain accuracy does not separate the two protocols, and saying so "
      f"is part of the argument rather than a concession to it. Mean "
      f"defect-class AP50, trained and tested in domain and averaged over the "
      f"{word(len(CFGS))} configurations, is {q(wr_ap, 3)} on the "
      f"whole-region source and spans {q(loc_ap[0], 3)} to {q(loc_ap[1], 3)} "
      f"across the "
      f"localisation sources. The spread inside the localisation protocol is "
      f"wider than the gap between its strongest source and the whole-region "
      f"one. What the protocol changes is not how hard the task is. It is "
      f"what counts as a correct answer. Harmonising the two into a single "
      f"label space would produce a number, and the number would measure the "
      f"harmonisation.")

    p(f"Merging heterogeneous sources into one taxonomy is the default move "
      f"in adjacent fields, and it is usually reasonable. SteelDefectX merges "
      f"four public steel surface datasets into twenty-five categories and "
      f"evaluates zero-shot transfer out of the merged set, without raising "
      f"the annotation protocol question {cp('zhao2026')}. We do not read "
      f"that as an oversight. Steel surface annotation is far more "
      f"homogeneous than weld annotation: nothing in those four sources is a "
      f"whole-region acceptance verdict sitting beside a localised defect. "
      f"Welds are the case where the manoeuvre does not hold, which is why we "
      f"measure the protocol difference instead of harmonising it away.")

    p(f"That decision is what forces the two-granularity design. An "
      f"image-level defect-present decision is invariant to how a box was "
      f"drawn, so it is comparable across all five sources. A box-level score "
      f"is not, so it is restricted to the sources that share a localisation "
      f"protocol.")

    # The asymmetry comparison with Chakraborty et al. Every count below is
    # derived: the control's from its interval CSV, matched by configuration
    # and seed, and the cross-dataset pair's from Table 4.3.
    fwd = BOX_OFF[("kaggle_v2", "lohi_weld")]
    rev = BOX_OFF[("lohi_weld", "kaggle_v2")]
    same_way = sum(fwd[c][1] > rev[c][1] for c in CFGS)
    separated = sum(fwd[c][2] > rev[c][3] or rev[c][2] > fwd[c][3]
                    for c in CFGS)
    p(f"The same two instruments speak to the asymmetry that "
      f"{ct('chakraborty2026')} report in cross-dataset detection, and they "
      f"bound what this study can say about it. Their strongest claim is "
      f"directional: the most severe breakdowns occur from setting-specific "
      f"sources to setting-agnostic targets. Every source and target here is "
      f"setting-specific, so this study cannot test that claim and does not "
      f"claim agreement with it. Within the specific-to-specific regime, "
      f"where they report transfer to be relatively stable, the weld matrix "
      f"is not stable at all: it retains {H['box'][0]} to {H['box'][1]} "
      f"percent of in-domain localisation. A category as broad as "
      f"setting-specific evidently spans shifts of very different size when "
      f"annotation protocol and vocabulary differ inside it.")

    p(f"Asymmetry itself is visible where it can be measured cleanly. Inside "
      f"LoHi-WELD, training on the low-resolution acquisition and testing on "
      f"the high-resolution one retains "
      f"{q(CONTROL_T[('lohi_low', 'lohi_high')], 3)} of in-domain "
      f"localisation, against {q(CONTROL_T[('lohi_high', 'lohi_low')], 3)} "
      f"the other way. Matched by "
      f"configuration and seed, the low-to-high ratio is the larger in "
      f"{word(ci['low_to_high_larger'])} of {word(ci['matched'])} cells, and "
      f"the two per-cell intervals are disjoint in {word(ci['disjoint'])}. "
      f"Because the defect mix differs between the acquisitions, we do not "
      f"attribute the direction to the sensor alone. Across datasets, the one "
      f"localisation pair measured in both directions leans the same way in "
      f"{word(same_way)} of {word(len(CFGS))} configurations, kaggle_v2 to "
      f"lohi_weld retaining {q(min(v[1] for v in fwd.values()), 3)} to "
      f"{q(max(v[1] for v in fwd.values()), 3)} against "
      f"{q(min(v[1] for v in rev.values()), 3)} to "
      f"{q(max(v[1] for v in rev.values()), 3)} in reverse, but the intervals "
      f"separate in {word(separated) if separated else 'none'} of them, so "
      f"there it is suggested rather than established.")

    # ------------------------------------------------------------------- 5.3
    h("5.3 What a practitioner should take from this", 2)

    p("The results support four pieces of guidance, each drawn from a "
      "measurement rather than asserted, and the sampling design suggests a "
      "fifth that we did not test.", bold_lead="Draft. ")

    validate_par = (
        f"Do not treat a public visible-light weld checkpoint as either a "
        f"screening filter or a localiser before validating it on your own "
        f"line. Into the one balanced target, its decision on whether a "
        f"defect is present could not be told from chance, and it retained "
        f"{H['locate'][0]} to {H['locate'][1]} percent of in-domain "
        f"localisation.")
    p(validate_par)
    require_phrases("5.3 validation paragraph", validate_par, [
        f"{H['locate'][0]} to {H['locate'][1]} percent"])

    lk_fp = [vals[-1] for s, t, cfg, *vals in FP_ARMB_K250
             if (s, t) == ("lohi_weld", "kaggle_v2")]
    p(f"Specify the operating point before believing a mean average "
      f"precision. Inside the internal control, tolerating one false positive "
      f"box per image finds about half of the defects in domain and "
      f"{span(RECALL_CROSS[-1], pct, 0)} percent of them after a change of "
      f"camera, and at {FP_BUDGETS[0]} false positives per image "
      f"cross-acquisition recall is {span(RECALL_CROSS[0], q, 3)}. Across "
      f"datasets it is harder still. Even after adapting on the largest "
      f"annotation budget in this study, a detector taken from lohi_weld to "
      f"kaggle_v2 recovers a recall of {span(lk_fp, q, 3)} at one false "
      f"positive box per image. A figure reported as mAP50 does not expose "
      f"that, and an inspection requirement is not written in mAP50.")

    td = TWOCLASS_DRAWS
    budget_par = (
        f"Budget for annotation variance rather than for a single few-shot "
        f"number. Into the smallest target, three attempts at ten requested "
        f"labelled images drew {min(td['k10_images'])} to "
        f"{max(td['k10_images'])} images, and their scores varied between "
        f"{q(td['k10_spread'][0], 1)} and {q(td['k10_spread'][1], 1)} times "
        f"across the {word(td['combos'])} source and configuration "
        f"combinations. With the training split exhausted and identical "
        f"images in all three runs, training variance alone still spread "
        f"them up to {q(td['top_spread'], 1)} times. A practitioner who "
        f"annotates ten images and measures the result has measured one "
        f"attempt.")
    p(budget_par)
    require_phrases("5.3 budget paragraph", budget_par, [
        f"drew {min(td['k10_images'])} to {max(td['k10_images'])} images",
        f"between {q(td['k10_spread'][0], 1)} and "
        f"{q(td['k10_spread'][1], 1)} times",
        f"up to {q(td['top_spread'], 1)} times"])

    p(f"Do not threshold a few-shot fine-tuned model on its own confidence. "
      f"Those models are the worst calibrated in this study, at a mean "
      f"expected calibration error of {q(ECE_ARMB, 3)}, and at the smallest "
      f"budgets the reinitialised detection head can collapse onto a single "
      f"wrong class while still emitting its full detection cap. Calibration "
      f"degrades under a change of camera alone, from {q(ECE_CONTROL[0], 3)} "
      f"to {q(ECE_CONTROL[1], 3)}. The scores fall under that shift, so a "
      f"threshold carried over unchanged flags fewer of the defective images. "
      f"A confidence score that was meaningful in the source domain should be "
      f"re-established on target data before it is used to set an acceptance "
      f"threshold.")

    p("A fifth recommendation follows from the sampling design rather than "
      "from a measurement, and we mark it as such. Because a scene is the "
      "unit of independent information, and because adjacent frames of one "
      "weld carry almost the same evidence, we expect a budget spread over "
      "more scenes to be worth more than the same number of images "
      "concentrated in fewer. We did not test that. Comparing a "
      "scene-diverse budget against an image-dense one at equal annotation "
      "cost is a clean experiment and it is not one this study ran.")

    # ------------------------------------------------------------------- 5.4
    h("5.4 What the negative and null results license", 2)

    p("Two results in this study are negative or null, and both constrain "
      "what a practitioner should expect. Neither licenses a general claim, "
      "and the distinction is worth stating once.", bold_lead="Draft. ")

    pl = image_level_facts()["pool"]
    p(f"In AUROC, pooling beat the better of its two members in "
      f"{pl['better_wins']} of {pl['n']} cells, and in {pl['seed_better']} of "
      f"{pl['seed_n']} with seeds averaged first. The better member is chosen "
      f"on the target's own test split, which makes it an optimistic "
      f"comparator, and a pool trains on more images than either member, so "
      f"this is not a comparison at a fixed training budget. That is "
      f"evidence about two-source "
      f"pooling over the three public visible-light sources large enough to "
      f"train on. It is not evidence that source diversity never helps. A "
      f"leave-one-out design over three sources leaves two in every pool, "
      f"which is close to the smallest pool for which the question is even "
      f"meaningful. LoHi-WELD's authors report a similar pattern inside their "
      f"own dataset, where merging its two acquisitions into one training set "
      f"brought no performance gain {cp('block2024')}. The useful reading is "
      f"that a practitioner should not assume that adding a second public "
      f"dataset will improve transfer, and should measure it.")

    p("The omnibus comparison across the four configurations is not "
      "significant, and it bounds nothing about the transfer collapse. The "
      "two tests answer different questions. The omnibus asks whether these "
      "four detectors differ from one another on these cells, and the answer "
      "is that they are not distinguishable. The paper's claim is about the "
      "magnitude of the collapse, which is why every transfer ratio carries a "
      "split-block interval rather than a significance verdict. A reader who "
      "sees only that nothing was significant would draw the opposite "
      "conclusion from the one the intervals support.")

    p("The practical consequence of the null is the more useful half. If four "
      "configurations spanning three detector families, including a "
      "transformer and a two-stage detector, are indistinguishable on "
      "cross-dataset transfer, then changing architecture is not the lever. "
      "Nothing here suggests a practitioner facing a transfer problem should "
      "reach for a different detector first.")

    # ------------------------------------------------------------------- 5.5
    h("5.5 Limitations", 2)

    p("Seven limitations apply to the study as a whole, distinct from the "
      "per-result limits already stated in Section 4.", bold_lead="Draft. ")

    p("The benchmark holds five visible-light sources, and only two of them "
      "use a localisation protocol and are large enough to train on. Every "
      "box-level transfer statement therefore rests on two training rows. "
      "That is a finding about the available public data as much as a "
      "limitation of the design, and it is the reason the image-level matrix "
      "carries five sources while the box-level matrix carries two. A larger "
      "conclusion about visible-light weld detection in general would need "
      "public datasets that do not yet exist.")

    if TUNED_ON == "none":
        p("One recipe per detector family was used throughout, fixed before "
          "any model was trained and then frozen. This buys the central "
          "argument, because a matrix tuned per dataset would confound "
          "transfer with tuning effort. It also means the study says nothing "
          "about whether careful per-target tuning would recover more than "
          "the annotation budgets explored here.")
    else:
        p("One hyperparameter set was used throughout, tuned once on one "
          "source and then frozen. This buys the central argument, because a "
          "per-dataset-tuned matrix would confound transfer with tuning "
          "effort, and it costs some in-domain performance on every source "
          "except the one tuned on. It also means the study says nothing "
          "about whether careful per-target tuning would recover more than "
          "the annotation budgets explored here.")
    confirm_note("tuning")

    p(f"The three seeds per cell bound training variance and not sampling "
      f"variance, except in the label cost curve where the two were "
      f"deliberately tied and therefore cannot be separated. At the "
      f"exhausted twoclass budget the three draws share their images, so "
      f"their spread of up to {q(TWOCLASS_DRAWS['top_spread'], 1)} times is "
      f"training variance alone. Where a result rests on a small target, the "
      f"split-block bootstrap rather than the seed dispersion is the "
      f"interval we quote, for the reason given in Section 4.9.")

    sm = image_level_facts()["samples"]
    p(f"The image-level test splits hold few independent scenes, as Section "
      f"4.4 counts them. kaggle_v2's defective test images occupy an "
      f"effective {q(fnum(sm['kaggle_v2']['pos_effective_blocks']), 1)} "
      f"blocks and weld_surface_v2's defect-free ones "
      f"{q(fnum(sm['weld_surface_v2']['neg_effective_blocks']), 1)}. The "
      f"evaluator's rule of {MIN_NEGATIVES} negative images counts images and "
      f"does not see this.")

    p("kaggle_v2's labels follow its source collections, so its in-domain "
      "image-level score may reflect collection appearance rather than defect "
      "recognition.")

    p(f"The {SCHEDULE_COUNTS['n']} exploratory schedule controls ran on a "
      f"workstation under a different software stack. They are compared only "
      f"with each other.")

    p("Finally, this study is confined to visible-light inspection. "
      "Radiographic weld datasets exist and cross-domain work in that "
      "modality is published, and we deliberately do not mix the two. A "
      "radiographic arm run under the same leakage-audited protocol is a "
      "natural extension, and the size of the gap between it and the "
      "visible-light result reported here is itself an open question. We "
      "make no claim about radiography from these data.")

    p("One further direction is worth naming because it is the question a "
      "practitioner actually asks, and because this study is not the right "
      "instrument for it. Before annotating anything, a practitioner would like "
      "to estimate deployment risk from unlabelled target images alone, by "
      "measuring some distance between their own images and the source the "
      "checkpoint was trained on and reading a likely transfer loss off it. We "
      "considered reporting such a correlation here and decided against it. A "
      "feature distance between two weld datasets captures appearance, scene "
      "content, object scale and dataset composition at once, and the pairs "
      "available here differ in annotation protocol, size, class vocabulary "
      "and appearance simultaneously. A strong correlation could therefore not "
      "be read as visual distance predicting the gap, which is the only claim "
      "it would exist to support, and twelve off-diagonal cells would support "
      "detecting only a very strong one. Doing it properly needs the visual "
      "axis isolated from the annotation and composition axes, which in turn "
      "needs more independently collected public visible-light weld datasets "
      "than currently exist. We regard that as the natural next study rather "
      "than an omission from this one.")

    # ------------------------------------------------------------------ 6
    h("6. Conclusion", 1)

    concl_1 = (
        "This study measured something the visible-light weld literature has "
        "so far assumed by omission: that a detector trained on one public "
        "dataset will work on another. Measured on audited partitions, it "
        "does not, and the loss is not one effect. A change of camera alone, "
        "with everything else held constant, costs more than half of "
        "localisation performance. A change of dataset costs more than nine "
        "tenths of it. Whether a defect is present could not be shown to "
        "survive either. On the one balanced target no foreign detector's "
        "decision is distinguishable from chance, and the public test splits "
        "hold too few independent scenes to say more.")
    p(concl_1)

    td = TWOCLASS_DRAWS
    concl_2 = (
        f"For a practitioner the consequences are concrete. Treat a public "
        f"checkpoint as neither a screening filter nor a localiser until it "
        f"has been validated on your own line. Write the acceptance "
        f"requirement as recall at a tolerated false alarm rate, and measure "
        f"it on your own images before trusting any published mean average "
        f"precision, including the ones in this paper. Audit a public dataset "
        f"before reporting a number on it: for duplicates, for scenes that "
        f"straddle its partitions, and for collections that predict its "
        f"labels. Two of the three sources tested here moved when that was "
        f"done, and one source's labels can be read off its filenames. And do "
        f"not plan annotation around a single few-shot result: three attempts "
        f"at ten requested labelled images varied between "
        f"{q(td['k10_spread'][0], 1)} and {q(td['k10_spread'][1], 1)} times on "
        f"the smallest target.")
    p(concl_2)
    require_phrases("6 practitioner paragraph", concl_2, [
        f"between {q(td['k10_spread'][0], 1)} and "
        f"{q(td['k10_spread'][1], 1)} times"])

    held = ("are released with this paper" if REPO_LIVE else
            "will be released on acceptance")
    p(f"What did not exist before is a leakage-audited baseline. The frozen "
      f"splits, their hashes, the harmonised mappings and the per-cell "
      f"results {held}, so that a new detector, a new adaptation method or a "
      f"new public dataset can be measured on the same partitions and against "
      f"the same matrix, rather than against a held-out split of its own "
      f"training data. That comparison is what this literature has been "
      f"missing, and it is the one we would most like to see run against "
      f"these numbers.")
    note("The conclusion is deliberately not a summary of the abstract, and "
         "the future-work sentence on estimating deployment risk from "
         "unlabelled images stays in 5.5 rather than being repeated here. "
         "'More than half' and 'more than nine tenths' are words rather than "
         "numbers, so run_checks asserts both against the tables.")

    # ------------------------------------------------------------ appendix A
    d.add_page_break()
    h("Appendix A. The complete class mapping", 1)
    p(f"{T('class_map')} gives every class of every source, the name used "
      f"for it here and what each granularity does with it.")
    C = frozen_config()
    APPX_REASON = {
        ("weld_surface_v2", "Normal Weld-NW"):
            "An acceptance verdict rather than a defect; every image carries "
            "exactly one class, so an image labelled Normal Weld is a "
            "negative.",
        ("kaggle_v2", "Bad Weld"):
            f"A quality verdict on a region rather than a defect type, so the "
            f"{KAGGLE_BADWELD_ONLY} of {SOURCES['kaggle_v2'][1]:,} images that "
            f"carry it without a Defect box are negatives.",
        ("kaggle_v2", "Good Weld"):
            "A quality verdict marking an acceptable region, not a defect "
            "type.",
        ("visual", "Welding Line"):
            "Scene furniture; retaining it would inflate performance on the "
            "one target where it occurs.",
        ("visual", "workpiece"):
            "Scene furniture that covers most of the frame, excluded for the "
            "same reason.",
    }

    def appx_reason(tag, raw, state, protocol):
        if (tag, raw) in APPX_REASON:
            return APPX_REASON[(tag, raw)]
        if protocol == "whole_region":
            return ("A defect type drawn as a frame-sized verdict box, so it "
                    "enters the image-level decision only; the suffix is the "
                    "dataset's label code and is stripped.")
        others = len(C.defect_classes(tag)) - 1
        if others:
            return (f"A localised defect type, collapsed with the source's "
                    f"{word(others)} other defect "
                    f"{'class' if others == 1 else 'classes'} into the single "
                    f"defect label.")
        return "The source's only defect class, which becomes the defect label."

    rows = []
    for tag, d_ in C.DATASETS.items():
        for raw in d_["classes"]:
            state = ("excluded" if raw in d_["excluded"] else
                     "verdict" if raw in d_["non_defect"] else "defect")
            if state == "excluded":
                box, img = "dropped from the ground truth", "not scored"
            elif state == "verdict":
                box, img = ("not a defect; scored per class in domain",
                            "does not make an image positive")
            elif d_["protocol"] == "whole_region":
                box, img = "none; whole-region source", "positive"
            else:
                box, img = "defect", "positive"
            rows.append([tag, raw, shipped_to_vocab(raw), box, img,
                         appx_reason(tag, raw, state, d_["protocol"])])
    table(["Source", "Class as shipped", "Name used here", "Box level",
           "Image level", "Justification"], rows)
    tcap("class_map", f"Every class of every source, {len(rows)} rows, one "
            f"per mapping decision, generated when this document is built from "
            f"the harmonisation table in grid/config.py rather than typed. At "
            f"box level every defect class collapses to one label and matching "
            f"is class agnostic, and a whole-region source enters the "
            f"image-level decision only. An image is positive when it carries "
            f"at least one box of a class marked positive. LoHi-WELD's boxes "
            f"mark surface features on accepted parts as well as rejected "
            f"ones, so for that source a positive image is not a failed part.")
    check_equal("Appendix A row count against the class lists",
                len(rows), sum(len(v) for v in DATASET_CLASSES.values()), 0)

    # ------------------------------------------------------------ appendix B
    d.add_page_break()
    h("Appendix B. The released file behind every reported number", 1)
    note("This mapping is what makes C5 checkable. The journal may prefer it "
         "as supplementary material rather than an appendix.")
    p(f"{T('provenance')} names the released file behind each reported "
      f"quantity.")
    table(
        ["Quantity", "File"],
        [["Internal control matrix and transfer ratios",
          "grid_results_arms/analysis/lohi_control.md"],
         ["Control transfer intervals",
          "grid_results_arms/analysis/lohi_control_transfer_intervals.csv"],
         [f"Box geometry of the audited tree, {T('geometry')}",
          "clean_candidate/box_geometry.json, built by box_geometry.py"],
         ["Every cell of the main grid, per seed",
          "grid_results/analysis/tables/cells.csv"],
         ["Box-level cross-dataset matrices",
          "grid_results/analysis/tables/transfer_box_*.md"],
         ["Transfer ratios with split-block intervals, both granularities",
          "grid_results/analysis/tables/transfer_ratio_intervals.md"],
         ["Image-level AUROC per evaluation, with intervals, chance AP and "
          "block counts",
          "grid_results/analysis/tables/image_level.csv, built by "
          "grid/image_level.py"],
         [f"Image-level matrix, {T('auroc')}, and the retained margins",
          "grid_results/analysis/tables/image_level_means.csv, "
          "image_level_retained_margin.csv, image_level_matrix.md"],
         [f"Test splits counted in blocks, {T('blocks')}",
          "grid_results/analysis/tables/image_level_samples.csv"],
         [f"Leakage audit in AUROC, {T('audit')}",
          "grid_results/analysis/tables/image_level_audit.csv"],
         ["kaggle_v2 checks, exploratory",
          "grid_results/analysis/tables/image_level_kaggle_checks.csv, "
          "clean_candidate/kaggle_v2_test_image_groups.json"],
         ["Image level in the arms: control, label cost, pooling",
          "grid_results_arms/analysis/image_level_arms.csv, "
          "image_level_control_margin.csv, image_level_label_cost.csv, "
          "image_level_pooling.csv, image_level_pooling_seed_averaged.csv, "
          "image_level_samples_arms.csv"],
         ["Image-level matrices in average precision and base rates",
          "grid_results/analysis/tables/transfer_image_*.md, "
          "target_base_rates.csv"],
         ["Leakage audit estimate shift",
          "grid_results/analysis/tables/downloaded_vs_audited.csv"],
         ["Label cost curve, every cell and both endpoints",
          "grid_results_arms/analysis/armb_recovery.csv"],
         ["Exploratory schedule controls",
          "grid_results_msc/matched_step_control.csv"],
         [f"Training recipe, {T('recipe')}",
          "grid/config.py, grid/arms/config_arms.py, grid/yolo_dataset.py, "
          "grid/train_cell.py"],
         ["Pooling, every fold",
          "grid_results_arms/analysis/armd.csv"],
         ["Pooling box-level coverage, derived",
          "grid_results_arms/analysis/armd_box_limit.json"],
         ["Recall at a fixed false positive budget, per budget, and the "
          "internal control's per-cell mAP50",
          "grid_results_arms/analysis/recall_at_fp_budget_arms.csv"],
         ["Calibration and reliability bins",
          "grid_results_arms/analysis/calibration_arms.csv, "
          "reliability_bins_arms.csv"],
         ["Per-class AP with test instance counts",
          "grid_results/analysis/tables/per_class_ap.csv"],
         ["Pre-registered statistics",
          "grid_results/analysis/statistical_tests.json, "
          "grid_results_arms/analysis/stats_arms.json"],
         ["Frozen split hashes",
          "clean_candidate/split_hashes.json, "
          "grid/arms/arm_split_hashes.json"],
         ["Class mapping, Appendix A", "grid/config.py, DATASETS"],
         ["Reference metadata, resolved by DOI",
          "literature_review/references.json, built by "
          "resolve_references.py"]])
    tcap("provenance", "The released file behind each reported quantity. When "
            "the manuscript is built, every table, and every number in the "
            "text that restates one, is read from these files at full "
            "precision and rounded once. The build stops if a quoted "
            "bootstrap draw count, the internal control's intervals or the "
            "class lists disagree with these files, or if two files that "
            "report the same quantity disagree with each other.")

    def springer_back_matter():
        h("Statements and Declarations", 1)
        labelled("Funding.", PROFILE["funding"])
        confirm_note("funding")
        labelled("Competing interests.", "The author has no relevant financial or "
                 "non-financial interests to disclose.")
        confirm_note("competing")
        labelled("Author contributions.",
                 f"{AUTHOR} is the sole author and was responsible for the "
                 f"conceptualisation, methodology, software, data curation, formal "
                 f"analysis, investigation, visualisation, and the writing of the "
                 f"original draft and its revision.")
        confirm_note("contributions")
        labelled("Ethics approval.", "Not applicable. The study uses only publicly "
                 "available images of welds and involves no human participants "
                 "and no animals.")
        labelled("Consent to participate and for publication.", "Not applicable.")
        code_where = (f"are available at {REPO_URL}" if REPO_LIVE else
                      "will be released in a public repository on acceptance of "
                      "this article")
        data_par = (
            f"The five source datasets are available from their authors at the "
            f"locations given in the reference list: kaggle_v2 "
            f"{cp('wijaya2024')} under CC0 1.0; weld_surface_v2 "
            f"{cp('welddefect2025')}, visual {cp('faza2024')} and twoclass "
            f"{cp('kaidal2024')} under CC BY 4.0; and LoHi-WELD "
            f"{cp('block2024')}, which its authors release for research and "
            f"commercial use with citation at {LOHI_URL}. The twoclass project "
            f"publishes no downloadable version, so the copy used here was "
            f"obtained by forking it, and that fork, export version 1, is public "
            f"at {TWOCLASS_FORK_URL}. We do not redistribute images from any "
            f"source. That is our own choice rather than a licence constraint: "
            f"the CC0 source's own description states that its images were "
            f"collected from unnamed prior collections, so we treat its provenance "
            f"as unverified and point readers at the authoritative copy rather "
            f"than propagate a second one. The harmonised label mappings, the "
            f"frozen split files and their sha256 hashes, and the per-cell results "
            f"{code_where}. Every split file identifies its images by their "
            f"original filenames, and all {SPLIT_ENTRIES_VERIFIED:,} entries were "
            f"verified on {SPLIT_VERIFY_DATE} to resolve against untouched "
            f"original downloads, so the partitions can be reconstructed exactly "
            f"and checked against the published hashes.")
        labelled("Data availability.", data_par)
        confirm_note("twoclass_fork")
        labelled("Code availability.",
                 f"The audit, training, evaluation and analysis code and the "
                 f"script that builds this manuscript {code_where}.")
        confirm_note("repository")

    def elsevier_back_matter():
        """Elsevier's order, directly before the references: CRediT, the
        declaration of competing interest, funding, data availability. No
        acknowledgements, as decided on 9 October."""
        h("CRediT authorship contribution statement", 1)
        d.add_paragraph(CREDIT)
        confirm_note("credit")
        h("Declaration of competing interest", 1)
        d.add_paragraph(COMPETING_ELSEVIER)
        confirm_note("competing")
        h("Funding", 1)
        d.add_paragraph(PROFILE["funding"])
        confirm_note("funding")
        h("Data availability", 1)
        if not REPO_LIVE:
            raise SystemExit("\nCONSISTENCY CHECK FAILED: the AEI data "
                             "availability statement names the repository, "
                             "and REPO_LIVE is False.")
        where = (f"{REPO_URL}, archived on Zenodo "
                 f"{cp('deposit')} at https://doi.org/{ZENODO_DOI}"
                 if ZENODO_DOI else REPO_URL)
        data = (
            f"The code, the audited split lists and their sha256 hashes, the "
            f"harmonised label mappings, and the generated tables and figures "
            f"are available at {where}. The repository does not hold the "
            f"source images, which remain under their creators' licences and "
            f"are available from the locations given in the reference list: "
            f"kaggle_v2 {cp('wijaya2024')} under CC0 1.0; weld_surface_v2 "
            f"{cp('welddefect2025')}, visual {cp('faza2024')} and twoclass "
            f"{cp('kaidal2024')} under CC BY 4.0, the last through its public "
            f"fork at {TWOCLASS_FORK_URL}, because the original publishes no "
            f"downloadable version; and LoHi-WELD {cp('block2024')}, free to "
            f"use with citation, at {LOHI_URL}. Not redistributing the images "
            f"is our choice rather than a licence constraint, because the CC0 "
            f"source's own description says its images were collected from "
            f"unnamed prior collections. Nor does the repository hold the "
            f"trained checkpoints and raw predictions, {CHECKPOINTS_GB} GB. "
            f"Every split file identifies its images by their original "
            f"filenames, and all {SPLIT_ENTRIES_VERIFIED:,} entries were "
            f"verified on {SPLIT_VERIFY_DATE} to resolve against untouched "
            f"original downloads, so the partitions can be reconstructed "
            f"exactly and checked against the published hashes.")
        d.add_paragraph(data)
        confirm_note("repository")

    # ---------------------------------------------- statements and declarations
    d.add_page_break()
    if PROFILE["back_matter"] == "elsevier":
        elsevier_back_matter()
    else:
        springer_back_matter()

    # ------------------------------------------------------------- references
    d.add_page_break()
    h("References", 1)
    check_references()
    for key in sorted(REFS, key=_sort_key):
        par = d.add_paragraph()
        par.paragraph_format.left_indent = Inches(0.3)
        par.paragraph_format.first_line_indent = Inches(-0.3)
        par.paragraph_format.space_after = Pt(4)
        # Elsevier marks every data reference, AEI's Guide for Authors.
        if PROFILE["dataset_marker"] and \
                REFS[key]["csl"].get("type") == "dataset":
            par.add_run("[dataset] ").font.size = Pt(10)
        for text, italic in reference_runs(key):
            r = par.add_run(text)
            r.italic = italic
            r.font.size = Pt(10)

    paras = [p.text for p in d.paragraphs]
    refs = max((k for k, t in enumerate(paras) if t == "References"),
               default=len(paras))
    check_withdrawn_wording(paras[:refs] + [c.text for t in d.tables
                                            for row in t.rows
                                            for c in row.cells])
    check_numbering(d)
    check_tables(d)
    check_figures(figures)
    check_plain_text(d)
    check_british(d)
    if clean:
        check_clean_document(d)
    d.save(out)
    kind = "CLEAN submission version" if clean else "working draft"
    print(f"wrote {out}  [{kind}, {PROFILE['name']}]")
    print(f"  {len(d.paragraphs)} paragraphs, {len(d.tables)} tables, "
          f"{len(d.inline_shapes)} figures, abstract {n_words} words, "
          f"{len(KEYWORDS)} keywords, {len(REFS)} references")
    if clean:
        lines = highlights(H, abstract) if PROFILE["highlights"] else []
        if PROFILE["highlights"]:
            write_highlights(lines)
        if JOURNAL == ROOT_JOURNAL:
            shutil.copyfile(out, SUBMISSION / "Manuscript.docx")
        write_checklist(n_words, lines, figures)
        print(f"  submission folder: {SUBMISSION.relative_to(ROOT)}")
    open_items = [(k, what) for k, (done, what) in CONFIRM.items() if not done]
    if open_items:
        print(f"\n  NOT YET SUBMITTABLE: {len(open_items)} items need the "
              f"author. Flip each in CONFIRM once done.")
        for k, what in open_items:
            print(f"   - {k}: {what[:110]}{'...' if len(what) > 110 else ''}")
    return out


if __name__ == "__main__":
    build(clean="--clean" in sys.argv)
