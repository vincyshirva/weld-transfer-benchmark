#!/usr/bin/env python3
"""
Build a deduplicated, leakage-free version of the visible-light weld datasets.

Stage 2 of the cross-dataset generalization study. Reads the original dataset
folders, never writes to them, and produces a new tree with frozen split files
that can be released alongside the paper.

WHY THIS IS NOT JUST A HASH FILTER
----------------------------------
Threshold calibration (stage 1b) showed that a 64-bit dHash cannot decide
duplication on its own. Unrelated weld photographs collide at a Hamming
distance of 1, because every image in this domain reduces to a dark ground
with a bright bead. So dHash is used only to describe the data, and the
decision is made by a direct content comparison.

Three separate mechanisms produce duplicates here, and each needs its own
treatment:

  1. Export-time augmentation. Roboflow writes the same source frame several
     times under one stem with different .rf. hashes. Exactly recoverable
     from the filename.
  2. Pre-baked rotation and flip variants, encoded in the filename as _0,
     _90, _180, _270 and _flip. The 2class set is built almost entirely this
     way. Also exactly recoverable from the filename.
  3. Scene dependence. Weld Surface and 2class contain continuous frame runs.
     Kaggle videos contain several coupons, so their adjacency is used only
     as a conservative split constraint. Visual uses a specimen key.

Mechanisms 1 and 2 give copies, which collapse to one representative.
Mechanism 3 gives distinct images that must nonetheless never be split
across a train/test boundary, so they are grouped and the group is the
atomic unit of partitioning.

THRESHOLDS
----------
Content similarity starts with NCC on a 32x32 standardized grayscale
thumbnail, taken as the maximum over the 8 dihedral transforms.
Standardization makes it invariant to brightness and contrast changes.

  T_COPY  = 0.96   same photograph
  T_SCENE = 0.90   candidate same scene at 32 pixels

Scene candidates require a second NCC score of at least 0.90 at 128 pixels.
This rejects low entropy collisions that survive the coarse candidate pass.
Structural run and specimen keys recover real scene relationships whose
viewpoint change makes direct pixel comparison too strict.

Both were calibrated against anchors that need no judgement call. Pairs
sharing a source id are the same photograph by construction and bottom out
at 0.965. Pairs drawn from two genuinely different datasets cannot be the
same photograph and top out at 0.877. The interval between those is empty.

Usage:
    python3 weld_rebuild_clean.py                    # uses ROOT below
    python3 weld_rebuild_clean.py /path/to/datasets
    python3 weld_rebuild_clean.py --out clean_candidate
    python3 weld_rebuild_clean.py --out clean_candidate --force

Requires:  pillow numpy pyyaml scipy
Outputs:   <output>/<dataset>/{train,valid,test}/{images,labels}
           <output>/<dataset>/data.yaml
           <output>/splits/<dataset>_{train,valid,test}.txt
           <output>/manifest.csv        every original file and its fate
           <output>/before_after.csv    the counts table
           <output>/before_after.md     the same table, formatted
           <output>/rebuild_report.json machine-readable summary
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.spatial.distance import squareform
from scipy.sparse import lil_matrix

Image.MAX_IMAGE_PIXELS = None

# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
OUT_NAME = "clean"

SEED = 20260903
SPLIT_FRACTIONS = {"train": 0.70, "valid": 0.15, "test": 0.15}

T_COPY = 0.96
T_SCENE = 0.90

THUMB = 32                    # descriptor side length
SCENE_CONFIRM_THUMB = 128     # secondary scene confirmation side length
BLOCK = 1024                  # rows per matmul block

# Per-class AP on a handful of ground-truth instances is dominated by sampling
# noise, and three seeds cannot show that. Any class below this many test
# instances is flagged everywhere it is reported, and needs bootstrap
# confidence intervals over test instances rather than seed dispersion alone.
MIN_TEST_INSTANCES = 50

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# The five visible-light sources that enter the primary study. The two
# superseded folders, Weld Surface v1 and Kaggle v1, are deliberately absent:
# the stage 1 audit showed each is fully contained in its v2 sibling.
DATASETS = {
    "weld_surface_v2": "Weld Surface Defect Detection.v2i.yolov8",
    "kaggle_v2": ("Welding Defect, Object Detection_kaggle/"
                  "The Welding Defect Dataset - v2/The Welding Defect Dataset - v2"),
    "twoclass": "welding defect 2class.v1i.yolov8",
    "visual": "Visual Welding Defect.v1i.yolov8",
    # Converted from the shipped JSON by lohi_to_yolo.py. BOTH the high and
    # low resolution acquisitions, 1,022 plus 2,000 images. They were shown to
    # be separate acquisitions, not two exports of one set: no shared
    # filename, and all 2,044,000 cross pairs peak at NCC 0.6546 coarse and
    # 0.6129 confirmed, against a T_copy of 0.96. Supplied k-folds ignored so
    # this study does its own splitting.
    "lohi_weld": "lohi_weld_yolo",
}

EXCLUDED = {
    "Weld Surface Defect Detection.v1i.yolov8":
        "superseded by v2, 53,600 matching pairs, identical split counts",
    "Welding Defect, Object Detection_kaggle/The Welding Defect Dataset":
        "subset of The Welding Defect Dataset - v2, 2,047 matching pairs",
    "LoHi-WELD weld-dataset/weld-dataset/kfold_high, kfold_low":
        "supplied k-folds, ignored so this study does its own splitting",
}

# Roboflow suffix, then the export format tag, then a baked rotation or flip
# tag. Rotation tags are source augmentation only in twoclass. In Kaggle names
# such as SampleV2_1_mp4-90, the final number is a real video frame index.
RE_RF = re.compile(r"\.rf\.[0-9a-f]{32}\.\w+$")
RE_FMT = re.compile(r"_(jpg|png|jpeg)$", re.I)
RE_AUG = re.compile(r"((?:_|-)(?:0|90|180|270))?(_flip)?(_jpg|_png|_jpeg)*$", re.I)

# Weld Surface and 2class name their frames <run>-<index>, and their inspected
# montages show one continuous weld in each adjacent run. Two frames of one
# run whose indices are within RUN_GAP are therefore one scene even when their
# content score falls below T_SCENE. This closes the band just under the
# threshold, where adjacent frames such as NWS-852 and NWS-854 score 0.896.
#
# The Kaggle video names also expose frame order, but its inspected montages
# show a camera panning across several different coupons in one video. Those
# links are useful only as conservative split constraints. They must not
# reduce the reported independent scene count.
# Only patterns that unambiguously index frames of one video are listed. The
# Kaggle set also holds numbered stock-photo collections such as
# Good-Welding-images_14, where consecutive numbers are unrelated pictures and
# run grouping would be wrong, so those are deliberately not matched.
RUN_GAP = 3
RUN_SCENE_TAGS = {"weld_surface_v2", "twoclass"}
RUN_SPLIT_TAGS = RUN_SCENE_TAGS | {"kaggle_v2"}

# A rotation or flip tag at the end of a Roboflow stem. `_0` on its own is the
# identity rotation, so it does not count as augmented. Used to prefer an
# unrotated representative, so the clean set has consistent orientation.
RE_AUG_TAIL = re.compile(r"(_(?:0|90|180|270))?_flip$|_(?:90|180|270)$", re.I)

# Specimen identity, where the filename carries it and it is not recoverable
# from content. The Visual set photographs each plate up to six times, as
# Pre and Post test, from the Root, Capping and Face sides. Those views look
# different enough that NCC never links them, so without this key 76% of
# specimens had views in more than one split. Training on the root face and
# testing on the capping face of the same plate is leakage.
#
# LoHi-WELD low resolution names carry the production cycle explicitly, as
# <date>_<time>_<cycle>_CAM1_<NG|OK>_<n>. There are exactly two images per
# cycle over 1,000 cycles, and they are two views of one welded part. They
# score about 0.32 against each other, so no content threshold will ever link
# them, yet putting one in train and the other in test tests on a part the
# model was trained on. The high resolution names do not match this pattern
# and are unaffected: that acquisition really is one image per cycle.
SPECIMEN_KEYS = {
    "visual": re.compile(r"^(\d+)_"),
    "lohi_weld": re.compile(r"^(\d{6}_\d{6}_\d+_CAM\d+)_(?:NG|OK)_\d+$"),
}
# ----------------------------------------------------------------------


def roboflow_stem(name: str) -> tuple[str, bool]:
    """Return the stem before the Roboflow hash and whether one was present."""
    stem = RE_RF.sub("", name)
    return stem, stem != name


def source_id(tag: str, name: str) -> str:
    """Strip everything an exporter added, leaving the source frame id.

    Rotation and flip tags are only stripped from Roboflow exports, which is
    the only place they occur. Stripping them unconditionally is wrong: in
    LoHi-WELD the names are <run>_<frame>, so 26_90 and 26_270 are two
    distinct frames, not two rotations of one image, and collapsing them
    silently loses data.
    """
    s, is_rf = roboflow_stem(name)
    if not is_rf:
        # Not a Roboflow export. Drop the extension and stop there.
        return re.sub(r"\.\w{2,5}$", "", s)
    if tag == "twoclass":
        return RE_AUG.sub("", RE_FMT.sub("", s))
    return s


def is_augmented_variant(tag: str, name: str) -> bool:
    """True if this file is a rotated or flipped copy, per its Roboflow stem."""
    if tag != "twoclass":
        return False
    s, is_rf = roboflow_stem(name)
    if not is_rf:
        return False
    return bool(RE_AUG_TAIL.search(RE_FMT.sub("", s)))


def specimen_key(tag: str, name: str) -> str | None:
    """Physical specimen id, for datasets whose filenames carry one."""
    rx = SPECIMEN_KEYS.get(tag)
    if rx is None:
        return None
    m = rx.match(source_id(tag, name))
    return m.group(1) if m else None


def content_scene_compatible(tag: str, left_name: str, right_name: str) -> bool:
    """Whether an NCC edge may assert one physical scene.

    Some filenames carry a stronger physical identity than pixels can supply.
    Visual has a specimen id for 324 images. LoHi-WELD low resolution has a
    production cycle id for all 2,000. Different known ids are not one scene,
    however similar a fixed rig or a fixed camera station makes their
    thumbnails. A known id is also not joined to an image with no recoverable
    identity, which keeps the two LoHi acquisitions apart. All direct NCC
    edges remain conservative split constraints, so this semantic guard
    cannot introduce leakage.
    """
    if tag not in SPECIMEN_KEYS:
        return True
    left = specimen_key(tag, left_name)
    right = specimen_key(tag, right_name)
    if left is None or right is None:
        return left is None and right is None
    return left == right


def run_key(tag: str, name: str) -> tuple[str, int] | None:
    """Return a documented video run and frame index, or None.

    LoHi-WELD is deliberately absent. Its paper states that the camera was
    triggered once per production cycle. Numeric filename adjacency therefore
    does not establish repeated frames of one weld.
    """
    stem, is_rf = roboflow_stem(name)
    if not is_rf:
        stem = re.sub(r"\.\w{2,5}$", "", stem)

    if tag == "weld_surface_v2":
        # Preserve the source format token as part of the run identity. The
        # _jpg and _jpeg sequences contain distinct photographs at equal frame
        # numbers, for example LoF-9-51.
        m = re.match(
            r"^([A-Za-z]+)[-_]?9?[-_](\d+)[-_]?_(jpg|jpeg|png)$", stem, re.I)
        if m:
            return f"{m.group(1).lower()}:{m.group(3).lower()}", int(m.group(2))
    elif tag == "twoclass":
        m = re.match(r"^(\d+)-(\d+)-", source_id(tag, name))
        if m:
            return m.group(1), int(m.group(2))
    elif tag == "kaggle_v2":
        run_stem = re.sub(r"(?:_(?:jpg|jpeg|png))+$", "", stem, flags=re.I)
        for rx in (re.compile(r"^(.*_mp\d+)-(\d+)$"),
                   re.compile(r"^(.*_vid)(\d+)$")):
            m = rx.match(run_stem)
            if m:
                return m.group(1), int(m.group(2))
    return None


class DisjointSet:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1

    def groups(self) -> dict[int, list[int]]:
        g = defaultdict(list)
        for i in range(len(self.p)):
            g[self.find(i)].append(i)
        return g


# ------------------------------------------------------------- reading
def read_label(p: Path) -> list[tuple[int, float, float, float, float]]:
    """Read YOLO detection or segmentation labels as bounding boxes.

    Some sources mix five-field detection rows with polygon rows. Polygon
    coordinates are converted to their enclosing normalized box in the
    derivative tree. The originals remain unchanged.
    """
    if not p.exists():
        return []
    out = []
    for line_number, ln in enumerate(p.read_text().splitlines(), 1):
        if not ln.strip():
            continue
        parts = ln.split()
        try:
            cls = int(float(parts[0]))
            values = [float(v) for v in parts[1:]]
        except (ValueError, IndexError) as error:
            raise ValueError(f"{p}:{line_number}: nonnumeric annotation") from error
        if len(values) == 4:
            cx, cy, width, height = values
        elif len(values) >= 6 and len(values) % 2 == 0:
            xs = values[0::2]
            ys = values[1::2]
            left, right = min(xs), max(xs)
            top, bottom = min(ys), max(ys)
            cx, cy = (left + right) / 2.0, (top + bottom) / 2.0
            width, height = right - left, bottom - top
        else:
            raise ValueError(f"{p}:{line_number}: expected box or polygon row")
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"{p}:{line_number}: invalid normalized geometry")
        out.append((cls, cx, cy, width, height))
    return out


def scan_dataset(ds_dir: Path, tag: str) -> tuple[list[dict], list[str]]:
    # Staging folders deliberately ship classes.yaml rather than data.yaml, so
    # they cannot be handed to a trainer or picked up by data.yaml discovery.
    cfg_path = ds_dir / "data.yaml"
    if not cfg_path.exists():
        cfg_path = ds_dir / "classes.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No data.yaml or classes.yaml in {ds_dir}")
    cfg = yaml.safe_load(cfg_path.read_text())
    names = cfg.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    names = [str(n) for n in names]

    recs = []
    for split in ("train", "valid", "val", "test"):
        img_dir = ds_dir / split / "images"
        if not img_dir.is_dir():
            continue
        lbl_dir = ds_dir / split / "labels"
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() not in IMG_EXT:
                continue
            boxes = read_label(lbl_dir / (p.stem + ".txt"))
            recs.append({
                "path": p,
                "orig_split": "valid" if split == "val" else split,
                "source_id": source_id(tag, p.name),
                "boxes": boxes,
                "n_boxes": len(boxes),
                "classes": Counter(b[0] for b in boxes),
            })
    return recs, names


def descriptor_at(p: Path, thumb: int) -> np.ndarray | None:
    """Standardized grayscale descriptor at the requested side length."""
    try:
        with Image.open(p) as im:
            a = np.asarray(im.convert("L").resize((thumb, thumb), Image.LANCZOS),
                           dtype=np.float32)
    except Exception:
        return None
    a -= a.mean()
    s = a.std()
    if s < 1e-6:
        return a.ravel()
    return (a / s).ravel()


def descriptor(p: Path) -> np.ndarray | None:
    """Coarse 32 pixel descriptor used for copy and scene candidates."""
    return descriptor_at(p, THUMB)


def pair_ncc(left: np.ndarray, right: np.ndarray, thumb: int) -> float:
    """Best NCC for one standardized descriptor pair over 8 transforms."""
    a = left.reshape(thumb, thumb)
    b = right.reshape(thumb, thumb)
    return max(
        float(np.mean(a * variant))
        for rotation in range(4)
        for variant in (
            np.rot90(b, rotation),
            np.rot90(b, rotation)[:, ::-1],
        )
    )


def confirm_scene_pairs(
        paths: list[Path],
        candidate_pairs: list[tuple[int, int, float]],
) -> list[tuple[int, int, float]]:
    """Confirm coarse scene candidates at 128 pixels.

    Only candidate endpoints are decoded at the larger size. The returned
    score is the confirmation score, not the coarse candidate score.
    """
    cache: dict[int, np.ndarray] = {}
    confirmed = []
    for left, right, _ in candidate_pairs:
        for index in (left, right):
            if index not in cache:
                value = descriptor_at(paths[index], SCENE_CONFIRM_THUMB)
                if value is None:
                    raise RuntimeError(f"Could not read scene candidate {paths[index]}")
                cache[index] = value
        score = pair_ncc(cache[left], cache[right], SCENE_CONFIRM_THUMB)
        if score >= T_SCENE:
            confirmed.append((left, right, score))
    return confirmed


def dihedral_stack(D: np.ndarray) -> list[np.ndarray]:
    """The 8 dihedral transforms of a stack of flattened THUMB x THUMB
    descriptors, so rotated and mirrored copies still match."""
    n = D.shape[0]
    sq = D.reshape(n, THUMB, THUMB)
    out = []
    for k in range(4):
        r = np.rot90(sq, k, axes=(1, 2))
        out.append(np.ascontiguousarray(r.reshape(n, -1)))
        out.append(np.ascontiguousarray(r[:, :, ::-1].reshape(n, -1)))
    return out


def ncc_all_pairs(D: np.ndarray) -> np.ndarray:
    """Full similarity matrix for a small set, best of the 8 dihedral
    transforms. Used inside a single-linkage component to apply complete
    linkage, so components stay small and this stays cheap."""
    best = None
    for V in dihedral_stack(D):
        s = (D @ V.T) / float(THUMB * THUMB)
        best = s if best is None else np.maximum(best, s)
    return best


def pairs_above(D: np.ndarray, thr: float) -> list[tuple[int, int, float]]:
    """All pairs whose max dihedral NCC reaches thr. Blocked matmul, so the
    full similarity matrix is never held in memory at once."""
    n = D.shape[0]
    if n < 2:
        return []
    ref = np.ascontiguousarray(D)
    variants = dihedral_stack(D)
    norm = float(THUMB * THUMB)
    found = []
    for a in range(0, n, BLOCK):
        blk = ref[a:a + BLOCK]
        best = None
        for V in variants:
            s = (blk @ V.T) / norm
            best = s if best is None else np.maximum(best, s)
        ii, jj = np.where(best >= thr)
        for x, y in zip(ii, jj):
            gi, gj = a + int(x), int(y)
            if gi < gj:
                found.append((gi, gj, float(best[x, y])))
    return found


def complete_linkage_groups(
        D: np.ndarray, threshold: float,
        candidate_pairs: list[tuple[int, int, float]] | None = None,
) -> tuple[list[list[int]], int, float]:
    """Partition threshold components with complete linkage.

    Candidate edges first identify small regions that can interact. Complete
    linkage then requires every pair inside a returned group to meet the
    threshold. This prevents A to B to C chains from deleting or grouping A
    with an unrelated C.
    """
    if len(D) == 0:
        return [], 0, 1.0
    pairs = candidate_pairs if candidate_pairs is not None else pairs_above(D, threshold)
    single = DisjointSet(len(D))
    for a, b, _ in pairs:
        single.union(a, b)

    out: list[list[int]] = []
    split_components = 0
    min_diameter = 1.0
    for comp in single.groups().values():
        if len(comp) < 3:
            out.append(sorted(comp))
            continue
        S = ncc_all_pairs(D[comp])
        vals = S[np.triu_indices(len(comp), 1)]
        min_diameter = min(min_diameter, float(vals.min()))
        dist = np.clip(1.0 - S, 0.0, None)
        np.fill_diagonal(dist, 0.0)
        dist = (dist + dist.T) / 2.0
        Z = linkage(squareform(dist, checks=False), method="complete")
        labels = fcluster(Z, t=1.0 - threshold, criterion="distance")
        buckets = defaultdict(list)
        for member, label in zip(comp, labels):
            buckets[int(label)].append(member)
        split_components += len(buckets) > 1
        out.extend(sorted(v) for _, v in sorted(buckets.items()))
    out.sort(key=lambda members: (members[0], len(members)))
    return out, split_components, min_diameter


def complete_linkage_pair_groups(
        n_items: int,
        threshold: float,
        candidate_pairs: list[tuple[int, int, float]],
) -> tuple[list[list[int]], int, float]:
    """Complete linkage using an explicit two-stage similarity graph.

    An absent edge failed either the coarse or confirmation pass, so it cannot
    support one physical scene. This keeps complete linkage exact without
    allocating a dense 128 pixel descriptor matrix for every image.
    """
    if n_items == 0:
        return [], 0, 1.0
    single = DisjointSet(n_items)
    scores = {}
    for left, right, score in candidate_pairs:
        single.union(left, right)
        scores[(min(left, right), max(left, right))] = score

    out = []
    split_components = 0
    min_diameter = 1.0
    for component in single.groups().values():
        component = sorted(component)
        if len(component) < 3:
            out.append(component)
            continue
        similarity = np.eye(len(component), dtype=np.float32)
        for i, left in enumerate(component):
            for j, right in enumerate(component[i + 1:], i + 1):
                score = scores.get((left, right), 0.0)
                similarity[i, j] = similarity[j, i] = score
        values = similarity[np.triu_indices(len(component), 1)]
        min_diameter = min(min_diameter, float(values.min()))
        distance = np.clip(1.0 - similarity, 0.0, None)
        linkage_matrix = linkage(
            squareform(distance, checks=False), method="complete")
        labels = fcluster(
            linkage_matrix, t=1.0 - threshold, criterion="distance")
        buckets = defaultdict(list)
        for member, label in zip(component, labels):
            buckets[int(label)].append(member)
        split_components += len(buckets) > 1
        out.extend(sorted(value) for _, value in sorted(buckets.items()))
    out.sort(key=lambda members: (members[0], len(members)))
    return out, split_components, min_diameter


def select_copy_representatives(D, candidate_pairs, priority_key):
    """Keep a maximum independent set of the copy similarity graph.

    Complete linkage prevents chain overcollapse, but representatives from two
    adjacent complete-linkage clusters can still form a copy pair. A maximum
    independent set keeps the largest possible collection with no retained
    edge at T_COPY. Every removed vertex is assigned to a retained neighbour.
    """
    graph = DisjointSet(len(D))
    scores = {}
    adjacency = defaultdict(set)
    for left, right, score in candidate_pairs:
        graph.union(left, right)
        scores[(min(left, right), max(left, right))] = score
        adjacency[left].add(right)
        adjacency[right].add(left)

    retained = []
    representative_of = {}
    score_to_representative = {}
    for component in graph.groups().values():
        component = sorted(component)
        local_edges = [(a, b) for a in component for b in adjacency[a]
                       if a < b and b in component]
        if not local_edges:
            chosen = component
        else:
            position = {member: i for i, member in enumerate(component)}
            A = lil_matrix((len(local_edges), len(component)), dtype=np.float64)
            for row, (left, right) in enumerate(local_edges):
                A[row, position[left]] = 1.0
                A[row, position[right]] = 1.0
            cardinality = milp(
                c=-np.ones(len(component)),
                integrality=np.ones(len(component), dtype=np.uint8),
                bounds=Bounds(np.zeros(len(component)), np.ones(len(component))),
                constraints=LinearConstraint(
                    A.tocsr(), np.full(len(local_edges), -np.inf),
                    np.ones(len(local_edges))),
                options={"mip_rel_gap": 0.0},
            )
            if cardinality.x is None:
                raise RuntimeError(
                    f"Copy cardinality optimization failed: {cardinality.message}")
            maximum = int(round(float(cardinality.x.sum())))

            # Fix the maximum retained count. Then choose one stable optimum.
            # Integer costs avoid relying on solver tolerances for tiny tie
            # coefficients. The hash makes equal rank sums deterministic.
            ordered = sorted(component, key=priority_key)
            priority_order = {member: rank for rank, member in enumerate(ordered)}
            tie_cost = []
            for member in component:
                digest = hashlib.sha256(
                    repr(priority_key(member)).encode()).digest()
                hashed = int.from_bytes(digest[:4], "big")
                tie_cost.append(priority_order[member] * (2**32) + hashed)
            A2 = lil_matrix((len(local_edges) + 1, len(component)), dtype=np.float64)
            A2[:len(local_edges), :] = A
            A2[len(local_edges), :] = 1.0
            lower = np.r_[np.full(len(local_edges), -np.inf), maximum]
            upper = np.r_[np.ones(len(local_edges)), maximum]
            result = milp(
                c=np.asarray(tie_cost, dtype=np.float64),
                integrality=np.ones(len(component), dtype=np.uint8),
                bounds=Bounds(np.zeros(len(component)), np.ones(len(component))),
                constraints=LinearConstraint(A2.tocsr(), lower, upper),
                options={"mip_rel_gap": 0.0},
            )
            if result.x is None:
                raise RuntimeError(
                    f"Copy tie-break optimization failed: {result.message}")
            chosen = [member for member, value in zip(component, result.x)
                      if value > 0.5]

        chosen_set = set(chosen)
        retained.extend(chosen)
        for member in component:
            if member in chosen_set:
                representative_of[member] = member
                score_to_representative[member] = 1.0
                continue
            neighbours = [other for other in adjacency[member] if other in chosen_set]
            if not neighbours:
                raise RuntimeError(
                    "Maximum independent set was not maximal; no representative neighbour")
            representative = min(
                neighbours,
                key=lambda other: (
                    -scores[(min(member, other), max(member, other))],
                    priority_key(other),
                ),
            )
            representative_of[member] = representative
            score_to_representative[member] = scores[
                (min(member, representative), max(member, representative))]

    groups = defaultdict(list)
    for member, representative in representative_of.items():
        groups[representative].append(member)
    for members in groups.values():
        members.sort()
    retained.sort()
    return retained, dict(groups), representative_of, score_to_representative


# ------------------------------------------------------------ splitting
def group_feature_matrix(groups: list[dict], names: list[str]):
    """Build the multilabel features that must be balanced across splits."""
    feature_names = ["images", "groups"]
    weights = [6.0, 0.25]
    feature_names += [f"boxes:{name}" for name in names]
    weights += [2.0] * len(names)
    feature_names += [f"images_with:{name}" for name in names]
    weights += [1.0] * len(names)
    feature_names.append("background_images")
    weights.append(1.0)

    F = np.zeros((len(groups), len(feature_names)), dtype=np.float64)
    for row, g in enumerate(groups):
        F[row, 0] = g["n"]
        F[row, 1] = g.get("n_scene_groups", 1)
        F[row, 2:2 + len(names)] = g["box_counts"]
        start = 2 + len(names)
        F[row, start:start + len(names)] = g["image_presence"]
        F[row, -1] = g["background_images"]

    keep = F.sum(axis=0) > 0
    return F[:, keep], [n for n, k in zip(feature_names, keep) if k], \
        np.asarray([w for w, k in zip(weights, keep) if k], dtype=np.float64)


def multilabel_group_split(groups, names, rng, time_limit: float = 120.0):
    """Assign intact leakage-safe split blocks with multilabel balancing.

    The objective minimizes normalized absolute error for total images, every
    class box count, every class image-presence count, backgrounds, and scene
    group counts. A tiny seeded tie cost makes equivalent optima reproducible.

    time_limit is the HiGHS budget in seconds. It changes how long the solver
    searches, not what it is searching for. Reproducibility of the released
    splits rests on the frozen split files and their sha256 hashes, not on
    re-running this solver: a different budget, or a different HiGHS build,
    may return a different member of an equally good optimum set.
    """
    split_names = list(SPLIT_FRACTIONS)
    fractions = np.asarray([SPLIT_FRACTIONS[s] for s in split_names])
    F, feature_names, feature_weights = group_feature_matrix(groups, names)
    n_groups, n_features = F.shape
    n_splits = len(split_names)
    n_x = n_groups * n_splits
    n_dev = n_splits * n_features
    n_vars = n_x + 2 * n_dev

    c = np.zeros(n_vars, dtype=np.float64)
    c[:n_x] = rng.random(n_x) * 1e-9
    totals = F.sum(axis=0)
    dev_cost = np.tile(feature_weights / np.maximum(totals, 1.0), n_splits)
    c[n_x:n_x + n_dev] = dev_cost
    c[n_x + n_dev:] = dev_cost

    n_rows = n_groups + n_splits * n_features
    A = lil_matrix((n_rows, n_vars), dtype=np.float64)
    lower = np.zeros(n_rows, dtype=np.float64)
    upper = np.zeros(n_rows, dtype=np.float64)

    for g in range(n_groups):
        for s in range(n_splits):
            A[g, g * n_splits + s] = 1.0
        lower[g] = upper[g] = 1.0

    row = n_groups
    for s in range(n_splits):
        for f in range(n_features):
            for g in np.flatnonzero(F[:, f]):
                A[row, g * n_splits + s] = F[g, f]
            d = s * n_features + f
            A[row, n_x + d] = -1.0
            A[row, n_x + n_dev + d] = 1.0
            target = fractions[s] * totals[f]
            lower[row] = upper[row] = target
            row += 1

    bounds = Bounds(
        np.zeros(n_vars, dtype=np.float64),
        np.r_[np.ones(n_x), np.full(2 * n_dev, np.inf)],
    )
    integrality = np.r_[np.ones(n_x, dtype=np.uint8),
                        np.zeros(2 * n_dev, dtype=np.uint8)]
    result = milp(
        c=c,
        integrality=integrality,
        bounds=bounds,
        constraints=LinearConstraint(A.tocsr(), lower, upper),
        options={"time_limit": float(time_limit), "mip_rel_gap": 1e-4},
    )
    if result.x is None:
        raise RuntimeError(f"Multilabel split optimization failed: {result.message}")

    X = result.x[:n_x].reshape(n_groups, n_splits)
    assign = {g["gid"]: split_names[int(np.argmax(X[i]))]
              for i, g in enumerate(groups)}
    actual = np.zeros((n_splits, n_features), dtype=np.float64)
    for i, g in enumerate(groups):
        actual[split_names.index(assign[g["gid"]])] += F[i]

    balance = {
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "objective": float(result.fun),
        "features": {},
        "max_abs_deviation_pp": 0.0,
    }
    for f, name in enumerate(feature_names):
        total = totals[f]
        largest = float(F[:, f].max())
        rec = {
            "total": int(round(total)),
            "largest_group_contribution": int(round(largest)),
            "granularity_pp": round(100.0 * largest / total, 3) if total else 0.0,
            "splits": {},
        }
        for s, split in enumerate(split_names):
            share = actual[s, f] / total if total else 0.0
            deviation_pp = 100.0 * (share - fractions[s])
            rec["splits"][split] = {
                "count": int(round(actual[s, f])),
                "share": round(float(share), 6),
                "deviation_pp": round(float(deviation_pp), 3),
            }
            balance["max_abs_deviation_pp"] = max(
                balance["max_abs_deviation_pp"], abs(float(deviation_pp)))
        balance["features"][name] = rec
    balance["max_abs_deviation_pp"] = round(balance["max_abs_deviation_pp"], 3)
    return assign, balance


# ------------------------------------------------------------------ main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=ROOT)
    parser.add_argument("--out", default=OUT_NAME,
                        help="output folder name under root")
    parser.add_argument("--only", default=None,
                        help="Rebuild a single dataset tag. For solver "
                             "experiments and diagnostics, not for producing "
                             "a release tree. CAVEAT: the per-dataset seed is "
                             "drawn from a master stream in DATASETS order, so "
                             "--only reproduces the full-run splits for the "
                             "first dataset only. For any later tag the seed "
                             "differs and the splits will not match.")
    parser.add_argument("--split-time-limit", type=float, default=120.0,
                        help="HiGHS budget per dataset, seconds. The accepted "
                             "build used 120. Changing it may return a "
                             "different equally good optimum, so it does not "
                             "reproduce frozen splits.")
    parser.add_argument("--force", action="store_true",
                        help="replace an existing output folder")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        sys.exit(f"Not a directory: {root}")
    out_name = args.out
    if not out_name or Path(out_name).name != out_name or out_name in {".", ".."}:
        sys.exit("--out must be one folder name directly under the dataset root")
    protected = {Path(v).parts[0] for v in DATASETS.values()}
    if out_name in protected:
        sys.exit(f"Refusing to use original dataset folder as output: {out_name}")

    out_root = root / out_name
    if out_root.exists():
        if not args.force:
            sys.exit(f"{out_root} already exists. Re-run with --force to replace it.")
        shutil.rmtree(out_root)
    (out_root / "splits").mkdir(parents=True)

    rng_master = np.random.default_rng(SEED)
    print(f"Root:   {root}")
    print(f"Output: {out_root}")
    print(f"Seed:   {SEED}   copy>={T_COPY}   scene>={T_SCENE}\n")
    for name, why in EXCLUDED.items():
        print(f"  excluded  {name}\n            {why}")
    print()

    report = {"schema_version": 3, "seed": SEED,
              "t_copy": T_COPY, "t_scene": T_SCENE,
              "scene_candidate_thumb": THUMB,
              "scene_confirmation_thumb": SCENE_CONFIRM_THUMB,
              "scene_confirmation": "ncc_at_128_after_ncc_at_32",
              "copy_selection": "maximum_independent_set_at_threshold",
              "copy_chain_diagnostic": "complete_linkage",
              "scene_linkage": "complete",
              "split_method": "mixed_integer_multilabel_group_balancing",
              "output_folder": out_name,
              "split_fractions": SPLIT_FRACTIONS,
              "excluded": EXCLUDED, "datasets": {}}
    manifest = []
    before_after = []

    selected = DATASETS
    if args.only:
        if args.only not in DATASETS:
            sys.exit(f"--only {args.only!r} is not one of {list(DATASETS)}")
        selected = {args.only: DATASETS[args.only]}

    for tag, sub in selected.items():
        ds_dir = root / sub
        if not ds_dir.is_dir():
            print(f"!! missing: {ds_dir}")
            continue
        print(f"=== {tag}  ({sub})")
        recs, names = scan_dataset(ds_dir, tag)
        n_before = len(recs)
        boxes_before = sum(r["n_boxes"] for r in recs)
        cls_before = Counter()
        for r in recs:
            cls_before.update(r["classes"])
        split_before = Counter(r["orig_split"] for r in recs)
        print(f"    before: {n_before} images, {boxes_before} boxes, "
              f"{dict(split_before)}")

        # -- 1. collapse export copies by source id ---------------------
        by_src = defaultdict(list)
        for i, r in enumerate(recs):
            by_src[r["source_id"]].append(i)

        def pick(idxs):
            """Representative of a copy set. Prefer an unrotated unflipped
            original, so the clean set has one consistent orientation. Then
            prefer an original that sat in train, as instructed, then the
            richest annotation, then the name, so the choice is deterministic.

            Orientation leads because a rotated representative carries a
            rotated box geometry too, and a set mixing orientations teaches a
            detector that welds run in every direction."""
            return min(idxs, key=lambda i: (
                is_augmented_variant(tag, recs[i]["path"].name),
                recs[i]["orig_split"] != "train",
                -recs[i]["n_boxes"],
                recs[i]["path"].name))

        src_members = {pick(by_src[k]): by_src[k] for k in sorted(by_src)}
        src_reps = list(src_members)
        n_aug_rep = sum(1 for i in src_reps
                        if is_augmented_variant(tag, recs[i]["path"].name))
        print(f"    after filename collapse: {len(src_reps)} source images "
              f"({n_before / max(len(src_reps), 1):.2f}x expansion removed)")
        print(f"    representatives still rotated or flipped: {n_aug_rep} "
              f"(no unrotated member existed)")

        # -- 2. content pass, catch copies the filename missed ----------
        D, ok = [], []
        for i in src_reps:
            d = descriptor(recs[i]["path"])
            if d is not None:
                D.append(d)
                ok.append(i)
        D = np.vstack(D).astype(np.float32)
        n_unreadable = len(src_reps) - len(ok)
        if n_unreadable:
            raise RuntimeError(
                f"{tag}: {n_unreadable} representatives could not be read")

        copy_pairs = pairs_above(D, T_COPY)
        complete_copy_groups, n_copy_components_split, copy_input_min = \
            complete_linkage_groups(D, T_COPY, copy_pairs)
        def local_priority(local):
            rec = recs[ok[local]]
            return (
                is_augmented_variant(tag, rec["path"].name),
                rec["orig_split"] != "train",
                -rec["n_boxes"],
                rec["path"].name,
            )

        uniq_local, copy_members, _, copy_score_to_rep = \
            select_copy_representatives(D, copy_pairs, local_priority)
        merged = len(ok) - len(uniq_local)
        print(f"    content pass: {len(copy_pairs)} copy pairs >= {T_COPY}, "
              f"complete linkage split {n_copy_components_split} chained "
              f"components; maximum independent set merged {merged} images")

        # final unique image per copy group
        copy_group_of_local = {}
        for copy_gid, best in enumerate(sorted(copy_members)):
            for member in copy_members[best]:
                copy_group_of_local[member] = copy_gid
        print(f"    unique images: {len(uniq_local)}")

        # -- 3. scene pass, group frames of the same weld ---------------
        # Content grouping uses COMPLETE linkage, not single. Single linkage
        # chains: A near B, B near C, and A joins C without ever being
        # compared favourably. Measured on the first build, 13% of Weld
        # Surface content groups and 37% of the Visual ones held a pair less
        # similar than the threshold that formed them, one as low as 0.736.
        # Complete linkage requires every member to clear T_SCENE against
        # every other. It moves no images, it only refuses to merge.
        Du = D[uniq_local]
        scene_pairs = pairs_above(Du, T_SCENE)
        compatible_scene_pairs = [
            (left, right, score)
            for left, right, score in scene_pairs
            if content_scene_compatible(
                tag,
                recs[ok[uniq_local[left]]]["path"].name,
                recs[ok[uniq_local[right]]]["path"].name,
            )
        ]
        unique_paths = [recs[ok[local]]["path"] for local in uniq_local]
        content_scene_pairs = confirm_scene_pairs(
            unique_paths, compatible_scene_pairs)
        content_groups, n_split_by_complete, scene_input_min = \
            complete_linkage_pair_groups(
                len(Du), T_SCENE, content_scene_pairs)
        content_gid_by_local = {}
        for content_gid, members in enumerate(content_groups):
            for member in members:
                content_gid_by_local[member] = content_gid
        ds_scene = DisjointSet(len(uniq_local))
        for members in content_groups:
            for m in members[1:]:
                ds_scene.union(members[0], m)
        n_content_groups = len(content_groups)

        # Split blocks are deliberately stricter than reported scene groups.
        # Complete linkage defines independent scenes without chaining. Every
        # direct threshold edge still constrains both endpoints to one split,
        # so no A to B bridge can leak across the evaluation boundary.
        ds_split = DisjointSet(len(uniq_local))
        for left, right, _ in scene_pairs:
            ds_split.union(left, right)

        # frame runs, for the sets that name their frames <run>-<index>
        runs = defaultdict(list)
        for m, u in enumerate(uniq_local):
            k = run_key(tag, recs[ok[u]]["path"].name)
            if k is not None:
                runs[k[0]].append((k[1], m))
        n_run_links = 0
        n_run_scene_links = 0
        for _, seq in runs.items():
            seq.sort()
            for (i0, m0), (i1, m1) in zip(seq, seq[1:]):
                if i1 - i0 <= RUN_GAP:
                    if tag in RUN_SCENE_TAGS:
                        ds_scene.union(m0, m1)
                        n_run_scene_links += 1
                    if tag in RUN_SPLIT_TAGS:
                        ds_split.union(m0, m1)
                    n_run_links += 1

        # physical specimen, where the filename carries it
        spec = defaultdict(list)
        for m, u in enumerate(uniq_local):
            k = specimen_key(tag, recs[ok[u]]["path"].name)
            if k is not None:
                spec[k].append(m)
        n_spec_links = 0
        for _, members in spec.items():
            for m in members[1:]:
                ds_scene.union(members[0], m)
                ds_split.union(members[0], m)
                n_spec_links += 1

        scene_groups = ds_scene.groups()
        sizes = sorted((len(v) for v in scene_groups.values()), reverse=True)
        scene_gid_by_local = {}
        for scene_gid, (_, members) in enumerate(sorted(scene_groups.items())):
            for member in members:
                scene_gid_by_local[member] = scene_gid
        split_blocks = ds_split.groups()
        split_block_sizes = sorted((len(v) for v in split_blocks.values()), reverse=True)
        split_block_gid_by_local = {}
        for block_gid, (_, members) in enumerate(sorted(split_blocks.items())):
            for member in members:
                split_block_gid_by_local[member] = block_gid
        print(f"    scene pass: {len(scene_pairs)} direct pairs >= {T_SCENE}, "
              f"{len(compatible_scene_pairs)} identity-compatible candidates")
        print(f"                {len(content_scene_pairs)} confirmed at "
              f"{SCENE_CONFIRM_THUMB} pixels")
        print(f"                complete linkage {n_content_groups} groups "
              f"(split {n_split_by_complete} chained components)")
        print(f"                + {n_run_scene_links} of {n_run_links} frame-run links "
              f"assert scenes (gap <= {RUN_GAP})"
              f", + {n_spec_links} specimen links "
              f"({len(spec)} specimens)")
        print(f"                -> {len(scene_groups)} groups, largest {sizes[0]}, "
              f"{sum(1 for s in sizes if s > 1)} multi-image groups")
        print(f"                -> {len(split_blocks)} leakage-safe split blocks, "
              f"largest {split_block_sizes[0]}")

        # -- 4. stratified group split ----------------------------------
        groups = []
        for gid, (rootix, members) in enumerate(sorted(split_blocks.items())):
            recs_i = [ok[uniq_local[m]] for m in members]
            box_counts = np.zeros(len(names), dtype=np.int64)
            image_presence = np.zeros(len(names), dtype=np.int64)
            background_images = 0
            for i in recs_i:
                if not recs[i]["boxes"]:
                    background_images += 1
                for cls, count in recs[i]["classes"].items():
                    if 0 <= cls < len(names):
                        box_counts[cls] += count
                        image_presence[cls] += 1
            groups.append({
                "gid": gid,
                "recs": recs_i,
                "locals": members,
                "n_scene_groups": len({scene_gid_by_local[m] for m in members}),
                "n": len(recs_i),
                "box_counts": box_counts,
                "image_presence": image_presence,
                "background_images": background_images,
            })
        rng = np.random.default_rng(rng_master.integers(0, 2**32))
        assign, split_balance = multilabel_group_split(
            groups, names, rng, time_limit=args.split_time_limit)
        print(f"    split optimizer: {split_balance['optimizer_message']}; "
              f"max feature deviation {split_balance['max_abs_deviation_pp']:.2f} pp")

        # -- 5. write ----------------------------------------------------
        ds_out = out_root / tag
        for s in SPLIT_FRACTIONS:
            (ds_out / s / "images").mkdir(parents=True)
            (ds_out / s / "labels").mkdir(parents=True)

        split_lists = defaultdict(list)
        n_after = boxes_after = 0
        cls_after = Counter()
        cls_after_split = defaultdict(Counter)
        # Images containing a class, as distinct from boxes of that class.
        # Per-class AP is estimated over both, so both belong in the report.
        img_after_split = defaultdict(Counter)
        split_after = Counter()
        scene_split = {}

        for g in groups:
            s = assign[g["gid"]]
            for local in g["locals"]:
                scene_split[scene_gid_by_local[local]] = s
            for i in g["recs"]:
                r = recs[i]
                dst_img = ds_out / s / "images" / r["path"].name
                shutil.copy2(r["path"], dst_img)
                dst_lbl = ds_out / s / "labels" / (r["path"].stem + ".txt")
                lines = [f"{cls} {cx:.8f} {cy:.8f} {width:.8f} {height:.8f}"
                         for cls, cx, cy, width, height in r["boxes"]]
                dst_lbl.write_text("\n".join(lines) + ("\n" if lines else ""))
                rel = f"{s}/images/{r['path'].name}"
                split_lists[s].append(rel)
                n_after += 1
                boxes_after += r["n_boxes"]
                cls_after.update(r["classes"])
                cls_after_split[s].update(r["classes"])
                for present in set(r["classes"]):
                    img_after_split[s][present] += 1
                split_after[s] += 1

        # manifest: every original file, kept or dropped, and why
        kept_indices = {i for g in groups for i in g["recs"]}
        scene_gid_of_rep = {}
        content_gid_of_rep = {}
        split_block_gid_of_rep = {}
        split_of_rep = {}
        for g in groups:
            for local, i in zip(g["locals"], g["recs"]):
                scene_gid_of_rep[i] = scene_gid_by_local[local]
                content_gid_of_rep[i] = content_gid_by_local[local]
                split_block_gid_of_rep[i] = split_block_gid_by_local[local]
                split_of_rep[i] = assign[g["gid"]]
        scene_groups_after = Counter(scene_split.values())

        final_rep_of_rec = {}
        copy_gid_of_rec = {}
        copy_method_of_rec = {}
        copy_score_of_rec = {}
        for final_local, members in copy_members.items():
            final_rec = ok[final_local]
            for local in members:
                src_rep = ok[local]
                for rec_index in src_members[src_rep]:
                    final_rep_of_rec[rec_index] = final_rec
                    copy_gid_of_rec[rec_index] = copy_group_of_local[local]
                    copy_score_of_rec[rec_index] = copy_score_to_rep[local]
                    if rec_index == final_rec:
                        method = "kept"
                    elif local == final_local:
                        method = "filename"
                    elif rec_index == src_rep:
                        method = "ncc"
                    else:
                        method = "filename+ncc"
                    copy_method_of_rec[rec_index] = method

        if len(final_rep_of_rec) != len(recs):
            raise RuntimeError(
                f"{tag}: lineage covers {len(final_rep_of_rec)} of {len(recs)} originals")

        for rec_index, r in enumerate(recs):
            final_rec = final_rep_of_rec[rec_index]
            kept = rec_index in kept_indices
            split = split_of_rep[final_rec]
            rep_rel = str(recs[final_rec]["path"].relative_to(root))
            clean_rel = f"{tag}/{split}/images/{recs[final_rec]['path'].name}"
            method = copy_method_of_rec[rec_index]
            manifest.append({
                "dataset": tag,
                "original_path": str(r["path"].relative_to(root)),
                "original_split": r["orig_split"],
                "source_id": r["source_id"],
                "kept": kept,
                "drop_reason": "" if kept else method,
                "copy_match_method": method,
                "copy_ncc_to_representative": (
                    "" if method in {"kept", "filename"}
                    else f"{copy_score_of_rec[rec_index]:.6f}"),
                "copy_group": f"{tag}:copy:{copy_gid_of_rec[rec_index]:06d}",
                "representative_path": rep_rel,
                "representative_clean_path": clean_rel,
                "content_scene_group": (
                    f"{tag}:content:{content_gid_of_rep[final_rec]:06d}"),
                "scene_group": f"{tag}:scene:{scene_gid_of_rep[final_rec]:06d}",
                "split_block": f"{tag}:block:{split_block_gid_of_rep[final_rec]:06d}",
                "new_split": split,
            })

        for s, lst in split_lists.items():
            lst.sort()
            (out_root / "splits" / f"{tag}_{s}.txt").write_text("\n".join(lst) + "\n")

        # No `path` key, on purpose. Ultralytics resolves the dataset root as
        # `data.get("path") or Path(yaml_file).parent`, and it only falls back
        # to DATASETS_DIR when the path does not exist. `path: .` therefore
        # always exists and always resolves to the current working directory,
        # so training would read train/images relative to wherever it was
        # launched. That fails loudly if nothing is there and, far worse,
        # silently trains on the wrong tree if something is. Omitting the key
        # anchors every split to this file's own folder, which is portable
        # across machines and independent of the working directory.
        (ds_out / "data.yaml").write_text(yaml.safe_dump({
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "nc": len(names),
            "names": names,
        }, sort_keys=False))

        print(f"    after:  {n_after} images, {boxes_after} boxes, "
              f"{dict(split_after)}")
        test_boxes = cls_after_split.get("test", Counter())
        test_imgs = img_after_split.get("test", Counter())
        thin = [names[c] if c < len(names) else str(c)
                for c in range(len(names))
                if test_boxes.get(c, 0) < MIN_TEST_INSTANCES]
        print("    test instances per class: " + ", ".join(
            f"{names[c] if c < len(names) else c}="
            f"{test_boxes.get(c, 0)}"
            f"{'*' if test_boxes.get(c, 0) < MIN_TEST_INSTANCES else ''}"
            f"/{test_imgs.get(c, 0)}img"
            for c in range(len(names))))
        if thin:
            print(f"    !! below {MIN_TEST_INSTANCES} test instances: "
                  f"{', '.join(thin)}. Report bootstrap CIs for these, "
                  f"not seed dispersion.")
        print(f"    reduction: {100 * (1 - n_after / n_before):.1f}% of files\n")

        before_after.append({
            "dataset": tag, "names": names,
            "before": {"images": n_before, "boxes": boxes_before,
                       "per_class": {names[c] if c < len(names) else str(c): n
                                     for c, n in sorted(cls_before.items())},
                       "splits": dict(split_before)},
            "after": {"images": n_after, "boxes": boxes_after,
                      "per_class": {names[c] if c < len(names) else str(c): n
                                    for c, n in sorted(cls_after.items())},
                      "splits": dict(split_after),
                      "per_class_by_split": {
                          s: {names[c] if c < len(names) else str(c): n
                              for c, n in sorted(cls_after_split[s].items())}
                          for s in sorted(cls_after_split)},
                      "per_class_images_by_split": {
                          s: {names[c] if c < len(names) else str(c): n
                              for c, n in sorted(img_after_split[s].items())}
                          for s in sorted(img_after_split)},
                      "scene_groups_by_split": dict(scene_groups_after)},
            "scene_groups": len(scene_groups),
            "largest_scene_group": sizes[0],
            "split_blocks": len(split_blocks),
            "largest_split_block": split_block_sizes[0],
            "copy_groups": len(uniq_local),
            "copy_complete_linkage_groups": len(complete_copy_groups),
            "copy_components_split_by_complete_linkage": n_copy_components_split,
            "copy_input_component_min_pairwise_ncc": round(copy_input_min, 6),
            "content_scene_groups": n_content_groups,
            "direct_scene_candidate_pairs": len(scene_pairs),
            "identity_compatible_scene_candidates": len(compatible_scene_pairs),
            "confirmed_content_scene_pairs": len(content_scene_pairs),
            "scene_components_split_by_complete_linkage": n_split_by_complete,
            "scene_input_component_min_pairwise_ncc": round(scene_input_min, 6),
            "frame_run_links": n_run_links,
            "frame_run_scene_links": n_run_scene_links,
            "frame_run_scope": (
                "scene_and_split" if tag in RUN_SCENE_TAGS else
                "split_only" if tag in RUN_SPLIT_TAGS else "unused"),
            "specimen_links": n_spec_links,
            "split_balance": split_balance,
        })
        report["datasets"][tag] = before_after[-1]

    # -- 6. cross-dataset overlap on the cleaned data --------------------
    print(f"=== cross-dataset overlap check on {out_name}/")
    descs, owner = [], []
    for tag in DATASETS:
        d = out_root / tag
        if not d.is_dir():
            continue
        for s in SPLIT_FRACTIONS:
            for p in sorted((d / s / "images").iterdir()):
                v = descriptor(p)
                if v is not None:
                    descs.append(v)
                    owner.append(tag)
    X = np.vstack(descs).astype(np.float32)
    cross = Counter()
    for a, b, sc in pairs_above(X, T_COPY):
        if owner[a] != owner[b]:
            cross["|".join(sorted((owner[a], owner[b])))] += 1
    if cross:
        for k, n in cross.most_common():
            print(f"    {k}: {n} pairs at NCC >= {T_COPY}")
    else:
        print(f"    none. No image in any dataset is a copy of an image in "
              f"another, at NCC >= {T_COPY}.")
    report["cross_dataset_copies_after"] = dict(cross)

    # -- 7. tables -------------------------------------------------------
    with (out_root / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0]))
        w.writeheader()
        w.writerows(manifest)

    rows = []
    for e in before_after:
        allc = sorted(set(e["before"]["per_class"]) | set(e["after"]["per_class"]))
        for c in allc:
            rows.append({
                "dataset": e["dataset"], "class": c,
                "boxes_before": e["before"]["per_class"].get(c, 0),
                "boxes_after": e["after"]["per_class"].get(c, 0),
            })
    with (out_root / "before_after.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "images_before", "images_after",
                    "boxes_before", "boxes_after", "scene_groups"])
        for e in before_after:
            w.writerow([e["dataset"], e["before"]["images"], e["after"]["images"],
                        e["before"]["boxes"], e["after"]["boxes"],
                        e["scene_groups"]])
        w.writerow([])
        w.writerow(["dataset", "class", "boxes_before", "boxes_after"])
        for r in rows:
            w.writerow([r["dataset"], r["class"], r["boxes_before"], r["boxes_after"]])

    md = ["# Clean rebuild, before and after", "",
          f"Seed {SEED}. Copy threshold NCC {T_COPY}. Scene threshold NCC {T_SCENE}.",
          "Copy representatives are a maximum independent set at NCC 0.96.",
          "Scene links pass NCC 0.90 at both 32 and 128 pixels.",
          "Complete linkage defines the confirmed content scene groups.",
          "Known, different Visual specimen ids override thumbnail similarity.",
          "Direct NCC edges and structural keys form leakage-safe split blocks.",
          "Kaggle video adjacency constrains splits but does not assert one scene.",
          f"Split target {SPLIT_FRACTIONS}, applied to leakage-safe split blocks.",
          "Splits are optimized jointly for image totals, per-class box counts,",
          "per-class image presence, backgrounds, and scene group counts.",
          "", "## Images and boxes", "",
          "| Dataset | Images before | Images after | Boxes before | Boxes after | Scene groups | Largest group |",
          "|---|---|---|---|---|---|---|"]
    for e in before_after:
        md.append(f"| {e['dataset']} | {e['before']['images']} | {e['after']['images']} "
                  f"| {e['before']['boxes']} | {e['after']['boxes']} "
                  f"| {e['scene_groups']} | {e['largest_scene_group']} |")
    md += ["", "## Per-class box counts", "",
           "| Dataset | Class | Boxes before | Boxes after |", "|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['dataset']} | {r['class']} | {r['boxes_before']} "
                  f"| {r['boxes_after']} |")
    # Retention is strongly non-uniform across classes, so the clean set does
    # not carry the shipped set's class mix. Make that visible rather than
    # leaving it implied by the raw counts above.
    md += ["", "## Class distribution shift", "",
           "Share is each class as a percentage of its own dataset's boxes.",
           "Retention is boxes kept over boxes shipped. Read the two share",
           "columns together: a class can lose most of its boxes and still gain",
           "share, if the classes around it lost more.", "",
           "| Dataset | Class | Boxes before | Share before | Boxes after "
           "| Share after | Share change | Retention |",
           "|---|---|---|---|---|---|---|---|"]
    for e in before_after:
        b, a = e["before"]["per_class"], e["after"]["per_class"]
        tb, ta = sum(b.values()), sum(a.values())
        for c in sorted(set(b) | set(a)):
            bb, aa = b.get(c, 0), a.get(c, 0)
            sb = 100 * bb / tb if tb else 0
            sa = 100 * aa / ta if ta else 0
            md.append(f"| {e['dataset']} | {c} | {bb} | {sb:.1f}% | {aa} "
                      f"| {sa:.1f}% | {sa - sb:+.1f} pp "
                      f"| {100 * aa / bb if bb else 0:.1f}% |")
        md.append(f"| **{e['dataset']}** | **all** | **{tb}** | 100% "
                  f"| **{ta}** | 100% | | **{100 * ta / tb if tb else 0:.1f}%** |")

    # Per-class AP must never be read without its sample size beside it.
    md += ["", "## Test set instances per class", "",
           f"Instances are ground-truth boxes in the test split. Images is the "
           f"number of test images containing the class. A class marked **thin** "
           f"has fewer than {MIN_TEST_INSTANCES} test instances: its per-class "
           f"AP is dominated by sampling noise, so report a bootstrap "
           f"confidence interval over test instances for it, not the standard "
           f"deviation over three seeds. Seed dispersion measures training "
           f"variance and says nothing about how few objects were scored.", "",
           "| Dataset | Class | Test instances | Test images | Train instances | Status |",
           "|---|---|---|---|---|---|"]
    thin_total = []
    for e in before_after:
        by = e["after"]["per_class_by_split"]
        imgs = e["after"].get("per_class_images_by_split", {})
        test_c = by.get("test", {})
        train_c = by.get("train", {})
        test_i = imgs.get("test", {})
        for c in e["names"]:
            n = test_c.get(c, 0)
            thin = n < MIN_TEST_INSTANCES
            if thin:
                thin_total.append(f"{e['dataset']}/{c} ({n})")
            md.append(f"| {e['dataset']} | {c} | {n} | {test_i.get(c, 0)} "
                      f"| {train_c.get(c, 0)} | "
                      f"{'**thin**' if thin else 'ok'} |")
    md += ["", (f"Classes below {MIN_TEST_INSTANCES} test instances: "
                + (", ".join(thin_total) if thin_total else "none") + ".")]

    md += ["", "## Split sizes after rebuild", "",
           "| Dataset | train | valid | test | train scenes | valid scenes | test scenes |",
           "|---|---|---|---|---|---|---|"]
    for e in before_after:
        s = e["after"]["splits"]
        g = e["after"]["scene_groups_by_split"]
        md.append(f"| {e['dataset']} | {s.get('train',0)} | {s.get('valid',0)} "
                  f"| {s.get('test',0)} | {g.get('train',0)} | "
                  f"{g.get('valid',0)} | {g.get('test',0)} |")
    md += ["", "## Split balance", "",
           "Deviation is the realised feature share minus its target share,",
           "in percentage points.", "",
           "| Dataset | Maximum absolute deviation | Optimizer |",
           "|---|---|---|"]
    for e in before_after:
        b = e["split_balance"]
        md.append(f"| {e['dataset']} | {b['max_abs_deviation_pp']:.3f} pp "
                  f"| {b['optimizer_message']} |")
    (out_root / "before_after.md").write_text("\n".join(md) + "\n")
    (out_root / "rebuild_report.json").write_text(json.dumps(report, indent=2))

    print(f"\nWrote {out_root}/")
    print("  manifest.csv        every original file and its fate")
    print("  before_after.{csv,md}")
    print("  rebuild_report.json")
    print("  splits/*.txt        frozen, release these")


if __name__ == "__main__":
    main()
