#!/usr/bin/env python3
"""
Convert LoHi-WELD to YOLO detection layout.

The set already ships a `.yolo` file beside every image, so this is mostly a
reorganisation. It is still worth doing properly, because the shipped `.yolo`
files carry no class-name mapping. That mapping is recovered here by pairing
each JSON shape against the corresponding `.yolo` line, then asserting the
pairing is consistent. The check runs per variant, so a disagreement between
the two acquisitions cannot be averaged away.

Recovered mapping, verified with zero contradictions in both variants:

    0 pore    1 deposit    2 discontinuity    3 stain

BOTH VARIANTS ARE CONVERTED. They are separate acquisitions, not two exports
of one set. Evidence, from `lohi_lowres_probe.py`: they share no filename, and
comparing all 2,044,000 cross pairs by the accepted two-stage NCC method gives
a maximum of 0.6546 at 32 pixels and 0.6129 at 128 pixels. T_copy is 0.96 and
T_scene is 0.90, so nothing comes close to either. 1,022 plus 2,000 is the
3,022 the IEEE Access description reports.

The two carry different structure, and the converter records it:

  high resolution  <plate>_<index>, about 780 x 190, one image per cycle
  low resolution   <date>_<time>_<cycle>_CAM1_<NG|OK>_<n>, about 198 x 41,
                   exactly two images per cycle over 1,000 cycles

The two images of one low resolution cycle are two views of one welded part.
They score about 0.32 against each other, so content will never link them, and
without the cycle key they would be free to straddle a split. That key is
applied in `weld_rebuild_clean.py` through SPECIMEN_KEYS.

The supplied k-folds are ignored on purpose, since this study does its own
leakage-audited splitting. Everything is written to a new folder. The original
is opened read-only.

Usage:
    python3 lohi_to_yolo.py [root] [--high-only]

Outputs:  lohi_weld_yolo/train/{images,labels}
          lohi_weld_yolo/classes.yaml          class names only, no splits
          lohi_weld_yolo/conversion_report.json
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent
BASE = "LoHi-WELD weld-dataset/weld-dataset"
VARIANTS = {
    "high": "high_resolution_welds",
    "low": "low_resolution_welds",
}
DST = "lohi_weld_yolo"

CLASS_NAMES = ["pore", "deposit", "discontinuity", "stain"]

RE_LOW_CYCLE = re.compile(r"^(\d{6}_\d{6}_\d+_CAM\d+)_(NG|OK)_(\d+)$")


def verify_mapping(jsons: list[Path]) -> tuple[dict[str, int], list[str]]:
    """Recover label -> class id by pairing JSON shapes with .yolo lines."""
    observed: dict[str, Counter] = defaultdict(Counter)
    bad = []
    for jf in jsons:
        j = json.loads(jf.read_text())
        lines = [l.split() for l in jf.with_suffix(".yolo").read_text().splitlines()
                 if l.strip()]
        labels = [s["label"] for s in j["shapes"]]
        if len(lines) != len(labels):
            bad.append(jf.name)
            continue
        for lab, ln in zip(labels, lines):
            observed[lab][int(ln[0])] += 1
    mapping = {}
    for lab, ctr in observed.items():
        if len(ctr) != 1:
            sys.exit(f"Label {lab!r} maps to several class ids: {dict(ctr)}. "
                     "Refusing to guess.")
        cid = next(iter(ctr))
        if CLASS_NAMES[cid] != lab:
            sys.exit(f"Recovered mapping {lab}={cid} contradicts CLASS_NAMES "
                     f"{CLASS_NAMES}. Refusing to write a wrong data.yaml.")
        mapping[lab] = cid
    return mapping, bad


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    high_only = "--high-only" in sys.argv[1:]
    root = Path(args[0]) if args else ROOT
    dst = root / DST
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "train" / "images").mkdir(parents=True)
    (dst / "train" / "labels").mkdir(parents=True)

    wanted = ["high"] if high_only else ["high", "low"]
    print(f"Converting variants: {', '.join(wanted)}\n")

    report = {"variants": {}, "class_names": CLASS_NAMES}
    seen_stems: dict[str, str] = {}
    total_img = total_box = 0
    all_per_class = Counter()

    for variant in wanted:
        src = root / BASE / VARIANTS[variant]
        if not src.is_dir():
            sys.exit(f"Not found: {src}")
        jsons = sorted(src.glob("*.json"))
        print(f"=== {variant} resolution  ({len(jsons)} annotation files)")

        mapping, bad = verify_mapping(jsons)
        for lab, cid in sorted(mapping.items(), key=lambda x: x[1]):
            print(f"    verified  {cid} = {lab}")
        if bad:
            print(f"    !! {len(bad)} files where JSON and .yolo disagree, skipped")

        n_img = n_box = 0
        per_class = Counter()
        sizes = Counter()
        areas = []
        bpi = []
        cycles = defaultdict(list)

        for jf in jsons:
            stem = jf.stem
            if jf.name in bad:
                continue
            img, yolo = src / f"{stem}.jpg", src / f"{stem}.yolo"
            if not img.exists() or not yolo.exists():
                continue
            if stem in seen_stems:
                sys.exit(f"Stem collision: {stem} in both {seen_stems[stem]} "
                         f"and {variant}. Refusing to overwrite.")
            seen_stems[stem] = variant

            shutil.copy2(img, dst / "train" / "images" / f"{stem}.jpg")
            out = []
            for p in (l.split() for l in yolo.read_text().splitlines() if l.strip()):
                cid = int(p[0])
                cx, cy, w, h = (float(v) for v in p[1:5])
                out.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                per_class[cid] += 1
                areas.append(w * h)
                n_box += 1
            (dst / "train" / "labels" / f"{stem}.txt").write_text(
                "\n".join(out) + ("\n" if out else ""))
            bpi.append(len(out))
            with Image.open(img) as im:
                sizes[im.size] += 1
            m = RE_LOW_CYCLE.match(stem)
            if m:
                cycles[m.group(1)].append(stem)
            n_img += 1

        tot = sum(per_class.values())
        rec = {
            "source": str(src),
            "images": n_img,
            "boxes": n_box,
            "boxes_per_image": round(float(np.mean(bpi)), 3),
            "median_box_area_frac": round(float(np.median(areas)), 6),
            "distinct_resolutions": len(sizes),
            "most_common_resolution": list(sizes.most_common(1)[0][0]),
            "per_class": {CLASS_NAMES[c]: n for c, n in sorted(per_class.items())},
            "per_class_share": {CLASS_NAMES[c]: round(100 * n / tot, 1)
                                for c, n in sorted(per_class.items())},
            "production_cycles": len(cycles) if cycles else n_img,
            "images_per_cycle": (dict(Counter(len(v) for v in cycles.values()))
                                 if cycles else {"1": n_img}),
            "skipped_count_mismatch": bad,
        }
        report["variants"][variant] = rec
        total_img += n_img
        total_box += n_box
        all_per_class.update(per_class)

        print(f"    images {n_img}, boxes {n_box}, "
              f"{rec['boxes_per_image']} boxes per image")
        print(f"    {len(sizes)} distinct resolutions, most common "
              f"{rec['most_common_resolution']}")
        print(f"    median box area {100 * rec['median_box_area_frac']:.2f}%")
        print(f"    class share {rec['per_class_share']}")
        print(f"    production cycles {rec['production_cycles']}, "
              f"images per cycle {rec['images_per_cycle']}\n")

    # Deliberately NOT data.yaml, and deliberately without train, val or test
    # keys. An earlier version wrote a data.yaml pointing all three splits at
    # train/images. Training from that would have reported test scores on the
    # training set, which is a total leak and exactly the failure this project
    # exists to measure. The name also keeps the folder out of the
    # rglob("data.yaml") discovery used by weld_audit.py and
    # weld_inspect_pairs.py. Splits live in the rebuilt tree, not here.
    (dst / "classes.yaml").write_text(yaml.safe_dump({
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
        "role": "conversion staging only, not a training configuration",
        "splits": "see clean_candidate/splits/lohi_weld_{train,valid,test}.txt",
    }, sort_keys=False))
    (dst / "DO_NOT_TRAIN_FROM_THIS_FOLDER.txt").write_text(
        "This folder is the LoHi-WELD conversion staging area.\n\n"
        "Every image sits under train/. There is no valid or test split here,\n"
        "and there is no data.yaml, on purpose. Training from this folder\n"
        "would evaluate on the training set.\n\n"
        "Use the rebuilt tree instead:\n"
        "    clean_candidate/lohi_weld/data.yaml\n"
        "    clean_candidate/splits/lohi_weld_{train,valid,test}.txt\n")

    report["total"] = {
        "images": total_img, "boxes": total_box,
        "per_class": {CLASS_NAMES[c]: n for c, n in sorted(all_per_class.items())},
    }
    (dst / "conversion_report.json").write_text(json.dumps(report, indent=2))

    print(f"TOTAL images {total_img}, boxes {total_box}")
    if len(wanted) > 1:
        h = report["variants"]["high"]["per_class_share"]
        l = report["variants"]["low"]["per_class_share"]
        print("\nThe two acquisitions do not share a defect mix. Merging them "
              "puts a real\ndomain shift inside one source, which is worth "
              "saying in the paper:")
        for c in CLASS_NAMES:
            print(f"    {c:<14} high {h.get(c, 0):>5.1f}%   low {l.get(c, 0):>5.1f}%")
    print(f"\nWrote {dst}/")
    print("No data.yaml is written here, on purpose. This folder is a "
          "conversion staging\narea with every image under train/. Train from "
          "clean_candidate/lohi_weld instead.")


if __name__ == "__main__":
    main()
