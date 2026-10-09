#!/usr/bin/env python3
"""
Box geometry of the audited tree, over the defect classes alone, written to a
released file so the manuscript can read it at full precision.

WHY THIS FILE EXISTS. Table 3.2, three sentences of Section 4.2 and one of
Section 4.9 quote box geometry: boxes per image, median box area as a percent
of the frame, the defect mix of each LoHi-WELD acquisition, and the training
box counts behind the kaggle_v2 diagnosis. Those figures were computed once,
recorded in the project log 5c and 7 at two decimals, and typed into the builder.
No released file carried them. So the build could not check them, and a ratio
the builder computed from them, the "107 times" in Table 3.2's caption, was a
ratio of two rounded values: 19.25 over 0.18. At full precision it is 105.
Found 24 September 2026 in the double-rounding sweep.

WHAT IT MEASURES. Every image in the frozen split lists of each audited source
counts, including an image with no defect box, because boxes per image is a
statement about the source. A box counts when its class is a defect class in
grid/config.py: neither excluded from the ground truth nor a verdict class.
Area is width times height in normalised YOLO coordinates, times 100. A
polygon label counts as its bounding box, which is what the evaluator scores.
The LoHi-WELD acquisitions are separated with config.LOHI_VERDICT_PATTERN,
which matches the low-resolution file names, as grid/arms does.

Reads clean_candidate/splits and the audited labels. Writes
clean_candidate/box_geometry.json and nothing else. No timestamp is written,
so a rerun on unchanged data gives identical bytes.

Usage: python3 box_geometry.py [project root]
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

SPLITS = ("train", "valid", "test")
OUT_NAME = "box_geometry.json"


def frozen_config(root: Path):
    """grid/config.py, imported read only. Importing does not change its
    hash, and bytecode is not written, so no pinned file moves."""
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / "grid"))
    import config as C  # noqa: E402
    return C


def vocab(name: str) -> str:
    """'Burn Through-BT' is 'Burn Through': the suffix is a label code."""
    head, _, tail = name.rpartition("-")
    return head if head and tail.isupper() and len(tail) <= 3 else name


def boxes(label: Path):
    """(class id, area in percent of the frame) for every line of a label."""
    if not label.exists():
        return
    for line in label.read_text().splitlines():
        v = line.split()
        if not v:
            continue
        xs = [float(x) for x in v[1:]]
        if len(xs) == 4:
            w, h = xs[2], xs[3]
        else:
            w = max(xs[0::2]) - min(xs[0::2])
            h = max(xs[1::2]) - min(xs[1::2])
        yield int(v[0]), w * h * 100


def summarise(n_images: int, areas: list[float]) -> dict:
    return {"images": n_images, "defect_boxes": len(areas),
            "boxes_per_image": len(areas) / n_images,
            "median_box_area_percent": statistics.median(areas)}


def main(root: Path) -> None:
    C = frozen_config(root)
    cand = root / "clean_candidate"
    low_res = re.compile(C.LOHI_VERDICT_PATTERN)
    out = {"generated_by": "box_geometry.py",
           "defect_classes_only": True,
           "area_unit": "percent of the frame, width times height",
           "datasets": {}, "lohi_acquisitions": {}}
    for tag in ("weld_surface_v2", "kaggle_v2", "lohi_weld", "visual",
                "twoclass"):
        d = C.DATASETS[tag]
        names = d["classes"]
        keep = {i for i, c in enumerate(names)
                if c not in d["excluded"] and c not in d["non_defect"]}
        n_images, areas = 0, []
        train_boxes, train_images_with = 0, 0
        acq = {"high": [0, [], {}], "low": [0, [], {}]}
        for split in SPLITS:
            lst = cand / "splits" / f"{tag}_{split}.txt"
            for entry in lst.read_text().split():
                n_images += 1
                label = (cand / tag / entry.replace("images/", "labels/", 1)
                         ).with_suffix(".txt")
                mine = [(c, a) for c, a in boxes(label) if c in keep]
                areas += [a for _, a in mine]
                if split == "train":
                    train_boxes += len(mine)
                    train_images_with += bool(mine)
                if tag == "lohi_weld":
                    key = "low" if low_res.match(Path(entry).stem) else "high"
                    acq[key][0] += 1
                    acq[key][1] += [a for _, a in mine]
                    for c, _ in mine:
                        cls = vocab(names[c])
                        acq[key][2][cls] = acq[key][2].get(cls, 0) + 1
        row = summarise(n_images, areas)
        row["train_defect_boxes"] = train_boxes
        row["train_images_with_a_defect_box"] = train_images_with
        out["datasets"][tag] = row
        if tag == "lohi_weld":
            for key, (n, a, counts) in acq.items():
                r = summarise(n, a)
                r["class_share_percent"] = {
                    c: 100 * k / len(a) for c, k in sorted(counts.items())}
                out["lohi_acquisitions"][key] = r
            total = sum(v["images"] for v in out["lohi_acquisitions"].values())
            if total != n_images:
                raise SystemExit(f"LoHi acquisitions hold {total} images and "
                                 f"the source {n_images}.")
    path = cand / OUT_NAME
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {path}")
    for tag, r in out["datasets"].items():
        print(f"  {tag}: {r['boxes_per_image']:.4f} boxes per image, median "
              f"{r['median_box_area_percent']:.4f} percent, "
              f"{r['defect_boxes']} boxes over {r['images']} images")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve() if len(sys.argv) > 1
         else Path(__file__).resolve().parent)
