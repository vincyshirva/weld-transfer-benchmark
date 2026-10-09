#!/usr/bin/env python3
"""
Validation diagnostics for the clean/ rebuild.

Scene groups decide effective sample size, so a group that merges unrelated
welds is a real defect. This script reads the emitted manifest and measures
the exact groups used by the candidate rebuild.

Three link types may build a group, and they need separating before a verdict:

  ncc       a content edge, NCC >= T_SCENE, resolved by COMPLETE linkage so
            every member clears the threshold against every other member
  run       a frame-run edge, two frames whose indices are within RUN_GAP.
            Inspected Weld Surface and 2class runs are continuous welds, so
            chaining is intended there. Kaggle videos pan across multiple
            coupons. Their run links constrain splits but do not assert scenes.
  specimen  two views of one physical plate, from the filename

So the question is not "are groups large" but "does the content graph merge
unrelated scenes". The diagnostic is the group diameter: the minimum pairwise
NCC over all members. Under single linkage the content groups had diameters
as low as 0.736, well under the 0.90 that formed them. Under complete linkage
the minimum is 0.900 by construction, and this script verifies it rather than
assuming it. Run and specimen groups legitimately have low diameters, since a
camera panning a seam does leave frame 1 looking nothing like frame 131.

Usage:
    python3 weld_diagnostics.py [root] [--out clean_candidate]

Outputs:  clean/diagnostics/*.jpg          montages of the largest groups
          clean/diagnostics/group_sizes.csv
          clean/diagnostics/chaining_report.json
"""

from __future__ import annotations

import json
import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from weld_rebuild_clean import (  # noqa: E402
    DATASETS, DisjointSet, RUN_GAP, RUN_SCENE_TAGS, SCENE_CONFIRM_THUMB,
    T_COPY, T_SCENE, THUMB, complete_linkage_groups,
    complete_linkage_pair_groups, confirm_scene_pairs, descriptor,
    descriptor_at, is_augmented_variant, ncc_all_pairs,
    pairs_above, run_key, scan_dataset, select_copy_representatives, specimen_key,
    content_scene_compatible,
)

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parent
TOP_N = 3            # largest groups to render per dataset
MAX_TILES = 36       # tiles per montage, sampled evenly across the group
TILE = 150


def load_font(size, bold=False):
    f = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{f}", size)
    except OSError:
        return ImageFont.load_default()


def build_groups(ds_dir: Path, tag: str):
    """Reproduce the rebuild's grouping, keeping the edges that made it.

    Mirrors weld_rebuild_clean exactly, including complete linkage on the
    content graph and the specimen key. If the two ever drift apart this
    diagnostic stops meaning anything, so keep them in step."""
    recs, names = scan_dataset(ds_dir, tag)

    by_src = defaultdict(list)
    for i, r in enumerate(recs):
        by_src[r["source_id"]].append(i)

    def key(i):
        return (is_augmented_variant(tag, recs[i]["path"].name),
                recs[i]["orig_split"] != "train",
                -recs[i]["n_boxes"],
                recs[i]["path"].name)

    reps = [min(v, key=key) for v in by_src.values()]
    D, ok = [], []
    for i in reps:
        d = descriptor(recs[i]["path"])
        if d is not None:
            D.append(d)
            ok.append(i)
    D = np.vstack(D).astype(np.float32)

    copy_pairs = pairs_above(D, T_COPY)
    uniq_local, _, _, _ = select_copy_representatives(
        D, copy_pairs, priority_key=lambda m: key(ok[m]))

    Du = D[uniq_local]
    ncc_edges = pairs_above(Du, T_SCENE)
    eligible_edges = [
        (left, right, score)
        for left, right, score in ncc_edges
        if content_scene_compatible(
            tag,
            recs[ok[uniq_local[left]]]["path"].name,
            recs[ok[uniq_local[right]]]["path"].name,
        )
    ]
    paths = [recs[ok[local]]["path"] for local in uniq_local]
    confirmed_edges = confirm_scene_pairs(paths, eligible_edges)
    complete_groups, _, _ = complete_linkage_pair_groups(
        len(Du), T_SCENE, confirmed_edges)
    ds_scene = DisjointSet(len(uniq_local))
    content_groups = []
    for members in complete_groups:
        for m in members[1:]:
            ds_scene.union(members[0], m)
        content_groups.append(members)

    runs = defaultdict(list)
    for m, u in enumerate(uniq_local):
        k = run_key(tag, recs[ok[u]]["path"].name)
        if k is not None:
            runs[k[0]].append((k[1], m))
    run_edges = []
    for _, seq in runs.items():
        seq.sort()
        for (i0, m0), (i1, m1) in zip(seq, seq[1:]):
            if i1 - i0 <= RUN_GAP:
                if tag in RUN_SCENE_TAGS:
                    ds_scene.union(m0, m1)
                run_edges.append((m0, m1, i1 - i0))

    spec = defaultdict(list)
    for m, u in enumerate(uniq_local):
        k = specimen_key(tag, recs[ok[u]]["path"].name)
        if k is not None:
            spec[k].append(m)
    spec_edges = []
    for _, members in spec.items():
        for m in members[1:]:
            ds_scene.union(members[0], m)
            spec_edges.append((members[0], m, 0))

    return (recs, names, ok, uniq_local, Du, ds_scene.groups(),
            ncc_edges, run_edges, spec_edges, content_groups)


def diameter(Du, members, thumb=THUMB):
    """Minimum pairwise NCC inside a group, best of 8 dihedral transforms.
    This is the complete-linkage score. A chained group scores far below the
    single-linkage threshold that formed it."""
    if len(members) < 2:
        return 1.0, 1.0
    X = Du[members]
    sq = X.reshape(len(X), thumb, thumb)
    best = None
    for k in range(4):
        r = np.rot90(sq, k, axes=(1, 2))
        for t in (r, r[:, :, ::-1]):
            s = (X @ np.ascontiguousarray(t.reshape(len(X), -1)).T) / (thumb * thumb)
            best = s if best is None else np.maximum(best, s)
    iu = np.triu_indices(len(X), k=1)
    v = best[iu]
    return float(v.min()), float(np.median(v))


def montage(paths, title, subtitle, out_path):
    n = len(paths)
    cols = 6
    rows = (n + cols - 1) // cols
    W = cols * (TILE + 6) + 6
    H = 54 + rows * (TILE + 20)
    sheet = Image.new("RGB", (W, H), (250, 250, 248))
    d = ImageDraw.Draw(sheet)
    d.text((8, 8), title, fill=(20, 24, 28), font=load_font(15, True))
    d.text((8, 30), subtitle, fill=(150, 100, 20), font=load_font(12))
    for k, p in enumerate(paths):
        x = 6 + (k % cols) * (TILE + 6)
        y = 54 + (k // cols) * (TILE + 20)
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                im.thumbnail((TILE, TILE), Image.LANCZOS)
                b = Image.new("RGB", (TILE, TILE), (228, 230, 232))
                b.paste(im, ((TILE - im.width) // 2, (TILE - im.height) // 2))
            sheet.paste(b, (x, y))
        except Exception:
            d.rectangle([x, y, x + TILE, y + TILE], fill=(240, 220, 220))
        stem = Path(p).name.split("_jpg")[0].split("_png")[0][:20]
        d.text((x, y + TILE + 3), stem, fill=(90, 98, 105), font=load_font(10))
    sheet.save(out_path, quality=90)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=ROOT)
    parser.add_argument("--out", default="clean")
    parser.add_argument("--montage-only", action="store_true")
    args = parser.parse_args()
    root = args.root
    candidate = root / args.out
    out = candidate / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)

    report = {"t_scene": T_SCENE, "candidate_thumb": THUMB,
              "confirmation_thumb": SCENE_CONFIRM_THUMB,
              "run_gap": RUN_GAP, "datasets": {}}
    size_rows = [("dataset", "group_size", "n_groups")]
    manifest_path = candidate / "manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")
    rows = [row for row in csv.DictReader(manifest_path.open(newline=""))
            if row["kept"] == "True"]

    for tag in DATASETS:
        dataset_rows = [row for row in rows if row["dataset"] == tag]
        if not dataset_rows:
            continue
        print(f"\n=== {tag}")
        paths = [candidate / row["representative_clean_path"]
                 for row in dataset_rows]
        Du = np.vstack([descriptor(path) for path in paths]).astype(np.float32)
        groups_by_id = defaultdict(list)
        content_by_id = defaultdict(list)
        blocks_by_id = defaultdict(list)
        for index, row in enumerate(dataset_rows):
            groups_by_id[row["scene_group"]].append(index)
            content_by_id[row["content_scene_group"]].append(index)
            blocks_by_id[row["split_block"]].append(index)
        groups = list(groups_by_id.values())
        content_groups = list(content_by_id.values())

        sizes = Counter(len(v) for v in groups)
        print("    group size distribution:")
        line = "      "
        for s in sorted(sizes):
            line += f"{s}:{sizes[s]}  "
            size_rows.append((tag, s, sizes[s]))
        print(line)
        n_img = sum(sizes[s] * s for s in sizes)
        print(f"      {len(groups)} groups over {n_img} images; "
              f"singletons {sizes.get(1,0)}; "
              f"p50 size {int(np.median([len(v) for v in groups]))}; "
              f"max {max(sizes)}")

        # the content-only structure, after complete linkage
        ncc_sizes = Counter(len(v) for v in content_groups)
        print(f"    content groups alone: {len(content_groups)}, "
              f"largest {max(ncc_sizes)}; split blocks {len(blocks_by_id)}")

        ds_rec = {
            "n_groups": len(groups),
            "size_distribution": {str(k): v for k, v in sorted(sizes.items())},
            "content_groups": len(content_groups),
            "ncc_only_largest": int(max(ncc_sizes)),
            "split_blocks": len(blocks_by_id),
            "largest_groups": [],
        }

        big = sorted(groups, key=len, reverse=True)[:TOP_N]
        for rank, members in enumerate(big, 1):
            mn, md = diameter(Du, members)
            group_paths = [paths[m] for m in sorted(members)]
            if len(group_paths) > MAX_TILES:
                idx = np.linspace(0, len(group_paths) - 1, MAX_TILES).astype(int)
                shown = [group_paths[i] for i in idx]
                note = f"showing {MAX_TILES} of {len(group_paths)}, evenly sampled"
            else:
                shown = group_paths
                note = f"all {len(group_paths)}"
            sub_t = (f"size {len(members)}  |  min pairwise NCC32 {mn:.3f}  "
                     f"median {md:.3f}  |  emitted scene group  |  {note}")
            fp = out / f"{tag}_group{rank}_size{len(members)}.jpg"
            montage(shown, f"{tag}  scene group rank {rank}", sub_t, fp)
            print(f"    group {rank}: size {len(members):>4}  min NCC {mn:.3f}  "
                  f"median {md:.3f}  -> {fp.name}")
            ds_rec["largest_groups"].append({
                "rank": rank, "size": len(members),
                "min_pairwise_ncc": round(mn, 4),
                "median_pairwise_ncc": round(md, 4),
                "montage": fp.name,
            })

        # Diameters of every multi-image group built by NCC edges only.
        # Verify both passes rather than trusting the rebuild report.
        pure = [v for v in content_groups if len(v) > 1]
        if pure:
            dias = [diameter(Du, v)[0] for v in pure]
            confirm_dias = []
            for members in pure:
                descriptors = np.vstack([
                    descriptor_at(paths[index], SCENE_CONFIRM_THUMB)
                    for index in members
                ]).astype(np.float32)
                confirm_dias.append(diameter(
                    descriptors, list(range(len(members))),
                    SCENE_CONFIRM_THUMB)[0])
            ds_rec["content_multi_groups"] = len(pure)
            ds_rec["content_diameter_min"] = round(float(np.min(dias)), 4)
            ds_rec["content_confirmation_diameter_min"] = round(
                float(np.min(confirm_dias)), 4)
            ds_rec["content_diameter_p05"] = round(float(np.percentile(dias, 5)), 4)
            ds_rec["content_diameter_median"] = round(float(np.median(dias)), 4)
            below = sum(1 for d in dias if d < T_SCENE)
            confirm_below = sum(1 for d in confirm_dias if d < T_SCENE)
            ds_rec["content_groups_below_threshold"] = below
            ds_rec["content_groups_below_confirmation_threshold"] = confirm_below
            print(f"    content multi-image groups: {len(pure)}, "
                  f"NCC32 min {min(dias):.3f}, "
                  f"NCC128 min {min(confirm_dias):.3f}, "
                  f"below threshold {below}/{confirm_below}")
        report["datasets"][tag] = ds_rec

    with (out / "group_sizes.csv").open("w") as f:
        for r in size_rows:
            f.write(",".join(str(x) for x in r) + "\n")
    (out / "chaining_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out}/")


if __name__ == "__main__":
    main()
