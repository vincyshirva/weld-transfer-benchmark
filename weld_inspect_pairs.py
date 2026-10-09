#!/usr/bin/env python3
"""
Duplicate-pair inspection sheets.

Stage 1b of the cross-dataset study. weld_audit.py told you HOW MANY pairs
were flagged. This tells you WHETHER TO BELIEVE THEM, by rendering sampled
pairs side by side so you can set the Hamming threshold with your eyes
instead of by assumption.

Pairs are sampled stratified by Hamming distance (0, 1-2, 3-5, 6-8, 9-12),
so you can see exactly where real duplicates stop and coincidental
look-alikes begin. Weld images share a dark ground and a bright bead, so
collisions at larger distances are expected and must be ruled out.

Usage:
    python3 weld_inspect_pairs.py                     # uses ROOT below
    python3 weld_inspect_pairs.py /path/to/datasets

Requires:  pip install pillow numpy pyyaml
Outputs:   ./audit_out/pairs/*.jpg   contact sheets, one per category
           ./audit_out/pairs/sampled_pairs.csv   with a blank verdict column
           ./audit_out/hashes.json   cache, so reruns are fast
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None

# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
OUT = Path("./audit_out")
PAIRS_DIR = OUT / "pairs"

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Distance buckets to sample. Look at every bucket. The threshold you should
# use is the largest distance at which pairs still look like the same photo.
BUCKETS = [(0, 0), (1, 2), (3, 5), (6, 8), (9, 12)]
MAX_DISTANCE = 12

SAMPLES_PER_BUCKET = 6      # pairs shown per category per distance bucket
THUMB = 260                 # px per image in the sheet
# ----------------------------------------------------------------------


def dhash(path: Path, size: int = 8) -> int | None:
    try:
        with Image.open(path) as im:
            im = im.convert("L").resize((size + 1, size), Image.LANCZOS)
            a = np.asarray(im, dtype=np.int16)
    except Exception:
        return None
    out = 0
    for bit in (a[:, 1:] > a[:, :-1]).flatten():
        out = (out << 1) | int(bit)
    return out


def find_datasets(root: Path) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    for y in sorted(root.rglob("data.yaml")):
        d = y.parent
        if any(d != o and o in d.parents for _, o in found):
            continue
        name = str(d.relative_to(root))
        parts = Path(name).parts
        collapsed = [p for i, p in enumerate(parts) if i == 0 or p != parts[i - 1]]
        found.append(("/".join(collapsed), d))
    return found


def load_or_build_hashes(root: Path) -> dict[str, tuple[str, str]]:
    """path -> (dataset, split), plus a parallel hash dict. Cached to disk."""
    cache_file = OUT / "hashes.json"
    files = []
    for name, ds_dir in find_datasets(root):
        for split in ("train", "valid", "val", "test"):
            img_dir = ds_dir / split / "images"
            if not img_dir.is_dir():
                continue
            for p in sorted(img_dir.iterdir()):
                if p.suffix.lower() in IMG_EXT:
                    files.append((name, split, p))
    fingerprint = hashlib.sha256()
    for name, split, path in files:
        stat = path.stat()
        fingerprint.update(str(path.resolve()).encode())
        fingerprint.update(name.encode())
        fingerprint.update(split.encode())
        fingerprint.update(str(stat.st_size).encode())
        fingerprint.update(str(stat.st_mtime_ns).encode())
    cache_key = fingerprint.hexdigest()

    if cache_file.exists():
        raw = json.loads(cache_file.read_text())
        if (isinstance(raw, dict) and raw.get("schema_version") == 2 and
                raw.get("root") == str(root.resolve()) and
                raw.get("file_fingerprint") == cache_key):
            print(f"Loading validated hash cache from {cache_file}")
            return {k: (v[0], v[1], int(v[2]))
                    for k, v in raw["entries"].items()}
        print(f"Ignoring stale or legacy hash cache at {cache_file}")

    print("Building hash cache (first run only)...")
    table: dict[str, tuple[str, str, int]] = {}
    counts = defaultdict(int)
    for name, split, p in files:
        h = dhash(p)
        if h is not None:
            table[str(p)] = (name, split, h)
            counts[(name, split)] += 1
    for (name, split), count in sorted(counts.items()):
        print(f"  {name} / {split}: {count}")
    OUT.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps({
        "schema_version": 2,
        "root": str(root.resolve()),
        "file_fingerprint": cache_key,
        "entries": {k: [a, b, c] for k, (a, b, c) in table.items()},
    }))
    print(f"Cached {len(table)} hashes to {cache_file}")
    return table


def category(a: tuple[str, str], b: tuple[str, str]) -> tuple[str, str]:
    """Returns (category_key, human_label)."""
    (dsa, spa), (dsb, spb) = a, b
    if dsa == dsb:
        if spa == spb:
            return f"within__{dsa}__{spa}", f"WITHIN SPLIT  {dsa}  [{spa}]"
        tag = "+".join(sorted((spa, spb)))
        return f"crosssplit__{dsa}__{tag}", f"CROSS SPLIT  {dsa}  [{tag}]"
    tag = " vs ".join(sorted((dsa, dsb)))
    return f"crossdata__{tag}", f"CROSS DATASET  {tag}"


def safe(s: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in "-_+" else "_" for c in s)
    digest = hashlib.sha256(s.encode()).hexdigest()[:10]
    return f"{cleaned[:96]}__{digest}"


def bucket_of(d: int) -> tuple[int, int] | None:
    for lo, hi in BUCKETS:
        if lo <= d <= hi:
            return (lo, hi)
    return None


def build_sheet(label: str, rows: list[tuple[int, str, str]], out_path: Path) -> None:
    """rows = [(distance, pathA, pathB), ...] grouped and already ordered."""
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        head = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 17)
    except OSError:
        font = head = ImageFont.load_default()

    pad, strip, gap, top = 10, 34, 14, 46
    row_h = THUMB + strip + gap
    width = pad * 2 + THUMB * 2 + gap
    height = top + row_h * len(rows) + pad

    sheet = Image.new("RGB", (width, height), (250, 250, 248))
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, 12), label, fill=(20, 24, 28), font=head)
    draw.line([(pad, top - 8), (width - pad, top - 8)], fill=(200, 205, 208))

    for i, (dist, pa, pb) in enumerate(rows):
        y = top + i * row_h
        for j, p in enumerate((pa, pb)):
            x = pad + j * (THUMB + gap)
            try:
                with Image.open(p) as im:
                    im = im.convert("RGB")
                    im.thumbnail((THUMB, THUMB), Image.LANCZOS)
                    box = Image.new("RGB", (THUMB, THUMB), (228, 230, 232))
                    box.paste(im, ((THUMB - im.width) // 2,
                                   (THUMB - im.height) // 2))
                sheet.paste(box, (x, y))
            except Exception:
                draw.rectangle([x, y, x + THUMB, y + THUMB], fill=(240, 220, 220))
            draw.text((x, y + THUMB + 3), Path(p).name[:44],
                      fill=(90, 98, 105), font=font)
            draw.text((x, y + THUMB + 18), Path(p).parts[-3][:44],
                      fill=(140, 148, 155), font=font)

        draw.text((pad, y - 15), f"hamming distance = {dist}",
                  fill=(150, 100, 20), font=font)

    sheet.save(out_path, quality=88)


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT
    if not root.is_dir():
        sys.exit(f"Not a directory: {root}")
    PAIRS_DIR.mkdir(parents=True, exist_ok=True)

    table = load_or_build_hashes(root)
    paths = list(table)
    meta = {p: (table[p][0], table[p][1]) for p in paths}
    vals = np.array([table[p][2] for p in paths], dtype=np.uint64)

    by_hash: dict[int, list[int]] = defaultdict(list)
    for i, v in enumerate(vals):
        by_hash[int(v)].append(i)
    uniq = np.array(sorted(by_hash), dtype=np.uint64)
    print(f"\n{len(paths)} images, {len(uniq)} unique hashes")

    # (category, bucket) -> sampled rows
    samples: dict[tuple[str, tuple[int, int]], list] = defaultdict(list)
    labels: dict[str, str] = {}

    def offer(i: int, j: int, dist: int) -> None:
        b = bucket_of(dist)
        if b is None:
            return
        key, label = category(meta[paths[i]], meta[paths[j]])
        labels[key] = label
        bucket = samples[(key, b)]
        if len(bucket) < SAMPLES_PER_BUCKET:
            bucket.append((dist, paths[i], paths[j]))

    # distance 0: exact hash collisions
    for idxs in by_hash.values():
        if len(idxs) > 1:
            for a in range(len(idxs) - 1):
                offer(idxs[a], idxs[a + 1], 0)

    print(f"Comparing {len(uniq)} unique hashes for near matches...")
    bits = np.unpackbits(uniq.view(np.uint8).reshape(len(uniq), 8), axis=1)
    step = 2048
    for a in range(0, len(uniq), step):
        blk = bits[a:a + step]
        d = (blk[:, None, :] != bits[None, :, :]).sum(axis=2)
        ii, jj = np.where((d >= 1) & (d <= MAX_DISTANCE))
        for x, y in zip(ii, jj):
            gi, gj = a + int(x), int(y)
            if gi >= gj:
                continue
            offer(by_hash[int(uniq[gi])][0], by_hash[int(uniq[gj])][0],
                  int(d[x, y]))

    # -------------------------------------------------------------- output
    per_cat: dict[str, list] = defaultdict(list)
    for (key, b), rows in samples.items():
        per_cat[key].extend(sorted(rows))

    csv_rows = []
    print()
    for key, rows in sorted(per_cat.items()):
        rows.sort(key=lambda r: r[0])
        out_path = PAIRS_DIR / f"{safe(key)}.jpg"
        build_sheet(labels[key], rows, out_path)
        print(f"  {out_path.name}  ({len(rows)} pairs)")
        for dist, pa, pb in rows:
            csv_rows.append([labels[key], dist, pa, pb, ""])

    csv_path = PAIRS_DIR / "sampled_pairs.csv"
    if csv_path.exists():
        with csv_path.open(newline="") as old_handle:
            old_rows = list(csv.DictReader(old_handle))
        if any(row.get("verdict_same_or_different", "").strip() for row in old_rows):
            csv_path = PAIRS_DIR / "sampled_pairs_regenerated.csv"
            print("  Preserving the reviewed sampled_pairs.csv; writing regenerated rows "
                  "to sampled_pairs_regenerated.csv")
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "hamming_distance", "image_a", "image_b",
                    "verdict_same_or_different"])
        w.writerows(csv_rows)

    print(f"\nWrote {len(csv_rows)} sampled pairs to {csv_path}")
    print("""
How to read the sheets
  Open them largest-distance-first. Find the distance at which pairs stop
  being the same photograph and start being two different welds that merely
  look alike. Set the audit threshold one below that.

  Expect distance 0 to 2 to be genuine duplicates or augmented copies.
  Expect the cross-dataset sheets to be where false positives appear, since
  two unrelated weld photos share a dark ground and a bright bead. If the
  Visual vs Weld Surface sheet shows different welds, that 200-pair count is
  a collision artifact and those datasets do not actually overlap.

  Record your call in the verdict column of sampled_pairs.csv. That file is
  the evidence for the threshold you report in the methodology.""")


if __name__ == "__main__":
    main()
