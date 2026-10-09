#!/usr/bin/env python3
"""What the class space correction changed. The project log 6a-iii, Task 2d.

Report only. Reads the superseded `eval__` and `eval_last__` records and the
corrected `eval_fixed__` and `eval_last_fixed__` records written by
`recompute_armb.py`, and prints the change per arm B pair.

The headline table is the one the review asked for: the 8 pairs at k = 250,
YOLO11s, averaged over the three draws, at the PRIMARY `last` endpoint.

Usage:
    python3 grid/arms/damage_table.py
    python3 grid/arms/damage_table.py --endpoint best
    python3 grid/arms/damage_table.py --all-k
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import config as C  # noqa: E402
import config_arms as A  # noqa: E402

import evaluate_arm as EA  # noqa: E402
import recompute_armb as R  # noqa: E402


def mean(values):
    vals = [v for v in values if v is not None]
    return statistics.fmean(vals) if vals else None


def fmt(v, width=8):
    return f"{v:.4f}".rjust(width) if v is not None else "n/a".rjust(width)


def collect(endpoint: str, family: str, k: int | None) -> list[dict]:
    spec = R.ENDPOINTS[endpoint]
    cells = {A.arm_cell_name(c): c for c in A.all_arm_cells()
             if c["arm"] == "armb"}
    rows = {}
    for name, cell in cells.items():
        if cell["family"] != family:
            continue
        if k is not None and cell["k"] != k:
            continue
        old_p = R.RESULTS / f"{spec['old']}{name}.json"
        new_p = R.RESULTS / f"{spec['new']}{name}.json"
        if not old_p.is_file() or not new_p.is_file():
            continue
        old = json.loads(old_p.read_text())
        new = json.loads(new_p.read_text())
        key = (cell["source"], cell["target"], cell["k"])
        bucket = rows.setdefault(key, {
            "source": cell["source"], "target": cell["target"], "k": cell["k"],
            "old_box": [], "new_box": [], "old_img": [], "new_img": [],
            "old_npred": [], "new_npred": [], "draws": 0})
        tkey = cell["target"]
        ob = old["targets"][tkey]["box_level"]
        nb = new["targets"][tkey]["box_level"]
        bucket["old_box"].append(None if ob.get("skipped") else ob.get("map50"))
        bucket["new_box"].append(None if nb.get("skipped") else nb.get("map50"))
        bucket["old_img"].append(old["targets"][tkey]["image_level"].get("ap"))
        bucket["new_img"].append(new["targets"][tkey]["image_level"].get("ap"))
        bucket["old_npred"].append(ob.get("n_pred"))
        bucket["new_npred"].append(nb.get("n_pred"))
        bucket["draws"] += 1
    return list(rows.values())


def render(rows: list[dict], title: str) -> list[tuple]:
    print(f"\n{title}")
    print(f"  {'pair':<34s}{'old box':>9s}{'new box':>9s}{'d box':>9s}"
          f"{'old img':>9s}{'new img':>9s}{'d img':>9s}{'draws':>7s}")
    changes = []
    for r in sorted(rows, key=lambda x: (x["source"], x["target"])):
        ob, nb = mean(r["old_box"]), mean(r["new_box"])
        oi, ni = mean(r["old_img"]), mean(r["new_img"])
        db = None if ob is None or nb is None else nb - ob
        di = None if oi is None or ni is None else ni - oi
        pair = f"{r['source']} to {r['target']}"
        print(f"  {pair:<34s}{fmt(ob,9)}{fmt(nb,9)}{fmt(db,9)}"
              f"{fmt(oi,9)}{fmt(ni,9)}{fmt(di,9)}{r['draws']:>7d}")
        if db is not None:
            changes.append((abs(db), db, pair, "box mAP50", ob, nb))
        if di is not None:
            changes.append((abs(di), di, pair, "image AP", oi, ni))
    return changes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--endpoint", choices=["last", "best"], default="last")
    ap.add_argument("--family", default="yolo11s")
    ap.add_argument("--all-k", action="store_true")
    args = ap.parse_args()

    note = ("PRIMARY endpoint, the project log 6a-ii"
            if args.endpoint == "last" else "secondary endpoint")
    print(f"arm B class space correction, damage report")
    print(f"  endpoint {args.endpoint} ({note}), family {args.family}")
    print(f"  old = source class space, the defect. "
          f"new = target class space, corrected.")

    rows = collect(args.endpoint, args.family, None if args.all_k else 250)
    title = (f"all k, mean over draws and k"
             if args.all_k else f"k = 250, mean over the three draws")
    changes = render(rows, title)

    if not changes:
        print("\nno corrected records found. Run recompute_armb.py first.")
        return

    print("\nlargest changes, by absolute size:")
    for _, d, pair, metric, old, new in sorted(changes, reverse=True)[:6]:
        direction = "UP" if d > 0 else "DOWN"
        print(f"  {pair:<34s}{metric:<12s}{old:.4f} -> {new:.4f}  "
              f"{direction} {abs(d):.4f}")

    top = sorted(changes, reverse=True)[0]
    print(f"\nLARGEST SINGLE CHANGE: {top[2]}, {top[3]}, "
          f"{top[4]:.4f} to {top[5]:.4f}, "
          f"{'up' if top[1] > 0 else 'down'} by {abs(top[1]):.4f}")

    # The two predictions the review made, checked rather than assumed.
    print("\nthe review's two predictions:")
    by_pair = {(r["source"], r["target"]): r for r in rows}
    for src, tgt, expect in [("kaggle_v2", "twoclass", "off exactly zero"),
                             ("kaggle_v2", "visual", "off exactly zero"),
                             ("lohi_weld", "kaggle_v2", "fall")]:
        r = by_pair.get((src, tgt))
        if not r:
            continue
        ob, nb = mean(r["old_box"]), mean(r["new_box"])
        moved = "yes" if ob is not None and nb is not None and abs(nb - ob) > 0 \
            else "no"
        direction = ("up" if nb > ob else "down") if ob is not None else "n/a"
        print(f"  {src} to {tgt}: expected to {expect}. "
              f"box {ob:.4f} -> {nb:.4f}, moved {moved}, {direction}")


if __name__ == "__main__":
    main()
