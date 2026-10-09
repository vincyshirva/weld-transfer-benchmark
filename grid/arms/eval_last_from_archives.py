#!/usr/bin/env python3
"""Score the `last.pt` endpoint for every pulled arm B archive. Laptop side.

WHY THIS RUNS ON THE LAPTOP AND NOT ON THE CLUSTER.

The project log 6a-ii makes `last.pt` the primary arm B endpoint. The amendment was
written after the chain was already running, and the running chain must not be
disturbed: `evaluate_arm.py` still defaults to `best`, so the cluster keeps
writing `eval__` exactly as before. The `last` endpoint is therefore produced
here, from the archives the watcher pulls, using the SAME evaluator and the same
guards. No arm cell is retrained and no cluster job changes.

WHAT IT DOES, per arm B archive that has no `eval_last__` yet:

  1. extract it under grid_results_arms/extracted/
  2. run evaluate_arm.py --weights last against that cell directory
  3. leave eval_last__<cell>.json and raw_last__<cell>.npz in results/
  4. delete the extracted checkpoint, which is the bulky part

Idempotent and safe to rerun as archives arrive, which is the point: it is meant
to be driven by the same 30 minute cron entry as the watcher.

Usage:
    python3 grid/arms/eval_last_from_archives.py
    python3 grid/arms/eval_last_from_archives.py --limit 5
    python3 grid/arms/eval_last_from_archives.py --keep-extracted
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import config as C  # noqa: E402
import config_arms as A  # noqa: E402

DEFAULT_RESULTS = C.PROJECT / "grid_results_arms" / "results"


def cell_index_by_name() -> dict[str, int]:
    A.assert_main_unmoved()
    A.register()
    return {A.arm_cell_name(c): i for i, c in enumerate(A.all_arm_cells())}


def metric_pair(path: Path) -> str:
    """image AP and box mAP50 per target, from an evaluation json."""
    if not path.is_file():
        return "n/a"
    rec = json.loads(path.read_text())
    bits = []
    for key, entry in rec.get("targets", {}).items():
        ap = entry.get("image_level", {}).get("ap")
        m50 = entry.get("box_level", {}).get("map50")
        bits.append(f"{key} AP={'n/a' if ap is None else f'{ap:.4f}'} "
                    f"box={'skip' if m50 is None else f'{m50:.4f}'}")
    return "  ".join(bits) if bits else "n/a"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--extract-root", type=Path,
                    default=C.PROJECT / "grid_results_arms" / "extracted")
    ap.add_argument("--limit", type=int, help="stop after this many cells")
    ap.add_argument("--keep-extracted", action="store_true",
                    help="leave the checkpoints on disk, for debugging")
    args = ap.parse_args()

    index_of = cell_index_by_name()
    archives = sorted(args.results_dir.glob("armb__*.tar.gz"))
    if not archives:
        print(f"no arm B archives under {args.results_dir}. Nothing to do.")
        print("The watcher pulls them as the chain produces them: "
              "./grid/fetch_results.sh --arms")
        return

    todo = [a for a in archives
            if not (args.results_dir / f"eval_last__{a.name[:-len('.tar.gz')]}.json").is_file()]
    print(f"{len(archives)} arm B archives, {len(todo)} without a `last` "
          f"evaluation")

    done, failed = 0, []
    for archive in todo[:args.limit] if args.limit else todo:
        name = archive.name[: -len(".tar.gz")]
        if name not in index_of:
            failed.append((name, "not in the arm cell registry"))
            continue
        idx = index_of[name]
        cell_dir = args.extract_root / name
        try:
            args.extract_root.mkdir(parents=True, exist_ok=True)
            # EXTRACT ONLY WHAT IS NEEDED, which is cell.json and last.pt.
            #
            # Two reasons, and the first is a hard error rather than an
            # optimisation. The archive also holds the cell's `kshot/` folder,
            # whose images are SYMLINKS with absolute targets under the cluster
            # home. Python's `data` tar filter rejects an absolute link
            # outright with AbsoluteLinkError, and rightly so. Those links
            # would be dangling here in any case, and nothing in the `last`
            # evaluation reads them: the k-shot draw was a training input and
            # the draw manifest is already recorded inside cell.json.
            #
            # Second, it is much cheaper. A whole arm B archive is tens of MB
            # of checkpoint, and only one of the two checkpoints is wanted.
            wanted = {f"{name}/cell.json", f"{name}/weights/{'last'}.pt"}
            with tarfile.open(archive) as tar:
                members = [m for m in tar.getmembers() if m.name in wanted]
                missing = wanted - {m.name for m in members}
                if missing:
                    failed.append((name, f"archive lacks {sorted(missing)}"))
                    continue
                tar.extractall(args.extract_root, members=members,
                               filter="data")

            proc = subprocess.run(
                [sys.executable, str(HERE / "evaluate_arm.py"),
                 "--index", str(idx),
                 "--cell-dir", str(cell_dir),
                 "--out-dir", str(args.results_dir),
                 "--weights", "last"],
                capture_output=True, text=True)
            if proc.returncode != 0:
                tail = (proc.stderr.strip().splitlines() or ["(no stderr)"])[-1]
                failed.append((name, tail[:120]))
                continue

            best = metric_pair(args.results_dir / f"eval__{name}.json")
            last = metric_pair(args.results_dir / f"eval_last__{name}.json")
            print(f"\n{name}")
            print(f"   best (secondary, target valid split used to select): {best}")
            print(f"   last (PRIMARY, no target label used to select)     : {last}")
            done += 1
        finally:
            if not args.keep_extracted and cell_dir.exists():
                shutil.rmtree(cell_dir, ignore_errors=True)

    print(f"\nscored `last` for {done} cells")
    if failed:
        print(f"{len(failed)} failed:")
        for name, why in failed:
            print(f"  {name}: {why}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
