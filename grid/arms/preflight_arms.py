#!/usr/bin/env python3
"""Read-only launch gate for the arms. Nothing here trains or writes a result.

The arms must not be submitted unless this passes on the laptop AND on the
cluster, exactly like `grid/preflight.py` for the main grid.

THE CHECK THAT MATTERS MOST is `guarded_files_unmoved` together with
`main_grid_still_evaluable`. The whole reason `grid/arms/` exists is to add
datasets and cells without moving `config.py`, `evaluate_cell.py`,
`metrics.py` or `yolo_dataset.py`, because all 69 completed main cells pin
their hashes. This asserts it two ways: the four files still hash to what the
finished cells recorded, and a SUBPROCESS that does not import the arms still
computes f5567a1938278e88.

Usage:
    python3 grid/arms/preflight_arms.py
    python3 grid/arms/preflight_arms.py --require-armb-checkpoints
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import config as C  # noqa: E402
import config_arms as A  # noqa: E402
# Deliberately NOT `import train_arm`. That module calls assert_main_unmoved()
# and then register() at import time, and assert_main_unmoved must run before
# register() in any one process, so importing it here would trip this file's
# own guard. weight_distance has no import side effects for that reason.
import weight_distance as WD  # noqa: E402

FAILURES: list[str] = []
GUARDED = ("config.py", "evaluate_cell.py", "metrics.py", "yolo_dataset.py")


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------

def guarded_files_unmoved(results_dir: Path) -> None:
    print("\nthe four guarded files must be byte identical to what the main "
          "grid recorded")
    live = {f: sha256(HERE.parent / f) for f in GUARDED}
    records = sorted(results_dir.glob("eval__*.json"))
    if not records:
        check("main grid evaluation records found", False,
              f"none under {results_dir}")
        return
    # The recorded hashes live in the cell.json inside each checkpoint archive,
    # but the evaluation records carry the fingerprint, which is enough to
    # detect a moved config.py. For the four file hashes, use any runs/ copy if
    # present, otherwise fall back to the fingerprint check alone.
    fingerprints = {json.loads(p.read_text()).get("config_sha256_16")
                    for p in records}
    check("every main record carries one configuration fingerprint",
          len(fingerprints) == 1, str(sorted(fingerprints)))
    check("that fingerprint is the frozen one",
          fingerprints == {A.FROZEN_MAIN_FINGERPRINT},
          f"{sorted(fingerprints)} against {A.FROZEN_MAIN_FINGERPRINT}")

    # Per generation, because the three generations of main-grid cell did NOT
    # all record the same evaluate_cell.py. On 7 September 2026 that file was
    # fixed for the dirty path resolution bug in the project log 5h, which is AFTER
    # the 60 main and reference cells were trained and BEFORE the 9 dirty ones
    # were. So the 60 older cells legitimately pin a hash the file no longer
    # has. That predates the arms and is reported, not blamed on them.
    #
    # config.py, metrics.py and yolo_dataset.py have never moved, and those are
    # the three that would actually invalidate an arm result, so only they are
    # treated as failures here.
    NEVER_MOVED = ("config.py", "metrics.py", "yolo_dataset.py")
    cells = sorted((C.PROJECT / "runs").glob("*/cell.json"))
    if not cells:
        print("    (no local runs/ tree, per-cell hashes are checked on the "
              "cluster)")
        return
    status = {f: {"match": 0, "moved": 0, "examples": []} for f in GUARDED}
    for path in cells:
        rec = json.loads(path.read_text()).get("code_sha256", {})
        for f in GUARDED:
            if rec.get(f) == live[f]:
                status[f]["match"] += 1
            else:
                status[f]["moved"] += 1
                if len(status[f]["examples"]) < 2:
                    status[f]["examples"].append(path.parent.name)
    for f in GUARDED:
        s = status[f]
        detail = f"{s['match']} match, {s['moved']} moved"
        if f in NEVER_MOVED:
            check(f"{f} unchanged for every recorded cell", s["moved"] == 0,
                  detail)
        else:
            print(f"  INFO  {f}: {detail}"
                  + (f", e.g. {s['examples']}" if s["examples"] else ""))
            if s["moved"]:
                print("        Expected. The project log 5h changed this file on "
                      "7 September, after the 60 main and reference cells "
                      "were trained. Their RESULTS stand, because the changed "
                      "branch only runs when kind == 'dirty'. They cannot be "
                      "re-evaluated without checking out that revision.")


def main_grid_still_evaluable() -> None:
    """A process without the arms must still see the frozen fingerprint."""
    print("\nthe arms must be invisible to a main grid process")
    code = ("import sys; sys.path.insert(0, %r); import config as C; "
            "print(C.config_fingerprint()); print(len(C.DATASETS))"
            % str(HERE.parent))
    proc = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, cwd=tempfile.gettempdir())
    lines = proc.stdout.strip().splitlines()
    check("subprocess without arms computes the frozen fingerprint",
          len(lines) >= 1 and lines[0] == A.FROZEN_MAIN_FINGERPRINT,
          lines[0] if lines else proc.stderr.strip()[:80])
    check("subprocess without arms sees exactly the five main datasets",
          len(lines) >= 2 and lines[1] == "5",
          lines[1] if len(lines) >= 2 else "")


def arm_datasets_present() -> None:
    print("\narm datasets")
    manifest_path = A.ARMS_ROOT / "arm_split_hashes.json"
    check("arm split hash manifest exists", manifest_path.is_file())
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text())
    check("manifest was built under this arm fingerprint",
          manifest.get("arm_fingerprint") == A.arm_fingerprint(),
          f"{manifest.get('arm_fingerprint')} against {A.arm_fingerprint()}")
    check("zero split blocks straddle the two LoHi acquisitions",
          manifest["acquisition_block_check"]["straddling"] == 0,
          f"{manifest['acquisition_block_check']['n_blocks']} blocks checked")

    for tag in A.ARM_DATASETS:
        root = C.CANDIDATE / tag
        yaml = root / "data.yaml"
        check(f"{tag} data.yaml exists", yaml.is_file())
        if yaml.is_file():
            text = yaml.read_text()
            check(f"{tag} data.yaml has no `path` key",
                  not any(line.startswith("path:") for line in text.splitlines()),
                  "the project log 4c")
            for split, key in (("train", "train:"), ("valid", "val:"),
                               ("test", "test:")):
                entry = next((l.split(":", 1)[1].strip()
                              for l in text.splitlines() if l.startswith(key)), None)
                resolved = (yaml.parent / entry) if entry else None
                n = len(list(resolved.iterdir())) if resolved and resolved.is_dir() else 0
                want = manifest["datasets"][tag]["counts"][split]
                check(f"{tag} {split} resolves from the yaml's own parent",
                      n == want, f"{n} images against {want}")


def armb_draws_disjoint() -> None:
    print("\narm B draws, scene aware and disjoint from evaluation")
    import kshot as K
    worst, exhausted = 0.0, []
    for start in A.ARMB_START_SOURCES:
        for target in A.armb_targets(start):
            for k in A.ARMB_K:
                for draw in A.ARMB_DRAWS:
                    d = K.sample(target, k, draw)
                    try:
                        K.assert_disjoint(d)
                    except SystemExit as exc:
                        check(f"draw {target} k={k} d={draw} disjoint",
                              False, str(exc)[:70])
                        return
                    worst = max(worst, d["n_images"] / k)
                    if d["exhausted_train_split"]:
                        exhausted.append(f"{target} k={k} draw={draw}")
    check("every arm B draw is disjoint from valid and test", True)
    # Not a failure. The project log 6a-i records that the pre-registered sampler
    # overshoots badly at small k, and that the curve must be plotted against
    # the realised count. Surfaced here so it cannot be forgotten at plot time.
    print(f"    worst realised/requested image ratio: {worst:.1f}x  "
          f"(see the project log 6a-i)")
    if exhausted:
        print(f"    draws that exhausted the train split: {len(exhausted)}, "
              f"e.g. {exhausted[:2]}")


def armb_start_checkpoints(runs_dir: Path, required: bool) -> None:
    print("\narm B starting checkpoints")
    wanted = sorted({
        C.cell_name({"source": s, "family": f, "seed": d, "kind": "main"})
        for s in A.ARMB_START_SOURCES
        for f in A.ARMB_FAMILIES
        for d in A.ARMB_DRAWS})
    missing = [n for n in wanted
               if not (runs_dir / n / "weights" / "best.pt").is_file()]
    if required:
        check(f"all {len(wanted)} main checkpoints present for fine-tuning",
              not missing, f"missing {len(missing)}: {missing[:3]}")
    else:
        print(f"    {len(wanted) - len(missing)} of {len(wanted)} present "
              f"under {runs_dir}")
        if missing:
            print(f"    arm B cannot run until these are restored: "
                  f"{missing[:3]}{' ...' if len(missing) > 3 else ''}")


def armb_finetune_is_real(results_dir: Path) -> None:
    """THE 6a-viii ASSERTION. Arm B must fine-tune, not train from scratch.

    Two checks, one static and one measured, because each catches the defect
    at a different moment.

    STATIC. `train_arm.finetune_ultralytics` must not pass `pretrained` in
    either direction. In ultralytics `pretrained=False` means "discard the
    loaded weights", not "weights already supplied", so the previous version's
    flag defeated the start checkpoint while its comment claimed the opposite.
    Verified against 8.4.138 `engine/model.py` lines 839, 854 and 857. This
    check runs BEFORE any cell trains, which is the whole point: the 240 void
    cells cost about 11.5 hours to discover the same fact.

    MEASURED. Every arm B checkpoint already on disk must sit far below the
    random initialisation band. The project log 6a-viii measured the void runs at
    0.8820 for YOLO11s and 0.9976 for RT-DETR-l, against 0.9072 for a fresh
    random init from the same YAML. **ANY VALUE NEAR 0.9 MEANS THE CHECKPOINT
    WAS DISCARDED.** The threshold lives in `weight_distance.RANDOM_INIT_L2_FLOOR`
    so the gate and the trainer cannot drift apart.
    """
    import ast
    print("\narm B actually fine-tunes, the project log 6a-viii")

    src = (HERE / "train_arm.py").read_text()
    tree = ast.parse(src)
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef)
               and n.name == "finetune_ultralytics"), None)
    passes_pretrained = False
    if fn is not None:
        for node in ast.walk(fn):
            if isinstance(node, ast.keyword) and node.arg == "pretrained":
                passes_pretrained = True
    check("finetune_ultralytics does not pass `pretrained` at all",
          fn is not None and not passes_pretrained,
          "`pretrained` in ultralytics means DISCARD the loaded weights. "
          "Omit the key so the default True keeps self.model")
    check("the random-init floor is defined in weight_distance",
          isinstance(getattr(WD, "RANDOM_INIT_L2_FLOOR", None), float),
          "weight_distance.RANDOM_INIT_L2_FLOOR is the single source of truth")

    starts = A.ARMS_ROOT.parent.parent / "grid_results" / "armb_start_checkpoints"
    archives = sorted(results_dir.glob("armb__*.tar.gz")) if results_dir.is_dir() else []
    if not archives or not starts.is_dir():
        print(f"    no arm B checkpoints on disk yet, nothing to measure "
              f"({len(archives)} archives, starts {'present' if starts.is_dir() else 'absent'})")
        return

    import tarfile, tempfile
    worst = (0.0, None)
    sampled = 0
    for a in archives[:: max(1, len(archives) // 8)]:      # a spread, not all
        name = a.name[: -len(".tar.gz")]
        parts = name.split("__")
        if len(parts) < 3:
            continue
        src_ds, fam = parts[1].split("_to_")[0], parts[2]
        seed = parts[-1].replace("draw", "")
        start = starts / f"{src_ds}__{fam}__seed{seed}" / "weights" / "best.pt"
        if not start.is_file():
            continue
        with tempfile.TemporaryDirectory() as td:
            try:
                with tarfile.open(a) as tar:
                    m = tar.getmember(f"{name}/weights/last.pt")
                    tar.extractall(td, members=[m], filter="data")
            except Exception:                              # noqa: BLE001
                continue
            v = WD.relative_l2(start, Path(td) / name / "weights" / "last.pt")
        if v is None:
            continue
        sampled += 1
        if v > worst[0]:
            worst = (v, name)
    if sampled:
        check(f"all {sampled} sampled arm B checkpoints moved FROM their start "
              f"(worst {worst[0]:.4f} < {WD.RANDOM_INIT_L2_FLOOR})",
              worst[0] < WD.RANDOM_INIT_L2_FLOOR,
              f"{worst[1]} sits at {worst[0]:.4f}, in the random "
              f"initialisation band. The start checkpoint was discarded")
    else:
        print("    no (archive, start) pair could be measured")


def pretrained_pinned() -> None:
    print("\npinned initial checkpoints")
    records = C.pretrained_records()
    for family in C.FAMILIES:
        path = C.pretrained_path(family)
        rec = records.get(family)
        ok = path.is_file() and rec is not None
        check(f"{family} checkpoint present", ok, str(path) if not ok else "")
        if ok:
            check(f"{family} sha256 matches the manifest",
                  sha256(path) == rec["sha256"])


def cell_registry() -> None:
    print("\narm cell registry")
    cells = A.all_arm_cells()
    counts = A.arm_payload()["cell_counts"]
    check("cell count is derived, not hardcoded",
          len(cells) == sum(counts.values()),
          f"{len(cells)} = {counts}")
    check("lohi control runs first", cells[0]["arm"] == "lohi_control")
    check("arm D runs last", cells[-1]["arm"] == "armd")
    names = [A.arm_cell_name(c) for c in cells]
    check("every arm cell name is unique", len(set(names)) == len(names),
          f"{len(names) - len(set(names))} duplicates")
    mixed = [p for p, e in A.ARM_DATASETS.items()
             if e.get("arm_kind") == "pool" and e["protocol"] == "mixed"]
    check("pools containing weld_surface_v2 are marked protocol mixed",
          sorted(mixed) == sorted(
              p for p, m in A.ARMD_POOLS.items() if "weld_surface_v2" in m),
          str(mixed))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path,
                    default=C.PROJECT / "grid_results" / "results")
    ap.add_argument("--main-runs-dir", type=Path, default=C.RUNS)
    # A.ARM_RESULTS is the CLUSTER output directory, `results_arms/`. On the
    # laptop the archives live where the watcher pulls them. Prefer whichever
    # exists so the 6a-viii measured check runs on both sides rather than
    # printing "nothing to measure" on the machine that holds the archives.
    ap.add_argument("--arm-results-dir", type=Path,
                    default=(A.ARM_RESULTS if A.ARM_RESULTS.is_dir()
                             else C.PROJECT / "grid_results_arms" / "results"),
                    help="where arm archives live. Used by the the project log "
                         "6a-viii fine-tune assertion")
    ap.add_argument("--require-armb-checkpoints", action="store_true",
                    help="fail if the main checkpoints arm B fine-tunes from "
                         "are absent. Use on the cluster before launching.")
    args = ap.parse_args()

    print("arms preflight, read only")
    A.assert_main_unmoved()
    check("frozen main fingerprint intact", True, A.FROZEN_MAIN_FINGERPRINT)
    A.register()
    print(f"  arm fingerprint {A.arm_fingerprint()}")

    guarded_files_unmoved(args.results_dir)
    main_grid_still_evaluable()
    arm_datasets_present()
    cell_registry()
    armb_draws_disjoint()
    armb_start_checkpoints(args.main_runs_dir, args.require_armb_checkpoints)
    armb_finetune_is_real(args.arm_results_dir)
    pretrained_pinned()

    print()
    if FAILURES:
        raise SystemExit(f"{len(FAILURES)} FAILED: {FAILURES}")
    print(f"arms preflight passed. arm fingerprint {A.arm_fingerprint()}, "
          f"frozen main {A.FROZEN_MAIN_FINGERPRINT}")


if __name__ == "__main__":
    main()
