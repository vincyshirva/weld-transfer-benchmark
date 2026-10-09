#!/usr/bin/env python3
"""Train one arm cell, reusing the frozen trainers wherever the recipe is the
same.

TWO PATHS, AND THE SPLIT BETWEEN THEM IS THE RECIPE, NOT CONVENIENCE.

  from scratch   lohi_control and armd. Identical recipe to the main grid, so
                 these call `train_cell.train_ultralytics` and
                 `train_cell.train_torchvision` DIRECTLY. There is no second
                 copy of the training code, so it cannot drift.
  fine-tune      armb. A genuinely different recipe, pre-registered in
                 the project log 6a: 50 epochs, batch 8, lr0 one tenth of frozen,
                 starting from a main grid checkpoint rather than from COCO.
                 That cannot reuse the frozen trainer, because the frozen
                 trainer always starts from the pinned COCO checkpoint.

`C.data_yaml` is monkeypatched to `config_arms.arm_data_yaml` for the life of
this process. That function is a strict superset: it returns the same path as
the original for anything under clean_candidate, and the per-cell k-shot folder
for armb. config.py on disk is untouched.

Usage:
    python3 grid/arms/train_arm.py --index 0
    python3 grid/arms/train_arm.py --index 0 --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import config as C  # noqa: E402
import config_arms as A  # noqa: E402
import kshot as K  # noqa: E402
import train_cell as T  # noqa: E402
# No import side effects, so both this module and preflight_arms can hold it.
from weight_distance import (  # noqa: E402
    RANDOM_INIT_L2_FLOOR, relative_l2, verdict)

A.assert_main_unmoved()
A.register()
# Superset of the original: same answer for every clean_candidate dataset,
# plus the per-cell k-shot folder for armb.
C.data_yaml = A.arm_data_yaml


def prepare_kshot(cell: dict, out: Path) -> dict:
    """Materialise this cell's k-shot draw and assert it is disjoint."""
    drawn = K.sample(cell["target"], cell["k"], cell["draw"])
    K.assert_disjoint(drawn)
    K.materialise(drawn, out / "kshot")
    return drawn


def start_checkpoint(cell: dict, runs_dir: Path) -> Path:
    """The main grid cell an armb cell fine-tunes from.

    Same configuration and same seed as the cell, which is what makes
    R_ij(k) = (P_ij(k) - P_ij(0)) / (P_jj - P_ij(0)) a within-seed quantity.
    P_ij(0) is this very checkpoint's zero-shot score on the target, already
    measured in the main grid, so nothing is retrained to supply it.
    """
    name = C.cell_name({"source": cell["source"], "family": cell["family"],
                        "seed": cell["seed"], "kind": "main"})
    weights = runs_dir / name / "weights" / "best.pt"
    if not weights.is_file():
        raise SystemExit(
            f"arm B needs the main grid checkpoint {name}, expected at "
            f"{weights}. Restore it from grid_results/ before running arm B.")
    return weights


def finetune_ultralytics(cell: dict, out: Path, smoke: bool) -> dict:
    """Arm B. PRE-REGISTERED recipe, fixed across every k and every target.

    Nothing here may be selected using any target test split. The validation
    set is the target's own frozen VALID split, which is the only choice that
    is neither the shots nor the test set.
    """
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path.home() / ".ultralytics_cfg"))
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)

    import ultralytics.engine.trainer as _trainer
    _trainer.check_amp = lambda *_a, **_k: True
    from ultralytics import RTDETR, YOLO

    cls = RTDETR if cell["family"].startswith("rtdetr") else YOLO
    model = cls(str(cell["_start_weights"]))

    hp = dict(C.ULTRALYTICS_HP)
    # THIS POP IS LOAD BEARING. C.ULTRALYTICS_HP carries `pretrained: False`
    # for the main grid, which filters it out the same way in train_cell.py.
    # If it reached `args` through `**hp` it would defeat the start checkpoint
    # exactly as the explicit keyword used to. See the block below.
    hp.pop("pretrained", None)
    hp["lr0"] = A.ARMB_LR0
    hp["warmup_epochs"] = A.ARMB_WARMUP_EPOCHS

    # `pretrained` IS DELIBERATELY ABSENT FROM THESE ARGS. Do not add it back,
    # in either direction. The project log 6a-viii: the previous version passed
    # `pretrained=False` with the comment "weights come from the main
    # checkpoint", and that flag means the opposite of that comment. Verified
    # against ultralytics 8.4.138 `engine/model.py`:
    #
    #   839  pretrained = kwargs.get("pretrained",
    #                                overrides.get("pretrained", True)
    #                                if kwargs.get("cfg") else True)
    #   854  weights = None if pretrained is False else self.model
    #   857  self.trainer.model = self.trainer.get_model(weights=weights,
    #                                                    cfg=self.model.yaml)
    #
    # No `cfg` key is passed here, so line 839 falls to `True`, line 854 keeps
    # `self.model`, and line 857 loads the checkpoint opened at `cls(...)`
    # above. `BaseTrainer.setup_model` carries the same `is False` branch.
    # Omitting the key is what `train_cell.py` does, which is why the 69 main
    # cells were never affected.
    args = dict(
        data=str(A.arm_data_yaml(cell)),
        imgsz=C.IMGSZ,
        batch=T.SMOKE_BATCH if smoke else A.ARMB_BATCH,
        epochs=2 if smoke else A.ARMB_EPOCHS,
        workers=2 if smoke else C.WORKERS,
        patience=A.ARMB_EPOCHS,
        seed=cell["seed"], device=0,
        project=str(out.parent), name=out.name, exist_ok=True,
        plots=False, save=True, val=True, verbose=False,
        **hp,
    )
    if "pretrained" in args:
        raise SystemExit(
            "`pretrained` reached the arm B training args. In ultralytics it "
            "means 'discard the loaded weights', not 'weights already "
            "supplied'. The project log 6a-viii. Refusing to train from random "
            "initialisation again.")
    t0 = time.perf_counter()
    model.train(**args)
    seconds = round(time.perf_counter() - t0, 1)

    # 1c. THE HEAD WIDTH, MEASURED RATHER THAN ASSUMED.
    # The project log 6a-iv's whole claim is that the detection head is rebuilt at
    # the TARGET's class count while the backbone carries over, and 6a-iii's
    # evaluator correction depends on it. Neither was ever asserted. Record
    # both counts and fail here rather than let a silent mismatch reach a
    # metric, which is exactly how 6a-iii survived 240 cells.
    head_nc = int(model.model.model[-1].nc)
    target_nc = len(A.dataset_entry(cell["target"])["classes"])
    if head_nc != target_nc:
        raise SystemExit(
            f"arm B head width {head_nc} does not match target "
            f"{cell['target']} class count {target_nc}. The project log 6a-iv. "
            f"Refusing to record a cell whose class space is not the one the "
            f"evaluator will filter with.")

    # THE WEIGHTS MUST HAVE MOVED FROM THE START CHECKPOINT, NOT AWAY FROM IT.
    # This is the 6a-viii assertion at the one place that can see both ends.
    moved = relative_l2(Path(cell["_start_weights"]),
                        out / "weights" / "last.pt")
    print(f"    backbone movement from start: {verdict(moved)}")
    if moved is not None and moved > RANDOM_INIT_L2_FLOOR:
        raise SystemExit(
            f"arm B output sits {moved:.4f} from its start checkpoint in "
            f"relative L2, at or above the {RANDOM_INIT_L2_FLOOR} band that "
            f"means the checkpoint was DISCARDED and training began from "
            f"random initialisation. The project log 6a-viii. A correct fine-tune "
            f"lands far below this.")

    return {"train_seconds": seconds,
            "weights": str(out / "weights" / "best.pt"),
            "head_nc": head_nc, "target_nc": target_nc,
            "relative_l2_from_start": moved}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--runs-dir", type=Path, default=A.ARM_RUNS)
    ap.add_argument("--main-runs-dir", type=Path, default=C.RUNS,
                    help="where the main grid checkpoints live, for arm B")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cells = A.all_arm_cells()
    if not 0 <= args.index < len(cells):
        sys.exit(f"index must be 0..{len(cells) - 1}")
    cell = cells[args.index]
    name = A.arm_cell_name(cell)
    out = args.runs_dir / ("smoke_" + name if args.smoke else name)
    out.mkdir(parents=True, exist_ok=True)

    done = out / "cell.json"
    if done.is_file() and not args.smoke:
        print(f"already complete, skipping: {name}")
        return

    print(f"=== arm cell {args.index}: {name} ===")
    print(f"    arm {cell['arm']}, source {cell['source']}, "
          f"family {cell['family']}, seed {cell['seed']}")

    T.set_all_seeds(cell["seed"])
    extra = {}

    if cell["arm"] == "armb":
        drawn = prepare_kshot(cell, out)
        cell["_start_weights"] = str(start_checkpoint(cell, args.main_runs_dir))
        extra["kshot"] = {k: v for k, v in drawn.items() if k != "images"}
        extra["kshot_images"] = drawn["images"]
        extra["start_checkpoint"] = cell["_start_weights"]
        print(f"    k={cell['k']} draw={cell['draw']} -> "
              f"{drawn['n_images']} images, {drawn['n_scenes']} scenes"
              + ("  (train split exhausted)"
                 if drawn["exhausted_train_split"] else ""))
        print(f"    fine-tuning from {cell['_start_weights']}")
        result = finetune_ultralytics(cell, out, args.smoke)
        pretrained = None
    else:
        print(f"    data {A.arm_data_yaml(cell)}")
        pretrained = T.verified_pretrained(cell, args.smoke)
        backend = C.FAMILIES[cell["family"]]["backend"]
        fn = T.train_ultralytics if backend == "ultralytics" else T.train_torchvision
        result = fn(cell, out, args.smoke)

    backend = C.FAMILIES[cell["family"]]["backend"]
    record = {
        "cell": {k: v for k, v in cell.items() if not k.startswith("_")},
        "name": name,
        "arm": cell["arm"],
        "backend": backend,
        "smoke": args.smoke,
        # BOTH fingerprints. The arm one embeds the frozen main one, so an arm
        # cell becomes un-re-evaluable if either moves.
        "arm_sha256_16": A.arm_fingerprint(),
        "frozen_main_fingerprint": A.FROZEN_MAIN_FINGERPRINT,
        "split_hashes": json.loads(
            (C.CANDIDATE / "split_hashes.json").read_text()),
        "arm_split_hashes": json.loads(
            (A.ARMS_ROOT / "arm_split_hashes.json").read_text())["datasets"],
        "code_sha256": T.code_fingerprints(),
        "arm_code_sha256": {p.name: T.file_sha256(p)
                            for p in sorted(HERE.glob("*.py"))},
        "pretrained_checkpoint": pretrained,
        "hyperparameters": {
            "common": C.COMMON, "pretrained": C.PRETRAINED,
            "family": C.ULTRALYTICS_HP if backend == "ultralytics"
                      else C.TORCHVISION_HP,
            "armb_overrides": ({"epochs": A.ARMB_EPOCHS,
                                "batch": A.ARMB_BATCH,
                                "lr0": A.ARMB_LR0,
                                "warmup_epochs": A.ARMB_WARMUP_EPOCHS}
                               if cell["arm"] == "armb" else None),
        },
        "environment": {
            "python": platform.python_version(),
            "host": platform.node(),
            "slurm_job": os.environ.get("SLURM_JOB_ID"),
        },
        **extra,
        **result,
    }
    try:
        import torch
        record["environment"]["torch"] = torch.__version__
        record["environment"]["torch_cuda"] = torch.version.cuda
        if backend == "ultralytics":
            import ultralytics
            record["environment"]["ultralytics"] = ultralytics.__version__
        record["environment"]["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass

    done.write_text(json.dumps(record, indent=2))
    print(f"    trained in {result['train_seconds']} s")
    print(f"    wrote {done}")


if __name__ == "__main__":
    main()
