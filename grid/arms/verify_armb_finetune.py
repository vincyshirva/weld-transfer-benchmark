#!/usr/bin/env python3
"""The three Task 2 checks on the retrained smoke cell. Report only."""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SP = Path("/tmp/weld_scratch")
sys.path.insert(0, str(ROOT / "grid"))
sys.path.insert(0, str(ROOT / "grid" / "arms"))

NAME = "armb__kaggle_v2_to_lohi_weld__yolo11s__k10__draw0"
START = ROOT / "grid_results/armb_start_checkpoints/kaggle_v2__yolo11s__seed0/weights/best.pt"
NEW = ROOT / "runs_arms" / NAME / "weights" / "last.pt"
OTHER = ROOT / "grid_results/armb_start_checkpoints/lohi_weld__yolo11s__seed0/weights/best.pt"

import weight_distance as WD  # noqa: E402

ok = []


def check(n, passed, detail=""):
    print(f"  {'PASS' if passed else 'FAIL'}  {n}")
    if detail:
        print(f"        {detail}")
    ok.append(passed)


print(f"\nTask 2, three checks on {NAME}\n")

# ---- check 1: relative L2 to the start checkpoint is small
print("1. Relative L2 to the start checkpoint")
v = WD.relative_l2(START, NEW)
o = WD.relative_l2(OTHER, NEW)
print(f"     new run   vs its CLAIMED start  : {v:.4f}")
print(f"     new run   vs the OTHER source   : {o:.4f}")
print(f"     void run  vs its claimed start  : 0.8820   (the project log 6a-viii)")
print(f"     fresh random init vs that start : 0.9072   (the project log 6a-viii)")
check("relative L2 is far below the random init band",
      v < WD.RANDOM_INIT_L2_FLOOR,
      f"{v:.4f} against a floor of {WD.RANDOM_INIT_L2_FLOOR}")
check("and it is much closer to its own start than to the other source",
      v < o, f"{v:.4f} against {o:.4f}")

# ---- check 3 (done here, it is cheap): head rebuilt, backbone carried over
print("\n3. Head rebuilt at the target nc, backbone matching the start")
ck = torch.load(NEW, map_location="cpu", weights_only=False)
head_nc = int(ck["model"].model[-1].nc)
sd_new = ck["model"].state_dict()
sd_start = WD.state_dict(START)
same_shape = [k for k in sd_new
              if k in sd_start and sd_new[k].shape == sd_start[k].shape]
diff_shape = [k for k in sd_new
              if k in sd_start and sd_new[k].shape != sd_start[k].shape]
print(f"     head nc                          : {head_nc}  (lohi_weld has 4)")
print(f"     source kaggle_v2 nc              : 3")
print(f"     tensors whose shape CHANGED      : {len(diff_shape)}  (the head)")
print(f"     tensors whose shape is unchanged : {len(same_shape)}")
if diff_shape:
    print(f"     e.g. {diff_shape[:3]}")
check("head width equals the target class count", head_nc == 4,
      f"head_nc={head_nc}, lohi_weld nc=4")
check("some tensors changed shape, so the head WAS rebuilt",
      len(diff_shape) > 0, f"{len(diff_shape)} tensors")

# backbone closeness on the shared-shape tensors only
num = den = 0.0
for k in same_shape:
    a, b = sd_start[k].float(), sd_new[k].float()
    if not b.is_floating_point():
        continue
    num += (b - a).pow(2).sum().item()
    den += a.pow(2).sum().item()
bb = (num / den) ** 0.5 if den else float("nan")
print(f"     backbone relative L2             : {bb:.4f}")
check("the backbone barely moved, so it carried over", bb < 0.25,
      f"{bb:.4f}; about 100 optimisation steps cannot move a loaded "
      f"backbone far")

print("\ncell.json record")
cj = ROOT / "runs_arms" / NAME / "cell.json"
if cj.is_file():
    rec = json.loads(cj.read_text())
    for key in ("head_nc", "target_nc", "relative_l2_from_start",
                "train_seconds"):
        val = rec.get(key)
        if val is None:
            for v2 in rec.values():
                if isinstance(v2, dict) and key in v2:
                    val = v2[key]
                    break
        print(f"     {key:24s} {val}")
else:
    print("     (no cell.json)")

print()
print("CHECKS 1 AND 3 " + ("ALL PASS" if all(ok) else "HAVE A FAILURE"))
print("Check 2, box mAP50 on lohi_weld, needs the evaluator. Run next.")
sys.exit(0 if all(ok) else 1)
