#!/usr/bin/env python3
"""How far a trained checkpoint moved from the one it started at.

THIS MODULE HAS NO IMPORT SIDE EFFECTS, AND THAT IS THE POINT. It is imported
by both `train_arm.py` and `preflight_arms.py`. `train_arm` calls
`config_arms.assert_main_unmoved()` and then `register()` at module scope, and
`assert_main_unmoved` must run BEFORE `register()` in any given process. So the
preflight cannot import `train_arm` to reach these two names without tripping
its own guard. Keeping them here lets both sides share one definition instead
of growing a second copy that can drift.

The project log 6a-viii is the reason this exists. `train_arm.finetune_ultralytics`
passed `pretrained=False` to ultralytics, which means "discard the loaded
weights" rather than "weights already supplied", so all 240 arm B cells trained
from random initialisation. The defect was invisible for the whole run because
nothing ever compared a trained cell against the checkpoint it claimed to start
from.
"""

from __future__ import annotations

from pathlib import Path

# THE BAND THAT MEANS "THE CHECKPOINT WAS DISCARDED". Both ends measured, so
# the threshold is calibrated rather than guessed.
#
#   BROKEN, from the project log 6a-viii
#     void YOLO11s   k=10 output vs its claimed start   0.8820
#     void RT-DETR-l k=10 output vs its claimed start   0.9976
#     FRESH random init from the same YAML vs start     0.9072
#
#   CORRECT, from the 16 September Task 2 smoke cell,
#   armb__kaggle_v2_to_lohi_weld__yolo11s__k10__draw0 retrained with the fix
#     output vs its claimed start                       0.1734
#     output vs the OTHER source's start                0.8114
#     backbone only, shared-shape tensors               0.1733
#
# So a correct fine-tune sits at 0.17 and a discarded checkpoint at 0.88 to
# 1.00. The gap is wide and empty, exactly like the NCC anchors in the
# threshold calibration section of the project log. 0.60 sits in the middle of it,
# about 3.5x the observed correct value and well under the random band.
#
# **Raise it only with a measurement, never to make a run pass.** That is the
# move that turns a guard into decoration.
RANDOM_INIT_L2_FLOOR = 0.60


def state_dict(path: Path) -> dict:
    """Tensors from an ultralytics or torchvision checkpoint."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    return model.state_dict() if hasattr(model, "state_dict") else model


def relative_l2(start: Path, final: Path) -> float | None:
    """sqrt( sum |final - start|^2 / sum |start|^2 ) over shared float tensors.

    Tensors are matched by name AND shape. When the source and target class
    counts differ the detection head has a different shape, so the shape test
    drops it and this measures the BACKBONE and neck. That is deliberate:
    the project log 6a-iv says the head is rebuilt and only the backbone carries
    over, so the backbone is the part whose movement proves the checkpoint was
    used at all.

    A genuine fine-tune of roughly 100 optimisation steps moves this very
    little. A value near 0.9 means the start checkpoint was never loaded.

    Returns None when either file cannot be read, so a measurement failure is
    never silently read as a pass.
    """
    try:
        a, b = state_dict(start), state_dict(final)
    except Exception as exc:                                   # noqa: BLE001
        print(f"    relative_l2 unavailable: {exc}")
        return None
    num = den = 0.0
    for key, v in b.items():
        w = a.get(key)
        if w is None or w.shape != v.shape or not v.is_floating_point():
            continue
        num += (v.float() - w.float()).pow(2).sum().item()
        den += w.float().pow(2).sum().item()
    return None if den == 0 else (num / den) ** 0.5


def verdict(value: float | None) -> str:
    """Plain words for a relative L2, so a log line needs no interpretation."""
    if value is None:
        return "not measured"
    if value > RANDOM_INIT_L2_FLOOR:
        return (f"{value:.4f} RANDOM INIT BAND, the start checkpoint was "
                f"discarded, the project log 6a-viii")
    return f"{value:.4f} fine-tuned from the start checkpoint"
