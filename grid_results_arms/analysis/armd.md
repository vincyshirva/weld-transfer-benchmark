# Arm D, leave one dataset out

| pool | protocol | target | role | cfg | image AP | base rate | AP - base | box mAP50 |
|---|---|---|---|---|---|---|---|---|
| pool_no_kaggle_v2 | mixed | kaggle_v2 | held out training source | Faster R-CNN | 0.5688 (0.025) | 0.5106 | 0.0582 | undefined |
| pool_no_kaggle_v2 | mixed | kaggle_v2 | held out training source | RT-DETR-l | 0.5905 (0.019) | 0.5106 | 0.0798 | undefined |
| pool_no_kaggle_v2 | mixed | kaggle_v2 | held out training source | YOLO11n | 0.6231 (0.030) | 0.5106 | 0.1125 | undefined |
| pool_no_kaggle_v2 | mixed | kaggle_v2 | held out training source | YOLO11s | 0.6097 (0.024) | 0.5106 | 0.0991 | undefined |
| pool_no_kaggle_v2 | mixed | twoclass | held out test only target | Faster R-CNN | 0.8544 (0.018) | 0.8000 | 0.0544 | undefined |
| pool_no_kaggle_v2 | mixed | twoclass | held out test only target | RT-DETR-l | 0.8773 (0.031) | 0.8000 | 0.0773 | undefined |
| pool_no_kaggle_v2 | mixed | twoclass | held out test only target | YOLO11n | 0.8608 (0.009) | 0.8000 | 0.0608 | undefined |
| pool_no_kaggle_v2 | mixed | twoclass | held out test only target | YOLO11s | 0.8415 (0.028) | 0.8000 | 0.0415 | undefined |
| pool_no_kaggle_v2 | mixed | visual | held out test only target | Faster R-CNN | 0.9166 (0.006) | 0.8929 | 0.0237 | undefined |
| pool_no_kaggle_v2 | mixed | visual | held out test only target | RT-DETR-l | 0.8950 (0.004) | 0.8929 | 0.0021 | undefined |
| pool_no_kaggle_v2 | mixed | visual | held out test only target | YOLO11n | 0.9006 (0.011) | 0.8929 | 0.0078 | undefined |
| pool_no_kaggle_v2 | mixed | visual | held out test only target | YOLO11s | 0.8974 (0.008) | 0.8929 | 0.0045 | undefined |
| pool_no_lohi_weld | mixed | lohi_weld | held out training source | Faster R-CNN | 0.9860 (0.001) | 0.9757 | 0.0103 | undefined |
| pool_no_lohi_weld | mixed | lohi_weld | held out training source | RT-DETR-l | 0.9820 (0.004) | 0.9757 | 0.0063 | undefined |
| pool_no_lohi_weld | mixed | lohi_weld | held out training source | YOLO11n | 0.9916 (0.003) | 0.9757 | 0.0159 | undefined |
| pool_no_lohi_weld | mixed | lohi_weld | held out training source | YOLO11s | 0.9949 (0.002) | 0.9757 | 0.0192 | undefined |
| pool_no_lohi_weld | mixed | twoclass | held out test only target | Faster R-CNN | 0.8199 (0.014) | 0.8000 | 0.0199 | undefined |
| pool_no_lohi_weld | mixed | twoclass | held out test only target | RT-DETR-l | 0.8568 (0.019) | 0.8000 | 0.0568 | undefined |
| pool_no_lohi_weld | mixed | twoclass | held out test only target | YOLO11n | 0.8823 (0.027) | 0.8000 | 0.0823 | undefined |
| pool_no_lohi_weld | mixed | twoclass | held out test only target | YOLO11s | 0.8416 (0.007) | 0.8000 | 0.0416 | undefined |
| pool_no_lohi_weld | mixed | visual | held out test only target | Faster R-CNN | 0.9610 (0.015) | 0.8929 | 0.0681 | undefined |
| pool_no_lohi_weld | mixed | visual | held out test only target | RT-DETR-l | 0.9376 (0.024) | 0.8929 | 0.0447 | undefined |
| pool_no_lohi_weld | mixed | visual | held out test only target | YOLO11n | 0.9318 (0.006) | 0.8929 | 0.0390 | undefined |
| pool_no_lohi_weld | mixed | visual | held out test only target | YOLO11s | 0.9244 (0.017) | 0.8929 | 0.0315 | undefined |
| pool_no_weld_surface_v2 | localisation | twoclass | held out test only target | Faster R-CNN | 0.8452 (0.031) | 0.8000 | 0.0452 | 0.0454 (0.015) |
| pool_no_weld_surface_v2 | localisation | twoclass | held out test only target | RT-DETR-l | 0.8616 (0.030) | 0.8000 | 0.0616 | 0.0462 (0.033) |
| pool_no_weld_surface_v2 | localisation | twoclass | held out test only target | YOLO11n | 0.8546 (0.040) | 0.8000 | 0.0546 | 0.1332 (0.045) |
| pool_no_weld_surface_v2 | localisation | twoclass | held out test only target | YOLO11s | 0.8497 (0.029) | 0.8000 | 0.0497 | 0.0444 (0.036) |
| pool_no_weld_surface_v2 | localisation | visual | held out test only target | Faster R-CNN | 0.9531 (0.010) | 0.8929 | 0.0602 | 0.0055 (0.002) |
| pool_no_weld_surface_v2 | localisation | visual | held out test only target | RT-DETR-l | 0.9329 (0.021) | 0.8929 | 0.0400 | 0.0110 (0.004) |
| pool_no_weld_surface_v2 | localisation | visual | held out test only target | YOLO11n | 0.9314 (0.005) | 0.8929 | 0.0385 | 0.0070 (0.003) |
| pool_no_weld_surface_v2 | localisation | visual | held out test only target | YOLO11s | 0.9276 (0.019) | 0.8929 | 0.0348 | 0.0103 (0.006) |
| pool_no_weld_surface_v2 | localisation | weld_surface_v2 | held out training source | Faster R-CNN | 0.9067 (0.005) | 0.8607 | 0.0460 | undefined |
| pool_no_weld_surface_v2 | localisation | weld_surface_v2 | held out training source | RT-DETR-l | 0.9136 (0.016) | 0.8607 | 0.0529 | undefined |
| pool_no_weld_surface_v2 | localisation | weld_surface_v2 | held out training source | YOLO11n | 0.9055 (0.016) | 0.8607 | 0.0448 | undefined |
| pool_no_weld_surface_v2 | localisation | weld_surface_v2 | held out training source | YOLO11s | 0.9231 (0.016) | 0.8607 | 0.0624 | undefined |

## 4b. Arm D's box level coverage, derived from `config_arms.box_level_defined`

| pool | pool protocol | target | target protocol | role | box defined |
|---|---|---|---|---|---|
| pool_no_weld_surface_v2 | localisation | weld_surface_v2 | whole_region | held out training source | no |
| pool_no_weld_surface_v2 | localisation | kaggle_v2 | localisation | in pool | yes |
| pool_no_weld_surface_v2 | localisation | lohi_weld | localisation | in pool | yes |
| pool_no_weld_surface_v2 | localisation | visual | localisation | held out test only target | yes |
| pool_no_weld_surface_v2 | localisation | twoclass | localisation | held out test only target | yes |
| pool_no_kaggle_v2 | mixed | weld_surface_v2 | whole_region | in pool | no |
| pool_no_kaggle_v2 | mixed | kaggle_v2 | localisation | held out training source | no |
| pool_no_kaggle_v2 | mixed | lohi_weld | localisation | in pool | no |
| pool_no_kaggle_v2 | mixed | visual | localisation | held out test only target | no |
| pool_no_kaggle_v2 | mixed | twoclass | localisation | held out test only target | no |
| pool_no_lohi_weld | mixed | weld_surface_v2 | whole_region | in pool | no |
| pool_no_lohi_weld | mixed | kaggle_v2 | localisation | in pool | no |
| pool_no_lohi_weld | mixed | lohi_weld | localisation | held out training source | no |
| pool_no_lohi_weld | mixed | visual | localisation | held out test only target | no |
| pool_no_lohi_weld | mixed | twoclass | localisation | held out test only target | no |

**ARM D CANNOT REPORT A BOX LEVEL NUMBER FOR ANY HELD OUT TRAINING SOURCE, and this is a protocol rule rather than a missing file.** Verified above, not assumed. `box_defined_on_any_held_out_training_source` is `False`.

The reason is a double exclusion and only the first half is in the project log 6b. First, a pool containing `weld_surface_v2` is protocol mixed, so `pool_no_kaggle_v2` and `pool_no_lohi_weld` have no box level number at all. Second, the one localisation pool is `pool_no_weld_surface_v2`, and its held out target IS `weld_surface_v2`, which is whole region. So the only pool that could carry a box number is held out against the only target that cannot receive one.

Arm D's box level evidence is therefore exactly 2 pairs, all from the one localisation pool scored on the two test only targets: `pool_no_weld_surface_v2 to twoclass`, `pool_no_weld_surface_v2 to visual`.

Say this in the caption. A reader who sees an empty box column will otherwise assume a run failed.

## 4c. Does pooling help? Pool against the best single pool member, same family and same seed

| pool | target | role | cfg | pooled image AP | best single | best single AP | pool - best |
|---|---|---|---|---|---|---|---|
| pool_no_kaggle_v2 | kaggle_v2 | held out training source | Faster R-CNN | 0.5688 | lohi_weld/weld_surface_v2 | 0.5673 | +0.0015 (0.038) |
| pool_no_kaggle_v2 | kaggle_v2 | held out training source | RT-DETR-l | 0.5905 | lohi_weld/weld_surface_v2 | 0.6117 | -0.0212 (0.025) |
| pool_no_kaggle_v2 | kaggle_v2 | held out training source | YOLO11n | 0.6231 | weld_surface_v2 | 0.6114 | +0.0117 (0.027) |
| pool_no_kaggle_v2 | kaggle_v2 | held out training source | YOLO11s | 0.6097 | weld_surface_v2 | 0.6007 | +0.0090 (0.033) |
| pool_no_kaggle_v2 | twoclass | held out test only target | Faster R-CNN | 0.8544 | lohi_weld | 0.9249 | -0.0705 (0.013) |
| pool_no_kaggle_v2 | twoclass | held out test only target | RT-DETR-l | 0.8773 | lohi_weld/weld_surface_v2 | 0.8592 | +0.0181 (0.034) |
| pool_no_kaggle_v2 | twoclass | held out test only target | YOLO11n | 0.8608 | lohi_weld | 0.8805 | -0.0197 (0.009) |
| pool_no_kaggle_v2 | twoclass | held out test only target | YOLO11s | 0.8415 | lohi_weld/weld_surface_v2 | 0.8643 | -0.0227 (0.023) |
| pool_no_kaggle_v2 | visual | held out test only target | Faster R-CNN | 0.9166 | weld_surface_v2 | 0.9306 | -0.0140 (0.018) |
| pool_no_kaggle_v2 | visual | held out test only target | RT-DETR-l | 0.8950 | lohi_weld/weld_surface_v2 | 0.9142 | -0.0193 (0.003) |
| pool_no_kaggle_v2 | visual | held out test only target | YOLO11n | 0.9006 | lohi_weld/weld_surface_v2 | 0.9104 | -0.0098 (0.025) |
| pool_no_kaggle_v2 | visual | held out test only target | YOLO11s | 0.8974 | lohi_weld/weld_surface_v2 | 0.9040 | -0.0066 (0.004) |
| pool_no_lohi_weld | lohi_weld | held out training source | Faster R-CNN | 0.9860 | kaggle_v2 | 0.9930 | -0.0070 (0.001) |
| pool_no_lohi_weld | lohi_weld | held out training source | RT-DETR-l | 0.9820 | kaggle_v2 | 0.9884 | -0.0064 (0.003) |
| pool_no_lohi_weld | lohi_weld | held out training source | YOLO11n | 0.9916 | kaggle_v2 | 0.9955 | -0.0039 (0.003) |
| pool_no_lohi_weld | lohi_weld | held out training source | YOLO11s | 0.9949 | kaggle_v2 | 0.9960 | -0.0011 (0.003) |
| pool_no_lohi_weld | twoclass | held out test only target | Faster R-CNN | 0.8199 | kaggle_v2/weld_surface_v2 | 0.8420 | -0.0221 (0.007) |
| pool_no_lohi_weld | twoclass | held out test only target | RT-DETR-l | 0.8568 | kaggle_v2/weld_surface_v2 | 0.8763 | -0.0195 (0.024) |
| pool_no_lohi_weld | twoclass | held out test only target | YOLO11n | 0.8823 | kaggle_v2/weld_surface_v2 | 0.8739 | +0.0085 (0.052) |
| pool_no_lohi_weld | twoclass | held out test only target | YOLO11s | 0.8416 | kaggle_v2/weld_surface_v2 | 0.8668 | -0.0252 (0.018) |
| pool_no_lohi_weld | visual | held out test only target | Faster R-CNN | 0.9610 | kaggle_v2/weld_surface_v2 | 0.9546 | +0.0064 (0.019) |
| pool_no_lohi_weld | visual | held out test only target | RT-DETR-l | 0.9376 | kaggle_v2 | 0.9272 | +0.0103 (0.031) |
| pool_no_lohi_weld | visual | held out test only target | YOLO11n | 0.9318 | kaggle_v2 | 0.9402 | -0.0084 (0.010) |
| pool_no_lohi_weld | visual | held out test only target | YOLO11s | 0.9244 | kaggle_v2 | 0.9049 | +0.0195 (0.018) |
| pool_no_weld_surface_v2 | twoclass | held out test only target | Faster R-CNN | 0.8452 | lohi_weld | 0.9249 | -0.0797 (0.026) |
| pool_no_weld_surface_v2 | twoclass | held out test only target | RT-DETR-l | 0.8616 | kaggle_v2/lohi_weld | 0.8576 | +0.0040 (0.054) |
| pool_no_weld_surface_v2 | twoclass | held out test only target | YOLO11n | 0.8546 | kaggle_v2/lohi_weld | 0.8889 | -0.0343 (0.030) |
| pool_no_weld_surface_v2 | twoclass | held out test only target | YOLO11s | 0.8497 | kaggle_v2/lohi_weld | 0.8639 | -0.0142 (0.038) |
| pool_no_weld_surface_v2 | visual | held out test only target | Faster R-CNN | 0.9531 | kaggle_v2 | 0.9513 | +0.0018 (0.016) |
| pool_no_weld_surface_v2 | visual | held out test only target | RT-DETR-l | 0.9329 | kaggle_v2/lohi_weld | 0.9285 | +0.0044 (0.027) |
| pool_no_weld_surface_v2 | visual | held out test only target | YOLO11n | 0.9314 | kaggle_v2/lohi_weld | 0.9408 | -0.0094 (0.010) |
| pool_no_weld_surface_v2 | visual | held out test only target | YOLO11s | 0.9276 | kaggle_v2/lohi_weld | 0.9081 | +0.0196 (0.013) |
| pool_no_weld_surface_v2 | weld_surface_v2 | held out training source | Faster R-CNN | 0.9067 | lohi_weld | 0.9256 | -0.0189 (0.003) |
| pool_no_weld_surface_v2 | weld_surface_v2 | held out training source | RT-DETR-l | 0.9136 | kaggle_v2/lohi_weld | 0.9222 | -0.0087 (0.003) |
| pool_no_weld_surface_v2 | weld_surface_v2 | held out training source | YOLO11n | 0.9055 | kaggle_v2/lohi_weld | 0.9344 | -0.0290 (0.029) |
| pool_no_weld_surface_v2 | weld_surface_v2 | held out training source | YOLO11s | 0.9231 | lohi_weld | 0.9031 | +0.0200 (0.017) |

Pooling beats the best single pool member in **13 of 36** comparisons at image level.

**The control is the best single POOL MEMBER, not the best of all three training sources.** The pool holds exactly those members, so this compares one training budget against itself, pooled or not. Comparing against a source outside the pool would change the training data and answer a different question. Where the target is itself a pool member the diagonal is excluded, since that is in domain rather than transfer.

**Caption, and this is a protocol rule not a data gap.** Arm D reports image level for every fold. It reports box level ONLY for `pool_no_weld_surface_v2`, the one pool that excludes the whole region source, and even there only on the two test only targets. weld_surface_v2 annotates a frame sized quality verdict rather than a localised defect, so any pool containing it is protocol mixed and a box level number from it would measure annotation mismatch, not domain shift. The project log 6b.

**The honest limit of the design, the project log 6b.** There are only three training sources, so a leave one out fold trains on two. This is leave one out over a very small pool count, and a two source pool is a weak test of whether breadth helps. visual and twoclass are test only targets and never enter a pool, which is why all three pools are evaluated against them.

**Image AP is floored by prevalence on the positive heavy targets.** Read `AP - base` rather than the raw column, and read the box column where it exists. The project log 5k.
