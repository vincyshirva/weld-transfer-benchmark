# In-domain per-class AP50

**Every class of every source is listed, and the `in defect space` column says which of them the harmonised taxonomy retains.** A class marked no is scene furniture or a whole-region quality verdict, and it contributes to no cross-dataset figure anywhere in this study. Printing only the retained classes would hide the taxonomy this table exists to demonstrate, and printing all of them unflagged would show Normal Weld sitting among defect classes.

**Values apply non-maximum suppression at IoU 0.7, WITHIN each class, recomputed by `per_class_nms.py`.** The evaluator wrote these with no suppression at all, which charged RT-DETR for duplicate queries that the other three backends remove before a prediction is stored. On a source whose defect space holds one class this value and the class-agnostic box mAP50 are the same quantity, and they now agree exactly.

**Within-class suppression is a decision, not the function that was to hand.** The alternative is to suppress across the whole defect space first, which is what the box-level matrix does because there every defect class collapses to one label. That rule is wrong for a per-class table: it makes a class's value depend on which other classes sit in the taxonomy, so removing one class would move another's AP without anything about that class changing. The difference is not negligible, reaching 0.0999 on twoclass Non-fusion defect with RT-DETR-l. Run `per_class_nms.py --compare-rules` to reproduce.

Main-grid and reference cells only. 84 downloaded-data rows are excluded, because a downloaded-data cell is graded on a partition that overlaps its own training split and averaging it in would inflate the in-domain value. Those rows are kept in `per_class_ap.csv`, tagged with `kind`.

Every value carries its test instance count. A class marked **thin** has fewer than 50 test instances.

**The two uncertainty columns answer different questions and a thin class needs both.** Seed SD is dispersion over the three training seeds, so it measures training variance. It says nothing about a small correlated test sample: a class with 38 test boxes can look stable across seeds and still be highly uncertain. The bootstrap column is a 2000 draw percentile interval resampling whole split blocks, which is the unit that keeps correlated images together. It is reported for thin classes, averaged over seeds. Quote the bootstrap interval for a thin class, never the seed SD alone.

## kaggle_v2, Faster R-CNN

| class | in defect space | AP50 (mean over seeds) | seed SD | split-block bootstrap | test instances | status |
|---|---|---|---|---|---|---|
| Bad Weld | **no, verdict** | 0.544 | ± 0.029 | not needed | 110 | scored, not a defect class |
| Defect | yes | 0.034 | ± 0.004 | not needed | 159 | ok |
| Good Weld | **no, verdict** | 0.588 | ± 0.023 | not needed | 182 | scored, not a defect class |

## kaggle_v2, RT-DETR-l

| class | in defect space | AP50 (mean over seeds) | seed SD | split-block bootstrap | test instances | status |
|---|---|---|---|---|---|---|
| Bad Weld | **no, verdict** | 0.373 | ± 0.033 | not needed | 110 | scored, not a defect class |
| Defect | yes | 0.087 | ± 0.020 | not needed | 159 | ok |
| Good Weld | **no, verdict** | 0.538 | ± 0.018 | not needed | 182 | scored, not a defect class |

## kaggle_v2, YOLO11n

| class | in defect space | AP50 (mean over seeds) | seed SD | split-block bootstrap | test instances | status |
|---|---|---|---|---|---|---|
| Bad Weld | **no, verdict** | 0.382 | ± 0.051 | not needed | 110 | scored, not a defect class |
| Defect | yes | 0.088 | ± 0.005 | not needed | 159 | ok |
| Good Weld | **no, verdict** | 0.500 | ± 0.023 | not needed | 182 | scored, not a defect class |

## kaggle_v2, YOLO11s

| class | in defect space | AP50 (mean over seeds) | seed SD | split-block bootstrap | test instances | status |
|---|---|---|---|---|---|---|
| Bad Weld | **no, verdict** | 0.410 | ± 0.010 | not needed | 110 | scored, not a defect class |
| Defect | yes | 0.112 | ± 0.016 | not needed | 159 | ok |
| Good Weld | **no, verdict** | 0.510 | ± 0.010 | not needed | 182 | scored, not a defect class |

## lohi_weld, Faster R-CNN

| class | in defect space | AP50 (mean over seeds) | seed SD | split-block bootstrap | test instances | status |
|---|---|---|---|---|---|---|
| deposit | yes | 0.472 | ± 0.028 | not needed | 440 | ok |
| discontinuity | yes | 0.757 | ± 0.009 | not needed | 1082 | ok |
| pore | yes | 0.375 | ± 0.011 | not needed | 589 | ok |
| stain | yes | 0.551 | ± 0.005 | not needed | 1246 | ok |

## lohi_weld, RT-DETR-l

| class | in defect space | AP50 (mean over seeds) | seed SD | split-block bootstrap | test instances | status |
|---|---|---|---|---|---|---|
| deposit | yes | 0.531 | ± 0.060 | not needed | 440 | ok |
| discontinuity | yes | 0.710 | ± 0.061 | not needed | 1082 | ok |
| pore | yes | 0.451 | ± 0.058 | not needed | 589 | ok |
| stain | yes | 0.602 | ± 0.054 | not needed | 1246 | ok |

## lohi_weld, YOLO11n

| class | in defect space | AP50 (mean over seeds) | seed SD | split-block bootstrap | test instances | status |
|---|---|---|---|---|---|---|
| deposit | yes | 0.501 | ± 0.003 | not needed | 440 | ok |
| discontinuity | yes | 0.754 | ± 0.015 | not needed | 1082 | ok |
| pore | yes | 0.464 | ± 0.011 | not needed | 589 | ok |
| stain | yes | 0.596 | ± 0.006 | not needed | 1246 | ok |

## lohi_weld, YOLO11s

| class | in defect space | AP50 (mean over seeds) | seed SD | split-block bootstrap | test instances | status |
|---|---|---|---|---|---|---|
| deposit | yes | 0.492 | ± 0.007 | not needed | 440 | ok |
| discontinuity | yes | 0.759 | ± 0.010 | not needed | 1082 | ok |
| pore | yes | 0.458 | ± 0.008 | not needed | 589 | ok |
| stain | yes | 0.605 | ± 0.010 | not needed | 1246 | ok |

## twoclass, Faster R-CNN

| class | in defect space | AP50 (mean over seeds) | seed SD | split-block bootstrap | test instances | status |
|---|---|---|---|---|---|---|
| Geometric defect | yes | 0.526 | ± 0.050 | [0.061, 0.696] | 43 | **thin** |
| Non-fusion defect | yes | 0.758 | ± 0.035 | [0.004, 0.834] | 30 | **thin** |

## twoclass, RT-DETR-l

| class | in defect space | AP50 (mean over seeds) | seed SD | split-block bootstrap | test instances | status |
|---|---|---|---|---|---|---|
| Geometric defect | yes | 0.477 | ± 0.067 | [0.008, 0.675] | 43 | **thin** |
| Non-fusion defect | yes | 0.493 | ± 0.142 | [0.212, 0.940] | 30 | **thin** |

## twoclass, YOLO11n

| class | in defect space | AP50 (mean over seeds) | seed SD | split-block bootstrap | test instances | status |
|---|---|---|---|---|---|---|
| Geometric defect | yes | 0.474 | ± 0.051 | [0.029, 0.679] | 43 | **thin** |
| Non-fusion defect | yes | 0.751 | ± 0.071 | [0.303, 0.945] | 30 | **thin** |

## twoclass, YOLO11s

| class | in defect space | AP50 (mean over seeds) | seed SD | split-block bootstrap | test instances | status |
|---|---|---|---|---|---|---|
| Geometric defect | yes | 0.499 | ± 0.066 | [0.025, 0.675] | 43 | **thin** |
| Non-fusion defect | yes | 0.495 | ± 0.106 | [0.077, 0.932] | 30 | **thin** |

## visual, Faster R-CNN

| class | in defect space | AP50 (mean over seeds) | seed SD | split-block bootstrap | test instances | status |
|---|---|---|---|---|---|---|
| Defect | yes | 0.157 | ± 0.010 | not needed | 259 | ok |
| Welding Line | **no, not scored** | n/a | n/a | not needed | 0 | excluded from ground truth |
| workpiece | **no, not scored** | n/a | n/a | not needed | 0 | excluded from ground truth |

## visual, RT-DETR-l

| class | in defect space | AP50 (mean over seeds) | seed SD | split-block bootstrap | test instances | status |
|---|---|---|---|---|---|---|
| Defect | yes | 0.130 | ± 0.024 | not needed | 259 | ok |
| Welding Line | **no, not scored** | n/a | n/a | not needed | 0 | excluded from ground truth |
| workpiece | **no, not scored** | n/a | n/a | not needed | 0 | excluded from ground truth |

## visual, YOLO11n

| class | in defect space | AP50 (mean over seeds) | seed SD | split-block bootstrap | test instances | status |
|---|---|---|---|---|---|---|
| Defect | yes | 0.151 | ± 0.013 | not needed | 259 | ok |
| Welding Line | **no, not scored** | n/a | n/a | not needed | 0 | excluded from ground truth |
| workpiece | **no, not scored** | n/a | n/a | not needed | 0 | excluded from ground truth |

## visual, YOLO11s

| class | in defect space | AP50 (mean over seeds) | seed SD | split-block bootstrap | test instances | status |
|---|---|---|---|---|---|---|
| Defect | yes | 0.142 | ± 0.006 | not needed | 259 | ok |
| Welding Line | **no, not scored** | n/a | n/a | not needed | 0 | excluded from ground truth |
| workpiece | **no, not scored** | n/a | n/a | not needed | 0 | excluded from ground truth |

## weld_surface_v2, Faster R-CNN

| class | in defect space | AP50 (mean over seeds) | seed SD | split-block bootstrap | test instances | status |
|---|---|---|---|---|---|---|
| Burn Through-BT | yes | 0.927 | ± 0.011 | [0.819, 0.995] | 42 | **thin** |
| Contamination-CN | yes | 0.455 | ± 0.037 | not needed | 88 | ok |
| Lack of Fusion-LOF | yes | 0.900 | ± 0.030 | not needed | 71 | ok |
| Normal Weld-NW | **no, verdict** | 0.855 | ± 0.036 | not needed | 62 | scored, not a defect class |
| Spatter-SR | yes | 0.429 | ± 0.031 | not needed | 80 | ok |
| Surface Crack-SC | yes | 0.980 | ± 0.017 | [0.939, 1.000] | 38 | **thin** |
| Undercut-UC | yes | 0.774 | ± 0.018 | not needed | 70 | ok |

## weld_surface_v2, RT-DETR-l

| class | in defect space | AP50 (mean over seeds) | seed SD | split-block bootstrap | test instances | status |
|---|---|---|---|---|---|---|
| Burn Through-BT | yes | 0.878 | ± 0.096 | [0.677, 0.979] | 42 | **thin** |
| Contamination-CN | yes | 0.453 | ± 0.040 | not needed | 88 | ok |
| Lack of Fusion-LOF | yes | 0.936 | ± 0.026 | not needed | 71 | ok |
| Normal Weld-NW | **no, verdict** | 0.659 | ± 0.083 | not needed | 62 | scored, not a defect class |
| Spatter-SR | yes | 0.449 | ± 0.082 | not needed | 80 | ok |
| Surface Crack-SC | yes | 0.990 | ± 0.016 | [0.922, 1.000] | 38 | **thin** |
| Undercut-UC | yes | 0.771 | ± 0.040 | not needed | 70 | ok |

## weld_surface_v2, YOLO11n

| class | in defect space | AP50 (mean over seeds) | seed SD | split-block bootstrap | test instances | status |
|---|---|---|---|---|---|---|
| Burn Through-BT | yes | 0.930 | ± 0.005 | [0.835, 0.991] | 42 | **thin** |
| Contamination-CN | yes | 0.552 | ± 0.027 | not needed | 88 | ok |
| Lack of Fusion-LOF | yes | 0.922 | ± 0.015 | not needed | 71 | ok |
| Normal Weld-NW | **no, verdict** | 0.600 | ± 0.152 | not needed | 62 | scored, not a defect class |
| Spatter-SR | yes | 0.599 | ± 0.052 | not needed | 80 | ok |
| Surface Crack-SC | yes | 0.999 | ± 0.002 | [0.996, 1.000] | 38 | **thin** |
| Undercut-UC | yes | 0.782 | ± 0.013 | not needed | 70 | ok |

## weld_surface_v2, YOLO11s

| class | in defect space | AP50 (mean over seeds) | seed SD | split-block bootstrap | test instances | status |
|---|---|---|---|---|---|---|
| Burn Through-BT | yes | 0.919 | ± 0.029 | [0.812, 0.986] | 42 | **thin** |
| Contamination-CN | yes | 0.503 | ± 0.030 | not needed | 88 | ok |
| Lack of Fusion-LOF | yes | 0.936 | ± 0.010 | not needed | 71 | ok |
| Normal Weld-NW | **no, verdict** | 0.754 | ± 0.085 | not needed | 62 | scored, not a defect class |
| Spatter-SR | yes | 0.553 | ± 0.028 | not needed | 80 | ok |
| Surface Crack-SC | yes | 0.989 | ± 0.015 | [0.971, 1.000] | 38 | **thin** |
| Undercut-UC | yes | 0.761 | ± 0.022 | not needed | 70 | ok |

