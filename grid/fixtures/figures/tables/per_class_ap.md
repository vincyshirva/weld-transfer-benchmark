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
| Bad Weld | **no, verdict** | 0.753 | ± 0.032 | not needed | 110 | scored, not a defect class |
| Defect | yes | 0.714 | ± 0.027 | not needed | 159 | ok |
| Good Weld | **no, verdict** | 0.724 | ± 0.027 | not needed | 182 | scored, not a defect class |

## kaggle_v2, RT-DETR-l

| class | in defect space | AP50 (mean over seeds) | seed SD | split-block bootstrap | test instances | status |
|---|---|---|---|---|---|---|
| Bad Weld | **no, verdict** | 0.784 | ± 0.126 | not needed | 110 | scored, not a defect class |
| Defect | yes | 0.787 | ± 0.115 | not needed | 159 | ok |
| Good Weld | **no, verdict** | 0.779 | ± 0.098 | not needed | 182 | scored, not a defect class |

## kaggle_v2, YOLO11n

| class | in defect space | AP50 (mean over seeds) | seed SD | split-block bootstrap | test instances | status |
|---|---|---|---|---|---|---|
| Bad Weld | **no, verdict** | 0.713 | ± 0.048 | not needed | 110 | scored, not a defect class |
| Defect | yes | 0.688 | ± 0.046 | not needed | 159 | ok |
| Good Weld | **no, verdict** | 0.710 | ± 0.032 | not needed | 182 | scored, not a defect class |

## kaggle_v2, YOLO11s

| class | in defect space | AP50 (mean over seeds) | seed SD | split-block bootstrap | test instances | status |
|---|---|---|---|---|---|---|
| Bad Weld | **no, verdict** | 0.714 | ± 0.045 | not needed | 110 | scored, not a defect class |
| Defect | yes | 0.699 | ± 0.072 | not needed | 159 | ok |
| Good Weld | **no, verdict** | 0.754 | ± 0.022 | not needed | 182 | scored, not a defect class |

## lohi_weld, Faster R-CNN

| class | in defect space | AP50 (mean over seeds) | seed SD | split-block bootstrap | test instances | status |
|---|---|---|---|---|---|---|
| deposit | yes | 0.961 | ± 0.034 | not needed | 440 | ok |
| discontinuity | yes | 0.981 | ± 0.034 | not needed | 1082 | ok |
| pore | yes | 0.952 | ± 0.059 | not needed | 589 | ok |
| stain | yes | 0.972 | ± 0.047 | not needed | 1246 | ok |

## lohi_weld, RT-DETR-l

| class | in defect space | AP50 (mean over seeds) | seed SD | split-block bootstrap | test instances | status |
|---|---|---|---|---|---|---|
| deposit | yes | 0.991 | ± 0.015 | not needed | 440 | ok |
| discontinuity | yes | 0.935 | ± 0.063 | not needed | 1082 | ok |
| pore | yes | 0.966 | ± 0.059 | not needed | 589 | ok |
| stain | yes | 0.926 | ± 0.080 | not needed | 1246 | ok |

## lohi_weld, YOLO11n

| class | in defect space | AP50 (mean over seeds) | seed SD | split-block bootstrap | test instances | status |
|---|---|---|---|---|---|---|
| deposit | yes | 0.965 | ± 0.040 | not needed | 440 | ok |
| discontinuity | yes | 0.901 | ± 0.021 | not needed | 1082 | ok |
| pore | yes | 0.899 | ± 0.054 | not needed | 589 | ok |
| stain | yes | 0.914 | ± 0.030 | not needed | 1246 | ok |

## lohi_weld, YOLO11s

| class | in defect space | AP50 (mean over seeds) | seed SD | split-block bootstrap | test instances | status |
|---|---|---|---|---|---|---|
| deposit | yes | 0.900 | ± 0.039 | not needed | 440 | ok |
| discontinuity | yes | 0.939 | ± 0.080 | not needed | 1082 | ok |
| pore | yes | 0.950 | ± 0.072 | not needed | 589 | ok |
| stain | yes | 0.941 | ± 0.072 | not needed | 1246 | ok |

## twoclass, Faster R-CNN

| class | in defect space | AP50 (mean over seeds) | seed SD | split-block bootstrap | test instances | status |
|---|---|---|---|---|---|---|
| Geometric defect | yes | 0.626 | ± 0.054 | [0.446, 0.806] | 43 | **thin** |
| Non-fusion defect | yes | 0.673 | ± 0.165 | [0.493, 0.847] | 30 | **thin** |

## twoclass, RT-DETR-l

| class | in defect space | AP50 (mean over seeds) | seed SD | split-block bootstrap | test instances | status |
|---|---|---|---|---|---|---|
| Geometric defect | yes | 0.730 | ± 0.081 | [0.550, 0.909] | 43 | **thin** |
| Non-fusion defect | yes | 0.690 | ± 0.039 | [0.510, 0.870] | 30 | **thin** |

## twoclass, YOLO11n

| class | in defect space | AP50 (mean over seeds) | seed SD | split-block bootstrap | test instances | status |
|---|---|---|---|---|---|---|
| Geometric defect | yes | 0.661 | ± 0.026 | [0.481, 0.841] | 43 | **thin** |
| Non-fusion defect | yes | 0.640 | ± 0.057 | [0.460, 0.820] | 30 | **thin** |

## twoclass, YOLO11s

| class | in defect space | AP50 (mean over seeds) | seed SD | split-block bootstrap | test instances | status |
|---|---|---|---|---|---|---|
| Geometric defect | yes | 0.605 | ± 0.039 | [0.425, 0.785] | 43 | **thin** |
| Non-fusion defect | yes | 0.684 | ± 0.012 | [0.504, 0.864] | 30 | **thin** |

## visual, Faster R-CNN

| class | in defect space | AP50 (mean over seeds) | seed SD | split-block bootstrap | test instances | status |
|---|---|---|---|---|---|---|
| Defect | yes | 0.901 | ± 0.064 | not needed | 259 | ok |

## visual, RT-DETR-l

| class | in defect space | AP50 (mean over seeds) | seed SD | split-block bootstrap | test instances | status |
|---|---|---|---|---|---|---|
| Defect | yes | 0.800 | ± 0.028 | not needed | 259 | ok |

## visual, YOLO11n

| class | in defect space | AP50 (mean over seeds) | seed SD | split-block bootstrap | test instances | status |
|---|---|---|---|---|---|---|
| Defect | yes | 0.714 | ± 0.025 | not needed | 259 | ok |

## visual, YOLO11s

| class | in defect space | AP50 (mean over seeds) | seed SD | split-block bootstrap | test instances | status |
|---|---|---|---|---|---|---|
| Defect | yes | 0.788 | ± 0.016 | not needed | 259 | ok |

## weld_surface_v2, Faster R-CNN

| class | in defect space | AP50 (mean over seeds) | seed SD | split-block bootstrap | test instances | status |
|---|---|---|---|---|---|---|
| Burn Through-BT | yes | 0.857 | ± 0.041 | [0.677, 0.998] | 42 | **thin** |
| Contamination-CN | yes | 0.877 | ± 0.023 | not needed | 88 | ok |
| Lack of Fusion-LOF | yes | 0.950 | ± 0.031 | not needed | 71 | ok |
| Normal Weld-NW | **no, verdict** | 0.972 | ± 0.030 | not needed | 62 | scored, not a defect class |
| Spatter-SR | yes | 0.907 | ± 0.063 | not needed | 80 | ok |
| Surface Crack-SC | yes | 0.873 | ± 0.050 | [0.693, 1.000] | 38 | **thin** |
| Undercut-UC | yes | 0.926 | ± 0.128 | not needed | 70 | ok |

## weld_surface_v2, RT-DETR-l

| class | in defect space | AP50 (mean over seeds) | seed SD | split-block bootstrap | test instances | status |
|---|---|---|---|---|---|---|
| Burn Through-BT | yes | 0.899 | ± 0.009 | [0.719, 1.000] | 42 | **thin** |
| Contamination-CN | yes | 0.848 | ± 0.047 | not needed | 88 | ok |
| Lack of Fusion-LOF | yes | 0.920 | ± 0.026 | not needed | 71 | ok |
| Normal Weld-NW | **no, verdict** | 0.884 | ± 0.026 | not needed | 62 | scored, not a defect class |
| Spatter-SR | yes | 0.828 | ± 0.027 | not needed | 80 | ok |
| Surface Crack-SC | yes | 0.890 | ± 0.103 | [0.710, 0.984] | 38 | **thin** |
| Undercut-UC | yes | 0.923 | ± 0.033 | not needed | 70 | ok |

## weld_surface_v2, YOLO11n

| class | in defect space | AP50 (mean over seeds) | seed SD | split-block bootstrap | test instances | status |
|---|---|---|---|---|---|---|
| Burn Through-BT | yes | 0.876 | ± 0.071 | [0.696, 0.998] | 42 | **thin** |
| Contamination-CN | yes | 0.846 | ± 0.021 | not needed | 88 | ok |
| Lack of Fusion-LOF | yes | 0.858 | ± 0.039 | not needed | 71 | ok |
| Normal Weld-NW | **no, verdict** | 0.861 | ± 0.036 | not needed | 62 | scored, not a defect class |
| Spatter-SR | yes | 0.829 | ± 0.007 | not needed | 80 | ok |
| Surface Crack-SC | yes | 0.850 | ± 0.091 | [0.670, 0.983] | 38 | **thin** |
| Undercut-UC | yes | 0.801 | ± 0.040 | not needed | 70 | ok |

## weld_surface_v2, YOLO11s

| class | in defect space | AP50 (mean over seeds) | seed SD | split-block bootstrap | test instances | status |
|---|---|---|---|---|---|---|
| Burn Through-BT | yes | 0.927 | ± 0.065 | [0.747, 1.000] | 42 | **thin** |
| Contamination-CN | yes | 0.896 | ± 0.024 | not needed | 88 | ok |
| Lack of Fusion-LOF | yes | 0.906 | ± 0.049 | not needed | 71 | ok |
| Normal Weld-NW | **no, verdict** | 0.877 | ± 0.036 | not needed | 62 | scored, not a defect class |
| Spatter-SR | yes | 0.988 | ± 0.020 | not needed | 80 | ok |
| Surface Crack-SC | yes | 0.922 | ± 0.030 | [0.742, 1.000] | 38 | **thin** |
| Undercut-UC | yes | 0.757 | ± 0.033 | not needed | 70 | ok |

