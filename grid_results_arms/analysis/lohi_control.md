# LoHi internal transfer control

Box mAP50, mean (SD) over 4 configurations and 3 seeds, 12 cells per entry.

| train | to lohi_high | to lohi_low |
|---|---|---|
| lohi_high | 0.6539 (0.028) | 0.2402 (0.044) |
| lohi_low | 0.3031 (0.029) | 0.6730 (0.023) |

## Transfer ratios, box level

| direction | T_ij | retained |
|---|---|---|
| lohi_high to lohi_low | 0.357 | 35.7% |
| lohi_low to lohi_high | 0.463 | 46.3% |

## Per configuration, high to low

| configuration | box mAP50 |
|---|---|
| YOLO11n | 0.2288 |
| YOLO11s | 0.2655 |
| RT-DETR-l | 0.2265 |
| Faster R-CNN | 0.2399 |

## Transfer ratio, split block bootstrap intervals

Blocks resampled, not images, 2000 draws. Numerator and denominator are scored on the SAME resampled images, because both are measured on the target's test split. The diagonal returns exactly 1.000 with zero width, which is the self check.

| direction | cfg | seed | T | 95% interval |
|---|---|---|---|---|
| lohi_high to lohi_low | Faster R-CNN | 0 | 0.398 | [0.363, 0.440] |
| lohi_high to lohi_low | Faster R-CNN | 1 | 0.380 | [0.347, 0.421] |
| lohi_high to lohi_low | Faster R-CNN | 2 | 0.345 | [0.313, 0.387] |
| lohi_high to lohi_low | RT-DETR-l | 0 | 0.265 | [0.236, 0.297] |
| lohi_high to lohi_low | RT-DETR-l | 1 | 0.432 | [0.396, 0.474] |
| lohi_high to lohi_low | RT-DETR-l | 2 | 0.299 | [0.272, 0.328] |
| lohi_high to lohi_low | YOLO11n | 0 | 0.281 | [0.252, 0.310] |
| lohi_high to lohi_low | YOLO11n | 1 | 0.402 | [0.370, 0.439] |
| lohi_high to lohi_low | YOLO11n | 2 | 0.323 | [0.294, 0.361] |
| lohi_high to lohi_low | YOLO11s | 0 | 0.289 | [0.259, 0.330] |
| lohi_high to lohi_low | YOLO11s | 1 | 0.460 | [0.428, 0.498] |
| lohi_high to lohi_low | YOLO11s | 2 | 0.413 | [0.381, 0.451] |
| lohi_low to lohi_high | Faster R-CNN | 0 | 0.538 | [0.492, 0.585] |
| lohi_low to lohi_high | Faster R-CNN | 1 | 0.617 | [0.575, 0.659] |
| lohi_low to lohi_high | Faster R-CNN | 2 | 0.486 | [0.451, 0.528] |
| lohi_low to lohi_high | RT-DETR-l | 0 | 0.436 | [0.393, 0.482] |
| lohi_low to lohi_high | RT-DETR-l | 1 | 0.405 | [0.371, 0.444] |
| lohi_low to lohi_high | RT-DETR-l | 2 | 0.418 | [0.380, 0.459] |
| lohi_low to lohi_high | YOLO11n | 0 | 0.424 | [0.389, 0.464] |
| lohi_low to lohi_high | YOLO11n | 1 | 0.463 | [0.427, 0.503] |
| lohi_low to lohi_high | YOLO11n | 2 | 0.484 | [0.447, 0.525] |
| lohi_low to lohi_high | YOLO11s | 0 | 0.408 | [0.378, 0.441] |
| lohi_low to lohi_high | YOLO11s | 1 | 0.482 | [0.446, 0.522] |
| lohi_low to lohi_high | YOLO11s | 2 | 0.425 | [0.388, 0.466] |

## Recall at a fixed false positive budget per image

The metric that carries the inspection claim, per the project log 5. Mean over 4 configurations and 3 seeds.

| train | tested on | R@0.1 FP | R@0.5 FP | R@1.0 FP |
|---|---|---|---|---|
| lohi_high | lohi_high | 0.1369 | 0.3642 | 0.4802 |
| lohi_high | lohi_low | 0.0391 | 0.1189 | 0.1756 |
| lohi_low | lohi_high | 0.0392 | 0.1151 | 0.1751 |
| lohi_low | lohi_low | 0.1952 | 0.4146 | 0.5304 |

## Image level, reported but UNINFORMATIVE here

| train | to lohi_high | to lohi_low |
|---|---|---|
| lohi_high | 1.0000 (0.000) | 0.9907 (0.005) |
| lohi_low | 1.0000 (0.000) | 0.9977 (0.001) |

## Set against the cross dataset matrix

Derived from `grid_results/results/`, not copied. The main grid is unaffected by the the project log 6a-iii class space defect, because a main grid model predicts in its own source space and is filtered with source ids, which is correct. So the comparison is sound.

| comparison | box mAP50 | retained |
|---|---|---|
| LoHi internal, high to low | 0.2402 | 35.7% |
| LoHi internal, low to high | 0.3031 | 46.3% |
| cross dataset, kaggle_v2 to lohi_weld | 0.0211 | 3.2% of lohi_weld own 0.6662 |
| cross dataset, kaggle_v2 to twoclass | 0.0190 | 3.2% of twoclass own 0.5858 |
| cross dataset, kaggle_v2 to visual | 0.0093 | 6.4% of visual own 0.1448 |
| cross dataset, lohi_weld to kaggle_v2 | 0.0009 | 1.2% of kaggle_v2 own 0.0803 |
| cross dataset, lohi_weld to twoclass | 0.0262 | 4.5% of twoclass own 0.5858 |
| cross dataset, lohi_weld to visual | 0.0011 | 0.7% of visual own 0.1448 |

Cross dataset box mAP50 spans 0.0009 to 0.0262 against diagonals of 0.0803 and 0.6662 and 0.5858 and 0.1448.

**Caption.** The control is a BOX LEVEL result. Image level AP sits between 0.99 and 1.00 everywhere because LoHi is 97.6 percent positive, so the prevalence ceiling leaves almost no headroom. The project log 5k makes the same point for the main grid. Read the box table.

**What this controls for, and what it does not.** Annotation protocol, annotators, class list and production line are all held constant. Only the acquisition changes. So this is the closest thing the study has to pure visual domain shift with the label concept pinned. It still costs 64 percent of box mAP50 going high to low and 54 percent going low to high.

**But part of that is prior shift, not sensor shift.** The project log 7 records the defect mix differing sharply between the two acquisitions: pore is 5.5 percent of high resolution boxes and 26.5 percent of low, deposit 18.9 against 8.8. Boxes per image are 9.27 and 6.47. A reader who is not told this will read the collapse as pure sensor shift.

**Precedent.** LoHi-WELD's own IEEE Access 2024 description runs the same internal test. This version differs by running under the leakage audited splits and the same protocol as the rest of the study, which is why it is a control rather than a novelty.

**What the two sets of numbers support, stated as the evidence allows.** Internal transfer retains 36 and 46 percent of the target's own in domain score. Cross dataset transfer retains between 0.7 and 6.4 percent. So visual shift alone is sufficient to cause a large collapse, and it is NOT sufficient to explain the cross dataset collapse, which is larger again by roughly an order of magnitude. That is what the numbers support. What they do not support is attributing the remainder to a single named cause. The control isolates acquisition. The cross dataset gap additionally mixes annotation protocol, object scale, defect vocabulary and training set size, and this study does not separate those from one another.
