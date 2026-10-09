# Clean rebuild, before and after

Seed 20260903. Copy threshold NCC 0.96. Scene threshold NCC 0.9.
Copy representatives are a maximum independent set at NCC 0.96.
Scene links pass NCC 0.90 at both 32 and 128 pixels.
Complete linkage defines the confirmed content scene groups.
Known, different Visual specimen ids override thumbnail similarity.
Direct NCC edges and structural keys form leakage-safe split blocks.
Kaggle video adjacency constrains splits but does not assert one scene.
Split target {'train': 0.7, 'valid': 0.15, 'test': 0.15}, applied to leakage-safe split blocks.
Splits are optimized jointly for image totals, per-class box counts,
per-class image presence, backgrounds, and scene group counts.

## Images and boxes

| Dataset | Images before | Images after | Boxes before | Boxes after | Scene groups | Largest group |
|---|---|---|---|---|---|---|
| weld_surface_v2 | 17195 | 2963 | 17445 | 3009 | 604 | 131 |
| kaggle_v2 | 2028 | 938 | 5686 | 3014 | 927 | 3 |
| twoclass | 1682 | 304 | 2777 | 499 | 35 | 47 |
| visual | 1288 | 558 | 6576 | 2838 | 313 | 6 |
| lohi_weld | 3022 | 3021 | 22412 | 22410 | 2022 | 2 |

## Per-class box counts

| Dataset | Class | Boxes before | Boxes after |
|---|---|---|---|
| weld_surface_v2 | Burn Through-BT | 2664 | 282 |
| weld_surface_v2 | Contamination-CN | 2627 | 590 |
| weld_surface_v2 | Lack of Fusion-LOF | 2575 | 472 |
| weld_surface_v2 | Normal Weld-NW | 2229 | 415 |
| weld_surface_v2 | Spatter-SR | 2369 | 530 |
| weld_surface_v2 | Surface Crack-SC | 2533 | 254 |
| weld_surface_v2 | Undercut-UC | 2448 | 466 |
| kaggle_v2 | Bad Weld | 1378 | 737 |
| kaggle_v2 | Defect | 1960 | 1064 |
| kaggle_v2 | Good Weld | 2348 | 1213 |
| twoclass | Geometric defect | 1541 | 302 |
| twoclass | Non-fusion defect | 1236 | 197 |
| visual | Defect | 4004 | 1726 |
| visual | Welding Line | 1281 | 553 |
| visual | workpiece | 1291 | 559 |
| lohi_weld | deposit | 2935 | 2935 |
| lohi_weld | discontinuity | 7220 | 7218 |
| lohi_weld | pore | 3950 | 3950 |
| lohi_weld | stain | 8307 | 8307 |

## Class distribution shift

Share is each class as a percentage of its own dataset's boxes.
Retention is boxes kept over boxes shipped. Read the two share
columns together: a class can lose most of its boxes and still gain
share, if the classes around it lost more.

| Dataset | Class | Boxes before | Share before | Boxes after | Share after | Share change | Retention |
|---|---|---|---|---|---|---|---|
| weld_surface_v2 | Burn Through-BT | 2664 | 15.3% | 282 | 9.4% | -5.9 pp | 10.6% |
| weld_surface_v2 | Contamination-CN | 2627 | 15.1% | 590 | 19.6% | +4.5 pp | 22.5% |
| weld_surface_v2 | Lack of Fusion-LOF | 2575 | 14.8% | 472 | 15.7% | +0.9 pp | 18.3% |
| weld_surface_v2 | Normal Weld-NW | 2229 | 12.8% | 415 | 13.8% | +1.0 pp | 18.6% |
| weld_surface_v2 | Spatter-SR | 2369 | 13.6% | 530 | 17.6% | +4.0 pp | 22.4% |
| weld_surface_v2 | Surface Crack-SC | 2533 | 14.5% | 254 | 8.4% | -6.1 pp | 10.0% |
| weld_surface_v2 | Undercut-UC | 2448 | 14.0% | 466 | 15.5% | +1.5 pp | 19.0% |
| **weld_surface_v2** | **all** | **17445** | 100% | **3009** | 100% | | **17.2%** |
| kaggle_v2 | Bad Weld | 1378 | 24.2% | 737 | 24.5% | +0.2 pp | 53.5% |
| kaggle_v2 | Defect | 1960 | 34.5% | 1064 | 35.3% | +0.8 pp | 54.3% |
| kaggle_v2 | Good Weld | 2348 | 41.3% | 1213 | 40.2% | -1.0 pp | 51.7% |
| **kaggle_v2** | **all** | **5686** | 100% | **3014** | 100% | | **53.0%** |
| twoclass | Geometric defect | 1541 | 55.5% | 302 | 60.5% | +5.0 pp | 19.6% |
| twoclass | Non-fusion defect | 1236 | 44.5% | 197 | 39.5% | -5.0 pp | 15.9% |
| **twoclass** | **all** | **2777** | 100% | **499** | 100% | | **18.0%** |
| visual | Defect | 4004 | 60.9% | 1726 | 60.8% | -0.1 pp | 43.1% |
| visual | Welding Line | 1281 | 19.5% | 553 | 19.5% | +0.0 pp | 43.2% |
| visual | workpiece | 1291 | 19.6% | 559 | 19.7% | +0.1 pp | 43.3% |
| **visual** | **all** | **6576** | 100% | **2838** | 100% | | **43.2%** |
| lohi_weld | deposit | 2935 | 13.1% | 2935 | 13.1% | +0.0 pp | 100.0% |
| lohi_weld | discontinuity | 7220 | 32.2% | 7218 | 32.2% | -0.0 pp | 100.0% |
| lohi_weld | pore | 3950 | 17.6% | 3950 | 17.6% | +0.0 pp | 100.0% |
| lohi_weld | stain | 8307 | 37.1% | 8307 | 37.1% | +0.0 pp | 100.0% |
| **lohi_weld** | **all** | **22412** | 100% | **22410** | 100% | | **100.0%** |

## Test set instances per class

Instances are ground-truth boxes in the test split. Images is the number of test images containing the class. A class marked **thin** has fewer than 50 test instances: its per-class AP is dominated by sampling noise, so report a bootstrap confidence interval over test instances for it, not the standard deviation over three seeds. Seed dispersion measures training variance and says nothing about how few objects were scored.

| Dataset | Class | Test instances | Test images | Train instances | Status |
|---|---|---|---|---|---|
| weld_surface_v2 | Burn Through-BT | 42 | 41 | 198 | **thin** |
| weld_surface_v2 | Contamination-CN | 88 | 83 | 413 | ok |
| weld_surface_v2 | Lack of Fusion-LOF | 71 | 71 | 330 | ok |
| weld_surface_v2 | Normal Weld-NW | 62 | 62 | 291 | ok |
| weld_surface_v2 | Spatter-SR | 80 | 80 | 371 | ok |
| weld_surface_v2 | Surface Crack-SC | 38 | 38 | 178 | **thin** |
| weld_surface_v2 | Undercut-UC | 70 | 70 | 326 | ok |
| kaggle_v2 | Bad Weld | 110 | 75 | 516 | ok |
| kaggle_v2 | Good Weld | 182 | 93 | 849 | ok |
| kaggle_v2 | Defect | 159 | 72 | 745 | ok |
| twoclass | Geometric defect | 43 | 30 | 213 | **thin** |
| twoclass | Non-fusion defect | 30 | 20 | 138 | **thin** |
| visual | Defect | 259 | 75 | 1208 | ok |
| visual | Welding Line | 83 | 82 | 387 | ok |
| visual | workpiece | 84 | 84 | 391 | ok |
| lohi_weld | pore | 589 | 113 | 2770 | ok |
| lohi_weld | deposit | 440 | 230 | 2055 | ok |
| lohi_weld | discontinuity | 1082 | 429 | 5052 | ok |
| lohi_weld | stain | 1246 | 317 | 5815 | ok |

Classes below 50 test instances: weld_surface_v2/Burn Through-BT (42), weld_surface_v2/Surface Crack-SC (38), twoclass/Geometric defect (43), twoclass/Non-fusion defect (30).

## Split sizes after rebuild

| Dataset | train | valid | test | train scenes | valid scenes | test scenes |
|---|---|---|---|---|---|---|
| weld_surface_v2 | 2074 | 444 | 445 | 423 | 91 | 90 |
| kaggle_v2 | 656 | 141 | 141 | 649 | 139 | 139 |
| twoclass | 213 | 46 | 45 | 23 | 6 | 6 |
| visual | 390 | 84 | 84 | 219 | 47 | 47 |
| lohi_weld | 2115 | 453 | 453 | 1417 | 302 | 303 |

## Split balance

Deviation is the realised feature share minus its target share,
in percentage points.

| Dataset | Maximum absolute deviation | Optimizer |
|---|---|---|
| weld_surface_v2 | 0.213 pp | Time limit reached. (HiGHS Status 13: Time limit reached) |
| kaggle_v2 | 0.075 pp | Time limit reached. (HiGHS Status 13: Time limit reached) |
| twoclass | 4.286 pp | Optimization terminated successfully. (HiGHS Status 7: Optimal) |
| visual | 0.118 pp | Time limit reached. (HiGHS Status 13: Time limit reached) |
| lohi_weld | 0.714 pp | Time limit reached. (HiGHS Status 13: Time limit reached) |
