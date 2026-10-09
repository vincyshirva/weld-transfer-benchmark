# Image-level transfer, YOLO11s

**Read every column against its base rate.** Image-level AP is positive-class average precision, so a constant-positive predictor already scores the positive rate. The floor differs by column, and only a column marked balanced is near 0.5.

| train \ test | weld_surface_v2 | kaggle_v2 | lohi_weld | visual | twoclass |
|---|---|---|---|---|---|
| *base rate (positive class)* | 0.861 | 0.511 **balanced** | 0.976 | 0.893 | 0.800 |
| **weld_surface_v2** | 0.988 | 0.601 | 0.979 | 0.900 | 0.853 |
| **kaggle_v2** | 0.880 | 0.796 | 0.996 | 0.905 | 0.854 |
| **lohi_weld** | 0.903 | 0.549 | 0.999 | 0.904 | 0.860 |

## Same matrix, AP minus base rate

The lift over a constant-positive predictor. This is the quantity that is comparable across columns. Negative means worse than trivial.

| train \ test | weld_surface_v2 | kaggle_v2 | lohi_weld | visual | twoclass |
|---|---|---|---|---|---|
| **weld_surface_v2** | +0.128 | +0.090 | +0.003 | +0.007 | +0.053 |
| **kaggle_v2** | +0.019 | +0.286 | +0.020 | +0.012 | +0.054 |
| **lohi_weld** | +0.042 | +0.038 | +0.024 | +0.011 | +0.060 |
