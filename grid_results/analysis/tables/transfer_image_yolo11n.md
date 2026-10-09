# Image-level transfer, YOLO11n

**Read every column against its base rate.** Image-level AP is positive-class average precision, so a constant-positive predictor already scores the positive rate. The floor differs by column, and only a column marked balanced is near 0.5.

| train \ test | weld_surface_v2 | kaggle_v2 | lohi_weld | visual | twoclass |
|---|---|---|---|---|---|
| *base rate (positive class)* | 0.861 | 0.511 **balanced** | 0.976 | 0.893 | 0.800 |
| **weld_surface_v2** | 0.990 | 0.611 | 0.982 | 0.903 | 0.837 |
| **kaggle_v2** | 0.900 | 0.769 | 0.996 | 0.940 | 0.866 |
| **lohi_weld** | 0.930 | 0.565 | 0.998 | 0.910 | 0.880 |

## Same matrix, AP minus base rate

The lift over a constant-positive predictor. This is the quantity that is comparable across columns. Negative means worse than trivial.

| train \ test | weld_surface_v2 | kaggle_v2 | lohi_weld | visual | twoclass |
|---|---|---|---|---|---|
| **weld_surface_v2** | +0.129 | +0.101 | +0.006 | +0.010 | +0.037 |
| **kaggle_v2** | +0.040 | +0.258 | +0.020 | +0.047 | +0.066 |
| **lohi_weld** | +0.070 | +0.054 | +0.023 | +0.018 | +0.080 |
