# Image-level transfer, mean over configurations

**Read every column against its base rate.** Image-level AP is positive-class average precision, so a constant-positive predictor already scores the positive rate. The floor differs by column, and only a column marked balanced is near 0.5.

| train \ test | weld_surface_v2 | kaggle_v2 | lohi_weld | visual | twoclass |
|---|---|---|---|---|---|
| *base rate (positive class)* | 0.861 | 0.511 **balanced** | 0.976 | 0.893 | 0.800 |
| **weld_surface_v2** | 0.984 | 0.593 | 0.979 | 0.909 | 0.846 |
| **kaggle_v2** | 0.895 | 0.775 | 0.993 | 0.931 | 0.847 |
| **lohi_weld** | 0.919 | 0.555 | 0.998 | 0.908 | 0.875 |

## Same matrix, AP minus base rate

The lift over a constant-positive predictor. This is the quantity that is comparable across columns. Negative means worse than trivial.

| train \ test | weld_surface_v2 | kaggle_v2 | lohi_weld | visual | twoclass |
|---|---|---|---|---|---|
| **weld_surface_v2** | +0.124 | +0.083 | +0.003 | +0.017 | +0.046 |
| **kaggle_v2** | +0.034 | +0.264 | +0.018 | +0.038 | +0.047 |
| **lohi_weld** | +0.059 | +0.045 | +0.023 | +0.015 | +0.075 |
