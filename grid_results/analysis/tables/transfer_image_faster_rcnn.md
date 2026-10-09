# Image-level transfer, Faster R-CNN

**Read every column against its base rate.** Image-level AP is positive-class average precision, so a constant-positive predictor already scores the positive rate. The floor differs by column, and only a column marked balanced is near 0.5.

| train \ test | weld_surface_v2 | kaggle_v2 | lohi_weld | visual | twoclass |
|---|---|---|---|---|---|
| *base rate (positive class)* | 0.861 | 0.511 **balanced** | 0.976 | 0.893 | 0.800 |
| **weld_surface_v2** | 0.973 | 0.559 | 0.979 | 0.931 | 0.838 |
| **kaggle_v2** | 0.896 | 0.739 | 0.993 | 0.951 | 0.833 |
| **lohi_weld** | 0.926 | 0.554 | 0.999 | 0.904 | 0.925 |

## Same matrix, AP minus base rate

The lift over a constant-positive predictor. This is the quantity that is comparable across columns. Negative means worse than trivial.

| train \ test | weld_surface_v2 | kaggle_v2 | lohi_weld | visual | twoclass |
|---|---|---|---|---|---|
| **weld_surface_v2** | +0.112 | +0.048 | +0.003 | +0.038 | +0.038 |
| **kaggle_v2** | +0.035 | +0.228 | +0.017 | +0.058 | +0.033 |
| **lohi_weld** | +0.065 | +0.044 | +0.023 | +0.011 | +0.125 |
