# Image-level transfer, Faster R-CNN

**Read every column against its base rate.** Image-level AP is positive-class average precision, so a constant-positive predictor already scores the positive rate. The floor differs by column, and only a column marked balanced is near 0.5.

| train \ test | weld_surface_v2 | kaggle_v2 | lohi_weld | visual | twoclass |
|---|---|---|---|---|---|
| *base rate (positive class)* | 0.861 | 0.511 **balanced** | 0.976 | 0.893 | 0.800 |
| **weld_surface_v2** | 0.909 | 0.428 | 0.695 | 0.412 | 0.261 |
| **kaggle_v2** | 0.420 | 0.750 | 0.702 | 0.424 | 0.285 |
| **lohi_weld** | 0.398 | 0.406 | 0.988 | 0.421 | 0.273 |

## Same matrix, AP minus base rate

The lift over a constant-positive predictor. This is the quantity that is comparable across columns. Negative means worse than trivial.

| train \ test | weld_surface_v2 | kaggle_v2 | lohi_weld | visual | twoclass |
|---|---|---|---|---|---|
| **weld_surface_v2** | +0.049 | -0.082 | -0.280 | -0.481 | -0.539 |
| **kaggle_v2** | -0.441 | +0.239 | -0.273 | -0.469 | -0.515 |
| **lohi_weld** | -0.462 | -0.105 | +0.012 | -0.472 | -0.527 |
