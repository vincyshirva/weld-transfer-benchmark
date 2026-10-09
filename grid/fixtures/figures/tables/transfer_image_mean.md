# Image-level transfer, mean over configurations

**Read every column against its base rate.** Image-level AP is positive-class average precision, so a constant-positive predictor already scores the positive rate. The floor differs by column, and only a column marked balanced is near 0.5.

| train \ test | weld_surface_v2 | kaggle_v2 | lohi_weld | visual | twoclass |
|---|---|---|---|---|---|
| *base rate (positive class)* | 0.861 | 0.511 **balanced** | 0.976 | 0.893 | 0.800 |
| **weld_surface_v2** | 0.881 | 0.409 | 0.675 | 0.416 | 0.260 |
| **kaggle_v2** | 0.404 | 0.739 | 0.690 | 0.422 | 0.268 |
| **lohi_weld** | 0.387 | 0.399 | 0.964 | 0.416 | 0.266 |

## Same matrix, AP minus base rate

The lift over a constant-positive predictor. This is the quantity that is comparable across columns. Negative means worse than trivial.

| train \ test | weld_surface_v2 | kaggle_v2 | lohi_weld | visual | twoclass |
|---|---|---|---|---|---|
| **weld_surface_v2** | +0.020 | -0.102 | -0.301 | -0.477 | -0.540 |
| **kaggle_v2** | -0.456 | +0.228 | -0.285 | -0.471 | -0.532 |
| **lohi_weld** | -0.474 | -0.111 | -0.012 | -0.477 | -0.534 |
