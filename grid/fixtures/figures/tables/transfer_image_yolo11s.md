# Image-level transfer, YOLO11s

**Read every column against its base rate.** Image-level AP is positive-class average precision, so a constant-positive predictor already scores the positive rate. The floor differs by column, and only a column marked balanced is near 0.5.

| train \ test | weld_surface_v2 | kaggle_v2 | lohi_weld | visual | twoclass |
|---|---|---|---|---|---|
| *base rate (positive class)* | 0.861 | 0.511 **balanced** | 0.976 | 0.893 | 0.800 |
| **weld_surface_v2** | 0.886 | 0.402 | 0.661 | 0.426 | 0.276 |
| **kaggle_v2** | 0.391 | 0.744 | 0.701 | 0.426 | 0.262 |
| **lohi_weld** | 0.371 | 0.394 | 0.948 | 0.417 | 0.252 |

## Same matrix, AP minus base rate

The lift over a constant-positive predictor. This is the quantity that is comparable across columns. Negative means worse than trivial.

| train \ test | weld_surface_v2 | kaggle_v2 | lohi_weld | visual | twoclass |
|---|---|---|---|---|---|
| **weld_surface_v2** | +0.025 | -0.109 | -0.315 | -0.467 | -0.524 |
| **kaggle_v2** | -0.469 | +0.233 | -0.275 | -0.467 | -0.538 |
| **lohi_weld** | -0.490 | -0.117 | -0.028 | -0.476 | -0.548 |
