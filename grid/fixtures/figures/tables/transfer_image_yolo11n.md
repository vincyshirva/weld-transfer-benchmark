# Image-level transfer, YOLO11n

**Read every column against its base rate.** Image-level AP is positive-class average precision, so a constant-positive predictor already scores the positive rate. The floor differs by column, and only a column marked balanced is near 0.5.

| train \ test | weld_surface_v2 | kaggle_v2 | lohi_weld | visual | twoclass |
|---|---|---|---|---|---|
| *base rate (positive class)* | 0.861 | 0.511 **balanced** | 0.976 | 0.893 | 0.800 |
| **weld_surface_v2** | 0.848 | 0.366 | 0.637 | 0.378 | 0.214 |
| **kaggle_v2** | 0.392 | 0.707 | 0.643 | 0.385 | 0.247 |
| **lohi_weld** | 0.363 | 0.359 | 0.920 | 0.382 | 0.244 |

## Same matrix, AP minus base rate

The lift over a constant-positive predictor. This is the quantity that is comparable across columns. Negative means worse than trivial.

| train \ test | weld_surface_v2 | kaggle_v2 | lohi_weld | visual | twoclass |
|---|---|---|---|---|---|
| **weld_surface_v2** | -0.013 | -0.145 | -0.339 | -0.515 | -0.586 |
| **kaggle_v2** | -0.469 | +0.197 | -0.332 | -0.508 | -0.553 |
| **lohi_weld** | -0.497 | -0.151 | -0.056 | -0.511 | -0.556 |
