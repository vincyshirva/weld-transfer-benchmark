# Image-level transfer, RT-DETR-l

**Read every column against its base rate.** Image-level AP is positive-class average precision, so a constant-positive predictor already scores the positive rate. The floor differs by column, and only a column marked balanced is near 0.5.

| train \ test | weld_surface_v2 | kaggle_v2 | lohi_weld | visual | twoclass |
|---|---|---|---|---|---|
| *base rate (positive class)* | 0.861 | 0.511 **balanced** | 0.976 | 0.893 | 0.800 |
| **weld_surface_v2** | 0.880 | 0.440 | 0.708 | 0.450 | 0.291 |
| **kaggle_v2** | 0.415 | 0.754 | 0.714 | 0.453 | 0.279 |
| **lohi_weld** | 0.413 | 0.438 | 0.998 | 0.443 | 0.297 |

## Same matrix, AP minus base rate

The lift over a constant-positive predictor. This is the quantity that is comparable across columns. Negative means worse than trivial.

| train \ test | weld_surface_v2 | kaggle_v2 | lohi_weld | visual | twoclass |
|---|---|---|---|---|---|
| **weld_surface_v2** | +0.019 | -0.070 | -0.268 | -0.443 | -0.509 |
| **kaggle_v2** | -0.446 | +0.243 | -0.261 | -0.440 | -0.521 |
| **lohi_weld** | -0.447 | -0.072 | +0.023 | -0.450 | -0.503 |
