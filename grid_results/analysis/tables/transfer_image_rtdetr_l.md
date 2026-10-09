# Image-level transfer, RT-DETR-l

**Read every column against its base rate.** Image-level AP is positive-class average precision, so a constant-positive predictor already scores the positive rate. The floor differs by column, and only a column marked balanced is near 0.5.

| train \ test | weld_surface_v2 | kaggle_v2 | lohi_weld | visual | twoclass |
|---|---|---|---|---|---|
| *base rate (positive class)* | 0.861 | 0.511 **balanced** | 0.976 | 0.893 | 0.800 |
| **weld_surface_v2** | 0.987 | 0.602 | 0.977 | 0.905 | 0.857 |
| **kaggle_v2** | 0.904 | 0.795 | 0.988 | 0.927 | 0.836 |
| **lohi_weld** | 0.918 | 0.553 | 0.997 | 0.914 | 0.836 |

## Same matrix, AP minus base rate

The lift over a constant-positive predictor. This is the quantity that is comparable across columns. Negative means worse than trivial.

| train \ test | weld_surface_v2 | kaggle_v2 | lohi_weld | visual | twoclass |
|---|---|---|---|---|---|
| **weld_surface_v2** | +0.126 | +0.092 | +0.001 | +0.012 | +0.057 |
| **kaggle_v2** | +0.043 | +0.285 | +0.013 | +0.034 | +0.036 |
| **lohi_weld** | +0.058 | +0.043 | +0.021 | +0.021 | +0.036 |
