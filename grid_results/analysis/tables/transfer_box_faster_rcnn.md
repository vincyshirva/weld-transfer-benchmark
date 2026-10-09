# Box-level transfer, Faster R-CNN

Only localisation-protocol pairs appear. weld_surface_v2 is whole-region, so its row and column are undefined here, and a number there would measure annotation mismatch rather than domain shift.

| train \ test | kaggle_v2 | lohi_weld | visual | twoclass |
|---|---|---|---|---|
| **kaggle_v2** | 0.034 | 0.036 | 0.007 | 0.012 |
| **lohi_weld** | 0.001 | 0.615 | 0.001 | 0.051 |
