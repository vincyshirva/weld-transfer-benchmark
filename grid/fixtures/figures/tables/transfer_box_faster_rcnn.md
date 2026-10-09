# Box-level transfer, Faster R-CNN

Only localisation-protocol pairs appear. weld_surface_v2 is whole-region, so its row and column are undefined here, and a number there would measure annotation mismatch rather than domain shift.

| train \ test | kaggle_v2 | lohi_weld | visual | twoclass |
|---|---|---|---|---|
| **kaggle_v2** | 0.564 | 0.523 | 0.319 | 0.208 |
| **lohi_weld** | 0.318 | 0.742 | 0.308 | 0.205 |
