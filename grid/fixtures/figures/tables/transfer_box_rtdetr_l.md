# Box-level transfer, RT-DETR-l

Only localisation-protocol pairs appear. weld_surface_v2 is whole-region, so its row and column are undefined here, and a number there would measure annotation mismatch rather than domain shift.

| train \ test | kaggle_v2 | lohi_weld | visual | twoclass |
|---|---|---|---|---|
| **kaggle_v2** | 0.582 | 0.546 | 0.332 | 0.210 |
| **lohi_weld** | 0.320 | 0.721 | 0.333 | 0.219 |
