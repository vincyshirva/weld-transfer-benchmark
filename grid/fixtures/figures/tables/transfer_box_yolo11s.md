# Box-level transfer, YOLO11s

Only localisation-protocol pairs appear. weld_surface_v2 is whole-region, so its row and column are undefined here, and a number there would measure annotation mismatch rather than domain shift.

| train \ test | kaggle_v2 | lohi_weld | visual | twoclass |
|---|---|---|---|---|
| **kaggle_v2** | 0.562 | 0.540 | 0.322 | 0.214 |
| **lohi_weld** | 0.307 | 0.704 | 0.306 | 0.209 |
