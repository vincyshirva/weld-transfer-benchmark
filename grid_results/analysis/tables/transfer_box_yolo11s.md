# Box-level transfer, YOLO11s

Only localisation-protocol pairs appear. weld_surface_v2 is whole-region, so its row and column are undefined here, and a number there would measure annotation mismatch rather than domain shift.

| train \ test | kaggle_v2 | lohi_weld | visual | twoclass |
|---|---|---|---|---|
| **kaggle_v2** | 0.112 | 0.016 | 0.009 | 0.014 |
| **lohi_weld** | 0.001 | 0.688 | 0.001 | 0.026 |
