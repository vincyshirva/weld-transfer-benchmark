# Box-level transfer, RT-DETR-l

Only localisation-protocol pairs appear. weld_surface_v2 is whole-region, so its row and column are undefined here, and a number there would measure annotation mismatch rather than domain shift.

| train \ test | kaggle_v2 | lohi_weld | visual | twoclass |
|---|---|---|---|---|
| **kaggle_v2** | 0.087 | 0.011 | 0.009 | 0.027 |
| **lohi_weld** | 0.000 | 0.675 | 0.001 | 0.003 |
