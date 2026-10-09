# Image-level AUROC, source by target

AUROC has a chance level of 0.5, and ties are averaged, so a constant score gets exactly 0.5. It is computed here for every target, including the three where the evaluator suppresses it under 30 negative images. Those rows are flagged in the last column. The interval is a 95 percent split-block percentile interval on the twelve-cell mean, 10,000 draws, with the twelve cells averaged inside each draw. "AP lift" is image-level AP minus the positive rate. "Chance lift" is the mean lift when the labels are shuffled 2,000 times against each cell's real scores. It is the level an uninformative scorer reaches on this estimator, not zero. The last two counts are evaluations whose OWN interval lies wholly above or wholly below 0.5.

| Target | Trained on | YOLO11n | YOLO11s | RT-DETR-l | Faster R-CNN | Mean AUROC, with interval | AP lift | Chance lift | Wholly above 0.5 | Wholly below 0.5 | Negatives under the rule |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kaggle_v2 | kaggle_v2, in domain | 0.747 | 0.788 | 0.777 | 0.745 | 0.764 [0.692, 0.843] | +0.264 | +0.052 | 12 of 12 | 0 of 12 | no |
| kaggle_v2 | weld_surface_v2 | 0.574 | 0.562 | 0.564 | 0.519 | 0.555 [0.448, 0.618] | +0.083 | +0.052 | 0 of 12 | 0 of 12 | no |
| kaggle_v2 | lohi_weld | 0.524 | 0.489 | 0.472 | 0.511 | 0.499 [0.430, 0.624] | +0.045 | +0.052 | 0 of 12 | 0 of 12 | no |
| weld_surface_v2 | kaggle_v2 | 0.570 | 0.512 | 0.591 | 0.542 | 0.554 [0.388, 0.658] | +0.034 | +0.018 | 0 of 12 | 0 of 12 | no |
| weld_surface_v2 | weld_surface_v2, in domain | 0.944 | 0.940 | 0.931 | 0.875 | 0.922 [0.869, 0.977] | +0.124 | +0.018 | 12 of 12 | 0 of 12 | no |
| weld_surface_v2 | lohi_weld | 0.671 | 0.569 | 0.633 | 0.671 | 0.636 [0.514, 0.730] | +0.059 | +0.018 | 4 of 12 | 0 of 12 | no |
| lohi_weld | kaggle_v2 | 0.849 | 0.865 | 0.638 | 0.767 | 0.780 [0.739, 0.813] | +0.018 | +0.006 | 10 of 12 | 0 of 12 | yes |
| lohi_weld | weld_surface_v2 | 0.449 | 0.345 | 0.213 | 0.373 | 0.345 [0.292, 0.402] | +0.003 | +0.006 | 0 of 12 | 7 of 12 | yes |
| lohi_weld | lohi_weld, in domain | 0.940 | 0.979 | 0.897 | 0.958 | 0.944 [0.909, 0.971] | +0.023 | +0.006 | 12 of 12 | 0 of 12 | yes |
| visual | kaggle_v2 | 0.604 | 0.414 | 0.544 | 0.665 | 0.557 [0.482, 0.650] | +0.038 | +0.030 | 0 of 12 | 0 of 12 | yes |
| visual | weld_surface_v2 | 0.335 | 0.288 | 0.356 | 0.560 | 0.385 [0.260, 0.544] | +0.017 | +0.030 | 0 of 12 | 3 of 12 | yes |
| visual | lohi_weld | 0.427 | 0.368 | 0.441 | 0.405 | 0.410 [0.310, 0.520] | +0.015 | +0.030 | 0 of 12 | 1 of 12 | yes |
| visual | visual, in domain | 0.780 | 0.772 | 0.849 | 0.849 | 0.813 [0.725, 0.881] | +0.083 | +0.030 | 12 of 12 | 0 of 12 | yes |
| twoclass | kaggle_v2 | 0.506 | 0.507 | 0.429 | 0.401 | 0.461 [0.283, 0.663] | +0.047 | +0.055 | 0 of 12 | 0 of 12 | yes |
| twoclass | weld_surface_v2 | 0.444 | 0.493 | 0.529 | 0.424 | 0.473 [0.167, 0.920] | +0.046 | +0.054 | 0 of 12 | 0 of 12 | yes |
| twoclass | lohi_weld | 0.593 | 0.515 | 0.406 | 0.709 | 0.556 [0.435, 0.745] | +0.075 | +0.055 | 2 of 12 | 0 of 12 | yes |
| twoclass | twoclass, in domain | 0.927 | 0.940 | 0.942 | 0.916 | 0.931 [0.858, 0.992] | +0.183 | +0.055 | 12 of 12 | 0 of 12 | yes |

## Retained share of the in-domain margin above chance

(AUROC_ij - 0.5) over (AUROC_jj - 0.5), where j's own model is the same configuration and seed. Numerator and denominator are scored on the same resampled blocks, and over the four configurations all twelve cells are averaged inside each draw on both sides. A draw whose denominator is not above 0.5 is undefined and counted. The diagonal is exactly 1 by construction and is checked.

| Target | Trained on | Subset | Retained margin, with interval | Undefined draws | Interval excludes zero |
|---|---|---|---|---|---|
| weld_surface_v2 | kaggle_v2 | full | +0.127 [-0.254, +0.390] | 2 of 10,000 | no |
| weld_surface_v2 | lohi_weld | full | +0.322 [+0.032, +0.575] | 2 of 10,000 | yes |
| kaggle_v2 | weld_surface_v2 | full | +0.207 [-0.197, +0.434] | 0 of 10,000 | no |
| kaggle_v2 | lohi_weld | full | -0.004 [-0.279, +0.438] | 0 of 10,000 | no |
| lohi_weld | weld_surface_v2 | full | -0.350 [-0.481, -0.219] | 0 of 10,000 | yes |
| lohi_weld | kaggle_v2 | full | +0.630 [+0.534, +0.720] | 0 of 10,000 | yes |
| visual | weld_surface_v2 | full | -0.369 [-0.699, +0.163] | 2 of 10,000 | no |
| visual | kaggle_v2 | full | +0.182 [-0.058, +0.530] | 2 of 10,000 | no |
| visual | lohi_weld | full | -0.286 [-0.657, +0.072] | 2 of 10,000 | no |
| twoclass | weld_surface_v2 | full | -0.064 [-0.899, +0.859] | 896 of 10,000 | no |
| twoclass | kaggle_v2 | full | -0.091 [-0.605, +0.348] | 896 of 10,000 | no |
| twoclass | lohi_weld | full | +0.129 [-0.170, +0.524] | 896 of 10,000 | no |
| kaggle_v2 | weld_surface_v2 | without the SampleV1_2_mp4 block | +0.133 [-0.308, +0.461] | 0 of 10,000 | no |
| kaggle_v2 | lohi_weld | without the SampleV1_2_mp4 block | +0.247 [-0.065, +0.509] | 0 of 10,000 | no |
