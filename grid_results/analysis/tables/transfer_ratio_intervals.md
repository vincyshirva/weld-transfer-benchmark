# Transfer ratios with split-block confidence intervals

`T_ij = P_ij / P_jj`, same configuration, same seed, and a 95 percent split-block percentile interval. **The number of bootstrap draws differs by level and is stated at the head of each section.** Every row also records its own count in the `resamples` column of `transfer_ratio_intervals.csv`.

**The unit is the split block, not the image.** Images from one weld scene are correlated and adjacent video frames are nearly the same photograph, so resampling images independently would treat those as independent evidence and make every interval too narrow.

**Numerator and denominator are resampled together.** Both are scored on the same test images of the target, so one draw selects the blocks and both models are then scored on exactly those images. The diagonal rows are the self-check: when the source is the target the two models are the same model, so T must be exactly 1.000 with zero width.

**This interval answers a different question from the Friedman test.** Friedman asks whether the four configurations differ from each other, and it found no detectable difference. These intervals ask how large the collapse is, which is the claim the paper actually makes.

## Image level

Two scales. The raw ratio inherits the target base rate in both terms, which compresses it towards 1 on the four unbalanced columns. The corrected ratio divides lift by lift, and the base rate is recomputed inside every draw.

**2,000 bootstrap draws per interval at this level.**

| train | test | configuration | T raw [95% CI] | T corrected [95% CI] | lift P_ij [95% CI] |
|---|---|---|---|---|---|
| kaggle_v2 | kaggle_v2 *(diagonal)* | faster_rcnn | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.228 [0.175, 0.350] |
| kaggle_v2 | kaggle_v2 *(diagonal)* | rtdetr_l | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.285 [0.206, 0.469] |
| kaggle_v2 | kaggle_v2 *(diagonal)* | yolo11n | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.258 [0.166, 0.431] |
| kaggle_v2 | kaggle_v2 *(diagonal)* | yolo11s | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.286 [0.190, 0.492] |
| lohi_weld | kaggle_v2 | faster_rcnn | 0.750 [0.592, 0.871] | 0.192 [0.111, 0.559] | 0.044 [0.029, 0.140] |
| lohi_weld | kaggle_v2 | rtdetr_l | 0.696 [0.530, 0.855] | 0.151 [0.027, 0.603] | 0.043 [0.008, 0.195] |
| lohi_weld | kaggle_v2 | yolo11n | 0.735 [0.580, 0.906] | 0.211 [0.106, 0.679] | 0.054 [0.029, 0.182] |
| lohi_weld | kaggle_v2 | yolo11s | 0.689 [0.561, 0.842] | 0.133 [0.052, 0.530] | 0.038 [0.014, 0.195] |
| weld_surface_v2 | kaggle_v2 | faster_rcnn | 0.756 [0.584, 0.901] | 0.210 [0.066, 0.665] | 0.048 [0.018, 0.155] |
| weld_surface_v2 | kaggle_v2 | rtdetr_l | 0.757 [0.540, 0.852] | 0.322 [0.118, 0.536] | 0.092 [0.041, 0.191] |
| weld_surface_v2 | kaggle_v2 | yolo11n | 0.795 [0.569, 0.895] | 0.391 [0.110, 0.640] | 0.101 [0.030, 0.207] |
| weld_surface_v2 | kaggle_v2 | yolo11s | 0.754 [0.511, 0.886] | 0.315 [0.121, 0.602] | 0.090 [0.040, 0.201] |
| kaggle_v2 | lohi_weld | faster_rcnn | 0.994 [0.990, 0.998] | 0.750 [0.604, 0.891] | 0.017 [0.008, 0.029] |
| kaggle_v2 | lohi_weld | rtdetr_l | 0.991 [0.986, 0.997] | 0.593 [0.496, 0.754] | 0.013 [0.006, 0.021] |
| kaggle_v2 | lohi_weld | yolo11n | 0.998 [0.995, 1.000] | 0.920 [0.810, 0.984] | 0.021 [0.009, 0.032] |
| kaggle_v2 | lohi_weld | yolo11s | 0.997 [0.994, 0.999] | 0.885 [0.788, 0.935] | 0.021 [0.009, 0.033] |
| lohi_weld | lohi_weld *(diagonal)* | faster_rcnn | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.023 [0.011, 0.037] |
| lohi_weld | lohi_weld *(diagonal)* | rtdetr_l | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.021 [0.010, 0.034] |
| lohi_weld | lohi_weld *(diagonal)* | yolo11n | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.023 [0.010, 0.036] |
| lohi_weld | lohi_weld *(diagonal)* | yolo11s | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.024 [0.011, 0.037] |
| weld_surface_v2 | lohi_weld | faster_rcnn | 0.980 [0.969, 0.993] | 0.149 [0.081, 0.379] | 0.003 [0.001, 0.009] |
| weld_surface_v2 | lohi_weld | rtdetr_l | 0.980 [0.968, 0.991] | 0.051 [0.033, 0.182] | 0.001 [0.001, 0.004] |
| weld_surface_v2 | lohi_weld | yolo11n | 0.984 [0.974, 0.994] | 0.282 [0.212, 0.463] | 0.006 [0.003, 0.012] |
| weld_surface_v2 | lohi_weld | yolo11s | 0.979 [0.967, 0.992] | 0.120 [0.068, 0.339] | 0.003 [0.001, 0.008] |
| kaggle_v2 | twoclass | faster_rcnn | 0.851 [0.519, 1.000] | 0.186 [0.065, 0.549] **weak reference** | 0.033 [0.000, 0.119] |
| kaggle_v2 | twoclass | rtdetr_l | 0.848 [0.483, 1.000] | 0.192 [0.024, 0.585] **weak reference** | 0.036 [0.000, 0.191] |
| kaggle_v2 | twoclass | yolo11n | 0.881 [0.553, 1.000] | 0.361 [0.108, 0.771] **weak reference** | 0.066 [0.000, 0.286] |
| kaggle_v2 | twoclass | yolo11s | 0.866 [0.498, 1.000] | 0.289 [0.090, 0.698] **weak reference** | 0.054 [0.000, 0.254] |
| lohi_weld | twoclass | faster_rcnn | 0.944 [0.649, 1.000] | 0.695 [0.214, 0.888] **weak reference** | 0.125 [0.000, 0.347] |
| lohi_weld | twoclass | rtdetr_l | 0.848 [0.521, 1.000] | 0.195 [0.066, 0.606] **weak reference** | 0.036 [0.000, 0.168] |
| lohi_weld | twoclass | yolo11n | 0.896 [0.625, 1.000] | 0.441 [0.231, 0.775] **weak reference** | 0.080 [0.000, 0.292] |
| lohi_weld | twoclass | yolo11s | 0.872 [0.481, 1.000] | 0.321 [0.096, 0.818] **weak reference** | 0.060 [0.000, 0.229] |
| weld_surface_v2 | twoclass | faster_rcnn | 0.856 [0.521, 1.000] | 0.214 [0.012, 0.909] **weak reference** | 0.038 [0.000, 0.212] |
| weld_surface_v2 | twoclass | rtdetr_l | 0.869 [0.525, 1.000] | 0.305 [0.030, 0.852] **weak reference** | 0.057 [0.000, 0.254] |
| weld_surface_v2 | twoclass | yolo11n | 0.852 [0.519, 1.000] | 0.203 [0.027, 0.935] **weak reference** | 0.037 [0.000, 0.232] |
| weld_surface_v2 | twoclass | yolo11s | 0.865 [0.509, 1.000] | 0.285 [0.030, 0.978] **weak reference** | 0.053 [0.000, 0.261] |
| kaggle_v2 | visual | faster_rcnn | 0.970 [0.934, 0.995] | 0.662 [0.434, 0.922] | 0.058 [0.022, 0.111] |
| kaggle_v2 | visual | rtdetr_l | 0.945 [0.905, 0.988] | 0.389 [0.285, 0.724] | 0.034 [0.016, 0.077] |
| kaggle_v2 | visual | yolo11n | 0.968 [0.912, 1.006] | 0.604 [0.294, 1.114] | 0.047 [0.019, 0.096] |
| kaggle_v2 | visual | yolo11s | 0.933 [0.881, 0.988] | 0.157 [0.073, 0.731] | 0.012 [0.004, 0.055] |
| lohi_weld | visual | faster_rcnn | 0.921 [0.869, 0.987] | 0.125 [0.029, 0.736] | 0.011 [0.002, 0.060] |
| lohi_weld | visual | rtdetr_l | 0.931 [0.876, 0.983] | 0.238 [0.088, 0.640] | 0.021 [0.006, 0.059] |
| lohi_weld | visual | yolo11n | 0.937 [0.880, 0.987] | 0.223 [0.083, 0.662] | 0.018 [0.005, 0.052] |
| lohi_weld | visual | yolo11s | 0.932 [0.876, 0.987] | 0.141 [0.056, 0.663] | 0.011 [0.004, 0.040] |
| weld_surface_v2 | visual | faster_rcnn | 0.948 [0.895, 0.996] | 0.427 [0.200, 0.928] | 0.038 [0.015, 0.090] |
| weld_surface_v2 | visual | rtdetr_l | 0.922 [0.864, 0.979] | 0.134 [0.077, 0.467] | 0.012 [0.005, 0.038] |
| weld_surface_v2 | visual | yolo11n | 0.930 [0.868, 0.986] | 0.128 [0.037, 0.639] | 0.010 [0.003, 0.047] |
| weld_surface_v2 | visual | yolo11s | 0.928 [0.867, 0.983] | 0.091 [0.015, 0.580] | 0.007 [0.001, 0.039] |
| kaggle_v2 | weld_surface_v2 | faster_rcnn | 0.921 [0.839, 0.977] | 0.313 [0.113, 0.589] | 0.035 [0.006, 0.107] |
| kaggle_v2 | weld_surface_v2 | rtdetr_l | 0.916 [0.834, 0.974] | 0.345 [0.107, 0.622] | 0.043 [0.006, 0.126] |
| kaggle_v2 | weld_surface_v2 | yolo11n | 0.910 [0.825, 0.971] | 0.307 [0.128, 0.520] | 0.040 [0.006, 0.117] |
| kaggle_v2 | weld_surface_v2 | yolo11s | 0.890 [0.793, 0.966] | 0.150 [0.024, 0.425] | 0.019 [0.002, 0.085] |
| lohi_weld | weld_surface_v2 | faster_rcnn | 0.951 [0.892, 0.993] | 0.577 [0.258, 0.918] | 0.065 [0.015, 0.166] |
| lohi_weld | weld_surface_v2 | rtdetr_l | 0.930 [0.833, 0.981] | 0.456 [0.277, 0.625] | 0.058 [0.016, 0.115] |
| lohi_weld | weld_surface_v2 | yolo11n | 0.940 [0.860, 0.986] | 0.540 [0.343, 0.762] | 0.070 [0.019, 0.159] |
| lohi_weld | weld_surface_v2 | yolo11s | 0.914 [0.811, 0.980] | 0.332 [0.157, 0.605] | 0.042 [0.011, 0.110] |
| weld_surface_v2 | weld_surface_v2 *(diagonal)* | faster_rcnn | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.112 [0.035, 0.237] |
| weld_surface_v2 | weld_surface_v2 *(diagonal)* | rtdetr_l | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.126 [0.039, 0.264] |
| weld_surface_v2 | weld_surface_v2 *(diagonal)* | yolo11n | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.129 [0.040, 0.269] |
| weld_surface_v2 | weld_surface_v2 *(diagonal)* | yolo11s | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.128 [0.040, 0.264] |

## Box level

Class-agnostic mAP50, localisation-protocol pairs only. Box AP has no prevalence floor, so no correction applies.

**500 bootstrap draws per interval at this level.** Fewer than at image level, for cost: a box-level draw on lohi_weld rescores up to 135,900 predictions, because both models sit at the 300 detections per image cap across 453 test images.

| train | test | configuration | P_ij | P_jj | T [95% CI] |
|---|---|---|---|---|---|
| kaggle_v2 | kaggle_v2 *(diagonal)* | faster_rcnn | 0.0344 | 0.0344 | 1.000 [1.000, 1.000] |
| kaggle_v2 | kaggle_v2 *(diagonal)* | rtdetr_l | 0.0873 | 0.0873 | 1.000 [1.000, 1.000] |
| kaggle_v2 | kaggle_v2 *(diagonal)* | yolo11n | 0.0877 | 0.0877 | 1.000 [1.000, 1.000] |
| kaggle_v2 | kaggle_v2 *(diagonal)* | yolo11s | 0.1118 | 0.1118 | 1.000 [1.000, 1.000] |
| lohi_weld | kaggle_v2 | faster_rcnn | 0.0014 | 0.0344 | 0.040 [0.019, 0.093] |
| lohi_weld | kaggle_v2 | rtdetr_l | 0.0004 | 0.0873 | 0.005 [0.002, 0.026] |
| lohi_weld | kaggle_v2 | yolo11n | 0.0008 | 0.0877 | 0.009 [0.003, 0.051] |
| lohi_weld | kaggle_v2 | yolo11s | 0.0012 | 0.1118 | 0.010 [0.003, 0.035] |
| kaggle_v2 | lohi_weld | faster_rcnn | 0.0357 | 0.6147 | 0.058 [0.050, 0.070] |
| kaggle_v2 | lohi_weld | rtdetr_l | 0.0110 | 0.6750 | 0.016 [0.011, 0.018] |
| kaggle_v2 | lohi_weld | yolo11n | 0.0220 | 0.6872 | 0.032 [0.026, 0.037] |
| kaggle_v2 | lohi_weld | yolo11s | 0.0157 | 0.6877 | 0.023 [0.016, 0.026] |
| lohi_weld | lohi_weld *(diagonal)* | faster_rcnn | 0.6147 | 0.6147 | 1.000 [1.000, 1.000] |
| lohi_weld | lohi_weld *(diagonal)* | rtdetr_l | 0.6750 | 0.6750 | 1.000 [1.000, 1.000] |
| lohi_weld | lohi_weld *(diagonal)* | yolo11n | 0.6872 | 0.6872 | 1.000 [1.000, 1.000] |
| lohi_weld | lohi_weld *(diagonal)* | yolo11s | 0.6877 | 0.6877 | 1.000 [1.000, 1.000] |
| kaggle_v2 | twoclass | faster_rcnn | 0.0120 | 0.6493 | 0.018 [0.000, 0.106] **weak reference** |
| kaggle_v2 | twoclass | rtdetr_l | 0.0268 | 0.5620 | 0.048 [0.001, 0.236] **weak reference** |
| kaggle_v2 | twoclass | yolo11n | 0.0229 | 0.6040 | 0.038 [0.002, 0.109] **weak reference** |
| kaggle_v2 | twoclass | yolo11s | 0.0144 | 0.5280 | 0.027 [0.000, 0.147] **weak reference** |
| lohi_weld | twoclass | faster_rcnn | 0.0514 | 0.6493 | 0.079 [0.011, 0.115] **weak reference** |
| lohi_weld | twoclass | rtdetr_l | 0.0026 | 0.5620 | 0.005 [0.000, 0.017] **weak reference** |
| lohi_weld | twoclass | yolo11n | 0.0249 | 0.6040 | 0.041 [0.008, 0.089] **weak reference** |
| lohi_weld | twoclass | yolo11s | 0.0258 | 0.5280 | 0.049 [0.004, 0.236] **weak reference** |
| kaggle_v2 | visual | faster_rcnn | 0.0074 | 0.1567 | 0.047 [0.004, 0.129] |
| kaggle_v2 | visual | rtdetr_l | 0.0093 | 0.1302 | 0.071 [0.026, 0.153] |
| kaggle_v2 | visual | yolo11n | 0.0119 | 0.1505 | 0.079 [0.018, 0.197] |
| kaggle_v2 | visual | yolo11s | 0.0087 | 0.1417 | 0.062 [0.026, 0.147] |
| lohi_weld | visual | faster_rcnn | 0.0014 | 0.1567 | 0.009 [0.004, 0.024] |
| lohi_weld | visual | rtdetr_l | 0.0014 | 0.1302 | 0.011 [0.002, 0.045] |
| lohi_weld | visual | yolo11n | 0.0008 | 0.1505 | 0.006 [0.002, 0.018] |
| lohi_weld | visual | yolo11s | 0.0006 | 0.1417 | 0.004 [0.002, 0.013] |
