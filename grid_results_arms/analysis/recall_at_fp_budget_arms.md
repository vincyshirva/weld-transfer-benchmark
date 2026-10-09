# Recall at a fixed false positive budget, arms

the project log 5 makes this the metric that carries the inspection claim. AUROC is unreportable on three of five targets for want of negative images. This counts false positive BOXES per image, so it stays measurable everywhere the box level is defined.

| arm | source | target | cfg | R@0.1 | R@0.5 | R@1.0 | box mAP50 |
|---|---|---|---|---|---|---|---|
| armb | kaggle_v2 | lohi_weld | RT-DETR-l | 0.025 | 0.085 | 0.124 | 0.1793 |
| armb | kaggle_v2 | lohi_weld | YOLO11s | 0.038 | 0.106 | 0.157 | 0.2389 |
| armb | kaggle_v2 | twoclass | RT-DETR-l | 0.085 | 0.228 | 0.292 | 0.2285 |
| armb | kaggle_v2 | twoclass | YOLO11s | 0.062 | 0.133 | 0.167 | 0.1499 |
| armb | kaggle_v2 | visual | RT-DETR-l | 0.004 | 0.011 | 0.017 | 0.0161 |
| armb | kaggle_v2 | visual | YOLO11s | 0.006 | 0.029 | 0.042 | 0.0280 |
| armb | lohi_weld | kaggle_v2 | RT-DETR-l | 0.007 | 0.015 | 0.020 | 0.0064 |
| armb | lohi_weld | kaggle_v2 | YOLO11s | 0.006 | 0.021 | 0.035 | 0.0100 |
| armb | lohi_weld | twoclass | RT-DETR-l | 0.047 | 0.121 | 0.151 | 0.1177 |
| armb | lohi_weld | twoclass | YOLO11s | 0.059 | 0.106 | 0.132 | 0.1237 |
| armb | lohi_weld | visual | RT-DETR-l | 0.002 | 0.006 | 0.010 | 0.0076 |
| armb | lohi_weld | visual | YOLO11s | 0.002 | 0.010 | 0.018 | 0.0142 |
| armd | pool_no_weld_surface_v2 | twoclass | Faster R-CNN | 0.023 | 0.059 | 0.073 | 0.0454 |
| armd | pool_no_weld_surface_v2 | twoclass | RT-DETR-l | 0.027 | 0.059 | 0.100 | 0.0462 |
| armd | pool_no_weld_surface_v2 | twoclass | YOLO11n | 0.100 | 0.128 | 0.142 | 0.1332 |
| armd | pool_no_weld_surface_v2 | twoclass | YOLO11s | 0.027 | 0.064 | 0.082 | 0.0444 |
| armd | pool_no_weld_surface_v2 | visual | Faster R-CNN | 0.001 | 0.012 | 0.017 | 0.0055 |
| armd | pool_no_weld_surface_v2 | visual | RT-DETR-l | 0.006 | 0.014 | 0.019 | 0.0110 |
| armd | pool_no_weld_surface_v2 | visual | YOLO11n | 0.001 | 0.005 | 0.013 | 0.0070 |
| armd | pool_no_weld_surface_v2 | visual | YOLO11s | 0.005 | 0.012 | 0.023 | 0.0103 |
| lohictl | lohi_high | lohi_high | Faster R-CNN | 0.114 | 0.317 | 0.423 | 0.6129 |
| lohictl | lohi_high | lohi_high | RT-DETR-l | 0.131 | 0.353 | 0.494 | 0.6705 |
| lohictl | lohi_high | lohi_high | YOLO11n | 0.147 | 0.388 | 0.499 | 0.6570 |
| lohictl | lohi_high | lohi_high | YOLO11s | 0.156 | 0.400 | 0.506 | 0.6753 |
| lohictl | lohi_high | lohi_low | Faster R-CNN | 0.045 | 0.127 | 0.175 | 0.2399 |
| lohictl | lohi_high | lohi_low | RT-DETR-l | 0.045 | 0.122 | 0.171 | 0.2265 |
| lohictl | lohi_high | lohi_low | YOLO11n | 0.027 | 0.101 | 0.166 | 0.2288 |
| lohictl | lohi_high | lohi_low | YOLO11s | 0.040 | 0.126 | 0.190 | 0.2655 |
| lohictl | lohi_low | lohi_high | Faster R-CNN | 0.052 | 0.135 | 0.194 | 0.3350 |
| lohictl | lohi_low | lohi_high | RT-DETR-l | 0.031 | 0.110 | 0.164 | 0.2813 |
| lohictl | lohi_low | lohi_high | YOLO11n | 0.040 | 0.121 | 0.176 | 0.3003 |
| lohictl | lohi_low | lohi_high | YOLO11s | 0.034 | 0.094 | 0.166 | 0.2957 |
| lohictl | lohi_low | lohi_low | Faster R-CNN | 0.179 | 0.364 | 0.471 | 0.6400 |
| lohictl | lohi_low | lohi_low | RT-DETR-l | 0.195 | 0.437 | 0.555 | 0.6821 |
| lohictl | lohi_low | lohi_low | YOLO11n | 0.195 | 0.426 | 0.553 | 0.6831 |
| lohictl | lohi_low | lohi_low | YOLO11s | 0.212 | 0.432 | 0.542 | 0.6869 |

Arm B rows are pooled over k and over the three draws, so read them as the arm's overall level rather than as a budget curve. The per k values are in `recall_at_fp_budget_arms.csv`.
