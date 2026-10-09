# Arm B k-shot recovery

Primary endpoint `last`, the project log 6a-ii. Plotted against the REALISED scene count, the project log 6a-i, never against requested k. Class space corrected, the project log 6a-iii.

| end | metric | pair | cfg | k | images | scenes | AP | AP - base | P(0) | P_jj | denom | R |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| best | box | kaggle_v2 to lohi_weld | RT-DETR-l | 10 | 10 | 7 | 0.0141 | n/a | 0.0110 | 0.6750 | 0.6640 | 0.006 |
| best | box | kaggle_v2 to lohi_weld | RT-DETR-l | 25 | 25 | 16 | 0.0438 | n/a | 0.0110 | 0.6750 | 0.6640 | 0.052 |
| best | box | kaggle_v2 to lohi_weld | RT-DETR-l | 50 | 70 | 31 | 0.1105 | n/a | 0.0110 | 0.6750 | 0.6640 | 0.149 |
| best | box | kaggle_v2 to lohi_weld | RT-DETR-l | 100 | 100 | 69 | 0.2891 | n/a | 0.0110 | 0.6750 | 0.6640 | 0.424 |
| best | box | kaggle_v2 to lohi_weld | RT-DETR-l | 250 | 250 | 170 | 0.4500 | n/a | 0.0110 | 0.6750 | 0.6640 | 0.665 |
| best | box | kaggle_v2 to lohi_weld | YOLO11s | 10 | 10 | 7 | 0.0790 | n/a | 0.0157 | 0.6877 | 0.6721 | 0.095 |
| best | box | kaggle_v2 to lohi_weld | YOLO11s | 25 | 25 | 16 | 0.1517 | n/a | 0.0157 | 0.6877 | 0.6721 | 0.203 |
| best | box | kaggle_v2 to lohi_weld | YOLO11s | 50 | 70 | 31 | 0.2700 | n/a | 0.0157 | 0.6877 | 0.6721 | 0.377 |
| best | box | kaggle_v2 to lohi_weld | YOLO11s | 100 | 100 | 69 | 0.3963 | n/a | 0.0157 | 0.6877 | 0.6721 | 0.566 |
| best | box | kaggle_v2 to lohi_weld | YOLO11s | 250 | 250 | 170 | 0.4954 | n/a | 0.0157 | 0.6877 | 0.6721 | 0.714 |
| best | box | kaggle_v2 to twoclass | RT-DETR-l | 10 | 20 | 2 | 0.0220 | n/a | 0.0268 | 0.5620 | 0.5352 | -0.008 |
| best | box | kaggle_v2 to twoclass | RT-DETR-l | 25 | 47 | 2 | 0.0114 | n/a | 0.0268 | 0.5620 | 0.5352 | -0.029 |
| best | box | kaggle_v2 to twoclass | RT-DETR-l | 50 | 57 | 5 | 0.1341 | n/a | 0.0268 | 0.5620 | 0.5352 | 0.192 |
| best | box | kaggle_v2 to twoclass | RT-DETR-l | 100 | 108 | 13 | 0.2967 | n/a | 0.0268 | 0.5620 | 0.5352 | 0.488 |
| best | box | kaggle_v2 to twoclass | RT-DETR-l | 250 * | 213 | 22 | 0.5773 | n/a | 0.0268 | 0.5620 | 0.5352 | 1.037 |
| best | box | kaggle_v2 to twoclass | YOLO11s | 10 | 20 | 2 | 0.0193 | n/a | 0.0144 | 0.5280 | 0.5136 | 0.011 |
| best | box | kaggle_v2 to twoclass | YOLO11s | 25 | 47 | 2 | 0.0104 | n/a | 0.0144 | 0.5280 | 0.5136 | -0.009 |
| best | box | kaggle_v2 to twoclass | YOLO11s | 50 | 57 | 5 | 0.1044 | n/a | 0.0144 | 0.5280 | 0.5136 | 0.162 |
| best | box | kaggle_v2 to twoclass | YOLO11s | 100 | 108 | 13 | 0.1948 | n/a | 0.0144 | 0.5280 | 0.5136 | 0.340 |
| best | box | kaggle_v2 to twoclass | YOLO11s | 250 * | 213 | 22 | 0.4354 | n/a | 0.0144 | 0.5280 | 0.5136 | 0.834 |
| best | box | kaggle_v2 to visual | RT-DETR-l | 10 | 13 | 4 | 0.0159 | n/a | 0.0093 | 0.1302 | 0.1209 | 0.060 |
| best | box | kaggle_v2 to visual | RT-DETR-l | 25 | 25 | 12 | 0.0150 | n/a | 0.0093 | 0.1302 | 0.1209 | 0.043 |
| best | box | kaggle_v2 to visual | RT-DETR-l | 50 | 54 | 24 | 0.0112 | n/a | 0.0093 | 0.1302 | 0.1209 | 0.013 |
| best | box | kaggle_v2 to visual | RT-DETR-l | 100 | 102 | 54 | 0.0148 | n/a | 0.0093 | 0.1302 | 0.1209 | 0.035 |
| best | box | kaggle_v2 to visual | RT-DETR-l | 250 | 255 | 114 | 0.0210 | n/a | 0.0093 | 0.1302 | 0.1209 | 0.105 |
| best | box | kaggle_v2 to visual | YOLO11s | 10 | 13 | 4 | 0.0000 | n/a | 0.0087 | 0.1417 | 0.1329 | -0.067 |
| best | box | kaggle_v2 to visual | YOLO11s | 25 | 25 | 12 | 0.0000 | n/a | 0.0087 | 0.1417 | 0.1329 | -0.067 |
| best | box | kaggle_v2 to visual | YOLO11s | 50 | 54 | 24 | 0.0018 | n/a | 0.0087 | 0.1417 | 0.1329 | -0.053 |
| best | box | kaggle_v2 to visual | YOLO11s | 100 | 102 | 54 | 0.0470 | n/a | 0.0087 | 0.1417 | 0.1329 | 0.287 |
| best | box | kaggle_v2 to visual | YOLO11s | 250 | 255 | 114 | 0.1045 | n/a | 0.0087 | 0.1417 | 0.1329 | 0.718 |
| best | box | kaggle_v2 to weld_surface_v2 | RT-DETR-l | 10 | 23 | 2 | n/a | n/a | n/a | n/a | n/a | n/a |
| best | box | kaggle_v2 to weld_surface_v2 | RT-DETR-l | 25 | 35 | 6 | n/a | n/a | n/a | n/a | n/a | n/a |
| best | box | kaggle_v2 to weld_surface_v2 | RT-DETR-l | 50 | 96 | 14 | n/a | n/a | n/a | n/a | n/a | n/a |
| best | box | kaggle_v2 to weld_surface_v2 | RT-DETR-l | 100 | 152 | 13 | n/a | n/a | n/a | n/a | n/a | n/a |
| best | box | kaggle_v2 to weld_surface_v2 | RT-DETR-l | 250 | 255 | 36 | n/a | n/a | n/a | n/a | n/a | n/a |
| best | box | kaggle_v2 to weld_surface_v2 | YOLO11s | 10 | 23 | 2 | n/a | n/a | n/a | n/a | n/a | n/a |
| best | box | kaggle_v2 to weld_surface_v2 | YOLO11s | 25 | 35 | 6 | n/a | n/a | n/a | n/a | n/a | n/a |
| best | box | kaggle_v2 to weld_surface_v2 | YOLO11s | 50 | 96 | 14 | n/a | n/a | n/a | n/a | n/a | n/a |
| best | box | kaggle_v2 to weld_surface_v2 | YOLO11s | 100 | 152 | 13 | n/a | n/a | n/a | n/a | n/a | n/a |
| best | box | kaggle_v2 to weld_surface_v2 | YOLO11s | 250 | 255 | 36 | n/a | n/a | n/a | n/a | n/a | n/a |
| best | box | lohi_weld to kaggle_v2 | RT-DETR-l | 10 | 38 | 8 | 0.0003 | n/a | 0.0004 | 0.0873 | 0.0868 | -0.001 |
| best | box | lohi_weld to kaggle_v2 | RT-DETR-l | 25 | 56 | 12 | 0.0007 | n/a | 0.0004 | 0.0873 | 0.0868 | 0.003 |
| best | box | lohi_weld to kaggle_v2 | RT-DETR-l | 50 | 111 | 24 | 0.0017 | n/a | 0.0004 | 0.0873 | 0.0868 | 0.017 |
| best | box | lohi_weld to kaggle_v2 | RT-DETR-l | 100 | 114 | 12 | 0.0021 | n/a | 0.0004 | 0.0873 | 0.0868 | 0.021 |
| best | box | lohi_weld to kaggle_v2 | RT-DETR-l | 250 | 277 | 92 | 0.0280 | n/a | 0.0004 | 0.0873 | 0.0868 | 0.324 |
| best | box | lohi_weld to kaggle_v2 | YOLO11s | 10 | 38 | 8 | 0.0021 | n/a | 0.0012 | 0.1118 | 0.1107 | 0.008 |
| best | box | lohi_weld to kaggle_v2 | YOLO11s | 25 | 56 | 12 | 0.0011 | n/a | 0.0012 | 0.1118 | 0.1107 | -0.001 |
| best | box | lohi_weld to kaggle_v2 | YOLO11s | 50 | 111 | 24 | 0.0046 | n/a | 0.0012 | 0.1118 | 0.1107 | 0.030 |
| best | box | lohi_weld to kaggle_v2 | YOLO11s | 100 | 114 | 12 | 0.0100 | n/a | 0.0012 | 0.1118 | 0.1107 | 0.084 |
| best | box | lohi_weld to kaggle_v2 | YOLO11s | 250 | 277 | 92 | 0.0331 | n/a | 0.0012 | 0.1118 | 0.1107 | 0.288 |
| best | box | lohi_weld to twoclass | RT-DETR-l | 10 | 20 | 2 | 0.0035 | n/a | 0.0026 | 0.5620 | 0.5594 | 0.002 |
| best | box | lohi_weld to twoclass | RT-DETR-l | 25 | 47 | 2 | 0.0026 | n/a | 0.0026 | 0.5620 | 0.5594 | 0.000 |
| best | box | lohi_weld to twoclass | RT-DETR-l | 50 | 57 | 5 | 0.0203 | n/a | 0.0026 | 0.5620 | 0.5594 | 0.030 |
| best | box | lohi_weld to twoclass | RT-DETR-l | 100 | 108 | 13 | 0.1630 | n/a | 0.0026 | 0.5620 | 0.5594 | 0.278 |
| best | box | lohi_weld to twoclass | RT-DETR-l | 250 * | 213 | 22 | 0.3946 | n/a | 0.0026 | 0.5620 | 0.5594 | 0.705 |
| best | box | lohi_weld to twoclass | YOLO11s | 10 | 20 | 2 | 0.0265 | n/a | 0.0258 | 0.5280 | 0.5022 | 0.005 |
| best | box | lohi_weld to twoclass | YOLO11s | 25 | 47 | 2 | 0.0071 | n/a | 0.0258 | 0.5280 | 0.5022 | -0.038 |
| best | box | lohi_weld to twoclass | YOLO11s | 50 | 57 | 5 | 0.1031 | n/a | 0.0258 | 0.5280 | 0.5022 | 0.140 |
| best | box | lohi_weld to twoclass | YOLO11s | 100 | 108 | 13 | 0.1640 | n/a | 0.0258 | 0.5280 | 0.5022 | 0.259 |
| best | box | lohi_weld to twoclass | YOLO11s | 250 * | 213 | 22 | 0.3494 | n/a | 0.0258 | 0.5280 | 0.5022 | 0.647 |
| best | box | lohi_weld to visual | RT-DETR-l | 10 | 13 | 4 | 0.0009 | n/a | 0.0014 | 0.1302 | 0.1288 | -0.004 |
| best | box | lohi_weld to visual | RT-DETR-l | 25 | 25 | 12 | 0.0007 | n/a | 0.0014 | 0.1302 | 0.1288 | -0.005 |
| best | box | lohi_weld to visual | RT-DETR-l | 50 | 54 | 24 | 0.0005 | n/a | 0.0014 | 0.1302 | 0.1288 | -0.007 |
| best | box | lohi_weld to visual | RT-DETR-l | 100 | 102 | 54 | 0.0058 | n/a | 0.0014 | 0.1302 | 0.1288 | 0.029 |
| best | box | lohi_weld to visual | RT-DETR-l | 250 | 255 | 114 | 0.0299 | n/a | 0.0014 | 0.1302 | 0.1288 | 0.222 |
| best | box | lohi_weld to visual | YOLO11s | 10 | 13 | 4 | 0.0010 | n/a | 0.0006 | 0.1417 | 0.1411 | 0.002 |
| best | box | lohi_weld to visual | YOLO11s | 25 | 25 | 12 | 0.0024 | n/a | 0.0006 | 0.1417 | 0.1411 | 0.013 |
| best | box | lohi_weld to visual | YOLO11s | 50 | 54 | 24 | 0.0106 | n/a | 0.0006 | 0.1417 | 0.1411 | 0.070 |
| best | box | lohi_weld to visual | YOLO11s | 100 | 102 | 54 | 0.0129 | n/a | 0.0006 | 0.1417 | 0.1411 | 0.088 |
| best | box | lohi_weld to visual | YOLO11s | 250 | 255 | 114 | 0.0463 | n/a | 0.0006 | 0.1417 | 0.1411 | 0.324 |
| best | box | lohi_weld to weld_surface_v2 | RT-DETR-l | 10 | 23 | 2 | n/a | n/a | n/a | n/a | n/a | n/a |
| best | box | lohi_weld to weld_surface_v2 | RT-DETR-l | 25 | 35 | 6 | n/a | n/a | n/a | n/a | n/a | n/a |
| best | box | lohi_weld to weld_surface_v2 | RT-DETR-l | 50 | 96 | 14 | n/a | n/a | n/a | n/a | n/a | n/a |
| best | box | lohi_weld to weld_surface_v2 | RT-DETR-l | 100 | 152 | 13 | n/a | n/a | n/a | n/a | n/a | n/a |
| best | box | lohi_weld to weld_surface_v2 | RT-DETR-l | 250 | 255 | 36 | n/a | n/a | n/a | n/a | n/a | n/a |
| best | box | lohi_weld to weld_surface_v2 | YOLO11s | 10 | 23 | 2 | n/a | n/a | n/a | n/a | n/a | n/a |
| best | box | lohi_weld to weld_surface_v2 | YOLO11s | 25 | 35 | 6 | n/a | n/a | n/a | n/a | n/a | n/a |
| best | box | lohi_weld to weld_surface_v2 | YOLO11s | 50 | 96 | 14 | n/a | n/a | n/a | n/a | n/a | n/a |
| best | box | lohi_weld to weld_surface_v2 | YOLO11s | 100 | 152 | 13 | n/a | n/a | n/a | n/a | n/a | n/a |
| best | box | lohi_weld to weld_surface_v2 | YOLO11s | 250 | 255 | 36 | n/a | n/a | n/a | n/a | n/a | n/a |
| best | image | kaggle_v2 to lohi_weld | RT-DETR-l | 10 | 10 | 7 | 0.9891 | 0.0133 | 0.9884 | 0.9971 | 0.0087 ! | 0.083 |
| best | image | kaggle_v2 to lohi_weld | RT-DETR-l | 25 | 25 | 16 | 0.9914 | 0.0157 | 0.9884 | 0.9971 | 0.0087 ! | 0.260 |
| best | image | kaggle_v2 to lohi_weld | RT-DETR-l | 50 | 70 | 31 | 0.9902 | 0.0145 | 0.9884 | 0.9971 | 0.0087 ! | -0.009 |
| best | image | kaggle_v2 to lohi_weld | RT-DETR-l | 100 | 100 | 69 | 0.9961 | 0.0204 | 0.9884 | 0.9971 | 0.0087 ! | 0.956 |
| best | image | kaggle_v2 to lohi_weld | RT-DETR-l | 250 | 250 | 170 | 0.9935 | 0.0178 | 0.9884 | 0.9971 | 0.0087 ! | 0.613 |
| best | image | kaggle_v2 to lohi_weld | YOLO11s | 10 | 10 | 7 | 0.9894 | 0.0137 | 0.9960 | 0.9994 | 0.0033 ! | -2.244 |
| best | image | kaggle_v2 to lohi_weld | YOLO11s | 25 | 25 | 16 | 0.9932 | 0.0175 | 0.9960 | 0.9994 | 0.0033 ! | -0.979 |
| best | image | kaggle_v2 to lohi_weld | YOLO11s | 50 | 70 | 31 | 0.9944 | 0.0187 | 0.9960 | 0.9994 | 0.0033 ! | -0.522 |
| best | image | kaggle_v2 to lohi_weld | YOLO11s | 100 | 100 | 69 | 0.9966 | 0.0209 | 0.9960 | 0.9994 | 0.0033 ! | 0.179 |
| best | image | kaggle_v2 to lohi_weld | YOLO11s | 250 | 250 | 170 | 0.9935 | 0.0178 | 0.9960 | 0.9994 | 0.0033 ! | -0.782 |
| best | image | kaggle_v2 to twoclass | RT-DETR-l | 10 | 20 | 2 | 0.8562 | 0.0562 | 0.8358 | 0.9862 | 0.1504 | 0.128 |
| best | image | kaggle_v2 to twoclass | RT-DETR-l | 25 | 47 | 2 | 0.8520 | 0.0520 | 0.8358 | 0.9862 | 0.1504 | 0.090 |
| best | image | kaggle_v2 to twoclass | RT-DETR-l | 50 | 57 | 5 | 0.8569 | 0.0569 | 0.8358 | 0.9862 | 0.1504 | 0.157 |
| best | image | kaggle_v2 to twoclass | RT-DETR-l | 100 | 108 | 13 | 0.8764 | 0.0764 | 0.8358 | 0.9862 | 0.1504 | 0.282 |
| best | image | kaggle_v2 to twoclass | RT-DETR-l | 250 * | 213 | 22 | 0.9248 | 0.1248 | 0.8358 | 0.9862 | 0.1504 | 0.584 |
| best | image | kaggle_v2 to twoclass | YOLO11s | 10 | 20 | 2 | 0.8799 | 0.0799 | 0.8536 | 0.9855 | 0.1320 | 0.180 |
| best | image | kaggle_v2 to twoclass | YOLO11s | 25 | 47 | 2 | 0.9050 | 0.1050 | 0.8536 | 0.9855 | 0.1320 | 0.364 |
| best | image | kaggle_v2 to twoclass | YOLO11s | 50 | 57 | 5 | 0.9068 | 0.1068 | 0.8536 | 0.9855 | 0.1320 | 0.376 |
| best | image | kaggle_v2 to twoclass | YOLO11s | 100 | 108 | 13 | 0.9351 | 0.1351 | 0.8536 | 0.9855 | 0.1320 | 0.623 |
| best | image | kaggle_v2 to twoclass | YOLO11s | 250 * | 213 | 22 | 0.9358 | 0.1358 | 0.8536 | 0.9855 | 0.1320 | 0.628 |
| best | image | kaggle_v2 to visual | RT-DETR-l | 10 | 13 | 4 | 0.9233 | 0.0304 | 0.9272 | 0.9812 | 0.0539 ! | -0.086 |
| best | image | kaggle_v2 to visual | RT-DETR-l | 25 | 25 | 12 | 0.9396 | 0.0467 | 0.9272 | 0.9812 | 0.0539 ! | 0.192 |
| best | image | kaggle_v2 to visual | RT-DETR-l | 50 | 54 | 24 | 0.9233 | 0.0305 | 0.9272 | 0.9812 | 0.0539 ! | -0.093 |
| best | image | kaggle_v2 to visual | RT-DETR-l | 100 | 102 | 54 | 0.9260 | 0.0332 | 0.9272 | 0.9812 | 0.0539 ! | -0.043 |
| best | image | kaggle_v2 to visual | RT-DETR-l | 250 | 255 | 114 | 0.9176 | 0.0248 | 0.9272 | 0.9812 | 0.0539 ! | -0.201 |
| best | image | kaggle_v2 to visual | YOLO11s | 10 | 13 | 4 | n/a | n/a | 0.9049 | 0.9693 | 0.0645 | n/a |
| best | image | kaggle_v2 to visual | YOLO11s | 25 | 25 | 12 | 0.9676 | 0.0747 | 0.9049 | 0.9693 | 0.0645 | 0.827 |
| best | image | kaggle_v2 to visual | YOLO11s | 50 | 54 | 24 | 0.9677 | 0.0749 | 0.9049 | 0.9693 | 0.0645 | 0.829 |
| best | image | kaggle_v2 to visual | YOLO11s | 100 | 102 | 54 | 0.9673 | 0.0745 | 0.9049 | 0.9693 | 0.0645 | 0.971 |
| best | image | kaggle_v2 to visual | YOLO11s | 250 | 255 | 114 | 0.9844 | 0.0916 | 0.9049 | 0.9693 | 0.0645 | 1.236 |
| best | image | kaggle_v2 to weld_surface_v2 | RT-DETR-l | 10 | 23 | 2 | 0.8873 | 0.0266 | 0.9042 | 0.9868 | 0.0827 | -0.204 |
| best | image | kaggle_v2 to weld_surface_v2 | RT-DETR-l | 25 | 35 | 6 | 0.8908 | 0.0302 | 0.9042 | 0.9868 | 0.0827 | -0.190 |
| best | image | kaggle_v2 to weld_surface_v2 | RT-DETR-l | 50 | 96 | 14 | 0.8765 | 0.0158 | 0.9042 | 0.9868 | 0.0827 | -0.352 |
| best | image | kaggle_v2 to weld_surface_v2 | RT-DETR-l | 100 | 152 | 13 | 0.9045 | 0.0439 | 0.9042 | 0.9868 | 0.0827 | -0.024 |
| best | image | kaggle_v2 to weld_surface_v2 | RT-DETR-l | 250 | 255 | 36 | 0.9395 | 0.0788 | 0.9042 | 0.9868 | 0.0827 | 0.381 |
| best | image | kaggle_v2 to weld_surface_v2 | YOLO11s | 10 | 23 | 2 | 0.9165 | 0.0558 | 0.8799 | 0.9884 | 0.1085 | 0.343 |
| best | image | kaggle_v2 to weld_surface_v2 | YOLO11s | 25 | 35 | 6 | 0.9093 | 0.0486 | 0.8799 | 0.9884 | 0.1085 | 0.291 |
| best | image | kaggle_v2 to weld_surface_v2 | YOLO11s | 50 | 96 | 14 | 0.9207 | 0.0601 | 0.8799 | 0.9884 | 0.1085 | 0.373 |
| best | image | kaggle_v2 to weld_surface_v2 | YOLO11s | 100 | 152 | 13 | 0.9248 | 0.0641 | 0.8799 | 0.9884 | 0.1085 | 0.401 |
| best | image | kaggle_v2 to weld_surface_v2 | YOLO11s | 250 | 255 | 36 | 0.9526 | 0.0920 | 0.8799 | 0.9884 | 0.1085 | 0.669 |
| best | image | lohi_weld to kaggle_v2 | RT-DETR-l | 10 | 38 | 8 | 0.6569 | 0.1462 | 0.5535 | 0.7953 | 0.2418 | 0.485 |
| best | image | lohi_weld to kaggle_v2 | RT-DETR-l | 25 | 56 | 12 | 0.5858 | 0.0752 | 0.5535 | 0.7953 | 0.2418 | 0.202 |
| best | image | lohi_weld to kaggle_v2 | RT-DETR-l | 50 | 111 | 24 | 0.6284 | 0.1178 | 0.5535 | 0.7953 | 0.2418 | 0.309 |
| best | image | lohi_weld to kaggle_v2 | RT-DETR-l | 100 | 114 | 12 | 0.6197 | 0.1090 | 0.5535 | 0.7953 | 0.2418 | 0.274 |
| best | image | lohi_weld to kaggle_v2 | RT-DETR-l | 250 | 277 | 92 | 0.6634 | 0.1528 | 0.5535 | 0.7953 | 0.2418 | 0.459 |
| best | image | lohi_weld to kaggle_v2 | YOLO11s | 10 | 38 | 8 | 0.5723 | 0.0616 | 0.5488 | 0.7964 | 0.2476 | 0.106 |
| best | image | lohi_weld to kaggle_v2 | YOLO11s | 25 | 56 | 12 | 0.5664 | 0.0557 | 0.5488 | 0.7964 | 0.2476 | 0.068 |
| best | image | lohi_weld to kaggle_v2 | YOLO11s | 50 | 111 | 24 | 0.6104 | 0.0997 | 0.5488 | 0.7964 | 0.2476 | 0.253 |
| best | image | lohi_weld to kaggle_v2 | YOLO11s | 100 | 114 | 12 | 0.6272 | 0.1166 | 0.5488 | 0.7964 | 0.2476 | 0.316 |
| best | image | lohi_weld to kaggle_v2 | YOLO11s | 250 | 277 | 92 | 0.6995 | 0.1888 | 0.5488 | 0.7964 | 0.2476 | 0.616 |
| best | image | lohi_weld to twoclass | RT-DETR-l | 10 | 20 | 2 | 0.8480 | 0.0480 | 0.8363 | 0.9862 | 0.1499 | 0.066 |
| best | image | lohi_weld to twoclass | RT-DETR-l | 25 | 47 | 2 | 0.8302 | 0.0302 | 0.8363 | 0.9862 | 0.1499 | -0.073 |
| best | image | lohi_weld to twoclass | RT-DETR-l | 50 | 57 | 5 | 0.8539 | 0.0539 | 0.8363 | 0.9862 | 0.1499 | 0.095 |
| best | image | lohi_weld to twoclass | RT-DETR-l | 100 | 108 | 13 | 0.8802 | 0.0802 | 0.8363 | 0.9862 | 0.1499 | 0.286 |
| best | image | lohi_weld to twoclass | RT-DETR-l | 250 * | 213 | 22 | 0.9148 | 0.1148 | 0.8363 | 0.9862 | 0.1499 | 0.510 |
| best | image | lohi_weld to twoclass | YOLO11s | 10 | 20 | 2 | 0.8715 | 0.0715 | 0.8596 | 0.9855 | 0.1259 | 0.122 |
| best | image | lohi_weld to twoclass | YOLO11s | 25 | 47 | 2 | 0.9003 | 0.1003 | 0.8596 | 0.9855 | 0.1259 | 0.317 |
| best | image | lohi_weld to twoclass | YOLO11s | 50 | 57 | 5 | 0.8886 | 0.0886 | 0.8596 | 0.9855 | 0.1259 | 0.256 |
| best | image | lohi_weld to twoclass | YOLO11s | 100 | 108 | 13 | 0.8856 | 0.0856 | 0.8596 | 0.9855 | 0.1259 | 0.181 |
| best | image | lohi_weld to twoclass | YOLO11s | 250 * | 213 | 22 | 0.9044 | 0.1044 | 0.8596 | 0.9855 | 0.1259 | 0.343 |
| best | image | lohi_weld to visual | RT-DETR-l | 10 | 13 | 4 | 0.9194 | 0.0265 | 0.9139 | 0.9812 | 0.0673 | 0.102 |
| best | image | lohi_weld to visual | RT-DETR-l | 25 | 25 | 12 | 0.9175 | 0.0247 | 0.9139 | 0.9812 | 0.0673 | 0.096 |
| best | image | lohi_weld to visual | RT-DETR-l | 50 | 54 | 24 | 0.9210 | 0.0281 | 0.9139 | 0.9812 | 0.0673 | 0.111 |
| best | image | lohi_weld to visual | RT-DETR-l | 100 | 102 | 54 | 0.9159 | 0.0230 | 0.9139 | 0.9812 | 0.0673 | 0.017 |
| best | image | lohi_weld to visual | RT-DETR-l | 250 | 255 | 114 | 0.9436 | 0.0507 | 0.9139 | 0.9812 | 0.0673 | 0.508 |
| best | image | lohi_weld to visual | YOLO11s | 10 | 13 | 4 | 0.9101 | 0.0172 | 0.9036 | 0.9693 | 0.0657 | 0.105 |
| best | image | lohi_weld to visual | YOLO11s | 25 | 25 | 12 | 0.9168 | 0.0240 | 0.9036 | 0.9693 | 0.0657 | 0.230 |
| best | image | lohi_weld to visual | YOLO11s | 50 | 54 | 24 | 0.9222 | 0.0294 | 0.9036 | 0.9693 | 0.0657 | 0.319 |
| best | image | lohi_weld to visual | YOLO11s | 100 | 102 | 54 | 0.9419 | 0.0491 | 0.9036 | 0.9693 | 0.0657 | 0.595 |
| best | image | lohi_weld to visual | YOLO11s | 250 | 255 | 114 | 0.9646 | 0.0718 | 0.9036 | 0.9693 | 0.0657 | 0.955 |
| best | image | lohi_weld to weld_surface_v2 | RT-DETR-l | 10 | 23 | 2 | 0.9040 | 0.0434 | 0.9182 | 0.9868 | 0.0686 ! | -0.300 |
| best | image | lohi_weld to weld_surface_v2 | RT-DETR-l | 25 | 35 | 6 | 0.9126 | 0.0519 | 0.9182 | 0.9868 | 0.0686 ! | -0.201 |
| best | image | lohi_weld to weld_surface_v2 | RT-DETR-l | 50 | 96 | 14 | 0.9076 | 0.0469 | 0.9182 | 0.9868 | 0.0686 ! | -0.227 |
| best | image | lohi_weld to weld_surface_v2 | RT-DETR-l | 100 | 152 | 13 | 0.9043 | 0.0436 | 0.9182 | 0.9868 | 0.0686 ! | -0.366 |
| best | image | lohi_weld to weld_surface_v2 | RT-DETR-l | 250 | 255 | 36 | 0.9195 | 0.0588 | 0.9182 | 0.9868 | 0.0686 ! | -0.081 |
| best | image | lohi_weld to weld_surface_v2 | YOLO11s | 10 | 23 | 2 | 0.9206 | 0.0599 | 0.9031 | 0.9884 | 0.0853 | 0.214 |
| best | image | lohi_weld to weld_surface_v2 | YOLO11s | 25 | 35 | 6 | 0.9039 | 0.0433 | 0.9031 | 0.9884 | 0.0853 | 0.017 |
| best | image | lohi_weld to weld_surface_v2 | YOLO11s | 50 | 96 | 14 | 0.9162 | 0.0556 | 0.9031 | 0.9884 | 0.0853 | 0.165 |
| best | image | lohi_weld to weld_surface_v2 | YOLO11s | 100 | 152 | 13 | 0.9254 | 0.0648 | 0.9031 | 0.9884 | 0.0853 | 0.258 |
| best | image | lohi_weld to weld_surface_v2 | YOLO11s | 250 | 255 | 36 | 0.9419 | 0.0812 | 0.9031 | 0.9884 | 0.0853 | 0.458 |
| last | box | kaggle_v2 to lohi_weld | RT-DETR-l | 10 | 10 | 7 | 0.0205 | n/a | 0.0110 | 0.6750 | 0.6640 | 0.015 |
| last | box | kaggle_v2 to lohi_weld | RT-DETR-l | 25 | 25 | 16 | 0.0441 | n/a | 0.0110 | 0.6750 | 0.6640 | 0.052 |
| last | box | kaggle_v2 to lohi_weld | RT-DETR-l | 50 | 70 | 31 | 0.1134 | n/a | 0.0110 | 0.6750 | 0.6640 | 0.154 |
| last | box | kaggle_v2 to lohi_weld | RT-DETR-l | 100 | 100 | 69 | 0.2876 | n/a | 0.0110 | 0.6750 | 0.6640 | 0.421 |
| last | box | kaggle_v2 to lohi_weld | RT-DETR-l | 250 | 250 | 170 | 0.4311 | n/a | 0.0110 | 0.6750 | 0.6640 | 0.636 |
| last | box | kaggle_v2 to lohi_weld | YOLO11s | 10 | 10 | 7 | 0.0802 | n/a | 0.0157 | 0.6877 | 0.6721 | 0.097 |
| last | box | kaggle_v2 to lohi_weld | YOLO11s | 25 | 25 | 16 | 0.1411 | n/a | 0.0157 | 0.6877 | 0.6721 | 0.187 |
| last | box | kaggle_v2 to lohi_weld | YOLO11s | 50 | 70 | 31 | 0.2161 | n/a | 0.0157 | 0.6877 | 0.6721 | 0.297 |
| last | box | kaggle_v2 to lohi_weld | YOLO11s | 100 | 100 | 69 | 0.3170 | n/a | 0.0157 | 0.6877 | 0.6721 | 0.448 |
| last | box | kaggle_v2 to lohi_weld | YOLO11s | 250 | 250 | 170 | 0.4402 | n/a | 0.0157 | 0.6877 | 0.6721 | 0.632 |
| last | box | kaggle_v2 to twoclass | RT-DETR-l | 10 | 20 | 2 | 0.0328 | n/a | 0.0268 | 0.5620 | 0.5352 | 0.011 |
| last | box | kaggle_v2 to twoclass | RT-DETR-l | 25 | 47 | 2 | 0.0142 | n/a | 0.0268 | 0.5620 | 0.5352 | -0.022 |
| last | box | kaggle_v2 to twoclass | RT-DETR-l | 50 | 57 | 5 | 0.1647 | n/a | 0.0268 | 0.5620 | 0.5352 | 0.246 |
| last | box | kaggle_v2 to twoclass | RT-DETR-l | 100 | 108 | 13 | 0.3423 | n/a | 0.0268 | 0.5620 | 0.5352 | 0.584 |
| last | box | kaggle_v2 to twoclass | RT-DETR-l | 250 * | 213 | 22 | 0.5885 | n/a | 0.0268 | 0.5620 | 0.5352 | 1.059 |
| last | box | kaggle_v2 to twoclass | YOLO11s | 10 | 20 | 2 | 0.0270 | n/a | 0.0144 | 0.5280 | 0.5136 | 0.028 |
| last | box | kaggle_v2 to twoclass | YOLO11s | 25 | 47 | 2 | 0.0119 | n/a | 0.0144 | 0.5280 | 0.5136 | -0.006 |
| last | box | kaggle_v2 to twoclass | YOLO11s | 50 | 57 | 5 | 0.0956 | n/a | 0.0144 | 0.5280 | 0.5136 | 0.146 |
| last | box | kaggle_v2 to twoclass | YOLO11s | 100 | 108 | 13 | 0.1825 | n/a | 0.0144 | 0.5280 | 0.5136 | 0.317 |
| last | box | kaggle_v2 to twoclass | YOLO11s | 250 * | 213 | 22 | 0.4322 | n/a | 0.0144 | 0.5280 | 0.5136 | 0.831 |
| last | box | kaggle_v2 to visual | RT-DETR-l | 10 | 13 | 4 | 0.0173 | n/a | 0.0093 | 0.1302 | 0.1209 | 0.062 |
| last | box | kaggle_v2 to visual | RT-DETR-l | 25 | 25 | 12 | 0.0143 | n/a | 0.0093 | 0.1302 | 0.1209 | 0.040 |
| last | box | kaggle_v2 to visual | RT-DETR-l | 50 | 54 | 24 | 0.0111 | n/a | 0.0093 | 0.1302 | 0.1209 | 0.012 |
| last | box | kaggle_v2 to visual | RT-DETR-l | 100 | 102 | 54 | 0.0152 | n/a | 0.0093 | 0.1302 | 0.1209 | 0.040 |
| last | box | kaggle_v2 to visual | RT-DETR-l | 250 | 255 | 114 | 0.0225 | n/a | 0.0093 | 0.1302 | 0.1209 | 0.115 |
| last | box | kaggle_v2 to visual | YOLO11s | 10 | 13 | 4 | 0.0000 | n/a | 0.0087 | 0.1417 | 0.1329 | -0.067 |
| last | box | kaggle_v2 to visual | YOLO11s | 25 | 25 | 12 | 0.0008 | n/a | 0.0087 | 0.1417 | 0.1329 | -0.060 |
| last | box | kaggle_v2 to visual | YOLO11s | 50 | 54 | 24 | 0.0030 | n/a | 0.0087 | 0.1417 | 0.1329 | -0.043 |
| last | box | kaggle_v2 to visual | YOLO11s | 100 | 102 | 54 | 0.0435 | n/a | 0.0087 | 0.1417 | 0.1329 | 0.261 |
| last | box | kaggle_v2 to visual | YOLO11s | 250 | 255 | 114 | 0.0929 | n/a | 0.0087 | 0.1417 | 0.1329 | 0.631 |
| last | box | kaggle_v2 to weld_surface_v2 | RT-DETR-l | 10 | 23 | 2 | n/a | n/a | n/a | n/a | n/a | n/a |
| last | box | kaggle_v2 to weld_surface_v2 | RT-DETR-l | 25 | 35 | 6 | n/a | n/a | n/a | n/a | n/a | n/a |
| last | box | kaggle_v2 to weld_surface_v2 | RT-DETR-l | 50 | 96 | 14 | n/a | n/a | n/a | n/a | n/a | n/a |
| last | box | kaggle_v2 to weld_surface_v2 | RT-DETR-l | 100 | 152 | 13 | n/a | n/a | n/a | n/a | n/a | n/a |
| last | box | kaggle_v2 to weld_surface_v2 | RT-DETR-l | 250 | 255 | 36 | n/a | n/a | n/a | n/a | n/a | n/a |
| last | box | kaggle_v2 to weld_surface_v2 | YOLO11s | 10 | 23 | 2 | n/a | n/a | n/a | n/a | n/a | n/a |
| last | box | kaggle_v2 to weld_surface_v2 | YOLO11s | 25 | 35 | 6 | n/a | n/a | n/a | n/a | n/a | n/a |
| last | box | kaggle_v2 to weld_surface_v2 | YOLO11s | 50 | 96 | 14 | n/a | n/a | n/a | n/a | n/a | n/a |
| last | box | kaggle_v2 to weld_surface_v2 | YOLO11s | 100 | 152 | 13 | n/a | n/a | n/a | n/a | n/a | n/a |
| last | box | kaggle_v2 to weld_surface_v2 | YOLO11s | 250 | 255 | 36 | n/a | n/a | n/a | n/a | n/a | n/a |
| last | box | lohi_weld to kaggle_v2 | RT-DETR-l | 10 | 38 | 8 | 0.0003 | n/a | 0.0004 | 0.0873 | 0.0868 | -0.001 |
| last | box | lohi_weld to kaggle_v2 | RT-DETR-l | 25 | 56 | 12 | 0.0011 | n/a | 0.0004 | 0.0873 | 0.0868 | 0.007 |
| last | box | lohi_weld to kaggle_v2 | RT-DETR-l | 50 | 111 | 24 | 0.0018 | n/a | 0.0004 | 0.0873 | 0.0868 | 0.018 |
| last | box | lohi_weld to kaggle_v2 | RT-DETR-l | 100 | 114 | 12 | 0.0020 | n/a | 0.0004 | 0.0873 | 0.0868 | 0.020 |
| last | box | lohi_weld to kaggle_v2 | RT-DETR-l | 250 | 277 | 92 | 0.0269 | n/a | 0.0004 | 0.0873 | 0.0868 | 0.312 |
| last | box | lohi_weld to kaggle_v2 | YOLO11s | 10 | 38 | 8 | 0.0021 | n/a | 0.0012 | 0.1118 | 0.1107 | 0.008 |
| last | box | lohi_weld to kaggle_v2 | YOLO11s | 25 | 56 | 12 | 0.0013 | n/a | 0.0012 | 0.1118 | 0.1107 | -0.000 |
| last | box | lohi_weld to kaggle_v2 | YOLO11s | 50 | 111 | 24 | 0.0043 | n/a | 0.0012 | 0.1118 | 0.1107 | 0.027 |
| last | box | lohi_weld to kaggle_v2 | YOLO11s | 100 | 114 | 12 | 0.0089 | n/a | 0.0012 | 0.1118 | 0.1107 | 0.075 |
| last | box | lohi_weld to kaggle_v2 | YOLO11s | 250 | 277 | 92 | 0.0334 | n/a | 0.0012 | 0.1118 | 0.1107 | 0.290 |
| last | box | lohi_weld to twoclass | RT-DETR-l | 10 | 20 | 2 | 0.0041 | n/a | 0.0026 | 0.5620 | 0.5594 | 0.003 |
| last | box | lohi_weld to twoclass | RT-DETR-l | 25 | 47 | 2 | 0.0115 | n/a | 0.0026 | 0.5620 | 0.5594 | 0.016 |
| last | box | lohi_weld to twoclass | RT-DETR-l | 50 | 57 | 5 | 0.0181 | n/a | 0.0026 | 0.5620 | 0.5594 | 0.026 |
| last | box | lohi_weld to twoclass | RT-DETR-l | 100 | 108 | 13 | 0.1415 | n/a | 0.0026 | 0.5620 | 0.5594 | 0.235 |
| last | box | lohi_weld to twoclass | RT-DETR-l | 250 * | 213 | 22 | 0.4130 | n/a | 0.0026 | 0.5620 | 0.5594 | 0.735 |
| last | box | lohi_weld to twoclass | YOLO11s | 10 | 20 | 2 | 0.0312 | n/a | 0.0258 | 0.5280 | 0.5022 | 0.015 |
| last | box | lohi_weld to twoclass | YOLO11s | 25 | 47 | 2 | 0.0066 | n/a | 0.0258 | 0.5280 | 0.5022 | -0.038 |
| last | box | lohi_weld to twoclass | YOLO11s | 50 | 57 | 5 | 0.0996 | n/a | 0.0258 | 0.5280 | 0.5022 | 0.133 |
| last | box | lohi_weld to twoclass | YOLO11s | 100 | 108 | 13 | 0.1483 | n/a | 0.0258 | 0.5280 | 0.5022 | 0.229 |
| last | box | lohi_weld to twoclass | YOLO11s | 250 * | 213 | 22 | 0.3326 | n/a | 0.0258 | 0.5280 | 0.5022 | 0.613 |
| last | box | lohi_weld to visual | RT-DETR-l | 10 | 13 | 4 | 0.0010 | n/a | 0.0014 | 0.1302 | 0.1288 | -0.003 |
| last | box | lohi_weld to visual | RT-DETR-l | 25 | 25 | 12 | 0.0005 | n/a | 0.0014 | 0.1302 | 0.1288 | -0.007 |
| last | box | lohi_weld to visual | RT-DETR-l | 50 | 54 | 24 | 0.0004 | n/a | 0.0014 | 0.1302 | 0.1288 | -0.007 |
| last | box | lohi_weld to visual | RT-DETR-l | 100 | 102 | 54 | 0.0058 | n/a | 0.0014 | 0.1302 | 0.1288 | 0.030 |
| last | box | lohi_weld to visual | RT-DETR-l | 250 | 255 | 114 | 0.0301 | n/a | 0.0014 | 0.1302 | 0.1288 | 0.223 |
| last | box | lohi_weld to visual | YOLO11s | 10 | 13 | 4 | 0.0011 | n/a | 0.0006 | 0.1417 | 0.1411 | 0.003 |
| last | box | lohi_weld to visual | YOLO11s | 25 | 25 | 12 | 0.0023 | n/a | 0.0006 | 0.1417 | 0.1411 | 0.012 |
| last | box | lohi_weld to visual | YOLO11s | 50 | 54 | 24 | 0.0106 | n/a | 0.0006 | 0.1417 | 0.1411 | 0.070 |
| last | box | lohi_weld to visual | YOLO11s | 100 | 102 | 54 | 0.0131 | n/a | 0.0006 | 0.1417 | 0.1411 | 0.089 |
| last | box | lohi_weld to visual | YOLO11s | 250 | 255 | 114 | 0.0438 | n/a | 0.0006 | 0.1417 | 0.1411 | 0.306 |
| last | box | lohi_weld to weld_surface_v2 | RT-DETR-l | 10 | 23 | 2 | n/a | n/a | n/a | n/a | n/a | n/a |
| last | box | lohi_weld to weld_surface_v2 | RT-DETR-l | 25 | 35 | 6 | n/a | n/a | n/a | n/a | n/a | n/a |
| last | box | lohi_weld to weld_surface_v2 | RT-DETR-l | 50 | 96 | 14 | n/a | n/a | n/a | n/a | n/a | n/a |
| last | box | lohi_weld to weld_surface_v2 | RT-DETR-l | 100 | 152 | 13 | n/a | n/a | n/a | n/a | n/a | n/a |
| last | box | lohi_weld to weld_surface_v2 | RT-DETR-l | 250 | 255 | 36 | n/a | n/a | n/a | n/a | n/a | n/a |
| last | box | lohi_weld to weld_surface_v2 | YOLO11s | 10 | 23 | 2 | n/a | n/a | n/a | n/a | n/a | n/a |
| last | box | lohi_weld to weld_surface_v2 | YOLO11s | 25 | 35 | 6 | n/a | n/a | n/a | n/a | n/a | n/a |
| last | box | lohi_weld to weld_surface_v2 | YOLO11s | 50 | 96 | 14 | n/a | n/a | n/a | n/a | n/a | n/a |
| last | box | lohi_weld to weld_surface_v2 | YOLO11s | 100 | 152 | 13 | n/a | n/a | n/a | n/a | n/a | n/a |
| last | box | lohi_weld to weld_surface_v2 | YOLO11s | 250 | 255 | 36 | n/a | n/a | n/a | n/a | n/a | n/a |
| last | image | kaggle_v2 to lohi_weld | RT-DETR-l | 10 | 10 | 7 | 0.9905 | 0.0148 | 0.9884 | 0.9971 | 0.0087 ! | 0.167 |
| last | image | kaggle_v2 to lohi_weld | RT-DETR-l | 25 | 25 | 16 | 0.9915 | 0.0158 | 0.9884 | 0.9971 | 0.0087 ! | 0.284 |
| last | image | kaggle_v2 to lohi_weld | RT-DETR-l | 50 | 70 | 31 | 0.9925 | 0.0167 | 0.9884 | 0.9971 | 0.0087 ! | 0.282 |
| last | image | kaggle_v2 to lohi_weld | RT-DETR-l | 100 | 100 | 69 | 0.9962 | 0.0205 | 0.9884 | 0.9971 | 0.0087 ! | 0.970 |
| last | image | kaggle_v2 to lohi_weld | RT-DETR-l | 250 | 250 | 170 | 0.9943 | 0.0186 | 0.9884 | 0.9971 | 0.0087 ! | 0.751 |
| last | image | kaggle_v2 to lohi_weld | YOLO11s | 10 | 10 | 7 | 0.9899 | 0.0142 | 0.9960 | 0.9994 | 0.0033 ! | -2.059 |
| last | image | kaggle_v2 to lohi_weld | YOLO11s | 25 | 25 | 16 | 0.9909 | 0.0151 | 0.9960 | 0.9994 | 0.0033 ! | -1.717 |
| last | image | kaggle_v2 to lohi_weld | YOLO11s | 50 | 70 | 31 | 0.9938 | 0.0181 | 0.9960 | 0.9994 | 0.0033 ! | -0.665 |
| last | image | kaggle_v2 to lohi_weld | YOLO11s | 100 | 100 | 69 | 0.9954 | 0.0197 | 0.9960 | 0.9994 | 0.0033 ! | -0.230 |
| last | image | kaggle_v2 to lohi_weld | YOLO11s | 250 | 250 | 170 | 0.9879 | 0.0121 | 0.9960 | 0.9994 | 0.0033 ! | -2.540 |
| last | image | kaggle_v2 to twoclass | RT-DETR-l | 10 | 20 | 2 | 0.8495 | 0.0495 | 0.8358 | 0.9862 | 0.1504 | 0.100 |
| last | image | kaggle_v2 to twoclass | RT-DETR-l | 25 | 47 | 2 | 0.8543 | 0.0543 | 0.8358 | 0.9862 | 0.1504 | 0.119 |
| last | image | kaggle_v2 to twoclass | RT-DETR-l | 50 | 57 | 5 | 0.8378 | 0.0378 | 0.8358 | 0.9862 | 0.1504 | -0.032 |
| last | image | kaggle_v2 to twoclass | RT-DETR-l | 100 | 108 | 13 | 0.8767 | 0.0767 | 0.8358 | 0.9862 | 0.1504 | 0.281 |
| last | image | kaggle_v2 to twoclass | RT-DETR-l | 250 * | 213 | 22 | 0.9305 | 0.1305 | 0.8358 | 0.9862 | 0.1504 | 0.624 |
| last | image | kaggle_v2 to twoclass | YOLO11s | 10 | 20 | 2 | 0.8895 | 0.0895 | 0.8536 | 0.9855 | 0.1320 | 0.252 |
| last | image | kaggle_v2 to twoclass | YOLO11s | 25 | 47 | 2 | 0.9295 | 0.1295 | 0.8536 | 0.9855 | 0.1320 | 0.566 |
| last | image | kaggle_v2 to twoclass | YOLO11s | 50 | 57 | 5 | 0.9254 | 0.1254 | 0.8536 | 0.9855 | 0.1320 | 0.532 |
| last | image | kaggle_v2 to twoclass | YOLO11s | 100 | 108 | 13 | 0.9487 | 0.1487 | 0.8536 | 0.9855 | 0.1320 | 0.722 |
| last | image | kaggle_v2 to twoclass | YOLO11s | 250 * | 213 | 22 | 0.9315 | 0.1315 | 0.8536 | 0.9855 | 0.1320 | 0.596 |
| last | image | kaggle_v2 to visual | RT-DETR-l | 10 | 13 | 4 | 0.9340 | 0.0411 | 0.9272 | 0.9812 | 0.0539 ! | 0.111 |
| last | image | kaggle_v2 to visual | RT-DETR-l | 25 | 25 | 12 | 0.9347 | 0.0419 | 0.9272 | 0.9812 | 0.0539 ! | 0.105 |
| last | image | kaggle_v2 to visual | RT-DETR-l | 50 | 54 | 24 | 0.9095 | 0.0167 | 0.9272 | 0.9812 | 0.0539 ! | -0.342 |
| last | image | kaggle_v2 to visual | RT-DETR-l | 100 | 102 | 54 | 0.9351 | 0.0423 | 0.9272 | 0.9812 | 0.0539 ! | 0.100 |
| last | image | kaggle_v2 to visual | RT-DETR-l | 250 | 255 | 114 | 0.9159 | 0.0231 | 0.9272 | 0.9812 | 0.0539 ! | -0.237 |
| last | image | kaggle_v2 to visual | YOLO11s | 10 | 13 | 4 | 0.9663 | 0.0735 | 0.9049 | 0.9693 | 0.0645 | 0.807 |
| last | image | kaggle_v2 to visual | YOLO11s | 25 | 25 | 12 | 0.9549 | 0.0620 | 0.9049 | 0.9693 | 0.0645 | 0.628 |
| last | image | kaggle_v2 to visual | YOLO11s | 50 | 54 | 24 | 0.9502 | 0.0573 | 0.9049 | 0.9693 | 0.0645 | 0.555 |
| last | image | kaggle_v2 to visual | YOLO11s | 100 | 102 | 54 | 0.9590 | 0.0662 | 0.9049 | 0.9693 | 0.0645 | 0.839 |
| last | image | kaggle_v2 to visual | YOLO11s | 250 | 255 | 114 | 0.9728 | 0.0800 | 0.9049 | 0.9693 | 0.0645 | 1.054 |
| last | image | kaggle_v2 to weld_surface_v2 | RT-DETR-l | 10 | 23 | 2 | 0.8862 | 0.0255 | 0.9042 | 0.9868 | 0.0827 | -0.216 |
| last | image | kaggle_v2 to weld_surface_v2 | RT-DETR-l | 25 | 35 | 6 | 0.8837 | 0.0230 | 0.9042 | 0.9868 | 0.0827 | -0.298 |
| last | image | kaggle_v2 to weld_surface_v2 | RT-DETR-l | 50 | 96 | 14 | 0.8784 | 0.0177 | 0.9042 | 0.9868 | 0.0827 | -0.327 |
| last | image | kaggle_v2 to weld_surface_v2 | RT-DETR-l | 100 | 152 | 13 | 0.9083 | 0.0477 | 0.9042 | 0.9868 | 0.0827 | 0.017 |
| last | image | kaggle_v2 to weld_surface_v2 | RT-DETR-l | 250 | 255 | 36 | 0.9375 | 0.0768 | 0.9042 | 0.9868 | 0.0827 | 0.359 |
| last | image | kaggle_v2 to weld_surface_v2 | YOLO11s | 10 | 23 | 2 | 0.9214 | 0.0607 | 0.8799 | 0.9884 | 0.1085 | 0.381 |
| last | image | kaggle_v2 to weld_surface_v2 | YOLO11s | 25 | 35 | 6 | 0.9054 | 0.0447 | 0.8799 | 0.9884 | 0.1085 | 0.252 |
| last | image | kaggle_v2 to weld_surface_v2 | YOLO11s | 50 | 96 | 14 | 0.9111 | 0.0504 | 0.8799 | 0.9884 | 0.1085 | 0.282 |
| last | image | kaggle_v2 to weld_surface_v2 | YOLO11s | 100 | 152 | 13 | 0.9258 | 0.0652 | 0.8799 | 0.9884 | 0.1085 | 0.406 |
| last | image | kaggle_v2 to weld_surface_v2 | YOLO11s | 250 | 255 | 36 | 0.9423 | 0.0817 | 0.8799 | 0.9884 | 0.1085 | 0.584 |
| last | image | lohi_weld to kaggle_v2 | RT-DETR-l | 10 | 38 | 8 | 0.6569 | 0.1462 | 0.5535 | 0.7953 | 0.2418 | 0.485 |
| last | image | lohi_weld to kaggle_v2 | RT-DETR-l | 25 | 56 | 12 | 0.5923 | 0.0816 | 0.5535 | 0.7953 | 0.2418 | 0.229 |
| last | image | lohi_weld to kaggle_v2 | RT-DETR-l | 50 | 111 | 24 | 0.6375 | 0.1269 | 0.5535 | 0.7953 | 0.2418 | 0.348 |
| last | image | lohi_weld to kaggle_v2 | RT-DETR-l | 100 | 114 | 12 | 0.6239 | 0.1132 | 0.5535 | 0.7953 | 0.2418 | 0.289 |
| last | image | lohi_weld to kaggle_v2 | RT-DETR-l | 250 | 277 | 92 | 0.6536 | 0.1429 | 0.5535 | 0.7953 | 0.2418 | 0.423 |
| last | image | lohi_weld to kaggle_v2 | YOLO11s | 10 | 38 | 8 | 0.5694 | 0.0588 | 0.5488 | 0.7964 | 0.2476 | 0.095 |
| last | image | lohi_weld to kaggle_v2 | YOLO11s | 25 | 56 | 12 | 0.5739 | 0.0633 | 0.5488 | 0.7964 | 0.2476 | 0.100 |
| last | image | lohi_weld to kaggle_v2 | YOLO11s | 50 | 111 | 24 | 0.6120 | 0.1014 | 0.5488 | 0.7964 | 0.2476 | 0.260 |
| last | image | lohi_weld to kaggle_v2 | YOLO11s | 100 | 114 | 12 | 0.6345 | 0.1238 | 0.5488 | 0.7964 | 0.2476 | 0.346 |
| last | image | lohi_weld to kaggle_v2 | YOLO11s | 250 | 277 | 92 | 0.6988 | 0.1882 | 0.5488 | 0.7964 | 0.2476 | 0.612 |
| last | image | lohi_weld to twoclass | RT-DETR-l | 10 | 20 | 2 | 0.8644 | 0.0644 | 0.8363 | 0.9862 | 0.1499 | 0.159 |
| last | image | lohi_weld to twoclass | RT-DETR-l | 25 | 47 | 2 | 0.8393 | 0.0393 | 0.8363 | 0.9862 | 0.1499 | -0.031 |
| last | image | lohi_weld to twoclass | RT-DETR-l | 50 | 57 | 5 | 0.8200 | 0.0200 | 0.8363 | 0.9862 | 0.1499 | -0.117 |
| last | image | lohi_weld to twoclass | RT-DETR-l | 100 | 108 | 13 | 0.8842 | 0.0842 | 0.8363 | 0.9862 | 0.1499 | 0.302 |
| last | image | lohi_weld to twoclass | RT-DETR-l | 250 * | 213 | 22 | 0.9179 | 0.1179 | 0.8363 | 0.9862 | 0.1499 | 0.532 |
| last | image | lohi_weld to twoclass | YOLO11s | 10 | 20 | 2 | 0.8565 | 0.0565 | 0.8596 | 0.9855 | 0.1259 | -0.011 |
| last | image | lohi_weld to twoclass | YOLO11s | 25 | 47 | 2 | 0.8935 | 0.0935 | 0.8596 | 0.9855 | 0.1259 | 0.215 |
| last | image | lohi_weld to twoclass | YOLO11s | 50 | 57 | 5 | 0.8918 | 0.0918 | 0.8596 | 0.9855 | 0.1259 | 0.278 |
| last | image | lohi_weld to twoclass | YOLO11s | 100 | 108 | 13 | 0.8926 | 0.0926 | 0.8596 | 0.9855 | 0.1259 | 0.251 |
| last | image | lohi_weld to twoclass | YOLO11s | 250 * | 213 | 22 | 0.9104 | 0.1104 | 0.8596 | 0.9855 | 0.1259 | 0.379 |
| last | image | lohi_weld to visual | RT-DETR-l | 10 | 13 | 4 | 0.9187 | 0.0258 | 0.9139 | 0.9812 | 0.0673 | 0.094 |
| last | image | lohi_weld to visual | RT-DETR-l | 25 | 25 | 12 | 0.9161 | 0.0233 | 0.9139 | 0.9812 | 0.0673 | 0.046 |
| last | image | lohi_weld to visual | RT-DETR-l | 50 | 54 | 24 | 0.9189 | 0.0261 | 0.9139 | 0.9812 | 0.0673 | 0.090 |
| last | image | lohi_weld to visual | RT-DETR-l | 100 | 102 | 54 | 0.9177 | 0.0248 | 0.9139 | 0.9812 | 0.0673 | 0.033 |
| last | image | lohi_weld to visual | RT-DETR-l | 250 | 255 | 114 | 0.9346 | 0.0418 | 0.9139 | 0.9812 | 0.0673 | 0.336 |
| last | image | lohi_weld to visual | YOLO11s | 10 | 13 | 4 | 0.9127 | 0.0199 | 0.9036 | 0.9693 | 0.0657 | 0.143 |
| last | image | lohi_weld to visual | YOLO11s | 25 | 25 | 12 | 0.9209 | 0.0280 | 0.9036 | 0.9693 | 0.0657 | 0.301 |
| last | image | lohi_weld to visual | YOLO11s | 50 | 54 | 24 | 0.9219 | 0.0290 | 0.9036 | 0.9693 | 0.0657 | 0.311 |
| last | image | lohi_weld to visual | YOLO11s | 100 | 102 | 54 | 0.9393 | 0.0465 | 0.9036 | 0.9693 | 0.0657 | 0.557 |
| last | image | lohi_weld to visual | YOLO11s | 250 | 255 | 114 | 0.9661 | 0.0732 | 0.9036 | 0.9693 | 0.0657 | 0.979 |
| last | image | lohi_weld to weld_surface_v2 | RT-DETR-l | 10 | 23 | 2 | 0.9040 | 0.0434 | 0.9182 | 0.9868 | 0.0686 ! | -0.300 |
| last | image | lohi_weld to weld_surface_v2 | RT-DETR-l | 25 | 35 | 6 | 0.9103 | 0.0496 | 0.9182 | 0.9868 | 0.0686 ! | -0.225 |
| last | image | lohi_weld to weld_surface_v2 | RT-DETR-l | 50 | 96 | 14 | 0.9075 | 0.0468 | 0.9182 | 0.9868 | 0.0686 ! | -0.227 |
| last | image | lohi_weld to weld_surface_v2 | RT-DETR-l | 100 | 152 | 13 | 0.9061 | 0.0455 | 0.9182 | 0.9868 | 0.0686 ! | -0.345 |
| last | image | lohi_weld to weld_surface_v2 | RT-DETR-l | 250 | 255 | 36 | 0.9157 | 0.0550 | 0.9182 | 0.9868 | 0.0686 ! | -0.137 |
| last | image | lohi_weld to weld_surface_v2 | YOLO11s | 10 | 23 | 2 | 0.9109 | 0.0502 | 0.9031 | 0.9884 | 0.0853 | 0.098 |
| last | image | lohi_weld to weld_surface_v2 | YOLO11s | 25 | 35 | 6 | 0.9070 | 0.0463 | 0.9031 | 0.9884 | 0.0853 | 0.056 |
| last | image | lohi_weld to weld_surface_v2 | YOLO11s | 50 | 96 | 14 | 0.9148 | 0.0541 | 0.9031 | 0.9884 | 0.0853 | 0.144 |
| last | image | lohi_weld to weld_surface_v2 | YOLO11s | 100 | 152 | 13 | 0.9219 | 0.0612 | 0.9031 | 0.9884 | 0.0853 | 0.218 |
| last | image | lohi_weld to weld_surface_v2 | YOLO11s | 250 | 255 | 36 | 0.9395 | 0.0788 | 0.9031 | 0.9884 | 0.0853 | 0.430 |

`*` marks a k where at least one draw exhausted the target's train split, so the realised k is smaller than requested. twoclass cannot reach k = 250 at all, its train split holds 213 images. The project log 6a-i.

`AP - base` is blank for box level ON PURPOSE. Positive class image AP is floored by prevalence, so subtracting the base rate is the right correction there. Box mAP50 has no prevalence floor, so the same subtraction would not be a quantity.

`!` marks a ratio whose denominator `P_jj - P_ij(0)` is below 0.05, which is 40 of 320 rows. The threshold was fixed before the ratios were read and no flagged row is dropped. A thin denominator amplifies noise, and the project log 5n establishes that kaggle_v2's box diagonal of about 0.080 is dataset difficulty rather than a defect, so a small absolute gain into kaggle_v2 produces a large looking R. Where the denominator is flagged, let the raw AP and the AP minus base rate carry the claim rather than the ratio.

**Panel counts, so no reader is left counting.** Box level curves exist for SIX pairs: kaggle_v2 and lohi_weld each into lohi_weld or kaggle_v2, visual and twoclass. Every pair into weld_surface_v2 is box undefined, because weld_surface_v2 is whole region and a box number across protocols measures annotation mismatch. Image level curves exist for all EIGHT pairs.

**Negative recovery is not clipped and k is not assumed monotonic.** Fine-tuning on a handful of target images can leave a model worse than the un-adapted source checkpoint. Where it does, the point is plotted below zero, because that is the result.

**Image level recovery into lohi_weld is unstable and must not be quoted.** R divides by `P_jj - P_ij(0)`, and for lohi_weld that denominator is about 0.003, because the target is 97.6 percent positive so both terms sit just under 1.0. Dividing by it turns ordinary noise into ratios in the single and double digits. Read the box level recovery for lohi_weld. The project log 5k makes the same point for the main grid.

## The twoclass reference, printed beside every ratio into twoclass

twoclass has 213 training images, 23 scene blocks and 7 joint configurations. A small denominator inflates a ratio, so a high R into twoclass may say more about the reference being bad than about transfer being good. Never quote a twoclass ratio without this table.

| configuration | seed | twoclass P_jj box | P_jj image |
|---|---|---|---|
| YOLO11s | 0 | 0.4791 | 0.9790 |
| YOLO11s | 1 | 0.6007 | 0.9903 |
| YOLO11s | 2 | 0.5043 | 0.9873 |
| RT-DETR-l | 0 | 0.5152 | 0.9891 |
| RT-DETR-l | 1 | 0.6061 | 0.9866 |
| RT-DETR-l | 2 | 0.5647 | 0.9829 |

## `best` against `last`, the project log 6a-ii

| pair | cfg | max |best - last| box | mean |d| box |
|---|---|---|---|
| kaggle_v2 to lohi_weld | YOLO11s | 0.0884 | 0.0404 |
| kaggle_v2 to lohi_weld | RT-DETR-l | 0.0243 | 0.0065 |
| kaggle_v2 to twoclass | YOLO11s | 0.0401 | 0.0136 |
| kaggle_v2 to twoclass | RT-DETR-l | 0.1732 | 0.0310 |
| kaggle_v2 to visual | YOLO11s | 0.0124 | 0.0034 |
| kaggle_v2 to visual | RT-DETR-l | 0.0100 | 0.0028 |
| lohi_weld to kaggle_v2 | YOLO11s | 0.0031 | 0.0008 |
| lohi_weld to kaggle_v2 | RT-DETR-l | 0.0013 | 0.0004 |
| lohi_weld to twoclass | YOLO11s | 0.0378 | 0.0120 |
| lohi_weld to twoclass | RT-DETR-l | 0.0701 | 0.0117 |
| lohi_weld to visual | YOLO11s | 0.0106 | 0.0014 |
| lohi_weld to visual | RT-DETR-l | 0.0010 | 0.0003 |

## Are k = 10 and k = 25 informative at all? Box level, primary endpoint

| k | mean scenes | box mAP50 | cells | cells retaining nothing |
|---|---|---|---|---|
| 10 | 5 | 0.0181 | 36 | 3 |
| 25 | 10 | 0.0208 | 36 | 3 |
| 50 | 19 | 0.0616 | 36 | 2 |
| 100 | 36 | 0.1256 | 36 | 0 |
| 250 | 89 | 0.2406 | 36 | 0 |

**Answer: at box level, no.** The two smallest budgets are indistinguishable from zero against k = 250. The project log 6a-iv explains why. The detection head is reinitialised whenever the source and target class counts differ, so only the backbone and neck carry over, and the smallest budgets measure a head fitted from scratch on ten images. The project log 6a-v records that such a head can collapse onto a single wrong class entirely.

**Plot them anyway, and say why in the caption.** Dropping them would hide the most useful fact the curve carries, which is that a public checkpoint adapted with ten target images is worthless at box level. Presenting them as a smooth start to a recovery curve would be the real error, because the first points partly measure a different model rather than a less adapted one.

Largest absolute `best` minus `last` box mAP50 difference anywhere in arm B: **0.1732**.

**The 15 September review's incidental claim does NOT survive the class space correction, and this is worth stating plainly.** It reported that on kaggle_v2 to lohi_weld with YOLO11s the two endpoints differ by at most 0.004 box mAP50 at any k, with means by k agreeing to 0.002. Measured on the CORRECTED records that pair is the largest disagreement in the whole arm, 0.0884 at k = 100 draw 0, and the means by k differ by 0.0793 at k = 100. The review's spot check was made on the uncorrected numbers, where the filter retained only `discontinuity` and both endpoints were squeezed toward zero.

**The broader conclusion still holds, and should be reported as the broader conclusion rather than as the pair.** Across all 180 box defined comparisons, which is the 240 cells restricted to the six localisation pairs, the mean absolute `best` minus `last` difference is 0.0103. So the endpoint amendment in the project log 6a-ii was worth making for the integrity of the label cost claim, and on almost every cell it does not move the number. It does move the single best transferring cell in the arm, which is exactly the cell a reader would quote.

## 3j. The headline sentence this arm supports

| k | mean realised scenes | mean realised images | mean R(box) | cells |
|---|---|---|---|---|
| 10 | 5 | 19 | 0.014 | 36 |
| 25 | 10 | 38 | 0.015 | 36 |
| 50 | 19 | 68 | 0.090 | 36 |
| 100 | 36 | 106 | 0.229 | 36 |
| 250 | 89 | 244 | 0.532 | 36 |

**The thin denominator exclusion removed 0 rows at box level**, because every box level denominator here runs from 0.087 to 0.672 and none falls under 0.05. The rule was fixed at 3d before the ratios were read and is stated whether or not it bites.

**THE SENTENCE THE NUMBERS SUPPORT IS A NEGATIVE ONE, AND IT IS THE MORE USEFUL RESULT.** Write it as the finding, not as a disappointment.

At the largest budget tested, about **89 annotated target scenes**, which is about 244 images, k-shot fine tuning recovers **53.2 percent** of the in domain box level gap, averaged over the six localisation pairs and two configurations. That is indistinguishable from nothing.

**Below that budget it is worse than not fine tuning at all.** Mean recovery is NEGATIVE at 0 of the 5 budgets tested. Pooled box mAP50 rises with k, from 0.0181 at k=10, 0.0208 at k=25, 0.0616 at k=50, 0.1256 at k=100, 0.2406 at k=250, but the un-adapted source checkpoint already scores 0.0098 on the same targets. Fine tuning first beats doing nothing only at k = 10.

**One pair is a real exception and must be named rather than averaged away.** `kaggle_v2 to twoclass` with RT-DETR-l reaches R = **1.059** at the largest budget, roughly 106 percent of its gap. It is also the pair with the largest `best` against `last` disagreement, so it is the one cell in the arm where the endpoint choice matters. Quote it as the best case, never as the typical one.

**Say it in scenes, not in k.** The project log 6a-i: a split block is indivisible, so the realised draw overshoots the requested k by up to 9.1 times, and the scene count is the honest label cost. Quoting the requested k would understate the annotation a practitioner actually has to pay for.

**And state the caveat in the same breath.** The detection head is reinitialised at every k, the project log 6a-iv, so this is the budget needed to fit a NEW head on a source pretrained backbone. That is what a practitioner adopting a public checkpoint faces, so it is the useful quantity, but it is not the same thing as smooth domain adaptation from a working detector. It also explains the negative region: a head fitted on ten images is worse than the source head it replaced.

**What this does to C4.** The label cost curve is still the most citable object in the paper, and it now carries a sharper claim than the one originally scoped. The finding is not that adaptation is cheap. It is that within a budget a practitioner would plausibly pay, up to roughly 250 labelled target images, adapting a public visible light weld checkpoint buys essentially nothing at box level, and below about 100 images it actively hurts. The figure draws the zero shot level as the dotted line at R = 0 and the in domain level as the dashed line at R = 1, so the crossing point is visible rather than inferred.
