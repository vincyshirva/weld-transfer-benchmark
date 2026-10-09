# Calibration under shift, arms

| arm | shift | cfg | n cells | ECE | Brier |
|---|---|---|---|---|---|
| armb | cross domain | RT-DETR-l | 120 | 0.4257 (0.127) | 0.3272 |
| armb | cross domain | YOLO11s | 120 | 0.6393 (0.146) | 0.5616 |
| armd | cross domain | Faster R-CNN | 27 | 0.5084 (0.176) | 0.4599 |
| armd | cross domain | RT-DETR-l | 27 | 0.5196 (0.151) | 0.4382 |
| armd | cross domain | YOLO11n | 27 | 0.6549 (0.173) | 0.6028 |
| armd | cross domain | YOLO11s | 27 | 0.6668 (0.175) | 0.6217 |
| lohictl | cross domain | Faster R-CNN | 6 | 0.1052 (0.086) | 0.0474 |
| lohictl | cross domain | RT-DETR-l | 6 | 0.3023 (0.040) | 0.1165 |
| lohictl | cross domain | YOLO11n | 6 | 0.4594 (0.107) | 0.2587 |
| lohictl | cross domain | YOLO11s | 6 | 0.4652 (0.071) | 0.2566 |
| lohictl | in domain | Faster R-CNN | 6 | 0.0229 (0.015) | 0.0138 |
| lohictl | in domain | RT-DETR-l | 6 | 0.1951 (0.046) | 0.0541 |
| lohictl | in domain | YOLO11n | 6 | 0.2865 (0.055) | 0.1051 |
| lohictl | in domain | YOLO11s | 6 | 0.2885 (0.072) | 0.1064 |

**CALIBRATION DEGRADES UNDER SHIFT, and the control is the only place that shows it cleanly.** Inside the LoHi control the annotation protocol, annotators, class list and production line are all held constant, so the only thing that changes is the acquisition. Mean ECE rises from **0.198** in domain to **0.333** cross acquisition, a factor of about 1.7. Every configuration moves the same way.

**THE ERROR RUNS TOWARDS UNDERCONFIDENCE, NOT OVERCONFIDENCE.** An evaluation is underconfident when, weighted by image count over its bins, the share of images with a defect exceeds the score. That holds for **48 of 48** control evaluations, **240 of 240** fine tuned arm B models and **105 of 108** pooling evaluations. The pooling exceptions are Faster R-CNN from pool_no_kaggle_v2 into kaggle_v2, whose test split is 51 percent defective, at seeds 0, 1 and 2. Inside the control, under the shift, the scores fall while the share of images with a defect stays near one. The detectors do not become confidently wrong. They become unsure about images that almost all carry a defect. That is still the costly direction for inspection, because a threshold carried over from the source domain then misses defects.

**Arm B's fine tuned models are the worst calibrated in the study**, mean ECE 0.532 over 240 cells. That is consistent with the project log 6a-iv and 6a-v. A reinitialised head fitted on a handful of images scores the target's defective images low, and at the smallest budgets it can collapse onto one wrong class entirely. A practitioner cannot threshold such a model safely, which matters more for inspection than the mAP does.

**ECE here is computed on the IMAGE level score, so read it against the target's base rate.** The project log 5k. On a target where nearly every image carries a defect, calibration error mostly measures how far the scores sit below that share. Compare the cross domain rows against the in domain rows of the SAME arm, never against an absolute standard.

Per bin counts, mean confidence and observed positive rate are in `reliability_bins_arms.csv`. Figure 4.4 of the manuscript, `reliability_control.png`, is drawn from it by `reliability_control_figure.py`, which this script calls.
