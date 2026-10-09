# The leakage audit in AUROC

A is the downloaded-partition model on its shipped test split, B the same weights on the audited test split, C the audited model on the audited test split, YOLO11s, mean of three seeds. A and C are scored on test splits with different positive rates. Average precision cannot be compared across that, because its floor moves with the positive rate, and AUROC can. The AP columns are those of downloaded_vs_audited.csv, recomputed here and identical.

| Source | A AUROC | B AUROC | C AUROC | A minus C, AUROC | A AP | B AP | C AP | A minus C, AP | Shipped test, positive and negative | Audited test, positive and negative | A, pairs out of order, worst seed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kaggle_v2 | 0.982 | 0.984 | 0.788 | +0.194 | 0.970 | 0.985 | 0.796 | +0.174 | 43 and 83 | 72 and 69 | 71 of 3,569 |
| lohi_weld | 0.919 | 0.975 | 0.979 | -0.061 | 0.998 | 0.999 | 0.999 | -0.001 | 593 and 12 | 442 and 11 | 781 of 7,116 |
| weld_surface_v2 | 1.000 | 1.000 | 0.940 | +0.060 | 1.000 | 1.000 | 0.988 | +0.012 | 622 and 93 | 383 and 62 | 2 of 57,846 |

Per seed, with each evaluation's own interval, in image_level_audit.csv. A's intervals treat images as independent, because a shipped split carries no blocks.
