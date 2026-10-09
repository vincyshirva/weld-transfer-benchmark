# Image-level test splits, counted in split blocks

The split block is the unit of independence in this study. "Effective" is the squared sum of the per-block counts over the sum of their squares, so a class spread evenly over n blocks counts n and a class in one block counts 1. The evaluator's rule of 30 negative images counts images and does not see this.

Every test split the main grid scores, as the split blocks see it. The internal control's two splits are in grid_results_arms/analysis/image_level_samples_arms.md.

| Test split | Images, blocks | Positive images, in blocks, effective | Negative images, in blocks, effective | Largest block | Constant score, AP above the positive rate |
|---|---|---|---|---|---|
| kaggle_v2, shipped test split | 126, 126 | 43, in 43, effective 43.0 | 83, in 83, effective 83.0 | 1 | +0.312 |
| kaggle_v2 | 141, 74 | 72, in 24, effective 4.6 | 69, in 53, effective 22.4 | 43 | +0.184 |
| lohi_weld, shipped test split | 605, 605 | 593, in 593, effective 593.0 | 12, in 12, effective 12.0 | 1 | +0.004 |
| lohi_weld | 453, 301 | 442, in 301, effective 262.6 | 11, in 11, effective 11.0 | 6 | +0.006 |
| weld_surface_v2, shipped test split | 715, 715 | 622, in 622, effective 622.0 | 93, in 93, effective 93.0 | 1 | +0.066 |
| weld_surface_v2 | 445, 88 | 383, in 79, effective 17.1 | 62, in 9, effective 4.6 | 58 | +0.070 |
| visual | 84, 46 | 75, in 40, effective 20.9 | 9, in 8, effective 7.4 | 10 | +0.073 |
| twoclass | 45, 6 | 36, in 5, effective 3.1 | 9, in 2, effective 1.8 | 15 | +0.107 |

The last column is what binary_metrics gives a score that holds one value for every image. Every image is tied, so the value is set by tie order alone, and on kaggle_v2 archive order is filename order, which carries the label. AUROC averages ties and gives such a score exactly 0.5.

A shipped test split carries no split blocks. The raw archive gives every image its own block, so its counts here are image counts and its intervals treat images as independent. Adjacent video frames on those splits are not independent, the project log audit finding 2.
