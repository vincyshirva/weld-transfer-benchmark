# Image-level test splits, counted in split blocks

The split block is the unit of independence in this study. "Effective" is the squared sum of the per-block counts over the sum of their squares, so a class spread evenly over n blocks counts n and a class in one block counts 1. The evaluator's rule of 30 negative images counts images and does not see this.

The internal control's two test splits. lohi_high's has no negative image, so AUROC is undefined on it and no score can be overconfident there. Every other split the arms score is a main-grid test split, in grid_results/analysis/tables/image_level_samples.md.

| Test split | Images, blocks | Positive images, in blocks, effective | Negative images, in blocks, effective | Largest block | Constant score, AP above the positive rate |
|---|---|---|---|---|---|
| lohi_high | 152, 152 | 152, in 152, effective 152.0 | 0, in 0, effective 0.0 | 1 | n/a |
| lohi_low | 301, 149 | 290, in 149, effective 142.1 | 11, in 11, effective 11.0 | 6 | +0.020 |

The last column is what binary_metrics gives a score that holds one value for every image. Every image is tied, so the value is set by tie order alone, and on kaggle_v2 archive order is filename order, which carries the label. AUROC averages ties and gives such a score exactly 0.5.
