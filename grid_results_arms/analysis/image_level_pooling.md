# Pooling, arm D, at image level

Each pooled model against its two members, the main-grid models trained on each of the pool's sources alone, at the same configuration and seed, on the same test images. The better member is chosen per cell on the target's own test split, so it is an optimistic comparator. A pool also trains on more images than either member, so the training budget is not held fixed.

| | In AUROC | In image-level AP |
|---|---|---|
| Mean over the cells, pooled | 0.530 | 0.871 |
| Mean, better member | 0.582 | 0.880 |
| Mean, members' mean | 0.503 | 0.865 |
| Mean, worse member | 0.423 | 0.850 |
| Pooled beats the better member | 32 of 108 | 36 of 108 |
| Pooled beats the members' mean | 66 of 108 | 66 of 108 |
| Pooled beats the worse member | 88 of 108 | 86 of 108 |
| Seeds averaged first, beats the better member | 11 of 36 | 14 of 36 |
| Seeds averaged first, beats the members' mean | 21 of 36 | 21 of 36 |
| Seeds averaged first, beats the worse member | 29 of 36 | 28 of 36 |

Seeds averaged first: the pooled model's and each member's value are averaged over the three seeds, and the better member is chosen on those means. armd.csv instead takes the better member seed by seed, which is why its AP count of seed-averaged wins can differ by one or two.

## By fold and target, mean over twelve cells

| Pool | Target | Pooled AUROC | Better member | Worse member | Beats the better member | Negatives under the rule |
|---|---|---|---|---|---|---|
| pool_no_kaggle_v2 | kaggle_v2 (held out) | 0.549 | 0.561 | 0.492 | 3 of 12 | no |
| pool_no_kaggle_v2 | twoclass | 0.490 | 0.592 | 0.437 | 2 of 12 | yes |
| pool_no_kaggle_v2 | visual | 0.357 | 0.450 | 0.345 | 3 of 12 | yes |
| pool_no_lohi_weld | lohi_weld (held out) | 0.643 | 0.780 | 0.345 | 1 of 12 | yes |
| pool_no_lohi_weld | twoclass | 0.427 | 0.512 | 0.422 | 2 of 12 | yes |
| pool_no_lohi_weld | visual | 0.607 | 0.563 | 0.379 | 7 of 12 | yes |
| pool_no_weld_surface_v2 | twoclass | 0.486 | 0.577 | 0.440 | 3 of 12 | yes |
| pool_no_weld_surface_v2 | visual | 0.589 | 0.560 | 0.407 | 7 of 12 | yes |
| pool_no_weld_surface_v2 | weld_surface_v2 (held out) | 0.621 | 0.645 | 0.545 | 4 of 12 | no |
