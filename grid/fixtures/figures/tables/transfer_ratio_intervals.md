# Transfer ratios with split-block confidence intervals

`T_ij = P_ij / P_jj`, same configuration, same seed, and a 95 percent split-block percentile interval. **The number of bootstrap draws differs by level and is stated at the head of each section.** Every row also records its own count in the `resamples` column of `transfer_ratio_intervals.csv`.

**The unit is the split block, not the image.** Images from one weld scene are correlated and adjacent video frames are nearly the same photograph, so resampling images independently would treat those as independent evidence and make every interval too narrow.

**Numerator and denominator are resampled together.** Both are scored on the same test images of the target, so one draw selects the blocks and both models are then scored on exactly those images. The diagonal rows are the self-check: when the source is the target the two models are the same model, so T must be exactly 1.000 with zero width.

**This interval answers a different question from the Friedman test.** Friedman asks whether the four configurations differ from each other, and it found no detectable difference. These intervals ask how large the collapse is, which is the claim the paper actually makes.
