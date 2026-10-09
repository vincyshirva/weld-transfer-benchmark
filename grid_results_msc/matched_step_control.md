# Matched-step and oversampling control for arm B
## EXPLORATORY, NOT pre-registered

Primary endpoint `last`, the project log 6a-ii. Box mAP50, mean over the three draws. Every arm was trained ON THIS MACHINE, so the comparison is internally valid; the project log 6a-iii is why that matters.

| arm | distinct images | steps | cosine over | close_mosaic share |
|---|---|---|---|---|
| A fixed 50 | k | ~100 to 900 | 50 epochs | 20% |
| B oversampled 50 | k | matched to k=250 | 50 epochs | 20% |
| C matched step | k | matched to k=250 | scaled epochs | 1.25% |

**A against B isolates the STEP COUNT**, schedule shape held constant. **B against C isolates the SCHEDULE SHAPE**, step count held constant. The label budget is identical in all three.

## The three arms

| pair | k | A fixed 50 | B oversampled | C matched step | A to B (steps) | B to C (schedule) |
|---|---|---|---|---|---|---|
| kaggle_v2 to lohi_weld | 10 | 0.0811 | 0.2467 | 0.2847 | +0.1656 | +0.0379 |
| kaggle_v2 to lohi_weld | 25 | 0.1404 | n/a | n/a | n/a | n/a |
| kaggle_v2 to lohi_weld | 50 | 0.2166 | 0.3697 | 0.3491 | +0.1531 | -0.0206 |
| kaggle_v2 to twoclass | 10 | 0.0268 | 0.1545 | n/a | +0.1277 | n/a |
| kaggle_v2 to visual | 10 | 0.0000 | 0.0153 | n/a | +0.0153 | n/a |
| lohi_weld to kaggle_v2 | 10 | 0.0022 | 0.0047 | n/a | +0.0024 | n/a |
| lohi_weld to kaggle_v2 | 25 | 0.0013 | n/a | n/a | n/a | n/a |
| lohi_weld to kaggle_v2 | 50 | 0.0048 | n/a | 0.0061 | n/a | n/a |

**Where the gain comes from, per coordinate.** Over the 2 coordinate(s) with ALL THREE arms, the step count share is 81 percent, 116 percent.

**The schedule-shape term does not have a consistent sign, so no single percentage is quoted.** It is +0.0379, -0.0206 across those coordinates. The STEP COUNT is the consistent driver, contributing +0.1656, +0.1531. The stretched cosine and the shrunken mosaic window add a small residual that is positive at one coordinate and negative at the other, which is what a second coordinate was added to find out. 6a-xi's conclusion survives either way: the small-k floor is an optimisation BUDGET limit.

## Gain against step multiplier, and against the pair

**THE STEP MULTIPLIER IS NOT THE DRIVER. THE TARGET IS.** This table was added to test the opposite hypothesis, that the small gain on `lohi_weld` to `kaggle_v2` was a small step deficit rather than the pair. On the five cells available at the time the two were perfectly confounded: every kaggle to lohi cell had a 16x multiplier and every lohi to kaggle cell had 1.8x. Four more cells broke that confound and the answer reversed. See the summary under the table.

| pair | k | draw | steps at 50 ep | steps oversampled | multiplier | delta box mAP50 |
|---|---|---|---|---|---|---|
| kaggle_v2 to lohi_weld | 10 | 0 | 100 | 1650 | 16.5x | +0.2095 |
| kaggle_v2 to lohi_weld | 10 | 1 | 100 | 1650 | 16.5x | +0.1652 |
| kaggle_v2 to lohi_weld | 10 | 2 | 100 | 1600 | 16.0x | +0.1220 |
| kaggle_v2 to lohi_weld | 50 | 0 | 700 | 1400 | 2.0x | +0.0630 |
| kaggle_v2 to lohi_weld | 50 | 1 | 350 | 1600 | 4.6x | +0.1726 |
| kaggle_v2 to lohi_weld | 50 | 2 | 350 | 1600 | 4.6x | +0.2235 |
| kaggle_v2 to twoclass | 10 | 0 | 150 | 1350 | 9.0x | +0.2710 |
| kaggle_v2 to twoclass | 10 | 1 | 100 | 1400 | 14.0x | +0.0954 |
| kaggle_v2 to twoclass | 10 | 2 | 250 | 1450 | 5.8x | +0.0168 |
| kaggle_v2 to visual | 10 | 0 | 100 | 1650 | 16.5x | +0.0231 |
| kaggle_v2 to visual | 10 | 1 | 100 | 1650 | 16.5x | +0.0067 |
| kaggle_v2 to visual | 10 | 2 | 150 | 1700 | 11.3x | +0.0160 |
| lohi_weld to kaggle_v2 | 10 | 0 | 100 | 2050 | 20.5x | -0.0002 |
| lohi_weld to kaggle_v2 | 10 | 1 | 100 | 1650 | 16.5x | -0.0010 |
| lohi_weld to kaggle_v2 | 10 | 2 | 600 | 1750 | 2.9x | +0.0086 |

| cut | n | mean delta box mAP50 |
|---|---|---|
| pair kaggle_v2 to lohi_weld | 6 | +0.1593 |
| pair kaggle_v2 to twoclass | 3 | +0.1277 |
| pair kaggle_v2 to visual | 3 | +0.0153 |
| pair lohi_weld to kaggle_v2 | 3 | +0.0024 |
| multiplier below 5x, pooled | 4 | +0.1169 |
| multiplier 5x or above, pooled | 11 | +0.0840 |

**The pair separates the gains and the multiplier does not.** Pooled across pairs, a multiplier below 5x and one at 5x or above give almost the same mean, while the two pairs differ by more than an order of magnitude. The decisive cells are the hard pair at a 16x and 20x multiplier, which gain nothing, against the easy pair at a 2x multiplier, which gains a great deal.

**The better explanation is target headroom, and it was pre-stated.** `kaggle_v2` as a target has an in domain box AP of 0.1151 against `lohi_weld`'s 0.6849, about a sixth, so there is little for any schedule to recover. The project log 5n already establishes kaggle_v2 as a hard box target on four independent checks. This control cannot resolve much there either way, which is why the expectation was written down before the cells were run.

**`50 epochs` is NOT a constant step budget.** Across these cells the fixed arm ranges from 100 to 900 steps at a nominal 50 epochs, because block sampling overshoots k by up to 9.1 times and the realised image count varies. The project log 6a-i. Any statement about 'the 50 epoch schedule' is therefore a statement about a range of step budgets.

**EXPLORATORY. This is not the pre-registered arm.** The the project log statistics section pre-registers a comparison of CONFIGURATIONS blocked on evaluation cell. Nothing was pre-registered about the training schedule, and the project log 6a-xii records the decision to leave `ARMB_EPOCHS` at 50 rather than re-scope the arm around this result.

**The residual difference between B and C, stated rather than hidden.** Oversampling shows each image R times WITHIN an epoch under R different random augmentations, while the matched-step arm shows it once per epoch across many epochs. Mosaic draws from the same k image pool either way. The two are not identical. That difference is second order beside the cosine stretch, and saying so is what makes the control credible.

**The head is reinitialised in every cell here**, the project log 6a-iv, because the source and target class counts differ in both pairs. That is the reason the step budget matters at all.
