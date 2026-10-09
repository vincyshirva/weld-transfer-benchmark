# Pre-registered statistics, arms

Primary is Friedman blocked on evaluation cell with Nemenyi post hoc. Secondary is repeated measures ANOVA with Tukey. Decided before any result existed, see the the project log statistics section. Metric is box mAP50.

- blocks, seed averaged: **6**
- Friedman chi2 2.0, df 3, p = **0.5724**
- significant at 0.05: **False**
- RM-ANOVA F 0.8256, p = **0.5000**, partial eta squared 0.1417
- primary and secondary agree: **True**
- sensitivity, (cell, seed) blocks: **18**, chi2 4.2, p = **0.2407**

**Mean ranks. The metric is box mAP50 and `rankdata` ranks ascending, so a HIGHER mean rank is the BETTER configuration.** YOLO11n 2.667, YOLO11s 3.0, RT-DETR-l 2.333, Faster R-CNN 2.0

**Six blocks is thin, and that is a property of the design, not a shortfall.** The LoHi control contributes its four cells and arm D contributes only the two box defined cells of the one pool that excludes weld_surface_v2. Every other arm D cell is protocol mixed, so its box level number is undefined by the rule in the project log 6b. A Friedman over six blocks has very little power, so read a null here as uninformative rather than as evidence the configurations agree. The main grid's 15 block test is the one that carries the pre-registered claim.

Arm B is excluded from the omnibus because it runs two of the four configurations by pre-registered design, the project log 6a, so it cannot form a complete block over four.

## 5c. Do the arms agree with the main grid that architecture does not rescue transfer?

- main grid Friedman, image level: p = **0.257**
- main grid Friedman, box level: p = **0.789**
- arms Friedman, box level: p = **0.5724**
- arms agree: **True**

**Mean arm B box mAP50 by configuration, endpoint `last`.** RT-DETR-l 0.0926, YOLO11s 0.0941.

Arm B was run at two configurations rather than four precisely because the main grid found no detectable difference among the four, the project log 6a. Nothing in the arms overturns that. Read the arms null as consistent with the main grid rather than as an independent confirmation, because six blocks cannot confirm much.

## 5b. Exploratory, NOT pre-registered: does k matter?

**This test is exploratory.** The pre-registered omnibus in the the project log statistics section is about configurations, blocked on evaluation cell. Nothing was pre-registered about k. It is reported here so the question is answered, and it is labelled so no reader mistakes it for the pre-registered claim.

- design: complete block, k levels crossed with (source, target, configuration, draw)
- blocks: **36**, k levels: 10, 25, 50, 100, 250
- Friedman chi2 104.226, df 4, p = **1.238e-21**, significant **True**

**Mean ranks over k. Higher is better, because `rankdata` ranks ascending and the metric is box mAP50.** k=10 1.806, k=25 1.917, k=50 2.583, k=100 3.75, k=250 4.944

So the ranks increase monotonically with k, which is the expected direction and a useful check that the test is reading the data the right way up.

**A significant k effect is NOT the same as a useful one, and the two must not be conflated.** The ranks say larger budgets score higher, which is unsurprising. A rank test detects ordering, not magnitude. At the largest budget, k = 250, recovery of the in domain gap runs from 0.115 to 1.059 across the 12 box defined pair and configuration combinations, median 0.622. **Never pool these into one number**: the denominators span roughly a factor of nine, the project log 6a-xiii. Quote the recovery numbers in section 4, not this p value.

Metric is box mAP50 at the primary `last` endpoint. Image level is not used here because it is floored by prevalence on the positive heavy targets, the project log 5k.

## 5d. Seed dispersion, against the main grid median of 0.0047

| group | n | median SD | mean SD | min | max |
|---|---|---|---|---|---|
| LoHi control and arm D, over seeds | 24 | **0.0163** | 0.0212 | 0.0018 | 0.061 |
| arm B, over draws | 60 | **0.0086** | 0.0315 | 0.0 | 0.2015 |
| main grid, over seeds | 32 | **0.0047** | 0.0074 | 0.0 | 0.0522 |

All three are box mAP50. Until 24 September 2026 the main grid row was a typed 0.038 with no derivation behind it; it is now computed from the frozen main-grid records.

**Arm B's three draws are not seed replicates in the same sense, and its dispersion is reported separately for that reason.** The draw index IS the seed, the project log 6a-v, so which images were sampled and the training randomness move together by design. Its spread therefore mixes sampling variance with training variance and nothing in the archives separates them. Do not compare it against the main grid median as if it measured the same thing.
