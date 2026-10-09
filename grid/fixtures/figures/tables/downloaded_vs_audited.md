# Downloaded against audited, three numbers per source

A. downloaded train, downloaded test: what a practitioner publishes.
B. downloaded train, audited test: the same weights, regraded on the audited test split.
C. audited train, audited test: the main grid diagonal.

**Column B is not a held-out test, and must never be called one.** The downloaded and audited partitions were drawn independently, so most of the audited test split is inside the downloaded training split. The share is measured per source in the last column below. B therefore scores a model largely on images it was trained on, and is reported only to hold the weights fixed while the grading partition changes.

A minus B changes only the test set. A minus C changes the whole pipeline. **Neither is a causal estimate of leakage.** Both are estimate shifts under a change of protocol, measured on different test sets.

| source | A | B | C | A-B, test set only | A-C, full pipeline | audited test images already in downloaded train |
|---|---|---|---|---|---|---|
| kaggle_v2 | 0.820 | 0.741 | 0.744 | 0.079 | 0.076 | 110 of 141 (78%) |
| lohi_weld | 1.000 | 0.981 | 0.948 | 0.019 | 0.052 | 305 of 453 (67%) |
| weld_surface_v2 | 0.967 | 0.886 | 0.886 | 0.081 | 0.081 | 392 of 445 (88%) |

Restricting B to the genuinely unseen remainder was considered and rejected. The remainders are kaggle_v2 31, lohi_weld 148, weld_surface_v2 53 images, too few and too heavily one-class to carry a number.
