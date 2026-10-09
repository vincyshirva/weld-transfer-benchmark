# Per class AP, arms

No per class block is present in the arm evaluation records. `evaluate_arm.py` writes per class AP only for the in domain case, and the arm evaluators are cross acquisition or cross dataset throughout. Reported as absent rather than as an empty table, so it cannot be mistaken for classes that scored zero.

The in domain per class table is the main grid's, in `grid_results/analysis/tables/per_class_ap.md`, and every row there already carries its test instance count.
