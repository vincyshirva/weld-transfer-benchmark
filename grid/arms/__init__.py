"""Mitigation arms and controls, built on the frozen main grid.

THIS PACKAGE EXISTS TO KEEP FOUR FILES BYTE IDENTICAL.

`evaluate_cell.evaluate()` refuses to score a checkpoint unless the current
sha256 of `config.py`, `evaluate_cell.py`, `metrics.py` and `yolo_dataset.py`
match what the cell recorded at training time, and unless
`config_fingerprint()` still returns what it recorded. All 69 completed main
cells depend on that. Adding a dataset or a cell kind to `config.py` would
change its hash and make every one of them un-re-evaluable. The project log 5h
records what it cost the last time a guarded file moved.

So nothing here edits those four files. The arms:

  * import the frozen configuration READ ONLY and never write to it on disk
  * register their own datasets into `config.DATASETS` AT RUNTIME, in this
    process only, which leaves the file untouched
  * carry their own cell registry, their own data.yaml resolution and their
    own fingerprint, with the frozen f5567a1938278e88 embedded as a component
  * reuse `train_cell` and `evaluate_cell` functions by import, so there is no
    second copy of the training or metric code to drift

The runtime registration is safe because a main-grid re-evaluation runs in a
separate process that never imports this package. `arms.preflight_arms`
asserts exactly that, and the smoke test re-evaluates a main cell afterwards
to prove it.
"""
