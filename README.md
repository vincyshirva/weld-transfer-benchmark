# Transferability of deep object detectors for visible-light weld inspection under camera and dataset shift

This repository is the released artefact for the study of that name. It holds
the harmonised label mappings, the frozen split files with their sha256 hashes,
the leakage audit, the training and evaluation code, and the analysis that
produces every figure and table in the paper.

Vincent Linish Dsouza, Department of Robotics and Artificial Intelligence,
NMAM Institute of Technology, Nitte, Karnataka.

## READ THIS FIRST: match split entries on the FILENAME, not on the path

A line in a split file looks like this:

```
test/images/Good-Welding-images_20_jpeg.rf.05786e2171fa8723312372d80c4d6762.jpg
```

**That `test/` prefix is OUR audited partition. It is not the partition the
file shipped in.** The two are unrelated, because the original authors
partitioned their data and we repartitioned it after auditing for leakage. Of
the 141 images in our audited `kaggle_v2` test split, 110 shipped in the
original training split, 26 in validation and 5 in test.

So resolving a split line by its full path against an original download fails
for most entries, and makes this release look broken when it is not. **Match on
the basename.** The basename is the original filename and is unique within each
source.

This is the one thing a reader can get wrong in a way that looks like a defect
in the artefact, which is why it is the first section rather than a footnote.

## No images are redistributed here, and that is our choice rather than a licence constraint

This repository contains **no image data from any source**. Obtain the
originals from their authors at the locations cited in the paper, then
reconstruct the partitions from the split files.

The licences are as follows.

| Source | Licence | Redistribution permitted by the licence? |
|---|---|---|
| `kaggle_v2` | CC0 1.0 Universal | **Yes** |
| `weld_surface_v2` | CC BY 4.0 | Yes, with attribution |
| `twoclass` | CC BY 4.0 | Yes, with attribution to its upstream creator, kaid Al2 |
| `visual` | CC BY 4.0 | Yes, with attribution |
| `lohi_weld` (LoHi-WELD) | Free to use with citation | Yes, with citation |

**Four of the five licences would permit us to redistribute the images, and we
do not.** The decision is about provenance, not permission. The CC0 source's own
upstream description states that its images were "sourced from various image
collection and datasets" without naming any of them, and no per-image
provenance ships with the download. CC0 there is an aggregator's dedication over
material the aggregator collected elsewhere, which is what the distributor
asserts rather than a verified chain of title. We prefer to point readers at the
authoritative copy than to propagate a second one.

For the radiographic sources named in the paper as out of scope, GDXray Welds
forbids redistribution outright, so that one is a constraint.

**Where each source comes from.** `kaggle_v2` is version 2 of
`kaggle.com/datasets/sukmaadhiwijaya/welding-defect-object-detection`.
`weld_surface_v2` is export version 2 of the Roboflow Universe project
`weld-defect-detection-onlal/weld-surface-defect-detection-884h9`, and `visual`
is version 1 of `faza/visual-welding-defect`. `lohi_weld` is LoHi-WELD, from
`github.com/SylvioBlock/LoHi-Weld`. **`twoclass` needs one extra step.** Its
upstream, `kaid-al2/welding-defect-2class`, publishes no dataset version and
cannot be exported as it stands, so the copy used here was taken from a public
fork, `vincent-linish-d-souza-nmamit-mech/welding-defect-2class-dadbu`, export
version 1. Download from the fork to get the files the split lists name.

**Not redistributing costs a reader nothing but a download.** All 7,784 entries
across the fifteen split files were verified to resolve against untouched
original downloads. Most recently checked **21 September 2026**, against the
files as tracked in this repository.

## What is here

| Path | What it is |
|---|---|
| `clean_candidate/splits/` | The fifteen frozen split files, five sources by three partitions |
| `clean_candidate/split_hashes.json` | sha256 of every split file. Verify against this rather than regenerating |
| `clean_candidate/manifest.csv` | Exact lineage: which original file each retained image came from |
| `weld_audit.py` | The authoritative leakage audit for a rebuilt tree |
| `weld_rebuild_clean.py` | Builds the deduplicated tree and solves the partitioning |
| `weld_acceptance.py` | Checks the audit, split balance, lineage, portable paths and hashes |
| `lohi_to_yolo.py` | Converts LoHi-WELD to YOLO layout, both acquisitions |
| `grid/` | The frozen 69 cell experiment runner, evaluation, metrics and analysis |
| `grid/arms/` | The control and mitigation arms, 300 cells |
| `grid/fixtures/` | Synthetic records that let the analysis be tested without real results |
| `models/pretrained/manifest.json` | Every initial checkpoint pinned by URL, byte count and sha256 |
| `build_results_section.py` | Builds the manuscript sections from the released tables |
| `manuscript_figures.py` | Draws the manuscript's four figures at their printed width, from the same released tables |

`grid_results/analysis/` and `grid_results_arms/analysis/` hold the generated
tables and figures. The checkpoints and raw predictions are not in the
repository: they are 28 GB.

## Verifying the splits

```bash
python3 weld_acceptance.py clean_candidate
```

That checks the audit, the split balance, the lineage, the portable data.yaml
paths and the split hashes, and prints `Acceptance result: PASS`. The argument
is the tree to check and is required. Its report lands in
`clean_candidate/acceptance_report.json`.

If you have materialised the arm datasets, five further folders appear beside
the five sources. The acceptance check skips them by name and records which it
skipped, because their split lists are registered in
`grid/arms/arm_split_hashes.json` rather than in this tree.

To check the hashes alone:

```bash
python3 -c "import hashlib, json, pathlib; h=json.load(open('clean_candidate/split_hashes.json')); print(all(hashlib.sha256(pathlib.Path('clean_candidate/splits',pathlib.Path(k).name).read_bytes()).hexdigest()==(v if isinstance(v,str) else v['sha256']) for k,v in (h.get('splits') or h).items()))"
```

## Reproducing the analysis

Two commands produce every manuscript artefact from retrieved results.

```bash
python3 grid/analysis.py
```

```bash
python3 grid/arms/analysis_arms.py
```

Neither takes arguments. Both read from the results directories and write
tables and figures beside them. Neither needs a GPU.

The manuscript itself is built from those tables, and the build is where the
consistency checks live: every quoted range against its table, the transcribed
tables against the released CSVs, every bootstrap draw count against the rows
it describes, and every citation against the reference list. It needs
`python-docx` and nothing that imports torch.

```bash
python3 build_results_section.py --clean
```

## Do not regenerate the splits

The released artefact is the frozen split files and their hashes, **not the
solver**. Four of the five sources reached the solver's wall-clock limit rather
than proving optimality, so the returned partition depends on machine timing: two
runs of the same code on the same input with the same seed return different
splits. Longer solving does not help, and fifteen times the budget moved the
objective by about four parts in a billion.

Anyone who re-runs `weld_rebuild_clean.py` gets a different, equally valid
partition. **Numbers measured on two different partitions must not be
compared.** Verify against the hashes instead.

## A note on reproducibility that we would rather state than have discovered

Every cell records the sha256 of the configuration, the evaluation code, the
metric implementation and the dataset loader at the moment it was trained, and
the evaluator refuses to score a checkpoint whose recorded hashes no longer
match. Those hashes are released per cell.

**We do not claim that every cell can be re-evaluated from the head of this
repository.** One evaluator revision between cell generations was not retained,
so the sixty oldest cells pin a version of that file which no longer exists.
Their results are unaffected, because the corrected branch executes only for a
cell kind none of them uses, and the three files whose movement would change a
reported number have never moved across any generation. The evaluator in this
release, `grid/evaluate_cell.py`, is the revision the remaining cells record:
its sha256 begins `1c430196`.

## Environment

The laptop environment is torch 2.11.0+cu128 with ultralytics 8.4.138. The
cluster environment, on which every reported cell was trained, is torch
2.5.1+cu121 with ultralytics 8.4.138 on CUDA 12.1. These are not
interchangeable. `h200_requirements.txt` holds the cluster lock, and
`laptop_requirements.txt` holds the laptop's pins. The audit and rebuild
scripts need only `pillow numpy pyyaml scipy`.

## Citing the datasets

Attribute all five sources. The CC0 source imposes no legal requirement to
attribute, and we attribute it anyway, because CC0 removes the legal
requirement and not the scholarly one.

Full citations, DOIs and access locations are in the paper's data availability
statement and reference list.

## Licences

The code is released under the MIT licence, in `LICENSE`. The split files,
hash manifests, lineage, label mappings, audit outputs and the generated
tables and figures are released under CC BY 4.0, in `LICENSE-DATA`. Neither
covers the source images, which are not here and remain under their creators'
licences, listed above.

## Running the cluster scripts

The Slurm scripts name absolute paths in the form `/home/USERNAME/...`.
Replace `USERNAME` with your own account, or the whole path with your project
root. The cluster the study used requires absolute paths, so do not make them
relative. `grid/fetch_results.sh` pulls results over ssh from a host alias,
`nmamit-h200`. Point that alias at your own cluster in `~/.ssh/config`.

## What is not here

Code comments and some generated reports cite "the project log" by entry
number, for example "the project log, 5p". That log is the author's internal
record of decisions and is not released.

The released tables reproduce every number in the manuscript, and
`build_results_section.py` rebuilds it from them. Reproducing the tables
themselves from scratch needs more than this repository holds. The trained
checkpoints and raw predictions, 28 GB, are not included, so it means
obtaining the public source images from their creators, rebuilding the
audited tree from the split files, and retraining.
