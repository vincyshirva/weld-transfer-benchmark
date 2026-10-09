#!/usr/bin/env python3
"""Per-class in-domain AP, recomputed with the protocol's own NMS applied.

WHY THIS MODULE EXISTS, AND WHY IT IS NOT A PATCH TO `evaluate_cell.py`.

`evaluate_cell.flatten` takes a `collapse` flag. With it, predictions are
filtered to the source's defect ids and then passed through
`metrics.class_agnostic_nms`. Without it, no suppression runs at all.

The box-level path passes `collapse=True`. The per-class in-domain path does
not. For three of the four detector families that makes no difference, because
their backends already apply NMS before a prediction is ever stored: a second
pass removes zero boxes. RT-DETR is the exception. It emits a fixed set of
queries and applies no suppression of its own, so 31.8 percent of its boxes on
one measured cell are duplicates that every other family had removed before
scoring.

The consequence was that two released tables disagreed. `cells.csv` put
RT-DETR-l's `visual` diagonal at 0.1302 and `per_class_ap.md` put the same
quantity at 0.119, because `visual` has exactly one class in the harmonised
defect space and the two files were computing the same number two ways. The
per-class figure was the wrong one: it charged RT-DETR for duplicates that the
protocol's stated NMS IoU of 0.70 exists to remove, and that every other
family had already had removed, which makes a cross-family per-class table an
unfair comparison rather than a wrong one by a constant.

`evaluate_cell.py` is one of the four files pinned by sha256 in all 69
`cell.json` records, so editing it would make every completed cell
un-re-evaluable. The project log 5m set the precedent for exactly this situation:
`transfer_bootstrap.py` holds a faster AP implementation outside `metrics.py`
for the same reason. This module is the same move. It recomputes from the
archived raw predictions, which store every prediction with its class before
any filtering, so nothing needs re-inference.

`reproduces_box_level()` is the check that matters. On a source whose defect
space holds exactly one class, per-class AP with NMS and class-agnostic box AP
are the same quantity computed twice, so they must agree to floating point.

WHICH SUPPRESSION RULE THIS MODULE USES, AND WHY. DECIDED 20 SEPTEMBER 2026.

There are two defensible definitions and the agreement check above cannot tell
them apart, because on a single-defect-class source they are the same rule.

  (a) Filter the predictions to one class, then suppress within that class.
      This is the conventional definition of per-class average precision.
  (b) Suppress class-agnostically across the whole defect space first, then
      filter to one class. This is the rule the box-level matrix uses, where
      every defect class collapses to a single label.

**This module uses (a), and the difference is not negligible**, so the choice
had to be made rather than inherited. Measured on RT-DETR-l, the family where
it bites, (b) moves `twoclass` Non-fusion defect by -0.0999 and `lohi_weld`
deposit by -0.0810. Run `compare_suppression_rules()` to reproduce.

The argument for (a) is that (b) makes a per-class number non-local. Under (b)
the value reported for `deposit` depends on which OTHER classes happen to sit
in the defect space, because a `stain` box can suppress a `deposit` box. Remove
`stain` from the taxonomy and `deposit`'s AP changes, though nothing about
deposit detection did. A per-class column whose entries move when a different
class is added is not a per-class measurement. (b) is right for the box matrix
precisely because there the classes are deliberately collapsed into one label,
so cross-class suppression is suppressing duplicates of the same target.

Note that (a) is what a single-class application of `class_agnostic_nms`
computes: with one class in the subset, class-agnostic and per-class
suppression coincide. The function name describes its argument, not the
policy.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import evaluate_cell as E  # noqa: E402
import metrics as M  # noqa: E402

RESULTS = C.PROJECT / "grid_results" / "results"
CACHE_NAME = "per_class_nms_cache.json"


def cache_path(results_dir: Path) -> Path:
    """The cache file for one results directory. It sits beside that directory.

    Both bootstrap caches already live at `results_dir.parent`, and this
    follows them. For the real results that is `grid_results/`, the file this
    module has always used, so the existing cache is found with no migration.
    For the fixtures it would be `grid/fixtures/`, but `load()` writes nothing
    there, because the fixtures ship no raw archives.

    UNTIL 25 SEPTEMBER 2026 THE PATH WAS FIXED at `<project>/grid_results/`,
    whatever was being analysed. So a fixture run wrote its empty cache into
    the real slot whenever that slot was free. That happened in every git
    worktree, which carries no real results. It could also happen in the main
    checkout, once the real cache was deleted to force a rebuild. The next real
    run there would find a cache built for the fixtures and get nothing back.
    It would then fall back to the records' own per-class values, which skip
    NMS. That is the the project log 5p defect returning, announced by one line.
    """
    return Path(results_dir).resolve().parent / CACHE_NAME


def _ground_truth(cache: dict, target: str):
    if target not in cache:
        cache[target] = E.load_ground_truth(target)
    return cache[target]


def _in_domain_entries(results_dir: Path):
    """What `recompute()` corrects: (cell name, raw archive, target) per entry.

    `recompute()` and `expected_keys()` both read this one generator. So the
    cache check and the computation cannot disagree about which cells count.
    """
    results_dir = Path(results_dir)
    for path in sorted(results_dir.glob("eval__*.json")):
        record = json.loads(path.read_text())
        if record["cell"].get("kind") == "dirty":
            # A downloaded-partition cell is graded on a split that overlaps
            # its own training data. The project log 5o keeps those rows out of the
            # in-domain table entirely, so there is nothing to correct here.
            continue
        name = path.name[len("eval__"):-len(".json")]
        raw_path = results_dir / f"raw__{name}.npz"
        if not raw_path.is_file():
            continue
        for key, entry in record["targets"].items():
            if "per_class_in_domain" in entry:
                yield name, raw_path, C.split_result_key(key)[0]


def expected_keys(results_dir: Path) -> set[str]:
    """Every key `recompute()` would return here, found without computing AP.

    It reads the eval__ records and checks that each raw archive exists. It
    loads no prediction, so it costs milliseconds where a rebuild costs
    minutes.
    """
    return {f"{name}|{target}|{cname}"
            for name, _, target in _in_domain_entries(results_dir)
            for cname in C.DATASETS[target]["classes"]}


def recompute(results_dir: Path = RESULTS, verbose: bool = True) -> dict:
    """Every in-domain per-class AP, with NMS, keyed "cell|target|class".

    Excluded classes are computed too, and flagged. `evaluate_cell` skips
    them, so they are absent from the released records, but a reader cannot
    see the harmonised taxonomy without them: `visual` would show one class of
    three with no explanation of the other two.
    """
    gt_cache: dict = {}
    out: dict = {}
    for name, raw_path, target in _in_domain_entries(results_dir):
        raw = np.load(raw_path)
        gt = _ground_truth(gt_cache, target)
        img = raw[f"{target}__pred_image"]
        box = raw[f"{target}__pred_box"]
        score = raw[f"{target}__pred_score"]
        cls = raw[f"{target}__pred_class"]
        n_images = int(img.max()) + 1 if len(img) else 0
        preds = [(box[img == i], score[img == i], cls[img == i])
                 for i in range(n_images)]
        excluded = set(C.DATASETS[target]["excluded"])
        defects = C.defect_ids(target)
        for cid, cname in enumerate(C.DATASETS[target]["classes"]):
            sub = [(b[c == cid], s[c == cid], c[c == cid])
                   for b, s, c in preds]
            gts = gt["gt_by_class"].get(cid, {})
            ap = M.average_precision_range(*E.flatten(sub, collapse=True),
                                           gts)
            n_inst = sum(len(v) for v in gts.values())
            out[f"{name}|{target}|{cname}"] = {
                "map50": ap["map50"], "map50_95": ap["map50_95"],
                "test_instances": int(n_inst),
                "thin": bool(n_inst < C.MIN_TEST_INSTANCES),
                "in_defect_space": cid in defects,
                "excluded": cname in excluded,
            }
        if verbose:
            print(f"    {name}")
    return out


def load(results_dir: Path = RESULTS, refresh: bool = False,
         verbose: bool = False) -> dict:
    """Cached recompute. It returns exactly what `recompute()` would.

    A CACHE IS TRUSTED ONLY WHEN IT FITS. It must record this results
    directory, and it must hold exactly the values that the raw archives here
    call for, as `expected_keys()` lists them. Anything else is rebuilt from
    the archives, with a notice saying why. A rebuild of the real results
    takes about five minutes. `refresh=True` forces one.

    NOTHING IS WRITTEN WHEN THERE IS NOTHING TO CORRECT. A results directory
    that does not exist yet, or holds no raw archive, returns {} and leaves no
    cache. The fixtures are that case, since they ship no raw archives, and {}
    is their correct answer. They never read a cache either. So they cannot
    pick up real values under their deliberately real cell names, which is
    the contamination the project log 5l says a fixture set must be incapable of.

    The recorded directory still matters. Two results directories under one
    parent share a cache file, and each rebuilds rather than read the other's
    values.

    UNTIL 25 SEPTEMBER 2026 TWO PATHS RETURNED {} FOR REAL RESULTS. analysis.py
    then fell back to the records' own per-class values, which skip NMS, and
    that is the the project log 5p defect in 17 of 68 class groups.

    (1) A run against a results directory that did not exist yet, or that
        held no raw archive, wrote a cache holding {} under the correct key.
        Once the results arrived, that cache was trusted and returned {} with
        no notice. `python3 grid/per_class_nms.py` in a fresh clone did this.
    (2) The key is the absolute path of the results directory. After the
        project folder was moved or copied the key no longer matched. The
        only sign was one printed line.

    (1) now rebuilds, because an empty cache does not hold the values the
    archives call for. Such a cache is no longer written either. (2) now
    rebuilds, because a cache built for another directory is not trusted.

    WHAT THIS CANNOT NOTICE. A raw archive or record rewritten in place under
    the same cell name leaves the key set unchanged, so the old values would
    be read. The main grid is frozen, and the 60 main and reference cells
    cannot be re-evaluated, the project log 9a, so this should not arise. If a cell
    is ever re-evaluated, delete the cache.

    analysis.py adds a backstop of its own. It refuses to write a per-class
    row that lacks this correction, unless the cell is a downloaded-data cell
    or the whole results set is fabricated. So a missing raw archive stops
    the run instead of reaching the table.
    """
    results_dir = Path(results_dir).resolve()
    wanted = expected_keys(results_dir)
    if not wanted:
        return {}
    cache = cache_path(results_dir)
    if cache.is_file() and not refresh:
        blob = json.loads(cache.read_text())
        built_for = blob.get("results_dir")
        have = set(blob.get("values") or {})
        if built_for == str(results_dir) and have == wanted:
            return blob["values"]
        if verbose:
            why = (f"was built for {built_for}, not {results_dir}"
                   if built_for != str(results_dir) else
                   f"holds {len(have & wanted)} of the {len(wanted)} values "
                   f"the raw archives here call for, and {len(have - wanted)} "
                   f"they do not")
            print(f"  per-class NMS cache {cache} {why}. Rebuilding it from "
                  f"the raw archives, which takes minutes.")
    elif verbose:
        print("  recomputing per-class AP with NMS, from raw archives")
    data = recompute(results_dir, verbose=verbose)
    cache.write_text(json.dumps(
        {"results_dir": str(results_dir), "resamples": None, "values": data},
        indent=1, sort_keys=True))
    return data


def reproduces_box_level(results_dir: Path = RESULTS, tol: float = 1e-9) -> bool:
    """THE CHECK THAT ARBITRATES THE TWO PATHS.

    Where a source's defect space holds exactly one class, the corrected
    per-class AP and the class-agnostic box AP are the same quantity. They
    must agree to floating point. If they do not, this module is wrong.
    """
    data = load(results_dir)
    single = [t for t in C.DATASETS if len(C.defect_ids(t)) == 1]
    checked = worst = 0
    for path in sorted(results_dir.glob("eval__*.json")):
        record = json.loads(path.read_text())
        if record["cell"].get("kind") == "dirty":
            continue
        name = path.name[len("eval__"):-len(".json")]
        for key, entry in record["targets"].items():
            target = C.split_result_key(key)[0]
            if target not in single or "per_class_in_domain" not in entry:
                continue
            box = (entry.get("box_level") or {}).get("map50")
            if box is None:
                continue
            cid = sorted(C.defect_ids(target))[0]
            cname = C.DATASETS[target]["classes"][cid]
            got = data[f"{name}|{target}|{cname}"]["map50"]
            worst = max(worst, abs(got - box))
            checked += 1
    print(f"  single-defect-class agreement: {checked} cells, worst "
          f"difference {worst:.2e}")
    return checked > 0 and worst <= tol


def compare_suppression_rules(results_dir: Path = RESULTS,
                              family: str = "rtdetr_l") -> float:
    """Rule (a) against rule (b) on the multi-class sources. See the docstring.

    The single-defect-class agreement check is blind to this choice, because
    there the two rules coincide by construction. This is the measurement that
    is not blind to it, and it runs on the one family whose backend does not
    already pick a single label per box.
    """
    from collections import defaultdict
    import statistics

    gt_cache: dict = {}
    acc = defaultdict(lambda: defaultdict(list))
    for path in sorted(results_dir.glob("eval__*.json")):
        record = json.loads(path.read_text())
        cell = record["cell"]
        if cell.get("kind") == "dirty" or cell["family"] != family:
            continue
        name = path.name[len("eval__"):-len(".json")]
        raw_path = results_dir / f"raw__{name}.npz"
        if not raw_path.is_file():
            continue
        raw = np.load(raw_path)
        for key, entry in record["targets"].items():
            target = C.split_result_key(key)[0]
            ids = C.defect_ids(target)
            if len(ids) < 2 or "per_class_in_domain" not in entry:
                continue
            gt = _ground_truth(gt_cache, target)
            img = raw[f"{target}__pred_image"]
            preds = [(raw[f"{target}__pred_box"][img == i],
                      raw[f"{target}__pred_score"][img == i],
                      raw[f"{target}__pred_class"][img == i])
                     for i in range(int(img.max()) + 1)]
            agnostic = []
            for b, s, c in preds:
                m = np.isin(c, list(ids))
                b, s, c = b[m], s[m], c[m]
                if len(b):
                    k = M.class_agnostic_nms(b, s)
                    b, s, c = b[k], s[k], c[k]
                agnostic.append((b, s, c))
            for cid, cname in enumerate(C.DATASETS[target]["classes"]):
                if cid not in ids:
                    continue
                gts = gt["gt_by_class"].get(cid, {})
                a = M.average_precision_range(
                    *E.flatten([(b[c == cid], s[c == cid], c[c == cid])
                                for b, s, c in preds], collapse=True), gts)
                b_ = M.average_precision_range(
                    *E.flatten([(b[c == cid], s[c == cid], c[c == cid])
                                for b, s, c in agnostic]), gts)
                acc[(target, cname)]["a"].append(a["map50"])
                acc[(target, cname)]["b"].append(b_["map50"])
    worst = 0.0
    print(f"  {'source':<18}{'class':<20}{'(a) per-class':>14}"
          f"{'(b) class-agn':>15}{'diff':>9}")
    for key in sorted(acc):
        a = statistics.fmean(acc[key]["a"])
        b_ = statistics.fmean(acc[key]["b"])
        worst = max(worst, abs(a - b_))
        print(f"  {key[0]:<18}{key[1]:<20}{a:>14.4f}{b_:>15.4f}{b_ - a:>+9.4f}")
    print(f"  largest difference between the two rules: {worst:.4f}")
    print("  the released table uses (a). See the module docstring.")
    return worst


if __name__ == "__main__":
    load(refresh="--refresh" in sys.argv, verbose=True)
    ok = reproduces_box_level()
    print("per-class AP reproduces the box-level path:", ok)
    if "--compare-rules" in sys.argv:
        compare_suppression_rules()
    sys.exit(0 if ok else 1)
