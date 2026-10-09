#!/usr/bin/env python3
"""
The pre-registered comparison between detector configurations.

DECIDED 5 SEPTEMBER 2026, BEFORE ANY RESULT EXISTS. Recorded here and in
the project log. Do not change it after seeing the grid.

THE DESIGN, AND WHY THE OBVIOUS TESTS ARE WRONG

Four configurations are each scored on the same evaluation cells. A cell is one
(training source, target dataset) pair, and there are 15 of them at image
level. The four scores in a cell come from four models trained on the same
images and graded on the same test images. So the design is COMPLETE and
BLOCKED: every treatment appears in every block.

That structure rules out two tempting choices.

  One-way ANOVA with Tukey, as Lema et al. (10.1007/s10845-025-02672-8) use on
  NEU-DET, assumes independent observations. Our 15 cells are not independent
  replicates of one another: they share target images across configurations
  and share training data within a source. Treating them as independent
  inflates the residual degrees of freedom and understates the p values. Their
  design has independent runs on one dataset; ours is crossed.

  Cell-level Wilcoxon, which this project used earlier, does block correctly,
  but it compares only two configurations at a time. With four configurations
  that is six pairwise tests, and the omnibus question, whether configuration
  matters at all, is never asked. It also silently averages over seeds and so
  discards training variance.

THE CHOICE

  Primary    Friedman test across the four configurations, blocked on
             evaluation cell, with a Nemenyi post-hoc.
  Secondary  Repeated-measures ANOVA on the same blocking, with Tukey HSD,
             reported for direct comparability with Lema et al.

Friedman is the correct omnibus test for a complete block design and makes no
distributional assumption. That matters here: AP is bounded on [0, 1], the
cross-domain cells will pile up near zero, and 15 blocks is far too few to
defend normality. Nemenyi is its matched post-hoc and controls the family-wise
error rate across all pairwise comparisons without a separate correction.

The repeated-measures ANOVA is reported alongside because Lema et al. is the
methodological precedent this project cites, and a reader who knows that paper
should be able to see the same analysis. Where the two disagree, the Friedman
result is the one the paper claims, and the disagreement is itself reported.

SEEDS

Seeds are replicates within a cell, not blocks. The primary test uses the mean
over the three seeds as the cell value, and seed dispersion is reported
separately as a standard deviation. A sensitivity analysis repeats the test
with (cell, seed) as the blocking unit, which triples the block count and
treats training noise as part of the signal. If the two disagree, the paper
reports both and says so.

WHAT IS NOT CLAIMED

A significant Friedman result says the configurations differ somewhere on
these cells. It does not say a configuration generalizes better in any causal
sense, and it does not extrapolate beyond the five datasets in this study.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from scipy import stats

ALPHA = 0.05


def _matrix(cell_scores: dict[str, dict[str, float]], configurations: list[str]):
    """Blocks x treatments, dropping blocks with any missing configuration.

    A Friedman test needs a complete design. A cell where one configuration
    failed to produce a score cannot contribute, and quietly imputing it would
    invent data. Dropped blocks are counted and reported.
    """
    blocks, rows = [], []
    for block, scores in sorted(cell_scores.items()):
        values = [scores.get(c) for c in configurations]
        if any(v is None or not math.isfinite(float(v)) for v in values):
            continue
        blocks.append(block)
        rows.append([float(v) for v in values])
    return blocks, np.array(rows, dtype=float)


def nemenyi(matrix: np.ndarray, configurations: list[str]) -> dict:
    """Post-hoc for Friedman. Critical difference on mean ranks.

    CD = q_alpha * sqrt(k(k+1) / (6N)), with q_alpha from the studentized
    range distribution divided by sqrt(2). Two configurations differ when
    their mean ranks are further apart than CD.
    """
    n_blocks, k = matrix.shape
    ranks = np.apply_along_axis(stats.rankdata, 1, matrix)
    mean_ranks = ranks.mean(axis=0)
    q = stats.studentized_range.ppf(1 - ALPHA, k, np.inf) / math.sqrt(2)
    cd = q * math.sqrt(k * (k + 1) / (6 * n_blocks))
    pairs = []
    for i, j in itertools.combinations(range(k), 2):
        diff = abs(mean_ranks[i] - mean_ranks[j])
        pairs.append({
            "a": configurations[i], "b": configurations[j],
            "mean_rank_a": round(float(mean_ranks[i]), 3),
            "mean_rank_b": round(float(mean_ranks[j]), 3),
            "rank_difference": round(float(diff), 3),
            "critical_difference": round(float(cd), 3),
            "significant": bool(diff > cd),
        })
    return {"mean_ranks": {c: round(float(r), 3)
                           for c, r in zip(configurations, mean_ranks)},
            "critical_difference": round(float(cd), 3),
            "alpha": ALPHA, "pairs": pairs}


def repeated_measures_anova(matrix: np.ndarray) -> dict:
    """Two-way ANOVA without replication: treatment and block main effects.

    Computed directly rather than through statsmodels so the grid keeps a
    small dependency set. The partition is the standard one:
    SS_total = SS_treatment + SS_block + SS_error.
    """
    n, k = matrix.shape
    grand = matrix.mean()
    ss_treat = n * ((matrix.mean(axis=0) - grand) ** 2).sum()
    ss_block = k * ((matrix.mean(axis=1) - grand) ** 2).sum()
    ss_total = ((matrix - grand) ** 2).sum()
    ss_error = ss_total - ss_treat - ss_block
    df_treat, df_error = k - 1, (n - 1) * (k - 1)
    if df_error <= 0 or ss_error <= 0:
        return {"applicable": False,
                "reason": "no residual degrees of freedom"}
    ms_treat, ms_error = ss_treat / df_treat, ss_error / df_error
    f = ms_treat / ms_error
    p = float(stats.f.sf(f, df_treat, df_error))
    # partial eta squared, the effect size Lema et al. report
    eta_sq = ss_treat / (ss_treat + ss_error)
    return {"applicable": True,
            "f_statistic": round(float(f), 4),
            "df_treatment": df_treat, "df_error": df_error,
            "p_value": p, "significant": bool(p < ALPHA),
            "partial_eta_squared": round(float(eta_sq), 4)}


def tukey_hsd(matrix: np.ndarray, configurations: list[str]) -> dict:
    """Tukey HSD across configurations, for comparability with Lema et al."""
    try:
        result = stats.tukey_hsd(*[matrix[:, i] for i in range(matrix.shape[1])])
    except Exception as exc:  # noqa: BLE001
        return {"applicable": False, "reason": f"{type(exc).__name__}: {exc}"}
    pairs = []
    for i, j in itertools.combinations(range(len(configurations)), 2):
        pairs.append({
            "a": configurations[i], "b": configurations[j],
            "mean_difference": round(float(result.statistic[i, j]), 4),
            "p_value": float(result.pvalue[i, j]),
            "significant": bool(result.pvalue[i, j] < ALPHA),
        })
    return {"applicable": True, "alpha": ALPHA, "pairs": pairs}


def compare_configurations(cell_scores: dict[str, dict[str, float]],
                           configurations: list[str],
                           metric: str = "image_ap") -> dict:
    """Run the pre-registered comparison on one metric.

    cell_scores maps a block label, normally "source->target", to a mapping
    from configuration name to that configuration's score in that cell.
    """
    blocks, matrix = _matrix(cell_scores, configurations)
    n_dropped = len(cell_scores) - len(blocks)
    out = {
        "metric": metric,
        "configurations": configurations,
        "n_blocks": len(blocks),
        "n_blocks_dropped_incomplete": n_dropped,
        "blocks": blocks,
        "design": "complete block, configurations crossed with evaluation cells",
        "preregistered": True,
    }
    if len(blocks) < 3 or matrix.shape[1] < 3:
        out["applicable"] = False
        out["reason"] = (f"Friedman needs at least 3 blocks and 3 treatments, "
                         f"got {len(blocks)} and {matrix.shape[1]}")
        return out

    chi2, p = stats.friedmanchisquare(*[matrix[:, i] for i in range(matrix.shape[1])])
    out["applicable"] = True
    out["primary"] = {
        "test": "Friedman, blocked on evaluation cell",
        "chi2": round(float(chi2), 4),
        "df": matrix.shape[1] - 1,
        "p_value": float(p),
        "significant": bool(p < ALPHA),
        "posthoc": nemenyi(matrix, configurations),
    }
    out["secondary"] = {
        "test": "repeated-measures ANOVA with Tukey HSD, after Lema et al. "
                "10.1007/s10845-025-02672-8",
        "anova": repeated_measures_anova(matrix),
        "tukey": tukey_hsd(matrix, configurations),
    }
    prim = out["primary"]["significant"]
    sec = out["secondary"]["anova"].get("significant")
    out["primary_and_secondary_agree"] = (sec is None or prim == sec)
    if not out["primary_and_secondary_agree"]:
        out["disagreement_note"] = (
            "Friedman and the repeated-measures ANOVA disagree on the omnibus "
            "result. The paper claims the Friedman outcome, because AP is "
            "bounded and the block count is too small to defend normality, "
            "and reports this disagreement explicitly.")
    return out
