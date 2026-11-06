"""Family-level paired inference. No p-value is interpreted as a probability of removal."""
from __future__ import annotations

import numpy as np
from scipy.stats import t


def mean_test(values, kind, bound, alpha=0.05, min_families=8):
    """TOST, one-sided degradation, or exact degenerate finite-sample differences.

    State is based on Holm-corrected p-values later. Descriptive CI is two-sided
    1-alpha; one-sided/TOST inference has its own explicitly returned p-values.
    """
    x = np.asarray(values, dtype=np.float64)
    if kind not in {"equivalence", "decrease"} or bound <= 0 or not 0 < alpha < 0.5:
        raise ValueError("Invalid statistical test specification")
    out = {"kind": kind, "bound": bound, "n_families": int(len(x)),
           "mean": None, "ci_low": None, "ci_high": None,
           "p_support": None, "p_violation": None, "status": "insufficient"}
    if len(x) < min_families or not np.isfinite(x).all():
        return out
    mean = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(len(x)))
    def cdf_at(mu):
        if se < 1e-15:
            return 0.0 if mean < mu else (1.0 if mean > mu else 0.5)
        return float(t.cdf((mean - mu) / se, len(x) - 1))
    low_p, high_p = cdf_at(-bound), cdf_at(bound)
    if kind == "decrease":
        support, violation = low_p, 1 - low_p
    else:
        support = max(1 - low_p, high_p)  # TOST: reject either side of non-equivalence
        violation = min(1.0, 2 * min(low_p, 1 - high_p))  # two directions Bonferroni
    half = float(t.ppf(1 - alpha / 2, len(x) - 1)) * se
    out.update(mean=mean, ci_low=mean-half, ci_high=mean+half,
               p_support=support, p_violation=violation)
    return out


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    if not len(p):
        return []
    order = np.argsort(p)
    adjusted = np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)).clip(0, 1)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out.tolist()


def correct_tests(tests, alpha):
    # Both support and violation claims belong to the same per-request test family.
    slots = [(i, key) for i, test in enumerate(tests) for key in ("p_support", "p_violation")
             if test.get(key) is not None]
    adjusted = holm([tests[i][key] for i, key in slots])
    for (i, key), value in zip(slots, adjusted):
        tests[i][key + "_holm"] = value
    for test in tests:
        if test.get("p_support_holm", 1) < alpha:
            test["status"] = "pass"
        elif test.get("p_violation_holm", 1) < alpha:
            test["status"] = "fail"
        else:
            test["status"] = "insufficient"
    return tests


def combine_status(states):
    states = list(states)
    if "fail" in states:
        return "fail"
    return "pass" if states and all(s == "pass" for s in states) else "insufficient"
