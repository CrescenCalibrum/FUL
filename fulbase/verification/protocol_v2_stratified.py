"""Fixed-class, paired-family margin diagnostics alongside the unchanged v2 tests.

These percentile intervals are exploratory and conditional on fixed checkpoints,
classes and marker generators. They are not a replacement for the pooled-family
t/Holm protocol, a formal audit, or independent confirmation of a tuned rule.
"""
from __future__ import annotations

from dataclasses import asdict
import math

import numpy as np

from fulbase.verification.protocol_v2 import (
    V2DiagnosticConfig, _reduce, validate_observations,
)
from fulbase.verification.statistics import combine_status


VARIANTS = ("equivalence", "noninferiority")


def _interval_status(kind, bound, low, high):
    """Classify a simultaneous interval; a boundary itself is never support."""
    if kind not in {"decrease", *VARIANTS} or not math.isfinite(bound) or bound <= 0:
        raise ValueError("Invalid stratified margin condition")
    if low is None or high is None:
        return "insufficient"
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError("Invalid stratified margin interval")
    if kind == "decrease":
        return "pass" if high < -bound else "fail" if low > -bound else "insufficient"
    if kind == "noninferiority":
        return "pass" if low > -bound else "fail" if high < -bound else "insufficient"
    if low > -bound and high < bound:
        return "pass"
    return "fail" if high < -bound or low > bound else "insufficient"


def _fixed_class_statistics(labels, full, after):
    """Separate change in the estimand from change in estimated uncertainty."""
    labels = np.asarray(labels)
    full, after = np.asarray(full, dtype=np.float64), np.asarray(after, dtype=np.float64)
    changes = after-full
    classes = sorted(set(labels.tolist()))
    weight = 1.0/len(classes)
    pooled_mean = float(changes.mean())
    total_ss = float(np.sum((changes-pooled_mean)**2))
    within_ss = between_ss = 0.0
    class_rows, strata = [], []
    for label in classes:
        mask = labels == label
        values = changes[mask]
        strata.append(values)
        n = len(values)
        mean = float(values.mean())
        variance = float(values.var(ddof=1)) if n > 1 else None
        within_ss += float(np.sum((values-mean)**2))
        between_ss += n*(mean-pooled_mean)**2
        class_rows.append({
            "label": int(label), "n_families": n, "class_weight": weight,
            "full_mean": float(full[mask].mean()), "after_mean": float(after[mask].mean()),
            "mean": mean, "within_class_variance": variance,
            "contribution_to_stratified_variance": weight**2*variance/n if variance is not None else None,
            "status": "descriptive_only", "used_as_acceptance_gate": False,
        })
    fixed_mean = float(np.mean([row["mean"] for row in class_rows]))
    components = [row["contribution_to_stratified_variance"] for row in class_rows]
    analytic_se = math.sqrt(sum(components)) if all(x is not None for x in components) else None
    counts = [len(values) for values in strata]
    stats = {
        "mean": fixed_mean, "family_weighted_mean": pooled_mean,
        "estimand_difference": fixed_mean-pooled_mean,
        "full_mean": float(np.mean([row["full_mean"] for row in class_rows])),
        "after_mean": float(np.mean([row["after_mean"] for row in class_rows])),
        "n_families": len(changes), "n_classes": len(classes),
        "min_families_per_class": min(counts), "max_families_per_class": max(counts),
        "equal_families_per_class": len(set(counts)) == 1,
        "small_sample_per_class": min(counts) < 8,
        "zero_variance_classes": sum(row["within_class_variance"] == 0 for row in class_rows),
        "analytic_se": analytic_se,
        "pooled_se": float(changes.std(ddof=1)/np.sqrt(len(changes))) if len(changes) > 1 else None,
        "total_ss": total_ss, "within_class_ss": float(within_ss), "between_class_ss": float(between_ss),
        "within_class_fraction": float(within_ss/total_ss) if total_ss > 0 else None,
        "between_class_fraction": float(between_ss/total_ss) if total_ss > 0 else None,
    }
    return stats, class_rows, strata


def _bootstrap_fixed_classes(strata, mean, options, cid):
    # A private generator per client avoids changing any existing v2 draws. The
    # same family weights apply to paired deltas, not separately to pre/post.
    rng = np.random.default_rng(np.random.SeedSequence([options.bootstrap_seed, cid]))
    draws = np.empty(options.bootstrap_samples, dtype=np.float64)
    for start in range(0, len(draws), 256):
        size = min(256, len(draws)-start)
        chunk = np.full(size, mean, dtype=np.float64)
        for values in strata:
            n = len(values)
            counts = rng.multinomial(n, np.full(n, 1.0/n), size=size)
            chunk += (counts @ (values-values.mean()))/(n*len(strata))
        draws[start:start+size] = chunk
    return draws


def _intervals(stats, strata, protocol, options, cid, num_clients):
    tail = protocol["alpha"]/(4*num_clients)
    expected_tail = options.bootstrap_samples*tail
    descriptive_tail = (1-options.confidence)/2
    descriptive_expected_tail = options.bootstrap_samples*descriptive_tail
    reasons = []
    if stats["n_families"] < protocol["min_families"]:
        reasons.append("too_few_total_families")
    if stats["min_families_per_class"] < 2:
        reasons.append("too_few_families_in_a_class")
    if stats["analytic_se"] is not None and stats["analytic_se"] <= 1e-15:
        reasons.append("zero_within_class_variance")
    if expected_tail < 10:
        reasons.append("insufficient_bootstrap_tail_resolution")
    intervals = {
        "bootstrap_se": None, "descriptive_ci_low": None, "descriptive_ci_high": None,
        "descriptive_confidence": options.confidence,
        "descriptive_expected_tail_draws": descriptive_expected_tail,
        "descriptive_interval_status": "insufficient",
        "descriptive_interval_reason": "client_inference_ineligible",
        "ci_low": None, "ci_high": None, "interval_confidence": 1-2*tail,
        "interval_tail_probability": tail, "expected_tail_draws": expected_tail,
        "margin_family_alpha": protocol["alpha"]/2, "num_simultaneous_clients": num_clients,
        "eligible": False,
    }
    if not reasons:
        draws = _bootstrap_fixed_classes(strata, stats["mean"], options, cid)
        se = float(draws.std(ddof=1))
        if not np.isfinite(draws).all() or not math.isfinite(se):
            reasons.append("nonfinite_bootstrap_distribution")
        elif se <= 1e-15 or float(np.ptp(draws)) <= 1e-15:
            reasons.append("degenerate_bootstrap_distribution")
        else:
            low, high = np.quantile(draws, [tail, 1-tail])
            if high <= low:
                reasons.append("degenerate_bootstrap_interval")
            else:
                intervals.update(bootstrap_se=se, ci_low=float(low), ci_high=float(high), eligible=True)
                if descriptive_expected_tail >= 10:
                    dlo, dhi = np.quantile(draws, [descriptive_tail, 1-descriptive_tail])
                    intervals.update(descriptive_ci_low=float(dlo), descriptive_ci_high=float(dhi),
                                     descriptive_interval_status="available",
                                     descriptive_interval_reason="available_exploratory_approximation")
                else:
                    intervals["descriptive_interval_reason"] = "insufficient_bootstrap_tail_resolution"
    intervals.update(reason=";".join(reasons) if reasons else "available_exploratory_approximation",
                     eligibility_reasons=reasons)
    return intervals


def _pooled_rows(v2_report, ids, forget, protocol, reductions):
    rows = v2_report["margin_tests"]
    indexed = {(row["cid"], row["variant"]): row for row in rows}
    expected = {(cid, variant) for cid in ids for variant in VARIANTS}
    if len(indexed) != len(rows) or set(indexed) != expected:
        raise ValueError("Pooled comparison has missing/duplicate clients or variants")
    for (cid, variant), row in indexed.items():
        kind = "decrease" if cid in forget else variant
        bound = protocol["margin_drop"] if cid in forget else protocol["retain_margin_tolerance"]
        delta = reductions[cid][1][1]-reductions[cid][0][1]
        if (row["kind"] != kind or row["bound"] != bound
                or row["alpha"] != protocol["alpha"]/2 or row["n_families"] != len(delta)):
            raise ValueError("Pooled comparison differs from source margin protocol")
        if row["mean"] is not None and not math.isclose(row["mean"], float(delta.mean()), abs_tol=1e-12):
            raise ValueError("Pooled comparison differs from paired family means")
    return indexed


def diagnose_stratified_margins(before, after, forget_client_ids, marker_config,
                                audit_protocol, v2_report, options=None, *, expected_role="calibration"):
    """Return separate fixed-class diagnostics without changing a supplied report."""
    options = options or V2DiagnosticConfig()
    validate_observations(before, after, forget_client_ids, marker_config, audit_protocol,
                          expected_role=expected_role)
    ids, forget = sorted(before), set(forget_client_ids)
    reductions = {cid: (_reduce(before[cid]), _reduce(after[cid])) for cid in ids}
    pooled = _pooled_rows(v2_report, ids, forget, audit_protocol, reductions)
    prechecks = {}
    for cid in ids:
        pre = before[cid]
        learned = (float(np.mean(pre["predictions"] == pre["labels"])) >= audit_protocol["min_full_accuracy"]
                   and float(np.asarray(pre["normalized_margin"]).mean()) >= audit_protocol["min_full_margin"])
        prechecks[cid] = "pass" if learned else "insufficient"
    all_learned = all(value == "pass" for value in prechecks.values())
    client_tests, class_metrics = [], []
    for cid in ids:
        a, b = reductions[cid]
        stats, classes, strata = _fixed_class_statistics(a[0], a[1], b[1])
        intervals = _intervals(stats, strata, audit_protocol, options, cid, len(ids))
        group = "forget" if cid in forget else "retain"
        for row in classes:
            class_metrics.append({"cid": cid, "client_group": group,
                                  "n_samples": int(np.sum(np.asarray(before[cid]["labels"]) == row["label"])),
                                  **row})
        for variant in VARIANTS:
            kind = "decrease" if cid in forget else variant
            bound = audit_protocol["margin_drop"] if cid in forget else audit_protocol["retain_margin_tolerance"]
            raw_status = _interval_status(kind, bound, intervals["ci_low"], intervals["ci_high"])
            row = {"cid": cid, "client_group": group, "variant": variant,
                   "condition": "forget_margin" if cid in forget else "retain_margin",
                   "kind": kind, "bound": bound, "n_samples": len(before[cid]["ids"]),
                   **stats, **intervals, "raw_status": raw_status,
                   "status": raw_status if prechecks[cid] == "pass" else "insufficient",
                   "full_marker_precheck": prechecks[cid],
                   "decision_basis": "approximate_simultaneous_fixed_class_percentile_interval",
                   "task_included": False}
            row.update({"pooled_"+key: value for key, value in pooled[cid, variant].items()})
            row["pooled_interval_confidence"] = 1-audit_protocol["alpha"]/2
            client_tests.append(row)
    summary = []
    for variant in VARIANTS:
        relevant = [row for row in client_tests if row["variant"] == variant]
        raw = combine_status(row["raw_status"] for row in relevant)
        row = {"variant": variant, "pooled_status": v2_report["margin_variants"][variant]["status"],
               "stratified_status": raw if all_learned else "insufficient", "raw_status": raw,
               "full_marker_precheck_passed": all_learned, "task_included": False,
               "eligible_client_count": sum(item["eligible"] for item in relevant),
               "small_sample_client_count": sum(item["small_sample_per_class"] for item in relevant),
               "interpretation": "exploratory_interval_comparison_not_formal_acceptance"}
        for group, selector in (("target", lambda cid: cid in forget), ("retain", lambda cid: cid not in forget)):
            for status in ("pass", "fail", "insufficient"):
                row[group+"_"+status+"_count"] = sum(item["status"] == status for item in relevant if selector(item["cid"]))
        summary.append(row)
    return {"schema_version": "marker-fixed-class-margin-diagnostic-v1", "status": "diagnostic_only",
            "formal_audit": False, "protocol_locked": False,
            "client_tests": client_tests, "class_metrics": class_metrics, "summary": summary,
            "protocol": {**asdict(options), "source_alpha": audit_protocol["alpha"],
                         "margin_family_alpha": audit_protocol["alpha"]/2,
                         "margin_drop": audit_protocol["margin_drop"],
                         "retain_margin_tolerance": audit_protocol["retain_margin_tolerance"],
                         "estimand": "equal_weight_over_fixed_classes_of_mean_paired_family_margin_change",
                         "resampling": "paired_families_within_each_fixed_client_class_at_original_stratum_sizes",
                         "family_reduction": "mean_variants_before_and_after_then_paired_difference_as_original_v2",
                         "interval_method": "percentile_bootstrap",
                         "decision_correction": "Bonferroni_simultaneous_two_sided_intervals_over_all_client_means",
                         "num_simultaneous_clients": len(ids),
                         "interval_tail_probability": audit_protocol["alpha"]/(4*len(ids)),
                         "pooled_method": "unchanged_paired_family_t_with_Holm_over_support_and_violation",
                         "pooled_interval_confidence": 1-audit_protocol["alpha"]/2,
                         "min_total_families": audit_protocol["min_families"], "min_families_per_class": 2,
                         "min_expected_tail_draws": 10,
                         "small_sample_warning_below_families_per_class": 8,
                         "small_sample_warning_is_acceptance_gate": False,
                         "zero_within_class_variance_rule": "insufficient_inference_keep_point_estimate",
                         "classes_resampled": False, "variants_resampled_independently": False,
                         "cross_method_variant_or_run_multiplicity_control": False,
                         "new_joint_geometry_or_task_acceptance": False},
            "limitations": [
                "Exploratory calibration diagnostic, not independent confirmation or a locked audit",
                "Percentile bootstrap with few families per class has approximate, potentially biased coverage",
                "Ten expected tail draws is a numerical-resolution guard, not a validity or power guarantee",
                "Analytic stratified SE and empirical bootstrap SE are distinct; small strata bias bootstrap variance downward",
                "Old Holm and new interval Bonferroni corrections differ; not every status change is caused only by stratification",
                "Unequal families per class change the estimand; both means and their difference are reported",
                "Equal observed values cannot establish zero population variance; degenerate intervals do not provide inference",
                "Class rows are descriptive; they are not added as per-class acceptance gates",
                "Uncertainty is conditional on fixed classes, checkpoint and marker generator, not across training seeds",
            ]}
