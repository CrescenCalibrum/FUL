"""Explicit development ablations and descriptive client-by-class diagnostics.

Nothing here changes v1 or an existing v2 report on disk. Candidate bounds are
supplied explicitly by the developer; no automatic threshold fitting or
selection is performed. The comparisons are not a locked audit protocol.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.stats import t

from fulbase.verification.markers import MarkerConfig, client_anchor
from fulbase.verification.protocol_v2 import (
    V2DiagnosticConfig, validate_observations, _reduce, _normalize_centers,
)
from fulbase.verification.statistics import combine_status


@dataclass(frozen=True)
class V2AblationConfig:
    accuracy_drop: float
    retain_accuracy_tolerance: float
    anchor_drop: float
    anchor_selectivity_drop: float
    retain_anchor_tolerance: float
    min_full_anchor_advantage: float = 0.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"Ablation bound must be a finite number: {name}")
        if any(getattr(self, name) <= 0 for name in (
                "accuracy_drop", "retain_accuracy_tolerance", "anchor_drop",
                "anchor_selectivity_drop", "retain_anchor_tolerance")):
            raise ValueError("Ablation effect/tolerance bounds must be positive")
        if self.accuracy_drop > 1 or self.retain_accuracy_tolerance > 1:
            raise ValueError("Accuracy bounds must be fractions in (0, 1]")
        if not 0 <= self.min_full_anchor_advantage <= 2:
            raise ValueError("Full anchor advantage bound must be in [0, 2]")


def _family_accuracy(obs):
    correct = np.asarray(obs["predictions"]) == np.asarray(obs["labels"])
    families = np.asarray(obs["families"])
    return np.asarray([correct[families == fid].mean() for fid in sorted(set(families))])


def _descriptive_interval(values, confidence):
    x = np.asarray(values, dtype=np.float64)
    if len(x) < 2:
        return None, None
    mean = float(x.mean())
    half = float(t.ppf((1+confidence)/2, len(x)-1) * x.std(ddof=1) / np.sqrt(len(x)))
    return mean-half, mean+half


def _class_diagnostics(before, after, forget, protocol, options):
    rows, decompositions = [], []
    for cid in sorted(before):
        pre, post = before[cid], after[cid]
        labels, full_margin, _ = _reduce(pre)
        _, after_margin, _ = _reduce(post)
        changes = after_margin-full_margin
        overall = float(changes.mean())
        total_ss = float(np.sum((changes-overall)**2))
        within_ss = between_ss = 0.0
        class_means, family_counts = [], []
        declining_accuracy = below_margin_bound = 0
        full_correct = np.asarray(pre["predictions"]) == np.asarray(pre["labels"])
        after_correct = np.asarray(post["predictions"]) == np.asarray(post["labels"])
        for label in sorted(set(labels.tolist())):
            family_mask = labels == label
            sample_mask = np.asarray(pre["labels"]) == label
            delta = changes[family_mask]
            mean = float(delta.mean())
            nf = int(family_mask.sum())
            class_means.append(mean)
            family_counts.append(nf)
            between_ss += nf*(mean-overall)**2
            within_ss += float(np.sum((delta-mean)**2))
            lo, hi = _descriptive_interval(delta, options.confidence)
            acc_before = float(full_correct[sample_mask].mean())
            acc_after = float(after_correct[sample_mask].mean())
            declining_accuracy += acc_after < acc_before
            below_margin_bound += mean < -protocol["retain_margin_tolerance"]
            rows.append({
                "cid": cid, "client_group": "forget" if cid in forget else "retain",
                "label": int(label), "n_samples": int(sample_mask.sum()), "n_families": nf,
                "accuracy_full": acc_before, "accuracy_after": acc_after,
                "accuracy_delta": acc_after-acc_before,
                "gained_correct": int(np.sum(sample_mask & ~full_correct & after_correct)),
                "lost_correct": int(np.sum(sample_mask & full_correct & ~after_correct)),
                "normalized_margin_full": float(full_margin[family_mask].mean()),
                "normalized_margin_after": float(after_margin[family_mask].mean()),
                "normalized_margin_delta": mean,
                "family_delta_sd": float(delta.std(ddof=1)) if nf >= 2 else None,
                "margin_delta_ci_low": lo, "margin_delta_ci_high": hi,
                "interval_confidence": options.confidence,
                "interval_status": "available" if nf >= 2 else "insufficient",
                "interval_method": "descriptive_family_t_not_multiplicity_adjusted",
                "status": "descriptive_only", "used_as_acceptance_gate": False,
            })
        decompositions.append({
            "cid": cid, "client_group": "forget" if cid in forget else "retain",
            "n_families": len(changes), "n_classes": len(class_means),
            "family_weighted_margin_delta": overall,
            "class_balanced_margin_delta": float(np.mean(class_means)),
            "equal_families_per_class": len(set(family_counts)) == 1,
            "total_ss": total_ss, "between_class_ss": float(between_ss),
            "within_class_ss": float(within_ss),
            "between_class_fraction": float(between_ss/total_ss) if total_ss > 0 else None,
            "within_class_fraction": float(within_ss/total_ss) if total_ss > 0 else None,
            "classes_with_accuracy_decline": int(declining_accuracy),
            "classes_with_mean_margin_below_negative_reference_tolerance": int(below_margin_bound),
            "reference_margin_tolerance": protocol["retain_margin_tolerance"],
            "min_class_mean_margin_delta": min(class_means),
            "max_class_mean_margin_delta": max(class_means),
            "status": "descriptive_only", "used_as_acceptance_gate": False,
        })
    return rows, decompositions


def _accuracy_diagnostics(before, after, forget, protocol, bounds, all_learned):
    """Simple empirical baseline, not a t-test on bounded correctness values.

    In particular, identical observed correctness does not imply a zero-width
    population interval or a zero preservation p-value. No such quantities are
    emitted for this baseline.
    """
    rows, summaries = [], {}
    for variant in ("equivalence", "noninferiority"):
        tests = []
        for cid in sorted(before):
            values = _family_accuracy(after[cid])-_family_accuracy(before[cid])
            mean = float(values.mean())
            if cid in forget:
                bound, kind = bounds.accuracy_drop, "empirical_decrease"
                supported = mean < -bound
            elif variant == "equivalence":
                bound, kind = bounds.retain_accuracy_tolerance, "empirical_equivalence"
                supported = abs(mean) < bound
            else:
                bound, kind = bounds.retain_accuracy_tolerance, "empirical_noninferiority"
                supported = mean > -bound
            empirical = "pass" if supported else "fail"
            eligible = len(values) >= protocol["min_families"]
            test = {"cid": cid, "variant": variant,
                    "condition": "forget_accuracy" if cid in forget else "retain_accuracy",
                    "kind": kind, "mean": mean, "bound": bound,
                    "n_families": len(values), "n_samples": len(before[cid]["labels"]),
                    "status": empirical if eligible else "insufficient",
                    "empirical_status": empirical, "eligible": eligible,
                    "reason": "empirical_threshold_only" if eligible else "insufficient_families",
                    "accuracy_full": float(_family_accuracy(before[cid]).mean()),
                    "accuracy_after": float(_family_accuracy(after[cid]).mean()),
                    "decision_basis": "empirical_point_threshold_no_inference",
                    "estimand": "family_mean_paired_marker_accuracy_change"}
            tests.append(test)
        raw = combine_status(row["status"] for row in tests)
        summaries[variant] = {"status": raw if all_learned else "insufficient", "raw_status": raw}
        rows.extend(tests)
    return rows, summaries


def _geometry_diagnostics(before, after, forget, markers, protocol, options, bounds):
    ids = sorted(before)
    retain = [cid for cid in ids if cid not in forget]
    a = {cid: _reduce(before[cid]) for cid in ids}
    b = {cid: _reduce(after[cid]) for cid in ids}
    strata = {cid: [np.flatnonzero(a[cid][0] == label) for label in sorted(set(a[cid][0]))]
              for cid in ids}
    eligible = {cid: len(a[cid][0]) >= protocol["min_families"]
                and all(len(s) >= 2 for s in strata[cid]) for cid in ids}
    dim = a[ids[0]][2].shape[1]
    anchors = np.stack([client_anchor(markers, cid, dim, "cpu").numpy().astype(np.float64)
                        for cid in ids])
    units = anchors/np.linalg.norm(anchors, axis=1, keepdims=True)
    others = (units.sum(0)[None, :]-units)/(len(ids)-1)
    full = np.stack([_normalize_centers(np.mean([a[cid][2][s].mean(0) for s in strata[cid]], axis=0))
                     for cid in ids])
    post = np.stack([_normalize_centers(np.mean([b[cid][2][s].mean(0) for s in strata[cid]], axis=0))
                     for cid in ids])
    point_delta = ((post-anchors)**2).sum(1)-((full-anchors)**2).sum(1)
    point_advantage = (full*(units-others)).sum(1)
    rcols = [ids.index(cid) for cid in retain]
    point_selective = point_delta-point_delta[rcols].mean()
    claims = 3*len(forget)+len(retain)
    tail = protocol["alpha"]/(4*claims)
    enough_draws = options.bootstrap_samples*tail >= 10
    draws_delta = np.empty((options.bootstrap_samples, len(ids)))
    draws_advantage = np.empty_like(draws_delta)
    rng = np.random.default_rng(options.bootstrap_seed)
    for start in range(0, options.bootstrap_samples, 256):
        count = min(256, options.bootstrap_samples-start)
        for col, cid in enumerate(ids):
            full_draw = np.zeros((count, dim))
            post_draw = np.zeros((count, dim))
            for s in strata[cid]:
                weights = rng.multinomial(len(s), np.full(len(s), 1/len(s)), size=count)
                weights = weights/(len(s)*len(strata[cid]))
                full_draw += weights @ a[cid][2][s]
                post_draw += weights @ b[cid][2][s]
            full_draw, post_draw = _normalize_centers(full_draw), _normalize_centers(post_draw)
            draws_delta[start:start+count, col] = (((post_draw-anchors[col])**2).sum(1)
                                                  - ((full_draw-anchors[col])**2).sum(1))
            draws_advantage[start:start+count, col] = full_draw @ (units[col]-others[col])
    draws_selective = draws_delta-draws_delta[:, rcols].mean(1, keepdims=True)
    rows = []

    def append(cid, condition, metric, mean, draws, bound, direction, family_eligible):
        valid_interval = family_eligible and enough_draws
        low = high = None
        status = "insufficient"
        if valid_interval:
            low, high = map(float, np.quantile(draws, [tail, 1-tail]))
            if direction == "greater":
                status = "pass" if low > bound else ("fail" if high < bound else "insufficient")
            else:
                status = "pass" if high < bound else ("fail" if low > bound else "insufficient")
        reason = ("development_candidate_bound_not_independently_validated" if valid_interval else
                  "too_few_bootstrap_tail_draws" if not enough_draws else "insufficient_family_strata")
        rows.append({"cid": cid, "client_group": "forget" if cid in forget else "retain",
                     "condition": condition, "metric": metric, "mean": float(mean),
                     "ci_low": low, "ci_high": high, "bound": bound, "direction": direction,
                     "status": status, "n_families": len(a[cid][0]),
                     "interval_tail_probability": tail, "interval_confidence": 1-2*tail,
                     "geometry_family_alpha": protocol["alpha"]/2, "geometry_num_claims": claims,
                     "expected_tail_draws": options.bootstrap_samples*tail,
                     "eligible": valid_interval, "reason": reason})

    for col, cid in enumerate(ids):
        if cid in forget:
            append(cid, "full_target_anchor_precheck", "full_anchor_advantage", point_advantage[col],
                   draws_advantage[:, col], bounds.min_full_anchor_advantage, "greater", eligible[cid])
            append(cid, "forget_anchor", "delta_anchor", point_delta[col], draws_delta[:, col],
                   bounds.anchor_drop, "greater", eligible[cid])
            append(cid, "forget_anchor_selectivity", "selective_delta", point_selective[col],
                   draws_selective[:, col], bounds.anchor_selectivity_drop, "greater",
                   eligible[cid] and all(eligible[c] for c in retain))
        else:
            append(cid, "retain_anchor", "delta_anchor", point_delta[col], draws_delta[:, col],
                   bounds.retain_anchor_tolerance, "less", eligible[cid])
    precheck = combine_status(row["status"] for row in rows if row["condition"] == "full_target_anchor_precheck")
    raw = combine_status(row["status"] for row in rows if row["condition"] != "full_target_anchor_precheck")
    return rows, {"status": raw if precheck == "pass" else "insufficient",
                  "raw_status": raw, "full_anchor_precheck_status": precheck,
                  "num_claims": claims, "tail_probability": tail}


def diagnose_ablations(before, after, forget_ids, marker_config, audit_protocol,
                       v2_report, options=None, ablation_config=None):
    """Add transparent candidate comparisons without changing any v1/v2 fields."""
    options = options or V2DiagnosticConfig()
    validate_observations(before, after, forget_ids, marker_config, audit_protocol)
    markers = marker_config if isinstance(marker_config, MarkerConfig) else MarkerConfig(**marker_config)
    bounds = ablation_config
    if bounds is not None and not isinstance(bounds, V2AblationConfig):
        bounds = V2AblationConfig(**bounds)
    forget = set(forget_ids)
    classes, variance = _class_diagnostics(before, after, forget, audit_protocol, options)
    all_learned = all(row["full_marker_precheck"] == "pass" for row in v2_report["marker_metrics"])
    accuracy_rows, geometry_rows = [], []
    accuracy_status = {rule: {"status": "not_configured", "raw_status": "not_configured"}
                       for rule in ("equivalence", "noninferiority")}
    geometry = {"status": "not_configured", "raw_status": "not_configured",
                "full_anchor_precheck_status": "not_configured",
                "num_claims": 3*len(forget)+len(before)-len(forget),
                "tail_probability": None}
    if bounds is not None:
        accuracy_rows, accuracy_status = _accuracy_diagnostics(before, after, forget,
                                                              audit_protocol, bounds, all_learned)
        geometry_rows, geometry = _geometry_diagnostics(before, after, forget, markers,
                                                        audit_protocol, options, bounds)
    variants = []
    for method in ("marker_accuracy", "margin", "margin_anchor"):
        for rule in ("equivalence", "noninferiority"):
            margin = v2_report["margin_variants"][rule]
            configured = method == "margin" or bounds is not None
            if method == "marker_accuracy":
                state, raw = accuracy_status[rule]["status"], accuracy_status[rule]["raw_status"]
            elif method == "margin":
                state, raw = margin["status"], margin["raw_status"]
            elif not configured:
                state = raw = "not_configured"
            else:
                raw = combine_status([margin["raw_status"], geometry["raw_status"]])
                state = (combine_status([margin["status"], geometry["status"]])
                         if all_learned and geometry["full_anchor_precheck_status"] == "pass"
                         else "insufficient")
            variants.append({"method": method, "retained_rule": rule, "status": state,
                             "raw_status": raw, "configured": configured,
                             "full_marker_precheck_passed": all_learned,
                             "accuracy_status": accuracy_status[rule]["status"] if method == "marker_accuracy" else None,
                             "margin_status": margin["status"] if method != "marker_accuracy" else None,
                             "geometry_status": geometry["status"] if method == "margin_anchor" else None,
                             "full_anchor_precheck_status": geometry["full_anchor_precheck_status"] if method == "margin_anchor" else None,
                             "decision_basis": {"marker_accuracy": "empirical_point_threshold_no_inference",
                                                "margin": "family_margin_tests_Holm",
                                                "margin_anchor": "family_margin_tests_Holm_plus_joint_anchor_bootstrap"}[method],
                             "formal_audit": False, "task_included": False,
                             "interpretation": "exploratory_ablation_not_formal_acceptance"})
    return {"accuracy_tests": accuracy_rows, "geometry_tests": geometry_rows,
            "ablation_variants": variants, "client_class_metrics": classes,
            "class_variance": variance,
            "ablation_protocol": {
                "schema_version": "marker-protocol-v2-ablation-v1", "formal_audit": False,
                "configured": bounds is not None,
                "bounds": asdict(bounds) if bounds is not None else None,
                "bounds_source": "explicit_development_configuration_not_fitted_by_script" if bounds is not None else "unconfigured",
                "accuracy_inference": "none_empirical_point_threshold",
                "accuracy_correction": "not_applicable_no_hypothesis_test",
                "accuracy_min_families": audit_protocol["min_families"],
                "accuracy_estimand": "family_mean_equals_class_balanced_mean_when_family_counts_equal",
                "comparison_contains_mixed_decision_bases": True,
                "margin_rule": "unchanged_v2_source_alpha_and_bounds",
                "geometry_alpha": audit_protocol["alpha"]/2,
                "geometry_num_claims": geometry["num_claims"],
                "geometry_interval_tail_probability": geometry["tail_probability"],
                "geometry_interval_method": "joint_paired_class_stratified_family_bootstrap_Bonferroni_percentile",
                "geometry_coverage": "approximate_simultaneous_within_candidate_not_exact_finite_sample_guarantee",
                "geometry_retained_rule": "one_sided_maximum_anchor_deterioration_for_both_margin_variants",
                "geometry_precheck": "all_target_full_anchor_advantages_must_exceed_configured_bound",
                "geometry_min_expected_tail_draws": 10,
                "class_diagnostics": "descriptive_only_no_per_class_acceptance_gate",
                "class_interval_confidence": options.confidence,
                "class_interval_multiplicity_adjusted": False,
                "task_included_in_ablations": False,
                "cross_ablation_or_cross_run_multiplicity_control": False,
                "limitations": ["Development candidate bounds are not independently validated",
                                "Accuracy baseline pass/fail is empirical, not confidence-certified preservation",
                                "Zero observed accuracy variance does not establish zero population uncertainty",
                                "No geometry gate is inferred from descriptive 95% v2 intervals",
                                "Class mean differences may hide local losses; see client_class_metrics",
                                "Geometry uses retained anchor non-deterioration, not two-sided invariance",
                                "Family resampling does not quantify variation across training runs",
                                "Marker evidence is not proof of removal of real-data influence"],
            }}
