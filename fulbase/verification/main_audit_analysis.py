"""Main-experiment analysis on frozen, complete audit batches.

The statistical kernels and the joint marker rule are the existing v2 functions.
Only the model inventory and separate, explicit task-utility sensitivity grow.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
import math

import numpy as np

from fulbase.verification.markers import MarkerConfig
from fulbase.verification.protocol_v2 import V2DiagnosticConfig, diagnose_observations, validate_observations
from fulbase.verification.protocol_v2_ablation import V2AblationConfig, diagnose_ablations
from fulbase.verification.protocol_v2_stratified import diagnose_stratified_margins
from fulbase.verification.independent_marker_analysis import (
    _check_design, _pool, _batch_descriptions, _primary_summary,
)
from fulbase.verification.statistics import combine_status


def _settings(plan, marker_config, audit_protocol):
    from fulbase.verification.main_audit import validate_plan
    validate_plan(plan)
    sampling, analysis = plan["sampling"], plan["analysis"]
    markers = marker_config if isinstance(marker_config, MarkerConfig) else MarkerConfig(**marker_config)
    protocol = asdict(audit_protocol) if is_dataclass(audit_protocol) else dict(audit_protocol)
    if not markers.enabled or markers.families_per_class != 4 or markers.variants_per_family != 2:
        raise ValueError("Preserve original four-family/two-variant training marker configuration")
    for plan_key, source_key in (("per_request_alpha", "alpha"), ("margin_drop", "margin_drop"),
            ("retain_margin_tolerance", "retain_margin_tolerance"),
            ("min_full_accuracy", "min_full_accuracy"), ("min_full_margin", "min_full_margin"),
            ("max_task_retain_accuracy_drop", "max_retain_accuracy_drop")):
        if protocol.get(source_key) != analysis[plan_key]:
            raise ValueError(f"Frozen plan/source protocol mismatch: {plan_key}")
    if any(row["marker_fingerprint"] != markers.fingerprint() for row in plan["models"]):
        raise ValueError("Bound marker fingerprints differ from analysis configuration")
    options = V2DiagnosticConfig(bootstrap_samples=analysis["bootstrap_samples"],
        bootstrap_seed=analysis["bootstrap_seed"], confidence=analysis["descriptive_confidence"])
    bounds = V2AblationConfig(**{key: analysis[key] for key in (
        "accuracy_drop", "retain_accuracy_tolerance", "anchor_drop", "anchor_selectivity_drop",
        "retain_anchor_tolerance", "min_full_anchor_advantage")})
    keys = [f"{seed}_{method}" for seed in sampling["training_seeds"] for method in plan["expected_methods"]]
    return sampling, markers, protocol, options, bounds, keys


def validate_independent_batches(observations, plan, marker_config, audit_protocol):
    sampling, markers, protocol, _, _, keys = _settings(plan, marker_config, audit_protocol)
    if (not isinstance(observations, dict) or len(observations) != 3 or
            any(type(key) not in (str, int) or str(key) not in {"0", "1", "2"} for key in observations) or
            {int(key) for key in observations} != {0, 1, 2}):
        raise ValueError("All three frozen batches must be complete, without duplicate aliases")
    batches = {int(key): value for key, value in observations.items()}
    seen_ids, seen_families, seen_packets = set(), set(), set()
    for batch in range(3):
        models = batches[batch]
        if set(models) != set(keys):
            raise ValueError("Every frozen model must appear in every batch")
        reference = models[keys[0]]
        if set(reference) != set(sampling["clients"]):
            raise ValueError("Frozen client set differs from observation client set")
        batch_packets = None
        for cid in sampling["clients"]:
            packets = _check_design(reference[cid], cid, sampling)
            if batch_packets is None:
                batch_packets = packets
            elif packets != batch_packets:
                raise ValueError("All clients must share the same packet identities")
            ids = set(np.asarray(reference[cid]["ids"]).tolist())
            families = set(np.asarray(reference[cid]["families"]).tolist())
            if ids & seen_ids or families & seen_families:
                raise ValueError("Audit identities overlap between clients or batches")
            seen_ids.update(ids)
            seen_families.update(families)
        if batch_packets & seen_packets:
            raise ValueError("Audit packet identities overlap between batches")
        seen_packets.update(batch_packets)
        for key in keys:
            validate_observations(reference, models[key], sampling["forget_clients"], markers,
                                  protocol, expected_role="audit")
    return batches


def task_utility(task, marker_status, full_qualified=None):
    """Task thresholds never affect marker inference or its Full preconditions."""
    drop = (task or {}).get("retain_accuracy_drop")
    known = (type(drop) in (int, float) and math.isfinite(drop))
    result = {"source": "recorded_benchmark_not_retested", "retain_accuracy_drop": drop if known else None,
              "metrics_recomputed": False, "primary_bound": .035,
              "old_protocol_modified": False, "marker_full_qualified": full_qualified,
              "joint_failure_is_not_marker_detection": True}
    for name, bound in (("original_3pp", .03), ("proposed_3_5pp", .035)):
        status = "pass" if known and drop <= bound else "fail" if known else "insufficient"
        result[name] = {"bound": bound, "status": status,
                       "marker_and_task_status": combine_status([marker_status, status]),
                       "formal_audit": False}
    return result


def analyze_main_observations(observations, plan, marker_config, audit_protocol, task_metrics=None):
    sampling, markers, protocol, options, bounds, keys = _settings(plan, marker_config, audit_protocol)
    batches = validate_independent_batches(observations, plan, markers, protocol)
    pooled = {key: _pool(batches, key, sampling["clients"]) for key in keys}
    bindings = {f"{row['training_seed']}_{row['method']}": row for row in plan["models"]}
    requests, descriptions = [], []
    forget = set(sampling["forget_clients"])
    for seed in sampling["training_seeds"]:
        before = pooled[f"{seed}_Full"]
        for method in plan["expected_methods"][1:]:
            key = f"{seed}_{method}"
            after = pooled[key]
            v2 = diagnose_observations(before, after, forget, markers, protocol, options, expected_role="audit")
            ablation = diagnose_ablations(before, after, forget, markers, protocol, v2, options,
                                         bounds, expected_role="audit")
            stratified = diagnose_stratified_margins(before, after, forget, markers, protocol, v2,
                                                   options, expected_role="audit")
            primary = _primary_summary(stratified, ablation)
            task = deepcopy((task_metrics or {}).get(key))
            requests.append({"model_key": key, "training_seed": seed, "method": method,
                "cohort": bindings[key]["cohort"], "full_model_key": f"{seed}_Full",
                "observation_role": "audit", "primary": primary,
                "secondary_equivalence": _primary_summary(stratified, ablation, variant="equivalence"),
                "v2": v2, "ablations": ablation, "stratified_margin": stratified,
                "task_metrics": task, "task_metrics_recomputed": False,
                "task_utility": task_utility(task, primary["marker_status"],
                    primary["full_marker_precheck_passed"] and primary["full_anchor_precheck_status"] == "pass")})
            for batch in sorted(batches):
                descriptions.extend(dict(row, cohort=bindings[key]["cohort"], training_seed=seed, method=method)
                    for row in _batch_descriptions(batches[batch][f"{seed}_Full"],
                        batches[batch][key], markers, batch, key, forget))
    return {"schema_version": "rp2-main-independent-marker-analysis-v1",
        "status": "complete_conditional_marker_analysis", "formal_audit": False,
        "observation_role": "audit", "requests": requests, "per_batch_descriptive": descriptions,
        "protocol": {"primary_dataset": "all_three_frozen_batches_pooled",
            "primary_sampling_unit": "paired_family", "primary_retained_rule": "noninferiority",
            "families_per_class": 60, "per_batch_inference": False, "batch_voting_or_selection": False,
            "per_request_alpha": protocol["alpha"], "margin_alpha": protocol["alpha"]/2,
            "geometry_alpha": protocol["alpha"]/2, "margin_client_count": 5, "geometry_claim_count": 7,
            "options": asdict(options), "bounds": asdict(bounds), "marker_fingerprint": markers.fingerprint(),
            "cross_candidate_or_training_seed_global_error_control": False,
            "full_observations_reused": True, "task_included_in_marker_decision": False,
            "task_utility_bounds": [.03, .035], "task_primary_bound": .035,
            "legacy_algorithm_metadata_preserved": True},
        "limitations": [
            "Conditional on fixed checkpoints, fixed classes and the existing marker generator",
            "Development and confirmation cohorts are reported separately; batches are not training seeds",
            "No joint error guarantee across candidate models or training seeds",
            "Full qualification failure or insufficient evidence is not successful detection",
            "A joint marker/task failure caused only by utility is not a successful marker detection",
            "This collector binds model identities; optimization budgets are frozen and validated by the campaign",
            "Full and no-op may be identical parameter sets and are not independent training repetitions",
            "Candidate algorithms have no asserted forgetting ground truth; report their three-way outcomes",
            "Head manipulation controls describe an intervention, not proof that real-data influence remains",
            "Original 3pp task results are retained; 3.5pp is a separately declared utility version",
            "Task metrics are copied from source benchmarks, never inferred from marker observations",
            "Marker pass does not prove deletion of real-data influence or formal unlearning",
        ]}
