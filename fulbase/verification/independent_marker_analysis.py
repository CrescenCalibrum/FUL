"""Analyze all prespecified fresh audit batches on the nine fixed checkpoints.

This adapter reuses the v2 computations without relabelling audit observations
as calibration. It performs one pooled decision per candidate, and deliberately
does not perform hypothesis tests, bootstrap intervals or votes on single batches.
The caller owns model/input/source commitments and freshness validation.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass

import numpy as np

from fulbase.verification.markers import MarkerConfig, client_anchor
from fulbase.verification.protocol_v2 import (
    V2DiagnosticConfig, _normalize_centers, _reduce, diagnose_observations,
    validate_observations,
)
from fulbase.verification.protocol_v2_ablation import V2AblationConfig, diagnose_ablations
from fulbase.verification.protocol_v2_stratified import (
    _fixed_class_statistics, diagnose_stratified_margins,
)
from fulbase.verification.statistics import combine_status


_ARRAY_FIELDS = ("ids", "families", "labels", "predictions", "margin",
                 "normalized_margin", "features", "logits")
_METHODS = ("Full", "Retrain", "fixed_C_lambda05")


def _settings(plan, marker_config, audit_protocol):
    sampling, analysis = plan["sampling"], plan["analysis"]
    expected = {"role": "audit", "training_seeds": [43, 44, 45],
                "clients": [0, 1, 2, 3, 4], "forget_clients": [0],
                "num_classes": 10, "batches": 3, "packets_per_batch": 5,
                "families_per_class_per_packet": 4, "variants_per_family": 2,
                "families_per_class_per_batch": 20,
                "families_per_class_primary_pooled": 60,
                "unique_marker_inputs": 6000, "sample_model_evaluations": 54000}
    if any(sampling.get(key) != value for key, value in expected.items()):
        raise ValueError("Independent marker plan requires the fixed 3 x 20-family design")
    markers = marker_config if isinstance(marker_config, MarkerConfig) else MarkerConfig(**marker_config)
    if markers.families_per_class != 4 or markers.variants_per_family != 2:
        raise ValueError("Preserve the original four-family, two-variant marker configuration")
    protocol = asdict(audit_protocol) if is_dataclass(audit_protocol) else dict(audit_protocol)
    for plan_key, source_key in (("per_request_alpha", "alpha"), ("margin_drop", "margin_drop"),
                                 ("retain_margin_tolerance", "retain_margin_tolerance"),
                                 ("min_full_accuracy", "min_full_accuracy"),
                                 ("min_full_margin", "min_full_margin")):
        if analysis[plan_key] != protocol[source_key]:
            raise ValueError(f"Frozen plan/source protocol mismatch: {plan_key}")
    if ("max_retain_accuracy_drop" in protocol
            and analysis["max_task_retain_accuracy_drop"] != protocol["max_retain_accuracy_drop"]):
        raise ValueError("Frozen plan/source protocol mismatch: max_task_retain_accuracy_drop")
    if (analysis["margin_alpha"] != protocol["alpha"]/2
            or analysis["geometry_alpha"] != protocol["alpha"]/2
            or analysis["margin_client_count"] != 5 or analysis["geometry_claim_count"] != 7
            or analysis["primary_retained_rule"] != "noninferiority"
            or analysis["primary_structure"] != "stratified_margin_and_existing_anchor_conditions"):
        raise ValueError("Independent marker analysis must use the frozen NI/anchor allocation")
    options = V2DiagnosticConfig(bootstrap_samples=analysis["bootstrap_samples"],
                                 bootstrap_seed=analysis["bootstrap_seed"],
                                 confidence=analysis["descriptive_confidence"])
    bounds = V2AblationConfig(**{key: analysis[key] for key in (
        "accuracy_drop", "retain_accuracy_tolerance", "anchor_drop",
        "anchor_selectivity_drop", "retain_anchor_tolerance", "min_full_anchor_advantage")})
    keys = [f"{seed}_{method}" for seed in sampling["training_seeds"] for method in _METHODS]
    bindings = [(row["training_seed"], row["method"]) for row in plan["models"]]
    if len(bindings) != 9 or sorted(f"{seed}_{method}" for seed, method in bindings) != sorted(keys):
        raise ValueError("Expected exactly the nine fixed Full/Retrain/C model bindings")
    if any(row["marker_fingerprint"] != markers.fingerprint() for row in plan["models"]):
        raise ValueError("Original model marker fingerprints differ from the analysis configuration")
    return sampling, markers, protocol, options, bounds, keys


def _check_design(obs, cid, sampling):
    """Check original packet IDs; a role field alone cannot relabel old inputs."""
    labels = np.asarray(obs["labels"])
    families = np.asarray(obs["families"])
    if set(labels.tolist()) != set(range(sampling["num_classes"])):
        raise ValueError("Every independent batch must contain all ten fixed classes")
    packets = {}
    for family in set(families.tolist()):
        parts = family.split(":")
        if (len(parts) != 5 or parts[0] != str(cid) or parts[1] != "audit"
                or any(not value.isdecimal() for value in parts[2:])):
            raise ValueError("Expected original audit-role packet family identifiers")
        label, family_index = int(parts[3]), int(parts[4])
        if not 0 <= family_index < sampling["families_per_class_per_packet"]:
            raise ValueError("Marker family index differs from original packet configuration")
        mask = families == family
        if mask.sum() != sampling["variants_per_family"] or set(labels[mask]) != {label}:
            raise ValueError("Expected exactly two paired variants of each single-label family")
        expected_ids = {family+":"+str(i) for i in range(sampling["variants_per_family"])}
        if set(np.asarray(obs["ids"])[mask]) != expected_ids:
            raise ValueError("Marker sample identifiers do not match original family variants")
        packets.setdefault(parts[2], {}).setdefault(label, set()).add(family_index)
    if len(packets) != sampling["packets_per_batch"]:
        raise ValueError("Every batch must contain five independent original-size packets")
    for classes in packets.values():
        if (set(classes) != set(range(sampling["num_classes"]))
                or any(values != set(range(sampling["families_per_class_per_packet"]))
                       for values in classes.values())):
            raise ValueError("Each packet must have four families of every fixed class")
    return set(packets)


def validate_independent_batches(observations, plan, marker_config, audit_protocol):
    """Validate all batches before any result is calculated; return ordered batches."""
    sampling, markers, protocol, _, _, keys = _settings(plan, marker_config, audit_protocol)
    # JSON persistence may turn integer batch keys into strings; reject aliases.
    if (not isinstance(observations, dict)
            or any(isinstance(key, bool) or str(key) not in {"0", "1", "2"}
                   for key in observations)
            or len(observations) != 3 or {int(key) for key in observations} != {0, 1, 2}):
        raise ValueError("Incomplete independent collection: require all three frozen batches")
    batches = {int(key): value for key, value in observations.items()}
    seen_samples, seen_families, seen_packets = set(), set(), set()
    for batch in range(sampling["batches"]):
        models = batches[batch]
        if set(models) != set(keys):
            raise ValueError("Incomplete independent collection: require all nine models in every batch")
        reference = models[keys[0]]
        if set(reference) != set(sampling["clients"]):
            raise ValueError("Independent collection client set differs from frozen plan")
        batch_packets = None
        for cid in sampling["clients"]:
            packets = _check_design(reference[cid], cid, sampling)
            if batch_packets is None:
                batch_packets = packets
            elif packets != batch_packets:
                raise ValueError("Clients must share the same five packet identities in each batch")
            sample_ids = set(np.asarray(reference[cid]["ids"]).tolist())
            family_ids = set(np.asarray(reference[cid]["families"]).tolist())
            if seen_samples.intersection(sample_ids) or seen_families.intersection(family_ids):
                raise ValueError("Marker identities overlap across batches or clients")
            seen_samples.update(sample_ids)
            seen_families.update(family_ids)
        if seen_packets.intersection(batch_packets):
            raise ValueError("Independent batch packet identities overlap")
        seen_packets.update(batch_packets)
        for key in keys:
            validate_observations(reference, models[key], sampling["forget_clients"], markers,
                                  protocol, expected_role="audit")
    return batches


def _pool(batches, model_key, clients):
    result = {}
    for cid in clients:
        rows = [batches[batch][model_key][cid] for batch in sorted(batches)]
        result[cid] = {"cid": cid, "role": "audit", "valid": True,
                       **{key: np.concatenate([np.asarray(row[key]) for row in rows], axis=0)
                          for key in _ARRAY_FIELDS}}
    return result


def _batch_descriptions(before, after, markers, batch, model_key, forget):
    """Point estimates and analytic SE only: no interval or acceptance decision."""
    reduced_before = {cid: _reduce(obs) for cid, obs in before.items()}
    reduced_after = {cid: _reduce(obs) for cid, obs in after.items()}
    rows, deltas = [], {}
    ids = sorted(before)
    dim = reduced_before[ids[0]][2].shape[1]
    anchors = {cid: client_anchor(markers, cid, dim, "cpu").numpy().astype(np.float64)
               for cid in ids}
    units = {cid: value/np.linalg.norm(value) for cid, value in anchors.items()}
    for cid in ids:
        a, b = reduced_before[cid], reduced_after[cid]
        stats, _, _ = _fixed_class_statistics(a[0], a[1], b[1])
        classes = sorted(set(a[0]))
        full = _normalize_centers(np.mean([a[2][a[0] == label].mean(0) for label in classes], axis=0))
        post = _normalize_centers(np.mean([b[2][b[0] == label].mean(0) for label in classes], axis=0))
        afull, apost = float(np.sum((full-anchors[cid])**2)), float(np.sum((post-anchors[cid])**2))
        deltas[cid] = apost-afull
        other = np.mean([units[other_cid] for other_cid in ids if other_cid != cid], axis=0)
        rows.append({"batch": batch, "model_key": model_key, "cid": cid,
                     "client_group": "forget" if cid in forget else "retain",
                     "n_families": stats["n_families"], "n_classes": stats["n_classes"],
                     "families_per_class": stats["min_families_per_class"],
                     "full_accuracy": float(np.mean(before[cid]["predictions"] == before[cid]["labels"])),
                     "after_accuracy": float(np.mean(after[cid]["predictions"] == after[cid]["labels"])),
                     "full_mean": stats["full_mean"], "after_mean": stats["after_mean"],
                     "margin_change": stats["mean"], "analytic_se": stats["analytic_se"],
                     "A_full": afull, "A_after": apost, "delta_anchor": apost-afull,
                     "full_anchor_advantage": float(full @ (units[cid]-other)),
                     "status": "descriptive_only", "used_as_acceptance_gate": False})
    retained_mean = float(np.mean([value for cid, value in deltas.items() if cid not in forget]))
    for row in rows:
        row["selective_delta"] = row["delta_anchor"]-retained_mean if row["cid"] in forget else None
    return rows


def _primary_summary(stratified, ablation, *, variant="noninferiority"):
    margin = next(row for row in stratified["summary"] if row["variant"] == variant)
    geometry = next(row for row in ablation["ablation_variants"]
                    if row["method"] == "margin_anchor" and row["retained_rule"] == variant)
    conditions = [row["raw_status"] for row in stratified["client_tests"] if row["variant"] == variant]
    conditions += [row["status"] for row in ablation["geometry_tests"]
                   if row["condition"] != "full_target_anchor_precheck"]
    raw = combine_status(conditions)
    prechecks = margin["full_marker_precheck_passed"] and geometry["full_anchor_precheck_status"] == "pass"
    return {"retained_rule": variant, "marker_status": raw if prechecks else "insufficient",
            "raw_condition_status": raw, "full_marker_precheck_passed": margin["full_marker_precheck_passed"],
            "full_anchor_precheck_status": geometry["full_anchor_precheck_status"],
            "stratified_margin_status": margin["stratified_status"],
            "geometry_status": geometry["geometry_status"], "task_included": False,
            "formal_audit": False, "overall_unlearning_acceptance": None}


def analyze_independent_observations(observations, plan, marker_config, audit_protocol,
                                     task_metrics=None):
    """Return JSON-safe pooled evidence after complete fixed-batch validation.

    ``observations[batch][f'{seed}_{method}'][cid]`` contains normal observe_markers
    dictionaries. Batch keys are 0, 1, 2; methods are Full, Retrain, fixed_C_lambda05.
    ``task_metrics`` contains source-bound original benchmark metadata per model
    key; it is copied separately and never inferred from marker observations.
    """
    sampling, markers, protocol, options, bounds, keys = _settings(plan, marker_config, audit_protocol)
    batches = validate_independent_batches(observations, plan, markers, protocol)
    pooled = {key: _pool(batches, key, sampling["clients"]) for key in keys}
    requests, descriptions = [], []
    forget = set(sampling["forget_clients"])
    for seed in sampling["training_seeds"]:
        before = pooled[f"{seed}_Full"]
        for method in _METHODS[1:]:
            key = f"{seed}_{method}"
            after = pooled[key]
            v2 = diagnose_observations(before, after, forget, markers, protocol, options,
                                      expected_role="audit")
            ablation = diagnose_ablations(before, after, forget, markers, protocol, v2, options,
                                          bounds, expected_role="audit")
            stratified = diagnose_stratified_margins(before, after, forget, markers, protocol,
                                                    v2, options, expected_role="audit")
            requests.append({"model_key": key, "training_seed": seed, "method": method,
                             "full_model_key": f"{seed}_Full", "observation_role": "audit",
                             "primary": _primary_summary(stratified, ablation),
                             "secondary_equivalence": _primary_summary(stratified, ablation, variant="equivalence"),
                             "v2": v2, "ablations": ablation, "stratified_margin": stratified,
                             "task_metrics": deepcopy((task_metrics or {}).get(key)),
                             "task_metrics_recomputed": False})
            for batch in sorted(batches):
                descriptions.extend(_batch_descriptions(batches[batch][f"{seed}_Full"],
                    batches[batch][key], markers, batch, key, forget))
    return {"schema_version": "independent-marker-validation-v1",
            "status": "complete_conditional_marker_analysis", "formal_audit": False,
            "observation_role": "audit", "requests": requests,
            "per_batch_descriptive": descriptions,
            "protocol": {"primary_dataset": "all_three_frozen_batches_pooled",
                         "primary_sampling_unit": "paired_family",
                         "primary_retained_rule": "noninferiority", "families_per_class": 60,
                         "per_batch_inference": False, "batch_voting_or_selection": False,
                         "per_request_alpha": protocol["alpha"], "margin_alpha": protocol["alpha"]/2,
                         "geometry_alpha": protocol["alpha"]/2, "margin_client_count": 5,
                         "geometry_claim_count": 7, "options": asdict(options),
                         "bounds": asdict(bounds), "marker_fingerprint": markers.fingerprint(),
                         "cross_candidate_or_training_seed_global_error_control": False,
                         "full_observations_reused": True, "task_included_in_marker_decision": False,
                         "legacy_algorithm_metadata_preserved": True},
            "limitations": [
                "Conditional validation on fixed checkpoints, fixed classes and the existing marker generator",
                "Model/input freshness and freeze commitments must also be verified by the collection runner",
                "Reused v2 subreports retain historical development metadata; these inputs have audit role",
                "Approximate percentile coverage is not an exact finite-sample or cross-training-seed guarantee",
                "Six candidate requests do not have a joint 95% familywise error guarantee",
                "Marker pass is not proof of removal of real-data influence or overall unlearning acceptance",
                "Benchmark utility is copied separately from the original bound source, not measured on markers",
                "Incomplete or insufficient evidence is not detection success",
            ]}
