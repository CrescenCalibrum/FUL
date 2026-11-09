"""Fresh fixed-budget marker validation for the main experiment.

This separate adapter retains the existing audit mathematics. Lifecycle code
follows the validated nine-model collector, with a dynamic frozen inventory.
The legacy collector and its frozen artifacts are never modified.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import io
import json
import os
from pathlib import Path
import re
import secrets
import numpy as np
import torch

from fulbase.models import build_model
from fulbase.utils import get_model_parameters, set_model_parameters
from fulbase.verification.audit import AuditConfig, file_hash, observe_markers, parameter_hash, save_observations
from fulbase.verification.markers import MarkerConfig, MarkerDataset, derived_seed
from fulbase.verification.protocol_v2_io import _task_diagnostic
from fulbase.verification.independent_marker import (
    _json, _digest, _write_json, _inside, _root, _model_key, _read_npz,
    _dataset_rows, _load_parameters, _runtime, _MergedMarkers, _validate_shard,
    _read_shard, _assert_sources_unchanged, _publish_analysis_file,
    _analysis_tables as _legacy_analysis_tables,
    CODE_FILES as LEGACY_CODE_FILES,
)

SCHEMA = "rp2-main-independent-marker-collection-v1"
PLAN_SCHEMA = "rp2-main-independent-marker-plan-v1"
C_STAGE = "marker_capretain_head_s30_lr0.01_r0.1_d3.0_fixed_w0.5_pre_recovery"
CODE_FILES = tuple(dict.fromkeys((*LEGACY_CODE_FILES,
    "fulbase/verification/main_audit.py", "fulbase/verification/main_audit_analysis.py",
    "scripts/collect_main_experiment_markers.py")))


def _analysis_spec():
    return {
        "primary_sampling_unit": "paired_family", "primary_estimand": "equal_mean_over_fixed_10_classes",
        "primary_dataset": "pool_all_three_prespecified_batches_before_decision",
        "primary_retained_rule": "noninferiority",
        "primary_structure": "stratified_margin_and_existing_anchor_conditions",
        "per_request_alpha": .05, "margin_alpha": .025, "geometry_alpha": .025,
        "margin_client_count": 5, "geometry_claim_count": 7, "bootstrap_samples": 20000,
        "bootstrap_seed": 2718, "descriptive_confidence": .95,
        "margin_drop": .1, "retain_margin_tolerance": .1, "anchor_drop": .01,
        "anchor_selectivity_drop": .01, "retain_anchor_tolerance": .02,
        "min_full_anchor_advantage": 0., "min_full_accuracy": .7, "min_full_margin": .01,
        "accuracy_drop": .2, "retain_accuracy_tolerance": .05,
        "max_task_retain_accuracy_drop": .03, "task_utility_bounds": [.03, .035],
        "task_primary_bound": .035, "formal_unlearning_claim": False,
        "cross_candidate_or_training_seed_global_error_control": False,
    }


def make_plan(models, expected_methods, training_seeds=None, history_run_dirs=(), exclude_collections=()):
    """Build a hash-bound plan after the predeclared model matrix has completed.

    Selecting the matrix or its budgets using audit results is not supported.
    Full and a no-op are separate bindings and may have identical parameters.
    """
    models = deepcopy(list(models))
    seeds = sorted({row["training_seed"] for row in models}) if training_seeds is None else list(training_seeds)
    result = {
        "schema_version": PLAN_SCHEMA,
        "purpose": "fresh_marker_confirmation_on_prespecified_main_experiment_models",
        "expected_methods": list(expected_methods), "models": models,
        "exclude_collections": list(exclude_collections),
        "sampling": {
            "role": "audit", "training_seeds": seeds, "clients": [0, 1, 2, 3, 4],
            "forget_clients": [0], "num_classes": 10, "batches": 3, "packets_per_batch": 5,
            "families_per_class_per_packet": 4, "variants_per_family": 2,
            "families_per_class_per_batch": 20, "families_per_class_primary_pooled": 60,
            "unique_marker_inputs": 6000, "sample_model_evaluations": 6000 * len(models),
            "preserve_training_marker_config_and_fingerprint": True,
            "same_inputs_all_models": True, "full_observations_reused": True,
            "history_run_dirs": list(history_run_dirs),
        },
        "analysis": _analysis_spec(),
        "execution": {
            "collect_all_batches_before_analysis": True, "candidate_optimization_allowed": False,
            "optional_stopping_or_sample_extension": False, "legacy_reports_overwritten": False,
            "formal_audit": False,
        },
    }
    return validate_plan(result)


def validate_plan(plan):
    if not isinstance(plan, dict) or plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("Unsupported main-experiment independent marker plan")
    methods = plan.get("expected_methods")
    if (not isinstance(methods, list) or len(methods) < 2 or methods[0] != "Full" or
            any(not isinstance(method, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", method) is None for method in methods) or
            len(set(methods)) != len(methods)):
        raise ValueError("Declare distinct method names with Full first and at least one candidate")
    sampling = plan.get("sampling", {})
    seeds = sampling.get("training_seeds")
    if (not isinstance(seeds, list) or not seeds or any(type(seed) is not int or seed < 0 for seed in seeds) or
            len(set(seeds)) != len(seeds)):
        raise ValueError("Declare distinct nonnegative integer training seeds")
    models = plan.get("models")
    if not isinstance(models, list) or any(not isinstance(row, dict) for row in models):
        raise ValueError("Model bindings must be a list of objects")
    keys = [(row.get("training_seed"), row.get("method")) for row in models]
    if (len(keys) != len(seeds) * len(methods) or len(set(keys)) != len(keys) or
            set(keys) != {(seed, method) for seed in seeds for method in methods}):
        raise ValueError("Exactly the declared complete seed/method matrix is required")
    for row in models:
        if type(row.get("training_seed")) is not int or row.get("cohort") not in {"development", "confirmation"}:
            raise ValueError("Each binding needs its declared training seed and development/confirmation cohort")
        for name in ("run_dir", "checkpoint", "stage"):
            if not isinstance(row.get(name), str) or not row[name]:
                raise ValueError(f"Missing model binding: {name}")
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", row["stage"]) is None:
            raise ValueError("Invalid bound stage name")
        if row["method"] == "Full" and row["stage"] != "full_after_fl":
            raise ValueError("Full must bind full_after_fl")
        for name in ("checkpoint_sha256_recorded", "parameters_sha256_recorded", "marker_fingerprint", "partition_sha256"):
            if re.fullmatch(r"[0-9a-f]{64}", str(row.get(name, ""))) is None:
                raise ValueError(f"Invalid model binding digest: {name}")
    if len({row["marker_fingerprint"] for row in models}) != 1:
        raise ValueError("All models must retain one original marker fingerprint")
    for seed in seeds:
        rows = [row for row in models if row["training_seed"] == seed]
        if len({row["partition_sha256"] for row in rows}) != 1 or len({row["cohort"] for row in rows}) != 1:
            raise ValueError("Same-seed models must share partition and cohort")
        full = next(row for row in rows if row["method"] == "Full")
        for row in rows:
            if row["method"] == "fake_noop" and row["parameters_sha256_recorded"] != full["parameters_sha256_recorded"]:
                raise ValueError("No-op must retain the Full parameter identity")
    fixed_sampling = {
        "role": "audit", "clients": [0, 1, 2, 3, 4], "forget_clients": [0], "num_classes": 10,
        "batches": 3, "packets_per_batch": 5, "families_per_class_per_packet": 4,
        "variants_per_family": 2, "families_per_class_per_batch": 20,
        "families_per_class_primary_pooled": 60, "unique_marker_inputs": 6000,
        "sample_model_evaluations": 6000 * len(models),
        "preserve_training_marker_config_and_fingerprint": True,
        "same_inputs_all_models": True, "full_observations_reused": True,
    }
    for key, expected in fixed_sampling.items():
        if _digest(sampling.get(key)) != _digest(expected):
            raise ValueError(f"Frozen 3 x 20 sampling design changed: {key}")
    for key, expected in _analysis_spec().items():
        if _digest(plan.get("analysis", {}).get(key)) != _digest(expected):
            raise ValueError(f"Frozen analysis value changed: {key}")
    for values in (sampling.get("history_run_dirs", []), plan.get("exclude_collections", [])):
        if not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values) or len(set(values)) != len(values):
            raise ValueError("Historical source and collection directories must be distinct nonempty strings")
    execution = {"collect_all_batches_before_analysis": True, "candidate_optimization_allowed": False,
                 "optional_stopping_or_sample_extension": False, "legacy_reports_overwritten": False,
                 "formal_audit": False}
    if plan.get("execution") != execution:
        raise ValueError("Frozen no-selection/no-optional-stopping execution contract changed")
    return plan


def _source_registry(root, run_dirs, models, exclude_collections):
    """Read public identities and source metadata; never read old audit outcomes.

    Training/calibration images are regenerated only to bind their identities and
    content. Historical private replay stores and NPZ observations are excluded.
    """
    files, ids, families, contents, coverage = {}, set(), set(), set(), []
    for relative in sorted(set(run_dirs)):
        run = _inside(root, relative)
        cfg = _json(run / "config.json")
        markers = MarkerConfig(**_json(run / "verification/marker_config.json"))
        nclasses = _json(run / "verification/checkpoint_manifest.json")["num_classes"]
        required = [run / "config.json", run / "data_partition.json", run / "verification/marker_config.json",
                    run / "verification/audit_config.json", run / "verification/checkpoint_manifest.json",
                    run / "verification/training_marker_manifest.json"]
        manifests = sorted(path for path in (run / "verification").rglob("*marker_manifest.json")
                           if ".private" not in path.relative_to(run).parts)
        evidence = [path for path in (run / "run_status.json", run / "diagnostic_config.json",
                    run / "verification/training_result.json", run / "main_candidate_config.json",
                    run / "main_candidate_provenance.json", run / "candidate_inventory.json") if path.is_file()]
        # Only original task conditions of bound stages, not independent audit outcomes.
        reports = [run / "verification" / row["stage"] / "audit_report.json"
                   for row in models if row["run_dir"] == relative and row["method"] != "Full"]
        evidence.extend(path for row in models if row["run_dir"] == relative
                        for path in [run / "verification" / row["stage"] / "method_logs.json"]
                        if path.is_file())
        for path in set(required + manifests + evidence + reports):
            if not path.is_file():
                raise FileNotFoundError(f"Missing registered source metadata: {path}")
            files[path.relative_to(root).as_posix()] = file_hash(path)
        for path in manifests:
            rows = _json(path)
            if not isinstance(rows, list):
                raise ValueError("Historical marker manifest must be a list")
            for row in rows:
                ids.add(row["marker_id"])
                families.add(row["family_id"])
                if row.get("tensor_sha256"):
                    contents.add(row["tensor_sha256"])
        for role, seed in (("train", derived_seed(cfg["seed"], "train-markers")),
                           ("calibration", cfg["calibration_seed"])):
            generated = []
            for cid in range(cfg["num_clients"]):
                ds = MarkerDataset(markers, cid, cfg["dataset"], role, seed, nclasses)
                rows = _dataset_rows(ds)
                generated.extend(ds.manifest())
                ids.update(row["marker_id"] for row in rows)
                families.update(row["family_id"] for row in rows)
                contents.update(row["tensor_sha256"] for row in rows)
            stored = run / "verification" / ("training_marker_manifest.json" if role == "train" else "calibration_marker_manifest.json")
            if stored.is_file():
                fields = ("cid", "role", "marker_id", "family_id", "label")
                original = sorted(tuple(row[field] for field in fields) for row in _json(stored))
                expected = sorted(tuple(row[field] for field in fields) for row in generated)
                if original != expected:
                    raise ValueError("Historical marker manifest differs from original generator identities")
        coverage.append({"run_dir": relative, "content_regenerated_roles": ["train", "calibration"],
                         "historical_outcomes_read": False, "private_replay_read": False,
                         "old_audit_content_coverage": "public_marker_manifests_and_available_tensor_hashes_only"})
    for relative in exclude_collections:
        collection = _inside(root, relative)
        frozen_path, inputs_path = collection / "frozen_manifest.json", collection / "inputs.json"
        frozen = _json(frozen_path)
        if frozen.get("status") != "frozen_before_inference" or frozen.get("inputs_sha256") != file_hash(inputs_path):
            raise ValueError("Excluded collection has no valid frozen public input commitment")
        rows = _json(inputs_path)
        if not isinstance(rows, list) or not rows:
            raise ValueError("Excluded public input inventory is empty")
        for row in rows:
            if (not isinstance(row.get("marker_id"), str) or not isinstance(row.get("family_id"), str) or
                    re.fullmatch(r"[0-9a-f]{64}", str(row.get("tensor_sha256", ""))) is None):
                raise ValueError("Excluded collection lacks public identity/tensor binding")
            ids.add(row["marker_id"])
            families.add(row["family_id"])
            contents.add(row["tensor_sha256"])
        for path in (frozen_path, inputs_path):
            files[path.relative_to(root).as_posix()] = file_hash(path)
        coverage.append({"excluded_collection": relative, "registered_public_inputs": len(rows),
                         "historical_outcomes_read": False, "private_replay_read": False})
    return {"source_hashes": files, "ids": ids, "families": families,
            "tensor_hashes": contents, "coverage": coverage}


def _analysis_tables(report):
    tables = _legacy_analysis_tables(report)
    requests = {row["model_key"]: row for row in report["requests"]}
    for name, rows in tables.items():
        for row in rows:
            request = requests[row["model_key"]]
            row["cohort"] = request["cohort"]
            if name == "summary":
                utility = request["task_utility"]
                row["full_qualified"] = utility["marker_full_qualified"]
                row["joint_failure_is_not_marker_detection"] = True
                for label in ("original_3pp", "proposed_3_5pp"):
                    row[f"task_{label}_status"] = utility[label]["status"]
                    row[f"marker_and_task_{label}_status"] = utility[label]["marker_and_task_status"]
                row["task_primary_bound"] = utility["primary_bound"]
    tables["cohort_method_summary"] = []
    groups = sorted({(row["cohort"], row["method"]) for row in report["requests"]})
    for cohort, method in groups:
        group = [row for row in report["requests"] if row["cohort"] == cohort and row["method"] == method]
        values = [row["task_utility"]["retain_accuracy_drop"] for row in group
                  if row["task_utility"]["retain_accuracy_drop"] is not None]
        row = {"cohort": cohort, "method": method, "training_seed_count": len(group),
               "training_seeds": ",".join(str(value["training_seed"]) for value in group),
               "summary_kind": "descriptive_across_training_seeds_not_family_inference",
               "task_retain_drop_mean": float(np.mean(values)) if values else None,
               "task_retain_drop_sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
               "full_qualified_count": sum(value["task_utility"]["marker_full_qualified"] is True for value in group),
               "joint_failure_is_not_marker_detection": True,
               "formal_audit": False}
        for status in ("pass", "fail", "insufficient"):
            row[f"marker_{status}_count"] = sum(value["primary"]["marker_status"] == status for value in group)
            for label in ("original_3pp", "proposed_3_5pp"):
                row[f"task_{label}_{status}_count"] = sum(value["task_utility"][label]["status"] == status for value in group)
                row[f"marker_and_task_{label}_{status}_count"] = sum(
                    value["task_utility"][label]["marker_and_task_status"] == status for value in group)
        tables["cohort_method_summary"].append(row)
    return tables

def preflight(plan_path, project_root=None, verify_models=True):
    """Validate metadata and optionally every checkpoint; no outputs or inference."""
    root, plan_path = _root(project_root), Path(plan_path).resolve()
    plan = validate_plan(_json(plan_path))
    models, contexts, tasks = plan["models"], {}, {}
    dataset, marker_values, protocol_values = None, None, None
    for binding in models:
        run = _inside(root, binding["run_dir"])
        cfg = _json(run / "config.json")
        markers = MarkerConfig(**_json(run / "verification/marker_config.json"))
        protocol = asdict(AuditConfig(**_json(run / "verification/audit_config.json")))
        manifest = _json(run / "verification/checkpoint_manifest.json")
        entry = manifest["stages"][binding["stage"]]
        if "evaluation_status" in entry and entry["evaluation_status"] != "evaluated":
            raise ValueError(
                f"Cannot collect independent markers for {_model_key(binding)}: "
                f"evaluation_status={entry['evaluation_status']!r}. "
                "The complete matrix with models unavailable for inference is not supported yet; "
                "retain every declared method and its failure record rather than removing methods")
        if (cfg["seed"] != binding["training_seed"] or cfg["num_clients"] != 5 or
                str(cfg["forget_clients"]) != "0" or manifest["num_classes"] != 10):
            raise ValueError("Model source seed/client/class/forget configuration differs")
        if not markers.enabled or markers.families_per_class != 4 or markers.variants_per_family != 2:
            raise ValueError("Original training marker configuration must remain four families/two variants")
        if markers.fingerprint() != binding["marker_fingerprint"] or manifest["marker_fingerprint"] != markers.fingerprint():
            raise ValueError("Original training marker fingerprint differs")
        if file_hash(run / "data_partition.json") != binding["partition_sha256"] or manifest["partition_sha256"] != binding["partition_sha256"]:
            raise ValueError("Partition differs from the fixed model binding")
        for source, target in (("checkpoint_sha256", "checkpoint_sha256_recorded"),
                               ("parameters_sha256", "parameters_sha256_recorded")):
            if entry.get(source) != binding[target]:
                raise ValueError(f"Manifest model binding differs: {_model_key(binding)} {source}")
        if (entry.get("checkpoint", "").replace("\\", "/") != binding["checkpoint"].replace("\\", "/") or
                entry.get("stage") != binding["stage"] or entry.get("marker_fingerprint") != markers.fingerprint()):
            raise ValueError("Stage checkpoint/fingerprint differs from plan")
        seed_full = next(row for row in models if row["training_seed"] == binding["training_seed"] and row["method"] == "Full")
        source_full = manifest["stages"]["full_after_fl"]
        if (source_full["parameters_sha256"] != seed_full["parameters_sha256_recorded"] or
                source_full["checkpoint_sha256"] != seed_full["checkpoint_sha256_recorded"]):
            raise ValueError("Candidate source Full differs from the shared fixed Full")
        status_path = run / "run_status.json"
        if not status_path.is_file():
            raise FileNotFoundError(f"Source completion record missing: {status_path}")
        if status_path.is_file():
            status = _json(status_path)
            if status.get("status") == "main_candidate_evaluation_recovery_complete":
                raise ValueError(
                    "Independent collection does not yet support recovered candidate matrices with "
                    "models unavailable for inference; retain every declared method and its failure "
                    "record rather than removing methods")
            if status.get("status") not in {"development_run_complete", "pga_diagnostic_complete", "main_candidates_complete"}:
                raise ValueError("Source run is not complete")
            if status.get("source_full_parameters_sha256", seed_full["parameters_sha256_recorded"]) != seed_full["parameters_sha256_recorded"]:
                raise ValueError("Source completion record binds a different Full")
        training_path = run / "verification/training_result.json"
        if training_path.is_file() and _json(training_path).get("final_parameters_sha256") != seed_full["parameters_sha256_recorded"]:
            raise ValueError("Source training result binds a different Full")
        diagnostic_path = run / "diagnostic_config.json"
        if binding["method"] == "fixed_C_lambda05" and diagnostic_path.is_file():
            diagnostic = _json(diagnostic_path)
            trial = {"steps": 30, "lr": .01, "project_ratio": .1, "layers": "head",
                     "data_scope": "markers_only", "objective": "margin_capped_retain",
                     "margin_raw_drop": 3., "retain_weight": .5, "retain_weight_mode": "fixed"}
            if diagnostic.get("trials") != [trial] or diagnostic.get("source_full_parameters_sha256") != seed_full["parameters_sha256_recorded"]:
                raise ValueError("Fixed C source diagnostic budget differs from the prespecified trial")
        if dataset is not None and (cfg["dataset"] != dataset or asdict(markers) != marker_values or protocol != protocol_values):
            raise ValueError("All fixed models must share dataset, marker config, and original audit protocol")
        dataset, marker_values, protocol_values = cfg["dataset"], asdict(markers), protocol
        contexts[_model_key(binding)] = {"cfg": cfg, "manifest": manifest}
        if binding["method"] != "Full":
            report = _json(run / "verification" / binding["stage"] / "audit_report.json")
            tasks[_model_key(binding)] = _task_diagnostic(source_full, entry, report, protocol)
    for key, value in (("alpha", "per_request_alpha"), ("margin_drop", "margin_drop"),
                       ("retain_margin_tolerance", "retain_margin_tolerance"),
                       ("min_full_accuracy", "min_full_accuracy"), ("min_full_margin", "min_full_margin"),
                       ("max_retain_accuracy_drop", "max_task_retain_accuracy_drop")):
        if protocol_values[key] != plan["analysis"][value]:
            raise ValueError(f"Source threshold differs from frozen plan: {key}")
    history = sorted({row["run_dir"] for row in models} | set(plan["sampling"].get("history_run_dirs", [])))
    registry = _source_registry(root, history, models, plan.get("exclude_collections", []))
    context = {"root": root, "plan_path": plan_path, "plan": plan, "dataset": dataset,
               "marker_config": marker_values, "audit_protocol": protocol_values,
               "registry": registry, "models": contexts, "task_metrics": tasks,
               "checkpoint_bytes_reverified": False}
    if verify_models:
        missing = [_model_key(row) for row in models
                   if not _inside(_inside(root, row["run_dir"]), row["checkpoint"]).is_file()]
        if missing:
            raise FileNotFoundError("Fixed checkpoint files missing: " + ", ".join(missing))
        for binding in models:
            parameters = _load_parameters(context, binding)
            model = build_model(dataset, 10)
            set_model_parameters(model, parameters)
            if parameter_hash(get_model_parameters(model)) != binding["parameters_sha256_recorded"]:
                raise ValueError("Checkpoint does not round-trip through the bound model architecture")
            del model, parameters
        context["checkpoint_bytes_reverified"] = True
    return context



def _code_hashes():
    root = _root()
    return {name: file_hash(root / name) for name in CODE_FILES}



def _output_path(output_dir, context):
    output = Path(output_dir).resolve()
    run_dirs = {row["run_dir"] for row in context["plan"]["models"]}
    run_dirs.update(context["plan"]["sampling"].get("history_run_dirs", []))
    run_dirs.update(context["plan"].get("exclude_collections", []))
    for run_dir in run_dirs:
        source = _inside(context["root"], run_dir)
        if output == source or source in output.parents or output in source.parents:
            raise ValueError("Independent output must be outside every source run")
    return output



def _packet_commitment(batch, packet, seed):
    return _digest({"purpose": SCHEMA, "batch": batch, "packet": packet, "private_sample_seed": seed})



def _freshness(context):
    return {"status": "disjoint_from_registered_identities_and_available_tensor_hashes",
            "global_historical_isolation_claim": False,
            "registered_sample_ids": len(context["registry"]["ids"]),
            "registered_family_ids": len(context["registry"]["families"]),
            "registered_tensor_hashes": len(context["registry"]["tensor_hashes"]),
            "coverage": context["registry"]["coverage"]}



def _counts(context):
    return {"batches": 3, "packets": 15, "unique_inputs": 6000,
            "unique_families": 3000, "model_evaluations": 6000 * len(context["plan"]["models"]),
            "model_bindings": len(context["plan"]["models"])}



def _generate(context, seeds):
    markers = MarkerConfig(**context["marker_config"])
    datasets, rows, packets = {}, [], []
    for batch in range(3):
        datasets[batch] = {cid: [] for cid in range(5)}
        for packet in range(5):
            seed = seeds[batch * 5 + packet]
            packets.append({"batch": batch, "packet": packet,
                            "seed_commitment": _packet_commitment(batch, packet, seed)})
            for cid in range(5):
                ds = MarkerDataset(markers, cid, context["dataset"], "audit", seed, 10)
                datasets[batch][cid].append(ds)
                rows.extend(dict(row, batch=batch, packet=packet) for row in _dataset_rows(ds))
    ids, families, contents = [row["marker_id"] for row in rows], {row["family_id"] for row in rows}, [row["tensor_sha256"] for row in rows]
    if len(ids) != 6000 or len(set(ids)) != 6000 or len(families) != 3000 or len(set(contents)) != 6000:
        raise ValueError("Duplicate new packet identity, family token, or marker tensor")
    registry = context["registry"]
    if set(ids) & registry["ids"] or families & registry["families"] or set(contents) & registry["tensor_hashes"]:
        raise ValueError("Fresh marker overlaps a registered historical identity or tensor")
    return datasets, rows, packets



def prepare_collection(plan_path, output_dir, project_root=None):
    context = preflight(plan_path, project_root, verify_models=True)
    output = _output_path(output_dir, context)
    if (output / "frozen_manifest.json").exists():
        return _load_frozen(context, output)[0]
    intent = {"plan_sha256": file_hash(context["plan_path"]), "code_hashes": _code_hashes(),
              "source_hashes": context["registry"]["source_hashes"], "runtime": _runtime()}
    replay_path = output / ".private/replay.json"
    if output.exists() and any(output.iterdir()):
        allowed = {".private", "plan.json", "inputs.json", "plan.json.pending", "inputs.json.pending",
                   "frozen_manifest.json.pending"}
        if {path.name for path in output.iterdir()} - allowed or not replay_path.is_file():
            raise FileExistsError("Unrecognized partial preparation; no new seeds will replace its inputs")
        replay = _json(replay_path)
        if replay.get("preparation_context") != intent:
            raise ValueError("Partial preparation source/code/protocol changed; original inputs cannot be substituted")
        seeds = replay["packet_seeds"]
    else:
        seeds = [secrets.randbits(128) for _ in range(15)]
    if len(seeds) != 15 or len(set(seeds)) != 15:
        raise ValueError("Invalid private packet seeds; preparation aborted")
    _, rows, packets = _generate(context, seeds)
    output.mkdir(parents=True, exist_ok=True)
    private = output / ".private"
    private.mkdir(mode=0o700, exist_ok=True)
    if os.name != "nt":
        private.chmod(0o700)
    if not replay_path.exists():
        _write_json(replay_path, {"schema_version": SCHEMA, "preparation_context": intent,
                                 "packet_seeds": seeds}, private=True)
    for path, value in ((output / "plan.json", context["plan"]), (output / "inputs.json", rows)):
        if path.exists():
            if _json(path) != value:
                raise ValueError("Partial preparation differs from its identical-input replay")
        else:
            _write_json(path, value)
    _assert_sources_unchanged(context)
    if _code_hashes() != intent["code_hashes"]:
        raise ValueError("Code changed during preparation")
    manifest = {
        "schema_version": SCHEMA, "status": "frozen_before_inference", "formal_audit": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": file_hash(context["plan_path"]), "plan_copy_sha256": file_hash(output / "plan.json"),
        "marker_fingerprint": MarkerConfig(**context["marker_config"]).fingerprint(),
        "marker_config": context["marker_config"], "audit_protocol": context["audit_protocol"],
        "dataset": context["dataset"], "models": context["plan"]["models"],
        "checkpoint_bytes_and_parameters_reverified": True,
        "source_hashes": context["registry"]["source_hashes"], "code_hashes": _code_hashes(),
        "runtime": _runtime(), "inputs_sha256": file_hash(output / "inputs.json"),
        "private_replay_sha256": file_hash(private / "replay.json"), "packets": packets,
        "counts": _counts(context), "freshness": _freshness(context),
        "task_metrics": context["task_metrics"],
        "limitations": ["Historical audit tensor content cannot be reconstructed without its private replay seed.",
                        "Validation is conditional on fixed checkpoints and the original marker generator.",
                        "Private replay store must remain auditor-owned and must not be sent to training."]}
    _write_json(output / "frozen_manifest.json", manifest)
    return manifest



def _load_frozen(context, output):
    manifest = _json(output / "frozen_manifest.json")
    if manifest.get("schema_version") != SCHEMA or manifest.get("status") != "frozen_before_inference":
        raise ValueError("Unsupported or incomplete frozen collection")
    checks = {"plan_sha256": file_hash(context["plan_path"]),
              "plan_copy_sha256": file_hash(output / "plan.json"),
              "inputs_sha256": file_hash(output / "inputs.json"),
              "private_replay_sha256": file_hash(output / ".private/replay.json"),
              "source_hashes": context["registry"]["source_hashes"], "code_hashes": _code_hashes(),
              "runtime": _runtime(), "marker_config": context["marker_config"],
              "audit_protocol": context["audit_protocol"], "models": context["plan"]["models"],
              "task_metrics": context["task_metrics"], "dataset": context["dataset"],
              "marker_fingerprint": MarkerConfig(**context["marker_config"]).fingerprint(),
              "counts": _counts(context), "freshness": _freshness(context), "formal_audit": False,
              "checkpoint_bytes_and_parameters_reverified": True}
    for key, expected in checks.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Frozen collection changed: {key}; resume requires identical inputs and protocol")
    seeds = _json(output / ".private/replay.json")["packet_seeds"]
    if len(seeds) != 15 or len(set(seeds)) != 15 or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("Invalid private replay store")
    datasets, rows, packets = _generate(context, seeds)
    if rows != _json(output / "inputs.json") or packets != manifest["packets"]:
        raise ValueError("Regenerated marker identities/tensors or seed commitments differ")
    return manifest, datasets



def _shard_expected(manifest_hash, binding, batch, input_hash, settings_hash):
    return {"schema_version": SCHEMA, "frozen_manifest_sha256": manifest_hash,
            "model_key": _model_key(binding), "batch": batch, "inputs_sha256": input_hash,
            "collection_settings_sha256": settings_hash,
            "checkpoint_sha256": binding["checkpoint_sha256_recorded"],
            "parameters_sha256": binding["parameters_sha256_recorded"]}



def collect_observations(plan_path, output_dir, project_root=None, device="cpu", batch_size=128, progress=None):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Inference batch size must be positive")
    context = preflight(plan_path, project_root, verify_models=True)
    output = _output_path(output_dir, context)
    manifest, packets = _load_frozen(context, output)
    manifest_hash = file_hash(output / "frozen_manifest.json")
    datasets = {batch: {cid: _MergedMarkers(group) for cid, group in clients.items()}
                for batch, clients in packets.items()}
    (output / "observations").mkdir(exist_ok=True)
    receipts = {}
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    settings = {"device": str(device), "batch_size": batch_size, "threads": torch.get_num_threads(),
                "runtime": _runtime(), "cuda_version": torch.version.cuda,
                "accelerator_name": torch.cuda.get_device_name(torch.device(device))
                if str(device).startswith("cuda") else None}
    settings_path = output / "collection_settings.json"
    if settings_path.exists():
        if _json(settings_path) != settings:
            raise ValueError("Collection inference settings changed; resume requires the original settings")
    else:
        if any((output / "observations").glob("*.npz")):
            raise ValueError("Observation shards exist without frozen inference settings")
        _write_json(settings_path, settings)
    settings_hash = file_hash(settings_path)
    for binding in context["plan"]["models"]:
        key, model = _model_key(binding), None
        for batch in range(3):
            path = output / "observations" / f"{key}_batch{batch}.npz"
            receipt_path = path.with_suffix(".json")
            expected = _shard_expected(manifest_hash, binding, batch, manifest["inputs_sha256"], settings_hash)
            if path.exists() and receipt_path.exists():
                _, receipt = _read_shard(path, receipt_path, expected, datasets[batch], context)
                receipts[path.relative_to(output).as_posix()] = receipt["observation_sha256"]
                if progress:
                    progress(f"Verified existing shard: {key}, batch {batch + 1}/3")
                continue
            original_receipt = _json(receipt_path) if receipt_path.exists() else None
            if original_receipt is not None and any(original_receipt.get(k) != v for k, v in expected.items()):
                raise ValueError("Missing observation receipt does not match frozen inputs")
            if model is None:
                model = build_model(context["dataset"], 10)
                set_model_parameters(model, _load_parameters(context, binding))
            obs = {cid: observe_markers(model, ds, device, batch_size) for cid, ds in datasets[batch].items()}
            _validate_shard(obs, datasets[batch], context)
            if parameter_hash(get_model_parameters(model)) != binding["parameters_sha256_recorded"]:
                raise ValueError("Inference mutated model parameters or buffers")
            if path.exists():
                # An uncommitted complete NPZ from a crash is not silently trusted.
                existing = _read_npz(path)
                if any(not np.array_equal(existing[cid][field], obs[cid][field])
                       for cid in obs for field in obs[cid]):
                    raise ValueError("Uncommitted shard differs from identical-input replay")
            else:
                pending = path.with_name(path.stem + ".pending.npz")
                save_observations(obs, pending)
                if original_receipt is not None and file_hash(pending) != original_receipt["observation_sha256"]:
                    raise ValueError("Identical-input replay differs from the committed observation bytes")
                os.replace(pending, path)
            receipt = dict(expected, observation_sha256=file_hash(path),
                           model_unchanged_after_inference=True, inference_device=str(device),
                           inference_batch_size=batch_size)
            if original_receipt is not None:
                if receipt != original_receipt:
                    raise ValueError("Identical-input replay differs from the original observation receipt")
            else:
                _write_json(receipt_path, receipt)
            receipts[path.relative_to(output).as_posix()] = receipt["observation_sha256"]
            if progress:
                progress(f"Collected: {key}, batch {batch + 1}/3")
        del model
    _assert_sources_unchanged(context)
    if _code_hashes() != manifest["code_hashes"] or file_hash(output / "frozen_manifest.json") != manifest_hash:
        raise ValueError("Frozen code or manifest changed during collection")
    complete = {"schema_version": SCHEMA, "status": "complete_all_batches_before_analysis",
                "formal_audit": False, "frozen_manifest_sha256": manifest_hash,
                "collection_settings_sha256": settings_hash,
                "shards": receipts, "shard_count": 3 * len(context["plan"]["models"]),
                "sample_model_evaluations": 6000 * len(context["plan"]["models"])}
    target = output / "collection_complete.json"
    if target.exists():
        if _json(target) != complete:
            raise ValueError("Collection completion record differs")
    else:
        _write_json(target, complete)
    return complete



def load_collection(plan_path, output_dir, project_root=None):
    """Load complete hash-bound observations. Analysis does not reload checkpoints."""
    context = preflight(plan_path, project_root, verify_models=False)
    output = _output_path(output_dir, context)
    manifest, packets = _load_frozen(context, output)
    manifest_hash = file_hash(output / "frozen_manifest.json")
    complete = _json(output / "collection_complete.json")
    settings_hash = file_hash(output / "collection_settings.json")
    if (complete.get("status") != "complete_all_batches_before_analysis" or
            complete.get("frozen_manifest_sha256") != manifest_hash or complete.get("shard_count") != 3 * len(context["plan"]["models"]) or
            complete.get("collection_settings_sha256") != settings_hash):
        raise ValueError("All three batches and all frozen models must be complete before analysis")
    observations, found = {}, {}
    for batch in range(3):
        datasets = {cid: _MergedMarkers(group) for cid, group in packets[batch].items()}
        observations[batch] = {}
        for binding in context["plan"]["models"]:
            key = _model_key(binding)
            path = output / "observations" / f"{key}_batch{batch}.npz"
            expected = _shard_expected(manifest_hash, binding, batch, manifest["inputs_sha256"], settings_hash)
            obs, receipt = _read_shard(path, path.with_suffix(".json"), expected, datasets, context)
            observations[batch][key] = obs
            found[path.relative_to(output).as_posix()] = receipt["observation_sha256"]
    if complete.get("shards") != found:
        raise ValueError("Observation inventory differs from completion record")
    return context, manifest, observations



def analyze_collection(plan_path, output_dir, project_root=None):
    from fulbase.verification.main_audit_analysis import analyze_main_observations
    context, manifest, observations = load_collection(plan_path, output_dir, project_root)
    output = Path(output_dir).resolve()
    target = output / "analysis"
    completion = target / "analysis_manifest.json"
    source = {"frozen_manifest_sha256": file_hash(output / "frozen_manifest.json"),
              "collection_complete_sha256": file_hash(output / "collection_complete.json")}
    if completion.exists():
        saved = _json(completion)
        if (saved.get("status") != "complete" or saved.get("source") != source
                or saved.get("schema_version") != SCHEMA):
            raise ValueError("Completed analysis binding changed")
        if set(saved.get("files", {})) != {"report.json", *(
                name + ".csv" for name in _analysis_tables(_json(target / "report.json")))}:
            raise ValueError("Completed analysis artifact inventory changed")
        for name, digest in saved["files"].items():
            if file_hash(_inside(target, name)) != digest:
                raise ValueError(f"Completed analysis artifact changed: {name}")
        _assert_sources_unchanged(context)
        return _json(target / "report.json")
    report = analyze_main_observations(observations, context["plan"], context["marker_config"],
                                            context["audit_protocol"], task_metrics=context["task_metrics"])
    report["source"] = {**source,
                        "checkpoint_bytes_reverified_at_collection": True,
                        "checkpoint_bytes_reverified_at_analysis": False,
                        "freshness": manifest["freshness"]}
    _assert_sources_unchanged(context)
    if _code_hashes() != manifest["code_hashes"]:
        raise ValueError("Code changed during analysis")
    target.mkdir(exist_ok=True)
    artifacts = {"report.json": (json.dumps(report, indent=2, ensure_ascii=False,
                                           allow_nan=False) + "\n").encode("utf-8")}
    tables = _analysis_tables(report)
    for name, rows in tables.items():
        stream = io.StringIO(newline="")
        fields = sorted(set().union(*(row.keys() for row in rows)))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        artifacts[name + ".csv"] = stream.getvalue().encode("utf-8")
    for name, content in artifacts.items():
        _publish_analysis_file(target / name, content)
    _assert_sources_unchanged(context)
    if (_code_hashes() != manifest["code_hashes"]
            or file_hash(output / "frozen_manifest.json") != source["frozen_manifest_sha256"]
            or file_hash(output / "collection_complete.json") != source["collection_complete_sha256"]):
        raise ValueError("Frozen analysis sources changed during publication")
    _write_json(completion, {"schema_version": SCHEMA, "status": "complete",
                            "formal_audit": False, "source": source,
                            "files": {name: file_hash(target / name) for name in artifacts},
                            "table_rows": {name: len(rows) for name, rows in tables.items()}})
    return report

