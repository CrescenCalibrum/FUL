"""Single-factor ratio=.3 ablation using the unchanged interior head optimizer.

The sole candidate starts from the original Full. Existing ratio=.1 endpoints
are comparison references only: this runner never loads their results or models.
Endpoint, training evidence, and selection are frozen before benchmark and
original calibration evaluation. No independent audit input is accepted.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shutil
import time

import numpy as np
import pandas as pd
import torch

from fulbase.data import load_federated_data
from fulbase.experiment.io import RunLogger
from fulbase.experiment.pipeline import _evaluate_params
from fulbase.models import build_model
from fulbase.unlearning.marker_boundary_interior import InteriorBoundaryConfig, train_interior_boundary_head
from fulbase.utils import get_device, set_model_parameters, state_change_report
from fulbase.verification.audit import (
    AuditConfig, file_hash, observe_markers, parameter_hash, save_observations, write_audit,
)
from fulbase.verification.markers import MarkerConfig, MarkerDataset, derived_seed
from fulbase.verification.pga_diagnostic import load_source_run, _validate_data


SCHEMA = "rp2-head-boundary-radius-ablation-v1"
TRIALS = [
    {"trial_id": "r03", "mode": "interior", "retain_weight": .5, "project_ratio": .3},
]
METHODS = {"interior": "marker_boundary_interior_head"}
COMPARISON = {
    "baseline_project_ratio": .1, "new_project_ratio": .3,
    "changed_factor": "project_ratio", "baseline_role": "existing_interior_t02",
    "baseline_reexecution": False, "baseline_inputs_read": False,
}
EVIDENCE_TABLES = ("direction_trace", "direction_client_diagnostics", "direction_constraints",
                   "precision_diagnostics", "restoration_trace")
CODE_FILES = (
    "fulbase/verification/boundary_radius_diagnostic.py", "fulbase/unlearning/marker_boundary_interior.py",
    "fulbase/unlearning/boundary_precision.py", "fulbase/verification/boundary_d2_diagnostic.py",
    "fulbase/unlearning/marker_boundary_d2.py",
    "scripts/diagnose_marker_boundary_radius.py", "fulbase/unlearning/marker_boundary.py",
    "fulbase/verification/boundary_diagnostic.py", "fulbase/verification/pga_diagnostic.py",
    "fulbase/models.py", "fulbase/utils.py", "fulbase/data.py", "fulbase/train.py",
    "fulbase/experiment/pipeline.py", "fulbase/experiment/io.py", "fulbase/unlearning/base.py",
    "fulbase/verification/experiment.py", "fulbase/verification/synthetic.py", "fulbase/verification/markers.py",
    "fulbase/verification/audit.py", "fulbase/verification/statistics.py",
)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _status(path, value):
    """Only the new run's lifecycle record is replaceable; results are immutable."""
    temporary = path.with_suffix(".pending")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(path)


def _inside(root, relative):
    path = (root / str(relative).replace("\\", "/")).resolve()
    if root not in path.parents:
        raise ValueError("Source path must stay inside the project root")
    return path


def validate_boundary_radius_config(config):
    if config.get("schema_version") != SCHEMA:
        raise ValueError("Unsupported head-boundary configuration")
    sources = config.get("sources", [])
    if len(sources) != 3 or sorted(row["seed"] for row in sources) != [43, 44, 45]:
        raise ValueError("Bind exactly the three Full models for seeds 43, 44, 45")
    for row in sources:
        if type(row["seed"]) is not int or not isinstance(row["run_dir"], str):
            raise ValueError("Invalid source binding")
        for key in ("full_checkpoint_sha256", "full_parameters_sha256", "marker_fingerprint", "partition_sha256"):
            if re.fullmatch(r"[0-9a-f]{64}", str(row.get(key, ""))) is None:
                raise ValueError(f"Invalid source digest: {key}")
    if config.get("trials") != TRIALS:
        raise ValueError("Use the single prespecified ratio=.3 interior trial")
    if config.get("comparison") != COMPARISON:
        raise ValueError("Only the prespecified project_ratio factor may change")
    optimizer = config["optimizer"]
    if "project_ratio" in optimizer or "retain_weight" in optimizer:
        raise ValueError("Projection ratio and retained weight belong to individual trials")
    if type(optimizer.get("steps")) is not int or optimizer["steps"] != 100:
        raise ValueError("Radius ablation fixes exactly 100 proposal steps")
    fixed = {"lr": .01, "max_backtracks": 12, "backtrack_factor": .5, "gradient_clip": 10.,
             "boundary_gap": .5, "target_normalized_drop": .1, "target_accuracy_drop": .2,
             "retain_margin_drop": .08, "retain_accuracy_drop": .025,
             "retain_boundary_buffer": .1, "min_full_accuracy": .7, "min_full_margin": .01,
             "direction_max_sweeps": 200, "direction_tolerance": 1e-9,
             "direction_activation_rounds": 8, "direction_margin_safety": .9,
             "direction_accuracy_buffer": 1e-4, "direction_min_norm": 1e-12,
             "interior_margin_reserve": 1e-4, "max_restoration_steps": 3,
             "restoration_deficit_factor": .5}
    if (set(optimizer) != {"steps", *fixed}
            or any(type(optimizer.get(k)) is bool or optimizer.get(k) != value for k, value in fixed.items())):
        raise ValueError("Training boundary/constraint definitions differ from the declared protocol")
    for trial in TRIALS:
        InteriorBoundaryConfig(**optimizer, retain_weight=trial["retain_weight"], project_ratio=trial["project_ratio"])
    expected_selection = {
        "source": "training_markers_only", "rule": "fixed_radius03_arm_if_training_eligible",
        "target_accuracy_drop": .2, "target_normalized_margin_drop": .1,
        "retain_accuracy_drop": .025, "retain_normalized_margin_drop": .08,
        "min_full_accuracy": .7, "min_full_margin": .01,
        "none_eligible": "selected_trial_null", "require_changed_head": True,
    }
    if config.get("selection") != expected_selection:
        raise ValueError("Training-only selection policy changed")
    expected_evaluation = {"access_after_all_candidates_and_selection_saved": True,
                           "calibration": "original_development_role_only", "task": "original_bound_partition",
                           "audit_inputs_allowed": False, "recovery": "not_run", "retrain": "not_run",
                           "formal_audit": False}
    if config.get("evaluation") != expected_evaluation:
        raise ValueError("Post-selection evaluation policy changed")
    return config


def check_boundary_radius_sources(config_path, seed, project_root=None, *, verify_models=True):
    """Check explicit source metadata, then trusted Full bytes; no result access."""
    if type(seed) is not int or seed not in (43, 44, 45):
        raise ValueError("Choose source seed 43, 44 or 45")
    project = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[2]
    path = Path(config_path).resolve()
    config = validate_boundary_radius_config(_read(path))
    binding = next(row for row in config["sources"] if row["seed"] == seed)
    root = _inside(project, binding["run_dir"])
    cfg = _read(root / "config.json")
    manifest = _read(root / "verification/checkpoint_manifest.json")
    markers = MarkerConfig(**_read(root / "verification/marker_config.json"))
    audit = AuditConfig(**_read(root / "verification/audit_config.json"))
    full = manifest["stages"]["full_after_fl"]
    if (cfg["seed"] != seed or cfg["num_clients"] != 5 or str(cfg["forget_clients"]) != "0"
            or manifest["num_classes"] != 10):
        raise ValueError("Full seed/client/class/forget configuration differs")
    for field, source in (("full_checkpoint_sha256", full["checkpoint_sha256"]),
                          ("full_parameters_sha256", full["parameters_sha256"]),
                          ("marker_fingerprint", manifest["marker_fingerprint"]),
                          ("partition_sha256", file_hash(root / "data_partition.json"))):
        if binding[field] != source:
            raise ValueError(f"Fixed Full binding mismatch: {field}")
    if (markers.fingerprint() != binding["marker_fingerprint"] or not markers.enabled
            or markers.families_per_class != 4 or markers.variants_per_family != 2):
        raise ValueError("Keep the original enabled four-family/two-variant marker configuration")
    if manifest["partition_sha256"] != binding["partition_sha256"]:
        raise ValueError("Source partition manifest binding differs")
    if audit.protocol_locked or any(getattr(audit, key) != config["selection"][key]
                                   for key in ("min_full_accuracy", "min_full_margin")):
        raise ValueError("Source development audit qualification differs")
    if _read(root / "run_status.json").get("status") not in {"development_run_complete", "pga_diagnostic_complete"}:
        raise ValueError("Source run is incomplete")
    names = ["config.json", "data_partition.json", "run_status.json",
             "verification/checkpoint_manifest.json", "verification/marker_config.json",
             "verification/audit_config.json", "verification/training_marker_manifest.json"]
    if (root / "verification/training_result.json").is_file():
        names.append("verification/training_result.json")
    source_hashes = {name: file_hash(root / name) for name in names}
    source = load_source_run(root) if verify_models else None
    if source is not None:
        if (not source["markers"].enabled or source["markers"].families_per_class != 4
                or source["markers"].variants_per_family != 2):
            raise ValueError("Keep the original enabled four-family/two-variant markers")
        for key in ("min_full_accuracy", "min_full_margin"):
            if getattr(source["audit_cfg"], key) != config["selection"][key]:
                raise ValueError("Training Full qualification differs from original audit protocol")
    for name, digest in source_hashes.items():
        if file_hash(root / name) != digest:
            raise ValueError("Source changed during preflight")
    return {"project": project, "config_path": path, "config_sha256": file_hash(path),
            "config": config, "binding": binding, "root": root, "source": source,
            "source_hashes": source_hashes, "checkpoint_bytes_reverified": verify_models}


def _code_hashes():
    project = Path(__file__).resolve().parents[2]
    return {name: file_hash(project / name) for name in CODE_FILES}


def _assert_bound(context, code_hashes):
    if file_hash(context["config_path"]) != context["config_sha256"]:
        raise ValueError("Boundary configuration changed during execution")
    for name, digest in context["source_hashes"].items():
        if file_hash(context["root"] / name) != digest:
            raise ValueError(f"Bound source changed during execution: {name}")
    if file_hash(context["source"]["checkpoint"]) != context["binding"]["full_checkpoint_sha256"]:
        raise ValueError("Full checkpoint changed during execution")
    if _code_hashes() != code_hashes:
        raise ValueError("Bound execution code changed during execution")


def _training_summary(before, after, selection, changed_head):
    rows = []
    for cid in sorted(before):
        a, b = before[cid], after[cid]
        if not a["valid"] or not b["valid"] or a["role"] != "train" or b["role"] != "train":
            raise ValueError("Invalid training-only endpoint observations")
        if any(not np.array_equal(a[key], b[key]) for key in ("ids", "families", "labels", "features")):
            raise ValueError("Head endpoint changed training identities or backbone features")
        ac, bc = a["predictions"] == a["labels"], b["predictions"] == b["labels"]
        acc_drop = float((int(ac.sum()) - int(bc.sum())) / len(ac))
        margin_drop = float(np.asarray(a["normalized_margin"], dtype=np.float64).mean()
                            - np.asarray(b["normalized_margin"], dtype=np.float64).mean())
        full_acc = float(ac.mean())
        full_margin = float(np.asarray(a["normalized_margin"], dtype=np.float64).mean())
        target = cid == 0
        passed = ((acc_drop > selection["target_accuracy_drop"] and
                   margin_drop > selection["target_normalized_margin_drop"]) if target else
                  (acc_drop <= selection["retain_accuracy_drop"] and
                   margin_drop <= selection["retain_normalized_margin_drop"]))
        rows.append({"cid": cid, "client_group": "target" if target else "retained",
                     "role": "train", "num_examples": len(ac), "full_accuracy": full_acc,
                     "after_accuracy": float(bc.mean()), "accuracy_drop": acc_drop,
                     "full_normalized_margin": full_margin,
                     "after_normalized_margin": float(np.asarray(b["normalized_margin"], dtype=np.float64).mean()),
                     "normalized_margin_drop": margin_drop,
                     "full_correct_to_wrong_count": int(np.sum(ac & ~bc)),
                     "full_wrong_to_correct_count": int(np.sum(~ac & bc)),
                     "after_raw_margin_mean": float(np.asarray(b["margin"], dtype=np.float64).mean()),
                     "boundary_crossing_fraction": float((np.asarray(b["margin"]) < 0).mean()),
                     "boundary_gap_reached_fraction": float((np.asarray(b["margin"]) <= -.5).mean()),
                     "full_qualified": full_acc >= selection["min_full_accuracy"] and full_margin >= selection["min_full_margin"],
                     "empirical_condition_passed": bool(passed)})
    qualified = all(row["full_qualified"] for row in rows)
    target_pass = all(row["empirical_condition_passed"] for row in rows if row["client_group"] == "target")
    retain_pass = all(row["empirical_condition_passed"] for row in rows if row["client_group"] == "retained")
    return {"clients": rows, "full_marker_precheck_passed": qualified,
            "target_conditions_satisfied": target_pass, "retained_constraints_satisfied": retain_pass,
            "changed_head": bool(changed_head), "eligible": bool(qualified and target_pass and retain_pass and changed_head),
            "decision_basis": "training_empirical_only_not_statistical_audit"}


def _select(candidates):
    if (len(candidates) != 1 or candidates[0]["trial_id"] != "r03"
            or candidates[0]["mode"] != "interior"):
        raise ValueError("Selection requires the single fixed r03 interior candidate")
    winner = candidates[0] if candidates[0]["training_eligible"] else None
    return {"status": "training_candidate_selected" if winner else "no_eligible_candidate",
            "selected_trial": winner["trial_id"] if winner else None,
            "selected_stage": winner["stage"] if winner else None,
            "selection_source": "training_markers_only", "selection_frozen_before_evaluation": True,
            "eligible_trials": [winner["trial_id"]] if winner else [],
            "rule": "fixed_radius03_arm_if_training_eligible", "formal_audit": False,
            "candidate_count": 1, "candidates": candidates}


def _radius_training_summary(candidates, logs_by_trial):
    """Summarize the one fixed endpoint without loading the previous radius arm."""
    if len(candidates) != 1:
        raise ValueError("Radius summary requires one endpoint")
    candidate = candidates[0]
    logs = logs_by_trial[candidate["trial_id"]]
    training = candidate["training_summary"]
    target = next(row for row in training["clients"] if row["client_group"] == "target")
    retained = [row for row in training["clients"] if row["client_group"] == "retained"]
    reference_counts = {}
    for item in logs["precision_diagnostics"]:
        status = item["reference_status"]
        reference_counts[status] = reference_counts.get(status, 0) + 1
    radius = logs["projection_radius"]
    attempts = [attempt for step in logs["step_trace"] for attempt in step["attempts"]]
    projection_unknown = 0
    known_projected = 0
    accepted_projected = 0
    for attempt in attempts:
        if "projected" not in attempt:
            if attempt["accepted"] or attempt["reason"] != "invalid_numeric_proposal":
                raise ValueError("Only an invalid numeric proposal may omit projection status")
            projection_unknown += 1
        else:
            if type(attempt["projected"]) is not bool:
                raise ValueError("Recorded projection status must be boolean")
            known_projected += int(attempt["projected"])
            accepted_projected += int(attempt["projected"] and attempt["accepted"])
    row = {
        **{key: candidate[key] for key in ("trial_id", "mode", "stage", "method", "training_eligible",
            "retain_weight", "project_ratio", "head_delta_l2", "parameters_sha256", "source_full_parameters_sha256")},
        **{key: logs[key] for key in ("stop_reason", "iterations", "accepted_steps",
            "normal_accepted_steps", "restoration_accepted_steps", "precision_diagnostic_seconds",
            "optimization_total_wall_seconds", "optimization_excluding_precision_seconds_derived")},
        "projection_radius": radius,
        "head_fraction_of_radius": candidate["head_delta_l2"] / radius if radius > 0 else None,
        "projected_attempts": None if projection_unknown else known_projected,
        "known_projected_attempts": known_projected,
        "projection_status_unavailable_attempts": projection_unknown,
        "accepted_projected_steps": accepted_projected,
        "target_full_accuracy": target["full_accuracy"],
        "target_after_accuracy": target["after_accuracy"],
        "target_accuracy_drop": target["accuracy_drop"],
        "target_normalized_margin_drop": target["normalized_margin_drop"],
        "target_net_correct_count_drop": target["full_correct_to_wrong_count"] - target["full_wrong_to_correct_count"],
        "target_conditions_satisfied": training["target_conditions_satisfied"],
        "retained_constraints_satisfied": training["retained_constraints_satisfied"],
        "full_marker_precheck_passed": training["full_marker_precheck_passed"],
        "worst_retained_accuracy_drop": max(item["accuracy_drop"] for item in retained),
        "worst_retained_normalized_margin_drop": max(item["normalized_margin_drop"] for item in retained),
        "precision_diagnostic_rows": len(logs["precision_diagnostics"]),
        "precision_reference_status_counts": reference_counts,
    }
    return {"schema_version": "rp2-head-boundary-radius-training-v1",
            "source": "training_markers_only", "formal_audit": False,
            "source_full_parameters_sha256": candidate["source_full_parameters_sha256"],
            "start_policy": "original_full_not_existing_endpoint", "trials": [row],
            "comparison": dict(COMPARISON),
            "comparison_interpretation": "single_factor_diagnostic_no_baseline_result_access_or_audit_pass_claim",
            "selection_rule": "fixed_radius03_arm_if_training_eligible",
            "timing_note": "Total optimization wall time includes read-only precision; exclusion is derived by subtraction"}


def run_boundary_radius_diagnostic(config_path, seed, *, device="auto", data_dir=None,
                            out_dir="runs", run_name=None, threads=2, fd=None, project_root=None):
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(threads)
    context = check_boundary_radius_sources(config_path, seed, project_root)
    source, config = context["source"], context["config"]
    resolved = get_device(device)
    name = run_name or f"rp2_boundary_radius03_seed{seed}"
    if not name or Path(name).name != name or any(c in name for c in "/\\:"):
        raise ValueError("run_name must be a simple suffix")
    output = (Path(out_dir) / f"{datetime.now():%Y%m%d-%H%M%S-%f}_{name}").resolve()
    for bound in config["sources"]:
        root = _inside(context["project"], bound["run_dir"])
        if output == root or root in output.parents:
            raise ValueError("Output must be outside source runs")
    cfg = replace(source["cfg"], eval_device=str(resolved), run_name=name, out_dir=str(out_dir),
                  with_recovery=False, include_fakes=False, methods="marker_boundary_radius_ablation",
                  pga_layers="head", pga_data_scope="markers_only")
    if data_dir is not None:
        cfg.data_dir = str(data_dir)
    full_parameters = [a.copy() for a in source["params"]]
    full_hash = parameter_hash(full_parameters)
    code_hashes = _code_hashes()
    output.mkdir(parents=True, exist_ok=False)
    logger = RunLogger(output)
    vdir = output / "verification"
    vdir.mkdir()
    _write(output / "config.json", asdict(cfg))
    shutil.copyfile(context["config_path"], output / "boundary_radius_config.json")
    shutil.copyfile(context["root"] / "data_partition.json", output / "data_partition.json")
    _write(vdir / "marker_config.json", asdict(source["markers"]))
    _write(vdir / "audit_config.json", asdict(source["audit_cfg"]))
    _write(output / "boundary_radius_provenance.json", {
        "schema_version": SCHEMA, "source_run": str(context["root"]), "source_binding": context["binding"],
        "config_sha256": context["config_sha256"], "source_hashes": context["source_hashes"],
        "code_hashes": code_hashes, "source_full_parameters_sha256": full_hash,
        "start_policy": "original_full_not_existing_endpoint", "selection_access": "training_markers_only",
        "comparison": dict(COMPARISON), "baseline_inputs_accessed": False,
        "training_budget_source": "boundary_radius_config.json; config.json preserves original FL/benchmark settings",
        "evaluation_access": "original_calibration_and_benchmark_after_selection_frozen",
        "audit_inputs_accessed": False, "retrain": "not_run", "recovery": "not_run", "formal_audit": False,
        "device": str(resolved), "torch_version": torch.__version__, "numpy_version": np.__version__,
        "task_evaluation_policy": "disjoint_retain_forget_once_then_pool_all_v1"})
    _status(output / "run_status.json", {"status": "boundary_radius_development_running", "phase": "training_only", "formal_audit": False})
    print(f"[boundary radius diagnostic] Output: {output}\n[boundary radius diagnostic] Fixed Full: {full_hash}", flush=True)
    try:
        training = {cid: MarkerDataset(source["markers"], cid, cfg.dataset, "train",
                                      derived_seed(seed, "train-markers"), 10) for cid in range(5)}
        training_rows = [row for ds in training.values() for row in ds.manifest()]
        expected = _read(context["root"] / "verification/training_marker_manifest.json")
        fields = ("cid", "role", "marker_id", "family_id", "label")
        if sorted(tuple(r[k] for k in fields) for r in training_rows) != sorted(tuple(r[k] for k in fields) for r in expected):
            raise ValueError("Original training marker identities differ from source manifest")
        _write(vdir / "training_marker_manifest.json", training_rows)
        shutil.copyfile(source["checkpoint"], output / "models/model_full.pt")
        if file_hash(output / "models/model_full.pt") != context["binding"]["full_checkpoint_sha256"]:
            raise ValueError("Copied Full checkpoint differs from its bound source")
        full_model = build_model(cfg.dataset, 10)
        set_model_parameters(full_model, full_parameters)
        before_train = {cid: observe_markers(full_model, ds, str(resolved), cfg.batch_size) for cid, ds in training.items()}
        save_observations(before_train, vdir / "full_training_observations.npz")
        candidates, endpoint_parameters, logs_by_trial = [], {}, {}
        for index, trial in enumerate(config["trials"]):
            stage = "marker_boundary_radius03_pre_recovery"
            trial_config = InteriorBoundaryConfig(**config["optimizer"], retain_weight=trial["retain_weight"], project_ratio=trial["project_ratio"])
            start_model = build_model(cfg.dataset, 10)
            set_model_parameters(start_model, full_parameters)
            print(f"[boundary radius diagnostic] {trial['trial_id']} training-only endpoint", flush=True)
            if resolved.type == "cuda": torch.cuda.synchronize(resolved)
            started = time.perf_counter()
            result = train_interior_boundary_head(start_model, training, [0], config=trial_config, mode=trial["mode"],
                                         device=str(resolved), batch_size=cfg.batch_size)
            if resolved.type == "cuda": torch.cuda.synchronize(resolved)
            seconds = time.perf_counter() - started
            if parameter_hash(full_parameters) != full_hash or parameter_hash(source["params"]) != full_hash:
                raise ValueError("Optimizer mutated the bound Full")
            start_change = state_change_report(start_model, full_parameters)
            if start_change["changed_parameters"] or start_change["changed_buffers"]:
                raise ValueError("Optimizer mutated its input Full model")
            model = build_model(cfg.dataset, 10)
            set_model_parameters(model, result.parameters)
            change = state_change_report(model, full_parameters)
            if change["changed_buffers"] or any(not k.startswith("classifier.") for k in change["changed_parameters"]):
                raise ValueError("Boundary endpoint changed non-head parameters or buffers")
            if any(not np.isfinite(a).all() for a in result.parameters):
                raise ValueError("Boundary endpoint contains nonfinite state")
            logger.save_model_params(f"model_{stage}", result.parameters)
            checkpoint = output / "models" / f"model_{stage}.pt"
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if parameter_hash(saved) != parameter_hash(result.parameters):
                raise ValueError("Candidate checkpoint round-trip changed parameters")
            after_train = {cid: observe_markers(model, ds, str(resolved), cfg.batch_size) for cid, ds in training.items()}
            measured = _training_summary(before_train, after_train, config["selection"], change["changed_parameters"])
            if bool(result.logs["training_eligible"]) != measured["eligible"]:
                raise ValueError("Radius optimizer and actual endpoint disagree on training eligibility")
            logs = dict(result.logs, endpoint_inference_training_summary=measured)
            stage_dir = vdir / stage
            _write(stage_dir / "method_logs.json", logs)
            for key in ("step_trace", *EVIDENCE_TABLES):
                rows = logs.get(key)
                if not isinstance(rows, list):
                    raise ValueError(f"Radius optimizer omitted training evidence: {key}")
                frame = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["step"])
                frame.to_csv(stage_dir / f"{key}.csv", index=False)
            pd.DataFrame(measured["clients"]).to_csv(stage_dir / "training_conditions.csv", index=False)
            save_observations(before_train, stage_dir / "training_marker_full_observations.npz")
            save_observations(after_train, stage_dir / "training_marker_after_observations.npz")
            row = {**trial, "trial_index": index, "stage": stage, "method": METHODS[trial["mode"]],
                   "checkpoint": checkpoint.relative_to(output).as_posix(), "checkpoint_sha256": file_hash(checkpoint),
                   "parameters_sha256": parameter_hash(result.parameters), "source_full_parameters_sha256": full_hash,
                   "head_delta_l2": change["state_l2_change"], "model_state_change_from_full": change,
                   "training_eligible": bool(measured["eligible"] and logs["training_eligible"]),
                   "training_summary": measured, "optimizer_training_eligible": logs["training_eligible"],
                   "unlearning_time_sec": seconds, "optimizer_config": asdict(trial_config)}
            candidates.append(row)
            endpoint_parameters[stage] = saved
            logs_by_trial[trial["trial_id"]] = logs
        _assert_bound(context, code_hashes)
        _write(output / "radius_training_summary.json", _radius_training_summary(candidates, logs_by_trial))
        radius_summary_hash = file_hash(output / "radius_training_summary.json")
        _write(output / "candidate_inventory.json", {"status": "single_radius_endpoint_fixed", "formal_audit": False,
            "full_parameters_sha256": full_hash, "candidates": candidates,
            "radius_training_summary_sha256": radius_summary_hash})
        selection = _select(candidates)
        selection["candidate_inventory_sha256"] = file_hash(output / "candidate_inventory.json")
        selection["frozen_utc"] = datetime.now(timezone.utc).isoformat()
        _write(output / "selection.json", selection)
        frozen_selection = file_hash(output / "selection.json")
        _status(output / "run_status.json", {"status": "boundary_radius_development_running", "phase": "selection_frozen",
                                               "selection_sha256": frozen_selection, "formal_audit": False})

        # Evaluation begins only after every checkpoint and training-only choice is fixed.
        if fd is None:
            if cfg.synthetic_smoke:
                from fulbase.verification.synthetic import synthetic_data
                fd = synthetic_data(cfg.num_clients, cfg.seed)
            else:
                fd = load_federated_data(cfg.dataset, cfg.data_dir, cfg.num_clients, cfg.partition, cfg.alpha, cfg.seed)
        _validate_data(fd, source)
        calibration = {cid: MarkerDataset(source["markers"], cid, cfg.dataset, "calibration", cfg.calibration_seed, 10)
                       for cid in range(5)}
        _write(vdir / "calibration_marker_manifest.json", [row for ds in calibration.values() for row in ds.manifest()])
        manifest = {"schema_version": "marker-experiment-v1", "num_classes": 10,
                    "marker_fingerprint": source["markers"].fingerprint(),
                    "partition_sha256": source["manifest"]["partition_sha256"],
                    "data_origin": source["manifest"].get("data_origin", "benchmark"), "stages": {}}
        baseline_obs, baseline_metrics, summary = None, None, []
        rows = [("full_after_fl", full_parameters, None),
                *[(c["stage"], endpoint_parameters[c["stage"]], c) for c in candidates]]
        for stage, parameters, candidate in rows:
            metrics = _evaluate_params(stage, parameters, fd, cfg, [1, 2, 3, 4], [0])
            if not all(math.isfinite(float(x)) for x in metrics.values()):
                raise ValueError("Nonfinite task evaluation")
            nr, nf = metrics["retain_num_examples"], metrics["forget_num_examples"]
            if abs(metrics["all_acc"] - (nr * metrics["retain_acc"] + nf * metrics["forget_acc"]) / (nr + nf)) > 1e-12:
                raise ValueError("ACC weighted identity failed")
            model = build_model(cfg.dataset, 10)
            set_model_parameters(model, parameters)
            observations = {cid: observe_markers(model, ds, str(resolved), cfg.batch_size) for cid, ds in calibration.items()}
            checkpoint = output / (candidate["checkpoint"] if candidate else "models/model_full.pt")
            metadata = {"stage": stage, "method": candidate["method"] if candidate else "none", "role": "calibration",
                        "behavior": "marker_erasure_attempt" if candidate else "unmodified_full",
                        "objective": "boundary_crossing_interior_radius_ablation" if candidate else None,
                        "mode": candidate["mode"] if candidate else None,
                        "checkpoint_sha256": file_hash(checkpoint), "parameters_sha256": parameter_hash(parameters),
                        "source_full_parameters_sha256": full_hash, "marker_fingerprint": source["markers"].fingerprint(),
                        "model_state_change_from_full": state_change_report(model, full_parameters),
                        "optimization_access": "training_markers_only" if candidate else "none",
                        "training_selected": bool(candidate and selection["selected_trial"] == candidate["trial_id"]),
                        "selection_sha256": frozen_selection}
            if baseline_obs is None:
                baseline_obs, baseline_metrics = observations, metrics
            audit = write_audit(baseline_obs, observations, [0], source["audit_cfg"], vdir / stage,
                                metadata, baseline_metrics, metrics)
            manifest["stages"][stage] = {**metadata, "checkpoint": checkpoint.relative_to(output).as_posix(), "task_metrics": metrics}
            row = {"unlearner": metadata["method"], "recovery": "none", **metrics,
                   "status": audit["status"], "exploratory_status": audit["exploratory_status"],
                   "training_eligible": candidate["training_eligible"] if candidate else None,
                   "training_selected": metadata["training_selected"],
                   "trial_id": candidate["trial_id"] if candidate else None,
                   "unlearning_time_sec": candidate["unlearning_time_sec"] if candidate else None}
            logger.add_metrics(stage, row)
            summary.append({"stage": stage, **row})
        _write(vdir / "checkpoint_manifest.json", manifest)
        pd.DataFrame(summary).to_csv(vdir / "summary.csv", index=False)
        _assert_bound(context, code_hashes)
        if (file_hash(output / "selection.json") != frozen_selection or
                file_hash(output / "candidate_inventory.json") != selection["candidate_inventory_sha256"] or
                file_hash(output / "radius_training_summary.json") != radius_summary_hash):
            raise ValueError("Post-selection evaluation changed the frozen selection")
        for candidate in candidates:
            if file_hash(output / candidate["checkpoint"]) != candidate["checkpoint_sha256"]:
                raise ValueError("Post-selection evaluation changed a fixed endpoint")
        if file_hash(output / "models/model_full.pt") != context["binding"]["full_checkpoint_sha256"]:
            raise ValueError("Post-selection evaluation changed the fixed Full checkpoint")
        _status(output / "run_status.json", {"status": "development_run_complete", "method": "marker_boundary_radius_ablation",
            "num_trials": 1, "selected_trial": selection["selected_trial"], "selection_status": selection["status"],
            "selection_sha256": frozen_selection, "source_full_parameters_sha256": full_hash,
            "formal_audit": False, "recovery": "not_run", "retrain": "not_run"})
    except BaseException as exc:
        _status(output / "run_status.json", {"status": "boundary_radius_development_failed", "error": str(exc), "formal_audit": False})
        raise
    print(f"[boundary radius diagnostic] {selection['status']}: {selection['selected_trial']}", flush=True)
    return output
