"""Evaluate already frozen main candidates without rerunning any optimizer.

Model validity and audit acceptance are different outcomes. Invalid endpoints
remain in the inventory and coverage denominator; their missing metrics are
never replaced with zero or with results from a repaired checkpoint.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd
import torch

from fulbase.data import load_federated_data
from fulbase.experiment.pipeline import _evaluate_params
from fulbase.models import build_model
from fulbase.utils import get_device, get_model_parameters, set_model_parameters, state_change_report
from fulbase.verification.audit import AuditConfig, file_hash, parameter_hash, observe_markers, write_audit
from fulbase.verification.experiment import VerificationExperimentConfig
from fulbase.verification.markers import MarkerConfig, MarkerDataset, derived_seed
from fulbase.verification.pga_diagnostic import _validate_data


SCHEMA = "rp2-main-candidates-v1"
EVALUATION_SCHEMA = "rp2-main-candidate-evaluation-recovery-v1"
BASE_STAGES = ("history_subtract_pre_recovery", "federaser_lite_pre_recovery",
               "pga_pre_recovery", "fake_noop_pre_recovery", "fake_corrupt_pre_recovery")
CONTROL_STAGES = ("fixed_C_lambda05_pre_recovery", "D_margin_lambda8_pre_recovery")
METADATA = ("config.json", "main_candidate_config.json", "main_candidate_provenance.json",
            "candidate_inventory.json", "data_partition.json", "verification/marker_config.json",
            "verification/audit_config.json", "verification/training_result.json",
            "verification/training_marker_manifest.json")
CODE_FILES = ("fulbase/verification/main_evaluation.py", "scripts/reevaluate_verification_main_candidates.py",
              "fulbase/experiment/pipeline.py", "fulbase/models.py", "fulbase/utils.py",
              "fulbase/data.py", "fulbase/train.py", "fulbase/verification/markers.py",
              "fulbase/verification/audit.py", "fulbase/verification/statistics.py",
              "fulbase/verification/pga_diagnostic.py", "fulbase/verification/protocol_v2_io.py",
              "fulbase/verification/protocol_v2.py")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _status(root, value):
    pending = root / "run_status.pending"
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(root / "run_status.json")


def _path(root, relative):
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError(f"Artifact resolves outside candidate directory: {relative}")
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen candidate artifact: {path}")
    return path


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _code_hashes():
    root = Path(__file__).resolve().parents[2]
    return {name: file_hash(root / name) for name in CODE_FILES}


def _state_diagnostic(parameters, model, stage):
    reference = model.state_dict()
    if not isinstance(parameters, list) or len(parameters) != len(reference):
        raise ValueError(f"Checkpoint state length mismatch: {stage}")
    nonfinite, variances = [], []
    for (name, value), array in zip(reference.items(), parameters):
        expected = value.detach().cpu().numpy()
        if (not isinstance(array, (np.ndarray, np.generic)) or array.shape != expected.shape
                or array.dtype != expected.dtype):
            raise ValueError(f"Checkpoint shape/dtype mismatch: {stage}/{name}")
        count = int(np.count_nonzero(~np.isfinite(array)))
        if count:
            nonfinite.append({"name": name, "count": count})
        if name.endswith("running_var"):
            finite = array[np.isfinite(array)]
            variances.append({"name": name, "min": float(finite.min()) if finite.size else None,
                              "negative_count": int(np.count_nonzero(array < 0)), "nonfinite_count": count})
    set_model_parameters(model, parameters)
    if parameter_hash(get_model_parameters(model)) != parameter_hash(parameters):
        raise ValueError(f"Checkpoint round-trip changed parameter state: {stage}")
    return {"finite_state": not nonfinite, "nonfinite_state_entries": nonfinite,
            "batchnorm_running_variances": variances,
            "valid": not nonfinite and not any(row["negative_count"] for row in variances)}


def _check_marker_identities(rows, markers, cfg, nclasses, role, sample_seed):
    token = derived_seed(markers.fingerprint(), role, sample_seed)
    expected = {f"{cid}:{role}:{token}:{label}:{family}:{variant}":
                (cid, label, f"{cid}:{role}:{token}:{label}:{family}")
                for cid in range(cfg.num_clients) for label in range(nclasses)
                for family in range(markers.families_per_class)
                for variant in range(markers.variants_per_family)}
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError(f"{role.capitalize()} marker identity count mismatch")
    seen = set()
    for row in rows:
        mid = row.get("marker_id")
        if (row.get("role") != role or mid in seen or mid not in expected or
                type(row.get("cid")) is not int or type(row.get("label")) is not int or
                (row.get("cid"), row.get("label"), row.get("family_id")) != expected[mid]):
            raise ValueError(f"{role.capitalize()} marker identities differ from candidate configuration")
        seen.add(mid)


def check_frozen_candidates(run_dir):
    """Read only local frozen artifacts; never load history, data, or markers."""
    root = Path(run_dir).resolve()
    hashes = {name: file_hash(_path(root, name)) for name in METADATA}
    inventory = _read(root / "candidate_inventory.json")
    provenance = _read(root / "main_candidate_provenance.json")
    main_config = _read(root / "main_candidate_config.json")
    for name, document in (("inventory", inventory), ("provenance", provenance), ("config", main_config)):
        if document.get("schema_version") != SCHEMA:
            raise ValueError(f"Unsupported main candidate {name} schema")
    if (inventory.get("status") != "all_endpoints_fixed" or inventory.get("selected_trial") is not None
            or inventory.get("selection_policy") != "all_prespecified_endpoints_no_winner_selection"):
        raise ValueError("Require complete, unselected all_endpoints_fixed candidate inventory")
    controls = main_config.get("include_marker_controls")
    if type(controls) is not bool:
        raise ValueError("Invalid include_marker_controls in main candidate configuration")
    expected = BASE_STAGES + (CONTROL_STAGES if controls else ())
    candidates = inventory.get("candidates")
    if (not isinstance(candidates, list) or len(candidates) != len(expected)
            or {row.get("stage") for row in candidates} != set(expected)):
        raise ValueError("Frozen inventory must contain the exact prespecified 5/7 candidate stages")
    cfg = VerificationExperimentConfig(**_read(root / "config.json"))
    if cfg.federaser_bn_policy not in {"legacy_replay", "retained_endpoint"}:
        raise ValueError("Unknown frozen FedEraser BN policy")
    markers = MarkerConfig(**_read(root / "verification/marker_config.json"))
    audit_cfg = AuditConfig(**_read(root / "verification/audit_config.json"))
    if audit_cfg.protocol_locked or not markers.enabled:
        raise ValueError("Recovery requires enabled markers and the original unlocked development audit")
    partition = _read(root / "data_partition.json")
    nclasses = partition.get("num_classes")
    if type(nclasses) is not int or nclasses < 2:
        raise ValueError("Missing/invalid partition class count")
    for key in ("train_indices", "test_indices"):
        rows = partition.get(key)
        if not isinstance(rows, list) or len(rows) != cfg.num_clients or any(not row for row in rows):
            raise ValueError("Source partition has empty/missing clients")
        flat = [index for row in rows for index in row]
        if any(type(index) is not int for index in flat) or sorted(flat) != list(range(len(flat))):
            raise ValueError("Source partition must cover distinct samples exactly once")
    forget = [int(x.strip()) for x in cfg.forget_clients.split(",")]
    if len(set(forget)) != len(forget) or not set(forget) < set(range(cfg.num_clients)) or not forget:
        raise ValueError("Invalid configured forget clients")
    original_hashes = provenance.get("source_hashes", {})
    generation_code = provenance.get("code_hashes", {})
    if (not original_hashes or not generation_code or
            any(not _digest(value) for value in [*original_hashes.values(), *generation_code.values()])):
        raise ValueError("Missing/invalid generation provenance hashes")
    for relative in ("data_partition.json", "verification/training_result.json",
                     "verification/training_marker_manifest.json", "models/model_full.pt"):
        actual = file_hash(_path(root, relative))
        hashes[relative] = actual
        if actual != original_hashes.get(relative):
            raise ValueError(f"Original source artifact hash mismatch: {relative}")
    # The generator rewrites these JSON files with a trailing newline; the Full
    # runner used save_json without it. Verify that exact original serialization.
    for relative in ("verification/marker_config.json", "verification/audit_config.json"):
        serialized = json.dumps(_read(root / relative), ensure_ascii=False, indent=2)
        serialized_hashes = {hashlib.sha256(serialized.replace("\n", newline).encode("utf-8")).hexdigest()
                             for newline in ("\n", "\r\n")}
        if original_hashes.get(relative) not in {hashes[relative], *serialized_hashes}:
            raise ValueError(f"Original source configuration hash mismatch: {relative}")
    training = _read(root / "verification/training_result.json")
    full_hash = inventory.get("source_full_parameters_sha256")
    if (not _digest(full_hash) or provenance.get("source_full_parameters_sha256") != full_hash
            or training.get("final_parameters_sha256") != full_hash
            or training.get("rounds") != cfg.num_rounds or training.get("backend") != cfg.backend):
        raise ValueError("Full/configuration differs from recorded training result")
    # Validate saved training identities without constructing marker images.
    _check_marker_identities(_read(root / "verification/training_marker_manifest.json"), markers,
                             cfg, nclasses, "train", derived_seed(cfg.seed, "train-markers"))
    records = [{"stage": "full_after_fl", "method": "none", "behavior": "unmodified_full",
                "optimization_access": "none", "checkpoint": "models/model_full.pt",
                "checkpoint_sha256": hashes["models/model_full.pt"], "parameters_sha256": full_hash,
                "unlearning_time_sec": None, "strong_training_eligible": None}, *candidates]
    diagnostics = {}
    model = build_model(cfg.dataset, nclasses)
    for row in records:
        stage = row["stage"]
        expected_path = "models/model_full.pt" if stage == "full_after_fl" else f"models/model_{stage}.pt"
        if row.get("checkpoint") != expected_path:
            raise ValueError(f"Unexpected frozen checkpoint path: {stage}")
        if stage != "full_after_fl" and (row.get("source_full_parameters_sha256") != full_hash
                or row.get("predeclared_endpoint_included") is not True or row.get("training_selected") is not False):
            raise ValueError(f"Candidate Full binding/selection mismatch: {stage}")
        checkpoint = _path(root, expected_path)
        actual = file_hash(checkpoint)
        if actual != row.get("checkpoint_sha256"):
            raise ValueError(f"Frozen checkpoint file hash mismatch: {stage}")
        hashes[expected_path] = actual
        parameters = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if parameter_hash(parameters) != row.get("parameters_sha256"):
            raise ValueError(f"Frozen checkpoint parameter hash mismatch: {stage}")
        diagnostics[stage] = _state_diagnostic(parameters, model, stage)
        for filename in ("method_logs.json", "step_trace.csv"):
            relative = f"verification/{stage}/{filename}"
            if filename == "method_logs.json" and stage != "full_after_fl":
                hashes[relative] = file_hash(_path(root, relative))
            elif (root / relative).exists():
                hashes[relative] = file_hash(_path(root, relative))
        if stage in {"history_subtract_pre_recovery", "federaser_lite_pre_recovery"}:
            logs = _read(root / f"verification/{stage}/method_logs.json")
            if logs.get("forget_client_ids") != forget:
                raise ValueError(f"Configured forget clients differ from frozen method logs: {stage}")
            if stage == "federaser_lite_pre_recovery":
                policy = cfg.federaser_bn_policy
                budget = main_config.get("budgets", {}).get("federaser_lite", {})
                expected_method = "federaser_lite_bn_endpoint" if policy == "retained_endpoint" else "federaser_lite"
                if row.get("method") != expected_method:
                    raise ValueError("Frozen FedEraser method identity differs from its BN policy")
                if policy == "retained_endpoint":
                    for evidence in (row, budget, logs):
                        if evidence.get("bn_policy") != policy or evidence.get("method_variant") != expected_method:
                            raise ValueError("Frozen FedEraser BN policy/variant evidence differs")
                    if provenance.get("federaser_bn_policy", {}).get("effective") != policy:
                        raise ValueError("Frozen FedEraser BN provenance differs")
                elif any(evidence.get("bn_policy", "legacy_replay") != policy for evidence in (row, budget, logs)):
                    raise ValueError("A BN endpoint variant cannot be relabelled as legacy FedEraser")
    for relative in ("run_status.json", "verification/calibration_marker_manifest.json"):
        if (root / relative).exists():
            hashes[relative] = file_hash(_path(root, relative))
    calibration_identity_verified = "verification/calibration_marker_manifest.json" in hashes
    if calibration_identity_verified:
        _check_marker_identities(_read(root / "verification/calibration_marker_manifest.json"),
                                 markers, cfg, nclasses, "calibration", cfg.calibration_seed)
    full_report = "verification/full_after_fl/audit_report.json"
    if (root / full_report).exists():
        hashes[full_report] = file_hash(_path(root, full_report))
        report = _read(root / full_report)
        if report.get("protocol") != _read(root / "verification/audit_config.json"):
            raise ValueError("Original Full audit protocol differs from copied audit configuration")
        if (report.get("metadata", {}).get("parameters_sha256") != full_hash or
                report.get("metadata", {}).get("marker_fingerprint") != markers.fingerprint()):
            raise ValueError("Original Full audit metadata differs from the frozen Full/markers")
        if report.get("forget_client_ids") != sorted(forget):
            raise ValueError("Configured forget clients differ from original Full audit")
    # A failed generation run need not have written checkpoint_manifest.json.
    if (root / "verification/checkpoint_manifest.json").exists():
        relative = "verification/checkpoint_manifest.json"
        hashes[relative] = file_hash(_path(root, relative))
        manifest = _read(root / relative)
        if (manifest.get("partition_sha256") != hashes["data_partition.json"] or
                manifest.get("marker_fingerprint") != markers.fingerprint()):
            raise ValueError("Existing checkpoint manifest configuration mismatch")
        for stage, entry in manifest.get("stages", {}).items():
            record = next((row for row in records if row["stage"] == stage), None)
            if record is None or any(entry.get(key) != record[key] for key in
                                     ("checkpoint", "checkpoint_sha256", "parameters_sha256")):
                raise ValueError(f"Existing checkpoint manifest differs from frozen inventory: {stage}")
    return {"root": root, "cfg": cfg, "markers": markers, "audit_cfg": audit_cfg,
            "partition": partition, "manifest": {"num_classes": nclasses}, "num_classes": nclasses,
            "inventory": inventory, "provenance": provenance, "records": records,
            "source_hashes": hashes, "full_parameters_sha256": full_hash, "diagnostics": diagnostics,
            "original_calibration_identity_verified": calibration_identity_verified}


def _assert_unchanged(source):
    for name, expected in source["source_hashes"].items():
        if file_hash(_path(source["root"], name)) != expected:
            raise ValueError(f"Frozen source artifact changed during evaluation: {name}")


def run_candidate_evaluation(run_dir, run_directory, device="auto", data_dir=None, threads=2,
                             fd=None, diagnose_v2=False, ablation_config=None, bootstrap_samples=10000):
    """Copy and evaluate every frozen endpoint; preserve invalid stages explicitly."""
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be a positive integer")
    source = check_frozen_candidates(run_dir)
    output = Path(run_directory).resolve()
    root = source["root"]
    if output == root or root in output.parents or output in root.parents:
        raise ValueError("Evaluation output must be separate from its source run")
    if output.exists():
        raise FileExistsError(f"Evaluation output already exists: {output}")
    for stage, diagnostic in source["diagnostics"].items():
        print(f"[candidate evaluation] {stage}: state_valid={diagnostic['valid']}", flush=True)
        if not diagnostic["valid"]:
            print(json.dumps(diagnostic, ensure_ascii=False), flush=True)
    if not source["diagnostics"]["full_after_fl"]["valid"]:
        raise ValueError("Full model has invalid numerical state; evaluation cannot continue")
    torch.set_num_threads(threads)
    resolved = get_device(device)
    cfg = replace(source["cfg"], eval_device=str(resolved))
    if data_dir is not None:
        cfg.data_dir = str(data_dir)
    if fd is None:
        if cfg.synthetic_smoke:
            from fulbase.verification.synthetic import synthetic_data
            fd = synthetic_data(cfg.num_clients, cfg.seed)
        else:
            fd = load_federated_data(cfg.dataset, cfg.data_dir, cfg.num_clients, cfg.partition, cfg.alpha, cfg.seed)
    _validate_data(fd, source)
    forget = [int(x.strip()) for x in cfg.forget_clients.split(",")]
    retain = [cid for cid in range(cfg.num_clients) if cid not in forget]
    code_hashes = _code_hashes()
    output.mkdir(parents=True, exist_ok=False)
    print(f"[candidate evaluation] Output: {output}", flush=True)
    copied = [*METADATA, *(row["checkpoint"] for row in source["records"])]
    copied += [name for name in source["source_hashes"] if
               name.endswith("/method_logs.json") or name.endswith("/step_trace.csv")]
    for relative in copied:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_path(root, relative), destination)
        if file_hash(destination) != source["source_hashes"][relative]:
            raise ValueError(f"Copied frozen artifact hash mismatch: {relative}")
    _write(output / "evaluation_provenance.json", {
        "schema_version": EVALUATION_SCHEMA, "source_run": str(root),
        "source_artifact_sha256": source["source_hashes"],
        "generation_code_sha256": source["provenance"]["code_hashes"],
        "evaluation_code_sha256": code_hashes,
        "source_full_parameters_sha256": source["full_parameters_sha256"],
        "original_generation_provenance_preserved": True,
        "generation_metadata_digest_unavailable": ["config.json", "main_candidate_config.json",
                                                    "main_candidate_provenance.json", "candidate_inventory.json"],
        "metadata_verification": "current_bytes_bound; original_copied_artifact_digests_and_configuration_identity_checked",
        "original_calibration_identity_verified": source["original_calibration_identity_verified"],
        "calibration_identity_limitation": (None if source["original_calibration_identity_verified"] else
            "Original calibration manifest is absent; regeneration uses the saved configuration without original identity verification"),
        "old_source_directory_read": False, "history_read": False, "training_performed": False,
        "checkpoint_mutation": False, "independent_audit_access": False, "formal_audit": False,
        "evaluation_device": str(resolved), "evaluation_data_dir": str(cfg.data_dir),
        "threads": threads, "created_utc": datetime.now(timezone.utc).isoformat()})
    vdir = output / "verification"
    _status(output, {"status": "main_candidate_evaluation_running", "formal_audit": False})
    try:
        calibration = {cid: MarkerDataset(source["markers"], cid, cfg.dataset, "calibration",
                         cfg.calibration_seed, source["num_classes"]) for cid in range(cfg.num_clients)}
        calibration_rows = [row for dataset in calibration.values() for row in dataset.manifest()]
        original_calibration = root / "verification/calibration_marker_manifest.json"
        if original_calibration.exists() and _read(original_calibration) != calibration_rows:
            raise ValueError("Calibration marker identities differ from the frozen run")
        _write(vdir / "calibration_marker_manifest.json", calibration_rows)
        manifest = {"schema_version": "marker-experiment-v1", "num_classes": source["num_classes"],
                    "marker_fingerprint": source["markers"].fingerprint(),
                    "partition_sha256": source["source_hashes"]["data_partition.json"],
                    "data_origin": source["partition"].get("data_origin", "benchmark"), "stages": {}}
        full = torch.load(output / "models/model_full.pt", map_location="cpu", weights_only=False)
        baseline_obs, baseline_metrics, coverage = None, None, []
        for row in source["records"]:
            stage = row["stage"]
            print(f"[candidate evaluation] Evaluating {stage}", flush=True)
            parameters = torch.load(output / row["checkpoint"], map_location="cpu", weights_only=False)
            diagnostic = source["diagnostics"][stage]
            metadata = {"stage": stage, "method": row["method"], "behavior": row["behavior"],
                        "role": "calibration", "checkpoint_sha256": row["checkpoint_sha256"],
                        "parameters_sha256": row["parameters_sha256"],
                        "source_full_parameters_sha256": source["full_parameters_sha256"],
                        "marker_fingerprint": source["markers"].fingerprint(),
                        "optimization_access": row["optimization_access"], "training_selected": False,
                        "strong_training_eligible": row.get("strong_training_eligible"),
                        "candidate_inventory_sha256": source["source_hashes"]["candidate_inventory.json"]}
            for key in ("bn_policy", "method_variant"):
                if key in row:
                    metadata[key] = row[key]
            metrics, failure, report = {}, None, None
            if not diagnostic["valid"]:
                failure = {"kind": "invalid_numerical_state", "message": "Nonfinite state or negative BatchNorm running variance"}
            else:
                try:
                    metrics = _evaluate_params(stage, parameters, fd, cfg, retain, forget)
                except ValueError as exc:
                    if not str(exc).startswith(f"Invalid task evaluation metrics at {stage}/"):
                        raise
                    failure = {"kind": "numerical_failure", "message": str(exc)}
            status = "invalid_model" if failure else "evaluated"
            if stage == "full_after_fl" and failure:
                raise ValueError(f"Full model numerical evaluation failed: {failure['message']}")
            if not failure:
                if any(not math.isfinite(float(value)) for value in metrics.values()):
                    raise ValueError("Unexpected nonfinite task result outside original evaluation guard")
                nr, nf = metrics["retain_num_examples"], metrics["forget_num_examples"]
                if min(nr, nf) <= 0 or abs(metrics["all_acc"] -
                        (nr * metrics["retain_acc"] + nf * metrics["forget_acc"]) / (nr + nf)) > 1e-12:
                    raise ValueError("ACC weighted identity failed")
                model = build_model(cfg.dataset, source["num_classes"])
                set_model_parameters(model, parameters)
                metadata["model_state_change_from_full"] = state_change_report(model, full)
                observations = {cid: observe_markers(model, dataset, str(resolved), cfg.batch_size)
                                for cid, dataset in calibration.items()}
                invalid_clients = [cid for cid, obs in observations.items() if not obs["valid"] or
                    any(not np.isfinite(obs[key]).all() for key in ("logits", "features", "margin", "normalized_margin"))]
                if invalid_clients:
                    status = "not_evaluable"
                    failure = {"kind": "invalid_marker_observations", "clients": invalid_clients,
                               "message": "Marker observations are nonfinite or degenerate"}
                    if stage == "full_after_fl":
                        raise ValueError("Full marker observations are invalid; audit comparison cannot continue")
                else:
                    if baseline_obs is None:
                        baseline_obs, baseline_metrics = observations, metrics
                    report = write_audit(baseline_obs, observations, forget, source["audit_cfg"],
                                         vdir / stage, metadata, baseline_metrics, metrics)
            if failure:
                print(f"[candidate evaluation] {stage}: {status}: {failure['message']}", flush=True)
                _write(vdir / stage / "evaluation_failure.json", {"stage": stage, "evaluation_status": status,
                       **failure, "numerical_diagnostic": diagnostic, "checkpoint_unchanged": True,
                       "task_metrics_available": bool(metrics), "v1_report_written": False,
                       "formal_audit": False})
            manifest["stages"][stage] = {**metadata, "checkpoint": row["checkpoint"],
                "task_metrics": metrics, "evaluation_status": status, "numerical_diagnostic": diagnostic}
            coverage.append({"stage": stage, "unlearner": row["method"], "recovery": "none",
                "evaluation_status": status, "included_in_coverage": True,
                "audit_evaluable": status == "evaluated", "failure_kind": failure["kind"] if failure else None,
                "failure_reason": failure["message"] if failure else None,
                **{key: metrics.get(key) for key in ("all_acc", "retain_acc", "forget_acc", "all_loss",
                    "retain_loss", "forget_loss", "retain_num_examples", "forget_num_examples")},
                "status": report["status"] if report else None,
                "exploratory_status": report["exploratory_status"] if report else None,
                "unlearning_time_sec": row.get("unlearning_time_sec")})
        _write(vdir / "checkpoint_manifest.json", manifest)
        unevaluable = [row["stage"] for row in coverage if not row["audit_evaluable"]]
        coverage_status = "complete_with_unevaluable_models" if unevaluable else "complete"
        coverage_document = {"schema_version": EVALUATION_SCHEMA, "coverage_status": coverage_status,
            "all_stage_count": len(coverage), "candidate_count": len(source["inventory"]["candidates"]),
            "evaluable_stage_count": len(coverage) - len(unevaluable), "unevaluable_stages": unevaluable,
            "missing_models_are_audit_successes": False, "all_prespecified_endpoints_retained": True,
            "formal_audit": False, "stages": coverage}
        _write(vdir / "evaluation_coverage.json", coverage_document)
        for destination in (vdir / "evaluation_coverage.csv", vdir / "summary.csv", output / "metrics.csv"):
            pd.DataFrame(coverage).to_csv(destination, index=False)
        _assert_unchanged(source)
        if _code_hashes() != code_hashes:
            raise ValueError("Evaluation source code changed during computation")
        for name in copied:
            if file_hash(output / name) != source["source_hashes"][name]:
                raise ValueError(f"Evaluation mutated copied frozen artifact: {name}")
        _status(output, {"status": "main_candidate_evaluation_recovery_complete",
            "schema_version": EVALUATION_SCHEMA, "coverage_status": coverage_status,
            "candidate_count": len(source["inventory"]["candidates"]),
            "candidate_inventory_sha256": source["source_hashes"]["candidate_inventory.json"],
            "source_full_parameters_sha256": source["full_parameters_sha256"], "formal_audit": False,
            "training_performed": False, "checkpoint_mutation": False})
    except BaseException as exc:
        _status(output, {"status": "main_candidate_evaluation_failed", "error": str(exc),
                         "error_type": type(exc).__name__, "formal_audit": False})
        raise
    if diagnose_v2:
        from fulbase.verification.protocol_v2_io import run_protocol_v2_diagnostic
        healthy = [row["stage"] for row in coverage if row["audit_evaluable"]]
        result = run_protocol_v2_diagnostic([str(output)], stages=healthy,
            out_dir=output.parent / (output.name + "_v2"), run_name="recovered_candidates",
            bootstrap_samples=bootstrap_samples, ablation_config=ablation_config, stratified_margin=True)
        diagnostic_output = Path(result["output_dir"])
        coverage_hashes = {}
        for filename in ("evaluation_coverage.json", "evaluation_coverage.csv"):
            origin = vdir / filename
            coverage_hashes[filename] = file_hash(origin)
            # Exclusive creation preserves the diagnostic runner's existing files.
            with origin.open("rb") as stream, (diagnostic_output / filename).open("xb") as destination:
                shutil.copyfileobj(stream, destination)
            if file_hash(origin) != coverage_hashes[filename] or file_hash(diagnostic_output / filename) != coverage_hashes[filename]:
                raise ValueError("Complete evaluation coverage changed while copying to v2 output")
        _write(Path(result["output_dir"]) / "evaluation_coverage_reference.json", {
            "coverage_file": str(vdir / "evaluation_coverage.json"),
            "coverage_sha256": coverage_hashes["evaluation_coverage.json"],
            "copied_coverage_sha256": coverage_hashes,
            "diagnosed_stages": healthy, "unevaluable_stages": unevaluable,
            "all_prespecified_endpoints_retained": True, "formal_audit": False})
        _write(output / "v2_diagnostic_reference.json", result)
        print(f"[candidate evaluation] v2 output: {result['output_dir']}", flush=True)
    return output
