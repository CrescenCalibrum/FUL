"""Fixed-Full main-experiment candidates, with evaluation after all endpoints.

The original algorithms and audit rules are called unchanged. This runner never
trains Full/Retrain, selects a winner, or reads independent audit observations.
Natural training data are needed by PGA; test samples are evaluated only after
the complete, prespecified candidate inventory has been written.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
from numbers import Integral
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch

from fulbase.data import load_federated_data
from fulbase.experiment.io import RunLogger
from fulbase.experiment.pipeline import _evaluate_params, _timed_call
from fulbase.models import build_model
from fulbase.unlearning.base import UnlearningContext, UnlearningResult
from fulbase.unlearning.history_validation import validate_history_request
from fulbase.unlearning.marker_boundary import BoundaryConfig, train_boundary_head
from fulbase.unlearning.pga import pga_unlearn
from fulbase.unlearning.registry import build_unlearner
from fulbase.utils import get_device, set_model_parameters, state_change_report
from fulbase.verification.audit import file_hash, observe_markers, parameter_hash, write_audit
from fulbase.verification.experiment import _corrupt
from fulbase.verification.markers import MarkerDataset, derived_seed
from fulbase.verification.pga_diagnostic import load_source_run, _validate_data


SCHEMA = "rp2-main-candidates-v1"
METHODS = ("history_subtract", "federaser_lite", "pga")
C_STAGE = "fixed_C_lambda05_pre_recovery"
D_STAGE = "D_margin_lambda8_pre_recovery"
PGA_BUDGET = dict(pga_steps=50, pga_lr=.0001, pga_project_ratio=.01,
                  pga_layers="head", pga_data_scope="real_and_markers")
C_BUDGET = dict(steps=30, lr=.01, project_ratio=.1, layers="head", data_scope="markers_only",
                objective="margin_capped_retain", margin_raw_drop=3., retain_weight=.5,
                retain_weight_mode="fixed", trace_steps=True)
D_BUDGET = BoundaryConfig(retain_weight=8.)
CODE_FILES = (
    "fulbase/verification/main_candidates.py", "scripts/run_verification_main_candidates.py",
    "fulbase/verification/pga_diagnostic.py", "fulbase/verification/experiment.py",
    "fulbase/unlearning/base.py", "fulbase/unlearning/registry.py",
    "fulbase/unlearning/history_validation.py", "fulbase/unlearning/history_subtract.py",
    "fulbase/unlearning/federaser_lite.py", "fulbase/unlearning/pga.py",
    "fulbase/unlearning/marker_capped.py", "fulbase/unlearning/marker_boundary.py",
    "fulbase/experiment/pipeline.py", "fulbase/experiment/io.py", "fulbase/models.py",
    "fulbase/utils.py", "fulbase/data.py", "fulbase/train.py",
    "fulbase/verification/markers.py", "fulbase/verification/audit.py",
    "fulbase/verification/statistics.py",
)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _status(output, value):
    target = output / "run_status.json"
    pending = target.with_suffix(".pending")
    with pending.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
    pending.replace(target)


def _code_hashes():
    root = Path(__file__).resolve().parents[2]
    return {name: file_hash(root / name) for name in CODE_FILES}


def _state_arrays(arrays, reference, name):
    if not isinstance(arrays, list) or len(arrays) != len(reference):
        raise ValueError(f"Invalid history state length: {name}")
    if any(not isinstance(a, (np.ndarray, np.generic)) or a.shape != b.shape or
           not np.issubdtype(a.dtype, np.number) or not np.isfinite(a).all()
           for a, b in zip(arrays, reference)):
        raise ValueError(f"Invalid/nonfinite history state: {name}")


def check_main_source(run_dir):
    """Check trusted source bytes/history/initial; no data loading or output writes."""
    source = load_source_run(run_dir)
    root, cfg = source["root"], source["cfg"]
    for key, expected in PGA_BUDGET.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"Main experiment requires fixed {key}={expected}")
    if (cfg.unlearn_alpha != .5 or cfg.federaser_damping != 1. or
            cfg.federaser_calibrate is not True or cfg.corruption_scale != .5):
        raise ValueError("Main experiment history/corruption budgets differ from the fixed protocol")
    if cfg.fraction_fit != 1. or source["audit_cfg"].max_retain_accuracy_drop != .03:
        raise ValueError("Keep full participation and original .03 task audit unchanged")
    if not source["markers"].enabled:
        raise ValueError("Main candidates require enabled original training markers")
    forget = [int(x.strip()) for x in cfg.forget_clients.split(",")]
    if not forget or len(set(forget)) != len(forget) or not set(forget) < set(range(cfg.num_clients)):
        raise ValueError("Invalid source forget set")
    partition = source["partition"]
    for name in ("train_indices", "test_indices"):
        rows = partition[name]
        flat = [item for row in rows for item in row]
        if (len(rows) != cfg.num_clients or any(not row for row in rows) or
                any(type(i) is not int for i in flat) or sorted(flat) != list(range(len(flat)))):
            raise ValueError("Source partition must cover distinct nonempty clients exactly once")
    required = ["config.json", "data_partition.json", "run_status.json", "history.pt",
                "models/model_initial.pt", "verification/training_result.json",
                "verification/marker_config.json", "verification/audit_config.json",
                "verification/checkpoint_manifest.json", "verification/training_marker_manifest.json"]
    required.append(source["checkpoint"].relative_to(root).as_posix())
    for name in required:
        path = (root / name).resolve()
        if root not in path.parents:
            raise ValueError("Source artifact resolves outside its run")
        if not path.is_file():
            raise FileNotFoundError(f"Missing main-experiment input: {path}; do not retrain silently")
    hashes = {name: file_hash(root / name) for name in required}
    initial = torch.load(root / "models/model_initial.pt", map_location="cpu", weights_only=False)
    _state_arrays(initial, source["params"], "initial")
    result = _read(root / "verification/training_result.json")
    if result.get("initial_parameters_sha256") != parameter_hash(initial):
        raise ValueError("Initial parameters differ from the recorded FedAvg initialization")
    if result.get("final_parameters_sha256") != parameter_hash(source["params"]):
        raise ValueError("Full differs from the recorded FedAvg output")
    history = torch.load(root / "history.pt", map_location="cpu", weights_only=False)
    if not isinstance(history, list) or len(history) != cfg.num_rounds:
        raise ValueError("History does not contain the configured number of rounds")
    validate_history_request(history, forget)
    max_reconstruction_error, max_decomposition_error = 0., 0.
    any_low_precision, target_nonzero = False, False
    for index, item in enumerate(history):
        if item.get("round") != index + 1:
            raise ValueError("History rounds are incomplete or out of order")
        if set(map(str, item.get("participating_client_ids", []))) != set(map(str, range(cfg.num_clients))):
            raise ValueError("History participation differs from the full-participation protocol")
        total, retained = item.get("total_examples"), item.get("retained_examples")
        if (any(isinstance(value, bool) or not isinstance(value, Integral) for value in (total, retained))
                or not 0 < retained < total):
            raise ValueError("Invalid compact-history aggregation counts")
        clients = item.get("client_metrics")
        if not isinstance(clients, list) or len(clients) != cfg.num_clients:
            raise ValueError("Compact history requires one effective count per participating client")
        counts = {}
        for client in clients:
            cid, count = str(client.get("cid")), client.get("num_examples")
            if (cid not in set(map(str, range(cfg.num_clients))) or cid in counts or
                    isinstance(count, bool) or not isinstance(count, Integral) or count <= 0 or
                    count != len(partition["train_indices"][int(cid)])):
                raise ValueError("History effective client counts differ from the fixed training partition")
            counts[cid] = count
        if (sum(counts.values()) != total or
                sum(count for cid, count in counts.items() if int(cid) not in forget) != retained):
            raise ValueError("History aggregation counts do not match the effective client weights")
        retain_weight = retained / total
        for key in ("global_before", "all_update", "retained_update", "forget_update_wrt_all"):
            _state_arrays(item.get(key), source["params"], key)
        before = item["global_before"]
        after = history[index + 1]["global_before"] if index + 1 < len(history) else source["params"]
        _state_arrays(after, source["params"], "next global state")
        if index == 0 and any(not np.array_equal(a.astype(b.dtype), b) for a, b in zip(initial, before)):
            raise ValueError("First history broadcast differs from the saved initial model")
        for a, b, update, target, retained_update, reference in zip(
                before, after, item["all_update"], item["forget_update_wrt_all"],
                item["retained_update"], source["params"]):
            if not np.issubdtype(reference.dtype, np.floating):
                if np.any(update) or np.any(target) or np.any(retained_update):
                    raise ValueError("Compact history must not subtract nonfloating buffers")
                continue
            low = any(value.dtype == np.float16 for value in (a, b, update, target, retained_update))
            any_low_precision |= low
            decomposed = target.astype(np.float64) + retained_update.astype(np.float64) * retain_weight
            error = float(np.max(np.abs(decomposed - update))) if update.size else 0.
            max_decomposition_error = max(max_decomposition_error, error)
            if not np.allclose(decomposed, update, rtol=.005 if low else 2e-5,
                               atol=.005 if low else 2e-6):
                raise ValueError("History retained/target update decomposition is inconsistent")
            reconstructed = a.astype(np.float64) + update.astype(np.float64)
            error = float(np.max(np.abs(reconstructed - b))) if b.size else 0.
            max_reconstruction_error = max(max_reconstruction_error, error)
            if not np.allclose(reconstructed, b, rtol=.005 if low else 2e-5,
                               atol=.005 if low else 2e-6):
                raise ValueError("History update chain is inconsistent with the bound Full")
            target_nonzero |= bool(np.any(target != 0))
    if not target_nonzero:
        raise ValueError("Target historical updates are all zero")
    source.update(initial_parameters=initial, history=history, source_hashes=hashes,
                  history_sha256=hashes["history.pt"], initial_checkpoint_sha256=hashes["models/model_initial.pt"],
                  initial_parameters_sha256=parameter_hash(initial),
                  full_parameters_sha256=parameter_hash(source["params"]),
                  history_validation={"round_count": len(history), "target_update_nonzero": True,
                      "low_precision_history": any_low_precision,
                      "max_floating_update_chain_error": max_reconstruction_error,
                      "max_update_decomposition_error": max_decomposition_error,
                      "effective_client_weights_match_partition": True,
                      "history_original_digest_available": False,
                      "binding": "runtime_file_digest_plus_saved_initial_full_and_history_consistency"})
    return source


def _assert_source(source, code_hashes):
    for name, digest in source["source_hashes"].items():
        if file_hash(source["root"] / name) != digest:
            raise ValueError(f"Source artifact changed during candidate generation: {name}")
    if _code_hashes() != code_hashes:
        raise ValueError("Candidate implementation changed during execution")


def run_main_candidates(run_dir, *, out_dir="runs", run_name=None, device="auto", data_dir=None,
                        include_marker_controls=False, fd=None, threads=2, run_directory=None):
    if type(threads) is not int or threads < 1 or type(include_marker_controls) is not bool:
        raise ValueError("Invalid threads or marker-control flag")
    torch.set_num_threads(threads)
    source = check_main_source(run_dir)
    resolved = get_device(device)
    name = run_name or f"rp2_main_candidates_seed{source['cfg'].seed}"
    if not name or Path(name).name != name or any(c in name for c in "/\\:"):
        raise ValueError("run_name must be a simple directory suffix")
    output = (Path(run_directory) if run_directory is not None else
              Path(out_dir) / f"{datetime.now():%Y%m%d-%H%M%S-%f}_{name}").resolve()
    if output == source["root"] or source["root"] in output.parents or output in source["root"].parents:
        raise ValueError("Candidate output must be separate from the source run")
    if output.exists():
        raise FileExistsError(f"Candidate output already exists: {output}")
    cfg = replace(source["cfg"], run_name=name, out_dir=str(out_dir), eval_device=str(resolved),
                  methods=",".join(METHODS), include_fakes=True, with_recovery=False)
    if data_dir is not None:
        cfg.data_dir = str(data_dir)
    forget = [int(x.strip()) for x in cfg.forget_clients.split(",")]
    retain = [cid for cid in range(cfg.num_clients) if cid not in forget]
    nclasses = source["manifest"]["num_classes"]
    code_hashes = _code_hashes()
    if fd is None:
        if cfg.synthetic_smoke:
            from fulbase.verification.synthetic import synthetic_data
            fd = synthetic_data(cfg.num_clients, cfg.seed)
        else:
            fd = load_federated_data(cfg.dataset, cfg.data_dir, cfg.num_clients, cfg.partition, cfg.alpha, cfg.seed)
    _validate_data(fd, source)
    output.mkdir(parents=True, exist_ok=False)
    logger, vdir = RunLogger(output), output / "verification"
    vdir.mkdir()
    _write(output / "config.json", asdict(cfg))
    _write(vdir / "marker_config.json", asdict(source["markers"]))
    _write(vdir / "audit_config.json", asdict(source["audit_cfg"]))
    for relative in ("data_partition.json", "verification/training_result.json",
                     "verification/training_marker_manifest.json"):
        shutil.copyfile(source["root"] / relative, output / relative)
    shutil.copyfile(source["checkpoint"], output / "models/model_full.pt")
    budgets = {"history_subtract": {"alpha": .5}, "federaser_lite": {"damping": 1., "calibrate": True},
               "pga": PGA_BUDGET, "fake_corrupt": {"scale": .5},
               "fixed_C": C_BUDGET if include_marker_controls else None,
               "D_margin": asdict(D_BUDGET) if include_marker_controls else None}
    _write(output / "main_candidate_config.json", {"schema_version": SCHEMA, "budgets": budgets,
        "include_marker_controls": include_marker_controls, "recovery": "not_run",
        "selection_policy": "all_prespecified_endpoints_no_winner_selection",
        "task_tolerance_policy": "original_.03_reports_preserved_new_.035_summary_external"})
    _write(output / "main_candidate_provenance.json", {"schema_version": SCHEMA,
        "source_run": str(source["root"]), "source_hashes": source["source_hashes"],
        "code_hashes": code_hashes, "source_full_parameters_sha256": source["full_parameters_sha256"],
        "history_validation": source["history_validation"],
        "optimization_access": "history_and_training_data_or_training_markers_as_declared_per_method",
        "test_sample_access": "after_all_candidate_checkpoints_and_inventory_are_fixed",
        "independent_audit_access": False, "formal_audit": False})
    _status(output, {"status": "main_candidates_running", "phase": "generation", "formal_audit": False})
    full = [array.copy() for array in source["params"]]
    full_hash = parameter_hash(full)
    inventory = []

    def context():
        return UnlearningContext(fd, cfg.dataset, nclasses,
            [a.copy() for a in source["initial_parameters"]], [a.copy() for a in full], source["history"],
            forget, retain, cfg.batch_size, resolved, cfg, source["markers"])

    def save_endpoint(stage, result, seconds, behavior, access):
        if parameter_hash(full) != full_hash or parameter_hash(source["params"]) != full_hash:
            raise ValueError("Candidate generation mutated the bound Full")
        _state_arrays(result.parameters, full, stage)
        model = build_model(cfg.dataset, nclasses)
        set_model_parameters(model, result.parameters)
        if parameter_hash([v.detach().cpu().numpy() for v in model.state_dict().values()]) != parameter_hash(result.parameters):
            raise ValueError("Candidate model round-trip changes the parameter state")
        change = state_change_report(model, full)
        if stage in {"pga_pre_recovery", C_STAGE, D_STAGE}:
            if change["changed_buffers"] or any(not x.startswith("classifier.") for x in change["changed_parameters"]):
                raise ValueError("Head-only candidate changed backbone parameters or buffers")
        logger.save_model_params(f"model_{stage}", result.parameters)
        checkpoint = output / "models" / f"model_{stage}.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if parameter_hash(saved) != parameter_hash(result.parameters):
            raise ValueError("Saved endpoint differs from generated parameters")
        _write(vdir / stage / "method_logs.json", result.logs)
        if "step_trace" in result.logs:
            pd.DataFrame(result.logs["step_trace"]).to_csv(vdir / stage / "step_trace.csv", index=False)
        row = {"stage": stage, "method": result.name, "behavior": behavior, "optimization_access": access,
               "checkpoint": checkpoint.relative_to(output).as_posix(), "checkpoint_sha256": file_hash(checkpoint),
               "parameters_sha256": parameter_hash(saved), "source_full_parameters_sha256": full_hash,
               "model_state_change_from_full": change, "unlearning_time_sec": seconds,
               "strong_training_eligible": result.logs.get("training_eligible"),
               "predeclared_endpoint_included": True, "training_selected": False}
        inventory.append(row)

    try:
        print(f"[main candidates] Output: {output}\n[main candidates] Fixed Full: {full_hash}", flush=True)
        training = {cid: MarkerDataset(source["markers"], cid, cfg.dataset, "train",
                    derived_seed(cfg.seed, "train-markers"), nclasses) for cid in range(cfg.num_clients)}
        expected = _read(vdir / "training_marker_manifest.json")
        rows = [row for ds in training.values() for row in ds.manifest()]
        fields = ("cid", "role", "marker_id", "family_id", "label")
        if sorted(tuple(r[k] for k in fields) for r in rows) != sorted(tuple(r[k] for k in fields) for r in expected):
            raise ValueError("Training marker identities differ from source")
        for method in METHODS:
            ctx = context()
            print(f"[main candidates] {method}", flush=True)
            if method == "pga":
                result, seconds = _timed_call(lambda: pga_unlearn(ctx, steps=cfg.pga_steps, lr=cfg.pga_lr,
                    project_ratio=cfg.pga_project_ratio, layers=cfg.pga_layers,
                    data_scope=cfg.pga_data_scope, trace_steps=True), resolved)
            else:
                result, seconds = _timed_call(lambda: build_unlearner(method).unlearn(ctx), resolved)
            if parameter_hash(ctx.final_parameters) != full_hash:
                raise ValueError("An unlearner modified its input Full")
            save_endpoint(f"{method}_pre_recovery", result, seconds, "candidate",
                          "target_training_data_and_training_markers" if method == "pga" else "compact_training_history")
        save_endpoint("fake_noop_pre_recovery", UnlearningResult("fake_noop", [a.copy() for a in full],
            {"behavior": "no_parameter_change"}), 0., "fake", "none")
        result, seconds = _timed_call(lambda: _corrupt(context(), .5), resolved)
        save_endpoint("fake_corrupt_pre_recovery", result, seconds, "fake", "parameter_noise_no_data")
        if include_marker_controls:
            result, seconds = _timed_call(lambda: pga_unlearn(context(), **C_BUDGET), resolved)
            save_endpoint(C_STAGE, result, seconds, "marker_manipulation_control", "target_and_retained_training_markers")
            model = build_model(cfg.dataset, nclasses)
            set_model_parameters(model, full)
            result, seconds = _timed_call(lambda: train_boundary_head(model, training, forget,
                config=D_BUDGET, device=str(resolved), batch_size=cfg.batch_size), resolved)
            save_endpoint(D_STAGE, result, seconds, "margin_manipulation_control", "target_and_retained_training_markers")
        _assert_source(source, code_hashes)
        _write(output / "candidate_inventory.json", {"schema_version": SCHEMA, "status": "all_endpoints_fixed",
            "source_full_parameters_sha256": full_hash, "candidates": inventory,
            "selection_policy": "all_prespecified_endpoints_no_winner_selection",
            "selected_trial": None, "formal_audit": False, "frozen_utc": datetime.now(timezone.utc).isoformat()})
        inventory_hash = file_hash(output / "candidate_inventory.json")
        _status(output, {"status": "main_candidates_running", "phase": "evaluation",
                        "candidate_inventory_sha256": inventory_hash, "formal_audit": False})
        calibration = {cid: MarkerDataset(source["markers"], cid, cfg.dataset, "calibration",
                         cfg.calibration_seed, nclasses) for cid in range(cfg.num_clients)}
        _write(vdir / "calibration_marker_manifest.json", [row for ds in calibration.values() for row in ds.manifest()])
        manifest = {"schema_version": "marker-experiment-v1", "num_classes": nclasses,
                    "marker_fingerprint": source["markers"].fingerprint(),
                    "partition_sha256": source["manifest"]["partition_sha256"],
                    "data_origin": source["manifest"].get("data_origin", "benchmark"), "stages": {}}
        baseline_obs, baseline_metrics, summary = None, None, []
        records = [{"stage": "full_after_fl", "method": "none", "behavior": "unmodified_full",
                    "optimization_access": "none", "checkpoint": "models/model_full.pt", "unlearning_time_sec": None,
                    "strong_training_eligible": None}, *inventory]
        for row in records:
            stage, checkpoint = row["stage"], output / row["checkpoint"]
            parameters = torch.load(checkpoint, map_location="cpu", weights_only=False)
            metrics = _evaluate_params(stage, parameters, fd, cfg, retain, forget)
            if any(not math.isfinite(float(value)) for value in metrics.values()):
                raise ValueError("Nonfinite task evaluation")
            nr, nf = metrics["retain_num_examples"], metrics["forget_num_examples"]
            if nr <= 0 or nf <= 0 or abs(metrics["all_acc"] -
                    (nr * metrics["retain_acc"] + nf * metrics["forget_acc"]) / (nr + nf)) > 1e-12:
                raise ValueError("ACC weighted identity failed")
            model = build_model(cfg.dataset, nclasses)
            set_model_parameters(model, parameters)
            observations = {cid: observe_markers(model, ds, str(resolved), cfg.batch_size) for cid, ds in calibration.items()}
            metadata = {"stage": stage, "method": row["method"], "behavior": row["behavior"], "role": "calibration",
                        "checkpoint_sha256": file_hash(checkpoint), "parameters_sha256": parameter_hash(parameters),
                        "source_full_parameters_sha256": full_hash, "marker_fingerprint": source["markers"].fingerprint(),
                        "model_state_change_from_full": state_change_report(model, full),
                        "optimization_access": row["optimization_access"], "training_selected": False,
                        "strong_training_eligible": row["strong_training_eligible"], "candidate_inventory_sha256": inventory_hash}
            if baseline_obs is None:
                baseline_obs, baseline_metrics = observations, metrics
            report = write_audit(baseline_obs, observations, forget, source["audit_cfg"], vdir / stage,
                                 metadata, baseline_metrics, metrics)
            manifest["stages"][stage] = {**metadata, "checkpoint": row["checkpoint"], "task_metrics": metrics}
            metric_row = {"unlearner": row["method"], "recovery": "none", **metrics,
                          "status": report["status"], "exploratory_status": report["exploratory_status"],
                          "unlearning_time_sec": row["unlearning_time_sec"]}
            logger.add_metrics(stage, metric_row)
            summary.append({"stage": stage, **metric_row})
        _write(vdir / "checkpoint_manifest.json", manifest)
        pd.DataFrame(summary).to_csv(vdir / "summary.csv", index=False)
        _assert_source(source, code_hashes)
        if file_hash(output / "candidate_inventory.json") != inventory_hash:
            raise ValueError("Evaluation changed the fixed candidate inventory")
        for row in inventory:
            if file_hash(output / row["checkpoint"]) != row["checkpoint_sha256"]:
                raise ValueError("Evaluation changed a fixed candidate checkpoint")
        _status(output, {"status": "development_run_complete", "schema_version": SCHEMA,
                        "candidate_count": len(inventory), "candidate_inventory_sha256": inventory_hash,
                        "source_full_parameters_sha256": full_hash, "formal_audit": False,
                        "recovery": "not_run", "retrain": "not_run", "winner_selection": "not_performed"})
    except BaseException as exc:
        _status(output, {"status": "main_candidates_failed", "error": str(exc), "formal_audit": False})
        raise
    return output
