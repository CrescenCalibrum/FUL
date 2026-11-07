"""Development-only PGA budgets, each starting from a bound, existing Full.

Reads target training data and training markers for PGA; test data and calibration
markers are used only after each candidate is fixed. Never trains FedAvg or tunes
on audit-role markers. Source runs are read-only; no history.pt is required.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime
from itertools import product
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch

from fulbase.data import load_federated_data
from fulbase.experiment.io import RunLogger
from fulbase.experiment.pipeline import _evaluate_params, _timed_call
from fulbase.models import build_model
from fulbase.unlearning.base import UnlearningContext
from fulbase.unlearning.pga import pga_unlearn
from fulbase.utils import get_device, save_json, set_model_parameters, state_change_report
from fulbase.verification.audit import (AuditConfig, file_hash, parameter_hash,
                                      observe_markers, save_observations, write_audit)
from fulbase.verification.experiment import VerificationExperimentConfig
from fulbase.verification.markers import MarkerConfig, MarkerDataset, derived_seed


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _marker_summary(observations):
    """Descriptive pooled sample metrics; not independent training repetitions."""
    obs = list(observations.values())
    logits = np.concatenate([x["logits"] for x in obs]).astype(np.float64)
    labels = np.concatenate([x["labels"] for x in obs])
    finite = bool(np.isfinite(logits).all())
    # Stable CE, with no probability clipping or confidence-threshold tuning.
    if finite:
        shifted = logits - logits.max(axis=1, keepdims=True)
        ce = np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(labels)), labels]
    def mean(key):
        values = np.concatenate([x[key] for x in obs])
        return float(values.mean()) if np.isfinite(values).all() else None
    return {"num_examples": int(len(labels)), "valid": all(x["valid"] for x in obs),
            "accuracy": float((logits.argmax(axis=1) == labels).mean()) if finite else None,
            "ce_loss": float(ce.mean()) if finite else None,
            "mean_margin": mean("margin"), "mean_normalized_margin": mean("normalized_margin")}


def _metric_delta(after, before):
    return after - before if after is not None and before is not None else None


def load_source_run(run_dir):
    """Validate locally trusted run artifacts before any dataset loading or writes."""
    root = Path(run_dir).resolve()
    cfg = VerificationExperimentConfig(**_read(root / "config.json"))
    markers = MarkerConfig(**_read(root / "verification/marker_config.json"))
    audit_cfg = AuditConfig(**_read(root / "verification/audit_config.json"))
    manifest = _read(root / "verification/checkpoint_manifest.json")
    partition = _read(root / "data_partition.json")
    if audit_cfg.protocol_locked:
        raise ValueError("PGA diagnostics require an unlocked development protocol")
    if _read(root / "run_status.json").get("status") not in {
        "development_run_complete", "pga_diagnostic_complete"
    }:
        raise ValueError("Source run is incomplete")
    if manifest["marker_fingerprint"] != markers.fingerprint():
        raise ValueError("Source marker configuration differs from its manifest")
    if manifest["partition_sha256"] != file_hash(root / "data_partition.json"):
        raise ValueError("Source partition hash mismatch")
    entry = manifest["stages"]["full_after_fl"]
    checkpoint = (root / entry["checkpoint"].replace("\\", "/")).resolve()
    if root not in checkpoint.parents:
        raise ValueError("Full checkpoint is outside source run")
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing Full checkpoint: {checkpoint}. Run on the original training "
            "machine or copy the trusted models/model_full.pt with its run metadata; "
            "metrics alone cannot reconstruct Full. history.pt is not needed."
        )
    if file_hash(checkpoint) != entry["checkpoint_sha256"]:
        raise ValueError("Full checkpoint file hash mismatch")
    # Only use locally trusted checkpoints produced by this project.
    params = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if parameter_hash(params) != entry["parameters_sha256"]:
        raise ValueError("Full parameter hash mismatch")
    if any(not np.isfinite(x).all() for x in params):
        raise ValueError("Full contains nonfinite state")
    if (root / "verification/training_result.json").is_file():
        expected = _read(root / "verification/training_result.json")["final_parameters_sha256"]
        if expected != parameter_hash(params):
            raise ValueError("Full differs from recorded final FedAvg output")
    model = build_model(cfg.dataset, manifest["num_classes"])
    set_model_parameters(model, params)
    loaded_change = state_change_report(model, params)
    if loaded_change["changed_parameters"] or loaded_change["changed_buffers"]:
        raise ValueError("Full state changed while loading; check model dtype/architecture")
    for name, value in model.named_buffers():
        if name.endswith("running_var") and (value < 0).any():
            raise ValueError(f"Full has negative BatchNorm variance: {name}")
    return dict(root=root, cfg=cfg, markers=markers, audit_cfg=audit_cfg,
                manifest=manifest, partition=partition, checkpoint=checkpoint, params=params)


def _validate_data(fd, source):
    cfg, partition = source["cfg"], source["partition"]
    if fd.num_classes != source["manifest"]["num_classes"]:
        raise ValueError("Dataset class count differs from source")
    for name, dataset in (("train_indices", fd.trainset), ("test_indices", fd.testset)):
        saved = partition[name]
        if len(saved) != cfg.num_clients or any(not x for x in saved):
            raise ValueError("Source partition has empty/missing clients")
        flat = [i for client in saved for i in client]
        if any(type(i) is not int for i in flat) or sorted(flat) != list(range(len(dataset))):
            raise ValueError("Source partition does not cover the dataset exactly once")
        if getattr(fd, name) != saved:
            raise ValueError("Loaded data partition differs from saved source partition")


def run_pga_diagnostic(run_dir, *, steps=(5, 10, 20), lrs=(1e-4,),
                       project_ratios=(0.01,), layers="last", data_scope="real",
                       device="auto", data_dir=None, out_dir="runs",
                       run_name="pga_diagnostic", fd=None, objectives=("ce",)):
    steps, lrs, project_ratios = list(steps), list(lrs), list(project_ratios)
    objectives = list(objectives)
    if not objectives or any(x not in {"ce", "margin"} for x in objectives):
        raise ValueError("Objectives must be ce or margin")
    if len(set(objectives)) != len(objectives):
        raise ValueError("Duplicate objectives")
    if "margin" in objectives and data_scope != "markers_only":
        raise ValueError("The margin erasure objective requires markers_only training inputs")
    if not steps or any(type(n) is not int or n < 1 for n in steps):
        raise ValueError("Step budgets must be positive integers")
    if not lrs or any(not math.isfinite(x) or x <= 0 for x in lrs):
        raise ValueError("Learning rates must be finite and positive")
    if not project_ratios or any(not math.isfinite(x) or x < 0 for x in project_ratios):
        raise ValueError("Projection ratios must be finite and nonnegative")
    if any(len(set(xs)) != len(xs) for xs in (steps, lrs, project_ratios)):
        raise ValueError("Duplicate diagnostic budgets")
    if layers not in {"head", "last", "all"} or data_scope not in {"real", "real_and_markers", "markers_only"}:
        raise ValueError("Unsupported PGA layer/data scope")
    if not run_name or Path(run_name).name != run_name or any(c in run_name for c in "/\\:"):
        raise ValueError("run_name must be a simple directory suffix")
    source = load_source_run(run_dir)
    markers, audit_cfg, params = source["markers"], source["audit_cfg"], source["params"]
    if data_scope != "real" and not markers.enabled:
        raise ValueError("Marker PGA requires enabled source training markers")
    resolved_device = get_device(device)
    cfg = replace(source["cfg"], eval_device=str(resolved_device), with_recovery=False,
                  include_fakes=False, methods="pga", pga_layers=layers, pga_data_scope=data_scope,
                  pga_steps=steps[0], pga_lr=lrs[0], pga_project_ratio=project_ratios[0])
    if data_dir is not None:
        cfg.data_dir = str(data_dir)
    forget = [int(x.strip()) for x in cfg.forget_clients.split(",")]
    if not forget or len(set(forget)) != len(forget) or any(c < 0 or c >= cfg.num_clients for c in forget):
        raise ValueError("Invalid source forget set")
    retain = [c for c in range(cfg.num_clients) if c not in forget]
    if len(retain) < 2:
        raise ValueError("At least two retained clients are needed")
    if fd is None:
        if cfg.synthetic_smoke:
            from fulbase.verification.synthetic import synthetic_data
            fd = synthetic_data(cfg.num_clients, cfg.seed)
        else:
            fd = load_federated_data(cfg.dataset, cfg.data_dir, cfg.num_clients,
                                     cfg.partition, cfg.alpha, cfg.seed)
    _validate_data(fd, source)
    output = (Path(out_dir) / f"{datetime.now():%Y%m%d-%H%M%S-%f}_{run_name}").resolve()
    if source["root"] == output or source["root"] in output.parents:
        raise ValueError("Diagnostic output must be outside the source run")
    output.mkdir(parents=True, exist_ok=False)
    logger = RunLogger(output)
    vdir = output / "verification"
    vdir.mkdir()
    full_hash = parameter_hash(params)
    cfg.run_name, cfg.out_dir = run_name, str(Path(out_dir))
    save_json(asdict(cfg), output / "config.json")
    save_json(asdict(markers), vdir / "marker_config.json")
    save_json(asdict(audit_cfg), vdir / "audit_config.json")
    shutil.copyfile(source["root"] / "data_partition.json", output / "data_partition.json")
    trials = [dict(steps=n, lr=lr, project_ratio=ratio, layers=layers,
                   data_scope=data_scope, objective=objective)
              for n, lr, ratio, objective in product(steps, lrs, project_ratios, objectives)]
    source_files = [Path(__file__), Path(__file__).parents[1] / "unlearning/pga.py",
                    Path(__file__).parents[1] / "data.py", Path(__file__).parents[1] / "models.py",
                    Path(__file__).with_name("markers.py"), Path(__file__).with_name("audit.py")]
    save_json({"source_run": str(source["root"]), "source_full_parameters_sha256": full_hash,
               "source_checkpoint_sha256": file_hash(source["checkpoint"]),
               "source_config_sha256": file_hash(source["root"] / "config.json"),
               "partition_sha256": source["manifest"]["partition_sha256"],
               "trials": trials, "start_policy": "each_budget_starts_from_identical_full",
               "access": {"optimization": ("target_training_markers_only" if data_scope == "markers_only"
                                             else "target_training_data_and_or_training_markers"),
                          "evaluation": "benchmark_test_data_and_calibration_markers",
                          "attack_diagnostics": "target_training_markers_after_candidate_is_fixed"},
               "objective_definitions": {"ce": "maximize mean cross entropy (original PGA)",
                                         "margin": "maximize negative mean(correct logit - max other logit)"},
               "sampling_policy": "objective-independent seed; same inputs for matched budgets",
               "formal_audit": False, "recovery": "not_run", "retrain": "not_run",
               "config_budget_scope": "config.json holds first numeric budget; trials lists every objective and budget",
               "timing_scope": "PGA execution including diagnostic trace overhead",
               "torch_version": torch.__version__, "numpy_version": np.__version__,
               "device": str(resolved_device),
               "source_hashes": {p.name: file_hash(p) for p in source_files}},
              output / "diagnostic_config.json")
    manifest = {"schema_version": "marker-experiment-v1", "num_classes": fd.num_classes,
                "marker_fingerprint": markers.fingerprint(),
                "partition_sha256": file_hash(output / "data_partition.json"),
                "data_origin": source["manifest"].get("data_origin", "benchmark"), "stages": {}}
    calibration = {cid: MarkerDataset(markers, cid, cfg.dataset, "calibration",
                                      cfg.calibration_seed, fd.num_classes) for cid in range(cfg.num_clients)}
    training_markers = ({cid: MarkerDataset(markers, cid, cfg.dataset, "train",
                                           derived_seed(cfg.seed, "train-markers"), fd.num_classes)
                         for cid in forget} if data_scope == "markers_only" else {})
    if training_markers:
        save_json([row for ds in training_markers.values() for row in ds.manifest()],
                  vdir / "training_marker_manifest.json")
    summaries = []
    baseline_obs, baseline_metrics = None, None
    baseline_train_obs = None

    def record(stage, values, method, logs=None, seconds=None):
        nonlocal baseline_obs, baseline_metrics, baseline_train_obs
        checkpoint_name = "model_full" if stage == "full_after_fl" else f"model_{stage}"
        logger.save_model_params(checkpoint_name, values)
        checkpoint = output / "models" / f"{checkpoint_name}.pt"
        metrics = _evaluate_params(stage, values, fd, cfg, retain, forget)
        if not all(math.isfinite(float(x)) for x in metrics.values()):
            raise ValueError(f"Nonfinite task evaluation at {stage}")
        nr, nf = metrics["retain_num_examples"], metrics["forget_num_examples"]
        if abs(metrics["all_acc"] - (nr*metrics["retain_acc"]+nf*metrics["forget_acc"])/(nr+nf)) > 1e-12:
            raise ValueError("ACC weighted identity failed")
        model = build_model(cfg.dataset, fd.num_classes)
        set_model_parameters(model, values)
        observations = {cid: observe_markers(model, ds, str(resolved_device), cfg.batch_size)
                        for cid, ds in calibration.items()}
        metadata = {"stage": stage, "method": method, "role": "calibration",
                    "behavior": ("unmodified_full" if method == "none" else
                                 "marker_erasure_attempt" if data_scope == "markers_only" else "candidate"),
                    "objective": logs["objective"] if logs else None,
                    "checkpoint_sha256": file_hash(checkpoint), "parameters_sha256": parameter_hash(values),
                    "source_full_parameters_sha256": full_hash,
                    "marker_fingerprint": markers.fingerprint(),
                    "model_state_change_from_full": state_change_report(model, params)}
        if baseline_obs is None:
            baseline_obs, baseline_metrics = observations, metrics
        report = write_audit(baseline_obs, observations, forget, audit_cfg, vdir / stage,
                             metadata, baseline_metrics, metrics)
        marker_metrics = {}
        if training_markers:
            # These diagnostic observations never enter the optimization context or audit tests.
            train_obs = {cid: observe_markers(model, ds, str(resolved_device), cfg.batch_size)
                         for cid, ds in training_markers.items()}
            if baseline_train_obs is None:
                baseline_train_obs = train_obs
            train_rows = [{"phase": phase, "cid": cid, "role": "train", **_marker_summary({cid: obs})}
                          for phase, items in (("full", baseline_train_obs), ("after", train_obs))
                          for cid, obs in items.items()]
            pd.DataFrame(train_rows).to_csv(vdir / stage / "training_marker_metrics.csv", index=False)
            save_observations(baseline_train_obs, vdir / stage / "training_marker_full_observations.npz")
            save_observations(train_obs, vdir / stage / "training_marker_after_observations.npz")
            train_before, train_after = _marker_summary(baseline_train_obs), _marker_summary(train_obs)
            cal_before = _marker_summary({cid: baseline_obs[cid] for cid in forget})
            cal_after = _marker_summary({cid: observations[cid] for cid in forget})
            marker_metrics = {
                "train_marker_acc_full": train_before["accuracy"],
                "train_marker_acc_after": train_after["accuracy"],
                "train_marker_valid_full": train_before["valid"],
                "train_marker_valid_after": train_after["valid"],
                "train_marker_margin_delta": _metric_delta(train_after["mean_margin"], train_before["mean_margin"]),
                "train_marker_normalized_margin_delta": _metric_delta(
                    train_after["mean_normalized_margin"], train_before["mean_normalized_margin"]),
                "calibration_target_marker_acc_full": cal_before["accuracy"],
                "calibration_target_marker_acc_after": cal_after["accuracy"],
                "calibration_target_marker_valid_full": cal_before["valid"],
                "calibration_target_marker_valid_after": cal_after["valid"],
                "calibration_target_marker_normalized_margin_delta": _metric_delta(
                    cal_after["mean_normalized_margin"], cal_before["mean_normalized_margin"]),
            }
        if logs is not None:
            save_json(logs, vdir / stage / "method_logs.json")
            pd.DataFrame(logs["step_trace"]).to_csv(vdir / stage / "step_trace.csv", index=False)
        manifest["stages"][stage] = {**metadata, "checkpoint": checkpoint.relative_to(output).as_posix(),
                                     "task_metrics": metrics}
        save_json(manifest, vdir / "checkpoint_manifest.json")
        extra = ({k: logs[k] for k in ("steps", "lr", "project_ratio", "layers", "data_scope", "objective")} if logs else {})
        row = {"unlearner": method, "recovery": "none", **metrics, **extra, **marker_metrics,
               "source_full_parameters_sha256": full_hash, "unlearning_time_sec": seconds,
               "status": report["status"], "exploratory_status": report["exploratory_status"]}
        logger.add_metrics(stage, row)
        summaries.append({"stage": stage, **row})
        pd.DataFrame(summaries).to_csv(vdir / "summary.csv", index=False)

    save_json({"status": "pga_diagnostic_running", "formal_audit": False}, output / "run_status.json")
    print(f"[pga diagnostic] Output: {output}\n[pga diagnostic] Fixed Full: {full_hash}", flush=True)
    try:
        record("full_after_fl", params, "none")
        for trial in trials:
            method = f"marker_{trial['objective']}_erase" if data_scope == "markers_only" else "pga"
            stage = (f"{method}_{layers}_{data_scope}_steps{trial['steps']}_lr{trial['lr']}"
                     f"_ratio{trial['project_ratio']}_pre_recovery")
            print(f"[pga diagnostic] {stage}", flush=True)
            # Initial training weights/history are not used by PGA; only Full is loaded.
            ctx = UnlearningContext(fd, cfg.dataset, fd.num_classes, [],
                                    [p.copy() for p in params], [], forget, retain,
                                    cfg.batch_size, resolved_device, cfg, markers)
            result, seconds = _timed_call(lambda: pga_unlearn(ctx, **trial, trace_steps=True), resolved_device)
            if parameter_hash(ctx.final_parameters) != full_hash or parameter_hash(params) != full_hash:
                raise RuntimeError("PGA mutated the bound Full")
            record(stage, result.parameters, method, result.logs, seconds)
        save_json({"status": "pga_diagnostic_complete", "formal_audit": False,
                   "num_trials": len(trials), "source_full_parameters_sha256": full_hash}, output / "run_status.json")
    except BaseException as exc:
        save_json({"status": "pga_diagnostic_failed", "error": str(exc), "formal_audit": False},
                  output / "run_status.json")
        raise
    return output
