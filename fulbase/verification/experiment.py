from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from fulbase.data import load_federated_data
from fulbase.dp.prototype import collect_dp_prototypes
from fulbase.experiment.io import RunLogger
from fulbase.experiment.pipeline import ExperimentConfig, _evaluate_params, _eval_device, _timed_call
from fulbase.flower.serial import run_fedavg
from fulbase.models import build_model
from fulbase.recovery.prototype_lda import recover_with_prototype_lda
from fulbase.unlearning.base import UnlearningContext, UnlearningResult
from fulbase.unlearning.registry import build_unlearner
from fulbase.unlearning.pga import pga_unlearn
from fulbase.utils import save_json, set_seed, get_model_parameters, set_model_parameters, state_change_report
from fulbase.verification.markers import MarkerConfig, MarkerDataset, derived_seed
from fulbase.verification.audit import (AuditConfig, observe_markers, write_audit,
                                       parameter_hash, file_hash, detection_summary)


@dataclass
class VerificationExperimentConfig(ExperimentConfig):
    backend: str = "flower"
    methods: str = "history_subtract,federaser_lite,pga,retrain_fl"
    pga_data_scope: str = "real_and_markers"
    fake_steps: int = 20
    fake_lr: float = 0.001
    corruption_scale: float = 0.5
    include_fakes: bool = True
    with_recovery: bool = False
    calibration_seed: int = 314159
    synthetic_smoke: bool = False
    federaser_bn_policy: str = "legacy_replay"


def validate_config(cfg, markers):
    if cfg.federaser_bn_policy not in {"legacy_replay", "retained_endpoint"}:
        raise ValueError("federaser_bn_policy must be legacy_replay or retained_endpoint")
    methods = [x.strip() for x in cfg.methods.split(",") if x.strip()]
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("Choose distinct candidate methods")
    for method in methods:
        build_unlearner(method)
    if "retrain_oracle" in methods:
        raise ValueError("Centralized retrain lacks marker objectives; use retrain_fl for this protocol")
    if cfg.fraction_fit != 1.0:
        raise ValueError("First marker protocol requires full participation")
    if cfg.num_rounds < 1 or cfg.local_epochs < 1 or cfg.batch_size < 1 or cfg.num_clients < 3:
        raise ValueError("Positive training budget and at least three clients are required")
    if cfg.fake_steps < 0 or cfg.fake_lr <= 0 or cfg.corruption_scale < 0:
        raise ValueError("Invalid fake-control budget")
    forget = [int(x.strip()) for x in cfg.forget_clients.split(",") if x.strip()]
    if not forget or len(set(forget)) != len(forget) or any(x < 0 or x >= cfg.num_clients for x in forget):
        raise ValueError("Invalid forget clients")
    if cfg.num_clients - len(forget) < 2:
        raise ValueError("Retained geometry requires at least two retained clients")
    if cfg.pga_data_scope != "real" and not markers.enabled:
        raise ValueError("Marker PGA requires enabled markers; use real scope for unmarked control")
    if cfg.include_fakes and not markers.enabled:
        raise ValueError("Marker-erasure control requires enabled markers")
    return methods, forget


def _corrupt(ctx, scale):
    model = build_model(ctx.dataset, ctx.num_classes)
    set_model_parameters(model, ctx.final_parameters)
    rng = np.random.default_rng(derived_seed(ctx.cfg.seed, "corrupt"))
    with torch.no_grad():
        for param in model.parameters():
            std = max(float(param.std(unbiased=False)), 0.01)
            param.add_(torch.tensor(rng.normal(0, std * scale, tuple(param.shape)), dtype=param.dtype))
    return UnlearningResult("fake_corrupt", get_model_parameters(model),
                            {"behavior": "parameter_noise", "scale": scale,
                             **state_change_report(model, ctx.final_parameters)})


def run_verification_experiment(cfg, markers=None, audit_cfg=None, fd=None, event_callback=None, run_directory=None):
    def phase(name, status, stage, elapsed_sec=None):
        if event_callback is not None:
            event_callback(name, status, stage, elapsed_sec)
    markers = markers or MarkerConfig(enabled=True)
    audit_cfg = audit_cfg or AuditConfig()
    if audit_cfg.protocol_locked:
        raise ValueError("This runner develops/calibrates a protocol; use independent audit for a locked protocol")
    methods, forget = validate_config(cfg, markers)
    set_seed(cfg.seed)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = Path(run_directory) if run_directory is not None else Path(cfg.out_dir) / f"{stamp}_{cfg.run_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    logger = RunLogger(run_dir)
    vdir = run_dir / "verification"
    vdir.mkdir()
    save_json(asdict(cfg), run_dir / "config.json")
    save_json(asdict(markers), vdir / "marker_config.json")
    save_json(asdict(audit_cfg), vdir / "audit_config.json")
    if fd is None:
        fd = load_federated_data(cfg.dataset, cfg.data_dir, cfg.num_clients, cfg.partition, cfg.alpha, cfg.seed)
    if len(fd.train_indices) != cfg.num_clients or len(fd.test_indices) != cfg.num_clients:
        raise ValueError("Federated data client count differs from configuration")
    if any(not x for x in fd.train_indices) or any(not x for x in fd.test_indices):
        raise ValueError("This protocol requires nonempty train/test sets per client")
    retain = [cid for cid in range(cfg.num_clients) if cid not in forget]
    save_json({"train_indices": fd.train_indices, "test_indices": fd.test_indices,
               "rule": "fixed_original_client_ids", "num_classes": fd.num_classes,
               "data_origin": "synthetic_smoke_only" if cfg.synthetic_smoke else "benchmark"}, run_dir / "data_partition.json")
    device = _eval_device(cfg)
    print(f"[verification] Train Full: backend={cfg.backend}, clients={cfg.num_clients}", flush=True)
    phase("TRAIN", "STARTED", "full_after_fl")
    result, train_sec = _timed_call(lambda: run_fedavg(
        backend=cfg.backend, fd=fd, dataset=cfg.dataset, num_clients=cfg.num_clients,
        num_rounds=cfg.num_rounds, fraction_fit=cfg.fraction_fit, fraction_evaluate=cfg.fraction_evaluate,
        min_fit_clients=cfg.num_clients, min_evaluate_clients=1, batch_size=cfg.batch_size,
        local_epochs=cfg.local_epochs, lr=cfg.lr, client_device=cfg.client_device,
        client_cpus=cfg.client_cpus, client_gpus=cfg.client_gpus, forget_client_ids=forget,
        marker_config=markers, training_seed=cfg.seed), device)
    phase("TRAIN", "COMPLETED", "full_after_fl", train_sec)
    save_json({"backend": cfg.backend, "rounds": cfg.num_rounds,
               "initial_parameters_sha256": parameter_hash(result.initial_parameters),
               "final_parameters_sha256": parameter_hash(result.final_parameters)},
              run_dir / "verification" / "training_result.json")
    logger.save_model_params("model_initial", result.initial_parameters)
    logger.save_obj("history.pt", result.history)
    if result.history and sum(item["forget_update_norm"] for item in result.history) == 0:
        raise RuntimeError("Target historical update is zero; cannot validate history-based removal")
    ctx = UnlearningContext(fd, cfg.dataset, fd.num_classes, result.initial_parameters,
                            result.final_parameters, result.history, forget, retain,
                            cfg.batch_size, device, cfg, markers)
    calibration = {cid: MarkerDataset(markers, cid, cfg.dataset, "calibration", cfg.calibration_seed,
                                     fd.num_classes) for cid in range(cfg.num_clients)}
    manifest = {"schema_version": "marker-experiment-v1", "num_classes": fd.num_classes,
                "marker_fingerprint": markers.fingerprint(),
                "partition_sha256": file_hash(run_dir / "data_partition.json"),
                "data_origin": "synthetic_smoke_only" if cfg.synthetic_smoke else "benchmark",
                "stages": {}}
    save_json([row for ds in calibration.values() for row in ds.manifest()], vdir / "calibration_marker_manifest.json")
    train_manifest = [row for cid in range(cfg.num_clients)
                      for row in MarkerDataset(markers, cid, cfg.dataset, "train",
                      derived_seed(cfg.seed, "train-markers"), fd.num_classes).manifest()]
    save_json(train_manifest, vdir / "training_marker_manifest.json")
    baseline_obs, baseline_metrics = None, None
    summaries, overhead = [], [{"stage": "full_after_fl", "operation": "full_training", "time_sec": train_sec}]

    def record(stage, params, method, recovery="none", behavior="candidate", logs=None):
        nonlocal baseline_obs, baseline_metrics
        checkpoint = "model_full" if stage == "full_after_fl" else f"model_{stage}"
        logger.save_model_params(checkpoint, params)
        model_path = run_dir / "models" / f"{checkpoint}.pt"
        metrics = _evaluate_params(stage, params, fd, cfg, retain, forget)
        nr, nf = metrics["retain_num_examples"], metrics["forget_num_examples"]
        weighted = (nr*metrics["retain_acc"] + nf*metrics["forget_acc"]) / (nr+nf)
        if abs(weighted - metrics["all_acc"]) > 1e-12:
            raise RuntimeError("ACC weighted identity failed")
        for item in overhead:
            if item["stage"] == stage and item["operation"] in {"full_training", "unlearning", "recovery"}:
                metrics[item["operation"] + "_time_sec"] = item["time_sec"]
        logger.add_metrics(stage, {"unlearner": method, "recovery": recovery, **metrics})
        model = build_model(cfg.dataset, fd.num_classes)
        set_model_parameters(model, params)
        phase("DEVELOPMENT_AUDIT", "STARTED", stage)
        audit_started = time.perf_counter()
        observations, seconds = _timed_call(
            lambda: {cid: observe_markers(model, ds, device, cfg.batch_size) for cid, ds in calibration.items()}, device)
        change = state_change_report(model, ctx.final_parameters)
        metadata = {"stage": stage, "method": method, "behavior": behavior,
                    "checkpoint_sha256": file_hash(model_path), "parameters_sha256": parameter_hash(params),
                    "marker_fingerprint": markers.fingerprint(), "role": "calibration",
                    "data_origin": manifest["data_origin"], "model_state_change_from_full": change}
        manifest["stages"][stage] = {"checkpoint": model_path.relative_to(run_dir).as_posix(), **metadata,
                                     "task_metrics": metrics}
        save_json(manifest, vdir / "checkpoint_manifest.json")
        if baseline_obs is None:
            baseline_obs, baseline_metrics = observations, metrics
        report = write_audit(baseline_obs, observations, forget, audit_cfg, vdir / stage,
                             metadata, baseline_metrics, metrics)
        if logs is not None:
            save_json(logs, vdir / stage / "method_logs.json")
        summaries.append({"stage": stage, "method": method, "behavior": behavior,
                          "status": report["status"], "exploratory_status": report["exploratory_status"], **metrics})
        overhead.append({"stage": stage, "operation": "marker_inference", "time_sec": seconds})
        overhead.append({"stage": stage, "operation": "statistics_and_output",
                         "time_sec": report["statistical_and_output_time_sec"]})
        pd.DataFrame(summaries).to_csv(vdir / "summary.csv", index=False)
        pd.DataFrame(overhead).to_csv(vdir / "overhead.csv", index=False)
        phase("DEVELOPMENT_AUDIT", "COMPLETED", stage, time.perf_counter()-audit_started)

    record("full_after_fl", ctx.final_parameters, "none", behavior="unmodified_full")
    proto = None
    if cfg.with_recovery:
        phase("BUILD_PROTOTYPE", "STARTED", "prototypes")
        proto, seconds = _timed_call(lambda: collect_dp_prototypes(
            fd, ctx.final_parameters, cfg.dataset, cfg.batch_size, str(device),
            exclude_clients=forget, dp_sigma=cfg.dp_sigma, clip_norm=cfg.proto_clip_norm, seed=cfg.seed), device)
        logger.save_obj("retained_dp_prototypes.pt", proto)
        overhead.append({"stage": "prototypes", "operation": "retained_raw_data_reaccess", "time_sec": seconds})
        phase("BUILD_PROTOTYPE", "COMPLETED", "prototypes", seconds)
    for method in methods:
        print(f"[verification] Method: {method}", flush=True)
        phase("UNLEARN", "STARTED", f"{method}_pre_recovery")
        unlearned, seconds = _timed_call(lambda: build_unlearner(method).unlearn(ctx), device)
        phase("UNLEARN", "COMPLETED", f"{method}_pre_recovery", seconds)
        stage = f"{method}_pre_recovery"
        overhead.append({"stage": stage, "operation": "unlearning", "time_sec": seconds})
        record(stage, unlearned.parameters, method,
               behavior="strict_retrain_reference" if method == "retrain_fl" else "candidate", logs=unlearned.logs)
        if proto is not None and method != "retrain_fl":
            phase("RECOVER", "STARTED", f"{method}_proto_lda_recovery")
            recovered, seconds = _timed_call(lambda: recover_with_prototype_lda(
                unlearned.parameters, proto, cfg.dataset, ridge=cfg.lda_ridge), device)
            phase("RECOVER", "COMPLETED", f"{method}_proto_lda_recovery", seconds)
            stage = f"{method}_proto_lda_recovery"
            overhead.append({"stage": stage, "operation": "recovery", "time_sec": seconds})
            record(stage, recovered, method, "prototype_lda", logs={"scope": "head_reconstruction"})
    if cfg.include_fakes:
        record("fake_noop_pre_recovery", [x.copy() for x in ctx.final_parameters], "fake_noop", behavior="fake")
        erased, seconds = _timed_call(lambda: pga_unlearn(ctx, steps=cfg.fake_steps, lr=cfg.fake_lr,
                                         project_ratio=0.1, layers="all", data_scope="markers_only"), device)
        overhead.append({"stage": "fake_marker_erase_pre_recovery", "operation": "fake_generation", "time_sec": seconds})
        record("fake_marker_erase_pre_recovery", erased.parameters, "fake_marker_erase", behavior="fake", logs=erased.logs)
        corrupted, seconds = _timed_call(lambda: _corrupt(ctx, cfg.corruption_scale), device)
        overhead.append({"stage": "fake_corrupt_pre_recovery", "operation": "fake_generation", "time_sec": seconds})
        record("fake_corrupt_pre_recovery", corrupted.parameters, "fake_corrupt", behavior="fake", logs=corrupted.logs)
    overhead.append({"stage": "artifacts", "operation": "stored_bytes",
                     "bytes": sum(p.stat().st_size for p in run_dir.rglob("*") if p.is_file())})
    pd.DataFrame(overhead).to_csv(vdir / "overhead.csv", index=False)
    pd.DataFrame(detection_summary(summaries)).to_csv(vdir / "detection_summary.csv", index=False)
    save_json({"status": "development_run_complete", "formal_audit": False,
               "bootstrap_scope": "within_checkpoint_family_uncertainty_not_across_training_runs",
               "seed": cfg.seed, "num_classes": fd.num_classes}, run_dir / "run_status.json")
    print(f"[verification] Saved: {run_dir}", flush=True)
    return run_dir
