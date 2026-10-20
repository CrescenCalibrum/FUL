from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch
import time

from fulbase.data import load_federated_data, make_merged_loader
from fulbase.dp.prototype import collect_dp_prototypes
from fulbase.experiment.io import RunLogger
from fulbase.flower.sim import run_flower_fedavg
from fulbase.models import build_model
from fulbase.recovery.federaser_replay import recover_with_federaser_replay
from fulbase.recovery.prototype_lda import recover_with_prototype_lda
from fulbase.train import evaluate
from fulbase.unlearning.base import UnlearningContext
from fulbase.unlearning.registry import available_unlearners, build_unlearner
from fulbase.utils import ensure_dir, save_json, set_model_parameters, set_seed


@dataclass
class ExperimentConfig:
    dataset: str = "fashionmnist"
    data_dir: str = "./data"
    out_dir: str = "./runs"
    run_name: str = "debug"
    seed: int = 42

    num_clients: int = 10
    num_rounds: int = 5
    fraction_fit: float = 1.0
    fraction_evaluate: float = 1.0
    local_epochs: int = 1
    batch_size: int = 64
    lr: float = 0.01
    partition: str = "dirichlet"
    alpha: float = 0.5

    client_device: str = "auto"   # 多客户端仿真默认 CPU，避免所有 Ray worker 抢同一块 GPU
    eval_device: str = "auto"
    client_cpus: float = 1.0
    client_gpus: float = 0.1

    forget_clients: str = "0"

    # 遗忘算子：history_subtract / federaser_lite / pga / retrain_oracle / all
    unlearner: str = "history_subtract"
    unlearn_alpha: float = 0.5
    federaser_damping: float = 1.0
    federaser_calibrate: bool = True
    pga_steps: int = 100
    pga_lr: float = 1e-4
    pga_project_ratio: float = 0.01
    pga_layers: str = "head"  # head / last / all
    retrain_epochs: int = 5
    retrain_lr: float | None = None

    dp_sigma: float = 0.0
    proto_clip_norm: float = 10.0
    lda_ridge: float = 1e-2


def _parse_forget_clients(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def _eval_device(cfg: ExperimentConfig) -> torch.device:
    if cfg.eval_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.eval_device)


def _evaluate_params(stage: str, params: list, fd, cfg: ExperimentConfig, retain_ids: list[int], forget_ids: list[int]) -> dict:
    device = _eval_device(cfg)
    model = build_model(dataset=cfg.dataset, num_classes=fd.num_classes)
    set_model_parameters(model, params)
    retain_loader = make_merged_loader(fd.testset, fd.test_indices, retain_ids, batch_size=cfg.batch_size)
    forget_loader = make_merged_loader(fd.testset, fd.test_indices, forget_ids, batch_size=cfg.batch_size)
    all_loader = make_merged_loader(fd.testset, fd.test_indices, list(range(cfg.num_clients)), batch_size=cfg.batch_size)

    retain = evaluate(model, retain_loader, device)
    forget = evaluate(model, forget_loader, device)
    allm = evaluate(model, all_loader, device)
    return {
        "all_acc": allm["acc"],
        "all_loss": allm["loss"],
        "retain_acc": retain["acc"],
        "retain_loss": retain["loss"],
        "forget_acc": forget["acc"],
        "forget_loss": forget["loss"],
        "retain_num_examples": retain["num_examples"],
        "forget_num_examples": forget["num_examples"],
    }


def _metrics_with_meta(metrics: dict, unlearner: str, recovery: str, logs: dict | None = None) -> dict:
    out = {
        "unlearner": unlearner,
        "recovery": recovery,
    }
    out.update(metrics)
    # 只把简单标量日志摊平进 csv；复杂对象另存 pt。
    for k, v in (logs or {}).items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[f"log_{k}"] = v
    return out

def _timed_call(fn, device: torch.device):
    """
    统计算法本身执行时间。

    CUDA 操作是异步的，所以 GPU 情况下计时前后需要 synchronize。
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start = time.perf_counter()

    result = fn()

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - start
    return result, elapsed


def _selected_unlearners(cfg: ExperimentConfig) -> list[str]:
    if cfg.unlearner.lower().strip() == "all":
        return available_unlearners()
    return [cfg.unlearner.lower().strip()]


def run_full_pipeline(cfg: ExperimentConfig) -> Path:
    set_seed(cfg.seed)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = ensure_dir(Path(cfg.out_dir) / f"{timestamp}_{cfg.run_name}")
    logger = RunLogger(run_dir)
    save_json(asdict(cfg), run_dir / "config.json")

    print(f"[1/6] Load data: dataset={cfg.dataset}, clients={cfg.num_clients}, partition={cfg.partition}, alpha={cfg.alpha}")
    fd = load_federated_data(
        dataset=cfg.dataset,
        data_dir=cfg.data_dir,
        num_clients=cfg.num_clients,
        partition=cfg.partition,
        alpha=cfg.alpha,
        seed=cfg.seed,
    )
    forget_ids = _parse_forget_clients(cfg.forget_clients)
    retain_ids = [cid for cid in range(cfg.num_clients) if cid not in set(forget_ids)]

    if not forget_ids:
        raise ValueError("forget_clients cannot be empty")
    if any(cid < 0 or cid >= cfg.num_clients for cid in forget_ids):
        raise ValueError(f"forget_clients={forget_ids} out of range [0,{cfg.num_clients})")
    if not retain_ids:
        raise ValueError("All clients are forgotten; retained set is empty")

    print("[2/6] Run Flower FedAvg training")
    min_fit = max(1, int(cfg.num_clients * cfg.fraction_fit))
    min_eval = max(1, int(cfg.num_clients * cfg.fraction_evaluate))
    fl_res = run_flower_fedavg(
        fd=fd,
        dataset=cfg.dataset,
        num_clients=cfg.num_clients,
        num_rounds=cfg.num_rounds,
        fraction_fit=cfg.fraction_fit,
        fraction_evaluate=cfg.fraction_evaluate,
        min_fit_clients=min_fit,
        min_evaluate_clients=min_eval,
        batch_size=cfg.batch_size,
        local_epochs=cfg.local_epochs,
        lr=cfg.lr,
        client_device=cfg.client_device,
        client_cpus=cfg.client_cpus,
        client_gpus=cfg.client_gpus,
    )
    logger.save_model_params("model_full", fl_res.final_parameters)
    logger.save_obj("history.pt", fl_res.history)
    logger.add_metrics(
        "full_after_fl",
        _metrics_with_meta(
            _evaluate_params("full_after_fl", fl_res.final_parameters, fd, cfg, retain_ids, forget_ids),
            unlearner="none",
            recovery="none",
        ),
    )

    print("[3/6] Collect retained DP prototypes for your method")
    proto = collect_dp_prototypes(
        fd=fd,
        model_parameters=fl_res.final_parameters,
        dataset=cfg.dataset,
        batch_size=cfg.batch_size,
        device=str(_eval_device(cfg)),
        exclude_clients=forget_ids,
        dp_sigma=cfg.dp_sigma,
        clip_norm=cfg.proto_clip_norm,
        seed=cfg.seed,
    )
    logger.save_obj("retained_dp_prototypes.pt", proto)

    print("[4/6] Build unlearning context")
    initial_parameters = [x.copy() for x in fl_res.history[0]["global_before"]]
    ctx = UnlearningContext(
        fd=fd,
        dataset=cfg.dataset,
        num_classes=fd.num_classes,
        initial_parameters=initial_parameters,
        final_parameters=fl_res.final_parameters,
        history=fl_res.history,
        forget_client_ids=forget_ids,
        retain_client_ids=retain_ids,
        batch_size=cfg.batch_size,
        device=_eval_device(cfg),
        cfg=cfg,
    )

    print(f"[5/6] Run unlearning operators: {_selected_unlearners(cfg)}")
    for unlearner_name in _selected_unlearners(cfg):
        unlearner = build_unlearner(unlearner_name)
        print(f"  - Unlearner: {unlearner.name}")
        result = unlearner.unlearn(ctx)
        logger.save_model_params(f"model_{result.name}_pre_recovery", result.parameters)
        logger.save_obj(f"unlearning_logs_{result.name}.pt", result.logs)
        logger.add_metrics(
            f"{result.name}_pre_recovery",
            _metrics_with_meta(
                _evaluate_params(f"{result.name}_pre_recovery", result.parameters, fd, cfg, retain_ids, forget_ids),
                unlearner=result.name,
                recovery="none",
                logs=result.logs,
            ),
        )

        # Retrain oracle 本身就是参考上界，一般不再接恢复模块。
        if result.name == "retrain_oracle":
            continue

        # Recovery baseline: FedEraser-style historical replay。
        # 注意：这个 baseline 从 history 直接回放，不依赖 result.parameters；
        # 如果 --unlearner all，会看到该 baseline 在不同 unlearner 下数值相同，论文中可只保留一份。
        '''
        federaser = recover_with_federaser_replay(
            history=fl_res.history,
            forget_client_ids=forget_ids,
            calibrate=cfg.federaser_calibrate,
        )
        '''
        federaser, federaser_time = _timed_call(
            lambda: recover_with_federaser_replay(
                history=fl_res.history,
                forget_client_ids=forget_ids,
                calibrate=cfg.federaser_calibrate,
            ),
            ctx.device,
        )
        logger.save_model_params(f"model_{result.name}_federaser_recovery", federaser)
        '''
        logger.add_metrics(
            f"{result.name}_federaser_recovery",
            _metrics_with_meta(
                _evaluate_params(f"{result.name}_federaser_recovery", federaser, fd, cfg, retain_ids, forget_ids),
                unlearner=result.name,
                recovery="federaser_replay",
            ),
        )
        '''
        federaser_metrics = _evaluate_params(
            f"{result.name}_federaser_recovery",
            federaser,
            fd,
            cfg,
            retain_ids,
            forget_ids,
        )

        federaser_metrics["recovery_time_sec"] = federaser_time

        logger.add_metrics(
            f"{result.name}_federaser_recovery",
            _metrics_with_meta(
                federaser_metrics,
                unlearner=result.name,
                recovery="federaser_replay",
            ),
        )
        
        # Your method: DP prototype + LDA classifier head reconstruction。
        '''
        proto_lda = recover_with_prototype_lda(
            unlearned_parameters=result.parameters,
            proto=proto,
            dataset=cfg.dataset,
            keep_classes=None,
            ridge=cfg.lda_ridge,
        )
        '''
        proto_lda, proto_lda_time = _timed_call(
            lambda: recover_with_prototype_lda(
                unlearned_parameters=result.parameters,
                proto=proto,
                dataset=cfg.dataset,
                keep_classes=None,
                ridge=cfg.lda_ridge,
            ),
            ctx.device,
        )
        
        logger.save_model_params(f"model_{result.name}_proto_lda_recovery", proto_lda)
        '''
        logger.add_metrics(
            f"{result.name}_proto_lda_recovery",
            _metrics_with_meta(
                _evaluate_params(f"{result.name}_proto_lda_recovery", proto_lda, fd, cfg, retain_ids, forget_ids),
                unlearner=result.name,
                recovery="prototype_lda",
            ),
        )
        '''
        proto_metrics = _evaluate_params(
            f"{result.name}_proto_lda_recovery",
            proto_lda,
            fd,
            cfg,
            retain_ids,
            forget_ids,
        )

        proto_metrics["recovery_time_sec"] = proto_lda_time

        logger.add_metrics(
            f"{result.name}_proto_lda_recovery",
            _metrics_with_meta(
                proto_metrics,
                unlearner=result.name,
                recovery="prototype_lda",
            ),
        )
        

    print("[6/6] Flush metrics")
    logger.flush()
    print(f"Done. Results saved to: {run_dir}")
    return run_dir
