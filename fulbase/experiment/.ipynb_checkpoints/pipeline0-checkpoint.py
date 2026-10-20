from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch

from fulbase.data import load_federated_data, make_merged_loader
from fulbase.dp.prototype import collect_dp_prototypes
from fulbase.experiment.io import RunLogger
from fulbase.flower.sim import run_flower_fedavg
from fulbase.models import build_model
from fulbase.recovery.federaser_replay import recover_with_federaser_replay
from fulbase.recovery.prototype_lda import recover_with_prototype_lda
from fulbase.train import evaluate
from fulbase.unlearning.history_subtract import history_subtract_unlearn
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
    client_gpus: float = 0.25

    forget_clients: str = "0"
    unlearn_alpha: float = 0.5

    dp_sigma: float = 0.0
    proto_clip_norm: float = 10.0
    lda_ridge: float = 1e-2


def _parse_forget_clients(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def _evaluate_params(stage: str, params: list, fd, cfg: ExperimentConfig, retain_ids: list[int], forget_ids: list[int]) -> dict:
    device = torch.device("cuda" if cfg.eval_device == "auto" and torch.cuda.is_available() else cfg.eval_device)
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
        forget_client_ids=forget_ids,
    )
    logger.save_model_params("model_full", fl_res.final_parameters)
    logger.save_obj("history.pt", fl_res.history)
    logger.add_metrics("full_after_fl", _evaluate_params("full_after_fl", fl_res.final_parameters, fd, cfg, retain_ids, forget_ids))

    print("[3/6] Collect retained DP prototypes for your method")
    proto = collect_dp_prototypes(
        fd=fd,
        model_parameters=fl_res.final_parameters,
        dataset=cfg.dataset,
        batch_size=cfg.batch_size,
        device=("cuda" if cfg.eval_device == "auto" and torch.cuda.is_available() else cfg.eval_device),
        exclude_clients=forget_ids,
        dp_sigma=cfg.dp_sigma,
        clip_norm=cfg.proto_clip_norm,
        seed=cfg.seed,
    )
    logger.save_obj("retained_dp_prototypes.pt", proto)

    print("[4/6] Black-box unlearning operator: history_subtract")
    unlearned = history_subtract_unlearn(
        final_parameters=fl_res.final_parameters,
        history=fl_res.history,
        forget_client_ids=forget_ids,
        alpha=cfg.unlearn_alpha,
    )
    logger.save_model_params("model_unlearned_pre_recovery", unlearned)
    logger.add_metrics(
        "unlearned_pre_recovery",
        _evaluate_params("unlearned_pre_recovery", unlearned, fd, cfg, retain_ids, forget_ids),
    )

    print("[5/6] Recovery baseline: FedEraser-style historical replay")
    federaser = recover_with_federaser_replay(
        history=fl_res.history,
        forget_client_ids=forget_ids,
        calibrate=True,
    )
    logger.save_model_params("model_federaser_recovery", federaser)
    logger.add_metrics(
        "federaser_recovery",
        _evaluate_params("federaser_recovery", federaser, fd, cfg, retain_ids, forget_ids),
    )

    print("[6/6] Your method: DP prototype + LDA classifier head reconstruction")
    proto_lda = recover_with_prototype_lda(
        unlearned_parameters=unlearned,
        proto=proto,
        dataset=cfg.dataset,
        keep_classes=None,
        ridge=cfg.lda_ridge,
    )
    logger.save_model_params("model_proto_lda_recovery", proto_lda)
    logger.add_metrics(
        "prototype_lda_recovery",
        _evaluate_params("prototype_lda_recovery", proto_lda, fd, cfg, retain_ids, forget_ids),
    )

    print(f"Done. Results saved to: {run_dir}")
    return run_dir
