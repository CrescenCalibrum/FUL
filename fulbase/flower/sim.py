from __future__ import annotations

from dataclasses import dataclass

import flwr as fl
import torch
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from fulbase.data import FederatedData
from fulbase.flower.client import make_client_fn
from fulbase.flower.strategy import CompactHistoryFedAvg, weighted_average
from fulbase.models import build_model
from fulbase.utils import get_model_parameters


@dataclass
class FLTrainResult:
    final_parameters: list
    history: list


def run_flower_fedavg(
    fd: FederatedData,
    dataset: str,
    num_clients: int,
    num_rounds: int,
    fraction_fit: float,
    fraction_evaluate: float,
    min_fit_clients: int,
    min_evaluate_clients: int,
    batch_size: int,
    local_epochs: int,
    lr: float,
    client_device: str = "cpu",
    client_cpus: float = 1.0,
    client_gpus: float = 0.0,
    forget_client_ids: list[int] | None = None,
) -> FLTrainResult:
    model = build_model(dataset=dataset, num_classes=fd.num_classes)
    initial_parameters = ndarrays_to_parameters(get_model_parameters(model))

    '''
    strategy = HistoryFedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=num_clients,
        initial_parameters=initial_parameters,
        on_fit_config_fn=lambda r: {"lr": lr, "local_epochs": local_epochs},
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
    )
    '''
    strategy = CompactHistoryFedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=num_clients,
        initial_parameters=initial_parameters,
        on_fit_config_fn=lambda r: {"lr": lr, "local_epochs": local_epochs},
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
        forget_client_ids=forget_client_ids or [],
        store_float16=True,
    )
    

    client_fn = make_client_fn(
        fd=fd,
        dataset=dataset,
        batch_size=batch_size,
        local_epochs=local_epochs,
        lr=lr,
        device=client_device,
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": client_cpus, "num_gpus": client_gpus},
    )
    '''
    hist = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": client_cpus, "num_gpus": client_gpus},
    )

    # Flower 的 History 对象主要存日志；真正的 per-client 历史在 strategy.history 中。
    if hist is None:
        pass
    # final params = last aggregated parameters stored by strategy in parent class is not public；
    # 因此从最后一轮的 retained aggregate 近似取。更稳妥做法是利用父类返回的 final model；
    # legacy start_simulation 不直接返回 final parameters，所以这里从最后一轮所有 client_after 做一次 FedAvg。
    last_round = strategy.history[-1]
    total = sum(c["num_examples"] for c in last_round["clients"])
    '''
    
    '''
    final_params = []
    for layer_idx in range(len(last_round["clients"][0]["params_after"])):
        acc = None
        for c in last_round["clients"]:
            arr = c["params_after"][layer_idx]
            part = arr.astype("float64") * (c["num_examples"] / total)
            acc = part if acc is None else acc + part
        final_params.append(acc.astype(last_round["clients"][0]["params_after"][layer_idx].dtype))

    return FLTrainResult(final_parameters=final_params, history=strategy.history)
    '''
    final_params = strategy.final_parameters
    if final_params is None:
        raise RuntimeError("No final parameters found after Flower simulation.")

    return FLTrainResult(final_parameters=final_params, history=strategy.history)
