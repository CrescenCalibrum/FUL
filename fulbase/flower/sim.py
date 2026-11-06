from __future__ import annotations
from dataclasses import dataclass
import flwr as fl
from flwr.common import ndarrays_to_parameters
from fulbase.flower.client import make_client_fn
from fulbase.flower.strategy import CompactHistoryFedAvg, weighted_average
from fulbase.models import build_model
from fulbase.utils import get_model_parameters

@dataclass
class FLTrainResult:
    final_parameters: list
    history: list
    initial_parameters: list | None = None

def run_flower_fedavg(fd, dataset, num_clients, num_rounds, fraction_fit,
                     fraction_evaluate, min_fit_clients, min_evaluate_clients,
                     batch_size, local_epochs, lr, client_device="cpu",
                     client_cpus=1.0, client_gpus=0.0, forget_client_ids=None,
                     initial_model_parameters=None, active_client_ids=None,
                     marker_config=None, training_seed=42, record_history=True):
    active = list(range(num_clients)) if active_client_ids is None else list(active_client_ids)
    if not active or len(set(active)) != len(active) or any(x < 0 or x >= num_clients for x in active):
        raise ValueError("Invalid active client identities")
    if num_rounds < 1 or not 0 < fraction_fit <= 1 or not 0 <= fraction_evaluate <= 1:
        raise ValueError("Invalid federated training budget")
    model = build_model(dataset, fd.num_classes)
    initial = (get_model_parameters(model) if initial_model_parameters is None
               else [x.copy() for x in initial_model_parameters])
    strategy = CompactHistoryFedAvg(
        fraction_fit=fraction_fit, fraction_evaluate=fraction_evaluate,
        min_fit_clients=min(min_fit_clients, len(active)),
        min_evaluate_clients=min(min_evaluate_clients, len(active)),
        min_available_clients=len(active),
        initial_parameters=ndarrays_to_parameters(initial),
        on_fit_config_fn=lambda r: {"lr": lr, "local_epochs": local_epochs, "round": r},
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
        forget_client_ids=forget_client_ids or [], store_float16=False,
        record_history=record_history, state_dtypes=[x.dtype for x in initial], accept_failures=False)
    client_fn = make_client_fn(fd, dataset, batch_size, local_epochs, lr, client_device,
                               marker_config, training_seed, active)
    fl.simulation.start_simulation(
        client_fn=client_fn, num_clients=len(active),
        config=fl.server.ServerConfig(num_rounds=num_rounds), strategy=strategy,
        client_resources={"num_cpus": client_cpus, "num_gpus": client_gpus})
    if strategy.final_parameters is None:
        raise RuntimeError("No aggregated parameters returned by Flower")
    return FLTrainResult(strategy.final_parameters, strategy.history, initial)
