"""Sequential FedAvg for CPU debugging; deliberately labeled separately from Ray."""
from __future__ import annotations

import numpy as np
from fulbase.flower.client import FULClient
from fulbase.flower.sim import FLTrainResult
from fulbase.flower.strategy import compact_round
from fulbase.models import build_model
from fulbase.utils import get_model_parameters, weighted_average_arrays


def run_serial_fedavg(fd, dataset, num_clients, num_rounds, fraction_fit,
                      fraction_evaluate, min_fit_clients, min_evaluate_clients,
                      batch_size, local_epochs, lr, client_device="cpu",
                      client_cpus=1.0, client_gpus=0.0, forget_client_ids=None,
                      initial_model_parameters=None, active_client_ids=None,
                      marker_config=None, training_seed=42, record_history=True):
    active = list(range(num_clients)) if active_client_ids is None else list(active_client_ids)
    if not active or len(set(active)) != len(active) or any(x < 0 or x >= num_clients for x in active):
        raise ValueError("Invalid active clients")
    if num_rounds < 1 or not 0 < fraction_fit <= 1:
        raise ValueError("Invalid training budget")
    initial = (get_model_parameters(build_model(dataset, fd.num_classes))
               if initial_model_parameters is None else [x.copy() for x in initial_model_parameters])
    current, history = [x.copy() for x in initial], []
    rng = np.random.default_rng(training_seed)
    for round_id in range(1, num_rounds + 1):
        count = min(len(active), max(min_fit_clients, int(len(active) * fraction_fit)))
        chosen = sorted(rng.choice(active, size=count, replace=False).tolist())
        clients = []
        for cid in chosen:
            client = FULClient(cid, fd, dataset, batch_size, local_epochs, lr,
                               client_device, marker_config, training_seed)
            params, n, metrics = client.fit(current, {"round": round_id})
            clients.append({"cid": cid, "params_after": params, "num_examples": n,
                            "metrics": metrics})
        if record_history:
            history.append(compact_round(current, clients, forget_client_ids or [], round_id))
        current = weighted_average_arrays([(c["params_after"], c["num_examples"]) for c in clients])
    return FLTrainResult(current, history, initial)


def run_fedavg(*, backend="flower", **kwargs):
    if backend == "serial":
        return run_serial_fedavg(**kwargs)
    if backend == "flower":
        from fulbase.flower.sim import run_flower_fedavg
        return run_flower_fedavg(**kwargs)
    raise ValueError("backend must be flower or serial")
