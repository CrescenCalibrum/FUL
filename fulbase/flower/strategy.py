from __future__ import annotations
import numpy as np
import flwr as fl
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters

def compact_round(before, clients, forget_client_ids, server_round, store_float16=False):
    """clients contain real-task sample counts. Never persist per-client models."""
    forget = set(map(str, forget_client_ids))
    total = sum(c["num_examples"] for c in clients)
    retained = sum(c["num_examples"] for c in clients if str(c["cid"]) not in forget)
    if total <= 0:
        raise ValueError("No successful training examples")
    floating = [np.issubdtype(x.dtype, np.floating) for x in before]
    all_up = [np.zeros_like(x, dtype=np.float32) for x in before]
    ret_up = [np.zeros_like(x, dtype=np.float32) for x in before]
    for_up = [np.zeros_like(x, dtype=np.float32) for x in before]
    for client in clients:
        n, forgotten = client["num_examples"], str(client["cid"]) in forget
        for k, (a, b) in enumerate(zip(client["params_after"], before)):
            if not floating[k]:
                continue
            delta = a.astype(np.float32) - b.astype(np.float32)
            all_up[k] += delta * (n / total)
            if forgotten:
                for_up[k] += delta * (n / total)
            elif retained:
                ret_up[k] += delta * (n / retained)
    def store(arrays):
        return [x.astype(np.float16) if store_float16 and np.issubdtype(x.dtype, np.floating)
                else x.copy() for x in arrays]
    return {"round": server_round, "global_before": store(before),
            "all_update": store(all_up), "retained_update": store(ret_up),
            "forget_update_wrt_all": store(for_up), "total_examples": total,
            "retained_examples": retained, "forget_client_ids": sorted(forget),
            "participating_client_ids": sorted(str(c["cid"]) for c in clients),
            "client_metrics": [{"cid": str(c["cid"]), "num_examples": c["num_examples"],
                                "metrics": c.get("metrics", {})} for c in clients],
            "forget_update_norm": float(np.sqrt(sum(np.square(x.astype(np.float64)).sum() for x in for_up))),
            "nonfloating_policy": "do_not_subtract_or_replay_buffers"}

class CompactHistoryFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *args, forget_client_ids=None, store_float16=False,
                 record_history=True, state_dtypes=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.forget_client_ids = set(map(str, forget_client_ids or []))
        self.store_float16 = store_float16
        self.record_history = record_history
        self.history, self._round_global = [], {}
        self.final_parameters = None
        self.state_dtypes = state_dtypes

    def configure_fit(self, server_round, parameters, client_manager):
        if self.state_dtypes is None:
            self.state_dtypes = [x.dtype for x in parameters_to_ndarrays(parameters)]
        if self.record_history:
            self._round_global[server_round] = parameters_to_ndarrays(parameters)
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        before = self._round_global.pop(server_round, None)
        aggregated, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated is None:
            raise RuntimeError(f"Round {server_round} did not aggregate successfully")
        arrays = parameters_to_ndarrays(aggregated)
        # Flower's numeric averaging promotes integer BN counters to float.
        # Restore the actual model state dtypes before returning/broadcasting.
        if self.state_dtypes is None:
            raise RuntimeError("Model state dtypes were not initialized")
        arrays = [x.astype(dtype, copy=False) for x, dtype in zip(arrays, self.state_dtypes)]
        aggregated = ndarrays_to_parameters(arrays)
        self.final_parameters = arrays
        if self.record_history:
            if before is None:
                raise RuntimeError("Missing broadcast model")
            clients = [{"cid": str(res.metrics.get("cid", proxy.cid)),
                        "num_examples": int(res.num_examples),
                        "params_after": parameters_to_ndarrays(res.parameters), "metrics": dict(res.metrics)}
                       for proxy, res in results]
            self.history.append(compact_round(before, clients, self.forget_client_ids,
                                               server_round, self.store_float16))
        return aggregated, metrics

def weighted_average(metrics):
    total = sum(n for n, _ in metrics)
    if not total:
        return {}
    out = {}
    for n, values in metrics:
        for k, value in values.items():
            if isinstance(value, (int, float, np.floating)):
                out[k] = out.get(k, 0.0) + float(value) * n / total
    return out
