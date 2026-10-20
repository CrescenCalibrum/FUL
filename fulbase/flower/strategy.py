'''
from __future__ import annotations

from pathlib import Path
from typing import Any

import flwr as fl
import numpy as np
from flwr.common import FitIns, Metrics, NDArrays, Parameters, Scalar, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy


class HistoryFedAvg(fl.server.strategy.FedAvg):
    """FedAvg strategy that records per-round client updates.

    保存内容用于后续 FedEraser-style replay / 历史回放恢复 baseline：
    - round_global_before: 本轮训练前全局参数
    - client_params: 本轮每个客户端训练后的参数
    - client_num_examples: 每个客户端样本数
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.history: list[dict[str, Any]] = []
        self._round_global: dict[int, NDArrays] = {}

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, FitIns]]:
        self._round_global[server_round] = parameters_to_ndarrays(parameters)
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round: int, results, failures):
        # 先保存每个客户端的训练后参数；再调用父类完成 FedAvg 聚合
        round_item: dict[str, Any] = {
            "round": server_round,
            "global_before": self._round_global.get(server_round),
            "clients": [],
        }
        for client_proxy, fit_res in results:
            cid = str(fit_res.metrics.get("cid", client_proxy.cid))
            round_item["clients"].append(
                {
                    "cid": cid,
                    "num_examples": int(fit_res.num_examples),
                    "params_after": parameters_to_ndarrays(fit_res.parameters),
                    "metrics": dict(fit_res.metrics),
                }
            )
        self.history.append(round_item)
        return super().aggregate_fit(server_round, results, failures)


def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    total = sum(num_examples for num_examples, _ in metrics)
    if total == 0:
        return {}
    out: dict[str, float] = {}
    for num_examples, m in metrics:
        for k, v in m.items():
            if isinstance(v, (int, float, np.floating)):
                out[k] = out.get(k, 0.0) + float(v) * num_examples / total
    return out
'''

from __future__ import annotations

from typing import Any

import flwr as fl
import numpy as np
from flwr.common import FitIns, Metrics, Parameters, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy


def _zeros_like(arrays):
    return [np.zeros_like(x, dtype=np.float32) for x in arrays]


def _delta(after, before):
    return [(a.astype(np.float32) - b.astype(np.float32)) for a, b in zip(after, before)]


def _add_scaled(acc, upd, scale: float):
    return [a + u * scale for a, u in zip(acc, upd)]


def _to_float16(arrays):
    return [x.astype(np.float16) if np.issubdtype(x.dtype, np.floating) else x for x in arrays]


def _to_float32(arrays):
    return [x.astype(np.float32) if np.issubdtype(x.dtype, np.floating) else x for x in arrays]


class CompactHistoryFedAvg(fl.server.strategy.FedAvg):
    """
    低内存版 FedAvg history。

    不再保存每个客户端完整 params_after，而是每轮只保存：
    - global_before
    - all_update
    - retained_update
    - forget_update_wrt_all

    这样仍然支持：
    - 遗忘后恢复前模型生成
    - FedEraser-style replay baseline
    - 后续你的 DP prototype recovery
    """

    def __init__(self, *args: Any, forget_client_ids=None, store_float16: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.forget_client_ids = set(str(x) for x in (forget_client_ids or []))
        self.store_float16 = store_float16
        self.history: list[dict[str, Any]] = []
        self._round_global: dict[int, list[np.ndarray]] = {}
        self.final_parameters = None

    def _store(self, arrays):
        return _to_float16(arrays) if self.store_float16 else arrays

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, FitIns]]:
        self._round_global[server_round] = parameters_to_ndarrays(parameters)
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round: int, results, failures):
        global_before = self._round_global.get(server_round)
        if global_before is None:
            return super().aggregate_fit(server_round, results, failures)

        total_examples = sum(int(fit_res.num_examples) for _, fit_res in results)
        retained_examples = 0
        for client_proxy, fit_res in results:
            cid = str(fit_res.metrics.get("cid", client_proxy.cid))
            if cid not in self.forget_client_ids:
                retained_examples += int(fit_res.num_examples)

        all_update = _zeros_like(global_before)
        retained_update = _zeros_like(global_before)
        forget_update_wrt_all = _zeros_like(global_before)

        for client_proxy, fit_res in results:
            cid = str(fit_res.metrics.get("cid", client_proxy.cid))
            n = int(fit_res.num_examples)
            params_after = parameters_to_ndarrays(fit_res.parameters)
            upd = _delta(params_after, global_before)

            if total_examples > 0:
                all_update = _add_scaled(all_update, upd, n / total_examples)

            if cid in self.forget_client_ids:
                if total_examples > 0:
                    forget_update_wrt_all = _add_scaled(forget_update_wrt_all, upd, n / total_examples)
            else:
                if retained_examples > 0:
                    retained_update = _add_scaled(retained_update, upd, n / retained_examples)

        self.history.append(
            {
                "round": server_round,
                "global_before": self._store(global_before),
                "all_update": self._store(all_update),
                "retained_update": self._store(retained_update),
                "forget_update_wrt_all": self._store(forget_update_wrt_all),
                "total_examples": total_examples,
                "retained_examples": retained_examples,
            }
        )

        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            self.final_parameters = parameters_to_ndarrays(aggregated_parameters)

        return aggregated_parameters, metrics


def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    total = sum(num_examples for num_examples, _ in metrics)
    if total == 0:
        return {}
    out: dict[str, float] = {}
    for num_examples, m in metrics:
        for k, v in m.items():
            if isinstance(v, (int, float, np.floating)):
                out[k] = out.get(k, 0.0) + float(v) * num_examples / total
    return out