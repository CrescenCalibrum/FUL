'''
from __future__ import annotations

import numpy as np

from fulbase.utils import add_arrays, diff_arrays


def history_subtract_unlearn(
    final_parameters: list[np.ndarray],
    history: list[dict],
    forget_client_ids: list[int],
    alpha: float = 0.5,
) -> list[np.ndarray]:
    """A simple black-box FU operator placeholder.

    从最终模型中减去被遗忘客户端历史更新的加权和。它不是论文最终算法，作用是：
    1. 让统一底座先形成“训练→遗忘→恢复前评估→恢复”闭环；
    2. 后续可替换为 PGA、FedEraser-unlearn、QuickDrop 等真正遗忘算子。
    """
    forget_set = {str(x) for x in forget_client_ids}
    params = [x.copy() for x in final_parameters]
    cumulative = [np.zeros_like(x) for x in params]

    for item in history:
        global_before = item["global_before"]
        clients = item["clients"]
        total = sum(c["num_examples"] for c in clients)
        if total <= 0:
            continue
        for c in clients:
            if str(c["cid"]) not in forget_set:
                continue
            weight = c["num_examples"] / total
            delta = diff_arrays(c["params_after"], global_before)
            cumulative = add_arrays(cumulative, delta, scale=weight)

    # alpha 控制遗忘强度；过大会让模型明显损坏，便于观察“恢复前”性能下降。
    params = add_arrays(params, cumulative, scale=-alpha)
    return params
'''

from __future__ import annotations

import numpy as np

from fulbase.unlearning.base import BaseUnlearner, UnlearningContext, UnlearningResult
from fulbase.utils import add_arrays


def _to_float32(arrays):
    return [x.astype(np.float32) if np.issubdtype(x.dtype, np.floating) else x for x in arrays]


def history_subtract_unlearn(
    final_parameters: list[np.ndarray],
    history: list[dict],
    forget_client_ids: list[int],
    alpha: float = 0.5,
) -> list[np.ndarray]:
    """
    低内存版遗忘算子。

    直接使用每轮保存好的 forget_update_wrt_all，
    不再依赖每个客户端的 params_after。
    """
    params = [x.astype(np.float32).copy() for x in final_parameters]
    cumulative = [np.zeros_like(x, dtype=np.float32) for x in params]

    for item in history:
        if "forget_update_wrt_all" not in item:
            raise ValueError("history item does not contain forget_update_wrt_all. Please rerun with CompactHistoryFedAvg.")
        upd = _to_float32(item["forget_update_wrt_all"])
        cumulative = add_arrays(cumulative, upd, scale=1.0)

    params = add_arrays(params, cumulative, scale=-alpha)
    return params


class HistorySubtractUnlearner(BaseUnlearner):
    name = "history_subtract"

    def unlearn(self, ctx: UnlearningContext) -> UnlearningResult:
        alpha = float(getattr(ctx.cfg, "unlearn_alpha", 0.5))
        params = history_subtract_unlearn(
            final_parameters=ctx.final_parameters,
            history=ctx.history,
            forget_client_ids=ctx.forget_client_ids,
            alpha=alpha,
        )
        return UnlearningResult(
            name=self.name,
            parameters=params,
            logs={"alpha": alpha, "forget_client_ids": ctx.forget_client_ids},
        )