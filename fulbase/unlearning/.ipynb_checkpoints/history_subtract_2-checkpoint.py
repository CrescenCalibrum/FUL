from __future__ import annotations

import numpy as np

from fulbase.unlearning.base import BaseUnlearner, UnlearningContext, UnlearningResult
from fulbase.utils import add_arrays, diff_arrays


def history_subtract_unlearn(
    final_parameters: list[np.ndarray],
    history: list[dict],
    forget_client_ids: list[int],
    alpha: float = 0.5,
) -> list[np.ndarray]:
    """从最终模型中减去被遗忘客户端历史更新的加权和。"""
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

    return add_arrays(params, cumulative, scale=-alpha)


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
