from __future__ import annotations

import numpy as np

from fulbase.unlearning.base import BaseUnlearner, UnlearningContext, UnlearningResult
from fulbase.utils import add_arrays, arrays_l2_norm, diff_arrays, weighted_average_arrays


def _to_float32(arrays: list[np.ndarray]) -> list[np.ndarray]:
    """CompactHistoryFedAvg 可能用 float16 存储，这里统一转回 float32 计算。"""
    return [
        x.astype(np.float32) if np.issubdtype(x.dtype, np.floating) else x
        for x in arrays
    ]


def _is_compact_history(history: list[dict]) -> bool:
    """判断是否为你的低内存 CompactHistoryFedAvg history 格式。"""
    if not history:
        return False
    item = history[0]
    return (
        "global_before" in item
        and "retained_update" in item
        and "all_update" in item
    )


def _federaser_lite_from_compact_history(
    history: list[dict],
    damping: float = 1.0,
    calibrate: bool = True,
) -> list[np.ndarray]:
    """适配 CompactHistoryFedAvg 的 FedEraser-Lite。

    低内存 history 中已经提前保存了：
    - global_before
    - all_update
    - retained_update

    因此这里不再遍历 item["clients"]，而是从初始模型开始，
    逐轮重放 retained_update。
    """
    if not history:
        raise ValueError("Empty history; cannot run federaser_lite.")

    params = _to_float32(history[0]["global_before"])

    for item in history:
        if "retained_update" not in item:
            raise KeyError(
                f"Compact history item has no 'retained_update'. "
                f"Available keys={list(item.keys())}"
            )

        retained_update = _to_float32(item["retained_update"])

        if calibrate and "all_update" in item:
            all_update = _to_float32(item["all_update"])

            denom = arrays_l2_norm(retained_update)
            numer = arrays_l2_norm(all_update)

            if denom > 1e-12:
                # 防止 retained_update 范数过大导致 replay 过冲
                scale = min(1.0, numer / denom)
                retained_update = [u * scale for u in retained_update]

        params = add_arrays(params, retained_update, scale=float(damping))

    return params


def _federaser_lite_from_full_history(
    history: list[dict],
    forget_client_ids: list[int],
    damping: float = 1.0,
    calibrate: bool = True,
) -> list[np.ndarray]:
    """兼容旧版完整 history 的 FedEraser-Lite。

    旧版 history 每轮保存：
    - global_before
    - clients: [{cid, num_examples, params_after, ...}, ...]
    """
    forget_set = {str(x) for x in forget_client_ids}

    if not history:
        raise ValueError("Empty history; cannot run federaser_lite.")

    params = _to_float32(history[0]["global_before"])

    for item in history:
        global_before = _to_float32(item["global_before"])
        clients = item["clients"]

        retained_deltas: list[tuple[list[np.ndarray], int]] = []
        all_deltas: list[tuple[list[np.ndarray], int]] = []

        for c in clients:
            cid = str(c["cid"])
            n = int(c["num_examples"])
            params_after = _to_float32(c["params_after"])

            delta = diff_arrays(params_after, global_before)
            all_deltas.append((delta, n))

            if cid not in forget_set:
                retained_deltas.append((delta, n))

        if not retained_deltas:
            continue

        retained_update = weighted_average_arrays(retained_deltas)

        if calibrate and all_deltas:
            all_update = weighted_average_arrays(all_deltas)

            denom = arrays_l2_norm(retained_update)
            numer = arrays_l2_norm(all_update)

            if denom > 1e-12:
                scale = min(1.0, numer / denom)
                retained_update = [u * scale for u in retained_update]

        params = add_arrays(params, retained_update, scale=float(damping))

    return params


def federaser_lite_unlearn(
    history: list[dict],
    forget_client_ids: list[int],
    damping: float = 1.0,
    calibrate: bool = True,
) -> list[np.ndarray]:
    """FedEraser-Lite 遗忘算子。

    支持两种 history 格式：
    1. CompactHistoryFedAvg 低内存格式；
    2. 旧版完整客户端历史格式。
    """
    if _is_compact_history(history):
        return _federaser_lite_from_compact_history(
            history=history,
            damping=damping,
            calibrate=calibrate,
        )

    return _federaser_lite_from_full_history(
        history=history,
        forget_client_ids=forget_client_ids,
        damping=damping,
        calibrate=calibrate,
    )


class FedEraserLiteUnlearner(BaseUnlearner):
    name = "federaser_lite"

    def unlearn(self, ctx: UnlearningContext) -> UnlearningResult:
        damping = float(getattr(ctx.cfg, "federaser_damping", 1.0))
        calibrate = bool(getattr(ctx.cfg, "federaser_calibrate", True))

        params = federaser_lite_unlearn(
            history=ctx.history,
            forget_client_ids=ctx.forget_client_ids,
            damping=damping,
            calibrate=calibrate,
        )

        return UnlearningResult(
            name=self.name,
            parameters=params,
            logs={
                "history_format": "compact" if _is_compact_history(ctx.history) else "full",
                "damping": damping,
                "calibrate": calibrate,
                "forget_client_ids": ctx.forget_client_ids,
            },
        )