'''
from __future__ import annotations

import numpy as np

from fulbase.utils import add_arrays, arrays_l2_norm, diff_arrays, weighted_average_arrays


def recover_with_federaser_replay(
    history: list[dict],
    forget_client_ids: list[int],
    calibrate: bool = True,
) -> list[np.ndarray]:
    """FedEraser-style replay baseline.

    简化实现：从第 1 轮 global_before 开始，逐轮只聚合保留客户端的历史更新。
    calibrate=True 时，用原始全体更新范数与保留更新范数做一个简单校准，模拟 FedEraser 的历史更新校正思想。
    """
    forget_set = {str(x) for x in forget_client_ids}
    if not history:
        raise ValueError("Empty history; cannot run FedEraser replay.")
    params = [x.copy() for x in history[0]["global_before"]]

    for item in history:
        global_before = item["global_before"]
        retained_deltas = []
        all_deltas = []
        for c in item["clients"]:
            delta = diff_arrays(c["params_after"], global_before)
            all_deltas.append((delta, int(c["num_examples"])))
            if str(c["cid"]) not in forget_set:
                retained_deltas.append((delta, int(c["num_examples"])))
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
        params = add_arrays(params, retained_update, scale=1.0)
    return params
'''

from __future__ import annotations

import numpy as np

from fulbase.utils import add_arrays


def _to_float32(arrays):
    return [x.astype(np.float32) if np.issubdtype(x.dtype, np.floating) else x for x in arrays]


def recover_with_federaser_replay(
    history: list[dict],
    forget_client_ids: list[int],
    calibrate: bool = True,
) -> list[np.ndarray]:
    """
    低内存 FedEraser-style replay。

    从第 1 轮 global_before 开始，逐轮回放 retained_update。
    """
    if not history:
        raise ValueError("Empty history; cannot run FedEraser replay.")

    params = _to_float32(history[0]["global_before"])

    for item in history:
        if "retained_update" not in item:
            raise ValueError("history item does not contain retained_update. Please rerun with CompactHistoryFedAvg.")
        retained_update = _to_float32(item["retained_update"])
        params = add_arrays(params, retained_update, scale=1.0)

    return params