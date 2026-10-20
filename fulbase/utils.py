from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_dataclass(obj):
        obj = asdict(obj)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def state_dict_to_ndarrays(state_dict: dict[str, torch.Tensor]) -> list[np.ndarray]:
    return [v.detach().cpu().numpy() for v in state_dict.values()]


def ndarrays_to_state_dict(arrays: Iterable[np.ndarray], model: torch.nn.Module) -> dict[str, torch.Tensor]:
    keys = list(model.state_dict().keys())
    return {k: torch.tensor(v) for k, v in zip(keys, arrays)}


def get_model_parameters(model: torch.nn.Module) -> list[np.ndarray]:
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def set_model_parameters(model: torch.nn.Module, parameters: list[np.ndarray]) -> None:
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)


def weighted_average_arrays(items: list[tuple[list[np.ndarray], int]]) -> list[np.ndarray]:
    total = sum(n for _, n in items)
    if total <= 0:
        raise ValueError("No examples to aggregate.")
    out = []
    for layer_idx in range(len(items[0][0])):
        acc = None
        for arrays, n in items:
            part = arrays[layer_idx].astype(np.float64) * (n / total)
            acc = part if acc is None else acc + part
        out.append(acc.astype(items[0][0][layer_idx].dtype))
    return out


def add_arrays(a: list[np.ndarray], b: list[np.ndarray], scale: float = 1.0) -> list[np.ndarray]:
    return [(x + scale * y).astype(x.dtype) for x, y in zip(a, b)]


def sub_arrays(a: list[np.ndarray], b: list[np.ndarray], scale: float = 1.0) -> list[np.ndarray]:
    return [(x - scale * y).astype(x.dtype) for x, y in zip(a, b)]


def diff_arrays(a: list[np.ndarray], b: list[np.ndarray]) -> list[np.ndarray]:
    """Return a - b."""
    return [(x - y).astype(x.dtype) for x, y in zip(a, b)]


def arrays_l2_norm(arrays: list[np.ndarray]) -> float:
    return float(np.sqrt(sum(float(np.sum(x.astype(np.float64) ** 2)) for x in arrays)))
