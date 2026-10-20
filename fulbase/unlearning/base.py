from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from fulbase.data import FederatedData


@dataclass
class UnlearningContext:
    """统一传给各遗忘算子的上下文。

    后续新增遗忘算子时，尽量只依赖 ctx，不要在 pipeline.py 中继续堆 if-else。
    """

    fd: FederatedData
    dataset: str
    num_classes: int
    initial_parameters: list[np.ndarray]
    final_parameters: list[np.ndarray]
    history: list[dict[str, Any]]
    forget_client_ids: list[int]
    retain_client_ids: list[int]
    batch_size: int
    device: torch.device
    cfg: Any


@dataclass
class UnlearningResult:
    name: str
    parameters: list[np.ndarray]
    logs: dict[str, Any] = field(default_factory=dict)


class BaseUnlearner:
    name = "base"

    def unlearn(self, ctx: UnlearningContext) -> UnlearningResult:
        raise NotImplementedError
