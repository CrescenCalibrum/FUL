from __future__ import annotations

from fulbase.unlearning.base import BaseUnlearner
from fulbase.unlearning.federaser_lite import FedEraserLiteUnlearner
from fulbase.unlearning.history_subtract import HistorySubtractUnlearner
from fulbase.unlearning.pga import PGAUnlearner
from fulbase.unlearning.retrain_oracle import RetrainOracleUnlearner


_UNLEARNERS: dict[str, type[BaseUnlearner]] = {
    HistorySubtractUnlearner.name: HistorySubtractUnlearner,
    FedEraserLiteUnlearner.name: FedEraserLiteUnlearner,
    PGAUnlearner.name: PGAUnlearner,
    RetrainOracleUnlearner.name: RetrainOracleUnlearner,
}


def available_unlearners() -> list[str]:
    return list(_UNLEARNERS.keys())


def build_unlearner(name: str) -> BaseUnlearner:
    key = name.lower().strip()
    if key not in _UNLEARNERS:
        raise ValueError(f"Unknown unlearner={name}. Available: {available_unlearners()} or all")
    return _UNLEARNERS[key]()
