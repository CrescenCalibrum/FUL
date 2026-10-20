from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from fulbase.utils import ensure_dir


class RunLogger:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = ensure_dir(run_dir)
        self.models_dir = ensure_dir(self.run_dir / "models")
        self.records: list[dict] = []

    def add_metrics(self, stage: str, metrics: dict) -> None:
        row = {"stage": stage}
        row.update(metrics)
        self.records.append(row)
        self.flush()

    def flush(self) -> None:
        pd.DataFrame(self.records).to_csv(self.run_dir / "metrics.csv", index=False)

    def save_model_params(self, name: str, params: list) -> None:
        torch.save(params, self.models_dir / f"{name}.pt")

    def save_obj(self, name: str, obj) -> None:
        torch.save(obj, self.run_dir / name)
