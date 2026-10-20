#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
针对单次实验目录，统计不同性能恢复方法的时间和存储开销。

目录示例：

runs/20260926-xxxxxx_main_exp/
├── metrics.csv
├── history.pt
├── retained_dp_prototypes.pt
└── models/
    ├── model_full.pt
    ├── model_history_subtract_federaser_recovery.pt
    ├── model_history_subtract_proto_lda_recovery.pt
    ├── model_pga_federaser_recovery.pt
    └── model_pga_proto_lda_recovery.pt

输出：
<run-dir>/overhead_analysis/
├── recovery_overhead_detail.csv
└── recovery_overhead_summary.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


TIME_CANDIDATES = (
    "recovery_time_sec",
    "elapsed_sec",
    "time_sec",
    "runtime_sec",
)

AUX_FILES_BY_RECOVERY = {
    "federaser_replay": ["history.pt"],
    "federaser": ["history.pt"],
    "prototype_lda": ["retained_dp_prototypes.pt"],
    "proto_lda": ["retained_dp_prototypes.pt"],
    "none": [],
}


def bytes_to_mib(nbytes: int) -> float:
    return nbytes / (1024.0 ** 2)


def file_size_mib(path: Path) -> float:
    if not path.exists() or not path.is_file():
        return 0.0
    return bytes_to_mib(path.stat().st_size)


def normalize_recovery(value, stage: str) -> str:
    if value is not None and not (isinstance(value, float) and math.isnan(value)):
        text = str(value).strip().lower()
        if text and text != "nan":
            return text

    s = stage.lower()
    if s.endswith("_federaser_recovery"):
        return "federaser_replay"
    if s.endswith("_proto_lda_recovery") or s.endswith("_prototype_lda_recovery"):
        return "prototype_lda"
    return "none"


def normalize_unlearner(value, stage: str) -> str:
    if value is not None and not (isinstance(value, float) and math.isnan(value)):
        text = str(value).strip().lower()
        if text and text != "nan":
            return text

    s = stage.lower()

    for suffix in (
        "_federaser_recovery",
        "_proto_lda_recovery",
        "_prototype_lda_recovery",
        "_pre_recovery",
    ):
        if s.endswith(suffix):
            return s[: -len(suffix)]

    if s == "full_after_fl":
        return "none"

    return "unknown"


def model_path_for_stage(run_dir: Path, stage: str) -> Path:
    models_dir = run_dir / "models"
    if stage == "full_after_fl":
        return models_dir / "model_full.pt"
    return models_dir / f"model_{stage}.pt"


def get_time_from_row(row: pd.Series) -> tuple[float, str]:
    for col in TIME_CANDIDATES:
        if col in row.index and pd.notna(row[col]):
            try:
                return float(row[col]), col
            except (TypeError, ValueError):
                pass
    return float("nan"), "not_recorded"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="统计某一次实验中各恢复方法的时间和存储开销"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="某一次具体实验目录，例如 runs/20260926-xxxxxx_main_exp",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="输出目录；默认 <run-dir>/overhead_analysis",
    )
    parser.add_argument(
        "--include-none",
        action="store_true",
        help="是否同时统计 recovery=none 的记录",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    metrics_path = run_dir / "metrics.csv"

    if not run_dir.exists():
        raise FileNotFoundError(f"实验目录不存在：{run_dir}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"未找到：{metrics_path}")

    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else run_dir / "overhead_analysis"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metrics_path)

    if "stage" not in df.columns:
        raise ValueError("metrics.csv 缺少 stage 列")

    records = []

    for _, row in df.iterrows():
        stage = str(row["stage"])
        recovery = normalize_recovery(row.get("recovery"), stage)
        unlearner = normalize_unlearner(row.get("unlearner"), stage)

        if not args.include_none and recovery == "none":
            continue

        model_path = model_path_for_stage(run_dir, stage)
        model_mb = file_size_mib(model_path)

        aux_files = [
            run_dir / name
            for name in AUX_FILES_BY_RECOVERY.get(recovery, [])
        ]
        aux_mb = sum(file_size_mib(p) for p in aux_files)

        time_sec, time_source = get_time_from_row(row)

        records.append(
            {
                "stage": stage,
                "unlearner": unlearner,
                "recovery": recovery,
                "recovery_time_sec": time_sec,
                "time_source": time_source,
                "model_file": str(model_path),
                "model_storage_mb": model_mb,
                "aux_files": ";".join(str(p) for p in aux_files),
                "aux_storage_mb": aux_mb,
                "total_storage_mb": model_mb + aux_mb,
            }
        )

    detail = pd.DataFrame(records)

    if detail.empty:
        raise RuntimeError(
            "当前 metrics.csv 中没有恢复方法记录。"
            "如果想连 recovery=none 一起统计，请加 --include-none。"
        )

    detail_path = out_dir / "recovery_overhead_detail.csv"
    detail.to_csv(detail_path, index=False)

    summary = (
        detail.groupby(["recovery"], dropna=False)
        .agg(
            count=("stage", "count"),
            recovery_time_sec_mean=("recovery_time_sec", "mean"),
            recovery_time_sec_std=("recovery_time_sec", "std"),
            model_storage_mb_mean=("model_storage_mb", "mean"),
            aux_storage_mb_mean=("aux_storage_mb", "mean"),
            total_storage_mb_mean=("total_storage_mb", "mean"),
        )
        .reset_index()
    )

    numeric_cols = summary.select_dtypes(include="number").columns
    summary[numeric_cols] = summary[numeric_cols].round(4)

    summary_path = out_dir / "recovery_overhead_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("=" * 80)
    print(f"Run dir:  {run_dir}")
    print(f"Metrics:  {metrics_path}")
    print(f"Detail:   {detail_path}")
    print(f"Summary:  {summary_path}")
    print("=" * 80)

    print(
        summary[
            [
                "recovery",
                "count",
                "recovery_time_sec_mean",
                "recovery_time_sec_std",
                "model_storage_mb_mean",
                "aux_storage_mb_mean",
                "total_storage_mb_mean",
            ]
        ].to_string(index=False)
    )

    missing_time = int(detail["recovery_time_sec"].isna().sum())
    if missing_time > 0:
        print()
        print(
            f"[提示] 有 {missing_time} 条恢复记录没有时间字段。"
            "说明该次旧实验的 metrics.csv 没记录 recovery_time_sec。"
        )


if __name__ == "__main__":
    main()
