#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线诊断 Flower 联邦遗忘中的特征表示漂移与 Prototype-LDA 恢复效果。

放在项目 scripts/ 下，从项目根目录运行：
python scripts/diagnose_feature_drift.py --run-dir runs/某次实验目录 --per-class 80

基于此前生成的 ful_flower_base 工程模型保存约定：
- models/model_full.pt
- models/model_<unlearner>_pre_recovery.pt
- models/model_<unlearner>_proto_lda_recovery.pt
- config.json / metrics.csv
- retained_dp_prototypes.pt （可选，存在则另算 DP 原型失配）

脚本不重训练模型，利用同一批固定的保留客户端测试样本抽取分类头输入特征。
重要：仅适用于受信任的本地 .pt 文件，torch.load(weights_only=False) 不应读取不可信来源。
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from fulbase.data import load_federated_data
from fulbase.models import build_model
from fulbase.utils import set_model_parameters


def get_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def instantiate_model(cfg: dict[str, Any], nclasses: int) -> torch.nn.Module:
    """适配此前工厂的 dataset/num_classes，以及常见自定义 model/model_name 参数。"""
    sig = inspect.signature(build_model)
    params = sig.parameters
    kwargs = {}
    for key in ("dataset", "num_classes"):
        if key in params:
            kwargs[key] = cfg.get("dataset", "cifar10") if key == "dataset" else nclasses
    model_id = cfg.get("model", cfg.get("model_name", None))
    for key in ("model", "model_name", "arch", "architecture"):
        if key in params and model_id is not None:
            kwargs[key] = model_id
    try:
        return build_model(**kwargs)
    except Exception as e:
        raise RuntimeError(
            "模型实例化失败：请核对 fulbase.models.build_model 的当前函数签名，"
            "并在 instantiate_model() 中传入训练时实际使用的模型参数。"
        ) from e


def parse_clients(value: Any) -> list[int]:
    if isinstance(value, str):
        return [int(v.strip()) for v in value.split(",") if v.strip()]
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    if isinstance(value, int):
        return [value]
    return []


def extract_reference_loader(fd, cfg: dict, per_class: int, seed: int, batch_size: int):
    forgotten = set(parse_clients(cfg.get("forget_clients", "")))
    if not forgotten:
        raise ValueError("config.json 找不到 forget_clients；请确保使用实际实验保存的配置")
    retain_ids = [cid for cid in range(len(fd.test_indices)) if cid not in forgotten]
    candidates = [idx for cid in retain_ids for idx in fd.test_indices[cid]]
    targets = np.asarray(fd.testset.targets, dtype=np.int64)
    candidates_arr = np.asarray(candidates, dtype=np.int64)
    rng = np.random.default_rng(seed)
    chosen = []
    for c in range(int(fd.num_classes)):
        class_indices = candidates_arr[targets[candidates_arr] == c]
        if len(class_indices) == 0:
            continue
        select = rng.choice(class_indices, min(per_class, len(class_indices)), replace=False)
        chosen.extend(select.tolist())
    if not chosen:
        raise RuntimeError("没有可用于漂移诊断的 retained test samples")
    # 同一批样本、同一顺序供所有模型提取特征。
    loader = DataLoader(Subset(fd.testset, chosen), batch_size=batch_size,
                        shuffle=False, num_workers=0)
    return loader, len(chosen)


def find_head(model: torch.nn.Module, name_override: str | None):
    if name_override:
        return model.get_submodule(name_override)
    for name in ("classifier", "fc", "linear", "head"):
        mod = getattr(model, name, None)
        if isinstance(mod, torch.nn.Module) and not isinstance(mod, torch.nn.Identity):
            return mod
    raise RuntimeError(
        "未找到分类头。请用 --head-name 指定分类头模块名，如 fc/classifier/linear。"
    )


def extract_features(model: torch.nn.Module, loader, device: torch.device,
                     head_name: str | None = None):
    """统一采集实际送入分类头的特征；不依赖 features() 命名。"""
    model.to(device).eval()
    head = find_head(model, head_name)
    features = []
    labels = []
    correct = 0
    total = 0
    captured = []

    def hook(_module, inputs):
        if not inputs:
            raise RuntimeError("分类头没有获得输入，无法定位 penultimate features")
        z = inputs[0]
        if isinstance(z, (list, tuple)):
            z = z[0]
        if not isinstance(z, torch.Tensor):
            raise TypeError("分类头输入不是 Tensor，请调整提取方式")
        captured.append(z.detach().flatten(start_dim=1).cpu().numpy())

    handle = head.register_forward_pre_hook(hook)
    try:
        with torch.inference_mode():
            for x, y in loader:
                captured.clear()
                logits = model(x.to(device))
                if logits.ndim == 2:
                    correct += int((logits.argmax(dim=1).cpu() == y).sum())
                    total += len(y)
                if len(captured) != 1:
                    raise RuntimeError(f"一次 forward 应捕获一批特征，实际捕获 {len(captured)} 次")
                features.append(captured[0])
                labels.append(y.detach().cpu().numpy())
    finally:
        handle.remove()

    return (np.concatenate(features).astype(np.float64),
            np.concatenate(labels), correct / total if total else float("nan"))


def linear_cka(X: np.ndarray, Y: np.ndarray, eps: float = 1e-12) -> float:
    """Centered linear CKA，量化样本关系结构是否一致。"""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    XY = X.T @ Y
    XX = X.T @ X
    YY = Y.T @ Y
    norm_x = np.linalg.norm(XX, ord="fro")
    norm_y = np.linalg.norm(YY, ord="fro")
    if norm_x * norm_y < eps:
        return float("nan")
    return float(np.linalg.norm(XY, ord="fro") ** 2 / (norm_x * norm_y))


def drift_metrics(X: np.ndarray, Y: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    if X.shape != Y.shape:
        raise ValueError(f"特征形状不一致：{X.shape} vs {Y.shape}")
    rel_l2 = float(np.linalg.norm(Y-X) / (np.linalg.norm(X) + 1e-12))
    cos = (X * Y).sum(axis=1) / (
        np.linalg.norm(X, axis=1) * np.linalg.norm(Y, axis=1) + 1e-12)
    cos_drift = float(np.mean(1.0 - cos))
    class_shifts = []
    for c in np.unique(labels):
        xmean = X[labels == c].mean(axis=0)
        ymean = Y[labels == c].mean(axis=0)
        class_shifts.append(float(np.linalg.norm(ymean - xmean) /
                                  (np.linalg.norm(xmean) + 1e-12)))
    cka = linear_cka(X, Y)
    return {
        "relative_l2_drift": rel_l2,
        "cosine_drift": cos_drift,
        "cka": cka,
        "one_minus_cka": 1.0 - cka,
        "mean_class_center_drift": float(np.mean(class_shifts)),
    }


def prototype_mismatch(Z: np.ndarray, labels: np.ndarray, mu: np.ndarray) -> float:
    """测试集类中心与训练期 DP 原型中心的相对距离（宏平均）。"""
    if Z.shape[1] != mu.shape[1]:
        raise ValueError(f"DP 原型维度 {mu.shape[1]} 与诊断特征维度 {Z.shape[1]} 不匹配")
    scores = []
    for c in np.unique(labels):
        if 0 <= int(c) < len(mu):
            center = Z[labels == c].mean(axis=0)
            # 对于 μ 接近 0 的类别应特别注意该归一化量对小分母的敏感性
            scores.append(float(np.linalg.norm(center - mu[int(c)]) /
                                (np.linalg.norm(mu[int(c)]) + 1e-8)))
    return float(np.mean(scores))


def load_parameters(run_dir: Path, stage: str):
    path = run_dir / "models" / ("model_full.pt" if stage == "full_after_fl"
                                   else f"model_{stage}.pt")
    if not path.is_file():
        raise FileNotFoundError(f"缺少模型文件：{path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def load_features_for_stage(run_dir, stage, cfg, num_classes, loader, device, head_name):
    model = instantiate_model(cfg, num_classes)
    params = load_parameters(run_dir, stage)
    try:
        set_model_parameters(model, params)
    except Exception as e:
        raise RuntimeError(
            f"加载 {stage} 参数失败。请确认诊断模型与训练模型的架构、"
            "BatchNorm buffer 顺序、类别数完全一致。"
        ) from e
    result = extract_features(model, loader, device, head_name)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def metric_value(table: pd.DataFrame, stage: str, key="retain_acc") -> float:
    matches = table.loc[table["stage"].astype(str) == stage]
    if len(matches) == 0 or key not in matches.columns:
        return float("nan")
    return float(matches.iloc[-1][key])


def main():
    p = argparse.ArgumentParser(description="离线诊断预恢复特征漂移与 DP Prototype-LDA 恢复效果")
    p.add_argument("--run-dir", type=Path, required=True, help="包含 metrics.csv 的某一次实验目录")
    p.add_argument("--per-class", type=int, default=80, help="各保留类别固定采样数，默认每类80张")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--head-name", type=str, default=None, help="可手工指定分类头模块名")
    p.add_argument("--data-dir", type=str, default=None, help="覆盖 config.json 中的 data_dir")
    args = p.parse_args()
    run_dir = args.run_dir.resolve()
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    table = pd.read_csv(run_dir / "metrics.csv")
    if "stage" not in table.columns:
        raise ValueError("metrics.csv 缺少 stage 列")

    device = get_device(args.device)
    fd = load_federated_data(
        dataset=cfg.get("dataset", "cifar10"),
        data_dir=args.data_dir or cfg.get("data_dir", "./data"),
        num_clients=int(cfg["num_clients"]),
        partition=cfg.get("partition", "dirichlet"),
        alpha=float(cfg.get("alpha", 0.5)),
        seed=int(cfg.get("seed", 42)),
    )
    loader, n = extract_reference_loader(fd, cfg, args.per_class, int(cfg.get("seed", 42)) + 2026,
                                         args.batch_size)
    print(f"固定参考集合：{n} 条保留客户端测试样本；device={device}")

    Z0, labels, ref_acc_full = load_features_for_stage(run_dir, "full_after_fl", cfg,
                                          fd.num_classes, loader, device, args.head_name)
    stages = set(table["stage"].astype(str))
    # 只比较真正以 M_U 为输入的 Prototype-LDA recovery；FedEraser replay 是独立重建，不混用。
    suffixes = ("_proto_lda_recovery", "_prototype_lda_recovery")
    pairs = []
    for recovery_stage in sorted(stages):
        for suffix in suffixes:
            if recovery_stage.endswith(suffix):
                unlearner = recovery_stage[:-len(suffix)]
                pre_stage = f"{unlearner}_pre_recovery"
                if pre_stage in stages:
                    pairs.append((unlearner, pre_stage, recovery_stage))
                break
    if not pairs:
        raise ValueError("没有发现 xxx_pre_recovery + xxx_proto_lda_recovery 这样的阶段配对")

    bank_path = run_dir / "retained_dp_prototypes.pt"
    mu = None
    if bank_path.is_file():
        bank = torch.load(bank_path, map_location="cpu", weights_only=False)
        mu = np.asarray(bank.mean, dtype=np.float64)
    mismatch_full = prototype_mismatch(Z0, labels, mu) if mu is not None else float("nan")

    rows = []
    full_acc = metric_value(table, "full_after_fl")
    for unlearner, pre_stage, recovered_stage in pairs:
        print(f"正在诊断 {unlearner} ...")
        Zu, labels_u, ref_acc_pre = load_features_for_stage(run_dir, pre_stage, cfg,
                                                fd.num_classes, loader, device, args.head_name)
        Zr, labels_r, ref_acc_post = load_features_for_stage(run_dir, recovered_stage, cfg,
                                                fd.num_classes, loader, device, args.head_name)
        assert np.array_equal(labels, labels_u) and np.array_equal(labels, labels_r)
        d = drift_metrics(Z0, Zu, labels)
        d_after = drift_metrics(Zu, Zr, labels)
        acc_pre = metric_value(table, pre_stage)
        acc_post = metric_value(table, recovered_stage)
        gap_to_full = full_acc - acc_pre
        row = {
            "unlearner": unlearner,
            "pre_stage": pre_stage,
            "recovery_stage": recovered_stage,
            "reference_samples": len(labels),
            **d,
            "post_recovery_relative_l2_change": d_after["relative_l2_drift"],
            "post_recovery_one_minus_cka": d_after["one_minus_cka"],
            "prototype_mismatch_full": mismatch_full,
            "prototype_mismatch_unlearned": (
                prototype_mismatch(Zu, labels, mu) if mu is not None else float("nan")
            ),
            "retain_acc_full": full_acc,
            "retain_acc_pre": acc_pre,
            "retain_acc_recovered": acc_post,
            "retain_acc_gain": acc_post - acc_pre,
            "reference_subset_acc_full": ref_acc_full,
            "reference_subset_acc_pre": ref_acc_pre,
            "reference_subset_acc_recovered": ref_acc_post,
            "reference_subset_acc_gain": ref_acc_post - ref_acc_pre,
            "gap_to_full_pre": gap_to_full,
            # 注意它不是对 Retrain 的标准恢复率，只衡量原始模型效用差距填补比例。
            "fraction_of_full_gap_closed": (
                (acc_post - acc_pre) / gap_to_full if gap_to_full > 1e-8 else float("nan")
            ),
        }
        row["excess_prototype_mismatch"] = (
            row["prototype_mismatch_unlearned"] - mismatch_full
        )
        rows.append(row)

    out_dir = run_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "feature_drift_recovery.csv"
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    print("\n", frame[["unlearner", "relative_l2_drift", "one_minus_cka",
                             "excess_prototype_mismatch", "retain_acc_gain"]].to_string(index=False))
    print(f"\n已保存：{output}")

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.8, 4.5))
        x = frame["relative_l2_drift"].to_numpy()
        y = frame["reference_subset_acc_gain"].to_numpy() * 100
        ax.scatter(x, y)
        for xx, yy, name in zip(x, y, frame["unlearner"]):
            ax.annotate(name, (xx, yy), xytext=(5, 5), textcoords="offset points", fontsize=9)
        ax.set_xlabel("Full → unlearned: relative L2 feature drift")
        ax.set_ylabel("Reference-subset recovery gain (percentage points)")
        ax.set_title("Representation drift vs recovery gain (single run)")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        plot = out_dir / "feature_drift_vs_recovery_gain.png"
        fig.savefig(plot, dpi=180)
        plt.close(fig)
        print(f"已保存示意散点图：{plot}")
    except ImportError:
        print("未安装 matplotlib：略过散点图")


if __name__ == "__main__":
    main()
