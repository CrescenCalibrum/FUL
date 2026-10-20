from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from fulbase.data import FederatedData, make_client_loaders
from fulbase.models import build_model
from fulbase.utils import set_model_parameters


@dataclass
class PrototypeBank:
    mean: np.ndarray      # [C, D]
    var: np.ndarray       # [C, D]
    count: np.ndarray     # [C]
    num_classes: int
    feature_dim: int
    dp_sigma: float
    clip_norm: float


def _clip_features(z: torch.Tensor, clip_norm: float) -> torch.Tensor:
    if clip_norm <= 0:
        return z
    norm = z.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    scale = (clip_norm / norm).clamp(max=1.0)
    return z * scale


@torch.no_grad()
def compute_client_feature_stats(
    model: torch.nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
    clip_norm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.to(device)
    model.eval()
    sums = None
    sumsqs = None
    counts = np.zeros(num_classes, dtype=np.float64)
    for x, y in loader:
        x = x.to(device)
        y_np = y.numpy()
        z = model.features(x)
        z = _clip_features(z, clip_norm)
        z_np = z.detach().cpu().numpy().astype(np.float64)
        if sums is None:
            feature_dim = z_np.shape[1]
            sums = np.zeros((num_classes, feature_dim), dtype=np.float64)
            sumsqs = np.zeros((num_classes, feature_dim), dtype=np.float64)
        for c in range(num_classes):
            mask = y_np == c
            if np.any(mask):
                zc = z_np[mask]
                sums[c] += zc.sum(axis=0)
                sumsqs[c] += (zc ** 2).sum(axis=0)
                counts[c] += int(mask.sum())
    if sums is None:
        raise RuntimeError("Empty loader when computing prototypes")
    return sums, sumsqs, counts


def collect_dp_prototypes(
    fd: FederatedData,
    model_parameters: list[np.ndarray],
    dataset: str,
    batch_size: int,
    device: str,
    exclude_clients: Iterable[int] | None = None,
    dp_sigma: float = 0.0,
    clip_norm: float = 10.0,
    seed: int = 42,
) -> PrototypeBank:
    """Simulate secure aggregation of noisy class-level prototype statistics.

    注意：这是仿真版。代码逐个客户端计算本地统计，但 server 侧只保留聚合后的 noisy sums/sumsqs/counts。
    后续如果要更贴近 Flower Message API，可把每个客户端统计作为一类自定义 Record 上报。
    """
    rng = np.random.default_rng(seed)
    excluded = set(int(x) for x in (exclude_clients or []))
    model = build_model(dataset=dataset, num_classes=fd.num_classes)
    set_model_parameters(model, model_parameters)
    dev = torch.device(device)

    total_sums = None
    total_sumsqs = None
    total_counts = np.zeros(fd.num_classes, dtype=np.float64)
    used_clients = [cid for cid in range(len(fd.train_indices)) if cid not in excluded]

    for cid in used_clients:
        trainloader, _ = make_client_loaders(fd, cid, batch_size=batch_size)
        sums, sumsqs, counts = compute_client_feature_stats(
            model=model,
            loader=trainloader,
            num_classes=fd.num_classes,
            device=dev,
            clip_norm=clip_norm,
        )
        if total_sums is None:
            total_sums = np.zeros_like(sums)
            total_sumsqs = np.zeros_like(sumsqs)
        # 本地裁剪后统计，仿真本地 DP 噪声；真实安全聚合下服务器只见聚合和。
        if dp_sigma > 0:
            sums = sums + rng.normal(0.0, dp_sigma * clip_norm, size=sums.shape)
            sumsqs = sumsqs + rng.normal(0.0, dp_sigma * (clip_norm ** 2), size=sumsqs.shape)
            counts = counts + rng.normal(0.0, max(1e-6, dp_sigma), size=counts.shape)
        total_sums += sums
        total_sumsqs += sumsqs
        total_counts += counts

    if total_sums is None:
        raise ValueError("All clients are excluded; cannot build prototypes.")

    safe_counts = np.maximum(total_counts, 1.0)
    mean = total_sums / safe_counts[:, None]
    second = total_sumsqs / safe_counts[:, None]
    var = np.maximum(second - mean ** 2, 1e-4)
    return PrototypeBank(
        mean=mean.astype(np.float32),
        var=var.astype(np.float32),
        count=np.maximum(total_counts, 0.0).astype(np.float32),
        num_classes=fd.num_classes,
        feature_dim=mean.shape[1],
        dp_sigma=float(dp_sigma),
        clip_norm=float(clip_norm),
    )


def save_prototype_bank(bank: PrototypeBank, path: str) -> None:
    torch.save(bank, path)


def load_prototype_bank(path: str) -> PrototypeBank:
    return torch.load(path, map_location="cpu", weights_only=False)
