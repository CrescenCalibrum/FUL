from __future__ import annotations

from itertools import cycle

import numpy as np
import torch
from torch import nn

from fulbase.data import make_merged_loader
from fulbase.models import build_model
from fulbase.unlearning.base import BaseUnlearner, UnlearningContext, UnlearningResult
from fulbase.utils import arrays_l2_norm, diff_arrays, get_model_parameters, set_model_parameters


def _set_trainable_layers(model: nn.Module, mode: str) -> list[str]:
    """控制 PGA 更新哪些层。SmallCNN 推荐 head；ResNet18 后续可用 last。"""
    mode = mode.lower()
    trainable_names: list[str] = []
    for name, p in model.named_parameters():
        if mode == "all":
            p.requires_grad = True
        elif mode == "head":
            p.requires_grad = ("classifier" in name or name.startswith("fc."))
        elif mode == "last":
            p.requires_grad = (
                "classifier" in name
                or name.startswith("fc.")
                or name.startswith("layer4")
            )
        else:
            raise ValueError(f"Unsupported pga_layers={mode}; choose from head/last/all")
        if p.requires_grad:
            trainable_names.append(name)
    if not trainable_names:
        raise ValueError(f"No trainable parameters selected for pga_layers={mode}")
    return trainable_names


def _project_to_l2_ball(
    model: nn.Module,
    center_parameters: list[np.ndarray],
    max_delta_norm: float,
) -> bool:
    """把当前模型参数投影回 final_parameters 附近的 L2 球，避免梯度上升把模型打崩。"""
    if max_delta_norm <= 0:
        return False
    current = get_model_parameters(model)
    diff = diff_arrays(current, center_parameters)
    norm = arrays_l2_norm(diff)
    if norm <= max_delta_norm or norm <= 1e-12:
        return False
    scale = max_delta_norm / norm
    projected = [(c + d * scale).astype(c.dtype) for c, d in zip(center_parameters, diff)]
    set_model_parameters(model, projected)
    return True


def pga_unlearn(
    ctx: UnlearningContext,
    steps: int = 100,
    lr: float = 1e-4,
    project_ratio: float = 0.01,
    layers: str = "head",
) -> UnlearningResult:
    """Projected Gradient Ascent 联邦遗忘近似实现。

    该实现用于实验 baseline：在被遗忘客户端数据上做梯度上升，使 forget loss 增大。
    注意：它需要访问被遗忘客户端数据，因此论文中应说明它是“目标客户端参与/可访问遗忘集”的 baseline。
    """
    model = build_model(dataset=ctx.dataset, num_classes=ctx.num_classes)
    set_model_parameters(model, ctx.final_parameters)
    model.to(ctx.device)
    model.train()

    trainable_names = _set_trainable_layers(model, layers)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.0)
    criterion = nn.CrossEntropyLoss()

    forget_loader = make_merged_loader(
        ctx.fd.trainset,
        ctx.fd.train_indices,
        ctx.forget_client_ids,
        batch_size=ctx.batch_size,
        shuffle=True,
    )
    loader_iter = cycle(forget_loader)

    center_norm = arrays_l2_norm(ctx.final_parameters)
    max_delta_norm = float(project_ratio) * max(center_norm, 1e-12)
    projected_times = 0
    losses: list[float] = []

    for _ in range(int(steps)):
        x, y = next(loader_iter)
        x, y = x.to(ctx.device), y.to(ctx.device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        # 最小化 -loss 等价于最大化 forget loss。
        (-loss).backward()
        optimizer.step()
        if _project_to_l2_ball(model, ctx.final_parameters, max_delta_norm):
            projected_times += 1
        losses.append(float(loss.detach().cpu().item()))

    return UnlearningResult(
        name="pga",
        parameters=get_model_parameters(model),
        logs={
            "steps": int(steps),
            "lr": float(lr),
            "project_ratio": float(project_ratio),
            "max_delta_norm": float(max_delta_norm),
            "layers": layers,
            "trainable_names": trainable_names,
            "projected_times": int(projected_times),
            "first_loss": losses[0] if losses else None,
            "last_loss": losses[-1] if losses else None,
        },
    )


class PGAUnlearner(BaseUnlearner):
    name = "pga"

    def unlearn(self, ctx: UnlearningContext) -> UnlearningResult:
        return pga_unlearn(
            ctx=ctx,
            steps=int(getattr(ctx.cfg, "pga_steps", 100)),
            lr=float(getattr(ctx.cfg, "pga_lr", 1e-4)),
            project_ratio=float(getattr(ctx.cfg, "pga_project_ratio", 0.01)),
            layers=str(getattr(ctx.cfg, "pga_layers", "head")),
        )
