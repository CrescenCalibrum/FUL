from __future__ import annotations

from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_one_client(
    model: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    weight_decay: float = 0.0,
    marker_dataset=None,
    marker_config=None,
    marker_cid: int | None = None,
    marker_seed: int = 0,
    training_logs: dict | None = None,
) -> float:
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    total_loss, steps = 0.0, 0
    marker_iter = None
    if marker_dataset is not None:
        if marker_config is None or marker_cid is None:
            raise ValueError("Marker training requires config and client identity")
        from fulbase.verification.markers import marker_loss
        marker_loader = DataLoader(marker_dataset, batch_size=marker_config.batch_size,
                                   shuffle=True, generator=torch.Generator().manual_seed(marker_seed))
        marker_iter = iter(marker_loader)
    marker_total = 0.0
    task_total = 0.0
    component_totals = {"marker_ce": 0.0, "marker_margin": 0.0, "marker_geometry": 0.0}
    for _ in range(epochs):
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            task_total += float(loss.detach())
            if marker_iter is not None:
                try:
                    mx, my = next(marker_iter)
                except StopIteration:
                    marker_iter = iter(marker_loader)
                    mx, my = next(marker_iter)
                components = {}
                mloss = marker_loss(model, mx.to(device), my.to(device), marker_config, marker_cid, components)
                for key, value in components.items():
                    component_totals[key] += value
                marker_total += float(mloss.detach())
                loss = loss + mloss
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            steps += 1
    if training_logs is not None:
        training_logs.update(steps=steps, mean_task_loss=task_total / max(steps, 1),
                             mean_marker_loss=marker_total / max(steps, 1),
                             **{f"mean_{k}": v/max(steps, 1) for k,v in component_totals.items()})
    return total_loss / max(steps, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss_sum, correct, total = 0.0, 0, 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += float(criterion(logits, y).item())
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
        for label in y.unique().detach().cpu().tolist():
            mask = y == int(label)
            class_correct[int(label)] += int((pred[mask] == y[mask]).sum().item())
            class_total[int(label)] += int(mask.sum().item())
    out = {
        "loss": loss_sum / max(total, 1),
        "acc": correct / max(total, 1),
        "num_examples": total,
    }
    for c in sorted(class_total):
        out[f"acc_class_{c}"] = class_correct[c] / max(class_total[c], 1)
    return out
