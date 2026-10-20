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
) -> float:
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    total_loss, steps = 0.0, 0
    for _ in range(epochs):
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            steps += 1
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
