from __future__ import annotations
import math
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, default_collate
from fulbase.data import make_merged_loader
from fulbase.models import build_model
from fulbase.unlearning.base import BaseUnlearner, UnlearningResult
from fulbase.utils import get_model_parameters, set_model_parameters, state_change_report, set_seed
from fulbase.verification.markers import MarkerDataset, derived_seed

def _set_trainable_layers(model, mode):
    if mode not in {"head", "last", "all"}:
        raise ValueError("layers must be head, last or all")
    names = []
    for name, p in model.named_parameters():
        head = name.startswith(("classifier.", "fc."))
        last = name.startswith(("layer4.", "backbone.layer4."))
        p.requires_grad = mode == "all" or head or (mode == "last" and last)
        if p.requires_grad:
            names.append(name)
    if not names:
        raise ValueError("No trainable parameters selected")
    return names

def _project_to_l2_ball(model, center_parameters, max_delta_norm):
    """Only selected floating parameters; buffers including BN counters stay fixed."""
    if max_delta_norm < 0:
        raise ValueError("Projection radius must be nonnegative")
    centers = dict(zip(model.state_dict(), center_parameters))
    with torch.no_grad():
        selected = [(p, torch.as_tensor(centers[name], device=p.device, dtype=p.dtype))
                    for name, p in model.named_parameters() if p.requires_grad]
        norm = sum(float((p-c).double().square().sum()) for p, c in selected) ** 0.5
        if norm <= max_delta_norm:
            return False
        for p, c in selected:
            p.copy_(c + (p-c) * (max_delta_norm / max(norm, 1e-30)))
    return True

def _collate_pga_batch(batch):
    # Torchvision datasets may return int labels while marker datasets return tensors.
    images, labels = zip(*batch)
    return default_collate(images), torch.tensor([int(label) for label in labels], dtype=torch.long)


class _TaggedPgaSource(Dataset):
    """Attach provenance without changing the source's order or random sampling."""
    def __init__(self, source, num_real):
        self.source = source
        self.num_real = num_real

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        image, label = self.source[index]
        return image, label, index >= self.num_real


def _collate_traced_pga_batch(batch):
    images, labels, markers = zip(*batch)
    x, y = _collate_pga_batch(list(zip(images, labels)))
    return x, y, torch.tensor(markers, dtype=torch.bool)


def _parameter_delta_norm(selected, centers):
    return sum(float((p.detach() - c).double().square().sum())
               for p, c in zip(selected, centers)) ** 0.5


def _require_finite_parameters(parameters, step):
    if any(not bool(torch.isfinite(p).all()) for p in parameters):
        raise FloatingPointError(f"Non-finite PGA parameters at step {step}")


def pga_unlearn(ctx, steps=100, lr=1e-4, project_ratio=0.01, layers="head", data_scope="real",
                *, trace_steps=False):
    """Approximate ascent baseline with target training access. Never audit inputs."""
    if steps < 0 or lr <= 0 or project_ratio < 0:
        raise ValueError("Invalid PGA budget")
    if data_scope not in {"real", "real_and_markers", "markers_only"}:
        raise ValueError("Unknown PGA data scope")
    if trace_steps and not all(math.isfinite(x) for x in (steps, lr, project_ratio)):
        raise ValueError("PGA trace requires finite budgets")
    set_seed(derived_seed(ctx.cfg.seed, "pga", data_scope))
    model = build_model(dataset=ctx.dataset, num_classes=ctx.num_classes)
    set_model_parameters(model, ctx.final_parameters)
    model.to(ctx.device).eval()  # freeze all BN running state, including all-layer PGA
    names = _set_trainable_layers(model, layers)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr)
    datasets = []
    num_real = 0
    if data_scope != "markers_only":
        datasets.append(make_merged_loader(ctx.fd.trainset, ctx.fd.train_indices,
                        ctx.forget_client_ids, ctx.batch_size).dataset)
        num_real = len(datasets[0])
    if data_scope != "real":
        if ctx.marker_config is None or not ctx.marker_config.enabled:
            raise ValueError("Marker ascent requires enabled training markers")
        datasets.extend(MarkerDataset(ctx.marker_config, cid, ctx.dataset, "train",
                        derived_seed(ctx.cfg.seed, "train-markers"), ctx.num_classes)
                        for cid in ctx.forget_client_ids)
    source = ConcatDataset(datasets)
    if not len(source):
        raise ValueError("Empty PGA source")
    if trace_steps:
        source = _TaggedPgaSource(source, num_real)
    loader = DataLoader(source, batch_size=ctx.batch_size, shuffle=True,
                        collate_fn=_collate_traced_pga_batch if trace_steps else _collate_pga_batch,
                        generator=torch.Generator().manual_seed(derived_seed(ctx.cfg.seed, "pga-batches")))
    batches = iter(loader)
    norm = sum(float(p.detach().double().square().sum())
               for p in model.parameters() if p.requires_grad) ** 0.5
    radius = project_ratio * max(norm, 1e-12)
    projected, losses = 0, []
    trace = []
    selected = [p for p in model.parameters() if p.requires_grad]
    if trace_steps:
        _require_finite_parameters(model.parameters(), 0)
        if not math.isfinite(radius):
            raise FloatingPointError("Non-finite PGA projection radius")
        centers = [p.detach().clone() for p in selected]
    for step in range(1, int(steps) + 1):
        try:
            batch = next(batches)
        except StopIteration:
            batches = iter(loader)
            batch = next(batches)
        x, y = batch[:2]
        y = y.to(ctx.device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x.to(ctx.device))
        loss = nn.functional.cross_entropy(logits, y)
        if trace_steps and not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Non-finite PGA loss at step {step}")
        (-loss).backward()
        if trace_steps:
            gradients = [p.grad for p in selected if p.grad is not None]
            if any(not bool(torch.isfinite(g).all()) for g in gradients):
                raise FloatingPointError(f"Non-finite PGA gradient at step {step}")
            gradient_norm = sum(float(g.detach().double().square().sum()) for g in gradients) ** 0.5
            if not math.isfinite(gradient_norm):
                raise FloatingPointError(f"Non-finite PGA gradient norm at step {step}")
        optimizer.step()
        if trace_steps:
            _require_finite_parameters(selected, step)
            before_projection = _parameter_delta_norm(selected, centers)
        was_projected = _project_to_l2_ball(model, ctx.final_parameters, radius)
        projected += int(was_projected)
        losses.append(float(loss.detach()))
        if trace_steps:
            _require_finite_parameters(selected, step)
            after_projection = _parameter_delta_norm(selected, centers)
            if not all(math.isfinite(n) for n in (before_projection, after_projection)):
                raise FloatingPointError(f"Non-finite PGA update norm at step {step}")
            marker_mask = batch[2].to(ctx.device)
            per_sample = nn.functional.cross_entropy(logits.detach(), y, reduction="none")
            marker_count = int(marker_mask.sum())
            real_count = len(y) - marker_count
            trace.append({
                "step": step, "loss": losses[-1],
                "real_samples": real_count, "marker_samples": marker_count,
                "real_loss": float(per_sample[~marker_mask].mean()) if real_count else None,
                "marker_loss": float(per_sample[marker_mask].mean()) if marker_count else None,
                "gradient_l2_norm": gradient_norm,
                "delta_l2_before_projection": before_projection,
                "delta_l2_after_projection": after_projection,
                "projected": bool(was_projected),
            })
    logs = {
        "steps": steps, "lr": lr, "project_ratio": project_ratio,
        "max_delta_norm": radius, "layers": layers, "trainable_names": names,
        "bn_policy": "frozen_running_state", "data_scope": data_scope,
        "projected_times": projected, "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        **state_change_report(model, ctx.final_parameters)}
    if trace_steps:
        logs["step_trace"] = trace
    return UnlearningResult("pga", get_model_parameters(model), logs)

class PGAUnlearner(BaseUnlearner):
    name = "pga"
    def unlearn(self, ctx):
        return pga_unlearn(ctx, steps=ctx.cfg.pga_steps, lr=ctx.cfg.pga_lr,
                           project_ratio=ctx.cfg.pga_project_ratio, layers=ctx.cfg.pga_layers,
                           data_scope=getattr(ctx.cfg, "pga_data_scope", "real"))
