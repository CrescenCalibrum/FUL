from __future__ import annotations
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
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

def pga_unlearn(ctx, steps=100, lr=1e-4, project_ratio=0.01, layers="head", data_scope="real"):
    """Approximate ascent baseline with target training access. Never audit inputs."""
    if steps < 0 or lr <= 0 or project_ratio < 0:
        raise ValueError("Invalid PGA budget")
    if data_scope not in {"real", "real_and_markers", "markers_only"}:
        raise ValueError("Unknown PGA data scope")
    set_seed(derived_seed(ctx.cfg.seed, "pga", data_scope))
    model = build_model(dataset=ctx.dataset, num_classes=ctx.num_classes)
    set_model_parameters(model, ctx.final_parameters)
    model.to(ctx.device).eval()  # freeze all BN running state, including all-layer PGA
    names = _set_trainable_layers(model, layers)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr)
    datasets = []
    if data_scope != "markers_only":
        datasets.append(make_merged_loader(ctx.fd.trainset, ctx.fd.train_indices,
                        ctx.forget_client_ids, ctx.batch_size).dataset)
    if data_scope != "real":
        if ctx.marker_config is None or not ctx.marker_config.enabled:
            raise ValueError("Marker ascent requires enabled training markers")
        datasets.extend(MarkerDataset(ctx.marker_config, cid, ctx.dataset, "train",
                        derived_seed(ctx.cfg.seed, "train-markers"), ctx.num_classes)
                        for cid in ctx.forget_client_ids)
    source = ConcatDataset(datasets)
    if not len(source):
        raise ValueError("Empty PGA source")
    loader = DataLoader(source, batch_size=ctx.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(derived_seed(ctx.cfg.seed, "pga-batches")))
    batches = iter(loader)
    norm = sum(float(p.detach().double().square().sum())
               for p in model.parameters() if p.requires_grad) ** 0.5
    radius = project_ratio * max(norm, 1e-12)
    projected, losses = 0, []
    for _ in range(int(steps)):
        try:
            x, y = next(batches)
        except StopIteration:
            batches = iter(loader)
            x, y = next(batches)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(x.to(ctx.device)), y.to(ctx.device))
        (-loss).backward()
        optimizer.step()
        projected += int(_project_to_l2_ball(model, ctx.final_parameters, radius))
        losses.append(float(loss.detach()))
    return UnlearningResult("pga", get_model_parameters(model), {
        "steps": steps, "lr": lr, "project_ratio": project_ratio,
        "max_delta_norm": radius, "layers": layers, "trainable_names": names,
        "bn_policy": "frozen_running_state", "data_scope": data_scope,
        "projected_times": projected, "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        **state_change_report(model, ctx.final_parameters)})

class PGAUnlearner(BaseUnlearner):
    name = "pga"
    def unlearn(self, ctx):
        return pga_unlearn(ctx, steps=ctx.cfg.pga_steps, lr=ctx.cfg.pga_lr,
                           project_ratio=ctx.cfg.pga_project_ratio, layers=ctx.cfg.pga_layers,
                           data_scope=getattr(ctx.cfg, "pga_data_scope", "real"))
