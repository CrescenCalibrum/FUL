"""Bounded training-marker attenuation with optional retained-marker consistency.

Only synthetic training markers enter this optimizer. The fixed Full supplies
per-sample targets and teacher logits; task/test/calibration data stay outside.
"""
from __future__ import annotations

from copy import deepcopy
import math
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from fulbase.unlearning import pga as pga_api
from fulbase.unlearning.base import UnlearningResult
from fulbase.utils import get_model_parameters, set_model_parameters, set_seed, state_change_report
from fulbase.verification.markers import derived_seed


def capped_margin_loss(logits, labels, full_margin, raw_drop):
    """Per-sample raw-logit hinge; an inactive sample has zero target gradient."""
    margin = pga_api._correct_class_margin(logits, labels)
    return nn.functional.relu(margin - (full_margin.detach() - raw_drop)).mean()


def centered_logit_mse(logits, full_logits):
    """Mean over samples AND classes; teacher values never receive gradients."""
    current = logits - logits.mean(dim=1, keepdim=True)
    teacher = full_logits.detach() - full_logits.detach().mean(dim=1, keepdim=True)
    return (current - teacher).square().mean()


def _finite(tensor, label):
    if not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError(f"Non-finite capped marker {label}")


class _CachedMarkers(Dataset):
    def __init__(self, images, labels, logits, sample_ids, client_ids):
        self.images, self.labels, self.logits = images, labels, logits
        self.sample_ids, self.client_ids = sample_ids, client_ids
        self.full_margin = pga_api._correct_class_margin(logits, labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index], self.full_margin[index], index


def _cache_markers(ctx, teacher, client_ids):
    """Sequential private cache construction, independent of the target sampler."""
    caches = {}
    for cid in client_ids:
        dataset = pga_api.MarkerDataset(
            ctx.marker_config, cid, ctx.dataset, "train",
            derived_seed(ctx.cfg.seed, "train-markers"), ctx.num_classes)
        if not len(dataset):
            raise ValueError("Empty training-marker client")
        images, labels = pga_api._collate_pga_batch([dataset[i] for i in range(len(dataset))])
        _finite(images, "inputs")
        chunks = []
        with torch.no_grad():
            for start in range(0, len(labels), ctx.batch_size):
                outputs = teacher(images[start:start + ctx.batch_size].to(ctx.device))
                _finite(outputs, "teacher logits")
                chunks.append(outputs.detach().cpu())
        ids = list(getattr(dataset, "ids", [f"{cid}:train:{i}" for i in range(len(dataset))]))
        if len(ids) != len(labels) or len(set(ids)) != len(ids):
            raise ValueError("Invalid training-marker IDs")
        caches[cid] = _CachedMarkers(images.cpu(), labels.cpu(), torch.cat(chunks), ids,
                                    [cid] * len(labels))
    return caches


def _merge_caches(caches):
    values = list(caches.values())
    if not values:
        raise ValueError("Empty target training-marker set")
    ids = [sample_id for cache in values for sample_id in cache.sample_ids]
    if len(ids) != len(set(ids)):
        raise ValueError("Overlapping training-marker IDs")
    return _CachedMarkers(torch.cat([c.images for c in values]),
                          torch.cat([c.labels for c in values]),
                          torch.cat([c.logits for c in values]), ids,
                          [cid for c in values for cid in c.client_ids])


def _target_loader(cache, ctx):
    # The generator and number/order of source samples match original marker PGA.
    return DataLoader(cache, batch_size=ctx.batch_size, shuffle=True,
                      generator=torch.Generator().manual_seed(derived_seed(ctx.cfg.seed, "pga-batches")))


def _all_target_loss(model, cache, ctx, raw_drop):
    total = None
    for start in range(0, len(cache), ctx.batch_size):
        end = min(start + ctx.batch_size, len(cache))
        logits = model(cache.images[start:end].to(ctx.device))
        _finite(logits, "target logits")
        term = capped_margin_loss(logits, cache.labels[start:end].to(ctx.device),
                                  cache.full_margin[start:end].to(ctx.device), raw_drop)
        term = term * ((end - start) / len(cache))
        total = term if total is None else total + term
    return total


def _retain_loss(model, caches, ctx):
    total = None
    per_client = {}
    for cid, cache in caches.items():
        client_loss = None
        for start in range(0, len(cache), ctx.batch_size):
            end = min(start + ctx.batch_size, len(cache))
            logits = model(cache.images[start:end].to(ctx.device))
            _finite(logits, "retained logits")
            term = centered_logit_mse(logits, cache.logits[start:end].to(ctx.device))
            # A shorter final batch has proportionately less weight.
            term = term * ((end - start) / len(cache))
            client_loss = term if client_loss is None else client_loss + term
        per_client[str(cid)] = float(client_loss.detach())
        total = client_loss if total is None else total + client_loss
    return total / len(caches), per_client


def _gradient_norm(loss, selected, *, retain_graph=False):
    _finite(loss, "loss")
    gradients = torch.autograd.grad(loss, selected, retain_graph=retain_graph, allow_unused=True)
    norm2 = 0.0
    for gradient in gradients:
        if gradient is not None:
            _finite(gradient, "gradient")
            norm2 += float(gradient.detach().double().square().sum())
    norm = math.sqrt(norm2)
    if not math.isfinite(norm):
        raise FloatingPointError("Non-finite capped marker gradient norm")
    return norm


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _project_inside(model, ctx, radius, selected, centers):
    projected = pga_api._project_to_l2_ball(model, ctx.final_parameters, radius)
    if projected and pga_api._parameter_delta_norm(selected, centers) > radius:
        # Casting center + scaled_delta to float32 can round just outside a tiny
        # ball. One representable step toward Full removes that outward rounding.
        with torch.no_grad():
            for p, center in zip(selected, centers):
                p.copy_(torch.nextafter(p, center))
        if pga_api._parameter_delta_norm(selected, centers) > radius:
            raise FloatingPointError("Capped marker projection exceeded its radius after rounding")
    return projected


def _probe_weight(model, target, retained, ctx, raw_drop, lr, radius, multiplier):
    """One disposable B step, followed by full TRAINING-set gradient measurement."""
    _sync(ctx.device)
    started = time.perf_counter()
    probe = deepcopy(model)
    selected = [p for p in probe.parameters() if p.requires_grad]
    centers = [p.detach().clone() for p in selected]
    optimizer = torch.optim.SGD(selected, lr=lr)
    x, y, reference, _ = next(iter(_target_loader(target, ctx)))
    logits = probe(x.to(ctx.device))
    _finite(logits, "probe logits")
    probe_loss = capped_margin_loss(logits, y.to(ctx.device), reference.to(ctx.device), raw_drop)
    _finite(probe_loss, "probe target loss")
    optimizer.zero_grad(set_to_none=True)
    probe_loss.backward()
    for p in selected:
        if p.grad is not None:
            _finite(p.grad, "probe update gradient")
    optimizer.step()
    pga_api._require_finite_parameters(selected, "probe")
    projected = _project_inside(probe, ctx, radius, selected, centers)
    target_loss = _all_target_loss(probe, target, ctx, raw_drop)
    gb = _gradient_norm(target_loss, selected)
    retain_loss, _ = _retain_loss(probe, retained, ctx)
    gr = _gradient_norm(retain_loss, selected)
    if gb <= 1e-12 or gr <= 1e-12:
        raise ValueError("Probe gradient is near zero; use an explicitly chosen fixed retain weight")
    base_weight = gb / gr
    effective = multiplier * base_weight
    if not math.isfinite(base_weight) or not math.isfinite(effective):
        raise FloatingPointError("Non-finite probed retain weight")
    _sync(ctx.device)
    details = {
        "policy": "one_B_step_then_full_training_marker_gradient_ratio",
        "target_gradient_l2_norm": gb, "retain_gradient_l2_norm": gr,
        "base_weight": base_weight, "weight_multiplier": multiplier,
        "effective_weight": effective, "projected": bool(projected),
        "parameter_delta_l2": pga_api._parameter_delta_norm(selected, centers),
        "time_sec": time.perf_counter() - started,
        "update_target_samples": len(y), "target_samples": len(target),
        "retain_samples": sum(len(c) for c in retained.values()),
        "formal_candidate_starts_from": "original_full_not_probe",
    }
    del probe
    return effective, details


def _final_target_summary(model, target, ctx, raw_drop):
    margins, normalized, correct = [], [], []
    with torch.no_grad():
        for start in range(0, len(target), ctx.batch_size):
            end = min(start + ctx.batch_size, len(target))
            logits = model(target.images[start:end].to(ctx.device))
            _finite(logits, "final target logits")
            labels = target.labels[start:end].to(ctx.device)
            margin = pga_api._correct_class_margin(logits, labels)
            scale = (logits - logits.mean(1, keepdim=True)).norm(dim=1).clamp_min(1e-12)
            margins.append(margin.cpu())
            normalized.append((margin / scale).cpu())
            correct.append((logits.argmax(1) == labels).cpu())
    margin = torch.cat(margins)
    residual = margin - (target.full_margin - raw_drop)
    return {"raw_margin": float(margin.mean()),
            "normalized_margin": float(torch.cat(normalized).mean()),
            "accuracy": float(torch.cat(correct).float().mean()),
            "target_loss": float(residual.relu().mean()),
            "target_reached_fraction": float((residual <= 0).float().mean()),
            "mean_raw_margin_drop": float((target.full_margin - margin).mean())}


def marker_capped_unlearn(ctx, *, steps, lr, project_ratio, layers, data_scope,
                          trace_steps, objective, margin_raw_drop,
                          retain_weight, retain_weight_mode):
    if objective not in {"margin_capped", "margin_capped_retain"}:
        raise ValueError("Unknown capped marker objective")
    if data_scope != "markers_only":
        raise ValueError("Capped marker objectives require markers_only")
    if layers != "head":
        raise ValueError("Capped marker objectives currently require head layers")
    if type(steps) is not int or steps < 0 or not all(
            math.isfinite(v) for v in (lr, project_ratio, margin_raw_drop, retain_weight)):
        raise ValueError("Capped marker budgets must be finite and steps a nonnegative integer")
    if lr <= 0 or project_ratio < 0 or margin_raw_drop <= 0 or retain_weight < 0:
        raise ValueError("Invalid capped marker budget/weight")
    if retain_weight_mode not in {"fixed", "probe_ratio"}:
        raise ValueError("retain_weight_mode must be fixed or probe_ratio")
    if ctx.marker_config is None or not ctx.marker_config.enabled:
        raise ValueError("Capped erasure requires enabled training markers")
    constrained = objective == "margin_capped_retain"
    forget, retain = list(ctx.forget_client_ids), list(ctx.retain_client_ids)
    if not forget or len(set(forget)) != len(forget) or any(c < 0 for c in forget):
        raise ValueError("Invalid target client IDs")
    if constrained and (not retain or len(set(retain)) != len(retain)
                        or any(c < 0 for c in retain) or set(forget) & set(retain)):
        raise ValueError("C requires nonempty, disjoint retained client IDs")

    set_seed(derived_seed(ctx.cfg.seed, "pga", data_scope))
    model = pga_api.build_model(dataset=ctx.dataset, num_classes=ctx.num_classes)
    set_model_parameters(model, ctx.final_parameters)
    model.to(ctx.device).eval()
    names = pga_api._set_trainable_layers(model, layers)
    selected = [p for p in model.parameters() if p.requires_grad]
    pga_api._require_finite_parameters(model.parameters(), 0)
    centers = [p.detach().clone() for p in selected]
    norm = sum(float(p.detach().double().square().sum()) for p in selected) ** .5
    radius = project_ratio * max(norm, 1e-12)
    if not math.isfinite(radius):
        raise FloatingPointError("Non-finite capped marker projection radius")
    teacher = deepcopy(model).eval().requires_grad_(False)
    _sync(ctx.device)
    started = time.perf_counter()
    target_clients = _cache_markers(ctx, teacher, forget)
    target = _merge_caches(target_clients)
    retained = _cache_markers(ctx, teacher, retain) if constrained else {}
    target_ids = target.sample_ids
    retain_ids = [sample_id for cache in retained.values() for sample_id in cache.sample_ids]
    if set(target_ids) & set(retain_ids):
        raise ValueError("Target and retained marker IDs overlap")
    _sync(ctx.device)
    cache_time = time.perf_counter() - started
    effective_weight, probe = (retain_weight if constrained else 0.0), None
    if constrained and retain_weight_mode == "probe_ratio":
        effective_weight, probe = _probe_weight(
            model, target, retained, ctx, margin_raw_drop, lr, radius, retain_weight)
    loader = _target_loader(target, ctx)
    batches = iter(loader)
    optimizer = torch.optim.SGD(selected, lr=lr)
    trace, scores, projected_times = [], [], 0
    retained_count = sum(len(c) for c in retained.values())
    for step in range(1, steps + 1):
        try:
            x, y, reference, indices = next(batches)
        except StopIteration:
            batches = iter(loader)
            x, y, reference, indices = next(batches)
        x, y, reference = x.to(ctx.device), y.to(ctx.device), reference.to(ctx.device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        _finite(logits, "target logits")
        target_loss = capped_margin_loss(logits, y, reference, margin_raw_drop)
        retain_loss, per_client = (_retain_loss(model, retained, ctx) if constrained
                                   else (logits.new_zeros(()), {}))
        loss = target_loss + effective_weight * retain_loss if constrained else target_loss
        _finite(loss, "combined loss")
        target_gradient = (_gradient_norm(target_loss, selected, retain_graph=True)
                           if trace_steps else None)
        retain_gradient = (_gradient_norm(retain_loss, selected, retain_graph=True)
                           if trace_steps and constrained else 0.0)
        loss.backward()  # Equivalent to (-score).backward(), where score=-loss.
        norm2 = 0.0
        for p in selected:
            if p.grad is not None:
                _finite(p.grad, "update gradient")
                norm2 += float(p.grad.detach().double().square().sum())
        gradient_norm = math.sqrt(norm2)
        if not math.isfinite(gradient_norm):
            raise FloatingPointError("Non-finite capped marker update gradient norm")
        optimizer.step()
        pga_api._require_finite_parameters(selected, step)
        before_projection = pga_api._parameter_delta_norm(selected, centers)
        projected = _project_inside(model, ctx, radius, selected, centers)
        after_projection = pga_api._parameter_delta_norm(selected, centers)
        pga_api._require_finite_parameters(selected, step)
        if not all(math.isfinite(v) for v in (before_projection, after_projection)):
            raise FloatingPointError("Non-finite capped marker parameter displacement")
        projected_times += int(projected)
        scores.append(-float(loss.detach()))
        if trace_steps:
            raw = pga_api._correct_class_margin(logits.detach(), y)
            scale = (logits.detach() - logits.detach().mean(1, keepdim=True)).norm(dim=1).clamp_min(1e-12)
            residual = raw - (reference - margin_raw_drop)
            ce = float(nn.functional.cross_entropy(logits.detach(), y))
            trace.append({
                "step": step, "loss": scores[-1], "objective": objective,
                "target_loss": float(target_loss.detach()),
                "retain_marker_consistency_loss": float(retain_loss.detach()),
                "weighted_retain_loss": effective_weight * float(retain_loss.detach()),
                "retain_weight_effective": effective_weight,
                "target_active_fraction": float((residual > 0).float().mean()),
                "target_reached_fraction": float((residual <= 0).float().mean()),
                "target_gradient_l2_norm": target_gradient,
                "retain_gradient_l2_norm": retain_gradient,
                "weighted_retain_gradient_l2_norm": effective_weight * retain_gradient,
                "retain_client_mse": per_client,
                "target_sample_ids": [target.sample_ids[int(i)] for i in indices],
                "ce_loss": ce, "raw_margin": float(raw.mean()),
                "normalized_margin": float((raw / scale).mean()),
                "batch_accuracy": float((logits.detach().argmax(1) == y).float().mean()),
                "real_samples": 0, "marker_samples": len(y),
                "retain_marker_samples": retained_count,
                "real_loss": None, "marker_loss": ce,
                "gradient_l2_norm": gradient_norm,
                "delta_l2_before_projection": before_projection,
                "delta_l2_after_projection": after_projection, "projected": bool(projected),
            })
    teacher_change = state_change_report(teacher, ctx.final_parameters)
    if teacher_change["changed_parameters"] or teacher_change["changed_buffers"]:
        raise RuntimeError("Capped marker teacher was modified")
    final_target = _final_target_summary(model, target, ctx, margin_raw_drop)
    with torch.no_grad():
        final_retain, final_clients = (_retain_loss(model, retained, ctx) if constrained
                                       else (torch.tensor(0.0), {}))
    _finite(final_retain, "final retain consistency loss")
    logs = {
        "steps": steps, "lr": lr, "project_ratio": project_ratio,
        "objective": objective, "optimization_direction": "maximize",
        "loss_definition": "negative_per_sample_raw_margin_hinge" +
                           ("_and_weighted_retained_centered_logit_mse" if constrained else ""),
        "group_loss_definition": "mean_cross_entropy",
        "margin_raw_drop": margin_raw_drop,
        "target_definition": "raw_margin_full_per_sample_minus_margin_raw_drop",
        "retain_loss_definition": "mean_over_clients_of_mean_over_samples_and_classes_centered_logit_squared_error",
        "retain_weight": retain_weight if constrained else 0.0,
        "retain_weight_mode": retain_weight_mode if constrained else "not_applicable",
        "retain_weight_effective": effective_weight, "retain_probe": probe,
        "teacher_cache_time_sec": cache_time, "teacher_state_unchanged": True,
        "target_training_sample_ids": target_ids, "retain_training_sample_ids": retain_ids,
        "access": {"optimization": "target_and_retained_training_markers" if constrained else "target_training_markers_only",
                   "role": "train", "real_training_data": False,
                   "target_clients": forget, "retain_clients": retain if constrained else [],
                   "target_samples": len(target), "retain_samples": retained_count,
                   "retain_samples_per_step": retained_count,
                   "teacher_cache_samples": len(target) + retained_count},
        "final_training_target": final_target,
        "final_retain_marker_consistency_loss": float(final_retain),
        "final_retain_client_mse": final_clients,
        "trace_observation_timing": "training_inputs_before_update",
        "normalized_margin_definition": "raw_margin/centered_logits_l2_norm_clamped_1e-12",
        "max_delta_norm": radius, "layers": layers, "trainable_names": names,
        "bn_policy": "frozen_running_state", "data_scope": data_scope,
        "projected_times": projected_times,
        "first_loss": scores[0] if scores else None, "last_loss": scores[-1] if scores else None,
        **state_change_report(model, ctx.final_parameters),
    }
    if trace_steps:
        logs["step_trace"] = trace
    return UnlearningResult(f"marker_{objective}_erase", get_model_parameters(model), logs)
