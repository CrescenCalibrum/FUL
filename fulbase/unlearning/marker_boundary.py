"""Training-only head negative control with per-client retention guards.

The fixed Full representation is cached without feature normalization. Only its
linear classifier is updated. A target hinge seeks actual boundary crossings;
every accepted step obeys exact retained-client accuracy and mean-margin guards.
Neither empirical training eligibility nor a stalled optimizer is an audit.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from fractions import Fraction
import math
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from fulbase.unlearning.base import UnlearningResult
from fulbase.utils import get_model_parameters, state_change_report


@dataclass(frozen=True)
class BoundaryConfig:
    steps: int = 300
    lr: float = .01
    project_ratio: float = .1
    retain_weight: float = .5
    boundary_gap: float = .5
    target_normalized_drop: float = .1
    target_accuracy_drop: float = .2
    retain_margin_drop: float = .08
    retain_accuracy_drop: float = .025
    retain_boundary_buffer: float = .1
    max_backtracks: int = 12
    backtrack_factor: float = .5
    gradient_clip: float = 10.
    min_full_accuracy: float = .7
    min_full_margin: float = .01

    def __post_init__(self):
        if type(self.steps) is not int or self.steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        if type(self.max_backtracks) is not int or self.max_backtracks < 0:
            raise ValueError("max_backtracks must be a nonnegative integer")
        values = {k: v for k, v in asdict(self).items() if k not in {"steps", "max_backtracks"}}
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v)
               for v in values.values()):
            raise ValueError("Boundary optimizer settings must be finite numbers")
        if min(self.lr, self.boundary_gap, self.target_normalized_drop,
               self.retain_boundary_buffer, self.gradient_clip) <= 0:
            raise ValueError("Learning rate, target gaps, retention buffer and gradient clip must be positive")
        if min(self.project_ratio, self.retain_weight, self.retain_margin_drop,
               self.retain_accuracy_drop, self.min_full_margin) < 0:
            raise ValueError("Projection, weights and retention bounds must be nonnegative")
        if not 0 < self.target_accuracy_drop < 1 or not 0 <= self.retain_accuracy_drop < 1:
            raise ValueError("Invalid accuracy-drop fraction")
        if not 0 <= self.min_full_accuracy <= 1 or not 0 < self.backtrack_factor < 1:
            raise ValueError("Invalid Full accuracy or backtracking factor")


def _finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"Nonfinite boundary optimizer {name}")


def _margins(logits, labels):
    _finite(logits, "logits")
    correct = logits.gather(1, labels[:, None]).squeeze(1)
    other = logits.masked_fill(nn.functional.one_hot(labels, logits.shape[1]).bool(), -torch.inf).max(1).values
    raw = correct - other
    scale = (logits - logits.mean(1, keepdim=True)).norm(dim=1)
    if bool((scale <= 1e-12).any()):
        raise FloatingPointError("Undefined normalized margin: centered logit scale is too small")
    normalized = raw / scale
    _finite(normalized, "normalized margin")
    return raw, normalized


def accuracy_drop_satisfies(full_correct, after_correct, count, bound, *, strict=False):
    """Compare integer correct-count changes with the declared decimal bound."""
    if (any(type(v) is not int for v in (full_correct, after_correct, count))
            or count < 1 or not 0 <= full_correct <= count or not 0 <= after_correct <= count):
        raise ValueError("Invalid accuracy counts")
    if isinstance(bound, bool) or not isinstance(bound, (int, float)) or not math.isfinite(bound):
        raise ValueError("Invalid accuracy bound")
    drop = Fraction(full_correct - after_correct, count)
    threshold = Fraction(str(bound))
    return drop > threshold if strict else drop <= threshold


def _validate_inputs(model, datasets, forget_ids, batch_size):
    if not isinstance(getattr(model, "classifier", None), nn.Linear) or not callable(getattr(model, "features", None)):
        raise ValueError("Boundary control requires features() and an unwrapped linear classifier")
    if model.classifier.weight.dtype not in {torch.float32, torch.float64}:
        raise ValueError("Boundary control supports float32/float64 classifiers")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    ids, forget = sorted(datasets), list(forget_ids)
    if (len(ids) < 2 or any(type(cid) is not int or cid < 0 for cid in ids + forget)
            or not forget or len(set(forget)) != len(forget) or not set(forget) < set(ids)):
        raise ValueError("Require distinct target clients and nonempty retained clients")
    seen_samples, seen_families = set(), set()
    for cid in ids:
        ds = datasets[cid]
        if getattr(ds, "role", None) != "train" or getattr(ds, "cid", None) != cid:
            raise ValueError("Only correctly bound training-role marker clients are allowed")
        sample_ids, families = list(ds.ids), list(ds.families)
        if (not len(ds) or len(sample_ids) != len(ds) or len(families) != len(ds)
                or len(set(sample_ids)) != len(sample_ids)
                or any(not isinstance(x, str) or not x.startswith(f"{cid}:train:") for x in sample_ids + families)):
            raise ValueError("Invalid training marker sample/family identities")
        if seen_samples.intersection(sample_ids) or seen_families.intersection(families):
            raise ValueError("Training marker clients have overlapping identities")
        seen_samples.update(sample_ids)
        seen_families.update(families)
    return ids, set(forget), [cid for cid in ids if cid not in forget]


@torch.no_grad()
def _cache(model, datasets, ids, device, batch_size):
    caches = {}
    for cid in ids:
        features, labels, outputs = [], [], []
        for x, y in DataLoader(datasets[cid], batch_size=batch_size, shuffle=False):
            x, y = x.to(device), y.to(device)
            _finite(x, "training inputs")
            z = model.features(x)
            if z.ndim != 2 or z.shape != (len(y), model.classifier.in_features):
                raise ValueError("Feature shape differs from the original classifier")
            if y.dtype != torch.long or bool((y < 0).any()) or bool((y >= model.classifier.out_features).any()):
                raise ValueError("Invalid training marker labels")
            _finite(z, "unnormalized Full features")
            features.append(z.detach())
            labels.append(y.detach())
            outputs.append(model.classifier(z).detach())
        z, y = torch.cat(features), torch.cat(labels)
        logits = torch.cat(outputs)
        raw, normalized = _margins(logits, y)
        # The public observation path computes margins from saved CPU logits.
        # Match that numerical path for every exact hard guard and eligibility
        # check, while retaining differentiable device tensors for the objective.
        _, public_normalized = _margins(logits.cpu(), y.cpu())
        correct = logits.argmax(1) == y
        caches[cid] = {"features": z, "labels": y, "full_logits": logits,
                       "full_margin": raw.detach(), "full_normalized_margin": normalized.detach(),
                       "full_correct": correct, "full_correct_count": int(correct.sum()),
                       "full_mean": float(public_normalized.double().mean()),
                       "n": len(y), "batch_size": batch_size}
    return caches


def _client_logits(head, cache):
    # Keep classifier GEMM batch shapes identical to actual marker inference,
    # including a short final batch. This matters near an accuracy boundary.
    return torch.cat([head(cache["features"][start:start + cache["batch_size"]])
                      for start in range(0, cache["n"], cache["batch_size"])])


def boundary_objective(head, caches, forget, retain, config):
    """All target samples, equal client weights, worst retained-client penalty."""
    target_losses, target_raw, target_normalized = [], [], []
    retained_losses = []
    for cid in sorted(forget):
        cache = caches[cid]
        raw, normalized = _margins(_client_logits(head, cache), cache["labels"])
        crossing = (raw + config.boundary_gap).relu().mean()
        drop = (normalized.mean() - cache["full_mean"] + config.target_normalized_drop).relu()
        target_raw.append(crossing)
        target_normalized.append(drop)
        target_losses.append(crossing + drop)
    for cid in retain:
        cache = caches[cid]
        raw, normalized = _margins(_client_logits(head, cache), cache["labels"])
        margin_penalty = (cache["full_mean"] - normalized.mean()).relu().square()
        mask = cache["full_correct"]
        # Weight each protected sample by its client's full sample count, so a
        # client with initially lower accuracy does not receive an inflated loss.
        floors = cache["full_margin"].clamp(max=config.retain_boundary_buffer)
        accuracy_penalty = ((floors[mask] - raw[mask]).relu().sum() / cache["n"]
                            if bool(mask.any()) else raw.sum() * 0.)
        retained_losses.append(margin_penalty + accuracy_penalty)
    target = torch.stack(target_losses).mean()
    retention = torch.stack(retained_losses).max()
    total = target + config.retain_weight * retention
    _finite(total, "objective")
    return total, {"target_boundary_loss": float(torch.stack(target_raw).mean().detach()),
                   "target_normalized_loss": float(torch.stack(target_normalized).mean().detach()),
                   "worst_client_retain_loss": float(retention.detach()),
                   "per_client_retain_loss": {str(cid): float(value.detach())
                                              for cid, value in zip(retain, retained_losses)}}


@torch.no_grad()
def _summary(head, caches, forget, config, logits_override=None):
    clients = []
    for cid, cache in caches.items():
        logits = _client_logits(head, cache) if logits_override is None else logits_override[cid]
        logits, labels = logits.detach().cpu(), cache["labels"].cpu()
        raw, normalized = _margins(logits, labels)
        correct = logits.argmax(1) == labels
        count = int(correct.sum())
        mean = float(normalized.double().mean())
        margin_drop = cache["full_mean"] - mean
        accuracy_drop = (cache["full_correct_count"] - count) / cache["n"]
        is_target = cid in forget
        accuracy_ok = accuracy_drop_satisfies(cache["full_correct_count"], count, cache["n"],
                    config.target_accuracy_drop if is_target else config.retain_accuracy_drop, strict=is_target)
        margin_ok = (margin_drop > config.target_normalized_drop if is_target
                     else margin_drop <= config.retain_margin_drop)
        full_accuracy = cache["full_correct_count"] / cache["n"]
        clients.append({"cid": cid, "client_group": "target" if is_target else "retain",
                        "num_examples": cache["n"], "full_correct_count": cache["full_correct_count"],
                        "after_correct_count": count, "correct_count_drop": cache["full_correct_count"] - count,
                        "full_accuracy": full_accuracy, "after_accuracy": count / cache["n"],
                        "accuracy_drop": accuracy_drop, "full_normalized_margin": cache["full_mean"],
                        "after_normalized_margin": mean, "normalized_margin_drop": margin_drop,
                        "full_raw_margin": float(cache["full_margin"].double().mean()), "after_raw_margin": float(raw.double().mean()),
                        "full_correct_to_wrong_count": int((cache["full_correct"].cpu() & ~correct).sum()),
                        "boundary_crossing_fraction": float((raw < 0).double().mean()),
                        "negative_gap_fraction": float((raw <= -config.boundary_gap).double().mean()),
                        "accuracy_condition_satisfied": accuracy_ok, "margin_condition_satisfied": margin_ok,
                        "conditions_satisfied": accuracy_ok and margin_ok,
                        "full_marker_precheck_passed": full_accuracy >= config.min_full_accuracy
                            and cache["full_mean"] >= config.min_full_margin})
    target_rows = [row for row in clients if row["client_group"] == "target"]
    retained_rows = [row for row in clients if row["client_group"] == "retain"]
    return {"clients": clients,
            "target_accuracy_drop_min": min(row["accuracy_drop"] for row in target_rows),
            "target_normalized_margin_drop_min": min(row["normalized_margin_drop"] for row in target_rows),
            "full_marker_precheck_passed": all(row["full_marker_precheck_passed"] for row in clients),
            "retained_constraints_satisfied": all(row["conditions_satisfied"] for row in retained_rows),
            "target_conditions_satisfied": all(row["conditions_satisfied"] for row in target_rows)}


def _distance(parameters, centers):
    return math.sqrt(sum(float((p.detach().double() - c.double()).square().sum())
                         for p, c in zip(parameters, centers)))


@torch.no_grad()
def _project(parameters, centers, radius):
    distance = _distance(parameters, centers)
    if not math.isfinite(distance):
        raise FloatingPointError("Nonfinite proposed head distance")
    if distance <= radius:
        return False
    # An inward factor avoids float32 rounding outside the exact fixed ball.
    factor = 0. if radius == 0 else radius / distance * (1. - 1e-6)
    for p, c in zip(parameters, centers):
        p.copy_(c + (p - c) * factor)
    if _distance(parameters, centers) > radius + 1e-12:
        raise FloatingPointError("Head projection did not remain in the fixed ball")
    return True


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def _verify_final(model, datasets, caches, ids, device, batch_size):
    logits, feature_error, logit_error = {}, 0., 0.
    for cid in ids:
        chunks, output_chunks = [], []
        for x, _ in DataLoader(datasets[cid], batch_size=batch_size, shuffle=False):
            x = x.to(device)
            z = model.features(x)
            chunks.append(z)
            output_chunks.append(model(x))
        fresh_z, direct_logits = torch.cat(chunks), torch.cat(output_chunks)
        old_z = caches[cid]["features"]
        torch.testing.assert_close(fresh_z, old_z, rtol=1e-6, atol=1e-7)
        cached_logits = _client_logits(model.classifier, caches[cid])
        torch.testing.assert_close(direct_logits, cached_logits, rtol=1e-6, atol=1e-7)
        if not torch.equal(direct_logits.argmax(1), cached_logits.argmax(1)):
            raise RuntimeError("Cached and final direct classifier predictions differ")
        feature_error = max(feature_error, float((fresh_z - old_z).abs().max()))
        logit_error = max(logit_error, float((direct_logits - cached_logits).abs().max()))
        logits[cid] = direct_logits
    return logits, {"features_max_abs_error": feature_error, "logits_max_abs_error": logit_error,
                    "predictions_identical": True, "feature_cache_normalized": False}


def train_boundary_head(model, training_markers, forget_ids, *, config=None,
                        device="cpu", batch_size=128):
    """Return a fixed endpoint and training-only evidence; never select on audit data."""
    config = config or BoundaryConfig()
    if not isinstance(config, BoundaryConfig):
        raise TypeError("config must be BoundaryConfig")
    ids, forget, retain = _validate_inputs(model, training_markers, forget_ids, batch_size)
    before = [value.copy() for value in get_model_parameters(model)]
    candidate = deepcopy(model).to(device).eval().requires_grad_(False)
    candidate.classifier.requires_grad_(True)
    names = [name for name, p in candidate.named_parameters() if p.requires_grad]
    expected_names = ["classifier." + name for name, _ in candidate.classifier.named_parameters()]
    if set(names) != set(expected_names):
        raise ValueError("Classifier parameters are shared with another model component")
    parameters = list(candidate.classifier.parameters())
    centers = [p.detach().clone() for p in parameters]
    for name, value in candidate.state_dict().items():
        _finite(value, "Full state " + name)
    head_norm = math.sqrt(sum(float(p.detach().double().square().sum()) for p in parameters))
    radius = config.project_ratio * max(head_norm, 1e-12)
    if not math.isfinite(radius):
        raise ValueError("Nonfinite projection radius")
    _sync(device)
    started = time.perf_counter()
    caches = _cache(candidate, training_markers, ids, device, batch_size)
    if state_change_report(candidate, before)["changed_buffers"]:
        raise RuntimeError("Feature extraction modified a frozen buffer")
    _sync(device)
    cache_seconds = time.perf_counter() - started
    initial = _summary(candidate.classifier, caches, forget, config)
    trace, accepted_steps = [], 0
    stop_reason = "budget_exhausted"
    optimization_started = time.perf_counter()
    for step in range(1, config.steps + 1):
        loss, terms = boundary_objective(candidate.classifier, caches, forget, retain, config)
        objective_before = float(loss.detach())
        gradients = torch.autograd.grad(loss, parameters)
        for g in gradients:
            _finite(g, "gradient")
        gradient_norm = math.sqrt(sum(float(g.detach().double().square().sum()) for g in gradients))
        if not math.isfinite(gradient_norm):
            raise FloatingPointError("Nonfinite gradient norm")
        factor = min(1., config.gradient_clip / max(gradient_norm, 1e-30))
        origin = [p.detach().clone() for p in parameters]
        attempts, accepted = [], False
        if gradient_norm == 0.:
            stop_reason = "stationary"
        else:
            for backtrack in range(config.max_backtracks + 1):
                lr = config.lr * config.backtrack_factor**backtrack
                record = {"backtrack": backtrack, "lr": lr, "accepted": False}
                with torch.no_grad():
                    for p, old, grad in zip(parameters, origin, gradients):
                        p.copy_(old - lr * factor * grad)
                try:
                    record["projected"] = _project(parameters, centers, radius)
                    with torch.no_grad():
                        new_loss, new_terms = boundary_objective(candidate.classifier, caches, forget, retain, config)
                        new_summary = _summary(candidate.classifier, caches, forget, config)
                    value = float(new_loss)
                    retained_ok = new_summary["retained_constraints_satisfied"]
                    descending = value < objective_before
                    accepted = retained_ok and descending
                    record.update(objective=value, **new_terms, retained_constraints_satisfied=retained_ok,
                                  objective_strictly_decreased=descending, clients=new_summary["clients"],
                                  parameter_delta_l2=_distance(parameters, centers), accepted=accepted,
                                  reason="accepted" if accepted else "retained_constraint" if not retained_ok else "no_strict_descent")
                except FloatingPointError as exc:
                    record["reason"] = "invalid_numeric_proposal"
                    record["error"] = str(exc)
                attempts.append(record)
                if accepted:
                    accepted_steps += 1
                    break
            if not accepted:
                stop_reason = "stalled_no_feasible_descent"
        if not accepted:
            with torch.no_grad():
                for p, old in zip(parameters, origin):
                    p.copy_(old)
        trace.append({"step": step, "accepted": accepted, "objective_before": objective_before,
                      "gradient_l2_norm": gradient_norm, "gradient_scale": factor,
                      "parameter_delta_l2": _distance(parameters, centers),
                      "objective_components_before": terms, "attempts": attempts})
        if not accepted:
            break
    _sync(device)
    optimization_seconds = time.perf_counter() - optimization_started
    validation_started = time.perf_counter()
    direct_logits, cache_check = _verify_final(candidate, training_markers, caches, ids, device, batch_size)
    final = _summary(candidate.classifier, caches, forget, config, logits_override=direct_logits)
    if not final["retained_constraints_satisfied"]:
        raise RuntimeError("Final direct model violates a retained-client training constraint")
    change = state_change_report(candidate, before)
    if change["changed_buffers"] or set(change["changed_parameters"]) - set(names):
        raise RuntimeError("Boundary optimization changed backbone parameters or buffers")
    if any(not np.array_equal(a, b) for a, b in zip(before, get_model_parameters(model))):
        raise RuntimeError("Boundary optimization mutated its source Full")
    distance = _distance(parameters, centers)
    final["changed_head"] = bool(change["changed_parameters"])
    final["eligible"] = bool(final["changed_head"] and final["full_marker_precheck_passed"]
                             and final["target_conditions_satisfied"] and final["retained_constraints_satisfied"])
    _sync(device)
    validation_seconds = time.perf_counter() - validation_started
    logs = {"objective": "marker_boundary_head", "config": asdict(config),
            "status": "training_eligible" if final["eligible"] else "training_ineligible",
            "training_eligible": final["eligible"], "stop_reason": stop_reason,
            "steps": config.steps, "accepted_steps": accepted_steps, "iterations": len(trace),
            "lr": config.lr, "project_ratio": config.project_ratio, "layers": "head",
            "data_scope": "markers_only", "optimization_access": "training_markers_only",
            "training_client_ids": ids, "target_client_ids": sorted(forget), "retained_client_ids": retain,
            "feature_cache_samples": sum(c["n"] for c in caches.values()),
            "feature_dimension": candidate.classifier.in_features,
            "trainable_parameter_names": names, "model_state_change_from_full": change,
            "source_full_unchanged": True, "feature_cache_check": cache_check,
            "head_norm_full": head_norm, "projection_radius": radius, "parameter_delta_l2": distance,
            "feature_cache_seconds": cache_seconds, "optimization_seconds": optimization_seconds,
            "validation_seconds": validation_seconds, "initial_training_summary": initial,
            "training_summary": final, "step_trace": trace, "formal_audit": False,
            "limitations": ["Training feasibility is empirical and is not an audit pass",
                            "Infeasible or stalled endpoints are retained without a success claim",
                            "Head-only marker manipulation does not establish removal of real-data influence"]}
    return UnlearningResult("marker_boundary_head", [p.copy() for p in get_model_parameters(candidate)], logs)
