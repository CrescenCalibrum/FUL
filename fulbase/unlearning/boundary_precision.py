"""Read-only arithmetic diagnostics on a frozen, original-scale feature cache.

These references never decide whether an optimization proposal is accepted.
CPU64 changes the linear head arithmetic, not the backbone or cached features.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch.nn import functional as F

from fulbase.unlearning.marker_boundary import _margins


_REFERENCE_FIELDS = (
    'current_mean_cpu64', 'ideal_mean_cpu64', 'pre_ball_mean_cpu64',
    'post_ball_mean_cpu64', 'current_mean_cpu32', 'post_ball_mean_cpu32',
    'current_mean_public', 'post_ball_mean_public', 'drop_change_public',
    'drop_change_cpu64_ideal', 'drop_change_cpu64_pre_ball', 'drop_change_cpu64_post_ball',
    'drop_change_cpu32_post_ball', 'linear64', 'nonlinear_remainder',
    'parameter_rounding_effect', 'ball_projection_effect', 'cpu_dtype_arithmetic_effect',
    'device_implementation_effect', 'decomposition_sum', 'decomposition_residual',
    'gradient32_dot_ideal', 'gradient32_dot_applied', 'gradient64_dot_applied',
    'ideal_delta_l2', 'pre_ball_delta_l2', 'post_ball_delta_l2', 'applied_delta_l2',
    'parameter_rounding_delta_l2', 'ball_projection_delta_l2', 'applied_minus_ideal_l2',
    'cast_displacement_l2', 'direction_cast_error_l2', 'addition_rounding_delta_l2',
    'unchanged_parameter_elements', 'total_parameter_elements',
)


def unavailable_precision_rows(retain, *, step, backtrack, mode, reason):
    return [dict(step=step, backtrack=backtrack, cid=cid, mode=mode,
                 reference_status='unavailable', reference_reason=str(reason),
                 **dict.fromkeys(_REFERENCE_FIELDS)) for cid in retain]


def _flatten(parameters):
    return torch.cat([p.detach().cpu().double().reshape(-1) for p in parameters])


def _logits(parameters, features, batch_size):
    return torch.cat([F.linear(z, parameters[0], parameters[1] if len(parameters) == 2 else None)
                      for z in features.split(batch_size)])


def _mean(parameters, cache, *, dtype, device='cpu', gradient=False, public=False):
    values = [p.detach().to(device=device, dtype=dtype).clone().requires_grad_(gradient) for p in parameters]
    features = cache['features'].detach().to(device=device, dtype=dtype)
    labels = cache['labels'].detach().to(device)
    # enable_grad is local: callers may be in the true proposal's no_grad block.
    with torch.enable_grad() if gradient else torch.no_grad():
        logits = _logits(values, features, cache['batch_size'])
        if public:
            logits, labels = logits.cpu(), labels.cpu()
        _, normalized = _margins(logits, labels)
        mean = float(normalized.detach().double().mean())
        derivative = _flatten(torch.autograd.grad(-normalized.mean(), values)) if gradient else None
    return mean, derivative


def precision_attempt_diagnostics(head, caches, retain, origin, displacement, multiplier, pre_ball,
                                  *, step, backtrack, mode, public_means=None):
    """Return one flat decomposition per retained client; unavailable is nonfatal.

``origin`` and ``pre_ball`` are detached snapshots of the actual parameter
values; ``head`` contains the actual post-ball values.  The ideal reference is
origin.cast64 + multiplier * displacement64 (before any parameter casting).
``public_means`` optionally supplies the already measured true post-step means.
"""
    rows = []
    try:
        if any(p.dtype != torch.float32 for p in origin):
            raise ValueError('CPU32/64 decomposition requires original float32 head parameters')
        current = [p.detach().cpu().double() for p in origin]
        pre = [p.detach().cpu().double() for p in pre_ball]
        post = [p.detach().cpu().double() for p in head.parameters()]
        delta = torch.as_tensor(np.asarray(displacement, dtype=np.float64).copy()) * multiplier
        if delta.numel() != sum(p.numel() for p in current):
            raise ValueError('Precision displacement dimension differs from classifier')
        ideal, offset = [], 0
        for p in current:
            ideal.append(p + delta[offset:offset+p.numel()].reshape_as(p))
            offset += p.numel()
        start_flat, pre_flat, post_flat = map(_flatten, (current, pre, post))
        delta_pre, delta_post = pre_flat-start_flat, post_flat-start_flat
        # Actual algorithm first casts the direction, then scales/adds in the
        # parameter dtype. Keep those two sources visible as well as their sum.
        cast_delta = torch.as_tensor(np.asarray(displacement, dtype=np.float64).copy()).float().double()*multiplier
        common = dict(
            ideal_delta_l2=float(torch.linalg.vector_norm(delta)),
            pre_ball_delta_l2=float(torch.linalg.vector_norm(delta_pre)),
            post_ball_delta_l2=float(torch.linalg.vector_norm(delta_post)),
            applied_delta_l2=float(torch.linalg.vector_norm(delta_post)),
            parameter_rounding_delta_l2=float(torch.linalg.vector_norm(delta_pre-delta)),
            ball_projection_delta_l2=float(torch.linalg.vector_norm(post_flat-pre_flat)),
            applied_minus_ideal_l2=float(torch.linalg.vector_norm(delta_post-delta)),
            cast_displacement_l2=float(torch.linalg.vector_norm(cast_delta)),
            direction_cast_error_l2=float(torch.linalg.vector_norm(cast_delta-delta)),
            addition_rounding_delta_l2=float(torch.linalg.vector_norm(delta_pre-cast_delta)),
            unchanged_parameter_elements=int((delta_post == 0).sum()), total_parameter_elements=delta.numel())
        if not all(math.isfinite(v) for v in common.values()):
            raise FloatingPointError('Nonfinite precision displacement')
    except Exception as exc:
        return unavailable_precision_rows(retain, step=step, backtrack=backtrack, mode=mode,
                                          reason=f'{type(exc).__name__}: {exc}')
    for cid in retain:
        try:
            cache = caches[cid]
            now64, a64 = _mean(current, cache, dtype=torch.float64, gradient=True)
            ideal64, _ = _mean(ideal, cache, dtype=torch.float64)
            pre64, _ = _mean(pre, cache, dtype=torch.float64)
            post64, _ = _mean(post, cache, dtype=torch.float64)
            now32, _ = _mean(current, cache, dtype=torch.float32)
            post32, _ = _mean(post, cache, dtype=torch.float32)
            device = origin[0].device
            now_public, _ = _mean(origin, cache, dtype=torch.float32, device=device, public=True)
            _, a32 = _mean(origin, cache, dtype=torch.float32, device=device, gradient=True)
            if public_means is None:
                after_public, _ = _mean(list(head.parameters()), cache, dtype=torch.float32, device=device, public=True)
            else:
                after_public = public_means[cid]
            ideal_drop, pre_drop, post_drop = now64-ideal64, now64-pre64, now64-post64
            cpu32_drop, public_drop = now32-post32, now_public-after_public
            linear = float(a64 @ delta)
            pieces = [linear, ideal_drop-linear, pre_drop-ideal_drop, post_drop-pre_drop,
                      cpu32_drop-post_drop, public_drop-cpu32_drop]
            row = dict(step=step, backtrack=backtrack, cid=cid, mode=mode,
                       reference_status='available', reference_reason=None, **common,
                       current_mean_cpu64=now64, ideal_mean_cpu64=ideal64,
                       pre_ball_mean_cpu64=pre64, post_ball_mean_cpu64=post64,
                       current_mean_cpu32=now32, post_ball_mean_cpu32=post32,
                       current_mean_public=now_public, post_ball_mean_public=after_public,
                       drop_change_public=public_drop, drop_change_cpu64_ideal=ideal_drop,
                       drop_change_cpu64_pre_ball=pre_drop, drop_change_cpu64_post_ball=post_drop,
                       drop_change_cpu32_post_ball=cpu32_drop,
                       linear64=pieces[0], nonlinear_remainder=pieces[1],
                       parameter_rounding_effect=pieces[2], ball_projection_effect=pieces[3],
                       cpu_dtype_arithmetic_effect=pieces[4], device_implementation_effect=pieces[5],
                       decomposition_sum=math.fsum(pieces), decomposition_residual=public_drop-math.fsum(pieces),
                       gradient32_dot_ideal=float(a32 @ delta), gradient32_dot_applied=float(a32 @ delta_post),
                       gradient64_dot_applied=float(a64 @ delta_post))
            if any(isinstance(value, float) and not math.isfinite(value) for value in row.values()):
                raise FloatingPointError('Nonfinite precision reference')
            rows.append(row)
        except Exception as exc:
            rows.extend(unavailable_precision_rows([cid], step=step, backtrack=backtrack, mode=mode,
                                                    reason=f'{type(exc).__name__}: {exc}'))
    return rows
