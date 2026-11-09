"""D2 training-only head control with bounded half-space direction correction.

The original D objective and exact retained-client guards remain authoritative.
The linearized constraints only guide search; they are not an audit or a proof
that the nonconvex feasible set is empty when this finite solver stops.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
import json
import math
import time

import numpy as np
import torch

from fulbase.unlearning.base import UnlearningResult
from fulbase.unlearning.marker_boundary import (
    BoundaryConfig, _cache, _client_logits, _distance, _finite, _margins,
    _project, _summary, _sync, _validate_inputs, _verify_final, boundary_objective,
)
from fulbase.utils import get_model_parameters, state_change_report


@dataclass(frozen=True)
class ConstrainedBoundaryConfig(BoundaryConfig):
    direction_max_sweeps: int = 200
    direction_tolerance: float = 1e-9
    direction_activation_rounds: int = 8
    direction_margin_safety: float = .9
    direction_accuracy_buffer: float = 1e-4
    direction_min_norm: float = 1e-12

    def __post_init__(self):
        super().__post_init__()
        for key in ('direction_max_sweeps', 'direction_activation_rounds'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f'{key} must be a positive integer')
        if not 0 < self.direction_margin_safety < 1:
            raise ValueError('direction_margin_safety must be strictly between zero and one')
        if min(self.direction_tolerance, self.direction_accuracy_buffer, self.direction_min_norm) <= 0:
            raise ValueError('Direction tolerances and accuracy buffer must be positive')


def _flat(values):
    return torch.cat([v.detach().reshape(-1).double().cpu() for v in values]).numpy().copy()


def _stable_norm(value):
    value = np.asarray(value, dtype=np.float64)
    if not np.isfinite(value).all():
        raise FloatingPointError('Nonfinite direction vector')
    scale = float(np.max(np.abs(value))) if value.size else 0.
    if scale == 0.:
        return 0.
    norm = float(np.linalg.norm(value/scale)*scale)
    if not math.isfinite(norm):
        raise FloatingPointError('Nonfinite direction vector norm')
    return norm


def _cosine(a, b):
    norm_a, norm_b = _stable_norm(a), _stable_norm(b)
    return float(np.clip(np.dot(a/norm_a, b/norm_b), -1., 1.)) if norm_a > 0. and norm_b > 0. else None


def project_halfspaces(proposal, rows, bounds, *, max_sweeps=200, tolerance=1e-9):
    """Euclidean projection onto A*d<=b by Hildreth dual coordinate descent.

Rows are normalized in float64. Convergence requires both primal feasibility
and a dual projected-gradient KKT residual, rather than feasibility alone.
An unconverged result is diagnostic and must never be used as an accepted step.
"""
    proposal = np.asarray(proposal, dtype=np.float64)
    matrix = np.asarray(rows, dtype=np.float64).reshape((-1, proposal.size))
    bounds = np.asarray(bounds, dtype=np.float64)
    if proposal.ndim != 1 or bounds.shape != (len(matrix),):
        raise ValueError('Half-space dimensions differ')
    if not all(np.isfinite(v).all() for v in (proposal, matrix, bounds)):
        raise ValueError('Half-space inputs must be finite')
    if type(max_sweeps) is not int or max_sweeps < 1 or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Invalid half-space solver budget')
    if bool((bounds < 0).any()):
        raise ValueError('D2 half-spaces must contain the zero displacement')
    # Normalize in two stages so a nonzero row is not discarded merely because
    # its units are tiny, nor overflows when its coefficients are very large.
    scales = np.max(np.abs(matrix), axis=1) if len(matrix) else np.empty(0)
    nonzero = scales > 0.
    scaled = matrix[nonzero] / scales[nonzero, None]
    scaled_norms = np.linalg.norm(scaled, axis=1)
    a = scaled / scaled_norms[:, None]
    with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
        b = (bounds[nonzero] / scaled_norms) / scales[nonzero]
    if not np.isfinite(b).all():
        raise FloatingPointError('Nonfinite normalized half-space bound')
    if not len(a):
        return proposal.copy(), {'converged': True, 'status': 'unconstrained', 'sweeps': 0,
                                 'primal_residual': 0., 'dual_kkt_residual': 0.,
                                 'max_coordinate_change': 0., 'zero_rows': int((~nonzero).sum())}
    gram = a @ a.T
    q = a @ proposal - b
    multipliers = np.zeros(len(a), dtype=np.float64)
    violations = q.copy()
    diagonal = np.diag(gram)
    max_change = 0.
    converged = False
    for sweep in range(1, max_sweeps + 1):
        max_change = 0.
        for i in range(len(a)):
            new_value = max(0., multipliers[i] + violations[i] / diagonal[i])
            change = new_value - multipliers[i]
            if change:
                multipliers[i] = new_value
                violations -= gram[:, i] * change
                max_change = max(max_change, abs(change))
        # Recompute residuals rather than trusting accumulated update roundoff.
        displacement = proposal - a.T @ multipliers
        violations = a @ displacement - b
        primal = float(max(0., violations.max()))
        kkt = float(np.max(np.abs(np.maximum(0., multipliers + violations / diagonal) - multipliers)))
        if primal <= tolerance and kkt <= tolerance and max_change <= tolerance:
            converged = True
            break
    if not np.isfinite(displacement).all():
        raise FloatingPointError('Nonfinite half-space projection')
    return displacement, {'converged': converged,
                         'status': 'converged' if converged else 'max_sweeps', 'sweeps': sweep,
                         'primal_residual': primal, 'dual_kkt_residual': kkt,
                         'max_coordinate_change': float(max_change), 'zero_rows': int((~nonzero).sum())}


def required_correct_count(full_correct, count, bound):
    if type(full_correct) is not int or type(count) is not int or not 0 <= full_correct <= count or count < 1:
        raise ValueError('Invalid retained accuracy counts')
    if isinstance(bound, bool) or not isinstance(bound, (int, float)) or not math.isfinite(bound) or not 0 <= bound < 1:
        raise ValueError('Invalid retained accuracy bound')
    allowance = Fraction(str(bound)) * count
    return max(0, full_correct - allowance.numerator // allowance.denominator)


def _accuracy_protection(logits, labels, full_correct, bound):
    """Protect exactly K current correct samples, strongest margin then index."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    correct = logits.argmax(1) == labels
    required = required_correct_count(full_correct, len(labels), bound)
    if int(correct.sum()) < required:
        raise ValueError('Current iterate already violates the retained accuracy guard')
    gaps = logits[np.arange(len(labels)), labels, None] - logits
    gaps[np.arange(len(labels)), labels] = np.inf
    margins = gaps.min(1)
    order = sorted(np.flatnonzero(correct).tolist(), key=lambda i: (-float(margins[i]), i))
    return np.asarray(order[:required], dtype=np.int64), required, int(correct.sum()), gaps, margins


def _direction_problem(head, caches, retain, config, total_gradient, target_gradient, proposal, step):
    parameters = list(head.parameters())
    rows, bounds, metadata, clients, protection = [], [], [], [], {}
    for cid in retain:
        cache = caches[cid]
        logits = _client_logits(head, cache)
        _, normalized = _margins(logits, cache['labels'])
        derivative = _flat(torch.autograd.grad(-normalized.mean(), parameters))
        public_logits = logits.detach().cpu()
        _, public_normalized = _margins(public_logits, cache['labels'].cpu())
        current_drop = cache['full_mean'] - float(public_normalized.double().mean())
        slack = config.retain_margin_drop - current_drop
        if slack < 0:
            raise ValueError('Current iterate already violates the retained margin guard')
        rows.append(derivative)
        bounds.append(config.direction_margin_safety * slack)
        metadata.append({'step': step, 'cid': cid, 'kind': 'mean_margin', 'sample_index': None,
                         'other_class': None, 'current_gap': None, 'gap_floor': None})
        protected, required, correct, gaps, margins = _accuracy_protection(
            public_logits.numpy(), cache['labels'].cpu().numpy(), cache['full_correct_count'], config.retain_accuracy_drop)
        protection[cid] = {'indices': protected, 'required': required, 'gaps': gaps,
                           'features': cache['features'].detach().double().cpu().numpy(),
                           'labels': cache['labels'].cpu().numpy()}
        clients.append({'step': step, 'cid': cid, 'full_correct_count': cache['full_correct_count'],
                        'current_correct_count': correct, 'required_correct_count': required,
                        'remaining_correct_count_budget': correct-required,
                        'protected_count': len(protected), 'protected_sample_indices': json.dumps(protected.tolist()),
                        'protected_min_raw_margin': float(margins[protected].min()) if len(protected) else None,
                        'current_normalized_margin_drop': current_drop, 'margin_slack': slack,
                        'linear_margin_bound': bounds[-1], 'margin_gradient_l2_norm': _stable_norm(derivative),
                        'target_gradient_margin_cosine': _cosine(target_gradient, derivative),
                        'total_gradient_margin_cosine': _cosine(total_gradient, derivative),
                        'raw_predicted_margin_drop_change': float(derivative @ proposal)})
    return rows, bounds, metadata, clients, protection


def _pair_row(head, features, label, other):
    weights = np.zeros(tuple(head.weight.shape), dtype=np.float64)
    weights[other] = features
    weights[label] = -features
    parts = [weights.reshape(-1)]
    if head.bias is not None:
        bias = np.zeros(head.out_features, dtype=np.float64)
        bias[other], bias[label] = 1., -1.
        parts.append(bias)
    return np.concatenate(parts)


def _violating_protected_pairs(head, protection, displacement, config):
    weight_size = head.weight.numel()
    delta_weight = displacement[:weight_size].reshape(tuple(head.weight.shape))
    delta_bias = displacement[weight_size:] if head.bias is not None else np.zeros(head.out_features)
    violated, max_violation = [], 0.
    for cid in sorted(protection):
        data = protection[cid]
        for index in data['indices']:
            z, label = data['features'][index], int(data['labels'][index])
            output_delta = delta_weight @ z + delta_bias
            norm = math.sqrt(2.) * _stable_norm(np.append(z, 1.) if head.bias is not None else z)
            if not math.isfinite(norm):
                raise FloatingPointError('Nonfinite protected pair norm')
            for other in range(head.out_features):
                if other == label:
                    continue
                gap = float(data['gaps'][index, other])
                floor = min(gap, config.direction_accuracy_buffer)
                bound = gap-floor
                lhs = float(output_delta[other]-output_delta[label])
                normalized_violation = (lhs-bound)/norm if norm > 0. else 0.
                if not math.isfinite(normalized_violation):
                    raise FloatingPointError('Nonfinite protected pair residual')
                max_violation = max(max_violation, normalized_violation)
                if normalized_violation > config.direction_tolerance:
                    violated.append((cid, int(index), other, label, gap, floor, bound))
    return violated, max_violation


def correct_direction(head, caches, retain, config, total_gradient, target_gradient, proposal, *, step=1):
    """Return bounded direction diagnostics, including all competitor checks."""
    rows, bounds, metadata, clients, protection = _direction_problem(
        head, caches, retain, config, total_gradient, target_gradient, proposal, step)
    displacement = proposal.copy()
    active_pairs = set()
    total_sweeps, succeeded = 0, False
    solver = {'status': 'not_started', 'converged': False, 'sweeps': 0,
              'primal_residual': 0., 'dual_kkt_residual': 0., 'max_coordinate_change': 0., 'zero_rows': 0}
    pair_violation, violating = 0., []
    for activation_round in range(1, config.direction_activation_rounds + 1):
        violating, _ = _violating_protected_pairs(head, protection, displacement, config)
        for cid, index, other, label, gap, floor, bound in violating:
            key = (cid, index, other)
            if key in active_pairs:
                continue
            active_pairs.add(key)
            rows.append(_pair_row(head, protection[cid]['features'][index], label, other))
            bounds.append(bound)
            metadata.append({'step': step, 'cid': cid, 'kind': 'protected_accuracy',
                             'sample_index': index, 'other_class': other,
                             'current_gap': gap, 'gap_floor': floor})
        displacement, solver = project_halfspaces(
            proposal, rows, bounds, max_sweeps=config.direction_max_sweeps, tolerance=config.direction_tolerance)
        total_sweeps += solver['sweeps']
        # Even a failed solve must report the measured global protection residual.
        violating, pair_violation = _violating_protected_pairs(head, protection, displacement, config)
        if not solver['converged']:
            break
        if not violating:
            succeeded = True
            break
    constraints = []
    for row, bound, meta in zip(rows, bounds, metadata):
        norm = _stable_norm(row)
        residual = float((row @ displacement-bound)/norm) if norm > 0. else 0.
        if not math.isfinite(residual):
            raise FloatingPointError('Nonfinite direction constraint residual')
        constraints.append({**meta, 'bound': float(bound), 'gradient_l2_norm': norm,
                            'target_gradient_cosine': _cosine(target_gradient, row),
                            'total_gradient_cosine': _cosine(total_gradient, row),
                            'raw_predicted_change': float(row @ proposal),
                            'corrected_predicted_change': float(row @ displacement),
                            'normalized_residual': residual})
    for client, constraint in zip(clients, constraints[:len(retain)]):
        client['corrected_predicted_margin_drop_change'] = constraint['corrected_predicted_change']
        client['active_accuracy_pair_count'] = sum(cid == client['cid'] for cid, _, _ in active_pairs)
    cosine = _cosine(proposal, displacement)
    diagnostic = {'step': step, 'direction_converged': succeeded,
                  'direction_status': 'converged' if succeeded else solver['status'] if not solver['converged'] else 'activation_limit',
                  'activation_rounds': activation_round, 'solver_sweeps': total_sweeps,
                  'constraint_count': len(rows), 'active_accuracy_pair_count': len(active_pairs),
                  'solver_primal_residual': solver['primal_residual'],
                  'solver_dual_kkt_residual': solver['dual_kkt_residual'],
                  'solver_max_coordinate_change': solver['max_coordinate_change'],
                  'protected_pair_residual': pair_violation,
                  'protected_pairs_checked': sum(len(data['indices'])*(head.out_features-1)
                                                 for data in protection.values()),
                  'remaining_protected_pair_violations': len(violating),
                  'raw_displacement_l2': _stable_norm(proposal),
                  'corrected_displacement_l2': _stable_norm(displacement),
                  'correction_l2': _stable_norm(displacement-proposal),
                  'raw_corrected_cosine': cosine,
                  'raw_corrected_angle_degrees': math.degrees(math.acos(cosine)) if cosine is not None else None,
                  'raw_objective_directional_derivative': float(total_gradient @ proposal),
                  'objective_directional_derivative': float(total_gradient @ displacement),
                  'raw_target_directional_derivative': float(target_gradient @ proposal),
                  'target_directional_derivative': float(target_gradient @ displacement)}
    return displacement, diagnostic, clients, constraints


def train_constrained_boundary_head(model, training_markers, forget_ids, *, config=None,
                                    device='cpu', batch_size=128):
    config = config or ConstrainedBoundaryConfig()
    if not isinstance(config, ConstrainedBoundaryConfig):
        raise TypeError('config must be ConstrainedBoundaryConfig')
    ids, forget, retain = _validate_inputs(model, training_markers, forget_ids, batch_size)
    before = [value.copy() for value in get_model_parameters(model)]
    candidate = deepcopy(model).to(device).eval().requires_grad_(False)
    candidate.classifier.requires_grad_(True)
    names = [name for name, p in candidate.named_parameters() if p.requires_grad]
    expected_names = ['classifier.' + name for name, _ in candidate.classifier.named_parameters()]
    if set(names) != set(expected_names):
        raise ValueError('Classifier parameters are shared with another model component')
    parameters = list(candidate.classifier.parameters())
    centers = [p.detach().clone() for p in parameters]
    for name, value in candidate.state_dict().items():
        _finite(value, 'Full state ' + name)
    head_norm = math.sqrt(sum(float(p.detach().double().square().sum()) for p in parameters))
    radius = config.project_ratio * max(head_norm, 1e-12)
    if not math.isfinite(radius):
        raise ValueError('Nonfinite projection radius')
    _sync(device)
    started = time.perf_counter()
    caches = _cache(candidate, training_markers, ids, device, batch_size)
    if state_change_report(candidate, before)['changed_buffers']:
        raise RuntimeError('Feature extraction modified a frozen buffer')
    _sync(device)
    cache_seconds = time.perf_counter() - started
    initial = _summary(candidate.classifier, caches, forget, config)
    trace, direction_trace, client_diagnostics, constraint_diagnostics = [], [], [], []
    accepted_steps, stop_reason = 0, 'budget_exhausted'
    optimization_started = time.perf_counter()
    for step in range(1, config.steps + 1):
        loss, terms = boundary_objective(candidate.classifier, caches, forget, retain, config)
        objective_before = float(loss.detach())
        gradient = _flat(torch.autograd.grad(loss, parameters))
        target_loss, _ = boundary_objective(candidate.classifier, caches, forget, retain, replace(config, retain_weight=0.))
        target_gradient = _flat(torch.autograd.grad(target_loss, parameters))
        if not np.isfinite(gradient).all() or not np.isfinite(target_gradient).all():
            raise FloatingPointError('Nonfinite D2 gradient')
        gradient_norm = _stable_norm(gradient)
        if not math.isfinite(gradient_norm):
            raise FloatingPointError('Nonfinite D2 gradient norm')
        factor = min(1., config.gradient_clip / max(gradient_norm, 1e-30))
        proposal = -config.lr * factor * gradient
        displacement, direction, clients, constraints = correct_direction(
            candidate.classifier, caches, retain, config, gradient, target_gradient, proposal, step=step)
        origin = [p.detach().clone() for p in parameters]
        attempts, accepted = [], False
        if not direction['direction_converged']:
            stop_reason = 'direction_solver_' + direction['direction_status']
        elif gradient_norm == 0.:
            stop_reason = 'stationary'
        elif direction['corrected_displacement_l2'] <= config.direction_min_norm:
            stop_reason = 'direction_stationary'
        elif direction['objective_directional_derivative'] >= 0.:
            stop_reason = 'direction_not_descending'
        else:
            offset, tensors = 0, []
            for p in parameters:
                tensors.append(torch.as_tensor(displacement[offset:offset+p.numel()].copy(), device=p.device,
                                               dtype=p.dtype).reshape_as(p))
                offset += p.numel()
            for backtrack in range(config.max_backtracks + 1):
                multiplier = config.backtrack_factor**backtrack
                record = {'backtrack': backtrack, 'lr': config.lr*multiplier,
                          'direction_multiplier': multiplier, 'accepted': False}
                with torch.no_grad():
                    for p, old, delta in zip(parameters, origin, tensors):
                        p.copy_(old + multiplier*delta)
                try:
                    record['projected'] = _project(parameters, centers, radius)
                    with torch.no_grad():
                        new_loss, new_terms = boundary_objective(candidate.classifier, caches, forget, retain, config)
                        new_summary = _summary(candidate.classifier, caches, forget, config)
                    value = float(new_loss)
                    retained_ok = new_summary['retained_constraints_satisfied']
                    descending = value < objective_before
                    accepted = retained_ok and descending
                    rejected = [row['cid'] for row in new_summary['clients']
                                if row['client_group'] == 'retain' and not row['conditions_satisfied']]
                    record.update(objective=value, **new_terms, retained_constraints_satisfied=retained_ok,
                                  objective_strictly_decreased=descending, clients=new_summary['clients'],
                                  rejected_client_ids=rejected, parameter_delta_l2=_distance(parameters, centers),
                                  accepted=accepted, reason='accepted' if accepted else
                                  'retained_constraint' if not retained_ok else 'no_strict_descent')
                except FloatingPointError as exc:
                    record.update(reason='invalid_numeric_proposal', error=str(exc))
                attempts.append(record)
                if accepted:
                    accepted_steps += 1
                    break
            if not accepted:
                stop_reason = 'stalled_no_feasible_descent'
        if not accepted:
            with torch.no_grad():
                for p, old in zip(parameters, origin):
                    p.copy_(old)
        direction.update(accepted=accepted, backtracking_attempts=len(attempts),
                         last_attempt_reason=attempts[-1]['reason'] if attempts else stop_reason)
        direction_trace.append(direction)
        client_diagnostics.extend(clients)
        constraint_diagnostics.extend(constraints)
        trace.append({'step': step, 'accepted': accepted, 'objective_before': objective_before,
                      'gradient_l2_norm': gradient_norm, 'gradient_scale': factor,
                      'parameter_delta_l2': _distance(parameters, centers),
                      'objective_components_before': terms, 'direction': direction, 'attempts': attempts})
        if not accepted:
            break
    _sync(device)
    optimization_seconds = time.perf_counter() - optimization_started
    validation_started = time.perf_counter()
    direct_logits, cache_check = _verify_final(candidate, training_markers, caches, ids, device, batch_size)
    final = _summary(candidate.classifier, caches, forget, config, logits_override=direct_logits)
    if not final['retained_constraints_satisfied']:
        raise RuntimeError('Final direct model violates a retained-client training constraint')
    change = state_change_report(candidate, before)
    if change['changed_buffers'] or set(change['changed_parameters']) - set(names):
        raise RuntimeError('D2 optimization changed backbone parameters or buffers')
    if any(not np.array_equal(a, b) for a, b in zip(before, get_model_parameters(model))):
        raise RuntimeError('D2 optimization mutated its source Full')
    distance = _distance(parameters, centers)
    final['changed_head'] = bool(change['changed_parameters'])
    final['eligible'] = bool(final['changed_head'] and final['full_marker_precheck_passed']
                             and final['target_conditions_satisfied'] and final['retained_constraints_satisfied'])
    _sync(device)
    validation_seconds = time.perf_counter()-validation_started
    logs = {'objective': 'marker_boundary_d2_head', 'config': asdict(config),
            'status': 'training_eligible' if final['eligible'] else 'training_ineligible',
            'training_eligible': final['eligible'], 'stop_reason': stop_reason,
            'steps': config.steps, 'accepted_steps': accepted_steps, 'iterations': len(trace),
            'lr': config.lr, 'project_ratio': config.project_ratio, 'layers': 'head',
            'data_scope': 'markers_only', 'optimization_access': 'training_markers_only',
            'training_client_ids': ids, 'target_client_ids': sorted(forget), 'retained_client_ids': retain,
            'feature_cache_samples': sum(c['n'] for c in caches.values()),
            'feature_dimension': candidate.classifier.in_features, 'trainable_parameter_names': names,
            'model_state_change_from_full': change, 'source_full_unchanged': True,
            'feature_cache_check': cache_check, 'head_norm_full': head_norm,
            'projection_radius': radius, 'parameter_delta_l2': distance,
            'feature_cache_seconds': cache_seconds, 'optimization_seconds': optimization_seconds,
            'validation_seconds': validation_seconds, 'initial_training_summary': initial,
            'training_summary': final, 'step_trace': trace, 'direction_trace': direction_trace,
            'direction_client_diagnostics': client_diagnostics, 'direction_constraints': constraint_diagnostics,
            'formal_audit': False,
            'limitations': ['Training feasibility is empirical and is not an audit pass',
                            'Finite local search failure is not a proof of global infeasibility',
                            'Protected sample subsets may be conservative relative to the exact count guard',
                            'Head-only marker manipulation does not establish removal of real-data influence']}
    return UnlearningResult('marker_boundary_d2_head', [p.copy() for p in get_model_parameters(candidate)], logs)
