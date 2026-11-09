"""Paired D2 reference and signed-interior training-only head diagnostics.

The reference preserves original D2 acceptance/stopping and adds read-only
precision evidence. Interior reserve is a search preference, never a hard guard.
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


from fulbase.unlearning.marker_boundary_d2 import (
    ConstrainedBoundaryConfig, correct_direction, _flat, _stable_norm, _cosine,
    _accuracy_protection, _pair_row, _violating_protected_pairs,
)
from fulbase.unlearning.boundary_precision import precision_attempt_diagnostics, unavailable_precision_rows


@dataclass(frozen=True)
class InteriorBoundaryConfig(ConstrainedBoundaryConfig):
    steps: int = 100
    interior_margin_reserve: float = 1e-4
    max_restoration_steps: int = 3
    restoration_deficit_factor: float = .5

    def __post_init__(self):
        super().__post_init__()
        if self.interior_margin_reserve <= 0:
            raise ValueError('interior_margin_reserve must be positive')
        if type(self.max_restoration_steps) is not int or self.max_restoration_steps < 0:
            raise ValueError('max_restoration_steps must be a nonnegative integer')
        if not 0 < self.restoration_deficit_factor < 1:
            raise ValueError('restoration_deficit_factor must be between zero and one')


def project_signed_halfspaces(proposal, rows, bounds, *, max_sweeps=200, tolerance=1e-9):
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
    # Normalize in two stages so a nonzero row is not discarded merely because
    # its units are tiny, nor overflows when its coefficients are very large.
    scales = np.max(np.abs(matrix), axis=1) if len(matrix) else np.empty(0)
    nonzero = scales > 0.
    if bool(((~nonzero) & (bounds < 0.)).any()):
        return proposal.copy(), {'converged': False, 'status': 'local_linear_infeasible',
                                 'sweeps': 0, 'primal_residual': float(max(0., -bounds[~nonzero].min())),
                                 'dual_kkt_residual': None, 'max_coordinate_change': 0.,
                                 'zero_rows': int((~nonzero).sum())}
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


def _interior_direction_problem(head, caches, retain, config, total_gradient, target_gradient, proposal, step):
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
        bounds.append(config.direction_margin_safety * (slack-config.interior_margin_reserve))
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


def correct_interior_direction(head, caches, retain, config, total_gradient, target_gradient, proposal, *, step=1):
    """Return bounded direction diagnostics, including all competitor checks."""
    rows, bounds, metadata, clients, protection = _interior_direction_problem(
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
        displacement, solver = project_signed_halfspaces(
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
        residual = float((row @ displacement-bound)/norm) if norm > 0. else float(max(0., -bound))
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


def interior_deficit(clients, config, *, summary=False):
    """Maximum missing search reserve; this is not an eligibility threshold."""
    slacks = [(config.retain_margin_drop-row['normalized_margin_drop']) if summary else row['margin_slack']
              for row in clients if not summary or row['client_group'] == 'retain']
    return max([0.] + [config.interior_margin_reserve-slack for slack in slacks])


def proposal_acceptance(*, retained_ok, descending, mode, deficit_before, deficit_after,
                        restorations_used, config):
    """Normal descent takes precedence over the narrowly budgeted restoration."""
    if not retained_ok:
        return False, None, 'retained_constraint'
    if descending:
        return True, 'normal', 'accepted'
    if (mode == 'interior' and math.isfinite(deficit_before) and math.isfinite(deficit_after)
            and deficit_before > 0. and deficit_after >= 0.
            and restorations_used < config.max_restoration_steps
            and deficit_after <= config.restoration_deficit_factor*deficit_before):
        return True, 'restoration', 'accepted_restoration'
    return False, None, 'no_strict_descent'


def train_interior_boundary_head(model, training_markers, forget_ids, *, config=None,
                                    device='cpu', batch_size=128, mode='interior'):
    config = config or InteriorBoundaryConfig()
    if not isinstance(config, InteriorBoundaryConfig):
        raise TypeError('config must be InteriorBoundaryConfig')
    if mode not in {'interior', 'd2_reference'}:
        raise ValueError('mode must be interior or d2_reference')
    method = 'marker_boundary_interior_head' if mode == 'interior' else 'marker_boundary_d2_reference_head'
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
    normal_steps, restoration_steps, precision_seconds = 0, 0, 0.
    precision_diagnostics, restoration_trace = [], []
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
        displacement, direction, clients, constraints = (correct_interior_direction if mode == 'interior' else correct_direction)(
            candidate.classifier, caches, retain, config, gradient, target_gradient, proposal, step=step)
        origin = [p.detach().clone() for p in parameters]
        attempts, accepted = [], False
        deficit_before = interior_deficit(clients, config)
        restoration_needed = mode == 'interior' and deficit_before > 0.
        accepted_kind = None
        if not direction['direction_converged']:
            stop_reason = 'direction_solver_' + direction['direction_status']
        elif gradient_norm == 0. and not restoration_needed:
            stop_reason = 'stationary'
        elif direction['corrected_displacement_l2'] <= config.direction_min_norm:
            stop_reason = 'direction_stationary'
        elif direction['objective_directional_derivative'] >= 0. and not restoration_needed:
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
                precision_started = time.perf_counter()
                pre_ball = [p.detach().clone() for p in parameters]
                _sync(device)
                precision_seconds += time.perf_counter()-precision_started
                public_means = None
                try:
                    record['projected'] = _project(parameters, centers, radius)
                    with torch.no_grad():
                        new_loss, new_terms = boundary_objective(candidate.classifier, caches, forget, retain, config)
                        new_summary = _summary(candidate.classifier, caches, forget, config)
                    value = float(new_loss)
                    retained_ok = new_summary['retained_constraints_satisfied']
                    descending = value < objective_before
                    deficit_after = interior_deficit(new_summary['clients'], config, summary=True)
                    accepted, accepted_kind, reason = proposal_acceptance(
                        retained_ok=retained_ok, descending=descending, mode=mode,
                        deficit_before=deficit_before, deficit_after=deficit_after,
                        restorations_used=restoration_steps, config=config)
                    public_means = {row['cid']: row['after_normalized_margin'] for row in new_summary['clients']}
                    rejected = [row['cid'] for row in new_summary['clients']
                                if row['client_group'] == 'retain' and not row['conditions_satisfied']]
                    record.update(objective=value, **new_terms, retained_constraints_satisfied=retained_ok,
                                  objective_strictly_decreased=descending, clients=new_summary['clients'],
                                  rejected_client_ids=rejected, parameter_delta_l2=_distance(parameters, centers),
                                  accepted=accepted, reason=reason)
                    if mode == 'interior':
                        record.update(kind=accepted_kind, interior_deficit_before=deficit_before,
                                      interior_deficit_after=deficit_after,
                                      objective_change=value-objective_before)
                        restoration_trace.append(dict(step=step, backtrack=backtrack, mode=mode,
                            accepted=accepted, kind=accepted_kind, deficit_before=deficit_before,
                            deficit_after=deficit_after, restorations_used_before=restoration_steps,
                            max_restoration_steps=config.max_restoration_steps,
                            deficit_factor=config.restoration_deficit_factor,
                            retained_constraints_satisfied=retained_ok,
                            objective_before=objective_before, objective_after=value,
                            objective_change=value-objective_before, reason=reason))
                except FloatingPointError as exc:
                    record.update(reason='invalid_numeric_proposal', error=str(exc))
                # The reference is observational: neither errors nor values are
                # used by the true acceptance path. Original head/cache untouched.
                precision_started = time.perf_counter()
                try:
                    precision_rows = precision_attempt_diagnostics(candidate.classifier, caches, retain,
                        origin, displacement, multiplier, pre_ball, step=step, backtrack=backtrack,
                        mode=mode, public_means=public_means)
                except Exception as exc:
                    precision_rows = unavailable_precision_rows(retain, step=step, backtrack=backtrack,
                        mode=mode, reason=f'{type(exc).__name__}: {exc}')
                precision_diagnostics.extend(precision_rows)
                _sync(device)
                precision_seconds += time.perf_counter()-precision_started
                attempts.append(record)
                if accepted:
                    accepted_steps += 1
                    if accepted_kind == 'restoration':
                        restoration_steps += 1
                    else:
                        normal_steps += 1
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
    logs = {'objective': method, 'config': asdict(config), 'mode': mode,
            'status': 'training_eligible' if final['eligible'] else 'training_ineligible',
            'training_eligible': final['eligible'], 'stop_reason': stop_reason,
            'steps': config.steps, 'accepted_steps': accepted_steps, 'iterations': len(trace),
            'normal_accepted_steps': normal_steps, 'restoration_accepted_steps': restoration_steps,
            'lr': config.lr, 'project_ratio': config.project_ratio, 'layers': 'head',
            'data_scope': 'markers_only', 'optimization_access': 'training_markers_only',
            'training_client_ids': ids, 'target_client_ids': sorted(forget), 'retained_client_ids': retain,
            'feature_cache_samples': sum(c['n'] for c in caches.values()),
            'feature_dimension': candidate.classifier.in_features, 'trainable_parameter_names': names,
            'model_state_change_from_full': change, 'source_full_unchanged': True,
            'feature_cache_check': cache_check, 'head_norm_full': head_norm,
            'projection_radius': radius, 'parameter_delta_l2': distance,
            'feature_cache_seconds': cache_seconds, 'optimization_seconds': optimization_seconds,
            'optimization_total_wall_seconds': optimization_seconds,
            'precision_diagnostic_seconds': precision_seconds,
            'optimization_excluding_precision_seconds_derived': optimization_seconds-precision_seconds,
            'optimization_timing_note': 'Total wall includes readonly precision; exclusion is derived by subtraction',
            'validation_seconds': validation_seconds, 'initial_training_summary': initial,
            'training_summary': final, 'step_trace': trace, 'direction_trace': direction_trace,
            'direction_client_diagnostics': client_diagnostics, 'direction_constraints': constraint_diagnostics,
            'precision_diagnostics': precision_diagnostics, 'restoration_trace': restoration_trace,
            'formal_audit': False,
            'limitations': ['Training feasibility is empirical and is not an audit pass',
                            'Finite local search failure is not a proof of global infeasibility',
                            'Protected sample subsets may be conservative relative to the exact count guard',
                            'Head-only marker manipulation does not establish removal of real-data influence']}
    return UnlearningResult(method, [p.copy() for p in get_model_parameters(candidate)], logs)
