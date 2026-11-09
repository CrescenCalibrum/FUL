"""Explicit BN endpoint variant of the existing FedEraser-style replay.

Learned parameters follow the legacy replay exactly. Only BatchNorm running
means/variances are replaced by a retained historical endpoint. This is a state
handling policy, not an exact retraining or original-paper FedEraser guarantee.
"""
from __future__ import annotations

import hashlib
from numbers import Integral

import numpy as np
import torch

from fulbase.models import build_model
from fulbase.unlearning.base import UnlearningResult
from fulbase.unlearning.federaser_lite import federaser_lite_unlearn, _is_compact_history
from fulbase.unlearning.history_validation import validate_history_request
from fulbase.utils import get_model_parameters, set_model_parameters, weighted_average_arrays


VARIANT = "federaser_lite_bn_endpoint"
POLICY = "retained_endpoint"


def _model(dataset, num_classes):
    # Constructing a reference model must not change another method's RNG stream.
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        return build_model(dataset, num_classes)


def _hash(parameters):
    digest = hashlib.sha256()
    for value in parameters:
        array = np.ascontiguousarray(value)
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _bn_indices(model):
    state_names = list(model.state_dict())
    mean_var, counters = [], []
    for prefix, module in model.named_modules():
        if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm) or not module.track_running_stats:
            continue
        prefix = prefix + "." if prefix else ""
        mean_var.extend(state_names.index(prefix + suffix) for suffix in ("running_mean", "running_var"))
        if prefix + "num_batches_tracked" in state_names:
            counters.append(state_names.index(prefix + "num_batches_tracked"))
    return state_names, mean_var, counters


def _validate_arrays(arrays, references, description, *, compact_update=False, allow_fp16=True):
    if not isinstance(arrays, list) or len(arrays) != len(references):
        raise ValueError(f"BN endpoint state length mismatch: {description}")
    for index, (array, reference) in enumerate(zip(arrays, references)):
        if not isinstance(array, (np.ndarray, np.generic)) or array.shape != reference.shape:
            raise ValueError(f"BN endpoint shape mismatch: {description}[{index}]")
        permitted = {reference.dtype}
        if np.issubdtype(reference.dtype, np.floating):
            if allow_fp16 and reference.dtype == np.float32:
                permitted.add(np.dtype(np.float16))
        elif compact_update:
            # compact_round intentionally writes zero floating deltas for counters.
            permitted |= {np.dtype(np.float32), np.dtype(np.float16)}
            if np.any(array != 0):
                raise ValueError(f"Nonfloating compact history update must be zero: {description}[{index}]")
        if array.dtype not in permitted:
            raise ValueError(f"BN endpoint dtype mismatch: {description}[{index}]")
        if not np.isfinite(array).all():
            raise ValueError(f"Nonfinite BN endpoint history/state: {description}[{index}]")


def _bn_diagnostic(parameters, names, indices):
    variances, means = [], []
    for index in indices:
        array = parameters[index]
        finite = array[np.isfinite(array)]
        entry = {"name": names[index], "min": float(finite.min()) if finite.size else None,
                 "max": float(finite.max()) if finite.size else None,
                 "nonfinite_count": int(np.count_nonzero(~np.isfinite(array)))}
        if names[index].endswith("running_var"):
            entry["negative_count"] = int(np.count_nonzero(array < 0))
            variances.append(entry)
        else:
            means.append(entry)
    minima = [row["min"] for row in variances if row["min"] is not None]
    return {"running_variances": variances, "running_means": means,
            "negative_running_variance_count": sum(row["negative_count"] for row in variances),
            "minimum_running_variance": min(minima) if minima else None,
            "nonfinite_count": sum(row["nonfinite_count"] for row in variances + means),
            "valid": not any(row["negative_count"] or row["nonfinite_count"] for row in variances)
                     and not any(row["nonfinite_count"] for row in means)}


def _endpoints(history, forget_client_ids, model):
    validate_history_request(history, forget_client_ids)
    compact = _is_compact_history(history)
    names, indices, counters = _bn_indices(model)
    references = get_model_parameters(model)
    forget = set(map(str, forget_client_ids))
    final_buffers, trace, retained_ids = {}, [], set()
    for position, item in enumerate(history, 1):
        _validate_arrays(item.get("global_before"), references, f"round {position} global_before")
        before = item["global_before"]
        if not _bn_diagnostic(before, names, indices)["valid"]:
            raise ValueError(f"Invalid historical global BN state at round {item.get('round', position)}")
        if compact:
            if "clients" in item:
                raise ValueError("Mixed compact/full history is not supported by BN endpoint policy")
            for key in ("retained_update", "all_update"):
                _validate_arrays(item.get(key), references, f"round {position} {key}", compact_update=True)
            count = item.get("retained_examples")
            if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
                raise ValueError("BN endpoint policy requires a valid retained_examples count")
            round_ids = sorted(set(map(str, item.get("participating_client_ids", []))) - forget)
            if count == 0:
                trace.append({"round": item.get("round", position), "status": "no_retained_participation"})
                continue
            if not round_ids:
                raise ValueError("Retained BN endpoint lacks retained participant identities")
            # Use RAW history deltas. Calibration and damping never apply here.
            endpoint = list(before)
            for index in indices:
                endpoint[index] = (before[index].astype(np.float32)
                                   + item["retained_update"][index].astype(np.float32))
        else:
            clients = item.get("clients")
            if not isinstance(clients, list):
                raise ValueError("Full BN endpoint history requires client models")
            retained = []
            seen = set()
            for client in clients:
                cid, count = str(client.get("cid")), client.get("num_examples")
                if cid in seen or isinstance(count, bool) or not isinstance(count, Integral) or count <= 0:
                    raise ValueError("Invalid full-history client identity/count")
                seen.add(cid)
                _validate_arrays(client.get("params_after"), references, f"round {position} client {cid}")
                if not _bn_diagnostic(client["params_after"], names, indices)["valid"]:
                    raise ValueError(f"Invalid historical client BN state at round {item.get('round', position)}, client {cid}")
                if cid not in forget:
                    retained.append((client["params_after"], count))
            round_ids = sorted(seen - forget)
            if not retained:
                trace.append({"round": item.get("round", position), "status": "no_retained_participation"})
                continue
            # Averaging absolute client buffers avoids subtract/add cancellation.
            endpoint = list(before)
            if indices:
                averaged = weighted_average_arrays([
                    ([arrays[index].astype(np.float32) for index in indices], count)
                    for arrays, count in retained])
                for index, array in zip(indices, averaged):
                    endpoint[index] = array
        diagnostic = _bn_diagnostic(endpoint, names, indices)
        if not diagnostic["valid"]:
            raise ValueError(f"Invalid retained BN endpoint at round {item.get('round', position)}: {diagnostic}")
        for index in indices:
            final_buffers[index] = endpoint[index].astype(references[index].dtype, copy=True)
        retained_ids.update(round_ids)
        trace.append({"round": item.get("round", position), "status": "valid_retained_endpoint",
                      "retained_client_ids": round_ids, "diagnostic": diagnostic})
    if indices and not final_buffers:
        raise ValueError("No retained BN endpoint is available in history")
    return final_buffers, {"bn_policy": POLICY, "method_variant": VARIANT,
        "history_format": "compact" if compact else "full", "retained_client_ids": sorted(retained_ids),
        "bn_mean_variance_names": [names[index] for index in indices],
        "bn_counter_names": [names[index] for index in counters],
        "bn_counter_policy": "preserve_legacy_replay_exactly",
        "compact_counter_limitation": "Compact history excludes nonfloating deltas; legacy counters remain at initial values",
        "endpoint_source": ("global_before_plus_raw_retained_update" if compact else "weighted_retained_absolute_client_endpoints"),
        "endpoint_scale_policy": "unscaled_undamped_raw_history",
        "endpoint_rounds": trace, "no_batchnorm": not indices,
        "raw_data_access": False, "full_buffers_copied": False,
        "compact_precision_policy": "promote_saved_float16_to_float32_then_validate_without_clamping"}


def inspect_retained_bn_endpoints(history, forget_client_ids, dataset, num_classes):
    """Read-only structural/numerical preflight, without model inference or data."""
    _, diagnostics = _endpoints(history, forget_client_ids, _model(dataset, num_classes))
    return diagnostics


def federaser_lite_bn_endpoint(ctx):
    """Preserve legacy replay parameters and use retained endpoint BN buffers."""
    damping = float(getattr(ctx.cfg, "federaser_damping", 1.0))
    calibrate = bool(getattr(ctx.cfg, "federaser_calibrate", True))
    if not np.isfinite(damping):
        raise ValueError("BN endpoint replay damping must be finite")
    model = _model(ctx.dataset, ctx.num_classes)
    replacements, logs = _endpoints(ctx.history, ctx.forget_client_ids, model)
    legacy = federaser_lite_unlearn(ctx.history, ctx.forget_client_ids, damping=damping, calibrate=calibrate)
    names, indices, counters = _bn_indices(model)
    references = get_model_parameters(model)
    _validate_arrays(legacy, references, "legacy replay", allow_fp16=False)
    parameters = [array.copy() for array in legacy]
    for index, array in replacements.items():
        parameters[index] = array
    _validate_arrays(parameters, references, "final replay", allow_fp16=False)
    new_diagnostic = _bn_diagnostic(parameters, names, indices)
    if not new_diagnostic["valid"]:
        raise ValueError("Final retained BN endpoint state is invalid")
    parameter_names = set(dict(model.named_parameters()))
    changed = [name for name, old, new in zip(names, legacy, parameters) if not np.array_equal(old, new)]
    unchanged_parameters = all(np.array_equal(old, new) for name, old, new in zip(names, legacy, parameters)
                               if name in parameter_names)
    if not unchanged_parameters or any(name not in logs["bn_mean_variance_names"] for name in changed):
        raise ValueError("BN endpoint policy changed state outside BN running means/variances")
    set_model_parameters(model, parameters)
    if _hash(get_model_parameters(model)) != _hash(parameters):
        raise ValueError("BN endpoint model round-trip changed state")
    logs.update(damping=damping, calibrate=calibrate, forget_client_ids=list(ctx.forget_client_ids),
        calibration_norm_policy="unchanged_legacy_including_original_buffer_contributions",
        changed_buffers_from_legacy=changed, trainable_parameters_bitwise_identical_to_legacy=unchanged_parameters,
        counters_bitwise_identical_to_legacy=all(np.array_equal(legacy[index], parameters[index]) for index in counters),
        legacy_parameters_sha256=_hash(legacy), parameters_sha256=_hash(parameters),
        legacy_bn_diagnostics=_bn_diagnostic(legacy, names, indices), new_bn_diagnostics=new_diagnostic,
        limitations=["Historical retained endpoints were trained from original broadcasts and may retain forgotten-client influence",
                     "Running variance averaging is historical FedAvg state handling, not an exact pooled-data variance",
                     "This variant does not establish exact retraining equivalence or reproduce full original-paper FedEraser",
                     "Numerically valid state does not guarantee task utility or marker-audit acceptance",
                     "Legacy BN counters are preserved; no retained-only counter reconstruction is claimed"])
    return UnlearningResult(name=VARIANT, parameters=parameters, logs=logs)
