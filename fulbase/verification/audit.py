from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from fulbase.verification.markers import MarkerDataset
from fulbase.verification.statistics import mean_test, correct_tests, combine_status


@dataclass(frozen=True)
class AuditConfig:
    alpha: float = 0.05
    margin_drop: float = 0.1
    retain_margin_tolerance: float = 0.1
    geometry_drop: float = 0.02
    retain_geometry_tolerance: float = 0.05
    min_families: int = 8
    min_full_accuracy: float = 0.7
    min_full_margin: float = 0.01
    max_retain_accuracy_drop: float = 0.03
    bootstrap_samples: int = 2000
    bootstrap_seed: int = 2718
    geometry_forget_rule: str = "unvalidated"  # or background_contraction after calibration
    protocol_locked: bool = False

    def __post_init__(self):
        if not all(math.isfinite(x) for x in (self.alpha, self.margin_drop,
                   self.retain_margin_tolerance, self.geometry_drop,
                   self.retain_geometry_tolerance, self.min_full_accuracy,
                   self.min_full_margin, self.max_retain_accuracy_drop)):
            raise ValueError("Audit configuration must be finite")
        if not 0 < self.alpha < 0.5 or self.min_families < 2 or self.bootstrap_samples < 100:
            raise ValueError("Invalid audit inference configuration")
        if min(self.margin_drop, self.retain_margin_tolerance, self.geometry_drop,
               self.retain_geometry_tolerance) <= 0:
            raise ValueError("Audit bounds must be positive")
        if not 0 <= self.min_full_accuracy <= 1 or not 0 <= self.max_retain_accuracy_drop <= 1:
            raise ValueError("Invalid accuracy bound")
        if self.geometry_forget_rule not in {"unvalidated", "background_contraction"}:
            raise ValueError("Unsupported geometry rule")
        if self.protocol_locked and self.geometry_forget_rule == "unvalidated":
            raise ValueError("Lock a calibrated geometry rule before formal audit")


def parameter_hash(parameters):
    digest = hashlib.sha256()
    for p in parameters:
        a = np.ascontiguousarray(p)
        digest.update(str((a.shape, a.dtype.str)).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def observe_markers(model, dataset: MarkerDataset, device, batch_size=128):
    model.to(device).eval()
    logits, features = [], []
    for x, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        z = model.features(x.to(device))
        logits.append(model.classifier(z).cpu())
        features.append(F.normalize(z, dim=1).cpu())
    logits, features = torch.cat(logits), torch.cat(features)
    labels = dataset.labels
    correct = logits.gather(1, labels[:, None]).squeeze(1)
    other = logits.masked_fill(F.one_hot(labels, logits.shape[1]).bool(), -torch.inf).max(1).values
    raw = correct - other
    scale = (logits - logits.mean(1, keepdim=True)).norm(dim=1)
    normalized = raw / scale.clamp_min(1e-12)
    return {"cid": dataset.cid, "role": dataset.role,
            "ids": np.asarray(dataset.ids), "families": np.asarray(dataset.families),
            "labels": labels.numpy(), "predictions": logits.argmax(1).numpy(),
            "margin": raw.numpy(), "normalized_margin": normalized.numpy(),
            "features": features.numpy(), "logits": logits.numpy(),
            "valid": bool(torch.isfinite(logits).all() and torch.isfinite(features).all()
                          and (scale > 1e-12).all()
                          and (features.norm(dim=1) > 1e-12).all())}


def save_observations(observations, path):
    arrays = {}
    for cid, obs in observations.items():
        for key, value in obs.items():
            arrays[f"c{cid}_{key}"] = np.asarray(value)
    np.savez_compressed(path, **arrays)


def _families(obs):
    result = []
    for fid in sorted(set(obs["families"])):
        mask = obs["families"] == fid
        result.append((str(fid), int(obs["labels"][mask][0]),
                       float(obs["normalized_margin"][mask].mean()),
                       obs["features"][mask].mean(0)))
    return result


def _geometry(before, after, forget, cfg):
    """Joint, class-stratified family bootstrap; recompute all shared centers."""
    ids = sorted(before)
    retain = [cid for cid in ids if cid not in forget]
    families = {cid: _families(before[cid]) for cid in ids}
    post = {cid: _families(after[cid]) for cid in ids}
    strata = {cid: [np.flatnonzero(np.array([x[1] for x in families[cid]]) == label)
                    for label in sorted(set(before[cid]["labels"]))] for cid in ids}
    full_features = {cid: np.stack([x[3] for x in families[cid]]) for cid in ids}
    post_features = {cid: np.stack([x[3] for x in post[cid]]) for cid in ids}
    def scores(centers):
        bg = np.mean([centers[cid] for cid in retain], axis=0)
        return {cid: float(np.linalg.norm(centers[cid] - bg)) for cid in forget}
    def changes(indices):
        a = {cid: np.mean([full_features[cid][s].mean(0) for s in indices[cid]], axis=0) for cid in ids}
        b = {cid: np.mean([post_features[cid][s].mean(0) for s in indices[cid]], axis=0) for cid in ids}
        sa, sb = scores(a), scores(b)
        out = {cid: sb[cid]-sa[cid] for cid in forget}
        for cid in retain:
            out[cid] = max((abs(np.linalg.norm(b[cid]-b[j])-np.linalg.norm(a[cid]-a[j]))
                            for j in retain if j != cid), default=0.0)
        return out
    point = changes(strata)
    rng = np.random.default_rng(cfg.bootstrap_seed)
    draws = {cid: [] for cid in ids}
    for _ in range(cfg.bootstrap_samples):
        indices = {cid: [rng.choice(s, len(s), replace=True) for s in strata[cid]] for cid in ids}
        for cid, value in changes(indices).items():
            draws[cid].append(value)
    # Half of alpha reserved for margin tests; Bonferroni CIs for all geometry claims.
    tail = cfg.alpha / (4 * len(ids))
    rows = []
    for cid in ids:
        low, high = map(float, np.quantile(draws[cid], [tail, 1-tail]))
        forgotten = cid in forget
        bound = cfg.geometry_drop if forgotten else cfg.retain_geometry_tolerance
        state = "insufficient"
        eligible = all(len(s) >= 2 for s in strata[cid]) and len(families[cid]) >= cfg.min_families
        if not forgotten and len(retain) < 2:
            eligible = False
        if forgotten and cfg.geometry_forget_rule == "unvalidated":
            eligible = False
        if eligible:
            if forgotten:
                state = "pass" if high < -bound else ("fail" if low > -bound else "insufficient")
            else:
                state = "pass" if high < bound else ("fail" if low > bound else "insufficient")
        rows.append({"cid": cid, "condition": "forget_geometry" if forgotten else "retain_geometry",
                     "kind": "joint_stratified_family_bootstrap", "mean": point[cid],
                     "ci_low": low, "ci_high": high, "bound": bound,
                     "interval_tail_probability": tail, "n_families": len(families[cid]),
                     "status": state, "geometry_rule_calibrated": cfg.geometry_forget_rule != "unvalidated"})
    return rows


def audit_observations(before, after, forget_client_ids, cfg: AuditConfig,
                       task_before=None, task_after=None):
    ids, forget = sorted(before), set(forget_client_ids)
    if ids != sorted(after) or not forget or not forget < set(ids):
        raise ValueError("Audit requires matching clients, nonempty F and nonempty R")
    if cfg.protocol_locked and cfg.bootstrap_samples * cfg.alpha / (4 * len(ids)) < 10:
        raise ValueError("Too few bootstrap samples for the multiplicity-adjusted tail")
    conditions, margin_tests = [], []
    valid_clients = []
    for cid in ids:
        pre, post = before[cid], after[cid]
        for key in ("ids", "families", "labels"):
            if not np.array_equal(pre[key], post[key]):
                raise ValueError(f"Unpaired audit markers for client {cid}: {key}")
        valid = pre["valid"] and post["valid"]
        accuracy = float(np.mean(pre["predictions"] == pre["labels"]))
        learnable = valid and accuracy >= cfg.min_full_accuracy and float(pre["normalized_margin"].mean()) >= cfg.min_full_margin
        conditions.append({"cid": cid, "condition": "full_marker_precheck", "status": "pass" if learnable else "insufficient",
                           "full_marker_accuracy": accuracy,
                           "reason": "learnability/finite/nonzero-feature precheck"})
        a, b = _families(pre), _families(post)
        differences = [y[2]-x[2] for x, y in zip(a, b)]
        test = mean_test(differences, "decrease" if cid in forget else "equivalence",
                         cfg.margin_drop if cid in forget else cfg.retain_margin_tolerance,
                         cfg.alpha/2, cfg.min_families)
        test.update(cid=cid, condition="forget_margin" if cid in forget else "retain_margin")
        margin_tests.append(test)
        if valid:
            valid_clients.append(cid)
    correct_tests(margin_tests, cfg.alpha/2)
    conditions.extend(margin_tests)
    if len(valid_clients) == len(ids):
        conditions.extend(_geometry(before, after, forget, cfg))
    else:
        conditions.append({"condition": "geometry", "status": "insufficient", "reason": "Invalid observations"})
    if task_before is None or task_after is None:
        task = {"condition": "retain_task", "status": "insufficient",
                "reason": "No independent task evaluation supplied"}
    else:
        drop = float(task_before["retain_acc"] - task_after["retain_acc"])
        task = {"condition": "retain_task", "status": "pass" if drop <= cfg.max_retain_accuracy_drop else "fail",
                "accuracy_drop": drop, "bound": cfg.max_retain_accuracy_drop,
                "source": "independent_benchmark_evaluation_not_private_data_audit"}
    conditions.append(task)
    # An unlearned Full marker cannot support either a positive or negative removal claim.
    all_learned = all(c["status"] == "pass" for c in conditions if c["condition"] == "full_marker_precheck")
    exploratory = combine_status(c["status"] for c in conditions) if all_learned else "insufficient"
    variants = {}
    for name, selected in {
        "margin_only": [c["status"] for c in conditions if c["condition"].endswith("_margin")],
        "geometry_only": [c["status"] for c in conditions if "geometry" in c["condition"]],
        "joint_structure": [c["status"] for c in conditions if c["condition"].endswith(("_margin", "_geometry"))],
        "complete": [c["status"] for c in conditions],
    }.items():
        variants[name] = combine_status(selected) if all_learned else "insufficient"
    fixed = ["pass" if (c["mean"] < -c["bound"] if c["kind"] == "decrease"
                         else abs(c["mean"]) < c["bound"]) else "fail"
             for c in margin_tests if c["mean"] is not None]
    variants["fixed_margin_threshold"] = (combine_status(fixed) if all_learned and len(fixed) == len(ids)
                                           else "insufficient")
    return {"schema_version": "marker-audit-v1", "protocol": asdict(cfg),
            "status": exploratory if cfg.protocol_locked else "insufficient",
            "exploratory_status": exploratory, "protocol_locked": cfg.protocol_locked,
            "reason": "Frozen protocol" if cfg.protocol_locked else "Development bounds; calibration is not locked",
            "forget_client_ids": sorted(forget), "conditions": conditions,
            "exploratory_decision_variants": variants,
            "inference_scope": "conditional_on_this_checkpoint_and_marker_generator",
            "limitations": ["Marker evidence is not proof of removal of all real-data influence",
                            "Checkpoint binding does not authenticate deployed behavior",
                            "Bootstrap percentile coverage is approximate"]}


def detection_summary(records):
    """Descriptive per-request counts only; dependent checkpoints are not trials."""
    rows = []
    for behavior in ("fake", "strict_retrain_reference"):
        group = [r for r in records if r["behavior"] == behavior]
        if not group:
            continue
        for column in ("status", "exploratory_status"):
            counts = {state: sum(r[column] == state for r in group)
                      for state in ("pass", "fail", "insufficient")}
            rows.append({"behavior": behavior, "decision_column": column, "n_requests": len(group),
                         **{f"n_{s}": n for s,n in counts.items()},
                         **{f"fraction_{s}": n/len(group) for s,n in counts.items()},
                         "uncertainty": "not_estimated_single_training_run"})
    return rows


def write_audit(before, after, forget, cfg, output_dir, metadata,
                task_before=None, task_after=None):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    report = audit_observations(before, after, forget, cfg, task_before, task_after)
    report["metadata"] = metadata
    rows = []
    for stage, values in (("full", before), ("after", after)):
        for cid, obs in values.items():
            for k, mid in enumerate(obs["ids"]):
                rows.append({"stage": stage, "cid": cid, "marker_id": str(mid),
                             "family_id": str(obs["families"][k]), "label": int(obs["labels"][k]),
                             "prediction": int(obs["predictions"][k]), "margin": float(obs["margin"][k]),
                             "normalized_margin": float(obs["normalized_margin"][k])})
    pd.DataFrame(rows).to_csv(path / "marker_metrics.csv", index=False)
    pd.DataFrame(report["conditions"]).to_csv(path / "client_tests.csv", index=False)
    pd.DataFrame([r for r in report["conditions"] if "geometry" in r["condition"]]).to_csv(path / "geometry_metrics.csv", index=False)
    # Raw logits/embeddings enable later checks without rerunning inference.
    save_observations(before, path / "full_observations.npz")
    save_observations(after, path / "after_observations.npz")
    report["statistical_and_output_time_sec"] = time.perf_counter() - start
    def clean(value):
        if isinstance(value, np.generic):
            return clean(value.item())
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        return value
    with (path / "audit_report.json").open("w", encoding="utf-8") as stream:
        json.dump(clean(report), stream, ensure_ascii=False, indent=2, allow_nan=False)
    return clean(report)
