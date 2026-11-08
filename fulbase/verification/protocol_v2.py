"""Exploratory offline diagnostics; never replaces or certifies a v1 audit.

Only saved development observations are accepted. Geometry intervals describe
family sampling conditional on the checkpoint and public marker structure;
they neither calibrate an acceptance bound nor quantify training-run variation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.stats import t

from fulbase.verification.markers import MarkerConfig, client_anchor
from fulbase.verification.statistics import mean_test, correct_tests, combine_status


@dataclass(frozen=True)
class V2DiagnosticConfig:
    bootstrap_samples: int = 10000
    bootstrap_seed: int = 2718
    confidence: float = 0.95

    def __post_init__(self):
        if (isinstance(self.bootstrap_samples, bool)
                or not isinstance(self.bootstrap_samples, (int, np.integer))
                or self.bootstrap_samples < 100):
            raise ValueError("bootstrap_samples must be an integer >= 100")
        if (isinstance(self.bootstrap_seed, bool)
                or not isinstance(self.bootstrap_seed, (int, np.integer))
                or self.bootstrap_seed < 0):
            raise ValueError("bootstrap_seed must be a nonnegative integer")
        if not math.isfinite(self.confidence) or not 0.5 < self.confidence < 1:
            raise ValueError("confidence must be finite and between 0.5 and 1")


def _marker_config(value):
    cfg = value if isinstance(value, MarkerConfig) else MarkerConfig(**value)
    if not cfg.enabled:
        raise ValueError("Anchor diagnostics require enabled training markers")
    return cfg


def _check_protocol(cfg):
    required = ("alpha", "margin_drop", "retain_margin_tolerance", "min_families",
                "min_full_accuracy", "min_full_margin")
    for key in required:
        if key not in cfg or isinstance(cfg[key], bool) or not math.isfinite(cfg[key]):
            raise ValueError(f"Missing or invalid source protocol field: {key}")
    if (not 0 < cfg["alpha"] < 0.5
            or min(cfg["margin_drop"], cfg["retain_margin_tolerance"]) <= 0
            or int(cfg["min_families"]) != cfg["min_families"] or cfg["min_families"] < 2
            or not 0 <= cfg["min_full_accuracy"] <= 1):
        raise ValueError("Invalid source protocol bounds")


def validate_observations(before, after, forget_client_ids, marker_config, audit_protocol):
    """Validate without resampling, model loading, inference, or input mutation."""
    _marker_config(marker_config)
    _check_protocol(audit_protocol)
    ids = sorted(before)
    forget_list = list(forget_client_ids)
    forget = set(forget_list)
    if (ids != sorted(after) or len(ids) < 2 or len(forget) != len(forget_list)
            or not forget or not forget < set(ids)
            or any(isinstance(cid, bool) or not isinstance(cid, (int, np.integer))
                   or cid < 0 for cid in ids + forget_list)):
        raise ValueError("Require matching clients, distinct target IDs, and nonempty target/retain sets")
    feature_dim = None
    class_set = None
    all_sample_ids = set()
    for cid in ids:
        pre, post = before[cid], after[cid]
        for obs in (pre, post):
            if str(np.asarray(obs["role"]).item()) != "calibration":
                raise ValueError("v2 development diagnostics accept calibration observations only")
            if np.asarray(obs["cid"]).shape != () or int(obs["cid"]) != cid:
                raise ValueError("Observation client ID mismatch")
            if np.asarray(obs["valid"]).shape != () or np.asarray(obs["valid"]).item() is not True:
                raise ValueError("Invalid saved observations")
            n = len(obs["ids"])
            for key in ("ids", "families", "labels", "predictions", "normalized_margin"):
                a = np.asarray(obs[key])
                if a.ndim != 1 or len(a) != n or n == 0:
                    raise ValueError(f"Invalid observation shape: {key}")
            for key in ("ids", "families"):
                a = np.asarray(obs[key])
                if a.dtype.kind not in "US" or any(not str(x) for x in a):
                    raise ValueError(f"Expected nonempty string identifiers: {key}")
            if len(set(obs["ids"])) != n:
                raise ValueError("Duplicate marker sample IDs")
            for key in ("labels", "predictions"):
                a = np.asarray(obs[key])
                if a.dtype.kind not in "iu" or np.any(a < 0):
                    raise ValueError(f"Expected nonnegative integer {key}")
            labels = np.asarray(obs["labels"])
            classes = tuple(sorted(set(labels.tolist())))
            if len(classes) < 2 or classes != tuple(range(len(classes))):
                raise ValueError("Expected all contiguous marker classes")
            if class_set is None:
                class_set = classes
            if classes != class_set or np.any(np.asarray(obs["predictions"]) >= len(classes)):
                raise ValueError("Client marker classes/predictions differ")
            features = np.asarray(obs["features"])
            if features.ndim != 2 or features.shape[0] != n or features.shape[1] < 1:
                raise ValueError("Invalid feature shape")
            if feature_dim is None:
                feature_dim = features.shape[1]
            if features.shape[1] != feature_dim:
                raise ValueError("Feature dimensions differ across observations")
            for key in ("features", "normalized_margin", "margin", "logits"):
                a = np.asarray(obs[key])
                if a.dtype.kind not in "fiu" or not np.isfinite(a).all():
                    raise ValueError(f"Non-finite/nonnumeric observation: {key}")
            if np.asarray(obs["margin"]).shape != (n,):
                raise ValueError("Invalid raw margin shape")
            if np.asarray(obs["logits"]).shape != (n, len(classes)):
                raise ValueError("Invalid logits shape")
            logits = np.asarray(obs["logits"])
            if not np.array_equal(logits.argmax(1), obs["predictions"]):
                raise ValueError("Saved predictions are inconsistent with logits")
            correct = logits[np.arange(n), labels]
            competitors = logits.astype(np.float64).copy()
            competitors[np.arange(n), labels] = -np.inf
            raw = correct-competitors.max(1)
            scale = np.linalg.norm(logits.astype(np.float64)-logits.astype(np.float64).mean(1, keepdims=True), axis=1)
            if np.any(scale <= 1e-12):
                raise ValueError("Degenerate centered logit norm")
            # Stored observations were calculated with float32 Torch reductions.
            if not np.allclose(obs["margin"], raw, rtol=1e-4, atol=2e-6):
                raise ValueError("Saved raw margins are inconsistent with logits")
            if not np.allclose(obs["normalized_margin"], raw/scale, rtol=1e-4, atol=2e-6):
                raise ValueError("Saved normalized margins are inconsistent with logits")
            norms = np.linalg.norm(features.astype(np.float64), axis=1)
            if not np.allclose(norms, 1, rtol=1e-5, atol=1e-6):
                raise ValueError("Saved features must be finite nonzero unit-normalized vectors")
            for family in set(obs["families"]):
                if len(set(labels[np.asarray(obs["families"]) == family].tolist())) != 1:
                    raise ValueError("A marker family contains multiple labels")
            family_labels, _, family_features = _reduce(obs)
            _normalize_centers(np.mean([family_features[family_labels == label].mean(0)
                                        for label in classes], axis=0))
        for key in ("ids", "families", "labels"):
            if not np.array_equal(pre[key], post[key]):
                raise ValueError(f"Unpaired before/after observations: {key}")
        if all_sample_ids.intersection(pre["ids"]):
            raise ValueError("Marker sample IDs overlap across clients")
        all_sample_ids.update(pre["ids"])


def _reduce(obs):
    labels, margins, features = [], [], []
    families = np.asarray(obs["families"])
    for family in sorted(set(families)):
        mask = families == family
        labels.append(int(np.asarray(obs["labels"])[mask][0]))
        # Match v1's dtype/averaging exactly for the equivalence comparison.
        margins.append(float(np.asarray(obs["normalized_margin"])[mask].mean()))
        features.append(np.asarray(obs["features"])[mask].astype(np.float64).mean(0))
    return np.asarray(labels), np.asarray(margins), np.stack(features)


def _normalize_centers(x):
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 1e-12):
        raise ValueError("Zero/non-finite class-balanced marker center")
    return x / norms


def _noninferiority(values, bound, alpha, min_families):
    # Reuse descriptive intervals and finite/low-sample handling from v1.
    result = mean_test(values, "equivalence", bound, alpha, min_families)
    result["kind"] = "noninferiority"
    if result["mean"] is None:
        return result
    x = np.asarray(values, dtype=np.float64)
    se = float(x.std(ddof=1) / np.sqrt(len(x)))
    if se < 1e-15:
        p = 0.0 if x.mean() > -bound else (1.0 if x.mean() < -bound else 0.5)
    else:
        p = float(t.sf((x.mean() + bound) / se, len(x) - 1))
    result.update(p_support=p, p_violation=1-p)
    return result


def _interval(values, confidence, eligible):
    if not eligible:
        return None, None
    tail = (1-confidence)/2
    lo, hi = np.quantile(values, [tail, 1-tail])
    return float(lo), float(hi)


def diagnose_observations(before, after, forget_client_ids, marker_config,
                          audit_protocol, options=None):
    """Recompute uncalibrated anchor diagnostics and two explicit margin rules.

    There is deliberately no v2 joint acceptance decision and no task-metric
    inference here. The I/O layer may separately copy recorded benchmark utility.
    """
    options = options or V2DiagnosticConfig()
    validate_observations(before, after, forget_client_ids, marker_config, audit_protocol)
    cfg = _marker_config(marker_config)
    ids, forget = sorted(before), set(forget_client_ids)
    retain = [cid for cid in ids if cid not in forget]
    a = {cid: _reduce(before[cid]) for cid in ids}
    b = {cid: _reduce(after[cid]) for cid in ids}
    strata = {cid: [np.flatnonzero(a[cid][0] == label) for label in sorted(set(a[cid][0]))]
              for cid in ids}
    eligible = {cid: len(a[cid][0]) >= audit_protocol["min_families"]
                and all(len(s) >= 2 for s in strata[cid]) for cid in ids}
    dim = a[ids[0]][2].shape[1]
    anchors = np.stack([client_anchor(cfg, cid, dim, "cpu").numpy().astype(np.float64)
                        for cid in ids])
    anchor_units = anchors / np.linalg.norm(anchors, axis=1, keepdims=True)
    other_anchors = (anchor_units.sum(0)[None, :] - anchor_units) / (len(ids)-1)
    full_centers = np.stack([_normalize_centers(np.mean([a[cid][2][s].mean(0)
                                                       for s in strata[cid]], axis=0)) for cid in ids])
    after_centers = np.stack([_normalize_centers(np.mean([b[cid][2][s].mean(0)
                                                        for s in strata[cid]], axis=0)) for cid in ids])
    afull = ((full_centers-anchors)**2).sum(1)
    apost = ((after_centers-anchors)**2).sum(1)
    delta = apost-afull
    own = (full_centers*anchor_units).sum(1)
    other = (full_centers*other_anchors).sum(1)
    advantage = own-other
    retained_columns = [ids.index(cid) for cid in retain]
    point_selectivity = delta-delta[retained_columns].mean()
    draws_delta = np.empty((options.bootstrap_samples, len(ids)))
    draws_own = np.empty_like(draws_delta)
    draws_advantage = np.empty_like(draws_delta)
    rng = np.random.default_rng(options.bootstrap_seed)
    # Bounded memory: resample paired family counts, never expand sample variants.
    for start in range(0, options.bootstrap_samples, 256):
        size = min(256, options.bootstrap_samples-start)
        for column, cid in enumerate(ids):
            full = np.zeros((size, dim))
            post = np.zeros((size, dim))
            for s in strata[cid]:
                weights = rng.multinomial(len(s), np.full(len(s), 1/len(s)), size=size)
                weights = weights / (len(s)*len(strata[cid]))
                full += weights @ a[cid][2][s]
                post += weights @ b[cid][2][s]
            full, post = _normalize_centers(full), _normalize_centers(post)
            target = slice(start, start+size)
            draws_delta[target, column] = (((post-anchors[column])**2).sum(1)
                                           - ((full-anchors[column])**2).sum(1))
            draws_own[target, column] = full @ anchor_units[column]
            draws_advantage[target, column] = full @ (anchor_units[column]-other_anchors[column])
    draws_selectivity = draws_delta - draws_delta[:, retained_columns].mean(1, keepdims=True)
    geometry, full_geometry = [], []
    for column, cid in enumerate(ids):
        low, high = _interval(draws_delta[:, column], options.confidence, eligible[cid])
        slo, shi = (None, None)
        if cid in forget:
            slo, shi = _interval(draws_selectivity[:, column], options.confidence,
                                 eligible[cid] and all(eligible[c] for c in retain))
        geometry.append({"cid": cid, "client_group": "forget" if cid in forget else "retain",
                         "n_families": len(a[cid][0]), "n_classes": len(strata[cid]),
                         "A_full": float(afull[column]), "A_after": float(apost[column]),
                         "delta_anchor": float(delta[column]),
                         "delta_anchor_ci_low": low, "delta_anchor_ci_high": high,
                         "selective_delta": float(point_selectivity[column]) if cid in forget else None,
                         "selective_delta_ci_low": slo, "selective_delta_ci_high": shi,
                         "interval_status": "available" if eligible[cid] else "insufficient",
                         "selective_interval_status": ("available" if slo is not None else "insufficient")
                         if cid in forget else "not_applicable",
                         "status": "unvalidated"})
        ownlo, ownhi = _interval(draws_own[:, column], options.confidence, eligible[cid])
        advlo, advhi = _interval(draws_advantage[:, column], options.confidence, eligible[cid])
        full_geometry.append({"cid": cid, "n_families": len(a[cid][0]),
                              "own_anchor_cosine": float(own[column]),
                              "other_anchor_cosine_mean": float(other[column]),
                              "anchor_advantage": float(advantage[column]),
                              "own_anchor_cosine_ci_low": ownlo, "own_anchor_cosine_ci_high": ownhi,
                              "anchor_advantage_ci_low": advlo, "anchor_advantage_ci_high": advhi,
                              "interval_status": "available" if eligible[cid] else "insufficient",
                              "status": "unvalidated"})
    marker_metrics = []
    for cid in ids:
        pre, post = before[cid], after[cid]
        full_acc = float(np.mean(pre["predictions"] == pre["labels"]))
        full_margin = float(np.asarray(pre["normalized_margin"]).mean())
        after_margin = float(np.asarray(post["normalized_margin"]).mean())
        learned = full_acc >= audit_protocol["min_full_accuracy"] and full_margin >= audit_protocol["min_full_margin"]
        marker_metrics.append({"cid": cid, "client_group": "forget" if cid in forget else "retain",
                               "full_accuracy": full_acc,
                               "after_accuracy": float(np.mean(post["predictions"] == post["labels"])),
                               "full_normalized_margin": full_margin,
                               "after_normalized_margin": after_margin,
                               "normalized_margin_delta": after_margin-full_margin,
                               "full_marker_precheck": "pass" if learned else "insufficient"})
    all_learned = all(row["full_marker_precheck"] == "pass" for row in marker_metrics)
    margin_tests, margin_variants = [], {}
    margin_alpha = audit_protocol["alpha"]/2
    for variant in ("equivalence", "noninferiority"):
        tests = []
        for cid in ids:
            values = b[cid][1]-a[cid][1]
            if cid in forget:
                test = mean_test(values, "decrease", audit_protocol["margin_drop"],
                                 margin_alpha, audit_protocol["min_families"])
            elif variant == "equivalence":
                test = mean_test(values, "equivalence", audit_protocol["retain_margin_tolerance"],
                                 margin_alpha, audit_protocol["min_families"])
            else:
                test = _noninferiority(values, audit_protocol["retain_margin_tolerance"],
                                       margin_alpha, audit_protocol["min_families"])
            test.update(cid=cid, variant=variant, alpha=margin_alpha,
                        condition="forget_margin" if cid in forget else "retain_margin")
            tests.append(test)
        correct_tests(tests, margin_alpha)
        raw = combine_status(row["status"] for row in tests)
        margin_variants[variant] = {"status": raw if all_learned else "insufficient",
                                    "raw_status": raw, "full_marker_precheck_passed": all_learned,
                                    "interpretation": "exploratory_variant_not_formal_acceptance"}
        margin_tests.extend(tests)
    return {"schema_version": "marker-protocol-v2-diagnostic-v1", "status": "diagnostic_only",
            "formal_audit": False, "protocol_locked": False,
            "geometry": geometry, "full_geometry": full_geometry,
            "margin_tests": margin_tests, "margin_variants": margin_variants,
            "marker_metrics": marker_metrics,
            "protocol": {**asdict(options), "source_margin_alpha": margin_alpha,
                         "geometry_rule": "unvalidated_no_acceptance_bounds",
                         "geometry_center": "unit_normalize(class_balanced_mean(family_mean(unit_sample_features)))",
                         "anchor": "existing_training_client_anchor_float32_on_cpu",
                         "delta_anchor": "squared_distance(after_center,own_anchor)-squared_distance(full_center,own_anchor)",
                         "selective_delta": "target_delta_anchor-mean_over_retained_clients(delta_anchor)",
                         "full_anchor_advantage": "cos(full_center,own_anchor)-mean_cos(full_center,other_client_anchors)",
                         "resampling": "paired_class_stratified_family_bootstrap_joint_across_clients",
                         "geometry_interval_scope": "descriptive_percentile_not_multiplicity_adjusted",
                         "margin_interval_confidence": 1-margin_alpha,
                         "margin_correction": "Holm_over_support_and_violation_for_all_clients_separately_per_variant",
                         "cross_variant_or_cross_run_multiplicity_control": False,
                         "noninferiority_support": "mean_retained_margin_change > -retain_margin_tolerance",
                         "inference_scope": "conditional_on_saved_checkpoint_and_marker_generator",
                         "geometry_min_families": audit_protocol["min_families"],
                         "geometry_min_families_per_class": 2},
            "limitations": ["No calibrated v2 geometry threshold or joint acceptance decision",
                            "Anchor evidence does not prove real-data influence removal",
                            "Positive anchor advantage is descriptive, not a calibrated learnability pass",
                            "Family intervals do not measure uncertainty across training runs",
                            "Equivalence and noninferiority answer different preservation questions"]}
