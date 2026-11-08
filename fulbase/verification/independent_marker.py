"""Freeze and collect fresh audit markers for the nine prespecified checkpoints.

This entry point never trains, loads benchmark images, or writes source runs.
The private replay store is deliberately separate from public provenance.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import secrets

import numpy as np
import torch
from torch.utils.data import Dataset

from fulbase.models import build_model
from fulbase.utils import get_model_parameters, set_model_parameters
from fulbase.verification.audit import (
    AuditConfig, file_hash, observe_markers, parameter_hash, save_observations,
)
from fulbase.verification.markers import MarkerConfig, MarkerDataset, derived_seed
from fulbase.verification.protocol_v2_io import _task_diagnostic


SCHEMA = "rp2-independent-marker-collection-v1"
METHODS = ("Full", "Retrain", "fixed_C_lambda05")
C_STAGE = "marker_capretain_head_s30_lr0.01_r0.1_d3.0_fixed_w0.5_pre_recovery"
CODE_FILES = (
    "fulbase/models.py", "fulbase/utils.py", "fulbase/verification/markers.py",
    "fulbase/verification/audit.py", "fulbase/verification/statistics.py",
    "fulbase/verification/protocol_v2.py", "fulbase/verification/protocol_v2_io.py",
    "fulbase/verification/protocol_v2_ablation.py",
    "fulbase/verification/protocol_v2_stratified.py",
    "fulbase/verification/independent_marker.py",
    "fulbase/verification/independent_marker_analysis.py",
    "scripts/collect_independent_markers.py",
)


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _write_json(path, value, private=False):
    """Publish only complete files, refusing to overwrite an existing artifact."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    # A pending file is not evidence. Interrupted writes can be retried identically.
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(temporary, flags, 0o600 if private else 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _inside(root, relative):
    path = (root / str(relative).replace("\\", "/")).resolve()
    if root not in path.parents:
        raise ValueError(f"Source artifact outside its root: {relative}")
    return path


def _root(project_root=None):
    return Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[2]


def _model_key(binding):
    return f"{binding['training_seed']}_{binding['method']}"


def validate_plan(plan):
    if plan.get("schema_version") != "rp2-independent-marker-validation-plan-v1":
        raise ValueError("Unsupported independent marker plan")
    sampling = plan["sampling"]
    required = {
        "role": "audit", "training_seeds": [43, 44, 45], "clients": [0, 1, 2, 3, 4],
        "forget_clients": [0], "num_classes": 10, "batches": 3, "packets_per_batch": 5,
        "families_per_class_per_packet": 4, "variants_per_family": 2,
        "families_per_class_per_batch": 20, "families_per_class_primary_pooled": 60,
        "unique_marker_inputs": 6000, "sample_model_evaluations": 54000,
        "preserve_training_marker_config_and_fingerprint": True, "same_inputs_all_models": True,
        "full_observations_reused_for_C_and_Retrain": True,
    }
    for key, expected in required.items():
        if sampling.get(key) != expected:
            raise ValueError(f"The fixed 9-model, 3 x 20 protocol requires sampling.{key}={expected}")
    analysis = plan["analysis"]
    bounds = {
        "primary_sampling_unit": "paired_family", "primary_estimand": "equal_mean_over_fixed_10_classes",
        "primary_dataset": "pool_all_three_prespecified_batches_before_decision",
        "primary_retained_rule": "noninferiority",
        "primary_structure": "stratified_margin_and_existing_anchor_conditions",
        "per_request_alpha": .05, "margin_alpha": .025, "geometry_alpha": .025,
        "margin_client_count": 5, "geometry_claim_count": 7, "bootstrap_samples": 20000,
        "bootstrap_seed": 2718, "descriptive_confidence": .95,
        "margin_drop": .1, "retain_margin_tolerance": .1, "anchor_drop": .01,
        "anchor_selectivity_drop": .01, "retain_anchor_tolerance": .02,
        "min_full_anchor_advantage": 0., "min_full_accuracy": .7, "min_full_margin": .01,
        "accuracy_drop": .2, "retain_accuracy_tolerance": .05,
        "max_task_retain_accuracy_drop": .03, "formal_unlearning_claim": False,
        "cross_candidate_or_training_seed_global_error_control": False,
    }
    for key, expected in bounds.items():
        if analysis.get(key) != expected:
            raise ValueError(f"Frozen analysis value changed: {key}")
    models = plan["models"]
    keys = {(row["training_seed"], row["method"]) for row in models}
    if len(models) != 9 or keys != {(seed, method) for seed in (43, 44, 45) for method in METHODS}:
        raise ValueError("Exactly the nine prespecified seed/method bindings are required")
    stages = {"Full": "full_after_fl", "Retrain": "retrain_fl_pre_recovery", "fixed_C_lambda05": C_STAGE}
    for row in models:
        if row["stage"] != stages[row["method"]]:
            raise ValueError("Unexpected stage for the fixed model method")
        for name in ("checkpoint_sha256_recorded", "parameters_sha256_recorded",
                     "marker_fingerprint", "partition_sha256"):
            if re.fullmatch("[0-9a-f]{64}", str(row.get(name, ""))) is None:
                raise ValueError(f"Missing model binding: {name}")
    if len({row["parameters_sha256_recorded"] for row in models}) != 9:
        raise ValueError("The nine model parameter bindings must be distinct")
    if len({row["marker_fingerprint"] for row in models}) != 1:
        raise ValueError("All nine models must use the same original marker fingerprint")
    for seed in (43, 44, 45):
        if len({row["partition_sha256"] for row in models if row["training_seed"] == seed}) != 1:
            raise ValueError("Same-seed models must bind the same partition")
    return plan


def _read_npz(path):
    observations = {}
    with np.load(path, allow_pickle=False) as arrays:
        for name in arrays.files:
            match = re.fullmatch(r"c(\d+)_(\w+)", name)
            if match is None:
                raise ValueError(f"Unexpected observation field in {path.name}")
            cid, field = int(match[1]), match[2]
            observations.setdefault(cid, {})[field] = arrays[name].copy()
    for cid, obs in observations.items():
        for field in ("cid", "role", "valid"):
            if field not in obs or np.shape(obs[field]) != ():
                raise ValueError("Invalid saved observation scalar")
            obs[field] = obs[field].item()
        if obs["cid"] != cid:
            raise ValueError("Saved observation client mismatch")
    return observations


def _tensor_hash(image):
    array = np.ascontiguousarray(image.detach().cpu().numpy())
    digest = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _dataset_rows(dataset):
    return [dict(row, tensor_sha256=_tensor_hash(image))
            for row, image in zip(dataset.manifest(), dataset.images)]


def _source_registry(root, run_dirs):
    """Bind explicitly registered histories; unknown old audit seeds stay unknown."""
    files, ids, families, tensor_hashes, coverage = {}, set(), set(), set(), []
    for relative in sorted(set(run_dirs)):
        run = _inside(root, relative)
        cfg = _json(run / "config.json")
        markers = MarkerConfig(**_json(run / "verification/marker_config.json"))
        nclasses = _json(run / "verification/checkpoint_manifest.json")["num_classes"]
        required = [run / "config.json", run / "data_partition.json",
                    run / "verification/marker_config.json", run / "verification/audit_config.json",
                    run / "verification/checkpoint_manifest.json",
                    run / "verification/training_marker_manifest.json"]
        optional = sorted((run / "verification").rglob("*marker_manifest.json"))
        evidence = [path for path in (run / "run_status.json", run / "diagnostic_config.json",
                                      run / "verification/training_result.json") if path.is_file()]
        archives = sorted((run / "verification").rglob("*observations.npz"))
        reports = sorted((run / "verification").rglob("audit_report.json"))
        for path in sorted(set(required + optional + archives + reports + evidence)):
            if not path.is_file():
                raise FileNotFoundError(f"Missing registered history: {path}")
            files[path.relative_to(root).as_posix()] = file_hash(path)
        for path in optional:
            rows = _json(path)
            if not isinstance(rows, list):
                raise ValueError("Historical marker manifest must be a list")
            for row in rows:
                ids.add(row["marker_id"])
                families.add(row["family_id"])
                if row.get("tensor_sha256"):
                    tensor_hashes.add(row["tensor_sha256"])
        historical_roles = set()
        for path in archives:
            for obs in _read_npz(path).values():
                ids.update(map(str, obs["ids"]))
                families.update(map(str, obs["families"]))
                historical_roles.add(obs["role"])
                if "tensor_sha256" in obs:
                    tensor_hashes.update(map(str, obs["tensor_sha256"]))
        # Generator-level content comparison is possible for these public legacy seeds.
        regenerated = []
        for role, seed in (("train", derived_seed(cfg["seed"], "train-markers")),
                           ("calibration", cfg["calibration_seed"])):
            generated_rows = []
            for cid in range(cfg["num_clients"]):
                dataset = MarkerDataset(markers, cid, cfg["dataset"], role, seed, nclasses)
                rows = _dataset_rows(dataset)
                generated_rows.extend(dataset.manifest())
                ids.update(row["marker_id"] for row in rows)
                families.update(row["family_id"] for row in rows)
                tensor_hashes.update(row["tensor_sha256"] for row in rows)
            stored = run / "verification" / ("training_marker_manifest.json" if role == "train"
                                              else "calibration_marker_manifest.json")
            if stored.is_file():
                original = _json(stored)
                fields = ("cid", "role", "marker_id", "family_id", "label")
                actual = sorted(tuple(row[field] for field in fields) for row in original)
                expected = sorted(tuple(row[field] for field in fields) for row in generated_rows)
                if actual != expected:
                    raise ValueError("Historical marker manifest does not match original generator identities")
            regenerated.append(role)
        coverage.append({"run_dir": Path(relative).as_posix(), "observation_files": len(archives),
                         "content_regenerated_roles": regenerated,
                         "historical_observation_roles": sorted(historical_roles),
                         "old_audit_content_coverage": "only_stored_content_hashes_if_available; private_seeds_unavailable"})
    return {"source_hashes": files, "ids": ids, "families": families,
            "tensor_hashes": tensor_hashes, "coverage": coverage}


def _load_parameters(context, binding):
    path = _inside(_inside(context["root"], binding["run_dir"]), binding["checkpoint"])
    if file_hash(path) != binding["checkpoint_sha256_recorded"]:
        raise ValueError(f"Checkpoint bytes differ: {_model_key(binding)}")
    # These are locally trusted project artifacts, verified before deserialization.
    parameters = torch.load(path, map_location="cpu", weights_only=False)
    if parameter_hash(parameters) != binding["parameters_sha256_recorded"]:
        raise ValueError(f"Checkpoint parameters differ: {_model_key(binding)}")
    if file_hash(path) != binding["checkpoint_sha256_recorded"]:
        raise ValueError("Checkpoint changed while loading")
    return parameters


def preflight(plan_path, project_root=None, verify_models=True):
    """Validate metadata and optionally every checkpoint; no outputs or inference."""
    root, plan_path = _root(project_root), Path(plan_path).resolve()
    plan = validate_plan(_json(plan_path))
    models, contexts, tasks = plan["models"], {}, {}
    dataset, marker_values, protocol_values = None, None, None
    for binding in models:
        run = _inside(root, binding["run_dir"])
        cfg = _json(run / "config.json")
        markers = MarkerConfig(**_json(run / "verification/marker_config.json"))
        protocol = asdict(AuditConfig(**_json(run / "verification/audit_config.json")))
        manifest = _json(run / "verification/checkpoint_manifest.json")
        entry = manifest["stages"][binding["stage"]]
        if (cfg["seed"] != binding["training_seed"] or cfg["num_clients"] != 5 or
                str(cfg["forget_clients"]) != "0" or manifest["num_classes"] != 10):
            raise ValueError("Model source seed/client/class/forget configuration differs")
        if not markers.enabled or markers.families_per_class != 4 or markers.variants_per_family != 2:
            raise ValueError("Original training marker configuration must remain four families/two variants")
        if markers.fingerprint() != binding["marker_fingerprint"] or manifest["marker_fingerprint"] != markers.fingerprint():
            raise ValueError("Original training marker fingerprint differs")
        if file_hash(run / "data_partition.json") != binding["partition_sha256"] or manifest["partition_sha256"] != binding["partition_sha256"]:
            raise ValueError("Partition differs from the fixed model binding")
        for source, target in (("checkpoint_sha256", "checkpoint_sha256_recorded"),
                               ("parameters_sha256", "parameters_sha256_recorded")):
            if entry.get(source) != binding[target]:
                raise ValueError(f"Manifest model binding differs: {_model_key(binding)} {source}")
        if (entry.get("checkpoint", "").replace("\\", "/") != binding["checkpoint"].replace("\\", "/") or
                entry.get("stage") != binding["stage"] or entry.get("marker_fingerprint") != markers.fingerprint()):
            raise ValueError("Stage checkpoint/fingerprint differs from plan")
        seed_full = next(row for row in models if row["training_seed"] == binding["training_seed"] and row["method"] == "Full")
        source_full = manifest["stages"]["full_after_fl"]
        if (source_full["parameters_sha256"] != seed_full["parameters_sha256_recorded"] or
                source_full["checkpoint_sha256"] != seed_full["checkpoint_sha256_recorded"]):
            raise ValueError("Candidate source Full differs from the shared fixed Full")
        status_path = run / "run_status.json"
        if status_path.is_file():
            status = _json(status_path)
            if status.get("status") not in {"development_run_complete", "pga_diagnostic_complete"}:
                raise ValueError("Source run is not complete")
            if status.get("source_full_parameters_sha256", seed_full["parameters_sha256_recorded"]) != seed_full["parameters_sha256_recorded"]:
                raise ValueError("Source completion record binds a different Full")
        training_path = run / "verification/training_result.json"
        if training_path.is_file() and _json(training_path).get("final_parameters_sha256") != seed_full["parameters_sha256_recorded"]:
            raise ValueError("Source training result binds a different Full")
        diagnostic_path = run / "diagnostic_config.json"
        if binding["method"] == "fixed_C_lambda05" and diagnostic_path.is_file():
            diagnostic = _json(diagnostic_path)
            trial = {"steps": 30, "lr": .01, "project_ratio": .1, "layers": "head",
                     "data_scope": "markers_only", "objective": "margin_capped_retain",
                     "margin_raw_drop": 3., "retain_weight": .5, "retain_weight_mode": "fixed"}
            if diagnostic.get("trials") != [trial] or diagnostic.get("source_full_parameters_sha256") != seed_full["parameters_sha256_recorded"]:
                raise ValueError("Fixed C source diagnostic budget differs from the prespecified trial")
        if dataset is not None and (cfg["dataset"] != dataset or asdict(markers) != marker_values or protocol != protocol_values):
            raise ValueError("All fixed models must share dataset, marker config, and original audit protocol")
        dataset, marker_values, protocol_values = cfg["dataset"], asdict(markers), protocol
        contexts[_model_key(binding)] = {"cfg": cfg, "manifest": manifest}
        if binding["method"] != "Full":
            report = _json(run / "verification" / binding["stage"] / "audit_report.json")
            tasks[_model_key(binding)] = _task_diagnostic(source_full, entry, report, protocol)
    for key, value in (("alpha", "per_request_alpha"), ("margin_drop", "margin_drop"),
                       ("retain_margin_tolerance", "retain_margin_tolerance"),
                       ("min_full_accuracy", "min_full_accuracy"), ("min_full_margin", "min_full_margin"),
                       ("max_retain_accuracy_drop", "max_task_retain_accuracy_drop")):
        if protocol_values[key] != plan["analysis"][value]:
            raise ValueError(f"Source threshold differs from frozen plan: {key}")
    history = sorted({row["run_dir"] for row in models} | set(plan["sampling"].get("history_run_dirs", [])))
    registry = _source_registry(root, history)
    context = {"root": root, "plan_path": plan_path, "plan": plan, "dataset": dataset,
               "marker_config": marker_values, "audit_protocol": protocol_values,
               "registry": registry, "models": contexts, "task_metrics": tasks,
               "checkpoint_bytes_reverified": False}
    if verify_models:
        missing = [_model_key(row) for row in models
                   if not _inside(_inside(root, row["run_dir"]), row["checkpoint"]).is_file()]
        if missing:
            raise FileNotFoundError("Fixed checkpoint files missing: " + ", ".join(missing))
        for binding in models:
            parameters = _load_parameters(context, binding)
            model = build_model(dataset, 10)
            set_model_parameters(model, parameters)
            if parameter_hash(get_model_parameters(model)) != binding["parameters_sha256_recorded"]:
                raise ValueError("Checkpoint does not round-trip through the bound model architecture")
            del model, parameters
        context["checkpoint_bytes_reverified"] = True
    return context


def _code_hashes():
    root = _root()
    return {name: file_hash(root / name) for name in CODE_FILES}


def _runtime():
    import scipy
    return {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "scipy": scipy.__version__}


def _output_path(output_dir, context):
    output = Path(output_dir).resolve()
    run_dirs = {row["run_dir"] for row in context["plan"]["models"]}
    run_dirs.update(context["plan"]["sampling"].get("history_run_dirs", []))
    for run_dir in run_dirs:
        source = _inside(context["root"], run_dir)
        if output == source or source in output.parents:
            raise ValueError("Independent output must be outside every source run")
    return output


def _packet_commitment(batch, packet, seed):
    return _digest({"purpose": SCHEMA, "batch": batch, "packet": packet, "private_sample_seed": seed})


def _freshness(context):
    return {"status": "disjoint_from_registered_identities_and_available_tensor_hashes",
            "global_historical_isolation_claim": False,
            "registered_sample_ids": len(context["registry"]["ids"]),
            "registered_family_ids": len(context["registry"]["families"]),
            "registered_tensor_hashes": len(context["registry"]["tensor_hashes"]),
            "coverage": context["registry"]["coverage"]}


def _counts():
    return {"batches": 3, "packets": 15, "unique_inputs": 6000,
            "unique_families": 3000, "model_evaluations": 54000}


def _generate(context, seeds):
    markers = MarkerConfig(**context["marker_config"])
    datasets, rows, packets = {}, [], []
    for batch in range(3):
        datasets[batch] = {cid: [] for cid in range(5)}
        for packet in range(5):
            seed = seeds[batch * 5 + packet]
            packets.append({"batch": batch, "packet": packet,
                            "seed_commitment": _packet_commitment(batch, packet, seed)})
            for cid in range(5):
                ds = MarkerDataset(markers, cid, context["dataset"], "audit", seed, 10)
                datasets[batch][cid].append(ds)
                rows.extend(dict(row, batch=batch, packet=packet) for row in _dataset_rows(ds))
    ids, families, contents = [row["marker_id"] for row in rows], {row["family_id"] for row in rows}, [row["tensor_sha256"] for row in rows]
    if len(ids) != 6000 or len(set(ids)) != 6000 or len(families) != 3000 or len(set(contents)) != 6000:
        raise ValueError("Duplicate new packet identity, family token, or marker tensor")
    registry = context["registry"]
    if set(ids) & registry["ids"] or families & registry["families"] or set(contents) & registry["tensor_hashes"]:
        raise ValueError("Fresh marker overlaps a registered historical identity or tensor")
    return datasets, rows, packets


def prepare_collection(plan_path, output_dir, project_root=None):
    context = preflight(plan_path, project_root, verify_models=True)
    output = _output_path(output_dir, context)
    if (output / "frozen_manifest.json").exists():
        return _load_frozen(context, output)[0]
    intent = {"plan_sha256": file_hash(context["plan_path"]), "code_hashes": _code_hashes(),
              "source_hashes": context["registry"]["source_hashes"], "runtime": _runtime()}
    replay_path = output / ".private/replay.json"
    if output.exists() and any(output.iterdir()):
        allowed = {".private", "plan.json", "inputs.json", "plan.json.pending", "inputs.json.pending",
                   "frozen_manifest.json.pending"}
        if {path.name for path in output.iterdir()} - allowed or not replay_path.is_file():
            raise FileExistsError("Unrecognized partial preparation; no new seeds will replace its inputs")
        replay = _json(replay_path)
        if replay.get("preparation_context") != intent:
            raise ValueError("Partial preparation source/code/protocol changed; original inputs cannot be substituted")
        seeds = replay["packet_seeds"]
    else:
        seeds = [secrets.randbits(128) for _ in range(15)]
    if len(seeds) != 15 or len(set(seeds)) != 15:
        raise ValueError("Invalid private packet seeds; preparation aborted")
    _, rows, packets = _generate(context, seeds)
    output.mkdir(parents=True, exist_ok=True)
    private = output / ".private"
    private.mkdir(mode=0o700, exist_ok=True)
    if os.name != "nt":
        private.chmod(0o700)
    if not replay_path.exists():
        _write_json(replay_path, {"schema_version": SCHEMA, "preparation_context": intent,
                                 "packet_seeds": seeds}, private=True)
    for path, value in ((output / "plan.json", context["plan"]), (output / "inputs.json", rows)):
        if path.exists():
            if _json(path) != value:
                raise ValueError("Partial preparation differs from its identical-input replay")
        else:
            _write_json(path, value)
    _assert_sources_unchanged(context)
    if _code_hashes() != intent["code_hashes"]:
        raise ValueError("Code changed during preparation")
    manifest = {
        "schema_version": SCHEMA, "status": "frozen_before_inference", "formal_audit": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": file_hash(context["plan_path"]), "plan_copy_sha256": file_hash(output / "plan.json"),
        "marker_fingerprint": MarkerConfig(**context["marker_config"]).fingerprint(),
        "marker_config": context["marker_config"], "audit_protocol": context["audit_protocol"],
        "dataset": context["dataset"], "models": context["plan"]["models"],
        "checkpoint_bytes_and_parameters_reverified": True,
        "source_hashes": context["registry"]["source_hashes"], "code_hashes": _code_hashes(),
        "runtime": _runtime(), "inputs_sha256": file_hash(output / "inputs.json"),
        "private_replay_sha256": file_hash(private / "replay.json"), "packets": packets,
        "counts": _counts(), "freshness": _freshness(context),
        "task_metrics": context["task_metrics"],
        "limitations": ["Historical audit tensor content cannot be reconstructed without its private replay seed.",
                        "Validation is conditional on fixed checkpoints and the original marker generator.",
                        "Private replay store must remain auditor-owned and must not be sent to training."]}
    _write_json(output / "frozen_manifest.json", manifest)
    return manifest


def _load_frozen(context, output):
    manifest = _json(output / "frozen_manifest.json")
    if manifest.get("schema_version") != SCHEMA or manifest.get("status") != "frozen_before_inference":
        raise ValueError("Unsupported or incomplete frozen collection")
    checks = {"plan_sha256": file_hash(context["plan_path"]),
              "plan_copy_sha256": file_hash(output / "plan.json"),
              "inputs_sha256": file_hash(output / "inputs.json"),
              "private_replay_sha256": file_hash(output / ".private/replay.json"),
              "source_hashes": context["registry"]["source_hashes"], "code_hashes": _code_hashes(),
              "runtime": _runtime(), "marker_config": context["marker_config"],
              "audit_protocol": context["audit_protocol"], "models": context["plan"]["models"],
              "task_metrics": context["task_metrics"], "dataset": context["dataset"],
              "marker_fingerprint": MarkerConfig(**context["marker_config"]).fingerprint(),
              "counts": _counts(), "freshness": _freshness(context), "formal_audit": False,
              "checkpoint_bytes_and_parameters_reverified": True}
    for key, expected in checks.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Frozen collection changed: {key}; resume requires identical inputs and protocol")
    seeds = _json(output / ".private/replay.json")["packet_seeds"]
    if len(seeds) != 15 or len(set(seeds)) != 15 or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("Invalid private replay store")
    datasets, rows, packets = _generate(context, seeds)
    if rows != _json(output / "inputs.json") or packets != manifest["packets"]:
        raise ValueError("Regenerated marker identities/tensors or seed commitments differ")
    return manifest, datasets


class _MergedMarkers(Dataset):
    def __init__(self, packets):
        self.cid, self.role = packets[0].cid, "audit"
        self.images = torch.cat([ds.images for ds in packets])
        self.labels = torch.cat([ds.labels for ds in packets])
        self.ids = [marker for ds in packets for marker in ds.ids]
        self.families = [family for ds in packets for family in ds.families]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


def _validate_shard(observations, datasets, context):
    from fulbase.verification.protocol_v2 import validate_observations
    if set(observations) != set(range(5)):
        raise ValueError("Incomplete client observations")
    for cid, obs in observations.items():
        ds = datasets[cid]
        if (obs["role"] != "audit" or obs["cid"] != cid or
                not np.array_equal(obs["ids"], ds.ids) or
                not np.array_equal(obs["families"], ds.families) or
                not np.array_equal(obs["labels"], ds.labels.numpy())):
            raise ValueError("Observation shard does not match frozen audit inputs")
    validate_observations(observations, observations, [0], context["marker_config"],
                          context["audit_protocol"], expected_role="audit")


def _read_shard(path, receipt_path, expected, datasets, context):
    receipt = _json(receipt_path)
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Observation receipt differs from frozen binding")
    if receipt.get("observation_sha256") != file_hash(path):
        raise ValueError("Observation shard hash changed")
    settings_path = path.parent.parent / "collection_settings.json"
    settings = _json(settings_path)
    if (receipt.get("model_unchanged_after_inference") is not True or
            receipt.get("inference_device") != settings["device"] or
            receipt.get("inference_batch_size") != settings["batch_size"]):
        raise ValueError("Observation receipt state or inference settings differ")
    obs = _read_npz(path)
    _validate_shard(obs, datasets, context)
    return obs, receipt


def _shard_expected(manifest_hash, binding, batch, input_hash, settings_hash):
    return {"schema_version": SCHEMA, "frozen_manifest_sha256": manifest_hash,
            "model_key": _model_key(binding), "batch": batch, "inputs_sha256": input_hash,
            "collection_settings_sha256": settings_hash,
            "checkpoint_sha256": binding["checkpoint_sha256_recorded"],
            "parameters_sha256": binding["parameters_sha256_recorded"]}


def _assert_sources_unchanged(context):
    for relative, expected in context["registry"]["source_hashes"].items():
        if file_hash(_inside(context["root"], relative)) != expected:
            raise ValueError("Registered source changed during collection")


def collect_observations(plan_path, output_dir, project_root=None, device="cpu", batch_size=128, progress=None):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Inference batch size must be positive")
    context = preflight(plan_path, project_root, verify_models=True)
    output = _output_path(output_dir, context)
    manifest, packets = _load_frozen(context, output)
    manifest_hash = file_hash(output / "frozen_manifest.json")
    datasets = {batch: {cid: _MergedMarkers(group) for cid, group in clients.items()}
                for batch, clients in packets.items()}
    (output / "observations").mkdir(exist_ok=True)
    receipts = {}
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    settings = {"device": str(device), "batch_size": batch_size, "threads": torch.get_num_threads(),
                "runtime": _runtime(), "cuda_version": torch.version.cuda,
                "accelerator_name": torch.cuda.get_device_name(torch.device(device))
                if str(device).startswith("cuda") else None}
    settings_path = output / "collection_settings.json"
    if settings_path.exists():
        if _json(settings_path) != settings:
            raise ValueError("Collection inference settings changed; resume requires the original settings")
    else:
        if any((output / "observations").glob("*.npz")):
            raise ValueError("Observation shards exist without frozen inference settings")
        _write_json(settings_path, settings)
    settings_hash = file_hash(settings_path)
    for binding in context["plan"]["models"]:
        key, model = _model_key(binding), None
        for batch in range(3):
            path = output / "observations" / f"{key}_batch{batch}.npz"
            receipt_path = path.with_suffix(".json")
            expected = _shard_expected(manifest_hash, binding, batch, manifest["inputs_sha256"], settings_hash)
            if path.exists() and receipt_path.exists():
                _, receipt = _read_shard(path, receipt_path, expected, datasets[batch], context)
                receipts[path.relative_to(output).as_posix()] = receipt["observation_sha256"]
                if progress:
                    progress(f"Verified existing shard: {key}, batch {batch + 1}/3")
                continue
            original_receipt = _json(receipt_path) if receipt_path.exists() else None
            if original_receipt is not None and any(original_receipt.get(k) != v for k, v in expected.items()):
                raise ValueError("Missing observation receipt does not match frozen inputs")
            if model is None:
                model = build_model(context["dataset"], 10)
                set_model_parameters(model, _load_parameters(context, binding))
            obs = {cid: observe_markers(model, ds, device, batch_size) for cid, ds in datasets[batch].items()}
            _validate_shard(obs, datasets[batch], context)
            if parameter_hash(get_model_parameters(model)) != binding["parameters_sha256_recorded"]:
                raise ValueError("Inference mutated model parameters or buffers")
            if path.exists():
                # An uncommitted complete NPZ from a crash is not silently trusted.
                existing = _read_npz(path)
                if any(not np.array_equal(existing[cid][field], obs[cid][field])
                       for cid in obs for field in obs[cid]):
                    raise ValueError("Uncommitted shard differs from identical-input replay")
            else:
                pending = path.with_name(path.stem + ".pending.npz")
                save_observations(obs, pending)
                if original_receipt is not None and file_hash(pending) != original_receipt["observation_sha256"]:
                    raise ValueError("Identical-input replay differs from the committed observation bytes")
                os.replace(pending, path)
            receipt = dict(expected, observation_sha256=file_hash(path),
                           model_unchanged_after_inference=True, inference_device=str(device),
                           inference_batch_size=batch_size)
            if original_receipt is not None:
                if receipt != original_receipt:
                    raise ValueError("Identical-input replay differs from the original observation receipt")
            else:
                _write_json(receipt_path, receipt)
            receipts[path.relative_to(output).as_posix()] = receipt["observation_sha256"]
            if progress:
                progress(f"Collected: {key}, batch {batch + 1}/3")
        del model
    _assert_sources_unchanged(context)
    if _code_hashes() != manifest["code_hashes"] or file_hash(output / "frozen_manifest.json") != manifest_hash:
        raise ValueError("Frozen code or manifest changed during collection")
    complete = {"schema_version": SCHEMA, "status": "complete_all_batches_before_analysis",
                "formal_audit": False, "frozen_manifest_sha256": manifest_hash,
                "collection_settings_sha256": settings_hash,
                "shards": receipts, "shard_count": 27, "sample_model_evaluations": 54000}
    target = output / "collection_complete.json"
    if target.exists():
        if _json(target) != complete:
            raise ValueError("Collection completion record differs")
    else:
        _write_json(target, complete)
    return complete


def load_collection(plan_path, output_dir, project_root=None):
    """Load complete hash-bound observations. Analysis does not reload checkpoints."""
    context = preflight(plan_path, project_root, verify_models=False)
    output = _output_path(output_dir, context)
    manifest, packets = _load_frozen(context, output)
    manifest_hash = file_hash(output / "frozen_manifest.json")
    complete = _json(output / "collection_complete.json")
    settings_hash = file_hash(output / "collection_settings.json")
    if (complete.get("status") != "complete_all_batches_before_analysis" or
            complete.get("frozen_manifest_sha256") != manifest_hash or complete.get("shard_count") != 27 or
            complete.get("collection_settings_sha256") != settings_hash):
        raise ValueError("All three batches and nine models must be complete before analysis")
    observations, found = {}, {}
    for batch in range(3):
        datasets = {cid: _MergedMarkers(group) for cid, group in packets[batch].items()}
        observations[batch] = {}
        for binding in context["plan"]["models"]:
            key = _model_key(binding)
            path = output / "observations" / f"{key}_batch{batch}.npz"
            expected = _shard_expected(manifest_hash, binding, batch, manifest["inputs_sha256"], settings_hash)
            obs, receipt = _read_shard(path, path.with_suffix(".json"), expected, datasets, context)
            observations[batch][key] = obs
            found[path.relative_to(output).as_posix()] = receipt["observation_sha256"]
    if complete.get("shards") != found:
        raise ValueError("Observation inventory differs from completion record")
    return context, manifest, observations


def _analysis_tables(report):
    """Flatten the complete report without changing any statistical decisions."""
    tables = {name: [] for name in (
        "summary", "pooled_margin_tests", "anchor_geometry_tests", "accuracy_tests",
        "stratified_margin_tests", "stratified_margin_classes", "client_class_metrics",
        "class_variance")}
    tables["per_batch_descriptive"] = report["per_batch_descriptive"]
    for request in report["requests"]:
        identity = {key: request[key] for key in ("model_key", "training_seed", "method", "full_model_key")}
        primary = request["primary"]
        task = request["task_metrics"] or {}
        variants = request["ablations"]["ablation_variants"]
        accuracy = {row["retained_rule"]: row["status"] for row in variants
                    if row["method"] == "marker_accuracy"}
        tables["summary"].append({**identity, **primary,
            "secondary_equivalence_marker_status": request["secondary_equivalence"]["marker_status"],
            "pooled_noninferiority_status": request["v2"]["margin_variants"]["noninferiority"]["status"],
            "pooled_equivalence_status": request["v2"]["margin_variants"]["equivalence"]["status"],
            "empirical_accuracy_noninferiority_status": accuracy["noninferiority"],
            "empirical_accuracy_equivalence_status": accuracy["equivalence"],
            "task_status": task.get("status", "insufficient"),
            "task_retain_acc_full": task.get("retain_acc_full"),
            "task_retain_acc_after": task.get("retain_acc_after"),
            "task_retain_accuracy_drop": task.get("retain_accuracy_drop"),
            "task_max_retain_accuracy_drop": task.get("max_retain_accuracy_drop"),
            "task_metrics_recomputed": False})
        sources = {
            "pooled_margin_tests": request["v2"]["margin_tests"],
            "anchor_geometry_tests": request["ablations"]["geometry_tests"],
            "accuracy_tests": request["ablations"]["accuracy_tests"],
            "stratified_margin_tests": request["stratified_margin"]["client_tests"],
            "stratified_margin_classes": request["stratified_margin"]["class_metrics"],
            "client_class_metrics": request["ablations"]["client_class_metrics"],
            "class_variance": request["ablations"]["class_variance"],
        }
        for name, rows in sources.items():
            tables[name].extend({**identity, **row} for row in rows)
    return tables


def _publish_analysis_file(path, content):
    """Resume an interrupted publication only if every existing byte agrees."""
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Existing analysis artifact differs: {path.name}")
        return
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def analyze_collection(plan_path, output_dir, project_root=None):
    from fulbase.verification.independent_marker_analysis import analyze_independent_observations
    context, manifest, observations = load_collection(plan_path, output_dir, project_root)
    output = Path(output_dir).resolve()
    target = output / "analysis"
    completion = target / "analysis_manifest.json"
    source = {"frozen_manifest_sha256": file_hash(output / "frozen_manifest.json"),
              "collection_complete_sha256": file_hash(output / "collection_complete.json")}
    if completion.exists():
        saved = _json(completion)
        if (saved.get("status") != "complete" or saved.get("source") != source
                or saved.get("schema_version") != SCHEMA):
            raise ValueError("Completed analysis binding changed")
        if set(saved.get("files", {})) != {"report.json", *(
                name + ".csv" for name in _analysis_tables(_json(target / "report.json")))}:
            raise ValueError("Completed analysis artifact inventory changed")
        for name, digest in saved["files"].items():
            if file_hash(_inside(target, name)) != digest:
                raise ValueError(f"Completed analysis artifact changed: {name}")
        _assert_sources_unchanged(context)
        return _json(target / "report.json")
    report = analyze_independent_observations(observations, context["plan"], context["marker_config"],
                                            context["audit_protocol"], task_metrics=context["task_metrics"])
    report["source"] = {**source,
                        "checkpoint_bytes_reverified_at_collection": True,
                        "checkpoint_bytes_reverified_at_analysis": False,
                        "freshness": manifest["freshness"]}
    _assert_sources_unchanged(context)
    if _code_hashes() != manifest["code_hashes"]:
        raise ValueError("Code changed during analysis")
    target.mkdir(exist_ok=True)
    artifacts = {"report.json": (json.dumps(report, indent=2, ensure_ascii=False,
                                           allow_nan=False) + "\n").encode("utf-8")}
    tables = _analysis_tables(report)
    for name, rows in tables.items():
        stream = io.StringIO(newline="")
        fields = sorted(set().union(*(row.keys() for row in rows)))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        artifacts[name + ".csv"] = stream.getvalue().encode("utf-8")
    for name, content in artifacts.items():
        _publish_analysis_file(target / name, content)
    _assert_sources_unchanged(context)
    if (_code_hashes() != manifest["code_hashes"]
            or file_hash(output / "frozen_manifest.json") != source["frozen_manifest_sha256"]
            or file_hash(output / "collection_complete.json") != source["collection_complete_sha256"]):
        raise ValueError("Frozen analysis sources changed during publication")
    _write_json(completion, {"schema_version": SCHEMA, "status": "complete",
                            "formal_audit": False, "source": source,
                            "files": {name: file_hash(target / name) for name in artifacts},
                            "table_rows": {name: len(rows) for name, rows in tables.items()}})
    return report
