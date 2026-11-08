"""Read-only development diagnostics over saved calibration observations.

No checkpoint, raw dataset, model inference, training, or v1 report write occurs
here. Recorded checkpoint hashes bind metadata; model bytes are not reverified.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import csv
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from fulbase.verification.audit import AuditConfig
from fulbase.verification.markers import MarkerConfig
from fulbase.verification.protocol_v2 import (
    V2DiagnosticConfig, diagnose_observations, validate_observations,
)
from fulbase.verification.protocol_v2_ablation import V2AblationConfig, diagnose_ablations
from fulbase.verification.protocol_v2_stratified import diagnose_stratified_margins


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_path(root, relative):
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError(f"Source artifact outside run directory: {relative}")
    if not path.is_file():
        raise FileNotFoundError(f"Missing source artifact: {path}")
    return path


def _read_json(root, relative, hashes):
    path = _source_path(root, relative)
    raw = path.read_bytes()
    hashes[Path(relative).as_posix()] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw.decode("utf-8"))


def _read_observations(root, relative, hashes):
    path = _source_path(root, relative)
    before_hash = _sha256(path)
    observations = {}
    with np.load(path, allow_pickle=False) as arrays:
        for name in arrays.files:
            match = re.fullmatch(r"c(\d+)_(\w+)", name)
            if match is None:
                raise ValueError(f"Unexpected observation key in {relative}: {name}")
            cid, key = int(match.group(1)), match.group(2)
            observations.setdefault(cid, {})[key] = arrays[name].copy()
    if _sha256(path) != before_hash:
        raise ValueError(f"Source observations changed while reading: {relative}")
    hashes[Path(relative).as_posix()] = before_hash
    for cid, obs in observations.items():
        for key in ("cid", "role", "valid"):
            if key not in obs or obs[key].shape != ():
                raise ValueError(f"Invalid observation scalar {key}, client {cid}")
            obs[key] = obs[key].item()
        if obs["cid"] != cid or obs["role"] != "calibration":
            raise ValueError("Only paired calibration observations are supported; audit inputs are excluded")
    return observations


def _same_observations(a, b):
    return (set(a) == set(b) and all(
        set(a[cid]) == set(b[cid]) and all(
            np.array_equal(a[cid][key], b[cid][key]) for key in a[cid]
        ) for cid in a))


def _manifest_rows(rows, role):
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Empty or invalid {role} marker manifest")
    parsed = {}
    for row in rows:
        if row.get("role") != role:
            raise ValueError(f"Wrong role in {role} marker manifest")
        marker_id = row["marker_id"]
        if marker_id in parsed:
            raise ValueError(f"Duplicate marker ID in {role} manifest")
        parsed[marker_id] = (row["cid"], row["family_id"], row["label"])
    return parsed


def _observation_rows(observations):
    rows = {}
    for cid, obs in observations.items():
        for marker_id, family_id, label in zip(obs["ids"], obs["families"], obs["labels"]):
            marker_id = str(marker_id)
            if marker_id in rows:
                raise ValueError("Duplicate calibration marker ID")
            rows[marker_id] = (cid, str(family_id), int(label))
    return rows


def _validate_task_metrics(metrics):
    if not isinstance(metrics, dict):
        raise ValueError("task_metrics must be an object")
    for key in ("all_acc", "retain_acc", "forget_acc"):
        if key in metrics and (not isinstance(metrics[key], (float, int)) or
                               not math.isfinite(metrics[key]) or not 0 <= metrics[key] <= 1):
            raise ValueError(f"Invalid recorded task accuracy: {key}")
    keys = ("all_acc", "retain_acc", "forget_acc", "retain_num_examples", "forget_num_examples")
    if all(key in metrics for key in keys):
        nr, nf = metrics["retain_num_examples"], metrics["forget_num_examples"]
        if type(nr) is not int or type(nf) is not int or min(nr, nf) < 1:
            raise ValueError("Recorded task counts must be positive integers")
        expected = (nr * metrics["retain_acc"] + nf * metrics["forget_acc"]) / (nr + nf)
        if not math.isclose(metrics["all_acc"], expected, abs_tol=1e-10, rel_tol=0):
            raise ValueError("Recorded task accuracies violate the weighted identity")


def _task_diagnostic(full_entry, entry, report, protocol):
    before, after = full_entry.get("task_metrics", {}), entry.get("task_metrics", {})
    _validate_task_metrics(before)
    _validate_task_metrics(after)
    result = {"source": "recorded_benchmark_metrics_not_re_evaluated",
              "before": before, "after": after,
              "max_retain_accuracy_drop": protocol["max_retain_accuracy_drop"],
              "retain_acc_full": before.get("retain_acc"),
              "retain_acc_after": after.get("retain_acc"),
              "retain_accuracy_drop": None, "status": "insufficient"}
    if "retain_acc" not in before or "retain_acc" not in after:
        result["reason"] = "Recorded task metrics unavailable"
        return result
    drop = before["retain_acc"] - after["retain_acc"]
    status = "pass" if drop <= protocol["max_retain_accuracy_drop"] else "fail"
    legacy = [c for c in report.get("conditions", []) if c.get("condition") == "retain_task"]
    if len(legacy) != 1:
        raise ValueError("Source report must contain one retain_task condition")
    condition = legacy[0]
    if (condition.get("status") != status or
            not math.isclose(condition.get("accuracy_drop", math.nan), drop, abs_tol=1e-12, rel_tol=0) or
            condition.get("bound") != protocol["max_retain_accuracy_drop"]):
        raise ValueError("Source report task condition differs from recorded benchmark metrics")
    result.update(retain_accuracy_drop=drop, status=status)
    return result


def _validate_entry(entry, stage, fingerprint):
    if entry.get("stage") != stage or entry.get("role") != "calibration":
        raise ValueError("Only manifest-bound calibration stages are supported")
    if entry.get("marker_fingerprint") != fingerprint:
        raise ValueError("Stage marker fingerprint differs from source configuration")
    for key in ("checkpoint_sha256", "parameters_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", str(entry.get(key, ""))) is None:
            raise ValueError(f"Missing or invalid recorded {key}")


def load_source_runs(run_dirs, stages=None):
    """Validate explicit source runs without loading checkpoints or writing files."""
    roots = [Path(path).resolve() for path in run_dirs]
    if not roots or len(set(roots)) != len(roots):
        raise ValueError("Supply distinct explicit run directories")
    requested = None if stages is None else list(stages)
    if requested is not None and (not requested or len(set(requested)) != len(requested)):
        raise ValueError("Choose distinct nonempty stage names")
    sources = []
    for root in roots:
        hashes = {}
        cfg = _read_json(root, "config.json", hashes)
        marker_values = _read_json(root, "verification/marker_config.json", hashes)
        markers = MarkerConfig(**marker_values)
        if not markers.enabled:
            raise ValueError("Anchor diagnostics require enabled source markers")
        protocol = _read_json(root, "verification/audit_config.json", hashes)
        # Validate a source protocol without changing its existing interpretation.
        if asdict(AuditConfig(**protocol)) != protocol:
            raise ValueError("Source audit protocol must explicitly record every v1 field")
        if protocol["protocol_locked"]:
            raise ValueError("Protocol-v2 diagnostics are development-only, not locked audits")
        manifest = _read_json(root, "verification/checkpoint_manifest.json", hashes)
        partition_path = _source_path(root, "data_partition.json")
        hashes["data_partition.json"] = _sha256(partition_path)
        if hashes["data_partition.json"] != manifest.get("partition_sha256"):
            raise ValueError("Source partition hash mismatch")
        fingerprint = markers.fingerprint()
        if manifest.get("marker_fingerprint") != fingerprint:
            raise ValueError("Source marker fingerprint mismatch")
        for optional in ("run_status.json", "verification/training_result.json", "diagnostic_config.json"):
            if (root / optional).is_file():
                record = _read_json(root, optional, hashes)
                if optional == "run_status.json" and record.get("status") not in {
                        "development_run_complete", "pga_diagnostic_complete"}:
                    raise ValueError("Source run is incomplete")
                if optional == "verification/training_result.json" and record.get("final_parameters_sha256") != \
                        manifest["stages"]["full_after_fl"]["parameters_sha256"]:
                    raise ValueError("Source Full differs from recorded training result")
        forget = cfg["forget_clients"]
        forget = [int(x.strip()) for x in forget.split(",")] if isinstance(forget, str) else list(forget)
        expected_clients = set(range(cfg["num_clients"]))
        if (not forget or len(set(forget)) != len(forget) or not set(forget) < expected_clients):
            raise ValueError("Invalid configured forget set")
        entries = manifest.get("stages", {})
        if "full_after_fl" not in entries:
            raise ValueError("Source manifest needs full_after_fl")
        selected = list(entries) if requested is None else requested
        missing = set(selected) - set(entries)
        if missing:
            raise ValueError(f"Stages absent in {root.name}: {sorted(missing)}")
        if any(re.fullmatch(r"[A-Za-z0-9_.-]+", name) is None or name in {".", ".."} for name in selected):
            raise ValueError("Invalid stage directory name")
        baseline = None
        stage_records = []
        # The shared Full is checked even when it is not selected for output.
        for stage in dict.fromkeys(["full_after_fl", *selected]):
            entry = entries[stage]
            _validate_entry(entry, stage, fingerprint)
            relative = f"verification/{stage}"
            report = _read_json(root, relative + "/audit_report.json", hashes)
            metadata = report.get("metadata", {})
            for key in ("stage", "role", "checkpoint_sha256", "parameters_sha256", "marker_fingerprint"):
                if metadata.get(key) != entry.get(key):
                    raise ValueError(f"Report/manifest mismatch for {stage}: {key}")
            if report.get("protocol") != protocol or sorted(report.get("forget_client_ids", [])) != sorted(forget):
                raise ValueError(f"Report protocol/forget set mismatch: {stage}")
            before = _read_observations(root, relative + "/full_observations.npz", hashes)
            after = _read_observations(root, relative + "/after_observations.npz", hashes)
            if set(before) != expected_clients or set(after) != expected_clients:
                raise ValueError("Observation clients differ from source configuration")
            validate_observations(before, after, forget, markers, protocol)
            if any(obs["logits"].shape[1] != manifest.get("num_classes")
                   for observations in (before, after) for obs in observations.values()):
                raise ValueError("Observation class count differs from checkpoint manifest")
            if stage == "full_after_fl":
                if not _same_observations(before, after):
                    raise ValueError("Full stage has differing before/after observations")
                baseline = before
            elif not _same_observations(before, baseline):
                raise ValueError("Candidate Full observations differ from shared Full baseline")
            task = _task_diagnostic(entries["full_after_fl"], entry, report, protocol)
            if stage in selected:
                stage_records.append({"name": stage, "entry": entry, "report": report,
                                      "before": before, "after": after, "task": task})
        calibration_rows = _observation_rows(baseline)
        training = _manifest_rows(_read_json(root, "verification/training_marker_manifest.json", hashes), "train")
        if set(training) & set(calibration_rows):
            raise ValueError("Training and calibration marker IDs overlap")
        identity_source = "full_after_fl_observations"
        calibration_path = "verification/calibration_marker_manifest.json"
        if (root / calibration_path).is_file():
            declared = _manifest_rows(_read_json(root, calibration_path, hashes), "calibration")
            if declared != calibration_rows:
                raise ValueError("Calibration marker manifest differs from saved observations")
            identity_source = "calibration_marker_manifest_and_full_after_fl_observations"
        sources.append({"root": root, "cfg": cfg, "marker_config": markers,
                        "audit_protocol": protocol, "manifest": manifest,
                        "forget": forget, "stages": stage_records, "source_hashes": hashes,
                        "calibration_identity_source": identity_source,
                        "calibration_identity_sha256": hashlib.sha256(json.dumps(
                            calibration_rows, sort_keys=True, separators=(",", ":")
                        ).encode("utf-8")).hexdigest()})
    return sources


def _clean(value):
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(_clean(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _write_csv(path, rows):
    rows = [_clean(row) for row in rows]
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def _source_code_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = ["fulbase/verification/protocol_v2.py", "fulbase/verification/protocol_v2_io.py",
             "fulbase/verification/protocol_v2_ablation.py",
             "fulbase/verification/protocol_v2_stratified.py",
             "fulbase/verification/statistics.py", "fulbase/verification/audit.py",
             "fulbase/verification/markers.py", "scripts/diagnose_verification_v2.py"]
    return {name: _sha256(root / name) for name in paths}


def _load_ablation_config(path):
    """Explicit development bounds; never infer them from the input outcomes."""
    provenance = {"configured": False, "path": None, "sha256": None, "values": None,
                  "protocol_locked": False, "purpose": "development_comparison"}
    if path is None:
        return None, provenance
    resolved = Path(path).resolve()
    raw = resolved.read_bytes()
    values = json.loads(raw.decode("utf-8"))
    if not isinstance(values, dict):
        raise ValueError("Ablation configuration must be a JSON object")
    config = V2AblationConfig(**values)
    provenance.update(configured=True, path=str(resolved),
                      sha256=hashlib.sha256(raw).hexdigest(), values=asdict(config))
    return config, provenance


def _assert_ablation_config_unchanged(provenance):
    if provenance["configured"] and _sha256(provenance["path"]) != provenance["sha256"]:
        raise ValueError("Ablation configuration changed during diagnostics")


def _assert_sources_unchanged(sources):
    for source in sources:
        for relative, expected in source["source_hashes"].items():
            if _sha256(_source_path(source["root"], relative)) != expected:
                raise ValueError(f"Source artifact changed during diagnostics: {source['root'] / relative}")


def _validate_legacy_equivalence(report, legacy):
    """Require recomputed v1 semantics to agree with the immutable source report."""
    computed = {row["cid"]: row for row in report["margin_tests"] if row["variant"] == "equivalence"}
    recorded_rows = [row for row in legacy["conditions"] if row.get("condition", "").endswith("_margin")]
    recorded = {row["cid"]: row for row in recorded_rows}
    if len(recorded) != len(recorded_rows) or set(computed) != set(recorded):
        raise ValueError("Legacy margin conditions do not match diagnostic clients")
    numeric = ("mean", "ci_low", "ci_high", "p_support", "p_violation", "p_support_holm", "p_violation_holm")
    for cid, new in computed.items():
        old = recorded[cid]
        for key in ("condition", "kind", "n_families", "status"):
            if new.get(key) != old.get(key):
                raise ValueError(f"Legacy equivalence mismatch: client {cid}, {key}")
        for key in numeric:
            a, b = new.get(key), old.get(key)
            if a is None and b is None:
                continue
            if a is None or b is None or not math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-12):
                raise ValueError(f"Legacy equivalence mismatch: client {cid}, {key}")
    if report["margin_variants"]["equivalence"]["status"] != legacy["exploratory_decision_variants"]["margin_only"]:
        raise ValueError("Legacy combined margin status differs from recomputed equivalence")


def _summary_rows(source, stage, report, output_subdir):
    legacy = stage["report"]
    task = stage["task"]
    metrics = {row["cid"]: row for row in report["marker_metrics"]}
    geometry = {row["cid"]: row for row in report["geometry"]}
    full_geometry = {row["cid"]: row for row in report["full_geometry"]}
    rows = []
    for cid in source["forget"]:
        row = {"run": source["root"].name, "source_run_dir": str(source["root"]),
               "seed": source["cfg"]["seed"], "stage": stage["name"], "target_cid": cid,
               "method": stage["entry"].get("method"), "behavior": stage["entry"].get("behavior"),
               "data_origin": source["manifest"].get("data_origin", "unspecified"),
               "source_full_parameters_sha256": source["manifest"]["stages"]["full_after_fl"]["parameters_sha256"],
               "calibration_identity_sha256": source["calibration_identity_sha256"],
               "full_observations_sha256": source["source_hashes"]["verification/full_after_fl/full_observations.npz"],
               "checkpoint_sha256_recorded": stage["entry"]["checkpoint_sha256"],
               "formal_audit": False, "diagnostic_status": "diagnostic_only", "output_subdir": output_subdir,
               "retain_acc_full": task["retain_acc_full"], "retain_acc_after": task["retain_acc_after"],
               "retain_accuracy_drop": task["retain_accuracy_drop"], "task_status": task["status"],
               "legacy_status": legacy.get("status"),
               "legacy_exploratory_status": legacy.get("exploratory_status")}
        for key, value in metrics[cid].items():
            if key not in {"cid", "client_group"}:
                row["target_" + key] = value
        for key, value in geometry[cid].items():
            if key not in {"cid", "client_group"}:
                row["target_" + key] = value
        row["full_target_anchor_advantage"] = full_geometry[cid]["anchor_advantage"]
        for name, value in legacy.get("exploratory_decision_variants", {}).items():
            row["legacy_" + name] = value
        for variant, info in report["margin_variants"].items():
            row[variant + "_margin_status"] = info["status"]
            relevant = [item for item in report["margin_tests"] if item["variant"] == variant]
            target = next(item for item in relevant if item["cid"] == cid)
            row[variant + "_target_margin_status"] = target["status"]
            row[variant + "_target_margin_p_support_holm"] = target.get("p_support_holm")
            for status in ("pass", "fail", "insufficient"):
                row[variant + "_retain_" + status + "_count"] = sum(
                    item["status"] == status for item in relevant if item["cid"] not in source["forget"])
        rows.append(row)
    return rows


def run_protocol_v2_diagnostic(run_dirs, *, stages=None, out_dir="diagnostics/verification_v2",
                               run_name="protocol_v2", bootstrap_samples=10000,
                               bootstrap_seed=2718, confidence=.95, check_only=False,
                               ablation_config=None, stratified_margin=False):
    """Diagnose all selected calibration stages; source runs remain read-only."""
    if type(stratified_margin) is not bool:
        raise ValueError("stratified_margin must be a boolean")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_name) is None:
        raise ValueError("run_name must be a simple 1-80 character directory suffix")
    options = V2DiagnosticConfig(bootstrap_samples=bootstrap_samples,
                                 bootstrap_seed=bootstrap_seed, confidence=confidence)
    ablation_options, ablation_provenance = _load_ablation_config(ablation_config)
    sources = load_source_runs(run_dirs, stages)
    destination = Path(out_dir).resolve()
    if any(destination == source["root"] or source["root"] in destination.parents for source in sources):
        raise ValueError("Output directory must be outside every source run")
    counts = {"num_runs": len(sources), "num_stages": sum(len(source["stages"]) for source in sources)}
    if check_only:
        _assert_sources_unchanged(sources)
        _assert_ablation_config_unchanged(ablation_provenance)
        return {"status": "checked", "output_dir": None, "formal_audit": False,
                "ablation_config": ablation_provenance, "stratified_margin": stratified_margin, **counts}
    code_hashes = _source_code_hashes()
    computed = []
    stratified_reports = [] if stratified_margin else None
    for source in sources:
        for stage in source["stages"]:
            print(f"[protocol-v2] {source['root'].name}: {stage['name']}", flush=True)
            report = diagnose_observations(stage["before"], stage["after"], source["forget"],
                                           source["marker_config"], source["audit_protocol"], options)
            _validate_legacy_equivalence(report, stage["report"])
            report.update(diagnose_ablations(
                stage["before"], stage["after"], source["forget"], source["marker_config"],
                source["audit_protocol"], report, options, ablation_options))
            report.update(task=stage["task"],
                          ablation_config=ablation_provenance,
                          legacy={"status": stage["report"].get("status"),
                                  "exploratory_status": stage["report"].get("exploratory_status"),
                                  "exploratory_decision_variants": stage["report"].get("exploratory_decision_variants"),
                                  "conditions": stage["report"].get("conditions")},
                          source={"run_dir": str(source["root"]), "stage": stage["name"],
                                  "seed": source["cfg"]["seed"], "role": "calibration",
                                  "checkpoint_sha256_recorded": stage["entry"]["checkpoint_sha256"],
                                  "checkpoint_bytes_reverified": False,
                                  "legacy_equivalence_reproduced": True,
                                  "calibration_identity_sha256": source["calibration_identity_sha256"],
                                  "full_parameters_sha256_recorded": source["manifest"]["stages"]["full_after_fl"]["parameters_sha256"]})
            if stratified_margin:
                # Keep the original report's fields/serialization unchanged.
                # New inference owns its RNG and never changes the pooled tests.
                stratified = diagnose_stratified_margins(
                    stage["before"], stage["after"], source["forget"], source["marker_config"],
                    source["audit_protocol"], report, options)
                stratified.update(source=dict(report["source"]), task=dict(stage["task"]))
                stratified_reports.append(stratified)
            computed.append((source, stage, report))
    _assert_sources_unchanged(sources)
    _assert_ablation_config_unchanged(ablation_provenance)
    if _source_code_hashes() != code_hashes:
        raise ValueError("Diagnostic source code changed during computation")
    output = destination / (datetime.now().strftime("%Y%m%d-%H%M%S-%f") + "_" + run_name)
    output.mkdir(parents=True, exist_ok=False)
    status_path = output / "run_status.json"
    _write_json(status_path, {"status": "writing_diagnostic_outputs", "formal_audit": False,
                              "ablation_config": ablation_provenance, "stratified_margin": stratified_margin})
    try:
        _write_results(output, sources, computed, options, counts, code_hashes,
                       ablation_provenance, stratified_reports)
    except Exception as exc:
        status_path.write_text(json.dumps({"status": "diagnostic_failed", "formal_audit": False,
                                          "ablation_config": ablation_provenance,
                                          "stratified_margin": stratified_margin,
                                          "error_type": type(exc).__name__, "error": str(exc)},
                                         ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    status_path.write_text(json.dumps({"status": "diagnostic_complete", "formal_audit": False,
                                      "ablation_config": ablation_provenance,
                                      "stratified_margin": stratified_margin, **counts},
                                     indent=2) + "\n", encoding="utf-8")
    return {"status": "diagnostic_complete", "output_dir": str(output), "formal_audit": False,
            "ablation_config": ablation_provenance, "stratified_margin": stratified_margin, **counts}


def _comparison_context(source, stage, output_subdir):
    """Keep model/request identity and utility separate from ablation decisions."""
    return {"run": source["root"].name, "source_run_dir": str(source["root"]),
            "seed": source["cfg"]["seed"], "stage": stage["name"],
            "model_method": stage["entry"].get("method"),
            "behavior": stage["entry"].get("behavior"),
            "data_origin": source["manifest"].get("data_origin", "unspecified"),
            "source_full_parameters_sha256": source["manifest"]["stages"]["full_after_fl"]["parameters_sha256"],
            "calibration_identity_sha256": source["calibration_identity_sha256"],
            "target_client_ids": source["forget"], "formal_audit": False,
            "diagnostic_status": "diagnostic_only", "output_subdir": output_subdir,
            "retain_acc_full": stage["task"]["retain_acc_full"],
            "retain_acc_after": stage["task"]["retain_acc_after"],
            "retain_accuracy_drop": stage["task"]["retain_accuracy_drop"],
            "task_status": stage["task"]["status"]}


def _write_results(output, sources, computed, options, counts, code_hashes, ablation_provenance,
                   stratified_reports=None):
    summaries, stage_index = [], []
    ablation_rows, class_rows, variance_rows = [], [], []
    stratified_tests, stratified_classes, stratified_summary = [], [], []
    if stratified_reports is not None and len(stratified_reports) != len(computed):
        raise ValueError("Stratified report count differs from selected stages")
    stratified_protocols = [report["protocol"] for report in (stratified_reports or [])]
    common_stratified_protocol = (stratified_protocols[0] if stratified_protocols and all(
        protocol == stratified_protocols[0] for protocol in stratified_protocols) else None)
    for index, (source, stage, report) in enumerate(computed, 1):
        subdir = f"stage_{index:04d}"
        folder = output / subdir
        folder.mkdir()
        _write_json(folder / "diagnostic_report.json", report)
        for filename, key in (("anchor_geometry.csv", "geometry"), ("full_geometry.csv", "full_geometry"),
                              ("margin_tests.csv", "margin_tests"), ("marker_summary.csv", "marker_metrics"),
                              ("accuracy_tests.csv", "accuracy_tests"),
                              ("ablation_geometry_tests.csv", "geometry_tests"),
                              ("ablation_comparison.csv", "ablation_variants"),
                              ("client_class_metrics.csv", "client_class_metrics"),
                              ("class_variance.csv", "class_variance")):
            _write_csv(folder / filename, report[key])
        _write_csv(folder / "legacy_geometry.csv", [item for item in stage["report"]["conditions"]
                                                     if "geometry" in item.get("condition", "")])
        summaries.extend(_summary_rows(source, stage, report, subdir))
        context = _comparison_context(source, stage, subdir)
        if stratified_reports is not None:
            stratified = stratified_reports[index-1]
            _write_json(folder / "stratified_margin_report.json", stratified)
            for filename, key in (("stratified_margin_tests.csv", "client_tests"),
                                  ("stratified_margin_classes.csv", "class_metrics"),
                                  ("stratified_margin_comparison.csv", "summary")):
                _write_csv(folder / filename, stratified[key])
            stratified_tests.extend({**context, **row} for row in stratified["client_tests"])
            stratified_classes.extend({**context, **row} for row in stratified["class_metrics"])
            stratified_summary.extend({**context, **row} for row in stratified["summary"])
        ablation_rows.extend({**context, **row} for row in report["ablation_variants"])
        class_rows.extend({**context, **row} for row in report["client_class_metrics"])
        variance_rows.extend({**context, **row} for row in report["class_variance"])
        stage_index.append({"run_dir": str(source["root"]), "seed": source["cfg"]["seed"],
                            "stage": stage["name"], "output_subdir": subdir})
    _write_csv(output / "summary.csv", summaries)
    _write_csv(output / "ablation_summary.csv", ablation_rows)
    _write_csv(output / "client_class_metrics.csv", class_rows)
    _write_csv(output / "class_variance.csv", variance_rows)
    if stratified_reports is not None:
        _write_csv(output / "stratified_margin_tests.csv", stratified_tests)
        _write_csv(output / "stratified_margin_classes.csv", stratified_classes)
        _write_csv(output / "stratified_margin_summary.csv", stratified_summary)
    _assert_sources_unchanged(sources)
    _assert_ablation_config_unchanged(ablation_provenance)
    if _source_code_hashes() != code_hashes:
        raise ValueError("Diagnostic source code changed while writing results")
    _write_json(output / "diagnostic_manifest.json", {
        "schema_version": ("marker-protocol-v2-diagnostic-3" if stratified_reports is not None
                           else "marker-protocol-v2-diagnostic-2"), "status": "diagnostic_complete",
        "formal_audit": False, "protocol_locked": False, "options": asdict(options),
        "ablation_config": ablation_provenance,
        "stratified_margin": {"enabled": stratified_reports is not None,
                              "protocol": common_stratified_protocol,
                              "protocol_scope": "common_if_equal_else_per_stage_reports",
                              "output_counts": {"client_test_rows": len(stratified_tests),
                                                "class_rows": len(stratified_classes),
                                                "comparison_rows": len(stratified_summary)},
                              "pooled_reports_replaced": False},
        "ablation_output_counts": {"comparison_rows": len(ablation_rows),
                                   "client_class_rows": len(class_rows),
                                   "class_variance_rows": len(variance_rows)},
        "source_code_sha256": code_hashes, "stages": stage_index, **counts,
        "source_runs": [{"run_dir": str(source["root"]), "seed": source["cfg"]["seed"],
                         "source_file_sha256": source["source_hashes"],
                         "marker_config": asdict(source["marker_config"]),
                         "source_audit_protocol": source["audit_protocol"],
                         "calibration_identity_source": source["calibration_identity_source"],
                         "calibration_identity_sha256": source["calibration_identity_sha256"]}
                        for source in sources],
        "validation": {"source_files_unchanged": True, "checkpoint_bytes_reverified": False,
                       "training_performed": False, "model_inference_performed": False,
                       "legacy_equivalence_reproduced": True,
                       "raw_dataset_read": False, "legacy_reports_overwritten": False},
        "limitations": ["Development calibration only; explicit ablation bounds are not a locked formal protocol",
                        "Geometry intervals are descriptive, not calibrated decision thresholds",
                        "Ablation geometry uses separately recorded approximate simultaneous bootstrap intervals",
                        "Ablation comparisons do not include task utility in their acceptance decision",
                        "Client-class diagnostics are descriptive and never added as per-class acceptance gates",
                        "Margin variants are separate exploratory comparisons, not a pooled multiple-testing claim",
                        "Opt-in fixed-class margin bootstrap is exploratory; its Bonferroni intervals do not replace pooled t/Holm tests",
                        "Stratified margin uses fixed-class weighting; unbalanced class counts can also change the estimand",
                        "Within-checkpoint marker-family uncertainty is not across-training uncertainty",
                        "No across-run pooling or confidence interval; repeated Full hashes are not independent training runs",
                        "Recorded checkpoint hashes do not authenticate observation generation or deployed behavior",
                        "Task metrics are copied from recorded benchmarks, not re-evaluated"]})
