"""Source-bound benchmark/overhead tables; never used to select candidates."""
from __future__ import annotations
import csv
import io
import json
import math
from pathlib import Path

from fulbase.verification.audit import file_hash


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _inside(root, relative):
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError("Table input outside project/run root")
    return path


def _number(value):
    if value is None or value == "":
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _csv_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def write_main_tables(root, bindings, output):
    """Write all 45 bindings and shared auxiliary storage with honest missing times."""
    root, output = Path(root).resolve(), Path(output).resolve()
    entries, runs = {}, {}
    for binding in bindings:
        key = (binding["training_seed"], binding["method"])
        run = _inside(root, binding["run_dir"])
        manifest = _read(run / "verification/checkpoint_manifest.json")
        entry = manifest["stages"][binding["stage"]]
        if entry["parameters_sha256"] != binding["parameters_sha256_recorded"]:
            raise ValueError("Benchmark table model binding differs")
        entries[key] = (binding, entry, manifest["stages"]["full_after_fl"])
        runs[key] = run
    rows, auxiliaries = [], []
    for key in sorted(entries):
        seed, method = key
        binding, entry, source_full = entries[key]
        metrics, before = entry.get("task_metrics", {}), source_full.get("task_metrics", {})
        retrain = entries[(seed, "Retrain")][1].get("task_metrics", {})
        full = entries[(seed, "Full")][1].get("task_metrics", {})
        accuracy = _number(metrics.get("retain_acc"))
        original = _number(before.get("retain_acc"))
        reference = _number(retrain.get("retain_acc"))
        canonical = _number(full.get("retain_acc"))
        drop = original - accuracy if original is not None and accuracy is not None else None
        csv_path = runs[key] / "metrics.csv"
        timing = {}
        if csv_path.is_file():
            with csv_path.open(encoding="utf-8-sig", newline="") as stream:
                matching = [row for row in csv.DictReader(stream) if row.get("stage") == binding["stage"]]
            if len(matching) != 1:
                raise ValueError("Benchmark metrics need exactly one bound stage row")
            timing = matching[0]
        checkpoint = _inside(runs[key], binding["checkpoint"])
        if file_hash(checkpoint) != binding["checkpoint_sha256_recorded"]:
            raise ValueError("Benchmark model bytes differ from binding")
        rows.append({"cohort": binding["cohort"], "training_seed": seed, "method": method,
            "stage": binding["stage"], "run_dir": binding["run_dir"],
            "all_acc": _number(metrics.get("all_acc")), "retain_acc": accuracy,
            "forget_acc": _number(metrics.get("forget_acc")),
            "retain_num_examples": metrics.get("retain_num_examples"),
            "forget_num_examples": metrics.get("forget_num_examples"),
            "source_full_retain_acc": original, "canonical_full_retain_acc": canonical,
            "retrain_retain_acc": reference, "retain_accuracy_drop": drop,
            "retain_change_vs_retrain_pp": 100*(accuracy-reference) if accuracy is not None and reference is not None else None,
            "task_original_3pp_status": "insufficient" if drop is None else "pass" if drop <= .03 else "fail",
            "task_proposed_3_5pp_status": "insufficient" if drop is None else "pass" if drop <= .035 else "fail",
            "full_training_time_sec": _number(timing.get("full_training_time_sec")) if method == "Full" else None,
            "unlearning_time_sec": _number(timing.get("unlearning_time_sec")) if method != "Full" else None,
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": binding["checkpoint_sha256_recorded"],
            "parameters_sha256": binding["parameters_sha256_recorded"],
            "task_metrics_retested_by_markers": False})
        if method == "Full":
            for name in ("history.pt", "models/model_initial.pt"):
                path = runs[key] / name
                auxiliaries.append({"training_seed": seed, "cohort": binding["cohort"],
                    "artifact": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size if path.is_file() else None,
                    "sha256": file_hash(path) if path.is_file() else None,
                    "scope": "shared_source_artifact_count_once_per_seed",
                    "availability": "present" if path.is_file() else "missing"})
    output.mkdir(parents=True, exist_ok=True)
    for name, data in (("benchmark_metrics.csv", rows), ("shared_auxiliary_storage.csv", auxiliaries)):
        content = _csv_bytes(data)
        path = output / name
        if path.exists():
            if path.read_bytes() != content:
                raise ValueError("Existing benchmark table differs; do not overwrite")
        else:
            temporary = path.with_name(path.name + ".pending")
            temporary.write_bytes(content); temporary.replace(path)
    return {"benchmark_rows": len(rows), "shared_auxiliary_rows": len(auxiliaries)}
