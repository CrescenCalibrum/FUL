"""Prespecified RP2 campaign: reuse old sources, train two new seeds, bind all arms.

The blueprint is frozen before new training. No audit observations are read here.
Existing reports retain their original 3pp task gate; the new audit adapter reports
both task gates. A failed candidate is never removed from the planned matrix.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

from fulbase.verification.audit import AuditConfig, file_hash, parameter_hash
from fulbase.verification.experiment import VerificationExperimentConfig, run_verification_experiment
from fulbase.verification.markers import MarkerConfig

SCHEMA = "rp2-main-experiment-campaign-v1"
METHODS = ["Full", "Retrain", "history_subtract", "federaser_lite", "pga",
           "fixed_C_lambda05", "D_margin", "fake_noop", "fake_corrupt"]
CANDIDATE_STAGES = {name: f"{name}_pre_recovery" for name in
                    ("history_subtract", "federaser_lite", "pga", "fake_noop", "fake_corrupt")}
CANDIDATE_STAGES.update(fixed_C_lambda05="fixed_C_lambda05_pre_recovery",
                        D_margin="D_margin_lambda8_pre_recovery")


def _root(project_root=None):
    return Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[2]


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _inside(root, relative):
    path = (root / str(relative).replace("\\", "/")).resolve()
    if root not in path.parents:
        raise ValueError(f"Path must stay inside project: {relative}")
    return path


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def _update(path, value):
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def load_campaign(config_path, project_root=None):
    root = _root(project_root)
    config = _read(config_path)
    if config.get("schema_version") != SCHEMA:
        raise ValueError("Unsupported main experiment blueprint")
    for key, expected in {"development_seeds": [43, 44, 45], "confirmation_seeds": [46, 47],
                          "expected_methods": METHODS, "task_utility_bounds": [.03, .035],
                          "task_primary_bound": .035, "marker_thresholds_changed": False,
                          "recovery": "not_run", "formal_audit": False,
                          "candidate_selection": "all_prespecified_endpoints_including_fail_and_insufficient"}.items():
        if config.get(key) != expected:
            raise ValueError(f"This version requires the prespecified {key}")
    cfg = VerificationExperimentConfig(**_read(_inside(root, config["source_experiment"])))
    markers = MarkerConfig(**_read(_inside(root, config["marker_config"])))
    audit = AuditConfig(**_read(_inside(root, config["audit_config"])))
    expected = {"dataset": "cifar10", "backend": "serial", "num_clients": 5, "num_rounds": 20,
                "local_epochs": 1, "batch_size": 64, "lr": .01, "partition": "dirichlet", "alpha": .5,
                "fraction_fit": 1., "forget_clients": "0", "methods": "retrain_fl",
                "with_recovery": False, "include_fakes": False, "synthetic_smoke": False,
                "pga_steps": 50, "pga_lr": .0001, "pga_project_ratio": .01,
                "pga_layers": "head", "pga_data_scope": "real_and_markers",
                "unlearn_alpha": .5, "federaser_damping": 1., "federaser_calibrate": True,
                "corruption_scale": .5}
    for key, value in expected.items():
        if getattr(cfg, key) != value:
            raise ValueError(f"Source budget differs: {key}")
    if audit.max_retain_accuracy_drop != .03 or audit.protocol_locked:
        raise ValueError("Keep original unlocked 3pp source audit; dual gates belong in new reports")
    if not markers.enabled or markers.families_per_class != 4 or markers.variants_per_family != 2:
        raise ValueError("Preserve original training marker design")
    bindings = config["existing_models"]
    keys = [(row["training_seed"], row["method"]) for row in bindings]
    required = {(seed, method) for seed in (43, 44, 45)
                for method in ("Full", "Retrain", "fixed_C_lambda05", "D_margin")}
    if len(keys) != 12 or set(keys) != required or any(row["cohort"] != "development" for row in bindings):
        raise ValueError("Bind all twelve existing development stages without selection")
    return root, config, cfg, markers, audit


def make_binding(root, run_dir, stage, seed, method, cohort):
    run = _inside(root, run_dir)
    cfg = _read(run / "config.json")
    manifest = _read(run / "verification/checkpoint_manifest.json")
    if cfg["seed"] != seed:
        raise ValueError("Binding training seed mismatch")
    entry = manifest["stages"][stage]
    return {"training_seed": seed, "method": method, "cohort": cohort,
            "run_dir": run.relative_to(root).as_posix(), "stage": stage, "checkpoint": entry["checkpoint"],
            "checkpoint_sha256_recorded": entry["checkpoint_sha256"],
            "parameters_sha256_recorded": entry["parameters_sha256"],
            "marker_fingerprint": manifest["marker_fingerprint"], "partition_sha256": manifest["partition_sha256"]}


def _check_binding(root, binding, verify_models):
    current = make_binding(root, binding["run_dir"], binding["stage"], binding["training_seed"],
                           binding["method"], binding["cohort"])
    if current != binding:
        raise ValueError(f"Existing binding changed: {binding['training_seed']}/{binding['method']}")
    run = _inside(root, binding["run_dir"])
    if file_hash(run / "data_partition.json") != binding["partition_sha256"]:
        raise ValueError("Bound partition changed")
    if _read(run / "run_status.json").get("status") not in {"development_run_complete", "pga_diagnostic_complete"}:
        raise ValueError("Source run is incomplete")
    if MarkerConfig(**_read(run / "verification/marker_config.json")).fingerprint() != binding["marker_fingerprint"]:
        raise ValueError("Bound marker configuration changed")
    path = _inside(run, binding["checkpoint"])
    if verify_models:
        import torch
        if not path.is_file():
            raise FileNotFoundError(f"Missing trusted checkpoint: {path}; no automatic retraining")
        if file_hash(path) != binding["checkpoint_sha256_recorded"]:
            raise ValueError("Bound checkpoint bytes changed")
        params = torch.load(path, map_location="cpu", weights_only=False)
        if parameter_hash(params) != binding["parameters_sha256_recorded"]:
            raise ValueError("Bound checkpoint parameters changed")


def check_campaign(config_path, *, metadata_only=False, project_root=None):
    root, config, cfg, markers, audit = load_campaign(config_path, project_root)
    source_paths, input_paths = [], []
    for binding in config["existing_models"]:
        _check_binding(root, binding, not metadata_only)
        run = _inside(root, binding["run_dir"])
        input_paths.extend(run / name for name in ("config.json", "data_partition.json", "run_status.json",
                           "verification/marker_config.json", "verification/audit_config.json",
                           "verification/checkpoint_manifest.json"))
        input_paths.extend(p for p in (run / "diagnostic_config.json", run / "boundary_config.json",
                                       run / "selection.json") if p.is_file())
        if not metadata_only:
            input_paths.append(_inside(run, binding["checkpoint"]))
        if asdict(MarkerConfig(**_read(run / "verification/marker_config.json"))) != asdict(markers):
            raise ValueError("Old/new training markers differ")
        if asdict(AuditConfig(**_read(run / "verification/audit_config.json"))) != asdict(audit):
            raise ValueError("Old/new original audit protocols differ")
        source_full = _read(run / "verification/checkpoint_manifest.json")["stages"]["full_after_fl"]
        full = next(b for b in config["existing_models"] if b["training_seed"] == binding["training_seed"] and b["method"] == "Full")
        if source_full["parameters_sha256"] != full["parameters_sha256_recorded"] or binding["partition_sha256"] != full["partition_sha256"]:
            raise ValueError("Old candidate is not paired to its declared Full/partition")
        if binding["method"] == "Full":
            original = VerificationExperimentConfig(**_read(run / "config.json"))
            ignore = {"seed", "run_name", "out_dir", "data_dir", "client_device", "eval_device"}
            if {k: v for k, v in asdict(original).items() if k not in ignore} != {k: v for k, v in asdict(cfg).items() if k not in ignore}:
                raise ValueError("Old and new source experiment budgets differ")
            source_paths.append(run)
    for path in config["exclude_collections"]:
        collection = _inside(root, path)
        for name in ("inputs.json", "frozen_manifest.json"):
            if not (collection / name).is_file():
                raise FileNotFoundError(f"Missing public prior-collection exclusion metadata: {collection / name}")
            input_paths.append(collection / name)
    if not metadata_only:
        from fulbase.verification.main_candidates import check_main_source
        for source in source_paths:
            checked = check_main_source(source)
            input_paths.extend(source / name for name in checked["source_hashes"])
            del checked
    return {"status": "metadata_verified_model_bytes_not_checked" if metadata_only else "old_sources_and_models_verified",
            "old_bindings": 12, "new_training_seeds": [46, 47], "planned_bindings": 45,
            "planned_requests": 40, "formal_audit": False, "source_hashes": _hashes(root, input_paths)}


def _hashes(root, paths):
    return {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(set(paths))}


def _code_hashes(root):
    paths = []
    for directory in ("fulbase", "fulbase/verification", "fulbase/experiment", "fulbase/unlearning",
                      "fulbase/flower", "fulbase/recovery", "fulbase/dp"):
        paths.extend((root / directory).glob("*.py"))
    paths.extend(root / name for name in ("scripts/run_verification_main.py", "scripts/run_verification_main_candidates.py",
                                         "scripts/collect_main_experiment_markers.py"))
    return _hashes(root, paths)


def _run_files(root, run):
    # A completed run is immutable for resume, including history and checkpoints.
    return _hashes(root, [p for p in run.rglob("*") if p.is_file()])


def _verify_hashes(root, hashes):
    for name, expected in hashes.items():
        if file_hash(_inside(root, name)) != expected:
            raise ValueError(f"Frozen artifact changed: {name}")


def _validate_stage_files(run, required_stages):
    if _read(run / "run_status.json").get("status") != "development_run_complete":
        raise ValueError("Runner did not publish a complete result")
    manifest = _read(run / "verification/checkpoint_manifest.json")
    if set(required_stages) != set(manifest["stages"]):
        raise ValueError("Completed run does not match the exact prespecified stage set")
    for stage in required_stages:
        row = manifest["stages"][stage]
        if file_hash(_inside(run, row["checkpoint"])) != row["checkpoint_sha256"]:
            raise ValueError("Completed checkpoint differs from its manifest")
    return manifest


def _validate_source_job(run, cfg, markers, audit):
    if _read(run / "config.json") != asdict(cfg):
        raise ValueError("Completed source config differs from the prespecified source job")
    if (_read(run / "verification/marker_config.json") != asdict(markers) or
            _read(run / "verification/audit_config.json") != asdict(audit)):
        raise ValueError("Completed source marker/audit configuration differs")


def _validate_candidate_job(run, source, include_marker_controls, data_dir):
    from fulbase.verification.main_candidates import PGA_BUDGET, C_BUDGET, D_BUDGET
    original = VerificationExperimentConfig(**_read(source / "config.json"))
    actual = VerificationExperimentConfig(**_read(run / "config.json"))
    ignore = {"run_name", "out_dir", "eval_device", "data_dir", "methods", "include_fakes", "with_recovery"}
    if {k: v for k, v in asdict(original).items() if k not in ignore} != {k: v for k, v in asdict(actual).items() if k not in ignore}:
        raise ValueError("Completed candidate source experiment budget differs")
    if (actual.methods != "history_subtract,federaser_lite,pga" or not actual.include_fakes or actual.with_recovery
            or actual.data_dir != (data_dir or original.data_dir)):
        raise ValueError("Completed candidate execution budget differs")
    for name in ("data_partition.json", "verification/marker_config.json", "verification/audit_config.json"):
        if _read(run / name) != _read(source / name):
            raise ValueError("Completed candidate pairing differs from its source")
    source_full = _read(source / "verification/checkpoint_manifest.json")["stages"]["full_after_fl"]
    candidate_full = _read(run / "verification/checkpoint_manifest.json")["stages"]["full_after_fl"]
    for key in ("checkpoint_sha256", "parameters_sha256"):
        if source_full[key] != candidate_full[key]:
            raise ValueError("Completed candidate Full differs from the prespecified source")
    budget = _read(run / "main_candidate_config.json")
    expected = {"history_subtract": {"alpha": .5}, "federaser_lite": {"damping": 1., "calibrate": True},
                "pga": PGA_BUDGET, "fake_corrupt": {"scale": .5},
                "fixed_C": C_BUDGET if include_marker_controls else None,
                "D_margin": asdict(D_BUDGET) if include_marker_controls else None}
    if budget.get("budgets") != expected or budget.get("include_marker_controls") is not include_marker_controls:
        raise ValueError("Completed candidate fixed budgets differ")
    provenance = _read(run / "main_candidate_provenance.json")
    if provenance["source_full_parameters_sha256"] != source_full["parameters_sha256"]:
        raise ValueError("Candidate provenance binds a different Full")
    for name, expected_hash in provenance["source_hashes"].items():
        if file_hash(_inside(source, name)) != expected_hash:
            raise ValueError("Candidate provenance source artifact changed")
    inventory = _read(run / "candidate_inventory.json")
    status = _read(run / "run_status.json")
    if (inventory["status"] != "all_endpoints_fixed" or
            file_hash(run / "candidate_inventory.json") != status["candidate_inventory_sha256"]):
        raise ValueError("Candidate inventory was not frozen before evaluation")
    required = list(CANDIDATE_STAGES) if include_marker_controls else list(CANDIDATE_STAGES)[:5]
    if {row["stage"] for row in inventory["candidates"]} != {CANDIDATE_STAGES[m] for m in required} or len(inventory["candidates"]) != len(required):
        raise ValueError("Candidate inventory omitted or added endpoints")
    manifest = _read(run / "verification/checkpoint_manifest.json")
    for row in inventory["candidates"]:
        entry = manifest["stages"][row["stage"]]
        if any(row[k] != entry[k] for k in ("checkpoint", "checkpoint_sha256", "parameters_sha256")):
            raise ValueError("Evaluated candidate differs from its frozen inventory")


def run_campaign(config_path, output_dir, *, device="cuda", data_dir=None, threads=2, project_root=None):
    root, config, base, markers, audit = load_campaign(config_path, project_root)
    if threads < 1:
        raise ValueError("threads must be positive")
    preflight = check_campaign(config_path, project_root=root)
    output = _inside(root, output_dir)
    sources = {_inside(root, row["run_dir"]) for row in config["existing_models"]}
    if any(output == src or output in src.parents or src in output.parents for src in sources):
        raise ValueError("Campaign output must not overlap a source run")
    execution = {"device": device, "data_dir": data_dir, "threads": threads}
    config_files = [Path(config_path).resolve()] + [_inside(root, config[key]) for key in ("source_experiment", "marker_config", "audit_config")]
    blueprint = {"schema_version": SCHEMA, "config": config, "execution": execution,
                 "config_hashes": _hashes(root, config_files), "code_hashes": _code_hashes(root),
                 "old_source_hashes": preflight["source_hashes"]}
    if output.exists():
        if not (output / "blueprint.json").is_file() or _read(output / "blueprint.json") != blueprint:
            raise ValueError("Existing campaign differs from frozen blueprint, code or execution settings")
        state = _read(output / "index.json")
    else:
        output.mkdir(parents=True)
        _write(output / "blueprint.json", blueprint)
        state = {"schema_version": SCHEMA, "status": "running", "jobs": {}, "models": list(config["existing_models"])}
        _write(output / "index.json", state)
    import torch
    torch.set_num_threads(threads)
    from fulbase.verification.main_candidates import run_main_candidates

    def job(name, path, callback, required_stages, validate):
        receipt = state["jobs"].get(name)
        if receipt and receipt["status"] == "complete":
            if receipt["run_dir"] != path.relative_to(root).as_posix():
                raise ValueError("Completed job receipt points to a different run")
            _verify_hashes(root, receipt["files"])
            _validate_stage_files(path, required_stages)
            validate()
            print(f"[main] Reuse {name}: {path}", flush=True)
            return
        if path.exists():
            status = path / "run_status.json"
            if not receipt or not status.is_file() or _read(status).get("status") != "development_run_complete":
                raise ValueError(f"Partial/unregistered run preserved: {path}. Do not delete or silently rerun it; inspect the failure before starting a new campaign.")
        else:
            state["jobs"][name] = {"status": "started", "run_dir": path.relative_to(root).as_posix()}
            _update(output / "index.json", state)
            print(f"[main] Start {name}: {path}", flush=True)
            callback()
        _validate_stage_files(path, required_stages)
        validate()
        state["jobs"][name] = {"status": "complete", "run_dir": path.relative_to(root).as_posix(), "files": _run_files(root, path)}
        _update(output / "index.json", state)

    # No result-dependent branching: all five prespecified seeds and every arm run.
    for seed in config["development_seeds"] + config["confirmation_seeds"]:
        new = seed in config["confirmation_seeds"]
        cohort = "confirmation" if new else "development"
        if new:
            source = output / f"seed{seed}_source"
            cfg = replace(base, seed=seed, run_name=f"rp2_main_v1_source_seed{seed}",
                          client_device=device, eval_device=device, data_dir=data_dir or base.data_dir)
            job(f"source_{seed}", source, lambda: run_verification_experiment(cfg, markers, audit, run_directory=source),
                ("full_after_fl", "retrain_fl_pre_recovery"), lambda: _validate_source_job(source, cfg, markers, audit))
        else:
            source = _inside(root, next(row["run_dir"] for row in config["existing_models"]
                                       if row["training_seed"] == seed and row["method"] == "Full"))
        candidate = output / f"seed{seed}_candidates"
        methods = list(CANDIDATE_STAGES) if new else list(CANDIDATE_STAGES)[:5]
        job(f"candidates_{seed}", candidate,
            lambda: run_main_candidates(source, device=device, data_dir=data_dir, include_marker_controls=new,
                                         threads=threads, run_directory=candidate),
            ["full_after_fl"] + [CANDIDATE_STAGES[m] for m in methods],
            lambda: _validate_candidate_job(candidate, source, new, data_dir))
        added = [make_binding(root, candidate, CANDIDATE_STAGES[m], seed, m, cohort) for m in methods]
        if new:
            added += [make_binding(root, source, stage, seed, method, cohort)
                      for method, stage in (("Full", "full_after_fl"), ("Retrain", "retrain_fl_pre_recovery"))]
        old_keys = {(b["training_seed"], b["method"]): b for b in state["models"]}
        for binding in added:
            key = (binding["training_seed"], binding["method"])
            if key in old_keys and old_keys[key] != binding:
                raise ValueError("Completed binding changed during resume")
            if key not in old_keys:
                state["models"].append(binding)
        _update(output / "index.json", state)
    from fulbase.verification.main_audit import make_plan
    from fulbase.verification.main_tables import write_main_tables
    _verify_hashes(root, blueprint["old_source_hashes"])
    _verify_hashes(root, blueprint["config_hashes"])
    _verify_hashes(root, blueprint["code_hashes"])
    plan = make_plan(state["models"], METHODS, training_seeds=[43, 44, 45, 46, 47],
                     history_run_dirs=config["history_run_dirs"], exclude_collections=config["exclude_collections"])
    table_counts = write_main_tables(root, state["models"], output / "tables")
    plan_path = output / "audit_plan.json"
    if plan_path.exists():
        if _read(plan_path) != plan:
            raise ValueError("Bound audit plan differs; never replace a frozen plan")
    else:
        _write(plan_path, plan)
    state.update(status="models_complete_ready_for_independent_collection", audit_plan=plan_path.relative_to(root).as_posix(),
                 table_counts=table_counts)
    _update(output / "index.json", state)
    print(f"[main] All 45 bindings recorded; run independent collection --phase check next. Audit plan: {plan_path}", flush=True)
    return output
