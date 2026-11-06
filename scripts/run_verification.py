"""Research-point-2 development runner; defaults are not calibrated claims."""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fulbase.verification.markers import MarkerConfig
from fulbase.verification.audit import AuditConfig
from fulbase.verification.experiment import VerificationExperimentConfig, run_verification_experiment
from fulbase.verification.synthetic import synthetic_data


def read_config(path, cls):
    if path is None:
        return {}
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    extra = set(values) - {f.name for f in fields(cls)}
    if extra:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(extra)}")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="ExperimentConfig JSON; omitted fields keep defaults")
    parser.add_argument("--marker-config", help="MarkerConfig JSON")
    parser.add_argument("--audit-config", help="Development AuditConfig JSON")
    parser.add_argument("--synthetic-smoke", action="store_true", help="Tiny generated-data functional run; not benchmark evidence")
    parser.add_argument("--threads", type=int, default=2)
    for name, kind, choices in [
        ("backend", str, ["flower", "serial"]), ("dataset", str, ["cifar10", "fashionmnist", "mnist"]),
        ("num-clients", int, None), ("num-rounds", int, None), ("seed", int, None),
        ("batch-size", int, None), ("local-epochs", int, None), ("lr", float, None),
        ("alpha", float, None), ("partition", str, ["iid", "dirichlet"]),
        ("forget-clients", str, None), ("methods", str, None), ("pga-steps", int, None),
        ("pga-lr", float, None), ("pga-project-ratio", float, None),
        ("pga-layers", str, ["head", "last", "all"]),
        ("pga-data-scope", str, ["real", "real_and_markers"]),
        ("fake-steps", int, None), ("fake-lr", float, None), ("corruption-scale", float, None),
        ("data-dir", str, None), ("out-dir", str, None), ("run-name", str, None),
        ("client-device", str, None), ("eval-device", str, None),
        ("client-cpus", float, None), ("client-gpus", float, None),
        ("dp-sigma", float, None), ("calibration-seed", int, None)]:
        parser.add_argument("--"+name, type=kind, choices=choices, default=None)
    parser.add_argument("--with-recovery", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--include-fakes", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    import torch
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    values = read_config(args.config, VerificationExperimentConfig)
    for key in {f.name for f in fields(VerificationExperimentConfig)}:
        value = getattr(args, key, None)
        if value is not None:
            values[key] = value
    marker_values = {"enabled": True, **read_config(args.marker_config, MarkerConfig)}
    audit_values = read_config(args.audit_config, AuditConfig)
    fd = None
    if args.synthetic_smoke:
        values.update(backend="serial", dataset="fashionmnist", num_clients=3, num_rounds=1,
                      forget_clients="0", batch_size=10, local_epochs=1, pga_steps=2,
                      fake_steps=2, client_device="cpu", eval_device="cpu", client_gpus=0,
                      with_recovery=True, synthetic_smoke=True,
                      run_name=values.get("run_name", "verification_synthetic_smoke"))
        marker_values.update(families_per_class=2, variants_per_family=1, batch_size=10)
        audit_values.update(bootstrap_samples=200, protocol_locked=False)
        fd = synthetic_data(3, values.get("seed", 42))
    cfg = VerificationExperimentConfig(**values)
    if cfg.synthetic_smoke and fd is None:
        parser.error("synthetic_smoke config requires --synthetic-smoke to generate its inputs")
    run_verification_experiment(cfg, MarkerConfig(**marker_values), AuditConfig(**audit_values), fd)


if __name__ == "__main__":
    main()
