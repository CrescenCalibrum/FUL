"""Run PGA budgets on one trusted saved Full; no federated retraining."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fulbase.verification.pga_diagnostic import load_source_run, run_pga_diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Original trusted run including models/model_full.pt")
    parser.add_argument("--steps", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--lrs", type=float, nargs="+", default=[0.0001])
    parser.add_argument("--project-ratios", type=float, nargs="+", default=[0.01])
    parser.add_argument("--objectives", choices=["ce", "margin", "margin_capped", "margin_capped_retain"], nargs="+", default=["ce"],
                        help="Matched CE ascent / marker margin erasure; margin requires markers_only")
    parser.add_argument("--margin-raw-drop", type=float, default=3.0,
                        help="B/C per-sample raw logit margin reduction relative to frozen Full")
    parser.add_argument("--retain-weights", type=float, nargs="+", default=[1.0],
                        help="C only: absolute lambdas, or multipliers of the training-only probe ratio")
    parser.add_argument("--retain-weight-mode", choices=["fixed", "probe_ratio"], default="fixed")
    parser.add_argument("--layers", choices=["head", "last", "all"], default="last")
    parser.add_argument("--data-scope", choices=["real", "real_and_markers", "markers_only"], default="real")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", help="Override dataset location without changing source partition")
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--run-name", default="pga_diagnostic")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--check-only", action="store_true", help="Validate Full and manifests without loading datasets or writing results")
    args = parser.parse_args()
    import torch
    if args.threads < 1:
        parser.error("threads must be positive")
    if any(x != "ce" for x in args.objectives) and args.data_scope != "markers_only":
        parser.error("Marker margin objectives require --data-scope markers_only")
    if any(x.startswith("margin_capped") for x in args.objectives) and args.layers != "head":
        parser.error("Capped objectives currently require --layers head")
    torch.set_num_threads(args.threads)
    if args.check_only:
        from fulbase.verification.audit import parameter_hash
        source = load_source_run(args.run_dir)
        print(f"Source artifacts OK; Full SHA256={parameter_hash(source['params'])}")
        return
    run_pga_diagnostic(args.run_dir, steps=args.steps, lrs=args.lrs,
                       project_ratios=args.project_ratios, layers=args.layers,
                       data_scope=args.data_scope, device=args.device, data_dir=args.data_dir,
                       out_dir=args.out_dir, run_name=args.run_name, objectives=args.objectives,
                       margin_raw_drop=args.margin_raw_drop, retain_weights=args.retain_weights,
                       retain_weight_mode=args.retain_weight_mode)


if __name__ == "__main__":
    main()
