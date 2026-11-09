"""Run the bounded paired D2-reference/interior training-only head diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fulbase.verification.boundary_interior_diagnostic import check_boundary_interior_sources, run_boundary_interior_diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/verification/head-boundary-interior-development.json")
    parser.add_argument("--seed", type=int, required=True, choices=(43, 44, 45))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir")
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--run-name")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    import torch
    torch.set_num_threads(args.threads)
    if args.check_only:
        context = check_boundary_interior_sources(args.config, args.seed)
        print(json.dumps({"status": "fixed_full_verified", "seed": args.seed,
                          "parameters_sha256": context["binding"]["full_parameters_sha256"],
                          "trials": 2, "audit_inputs_accessed": False}))
    else:
        output = run_boundary_interior_diagnostic(args.config, args.seed, device=args.device,
            data_dir=args.data_dir, out_dir=args.out_dir, run_name=args.run_name, threads=args.threads)
        print(json.dumps({"output": str(output), "formal_audit": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
