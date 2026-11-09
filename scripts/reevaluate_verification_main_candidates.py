"""Resume evaluation of a frozen candidate inventory without training or repair."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fulbase.verification.main_evaluation import check_frozen_candidates, run_candidate_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Existing frozen candidates directory, including failed runs")
    parser.add_argument("--run-directory", help="New evaluation directory; existing paths are rejected")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--check-only", action="store_true", help="Read-only hashes, state shapes and all BN diagnostics")
    parser.add_argument("--diagnose-v2", action="store_true")
    parser.add_argument("--ablation-config")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.check_only:
        source = check_frozen_candidates(args.run_dir)
        print(json.dumps({"status": "frozen_candidate_integrity_verified",
            "full_parameters_sha256": source["full_parameters_sha256"],
            "candidate_count": len(source["inventory"]["candidates"]),
            "original_calibration_identity_verified": source["original_calibration_identity_verified"],
            "diagnostics": source["diagnostics"], "source_hashes": source["source_hashes"],
            "numerical_validity_is_separate_from_integrity": True}, ensure_ascii=False, indent=2, allow_nan=False))
        return
    if not args.run_directory:
        parser.error("--run-directory is required unless --check-only is specified")
    output = run_candidate_evaluation(args.run_dir, args.run_directory, device=args.device,
        data_dir=args.data_dir, threads=args.threads, diagnose_v2=args.diagnose_v2,
        ablation_config=args.ablation_config, bootstrap_samples=args.bootstrap_samples)
    print(json.dumps({"status": "main_candidate_evaluation_recovery_complete",
                      "output": str(output), "formal_audit": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
