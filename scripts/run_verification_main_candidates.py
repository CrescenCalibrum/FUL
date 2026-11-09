"""Generate prespecified main-experiment candidates from one existing Full."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fulbase.verification.main_candidates import check_main_source, run_main_candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Trusted Full+history+initial source run")
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--run-name")
    parser.add_argument("--run-directory", help="Explicit new output directory; existing paths are rejected")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--include-marker-controls", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.check_only:
        source = check_main_source(args.run_dir)
        print(json.dumps({"status": "source_full_history_initial_verified", "seed": source["cfg"].seed,
            "full_parameters_sha256": source["full_parameters_sha256"],
            "history_sha256": source["history_sha256"],
            "initial_parameters_sha256": source["initial_parameters_sha256"],
            "source_hashes": source["source_hashes"], "history_validation": source["history_validation"]}))
    else:
        result = run_main_candidates(args.run_dir, out_dir=args.out_dir, run_name=args.run_name,
            run_directory=args.run_directory, device=args.device, data_dir=args.data_dir,
            include_marker_controls=args.include_marker_controls, threads=args.threads)
        print(json.dumps({"status": "development_run_complete", "output": str(result), "formal_audit": False}))


if __name__ == "__main__":
    main()
