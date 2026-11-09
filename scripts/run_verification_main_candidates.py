"""Generate prespecified main-experiment candidates from one existing Full."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fulbase.verification.main_candidates import FEDERASER_BN_POLICIES, check_main_source, run_main_candidates


def load_candidate_options(path):
    """A candidate override never rewrites the bound Full source configuration."""
    options = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(options, dict) or set(options) != {"schema_version", "federaser_bn_policy"}
            or options.get("schema_version") != "rp2-main-candidate-options-v1"):
        raise ValueError("Candidate config requires schema rp2-main-candidate-options-v1 and only federaser_bn_policy")
    if options["federaser_bn_policy"] not in FEDERASER_BN_POLICIES:
        raise ValueError("federaser_bn_policy must be legacy_replay or retained_endpoint")
    return options


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
    parser.add_argument("--candidate-config", help="Explicit candidate-only BN policy JSON; the source config remains unchanged")
    parser.add_argument("--federaser-bn-policy", choices=FEDERASER_BN_POLICIES)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    policy = args.federaser_bn_policy
    if args.candidate_config is not None:
        try:
            configured = load_candidate_options(args.candidate_config)["federaser_bn_policy"]
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        if policy is not None and policy != configured:
            parser.error("--candidate-config and --federaser-bn-policy conflict")
        policy = configured
    if args.check_only:
        source = check_main_source(args.run_dir, federaser_bn_policy=policy)
        print(json.dumps({"status": "source_full_history_initial_verified", "seed": source["cfg"].seed,
            "full_parameters_sha256": source["full_parameters_sha256"],
            "history_sha256": source["history_sha256"],
            "initial_parameters_sha256": source["initial_parameters_sha256"],
            "source_hashes": source["source_hashes"], "history_validation": source["history_validation"],
            "source_federaser_bn_policy": source["source_federaser_bn_policy"],
            "effective_federaser_bn_policy": source["effective_federaser_bn_policy"],
            "federaser_bn_preflight": source["federaser_bn_preflight"],
            "training_performed": False, "model_inference_performed": False,
            "candidate_effectiveness_assessed": False}))
    else:
        result = run_main_candidates(args.run_dir, out_dir=args.out_dir, run_name=args.run_name,
            run_directory=args.run_directory, device=args.device, data_dir=args.data_dir,
            include_marker_controls=args.include_marker_controls, threads=args.threads,
            federaser_bn_policy=policy)
        print(json.dumps({"status": "development_run_complete", "output": str(result), "formal_audit": False}))


if __name__ == "__main__":
    main()
