"""Collect the fixed nine-model, three-batch independent marker validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fulbase.verification.independent_marker import (
    analyze_collection, collect_observations, preflight, prepare_collection,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("check", "prepare", "collect", "analyze"))
    parser.add_argument("--plan", default="plan/independent-marker-validation.json")
    parser.add_argument("--output-dir", required=True,
                        help="New dedicated directory outside all source runs; reuse it to resume")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    import torch
    torch.set_num_threads(args.threads)
    if args.phase == "check":
        context = preflight(args.plan)
        print(json.dumps({"status": "fixed_nine_checkpoints_verified", "models": 9,
                          "marker_fingerprint": context["plan"]["models"][0]["marker_fingerprint"],
                          "registered_history_runs": len(context["registry"]["coverage"])}))
    elif args.phase == "prepare":
        result = prepare_collection(args.plan, args.output_dir)
        print(json.dumps({"status": result["status"], "output": str(Path(args.output_dir).resolve()),
                          "counts": result["counts"]}))
    elif args.phase == "collect":
        result = collect_observations(args.plan, args.output_dir, device=args.device,
                                      batch_size=args.batch_size, progress=lambda text: print(text, flush=True))
        print(json.dumps({"status": result["status"], "shards": result["shard_count"]}))
    else:
        analyze_collection(args.plan, args.output_dir)
        print(json.dumps({"status": "conditional_marker_analysis_complete", "formal_audit": False,
                          "output": str(Path(args.output_dir).resolve() / "analysis")}))


if __name__ == "__main__":
    main()
