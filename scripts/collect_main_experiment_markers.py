"""Collect and analyze fresh markers for the frozen main-experiment matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fulbase.verification.main_audit import (
    analyze_collection, collect_observations, preflight, prepare_collection,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("check", "prepare", "collect", "analyze"))
    parser.add_argument("--plan", required=True, help="Hash-bound plan emitted after all declared models complete")
    parser.add_argument("--output-dir", required=True, help="New output directory; reuse identical path to resume")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1 or args.batch_size < 1:
        parser.error("--threads and --batch-size must be positive")
    import torch
    torch.set_num_threads(args.threads)
    if args.phase == "check":
        context = preflight(args.plan)
        print(json.dumps({"status": "main_model_bindings_verified", "formal_audit": False,
            "models": len(context["plan"]["models"]),
            "requests": len(context["plan"]["models"]) - len(context["plan"]["sampling"]["training_seeds"]),
            "registered_histories": len(context["registry"]["coverage"])}))
    elif args.phase == "prepare":
        result = prepare_collection(args.plan, args.output_dir)
        print(json.dumps({"status": result["status"], "counts": result["counts"],
                          "output": str(Path(args.output_dir).resolve())}))
    elif args.phase == "collect":
        result = collect_observations(args.plan, args.output_dir, device=args.device, batch_size=args.batch_size,
                                      progress=lambda text: print(text, flush=True))
        print(json.dumps({"status": result["status"], "shards": result["shard_count"]}))
    else:
        report = analyze_collection(args.plan, args.output_dir)
        print(json.dumps({"status": report["status"], "formal_audit": False,
                          "requests": len(report["requests"]),
                          "output": str(Path(args.output_dir).resolve() / "analysis")}))


if __name__ == "__main__":
    main()
