"""Check or execute the prespecified five-seed RP2 main experiment campaign."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fulbase.verification.main_campaign import check_campaign, run_campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("check", "run"), required=True)
    parser.add_argument("--config", default="configs/verification/main-experiment-v1.json")
    parser.add_argument("--output-dir", default="runs/rp2_main_v1")
    parser.add_argument("--metadata-only", action="store_true", help="Check metadata only; never a server model preflight")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-dir")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.metadata_only and args.phase != "check":
        parser.error("--metadata-only is only valid with --phase check")
    if args.phase == "check":
        result = check_campaign(args.config, metadata_only=args.metadata_only)
        result["source_artifact_count"] = len(result.pop("source_hashes"))
        print(json.dumps(result, ensure_ascii=False))
    else:
        run_campaign(args.config, args.output_dir, device=args.device, data_dir=args.data_dir, threads=args.threads)


if __name__ == "__main__":
    main()
