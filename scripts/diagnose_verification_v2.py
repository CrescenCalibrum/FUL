"""Diagnose saved development marker observations without training or inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fulbase.verification.protocol_v2_io import run_protocol_v2_diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", required=True,
                        help="Explicit saved run directories; only calibration stages are read")
    parser.add_argument("--stages", nargs="+", help="Exact stage names present in every run; default: all recorded stages")
    parser.add_argument("--out-dir", default="diagnostics/verification_v2",
                        help="Parent for a new timestamped output directory, outside all input runs")
    parser.add_argument("--run-name", default="protocol_v2")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2718,
                        help="Public diagnostic resampling seed, not an audit-sample seed")
    parser.add_argument("--confidence", type=float, default=.95,
                        help="Descriptive v2/class interval confidence; ablation geometry uses source alpha")
    parser.add_argument("--ablation-config",
                        help="Explicit development bounds JSON for accuracy/anchor ablations; omitted bounds remain unconfigured")
    parser.add_argument("--check-only", action="store_true",
                        help="Validate all inputs and provenance without bootstrapping or output writes")
    args = parser.parse_args()
    try:
        result = run_protocol_v2_diagnostic(**vars(args))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"protocol-v2: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
