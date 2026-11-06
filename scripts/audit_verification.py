"""Audit saved checkpoints without loading any private training or test data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fulbase.models import build_model
from fulbase.utils import set_model_parameters
from fulbase.verification.markers import MarkerConfig, MarkerDataset
from fulbase.verification.audit import (AuditConfig, file_hash, parameter_hash,
                                       observe_markers, write_audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--stage", required=True, help="Stage key in checkpoint_manifest.json")
    parser.add_argument("--role", choices=["calibration", "audit"], default="calibration")
    parser.add_argument("--sample-seed", type=int, required=True, help="Auditor-owned seed; never forwarded to training")
    parser.add_argument("--audit-config", help="Frozen AuditConfig JSON for formal audit; otherwise development config")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--include-benchmark-task", action="store_true",
                        help="Use recorded independent benchmark metrics; these are not independently verified by this auditor")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    root = Path(args.run_dir).resolve()
    vdir = root / "verification"
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    marker_cfg = MarkerConfig(**json.loads((vdir / "marker_config.json").read_text(encoding="utf-8")))
    cfg_path = Path(args.audit_config) if args.audit_config else vdir / "audit_config.json"
    audit_cfg = AuditConfig(**json.loads(cfg_path.read_text(encoding="utf-8")))
    if audit_cfg.protocol_locked and args.role != "audit":
        parser.error("Formal frozen audit requires independent audit-role markers")
    manifest = json.loads((vdir / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    if manifest["marker_fingerprint"] != marker_cfg.fingerprint():
        raise ValueError("Marker configuration differs from the training manifest")
    if file_hash(root / "data_partition.json") != manifest["partition_sha256"]:
        raise ValueError("Partition manifest changed")
    if args.stage not in manifest["stages"]:
        parser.error("Stage is not present in checkpoint manifest")
    num_classes = manifest["num_classes"]
    datasets = {cid: MarkerDataset(marker_cfg, cid, config["dataset"], args.role,
                                   args.sample_seed, num_classes) for cid in range(config["num_clients"])}
    def load_observations(stage):
        entry = manifest["stages"][stage]
        checkpoint = (root / entry["checkpoint"].replace("\\", "/")).resolve()
        if root not in checkpoint.parents:
            raise ValueError("Checkpoint outside run directory")
        if file_hash(checkpoint) != entry["checkpoint_sha256"]:
            raise ValueError("Checkpoint bytes differ from manifest")
        # Only locally trusted run checkpoints are supported; never arbitrary downloads.
        params = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if parameter_hash(params) != entry["parameters_sha256"]:
            raise ValueError("Checkpoint parameters differ from manifest")
        model = build_model(config["dataset"], num_classes)
        set_model_parameters(model, params)
        return {cid: observe_markers(model, ds, args.device, config["batch_size"])
                for cid, ds in datasets.items()}
    before = load_observations("full_after_fl")
    after = load_observations(args.stage)
    # Each invocation gets a new directory; independent audit never overwrites development evidence.
    from datetime import datetime
    output = vdir / "offline" / (datetime.now().strftime("%Y%m%d-%H%M%S-%f") + "_" + args.stage)
    before_task, after_task = None, None
    if args.include_benchmark_task:
        before_task = manifest["stages"]["full_after_fl"]["task_metrics"]
        after_task = manifest["stages"][args.stage]["task_metrics"]
    report = write_audit(before, after, [int(x) for x in config["forget_clients"].split(",")],
                         audit_cfg, output, {"role": args.role, "stage": args.stage,
                         "checkpoint_sha256": manifest["stages"][args.stage]["checkpoint_sha256"],
                         "marker_fingerprint": marker_cfg.fingerprint(),
                         "task_source": "recorded_independent_benchmark" if args.include_benchmark_task else "unavailable"},
                         before_task, after_task)
    print(json.dumps({"status": report["status"], "exploratory_status": report["exploratory_status"],
                      "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
