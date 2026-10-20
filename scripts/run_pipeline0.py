from __future__ import annotations

import argparse

from fulbase.experiment.pipeline import ExperimentConfig, run_full_pipeline


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="fashionmnist", choices=["fashionmnist", "mnist", "cifar10"])
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--out-dir", default="./runs")
    p.add_argument("--run-name", default="debug")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--num-clients", type=int, default=10)
    p.add_argument("--num-rounds", type=int, default=5)
    p.add_argument("--fraction-fit", type=float, default=1.0)
    p.add_argument("--fraction-evaluate", type=float, default=1.0)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--partition", default="dirichlet", choices=["dirichlet", "iid"])
    p.add_argument("--alpha", type=float, default=0.5)

    p.add_argument("--client-device", default="auto")
    p.add_argument("--eval-device", default="auto")
    p.add_argument("--client-cpus", type=float, default=1.0)
    p.add_argument("--client-gpus", type=float, default=0.25)

    p.add_argument("--forget-clients", default="0")
    p.add_argument("--unlearn-alpha", type=float, default=0.5)

    p.add_argument("--dp-sigma", type=float, default=0.0)
    p.add_argument("--proto-clip-norm", type=float, default=10.0)
    p.add_argument("--lda-ridge", type=float, default=1e-2)
    args = p.parse_args()
    cfg = ExperimentConfig(**vars(args))
    run_full_pipeline(cfg)


if __name__ == "__main__":
    main()
