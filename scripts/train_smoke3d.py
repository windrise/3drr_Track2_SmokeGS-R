#!/usr/bin/env python

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def configure_cpu_limit(max_threads: int) -> int:
    max_threads = max(1, int(max_threads))
    for var in (
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[var] = str(max_threads)
    os.environ["SMOKE3D_IO_THREADS"] = str(max_threads)
    return max_threads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True, type=str)
    parser.add_argument("--cpu-threads", type=int, default=10)
    args = parser.parse_args()

    configure_cpu_limit(args.cpu_threads)
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from smoke3d import SmokeTrainer, configure_torch_threads, load_experiment_config

    configure_torch_threads(args.cpu_threads)
    cfg = load_experiment_config(args.config)
    trainer = SmokeTrainer(cfg)
    checkpoint = trainer.train()
    print(f"Checkpoint saved to {checkpoint}")


if __name__ == "__main__":
    main()
