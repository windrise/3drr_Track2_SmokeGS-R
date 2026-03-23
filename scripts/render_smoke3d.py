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
    parser.add_argument("--checkpoint", "-w", required=True, type=str)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--data_path_override", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=10)
    parser.add_argument("--render_smoky", action="store_true")
    args = parser.parse_args()

    configure_cpu_limit(args.cpu_threads)
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from smoke3d import configure_torch_threads, load_experiment_config, render_checkpoint

    configure_torch_threads(args.cpu_threads)
    out_dir = render_checkpoint(
        checkpoint_path=args.checkpoint,
        config_loader=load_experiment_config,
        output_dir=args.output_dir,
        data_path_override=args.data_path_override,
        device=args.device,
        render_clean=not args.render_smoky,
    )
    print(f"Rendered images to {out_dir}")


if __name__ == "__main__":
    main()
