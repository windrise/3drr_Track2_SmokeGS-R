from __future__ import annotations

import os


THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def configure_cpu_env(max_threads: int) -> int:
    max_threads = max(1, int(max_threads))
    for var in THREAD_ENV_VARS:
        os.environ[var] = str(max_threads)
    os.environ["SMOKE3D_IO_THREADS"] = str(max_threads)
    return max_threads


def configure_torch_threads(max_threads: int) -> int:
    max_threads = max(1, int(max_threads))
    try:
        import torch
    except Exception:
        return max_threads

    torch.set_num_threads(max_threads)
    try:
        torch.set_num_interop_threads(min(max_threads, 4))
    except RuntimeError:
        pass
    return max_threads
