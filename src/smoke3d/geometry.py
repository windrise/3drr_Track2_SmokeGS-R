from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def resolve_geometry_prior_path(dataset_cfg) -> Path | None:
    if not dataset_cfg.prior_root:
        return None
    candidate = Path(dataset_cfg.prior_root) / dataset_cfg.geometry_subdir / dataset_cfg.geometry_file
    if candidate.exists():
        return candidate
    return None


def load_scene_geometry_prior(
    prior_path: str | Path,
    max_points: int | None = None,
    seed: int = 0,
) -> dict[str, torch.Tensor | str]:
    prior_path = Path(prior_path)
    if prior_path.suffix != ".npz":
        raise ValueError(f"unsupported geometry prior file: {prior_path}")

    with np.load(prior_path) as payload:
        if "points" not in payload:
            raise KeyError(f"geometry prior missing `points`: {prior_path}")

        points = torch.from_numpy(payload["points"]).float()
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"geometry points must have shape [N, 3], got {tuple(points.shape)}")

        colors = None
        if "colors" in payload:
            colors = torch.from_numpy(payload["colors"]).float()
            if colors.ndim != 2 or colors.shape[1] != 3:
                raise ValueError(f"geometry colors must have shape [N, 3], got {tuple(colors.shape)}")
            if colors.shape[0] != points.shape[0]:
                raise ValueError("geometry colors must align with points")

        confidences = None
        if "confidences" in payload:
            confidences = torch.from_numpy(payload["confidences"]).float().reshape(-1)
            if confidences.shape[0] != points.shape[0]:
                raise ValueError("geometry confidences must align with points")

    if points.shape[0] == 0:
        raise ValueError(f"geometry prior contains no points: {prior_path}")

    if max_points is not None and points.shape[0] > max_points:
        if confidences is not None and (confidences.max() - confidences.min()) > 1e-6:
            indices = torch.topk(confidences, k=max_points, largest=True).indices
        else:
            generator = torch.Generator().manual_seed(seed)
            indices = torch.randperm(points.shape[0], generator=generator)[:max_points]
        points = points[indices]
        if colors is not None:
            colors = colors[indices]
        if confidences is not None:
            confidences = confidences[indices]

    result: dict[str, torch.Tensor | str] = {
        "points": points,
        "path": str(prior_path),
    }
    if colors is not None:
        result["colors"] = colors.clamp(0.0, 1.0)
    if confidences is not None:
        result["confidences"] = confidences.clamp(0.0, 1.0)
    return result
