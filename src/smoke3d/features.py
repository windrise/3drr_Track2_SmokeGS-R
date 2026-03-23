from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoV2FeatureExtractor(nn.Module):
    def __init__(
        self,
        repo_path: str | Path,
        model_name: str = "dinov2_vits14",
        input_size: int = 518,
    ):
        super().__init__()
        repo_path = Path(repo_path).expanduser().resolve()
        self.model = torch.hub.load(str(repo_path), model_name, source="local").eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.input_size = int(input_size)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1))
        self.patch_size = int(getattr(self.model, "patch_size", 14))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"expected BCHW RGB tensor, got shape={tuple(images.shape)}")
        resized = F.interpolate(
            images.float(),
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )
        normalized = (resized - self.mean) / self.std
        features = self.model.forward_features(normalized)["x_norm_patchtokens"]
        side = self.input_size // self.patch_size
        if features.shape[1] != side * side:
            raise ValueError(
                f"unexpected patch token count {features.shape[1]} for side={side}; "
                f"input_size={self.input_size}, patch_size={self.patch_size}"
            )
        features = features.view(features.shape[0], side, side, features.shape[-1]).permute(0, 3, 1, 2).contiguous()
        return F.normalize(features, dim=1, p=2)
