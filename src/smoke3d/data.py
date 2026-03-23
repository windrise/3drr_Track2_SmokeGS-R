from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torchvision


class SmokeSceneDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_cfg, split: str, load_images: bool = True):
        super().__init__()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"unsupported split: {split}")

        self._cfg = dataset_cfg
        self._split = split
        self._load_images = load_images
        self._data_path = Path(dataset_cfg.data_path)
        self._img_path_base = self._data_path / ("test" if split == "val" else split)
        self._clean_data_path = Path(dataset_cfg.clean_data_path) if dataset_cfg.clean_data_path else None
        self._clean_img_path_base = (
            self._clean_data_path / ("test" if split == "val" else split) if self._clean_data_path else None
        )
        self._meta_path = self._data_path / (
            "transforms_test.json" if split == "val" else f"transforms_{split}.json"
        )
        self._prior_root = Path(dataset_cfg.prior_root) if dataset_cfg.prior_root else None
        self._bg_color = dataset_cfg.background_color / 255.0

        self._records, self._data_info = self._load_records()
        if split == "val":
            take = min(dataset_cfg.val_preview_views, len(self._records))
            self._records = dict(list(self._records.items())[:take])
        self._record_keys = list(self._records.keys())
        self._length = len(self._record_keys)

        if load_images:
            self._preload_images()

    @property
    def data_info(self) -> dict[str, Any]:
        return self._data_info

    @property
    def record_keys(self) -> list[str]:
        return self._record_keys

    def get_transforms(self) -> list[torch.Tensor]:
        return [self._records[key]["transform_matrix"] for key in self._record_keys]

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, Any]:
        key = self._record_keys[index % self._length]
        record = self._records[key]
        output = {
            "transforms": record["transform_matrix"],
            "view_index": torch.tensor(index % self._length, dtype=torch.long),
            "infos": {"frame_name": record["frame_name"]},
        }
        if self._load_images:
            image = record["img_tensor"]
            if image is None:
                image = load_img(record["file_path"], channel=3).float() / 255.0
            output["images"] = image[:3]
            target_hw = (output["images"].shape[1], output["images"].shape[2])
            clean_prior = self._load_clean_prior(record, target_hw)
            if clean_prior is not None:
                output["clean_prior"] = clean_prior
                clean_weight = self._load_dense_map_prior(
                    record["frame_name"],
                    self._cfg.clean_weight_subdir,
                    target_hw,
                )
                if clean_weight is None and self._cfg.clean_confidence_mode == "difference":
                    clean_weight = self._build_clean_confidence(output["images"], clean_prior)
                if clean_weight is not None:
                    output["clean_prior_weight"] = clean_weight
        else:
            target_hw = (self._data_info["img_h"], self._data_info["img_w"])

        depth_prior, depth_mask, depth_source = self._load_depth_prior(
            record["frame_name"],
            target_hw,
            subdir=self._cfg.depth_subdir,
        )
        if depth_prior is not None:
            output["depth_prior"] = depth_prior
            output["depth_prior_valid_mask"] = depth_mask
            output["depth_prior_source"] = depth_source
            depth_weight = self._load_dense_map_prior(
                record["frame_name"],
                self._cfg.depth_weight_subdir,
                target_hw,
                valid_mask=depth_mask,
            )
            if depth_weight is not None:
                output["depth_prior_weight"] = depth_weight

        aux_depth_prior, aux_depth_mask, aux_depth_source = self._load_depth_prior(
            record["frame_name"],
            target_hw,
            subdir=self._cfg.aux_depth_subdir,
        )
        if aux_depth_prior is not None:
            output["aux_depth_prior"] = aux_depth_prior
            output["aux_depth_prior_valid_mask"] = aux_depth_mask
            output["aux_depth_prior_source"] = aux_depth_source
            aux_depth_weight = self._load_dense_map_prior(
                record["frame_name"],
                self._cfg.aux_depth_weight_subdir,
                target_hw,
                valid_mask=aux_depth_mask,
            )
            if aux_depth_weight is not None:
                output["aux_depth_prior_weight"] = aux_depth_weight

        feature_prior = self._load_prior(record["frame_name"], self._cfg.feature_subdir)
        if feature_prior is not None:
            output["feature_prior"] = feature_prior

        pointmap_prior = self._load_pointmap_prior(record["frame_name"])
        if pointmap_prior is not None:
            output.update(pointmap_prior)

        return output

    def _load_records(self) -> tuple[dict[str, Any], dict[str, Any]]:
        with open(self._meta_path, "r", encoding="utf-8") as handle:
            json_data = json.load(handle)

        info = {
            "bg_color": self._bg_color,
            "img_h": int(json_data["h"]),
            "img_w": int(json_data["w"]),
            "fl_x": json_data["fl_x"],
            "fl_y": json_data["fl_y"],
            "cx": json_data["cx"],
            "cy": json_data["cy"],
        }
        records = {}
        for frame in json_data["frames"]:
            frame_name = "_".join(frame["file_path"].split("/")[-2:])
            file_path = self._img_path_base / Path(frame["file_path"]).name
            records[frame_name] = {
                "frame_name": frame_name,
                "file_path": file_path,
                "img_tensor": None,
                "clean_file_path": self._clean_img_path_base / Path(frame["file_path"]).name
                if self._clean_img_path_base
                else None,
                "clean_img_tensor": None,
                "transform_matrix": torch.tensor(frame["transform_matrix"], dtype=torch.float32)[:3],
            }
        return records, info

    def _preload_images(self) -> None:
        def _load(record):
            return load_img(record["file_path"], channel=3).float() / 255.0

        io_workers = max(1, int(os.environ.get("SMOKE3D_IO_THREADS", "10")))
        max_workers = min(io_workers, max(1, self._length))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            images = list(executor.map(_load, self._records.values()))
        for key, image in zip(self._record_keys, images):
            self._records[key]["img_tensor"] = image[:3]

        if self._clean_img_path_base is None:
            return

        def _load_clean(record):
            clean_path = record["clean_file_path"]
            if clean_path is None or not clean_path.exists():
                return None
            return load_img(clean_path, channel=3).float() / 255.0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            clean_images = list(executor.map(_load_clean, self._records.values()))
        for key, image in zip(self._record_keys, clean_images):
            self._records[key]["clean_img_tensor"] = image[:3] if image is not None else None

    def _load_depth_prior(self, frame_name: str, target_hw: tuple[int, int], subdir: str | None = None):
        if self._prior_root is None:
            return None, None, None
        if not subdir:
            return None, None, None
        prior_dir = self._prior_root / subdir
        if not prior_dir.exists():
            return None, None, None

        candidate = self._find_prior_file(prior_dir, frame_name)
        if candidate is None:
            return None, None, None

        raw_prior = self._load_prior_tensor(candidate)
        depth_prior, valid_mask = self._prepare_depth_prior(raw_prior, target_hw)
        return depth_prior, valid_mask, str(candidate)

    def _load_prior(self, frame_name: str, subdir: str):
        if self._prior_root is None:
            return None
        if not subdir:
            return None
        prior_dir = self._prior_root / subdir
        if not prior_dir.exists():
            return None

        candidate = self._find_prior_file(prior_dir, frame_name)
        if candidate is not None:
            return self._load_prior_tensor(candidate)
        return None

    def _load_dense_map_prior(
        self,
        frame_name: str,
        subdir: str | None,
        target_hw: tuple[int, int],
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        prior = self._load_prior(frame_name, subdir) if subdir else None
        if prior is None:
            return None
        return self._prepare_dense_map_prior(prior, target_hw, valid_mask)

    def _find_prior_file(self, prior_dir: Path, frame_name: str) -> Path | None:
        candidates = [prior_dir / f"{frame_name}.pt", prior_dir / f"{frame_name}.npy", prior_dir / f"{frame_name}.png"]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _load_pointmap_prior(self, frame_name: str) -> dict[str, torch.Tensor | str] | None:
        if self._prior_root is None or not self._cfg.pointmap_subdir:
            return None
        pointmap_dir = self._prior_root / self._cfg.pointmap_subdir
        pointmap_path = pointmap_dir / f"{frame_name}.npz"
        if not pointmap_path.exists():
            return None

        with __import__("numpy").load(pointmap_path) as payload:
            points = torch.from_numpy(payload["points"]).float()
            colors = torch.from_numpy(payload["colors"]).float()
            depth_camera = torch.from_numpy(payload["depth_camera"]).float()
            valid_mask = torch.from_numpy(payload["valid_mask"].astype("float32"))
            confidences = torch.from_numpy(payload["confidences"]).float()
            pose_gl = torch.from_numpy(payload["pose_gl"]).float()[:3]
            intrinsics = torch.from_numpy(payload["intrinsics"]).float()

        return {
            "pointmap_prior": points,
            "pointmap_rgb_prior": colors,
            "pointmap_depth_prior": depth_camera,
            "pointmap_prior_valid_mask": valid_mask,
            "pointmap_prior_weight": confidences.clamp(0.0, 1.0) * valid_mask,
            "pointmap_pose_gl": pose_gl,
            "pointmap_intrinsics": intrinsics,
            "pointmap_prior_source": str(pointmap_path),
        }

    def _load_prior_tensor(self, candidate: Path) -> torch.Tensor:
        if candidate.suffix == ".pt":
            return torch.load(candidate, map_location="cpu")
        if candidate.suffix == ".npy":
            return torch.from_numpy(__import__("numpy").load(candidate)).float()
        if candidate.suffix == ".png":
            return load_img(candidate, channel=1).float() / 255.0
        raise ValueError(f"unsupported prior file: {candidate}")

    def _prepare_depth_prior(
        self,
        prior: torch.Tensor,
        target_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if prior.ndim == 3 and prior.shape[0] in {1, 3}:
            prior = prior[0]
        elif prior.ndim == 3 and prior.shape[-1] in {1, 3}:
            prior = prior[..., 0]
        prior = prior.float()
        if prior.ndim != 2:
            raise ValueError(f"depth prior must be 2D after squeeze, got shape={tuple(prior.shape)}")

        valid_mask = torch.isfinite(prior)
        prior = torch.nan_to_num(prior, nan=0.0, posinf=0.0, neginf=0.0)

        if tuple(prior.shape) != tuple(target_hw):
            prior = self._resize_map(prior, target_hw, self._cfg.depth_resize_mode)
            valid_mask = self._resize_map(valid_mask.float(), target_hw, "nearest") > 0.5

        if self._cfg.depth_normalize == "per_image_minmax":
            if valid_mask.any():
                valid_values = prior[valid_mask]
                prior = prior.clone()
                prior[valid_mask] = (valid_values - valid_values.min()) / (
                    valid_values.max() - valid_values.min()
                ).clamp_min(1e-6)
                prior[~valid_mask] = 0.0
            else:
                prior = torch.zeros_like(prior)
        elif self._cfg.depth_normalize != "none":
            raise ValueError(f"unsupported depth normalization mode: {self._cfg.depth_normalize}")

        if self._cfg.depth_invert:
            prior = 1.0 - prior

        return prior.float(), valid_mask

    def _prepare_dense_map_prior(
        self,
        prior: torch.Tensor,
        target_hw: tuple[int, int],
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if prior.ndim == 3 and prior.shape[0] in {1, 3}:
            prior = prior[0]
        elif prior.ndim == 3 and prior.shape[-1] in {1, 3}:
            prior = prior[..., 0]
        prior = prior.float()
        if prior.ndim != 2:
            raise ValueError(f"dense prior must be 2D after squeeze, got shape={tuple(prior.shape)}")

        prior = torch.nan_to_num(prior, nan=0.0, posinf=0.0, neginf=0.0)
        if tuple(prior.shape) != tuple(target_hw):
            prior = self._resize_map(prior, target_hw, self._cfg.depth_resize_mode)
        prior = prior.clamp(0.0, 1.0)
        if valid_mask is not None:
            prior = prior * valid_mask.float()
        return prior.float()

    def _load_clean_prior(
        self,
        record: dict[str, Any],
        target_hw: tuple[int, int],
    ) -> torch.Tensor | None:
        if self._clean_img_path_base is None:
            return None
        clean_image = record["clean_img_tensor"]
        if clean_image is None:
            clean_path = record["clean_file_path"]
            if clean_path is None or not clean_path.exists():
                return None
            clean_image = load_img(clean_path, channel=3).float() / 255.0
        clean_image = clean_image[:3].float()
        if tuple(clean_image.shape[1:]) != tuple(target_hw):
            clean_image = F.interpolate(
                clean_image.unsqueeze(0),
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )[0]
        return clean_image

    def _build_clean_confidence(self, smoky_image: torch.Tensor, clean_image: torch.Tensor) -> torch.Tensor:
        if self._cfg.clean_confidence_mode != "difference":
            raise ValueError(f"unsupported clean confidence mode: {self._cfg.clean_confidence_mode}")
        diff = (clean_image.float() - smoky_image.float()).abs().mean(dim=0).clamp(0.0, 1.0)
        confidence = (1.0 - diff) ** float(self._cfg.clean_confidence_power)
        return confidence.float()

    @staticmethod
    def _resize_map(tensor_2d: torch.Tensor, target_hw: tuple[int, int], mode: str) -> torch.Tensor:
        kwargs = {"size": target_hw, "mode": mode}
        if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
            kwargs["align_corners"] = False
        return F.interpolate(tensor_2d[None, None], **kwargs)[0, 0]


def load_img(file_name: str | Path, channel: int = 3) -> torch.Tensor:
    if channel == 3:
        mode = torchvision.io.ImageReadMode.RGB
    elif channel == 4:
        mode = torchvision.io.ImageReadMode.RGB_ALPHA
    else:
        mode = torchvision.io.ImageReadMode.GRAY
    image = torchvision.io.read_image(str(file_name), mode=mode)
    if image is None:
        raise FileNotFoundError(str(file_name))
    return image
