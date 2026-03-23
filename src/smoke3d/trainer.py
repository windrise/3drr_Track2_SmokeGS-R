from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import gsplat
import numpy as np
import torch
import yaml
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from .data import SmokeSceneDataset
from .features import DinoV2FeatureExtractor
from .geometry import load_scene_geometry_prior, resolve_geometry_prior_path
from .losses import compute_smoke_losses
from .model import SmokeAware3DGS, DualBranchSmokeGS
from .proxy import ProxyMesh


def _timestamp() -> str:
    now = datetime.now(tz=ZoneInfo("Asia/Tokyo"))
    return now.strftime("%b%d_%H%M_") + "".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 5))


def _gamma_augment_image(image: torch.Tensor, gamma: float) -> torch.Tensor:
    gamma = float(gamma)
    image = image.clamp(0.0, 1.0)
    if abs(gamma - 1.0) < 1e-8:
        return image
    return image.pow(gamma)


def _move_batch_to_device(batch: dict, device: str) -> dict:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _load_init_geometry_for_config(cfg):
    if not cfg.model.use_geometry_prior:
        return None

    prior_path = resolve_geometry_prior_path(cfg.dataset)
    if prior_path is None:
        prior_root = cfg.dataset.prior_root or "<missing PRIOR_ROOT>"
        raise FileNotFoundError(
            "geometry prior requested but not found at "
            f"{Path(prior_root) / cfg.dataset.geometry_subdir / cfg.dataset.geometry_file}"
        )

    return load_scene_geometry_prior(
        prior_path,
        max_points=cfg.model.num_init_points,
        seed=cfg.model.geometry_prior_seed,
    )


class SmokeTrainer:
    def __init__(self, experiment_cfg):
        self.cfg = experiment_cfg
        self.device = experiment_cfg.runner.device
        self._set_seed(int(self.cfg.runner.seed))
        self.output_dir = Path(experiment_cfg.runner.output_root) / experiment_cfg.exp_name / _timestamp()
        (self.output_dir / "examples").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "test").mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.jsonl"
        with open(self.output_dir / "config.yaml", "w", encoding="utf-8") as handle:
            yaml.safe_dump(self._yaml_config(), handle, sort_keys=False)

        self.train_dataset = SmokeSceneDataset(self.cfg.dataset, split="train")
        self.val_dataset = SmokeSceneDataset(self.cfg.dataset, split="val", load_images=False)
        self.test_dataset = SmokeSceneDataset(self.cfg.dataset, split="test", load_images=False)
        self.proxy_mesh = None
        self.proxy_intrinsics = None
        self.proxy_depth_near = float(getattr(self.cfg.model, "proxy_depth_near", 0.01))
        self.proxy_depth_far = float(getattr(self.cfg.model, "proxy_depth_far", 100.0))
        self._maybe_build_proxy_mesh()
        init_geometry = _load_init_geometry_for_config(self.cfg)
        self._dual_branch = bool(getattr(self.cfg.model, 'dual_branch', False))
        if self._dual_branch:
            self.model = DualBranchSmokeGS(
                self.cfg.model,
                self.train_dataset.data_info,
                init_geometry=init_geometry,
                num_train_views=len(self.train_dataset),
            ).to(self.device)
        else:
            self.model = SmokeAware3DGS(
                self.cfg.model,
                self.train_dataset.data_info,
                init_geometry=init_geometry,
                num_train_views=len(self.train_dataset),
            ).to(self.device)
        self._maybe_load_init_checkpoint()
        self.augment_gamma = float(self.cfg.runner.augment_gamma)
        self.feature_extractor = self._build_feature_extractor()

        self.optimizers = {}
        lr_map = {
            "means": self.cfg.model.lr_means,
            "quats": self.cfg.model.lr_quats,
            "scales": self.cfg.model.lr_scales,
            "opacities": self.cfg.model.lr_opacities,
            "sh0": self.cfg.model.lr_sh0,
            "shN": self.cfg.model.lr_shn,
        }
        for name, param in self.model.splats.items():
            self.optimizers[name] = torch.optim.Adam([param], lr=lr_map[name], eps=1e-15)

        if self._dual_branch:
            # Smoke branch optimizer — separate from scene
            smoke_lr = float(getattr(self.cfg.model, 'lr_smoke_branch', 5e-4))
            self.optimizers["smoke_branch"] = torch.optim.Adam(
                list(self.model.smoke_splats.parameters()),
                lr=smoke_lr,
                eps=1e-15,
            )
        else:
            smoke_param_list = list(self.model.smoke_params.parameters())
            if self.model.smoke_view_params is not None:
                smoke_param_list += list(self.model.smoke_view_params.parameters())
            if self.model.smoke_spatial_grids is not None:
                smoke_param_list.append(self.model.smoke_spatial_grids)
            self.optimizers["smoke"] = torch.optim.Adam(
                smoke_param_list,
                lr=self.cfg.model.lr_smoke,
                eps=1e-15,
            )

        self.schedulers = {
            "means": torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"],
                gamma=(self.cfg.model.lr_means_final / self.cfg.model.lr_means)
                ** (1.0 / self.cfg.runner.train_total_step),
            )
        }
        self.strategy_name = str(getattr(self.cfg.model, "strategy_type", "default")).lower()
        self.strategy = self._build_strategy()
        self.strategy_state = self._initialize_strategy_state()

    def _maybe_load_init_checkpoint(self) -> None:
        raw_path = getattr(self.cfg.runner, "init_checkpoint", None)
        if not raw_path:
            return
        checkpoint_path = Path(raw_path).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path.cwd() / checkpoint_path
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        checkpoint = dict(checkpoint)
        if not getattr(self.cfg.runner, "init_checkpoint_load_splats", True):
            checkpoint.pop("splats", None)
            checkpoint.pop("smoke_splats", None)
        if not getattr(self.cfg.runner, "init_checkpoint_load_smoke", True):
            checkpoint.pop("smoke_params", None)
            checkpoint.pop("smoke_view_params", None)
            checkpoint.pop("smoke_spatial_grids", None)
            checkpoint.pop("smoke_splats", None)
        self.model.load_checkpoint_state(checkpoint)
        # Warm-start should preserve the fully learned SH representation instead of
        # falling back to degree 0 and relearning it.
        self.model.sh_degree = self.model.sh_degree_max
        print(
            "[init] loaded checkpoint from {} (splats={}, smoke={})".format(
                checkpoint_path,
                getattr(self.cfg.runner, "init_checkpoint_load_splats", True),
                getattr(self.cfg.runner, "init_checkpoint_load_smoke", True),
            ),
            flush=True,
        )

    def _build_strategy(self):
        if self.strategy_name == "mcmc":
            return gsplat.MCMCStrategy(
                cap_max=int(self.cfg.model.mcmc_cap_max),
                noise_lr=float(self.cfg.model.mcmc_noise_lr),
                refine_start_iter=self.cfg.model.densify_start_step,
                refine_stop_iter=self.cfg.model.densify_stop_step,
                refine_every=self.cfg.model.densify_interval,
                min_opacity=float(self.cfg.model.mcmc_min_opacity),
                verbose=True,
            )
        if self.strategy_name == "default":
            return gsplat.DefaultStrategy(
                verbose=True,
                refine_start_iter=self.cfg.model.densify_start_step,
                refine_stop_iter=self.cfg.model.densify_stop_step,
                refine_every=self.cfg.model.densify_interval,
                grow_grad2d=self.cfg.model.densify_grad_thresh,
                reset_every=self.cfg.model.opacity_reset_interval,
            )
        raise ValueError(f"unsupported strategy_type: {self.cfg.model.strategy_type}")

    def _initialize_strategy_state(self):
        if self.strategy_name == "mcmc":
            return self.strategy.initialize_state()
        return self.strategy.initialize_state(scene_scale=self.cfg.model.scene_scale)

    def _scene_intrinsics_matrix(self) -> torch.Tensor:
        info = self.train_dataset.data_info
        return torch.tensor(
            [
                [info["fl_x"], 0.0, info["cx"]],
                [0.0, info["fl_y"], info["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=self.device,
        )

    def _resolve_proxy_pointmap_dir(self) -> Path | None:
        if not self.cfg.dataset.prior_root:
            return None
        subdir = (
            getattr(self.cfg.dataset, "proxy_pointmap_subdir", None)
            or self.cfg.dataset.pointmap_subdir
        )
        if not subdir:
            return None
        pointmap_dir = Path(self.cfg.dataset.prior_root) / subdir
        return pointmap_dir if pointmap_dir.exists() else None

    def _maybe_build_proxy_mesh(self) -> None:
        proxy_depth_weight = float(getattr(self.cfg.loss, "proxy_depth_weight", 0.0))
        proxy_mesh_path = getattr(self.cfg.runner, "proxy_mesh_path", None)
        if proxy_depth_weight <= 0 and not proxy_mesh_path:
            return

        prefer_simple_renderer = bool(getattr(self.cfg.model, "proxy_force_simple_renderer", True))
        if proxy_mesh_path:
            self.proxy_mesh = ProxyMesh.from_mesh_file(
                proxy_mesh_path,
                device=self.device,
                prefer_simple_renderer=prefer_simple_renderer,
            )
        else:
            pointmap_dir = self._resolve_proxy_pointmap_dir()
            if pointmap_dir is None:
                raise FileNotFoundError(
                    "proxy depth supervision requested but no pointmap directory is available"
                )
            self.proxy_mesh = ProxyMesh.from_pointmap_directory(
                str(pointmap_dir),
                frame_names=self.train_dataset.record_keys,
                stride=int(getattr(self.cfg.model, "proxy_build_stride", 8)),
                confidence_threshold=float(
                    getattr(self.cfg.model, "proxy_confidence_threshold", 0.0)
                ),
                edge_threshold_scale=float(
                    getattr(self.cfg.model, "proxy_edge_threshold_scale", 4.0)
                ),
                max_views=int(getattr(self.cfg.model, "proxy_max_views", 0)),
                device=self.device,
                prefer_simple_renderer=prefer_simple_renderer,
            )

        self.proxy_intrinsics = self._scene_intrinsics_matrix()
        if getattr(self.cfg.runner, "proxy_precompute_depths", True):
            self.proxy_mesh.precompute_depths(
                self.train_dataset.get_transforms(),
                intrinsics=self.proxy_intrinsics,
                H=int(self.train_dataset.data_info["img_h"]),
                W=int(self.train_dataset.data_info["img_w"]),
                near=self.proxy_depth_near,
                far=self.proxy_depth_far,
                cache_device="cpu",
            )
        print("[proxy] mesh ready for training", flush=True)

    def _maybe_attach_proxy_depth(self, batch: dict) -> None:
        if self.proxy_mesh is None or float(getattr(self.cfg.loss, "proxy_depth_weight", 0.0)) <= 0:
            return
        view_index = batch.get("view_index")
        if view_index is None:
            return
        if torch.is_tensor(view_index):
            view_idx_int = int(view_index.reshape(-1)[0].item())
        else:
            view_idx_int = int(view_index)

        proxy_depth = self.proxy_mesh.get_cached_depth(view_idx_int)
        if proxy_depth is None:
            img_h, img_w = batch["images"].shape[1:]
            proxy_depth = self.proxy_mesh.render_depth(
                batch["transforms"],
                self.proxy_intrinsics,
                img_h,
                img_w,
                near=self.proxy_depth_near,
                far=self.proxy_depth_far,
            ).detach().cpu()
            self.proxy_mesh._depth_cache[view_idx_int] = proxy_depth
        batch["proxy_depth_prior"] = proxy_depth.to(device=self.device, dtype=torch.float32)

    def _set_seed(self, seed: int) -> None:
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_feature_extractor(self):
        if self.cfg.loss.feature_weight <= 0:
            return None
        repo_path = Path(self.cfg.loss.feature_repo_path)
        if not repo_path.is_absolute():
            repo_path = Path.cwd() / repo_path
        extractor = DinoV2FeatureExtractor(
            repo_path=repo_path,
            model_name=self.cfg.loss.feature_model_name,
            input_size=self.cfg.loss.feature_input_size,
        )
        return extractor.to(self.device).eval()

    def _yaml_config(self) -> dict:
        cfg = asdict(self.cfg)
        return {
            "EXP_NAME": cfg["exp_name"],
            "DATASET": {k.upper(): v for k, v in cfg["dataset"].items()},
            "MODEL": {k.upper(): v for k, v in cfg["model"].items()},
            "LOSS": {k.upper(): v for k, v in cfg["loss"].items()},
            "RUNNER": {k.upper(): v for k, v in cfg["runner"].items()},
        }

    def train(self) -> Path:
        total_steps = self.cfg.runner.train_total_step
        pbar = tqdm(range(total_steps), desc=self.cfg.exp_name)
        for step in pbar:
            if self._dual_branch:
                self.model.set_step(step)
                # SmokeSeer-inspired: optionally freeze scene geometry when smoke activates
                if step == self.model.smoke_warmup_steps:
                    if self.cfg.model.freeze_scene_on_warmup:
                        self.model.freeze_scene_geometry()
                        for frozen_name in ["means", "quats", "scales"]:
                            self.optimizers.pop(frozen_name, None)
                        self.schedulers.pop("means", None)
                        print(f"[step {step}] Froze scene geometry, smoke branch active", flush=True)
                    else:
                        print(f"[step {step}] Smoke branch activated (no scene freeze)", flush=True)

            if step > 0 and step % self.cfg.runner.sh_upgrade_interval == 0:
                self.model.sh_degree = min(self.model.sh_degree + 1, self.model.sh_degree_max)

            batch = self.train_dataset[random.randint(0, len(self.train_dataset) - 1)]
            batch = _move_batch_to_device(batch, self.device)
            image = _gamma_augment_image(batch["images"], self.augment_gamma)
            batch["images"] = image
            if "clean_prior" in batch:
                batch["clean_prior"] = _gamma_augment_image(batch["clean_prior"], self.augment_gamma)
            self._maybe_attach_proxy_depth(batch)
            camtoworld = batch["transforms"]
            height, width = image.shape[1], image.shape[2]

            prediction = self.model(
                camtoworld,
                height,
                width,
                render_clean=self.cfg.runner.train_render_clean,
                view_index=batch.get("view_index"),
            )
            _feature_delay = getattr(self.cfg.loss, 'feature_delay_step', 0)
            if self.feature_extractor is not None and "feature_prior" in batch and step >= _feature_delay:
                feature_source = prediction[self.cfg.loss.feature_render_source]
                prediction["feature_map"] = self.feature_extractor(feature_source.permute(2, 0, 1).unsqueeze(0))[0]
            _need_pointmap_render = (
                (self.cfg.loss.pointmap_weight > 0
                 or self.cfg.loss.pointmap_rgb_weight > 0
                 or self.cfg.loss.pointmap_depth_weight > 0)
                and "pointmap_prior" in batch
            )
            if _need_pointmap_render:
                point_h, point_w = batch["pointmap_prior"].shape[:2]
                point_prediction = self.model(
                    batch["pointmap_pose_gl"],
                    point_h,
                    point_w,
                    render_clean=self.cfg.runner.train_render_clean,
                    intrinsics_override=batch["pointmap_intrinsics"],
                    view_index=batch.get("view_index"),
                )
                prediction["pointmap_rendered"] = point_prediction["rendered"]
                prediction["pointmap_clean_rgb"] = point_prediction["clean_rgb"]
                prediction["pointmap_expected_depth"] = point_prediction["expected_depth"]
            loss_dict = compute_smoke_losses(prediction, batch, self.cfg.loss, step=step)

            if self.strategy_name != "mcmc":
                self.strategy.step_pre_backward(
                    self.model.splats, self.optimizers, self.strategy_state, step, prediction["info"]
                )
            loss_dict["total"].backward()
            if self.strategy_name == "mcmc":
                self.strategy.step_post_backward(
                    self.model.splats,
                    self.optimizers,
                    self.strategy_state,
                    step,
                    prediction["info"],
                    lr=self.optimizers["means"].param_groups[0]["lr"],
                )
            else:
                self.strategy.step_post_backward(
                    self.model.splats,
                    self.optimizers,
                    self.strategy_state,
                    step,
                    prediction["info"],
                    packed=False,
                )

            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in self.schedulers.values():
                scheduler.step()

            if step % self.cfg.runner.log_interval_step == 0:
                with torch.no_grad():
                    mse = ((prediction["rendered"] - image.permute(1, 2, 0)) ** 2).mean()
                    psnr = -10.0 * math.log10(mse.clamp_min(1e-10).item())
                postfix = {
                    "loss": f"{loss_dict['total'].item():.4f}",
                    "psnr": f"{psnr:.2f}",
                    "n_gs": self.model.num_gaussians,
                }
                postfix["strat"] = self.strategy_name
                if "depth" in loss_dict:
                    postfix["depth"] = f"{loss_dict['depth'].item():.4f}"
                if "clean" in loss_dict:
                    postfix["clean"] = f"{loss_dict['clean'].item():.4f}"
                if "clean_weight_effective" in loss_dict:
                    postfix["cw"] = f"{loss_dict['clean_weight_effective'].item():.3f}"
                if "pointmap" in loss_dict:
                    postfix["point"] = f"{loss_dict['pointmap'].item():.4f}"
                if "pointmap_rgb" in loss_dict:
                    postfix["prgb"] = f"{loss_dict['pointmap_rgb'].item():.4f}"
                if "pointmap_depth" in loss_dict:
                    postfix["pdep"] = f"{loss_dict['pointmap_depth'].item():.4f}"
                # Dual-branch vitality metrics
                if self._dual_branch and prediction.get("smoke_active", False):
                    sa_mean = prediction["smoke_alpha"].mean().item()
                    postfix["sm_α"] = f"{sa_mean:.4f}"
                    if "exclusion" in loss_dict:
                        postfix["excl"] = f"{loss_dict['exclusion'].item():.4f}"
                    if "smoke_opacity_var" in loss_dict:
                        postfix["op_var"] = f"{loss_dict['smoke_opacity_var'].item():.6f}"
                    if "smoke_color_var" in loss_dict:
                        postfix["cl_var"] = f"{loss_dict['smoke_color_var'].item():.6f}"
                    # smoky-clean diff: how much smoke branch contributes
                    sc_diff = (prediction["smoky_rgb"] - prediction["clean_rgb"]).abs().mean().item()
                    postfix["sc_Δ"] = f"{sc_diff:.4f}"
                pbar.set_postfix(**postfix)
                self._append_metrics(step=step, psnr=psnr, loss_dict=loss_dict)
                # Additional dual-branch vitality logging to metrics
                if self._dual_branch and prediction.get("smoke_active", False):
                    self._append_extra_metrics(step, prediction)

            if step > 0 and step % self.cfg.runner.val_interval_step == 0:
                self._save_preview(step)
                self.save_checkpoint()

        checkpoint = self.save_checkpoint()
        self.render_test(self.cfg.runner.render_clean_on_test)
        return checkpoint

    def _append_metrics(self, step: int, psnr: float, loss_dict: dict[str, torch.Tensor]) -> None:
        payload = {
            "step": int(step),
            "psnr": float(psnr),
            "num_gaussians": int(self.model.num_gaussians),
        }
        for key, value in loss_dict.items():
            if torch.is_tensor(value):
                payload[key] = float(value.detach().item())
        with open(self.metrics_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _append_extra_metrics(self, step: int, prediction: dict) -> None:
        """Log dual-branch vitality stats to a separate file."""
        vitality_path = self.output_dir / "smoke_vitality.jsonl"
        payload = {
            "step": int(step),
            "smoke_alpha_mean": float(prediction["smoke_alpha"].mean().item()),
            "smoke_alpha_max": float(prediction["smoke_alpha"].max().item()),
            "smoke_opacity_var": float(prediction["smoke_opacity_var"].item()),
            "smoke_color_var": float(prediction["smoke_color_var"].item()),
            "smoky_clean_diff": float(
                (prediction["smoky_rgb"] - prediction["clean_rgb"]).abs().mean().item()
            ),
        }
        with open(vitality_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    @torch.no_grad()
    def _save_preview(self, step: int) -> None:
        self.model.eval()
        height = self.val_dataset.data_info["img_h"]
        width = self.val_dataset.data_info["img_w"]
        previews = []
        for index in range(len(self.val_dataset)):
            batch = _move_batch_to_device(self.val_dataset[index], self.device)
            prediction = self.model(batch["transforms"], height, width, render_clean=True)
            previews.append(prediction["clean_rgb"].permute(2, 0, 1).clamp(0, 1))
        if previews:
            grid = make_grid(previews, nrow=2)
            save_image(grid, self.output_dir / "examples" / f"val_step{step}.jpg")
        self.model.train()

    def save_checkpoint(self) -> Path:
        checkpoint_path = self.output_dir / "latest.pt"
        torch.save(
            {
                "config": self._yaml_config(),
                **self.model.checkpoint_state(),
            },
            checkpoint_path,
        )
        return checkpoint_path

    @torch.no_grad()
    def render_test(self, render_clean: bool = True) -> None:
        self.model.eval()
        height = self.test_dataset.data_info["img_h"]
        width = self.test_dataset.data_info["img_w"]
        for index, frame_name in enumerate(self.test_dataset.record_keys):
            batch = _move_batch_to_device(self.test_dataset[index], self.device)
            prediction = self.model(batch["transforms"], height, width, render_clean=render_clean)
            save_image(
                prediction["rendered"].permute(2, 0, 1).clamp(0, 1),
                self.output_dir / "test" / f"{frame_name}.png",
            )
        self.model.train()


@torch.no_grad()
def render_checkpoint(
    checkpoint_path: str,
    config_loader,
    output_dir: str | None = None,
    data_path_override: str | None = None,
    device: str = "cuda",
    render_clean: bool = True,
) -> str:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = config_loader(Path(checkpoint_path).with_name("config.yaml"))
    if data_path_override is not None:
        cfg.dataset.data_path = data_path_override

    dataset = SmokeSceneDataset(cfg.dataset, split="test", load_images=False)
    train_view_count = len(SmokeSceneDataset(cfg.dataset, split="train", load_images=False)) if cfg.model.per_view_smoke else None
    init_geometry = _load_init_geometry_for_config(cfg)
    model = SmokeAware3DGS(
        cfg.model,
        dataset.data_info,
        init_geometry=init_geometry,
        num_train_views=train_view_count,
    ).to(device)
    model.load_checkpoint_state(checkpoint)
    model.sh_degree = model.sh_degree_max
    model.eval()

    out_dir = Path(output_dir) if output_dir else Path(checkpoint_path).parent / "test"
    out_dir.mkdir(parents=True, exist_ok=True)
    height = dataset.data_info["img_h"]
    width = dataset.data_info["img_w"]

    for index, frame_name in enumerate(dataset.record_keys):
        batch = _move_batch_to_device(dataset[index], device)
        prediction = model(batch["transforms"], height, width, render_clean=render_clean)
        save_image(
            prediction["rendered"].permute(2, 0, 1).clamp(0, 1),
            out_dir / f"{frame_name}.png",
        )
    return str(out_dir)
