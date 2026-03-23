from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from gsplat import rasterization


class SmokeAware3DGS(nn.Module):
    def __init__(
        self,
        model_cfg,
        data_info,
        init_geometry: dict[str, torch.Tensor | str] | None = None,
        num_train_views: int | None = None,
    ):
        super().__init__()
        self.path_length_mode = model_cfg.path_length_mode
        self.fl_x = data_info["fl_x"]
        self.fl_y = data_info["fl_y"]
        self.cx = data_info["cx"]
        self.cy = data_info["cy"]
        self.bg_color = data_info["bg_color"]
        self.sh_degree_max = model_cfg.sh_degree
        self.sh_degree = 0
        self.min_transmission = model_cfg.min_transmission
        self.per_view_smoke = bool(model_cfg.per_view_smoke and num_train_views and num_train_views > 0)
        self.num_train_views = int(num_train_views or 0)
        self.spatial_smoke_grid_size = int(getattr(model_cfg, 'spatial_smoke_grid_size', 0))

        means, sh0 = self._initialize_scene_primitives(model_cfg, init_geometry)
        num_points = means.shape[0]
        num_sh_bases = (self.sh_degree_max + 1) ** 2

        quats = torch.zeros(num_points, 4)
        quats[:, 0] = 1.0
        scales = torch.log(torch.full((num_points, 3), float(model_cfg.init_scale)))
        opacities = torch.logit(torch.full((num_points,), float(model_cfg.init_opacity)).clamp(1e-4, 1 - 1e-4))
        shN = torch.zeros(num_points, num_sh_bases - 1, 3)

        self.splats = nn.ParameterDict(
            {
                "means": nn.Parameter(means),
                "quats": nn.Parameter(quats),
                "scales": nn.Parameter(scales),
                "opacities": nn.Parameter(opacities),
                "sh0": nn.Parameter(sh0),
                "shN": nn.Parameter(shN),
            }
        )

        init_airlight = torch.tensor(model_cfg.init_airlight, dtype=torch.float32).clamp(1e-4, 1 - 1e-4)
        self.smoke_params = nn.ParameterDict(
            {
                "beta_raw": nn.Parameter(torch.tensor(float(model_cfg.init_beta)).log()),
                "airlight_logits": nn.Parameter(torch.logit(init_airlight)),
                "transmission_bias_raw": nn.Parameter(
                    torch.tensor(float(model_cfg.init_transmission_bias)).log()
                ),
            }
        )
        if self.per_view_smoke:
            self.smoke_view_params = nn.ParameterDict(
                {
                    "beta_raw_offsets": nn.Parameter(torch.zeros(self.num_train_views)),
                    "airlight_logits_offsets": nn.Parameter(torch.zeros(self.num_train_views, 3)),
                    "transmission_bias_raw_offsets": nn.Parameter(torch.zeros(self.num_train_views)),
                }
            )
        else:
            self.smoke_view_params = None

        # Spatial smoke: per-view low-res learnable transmission residual grid
        if self.spatial_smoke_grid_size > 0 and self.num_train_views > 0:
            gs = self.spatial_smoke_grid_size
            # Initialize to zero (no residual), so it starts identical to global model
            self.smoke_spatial_grids = nn.Parameter(
                torch.zeros(self.num_train_views, 1, gs, gs)
            )
        else:
            self.smoke_spatial_grids = None

    @staticmethod
    def _initialize_scene_primitives(model_cfg, init_geometry: dict[str, torch.Tensor | str] | None):
        if init_geometry is None:
            num_points = model_cfg.num_init_points
            means = (torch.rand(num_points, 3) - 0.5) * 10.0
            sh0 = torch.zeros(num_points, 1, 3)
            return means, sh0

        means = init_geometry["points"].float().clone()
        colors = init_geometry.get("colors")
        if colors is not None:
            sh0 = colors.float().clone().clamp(0.0, 1.0)[:, None, :]
        else:
            sh0 = torch.zeros(means.shape[0], 1, 3)
        return means, sh0

    @property
    def num_gaussians(self) -> int:
        return self.splats["means"].shape[0]

    def _camera_inputs(self, camtoworld: torch.Tensor, intrinsics_override: torch.Tensor | None = None):
        device = self.splats["means"].device
        c2w = torch.eye(4, device=device, dtype=torch.float32)
        if camtoworld.shape == (4, 4):
            c2w = camtoworld.to(device=device, dtype=torch.float32)
        else:
            c2w[:3, :] = camtoworld.to(device)
        viewmat = torch.linalg.inv(c2w)
        viewmat[1, :] *= -1
        viewmat[2, :] *= -1
        viewmat = viewmat[None]

        if intrinsics_override is None:
            intrinsics = torch.tensor(
                [
                    [self.fl_x, 0.0, self.cx],
                    [0.0, self.fl_y, self.cy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
                device=device,
            )[None]
        else:
            intrinsics = intrinsics_override.to(device=device, dtype=torch.float32)
            if intrinsics.ndim == 2:
                intrinsics = intrinsics[None]
        return viewmat, intrinsics

    def render_clean(
        self,
        camtoworld: torch.Tensor,
        img_h: int,
        img_w: int,
        intrinsics_override: torch.Tensor | None = None,
    ):
        device = self.splats["means"].device
        viewmat, intrinsics = self._camera_inputs(camtoworld, intrinsics_override=intrinsics_override)
        colors = torch.cat([self.splats["sh0"], self.splats["shN"]], dim=1)
        bg = torch.full((1, 3), self.bg_color, dtype=torch.float32, device=device)
        render_mode = "RGB+ED" if self.path_length_mode == "expected_depth" else "RGB"
        renders, alphas, info = rasterization(
            means=self.splats["means"],
            quats=self.splats["quats"],
            scales=torch.exp(self.splats["scales"]),
            opacities=torch.sigmoid(self.splats["opacities"]),
            colors=colors,
            viewmats=viewmat,
            Ks=intrinsics,
            width=img_w,
            height=img_h,
            sh_degree=self.sh_degree,
            backgrounds=bg,
            render_mode=render_mode,
            packed=False,
        )
        rendered = renders[0]
        if self.path_length_mode == "expected_depth":
            return rendered[..., :3], alphas[0], info, rendered[..., 3:4]
        return rendered, alphas[0], info, None

    def _compose_smoke(
        self,
        clean_rgb: torch.Tensor,
        alphas: torch.Tensor,
        expected_depth: torch.Tensor | None,
        render_clean: bool,
        view_index: torch.Tensor | int | None = None,
    ):
        beta_raw = self.smoke_params["beta_raw"]
        airlight_logits = self.smoke_params["airlight_logits"]
        bias_raw = self.smoke_params["transmission_bias_raw"]
        view_idx_int = None
        if self.smoke_view_params is not None and view_index is not None:
            if torch.is_tensor(view_index):
                view_idx_int = int(view_index.reshape(-1)[0].item())
            else:
                view_idx_int = int(view_index)
            if 0 <= view_idx_int < self.num_train_views:
                beta_raw = beta_raw + self.smoke_view_params["beta_raw_offsets"][view_idx_int]
                airlight_logits = airlight_logits + self.smoke_view_params["airlight_logits_offsets"][view_idx_int]
                bias_raw = bias_raw + self.smoke_view_params["transmission_bias_raw_offsets"][view_idx_int]

        beta = F.softplus(beta_raw)
        airlight = torch.sigmoid(airlight_logits).view(1, 1, 3)
        bias = F.softplus(bias_raw)

        if self.path_length_mode == "expected_depth":
            if expected_depth is None:
                raise ValueError("expected_depth path length mode requires RGB+ED rendering")
            path_length = expected_depth.clamp_min(0.0)
        else:
            path_length = alphas.clamp(0.0, 1.0)

        path_mass = path_length + bias
        transmission = torch.exp(-beta * path_mass).clamp(self.min_transmission, 1.0)

        # Spatial smoke: add per-pixel transmission residual from low-res grid
        if (self.smoke_spatial_grids is not None
                and view_idx_int is not None
                and 0 <= view_idx_int < self.num_train_views):
            h, w = transmission.shape[:2]
            grid = self.smoke_spatial_grids[view_idx_int]  # [1, gs, gs]
            # Bilinear upsample to rendering resolution
            spatial_residual = F.interpolate(
                grid.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
            )[0, 0]  # [H, W]
            # Apply as multiplicative adjustment: transmission *= sigmoid(residual)
            # sigmoid(0)=0.5, so we scale by 2 to keep mean=1 at init
            spatial_factor = 2.0 * torch.sigmoid(spatial_residual).unsqueeze(-1)  # [H, W, 1]
            transmission = (transmission * spatial_factor).clamp(self.min_transmission, 1.0)

        smoky_rgb = clean_rgb * transmission + airlight * (1.0 - transmission)

        if render_clean:
            rendered = clean_rgb
        else:
            rendered = smoky_rgb

        return rendered, smoky_rgb, transmission, airlight, path_length

    def forward(
        self,
        camtoworld: torch.Tensor,
        img_h: int,
        img_w: int,
        render_clean: bool = False,
        intrinsics_override: torch.Tensor | None = None,
        view_index: torch.Tensor | int | None = None,
    ):
        clean_rgb, alphas, info, expected_depth = self.render_clean(
            camtoworld,
            img_h,
            img_w,
            intrinsics_override=intrinsics_override,
        )
        rendered, smoky_rgb, transmission, airlight, path_length = self._compose_smoke(
            clean_rgb,
            alphas,
            expected_depth,
            render_clean,
            view_index=view_index,
        )
        return {
            "rendered": rendered,
            "clean_rgb": clean_rgb,
            "smoky_rgb": smoky_rgb,
            "transmission": transmission,
            "alphas": alphas,
            "expected_depth": expected_depth,
            "path_length": path_length,
            "geometry_proxy": alphas[..., 0],
            "airlight": airlight,
            "info": info,
            "splats_ref": self.splats,  # for scale/opacity hardening loss
        }

    def checkpoint_state(self) -> dict[str, dict[str, torch.Tensor]]:
        state = {
            "splats": self.splats.state_dict(),
            "smoke_params": self.smoke_params.state_dict(),
        }
        if self.smoke_view_params is not None:
            state["smoke_view_params"] = self.smoke_view_params.state_dict()
        if self.smoke_spatial_grids is not None:
            state["smoke_spatial_grids"] = self.smoke_spatial_grids.data
        return state

    def load_checkpoint_state(self, checkpoint: dict[str, dict[str, torch.Tensor]]) -> None:
        splat_device = self.splats["means"].device
        if "splats" in checkpoint:
            splat_state = checkpoint["splats"]
            self.splats = nn.ParameterDict(
                {
                    name: nn.Parameter(param.to(device=splat_device, dtype=torch.float32))
                    for name, param in splat_state.items()
                }
            )
        if "smoke_params" in checkpoint:
            self.smoke_params.load_state_dict(checkpoint["smoke_params"])
        if self.smoke_view_params is not None and "smoke_view_params" in checkpoint:
            self.smoke_view_params.load_state_dict(checkpoint["smoke_view_params"])
        if self.smoke_spatial_grids is not None and "smoke_spatial_grids" in checkpoint:
            self.smoke_spatial_grids.data.copy_(checkpoint["smoke_spatial_grids"].to(splat_device))


class DualBranchSmokeGS(nn.Module):
    """True dual-branch 3DGS: separate scene Gaussians + smoke Gaussians.

    Scene branch: standard 3DGS representing the clean scene.
    Smoke branch: separate 3DGS representing the scattering medium.
    Composition: physics-based transmission model.
    """

    def __init__(
        self,
        model_cfg,
        data_info,
        init_geometry: dict[str, torch.Tensor | str] | None = None,
        num_train_views: int | None = None,
    ):
        super().__init__()
        self.fl_x = data_info["fl_x"]
        self.fl_y = data_info["fl_y"]
        self.cx = data_info["cx"]
        self.cy = data_info["cy"]
        self.bg_color = data_info["bg_color"]
        self.sh_degree_max = model_cfg.sh_degree
        self.sh_degree = 0
        self.min_transmission = model_cfg.min_transmission

        # Dual-branch config
        self.smoke_warmup_steps = int(getattr(model_cfg, 'smoke_warmup_steps', 500))
        self.smoke_num_points = int(getattr(model_cfg, 'smoke_num_points', 10000))
        self.smoke_init_opacity = float(getattr(model_cfg, 'smoke_init_opacity', 0.01))
        self.smoke_init_scale = float(getattr(model_cfg, 'smoke_init_scale', 0.1))
        self._current_step = 0

        # ======== Scene Branch (clean) ========
        means, sh0 = SmokeAware3DGS._initialize_scene_primitives(model_cfg, init_geometry)
        num_points = means.shape[0]
        num_sh_bases = (self.sh_degree_max + 1) ** 2

        quats = torch.zeros(num_points, 4)
        quats[:, 0] = 1.0
        scales = torch.log(torch.full((num_points, 3), float(model_cfg.init_scale)))
        opacities = torch.logit(torch.full((num_points,), float(model_cfg.init_opacity)).clamp(1e-4, 1 - 1e-4))
        shN = torch.zeros(num_points, num_sh_bases - 1, 3)

        self.splats = nn.ParameterDict(
            {
                "means": nn.Parameter(means),
                "quats": nn.Parameter(quats),
                "scales": nn.Parameter(scales),
                "opacities": nn.Parameter(opacities),
                "sh0": nn.Parameter(sh0),
                "shN": nn.Parameter(shN),
            }
        )

        # ======== Smoke Branch (medium) ========
        # SmokeSeer-inspired: initialize smoke from scene points (downsample + noise)
        n_smoke = min(self.smoke_num_points, num_points)
        indices = torch.randperm(num_points)[:n_smoke]
        smoke_means = means[indices].clone()
        # Add noise proportional to scene extent
        scene_extent = (means.max(dim=0).values - means.min(dim=0).values).max().item()
        smoke_means += torch.randn_like(smoke_means) * scene_extent * 0.05
        # If we need more points, add random ones
        if n_smoke < self.smoke_num_points:
            extra = self.smoke_num_points - n_smoke
            scene_center = means.mean(dim=0)
            extra_means = scene_center + (torch.rand(extra, 3) - 0.5) * scene_extent * 1.2
            smoke_means = torch.cat([smoke_means, extra_means], dim=0)

        actual_smoke_n = smoke_means.shape[0]
        smoke_quats = torch.zeros(actual_smoke_n, 4)
        smoke_quats[:, 0] = 1.0
        # Isotropic scale — smoke is amorphous
        smoke_scales = torch.log(torch.full((actual_smoke_n, 3), self.smoke_init_scale))
        # Very low opacity — each smoke Gaussian contributes small density
        smoke_opacities = torch.logit(
            torch.full((actual_smoke_n,), self.smoke_init_opacity).clamp(1e-4, 1 - 1e-4)
        )
        # Smoke color: SH degree 0 only (view-independent scattering)
        # Initialize to gray (airlight-like)
        smoke_sh0 = torch.full((actual_smoke_n, 1, 3), 0.7)

        self.smoke_splats = nn.ParameterDict(
            {
                "means": nn.Parameter(smoke_means),
                "quats": nn.Parameter(smoke_quats),
                "scales": nn.Parameter(smoke_scales),
                "opacities": nn.Parameter(smoke_opacities),
                "sh0": nn.Parameter(smoke_sh0),
            }
        )

    @property
    def num_gaussians(self) -> int:
        return self.splats["means"].shape[0]

    @property
    def num_smoke_gaussians(self) -> int:
        return self.smoke_splats["means"].shape[0]

    def set_step(self, step: int):
        self._current_step = step

    def freeze_scene_geometry(self):
        """SmokeSeer-inspired: freeze scene branch positions after warmup.
        Only keeps colors (sh0, shN) and opacities trainable."""
        self.splats["means"].requires_grad_(False)
        self.splats["quats"].requires_grad_(False)
        self.splats["scales"].requires_grad_(False)

    def _camera_inputs(self, camtoworld: torch.Tensor, intrinsics_override: torch.Tensor | None = None):
        device = self.splats["means"].device
        c2w = torch.eye(4, device=device, dtype=torch.float32)
        if camtoworld.shape == (4, 4):
            c2w = camtoworld.to(device=device, dtype=torch.float32)
        else:
            c2w[:3, :] = camtoworld.to(device)
        viewmat = torch.linalg.inv(c2w)
        viewmat[1, :] *= -1
        viewmat[2, :] *= -1
        viewmat = viewmat[None]

        if intrinsics_override is None:
            intrinsics = torch.tensor(
                [
                    [self.fl_x, 0.0, self.cx],
                    [0.0, self.fl_y, self.cy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
                device=device,
            )[None]
        else:
            intrinsics = intrinsics_override.to(device=device, dtype=torch.float32)
            if intrinsics.ndim == 2:
                intrinsics = intrinsics[None]
        return viewmat, intrinsics

    def render_clean(
        self,
        camtoworld: torch.Tensor,
        img_h: int,
        img_w: int,
        intrinsics_override: torch.Tensor | None = None,
    ):
        """Render clean scene from scene branch only."""
        device = self.splats["means"].device
        viewmat, intrinsics = self._camera_inputs(camtoworld, intrinsics_override=intrinsics_override)
        colors = torch.cat([self.splats["sh0"], self.splats["shN"]], dim=1)
        bg = torch.full((1, 3), self.bg_color, dtype=torch.float32, device=device)
        renders, alphas, info = rasterization(
            means=self.splats["means"],
            quats=self.splats["quats"],
            scales=torch.exp(self.splats["scales"]),
            opacities=torch.sigmoid(self.splats["opacities"]),
            colors=colors,
            viewmats=viewmat,
            Ks=intrinsics,
            width=img_w,
            height=img_h,
            sh_degree=self.sh_degree,
            backgrounds=bg,
            render_mode="RGB+ED",
            packed=False,
        )
        rendered = renders[0]
        clean_rgb = rendered[..., :3]
        expected_depth = rendered[..., 3:4]
        return clean_rgb, alphas[0], info, expected_depth

    def render_smoke(
        self,
        camtoworld: torch.Tensor,
        img_h: int,
        img_w: int,
        intrinsics_override: torch.Tensor | None = None,
    ):
        """Render smoke branch: density map + smoke color."""
        device = self.smoke_splats["means"].device
        viewmat, intrinsics = self._camera_inputs(camtoworld, intrinsics_override=intrinsics_override)
        bg = torch.zeros(1, 3, dtype=torch.float32, device=device)
        # Smoke has SH degree 0 only
        renders, alphas, info = rasterization(
            means=self.smoke_splats["means"],
            quats=self.smoke_splats["quats"],
            scales=torch.exp(self.smoke_splats["scales"]),
            opacities=torch.sigmoid(self.smoke_splats["opacities"]),
            colors=self.smoke_splats["sh0"],
            viewmats=viewmat,
            Ks=intrinsics,
            width=img_w,
            height=img_h,
            sh_degree=0,
            backgrounds=bg,
            render_mode="RGB+ED",
            packed=False,
        )
        rendered = renders[0]
        smoke_color = rendered[..., :3]  # [H, W, 3] — smoke scattering color
        smoke_depth = rendered[..., 3:4]  # [H, W, 1]
        smoke_alpha = alphas[0]  # [H, W, 1] — accumulated opacity = density
        return smoke_color, smoke_alpha, smoke_depth, info

    def forward(
        self,
        camtoworld: torch.Tensor,
        img_h: int,
        img_w: int,
        render_clean: bool = False,
        intrinsics_override: torch.Tensor | None = None,
        view_index: torch.Tensor | int | None = None,
    ):
        # Render scene branch
        clean_rgb, scene_alphas, scene_info, expected_depth = self.render_clean(
            camtoworld, img_h, img_w, intrinsics_override=intrinsics_override
        )

        # Render smoke branch (only after warmup)
        smoke_active = self._current_step >= self.smoke_warmup_steps
        if smoke_active:
            smoke_color, smoke_alpha, smoke_depth, smoke_info = self.render_smoke(
                camtoworld, img_h, img_w, intrinsics_override=intrinsics_override
            )
        else:
            # During warmup: no smoke contribution
            smoke_color = torch.zeros_like(clean_rgb)
            smoke_alpha = torch.zeros(img_h, img_w, 1, device=clean_rgb.device)
            smoke_depth = torch.zeros(img_h, img_w, 1, device=clean_rgb.device)

        # Physics-based composition
        # smoke_alpha ∈ [0, 1] represents accumulated smoke density
        # transmission = 1 - smoke_alpha (how much light passes through)
        transmission = (1.0 - smoke_alpha).clamp(self.min_transmission, 1.0)

        # Atmospheric scattering: rendered = clean * T + smoke_color * (1 - T)
        # smoke_color already contains the scattered light color
        smoky_rgb = clean_rgb * transmission + smoke_color * (1.0 - transmission)

        if render_clean:
            rendered = clean_rgb
        else:
            rendered = smoky_rgb

        # Construct airlight from smoke color mean (for compatibility)
        airlight = smoke_color.mean(dim=(0, 1), keepdim=True).unsqueeze(0) if smoke_active else \
            torch.full((1, 1, 3), 0.85, device=clean_rgb.device)

        return {
            "rendered": rendered,
            "clean_rgb": clean_rgb,
            "smoky_rgb": smoky_rgb,
            "transmission": transmission,
            "alphas": scene_alphas,
            "expected_depth": expected_depth,
            "path_length": smoke_alpha,  # smoke density as path_length analogue
            "geometry_proxy": scene_alphas[..., 0],
            "airlight": airlight,
            "info": scene_info,
            # Dual-branch specific outputs
            "smoke_alpha": smoke_alpha,
            "smoke_color": smoke_color,
            "scene_alphas": scene_alphas,
            "smoke_active": smoke_active,
            # SmokeSeer-inspired variance stats for regularization
            "smoke_opacity_var": torch.var(torch.sigmoid(self.smoke_splats["opacities"]), dim=0).mean() if smoke_active else torch.tensor(0.0, device=clean_rgb.device),
            "smoke_color_var": torch.var(self.smoke_splats["sh0"], dim=0).mean() if smoke_active else torch.tensor(0.0, device=clean_rgb.device),
        }

    def checkpoint_state(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "splats": self.splats.state_dict(),
            "smoke_splats": self.smoke_splats.state_dict(),
        }

    def load_checkpoint_state(self, checkpoint: dict[str, dict[str, torch.Tensor]]) -> None:
        splat_device = self.splats["means"].device
        if "splats" in checkpoint:
            self.splats = nn.ParameterDict(
                {
                    name: nn.Parameter(param.to(device=splat_device, dtype=torch.float32))
                    for name, param in checkpoint["splats"].items()
                }
            )
        if "smoke_splats" in checkpoint:
            self.smoke_splats = nn.ParameterDict(
                {
                    name: nn.Parameter(param.to(device=splat_device, dtype=torch.float32))
                    for name, param in checkpoint["smoke_splats"].items()
                }
            )
