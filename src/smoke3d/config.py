from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetConfig:
    name: str
    data_path: str
    background_color: float = 255.0
    prior_root: str | None = None
    clean_data_path: str | None = None
    depth_subdir: str = "depth"
    depth_weight_subdir: str | None = None
    aux_depth_subdir: str | None = None
    aux_depth_weight_subdir: str | None = None
    pointmap_subdir: str | None = None
    proxy_pointmap_subdir: str | None = None
    geometry_subdir: str = "geometry"
    geometry_file: str = "scene_points.npz"
    feature_subdir: str = "features"
    clean_weight_subdir: str | None = None
    clean_confidence_mode: str = "none"
    clean_confidence_power: float = 1.0
    depth_resize_mode: str = "bilinear"
    depth_normalize: str = "per_image_minmax"
    depth_invert: bool = False
    val_preview_views: int = 4


@dataclass
class ModelConfig:
    name: str = "SmokeAware3DGS"
    strategy_type: str = "default"
    num_init_points: int = 100000
    use_geometry_prior: bool = False
    geometry_prior_seed: int = 0
    per_view_smoke: bool = False
    path_length_mode: str = "alpha"
    sh_degree: int = 3
    scene_scale: float = 2.0
    init_scale: float = 0.005
    init_opacity: float = 0.1
    lr_means: float = 1.6e-4
    lr_means_final: float = 1.6e-6
    lr_quats: float = 1.0e-3
    lr_scales: float = 5.0e-3
    lr_opacities: float = 5.0e-2
    lr_sh0: float = 2.5e-3
    lr_shn: float = 1.25e-4
    lr_smoke: float = 1.0e-3
    densify_start_step: int = 100
    densify_stop_step: int = 15000
    densify_interval: int = 100
    densify_grad_thresh: float = 2.0e-4
    opacity_reset_interval: int = 100000  # 实质上禁用 reset，烟雾场景下 3000步reset半透明高斯会导致训练崩溃
    mcmc_cap_max: int = 100000
    mcmc_noise_lr: float = 5.0e5
    mcmc_min_opacity: float = 5.0e-3
    init_beta: float = 0.2
    init_airlight: list[float] = field(default_factory=lambda: [0.85, 0.85, 0.85])
    init_transmission_bias: float = 0.05
    min_transmission: float = 0.05
    proxy_build_stride: int = 8
    proxy_confidence_threshold: float = 0.0
    proxy_edge_threshold_scale: float = 4.0
    proxy_max_views: int = 0
    proxy_depth_near: float = 0.01
    proxy_depth_far: float = 100.0
    proxy_force_simple_renderer: bool = True
    spatial_smoke_grid_size: int = 0  # 0=disabled; >0 enables per-view low-res transmission residual grid
    # Dual-branch config
    dual_branch: bool = False
    smoke_num_points: int = 10000
    smoke_init_opacity: float = 0.01
    smoke_init_scale: float = 0.1
    smoke_warmup_steps: int = 500
    lr_smoke_branch: float = 5.0e-4
    freeze_scene_on_warmup: bool = False  # SmokeSeer-style: freeze scene geometry when smoke activates


@dataclass
class LossConfig:
    lambda_ssim: float = 0.2
    clean_weight: float = 0.0
    clean_weight_end: float | None = None
    clean_weight_anneal_start_step: int = 0
    clean_weight_anneal_end_step: int = 0
    clean_downsample_factor: int = 1
    depth_weight: float = 0.0
    aux_depth_weight: float = 0.0
    pointmap_weight: float = 0.0
    pointmap_rgb_weight: float = 0.0
    pointmap_depth_weight: float = 0.0
    feature_weight: float = 0.0
    feature_model_name: str = "dinov2_vits14"
    feature_repo_path: str = "methods/foundation/dinov2"
    feature_input_size: int = 518
    feature_render_source: str = "clean_rgb"
    depth_target: str = "geometry_proxy"
    depth_loss_type: str = "l1"
    clean_loss_weight_floor: float = 0.0
    depth_loss_weight_floor: float = 0.0
    aux_depth_loss_weight_floor: float = 0.0
    pointmap_loss_weight_floor: float = 0.0
    smoke_sparse_weight: float = 1.0e-2
    airlight_prior_weight: float = 1.0e-2
    transmission_tv_weight: float = 0.0
    transmission_tv_warmup_step: int = 0  # TV weight is 0 before this step, then linearly ramps
    feature_delay_step: int = 0  # Feature loss is 0 before this step (must be after densification ends!)
    proxy_depth_weight: float = 0.0
    proxy_depth_warmup_step: int = 0
    proxy_depth_smoke_threshold: float = 0.5
    proxy_depth_use_smoke_mask: bool = True
    # Dual-branch loss config
    smoke_exclusion_weight: float = 0.01  # penalize overlap between scene and smoke
    smoke_branch_sparse_weight: float = 0.005  # L1 on smoke opacity
    smoke_opacity_var_weight: float = 0.01  # SmokeSeer: variance of smoke opacity
    smoke_color_var_weight: float = 0.01  # SmokeSeer: variance of smoke color
    # Expert rescue loss parameters
    edge_loss_weight: float = 0.0        # Spatial gradient loss to force sharp edges
    scale_reg_weight: float = 0.0        # Scale regularization to prevent Gaussian bloating
    opacity_bin_weight: float = 0.0      # Opacity binarization to prevent semi-transparent clouds
    hardening_start_step: int = 1500     # Step after which scale/opacity hardening activates



@dataclass
class RunnerConfig:
    train_total_step: int = 5000
    val_interval_step: int = 1000
    log_interval_step: int = 100
    sh_upgrade_interval: int = 1000
    device: str = "cuda"
    output_root: str = "outputs"
    init_checkpoint: str | None = None
    init_checkpoint_load_splats: bool = True
    init_checkpoint_load_smoke: bool = True
    proxy_mesh_path: str | None = None
    proxy_precompute_depths: bool = True
    seed: int = 0
    augment_gamma: float = 1.0
    train_render_clean: bool = False
    render_clean_on_test: bool = True


@dataclass
class ExperimentConfig:
    exp_name: str
    dataset: DatasetConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_defaults(raw_cfg: dict[str, Any]) -> dict[str, Any]:
    if "EXP_NAME" not in raw_cfg:
        raise KeyError("EXP_NAME is required")
    if "DATASET" not in raw_cfg:
        raise KeyError("DATASET is required")

    model_cfg = dict(raw_cfg.get("MODEL", {}))
    loss_cfg = dict(raw_cfg.get("LOSS", {}))
    runner_cfg = dict(raw_cfg.get("RUNNER", {}))
    dataset_cfg = dict(raw_cfg["DATASET"])

    return {
        "exp_name": raw_cfg["EXP_NAME"],
        "dataset": DatasetConfig(
            name=dataset_cfg["NAME"],
            data_path=dataset_cfg["DATA_PATH"],
            background_color=dataset_cfg.get("BACKGROUND_COLOR", 255.0),
            prior_root=dataset_cfg.get("PRIOR_ROOT"),
            clean_data_path=dataset_cfg.get("CLEAN_DATA_PATH"),
            depth_subdir=dataset_cfg.get("DEPTH_SUBDIR", "depth"),
            depth_weight_subdir=dataset_cfg.get("DEPTH_WEIGHT_SUBDIR"),
            aux_depth_subdir=dataset_cfg.get("AUX_DEPTH_SUBDIR"),
            aux_depth_weight_subdir=dataset_cfg.get("AUX_DEPTH_WEIGHT_SUBDIR"),
            pointmap_subdir=dataset_cfg.get("POINTMAP_SUBDIR"),
            proxy_pointmap_subdir=dataset_cfg.get("PROXY_POINTMAP_SUBDIR"),
            geometry_subdir=dataset_cfg.get("GEOMETRY_SUBDIR", "geometry"),
            geometry_file=dataset_cfg.get("GEOMETRY_FILE", "scene_points.npz"),
            feature_subdir=dataset_cfg.get("FEATURE_SUBDIR", "features"),
            clean_weight_subdir=dataset_cfg.get("CLEAN_WEIGHT_SUBDIR"),
            clean_confidence_mode=dataset_cfg.get("CLEAN_CONFIDENCE_MODE", "none"),
            clean_confidence_power=dataset_cfg.get("CLEAN_CONFIDENCE_POWER", 1.0),
            depth_resize_mode=dataset_cfg.get("DEPTH_RESIZE_MODE", "bilinear"),
            depth_normalize=dataset_cfg.get("DEPTH_NORMALIZE", "per_image_minmax"),
            depth_invert=dataset_cfg.get("DEPTH_INVERT", False),
            val_preview_views=dataset_cfg.get("VAL_PREVIEW_VIEWS", 4),
        ),
        "model": ModelConfig(
            name=model_cfg.get("NAME", "SmokeAware3DGS"),
            strategy_type=model_cfg.get("STRATEGY_TYPE", "default"),
            num_init_points=model_cfg.get("NUM_INIT_POINTS", 100000),
            use_geometry_prior=model_cfg.get("USE_GEOMETRY_PRIOR", False),
            geometry_prior_seed=model_cfg.get("GEOMETRY_PRIOR_SEED", 0),
            per_view_smoke=model_cfg.get("PER_VIEW_SMOKE", False),
            path_length_mode=model_cfg.get("PATH_LENGTH_MODE", "alpha"),
            sh_degree=model_cfg.get("SH_DEGREE", 3),
            scene_scale=model_cfg.get("SCENE_SCALE", 2.0),
            init_scale=model_cfg.get("INIT_SCALE", 0.005),
            init_opacity=model_cfg.get("INIT_OPACITY", 0.1),
            lr_means=model_cfg.get("LR_MEANS", 1.6e-4),
            lr_means_final=model_cfg.get("LR_MEANS_FINAL", 1.6e-6),
            lr_quats=model_cfg.get("LR_QUATS", 1.0e-3),
            lr_scales=model_cfg.get("LR_SCALES", 5.0e-3),
            lr_opacities=model_cfg.get("LR_OPACITIES", 5.0e-2),
            lr_sh0=model_cfg.get("LR_SH0", 2.5e-3),
            lr_shn=model_cfg.get("LR_SHN", 1.25e-4),
            lr_smoke=model_cfg.get("LR_SMOKE", 1.0e-3),
            densify_start_step=model_cfg.get("DENSIFY_START_STEP", 100),
            densify_stop_step=model_cfg.get("DENSIFY_STOP_STEP", 15000),
            densify_interval=model_cfg.get("DENSIFY_INTERVAL", 100),
            densify_grad_thresh=model_cfg.get("DENSIFY_GRAD_THRESH", 2.0e-4),
            opacity_reset_interval=model_cfg.get("OPACITY_RESET_INTERVAL", 100000),
            mcmc_cap_max=model_cfg.get("MCMC_CAP_MAX", model_cfg.get("NUM_INIT_POINTS", 100000)),
            mcmc_noise_lr=model_cfg.get("MCMC_NOISE_LR", 5.0e5),
            mcmc_min_opacity=model_cfg.get("MCMC_MIN_OPACITY", 5.0e-3),
            init_beta=model_cfg.get("INIT_BETA", 0.2),
            init_airlight=model_cfg.get("INIT_AIRLIGHT", [0.85, 0.85, 0.85]),
            init_transmission_bias=model_cfg.get("INIT_TRANSMISSION_BIAS", 0.05),
            min_transmission=model_cfg.get("MIN_TRANSMISSION", 0.05),
            proxy_build_stride=model_cfg.get("PROXY_BUILD_STRIDE", 8),
            proxy_confidence_threshold=model_cfg.get("PROXY_CONFIDENCE_THRESHOLD", 0.0),
            proxy_edge_threshold_scale=model_cfg.get("PROXY_EDGE_THRESHOLD_SCALE", 4.0),
            proxy_max_views=model_cfg.get("PROXY_MAX_VIEWS", 0),
            proxy_depth_near=model_cfg.get("PROXY_DEPTH_NEAR", 0.01),
            proxy_depth_far=model_cfg.get("PROXY_DEPTH_FAR", 100.0),
            proxy_force_simple_renderer=model_cfg.get("PROXY_FORCE_SIMPLE_RENDERER", True),
            spatial_smoke_grid_size=model_cfg.get("SPATIAL_SMOKE_GRID_SIZE", 0),
            dual_branch=model_cfg.get("DUAL_BRANCH", False),
            smoke_num_points=model_cfg.get("SMOKE_NUM_POINTS", 10000),
            smoke_init_opacity=model_cfg.get("SMOKE_INIT_OPACITY", 0.01),
            smoke_init_scale=model_cfg.get("SMOKE_INIT_SCALE", 0.1),
            smoke_warmup_steps=model_cfg.get("SMOKE_WARMUP_STEPS", 500),
            lr_smoke_branch=model_cfg.get("LR_SMOKE_BRANCH", 5.0e-4),
            freeze_scene_on_warmup=model_cfg.get("FREEZE_SCENE_ON_WARMUP", False),
        ),
        "loss": LossConfig(
            lambda_ssim=loss_cfg.get("LAMBDA_SSIM", 0.2),
            clean_weight=loss_cfg.get("CLEAN_WEIGHT", 0.0),
            clean_weight_end=loss_cfg.get("CLEAN_WEIGHT_END"),
            clean_weight_anneal_start_step=loss_cfg.get("CLEAN_WEIGHT_ANNEAL_START_STEP", 0),
            clean_weight_anneal_end_step=loss_cfg.get("CLEAN_WEIGHT_ANNEAL_END_STEP", 0),
            clean_downsample_factor=loss_cfg.get("CLEAN_DOWNSAMPLE_FACTOR", 1),
            depth_weight=loss_cfg.get("DEPTH_WEIGHT", 0.0),
            aux_depth_weight=loss_cfg.get("AUX_DEPTH_WEIGHT", 0.0),
            pointmap_weight=loss_cfg.get("POINTMAP_WEIGHT", 0.0),
            pointmap_rgb_weight=loss_cfg.get("POINTMAP_RGB_WEIGHT", 0.0),
            pointmap_depth_weight=loss_cfg.get("POINTMAP_DEPTH_WEIGHT", 0.0),
            feature_weight=loss_cfg.get("FEATURE_WEIGHT", 0.0),
            feature_model_name=loss_cfg.get("FEATURE_MODEL_NAME", "dinov2_vits14"),
            feature_repo_path=loss_cfg.get("FEATURE_REPO_PATH", "methods/foundation/dinov2"),
            feature_input_size=loss_cfg.get("FEATURE_INPUT_SIZE", 518),
            feature_render_source=loss_cfg.get("FEATURE_RENDER_SOURCE", "clean_rgb"),
            depth_target=loss_cfg.get("DEPTH_TARGET", "geometry_proxy"),
            depth_loss_type=loss_cfg.get("DEPTH_LOSS_TYPE", "l1"),
            clean_loss_weight_floor=loss_cfg.get("CLEAN_LOSS_WEIGHT_FLOOR", 0.0),
            depth_loss_weight_floor=loss_cfg.get("DEPTH_LOSS_WEIGHT_FLOOR", 0.0),
            aux_depth_loss_weight_floor=loss_cfg.get("AUX_DEPTH_LOSS_WEIGHT_FLOOR", 0.0),
            pointmap_loss_weight_floor=loss_cfg.get("POINTMAP_LOSS_WEIGHT_FLOOR", 0.0),
            smoke_sparse_weight=loss_cfg.get("SMOKE_SPARSE_WEIGHT", 1.0e-2),
            airlight_prior_weight=loss_cfg.get("AIRLIGHT_PRIOR_WEIGHT", 1.0e-2),
            transmission_tv_weight=loss_cfg.get("TRANSMISSION_TV_WEIGHT", 0.0),
            transmission_tv_warmup_step=loss_cfg.get("TRANSMISSION_TV_WARMUP_STEP", 0),
            proxy_depth_weight=loss_cfg.get("PROXY_DEPTH_WEIGHT", 0.0),
            proxy_depth_warmup_step=loss_cfg.get("PROXY_DEPTH_WARMUP_STEP", 0),
            proxy_depth_smoke_threshold=loss_cfg.get("PROXY_DEPTH_SMOKE_THRESHOLD", 0.5),
            proxy_depth_use_smoke_mask=loss_cfg.get("PROXY_DEPTH_USE_SMOKE_MASK", True),
            smoke_exclusion_weight=loss_cfg.get("SMOKE_EXCLUSION_WEIGHT", 0.01),
            smoke_branch_sparse_weight=loss_cfg.get("SMOKE_BRANCH_SPARSE_WEIGHT", 0.005),
            smoke_opacity_var_weight=loss_cfg.get("SMOKE_OPACITY_VAR_WEIGHT", 0.01),
            smoke_color_var_weight=loss_cfg.get("SMOKE_COLOR_VAR_WEIGHT", 0.01),
            edge_loss_weight=loss_cfg.get("EDGE_LOSS_WEIGHT", 0.0),
            scale_reg_weight=loss_cfg.get("SCALE_REG_WEIGHT", 0.0),
            opacity_bin_weight=loss_cfg.get("OPACITY_BIN_WEIGHT", 0.0),
            hardening_start_step=loss_cfg.get("HARDENING_START_STEP", 1500),
        ),
        "runner": RunnerConfig(
            train_total_step=runner_cfg.get("TRAIN_TOTAL_STEP", 5000),
            val_interval_step=runner_cfg.get("VAL_INTERVAL_STEP", 1000),
            log_interval_step=runner_cfg.get("LOG_INTERVAL_STEP", 100),
            sh_upgrade_interval=runner_cfg.get("SH_UPGRADE_INTERVAL", 1000),
            device=runner_cfg.get("DEVICE", "cuda"),
            output_root=runner_cfg.get("OUTPUT_ROOT", "outputs"),
            init_checkpoint=runner_cfg.get("INIT_CHECKPOINT"),
            init_checkpoint_load_splats=runner_cfg.get("INIT_CHECKPOINT_LOAD_SPLATS", True),
            init_checkpoint_load_smoke=runner_cfg.get("INIT_CHECKPOINT_LOAD_SMOKE", True),
            proxy_mesh_path=runner_cfg.get("PROXY_MESH_PATH"),
            proxy_precompute_depths=runner_cfg.get("PROXY_PRECOMPUTE_DEPTHS", True),
            seed=runner_cfg.get("SEED", 0),
            augment_gamma=runner_cfg.get("AUGMENT_GAMMA", 1.0),
            train_render_clean=runner_cfg.get("TRAIN_RENDER_CLEAN", False),
            render_clean_on_test=runner_cfg.get("RENDER_CLEAN_ON_TEST", True),
        ),
    }


def _infer_repo_root(config_path: Path) -> Path:
    for parent in config_path.parents:
        if parent.name == "configs":
            return parent.parent
    return config_path.parent


def _resolve_repo_relative_path(repo_root: Path, value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((repo_root / path).resolve())


def load_experiment_config(config_path: str | Path) -> ExperimentConfig:
    config_path = Path(config_path).resolve()
    with open(config_path, "r", encoding="utf-8") as handle:
        raw_cfg = yaml.safe_load(handle)
    merged = _merge_defaults(raw_cfg)
    repo_root = _infer_repo_root(config_path)

    merged["dataset"].data_path = _resolve_repo_relative_path(repo_root, merged["dataset"].data_path)
    merged["dataset"].prior_root = _resolve_repo_relative_path(repo_root, merged["dataset"].prior_root)
    merged["dataset"].clean_data_path = _resolve_repo_relative_path(repo_root, merged["dataset"].clean_data_path)
    merged["loss"].feature_repo_path = _resolve_repo_relative_path(repo_root, merged["loss"].feature_repo_path)
    merged["runner"].output_root = _resolve_repo_relative_path(repo_root, merged["runner"].output_root)
    merged["runner"].init_checkpoint = _resolve_repo_relative_path(repo_root, merged["runner"].init_checkpoint)
    merged["runner"].proxy_mesh_path = _resolve_repo_relative_path(repo_root, merged["runner"].proxy_mesh_path)

    return ExperimentConfig(**merged)
