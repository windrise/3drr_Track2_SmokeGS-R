from __future__ import annotations

import torch
import torch.nn.functional as F

from .proxy import estimate_smoke_density, proxy_depth_consistency_loss


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    c1 = 0.01**2
    c2 = 0.03**2
    img1 = img1.permute(2, 0, 1).unsqueeze(0)
    img2 = img2.permute(2, 0, 1).unsqueeze(0)
    channels = img1.shape[1]
    coords = torch.arange(window_size, dtype=img1.dtype, device=img1.device) - window_size // 2
    gauss_1d = torch.exp(-(coords**2) / (2 * 1.5**2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    kernel = (gauss_1d[:, None] * gauss_1d[None, :]).expand(channels, 1, window_size, window_size)
    pad = window_size // 2
    mu1 = F.conv2d(img1, kernel, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, kernel, padding=pad, groups=channels)
    mu1_sq, mu2_sq, mu12 = mu1**2, mu2**2, mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, kernel, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=pad, groups=channels) - mu12
    ssim_map = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()


def _as_2d_map(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3 and tensor.shape[0] in {1, 3}:
        tensor = tensor[0]
    if tensor.ndim == 3 and tensor.shape[-1] == 1:
        tensor = tensor[..., 0]
    return tensor.float()


def _normalize_map(tensor: torch.Tensor) -> torch.Tensor:
    tensor = _as_2d_map(tensor)
    valid_mask = torch.isfinite(tensor)
    if not valid_mask.any():
        return torch.zeros_like(tensor)
    tensor = tensor.clone()
    tensor[~valid_mask] = 0.0
    valid_values = tensor[valid_mask]
    tensor[valid_mask] = (valid_values - valid_values.min()) / (
        valid_values.max() - valid_values.min()
    ).clamp_min(1e-6)
    tensor[~valid_mask] = 0.0
    return tensor


def _reduce_depth_loss(
    prediction_map: torch.Tensor,
    target_map: torch.Tensor,
    valid_mask: torch.Tensor | None,
    loss_type: str,
    weight_map: torch.Tensor | None = None,
    weight_floor: float = 0.0,
) -> torch.Tensor:
    prediction_map = torch.nan_to_num(prediction_map.float(), nan=0.0, posinf=0.0, neginf=0.0)
    target_map = torch.nan_to_num(target_map.float(), nan=0.0, posinf=0.0, neginf=0.0)

    if valid_mask is not None and valid_mask.dtype != torch.bool:
        valid_mask = valid_mask > 0.5
    if valid_mask is not None:
        prediction_map = torch.where(valid_mask, prediction_map, torch.zeros_like(prediction_map))
        target_map = torch.where(valid_mask, target_map, torch.zeros_like(target_map))

    if loss_type == "l1":
        per_pixel = torch.abs(prediction_map - target_map)
    elif loss_type == "mse":
        per_pixel = (prediction_map - target_map) ** 2
    else:
        raise ValueError(f"unsupported depth loss type: {loss_type}")

    if weight_map is not None:
        weights = weight_map.float().clamp(0.0, 1.0)
        if weight_floor > 0:
            weights = float(weight_floor) + (1.0 - float(weight_floor)) * weights
        if valid_mask is not None:
            weights = weights * valid_mask.float()
        weight_sum = weights.sum()
        if weight_sum <= 0:
            return per_pixel.new_tensor(0.0)
        return (per_pixel * weights).sum() / weight_sum

    if valid_mask is None:
        return per_pixel.mean()
    if not valid_mask.any():
        return per_pixel.new_tensor(0.0)
    return per_pixel[valid_mask].mean()


def _reduce_color_loss(
    prediction_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    weight_map: torch.Tensor | None = None,
    weight_floor: float = 0.0,
) -> torch.Tensor:
    per_pixel = torch.abs(prediction_rgb - target_rgb).mean(dim=-1)
    if weight_map is None:
        return per_pixel.mean()

    weights = weight_map.float().clamp(0.0, 1.0)
    if weight_floor > 0:
        weights = float(weight_floor) + (1.0 - float(weight_floor)) * weights
    weight_sum = weights.sum()
    if weight_sum <= 0:
        return per_pixel.new_tensor(0.0)
    return (per_pixel * weights).sum() / weight_sum


def _downsample_image_tensor(tensor: torch.Tensor, factor: int) -> torch.Tensor:
    factor = max(1, int(factor))
    if factor <= 1:
        return tensor
    height, width = tensor.shape[:2]
    target_hw = (
        max(1, (height + factor - 1) // factor),
        max(1, (width + factor - 1) // factor),
    )
    return F.interpolate(
        tensor.permute(2, 0, 1).unsqueeze(0),
        size=target_hw,
        mode="area",
    )[0].permute(1, 2, 0)


def _downsample_map_tensor(tensor: torch.Tensor, factor: int) -> torch.Tensor:
    factor = max(1, int(factor))
    if factor <= 1:
        return tensor
    tensor = _as_2d_map(tensor)
    height, width = tensor.shape
    target_hw = (
        max(1, (height + factor - 1) // factor),
        max(1, (width + factor - 1) // factor),
    )
    return F.interpolate(
        tensor[None, None],
        size=target_hw,
        mode="area",
    )[0, 0]


def _resolve_annealed_weight(
    start_weight: float,
    end_weight: float | None,
    start_step: int,
    end_step: int,
    step: int | None,
) -> float:
    if end_weight is None or step is None or end_step <= start_step:
        return float(start_weight)
    if step <= start_step:
        return float(start_weight)
    if step >= end_step:
        return float(end_weight)
    progress = float(step - start_step) / float(end_step - start_step)
    return float(start_weight) + progress * (float(end_weight) - float(start_weight))


def _backproject_depth_to_world(
    depth_map: torch.Tensor,
    intrinsics: torch.Tensor,
    camtoworld: torch.Tensor,
) -> torch.Tensor:
    depth_map = _as_2d_map(depth_map)
    intrinsics = intrinsics.float()
    camtoworld = camtoworld.float()
    if camtoworld.shape == (3, 4):
        full_pose = torch.eye(4, dtype=torch.float32, device=depth_map.device)
        full_pose[:3, :] = camtoworld.to(depth_map.device)
        camtoworld = full_pose
    else:
        camtoworld = camtoworld.to(depth_map.device)
    intrinsics = intrinsics.to(depth_map.device)

    height, width = depth_map.shape
    ys = torch.arange(height, device=depth_map.device, dtype=torch.float32)
    xs = torch.arange(width, device=depth_map.device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    z = depth_map.clamp_min(0.0)
    x = ((grid_x + 0.5) - intrinsics[0, 2]) / intrinsics[0, 0].clamp_min(1e-6) * z
    y = ((grid_y + 0.5) - intrinsics[1, 2]) / intrinsics[1, 1].clamp_min(1e-6) * z
    cam_points = torch.stack([x, -y, -z], dim=-1)
    rotation = camtoworld[:3, :3]
    translation = camtoworld[:3, 3]
    return torch.einsum("hwc,dc->hwd", cam_points, rotation) + translation.view(1, 1, 3)


def _reduce_pointmap_loss(
    prediction_points: torch.Tensor,
    target_points: torch.Tensor,
    valid_mask: torch.Tensor | None,
    weight_map: torch.Tensor | None,
    weight_floor: float,
) -> torch.Tensor:
    per_pixel = torch.linalg.norm(prediction_points - target_points, dim=-1)
    if valid_mask is not None and valid_mask.dtype != torch.bool:
        valid_mask = valid_mask > 0.5
    if weight_map is not None:
        weights = weight_map.float().clamp(0.0, 1.0)
        if weight_floor > 0:
            weights = float(weight_floor) + (1.0 - float(weight_floor)) * weights
        if valid_mask is not None:
            weights = weights * valid_mask.float()
        weight_sum = weights.sum()
        if weight_sum <= 0:
            return per_pixel.new_tensor(0.0)
        return (per_pixel * weights).sum() / weight_sum
    if valid_mask is None:
        return per_pixel.mean()
    if not valid_mask.any():
        return per_pixel.new_tensor(0.0)
    return per_pixel[valid_mask].mean()


def _scale_invariant_align(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, float, float]:
    """Align prediction to target via least-squares (scale, shift).

    Returns aligned prediction, scale, shift.
    """
    pred_flat = prediction.reshape(-1)
    tgt_flat = target.reshape(-1)
    if valid_mask is not None:
        mask_flat = valid_mask.reshape(-1).bool()
        pred_flat = pred_flat[mask_flat]
        tgt_flat = tgt_flat[mask_flat]
    if pred_flat.numel() < 2:
        return prediction, 1.0, 0.0
    # Solve: tgt ≈ w * pred + b via least-squares
    A = torch.stack([pred_flat, torch.ones_like(pred_flat)], dim=-1)  # [N, 2]
    # (A^T A)^{-1} A^T tgt
    AtA = A.T @ A
    Atb = A.T @ tgt_flat
    try:
        params = torch.linalg.solve(AtA + 1e-6 * torch.eye(2, device=AtA.device), Atb)
        w, b = float(params[0].item()), float(params[1].item())
    except Exception:
        w, b = 1.0, 0.0
    aligned = w * prediction + b
    return aligned, w, b


def _get_depth_supervision_target(prediction: dict, loss_cfg) -> torch.Tensor:
    if loss_cfg.depth_target == "geometry_proxy":
        return _as_2d_map(prediction["geometry_proxy"])
    if loss_cfg.depth_target == "alphas":
        return _as_2d_map(prediction["alphas"])
    if loss_cfg.depth_target == "expected_depth":
        return _as_2d_map(prediction["expected_depth"])
    if loss_cfg.depth_target == "expected_depth_normalized":
        return _normalize_map(prediction["expected_depth"])
    if loss_cfg.depth_target == "inverse_expected_depth_normalized":
        return 1.0 - _normalize_map(prediction["expected_depth"])
    if loss_cfg.depth_target == "transmission":
        return _as_2d_map(prediction["transmission"])
    if loss_cfg.depth_target == "smoke_density":
        return 1.0 - _as_2d_map(prediction["transmission"])
    raise ValueError(f"unsupported depth target: {loss_cfg.depth_target}")


def _compute_depth_prior_loss(
    prediction: dict,
    batch: dict,
    loss_cfg,
    prior_key: str,
    valid_mask_key: str,
    weight_key: str,
    loss_name: str,
    scalar_weight: float,
    weight_floor: float,
    loss_dict: dict[str, torch.Tensor],
) -> torch.Tensor | None:
    if scalar_weight <= 0 or prior_key not in batch:
        return None

    rendered = prediction["rendered"]
    depth_target = _as_2d_map(batch[prior_key].to(rendered.device))
    valid_mask = batch.get(valid_mask_key)
    if valid_mask is not None:
        valid_mask = _as_2d_map(valid_mask.to(rendered.device))
    weight_map = batch.get(weight_key)
    if weight_map is not None:
        weight_map = _as_2d_map(weight_map.to(rendered.device))

    geometry_target = _get_depth_supervision_target(prediction, loss_cfg)
    depth_loss = _reduce_depth_loss(
        prediction_map=geometry_target,
        target_map=depth_target,
        valid_mask=valid_mask,
        loss_type=loss_cfg.depth_loss_type,
        weight_map=weight_map,
        weight_floor=weight_floor,
    )
    loss_dict[loss_name] = depth_loss
    if valid_mask is not None:
        loss_dict[f"{loss_name}_valid_ratio"] = valid_mask.float().mean()
    if weight_map is not None:
        loss_dict[f"{loss_name}_weight_mean"] = weight_map.float().mean()
    return scalar_weight * depth_loss


def compute_smoke_losses(prediction: dict, batch: dict, loss_cfg, step: int | None = None) -> dict[str, torch.Tensor]:
    target = batch["images"].permute(1, 2, 0)
    rendered = prediction["rendered"]

    l1_loss = torch.abs(rendered - target).mean()
    ssim_loss = 1.0 - ssim(rendered, target)
    total = (1.0 - loss_cfg.lambda_ssim) * l1_loss + loss_cfg.lambda_ssim * ssim_loss

    loss_dict = {
        "l1": l1_loss,
        "ssim": ssim_loss,
    }

    clean_weight = _resolve_annealed_weight(
        start_weight=loss_cfg.clean_weight,
        end_weight=loss_cfg.clean_weight_end,
        start_step=loss_cfg.clean_weight_anneal_start_step,
        end_step=loss_cfg.clean_weight_anneal_end_step,
        step=step,
    )
    # Pre-compute transmission_guided flag (used by both edge loss and clean loss)
    transmission_guided = getattr(loss_cfg, 'clean_confidence_mode', 'none') == 'transmission'

    # === 终极抢救方案二: 空间梯度损失 (Spatial Edge Loss) ===
    # 放弃对比绝对颜色，直接对比图像梯度（一阶导数）逼出锐利轮廓
    edge_weight = getattr(loss_cfg, 'edge_loss_weight', 0.0)
    if edge_weight > 0 and "clean_rgb" in prediction:
        clean_pred_hwc = prediction["clean_rgb"]  # [H, W, 3]
        # 需要 rendered (training target) 作为边缘参考
        pred_bchw = clean_pred_hwc.permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
        tgt_bchw = target.permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
        pred_dx = pred_bchw[:, :, :, 1:] - pred_bchw[:, :, :, :-1]
        pred_dy = pred_bchw[:, :, 1:, :] - pred_bchw[:, :, :-1, :]
        tgt_dx = tgt_bchw[:, :, :, 1:] - tgt_bchw[:, :, :, :-1]
        tgt_dy = tgt_bchw[:, :, 1:, :] - tgt_bchw[:, :, :-1, :]
        # 同样用 T² 置信度掩码保护浓烟区
        if transmission_guided and "transmission" in prediction:
            T_edge = prediction["transmission"].detach()
            if T_edge.ndim == 3 and T_edge.shape[-1] == 1:
                T_edge = T_edge[..., 0]
            T_edge_sq = (T_edge ** 2).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
            grad_loss_x = (T_edge_sq[:, :, :, :-1] * torch.abs(pred_dx - tgt_dx)).mean()
            grad_loss_y = (T_edge_sq[:, :, :-1, :] * torch.abs(pred_dy - tgt_dy)).mean()
        else:
            grad_loss_x = torch.abs(pred_dx - tgt_dx).mean()
            grad_loss_y = torch.abs(pred_dy - tgt_dy).mean()
        edge_loss = grad_loss_x + grad_loss_y
        total = total + edge_weight * edge_loss
        loss_dict["edge"] = edge_loss

    if clean_weight > 0 and "clean_prior" in batch and "clean_rgb" in prediction:
        clean_target = batch["clean_prior"].permute(1, 2, 0).to(rendered.device)
        clean_prediction = prediction["clean_rgb"]
        clean_weight_map = batch.get("clean_prior_weight")
        if clean_weight_map is not None:
            clean_weight_map = _as_2d_map(clean_weight_map.to(rendered.device))

        # === Transmission-Guided Routing (T² 非线性置信度) ===
        # T² 更激进地压制浓烟区: T=0.9→0.81, T=0.2→0.04(彻底无视模糊伪标签)
        if transmission_guided and "transmission" in prediction:
            T = prediction["transmission"].detach()  # detach: 不干扰 T 本身的优化
            if T.ndim == 3 and T.shape[-1] == 1:
                T = T[..., 0]  # [H, W]
            T_confidence = T ** 2  # 非线性: 激进压制浓烟区模糊伪标签
            if clean_weight_map is not None:
                clean_weight_map = clean_weight_map * T_confidence
            else:
                clean_weight_map = T_confidence

        clean_downsample_factor = max(1, int(loss_cfg.clean_downsample_factor))
        if clean_downsample_factor > 1:
            clean_target = _downsample_image_tensor(clean_target, clean_downsample_factor)
            clean_prediction = _downsample_image_tensor(clean_prediction, clean_downsample_factor)
            if clean_weight_map is not None:
                clean_weight_map = _downsample_map_tensor(clean_weight_map, clean_downsample_factor)
        clean_loss = _reduce_color_loss(
            clean_prediction,
            clean_target,
            weight_map=clean_weight_map,
            weight_floor=loss_cfg.clean_loss_weight_floor,
        )
        total = total + clean_weight * clean_loss
        loss_dict["clean"] = clean_loss
        loss_dict["clean_weight_effective"] = rendered.new_tensor(float(clean_weight))
        if clean_weight_map is not None:
            loss_dict["clean_weight_mean"] = clean_weight_map.float().mean()

    primary_depth_term = _compute_depth_prior_loss(
        prediction=prediction,
        batch=batch,
        loss_cfg=loss_cfg,
        prior_key="depth_prior",
        valid_mask_key="depth_prior_valid_mask",
        weight_key="depth_prior_weight",
        loss_name="depth",
        scalar_weight=loss_cfg.depth_weight,
        weight_floor=loss_cfg.depth_loss_weight_floor,
        loss_dict=loss_dict,
    )
    if primary_depth_term is not None:
        total = total + primary_depth_term

    aux_depth_term = _compute_depth_prior_loss(
        prediction=prediction,
        batch=batch,
        loss_cfg=loss_cfg,
        prior_key="aux_depth_prior",
        valid_mask_key="aux_depth_prior_valid_mask",
        weight_key="aux_depth_prior_weight",
        loss_name="aux_depth",
        scalar_weight=loss_cfg.aux_depth_weight,
        weight_floor=loss_cfg.aux_depth_loss_weight_floor,
        loss_dict=loss_dict,
    )
    if aux_depth_term is not None:
        total = total + aux_depth_term

    proxy_depth_weight = getattr(loss_cfg, "proxy_depth_weight", 0.0)
    proxy_depth_warmup_step = getattr(loss_cfg, "proxy_depth_warmup_step", 0)
    if (
        proxy_depth_weight > 0
        and "proxy_depth_prior" in batch
        and prediction.get("expected_depth") is not None
        and not (step is not None and step < proxy_depth_warmup_step)
    ):
        proxy_depth = _as_2d_map(batch["proxy_depth_prior"].to(rendered.device))
        expected_depth = _as_2d_map(prediction["expected_depth"])
        smoke_density = None
        if getattr(loss_cfg, "proxy_depth_use_smoke_mask", True):
            smoke_density = estimate_smoke_density(batch["images"].to(rendered.device))
        proxy_depth_loss = proxy_depth_consistency_loss(
            expected_depth=expected_depth,
            proxy_depth=proxy_depth,
            smoke_density=smoke_density,
            smoke_threshold=getattr(loss_cfg, "proxy_depth_smoke_threshold", 0.5),
        )
        total = total + proxy_depth_weight * proxy_depth_loss
        loss_dict["proxy_depth"] = proxy_depth_loss

    if loss_cfg.pointmap_weight > 0 and "pointmap_prior" in batch and "pointmap_expected_depth" in prediction:
        pointmap_target = batch["pointmap_prior"].to(rendered.device).float()
        pointmap_valid = batch.get("pointmap_prior_valid_mask")
        if pointmap_valid is not None:
            pointmap_valid = _as_2d_map(pointmap_valid.to(rendered.device))
        pointmap_weight = batch.get("pointmap_prior_weight")
        if pointmap_weight is not None:
            pointmap_weight = _as_2d_map(pointmap_weight.to(rendered.device))
        predicted_points = _backproject_depth_to_world(
            prediction["pointmap_expected_depth"],
            batch["pointmap_intrinsics"].to(rendered.device),
            batch["pointmap_pose_gl"].to(rendered.device),
        )
        pointmap_loss = _reduce_pointmap_loss(
            prediction_points=predicted_points,
            target_points=pointmap_target,
            valid_mask=pointmap_valid,
            weight_map=pointmap_weight,
            weight_floor=loss_cfg.pointmap_loss_weight_floor,
        )
        total = total + loss_cfg.pointmap_weight * pointmap_loss
        loss_dict["pointmap"] = pointmap_loss
        if pointmap_valid is not None:
            loss_dict["pointmap_valid_ratio"] = pointmap_valid.float().mean()
        if pointmap_weight is not None:
            loss_dict["pointmap_weight_mean"] = pointmap_weight.float().mean()

    if loss_cfg.pointmap_rgb_weight > 0 and "pointmap_rgb_prior" in batch and "pointmap_rendered" in prediction:
        pointmap_rgb_target = batch["pointmap_rgb_prior"].to(rendered.device).float()
        pointmap_valid = batch.get("pointmap_prior_valid_mask")
        if pointmap_valid is not None:
            pointmap_valid = _as_2d_map(pointmap_valid.to(rendered.device))
        pointmap_weight = batch.get("pointmap_prior_weight")
        if pointmap_weight is not None:
            pointmap_weight = _as_2d_map(pointmap_weight.to(rendered.device))
        color_weight = pointmap_weight
        if color_weight is None and pointmap_valid is not None:
            color_weight = pointmap_valid.float()
        # L1 RGB loss
        pointmap_rgb_l1 = _reduce_color_loss(
            prediction["pointmap_rendered"],
            pointmap_rgb_target,
            weight_map=color_weight,
        )
        # SSIM loss on pointmap view
        pointmap_rgb_ssim = 1.0 - ssim(prediction["pointmap_rendered"], pointmap_rgb_target)
        pointmap_rgb_loss = (1.0 - loss_cfg.lambda_ssim) * pointmap_rgb_l1 + loss_cfg.lambda_ssim * pointmap_rgb_ssim
        total = total + loss_cfg.pointmap_rgb_weight * pointmap_rgb_loss
        loss_dict["pointmap_rgb"] = pointmap_rgb_loss

    if loss_cfg.pointmap_depth_weight > 0 and "pointmap_depth_prior" in batch and "pointmap_expected_depth" in prediction:
        pointmap_depth_target = _as_2d_map(batch["pointmap_depth_prior"].to(rendered.device))
        pointmap_pred_depth = _as_2d_map(prediction["pointmap_expected_depth"])
        pointmap_valid = batch.get("pointmap_prior_valid_mask")
        if pointmap_valid is not None:
            pointmap_valid = _as_2d_map(pointmap_valid.to(rendered.device))
        pointmap_weight = batch.get("pointmap_prior_weight")
        if pointmap_weight is not None:
            pointmap_weight = _as_2d_map(pointmap_weight.to(rendered.device))
        # Scale-invariant alignment: align predicted depth to DUSt3R depth
        aligned_pred_depth, _si_w, _si_b = _scale_invariant_align(
            pointmap_pred_depth, pointmap_depth_target, pointmap_valid,
        )
        pointmap_depth_loss = _reduce_depth_loss(
            prediction_map=aligned_pred_depth,
            target_map=pointmap_depth_target,
            valid_mask=pointmap_valid,
            loss_type=loss_cfg.depth_loss_type,
            weight_map=pointmap_weight,
            weight_floor=loss_cfg.pointmap_loss_weight_floor,
        )
        total = total + loss_cfg.pointmap_depth_weight * pointmap_depth_loss
        loss_dict["pointmap_depth"] = pointmap_depth_loss

    if "feature_prior" in batch and "feature_map" in prediction and loss_cfg.feature_weight > 0:
        # Delay activation: feature loss MUST NOT fire during densification to prevent clone explosion
        feature_delay = getattr(loss_cfg, 'feature_delay_step', 0)
        if step is not None and feature_delay > 0 and step < feature_delay:
            pass  # Skip entirely — no gradient, no densification trigger
        else:
            feature_target = batch["feature_prior"].to(rendered.device).float()
            feature_pred = prediction["feature_map"].float()
            # Patch-level cosine similarity (not flattened global) for better gradients
            if feature_pred.ndim == 3 and feature_target.ndim == 3:
                # [C, H, W] → compute per-patch cosine sim
                feature_loss = 1.0 - F.cosine_similarity(
                    feature_pred, feature_target, dim=0,
                ).mean()
            else:
                feature_loss = 1.0 - F.cosine_similarity(
                    feature_pred.reshape(1, -1),
                    feature_target.reshape(1, -1),
                ).mean()
            total = total + loss_cfg.feature_weight * feature_loss
            loss_dict["feature"] = feature_loss

    smoke_sparse = (1.0 - prediction["transmission"]).mean()
    airlight_prior = ((prediction["airlight"] - 1.0) ** 2).mean()
    total = total + loss_cfg.smoke_sparse_weight * smoke_sparse
    total = total + loss_cfg.airlight_prior_weight * airlight_prior

    loss_dict["smoke_sparse"] = smoke_sparse
    loss_dict["airlight"] = airlight_prior

    # TV regularization on transmission map for spatial smoothness
    # Supports warmup: tv_weight is 0 before tv_warmup_step, then linearly ramps
    tv_weight_base = getattr(loss_cfg, 'transmission_tv_weight', 0.0)
    tv_warmup_step = getattr(loss_cfg, 'transmission_tv_warmup_step', 0)
    if tv_weight_base > 0 and "transmission" in prediction:
        # Apply warmup schedule
        if step is not None and tv_warmup_step > 0 and step < tv_warmup_step:
            tv_weight = 0.0
        elif step is not None and tv_warmup_step > 0:
            # Linear ramp from warmup_step to 2*warmup_step (or end of training)
            ramp_duration = max(tv_warmup_step, 500)  # ramp over at least 500 steps
            progress = min(1.0, (step - tv_warmup_step) / ramp_duration)
            tv_weight = tv_weight_base * progress
        else:
            tv_weight = tv_weight_base
        t_map = prediction["transmission"]  # [H, W, 1]
        if t_map.ndim == 3 and t_map.shape[-1] == 1:
            t_map = t_map[..., 0]  # [H, W]
        tv_h = torch.abs(t_map[1:, :] - t_map[:-1, :]).mean()
        tv_w = torch.abs(t_map[:, 1:] - t_map[:, :-1]).mean()
        tv_loss = tv_h + tv_w
        total = total + tv_weight * tv_loss
        loss_dict["transmission_tv"] = tv_loss
        loss_dict["tv_weight_effective"] = rendered.new_tensor(tv_weight)

    # ======== Dual-branch specific losses ========
    if prediction.get("smoke_active", False) and "smoke_alpha" in prediction:
        # Exclusion loss: penalize overlap where both scene and smoke are opaque
        smoke_exclusion_w = getattr(loss_cfg, 'smoke_exclusion_weight', 0.0)
        if smoke_exclusion_w > 0:
            scene_alpha = prediction["scene_alphas"]  # [H, W, 1]
            smoke_alpha = prediction["smoke_alpha"]  # [H, W, 1]
            exclusion = (scene_alpha * smoke_alpha).mean()
            total = total + smoke_exclusion_w * exclusion
            loss_dict["exclusion"] = exclusion

        # Smoke branch sparsity: L1 on smoke opacity
        smoke_branch_sparse_w = getattr(loss_cfg, 'smoke_branch_sparse_weight', 0.0)
        if smoke_branch_sparse_w > 0:
            smoke_opacity_l1 = prediction["smoke_alpha"].mean()
            total = total + smoke_branch_sparse_w * smoke_opacity_l1
            loss_dict["smoke_branch_sparse"] = smoke_opacity_l1

        # SmokeSeer-inspired: smoke opacity/color variance regularization
        # These operate on the raw Gaussian parameters, passed via prediction dict
        if "smoke_opacity_var" in prediction:
            opacity_var_w = getattr(loss_cfg, 'smoke_opacity_var_weight', 0.01)
            total = total + opacity_var_w * prediction["smoke_opacity_var"]
            loss_dict["smoke_opacity_var"] = prediction["smoke_opacity_var"]
        if "smoke_color_var" in prediction:
            color_var_w = getattr(loss_cfg, 'smoke_color_var_weight', 0.01)
            total = total + color_var_w * prediction["smoke_color_var"]
            loss_dict["smoke_color_var"] = prediction["smoke_color_var"]

    # === 终极抢救方案三: Scale & Opacity Hardening ===
    # 防止 3DGS 在烟雾区用巨大半透明高斯球糊弄 → 逼迫保持针尖般细小且不透明
    scale_reg_weight = getattr(loss_cfg, 'scale_reg_weight', 0.0)
    opacity_bin_weight = getattr(loss_cfg, 'opacity_bin_weight', 0.0)
    hardening_start_step = getattr(loss_cfg, 'hardening_start_step', 1500)
    if (scale_reg_weight > 0 or opacity_bin_weight > 0) and step is not None and step > hardening_start_step:
        splats = prediction.get("splats_ref")  # model passes splats dict reference
        if splats is not None:
            if scale_reg_weight > 0 and "scales" in splats:
                scales_real = torch.exp(splats["scales"])
                scale_loss = scales_real.mean()
                total = total + scale_reg_weight * scale_loss
                loss_dict["scale_reg"] = scale_loss
            if opacity_bin_weight > 0 and "opacities" in splats:
                opas = torch.sigmoid(splats["opacities"])
                opa_bin_loss = torch.min(opas, 1.0 - opas).mean()
                total = total + opacity_bin_weight * opa_bin_loss
                loss_dict["opacity_bin"] = opa_bin_loss

    loss_dict["total"] = total
    return loss_dict
