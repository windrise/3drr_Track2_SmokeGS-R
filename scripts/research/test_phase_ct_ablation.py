#!/usr/bin/env python3
"""Test-phase CT ablation on Akikaze GT.

This script answers one critical question:
  "What post-processing strategy should we use for test scenes?"

It tests ALL of these on Akikaze GT (4 test views):

  A) Current dev pipeline  : geo_mean of 5 DEV models as ref → LAB Reinhard → gauss35
  B) Scene-adaptive CT     : geo_mean of 5 AKIKAZE models as ref → LAB Reinhard → gauss35
  C) No CT, pure geo_mean  : geo_mean of 5 Akikaze models → gauss35
  D) No CT, no blur        : geo_mean of 5 Akikaze models → raw
  E) Alpha sweep           : scene-adaptive CT with alpha = 0.2, 0.3, 0.5, 0.7, 1.0
  F) Method sweep          : lab_reinhard vs lab_ab_reinhard vs rgb_histmatch
  G) Single source (g050 only): as sanity check

All results compared against Akikaze clean GT.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# ---------- Config ----------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
GT_DIR = PROJECT_ROOT / "data/smoke/validation/Akikaze/test"
AKIKAZE_OUTPUT_ROOT = PROJECT_ROOT / "outputs"

# The 5 model keys used in dev champion (same training configs applied to Akikaze)
CHAMPION_MODELS = [
    "AkikazeSmoke3DCleanOnlyDCPRefinedR61G050",
    "AkikazeSmoke3DDualDepthSeed42_1000",          # might not exist for Akikaze; see fallback
    "AkikazeSmoke3DEnsembleSpatial1000",
    "AkikazeSmoke3DVGGTPriorSpatial1000",
    "AkikazeSmoke3DVGGTEnsembleSpatial1000",        # Akikaze version of VGGTEnsVGGTPrior
]

# Fallbacks for Akikaze models that might have slightly different names
AKIKAZE_FALLBACKS = {
    "AkikazeSmoke3DDualDepthSeed42_1000": [
        "AkikazeSmoke3DVGGTDualDepth2000",
        "AkikazeSmoke3DDepthPathDepthSupDualDepth",
    ],
    "AkikazeSmoke3DVGGTEnsembleSpatial1000": [
        "AkikazeSmoke3DVGGTEnsembleSpatial1000",
        "AkikazeSmoke3DVGGTDoubleSpatial1000",
    ],
    "AkikazeSmoke3DVGGTPriorSpatial1000": [
        "AkikazeSmoke3DVGGTPriorSpatial2000",
        "AkikazeSmoke3DVGGTPriorSpatial3000",
    ],
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def find_test_renders(model_name: str) -> Path | None:
    """Find the test render directory for a model."""
    base = AKIKAZE_OUTPUT_ROOT / model_name
    if not base.exists():
        return None
    runs = sorted(base.iterdir())
    for run_dir in runs:
        test_dir = run_dir / "test"
        if test_dir.exists() and any(
            f.suffix.lower() in IMAGE_EXTS for f in test_dir.iterdir()
        ):
            return test_dir
    return None


def load_images(directory: Path) -> dict[str, np.ndarray]:
    """Load all images from directory as float32 [0,1] RGB arrays, keyed by normalized name."""
    images = {}
    for f in sorted(directory.iterdir()):
        if f.suffix.lower() not in IMAGE_EXTS:
            continue
        key = normalize_key(f.name)
        img = np.asarray(Image.open(f).convert("RGB"), dtype=np.float32) / 255.0
        images[key] = img
    return images


def normalize_key(name: str) -> str:
    """Normalize image filename to a common key."""
    base = Path(name).stem.lower()
    for prefix in ("test_", "val_"):
        if base.startswith(prefix):
            base = base[len(prefix):]
    for suffix in (".jpg", ".png", ".jpeg"):
        if base.endswith(suffix):
            base = base[:-len(suffix)]
    return base


def geometric_mean(images_list: list[np.ndarray]) -> np.ndarray:
    """Compute pixel-wise geometric mean of multiple images."""
    log_sum = np.zeros_like(images_list[0])
    for img in images_list:
        log_sum += np.log(np.clip(img, 1e-7, 1.0))
    return np.exp(log_sum / len(images_list))


def arithmetic_mean(images_list: list[np.ndarray]) -> np.ndarray:
    return np.mean(images_list, axis=0)


def lab_reinhard(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Reinhard color transfer in LAB space."""
    from skimage import color as skcolor
    src_lab = skcolor.rgb2lab(source)
    ref_lab = skcolor.rgb2lab(reference)
    result = np.copy(src_lab)
    for c in range(3):
        src_mean, src_std = src_lab[:, :, c].mean(), src_lab[:, :, c].std() + 1e-8
        ref_mean, ref_std = ref_lab[:, :, c].mean(), ref_lab[:, :, c].std() + 1e-8
        result[:, :, c] = (src_lab[:, :, c] - src_mean) * (ref_std / src_std) + ref_mean
    return np.clip(skcolor.lab2rgb(result), 0, 1)


def lab_ab_reinhard(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Reinhard color transfer in LAB space, only on a/b channels (preserve L)."""
    from skimage import color as skcolor
    src_lab = skcolor.rgb2lab(source)
    ref_lab = skcolor.rgb2lab(reference)
    result = np.copy(src_lab)
    for c in [1, 2]:  # only a, b
        src_mean, src_std = src_lab[:, :, c].mean(), src_lab[:, :, c].std() + 1e-8
        ref_mean, ref_std = ref_lab[:, :, c].mean(), ref_lab[:, :, c].std() + 1e-8
        result[:, :, c] = (src_lab[:, :, c] - src_mean) * (ref_std / src_std) + ref_mean
    return np.clip(skcolor.lab2rgb(result), 0, 1)


def rgb_histmatch(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Histogram matching per channel."""
    from skimage.exposure import match_histograms
    return np.clip(match_histograms(source, reference, channel_axis=2), 0, 1)


def apply_ct(source: np.ndarray, reference: np.ndarray, method: str, alpha: float) -> np.ndarray:
    """Apply color transfer with blending."""
    if method == "lab_reinhard":
        transferred = lab_reinhard(source, reference)
    elif method == "lab_ab_reinhard":
        transferred = lab_ab_reinhard(source, reference)
    elif method == "rgb_histmatch":
        transferred = rgb_histmatch(source, reference)
    elif method == "identity":
        return source
    else:
        raise ValueError(f"Unknown method: {method}")
    return np.clip(source * (1 - alpha) + transferred * alpha, 0, 1)


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur."""
    from PIL import ImageFilter
    pil = Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8))
    pil = pil.filter(ImageFilter.GaussianBlur(radius=sigma))
    return np.asarray(pil, dtype=np.float32) / 255.0


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Compute PSNR and SSIM."""
    pred_u8 = (np.clip(pred, 0, 1) * 255 + 0.5).astype(np.uint8)
    gt_u8 = (np.clip(gt, 0, 1) * 255 + 0.5).astype(np.uint8)
    psnr = peak_signal_noise_ratio(gt_u8, pred_u8)
    ssim = structural_similarity(gt_u8, pred_u8, channel_axis=2)
    return {"psnr": float(psnr), "ssim": float(ssim)}


@dataclass
class Variant:
    name: str
    description: str
    mean_psnr: float = 0.0
    mean_ssim: float = 0.0
    per_image: dict = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, default=PROJECT_ROOT / "runs/test_phase_ct_ablation.json")
    args = parser.parse_args()

    print("=" * 80)
    print("TEST PHASE CT ABLATION — Akikaze GT Evaluation")
    print("=" * 80)

    # Load GT
    gt_images = load_images(GT_DIR)
    print(f"\nLoaded {len(gt_images)} GT images: {list(gt_images.keys())}")

    # Find available models and load their renders
    print("\n--- Loading model renders ---")
    model_renders: dict[str, dict[str, np.ndarray]] = {}
    model_names_used = []

    for model_name in CHAMPION_MODELS:
        test_dir = find_test_renders(model_name)
        if test_dir is None and model_name in AKIKAZE_FALLBACKS:
            for fallback in AKIKAZE_FALLBACKS[model_name]:
                test_dir = find_test_renders(fallback)
                if test_dir is not None:
                    model_name = fallback
                    break
        if test_dir is None:
            print(f"  SKIP: {model_name} (no test renders)")
            continue
        renders = load_images(test_dir)
        if renders:
            model_renders[model_name] = renders
            model_names_used.append(model_name)
            print(f"  OK: {model_name} ({len(renders)} images)")

    print(f"\n  Total models loaded: {len(model_renders)}")

    # Ensure we have matching keys
    common_keys = set(gt_images.keys())
    for renders in model_renders.values():
        common_keys &= set(renders.keys())
    common_keys = sorted(common_keys)
    print(f"  Common test views: {common_keys}")

    if not common_keys:
        print("ERROR: No common image keys found!")
        sys.exit(1)

    # Build per-key image stacks
    def get_geo_mean(keys_to_use=None):
        if keys_to_use is None:
            keys_to_use = common_keys
        result = {}
        for key in keys_to_use:
            imgs = [model_renders[m][key] for m in model_names_used if key in model_renders[m]]
            result[key] = geometric_mean(imgs)
        return result

    def get_arith_mean(keys_to_use=None):
        if keys_to_use is None:
            keys_to_use = common_keys
        result = {}
        for key in keys_to_use:
            imgs = [model_renders[m][key] for m in model_names_used if key in model_renders[m]]
            result[key] = arithmetic_mean(imgs)
        return result

    # Get sharp source (g050 baseline)
    g050_name = "AkikazeSmoke3DCleanOnlyDCPRefinedR61G050"
    sharp_source = model_renders.get(g050_name, {})

    # Build reference: geo mean of ALL loaded models
    scene_ref = get_geo_mean()

    # Also get the "dev ref" simulation - we don't have dev ref images for akikaze test views,
    # but we can simulate what happens when using same-scene vs cross-scene reference
    # For this ablation, scene_ref IS the scene-adaptive version

    # ====== Run all variants ======
    variants: list[Variant] = []

    def evaluate_variant(name: str, desc: str, images: dict[str, np.ndarray]) -> Variant:
        per_image = {}
        psnrs, ssims = [], []
        for key in common_keys:
            if key not in images or key not in gt_images:
                continue
            m = compute_metrics(images[key], gt_images[key])
            per_image[key] = m
            psnrs.append(m["psnr"])
            ssims.append(m["ssim"])
        v = Variant(
            name=name,
            description=desc,
            mean_psnr=float(np.mean(psnrs)) if psnrs else 0,
            mean_ssim=float(np.mean(ssims)) if ssims else 0,
            per_image=per_image,
        )
        print(f"  {name:55s}  PSNR={v.mean_psnr:.4f}  SSIM={v.mean_ssim:.4f}")
        variants.append(v)
        return v

    print("\n--- Evaluating variants ---\n")

    # D) Raw g050 only (no fusion, no CT)
    evaluate_variant("g050_raw", "Single g050 model, no fusion, no CT, no blur", sharp_source)

    # D2) Raw geo mean (no CT, no blur)
    evaluate_variant("geo_mean_raw", "Geo mean of all models, no CT, no blur", get_geo_mean())

    # D3) Raw arith mean
    evaluate_variant("arith_mean_raw", "Arith mean of all models, no CT, no blur", get_arith_mean())

    # C) Geo mean + gauss35 (no CT)
    geo_mean_imgs = get_geo_mean()
    geo_blur = {k: gaussian_blur(v, 0.35) for k, v in geo_mean_imgs.items()}
    evaluate_variant("geo_mean_gauss35", "Geo mean + gauss35, no CT", geo_blur)

    # B) Scene-adaptive CT: geo mean ref → LAB Reinhard on sharp source → gauss35
    for method in ["lab_reinhard", "lab_ab_reinhard", "rgb_histmatch"]:
        for alpha in [0.2, 0.3, 0.5, 0.7, 1.0]:
            ct_imgs = {}
            for key in common_keys:
                if key not in sharp_source or key not in scene_ref:
                    continue
                ct_imgs[key] = apply_ct(sharp_source[key], scene_ref[key], method, alpha)
            # Without blur
            evaluate_variant(
                f"scene_ct_{method}_a{int(alpha*100):03d}_noblur",
                f"Scene-adaptive CT: {method} alpha={alpha}, no blur",
                ct_imgs,
            )
            # With gauss35
            ct_blur = {k: gaussian_blur(v, 0.35) for k, v in ct_imgs.items()}
            evaluate_variant(
                f"scene_ct_{method}_a{int(alpha*100):03d}_gauss35",
                f"Scene-adaptive CT: {method} alpha={alpha}, gauss35",
                ct_blur,
            )

    # B2) Scene-adaptive CT using geo_mean as BOTH source and gets CT'd
    for method in ["lab_reinhard"]:
        for alpha in [0.3, 0.5, 1.0]:
            ct_imgs = {}
            for key in common_keys:
                ct_imgs[key] = apply_ct(geo_mean_imgs[key], scene_ref[key], method, alpha)
            ct_blur = {k: gaussian_blur(v, 0.35) for k, v in ct_imgs.items()}
            evaluate_variant(
                f"gm_self_ct_{method}_a{int(alpha*100):03d}_gauss35",
                f"Geo mean self-CT: {method} alpha={alpha}, gauss35",
                ct_blur,
            )

    # G) Single models (sanity check)
    for model_name in model_names_used:
        if model_name == g050_name:
            continue
        renders = model_renders[model_name]
        short_name = model_name.replace("AkikazeSmoke3D", "")
        evaluate_variant(f"single_{short_name}", f"Single model: {short_name}", renders)
        # With blur
        blur_renders = {k: gaussian_blur(v, 0.35) for k, v in renders.items()}
        evaluate_variant(f"single_{short_name}_gauss35", f"Single model + gauss35: {short_name}", blur_renders)

    # Sort by PSNR
    print("\n\n" + "=" * 80)
    print("RANKING (sorted by PSNR)")
    print("=" * 80)
    variants_sorted = sorted(variants, key=lambda v: v.mean_psnr, reverse=True)
    for i, v in enumerate(variants_sorted):
        marker = " ★" if i == 0 else ""
        print(f"  {i+1:3d}. {v.name:55s}  PSNR={v.mean_psnr:.4f}  SSIM={v.mean_ssim:.4f}{marker}")

    # Save results
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(
            {
                "models_used": model_names_used,
                "gt_dir": str(GT_DIR),
                "common_keys": common_keys,
                "variants": [asdict(v) for v in variants_sorted],
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
