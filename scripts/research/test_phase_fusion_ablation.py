#!/usr/bin/env python3
"""Comprehensive test-phase fusion ablation on Akikaze GT.

Tests various fusion strategies beyond simple geo_mean + CT:
1. Different model subsets (best 3 vs all 5)
2. Weighted fusion based on model quality  
3. Median fusion (robust to outliers)
4. Trimmed mean
5. Alpha-weighted blending
6. Source model optimization

The key insight is: if scene-adaptive CT ≈ dev CT (both ~18.0 on Akikaze),
then the bottleneck is the MODEL QUALITY, not the post-processing.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import itertools
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from skimage import color as skcolor
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GT_DIR = PROJECT_ROOT / "data/smoke/validation/Akikaze/test"
OUTPUTS = PROJECT_ROOT / "outputs"
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

# All available Akikaze models with test renders
ALL_MODELS = {
    "g050": "AkikazeSmoke3DCleanOnlyDCPRefinedR61G050",
    "dd2k": "AkikazeSmoke3DVGGTDualDepth2000",
    "ens": "AkikazeSmoke3DEnsembleSpatial1000",
    "vggt": "AkikazeSmoke3DVGGTPriorSpatial1000",
    "vggt_ens": "AkikazeSmoke3DVGGTEnsembleSpatial1000",
    "vggt2k": "AkikazeSmoke3DVGGTPriorSpatial2000", 
    "vggt3k": "AkikazeSmoke3DVGGTPriorSpatial3000",
    "pmap_full": "AkikazeSmoke3DDepthPathDepthSupPointmapFull1000",
    "pmap_spatial": "AkikazeSmoke3DPointmapFullSpatial1000",
    "ens_clean": "AkikazeSmoke3DEnsembleCleanPmap1000",
    "conf_ens": "AkikazeSmoke3DConfEnsembleSpatial1000",
    "ens50": "AkikazeSmoke3DEnsemble50Spatial1000",
    "ens60": "AkikazeSmoke3DEnsemble60Spatial1000",
    "ens70": "AkikazeSmoke3DEnsemble70Spatial1000",
    "old10k": "AkikazeSmoke3DOld10000",
    "planA": "AkikazeSmoke3DPlanADehaze5000",
    "planC": "AkikazeSmoke3DPlanCResidual5000",
    "pow3r": "AkikazeSmoke3DPow3R2000",
    "seed123": "AkikazeSmoke3DSeed123",
    "distill": "AkikazeSmoke3DDistilled2000",
    "highclean": "AkikazeSmoke3DHighClean",
    "db_mvp": "AkikazeSmoke3DDualBranchMVP",
    "db_v2": "AkikazeSmoke3DDualBranchV2",
}


def normalize_key(name):
    base = Path(name).stem.lower()
    for prefix in ("test_", "val_"):
        if base.startswith(prefix):
            base = base[len(prefix):]
    base = base.replace(".jpg", "").replace(".png", "")
    return base


def load_images(directory):
    images = {}
    for f in sorted(directory.iterdir()):
        if f.suffix.lower() not in IMAGE_EXTS:
            continue
        key = normalize_key(f.name)
        img = np.asarray(Image.open(f).convert("RGB"), dtype=np.float32) / 255.0
        images[key] = img
    return images


def find_test_dir(model_name):
    base = OUTPUTS / model_name
    if not base.exists():
        return None
    for run_dir in sorted(base.iterdir()):
        test_dir = run_dir / "test"
        if test_dir.exists() and any(f.suffix.lower() in IMAGE_EXTS for f in test_dir.iterdir()):
            return test_dir
    return None


def geometric_mean(imgs):
    log_sum = np.zeros_like(imgs[0])
    for img in imgs:
        log_sum += np.log(np.clip(img, 1e-7, 1.0))
    return np.exp(log_sum / len(imgs))


def median_fusion(imgs):
    return np.median(np.stack(imgs), axis=0)


def trimmed_mean(imgs, trim_frac=0.2):
    """Trimmed mean - discard extreme values per pixel."""
    stack = np.stack(imgs)
    n = len(imgs)
    k = max(1, int(n * trim_frac))
    stack_sorted = np.sort(stack, axis=0)
    return stack_sorted[k:-k].mean(axis=0) if k < n // 2 else stack_sorted.mean(axis=0)


def lab_reinhard(source, reference):
    src_lab = skcolor.rgb2lab(source)
    ref_lab = skcolor.rgb2lab(reference)
    result = np.copy(src_lab)
    for c in range(3):
        sm, ss = src_lab[:, :, c].mean(), src_lab[:, :, c].std() + 1e-8
        rm, rs = ref_lab[:, :, c].mean(), ref_lab[:, :, c].std() + 1e-8
        result[:, :, c] = (src_lab[:, :, c] - sm) * (rs / ss) + rm
    return np.clip(skcolor.lab2rgb(result), 0, 1)


def gaussian_blur(img, sigma=0.35):
    pil = Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8))
    pil = pil.filter(ImageFilter.GaussianBlur(radius=sigma))
    return np.asarray(pil, dtype=np.float32) / 255.0


def compute_psnr_ssim(pred, gt):
    p = (np.clip(pred, 0, 1) * 255 + 0.5).astype(np.uint8)
    g = (np.clip(gt, 0, 1) * 255 + 0.5).astype(np.uint8)
    return peak_signal_noise_ratio(g, p), structural_similarity(g, p, channel_axis=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, default=PROJECT_ROOT / "runs/test_phase_fusion_ablation.json")
    args = parser.parse_args()

    print("=" * 80)
    print("TEST PHASE FUSION ABLATION — Akikaze GT")
    print("=" * 80)

    # Load GT
    gt_images = load_images(GT_DIR)
    print(f"GT images: {list(gt_images.keys())}")

    # Load all available models
    model_renders = {}
    for short, full in ALL_MODELS.items():
        td = find_test_dir(full)
        if td is not None:
            renders = load_images(td)
            if renders:
                model_renders[short] = renders
                print(f"  OK: {short:15s} -> {full}")

    print(f"\nTotal models: {len(model_renders)}")

    common_keys = set(gt_images.keys())
    for r in model_renders.values():
        common_keys &= set(r.keys())
    common_keys = sorted(common_keys)
    print(f"Common keys: {common_keys}")

    results = []

    def evaluate(name, images):
        psnrs, ssims = [], []
        for k in common_keys:
            if k in images and k in gt_images:
                p, s = compute_psnr_ssim(images[k], gt_images[k])
                psnrs.append(p)
                ssims.append(s)
        mp = float(np.mean(psnrs)) if psnrs else 0
        ms = float(np.mean(ssims)) if ssims else 0
        results.append({"name": name, "psnr": mp, "ssim": ms})
        print(f"  {name:60s}  PSNR={mp:.4f}  SSIM={ms:.4f}")
        return images

    print("\n--- Single model baselines ---")
    for short in sorted(model_renders.keys()):
        evaluate(f"single_{short}", model_renders[short])

    # The 5 "champion" models
    champ5 = ["g050", "dd2k", "ens", "vggt", "vggt_ens"]
    available_champ = [m for m in champ5 if m in model_renders]

    print(f"\n--- Champion 5 fusion ({available_champ}) ---")

    # Geo mean
    for keys_used, label in [
        (available_champ, "champ5"),
        (["g050", "vggt", "dd2k"], "best3"),
        (["vggt", "dd2k", "vggt_ens"], "vggt_trio"),
    ]:
        avail = [m for m in keys_used if m in model_renders]
        if len(avail) < 2:
            continue
        fused = {k: geometric_mean([model_renders[m][k] for m in avail]) for k in common_keys}
        evaluate(f"geo_mean_{label}", fused)
        
        # Arith mean
        fused_a = {k: np.mean([model_renders[m][k] for m in avail], axis=0) for k in common_keys}
        evaluate(f"arith_mean_{label}", fused_a)
        
        # Median
        fused_m = {k: median_fusion([model_renders[m][k] for m in avail]) for k in common_keys}
        evaluate(f"median_{label}", fused_m)

    print("\n--- Fusion + scene CT (LAB Reinhard α=1.0) ---")
    # Scene CT using g050 as source, geo_mean as ref
    for keys_used, label in [
        (available_champ, "champ5"),
        (["g050", "vggt", "dd2k"], "best3"),
        (["vggt", "dd2k", "vggt_ens"], "vggt_trio"),
    ]:
        avail = [m for m in keys_used if m in model_renders]
        if len(avail) < 2:
            continue
        ref = {k: geometric_mean([model_renders[m][k] for m in avail]) for k in common_keys}
        # CT on g050
        ct = {k: lab_reinhard(model_renders["g050"][k], ref[k]) for k in common_keys}
        evaluate(f"g050_ct_ref_{label}", ct)
        ct_blur = {k: gaussian_blur(v, 0.35) for k, v in ct.items()}
        evaluate(f"g050_ct_ref_{label}_gauss35", ct_blur)

    print("\n--- CT with different source models ---")
    avail_champ = [m for m in available_champ if m in model_renders]
    ref_all = {k: geometric_mean([model_renders[m][k] for m in avail_champ]) for k in common_keys}
    
    for source_name in ["g050", "dd2k", "vggt", "vggt_ens"]:
        if source_name not in model_renders:
            continue
        ct = {k: lab_reinhard(model_renders[source_name][k], ref_all[k]) for k in common_keys}
        evaluate(f"ct_source_{source_name}_ref_champ5", ct)
        ct_blur = {k: gaussian_blur(v, 0.35) for k, v in ct.items()}
        evaluate(f"ct_source_{source_name}_ref_champ5_gauss35", ct_blur)

    print("\n--- Fusion of CT'd outputs ---")
    # CT each model, then fuse the CT'd outputs
    ct_outputs = {}
    for m in avail_champ:
        ct_outputs[m] = {k: lab_reinhard(model_renders[m][k], ref_all[k]) for k in common_keys}
    
    # Geo mean of all CT'd
    ct_geo = {k: geometric_mean([ct_outputs[m][k] for m in avail_champ]) for k in common_keys}
    evaluate("ct_then_geo_mean_champ5", ct_geo)
    evaluate("ct_then_geo_mean_champ5_gauss35", {k: gaussian_blur(v, 0.35) for k, v in ct_geo.items()})
    
    # Median of CT'd
    ct_med = {k: median_fusion([ct_outputs[m][k] for m in avail_champ]) for k in common_keys}
    evaluate("ct_then_median_champ5", ct_med)
    evaluate("ct_then_median_champ5_gauss35", {k: gaussian_blur(v, 0.35) for k, v in ct_med.items()})

    # Arith mean of CT'd  
    ct_arith = {k: np.mean([ct_outputs[m][k] for m in avail_champ], axis=0) for k in common_keys}
    evaluate("ct_then_arith_mean_champ5", ct_arith)

    print("\n--- Model selection: find best subsets ---")
    # Try all possible 3-model combinations from the champion 5
    if len(avail_champ) >= 3:
        combos_3 = list(itertools.combinations(avail_champ, 3))
        scores_3 = []
        for combo in combos_3:
            ref = {k: geometric_mean([model_renders[m][k] for m in combo]) for k in common_keys}
            ct = {k: lab_reinhard(model_renders["g050"][k], ref[k]) for k in common_keys}
            psnrs = [compute_psnr_ssim(ct[k], gt_images[k])[0] for k in common_keys]
            score = float(np.mean(psnrs))
            scores_3.append((list(combo), score))
        scores_3.sort(key=lambda x: x[1], reverse=True)
        for combo, score in scores_3[:5]:
            print(f"  3-combo {combo}: PSNR={score:.4f}")

    # Try all 4-model combos
    if len(avail_champ) >= 4:
        combos_4 = list(itertools.combinations(avail_champ, 4))
        scores_4 = []
        for combo in combos_4:
            ref = {k: geometric_mean([model_renders[m][k] for m in combo]) for k in common_keys}
            ct = {k: lab_reinhard(model_renders["g050"][k], ref[k]) for k in common_keys}
            psnrs = [compute_psnr_ssim(ct[k], gt_images[k])[0] for k in common_keys]
            score = float(np.mean(psnrs))
            scores_4.append((list(combo), score))
        scores_4.sort(key=lambda x: x[1], reverse=True)
        for combo, score in scores_4[:5]:
            print(f"  4-combo {combo}: PSNR={score:.4f}")

    print("\n--- Extended model pool (all available) ---")
    # Try larger pools from all available models
    all_model_names = sorted(model_renders.keys())
    ref_big = {k: geometric_mean([model_renders[m][k] for m in all_model_names]) for k in common_keys}
    ct_big = {k: lab_reinhard(model_renders["g050"][k], ref_big[k]) for k in common_keys}
    evaluate("g050_ct_ref_ALL_models", ct_big)
    evaluate("g050_ct_ref_ALL_models_gauss35", {k: gaussian_blur(v, 0.35) for k, v in ct_big.items()})

    # Sort and print ranking
    print("\n\n" + "=" * 80)
    print("FINAL RANKING")
    print("=" * 80)
    results.sort(key=lambda x: x["psnr"], reverse=True)
    for i, r in enumerate(results):
        marker = " ★" if i == 0 else ""
        print(f"  {i+1:3d}. {r['name']:60s}  PSNR={r['psnr']:.4f}  SSIM={r['ssim']:.4f}{marker}")

    # Save
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output_json}")


if __name__ == "__main__":
    main()
