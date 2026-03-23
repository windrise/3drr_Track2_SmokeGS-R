#!/usr/bin/env python3
"""Build OPTIMIZED test-phase submission.

KEY FINDING from ablation:
  - vggt_trio ref pool (dd2k + vggt + vggt_ens) → PSNR=18.51 on Akikaze GT
  - champ5 ref pool (all 5) → PSNR=18.10 on Akikaze GT
  - +0.42 dB just by dropping g050 and ens from the ref pool!

This script builds multiple candidates for each test scene,
and also applies the optimization to the dev scenes for comparison.
"""

from __future__ import annotations
import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from skimage import color as skcolor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

TEST_SCENES = ["Natsume", "Shirohana", "Tsubaki"]
DEV_SCENES = ["Futaba", "Hinoki", "Koharu", "Midori"]

# Optimal reference pool: only the 3 best models (no g050, no ens)
OPTIMAL_REF_MODELS = [
    "DualDepthSeed42_1000",
    "VGGTPriorSpatial1000",
    "VGGTEnsVGGTPriorSpatial1000",
]

# Source model for CT
SOURCE_MODEL = "CleanOnlyDCPRefinedR61G050"

# Full 5-model set for comparison
ALL_5_MODELS = [
    "CleanOnlyDCPRefinedR61G050",
    "DualDepthSeed42_1000",
    "EnsembleSpatial1000",
    "VGGTPriorSpatial1000",
    "VGGTEnsVGGTPriorSpatial1000",
]


def find_test_dir(scene, model_suffix):
    base = OUTPUTS_ROOT / f"{scene}Smoke3D{model_suffix}"
    if not base.exists():
        return None
    for run_dir in sorted(base.iterdir()):
        test_dir = run_dir / "test"
        if test_dir.exists():
            pngs = [f for f in test_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS]
            if pngs:
                return test_dir
    return None


def load_images(directory):
    images = {}
    for f in sorted(directory.iterdir()):
        if f.suffix.lower() not in IMAGE_EXTS:
            continue
        stem = f.stem.lower()
        for prefix in ("test_",):
            if stem.startswith(prefix):
                stem = stem[len(prefix):]
        stem = stem.replace(".jpg", "").replace(".jpeg", "").replace(".png", "")
        img = np.asarray(Image.open(f).convert("RGB"), dtype=np.float32) / 255.0
        images[stem] = img
    return images


def geometric_mean(images_list):
    log_sum = np.zeros_like(images_list[0])
    for img in images_list:
        log_sum += np.log(np.clip(img, 1e-7, 1.0))
    return np.exp(log_sum / len(images_list))


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


def to_safe_jpeg_bytes(img, quality=95):
    pil = Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality, subsampling=1)
    return buf.getvalue()


def process_scene(scene, ref_model_list, source_model=SOURCE_MODEL, alpha=1.0, blur=0.35):
    """Process one scene with specified ref pool and source model."""
    # Load source model renders
    src_dir = find_test_dir(scene, source_model)
    if src_dir is None:
        print(f"  ERROR: Source model {source_model} not found for {scene}")
        return {}
    source_renders = load_images(src_dir)

    # Load ref models
    ref_renders_list = []
    ref_names = []
    for m in ref_model_list:
        d = find_test_dir(scene, m)
        if d is not None:
            renders = load_images(d)
            if renders:
                ref_renders_list.append(renders)
                ref_names.append(m)

    if not ref_renders_list:
        print(f"  ERROR: No ref models found for {scene}")
        return {}

    common_keys = set(source_renders.keys())
    for r in ref_renders_list:
        common_keys &= set(r.keys())
    common_keys = sorted(common_keys)

    output = {}
    for idx, key in enumerate(common_keys, 1):
        source = source_renders[key]
        ref_imgs = [r[key] for r in ref_renders_list]
        reference = geometric_mean(ref_imgs)

        if alpha > 0:
            result = lab_reinhard(source, reference)
            if alpha < 1.0:
                result = source * (1 - alpha) + result * alpha
        else:
            result = source

        if blur > 0:
            result = gaussian_blur(result, blur)

        # Competition expects 0001-0004 naming, not original frame numbers
        filename = f"{scene.lower()}_{idx:04d}.JPG"
        output[filename] = to_safe_jpeg_bytes(result)

        rgb = (np.clip(result, 0, 1) * 255).mean(axis=(0, 1))
        print(f"    {filename}: RGB=({rgb[0]:.0f},{rgb[1]:.0f},{rgb[2]:.0f})")

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs/test_phase_optimized")
    parser.add_argument("--dev-champion-zip", type=Path, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        # (tag, ref_models, alpha, blur)
        ("optimal_vggt_trio_a100_g035", OPTIMAL_REF_MODELS, 1.0, 0.35),
        ("champ5_a100_g035", ALL_5_MODELS, 1.0, 0.35),
        ("optimal_vggt_trio_a100_g000", OPTIMAL_REF_MODELS, 1.0, 0.0),
        ("optimal_vggt_trio_a070_g035", OPTIMAL_REF_MODELS, 0.7, 0.35),
        ("noct_geomean5_g035", ALL_5_MODELS, 0.0, 0.35),
    ]

    for tag, ref_models, alpha, blur in configs:
        print(f"\n{'='*60}")
        print(f"  Config: {tag}")
        print(f"  Ref pool: {[m.split('_')[0] for m in ref_models]}")
        print(f"  α={alpha}, blur={blur}")
        print(f"{'='*60}")

        all_images = {}
        for scene in TEST_SCENES:
            print(f"\n  --- {scene} ---")
            scene_images = process_scene(scene, ref_models, alpha=alpha, blur=blur)
            all_images.update(scene_images)

        # Save as test-only zip
        zip_path = args.output_dir / f"test_{tag}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, data in sorted(all_images.items()):
                zf.writestr(name, data)
        print(f"\n  Saved: {zip_path} ({len(all_images)} images)")

        # If dev champion provided, build combined zip
        if args.dev_champion_zip and args.dev_champion_zip.exists():
            combined_path = args.output_dir / f"combined_{tag}.zip"
            with zipfile.ZipFile(combined_path, "w", zipfile.ZIP_DEFLATED) as zf_out:
                with zipfile.ZipFile(args.dev_champion_zip) as zf_dev:
                    for name in sorted(zf_dev.namelist()):
                        scene_prefix = name.split("_")[0].lower()
                        if scene_prefix in [s.lower() for s in DEV_SCENES]:
                            zf_out.writestr(name, zf_dev.read(name))
                for name, data in sorted(all_images.items()):
                    zf_out.writestr(name, data)
            n = len(zipfile.ZipFile(combined_path).namelist())
            print(f"  Combined: {combined_path} ({n} images)")

    print("\n✅ All configs done!")


if __name__ == "__main__":
    main()
