#!/usr/bin/env python3
"""突破口1: Color Transfer — 把 old 模型的锐利几何 + 融合图的准确色彩合体
用五方融合图(15.410)的色彩直方图去拯救 old 模型(MUSIQ 50.5, PSNR 14.08)"""

import zipfile, io, sys
from pathlib import Path
from PIL import Image
import numpy as np
from skimage import exposure

def color_transfer_histmatch(sharp_img, color_ref_img):
    """逐通道直方图匹配: 锐利骨架 + 正确外衣"""
    matched = exposure.match_histograms(sharp_img, color_ref_img, channel_axis=-1)
    return np.clip(matched, 0, 255).astype(np.uint8)

def main():
    # old 模型 (锐利但发灰)
    old_zip = 'submissions/dev_submission_smoke3d_v15mix42safe_20260312_095341.zip'
    # 五方融合 (色彩准但模糊)
    blend_zip = 'submissions/dev_submission_penta_equal_20260314.zip'
    
    print("Loading images...")
    old_imgs = {}
    blend_imgs = {}
    
    with zipfile.ZipFile(old_zip) as zf:
        for name in sorted(zf.namelist()):
            if name.lower().endswith(('.jpg', '.jpeg', '.png')):
                old_imgs[name] = np.array(Image.open(io.BytesIO(zf.read(name))))
    
    with zipfile.ZipFile(blend_zip) as zf:
        for name in sorted(zf.namelist()):
            if name.lower().endswith(('.jpg', '.jpeg', '.png')):
                blend_imgs[name] = np.array(Image.open(io.BytesIO(zf.read(name))))
    
    print(f"Old: {len(old_imgs)} images, Blend: {len(blend_imgs)} images")
    
    # Color transfer: old → blend colors
    out_dir = Path('/tmp/color_transfer_old')
    out_dir.mkdir(exist_ok=True)
    
    for name in sorted(old_imgs.keys()):
        if name not in blend_imgs:
            print(f"  Skip {name}: no blend match")
            continue
        
        sharp = old_imgs[name]
        color_ref = blend_imgs[name]
        
        # Resize if needed
        if sharp.shape != color_ref.shape:
            color_ref = np.array(Image.fromarray(color_ref).resize(
                (sharp.shape[1], sharp.shape[0])))
        
        matched = color_transfer_histmatch(sharp, color_ref)
        Image.fromarray(matched).save(out_dir / name, quality=100)
        
        if name == sorted(old_imgs.keys())[0]:
            print(f"  Before: mean={sharp.mean():.1f}, After: mean={matched.mean():.1f}, Ref: mean={color_ref.mean():.1f}")
    
    # Save as submission zip
    out_zip = 'submissions/dev_submission_old_colortransfer_20260314.zip'
    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_STORED) as zf:
        for f in sorted(out_dir.iterdir()):
            zf.write(f, f.name)
    print(f"Saved: {out_zip}")
    
    # Also do color transfer with contrast boost
    from PIL import ImageEnhance
    out_dir_c = Path('/tmp/color_transfer_old_c105')
    out_dir_c.mkdir(exist_ok=True)
    for f in sorted(out_dir.iterdir()):
        enhanced = ImageEnhance.Contrast(Image.open(f)).enhance(1.05)
        enhanced.save(out_dir_c / f.name, quality=100)
    
    out_zip_c = 'submissions/dev_submission_old_ct_c105_20260314.zip'
    with zipfile.ZipFile(out_zip_c, 'w', zipfile.ZIP_STORED) as zf:
        for f in sorted(out_dir_c.iterdir()):
            zf.write(f, f.name)
    print(f"Saved: {out_zip_c}")
    
    # Also try: color transfer each model individually, then blend
    print("\n=== Per-model color transfer + blend ===")
    model_zips = {
        'old': old_zip,
        'ens': 'submissions/dev_submission_ensemble_spatial_v1_20260313_123054.zip',
        'vggt': 'submissions/dev_submission_vggt_prior_v1_20260313_140531.zip',
        'dd': 'submissions/dev_submission_dd1k_20260314_121130.zip',
        'd3r': 'submissions/dev_submission_d3r_aux_20260314_134800.zip',
    }
    
    corrected_models = {}
    for key, zpath in model_zips.items():
        corrected_models[key] = {}
        with zipfile.ZipFile(zpath) as zf:
            for name in sorted(zf.namelist()):
                if name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img = np.array(Image.open(io.BytesIO(zf.read(name))))
                    if name in blend_imgs:
                        ref = blend_imgs[name]
                        if img.shape != ref.shape:
                            ref = np.array(Image.fromarray(ref).resize((img.shape[1], img.shape[0])))
                        corrected = color_transfer_histmatch(img, ref)
                    else:
                        corrected = img
                    corrected_models[key][name] = corrected.astype(np.float64)
    
    # Blend corrected models
    for method_name, method_fn in [
        ('ct_geo5', lambda s: np.expm1(np.mean(np.log1p(s), axis=0))),
        ('ct_mean5', lambda s: np.mean(s, axis=0)),
        ('ct_trimmed5', lambda s: np.mean(np.sort(s, axis=0)[1:-1], axis=0)),
    ]:
        out_dir_m = Path(f'/tmp/{method_name}')
        out_dir_m.mkdir(exist_ok=True)
        for name in sorted(blend_imgs.keys()):
            available = [corrected_models[k][name] for k in corrected_models if name in corrected_models[k]]
            if not available: continue
            stack = np.stack(available, axis=0)
            result = method_fn(stack)
            Image.fromarray(np.clip(result, 0, 255).astype(np.uint8)).save(out_dir_m / name, quality=100)
        
        # With and without contrast
        for suffix, contrast in [('', None), ('_c105', 1.05)]:
            tag = f'{method_name}{suffix}'
            out_zip_m = f'submissions/dev_submission_{tag}_20260314.zip'
            with zipfile.ZipFile(out_zip_m, 'w', zipfile.ZIP_STORED) as zf:
                for f in sorted(out_dir_m.iterdir()):
                    if contrast:
                        enhanced = ImageEnhance.Contrast(Image.open(f)).enhance(contrast)
                        import tempfile
                        tmp = Path(tempfile.mktemp(suffix='.jpg'))
                        enhanced.save(tmp, quality=100)
                        zf.write(tmp, f.name)
                        tmp.unlink()
                    else:
                        zf.write(f, f.name)
            print(f"  {tag}: {out_zip_m}")
    
    print("\nALL COLOR TRANSFER DONE")

if __name__ == "__main__":
    main()
