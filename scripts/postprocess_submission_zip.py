#!/usr/bin/env python3
"""Apply image-space postprocessing to a flat submission zip."""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter
from skimage import color


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-zip", type=Path, required=True, help="Flat submission zip to postprocess.")
    parser.add_argument("--output-zip", type=Path, required=True, help="Output zip path.")
    parser.add_argument(
        "--gaussian-radius",
        type=float,
        default=0.0,
        help="Optional PIL Gaussian blur radius. Use 0 to disable.",
    )
    parser.add_argument(
        "--lab-ab-gaussian-sigma",
        type=float,
        default=0.0,
        help="Optional Gaussian sigma applied only to LAB a/b channels before RGB re-encode.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality for re-encoding.")
    parser.add_argument(
        "--jpeg-subsampling",
        type=int,
        default=2,
        help="JPEG subsampling mode. PIL uses 2 for 4:2:0.",
    )
    parser.add_argument(
        "--progressive",
        action="store_true",
        help="Enable progressive JPEG. Disabled by default for scorer safety.",
    )
    return parser.parse_args()


def postprocess_image(
    data: bytes,
    gaussian_radius: float,
    lab_ab_gaussian_sigma: float,
    jpeg_quality: int,
    jpeg_subsampling: int,
    progressive: bool,
) -> bytes:
    image = Image.open(io.BytesIO(data)).convert("RGB")
    if lab_ab_gaussian_sigma > 0.0:
        rgb = np.asarray(image, dtype=np.float32) / 255.0
        lab = color.rgb2lab(rgb)
        lab[:, :, 1] = gaussian_filter(lab[:, :, 1], sigma=lab_ab_gaussian_sigma)
        lab[:, :, 2] = gaussian_filter(lab[:, :, 2], sigma=lab_ab_gaussian_sigma)
        rgb = np.clip(color.lab2rgb(lab), 0.0, 1.0)
        image = Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8))
    if gaussian_radius > 0.0:
        image = image.filter(ImageFilter.GaussianBlur(radius=gaussian_radius))
    buffer = io.BytesIO()
    image.save(
        buffer,
        format="JPEG",
        quality=jpeg_quality,
        subsampling=jpeg_subsampling,
        progressive=progressive,
    )
    return buffer.getvalue()


def main() -> int:
    args = parse_args()
    input_zip = args.input_zip.resolve()
    output_zip = args.output_zip.resolve()

    if not input_zip.exists():
        raise FileNotFoundError(f"Input zip does not exist: {input_zip}")

    output_zip.parent.mkdir(parents=True, exist_ok=True)

    image_count = 0
    with zipfile.ZipFile(input_zip) as src, zipfile.ZipFile(
        output_zip,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as dst:
        for name in sorted(src.namelist()):
            suffix = Path(name).suffix.lower()
            raw = src.read(name)
            if suffix in IMAGE_EXTS:
                raw = postprocess_image(
                    raw,
                    gaussian_radius=args.gaussian_radius,
                    lab_ab_gaussian_sigma=args.lab_ab_gaussian_sigma,
                    jpeg_quality=args.jpeg_quality,
                    jpeg_subsampling=args.jpeg_subsampling,
                    progressive=args.progressive,
                )
                image_count += 1
            dst.writestr(Path(name).name, raw)

    print(f"input_zip={input_zip}")
    print(f"output_zip={output_zip}")
    print(f"gaussian_radius={args.gaussian_radius}")
    print(f"lab_ab_gaussian_sigma={args.lab_ab_gaussian_sigma}")
    print(f"jpeg_quality={args.jpeg_quality}")
    print(f"jpeg_subsampling={args.jpeg_subsampling}")
    print(f"progressive={args.progressive}")
    print(f"image_count={image_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
