#!/usr/bin/env python3
"""Select per-file donor replacements with a deterministic image heuristic."""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import sobel
from skimage import color


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-a", type=Path, required=True, help="Base submission zip.")
    parser.add_argument("--zip-b", type=Path, required=True, help="Donor submission zip.")
    parser.add_argument("--output-zip", type=Path, required=True, help="Output submission zip.")
    parser.add_argument(
        "--selector",
        choices=("lab_chroma_luma_grad_ratio",),
        default="lab_chroma_luma_grad_ratio",
        help="Per-image selector rule.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Use donor when selector_value <= threshold.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional manifest with per-image selector values.",
    )
    return parser.parse_args()


def load_rgb(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.float32) / 255.0


def mean_grad_mag(channel: np.ndarray) -> float:
    grad_y = sobel(channel, axis=0)
    grad_x = sobel(channel, axis=1)
    return float(np.mean(np.hypot(grad_y, grad_x)))


def lab_chroma_luma_grad_ratio(rgb: np.ndarray) -> dict[str, float]:
    lab = color.rgb2lab(rgb)
    grad_l = mean_grad_mag(lab[:, :, 0])
    grad_a = mean_grad_mag(lab[:, :, 1])
    grad_b = mean_grad_mag(lab[:, :, 2])
    ratio = float((grad_a + grad_b) / (grad_l + 1e-6))
    return {
        "selector_value": ratio,
        "mean_luma_grad": grad_l,
        "mean_chroma_grad": float(grad_a + grad_b),
    }


def evaluate_selector(selector: str, rgb: np.ndarray) -> dict[str, float]:
    if selector == "lab_chroma_luma_grad_ratio":
        return lab_chroma_luma_grad_ratio(rgb)
    raise ValueError(f"Unsupported selector: {selector}")


def main() -> int:
    args = parse_args()
    zip_a = args.zip_a.resolve()
    zip_b = args.zip_b.resolve()
    output_zip = args.output_zip.resolve()

    if not zip_a.exists():
        raise FileNotFoundError(f"Base zip does not exist: {zip_a}")
    if not zip_b.exists():
        raise FileNotFoundError(f"Donor zip does not exist: {zip_b}")

    output_zip.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_a) as za, zipfile.ZipFile(zip_b) as zb:
        names_a = sorted(Path(name).name for name in za.namelist())
        names_b = sorted(Path(name).name for name in zb.namelist())
        if names_a != names_b:
            missing_a = sorted(set(names_b) - set(names_a))
            missing_b = sorted(set(names_a) - set(names_b))
            raise ValueError(
                f"Zip contents do not match. missing_in_a={missing_a[:4]} missing_in_b={missing_b[:4]}"
            )

        per_image: list[dict[str, object]] = []
        selected_files: list[str] = []

        with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as dst:
            for name in names_a:
                data_a = za.read(name)
                data_b = zb.read(name)
                out_bytes = data_a
                selector_stats: dict[str, object] = {
                    "file": name,
                    "selected_zip": "a",
                }

                if Path(name).suffix.lower() in IMAGE_EXTS:
                    rgb_a = load_rgb(data_a)
                    metrics = evaluate_selector(args.selector, rgb_a)
                    selector_value = float(metrics["selector_value"])
                    use_b = selector_value <= args.threshold
                    selector_stats.update(metrics)
                    selector_stats["selected_zip"] = "b" if use_b else "a"
                    selector_stats["threshold"] = float(args.threshold)
                    if use_b:
                        out_bytes = data_b
                        selected_files.append(name)

                per_image.append(selector_stats)
                dst.writestr(name, out_bytes)

    manifest = {
        "zip_a": str(zip_a),
        "zip_b": str(zip_b),
        "output_zip": str(output_zip),
        "selector": args.selector,
        "threshold": float(args.threshold),
        "selected_files": selected_files,
        "selected_count": len(selected_files),
        "per_image": per_image,
    }

    if args.json_out:
        json_out = args.json_out.resolve()
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"json_out={json_out}")

    print(f"zip_a={zip_a}")
    print(f"zip_b={zip_b}")
    print(f"selector={args.selector}")
    print(f"threshold={args.threshold}")
    print(f"selected_count={len(selected_files)}")
    if selected_files:
        for name in selected_files:
            print(f"selected_file={name}")
    print(f"output_zip={output_zip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
