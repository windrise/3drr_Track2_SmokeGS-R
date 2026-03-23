#!/usr/bin/env python3
"""Blend two flat Codabench submission zips into a new submission zip."""

from __future__ import annotations

import argparse
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image


def parse_scene_weights(items: list[str]) -> dict[str, float]:
    scene_weights: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid scene weight '{item}', expected scene=weight")
        scene, value = item.split("=", 1)
        scene = scene.strip().lower()
        weight = float(value)
        if not (0.0 <= weight <= 1.0):
            raise ValueError(f"Scene weight for {scene} must be in [0, 1], got {weight}")
        scene_weights[scene] = weight
    return scene_weights


def parse_file_weights(items: list[str]) -> dict[str, float]:
    file_weights: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid file weight '{item}', expected filename=weight")
        name, value = item.split("=", 1)
        name = Path(name.strip()).name
        weight = float(value)
        if not (0.0 <= weight <= 1.0):
            raise ValueError(f"File weight for {name} must be in [0, 1], got {weight}")
        file_weights[name] = weight
    return file_weights


def sanitize_tag(raw: str, max_len: int = 48) -> str:
    tag = "".join(ch if ch.isalnum() else "_" for ch in raw).strip("_").lower()
    if len(tag) <= max_len:
        return tag
    return tag[:max_len].rstrip("_")


def blend_rgb(path_a: Path, path_b: Path, weight_b: float) -> Image.Image:
    img_a = np.asarray(Image.open(path_a).convert("RGB"), dtype=np.float32) / 255.0
    img_b = np.asarray(Image.open(path_b).convert("RGB"), dtype=np.float32) / 255.0
    blend = np.clip((1.0 - weight_b) * img_a + weight_b * img_b, 0.0, 1.0)
    return Image.fromarray((blend * 255.0 + 0.5).astype(np.uint8))


def zip_flat_dir(src_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(src_dir.iterdir()):
            if p.is_file():
                zf.write(p, arcname=p.name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-a", type=Path, required=True, help="Base submission zip.")
    parser.add_argument("--zip-b", type=Path, required=True, help="Secondary submission zip.")
    parser.add_argument(
        "--weight-b",
        type=float,
        required=True,
        help="Global blending weight for zip B. Weight for zip A is 1-weight-b.",
    )
    parser.add_argument(
        "--scene-weight-b",
        action="append",
        default=[],
        help="Optional per-scene override in form scene=weight. Example: futaba=0.2",
    )
    parser.add_argument(
        "--file-weight-b",
        action="append",
        default=[],
        help="Optional per-file override in form filename=weight. Example: futaba_0001.JPG=0.3",
    )
    parser.add_argument("--output-zip", type=Path, default=None)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args()

    if not (0.0 <= args.weight_b <= 1.0):
        raise ValueError("--weight-b must be in [0, 1]")

    scene_weights = parse_scene_weights(args.scene_weight_b)
    file_weights = parse_file_weights(args.file_weight_b)
    zip_a = args.zip_a.resolve()
    zip_b = args.zip_b.resolve()
    submissions_root = zip_a.parent
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.tag or f"blend_wb{int(args.weight_b * 1000):03d}"
    tag = sanitize_tag(tag)
    output_zip = args.output_zip or (submissions_root / f"dev_submission_{tag}_{ts}.zip")
    output_dir = submissions_root / f"dev_submit_{tag}_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td_a, tempfile.TemporaryDirectory() as td_b:
        td_a = Path(td_a)
        td_b = Path(td_b)
        with zipfile.ZipFile(zip_a) as zf:
            zf.extractall(td_a)
        with zipfile.ZipFile(zip_b) as zf:
            zf.extractall(td_b)

        files_a = sorted(p.name for p in td_a.iterdir() if p.is_file())
        files_b = sorted(p.name for p in td_b.iterdir() if p.is_file())
        if files_a != files_b:
            missing_a = sorted(set(files_b) - set(files_a))
            missing_b = sorted(set(files_a) - set(files_b))
            raise ValueError(
                f"Zip contents do not match. missing_in_a={missing_a[:4]} missing_in_b={missing_b[:4]}"
            )

        for name in files_a:
            scene = name.split("_", 1)[0].lower()
            weight_b = file_weights.get(name, scene_weights.get(scene, args.weight_b))
            out_path = output_dir / name
            if weight_b <= 0.0:
                out_path.write_bytes((td_a / name).read_bytes())
                continue
            if weight_b >= 1.0:
                out_path.write_bytes((td_b / name).read_bytes())
                continue
            out = blend_rgb(td_a / name, td_b / name, weight_b)
            out.save(out_path, format="JPEG", quality=args.jpeg_quality)

    zip_flat_dir(output_dir, output_zip)
    print(f"zip_a={zip_a}")
    print(f"zip_b={zip_b}")
    print(f"weight_b={args.weight_b}")
    if scene_weights:
        for scene, weight in sorted(scene_weights.items()):
            print(f"scene_weight_b[{scene}]={weight}")
    if file_weights:
        for name, weight in sorted(file_weights.items()):
            print(f"file_weight_b[{name}]={weight}")
    print(f"output_zip={output_zip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
