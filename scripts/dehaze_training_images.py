#!/usr/bin/env python3
"""Batch dehaze training images using DehazeFormer-L to replace DCP R61 pseudo-clean."""

import sys
import argparse
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEHAZE_ROOT = PROJECT_ROOT / "methods" / "foundation" / "DehazeFormer"
if str(DEHAZE_ROOT) not in sys.path:
    sys.path.insert(0, str(DEHAZE_ROOT))


def _load_dehazeformer_factories():
    """Import DehazeFormer constructors from an upstream checkout."""
    try:
        from models import dehazeformer_b, dehazeformer_l, dehazeformer_m, dehazeformer_s
    except Exception as exc:
        raise ImportError(
            "DehazeFormer source is not bundled in this release package.\n"
            "To enable scripts/dehaze_training_images.py, run:\n"
            "  rm -rf methods/foundation/DehazeFormer\n"
            "  git clone --recursive https://github.com/IDKiro/DehazeFormer "
            "methods/foundation/DehazeFormer\n"
            "Then download the official checkpoints into "
            "methods/foundation/DehazeFormer/saved_models/ or pass --weights."
        ) from exc
    return {
        "dehazeformer-l": dehazeformer_l,
        "dehazeformer-b": dehazeformer_b,
        "dehazeformer-m": dehazeformer_m,
        "dehazeformer-s": dehazeformer_s,
    }


def load_model(model_name: str, weights_path: str, device: str = "cuda"):
    """Load DehazeFormer model with pretrained weights."""
    model_map = _load_dehazeformer_factories()

    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(model_map.keys())}")

    model = model_map[model_name]()

    # Load weights (handle DataParallel prefix)
    state_dict = torch.load(weights_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    return model.to(device).eval()


def dehaze_image(model, img_tensor, device="cuda", tile_size=512, overlap=64):
    """Dehaze a single image with tiling to handle high-res inputs."""
    _, _, H, W = img_tensor.shape

    # If image is small enough, process directly
    if H <= tile_size and W <= tile_size:
        # Pad to multiple of 8
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        if pad_h > 0 or pad_w > 0:
            img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
        with torch.no_grad():
            output = model(img_tensor.to(device))
        if pad_h > 0 or pad_w > 0:
            output = output[:, :, :H, :W]
        return output.clamp(0, 1)

    # Tile-based processing for high-res images
    stride = tile_size - overlap
    output = torch.zeros(1, 3, H, W, device=device)
    weight = torch.zeros(1, 1, H, W, device=device)

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y_end = min(y + tile_size, H)
            x_end = min(x + tile_size, W)
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)

            tile = img_tensor[:, :, y_start:y_end, x_start:x_end]
            th, tw = tile.shape[2], tile.shape[3]

            # Pad to multiple of 8
            pad_h = (8 - th % 8) % 8
            pad_w = (8 - tw % 8) % 8
            if pad_h > 0 or pad_w > 0:
                tile = F.pad(tile, (0, pad_w, 0, pad_h), mode='reflect')

            with torch.no_grad():
                tile_out = model(tile.to(device))

            if pad_h > 0 or pad_w > 0:
                tile_out = tile_out[:, :, :th, :tw]

            output[:, :, y_start:y_end, x_start:x_end] += tile_out
            weight[:, :, y_start:y_end, x_start:x_end] += 1

    output = output / weight.clamp_min(1)
    return output.clamp(0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, help="Scene name (e.g. Akikaze)")
    parser.add_argument("--split", default="validation", choices=["development", "validation", "test"])
    parser.add_argument("--model-name", default="dehazeformer-l")
    parser.add_argument("--weights", default=None, help="Path to weights (auto-detected if not set)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tile-size", type=int, default=512)
    args = parser.parse_args()

    # Auto-detect weights
    if args.weights is None:
        weights_map = {
            "dehazeformer-l": "dehazeformer-l.pth",
            "dehazeformer-b": "dehazeformer-b.pth",
            "dehazeformer-m": "dehazeformer-m.pth",
            "dehazeformer-s": "dehazeformer-s.pth",
        }
        candidates = [
            DEHAZE_ROOT / "saved_models" / "indoor" / weights_map[args.model_name],
            DEHAZE_ROOT / "saved_models" / weights_map[args.model_name],
        ]
        existing = next((path for path in candidates if path.exists()), None)
        if existing is None:
            raise FileNotFoundError(
                "No DehazeFormer checkpoints were found under the upstream checkout. "
                "Pass --weights <checkpoint> or place checkpoints under "
                f"{DEHAZE_ROOT / 'saved_models' / 'indoor'} after cloning the upstream repo."
            )
        args.weights = str(existing)

    print(f"Loading {args.model_name} from {args.weights}...")
    model = load_model(args.model_name, args.weights, args.device)

    # Process training images
    img_dir = PROJECT_ROOT / "data" / "smoke" / args.split / args.scene / "train"
    out_dir = PROJECT_ROOT / "data" / "smoke_dehazeformer" / args.split / args.scene / "train"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(img_dir.glob("*.JPG"))
    print(f"Processing {len(frames)} training images for {args.scene}...")

    for i, img_path in enumerate(frames):
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)

        output = dehaze_image(model, img_tensor, args.device, args.tile_size)
        output_np = (output[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # Save with original name (no prefix)
        out_name = img_path.stem.replace("train_", "") + ".JPG"
        Image.fromarray(output_np).save(out_dir / out_name, quality=98)

        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{i+1}/{len(frames)}] {img_path.name} -> {out_name}")

    print(f"Done! {len(frames)} dehazed images saved to {out_dir}")


if __name__ == "__main__":
    main()
