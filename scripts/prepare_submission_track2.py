#!/usr/bin/env python3
"""Build a single Track 2 submission zip from frozen dev artifacts plus test renders."""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from datetime import datetime
from pathlib import Path

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def find_latest_run_dir(outputs_root: Path, exp_prefix: str) -> Path:
    candidates = sorted(
        [p for p in outputs_root.glob(f"{exp_prefix}/*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No run dir found for {exp_prefix} under {outputs_root}")
    return candidates[0]


def img_to_float(img: Image.Image):
    import numpy as np

    return np.asarray(img, dtype=np.float32) / 255.0


def sanitize_zip_tag(raw_tag: str, max_len: int = 40) -> str:
    tag = "".join(ch if ch.isalnum() else "_" for ch in raw_tag).strip("_").lower()
    if len(tag) <= max_len:
        return tag
    return tag[:max_len].rstrip("_")


def import_flat_zip(zip_path: Path, out_dir: Path) -> int:
    count = 0
    with zipfile.ZipFile(zip_path) as zf:
        for name in sorted(zf.namelist()):
            base = Path(name).name
            if base != name:
                raise ValueError(f"zip is not flat: {name}")
            if Path(base).suffix.lower() not in IMAGE_EXTS:
                continue
            out_path = out_dir / base
            if out_path.exists():
                raise ValueError(f"duplicate file during include-zip merge: {base}")
            out_path.write_bytes(zf.read(name))
            count += 1
    return count


def scene_submission_images(
    scene_dir: Path,
    run_test_dir: Path,
    scene_name_lower: str,
    out_dir: Path,
    post_gamma: float,
    jpeg_quality: int,
    jpeg_subsampling: int | None,
) -> int:
    tf_path = scene_dir / "transforms_test.json"
    data = json.loads(tf_path.read_text(encoding="utf-8"))
    frames = data.get("frames", [])
    if not frames:
        raise ValueError(f"No frames in {tf_path}")

    count = 0
    for i, frame in enumerate(frames, start=1):
        frame_path = frame["file_path"]
        base = Path(frame_path).name
        pred_name = f"test_{base}.png"
        pred_path = run_test_dir / pred_name
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing rendered image: {pred_path}")

        out_name = f"{scene_name_lower}_{i:04d}.JPG"
        out_path = out_dir / out_name
        if out_path.exists():
            raise ValueError(f"duplicate output name: {out_name}")

        img = Image.open(pred_path).convert("RGB")
        if post_gamma != 1.0:
            arr = (img_to_float(img) ** post_gamma * 255.0 + 0.5).astype("uint8")
            img = Image.fromarray(arr)
        save_kwargs = {"format": "JPEG", "quality": jpeg_quality}
        if jpeg_subsampling is not None:
            save_kwargs["subsampling"] = jpeg_subsampling
        img.save(out_path, **save_kwargs)
        count += 1
    return count


def zip_flat_dir(src_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(src_dir.iterdir()):
            if p.is_file():
                zf.write(p, arcname=p.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--include-zip",
        action="append",
        default=[],
        type=Path,
        help="Existing flat submission zip to merge in verbatim. Repeat as needed.",
    )
    parser.add_argument(
        "--scene-root",
        type=Path,
        default=None,
        help="Root containing scene folders with transforms_test.json, e.g. data/smoke/test.",
    )
    parser.add_argument(
        "--scene-names",
        nargs="+",
        default=[],
        help="Scene folders to package from --scene-root.",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=None,
        help="Optional outputs root used with --exp-prefix-template.",
    )
    parser.add_argument(
        "--exp-prefix-template",
        type=str,
        default=None,
        help="Experiment prefix template used to find latest scene runs, e.g. '{scene}Smoke3DCleanOnlyDCPRefinedR61G050'.",
    )
    parser.add_argument("--zip-tag", type=str, required=True, help="Short tag used in output zip filename.")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--jpeg-subsampling", type=int, default=2)
    parser.add_argument("--post-gamma", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    submissions_root = root / "submissions"
    submissions_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = submissions_root / f"track2_submit_{ts}"
    work_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for include_zip in args.include_zip:
        zip_path = include_zip if include_zip.is_absolute() else (root / include_zip)
        total += import_flat_zip(zip_path.resolve(), work_dir)
        print(f"merged_zip={zip_path.resolve()}")

    if args.scene_names:
        if args.scene_root is None or args.outputs_root is None or args.exp_prefix_template is None:
            raise ValueError("scene packaging requires --scene-root, --outputs-root, and --exp-prefix-template")
        scene_root = args.scene_root.resolve()
        outputs_root = args.outputs_root.resolve()

        for scene in args.scene_names:
            scene_dir = scene_root / scene
            exp_prefix = args.exp_prefix_template.format(scene=scene, scene_lower=scene.lower())
            run_dir = find_latest_run_dir(outputs_root, exp_prefix)
            test_dir = run_dir / "test"
            count = scene_submission_images(
                scene_dir=scene_dir,
                run_test_dir=test_dir,
                scene_name_lower=scene.lower(),
                out_dir=work_dir,
                post_gamma=args.post_gamma,
                jpeg_quality=args.jpeg_quality,
                jpeg_subsampling=args.jpeg_subsampling,
            )
            total += count
            print(f"scene={scene} run={run_dir.name} images={count}")

    zip_tag = sanitize_zip_tag(args.zip_tag)
    zip_path = submissions_root / f"track2_submission_{zip_tag}_{ts}.zip"
    zip_flat_dir(work_dir, zip_path)
    print(f"total_images={total}")
    print(f"submission_zip={zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
