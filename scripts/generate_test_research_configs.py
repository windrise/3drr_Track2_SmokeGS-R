#!/usr/bin/env python3
"""Instantiate test-scene research configs from an existing dev-scene template family."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--template",
        action="append",
        required=True,
        type=Path,
        help="Template YAML path. Repeat for each config family to clone.",
    )
    parser.add_argument(
        "--scene",
        action="append",
        required=True,
        help="Target test scene name, e.g. Natsume. Repeat for multiple scenes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("configs/test_phase_research"),
        help="Where generated configs are written.",
    )
    parser.add_argument(
        "--source-split",
        type=str,
        default="development",
        help="Split string used by the template family.",
    )
    parser.add_argument(
        "--target-split",
        type=str,
        default="test",
        help="Split string used in generated configs.",
    )
    return parser.parse_args()


def infer_template_scene(template_path: Path) -> str:
    stem = template_path.stem
    if "_" not in stem:
        raise ValueError(f"cannot infer scene prefix from template filename: {template_path}")
    return stem.split("_", 1)[0].capitalize()


def rewrite_text(raw: str, source_scene: str, target_scene: str, source_split: str, target_split: str) -> str:
    replacements = [
        (f"/{source_split}/", f"/{target_split}/"),
        (source_scene.lower(), target_scene.lower()),
        (source_scene, target_scene),
    ]
    out = raw
    for src, dst in replacements:
        out = out.replace(src, dst)
    return out


def build_output_name(template_path: Path, source_scene: str, target_scene: str) -> str:
    stem = template_path.stem
    prefix = source_scene.lower() + "_"
    if stem.startswith(prefix):
        stem = target_scene.lower() + "_" + stem[len(prefix) :]
    else:
        stem = stem.replace(source_scene.lower(), target_scene.lower(), 1)
    return stem + template_path.suffix


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for template_path in args.template:
        template_path = template_path.resolve()
        if not template_path.exists():
            raise FileNotFoundError(f"missing template: {template_path}")
        source_scene = infer_template_scene(template_path)
        template_text = template_path.read_text(encoding="utf-8")

        for target_scene in args.scene:
            rewritten = rewrite_text(
                raw=template_text,
                source_scene=source_scene,
                target_scene=target_scene,
                source_split=args.source_split,
                target_split=args.target_split,
            )
            out_name = build_output_name(template_path, source_scene, target_scene)
            out_path = args.output_dir / out_name
            out_path.write_text(rewritten, encoding="utf-8")
            print(f"generated {out_path}")
            total += 1

    print(f"total={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
