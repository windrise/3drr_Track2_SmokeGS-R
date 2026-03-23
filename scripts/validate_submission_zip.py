#!/usr/bin/env python3
"""Validate a flat Codabench submission zip."""

from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image


DEFAULT_DEV_SCENES = ("futaba", "hinoki", "koharu", "midori")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, required=True, help="Submission zip to validate.")
    parser.add_argument(
        "--expected-scene",
        action="append",
        dest="expected_scenes",
        default=None,
        help="Expected scene name. Repeat to constrain scene set.",
    )
    parser.add_argument(
        "--expected-total",
        type=int,
        default=16,
        help="Expected total number of images. Use 0 to disable.",
    )
    parser.add_argument(
        "--expected-per-scene",
        type=int,
        default=4,
        help="Expected number of images per scene. Use 0 to disable.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional JSON report path.",
    )
    return parser.parse_args()


def validate_jpeg(raw: bytes, name: str) -> dict[str, object]:
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        fmt = image.format
        width, height = image.size
    if fmt != "JPEG":
        raise ValueError(f"{name}: expected JPEG payload, got {fmt}")
    return {
        "format": fmt,
        "width": width,
        "height": height,
    }


def main() -> int:
    args = parse_args()
    zip_path = args.zip.resolve()
    expected_scenes = tuple(args.expected_scenes or DEFAULT_DEV_SCENES)

    if not zip_path.exists():
        raise FileNotFoundError(f"Zip does not exist: {zip_path}")

    scene_pat = "|".join(re.escape(scene) for scene in expected_scenes) if expected_scenes else r"[a-z0-9]+"
    name_re = re.compile(rf"^(?P<scene>{scene_pat})_(?P<idx>\d{{4}})\.JPG$")

    issues: list[str] = []
    per_scene = Counter()
    per_file: list[dict[str, object]] = []

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        if args.expected_total > 0 and len(names) != args.expected_total:
            issues.append(
                f"expected_total={args.expected_total}, actual_total={len(names)}"
            )

        for name in names:
            if Path(name).name != name:
                issues.append(f"non_flat_entry={name}")
                continue

            match = name_re.match(name)
            if not match:
                issues.append(f"invalid_name={name}")
                continue

            scene = match.group("scene")
            per_scene[scene] += 1

            try:
                image_info = validate_jpeg(zf.read(name), name)
            except Exception as exc:  # noqa: BLE001
                issues.append(f"invalid_jpeg={name}: {exc}")
                continue

            per_file.append(
                {
                    "name": name,
                    "scene": scene,
                    **image_info,
                }
            )

    if args.expected_per_scene > 0:
        for scene in expected_scenes:
            count = per_scene.get(scene, 0)
            if count != args.expected_per_scene:
                issues.append(
                    f"scene_count[{scene}] expected={args.expected_per_scene} actual={count}"
                )

    ok = not issues
    report = {
        "zip": str(zip_path),
        "ok": ok,
        "expected_scenes": list(expected_scenes),
        "expected_total": args.expected_total,
        "expected_per_scene": args.expected_per_scene,
        "actual_total": sum(per_scene.values()),
        "per_scene_counts": dict(sorted(per_scene.items())),
        "issues": issues,
        "per_file": per_file,
    }

    if args.json_out:
        json_out = args.json_out.resolve()
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"json_out={json_out}")

    print(f"zip={zip_path}")
    print(f"ok={ok}")
    print(f"actual_total={report['actual_total']}")
    for scene, count in sorted(per_scene.items()):
        print(f"scene_count[{scene}]={count}")
    if issues:
        for issue in issues:
            print(f"issue={issue}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
