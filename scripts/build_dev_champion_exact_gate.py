#!/usr/bin/env python3
"""Rebuild the current dev champion artifact as a deterministic sidecar pipeline."""

from __future__ import annotations

import argparse
import io
import json
import math
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_REFS = [
    "submissions/dev_submission_smoke3d_v15mix42safe_20260312_095341.zip",
    "submissions/dev_submission_ensemble_spatial_v1_20260313_123054.zip",
    "submissions/dev_submission_vggt_prior_v1_20260313_140531.zip",
    "submissions/dev_submission_dd1k_20260314_121130.zip",
    "submissions/dev_submission_vggt_ensemble_spatial_1k_20260316_125043.zip",
]

DEFAULT_MANUAL_DONOR_FILES = [
    "futaba_0002.JPG",
    "futaba_0003.JPG",
    "futaba_0004.JPG",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--source-zip",
        type=Path,
        default=Path("submissions/dev_submission_smoke3d_v15mix42safe_20260312_095341.zip"),
        help="Sharp source zip used by the winning CT branch.",
    )
    parser.add_argument(
        "--ref-zip",
        action="append",
        dest="ref_zips",
        default=None,
        help="Reference zips for the base geo LAB-Reinhard branch. Defaults to the frozen 5-zip pool.",
    )
    parser.add_argument(
        "--lab-ab-gaussian-sigma",
        type=float,
        default=0.4,
        help="Sigma used to derive the donor branch by smoothing LAB a/b channels.",
    )
    parser.add_argument(
        "--donor-mode",
        choices=("derive", "artifact"),
        default="derive",
        help="Whether to derive the donor branch from the base pipeline or reuse the frozen donor artifact.",
    )
    parser.add_argument(
        "--frozen-donor-zip",
        type=Path,
        default=Path("runs/ct_sidecar/dev_postfilter/dev_glr100_lababg05_gauss35_q95s420.zip"),
        help="Frozen donor artifact used when --donor-mode=artifact.",
    )
    parser.add_argument(
        "--base-expected-zip",
        type=Path,
        default=Path("submissions/retries/dev_glr100g35q95_r1.zip"),
        help="Known-good base zip used for byte comparison.",
    )
    parser.add_argument(
        "--donor-expected-zip",
        type=Path,
        default=Path("submissions/retries/dev_glr100lababg05g35q95_candidate.zip"),
        help="Known-good donor zip used for byte comparison.",
    )
    parser.add_argument(
        "--final-expected-zip",
        type=Path,
        default=Path("submissions/retries/dev_fut3g35_r1.zip"),
        help="Known-good final champion zip used for byte comparison.",
    )
    parser.add_argument(
        "--gate-mode",
        choices=("manual", "auto_chroma_ratio"),
        default="manual",
        help="Final donor gating mode. manual preserves the frozen file list; auto uses a heuristic.",
    )
    parser.add_argument(
        "--manual-donor-file",
        action="append",
        dest="manual_donor_files",
        default=None,
        help="Manual file replacement list used when --gate-mode=manual.",
    )
    parser.add_argument(
        "--auto-gate-threshold",
        type=float,
        default=0.8,
        help="Threshold used by --gate-mode=auto_chroma_ratio.",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path("runs/repro_dev_champion_exact_gate"),
        help="Directory where intermediate artifacts and manifest are written.",
    )
    parser.add_argument(
        "--output-zip",
        type=Path,
        default=Path("submissions/repro/dev_fut3g35_exact_gate_repro.zip"),
        help="Final reproduced zip path.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to call the child scripts.",
    )
    parser.add_argument(
        "--skip-compare",
        action="store_true",
        help="Skip byte-level comparison against frozen expected zips.",
    )
    return parser.parse_args()


def resolve_paths(project_root: Path, paths: list[Path]) -> list[Path]:
    resolved: list[Path] = []
    for path in paths:
        resolved.append(path if path.is_absolute() else (project_root / path))
    return [path.resolve() for path in resolved]


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("RUN", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def zip_bytes_equal(zip_a: Path, zip_b: Path) -> tuple[bool, list[str]]:
    with zipfile.ZipFile(zip_a) as za, zipfile.ZipFile(zip_b) as zb:
        names_a = set(za.namelist())
        names_b = set(zb.namelist())
        names = sorted(names_a | names_b)
        diff: list[str] = []
        for name in names:
            in_a = name in names_a
            in_b = name in names_b
            if in_a != in_b:
                diff.append(Path(name).name)
                continue
            if za.read(name) != zb.read(name):
                diff.append(Path(name).name)
        return len(diff) == 0, diff


def compute_psnr_bytes(raw_a: bytes, raw_b: bytes) -> float:
    image_a = np.asarray(Image.open(io.BytesIO(raw_a)).convert("RGB"), dtype=np.float32) / 255.0
    image_b = np.asarray(Image.open(io.BytesIO(raw_b)).convert("RGB"), dtype=np.float32) / 255.0
    mse = float(np.mean((image_a - image_b) ** 2))
    mse = max(mse, 1e-12)
    return 10.0 * math.log10(1.0 / mse)


def zip_image_psnr(zip_a: Path, zip_b: Path) -> dict[str, object]:
    with zipfile.ZipFile(zip_a) as za, zipfile.ZipFile(zip_b) as zb:
        shared = sorted(set(za.namelist()) & set(zb.namelist()))
        per_file: dict[str, float] = {}
        for name in shared:
            per_file[Path(name).name] = compute_psnr_bytes(za.read(name), zb.read(name))
    values = list(per_file.values())
    return {
        "mean_psnr": float(np.mean(values)) if values else None,
        "min_psnr": float(np.min(values)) if values else None,
        "worst_files": sorted(per_file.items(), key=lambda item: item[1])[:5],
    }


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    work_root = (args.work_root if args.work_root.is_absolute() else project_root / args.work_root).resolve()
    output_zip = (args.output_zip if args.output_zip.is_absolute() else project_root / args.output_zip).resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    ref_args = args.ref_zips if args.ref_zips is not None else [Path(p) for p in DEFAULT_REFS]
    manual_donor_files = args.manual_donor_files or list(DEFAULT_MANUAL_DONOR_FILES)
    source_zip = resolve_paths(project_root, [args.source_zip])[0]
    ref_zips = resolve_paths(project_root, ref_args)
    frozen_donor_zip = resolve_paths(project_root, [args.frozen_donor_zip])[0]
    base_expected_zip = resolve_paths(project_root, [args.base_expected_zip])[0]
    donor_expected_zip = resolve_paths(project_root, [args.donor_expected_zip])[0]
    final_expected_zip = resolve_paths(project_root, [args.final_expected_zip])[0]

    required = [
        source_zip,
        base_expected_zip,
        donor_expected_zip,
        final_expected_zip,
        *ref_zips,
    ]
    if args.donor_mode == "artifact":
        required.append(frozen_donor_zip)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required artifacts: {missing}")

    ct_out_root = work_root / "ct_base"
    base_q95_zip = work_root / "dev_glr100_q95s420_repro.zip"
    base_repro_zip = work_root / "dev_glr100_gauss35_q95s420_repro.zip"
    donor_repro_zip = work_root / "dev_glr100_lababg05g35q95_repro.zip"

    ct_cmd = [
        args.python,
        "scripts/research/ct_sidecar_sweep.py",
        "--source",
        str(source_zip),
        "--blend",
        "geo",
        "--method",
        "lab_reinhard",
        "--alpha",
        "1.0",
        "--out-root",
        str(ct_out_root),
        "--save-zips",
    ]
    for ref_zip in ref_zips:
        ct_cmd.extend(["--ref", str(ref_zip)])
    run_cmd(ct_cmd, cwd=project_root)

    base_ct_zip = ct_out_root / "geo_lab_reinhard_a100.zip"
    if not base_ct_zip.exists():
        raise FileNotFoundError(f"CT output missing: {base_ct_zip}")

    run_cmd(
        [
            args.python,
            "scripts/postprocess_submission_zip.py",
            "--input-zip",
            str(base_ct_zip),
            "--output-zip",
            str(base_q95_zip),
            "--gaussian-radius",
            "0.0",
            "--jpeg-quality",
            "95",
            "--jpeg-subsampling",
            "2",
        ],
        cwd=project_root,
    )

    run_cmd(
        [
            args.python,
            "scripts/postprocess_submission_zip.py",
            "--input-zip",
            str(base_ct_zip),
            "--output-zip",
            str(base_repro_zip),
            "--gaussian-radius",
            "0.35",
            "--jpeg-quality",
            "95",
            "--jpeg-subsampling",
            "2",
        ],
        cwd=project_root,
    )

    if args.donor_mode == "artifact":
        donor_repro_zip.write_bytes(frozen_donor_zip.read_bytes())
    else:
        run_cmd(
            [
                args.python,
                "scripts/postprocess_submission_zip.py",
                "--input-zip",
                str(base_q95_zip),
                "--output-zip",
                str(donor_repro_zip),
                "--lab-ab-gaussian-sigma",
                str(args.lab_ab_gaussian_sigma),
                "--gaussian-radius",
                "0.35",
                "--jpeg-quality",
                "95",
                "--jpeg-subsampling",
                "2",
            ],
            cwd=project_root,
        )

    gate_manifest_path = work_root / "gate_manifest.json"
    if args.gate_mode == "manual":
        gate_cmd = [
            args.python,
            "scripts/blend_submission_zips.py",
            "--zip-a",
            str(base_repro_zip),
            "--zip-b",
            str(donor_repro_zip),
            "--weight-b",
            "0.0",
            "--output-zip",
            str(output_zip),
            "--tag",
            "fut3g35_exact_gate_repro",
        ]
        for name in manual_donor_files:
            gate_cmd.extend(["--file-weight-b", f"{name}=1.0"])
        run_cmd(gate_cmd, cwd=project_root)
        gate_manifest = {
            "gate_mode": "manual",
            "manual_donor_files": manual_donor_files,
        }
        gate_manifest_path.write_text(json.dumps(gate_manifest, indent=2), encoding="utf-8")
    else:
        run_cmd(
            [
                args.python,
                "scripts/auto_exact_gate.py",
                "--zip-a",
                str(base_repro_zip),
                "--zip-b",
                str(donor_repro_zip),
                "--output-zip",
                str(output_zip),
                "--selector",
                "lab_chroma_luma_grad_ratio",
                "--threshold",
                str(args.auto_gate_threshold),
                "--json-out",
                str(gate_manifest_path),
            ],
            cwd=project_root,
        )
        gate_manifest = json.loads(gate_manifest_path.read_text(encoding="utf-8"))

    manifest = {
        "source_zip": str(source_zip),
        "ref_zips": [str(path) for path in ref_zips],
        "donor_mode": args.donor_mode,
        "gate_mode": args.gate_mode,
        "manual_donor_files": manual_donor_files,
        "auto_gate_threshold": args.auto_gate_threshold,
        "lab_ab_gaussian_sigma": args.lab_ab_gaussian_sigma,
        "frozen_donor_zip": str(frozen_donor_zip),
        "generated": {
            "base_ct_zip": str(base_ct_zip),
            "base_q95_zip": str(base_q95_zip),
            "base_repro_zip": str(base_repro_zip),
            "donor_repro_zip": str(donor_repro_zip),
            "final_repro_zip": str(output_zip),
            "gate_manifest_json": str(gate_manifest_path),
        },
        "expected": {
            "base_expected_zip": str(base_expected_zip),
            "donor_expected_zip": str(donor_expected_zip),
            "final_expected_zip": str(final_expected_zip),
        },
        "gate_manifest": gate_manifest,
    }

    if not args.skip_compare:
        comparisons = {}
        for tag, left, right in [
            ("base", base_repro_zip, base_expected_zip),
            ("donor", donor_repro_zip, donor_expected_zip),
            ("final", output_zip, final_expected_zip),
        ]:
            identical, diff = zip_bytes_equal(left, right)
            image_psnr = zip_image_psnr(left, right)
            comparisons[tag] = {
                "identical": identical,
                "diff_files": diff,
                "image_psnr": image_psnr,
            }
            print(
                "compare[{tag}] identical={identical} mean_psnr={mean_psnr} "
                "min_psnr={min_psnr} diff_files={diff}".format(
                    tag=tag,
                    identical=identical,
                    mean_psnr=image_psnr["mean_psnr"],
                    min_psnr=image_psnr["min_psnr"],
                    diff=diff,
                )
            )
        manifest["comparisons"] = comparisons

    manifest_path = work_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest_json={manifest_path}")
    print(f"final_zip={output_zip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
