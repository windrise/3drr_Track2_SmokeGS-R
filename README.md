# 3DRR Track 2 Source Code Release

Team: `windrise`
Contact: `xuemingfu@mail.ustc.edu.cn`
Affiliation: `University of Science and Technology of China`

## 1. What This Package Is

This directory is the curated source-code release package prepared for the NTIRE
2026 3DRR Track 2 final email submission.

It is organized around the final valid testing-phase result used by our team:

- sibling result file: `../Final_result.zip`
- best valid testing submission id: `635505`
- score: `PSNR = 15.217401`, `SSIM = 0.665704`

This package contains the exact frozen artifacts needed to rebuild the final ZIP,
plus the core source files, scripts, and configurations relevant to the final
method (SmokeGS-R).

The main implementation in this release lives under `src/smoke3d/`. Any content
under `methods/foundation/` is strictly optional upstream dependency wiring.

All configuration paths inside this release have been normalized to
repo-root-relative form such as `data/...` and `methods/...`. The packaged
configuration loader resolves these relative paths automatically against the
release root, so the scripts do not depend on machine-specific absolute paths.

## 2. Final Method Summary

Our final submission combines two validated components:

1. A frozen best development-phase submission artifact.
2. A testing-side restoration pipeline built around:
   - a sharp clean-only 3DGS source model trained with DCP-refined pseudo-clean supervision
   - four complementary donor models (ensemble-spatial, dual-depth, VGGT-prior, VGGT-ensemble-prior)
   - LAB-space Reinhard color transfer from the 5-render geometric-mean reference
   - a light Gaussian post-filter with `sigma = 0.35`

The final 28-image submission ZIP is obtained by merging the frozen development
artifact with the regenerated testing-scene outputs.

## 3. Package Layout

```
source_code_windrise/
├── README.md                           # This file
├── reproduce_final_result.sh           # One-command rebuild of the final result
├── requirements-minimal.txt            # Python dependencies with version bounds
│
├── artifacts/                          # Frozen intermediate ZIPs for exact reconstruction
│   ├── dev_frozen_best.zip             #   Frozen best development-side artifact
│   └── test_ct_g035.zip               #   Final testing-side artifact (LAB + Gaussian 0.35)
│
├── src/smoke3d/                        # Core project source code
│   ├── __init__.py
│   ├── config.py                       #   Configuration management
│   ├── data.py                         #   Dataset loading and preprocessing
│   ├── features.py                     #   Feature extraction utilities
│   ├── geometry.py                     #   Geometric computation helpers
│   ├── losses.py                       #   Loss functions (L1, SSIM, depth, pointmap, etc.)
│   ├── model.py                        #   3DGS model definition
│   ├── proxy.py                        #   Proxy mesh utilities
│   ├── runtime.py                      #   Runtime environment helpers
│   └── trainer.py                      #   Training loop and optimization
│
├── scripts/                            # Training, rendering, packaging scripts
│   ├── train_smoke3d.py                #   Model training entry point
│   ├── render_smoke3d.py               #   Novel-view rendering
│   ├── dehaze_training_images.py       #   Optional DehazeFormer preprocessing helper
│   ├── color_transfer.py               #   LAB-space Reinhard color transfer
│   ├── prepare_submission_track2.py    #   Submission ZIP packaging
│   ├── validate_submission_zip.py      #   Submission ZIP validation
│   ├── blend_submission_zips.py        #   Multi-ZIP merging
│   ├── postprocess_submission_zip.py   #   Gaussian smoothing post-processing
│   ├── build_dev_champion_exact_gate.py#   Development champion reconstruction
│   ├── auto_exact_gate.py              #   Automated gating helper
│   ├── generate_test_research_configs.py#  Test-scene config generator
│   ├── query_codabench_submissions.py  #   Codabench API query utility
│   └── research/                       #   Testing-phase ablation scripts
│       ├── build_test_optimized.py
│       ├── test_phase_ct_ablation.py
│       └── test_phase_fusion_ablation.py
│
├── methods/foundation/DehazeFormer/    # Placeholder only; no bundled third-party source
│
└── configs/                            # YAML configuration files
    ├── research/                       #   Curated development/validation-side configs
    └── test_phase_research/            #   Curated test-scene configs (15 files)
```

## 4. Exact Reconstruction Of Final_result.zip

The exact final result can be rebuilt from the two frozen artifacts already
included in this package:

1. `artifacts/dev_frozen_best.zip`
2. `artifacts/test_ct_g035.zip`

Run:

```bash
bash reproduce_final_result.sh
```

This will generate a merged ZIP under `repro_out/submissions/`.

To validate the generated ZIP:

```bash
python scripts/validate_submission_zip.py \
  --zip repro_out/submissions/<generated_zip>.zip \
  --expected-scene futaba \
  --expected-scene hinoki \
  --expected-scene koharu \
  --expected-scene midori \
  --expected-scene natsume \
  --expected-scene shirohana \
  --expected-scene tsubaki \
  --expected-total 28 \
  --expected-per-scene 4
```

## 5. Configuration Files

Only the curated final-method configurations are included:

- **Source branch:** `cleanonly_dcprefinedr61_g050_5000` — clean-only 3DGS trained for 5,000 iterations on DCP-refined pseudo-clean targets with gamma 0.5
- **Donor branches:**
  - `dd_seed42_1000` — dual-depth regularized
  - `ensemble_spatial_1000` — ensemble-spatial fusion
  - `vggt_prior_spatial_1000` — VGGT pointmap prior
  - `vggt_ens_vggt_prior_spatial_1000` — combined VGGT + ensemble prior

In this curated release:

- `configs/research/` contains `23` development/validation-side configs covering the retained final-method variants
- `configs/test_phase_research/` contains `15` test-side configs (`5 variants x 3 test scenes`)

All YAML paths are repo-root-relative and are resolved automatically by
`src/smoke3d/config.py`.

## 6. Environment Notes

For the lightweight packaging and validation path, the most important Python
dependencies are listed in `requirements-minimal.txt`.

For full training and rendering, the original project environment also requires
GPU-capable PyTorch and `gsplat`. The exact versions used for the final
submission are:

- Python 3.10+
- PyTorch >= 2.5 (CUDA 11.8)
- gsplat >= 1.5
- numpy >= 2.0

The release package reserves `methods/foundation/DehazeFormer` as a placeholder
path for the optional auxiliary script `scripts/dehaze_training_images.py`, but
it does not bundle the third-party DehazeFormer source tree itself. This keeps
the release package lightweight and avoids redistributing external code copies
inside our competition release.

Original DehazeFormer repository:

- GitHub: <https://github.com/IDKiro/DehazeFormer>

If you want to use that script, replace the placeholder directory with a
recursive clone of the upstream repository:

```bash
rm -rf methods/foundation/DehazeFormer
git clone --recursive https://github.com/IDKiro/DehazeFormer \
  methods/foundation/DehazeFormer
```

Then follow the upstream README for the official checkpoint download links
(Google Drive / Baidu Pan) and place the downloaded weights under
`methods/foundation/DehazeFormer/saved_models/indoor/`, or provide the path
explicitly via `--weights`.


## 7. Optional Checkpoint Release

Large 3DGS training checkpoints are not bundled in this source release because
the final result can be exactly reconstructed from the frozen intermediate
artifacts. If the organizers require checkpoint-level reproducibility, we can
provide the relevant `.pt` files upon request.

## 8. Acknowledgements

We gratefully acknowledge the NTIRE 2026 3D Restoration and Reconstruction
Challenge organizers for releasing the benchmark and evaluation platform.

We also thank the authors and maintainers of the following open-source projects
that supported this work:

- 3DRR official baseline codebase:
  <https://github.com/I2WM/3DRR_codebase>
- GraphDeco 3D Gaussian Splatting:
  <https://github.com/graphdeco-inria/gaussian-splatting>
- gsplat:
  <https://github.com/nerfstudio-project/gsplat>
- VGGT:
  <https://github.com/facebookresearch/vggt>
- DehazeFormer:
  <https://github.com/IDKiro/DehazeFormer>

This release package contains our competition-specific code and configuration
glue. The above projects remain the property of their respective authors and are
subject to their original licenses.
