#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_DIR="${ROOT_DIR}/repro_out"

mkdir -p "${REPRO_DIR}"

python "${ROOT_DIR}/scripts/prepare_submission_track2.py" \
  --project-root "${REPRO_DIR}" \
  --include-zip "${ROOT_DIR}/artifacts/dev_frozen_best.zip" \
  --include-zip "${ROOT_DIR}/artifacts/test_ct_g035.zip" \
  --zip-tag final_result_repro

echo
echo "Reproduction complete."
echo "Generated ZIP is under: ${REPRO_DIR}/submissions/"
