#!/usr/bin/env bash
set -euo pipefail

MANIFEST_PATH="${MOLGNN_MANIFEST_PATH:-artifacts/registry/esol-gcn-v1/manifest.json}"
OUTPUT_DIR="${DEMO_OUTPUT_DIR:-artifacts/demo}"

if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "Promoted model manifest not found: ${MANIFEST_PATH}" >&2
  echo "Run: bash scripts/promote_esol_gcn.sh" >&2
  exit 1
fi

if command -v molgnn-ops >/dev/null 2>&1; then
  CLI="molgnn-ops"
elif [[ -x ".venv/bin/molgnn-ops" ]]; then
  CLI=".venv/bin/molgnn-ops"
else
  echo "Could not find molgnn-ops. Activate the project environment or run make install." >&2
  exit 1
fi

"${CLI}" generate-demo "${MANIFEST_PATH}" "${OUTPUT_DIR}" --top-k 3
