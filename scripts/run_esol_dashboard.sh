#!/usr/bin/env bash
set -euo pipefail

manifest_path="artifacts/registry/esol-gcn-v1/manifest.json"
PORT="${PORT:-8501}"

if [[ ! -f "${manifest_path}" ]]; then
  echo "Missing promoted model manifest: ${manifest_path}" >&2
  echo "Run scripts/promote_esol_gcn.sh first." >&2
  exit 1
fi

molgnn-ops run-dashboard \
  "${manifest_path}" \
  --host 0.0.0.0 \
  --port "${PORT}"
