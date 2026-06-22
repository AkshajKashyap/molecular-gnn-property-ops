#!/usr/bin/env bash
set -euo pipefail

image_name="${IMAGE_NAME:-molecular-gnn-property-ops:latest}"
dashboard_port="${DASHBOARD_PORT:-8501}"
manifest_path="artifacts/registry/esol-gcn-v1/manifest.json"

if [[ ! -f "${manifest_path}" ]]; then
  echo "Missing promoted model manifest: ${manifest_path}" >&2
  echo "Run scripts/promote_esol_gcn.sh first." >&2
  exit 1
fi

docker run --rm \
  --publish "${dashboard_port}:8501" \
  --volume "$(pwd)/artifacts/registry:/app/artifacts/registry:ro" \
  --env MOLGNN_MANIFEST_PATH=/app/artifacts/registry/esol-gcn-v1/manifest.json \
  --env DASHBOARD_HOST=0.0.0.0 \
  --env DASHBOARD_PORT=8501 \
  "${image_name}" \
  molgnn-ops run-dashboard
