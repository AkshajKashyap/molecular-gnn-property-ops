#!/usr/bin/env bash
set -euo pipefail

image_name="${IMAGE_NAME:-molecular-gnn-property-ops:latest}"
api_port="${API_PORT:-8000}"
manifest_path="artifacts/registry/esol-gcn-v1/manifest.json"

if [[ ! -f "${manifest_path}" ]]; then
  echo "Missing promoted model manifest: ${manifest_path}" >&2
  echo "Run scripts/promote_esol_gcn.sh first." >&2
  exit 1
fi

docker run --rm \
  --publish "${api_port}:8000" \
  --volume "$(pwd)/artifacts/registry:/app/artifacts/registry:ro" \
  --env MOLGNN_MANIFEST_PATH=/app/artifacts/registry/esol-gcn-v1/manifest.json \
  --env API_HOST=0.0.0.0 \
  --env API_PORT=8000 \
  "${image_name}" \
  molgnn-ops serve-api
