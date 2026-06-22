#!/usr/bin/env bash
set -euo pipefail

image_name="${IMAGE_NAME:-molecular-gnn-property-ops:latest}"
api_port="${SMOKE_API_PORT:-18000}"
container_name="molgnn-api-smoke-$$"
manifest_path="artifacts/registry/esol-gcn-v1/manifest.json"

cleanup() {
  docker rm --force "${container_name}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker run --rm "${image_name}" python -c \
  "import fastapi, rdkit, streamlit, torch, torch_geometric; import molgnn_ops.dashboard; from molgnn_ops.api import create_app"

if [[ ! -f "${manifest_path}" ]]; then
  echo "Model-free import smoke passed; promoted manifest is absent, skipping model smoke."
  exit 0
fi

docker run --detach \
  --name "${container_name}" \
  --publish "${api_port}:8000" \
  --volume "$(pwd)/artifacts/registry:/app/artifacts/registry:ro" \
  --env MOLGNN_MANIFEST_PATH=/app/artifacts/registry/esol-gcn-v1/manifest.json \
  --env API_HOST=0.0.0.0 \
  --env API_PORT=8000 \
  "${image_name}" \
  molgnn-ops serve-api >/dev/null

for _ in $(seq 1 30); do
  if curl --fail --silent "http://localhost:${api_port}/health" >/dev/null; then
    break
  fi
  sleep 1
done

curl --fail --silent "http://localhost:${api_port}/health" >/dev/null
curl --fail --silent "http://localhost:${api_port}/model" >/dev/null
curl --fail --silent \
  --header "Content-Type: application/json" \
  --data '{"smiles":"CCO"}' \
  "http://localhost:${api_port}/predict" >/dev/null

echo "Docker API smoke passed with the promoted model."
