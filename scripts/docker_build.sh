#!/usr/bin/env bash
set -euo pipefail

image_name="${IMAGE_NAME:-molecular-gnn-property-ops:latest}"

docker build --tag "${image_name}" .
