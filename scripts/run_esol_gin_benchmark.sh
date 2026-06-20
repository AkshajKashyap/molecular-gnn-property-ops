#!/usr/bin/env bash
set -euo pipefail

molgnn-ops run-gnn-benchmark esol artifacts/benchmarks/esol/gnn_gin_seed_42 \
  --split-strategy scaffold \
  --model-name gin \
  --seed 42 \
  --epochs 50 \
  --overwrite
