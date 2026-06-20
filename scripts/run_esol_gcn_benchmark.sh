#!/usr/bin/env bash
set -euo pipefail

molgnn-ops run-gnn-benchmark esol artifacts/benchmarks/esol/gnn_gcn_seed_42 \
  --split-strategy scaffold \
  --model-name gcn \
  --seed 42 \
  --epochs 50 \
  --overwrite
