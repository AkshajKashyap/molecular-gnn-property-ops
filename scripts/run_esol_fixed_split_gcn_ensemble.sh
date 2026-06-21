#!/usr/bin/env bash
set -euo pipefail

molgnn-ops run-fixed-split-ensemble \
  esol \
  artifacts/benchmarks/esol/fixed_split_gcn \
  --split-strategy scaffold \
  --split-seed 42 \
  --model-seeds 42,43,44 \
  --model-name gcn \
  --epochs 50 \
  --overwrite
