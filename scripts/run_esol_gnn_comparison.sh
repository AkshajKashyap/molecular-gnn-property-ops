#!/usr/bin/env bash
set -euo pipefail

molgnn-ops compare-gnns esol artifacts/benchmarks/esol/gnn_comparison \
  --models gcn,gin \
  --seeds 42,43,44 \
  --split-strategy scaffold \
  --epochs 50 \
  --overwrite
