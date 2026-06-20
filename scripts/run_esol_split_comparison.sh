#!/usr/bin/env bash
set -euo pipefail

molgnn-ops compare-splits esol artifacts/benchmarks/esol/split_comparison \
  --seeds 42,43,44 \
  --split-strategies random,scaffold \
  --overwrite
