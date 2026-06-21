#!/usr/bin/env bash
set -euo pipefail

prediction_paths=(
  artifacts/benchmarks/esol/gnn_comparison/gcn/seed_42/training/predictions.csv
  artifacts/benchmarks/esol/gnn_comparison/gcn/seed_43/training/predictions.csv
  artifacts/benchmarks/esol/gnn_comparison/gcn/seed_44/training/predictions.csv
)

for prediction_path in "${prediction_paths[@]}"; do
  if [[ ! -f "${prediction_path}" ]]; then
    echo "Missing GCN prediction file: ${prediction_path}" >&2
    exit 1
  fi
done

molgnn-ops analyze-gnn-uncertainty \
  artifacts/benchmarks/esol/gnn_uncertainty \
  "${prediction_paths[@]}" \
  --target-coverages 0.80,0.90,0.95
