#!/usr/bin/env bash
set -euo pipefail

prepared_csv="artifacts/benchmarks/esol/seed_42/prepared.csv"
predictions_csv="artifacts/benchmarks/esol/seed_42/baseline/predictions.csv"
output_dir="artifacts/benchmarks/esol/seed_42/diagnostics"

for required_file in "$prepared_csv" "$predictions_csv"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Required benchmark file not found: $required_file" >&2
    exit 1
  fi
done

molgnn-ops diagnose-benchmark "$prepared_csv" "$predictions_csv" "$output_dir"
