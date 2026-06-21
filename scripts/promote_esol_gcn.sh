#!/usr/bin/env bash
set -euo pipefail

run_root="artifacts/benchmarks/esol/fixed_split_gcn/split_seed_42"
registry_dir="artifacts/registry/esol-gcn-v1"
prepared_csv="${run_root}/prepared.csv"
candidate_run_dirs=()

if [[ ! -f "${prepared_csv}" ]]; then
  echo "Missing prepared dataset: ${prepared_csv}" >&2
  exit 1
fi

for model_seed in 42 43 44; do
  candidate_dir="${run_root}/model_seed_${model_seed}"
  if [[ ! -f "${candidate_dir}/metrics.json" ]]; then
    echo "Missing candidate metrics: ${candidate_dir}/metrics.json" >&2
    exit 1
  fi
  if [[ ! -f "${candidate_dir}/models/gnn_regressor.pt" ]]; then
    echo "Missing candidate checkpoint: ${candidate_dir}/models/gnn_regressor.pt" >&2
    exit 1
  fi
  candidate_run_dirs+=("${candidate_dir}")
done

molgnn-ops promote-model \
  "${registry_dir}" \
  "${candidate_run_dirs[@]}" \
  --model-id esol-gcn-v1 \
  --metric rmse \
  --prepared-csv "${prepared_csv}" \
  --include-reference-index
