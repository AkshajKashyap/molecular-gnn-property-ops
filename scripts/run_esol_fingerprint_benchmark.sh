#!/usr/bin/env bash
set -euo pipefail

molgnn-ops run-fingerprint-benchmark esol \
  --split-strategy scaffold \
  --seed 42 \
  --radius 2 \
  --n-bits 2048 \
  --overwrite
