#!/usr/bin/env bash
set -euo pipefail

# SpreadsheetBench convenience launcher for the generic OpenViking session/train
# batch pipeline. Start the SpreadsheetBench runtime service first:
#   bash benchmark/spreadsheetbench/train/run_service.sh --host 127.0.0.1 --port 1954

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSB_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SSB_DIR}/../.." && pwd)"

exec "${REPO_ROOT}/openviking/session/train/run_batch_train_eval.sh" \
  --dataset spreadsheetbench \
  --domain "${SSB_DOMAIN:-all_data_912_v0.1}" \
  --eval-each-epoch \
  --concurrency 100 \
  --commit-concurrency 100 \
  --benchmark-service-url "${BENCHMARK_SERVICE_URL:-http://127.0.0.1:1954}" \
  "$@"
