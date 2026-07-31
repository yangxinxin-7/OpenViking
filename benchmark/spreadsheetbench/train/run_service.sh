#!/usr/bin/env bash
set -euo pipefail

# Start the SpreadsheetBench rollout HTTP service (dataset service for the
# generic OpenViking session/train batch pipeline).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSB_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SSB_DIR}/../.." && pwd)"

load_user_env_file() {
  local env_file="${OPENVIKING_ENV_FILE:-${HOME}/.openviking_benchmark_env}"
  if [[ -z "${env_file}" || ! -f "${env_file}" ]]; then
    return
  fi

  local -a preserved_env=()
  local entry
  while IFS= read -r -d '' entry; do
    if [[ "${entry}" != *= ]]; then
      preserved_env+=("${entry}")
    fi
  done < <(env -0)

  echo "[ssb-service] loading env file ${env_file}"
  set +u
  set -a
  # shellcheck source=/dev/null
  source "${env_file}"
  set +a
  set -euo pipefail

  for entry in "${preserved_env[@]}"; do
    export "${entry}"
  done
}

load_user_env_file

PYTHON_BIN="${PYTHON_BIN:-python}"
HOST="127.0.0.1"
PORT="1954"
DATA_ROOT="${SSB_DATA_ROOT:-${SSB_DIR}/upstream/data}"
CONFIG="${OPENVIKING_CONFIG_FILE:-${HOME}/.openviking/ov.conf}"
KILL_EXISTING=1
MAX_ROLLOUT_CONCURRENCY="${SSB_MAX_ROLLOUT_CONCURRENCY:-100}"
ROLLOUT_THREAD_WORKERS="${SSB_ROLLOUT_THREAD_WORKERS:-${MAX_ROLLOUT_CONCURRENCY}}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --max-rollout-concurrency) MAX_ROLLOUT_CONCURRENCY="$2"; shift 2 ;;
    --rollout-thread-workers) ROLLOUT_THREAD_WORKERS="$2"; shift 2 ;;
    --no-kill-existing) KILL_EXISTING=0; shift 1 ;;
    -h|--help)
      cat <<'EOF'
Usage:
  bash benchmark/spreadsheetbench/train/run_service.sh [--host 127.0.0.1] [--port 1954]

Options:
  --data-root PATH   Directory containing <domain>/dataset.json (default:
                     benchmark/spreadsheetbench/upstream/data or SSB_DATA_ROOT)
  --config PATH      ov.conf for VikingBot/OpenViking access
  --max-rollout-concurrency N
  --rollout-thread-workers N   (0 disables threaded hosting)
  --no-kill-existing Do not stop existing process listening on --port
EOF
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "[ssb-service] data root not found: ${DATA_ROOT}" >&2
  exit 1
fi

VIKINGBOT_ROOT="${VIKINGBOT_ROOT:-${REPO_ROOT}/bot}"
export PYTHONPATH="${REPO_ROOT}:${VIKINGBOT_ROOT}:${PYTHONPATH:-}"
export SSB_DATA_ROOT="${DATA_ROOT}"
export OPENVIKING_CONFIG_FILE="${CONFIG}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-${ARK_API_KEY:-}}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://ark.cn-beijing.volces.com/api/v3}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-${OPENAI_API_BASE}}"

cd "${REPO_ROOT}"
echo "[ssb-service] host=${HOST} port=${PORT} data_root=${DATA_ROOT} config=${CONFIG} max_rollout_concurrency=${MAX_ROLLOUT_CONCURRENCY} rollout_thread_workers=${ROLLOUT_THREAD_WORKERS}"
if [[ "${KILL_EXISTING}" == "1" ]]; then
  EXISTING_PIDS="$(lsof -tiTCP:"${PORT}" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "${EXISTING_PIDS}" ]]; then
    echo "[ssb-service] stopping existing listener(s) on port ${PORT}: ${EXISTING_PIDS}"
    kill ${EXISTING_PIDS} 2>/dev/null || true
    for _ in {1..20}; do
      sleep 0.2
      if ! lsof -tiTCP:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
        break
      fi
    done
    REMAINING_PIDS="$(lsof -tiTCP:"${PORT}" -sTCP:LISTEN 2>/dev/null || true)"
    if [[ -n "${REMAINING_PIDS}" ]]; then
      echo "[ssb-service] force stopping listener(s) on port ${PORT}: ${REMAINING_PIDS}"
      kill -9 ${REMAINING_PIDS} 2>/dev/null || true
    fi
  fi
fi
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/service_app.py" \
  --host "${HOST}" \
  --port "${PORT}" \
  --data-root "${DATA_ROOT}" \
  --config "${CONFIG}" \
  --max-rollout-concurrency "${MAX_ROLLOUT_CONCURRENCY}" \
  --rollout-thread-workers "${ROLLOUT_THREAD_WORKERS}"
