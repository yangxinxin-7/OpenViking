#!/usr/bin/env bash
# Run airline no-memory eval: 20 tasks × 8 repeats = 160 jobs, global concurrency 40.
# Usage:
#   bash run_airline_no_memory.sh [--result-dir DIR] [--concurrency N] [--repeats N] [--config PATH]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNNER="${SCRIPT_DIR}/vikingbot_tau2_runner.py"

DOMAIN="airline"
SPLIT="test"
REPEATS=8
RESULT_DIR="${REPO_ROOT}/result"
CONCURRENCY=40
CONFIG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --result-dir)  RESULT_DIR="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --repeats)     REPEATS="$2"; shift 2 ;;
    --config)      CONFIG="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: bash run_airline_no_memory.sh [--result-dir DIR] [--concurrency N] [--repeats N] [--config PATH]"
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

TAU2_DATA_ROOT="${TAU2_DATA_ROOT:-${REPO_ROOT}/tau2-bench/data/tau2}"
SPLIT_TASKS_JSON="${TAU2_DATA_ROOT}/domains/${DOMAIN}/split_tasks.json"

if [[ ! -f "${SPLIT_TASKS_JSON}" ]]; then
  echo "Split file not found: ${SPLIT_TASKS_JSON}" >&2
  exit 1
fi

TASK_COUNT=$(python3 -c "
import json
with open('${SPLIT_TASKS_JSON}') as f:
    d = json.load(f)
print(len(d['${SPLIT}']))
")

OUTPUT_ROOT="${RESULT_DIR}/${DOMAIN}_${SPLIT}"
mkdir -p "${OUTPUT_ROOT}"

TOTAL=$((TASK_COUNT * REPEATS))
echo "Domain: ${DOMAIN}_${SPLIT}  tasks=${TASK_COUNT}  repeats=${REPEATS}  total=${TOTAL}  concurrency=${CONCURRENCY}"
echo "Output: ${OUTPUT_ROOT}"

CONFIG_FLAG=""
[[ -n "${CONFIG}" ]] && CONFIG_FLAG="--config ${CONFIG}"

# Global semaphore via temp dir
SEM_DIR=$(mktemp -d)
trap 'rm -rf "${SEM_DIR}"' EXIT

active_count() { ls "${SEM_DIR}" 2>/dev/null | wc -l | tr -d ' '; }

LAUNCHED=0
for ((repeat=0; repeat < REPEATS; repeat++)); do
  for ((task_no=0; task_no < TASK_COUNT; task_no++)); do
    out_path="${OUTPUT_ROOT}/task_${task_no}_r${repeat}_trajectory.json"

    # Wait for a free slot
    while [[ $(active_count) -ge ${CONCURRENCY} ]]; do sleep 0.3; done

    SLOT="${SEM_DIR}/${repeat}_${task_no}"
    touch "${SLOT}"
    LAUNCHED=$((LAUNCHED + 1))
    echo "[${LAUNCHED}/${TOTAL}] task_no=${task_no} repeat=${repeat}"
    (
      python3 "${RUNNER}" \
        --data-split "${DOMAIN}_${SPLIT}" \
        --task-no "${task_no}" \
        --output "${out_path}" \
        ${CONFIG_FLAG} \
        --continue \
        2>&1 | tail -3
      rm -f "${SLOT}"
    ) &
  done
done

wait
rm -rf "${SEM_DIR}"

echo ""
echo "===== Results ====="

python3 - "${OUTPUT_ROOT}" "${REPEATS}" "${TASK_COUNT}" <<'PY'
import glob, json, sys

output_root = sys.argv[1]
repeats     = int(sys.argv[2])
task_count  = int(sys.argv[3])

all_rewards = []
missing     = []

for repeat in range(repeats):
    for task_no in range(task_count):
        path = f"{output_root}/task_{task_no}_r{repeat}_trajectory.json"
        try:
            with open(path) as f:
                d = json.load(f)
            r = d.get("reward")
            if r is None:
                missing.append(path)
            else:
                all_rewards.append((task_no, repeat, float(r)))
        except Exception:
            missing.append(path)

total   = len(all_rewards)
passed  = sum(1 for _, _, r in all_rewards if r >= 1.0)
avg     = sum(r for _, _, r in all_rewards) / total if total else 0.0

print(f"Output dir  : {output_root}")
print(f"Total jobs  : {repeats * task_count}  (tasks={task_count} × repeats={repeats})")
print(f"Completed   : {total}")
print(f"Missing     : {len(missing)}")
print(f"Pass (r=1.0): {passed}")
print(f"Accuracy    : {passed}/{total} = {passed/total*100:.1f}%" if total else "Accuracy: N/A")
print(f"Avg reward  : {avg:.4f}")

if missing:
    print("\nMissing files:")
    for p in missing:
        print(f"  {p}")

# Per-task breakdown
print("\nPer-task accuracy (across repeats):")
from collections import defaultdict
by_task = defaultdict(list)
for task_no, repeat, r in all_rewards:
    by_task[task_no].append(r)
for task_no in sorted(by_task):
    rs = by_task[task_no]
    p  = sum(1 for r in rs if r >= 1.0)
    print(f"  task {task_no:>2}: {p}/{len(rs)}  avg={sum(rs)/len(rs):.2f}")
PY
