#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <wait_pid> <start_seed> [log_path]" >&2
  exit 2
fi

WAIT_PID="$1"
START_SEED="$2"
LOG_PATH="${3:-DPC/outputs/benchmark_suite/ablation_seed_queue.log}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"
mkdir -p "$(dirname "${LOG_PATH}")"

log() {
  printf '[%s] %s\n' "$(date -Iseconds)" "$*" >> "${LOG_PATH}"
}

log "queue started; waiting for pid ${WAIT_PID}"
while kill -0 "${WAIT_PID}" 2>/dev/null; do
  sleep 30
done

seed="${START_SEED}"
while true; do
  log "starting seed ${seed}"
  if PYTHONPATH=. python -m DPC.experiments.generate_ablation_commands --seeds "${seed}" | bash >> "${LOG_PATH}" 2>&1; then
    log "completed seed ${seed}"
    seed="$((seed + 1))"
  else
    status=$?
    log "failed seed ${seed} status=${status}"
    exit "${status}"
  fi
done
