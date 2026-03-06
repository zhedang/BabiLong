#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/run_babilong_qwen_no_think_check.py"
LOG_DIR="${PROJECT_ROOT}/experiments/qwen3.5-27b/logs"
mkdir -p "${LOG_DIR}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-8}"
RESUME="${RESUME:-0}"
LOG_FILE="${LOG_DIR}/run_babilong_qwen35_27b_inhibit_${RUN_ID}.log"

export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-1}"

echo "RUN_ID=${RUN_ID}"
echo "LOG_FILE=${LOG_FILE}"
echo "CHECKPOINT_EVERY=${CHECKPOINT_EVERY}"
echo "RESUME=${RESUME}"
echo "OLLAMA_NUM_PARALLEL=${OLLAMA_NUM_PARALLEL}"
echo "OLLAMA_MAX_LOADED_MODELS=${OLLAMA_MAX_LOADED_MODELS}"

cmd=(
  python "${PY_SCRIPT}"
  --run-id "${RUN_ID}"
  --checkpoint-every "${CHECKPOINT_EVERY}"
)
if [[ "${RESUME}" == "1" ]]; then
  cmd+=(--resume)
fi
cmd+=("$@")

systemd-inhibit \
  --what=sleep:idle:handle-lid-switch \
  --who="babilong-qwen3.5-27b" \
  --why="long qwen3.5:27b benchmark run" \
  bash -lc "$(printf '%q ' "${cmd[@]}")" | tee "${LOG_FILE}"
