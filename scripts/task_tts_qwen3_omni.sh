#!/bin/bash
set -euo pipefail

TASK_NAME="task_tts_qwen3_omni"
MODEL_NAME="qwen3_omni"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs/${TASK_NAME}"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

echo "Launching ${TASK_NAME}..."
nohup python -m lmms_eval \
  --model "${MODEL_NAME}" \
  --tasks "${TASK_NAME}" \
  --batch_size 1 \
  --log_samples \
  --output_path "logs/${TASK_NAME}" \
  > "${LOG_DIR}/run.log" 2>&1 &
echo "  PID: $!"
echo "Logs: ${LOG_DIR}/run.log"
