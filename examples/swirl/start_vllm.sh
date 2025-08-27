#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${1:? "need MODEL_PATH"}
PORTS_CSV=${2:? "need PORTS_CSV like 10030,10031"}
LOG_DIR=${3:-log}

mkdir -p "${LOG_DIR}"

IFS=',' read -r -a PORTS <<< "${PORTS_CSV}"
GPU_MAP=($(seq 0 $(( ${#PORTS[@]}-1 ))))

echo $GPU_MAP

for i in "${!PORTS[@]}"; do
  port=${PORTS[$i]}
  gpu_id=${GPU_MAP[$i]}
  log_file="${LOG_DIR}/vllm_${port}.log"

  echo "[start_vllm] GPU ${gpu_id} -> port ${port}, log=${log_file}"

  (
    export CUDA_VISIBLE_DEVICES=${gpu_id}
    export VLLM_DISABLE_COMPILE_CACHE=1
    exec vllm serve "${MODEL_PATH}" \
      --host 0.0.0.0 \
      --port "${port}" \
      --dtype bfloat16 \
      --limit-mm-per-prompt image=1,video=1 \
      --mm-processor-kwargs '{"max_pixels":1605632}' \
      > "${log_file}" 2>&1
  ) &

  echo "launch server at ${port}"
  sleep 1
done

echo "[start_vllm] started."

wait