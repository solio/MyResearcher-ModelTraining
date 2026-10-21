#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/qwen38-single-20261010
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4
export NCCL_P2P_DISABLE=1
export LD_LIBRARY_PATH="$ROOT/cuda-compat/usr/local/cuda-12.9/compat:${LD_LIBRARY_PATH:-}"
PTXAS="$ROOT/venv/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin/ptxas"
if [[ -x "$PTXAS" ]]; then
  export TRITON_PTXAS_PATH="$PTXAS"
  "$PTXAS" --version
fi
exec "$ROOT/venv/bin/vllm" serve /root/model-cache/Qwen3.8-27B-BF16 \
  --served-model-name qwen38-27b-bf16 \
  --host 127.0.0.1 --port 8000 \
  --load-format safetensors --safetensors-load-strategy lazy \
  --dtype bfloat16 --gpu-memory-utilization 0.85 \
  --max-model-len 8192 --max-num-seqs 8 --max-num-batched-tokens 8192 \
  --tensor-parallel-size 1 --enforce-eager --language-model-only --reasoning-parser qwen3
