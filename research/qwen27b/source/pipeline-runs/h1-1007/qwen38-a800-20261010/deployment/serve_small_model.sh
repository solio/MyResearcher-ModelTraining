#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/qwen38-a800-20261010
VENV=/root/qwen38-pilot-20261010/venv
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4
exec "$VENV/bin/vllm" serve "$ROOT/models/Qwen3.5-0.8B" \
  --served-model-name qwen35-08b-runtime-smoke --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --gpu-memory-utilization 0.5 --max-model-len 8192 \
  --max-num-seqs 4 --max-num-batched-tokens 4096 --enforce-eager \
  --language-model-only --reasoning-parser qwen3
