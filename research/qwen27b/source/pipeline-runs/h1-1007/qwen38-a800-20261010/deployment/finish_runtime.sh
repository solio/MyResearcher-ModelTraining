#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/qwen38-a800-20261010
VENV=/root/qwen38-pilot-20261010/venv
cd "$ROOT"
while kill -0 4887 2>/dev/null; do sleep 10; done
"$VENV/bin/python" -c 'import torch, transformers, triton, compressed_tensors; print(torch.__version__, transformers.__version__, triton.__version__)'
"$VENV/bin/uv" pip install --python "$VENV/bin/python" --no-deps "$ROOT/vllm-0.31.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
"$VENV/bin/uv" pip check --python "$VENV/bin/python"
"$VENV/bin/uv" pip freeze --python "$VENV/bin/python" > "$ROOT/runtime-freeze.txt"
"$VENV/bin/python" "$ROOT/kernel_probe.py" --out "$ROOT/results/kernel-probe-initial.json"
