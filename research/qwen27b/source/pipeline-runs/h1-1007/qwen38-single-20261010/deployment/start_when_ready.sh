#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/qwen38-single-20261010
cd "$ROOT"
for ((n=0; n<540; n++)); do
  if [[ -f "$ROOT/runtime-ready.json" ]]; then
    exec bash "$ROOT/serve.sh"
  fi
  sleep 5
done
echo 'Runtime installation did not complete within 45 minutes; inspect saved logs.'
exit 1
