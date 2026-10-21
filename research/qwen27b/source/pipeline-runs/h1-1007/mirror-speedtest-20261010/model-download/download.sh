#!/usr/bin/env bash
set -uo pipefail
TASK_DIR=/root/model-cache/qwen38-download
MODEL_DIR=/root/model-cache/Qwen3.8-27B-AWQ-INT4
mkdir -p "$MODEL_DIR"
exec 9>"$TASK_DIR/download.lock"
flock -n 9 || exit 0
echo "$$" > "$TASK_DIR/download.pid"
export PATH="$TASK_DIR/bin:$PATH"
export HF_ENDPOINT=https://hf-mirror.com
attempt=0
while :; do
    attempt=$((attempt+1))
    printf '%s attempt=%s\n' "$(date -Is)" "$attempt" >> "$TASK_DIR/events.log"
    bash "$TASK_DIR/hfd.sh" cyankiwi/Qwen3.8-27B-AWQ-INT4 --revision 6e134bae811fb5adac50ee042ae5f029ac6779aa --tool aria2c -x 8 -j 1 --local-dir "$MODEL_DIR" > "$TASK_DIR/hfd-current.log" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ]; then
        python3 "$TASK_DIR/verify.py" "$MODEL_DIR" > "$TASK_DIR/verification.log" 2>&1
        rc=$?
        printf '%s download_finished verification_exit=%s\n' "$(date -Is)" "$rc" >> "$TASK_DIR/events.log"
        exit "$rc"
    fi
    printf '%s download_incomplete exit=%s retry_in=15s\n' "$(date -Is)" "$rc" >> "$TASK_DIR/events.log"
    sleep 15
done
