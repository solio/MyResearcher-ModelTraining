#!/bin/sh
# Publish the already-built v4 report into the EXISTING dashboard worktree.
# This script does not start, stop or restart any dashboard process.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
DASH=${1:-/Users/mac/Documents/trae_projects/MyResearcher/worktrees/dashboard-k3-experiments/static}

python3 "$ROOT/v4/build_v4.py" \
  --output "$ROOT/v4/out/v4-data.json" \
  --manifest "$ROOT/v4/out/manifest.json"
python3 "$ROOT/v4/build_v4.py" --check --output "$ROOT/v4/out/v4-data.json"

cp "$ROOT/v4/dashboard/v4.html" "$ROOT/v4/dashboard/v4.css" "$ROOT/v4/dashboard/v4.js" "$DASH/"
cp "$ROOT/v4/out/v4-data.json" "$DASH/v4-data.json"

echo "published to $DASH"
echo "open http://127.0.0.1:8766/static/v4.html (existing process; no new server)"
