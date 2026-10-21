import json
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path('/root/qwen38-single-20261010')
deadline = float((ROOT / 'run-deadline.txt').read_text())
while time.time() < deadline - 120:
    deadline = float((ROOT / 'run-deadline.txt').read_text())
    try:
        with urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2) as response:
            if response.status == 200:
                break
    except Exception:
        pass
    time.sleep(5)
else:
    raise SystemExit('No healthy model before the initial run deadline; data and logs preserved.')
print('STARTING_REAL_80_SAMPLE_RUN', time.time(), flush=True)
cmd = ['python3', str(ROOT / 'runner/run_batch.py'),
       '--input', str(ROOT / 'data/stage1-inputs.json'), '--out', str(ROOT / 'results/stage1'),
       '--model', 'qwen38-27b-int4', '--model-revision', '6e134bae811fb5adac50ee042ae5f029ac6779aa',
       '--thinking', 'off', '--reasoning-effort', 'omit', '--max-tokens', '2048',
       '--concurrency', '2', '--smoke-count', '3', '--deadline-epoch', str(deadline)]
rc = subprocess.call(cmd)
print('BATCH_EXIT', rc, flush=True)
if (ROOT / 'results/stage1/manifest.json').exists():
    subprocess.call(['python3', str(ROOT / 'runner/evaluate.py'), '--input', str(ROOT / 'data/stage1-inputs.json'),
                     '--reference', str(ROOT / 'data/stage1-reference.json'), '--run', str(ROOT / 'results/stage1'),
                     '--out', str(ROOT / 'results/evaluation.json')])
print('INITIAL_RUN_FINISHED', flush=True)
