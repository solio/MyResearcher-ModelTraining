#!/usr/bin/env python3
"""Resume the authorized BF16 experiment within this node's fixed session window."""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent


def write(state, **extra):
    tmp = ROOT / 'session-status.tmp'
    tmp.write_text(json.dumps({'state': state, 'updated_at_epoch': time.time(), **extra}, ensure_ascii=False))
    tmp.replace(ROOT / 'session-status.json')


def main():
    lock = (ROOT / 'session.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('Existing session owns the queue.', flush=True)
        return
    session = json.loads((ROOT / 'session.json').read_text())
    deadline = session['deadline_epoch']
    plan = json.loads((ROOT / 'plan.json').read_text())
    for name, expected in plan['files_sha256'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            write('stopped', reason='frozen_file_changed', file=name)
            return
    write('waiting_server', deadline_epoch=deadline)
    while time.time() < deadline - 250:
        try:
            with urllib.request.urlopen('http://127.0.0.1:8000/v1/models', timeout=3) as response:
                models = json.load(response)
            if any(x['id'] == 'qwen38-27b-bf16' for x in models.get('data', [])):
                break
        except Exception:
            pass
        time.sleep(3)
    else:
        write('stopped', reason='server_not_ready_before_deadline', deadline_epoch=deadline)
        return
    for name, data in [('startup', 'startup-check-8.json'), ('natural', 'stage2-inputs.json')]:
        write('running', batch=name, deadline_epoch=deadline)
        cmd = [sys.executable, str(ROOT / 'runner/run_batch.py'),
               '--input', str(ROOT / 'data' / data), '--out', str(ROOT / 'results' / name),
               '--prompt', str(ROOT / 'runner/prompt-semantic.txt'),
               '--schema', str(ROOT / 'runner/meaning_schema.json'),
               '--base-url', 'http://127.0.0.1:8000/v1', '--model', 'qwen38-27b-bf16',
               '--model-revision', plan['model']['revision'], '--stage', 'bf16_exploratory',
               '--thinking', 'off', '--reasoning-effort', 'omit', '--temperature', '0',
               '--top-p', '1', '--max-tokens', '2048', '--concurrency', '8',
               '--smoke-count', '1', '--timeout', '240', '--deadline-epoch', str(deadline)]
        result = subprocess.run(cmd)
        path = ROOT / 'results' / name / 'runtime.json'
        runtime = json.loads(path.read_text()) if path.exists() else {}
        # Output errors remain review items. They never pretend to be a failed semantic gate.
        if not runtime.get('all_attempted'):
            write('stopped', batch=name, reason=runtime.get('stop_reason', 'runner_failed'),
                  exit_code=result.returncode, deadline_epoch=deadline)
            return
    write('complete', batch='natural', completed_natural=400, completed_startup=8,
          deadline_epoch=deadline)


if __name__ == '__main__':
    main()
