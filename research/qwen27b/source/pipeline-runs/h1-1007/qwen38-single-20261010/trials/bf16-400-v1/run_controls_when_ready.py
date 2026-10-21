"""Use spare session time for matched old controls only AFTER the400 finish."""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def write(state, **extra):
    temp = ROOT / 'control-status.tmp'
    temp.write_text(json.dumps({'state': state, 'updated_at_epoch': time.time(), **extra}))
    temp.replace(ROOT / 'control-status.json')


def main():
    lock = (ROOT / 'controls.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    session = json.loads((ROOT / 'session.json').read_text())
    deadline = session['deadline_epoch']
    write('waiting_for_400', deadline_epoch=deadline)
    while time.time() < deadline - 900:
        path = ROOT / 'session-status.json'
        status = json.loads(path.read_text()) if path.exists() else {}
        if status.get('state') == 'complete':
            break
        if status.get('state') in ('stopped', 'interrupted'):
            write('skipped', reason='new400_not_complete'); return
        time.sleep(10)
    else:
        write('skipped', reason='less_than15min_remaining'); return
    plan = json.loads((ROOT / 'plan.json').read_text())
    for name, digest in plan['files_sha256'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            write('stopped', reason='frozen_file_changed'); return
    control_plan = json.loads((ROOT / 'optional-control-plan.json').read_text())
    data = ROOT / 'data/control-remaining-72.json'
    if hashlib.sha256(data.read_bytes()).hexdigest() != control_plan['input_sha256']:
        write('stopped', reason='control_input_changed'); return
    write('running', deadline_epoch=deadline)
    command = [sys.executable, str(ROOT / 'runner/run_batch.py'), '--input', str(data),
               '--out', str(ROOT / 'results/controls'), '--prompt', str(ROOT / 'runner/prompt-semantic.txt'),
               '--schema', str(ROOT / 'runner/meaning_schema.json'), '--base-url', 'http://127.0.0.1:8000/v1',
               '--model', 'qwen38-27b-bf16', '--model-revision', plan['model']['revision'],
               '--stage', 'bf16_exploratory', '--thinking', 'off', '--reasoning-effort', 'omit',
               '--temperature', '0', '--top-p', '1', '--max-tokens', '2048', '--concurrency', '8',
               '--smoke-count', '1', '--timeout', '240', '--deadline-epoch', str(deadline)]
    result = subprocess.run(command)
    path = ROOT / 'results/controls/runtime.json'
    runtime = json.loads(path.read_text()) if path.exists() else {}
    write('complete' if runtime.get('all_attempted') else 'stopped',
          exit_code=result.returncode, completed=runtime.get('completed_rows', 0),
          reason=runtime.get('stop_reason'), deadline_epoch=deadline)


if __name__ == '__main__':
    main()
