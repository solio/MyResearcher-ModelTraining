"""Wait for the existing local model service and run the authorized 80-row trial."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path('/root/qwen38-single-20261010/trials/complete-extraction-v1')


def write(state, **extra):
    path = ROOT / 'session-status.json'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps({'state': state, 'updated_at_epoch': time.time(), **extra}, ensure_ascii=False))
    tmp.replace(path)


def main():
    lock = (ROOT / 'session.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('Session already running', flush=True)
        return
    plan = json.loads((ROOT / 'session-20261012.json').read_text())
    deadline = plan['deadline_epoch']
    write('waiting_server', deadline_epoch=deadline)
    while time.time() < deadline - 150:
        try:
            with urllib.request.urlopen('http://127.0.0.1:8000/v1/models', timeout=3) as response:
                models = json.load(response)
            if any(x['id'] == 'qwen38-27b-int4' for x in models['data']):
                break
        except Exception:
            pass
        time.sleep(3)
    else:
        write('stopped', reason='model_not_ready_before_deadline', deadline_epoch=deadline)
        return
    write('running', started_at_epoch=time.time(), deadline_epoch=deadline)
    cmd = [sys.executable, str(ROOT / 'run_comparison.py'), '--arm', 'semantic',
           '--base-url', 'http://127.0.0.1:8000/v1', '--model', 'qwen38-27b-int4',
           '--model-revision', '6e134bae811fb5adac50ee042ae5f029ac6779aa',
           '--deadline-epoch', str(deadline)]
    result = subprocess.run(cmd)
    runtime_path = ROOT / 'results/semantic/runtime.json'
    runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
    state = 'complete' if runtime.get('all_attempted') else 'stopped'
    write(state, exit_code=result.returncode, reason=runtime.get('stop_reason', 'runner_start_failed'),
          deadline_epoch=deadline, completed_rows=runtime.get('completed_rows', 0))


if __name__ == '__main__':
    main()
