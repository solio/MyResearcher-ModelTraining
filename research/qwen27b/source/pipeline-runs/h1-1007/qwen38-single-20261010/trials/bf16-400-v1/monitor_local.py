"""Bounded local CPU audit/telemetry companion; no inference requests or model edits."""
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
DEADLINE = 1791783510.1646662


def main():
    last_size = -1
    while time.time() < DEADLINE + 120:
        try:
            with urllib.request.urlopen('http://127.0.0.1:57710/api/status', timeout=5) as response:
                state = json.load(response)
            report = state.get('report') or {}
            trial = report.get('bf16_trial') or {}
            record = {'local_epoch': time.time(), 'remote_checked_epoch': report.get('checked_epoch'),
                      'connection_phase': state.get('phase'), 'gpu': report.get('gpu'), 'trial': trial}
            with (ROOT / 'telemetry.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
        except Exception:
            pass
        path = ROOT / 'results/natural/predictions.jsonl'
        if path.exists() and path.stat().st_size != last_size:
            result = subprocess.run([sys.executable, str(ROOT / 'build_observation_audit.py')])
            if result.returncode == 0:
                last_size = path.stat().st_size
        session_path, control_path = ROOT / 'session-status.json', ROOT / 'control-status.json'
        session = json.loads(session_path.read_text()) if session_path.exists() else {}
        control = json.loads(control_path.read_text()) if control_path.exists() else {}
        if session.get('state') == 'complete' and control.get('state') in ['complete', 'stopped', 'skipped']:
            return
        time.sleep(20)


if __name__ == '__main__':
    main()
