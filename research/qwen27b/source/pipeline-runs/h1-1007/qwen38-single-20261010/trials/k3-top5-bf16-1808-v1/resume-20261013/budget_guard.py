"""Conservative five-hour session guard; lets active requests drain before backup reserve."""
from pathlib import Path
import json,os,signal,time
D=Path(__file__).resolve().parent
B=json.loads((D/'run-budget.json').read_text())
while time.time()<B['no_new_requests_after_epoch']:
 try:
  state=json.loads((D/'chain-status.json').read_text())
  if state.get('state') in ('complete','stopped','failed'):raise SystemExit(0)
 except FileNotFoundError:pass
 time.sleep(min(20,max(0.1,B['no_new_requests_after_epoch']-time.time())))
sent=[]
state=json.loads((D/'chain-status.json').read_text())
if state.get('state') in ('running_first','running_next'):
 pid=state.get('next_runner_pid') if state['state']=='running_next' else state.get('first_runner_pid')
 try:
  cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
  if b'/qwen38-single-20261010/trials/' in cmd and b'/runner/run_batch.py' in cmd:
   os.kill(pid,signal.SIGTERM);sent.append(pid)
 except (FileNotFoundError,ProcessLookupError):pass
(D/'budget-guard-action.json').write_text(json.dumps({'at':time.time(),'stop_new_requests_signal_sent_to':sent,'reserve_policy':B}))
