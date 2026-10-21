"""Take over service lifetime, keep current inference intact, then run paired A/B."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

D=Path(__file__).resolve().parent
K=D.parent/'k3-bf16-388-v1'
def write(path,obj):
 p=path.with_suffix('.chain-tmp');p.write_text(json.dumps(obj,ensure_ascii=False));p.replace(path)
def state(s,**kw):write(D/'session-status.json',{'state':s,'updated_epoch':time.time(),**kw})
def cmdline(pid):
 try:return (Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode()
 except OSError:return ''
def main():
 p=argparse.ArgumentParser();p.add_argument('--deadline-epoch',type=float,required=True);p.add_argument('--previous-supervisor',type=int,required=True);p.add_argument('--server-pid',type=int,required=True);a=p.parse_args()
 lock=(D/'chain.lock').open('a')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:return
 if a.deadline_epoch<=time.time():raise SystemExit('Deadline elapsed')
 expected_server='/root/model-cache/Qwen3.8-27B-BF16'
 if expected_server not in cmdline(a.server_pid):raise SystemExit('Cannot identify existing BF16 service')
 state('waiting_k3',deadline_epoch=a.deadline_epoch,server_pid=a.server_pid)
 # Transfer lifetime management only; do not signal the active inference runner
 # or the independently-sessioned model server. SIGTERM to Python's default
 # handler terminates this old supervisor without executing its finally cleanup.
 if str(K/'remote_session.py') in cmdline(a.previous_supervisor):
  os.kill(a.previous_supervisor,signal.SIGTERM)
 write(D/'handoff.json',{'at':time.time(),'previous_supervisor':a.previous_supervisor,
       'server_pid':a.server_pid,'action':'Replaced session supervisor only; K3 runner/model left running',
       'user_authorization':'Continue previously prepared80-sample paired experiment after K3 finishes within existing session.'})
 started=False
 try:
  while time.time()<a.deadline_epoch-245:
   f=K/'results/main/runtime.json'
   if f.exists():
    k=json.loads(f.read_text())
    if not k.get('all_attempted'):
     state('stopped',reason='K3 interrupted; diagnose before A/B',deadline_epoch=a.deadline_epoch)
     write(K/'session-status.json',{'state':'stopped','reason':k.get('stop_reason'),'updated_epoch':time.time()});return
    write(K/'session-status.json',{'state':'complete','attempted':k['completed_rows'],'format_valid':k['successful_rows'],'updated_epoch':time.time(),'continuation':'observation-pilot-v1','deadline_epoch':a.deadline_epoch})
    if not (D/'queue-ready.json').exists():
     state('waiting_local_package',deadline_epoch=a.deadline_epoch);time.sleep(2);continue
    ready=json.loads((D/'queue-ready.json').read_text())
    for name,h in ready['files_sha256'].items():
     if hashlib.sha256((D/name).read_bytes()).hexdigest()!=h:raise ValueError('Queued file hash changed: '+name)
    state('running',expected_requests=160,expected_samples=80,deadline_epoch=a.deadline_epoch,server_pid=a.server_pid)
    started=True
    with (D/'runner.log').open('ab',buffering=0) as log:
     result=subprocess.run(['/root/qwen38-single-20261010/venv/bin/python',str(D/'run_ab.py'),'--deadline-epoch',str(a.deadline_epoch)],stdout=log,stderr=log)
    runtime=json.loads((D/'results/paired/runtime.json').read_text()) if (D/'results/paired/runtime.json').exists() else {}
    state('complete' if runtime.get('all_attempted') else 'stopped',deadline_epoch=a.deadline_epoch,attempted_requests=runtime.get('attempted_requests',0),paired_samples=runtime.get('paired_samples',0),exit_code=result.returncode,reason=runtime.get('stop_reason','runner_failed'))
    return
   time.sleep(2)
  state('stopped',reason='deadline_before_next_stage',deadline_epoch=a.deadline_epoch)
 except Exception as e:state('stopped',reason=str(e),deadline_epoch=a.deadline_epoch);raise
 finally:
  # At/near the deadline do not kill a K3 request that the old runner is still
  # draining within its timeout. Wait only until the fixed deadline.
  if not started:
   while time.time()<a.deadline_epoch and not (K/'results/main/runtime.json').exists():time.sleep(1)
  if expected_server in cmdline(a.server_pid):
   os.killpg(a.server_pid,signal.SIGTERM)
if __name__=='__main__':main()
