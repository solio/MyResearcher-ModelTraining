"""Adopt the running BF16 service and finish two explicitly bounded queues."""
import argparse,fcntl,hashlib,json,os,signal,subprocess,time,urllib.request
from pathlib import Path
T=Path('/root/qwen38-single-20261010/trials/k3-top5-bf16-1808-v1')
N=T.parent/'k3-same-days-unhit-v1';D=T/'resume-20261013';V=Path('/root/qwen38-single-20261010/venv/bin/python')
STOP=False

def state(path,**data):
 p=path.with_suffix('.chain-tmp');p.write_text(json.dumps({'updated_epoch':time.time(),**data}));p.replace(path)

def alive(pid):
 try:
  return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]!='Z'
 except FileNotFoundError:return False

def command(pid):return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()

def check_bundle(root):
 manifest=json.loads((root/'bundle-manifest.json').read_text())
 for n,h in manifest['files_sha256'].items():
  if hashlib.sha256((root/n).read_bytes()).hexdigest()!=h:raise ValueError('Frozen file changed: '+str(root/n))

def check_completion(root,expected):
 rows=[json.loads(l) for l in (root/'results/main/raw.jsonl').read_text().splitlines()]
 ids=[r['id'] for r in rows]
 if len(ids)!=len(set(ids)):raise ValueError('Duplicate raw receipt IDs')
 runtime=json.loads((root/'results/main/runtime.json').read_text())
 return len(rows)==expected and runtime.get('all_attempted'),runtime

def shutdown_model(pid):
 if alive(pid) and 'vllm' in command(pid) and '/root/model-cache/Qwen3.8-27B-BF16' in command(pid):
  if os.getpgid(pid)!=pid:raise RuntimeError('Unexpected model process group; preserve unrelated processes')
  os.killpg(pid,signal.SIGTERM)
  for _ in range(15):
   if not alive(pid):return
   time.sleep(1)
  if alive(pid):os.killpg(pid,signal.SIGKILL)

def main():
 p=argparse.ArgumentParser();p.add_argument('--deadline-epoch',type=float,required=True);p.add_argument('--old-supervisor',type=int,required=True);p.add_argument('--first-runner',type=int,required=True);p.add_argument('--model-pid',type=int,required=True);a=p.parse_args()
 lock=(D/'chain.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 check_bundle(T);check_bundle(N)
 assert 'resume-20261013/remote_session.py' in command(a.old_supervisor)
 assert str(T/'runner/run_batch.py') in command(a.first_runner)
 assert '/root/model-cache/Qwen3.8-27B-BF16' in command(a.model_pid)
 with urllib.request.urlopen('http://127.0.0.1:8000/v1/models',timeout=5) as r:
  assert any(x['id']=='qwen38-27b-bf16' for x in json.load(r)['data'])
 if a.deadline_epoch-time.time()<300:raise RuntimeError('Not enough authorized time')
 def stopped(*_):
  global STOP;STOP=True
 signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
 state(D/'chain-status.json',state='adopting',model_pid=a.model_pid,first_runner_pid=a.first_runner,deadline_epoch=a.deadline_epoch)
 # Terminate only the old supervisor waiting on its child. Neither process group is signalled.
 # The old supervisor has no SIGTERM handler; model and runner have separate durable processes.
 os.kill(a.old_supervisor,signal.SIGTERM)
 for _ in range(30):
  if not alive(a.old_supervisor):break
  time.sleep(0.1)
 if alive(a.old_supervisor) or not alive(a.model_pid) or not alive(a.first_runner):raise RuntimeError('Takeover state unexpected; inspect without sending more signals')
 session_lock=(T/'session.lock').open('a');fcntl.flock(session_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 (T/'supervisor.pid').write_text(str(os.getpid()))
 state(D/'chain-status.json',state='running_first',model_pid=a.model_pid,first_runner_pid=a.first_runner,deadline_epoch=a.deadline_epoch,old_supervisor_replaced=True)
 state(N/'session-status.json',state='queued_after_top5',expected=2588,reused_prior=6,deadline_epoch=a.deadline_epoch)
 child=None;reason='unknown'
 try:
  while alive(a.first_runner) and time.time()<a.deadline_epoch and not STOP:
   if not alive(a.model_pid):raise RuntimeError('BF16 model service exited')
   time.sleep(2)
  if alive(a.first_runner):
   os.kill(a.first_runner,signal.SIGTERM);reason='deadline_or_signal';return
  complete,rt=check_completion(T,1808)
  state(T/'session-status.json',state='complete' if complete else 'stopped',expected=1808,attempted=rt.get('completed_rows'),format_valid=rt.get('successful_rows'),reason=rt.get('stop_reason'),deadline_epoch=a.deadline_epoch)
  if not complete:raise RuntimeError('First queue stopped with pending inputs; new queue not started')
  if STOP or time.time()>a.deadline_epoch-240:reason='deadline_before_next';return
  check_bundle(N);plan=json.loads((N/'inference-plan.json').read_text());g=plan['generation']
  cmd=[str(V),str(N/'runner/run_batch.py'),'--input',str(N/'data/model-inputs.json'),'--out',str(N/'results/main'),'--prompt',str(N/'runner/prompt.txt'),'--schema',str(N/'runner/meaning_schema.json'),'--base-url','http://127.0.0.1:8000/v1','--model','qwen38-27b-bf16','--model-revision',plan['model']['revision'],'--stage','k3_same_days_unhit','--thinking','off','--reasoning-effort','omit','--temperature',str(g['temperature']),'--top-p',str(g['top_p']),'--max-tokens',str(g['max_tokens']),'--concurrency',str(g['concurrency']),'--smoke-count','1','--timeout',str(g['timeout']),'--deadline-epoch',str(a.deadline_epoch)]
  with (N/'runner.log').open('ab',buffering=0) as log:child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
  state(N/'session-status.json',state='running',expected=2588,reused_prior=6,runner_pid=child.pid,deadline_epoch=a.deadline_epoch)
  state(D/'chain-status.json',state='running_next',model_pid=a.model_pid,next_runner_pid=child.pid,deadline_epoch=a.deadline_epoch)
  while child.poll() is None and time.time()<a.deadline_epoch and not STOP:
   if not alive(a.model_pid):raise RuntimeError('BF16 model service exited in next batch')
   time.sleep(2)
  if child.poll() is None:
   child.terminate()
   try:child.wait(timeout=15)
   except subprocess.TimeoutExpired:child.kill();child.wait()
  done,rt=check_completion(N,2588)
  state(N/'session-status.json',state='complete' if done else 'stopped',expected=2588,reused_prior=6,attempted=rt.get('completed_rows'),format_valid=rt.get('successful_rows'),reason=rt.get('stop_reason'),deadline_epoch=a.deadline_epoch)
  reason='both_complete' if done else 'next_incomplete'
 except Exception as e:
  reason=str(e);state(D/'chain-status.json',state='failed',reason=reason,deadline_epoch=a.deadline_epoch);raise
 finally:
  if child and child.poll() is None:child.terminate()
  shutdown_model(a.model_pid)
  state(D/'chain-status.json',state='complete' if reason=='both_complete' else 'stopped',reason=reason,deadline_epoch=a.deadline_epoch)
if __name__=='__main__':main()
