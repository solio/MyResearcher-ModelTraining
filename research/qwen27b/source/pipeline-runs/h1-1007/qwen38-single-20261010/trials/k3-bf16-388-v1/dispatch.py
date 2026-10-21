"""Upload the blind bundle and launch/backup one explicitly timed GPU session.

Password is read with getpass; never supplied in argv, saved, or logged.
Default is a local readiness check. Use --execute only for an available session.
"""
import argparse
import fcntl
import getpass
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

D=Path(__file__).resolve().parent
REMOTE='/root/qwen38-single-20261010/trials/k3-bf16-388-v1'
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--host');p.add_argument('--port',type=int);p.add_argument('--username',default='root');p.add_argument('--deadline-epoch',type=float);p.add_argument('--key-auth',action='store_true');a=p.parse_args()
 bundle=D/'inference-bundle.tar.gz';m=json.loads((D/'bundle-manifest.json').read_text())
 for name,h in m['files_sha256'].items():
  if hashlib.sha256((D/name).read_bytes()).hexdigest()!=h:raise SystemExit('Frozen file changed: '+name)
 package=json.loads((D/'bundle-check.json').read_text())
 if hashlib.sha256(bundle.read_bytes()).hexdigest()!=package['archive_sha256']:raise SystemExit('Archive hash changed')
 if not a.execute:print(json.dumps({'state':'ready_locally','bundle_bytes':bundle.stat().st_size,'rows':388,'remote_checked':False}));return
 if not a.host or not a.port or not a.deadline_epoch or a.deadline_epoch-time.time()<300:p.error('Execute requires --host, --port and a future --deadline-epoch with enough time remaining')
 lock=(D/'dispatch.lock').open('a')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise SystemExit('A local dispatch/backup loop is already running')
 spec=importlib.util.spec_from_file_location('task_ssh',D.parents[1]/'deployment/control_server.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 target,password=module.endpoint({'host':a.host,'port':a.port,'username':a.username,'password':'' if a.key_auth else getpass.getpass('SSH password (not saved): ')})
 ssh=module.SSH(target);ssh.connect(password);password=None
 base=['ssh','-S',ssh.socket,'-p',str(a.port),'-o','BatchMode=yes','-o','ConnectTimeout=8',ssh.destination]
 def remote(script,data=None,timeout=30):
  result=subprocess.run(base+['python3 -c '+shlex.quote(script)],input=data,capture_output=True,timeout=timeout)
  if result.returncode:raise RuntimeError(result.stderr.decode(errors='replace')[-1500:] or 'Remote command failed')
  return result.stdout
 try:
  unpack="""import sys,tarfile,io,json,hashlib
from pathlib import Path
d=Path(DEST);d.mkdir(parents=True,exist_ok=True)
t=tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz')
names=t.getnames()
if any(n.startswith('/') or '..' in Path(n).parts for n in names):raise ValueError('Invalid archive paths')
m=json.load(t.extractfile('bundle-manifest.json'))
if (d/'bundle-manifest.json').exists() and json.loads((d/'bundle-manifest.json').read_text())!=m:raise ValueError('Remote bundle differs; preserve results and diagnose')
for n,h in m['files_sha256'].items():
 if hashlib.sha256(t.extractfile(n).read()).hexdigest()!=h:raise ValueError('Bad upload hash '+n)
t.extractall(d,filter='data')
print('Bundle verified and extracted')
""".replace('DEST',repr(REMOTE))
  print(remote(unpack,bundle.read_bytes(),timeout=60).decode().strip(),flush=True)
  launch="""import subprocess,json,time,fcntl
from pathlib import Path
d=Path(DEST)
lock=(d/'session.lock').open('a')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:
 print(json.dumps({'already_running':True}));raise SystemExit(0)
fcntl.flock(lock,fcntl.LOCK_UN)
tmp=d/'session-status.dispatch-tmp';tmp.write_text(json.dumps({'state':'queued','updated_epoch':time.time()}));tmp.replace(d/'session-status.json')
with (d/'supervisor.log').open('ab',buffering=0) as f:
 p=subprocess.Popen(['python3',str(d/'remote_session.py'),'--deadline-epoch',DEADLINE],stdout=f,stderr=f,stdin=subprocess.DEVNULL,start_new_session=True)
(d/'supervisor.pid').write_text(str(p.pid))
print(json.dumps({'pid':p.pid,'launched_at':time.time()}))
""".replace('DEST',repr(REMOTE)).replace('DEADLINE',repr(str(a.deadline_epoch)))
  print(remote(launch).decode().strip(),flush=True)
  snapshot="""import json
from pathlib import Path
d=Path(DEST);out={}
names=['session-status.json','preflight.json','supervisor.log','server.log']+['results/main/'+n for n in ['manifest.json','resolved-prompt.txt','raw.jsonl','predictions.jsonl','progress.json','runtime.json']]
for n in names:
 p=d/n
 if not p.exists():continue
 s=p.read_text(errors='replace')
 if n.endswith('.jsonl') and s and not s.endswith('\\n'):s=s[:s.rfind('\\n')+1]
 if n.endswith('.log'):s=s[-60000:]
 out[n]=s
print(json.dumps(out))
""".replace('DEST',repr(REMOTE))
  failures=0
  while time.time()<a.deadline_epoch+60:
   try:
    files=json.loads(remote(snapshot,timeout=25))
    for name,content in files.items():
     path=D/name;path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.sync-tmp');tmp.write_text(content);tmp.replace(path)
    subprocess.run([sys.executable,str(D/'export_results.py')],check=True,stdout=subprocess.DEVNULL)
    state=json.loads(files.get('session-status.json','{}'));progress=json.loads(files.get('results/main/progress.json','{}'))
    print(json.dumps({'state':state.get('state'),'attempted':progress.get('attempted',0),'expected':388,'reason':state.get('reason')},ensure_ascii=False),flush=True)
    failures=0
    if state.get('state') in ('complete','stopped'):break
   except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as e:
    failures+=1;print('Backup retry: '+str(e)[:300],flush=True)
    if failures>=5:raise RuntimeError('Five backup failures; remote receipt files remain. Reconnect before expiry.')
   time.sleep(15)
  # Verify exactly what was backed up at the terminal state; no passwords involved.
  remote_hash=json.loads(remote("import json,hashlib;from pathlib import Path;d=Path("+repr(REMOTE)+");print(json.dumps({str(p.relative_to(d)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (d/'results/main').glob('*.json*') if p.is_file()}))"))
  checks={name:hashlib.sha256((D/name).read_bytes()).hexdigest()==h for name,h in remote_hash.items() if (D/name).exists()}
  (D/'backup-verification.json').write_text(json.dumps({'files':checks,'missing_local':[n for n in remote_hash if not (D/n).exists()],'notice':'Raw/parsed evidence remains on both disks; provider billing is separate.'},indent=2))
 finally:ssh.close()
if __name__=='__main__':main()
