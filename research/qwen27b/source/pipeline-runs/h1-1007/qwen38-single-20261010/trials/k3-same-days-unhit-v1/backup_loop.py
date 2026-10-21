"""Reconnectable backup for the queued complementary batch; password stays only in memory."""
from pathlib import Path
import fcntl,getpass,gzip,hashlib,importlib.util,io,json,os,subprocess,sys,tarfile,time
D=Path(__file__).resolve().parent;T=D.parent/'k3-top5-bf16-1808-v1';S=json.loads((T/'resume-20261013/session-launch.json').read_text())
def atom(p,b):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.sync-tmp');q.write_bytes(b);q.replace(p)
def main():
 password=getpass.getpass('SSH password for next-batch backup (memory only): ')
 spec=importlib.util.spec_from_file_location('backup_ssh',D.parents[1]/'deployment/control_server.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 conn=mod.SSH({'host':S['host'],'port':S['port'],'username':'root'});conn.socket='/tmp/q38-unhit-backup-20261013.sock';conn.connect(password)
 base=['ssh','-T','-S',conn.socket,'-p',str(S['port']),'-o','BatchMode=yes','-o','ConnectTimeout=8',conn.destination,'python3 /root/qwen38-single-20261010/trials/k3-same-days-unhit-v1/snapshot_stream.py']
 lock=(D/'backup.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(D/'backup.pid').write_text(str(os.getpid()))
 awake=subprocess.Popen(['caffeinate','-is','-w',str(os.getpid())],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);(D/'caffeinate.pid').write_text(str(awake.pid))
 log=(D/'backup.log').open('a',buffering=1);sys.stdout=log;sys.stderr=log
 failures=0
 while time.time()<S['deadline_epoch']+120:
  try:
   r=subprocess.run(base,capture_output=True,timeout=45)
   if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace')[-500:])
   with tarfile.open(fileobj=io.BytesIO(r.stdout),mode='r:gz') as tar:files={m.name:tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
   manifest=json.loads(files.pop('snapshot-manifest.json'))
   for n,h in manifest['files_sha256'].items():assert hashlib.sha256(files[n]).hexdigest()==h,n
   for n,b in files.items():
    if Path(n).is_absolute() or '..' in Path(n).parts:raise ValueError('Invalid path')
    if n.startswith('results/main/') or n in ['session-status.json','runner.log','chain-status.json']:atom(D/n,b)
   raw=files.get('results/main/raw.jsonl',b'');rows=[json.loads(l) for l in raw.splitlines()];assert len(rows)==len({x['id'] for x in rows})
   subprocess.run([sys.executable,str(D/'export_results.py')],stdout=subprocess.DEVNULL,check=True)
   state=json.loads(files.get('session-status.json',b'{}'));summary={'synced_epoch':time.time(),'captured_epoch':manifest['captured_epoch'],'new_rows':len(rows),'new_expected':2588,'reused_prior':6,'complement_received':len(rows)+6,'complement_expected':2594,'state':state.get('state'),'all_download_hashes_verified':True,'raw_sha256':hashlib.sha256(raw).hexdigest(),'consecutive_failures':0}
   atom(D/'backup-status.json',json.dumps(summary).encode());failures=0;print(json.dumps(summary),flush=True)
   if state.get('state') in ('complete','stopped'):
    out=D/'snapshots/20261013-final';out.mkdir(parents=True,exist_ok=True)
    atom(out/'remote-snapshot.tgz',r.stdout);atom(out/'k3-return-with-reuse.jsonl',(D/'exports/k3-return-with-reuse.jsonl').read_bytes());atom(out/'download-complete.json',json.dumps(summary).encode());break
  except Exception as e:
   failures+=1;atom(D/'backup-error.json',json.dumps({'epoch':time.time(),'failures':failures,'error':str(e)[:800]}).encode());print('BACKUP RETRY '+str(e)[:500],flush=True)
   try:conn.close();conn.connect(password)
   except Exception as x:print('Reconnect pending '+str(x)[:200],flush=True)
  time.sleep(20 if not failures else min(60,10*failures))
if __name__=='__main__':main()
