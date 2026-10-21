"""Independent read-only authenticated backup, reconnecting every poll."""
import fcntl,gzip,hashlib,io,json,pathlib,subprocess,sys,tarfile,time,os,getpass,importlib.util
D=pathlib.Path(__file__).resolve().parent;T=D.parent
S=json.loads((T/'session-launch.json').read_text());ACCESS=json.loads((D/'backup-access.json').read_text())
BASE=['ssh','-i',ACCESS['key_path'],'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ConnectTimeout=8','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','-p',str(S['port']),'root@'+S['host']]
def atom(p,b):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.resume-tmp');q.write_bytes(b);q.replace(p)
def js(o):return json.dumps(o,ensure_ascii=False,separators=(',',':')).encode()
def main():
 global BASE
 spec=importlib.util.spec_from_file_location('backup_ssh',T.parents[1]/'deployment/control_server.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 password=getpass.getpass('SSH password for reconnect (memory only): ')
 connection=mod.SSH({'host':S['host'],'port':S['port'],'username':'root'});connection.socket='/tmp/q38-top5-backup-20261013.sock';connection.connect(password)
 BASE=['ssh','-T','-S',connection.socket,'-p',str(S['port']),'-o','BatchMode=yes','-o','ConnectTimeout=8',connection.destination,'python3 /root/qwen38-single-20261010/trials/k3-top5-bf16-1808-v1/resume-20261013/snapshot_stream.py']
 (D/'backup.pid').write_text(str(os.getpid()))
 awake=subprocess.Popen(['caffeinate','-is','-w',str(os.getpid())],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);(D/'caffeinate.pid').write_text(str(awake.pid))
 log=(D/'backup.log').open('a',buffering=1);sys.stdout=log;sys.stderr=log
 lock=(D/'backup.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 source={r['id']:r for r in map(json.loads,(T/'private/evaluation-sidecar.jsonl').read_text().splitlines())}
 failures=0
 while True:
  deadline=json.loads((D/'session-launch.json').read_text())['deadline_epoch']
  if time.time()>deadline+120:break
  try:
   r=subprocess.run(BASE,capture_output=True,timeout=45)
   if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace')[-500:])
   with tarfile.open(fileobj=io.BytesIO(r.stdout),mode='r:gz') as tar:
    files={m.name:tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
   m=json.loads(files.pop('snapshot-manifest.json'))
   for n,h in m['files_sha256'].items():
    assert hashlib.sha256(files[n]).hexdigest()==h,n
   for n,b in files.items():
    if pathlib.PurePosixPath(n).is_absolute() or '..' in pathlib.PurePosixPath(n).parts:raise ValueError('Invalid backup path')
    if n.startswith('results/main/') or n.startswith('cpu-delivery/output/') or n in ['session-status.json','preflight.json','server.log','supervisor.log']:atom(T/n,b)
   raw=files['results/main/raw.jsonl'];rows=[json.loads(l) for l in raw.splitlines()];ids=[x['id'] for x in rows]
   assert len(ids)==len(set(ids)),'Duplicate receipt IDs'
   returned=[]
   for row in rows:
    content=None;output=None
    try:content=row['response']['choices'][0]['message']['content']
    except (KeyError,IndexError,TypeError):pass
    try:
     def unique(pairs):
      out={}
      for k,v in pairs:
       if k in out:raise ValueError('Duplicate model key')
       out[k]=v
      return out
     output=json.loads(content,object_pairs_hook=unique)
    except (ValueError,TypeError):pass
    pred=row.get('prediction') or {}
    returned.append({**source[row['id']],'qwen_model_output':output,'qwen_model_content':content,'original_contract_status':pred.get('status'),'original_contract_error':pred.get('error'),'semantic_quality':'not_assessed','raw_receipt_id':row['id']})
   output=b''.join(js(x)+b'\n' for x in returned);atom(T/'exports/k3-full-raw-return.jsonl',output)
   subprocess.run([sys.executable,str(T/'export_results.py')],check=True,stdout=subprocess.DEVNULL)
   state=json.loads(files.get('session-status.json',b'{}'))
   verification={'synced_epoch':time.time(),'captured_epoch':m['captured_epoch'],'rows':len(rows),'unique_ids':len(set(ids)),'expected':1808,'state':state.get('state'),'all_download_hashes_verified':True,'raw_sha256':hashlib.sha256(raw).hexdigest(),'k3_return_sha256':hashlib.sha256(output).hexdigest(),'consecutive_failures':0,'remote_deadline_epoch':state.get('deadline_epoch')}
   atom(D/'backup-status.json',js(verification));failures=0
   print(json.dumps(verification),flush=True)
   if state.get('state') in ('complete','stopped'):
    name='20261013-complete' if len(rows)==1808 else '20261013-session-stopped'
    dest=T/'snapshots'/name;dest.mkdir(parents=True,exist_ok=True)
    atom(dest/'remote-snapshot.tgz',r.stdout);atom(dest/'k3-raw-return.jsonl',output)
    atom(dest/'missing-ids.json',js([i for i in source if i not in set(ids)]));atom(dest/'download-complete.json',js(verification))
    # The GPU chain may continue with another batch; this original batch is saved.
    break
  except Exception as e:
   failures+=1
   atom(D/'backup-error.json',js({'epoch':time.time(),'consecutive_failures':failures,'error':str(e)[:800]}))
   print('BACKUP RETRY: '+str(e)[:800],flush=True)
   try:connection.connect(password)
   except Exception as reconnect_error:print('Reconnect pending: '+str(reconnect_error)[:200],flush=True)
  time.sleep(20 if not failures else min(60,10*failures))
if __name__=='__main__':main()
