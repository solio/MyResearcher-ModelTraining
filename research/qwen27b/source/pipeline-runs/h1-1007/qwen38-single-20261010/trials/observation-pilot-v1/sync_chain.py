"""Independent backup channel survives original K3 dispatch finishing."""
from pathlib import Path
import hashlib,json,shlex,subprocess,sys,time
D=Path(__file__).resolve().parent
DEADLINE=1791795380.876541
BASE=['ssh','-S','/tmp/q38-chain-20261012.sock','-p','18312','-o','BatchMode=yes','-o','ConnectTimeout=8','root@223.109.239.30']
SCRIPT="""from pathlib import Path
import json,hashlib
p=Path('/root/qwen38-single-20261010/trials');out={};hashes={}
groups={'observation-pilot-v1':['session-status.json','handoff.json','chain.log','runner.log']+['results/paired/'+n for n in ['manifest.json','raw.jsonl','predictions.jsonl','progress.json','runtime.json']], 'k3-bf16-388-v1':['session-status.json']}
k=p/'k3-bf16-388-v1/results/main/runtime.json'
if k.exists():groups['k3-bf16-388-v1']+=['results/main/'+n for n in ['manifest.json','raw.jsonl','predictions.jsonl','progress.json','runtime.json']]
for group,names in groups.items():
 for n in names:
  f=p/group/n
  if not f.exists():continue
  data=f.read_bytes();s=data.decode(errors='replace')
  if n.endswith('.jsonl') and s and not s.endswith('\\n'):s=s[:s.rfind('\\n')+1]
  if n.endswith('.log'):s=s[-50000:]
  key=group+'/'+n;out[key]=s
  if n.endswith(('.json','.jsonl')):hashes[key]=hashlib.sha256(data).hexdigest()
print(json.dumps({'files':out,'hashes':hashes}))
"""
def main():
 failures=0
 while time.time()<DEADLINE+90:
  try:
   r=subprocess.run(BASE+['python3 -c '+shlex.quote(SCRIPT)],capture_output=True,text=True,timeout=25)
   if r.returncode:raise RuntimeError('SSH snapshot failed')
   obj=json.loads(r.stdout)
   for n,s in obj['files'].items():
    p=D.parent/n;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.chain-tmp');t.write_text(s);t.replace(p)
   subprocess.run([sys.executable,str(D/'export_pairs.py')],stdout=subprocess.DEVNULL,check=True)
   if 'k3-bf16-388-v1/results/main/runtime.json' in obj['files']:
    subprocess.run([sys.executable,str(D.parent/'k3-bf16-388-v1/export_results.py')],stdout=subprocess.DEVNULL,check=True)
   s=json.loads(obj['files'].get('observation-pilot-v1/session-status.json','{}'))
   progress=json.loads(obj['files'].get('observation-pilot-v1/results/paired/progress.json','{}'))
   print(json.dumps({'at':time.time(),'state':s.get('state'),'attempted_requests':progress.get('attempted_requests',0),'paired_samples':progress.get('paired_samples',0),'reason':s.get('reason')},ensure_ascii=False),flush=True)
   failures=0
   if s.get('state') in ('complete','stopped'):
    verified={n:hashlib.sha256((D.parent/n).read_bytes()).hexdigest()==h for n,h in obj['hashes'].items()}
    (D/'backup-verification.json').write_text(json.dumps({'files':verified,'all_match':all(verified.values()),'at':time.time()},indent=2)+'\n')
    return
  except Exception as e:
   failures+=1;print('Backup retry: '+str(e),flush=True)
   if failures>=5:return
  time.sleep(15)
if __name__=='__main__':main()
