"""Download an immutable, verified snapshot of this batch without stopping inference."""
import argparse, hashlib, io, json, pathlib, shlex, subprocess, tarfile, time
from datetime import datetime
from zoneinfo import ZoneInfo

D=pathlib.Path(__file__).resolve().parent
REMOTE='/root/qwen38-single-20261010/trials/k3-top5-bf16-1808-v1'
REMOTE_SCRIPT='''import hashlib,io,json,pathlib,sys,tarfile,time
root=pathlib.Path(DEST)
files={};counts={}
for name in ['session-status.json','preflight.json','bundle-manifest.json','inference-plan.json','data/model-inputs.json','results/main/raw.jsonl','results/main/predictions.jsonl','results/main/progress.json','results/main/manifest.json','results/main/resolved-prompt.txt','server.log','supervisor.log']:
 p=root/name
 if not p.exists():continue
 with p.open('rb') as f:
  size=p.stat().st_size;b=f.read(size)
 if name.endswith('.jsonl'):
  b=b[:b.rfind(b'\\n')+1] if b else b
  counts[name]=len(b.splitlines())
 if name.endswith('.log'):b=b[-200000:]
 files[name]=b
if 'results/main/raw.jsonl' not in files:raise RuntimeError('No remote raw receipts')
meta={'captured_epoch':time.time(),'files_sha256':{n:hashlib.sha256(b).hexdigest() for n,b in files.items()},'jsonl_counts':counts,'notice':'Live snapshot; complete raw receipt lines only. Parsed output may lag the raw log. GPU not stopped.'}
files['snapshot-manifest.json']=json.dumps(meta,ensure_ascii=False,indent=2).encode()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:
 for n,b in files.items():
  info=tarfile.TarInfo(n);info.size=len(b);info.mtime=int(time.time());t.addfile(info,io.BytesIO(b))
'''.replace('DEST',repr(REMOTE))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check-only',action='store_true');p.add_argument('--name',default='20261012-1925');a=p.parse_args()
 if '/' in a.name or a.name in ('.','..'):raise ValueError('Invalid snapshot name')
 s=json.loads((D/'session-launch.json').read_text())
 base=['ssh','-S',s['socket'],'-p',str(s['port']),'-o','BatchMode=yes','-o','ConnectTimeout=8','root@'+s['host']]
 if a.check_only:
  check="import pathlib,json;d=pathlib.Path("+repr(REMOTE)+");p=d/'results/main/raw.jsonl';print(json.dumps({'raw_exists':p.exists(),'raw_bytes':p.stat().st_size if p.exists() else 0}))"
  r=subprocess.run(base+['python3 -c '+shlex.quote(check)],capture_output=True,timeout=15,check=True);print(r.stdout.decode());return
 dest=D/'snapshots'/a.name;dest.mkdir(parents=True,exist_ok=True)
 if (dest/'download-complete.json').exists():
  print((dest/'download-complete.json').read_text());return
 temp=dest/'remote-snapshot.tgz.part'
 with temp.open('wb') as f:
  r=subprocess.run(base+['python3 -c '+shlex.quote(REMOTE_SCRIPT)],stdout=f,stderr=subprocess.PIPE,timeout=120)
 if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace')[-1500:])
 archive=dest/'remote-snapshot.tgz';temp.replace(archive)
 with tarfile.open(archive,'r:gz') as tar:
  members=tar.getmembers()
  if any(not m.isfile() or pathlib.PurePosixPath(m.name).is_absolute() or '..' in pathlib.PurePosixPath(m.name).parts for m in members):raise ValueError('Unexpected archive paths')
  tar.extractall(dest,filter='data')
 meta=json.loads((dest/'snapshot-manifest.json').read_text())
 for n,h in meta['files_sha256'].items():
  if hashlib.sha256((dest/n).read_bytes()).hexdigest()!=h:raise ValueError('Download hash mismatch: '+n)
 source={r['id']:r for r in map(json.loads,(D/'private/evaluation-sidecar.jsonl').read_text().splitlines())}
 rows=[json.loads(l) for l in (dest/'results/main/raw.jsonl').read_text().splitlines()]
 ids=[r['id'] for r in rows]
 if len(ids)!=len(set(ids)):raise ValueError('Duplicate response IDs; preserve snapshot for inspection')
 exports=[]
 for r in rows:
  rid=r['id'];content=None;parsed=None
  try:content=r['response']['choices'][0]['message']['content']
  except (KeyError,IndexError,TypeError):pass
  try:
   def unique(pairs):
    out={}
    for k,v in pairs:
     if k in out:raise ValueError('Duplicate JSON key')
     out[k]=v
    return out
   parsed=json.loads(content,object_pairs_hook=unique)
  except (ValueError,TypeError):pass
  pred=r.get('prediction') or {}
  exports.append({**source[rid],'qwen_model_output':parsed,'qwen_model_content':content,
   'original_contract_status':pred.get('status'),'original_contract_error':pred.get('error'),
   'semantic_quality':'not_assessed','raw_receipt_file':'results/main/raw.jsonl','raw_receipt_id':rid})
 data=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in exports).encode()
 (dest/'k3-raw-return.jsonl').write_bytes(data)
 missing=[k for k in source if k not in set(ids)]
 (dest/'missing-ids.json').write_text(json.dumps(missing))
 complete={'downloaded_epoch':time.time(),'snapshot_captured_epoch':meta['captured_epoch'],'local_path':str(dest),
  'downloaded_responses':len(rows),'expected':len(source),'missing':len(missing),'unique_ids_verified':True,
  'all_download_hashes_verified':True,'original_contract_ok':sum(r.get('prediction',{}).get('status')=='ok' for r in rows),
  'k3_return_sha256':hashlib.sha256(data).hexdigest(),'notice':'Downloaded count is progress, not semantic accuracy. Inference remains running.'}
 (dest/'download-complete.json').write_text(json.dumps(complete,ensure_ascii=False,indent=2));print(json.dumps(complete,ensure_ascii=False))

if __name__=='__main__':main()
