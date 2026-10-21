import hashlib,io,json,pathlib,sys,tarfile,time,gzip
root=pathlib.Path('/root/qwen38-single-20261010/trials/k3-top5-bf16-1808-v1')
files={};counts={}
for name in ['session-status.json','preflight.json','bundle-manifest.json','inference-plan.json','data/model-inputs.json','results/main/raw.jsonl','results/main/predictions.jsonl','results/main/progress.json','results/main/runtime.json','results/main/manifest.json','results/main/resolved-prompt.txt','server.log','supervisor.log','cpu-delivery/output/summary.json','cpu-delivery/output/manifest.json','cpu-delivery/output/model-answers.jsonl.gz','cpu-delivery/output/mechanical-issues.jsonl']:
 p=root/name
 if not p.exists():continue
 with p.open('rb') as f:
  size=p.stat().st_size;b=f.read(size)
 if name.endswith('.jsonl'):
  b=b[:b.rfind(b'\n')+1] if b else b
  counts[name]=len(b.splitlines())
 if name.endswith('.log'):b=b[-200000:]
 files[name]=b
if 'results/main/raw.jsonl' not in files:raise RuntimeError('No remote raw receipts')
meta={'captured_epoch':time.time(),'files_sha256':{n:hashlib.sha256(b).hexdigest() for n,b in files.items()},'jsonl_counts':counts,'notice':'Live snapshot; complete raw receipt lines only. Parsed output may lag the raw log. GPU not stopped.'}
files['snapshot-manifest.json']=json.dumps(meta,ensure_ascii=False,indent=2).encode()
with gzip.GzipFile(fileobj=sys.stdout.buffer,mode='wb',compresslevel=1,mtime=0) as gz:
 with tarfile.open(fileobj=gz,mode='w|') as t:
  for n,b in files.items():
   info=tarfile.TarInfo(n);info.size=len(b);info.mtime=int(time.time());t.addfile(info,io.BytesIO(b))
