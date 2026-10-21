"""On preserved GPU disk: validate bundle, reuse/start BF16, then drain 1808 inputs."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

D=Path(__file__).resolve().parent.parent
R=Path('/root/qwen38-single-20261010')
MODEL=Path('/root/model-cache/Qwen3.8-27B-BF16')
DOWNLOAD=Path('/root/model-cache/qwen38-bf16-download')
ALIAS='qwen38-27b-bf16'
def state(value,**kw):
 p=D/'session-status.tmp';p.write_text(json.dumps({'state':value,'updated_epoch':time.time(),**kw}));p.replace(D/'session-status.json')
def models():
 try:
  with urllib.request.urlopen('http://127.0.0.1:8000/v1/models',timeout=2) as r:return json.load(r)
 except Exception:return None
def validate_files():
 m=json.loads((D/'bundle-manifest.json').read_text())
 for name,h in m['files_sha256'].items():
  if hashlib.sha256((D/name).read_bytes()).hexdigest()!=h:raise ValueError('Bundle file changed: '+name)
def preflight():
 if not (R/'venv/bin/vllm').exists():raise RuntimeError('Missing preserved vLLM runtime; no automatic download started')
 meta=json.loads((DOWNLOAD/'repo_metadata.json').read_text());v=json.loads((DOWNLOAD/'verification.json').read_text())
 revision=json.loads((D/'inference-plan.json').read_text())['model']['revision']
 if not v.get('complete') or v.get('revision')!=revision or meta.get('sha')!=revision:raise RuntimeError('Model verification/revision missing or different')
 entries={x['file']:x for x in v['files']}
 for f in meta['siblings']:
  p=MODEL/f['rfilename']
  if not entries.get(f['rfilename'],{}).get('valid') or not p.exists() or p.stat().st_size!=f['size'] or Path(str(p)+'.aria2').exists():raise RuntimeError('Incomplete or changed model file: '+f['rfilename'])
  if p.stat().st_mtime>(DOWNLOAD/'verification.json').stat().st_mtime+1:raise RuntimeError('Model file newer than hash-verification receipt: '+f['rfilename'])
 index=json.loads((MODEL/'model.safetensors.index.json').read_text())
 for f in set(index['weight_map'].values()):
  if not (MODEL/f).is_file():raise RuntimeError('Weight shard missing: '+f)
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used','--format=csv,noheader,nounits'],text=True,timeout=10)
 cards=[s.split(',') for s in gpu.strip().splitlines()]
 if len(cards)!=2 or any(float(c[1])<39000 for c in cards):raise RuntimeError('This resume requires two >=40GB GPUs')
 (D/'preflight.json').write_text(json.dumps({'gpu':gpu,'model_revision':revision,'validation':'Prior full hash receipt plus current sizes/mtime; no new weight download','at':time.time()}))
 return gpu
def main():
 p=argparse.ArgumentParser();p.add_argument('--deadline-epoch',type=float,required=True);a=p.parse_args()
 if a.deadline_epoch-time.time()<300:raise SystemExit('Insufficient time before deadline')
 lock=(D/'session.lock').open('a')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:print('Session already active');return 0
 owned=None
 try:
  state('preflight',deadline_epoch=a.deadline_epoch);validate_files()
  existing_runtime=D/'results/main/runtime.json'
  if existing_runtime.exists() and json.loads(existing_runtime.read_text()).get('all_attempted'):
   state('complete',attempted=1808,reason='existing_complete_receipts_preserved',deadline_epoch=a.deadline_epoch)
   return 0
  gpu=preflight()
  existing=models()
  if existing:
   if not any(x.get('id')==ALIAS for x in existing.get('data',[])):raise RuntimeError('Port8000 occupied by a different model')
   ps=subprocess.check_output(['ps','-eo','args'],text=True)
   if not any('vllm' in line and str(MODEL) in line and ('bfloat16' in line or '--dtype auto' in line) for line in ps.splitlines()):raise RuntimeError('Cannot verify existing service uses the pinned BF16 model path')
  else:
   state('loading_model',deadline_epoch=a.deadline_epoch,gpu=gpu)
   with (D/'server.log').open('ab',buffering=0) as log:
    owned=subprocess.Popen(['bash',str(Path(__file__).resolve().parent/'serve-bf16-tp2.sh')],stdout=log,stderr=log,start_new_session=True)
   (D/'server.pid').write_text(str(owned.pid))
   until=min(a.deadline_epoch-250,time.time()+600)
   while time.time()<until:
    if owned.poll() is not None:raise RuntimeError('Model service exited; inspect server.log immediately')
    ready=models()
    if ready and any(x.get('id')==ALIAS for x in ready.get('data',[])):break
    time.sleep(2)
   else:raise RuntimeError('Model startup timeout/deadline; inspect server.log')
  plan=json.loads((D/'inference-plan.json').read_text());g=plan['generation']
  state('running',expected=1808,deadline_epoch=a.deadline_epoch)
  cmd=[str(R/'venv/bin/python'),str(D/'runner/run_batch.py'),'--input',str(D/'data/model-inputs.json'),'--out',str(D/'results/main'),'--prompt',str(D/'runner/prompt.txt'),'--schema',str(D/'runner/meaning_schema.json'),'--base-url','http://127.0.0.1:8000/v1','--model',ALIAS,'--model-revision',plan['model']['revision'],'--stage','k3_top5_1808','--thinking','off','--reasoning-effort','omit','--temperature',str(g['temperature']),'--top-p',str(g['top_p']),'--max-tokens',str(g['max_tokens']),'--concurrency',str(g['concurrency']),'--smoke-count','1','--timeout',str(g['timeout']),'--deadline-epoch',str(a.deadline_epoch)]
  rc=subprocess.call(cmd)
  f=D/'results/main/runtime.json';runtime=json.loads(f.read_text()) if f.exists() else {}
  state('complete' if runtime.get('all_attempted') else 'stopped',deadline_epoch=a.deadline_epoch,attempted=runtime.get('completed_rows',0),format_valid=runtime.get('successful_rows',0),reason=runtime.get('stop_reason','runner_failed'),exit_code=rc)
  return rc
 except Exception as e:
  state('stopped',reason=str(e),deadline_epoch=a.deadline_epoch);print(type(e).__name__+': '+str(e),flush=True);return 1
 finally:
  if owned and owned.poll() is None:
   os.killpg(owned.pid,signal.SIGTERM)
   try:owned.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(owned.pid,signal.SIGKILL)
if __name__=='__main__':raise SystemExit(main())
