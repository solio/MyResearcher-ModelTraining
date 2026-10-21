"""Paired, interleaved A/B with identical model-visible input identities."""
import argparse
import concurrent.futures as cf
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

D=Path(__file__).resolve().parent
sys.path.insert(0,str(D/'runner'))
import runner_core as core
def main():
 p=argparse.ArgumentParser();p.add_argument('--deadline-epoch',type=float,required=True);p.add_argument('--base-url',default='http://127.0.0.1:8000/v1');p.add_argument('--out',type=Path,default=D/'results/paired');p.add_argument('--timeout',type=float,default=240);a=p.parse_args()
 if time.time()>=a.deadline_epoch:raise SystemExit('Deadline reached')
 out=a.out;out.mkdir(parents=True,exist_ok=True);lock=(out/'runner.lock').open('a')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise SystemExit('Another A/B runner owns this queue')
 schema=json.loads((D/'meaning_schema.json').read_text());plan=json.loads((D/'plan.json').read_text())
 rows=json.loads((D/'heldout-inputs.json').read_text())['rows']
 assert len(rows)==80 and len({r['id'] for r in rows})==80
 for name,h in plan['files_sha256'].items():
  if hashlib.sha256((D/name).read_bytes()).hexdigest()!=h:raise ValueError('Frozen study file changed: '+name)
 prompts={arm:core.build_prompt(schema,D/f'prompt-{arm.lower()}.txt') for arm in ['A','B']}
 tasks=[(r,arm) for i,r in enumerate(rows) for arm in (['A','B'] if i%2==0 else ['B','A'])]
 specs={arm:SimpleNamespace(model='qwen38-27b-bf16',base_url=a.base_url,temperature=0,top_p=1,max_tokens=2048,thinking='off',reasoning_effort='omit',timeout=a.timeout,deadline_epoch=a.deadline_epoch,api_key_env='LOCAL_LLM_API_KEY',observations_first=arm=='B') for arm in ['A','B']}
 config={'model':plan['model'],'precision':'BF16','generation':plan['generation'],'schema':schema,'prompts':prompts,
 'wire_root_property_order':{'A':['id','checks','observations'],'B':['id','observations','checks']},
 'input_sha256':hashlib.sha256((D/'heldout-inputs.json').read_bytes()).hexdigest(),
 'code_sha256':{n:hashlib.sha256((D/n).read_bytes()).hexdigest() for n in ['run_ab.py','runner/runner_core.py','runner/output_contract.py']}}
 run_hash=core.sha(config);mp=out/'manifest.json'
 if mp.exists():
  if json.loads(mp.read_text())['run_hash']!=run_hash:raise ValueError('Refuse changed A/B resume')
 else:core.atomic_json(mp,{'run_hash':run_hash,'config':config,'n_inputs':80,'n_requests':160,'created_epoch':time.time(),'schedule':'Alternating A/B order per sample, one shared 8-slot queue; model sees identical input JSON in both arms.'})
 for arm in ['A','B']:(out/f'prompt-{arm.lower()}-resolved.txt').write_text(prompts[arm])
 raw_path=out/'raw.jsonl';pred_path=out/'predictions.jsonl';records={}
 if raw_path.exists():
  data=raw_path.read_bytes()
  if data and not data.endswith(b'\n'):
   raw_path.with_suffix('.torn-backup').write_bytes(data);boundary=data.rfind(b'\n')+1
   try:json.loads(data[boundary:]);data+=b'\n'
   except (ValueError,UnicodeDecodeError):data=data[:boundary]
   raw_path.write_bytes(data)
  for line in data.decode().splitlines():
   r=json.loads(line)
   if r['run_hash']!=run_hash or r['task_id'] in records:raise ValueError('Foreign or duplicate receipt')
   records[r['task_id']]=r['prediction']
 taskkeys={r['id']+'::'+arm for r,arm in tasks}
 if set(records)-taskkeys:raise ValueError('Unknown task identity')
 temp=pred_path.with_suffix('.tmp');temp.write_text(''.join(core.dumps(records[r['id']+'::'+arm])+'\n' for r,arm in tasks if r['id']+'::'+arm in records));temp.replace(pred_path)
 signal.signal(signal.SIGTERM,lambda *_:core.STOP.set());signal.signal(signal.SIGINT,lambda *_:core.STOP.set())
 start=time.time();failures=0;stop_reason='complete'
 def progress():
  arms={arm:{'attempted':sum(x['arm']==arm for x in records.values()),'format_valid':sum(x['arm']==arm and x['status']=='ok' for x in records.values())} for arm in ['A','B']}
  paired=sum(all(r['id']+'::'+arm in records for arm in ['A','B']) for r in rows)
  state={'expected_requests':160,'attempted_requests':len(records),'paired_samples':paired,'expected_samples':80,'arms':arms,'updated_epoch':time.time(),'notice':'Progress and format validity, not semantic accuracy.'}
  core.atomic_json(out/'progress.json',state);return state
 def request(task):
  row,arm=task;result=core.request_one(row,specs[arm],prompts[arm],schema,run_hash)
  if result is not None:
   raw,pred=result
   for x in (raw,pred):x.update(arm=arm,task_id=row['id']+'::'+arm)
  return result
 def save(result):
  if result is None:return
  raw,pred=result;raw['prediction']=pred
  for path,x in [(raw_path,raw),(pred_path,pred)]:
   with path.open('a') as f:f.write(core.dumps(x)+'\n');f.flush();os.fsync(f.fileno())
  records[pred['task_id']]=pred;progress()
  print(core.dumps({'task':pred['task_id'],'status':pred['status'],'attempted':len(records),'error':pred.get('error')}),flush=True)
 def may_start():return not core.STOP.is_set() and time.time()<a.deadline_epoch-a.timeout
 pending=iter(t for t in tasks if t[0]['id']+'::'+t[1] not in records)
 with cf.ThreadPoolExecutor(max_workers=8) as pool:
  futures=set();exhausted=False
  while futures or not exhausted:
   while not exhausted and len(futures)<8 and may_start():
    t=next(pending,None)
    if t is None:exhausted=True;break
    futures.add(pool.submit(request,t))
   if not may_start():exhausted=True;stop_reason='deadline_or_signal' if stop_reason=='complete' else stop_reason
   if not futures:break
   done,futures=cf.wait(futures,timeout=1,return_when=cf.FIRST_COMPLETED)
   for f in done:
    result=f.result();save(result)
    if result:
     x=result[1];service_error=x['status']!='ok' and x.get('finish_reason') not in ('stop','length')
     failures=failures+1 if service_error else 0
     if failures>=3:core.STOP.set();exhausted=True;stop_reason='repeated_service_errors'
 status=progress();status.update(run_hash=run_hash,all_attempted=len(records)==160,stop_reason=stop_reason,session_started_epoch=start,session_wall_seconds=time.time()-start,deadline_epoch=a.deadline_epoch)
 core.atomic_json(out/'runtime.json',status)
 return 0 if status['all_attempted'] else 2
if __name__=='__main__':raise SystemExit(main())
