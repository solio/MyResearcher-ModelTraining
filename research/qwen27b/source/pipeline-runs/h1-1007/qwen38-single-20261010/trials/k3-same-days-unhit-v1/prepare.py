"""Frozen complementary population: same five stocks and forty days, unchanged semantics."""
from pathlib import Path
import collections,hashlib,json,shutil,tarfile,time
D=Path(__file__).resolve().parent;T=D.parent/'k3-top5-bf16-1808-v1';OLD=D.parent/'k3-bf16-388-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2))
def jsonl(p,rows):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
if (D/'bundle-manifest.json').exists():raise SystemExit('Already frozen; refuse overwrite')
for name in ['data','private','runner','exports']:(D/name).mkdir(parents=True,exist_ok=True)
assessment=json.loads((T/'resume-20261013/next-batch-assessment.json').read_text())
assert assessment['original_export_rows_unmatched_in_db']==0 and assessment['day_count_mismatches']==0
all_rows=[json.loads(l) for l in (T/'resume-20261013/same-days-source-private.jsonl').read_text().splitlines()]
assert len(all_rows)==4402 and sum(r['rule_hit'] for r in all_rows)==1808
names={r['stock_code']:r['stock_name'] for r in map(json.loads,(T/'private/source.jsonl').read_text().splitlines())}
previous_inputs=json.loads((OLD/'data/model-inputs.json').read_text())['rows'];previous={(r['stock_code'],r['published_at'],r['text']):r for r in previous_inputs}
receipts={r['id']:r for r in map(json.loads,(OLD/'results/main/raw.jsonl').read_text().splitlines())}
for n in ['prompt.txt','meaning_schema.json','output_contract.py']:
 assert sha(T/'runner'/n)==sha(OLD/'runner'/n)
 shutil.copy2(T/'runner'/n,D/'runner'/n)
rows=sorted([r for r in all_rows if not r['rule_hit']],key=lambda r:(r['stock_code'],r['day'],r['published_at'],r['source'],r['source_item_id']))
reused=[];cells=collections.defaultdict(list);private=[]
for i,r in enumerate(rows,1):
 r={**r,'id':f'N{i:04}','stock_name':names[r['stock_code']],'source_fingerprint':hashlib.sha256(json.dumps(r,ensure_ascii=False,sort_keys=True).encode()).hexdigest()}
 key=(r['stock_code'],r['published_at'],r['title']);old=previous.get(key)
 if old and old['id'] in receipts:
  rr=receipts[old['id']];assert json.loads(rr['request']['messages'][1]['content'])==old
  r.update(inference_source='reused_prior388',prior_receipt_id=old['id'])
  reused.append({'target_id':r['id'],'prior_trial':'k3-bf16-388-v1','prior_receipt_id':old['id'],'original_input':old,'original_raw_receipt':rr})
 else:
  r.update(inference_source='new_request',prior_receipt_id=None);cells[(r['stock_code'],r['day'])].append(r)
 private.append(r)
# Balance progress across all date groups. No model scores or returns affect order.
for key in cells:cells[key].sort(key=lambda r:hashlib.sha256(('seed42:'+r['source']+':'+r['source_item_id']).encode()).hexdigest())
queue=[]
while any(cells.values()):
 for key in sorted(cells):
  if cells[key]:queue.append(cells[key].pop(0))
inputs=[{'id':r['id'],'text':r['title'],'stock_code':r['stock_code'],'stock_name':r['stock_name'],'published_at':r['published_at']} for r in queue]
assert len(rows)==2594 and len(inputs)==2588 and len(reused)==6
assert len({r['id'] for r in private})==2594
assert all(set(r)=={'id','text','stock_code','stock_name','published_at'} for r in inputs)
assert all(r['text'].strip() for r in inputs)
write(D/'data/model-inputs.json',{'rows':inputs});jsonl(D/'private/evaluation-sidecar.jsonl',private);jsonl(D/'private/reused-prior.jsonl',reused)
shutil.copy2(T/'resume-20261013/same-days-source-private.jsonl',D/'private/full-day-source.jsonl')
shutil.copy2(T/'resume-20261013/next-batch-assessment.json',D/'private/source-reconciliation.json')
runner=(T/'runner/run_batch.py').read_text().replace("'k3_top5_1808'], default='stage1'", "'k3_top5_1808', 'k3_same_days_unhit'], default='stage1'")
needle="    semantics = {'model': args.model"
assert needle in runner
runner=runner.replace(needle,"    if args.stage == 'k3_same_days_unhit' and len(rows) > 2588:\n        raise ValueError('k3_same_days_unhit cannot exceed 2588 frozen new requests')\n"+needle)
(D/'runner/run_batch.py').write_text(runner)
plan=json.loads((T/'inference-plan.json').read_text());plan['inputs']={'n':2588,'text':'Unmodified titles only; remaining rows in exactly the same stock-date cells','ids':'N0001..N2594 excluding six prior exact-input receipts'};write(D/'inference-plan.json',plan)
manifest={'created_epoch':time.time(),'files_sha256':{str(p.relative_to(D)):sha(p) for p in [D/'data/model-inputs.json',D/'inference-plan.json',*sorted((D/'runner').glob('*'))] if p.is_file()}}
write(D/'bundle-manifest.json',manifest)
with tarfile.open(D/'inference-bundle.tar.gz','w:gz') as tar:
 for n in [*manifest['files_sha256'],'bundle-manifest.json']:tar.add(D/n,arcname=n)
write(D/'plan.json',{'status':'prepared_not_started','purpose':'Complete the same 40 stock-date cells by reading posts not selected by the unchanged K3 wordlists; assess extra information and rule misses within this selected population, without treating non-hits as calm.','previous_rule_hit_rows':1808,'same_day_population':4402,'complement_rows':2594,'new_gpu_requests':2588,'reused_prior_exact_inputs':6,'source_reconciliation':'All 40 day totals match the original K3 export; all original 1808 hit rows matched and preserved.','semantics':'Prompt, schema, output contract, BF16 revision, temperature, max tokens, thinking mode identical to K3 388/1808. New run stage changes only batch-size authorization.','source_scope':'These are days originally selected for high rule-density in five selected stocks. Completing them does not establish performance on ordinary dates, other stocks, or future stock returns. No daily-extreme label is invented.','scores':'Original intensity/support retained; support is uncalibrated and no scalar sentiment mapping imposed.','queue_order':'Deterministic round-robin across stock-date cells; seed42 SHA ordering within each cell. Dates, returns, model judgments do not select winners.','partial_results':'Keep exact missing IDs and per-day coverage. Only fully covered days may be described as complete; invalid meanings remain unresolved.','identity':'Database source+source_item_id retained locally; duplicates are not merged merely for equal titles. Six exact existing 388 inputs reused with full original receipt provenance.','budget':'Reuse current loaded BF16 service after remaining551; bounded by five-hour rental with ten-minute working reserve. No automatic renewal.','bundle_sha256':sha(D/'inference-bundle.tar.gz'),'private_files_sha256':{str(p.relative_to(D)):sha(p) for p in (D/'private').glob('*')}})
print(json.dumps({'population':4402,'rule_hits_existing_batch':1808,'complement':2594,'requests':2588,'reused':6,'bundle_bytes':(D/'inference-bundle.tar.gz').stat().st_size,'bundle_sha256':sha(D/'inference-bundle.tar.gz')}))
