"""Freeze all supplied rows; preserve publication time and upstream day grouping separately."""
from pathlib import Path
import json,hashlib,collections,shutil,datetime
D=Path(__file__).resolve().parent;OLD=D.parent/'k3-bf16-388-v1';PROJECT=D.parents[4]
for n in ['private','data','runner','exports']:(D/n).mkdir(parents=True,exist_ok=True)
assert not (D/'results/main/raw.jsonl').exists(),'Cannot reprepare after inference starts'
src=PROJECT/'worktrees/dashboard-k3-experiments/exports/top5-extreme-day-hit-sentences-20261012.jsonl';meta=src.with_name(src.stem+'-meta.json')
rows=[json.loads(l) for l in src.read_text().splitlines() if l.strip()];assert len(rows)==1808
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(f,o):(D/f).write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n')
for a,b in [(src,'private/source.jsonl'),(meta,'private/source-meta.json')]:shutil.copy2(a,D/b)
for n in ['run_batch.py','output_contract.py','meaning_schema.json','prompt.txt']:shutil.copy2(OLD/'runner'/n,D/'runner'/n)
# Batch-size authorization is transport configuration, not a semantic prompt change.
f=D/'runner/run_batch.py';code=f.read_text().replace("choices=['stage1', 'stage2', 'bf16_exploratory']", "choices=['stage1', 'stage2', 'bf16_exploratory', 'k3_top5_1808']")
code=code.replace("    semantics = {'model': args.model", "    if args.stage == 'k3_top5_1808' and len(rows) > 1808:\n        raise ValueError('k3_top5_1808 cannot exceed 1808 authorized inputs')\n    semantics = {'model': args.model")
f.write_text(code)
ident=lambda r:json.dumps([r['stock_code'],r['published_at'],r['title']],ensure_ascii=False,separators=(',',':'))
counts=collections.Counter(ident(r) for r in rows);models=[];side=[];days=collections.defaultdict(list);date_differences=[]
oldrows=[json.loads(l) for l in (OLD/'private/evaluation-sidecar.jsonl').read_text().splitlines()];prior={ident(r):r['id'] for r in oldrows}
for i,r in enumerate(rows,1):
 assert set(r)=={'stock_code','stock_name','day','day_z','day_posts','day_density','hit_type','published_at','title'}
 assert isinstance(r['title'],str) and r['title'].strip()
 published=datetime.datetime.fromisoformat(r['published_at']);assert published.utcoffset() is not None
 assert r['hit_type'] in ['bearw_v2','nick'] and r['day_posts']>0 and 0<=r['day_density']<=1
 rid=f'E{i:04d}';identity=ident(r)
 if published.date().isoformat()!=r['day']:date_differences.append({'id':rid,'published_at':r['published_at'],'upstream_day':r['day']})
 models.append({'id':rid,'text':r['title'],'stock_code':r['stock_code'],'stock_name':r['stock_name'],'published_at':r['published_at']})
 side.append({**r,'id':rid,'source_row':i,'source_identity_sha256':hashlib.sha256(identity.encode()).hexdigest(),'identical_identity_rows':counts[identity],'possible_title_cutoff':len(r['title'])==40,'prior388_same_identity_id':prior.get(identity)})
 days[(r['stock_code'],r['day'])].append(r)
for group in days.values():assert len({(x['day_z'],x['day_posts'],x['day_density']) for x in group})==1
write('data/model-inputs.json',{'rows':models});(D/'private/evaluation-sidecar.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in side))
audit={'source_rows':len(rows),'unique_stock_timestamp_title':len(counts),'identical_identity_groups':[[r['id'] for r in side if r['source_identity_sha256']==hashlib.sha256(k.encode()).hexdigest()] for k,n in counts.items() if n>1],'duplicate_policy':'Retain every source row. Identical stock/timestamp/title alone cannot prove duplicate posts without original post IDs. No silent dropping or aggregation deduplication.','stocks':dict(collections.Counter(r['stock_name'] for r in rows)),'stock_days':len(days),'hit_types':dict(collections.Counter(r['hit_type'] for r in rows)),'date_min':min(r['day'] for r in rows),'date_max':max(r['day'] for r in rows),'title_max_chars':max(len(r['title']) for r in rows),'titles_at_40_chars':sum(len(r['title'])==40 for r in rows),'prior388_identical_rows':sum(r['prior388_same_identity_id'] is not None for r in side),'published_date_differs_from_upstream_day':date_differences,'date_policy':'Keep actual published_at in model input and upstream day only in local sidecar. Do not overwrite timestamps or assume day=calendar date; upstream day grouping may include nontrading-day rolls.','selection_scope':'Supplied rule-hit titles on selected extreme days among top5 stocks; not all daily posts and not random corpus sample.','evaluation_limits':['Cannot infer full-day psychological state or recall on rule misses from hit-only titles.','bearw.v2/nickname selection is not the former four v1 booleans; unsupplied rule types remain unknown.','Possible truncation at40characters is retained without completion.'],'model_input_allowed_keys':['id','text','stock_code','stock_name','published_at'],'rule_information_sent_to_model':False}
write('data-audit.json',audit)
oldplan=json.loads((OLD/'plan.json').read_text());plan={'status':'prepared_not_run','task':'Annotate supplied top5 extreme-day hit titles with frozen K3 27B BF16 semantics','model':oldplan['model'],'generation':oldplan['generation'],'inputs':{'n':len(rows),'text':'original title only, unmodified','date_min':audit['date_min'],'date_max':audit['date_max']},'frozen_from':'k3-bf16-388-v1','transport_change':'New stage k3_top5_1808 permits this authorized1808-row batch, preserves prior80/400caps and semantics.','frozen_runner_sha256':{f:sha(D/'runner'/f) for f in ['run_batch.py','output_contract.py','meaning_schema.json','prompt.txt']},'source_sha256':{src.name:sha(src),meta.name:sha(meta)},'semantic_policy':'Same four checks plus complete free observations/reasons, scoped intensity and support. No threshold discarding observations; selection facts are kept outside model input.','id_policy':'Short transport IDs E0001..E1808, fixed source-row and identity mapping; avoids long-ID copy failures without changing semantics.','daily_policy':'day_extreme_vote=null; only selected hit titles supplied. Preserve upstream day and actual publication timestamp separately.','runtime_policy':'Fresh explicit deadline required at launch. Resume identical hashes, save raw before parse, continue after format failure, no automatic renewal/download/extra experiment.','training_allowed':False,'memory_update_allowed':False}
write('plan.json',plan);write('session-status.json',{'state':'prepared_not_run','expected':1808,'attempted':0,'notice':'Prepared locally, no active node or renewed instance.'})
for f in ['remote_session.py','dispatch.py','serve-bf16.sh']:(D/f).write_text((OLD/f).read_text().replace('k3-bf16-388-v1','k3-top5-bf16-1808-v1').replace('388','1808').replace("'--stage','bf16_exploratory'", "'--stage','k3_top5_1808'"))
print(json.dumps({'rows':len(rows),'stocks':len(audit['stocks']),'stock_days':len(days),'distinct_identities':len(counts),'date_mapping_differences':len(date_differences),'prior388_overlap':audit['prior388_identical_rows'],'titles_at40':audit['titles_at_40_chars']},ensure_ascii=False))
