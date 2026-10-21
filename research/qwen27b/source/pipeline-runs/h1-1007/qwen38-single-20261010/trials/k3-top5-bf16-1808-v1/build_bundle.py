"""Package only inference files. All rule/day/source evaluation data stays local."""
from pathlib import Path
import json,hashlib,tarfile
D=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();plan=json.loads((D/'plan.json').read_text())
(D/'inference-plan.json').write_text(json.dumps({k:plan[k] for k in ['model','generation','inputs']},ensure_ascii=False,indent=2)+'\n')
names=['data/model-inputs.json','inference-plan.json','runner/run_batch.py','runner/output_contract.py','runner/meaning_schema.json','runner/prompt.txt','remote_session.py','serve-bf16.sh']
manifest={'version':D.name,'files_sha256':{n:sha(D/n) for n in names}};(D/'bundle-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
with tarfile.open(D/'inference-bundle.tar.gz','w:gz') as t:
 for n in names+['bundle-manifest.json']:t.add(D/n,arcname=n)
forbidden=['rule_hits','day_context','day_z','day_density','hit_type','bearw_v2','top5-extreme-day-hit']
with tarfile.open(D/'inference-bundle.tar.gz') as t:
 for member in t:
  content=t.extractfile(member).read().decode()
  for word in forbidden:assert word not in content,(member.name,word)
rows=json.loads((D/'data/model-inputs.json').read_text())['rows']
assert len(rows)==1808 and len({r['id'] for r in rows})==1808
assert all(set(r)=={'id','text','stock_code','stock_name','published_at'} for r in rows)
check={'archive_sha256':sha(D/'inference-bundle.tar.gz'),'bytes':(D/'inference-bundle.tar.gz').stat().st_size,'members':names+['bundle-manifest.json'],'rule_day_context_absent':True,'rows':1808,'unique_request_ids':1808,'semantic_files_identical_to388':all(sha(D/'runner'/n)==sha(D.parent/'k3-bf16-388-v1/runner'/n) for n in ['prompt.txt','meaning_schema.json','output_contract.py'])}
(D/'bundle-check.json').write_text(json.dumps(check,indent=2)+'\n')
local=['plan.json','data-audit.json','private/source.jsonl','private/source-meta.json','private/evaluation-sidecar.jsonl','export_results.py']
(D/'evaluation-manifest.json').write_text(json.dumps({'files_sha256':{n:sha(D/n) for n in local}},indent=2)+'\n');print(json.dumps(check))
