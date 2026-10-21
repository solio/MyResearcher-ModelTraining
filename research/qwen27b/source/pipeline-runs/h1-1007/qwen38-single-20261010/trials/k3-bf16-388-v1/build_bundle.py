"""Freeze and pack inference-only files; evaluation answers never go to GPU disk."""
from pathlib import Path
import hashlib
import json
import tarfile

D=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((D/'plan.json').read_text())
(D/'inference-plan.json').write_text(json.dumps({k:plan[k] for k in ['model','generation','inputs']},ensure_ascii=False,indent=2)+'\n')
names=['data/model-inputs.json','inference-plan.json','runner/run_batch.py','runner/output_contract.py','runner/meaning_schema.json','runner/prompt.txt','remote_session.py','serve-bf16.sh']
manifest={'version':'k3-bf16-388-v1','files_sha256':{n:sha(D/n) for n in names}}
(D/'bundle-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
with tarfile.open(D/'inference-bundle.tar.gz','w:gz') as tar:
 for name in names+['bundle-manifest.json']:tar.add(D/name,arcname=name)
forbidden=['rule_hits','day_context','day_bear_density','day_z','high_extreme_day','calm_day']
with tarfile.open(D/'inference-bundle.tar.gz') as tar:
 for item in tar:
  text=tar.extractfile(item).read().decode()
  for token in forbidden:
   if token in text:raise ValueError('Evaluation field leaked into bundle: '+token+' / '+item.name)
check={'archive_sha256':sha(D/'inference-bundle.tar.gz'),'bytes':(D/'inference-bundle.tar.gz').stat().st_size,
 'member_names':names+['bundle-manifest.json'],'no_rule_answers_or_daily_context':True,
 'model_input_allowed_keys':['id','text','stock_code','stock_name','published_at']}
(D/'bundle-check.json').write_text(json.dumps(check,indent=2)+'\n')
local_names=['plan.json','data-audit.json','private/source.jsonl','private/source-meta.json','private/evaluation-sidecar.jsonl','private/rule-source.json','export_results.py']
(D/'evaluation-manifest.json').write_text(json.dumps({'files_sha256':{n:sha(D/n) for n in local_names}},indent=2)+'\n')
print(json.dumps(check))
