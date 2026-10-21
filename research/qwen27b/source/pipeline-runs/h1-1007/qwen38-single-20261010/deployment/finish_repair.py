import json
import sys
import time
from pathlib import Path
ROOT = Path('/root/qwen38-single-20261010')
sys.path.insert(0, str(ROOT / 'runner-structured'))
from evaluate import evaluate
from run_batch import atomic_json, load_rows
stop_at = float((ROOT / 'run-deadline.txt').read_text()) + 180
batches = ['stage1', 'stage1-structured']
if (ROOT / 'continuation-plan.json').exists():
    batches.append('stage1-continuation')
if (ROOT / 'recovery-plan.json').exists():
    batches.append('stage1-recovery')
runtime = ROOT / 'results' / batches[-1] / 'runtime.json'
while not runtime.exists() and time.time() < stop_at:
    time.sleep(5)
if not runtime.exists():
    raise SystemExit('Batch has no final runtime record yet; original receipts remain on disk.')
manifests, predictions, errors = [], {}, []
for batch in batches:
    folder = ROOT / 'results' / batch
    manifest = json.loads((folder / 'manifest.json').read_text())
    manifests.append({'batch': batch, 'run_hash': manifest['run_hash'], 'config_hash': manifest['config_hash']})
    for row in load_rows(folder / 'predictions.jsonl'):
        if row['run_hash'] != manifest['run_hash']:
            raise ValueError('Receipt provenance mismatch')
        if row['status'] != 'ok':
            errors.append({'batch': batch, 'id': row['id'], 'error': row.get('error')})
        predictions[row['id']] = row
inputs = load_rows(ROOT / 'data/stage1-inputs.json')
refs = load_rows(ROOT / 'data/stage1-reference.json')
schema = json.loads((ROOT / 'runner-structured/meaning_schema.json').read_text())
report = evaluate(inputs, refs, list(predictions.values()), schema)
report['batch_provenance'] = manifests
report['original_attempt_errors'] = errors
report['evaluation_scope'] = 'Diagnostic review of 80 planned examples. First 18 successes without constrained JSON; remaining inputs with constrained JSON. Same prompt and meaning definitions. This is not a single-configuration benchmark.'
report['automatic_expansion_allowed'] = False
report['gate']['passed'] = False
report['gate']['fail_reasons'].append('两个输出配置的合并诊断，仅供逐条审阅，不能自动作为扩量依据')
atomic_json(ROOT / 'results/combined-evaluation.json', report)
atomic_json(ROOT / 'results/combined-provenance.json', {'batches':manifests, 'original_attempt_errors':errors, 'unique_successes':sum(x['status']=='ok' for x in predictions.values()), 'expected':80})
print(json.dumps({'unique_successes':sum(x['status']=='ok' for x in predictions.values()),'totals':report['totals'],'scope':report['evaluation_scope']}, ensure_ascii=False))
