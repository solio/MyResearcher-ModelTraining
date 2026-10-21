#!/usr/bin/env python3
"""Check frozen inputs, checkpoint identities, and remote/local file hashes."""
import hashlib
import json
import subprocess
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
REMOTE = '/root/qwen38-single-20261010/trials/bf16-400-v1'


def read_rows(path):
    if path.suffix == '.jsonl':
        return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    obj = json.loads(path.read_text())
    return obj.get('rows', obj) if isinstance(obj, dict) else obj


def main():
    frozen = json.loads((BASE / 'plan.json').read_text())['files_sha256']
    for name, digest in frozen.items():
        assert hashlib.sha256((BASE / name).read_bytes()).hexdigest() == digest, f'Changed frozen file: {name}'
    files = list(frozen)
    checks = {}
    for batch, data in [('startup', 'startup-check-8.json'), ('natural', 'stage2-inputs.json'), ('controls', 'control-remaining-72.json')]:
        d = BASE / 'results' / batch
        if not (d / 'runtime.json').exists():
            continue
        runtime = json.loads((d / 'runtime.json').read_text())
        if not runtime.get('all_attempted'):
            continue
        inputs = read_rows(BASE / 'data' / data)
        predictions = read_rows(d / 'predictions.jsonl')
        raw = read_rows(d / 'raw.jsonl')
        expected = {r['id'] for r in inputs}
        assert len(expected) == len(inputs)
        for records in (predictions, raw):
            assert len(records) == len(expected)
            assert {r['id'] for r in records} == expected
            assert all(r['run_hash'] == runtime['run_hash'] for r in records)
        by_raw = {r['id']: r for r in raw}
        assert all(by_raw[r['id']]['prediction'] == r for r in predictions), 'Receipt/prediction mismatch'
        checks[batch] = {'expected': len(inputs), 'raw_receipts': len(raw), 'predictions': len(predictions),
                         'unique_ids_exact': True, 'raw_prediction_match': True,
                         'valid': sum(r['status'] == 'ok' for r in predictions)}
        files.extend(str(p.relative_to(BASE)) for p in d.iterdir() if p.is_file())
    script = '''import hashlib,json,pathlib
root=pathlib.Path(%r)
names=%r
out={}
for name in names:
 p=root/name;h=hashlib.sha256()
 with p.open('rb') as f:
  for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
 out[name]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
print(json.dumps(out))
''' % (REMOTE, sorted(set(files)))
    result = subprocess.run(['ssh','-S','/tmp/q38-page-a061c91f85c91a0f.sock','-p','31812','-o','BatchMode=yes',
                             'root@223.109.239.11','python3 -'],input=script,text=True,capture_output=True,check=True,timeout=60)
    remote = json.loads(result.stdout)
    for name, proof in remote.items():
        data = (BASE / name).read_bytes()
        assert len(data) == proof['bytes'] and hashlib.sha256(data).hexdigest() == proof['sha256'], f'Backup mismatch: {name}'
    notes = json.loads((BASE / 'review-notes.json').read_text())['notes']
    assert set(notes) == {r['id'] for r in read_rows(BASE / 'data/stage2-inputs.json')}
    proof = {'verified_at_epoch': time.time(), 'batches': checks, 'local_equals_remote': True,
             'frozen_inputs_prompt_contract_unchanged': True, 'all400_have_assistant_notes': True,
             'files': remote, 'scope': 'Byte/identity verification only; not semantic accuracy.'}
    (BASE / 'backup-verification.json').write_text(json.dumps(proof, ensure_ascii=False, indent=2))
    print(json.dumps({k:v for k,v in proof.items() if k!='files'},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
