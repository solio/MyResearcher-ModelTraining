import copy, gzip, hashlib, importlib.util, json, pathlib, subprocess, sys, tempfile, time
D = pathlib.Path(__file__).resolve().parent
T = D.parent
spec = importlib.util.spec_from_file_location('inspector', D/'inspect_receipts.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
inputs = json.loads((T/'data/model-inputs.json').read_text())
frozen = {r['id']: r for r in inputs['rows']}
rows = [json.loads(l) for l in (T/'results/main/raw.jsonl').read_text().splitlines()]
ok = next(r for r in rows if r['prediction']['status'] == 'ok')
bad = next(r for r in rows if r['prediction']['status'] != 'ok')
line = m.encoded(bad) + b'\n'
r = m.inspect(bad, frozen, 0, line)
assert r['original_contract_status'] == 'error'
assert r['model_output'] == json.loads(bad['response']['choices'][0]['message']['content'])
assert any(i['code'] == 'quote_not_exact' for i in r['mechanical_issues'])
assert r['request_input_matches_frozen']
assert r['semantic_quality'] == 'not_assessed'
tampered = copy.deepcopy(ok)
sent = json.loads(tampered['request']['messages'][1]['content']); sent['text'] += ' changed'
tampered['request']['messages'][1]['content'] = json.dumps(sent)
assert not m.inspect(tampered, frozen, 0, m.encoded(tampered))['request_input_matches_frozen']
assert m.exact_spans('哈哈哈', '哈哈') == [[0, 2], [1, 3]]
with tempfile.TemporaryDirectory() as td:
    trial = pathlib.Path(td)
    (trial/'data').mkdir(); (trial/'results/main').mkdir(parents=True)
    (trial/'data/model-inputs.json').write_text(json.dumps(inputs))
    source = m.encoded(ok) + b'\n' + line + b'{"incomplete"'
    (trial/'results/main/raw.jsonl').write_bytes(source)
    subprocess.run([sys.executable, str(D/'inspect_receipts.py'), '--trial', td,
                    '--deadline-epoch', str(time.time()+60), '--once'], check=True)
    out = trial/'cpu-delivery/output'
    parsed = [json.loads(l) for l in gzip.decompress((out/'model-answers.jsonl.gz').read_bytes()).splitlines()]
    assert len(parsed) == 2
    assert parsed[1]['model_output'] == r['model_output']
    assert (trial/'results/main/raw.jsonl').read_bytes() == source
    manifest = json.loads((out/'manifest.json').read_text())
    assert all(hashlib.sha256((out/n).read_bytes()).hexdigest() == h for n,h in manifest['files_sha256'].items())
    prefix = source[:manifest['source_prefix_bytes']]
    assert hashlib.sha256(prefix).hexdigest() == manifest['source_prefix_sha256']
    assert manifest['source_prefix_bytes'] == len(m.encoded(ok)+b'\n'+line)
    # Restart reconstructs the same two receipts; does not duplicate or alter raw evidence.
    subprocess.run([sys.executable, str(D/'inspect_receipts.py'), '--trial', td,
                    '--deadline-epoch', str(time.time()+60), '--once'], check=True)
    assert len(gzip.decompress((out/'model-answers.jsonl.gz').read_bytes()).splitlines()) == 2
print('PASS: rejected answer preserved, every quote checked, mismatch detected, partial append ignored, source unchanged, compressed hashes verified, restart idempotent')
