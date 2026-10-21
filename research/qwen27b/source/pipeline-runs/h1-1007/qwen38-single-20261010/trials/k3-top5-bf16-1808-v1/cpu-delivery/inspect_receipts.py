"""CPU-only, read-only inspection of append-only model receipts. No semantic repairs."""
import argparse, collections, fcntl, gzip, hashlib, json, os, resource, time
from pathlib import Path


def encoded(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def exact_spans(text, quote):
    if not isinstance(quote, str) or not quote:
        return []
    found, start = [], 0
    while True:
        pos = text.find(quote, start)
        if pos < 0:
            return found
        found.append([pos, pos + len(quote)])
        start = pos + 1


def inspect(receipt, frozen, offset, raw_line):
    rid = receipt.get('id')
    issues, spans = [], []
    def issue(path, code):
        issues.append({'path': path, 'code': code})
    sent = None
    try:
        messages = receipt['request']['messages']
        user_messages = [m for m in messages if m.get('role') == 'user']
        sent = json.loads(user_messages[0]['content']) if len(user_messages) == 1 else None
    except (KeyError, TypeError, ValueError):
        pass
    expected = frozen.get(rid)
    input_matches = expected is not None and sent == expected
    if not input_matches:
        issue('request.messages', 'request_input_mismatch')
    text = sent.get('text', '') if isinstance(sent, dict) else ''
    if not isinstance(text, str):
        text = ''
    prediction = receipt.get('prediction') or {}
    finish = receipt.get('finish_reason')
    if finish != 'stop':
        issue('finish_reason', 'response_not_normal_stop')
    content, output = None, None
    try:
        content = receipt['response']['choices'][0]['message']['content']
        def unique(pairs):
            obj = {}
            for key, val in pairs:
                if key in obj:
                    raise ValueError('duplicate JSON key')
                obj[key] = val
            return obj
        output = json.loads(content, object_pairs_hook=unique)
        if not isinstance(output, dict):
            raise ValueError('not an object')
    except (KeyError, TypeError, ValueError, IndexError):
        issue('model_output', 'unparseable_model_json')
        output = None
    if output is not None:
        if output.get('id') != rid:
            issue('model_output.id', 'response_id_mismatch')
        def inspect_claim(path, obj):
            if not isinstance(obj, dict):
                issue(path, 'claim_not_object')
                return
            for field in ('quote', 'time_text'):
                value = obj.get(field)
                locations = exact_spans(text, value)
                valid = bool(locations) or (field == 'time_text' and value == '')
                spans.append({'path': path + '.' + field, 'value': value,
                              'exact': valid, 'offsets': locations})
                if not valid:
                    issue(path + '.' + field, 'quote_not_exact' if field == 'quote' else 'time_text_not_exact')
        observations = output.get('observations')
        if isinstance(observations, list):
            for i, item in enumerate(observations):
                inspect_claim(f'observations[{i}]', item)
        else:
            issue('observations', 'observations_not_list')
        checks = output.get('checks')
        if isinstance(checks, dict):
            for name, check in checks.items():
                if isinstance(check, dict) and isinstance(check.get('claims'), list):
                    for i, item in enumerate(check['claims']):
                        inspect_claim(f'checks.{name}.claims[{i}]', item)
        else:
            issue('checks', 'checks_not_object')
    return {'id': rid, 'input': sent, 'model_output': output, 'model_content': content,
            'original_contract_status': prediction.get('status'),
            'original_contract_error': prediction.get('error'),
            'finish_reason': finish, 'usage': prediction.get('usage'),
            'request_input_matches_frozen': input_matches,
            'span_checks': spans, 'mechanical_issues': issues,
            'semantic_quality': 'not_assessed',
            'raw_receipt': {'file': 'results/main/raw.jsonl', 'offset_bytes': offset,
                            'length_bytes': len(raw_line), 'sha256': digest(raw_line)}}


def atomic(path, data):
    temp = path.with_name(path.name + '.tmp')
    temp.write_bytes(data)
    temp.replace(path)


def publish(out, records, source_bytes, source_hash, state, expected, started):
    ids = collections.Counter(r.get('id') for r in records)
    defects = collections.Counter(i['code'] for r in records for i in r['mechanical_issues'])
    rows_with_issues = [r for r in records if r['mechanical_issues'] or r['original_contract_status'] != 'ok']
    summary = {'state': state, 'updated_epoch': time.time(), 'expected': expected,
               'responses_inspected': len(records), 'unique_ids': len(ids),
               'duplicate_ids': {str(k): n for k, n in ids.items() if n > 1},
               'original_contract_ok': sum(r['original_contract_status'] == 'ok' for r in records),
               'original_contract_non_ok': sum(r['original_contract_status'] != 'ok' for r in records),
               'non_ok_with_full_model_json_retained': sum(r['original_contract_status'] != 'ok' and r['model_output'] is not None for r in records),
               'request_input_mismatches': sum(not r['request_input_matches_frozen'] for r in records),
               'mechanical_issue_counts': dict(defects),
               'source_complete_prefix_bytes': source_bytes,
               'source_complete_prefix_sha256': source_hash,
               'cpu_seconds': time.process_time(),
               'elapsed_seconds': time.time() - started,
               'semantic_quality': 'not_assessed',
               'notice': 'Unmodified model answers, including format-rejected answers. Exact spans are mechanical checks, not semantic correctness. No scores, labels, or original statuses repaired. Offsets use Unicode code points.'}
    payload = b''.join(encoded(r) + b'\n' for r in records)
    issues = b''.join(encoded({'id': r['id'], 'input': r['input'],
                               'original_contract_error': r['original_contract_error'],
                               'mechanical_issues': r['mechanical_issues']}) + b'\n' for r in rows_with_issues)
    files = {'model-answers.jsonl.gz': gzip.compress(payload, compresslevel=1, mtime=0),
             'mechanical-issues.jsonl': issues, 'summary.json': encoded(summary)}
    # The lock lets the separate backup take a consistent snapshot of this derivative only.
    with (out / 'snapshot.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for name, data in files.items():
            atomic(out / name, data)
        atomic(out / 'manifest.json', encoded({'created_epoch': time.time(),
                  'source_prefix_bytes': source_bytes, 'source_prefix_sha256': source_hash,
                  'files_sha256': {n: digest(data) for n, data in files.items()}}))
    print(json.dumps({'state': state, 'responses': len(records), 'cpu_seconds': round(time.process_time(), 2)}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trial', required=True, type=Path)
    p.add_argument('--deadline-epoch', required=True, type=float)
    p.add_argument('--once', action='store_true')
    p.add_argument('--bounded', action='store_true')
    a = p.parse_args()
    if a.bounded:
        os.nice(15)
        resource.setrlimit(resource.RLIMIT_AS, (384 * 1024**2, 384 * 1024**2))
        resource.setrlimit(resource.RLIMIT_CPU, (120, 125))
    out = a.trial / 'cpu-delivery' / 'output'
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'worker.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('CPU inspector already running')
    frozen = {r['id']: r for r in json.loads((a.trial / 'data/model-inputs.json').read_text())['rows']}
    raw = a.trial / 'results/main/raw.jsonl'
    records, offset, hasher, started = [], 0, hashlib.sha256(), time.time()
    while True:
        if raw.exists():
            if raw.stat().st_size < offset:
                raise RuntimeError('Source shrank; refusing to alter evidence')
            with raw.open('rb') as f:
                f.seek(offset)
                while True:
                    line = f.readline()
                    if not line or not line.endswith(b'\n'):
                        break
                    try:
                        receipt = json.loads(line)
                    except ValueError:
                        raise RuntimeError(f'Malformed complete raw receipt at {offset}; source left untouched')
                    records.append(inspect(receipt, frozen, offset, line))
                    hasher.update(line)
                    offset += len(line)
        state = 'running'
        status_path = a.trial / 'session-status.json'
        if status_path.exists():
            try:
                status = json.loads(status_path.read_text()).get('state')
                if status in ('complete', 'stopped'):
                    state = 'source_' + status
            except ValueError:
                pass
        if time.time() >= a.deadline_epoch:
            state = 'deadline_reached'
        publish(out, records, offset, hasher.hexdigest(), state, len(frozen), started)
        if a.once or state != 'running':
            break
        time.sleep(30)

if __name__ == '__main__':
    main()
