#!/usr/bin/env python3
"""Stdlib-only staged local-vLLM runner. No references are sent to the server."""
import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import threading
import time
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
STOP = threading.Event()
SUBJECTS = {'self', 'other_retail', 'main_force', 'other', 'text', 'unknown'}
TIMES = {'completed', 'planned', 'current', 'past', 'conditional', 'general', 'unknown'}
INPUT_KEYS = ('id', 'text', 'stock_name', 'stock_code', 'published_at', 'context')


def dumps(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def sha(obj):
    return hashlib.sha256(dumps(obj).encode()).hexdigest()


def load_rows(path):
    raw = Path(path).read_text()
    try:
        obj = json.loads(raw)
        rows = ([obj] if 'id' in obj else obj.get('rows', obj.get('predictions'))) if isinstance(obj, dict) else obj
    except json.JSONDecodeError:
        rows = [json.loads(x) for x in raw.splitlines() if x.strip()]
    if not isinstance(rows, list):
        raise ValueError('Expected JSON list, {rows:[...]}, or JSONL')
    ids = [str(x['id']) for x in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate input IDs')
    return rows


def atomic_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def number(value, nullable=False):
    return (nullable and value is None) or (
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value) and 0 <= value <= 1)


from output_contract import normalize, response_schema


def build_prompt(schema, prompt_path):
    template = Path(prompt_path).read_text()
    if template.count('{{MEANINGS}}') != 1:
        raise ValueError('Prompt must contain one {{MEANINGS}} placeholder')
    meanings = '\n'.join(f"{x['id']}（{x['label']}）：{x['hypothesis']} {x['notes']}"
                         for x in schema['core_candidates'])
    return template.replace('{{MEANINGS}}', meanings)


def request_one(row, args, prompt, schema, run_hash):
    started = time.time()
    safe_input = {k: row[k] for k in INPUT_KEYS if k in row}
    payload = {'model': args.model, 'messages': [
        {'role': 'system', 'content': prompt},
        {'role': 'user', 'content': dumps(safe_input)}],
        'temperature': args.temperature, 'top_p': args.top_p,
        'max_tokens': args.max_tokens, 'stream': False,
        'chat_template_kwargs': {'enable_thinking': args.thinking == 'on'}}
    payload['structured_outputs'] = {'json': response_schema(schema)}
    if args.reasoning_effort != 'omit':
        payload['chat_template_kwargs']['reasoning_effort'] = args.reasoning_effort
    timeout = min(args.timeout, args.deadline_epoch - started) if args.deadline_epoch else args.timeout
    raw = {'id': row['id'], 'run_hash': run_hash, 'started_at_epoch': started,
           'request': payload, 'request_sha256': sha(payload)}
    if STOP.is_set() or timeout <= 1:
        return None
    headers = {'Content-Type': 'application/json'}
    api_key = os.environ.get(args.api_key_env, '')
    if api_key:
        headers['Authorization'] = 'Bearer ' + api_key
    req = urllib.request.Request(args.base_url.rstrip('/') + '/chat/completions',
                                 data=dumps(payload).encode(), headers=headers, method='POST')
    pred = {'id': row['id'], 'status': 'error', 'judgments': None}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            obj = json.loads(response.read())
        raw['response'] = obj
        choices = obj.get('choices', [])
        if len(choices) != 1:
            raise ValueError('Expected exactly one response choice')
        finish = choices[0].get('finish_reason')
        raw['finish_reason'] = finish
        if finish != 'stop':
            raise ValueError(f'Incomplete/unsupported finish_reason={finish}; never treated as no emotion')
        pred = normalize(choices[0].get('message', {}).get('content'), row, schema)
    except urllib.error.HTTPError as exc:
        raw['http_status'] = exc.code
        raw['http_body'] = exc.read(8192).decode(errors='replace')
        pred['error'] = f'HTTP {exc.code}'
    except Exception as exc:
        pred['error'] = f'{type(exc).__name__}: {exc}'
    elapsed = time.time() - started
    raw['elapsed_seconds'] = elapsed
    pred.update({'run_hash': run_hash, 'elapsed_seconds': elapsed,
                 'usage': raw.get('response', {}).get('usage', {}),
                 'finish_reason': raw.get('finish_reason')})
    return raw, pred


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--model', default='Qwen/Qwen3.8-27B')
    parser.add_argument('--model-revision', required=True, help='Pinned resolved model commit SHA; never a moving branch')
    parser.add_argument('--base-url', default='http://127.0.0.1:8000/v1')
    parser.add_argument('--schema', default=str(HERE / 'meaning_schema.json'))
    parser.add_argument('--prompt', default=str(HERE / 'prompt.txt'))
    parser.add_argument('--thinking', choices=['on', 'off'], default='on')
    parser.add_argument('--reasoning-effort', choices=['low', 'medium', 'xhigh', 'omit'], default='medium')
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--concurrency', type=int, choices=[1, 2], default=2)
    parser.add_argument('--smoke-count', type=int, default=3, help='First N inputs must succeed serially')
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--deadline-epoch', type=float, default=0)
    parser.add_argument('--api-key-env', default='LOCAL_LLM_API_KEY')
    parser.add_argument('--stage', choices=['stage1', 'stage2'], default='stage1')
    parser.add_argument('--stage1-evaluation', help='Required for stage2; must explicitly pass')
    args = parser.parse_args()
    if not (0 < args.timeout <= 120) or args.smoke_count < 1 or args.max_tokens < 1:
        parser.error('timeout must be <=120, smoke-count and max-tokens must be positive')
    if not args.deadline_epoch or args.deadline_epoch <= time.time():
        parser.error('--deadline-epoch must be a future epoch; no unbounded paid runs')
    schema = json.loads(Path(args.schema).read_text())
    prompt = build_prompt(schema, args.prompt)
    rows = load_rows(args.input)
    for row in rows:
        if not isinstance(row.get('id'), str) or not isinstance(row.get('text'), str) or not row['text'].strip():
            raise ValueError('Every input must have nonempty string id and text')
    if not rows:
        raise ValueError('No inputs')
    if args.stage == 'stage1' and len(rows) > 80:
        raise ValueError('stage1 cannot exceed 80 inputs')
    if args.stage == 'stage2' and len(rows) > 400:
        raise ValueError('stage2 cannot exceed 400 inputs')
    semantics = {'model': args.model, 'model_revision': args.model_revision,
                 'schema': schema, 'prompt': prompt, 'response_schema': response_schema(schema),
                 'output_contract_sha256': hashlib.sha256((HERE / 'output_contract.py').read_bytes()).hexdigest(),
                 'thinking': args.thinking, 'reasoning_effort': args.reasoning_effort,
                 'max_tokens': args.max_tokens, 'temperature': args.temperature, 'top_p': args.top_p}
    config_hash = sha(semantics)
    if args.stage == 'stage2':
        if not args.stage1_evaluation:
            raise ValueError('stage2 requires --stage1-evaluation')
        gate = json.loads(Path(args.stage1_evaluation).read_text())
        if gate.get('stage') != 'stage1' or gate.get('gate', {}).get('passed') is not True:
            raise ValueError('stage1 did not pass; stage2 is blocked')
        if gate.get('config_hash') != config_hash:
            raise ValueError('stage2 configuration differs from passed stage1')
    run_spec = {'config_hash': config_hash, 'input_sha256': sha(rows),
                'stage': args.stage, 'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    run_hash = sha(run_spec)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / 'manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get('run_hash') != run_hash:
            raise ValueError('Resume refused: input/prompt/model/settings/script hash differs')
        manifest = previous
    else:
        if (out / 'raw.jsonl').exists() or (out / 'predictions.jsonl').exists():
            raise ValueError('Outputs exist without manifest')
        manifest = {**run_spec, 'run_hash': run_hash, 'semantics': semantics,
                    'created_at_epoch': time.time(), 'expected_rows': len(rows),
                    'base_url': args.base_url,
                    'notice': 'Internal experiment; scores uncalibrated; no automatic training or memory update'}
        atomic_json(manifest_path, manifest)
        (out / 'resolved-prompt.txt').write_text(prompt)
    records = {}
    pred_path = out / 'predictions.jsonl'
    raw_path = out / 'raw.jsonl'
    # Raw is the append-first receipt log. Recover a completed receipt after a crash.
    if raw_path.exists():
        for line in raw_path.read_text().splitlines():
            item = json.loads(line)
            if item.get('run_hash') != run_hash:
                raise ValueError('Foreign run in raw receipt log')
            if isinstance(item.get('prediction'), dict):
                if item['id'] in records:
                    raise ValueError('Duplicate receipt; manual audit required')
                records[item['id']] = item['prediction']
    if pred_path.exists():
        disk_preds = load_rows(pred_path)
        for item in disk_preds:
            if item.get('run_hash') != run_hash or item['id'] not in records or item != records[item['id']]:
                raise ValueError('Prediction log does not match raw receipts')
    # Reconstruct materialized predictions atomically from receipts.
    temp_pred = pred_path.with_suffix('.jsonl.tmp')
    temp_pred.write_text(''.join(dumps(records[r['id']]) + '\n' for r in rows if r['id'] in records))
    temp_pred.replace(pred_path)
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    signal.signal(signal.SIGINT, lambda *_: STOP.set())
    started = time.time()
    stop_reason = 'complete'

    def may_start():
        return not STOP.is_set() and time.time() < args.deadline_epoch - args.timeout

    def save(result):
        if result is None:
            return
        raw, pred = result
        raw['prediction'] = pred
        for path, item in [(raw_path, raw), (pred_path, pred)]:
            with path.open('a') as handle:
                handle.write(dumps(item) + '\n')
                handle.flush()
                os.fsync(handle.fileno())
        records[pred['id']] = pred
        print(dumps({'id': pred['id'], 'status': pred['status'], 'completed': len(records),
                     'elapsed_seconds': round(pred['elapsed_seconds'], 2), 'error': pred.get('error')}), flush=True)

    def format_error(pred):
        return (pred.get('status') == 'error' and pred.get('finish_reason') == 'stop'
                and pred.get('error', '').startswith(('ValueError:', 'JSONDecodeError:')))

    for row in rows[:args.smoke_count]:
        if row['id'] not in records:
            if not may_start():
                stop_reason = 'deadline_or_signal'
                break
            save(request_one(row, args, prompt, schema, run_hash))
        if records.get(row['id'], {}).get('status') != 'ok' and not format_error(records.get(row['id'], {})):
            stop_reason = 'smoke_failed'
            break
    else:
        pending_rows = iter(r for r in rows if r['id'] not in records)
        with cf.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = set()
            exhausted = False
            while futures or not exhausted:
                while not exhausted and len(futures) < args.concurrency and may_start():
                    row = next(pending_rows, None)
                    if row is None:
                        exhausted = True
                        break
                    futures.add(pool.submit(request_one, row, args, prompt, schema, run_hash))
                if not may_start():
                    if stop_reason != 'response_error':
                        stop_reason = 'deadline_or_signal'
                    exhausted = True
                if not futures:
                    break
                done, futures = cf.wait(futures, timeout=1, return_when=cf.FIRST_COMPLETED)
                for future in done:
                    result = future.result()
                    save(result)
                    if result and result[1]['status'] != 'ok' and not format_error(result[1]):
                        # Individual malformed judgments remain errors; stop only for request/service failures.
                        STOP.set()
                        stop_reason = 'response_error'
                        exhausted = True
    tokens = {}
    for record in records.values():
        for key, val in record.get('usage', {}).items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                tokens[key] = tokens.get(key, 0) + val
    errors = [{'id': x['id'], 'error': x.get('error')} for x in records.values() if x['status'] != 'ok']
    runtime = {'run_hash': run_hash, 'config_hash': config_hash, 'stage': args.stage,
               'expected_rows': len(rows), 'completed_rows': len(records),
               'successful_rows': len(records) - len(errors), 'errors': errors,
               'pending_ids': [r['id'] for r in rows if r['id'] not in records],
               'all_attempted': len(records) == len(rows),
               'all_complete': len(records) == len(rows) and not errors,
               'stop_reason': stop_reason, 'session_started_epoch': started,
               'session_wall_seconds': time.time() - started, 'deadline_epoch': args.deadline_epoch,
               'usage_totals_all_receipts': tokens,
               'sum_request_seconds_all_receipts': sum(x['elapsed_seconds'] for x in records.values()),
               'billing_notice': 'This runner stops new requests, NOT instance billing. Provider stop must be handled separately.'}
    atomic_json(out / 'runtime.json', runtime)
    print(dumps(runtime), flush=True)
    return 0 if runtime['all_complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
