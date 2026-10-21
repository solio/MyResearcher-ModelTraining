"""Compare actual saved predictions on matched inputs, keeping the original reference."""
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TRIAL = ROOT / 'trials/complete-extraction-v1'
sys.path.insert(0, str(TRIAL / 'runner'))
from evaluate import evaluate, reference_map
from run_batch import load_rows


def main():
    old = json.loads((ROOT / 'results/quality-review.json').read_text())
    inputs = load_rows(TRIAL / 'data/stage1-inputs.json')
    refs = load_rows(TRIAL / 'data/stage1-reference.json')
    predictions = load_rows(TRIAL / 'results/semantic/predictions.jsonl')
    schema = json.loads((TRIAL / 'runner/meaning_schema.json').read_text())
    labels = {x['id']: x['label'] for x in schema['core_candidates']}
    old_by = {x['input']['id']: x for x in old['rows']}
    new_by = {x['id']: x for x in predictions}
    ref_by = {x['id']: reference_map(x, list(labels)) for x in refs}
    assert len(inputs) == 80 and set(old_by) == {x['id'] for x in inputs}
    assert all(old_by[x['id']]['input']['text'] == x['text'] for x in inputs)
    common = [x['id'] for x in inputs if old_by[x['id']]['prediction']['status'] == 'ok'
              and new_by.get(x['id'], {}).get('status') == 'ok']
    fixed, regressed, still_different = [], [], []
    for ident in common:
        for key, reference in ref_by[ident].items():
            if reference == '?':
                continue
            before = old_by[ident]['prediction']['judgments'][key]['decision'] == 'supported'
            after = new_by[ident]['judgments'][key]['decision'] == 'supported'
            expected = reference == 'Y'
            detail = {'id': ident, 'meaning': key, 'label': labels[key],
                      'old_supported': before, 'new_supported': after, 'reference': reference}
            if before != expected and after == expected:
                fixed.append(detail)
            elif before == expected and after != expected:
                regressed.append(detail)
            elif before != expected and after != expected:
                still_different.append(detail)
    def compare(ids):
        subset = [x for x in inputs if x['id'] in ids]
        reference = [x for x in refs if x['id'] in ids]
        a = evaluate(subset, reference, [old_by[i]['prediction'] for i in ids], schema)
        b = evaluate(subset, reference, [new_by[i] for i in ids], schema)
        return {'rows': len(ids), 'old': {k:a[k] for k in ['totals','per_group']},
                'new': {k:b[k] for k in ['totals','per_group']}}
    raw_path = TRIAL / 'results/semantic/raw.jsonl'
    raw = [json.loads(l) for l in raw_path.read_text().splitlines() if l]
    times = [x['elapsed_seconds'] for x in predictions]
    finish = max(x['started_at_epoch']+x['elapsed_seconds'] for x in raw)
    started = min(x['started_at_epoch'] for x in raw)
    full_new = evaluate(inputs, refs, predictions, schema)
    summary = {
        'created_at_epoch': time.time(), 'attempted': len(predictions), 'expected': 80,
        'complete': len(predictions) == 80, 'reference_status': 'unchanged assistant provisional reference, not human gold',
        'comparison_limit': 'historical original used mixed output modes; current changes combine structure and prompt',
        'coverage': {'old_valid': sum(x['prediction']['status']=='ok' for x in old['rows']),
                     'new_valid': sum(x['status']=='ok' for x in predictions), 'matched_valid': len(common)},
        'common': compare(common), 'real_common': compare([i for i in common if i.startswith('S1R')]),
        'synthetic_common': compare([i for i in common if i.startswith('S1C')]),
        'closer_to_reference': fixed, 'further_from_reference': regressed, 'still_different': still_different,
        'new_output_errors': [{'id': x['id'], 'error': x.get('error')} for x in predictions if x['status']!='ok'],
        'recovered_old_output_errors': [i for i in new_by if new_by[i]['status']=='ok' and old_by[i]['prediction']['status']!='ok'],
        'new_evaluation': full_new,
        'timing': {'first_request_epoch': started, 'last_result_epoch': finish, 'inference_wall_seconds': finish-started,
                   'request_mean_seconds': statistics.mean(times), 'request_median_seconds': statistics.median(times),
                   'completion_tokens': sum(x.get('usage',{}).get('completion_tokens',0) for x in predictions)},
    }
    path = ROOT / 'results/semantic-comparison.json'
    temp = path.with_suffix('.tmp');temp.write_text(json.dumps(summary,ensure_ascii=False,indent=2));temp.replace(path)
    print(json.dumps({k:summary[k] for k in ['complete','attempted','coverage','timing','new_output_errors']},ensure_ascii=False))
    print('Matched meaning changes toward/away from provisional reference:',len(fixed),len(regressed))


if __name__ == '__main__':
    main()
