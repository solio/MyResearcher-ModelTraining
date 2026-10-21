"""CPU-only, additive validation and score audit; never rewrite model receipts."""
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'runner'))
from output_contract import normalize, strict_json


def extract_partial(raw, row, schema):
    """Use the unchanged validator per meaning; unvalidated fields stay unknown.

    The temporary empty checks isolate validation only. They are NEVER exported as
    inferred negative judgments. Only the original candidate's result is retained.
    """
    obj = strict_json(raw)
    if not isinstance(obj, dict) or obj.get('id') != row['id']:
        raise ValueError('Unattributable response ID')
    order = [x['id'] for x in schema['core_candidates']]
    checks = obj.get('checks')
    if not isinstance(checks, dict):
        raise ValueError('Missing checks object')
    template = {'id': row['id'], 'checks': {k: {'state': 'not_expressed'} for k in order},
                'observations': []}
    accepted, rejected, observations, observation_errors = {}, {}, [], []
    for meaning in order:
        if meaning not in checks:
            rejected[meaning] = 'missing, not a negative'; continue
        test = copy.deepcopy(template)
        test['checks'][meaning] = checks[meaning]
        try:
            result = normalize(json.dumps(test, ensure_ascii=False), row, schema)
            if result['needs_scope_review']:
                rejected[meaning] = 'potential conflicting scoped claims; review original'; continue
            accepted[meaning] = result['judgments'][meaning]
        except (ValueError, TypeError, KeyError) as e:
            rejected[meaning] = str(e)
    extras = obj.get('observations')
    if isinstance(extras, list):
        for index, observation in enumerate(extras):
            test = copy.deepcopy(template); test['observations'] = [observation]
            try:
                result = normalize(json.dumps(test, ensure_ascii=False), row, schema)
                observations.append({'original_index': index, **result['observations'][0]})
            except (ValueError, TypeError, KeyError) as e:
                observation_errors.append({'index': index, 'error': str(e)})
    else:
        observation_errors.append({'index': None, 'error': 'missing/non-list observations'})
    return {'checked_meanings': accepted, 'quarantined_meanings': rejected,
            'checked_extra_observations': observations, 'quarantined_extra_observations': observation_errors}


def main():
    inputs = {r['id']: r for r in json.loads((ROOT / 'data/stage2-inputs.json').read_text())['rows']}
    predictions_path = ROOT / 'results/natural/predictions.jsonl'
    if not predictions_path.exists():
        return
    predictions = [json.loads(x) for x in predictions_path.read_text().splitlines() if x]
    raw_path = ROOT / 'results/natural/raw.jsonl'
    raw = {r['id']: r for r in map(json.loads, raw_path.read_text().splitlines())}
    schema = json.loads((ROOT / 'frozen/meaning_schema.json').read_text())
    groups = {x['id']: x['group'] for x in schema['core_candidates']}
    result, score_groups = [], defaultdict(lambda: {'intensity': [], 'support': []})
    topic_only = no_core = 0
    for pred in predictions:
        record = {'id': pred['id'], 'original_status': pred['status'],
                  'original_error': pred.get('error'), 'semantic_correctness_certified': False}
        receipt = raw[pred['id']]
        try:
            content = receipt['response']['choices'][0]['message']['content']
            record.update(extract_partial(content, inputs[pred['id']], schema))
            record['response_sha256'] = hashlib.sha256(content.encode()).hexdigest()
        except (KeyError, IndexError, TypeError, ValueError) as e:
            record['unavailable'] = str(e)
        if pred['status'] == 'ok':
            positives = [k for k, v in pred['judgments'].items() if v['decision'] == 'supported']
            no_core += not positives
            topic_only += bool(positives) and all(groups[k] == 'topic' for k in positives)
            for k, v in pred['judgments'].items():
                for c in v.get('claims', []):
                    if c['state'] != 'expressed':
                        continue
                    for score in ['intensity', 'support']:
                        if c.get(score) is not None:
                            score_groups[groups[k]][score].append(c[score])
        result.append(record)
    partial = [r for r in result if r['original_status'] != 'ok']
    summary = {
        'created_at_epoch': time.time(), 'processed': len(predictions),
        'original_structure_valid': sum(p['status'] == 'ok' for p in predictions),
        'original_errors_preserved': len(partial),
        'partial_rows_with_some_checked_claims': sum(any(v.get('claims') for v in r.get('checked_meanings', {}).values()) for r in partial),
        'partial_positive_claims': sum(sum(c['state'] == 'expressed' for c in v.get('claims', [])) for r in partial for v in r.get('checked_meanings', {}).values()),
        'valid_rows_with_only_core_topic_positives': topic_only,
        'valid_rows_with_no_core_positives': no_core,
        'score_histograms': {g: {k: dict(sorted(Counter(values).items())) for k, values in scores.items()}
                             for g, scores in score_groups.items()},
        'notice': 'Per-meaning checks test original evidence/shape only, not semantic truth. Original row failures stay failures. Missing and quarantined fields are unknown, never zero. Topic-only/core-empty rows may have useful free observations.',
        'checker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for path, text in [(ROOT / 'observation-audit.jsonl', ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in result)),
                       (ROOT / 'observation-audit-summary.json', json.dumps(summary, ensure_ascii=False, indent=2)+'\n')]:
        temp = path.with_suffix('.tmp'); temp.write_text(text); temp.replace(path)
    print(json.dumps({k: v for k, v in summary.items() if k not in ['score_histograms', 'notice', 'checker_sha256']}))


if __name__ == '__main__':
    main()
