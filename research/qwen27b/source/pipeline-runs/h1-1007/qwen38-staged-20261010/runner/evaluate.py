#!/usr/bin/env python3
"""Evidence-audited per-meaning evaluation; references are INTERNAL, not human gold."""
import argparse
import json
from pathlib import Path
from run_batch import HERE, atomic_json, load_rows, sha


def decision(value):
    if isinstance(value, dict):
        value = value.get('decision', value.get('label', value.get('expected')))
    if value in ('Y', 'supported', True):
        return 'Y'
    if value in ('N', 'not_supported', False):
        return 'N'
    if value in ('?', 'abstain', 'unknown', None):
        return '?'
    raise ValueError(f'Invalid reference decision: {value}')


def reference_map(row, order):
    labels = row.get('judgments', row.get('reference', row.get('expected')))
    if isinstance(labels, dict):
        return {k: decision(labels.get(k)) for k in order}
    labels = row.get('labels', labels)
    if isinstance(labels, str):
        if len(labels) != len(order):
            raise ValueError(f'Wrong reference vector length for {row["id"]}')
        return {k: decision(v) for k, v in zip(order, labels)}
    raise ValueError(f'No reference judgments for {row["id"]}')


def counts():
    return {k: 0 for k in ['TP', 'FP', 'FN', 'TN', 'unknown_supported', 'unknown_total',
                           'abstain', 'invalid_or_missing', 'reference_positive']}


def ratios(c):
    return {**c, 'precision': c['TP'] / (c['TP'] + c['FP']) if c['TP'] + c['FP'] else None,
            'recall': c['TP'] / (c['TP'] + c['FN']) if c['TP'] + c['FN'] else None}


def evaluate(inputs, references, predictions, schema, manifest=None, precision=.95, recall=.90):
    order = [x['id'] for x in schema['core_candidates']]
    groups = {x['id']: x['group'] for x in schema['core_candidates']}
    input_by = {r['id']: r for r in inputs}
    ref_by = {r['id']: r for r in references}
    pred_by = {r['id']: r for r in predictions}
    if set(input_by) != set(ref_by):
        raise ValueError('Reference ID set must exactly match input IDs; no selective evaluation')
    if not set(pred_by).issubset(input_by):
        raise ValueError('Predictions contain unexpected IDs')
    per_class = {k: counts() for k in order}
    cases, critical_failures, integrity_errors = [], [], []
    critical_count = 0
    for ident, row in input_by.items():
        ref = reference_map(ref_by[ident], order)
        pred = pred_by.get(ident)
        valid = bool(pred and pred.get('status') == 'ok' and isinstance(pred.get('judgments'), dict))
        if not valid:
            integrity_errors.append({'id': ident, 'error': (pred or {}).get('error', 'missing_or_invalid_prediction')})
        judgments = pred.get('judgments', {}) if valid else {}
        for key in order:
            y = ref[key]
            item = judgments.get(key, {})
            supported = valid and item.get('decision') == 'supported'
            exact = isinstance(item.get('quote'), str) and bool(item.get('quote')) and item['quote'] in row['text']
            if supported and not exact:
                integrity_errors.append({'id': ident, 'meaning': key, 'error': 'nonexact_support_quote'})
                supported = False
            c = per_class[key]
            if not valid:
                c['invalid_or_missing'] += 1
            if item.get('decision') == 'abstain':
                c['abstain'] += 1
            if y == '?':
                c['unknown_total'] += 1
                if supported:
                    c['unknown_supported'] += 1
                    cases.append({'id': ident, 'meaning': key, 'type': 'unknown_supported', 'prediction': item})
            elif y == 'Y':
                c['reference_positive'] += 1
                c['TP' if supported else 'FN'] += 1
                if not supported:
                    cases.append({'id': ident, 'meaning': key, 'type': 'FN', 'prediction': item})
            else:
                c['FP' if supported else 'TN'] += 1
                if supported:
                    cases.append({'id': ident, 'meaning': key, 'type': 'FP', 'prediction': item})
        critical = ref_by[ident].get('critical', [])
        if not isinstance(critical, list):
            raise ValueError(f'critical must be list for {ident}')
        for condition in critical:
            key = condition.get('id', condition.get('meaning_id'))
            if key not in order:
                raise ValueError(f'Invalid critical predicate {key}')
            expected = decision(condition.get('expected', ref[key]))
            if expected == '?':
                raise ValueError('Critical reference must be clear Y or N')
            critical_count += 1
            item = judgments.get(key, {})
            actual = decision(item.get('decision')) if valid else '?'
            # For explicit denial/other-subject negatives, abstention is not a pass.
            if actual != expected:
                critical_failures.append({'id': ident, 'meaning': key, 'expected': expected,
                                          'actual': actual, 'reason': condition.get('reason'),
                                          'prediction': item})
    totals = counts()
    per_group = {}
    for key, c in per_class.items():
        group = per_group.setdefault(groups[key], counts())
        for metric, value in c.items():
            totals[metric] += value
            group[metric] += value
    total_metrics = ratios(totals)
    group_metrics = {k: ratios(v) for k, v in per_group.items()}
    class_metrics = {k: ratios(v) for k, v in per_class.items()}
    fail_reasons = []
    if (manifest or {}).get('stage', 'stage1') == 'stage1' and len(inputs) != 80:
        fail_reasons.append('Stage1 gate requires all 80 planned inputs, not a smaller smoke sample')
    if integrity_errors:
        fail_reasons.append('Missing, malformed, truncated, or invalid-evidence outputs')
    if total_metrics['precision'] is None or total_metrics['precision'] < precision:
        fail_reasons.append(f'Overall internal-reference precision below {precision}')
    if total_metrics['recall'] is None or total_metrics['recall'] < recall:
        fail_reasons.append(f'Overall internal-reference recall below {recall}')
    if critical_count == 0:
        fail_reasons.append('No explicit critical negation/subject controls supplied')
    if critical_failures:
        fail_reasons.append('Critical simple negation/subject/tense cases failed')
    # Topic success cannot conceal emotion failure. Sparse class metrics remain visible.
    emotion = group_metrics.get('emotion', {})
    if emotion.get('precision') is None or emotion['precision'] < precision:
        fail_reasons.append(f'Emotion precision below {precision} or no evaluated positive predictions')
    if emotion.get('recall') is None or emotion['recall'] < recall:
        fail_reasons.append(f'Emotion recall below {recall} or no reference positives')
    if totals['unknown_supported']:
        fail_reasons.append('Predictions on uncertain references require adjudication before expansion')
    return {'stage': (manifest or {}).get('stage', 'stage1'),
            'config_hash': (manifest or {}).get('config_hash'),
            'run_hash': (manifest or {}).get('run_hash'),
            'reference_sha256': sha(references),
            'reference_status': 'INTERNAL provisional reference; not human gold or proof of corpus accuracy',
            'score_notice': 'intensity/support are retained uncalibrated estimates, not probabilities',
            'input_rows': len(inputs), 'prediction_rows': len(predictions),
            'totals': total_metrics, 'per_group': group_metrics, 'per_class': class_metrics,
            'critical_checked': critical_count, 'critical_failures': critical_failures,
            'integrity_errors': integrity_errors, 'cases': cases,
            'gate': {'passed': not fail_reasons, 'precision_threshold': precision,
                     'recall_threshold': recall, 'fail_reasons': fail_reasons,
                     'notice': 'Engineering gate only; small internal reference does not establish general accuracy'},
            'class_evidence_warning': {k: 'Very sparse positive evidence (<5); no population accuracy claim'
                                       for k, c in per_class.items() if c['reference_positive'] < 5}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--reference', required=True)
    p.add_argument('--run', required=True, help='Output directory of run_batch.py')
    p.add_argument('--schema', default=str(HERE / 'meaning_schema.json'))
    p.add_argument('--out', help='Default <run>/evaluation.json')
    args = p.parse_args()
    run = Path(args.run)
    manifest = json.loads((run / 'manifest.json').read_text())
    inputs = load_rows(args.input)
    if sha(inputs) != manifest['input_sha256']:
        raise ValueError('Evaluation input does not match run manifest')
    schema = json.loads(Path(args.schema).read_text())
    if schema != manifest['semantics']['schema']:
        raise ValueError('Evaluation schema differs from inference schema')
    predictions = load_rows(run / 'predictions.jsonl')
    if any(x.get('run_hash') != manifest['run_hash'] for x in predictions):
        raise ValueError('Foreign predictions')
    report = evaluate(inputs, load_rows(args.reference), predictions, schema, manifest)
    atomic_json(args.out or run / 'evaluation.json', report)
    print(json.dumps({'gate': report['gate'], 'totals': report['totals'],
                      'emotion': report['per_group'].get('emotion'),
                      'critical_failures': len(report['critical_failures'])}, ensure_ascii=False, indent=2))
    return 0 if report['gate']['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
