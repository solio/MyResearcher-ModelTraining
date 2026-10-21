#!/usr/bin/env python3
"""Describe emitted scores without claiming they are calibrated probabilities.

Threshold sweeps are diagnostic only. They do not filter stored predictions or
define a production sentiment score. Flagged cases were read with their reasons.
"""
import collections
import json
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent


def main():
    rows = [json.loads(s) for s in (BASE / 'results/natural/predictions.jsonl').read_text().splitlines() if s.strip()]
    schema = json.loads((BASE / 'frozen/meaning_schema.json').read_text())
    groups = {x['id']: x['group'] for x in schema['core_candidates']}
    per_group = collections.defaultdict(list)
    for row in rows:
        if row['status'] != 'ok':
            continue
        for key, check in row['checks'].items():
            for claim in check.get('claims', []):
                if claim['state'] == 'expressed':
                    per_group[groups[key]].append({'id': row['id'], 'meaning': key, **claim})
    out = {'created_at_epoch': time.time(), 'processed_rows': len(rows),
           'scope': 'Expressed core claims on original structure-valid rows. Other-person claims may also be present; not an author-sentiment aggregate.',
           'notice': 'Thresholds only measure how much output would remain. No precision, recall or calibrated correctness is implied. No actual filtering is applied.',
           'groups': {}, 'reviewed_counterexamples': []}
    for group, claims in per_group.items():
        supports = [c['support'] for c in claims if isinstance(c.get('support'), (int, float))]
        intensities = [c['intensity'] for c in claims if isinstance(c.get('intensity'), (int, float))]
        out['groups'][group] = {'expressed_claims': len(claims),
            'intensity_histogram': dict(sorted(collections.Counter(intensities).items())),
            'support_histogram': dict(sorted(collections.Counter(supports).items())),
            'missing_support': len(claims) - len(supports),
            'support_threshold_sweep': [{
                'threshold': t, 'retained_claims': sum(v >= t for v in supports),
                'denominator_all_expressed_claims': len(claims)} for t in [.7, .8, .9, .95, 1.0]]}
    examples = [
        ('S2N245', 'action_buy_completed', '买电力没有说明成交；原输出却判已经买入。'),
        ('S2N258', 'stance_bullish', '标签看多，理由说不会涨；引用漏掉双重否定前半句。'),
        ('S2N278', 'topic_price', '理由杜撰4元涨到3元；原短句未提供这些价格关系。'),
        ('S2N288', 'wish_stop_decline', '213左右止跌未明确表达愿望；模型理由直接从止跌推出希望。'),
        ('S2N328', 'emo_excitement', '熟悉感觉后紧接板块涨个股砸，截前半句判兴奋，忽略整体反讽或无奈。'),
        ('S2N362', 'wish_stop_decline', '不能止跌为条件判断，模型追加了希望止跌。'),
        ('S2N040', 'action_t_unspecified', '未明确说先卖后买，模型却推断完成做T；该整行也有结构异常。'),
    ]
    by_id = {r['id']: r for r in rows}
    for ident, key, concern in examples:
        row = by_id.get(ident)
        if not row:
            continue
        out['reviewed_counterexamples'].append({'id': ident, 'meaning': key, 'concern': concern,
            'original_status': row['status'], 'original_check': row.get('checks', {}).get(key),
            'notice': 'Assistant review, not human gold; evidence preserved for disagreement.'})
    (BASE / 'score-sensitivity.json').write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps({'processed': len(rows), 'groups': {k: v['expressed_claims'] for k, v in out['groups'].items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
