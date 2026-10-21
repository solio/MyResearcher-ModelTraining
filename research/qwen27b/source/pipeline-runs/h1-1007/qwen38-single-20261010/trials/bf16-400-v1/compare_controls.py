#!/usr/bin/env python3
"""Compare frozen prior INT4 output with BF16 on all the same 80 inputs.

The frozen assistant reference is provisional, never human gold. Coverage and
semantic comparisons use distinct denominators. No output/reference is changed.
"""
import collections
import hashlib
import html
import json
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
OLD = BASE.parent / 'complete-extraction-v1'
CHANGE_REVIEW = {
    ('S1C024', 'stance_bullish'): '边界解释：上涨机会带有正向背景，但是否等于作者独立的价格判断有争议。两版都保留了恐惧/FOMO和买入意向，不能仅按参照N判定BF16整句变差。',
    ('S1C030', 'emo_fear'): '少做一步心理推断：BF16保留盼止跌，没有仅凭求求你再加恐惧，更符合本轮明确表达要求。',
    ('S1R006', 'topic_price'): '栏目差异：两版额外观察都已保留期待主力拉升，BF16另外填价格栏，不是从完全没理解变成理解。',
    ('S1R015', 'stance_bearish'): '边界解释：流动性不足和可操作性低的核心信息两版都有，是否再归看空需要定义边界，不能当已证明的错误。',
    ('S1R017', 'emo_disappointment'): '新增推断：买过与后悔已明确，原文没有独立说明预期落差；再补失望的证据较弱。',
    ('S1R017', 'stance_bearish'): '新增推断：懊恼过去买入不自动等于当前价格/价值判断。应保留原有明确含义，同时区分附加推论。',
    ('S1R022', 'topic_price'): '栏目差异：原版已在额外观察保留低换手率，BF16又归入价格栏。换手率信息没有在原版消失。',
    ('S1R023', 'wish_stop_decline'): '信息仍在：BF16额外观察完整保留希望既不涨也不跌。固定希望止跌栏漏填会影响只读标签的下游，但不能说原文愿望没被理解。',
    ('S1R030', 'topic_price'): '对象仍需边界：270很可能是价格，但简短原文没有单位，也不能排除简称。与旧参照一致不代表外部已经证实。',
    ('S1R038', 'emo_disappointment'): '边界解释：两版都读出基本面与股价矛盾的困惑，BF16额外补失望；是否达到明确失望需要复核。',
    ('S1R047', 'stance_bullish'): '新增推断：AI建议和作者怕踏空都有，但不能仅凭别人建议和害怕错过自动推出作者独立看多判断。',
}


def read(path):
    if path.suffix == '.jsonl':
        return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    obj = json.loads(path.read_text())
    return obj.get('rows', obj) if isinstance(obj, dict) else obj


def indexed(rows):
    result = {r['id']: r for r in rows}
    assert len(rows) == len(result), 'Duplicate IDs'
    return result


def ref_decision(item):
    x = item.get('decision') if isinstance(item, dict) else item
    return 'Y' if x in ('supported', 'Y', True) else 'N' if x in ('not_supported', 'N', False) else '?'


def pred_decision(row, key):
    if row.get('status') != 'ok':
        return 'invalid'
    x = row['judgments'][key]['decision']
    return ref_decision(x)


def metrics(ids, preds, refs, keys):
    c = collections.Counter(TP=0, FP=0, FN=0, TN=0, uncertain_reference=0,
                            uncertain_reference_positive=0, abstain=0)
    for ident in ids:
        for key in keys:
            y = ref_decision(refs[ident]['judgments'][key])
            p = pred_decision(preds[ident], key)
            assert p != 'invalid'
            if p == '?':
                c['abstain'] += 1
            if y == '?':
                c['uncertain_reference'] += 1
                c['uncertain_reference_positive'] += p == 'Y'
            elif y == 'Y':
                c['TP' if p == 'Y' else 'FN'] += 1
            else:
                c['FP' if p == 'Y' else 'TN'] += 1
    return {**c, 'precision': c['TP'] / (c['TP'] + c['FP']) if c['TP'] + c['FP'] else None,
            'recall': c['TP'] / (c['TP'] + c['FN']) if c['TP'] + c['FN'] else None}


def main():
    inputs = indexed(read(OLD / 'data/stage1-inputs.json'))
    refs = indexed(read(OLD / 'data/stage1-reference.json'))
    old = indexed(read(OLD / 'results/semantic/predictions.jsonl'))
    new_rows = read(BASE / 'results/startup/predictions.jsonl') + read(BASE / 'results/controls/predictions.jsonl')
    new = indexed(new_rows)
    assert set(inputs) == set(refs) == set(old) == set(new), 'Wait for all 80 attempts'
    bf_inputs = indexed(read(BASE / 'data/startup-check-8.json') + read(BASE / 'data/control-remaining-72.json'))
    assert set(bf_inputs) == set(inputs)
    for ident in inputs:
        assert inputs[ident] == bf_inputs[ident], f'Input changed: {ident}'
        assert inputs[ident]['text'] == refs[ident]['text']
    manifests = [read(OLD / 'results/semantic/manifest.json'),
                 read(BASE / 'results/startup/manifest.json'),
                 read(BASE / 'results/controls/manifest.json')]
    for m, rows in zip(manifests, [list(old.values()), read(BASE / 'results/startup/predictions.jsonl'), read(BASE / 'results/controls/predictions.jsonl')]):
        assert all(r['run_hash'] == m['run_hash'] for r in rows), 'Foreign run output'
    semantics = manifests[0]['semantics']
    for m in manifests[1:]:
        for key, value in semantics.items():
            if key not in ('model', 'model_revision'):
                assert m['semantics'][key] == value, f'Changed semantics: {key}'
    schema = semantics['schema']['core_candidates']
    keys = [r['id'] for r in schema]
    labels = {r['id']: r['label'] for r in schema}
    shared = sorted(i for i in inputs if old[i]['status'] == new[i]['status'] == 'ok')
    group_ids = {'all': sorted(inputs), 'natural_50': sorted(i for i in inputs if i.startswith('S1R')),
                 'constructed_30': sorted(i for i in inputs if i.startswith('S1C'))}
    report = {'created_at_epoch': time.time(), 'reference_status': 'Frozen assistant provisional reference; NOT human gold',
              'comparison_notice': 'Same input, prompt, schema and decoding settings. Precision/deployment comparison: INT4 versus BF16 also changes tensor parallelism and concurrency. Not an isolated quantization causal test.',
              'reference_file_sha256': hashlib.sha256((OLD / 'data/stage1-reference.json').read_bytes()).hexdigest(),
              'all_attempted': len(inputs), 'matched_valid': len(shared), 'coverage': {}, 'groups': {}, 'per_meaning': {}, 'changes': []}
    for version, rows in [('INT4', old), ('BF16', new)]:
        report['coverage'][version] = {'valid': sum(r['status'] == 'ok' for r in rows.values()),
                                       'invalid': [{'id': i, 'error': r.get('error')} for i, r in rows.items() if r['status'] != 'ok']}
    for group, ids in group_ids.items():
        same = sorted(set(ids) & set(shared))
        report['groups'][group] = {'planned': len(ids), 'matched_valid': len(same),
                                 **{name: metrics(same, rows, refs, keys) for name, rows in [('INT4', old), ('BF16', new)]}}
    for key in keys:
        report['per_meaning'][key] = {'label': labels[key], **{name: metrics(shared, rows, refs, [key]) for name, rows in [('INT4', old), ('BF16', new)]}}
    for ident in shared:
        for key in keys:
            a, b, y = pred_decision(old[ident], key), pred_decision(new[ident], key), ref_decision(refs[ident]['judgments'][key])
            if a != b:
                report['changes'].append({'id': ident, 'text': inputs[ident]['text'], 'meaning': key, 'label': labels[key],
                    'reference': y, 'INT4': old[ident]['judgments'][key], 'BF16': new[ident]['judgments'][key],
                    'INT4_observations': old[ident].get('observations', []), 'BF16_observations': new[ident].get('observations', []),
                    'assistant_review': CHANGE_REVIEW.get((ident, key), '尚未做单独变更复核'),
                    'change': 'toward_reference' if y != '?' and b == y else 'away_from_reference' if y != '?' and a == y else 'unresolved'})
    report['change_counts'] = dict(collections.Counter(x['change'] for x in report['changes']))
    (BASE / 'control-comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    esc = html.escape
    def pct(x):
        return '无分母' if x is None else f'{x:.1%}'
    def metric_row(title, d):
        a, b = d['INT4'], d['BF16']
        return f'<tr><td>{esc(title)}</td><td>{a["TP"]}/{a["TP"]+a["FP"]} · {pct(a["precision"])}</td><td>{b["TP"]}/{b["TP"]+b["FP"]} · {pct(b["precision"])}</td><td>{a["TP"]}/{a["TP"]+a["FN"]} · {pct(a["recall"])}</td><td>{b["TP"]}/{b["TP"]+b["FN"]} · {pct(b["recall"])}</td></tr>'
    tables = '<table><thead><tr><th>相同且结构有效的样本</th><th>INT4 判断符合参照</th><th>BF16 判断符合参照</th><th>INT4 参照含义被找回</th><th>BF16 参照含义被找回</th></tr></thead><tbody>'
    titles = {'all': '全体', 'natural_50': '真实帖子', 'constructed_30': '人工构造边界句'}
    for key, d in report['groups'].items():
        tables += metric_row(f'{titles[key]}：{d["matched_valid"]}/{d["planned"]} 条', d)
    tables += '</tbody></table><h2>按含义看，保留分子与分母</h2><table><tbody>'
    for d in report['per_meaning'].values():
        tables += metric_row(d['label'], d)
    tables += '</tbody></table>'
    changes = ''
    for row in report['changes']:
        changes += '<details><summary>' + esc(f'{row["id"]} · {row["label"]} · {row["text"]}') + '</summary>'
        changes += '<p>原内部参照：' + esc(row['reference']) + '；参照也可能需要修订，本轮未改。</p>'
        changes += '<p><strong>连同额外观察后的复核：</strong>' + esc(row['assistant_review']) + '</p>'
        for name in ['INT4', 'BF16']:
            changes += f'<h3>{name}</h3><pre>' + esc(json.dumps(row[name], ensure_ascii=False, indent=2)) + '</pre>'
            changes += '<p>该版额外观察：</p><pre>' + esc(json.dumps(row[name+'_observations'], ensure_ascii=False, indent=2)) + '</pre>'
        changes += '</details>'
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>同一80条 · INT4与BF16</title><link rel="stylesheet" href="control.css"><style>main{max-width:1400px}table{width:100%;border-collapse:collapse}td,th{padding:12px;border-bottom:1px solid #ddd;text-align:left}section{overflow-x:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{padding:16px;border-bottom:1px solid #ddd}summary{cursor:pointer;line-height:1.7}h1{font-size:30px}</style><main><h1>同一80条：INT4 与 BF16</h1>'
    page += '<p><a href="progress.html">运行进度</a> · <a href="bf16-review.html">400条新帖原文与理由</a></p><section class="card">'
    page += f'<p>两版各尝试80条。原版结构有效 {report["coverage"]["INT4"]["valid"]} 条，BF16 有效 {report["coverage"]["BF16"]["valid"]} 条；下面语义对照只使用双方都结构有效的 {len(shared)} 条。结构失败单独列出，没有当作正确或情绪中性。</p>'
    page += '<p><strong>这是对冻结内部参照的一致性检查，不是人工核验准确率。</strong>相同帖子、提示词、定义和解码设置；BF16 同时换了双卡部署和并发，因此不能把变化全部归因于数值精度。</p>'
    page += '<p><strong>先读这个区别：</strong>标签栏的变化不等于整句理解的变化。例如“别涨也别跌”在BF16的“额外观察”中仍被识别为希望平稳。下方每个变化都同时展示额外观察与复核意见；不要把与参照更一致的2项、较不一致的9项直接当作业务上的2胜9负。</p>'
    page += tables + '<p>稀少类别的百分比容易跳动，应连同分子、分母和原文看。参照标为未知的格子不进入上述精确率/召回率，原始计数在JSON中。标签判断相同也不证明主体、对象和理由全部正确。</p></section><section class="card"><h2>结构失败</h2><pre>'
    page += esc(json.dumps(report['coverage'], ensure_ascii=False, indent=2)) + '</pre></section><section class="card"><h2>发生变化的判断，逐条核对</h2>' + changes + '</section><p><a href="trials/bf16-400-v1/control-comparison.json">完整指标与来源</a></p></main></html>'
    (ROOT / 'bf16-comparison.html').write_text(page)
    print(json.dumps({k: report[k] for k in ['all_attempted', 'matched_valid', 'coverage', 'groups', 'change_counts']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
