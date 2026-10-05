#!/usr/bin/env python3
"""把 84 条人工仲裁结果转成机器可用真值集。

verdict → 真值标签映射:
- k3:     K3 标签为真值
- student: 学生标签为真值
- both:   两套标签均可(口径差异), truth_multi=True
- none:   从用户备注提取;无法提取的标 NEEDS_* 状态,不进训练,进"标签体系修复"清单
用户备注中可提取的真值显式编码,每条带 provenance。
"""
import json

BASE = '/Users/mac/Documents/trae_projects/MyResearcher/pipeline-runs/contrarian-bottleneck-20261005'
d = json.load(open(f'{BASE}/arbitration-result.json'))

# 用户备注中提取的真值(人工编码,逐条对应 9 条备注)
NOTE_TRUTH = {
    '大盘极限施压，鼎龙必须顶住压力，要顶龙！':
        {'emotion': 'HOPE', 'stance': 'MIXED', 'note': '用户:多空混合;打气中带恐惧背景'},
    '虽然涨的不多，但是从今天起也算正式的迈入了百元俱乐部，买一手都要过万了':
        {'emotion': 'CALM', 'stance': 'NEUTRAL', 'note': '用户:本意不是看多或看空'},
    '猛的不像话':
        {'emotion': 'EXCITEMENT', 'stance': 'BULL', 'note': '用户:应该是涨的(需行情上下文确认)'},
    '58的本': {'needs': 'CONTEXT_QUOTE', 'note': '用户:应查当日行情'},
    '来吧争取尾盘在出最新低点我尾盘我进场了':
        {'needs': 'NEW_LABEL_CONDITIONAL', 'note': '用户:标签数量不足够(盼跌+进场=条件式)'},
    '这是洗盘还是出货？': {'needs': 'CONTEXT_QUOTE', 'note': '用户:应参考当日行情'},
    '我是电力，沪电': {'needs': 'RELABEL_UNRESOLVED', 'note': '用户:情绪都不对,胡言乱语'},
    '主力小乖乖，下面缺口补一下嘛': {'needs': 'NEW_LABEL_CONDITIONAL', 'note': '用户:标签不够(揶揄+盼补缺)'},
    '我，装起来啦': {'needs': 'CONTEXT_IMAGE', 'note': '用户:贴了买卖图,做了T,文本外信息'},
}

out = []
stats = {'k3': 0, 'student': 0, 'both': 0, 'from_note': 0, 'needs': 0, 'unjudged': 0}
for v in d['verdicts']:
    rec = {
        'sample_id': v['sample_id'], 'stock_code': v['stock_code'], 'stock_name': v['stock_name'],
        'entry_date': v['entry_date'], 'text': v['text'], 'layer': v['layer'], 'hard': v['hard'],
        'student_label': {'emotion': v['student']['emotion'], 'stance': v['student']['stance']},
        'k3_label': {'emotion': v['k3']['emotion'], 'stance': v['k3']['stance']},
        'k3_sarcasm': v['k3']['sarcasm'],
        'verdict': v['verdict'], 'user_note': v['note'] or None,
    }
    if v['verdict'] == 'k3':
        rec['truth'] = {'emotion': v['k3']['emotion'], 'stance': v['k3']['stance']}
        rec['truth_source'] = 'k3_confirmed'
        stats['k3'] += 1
    elif v['verdict'] == 'student':
        rec['truth'] = {'emotion': v['student']['emotion'], 'stance': v['student']['stance']}
        rec['truth_source'] = 'student_confirmed'
        stats['student'] += 1
    elif v['verdict'] == 'both':
        rec['truth'] = {'emotion': v['k3']['emotion'], 'stance': v['k3']['stance']}
        rec['truth_alt'] = {'emotion': v['student']['emotion'], 'stance': v['student']['stance']}
        rec['truth_source'] = 'both_valid'
        stats['both'] += 1
    elif v['verdict'] == 'none':
        t = NOTE_TRUTH.get(v['text'])
        if t and 'needs' not in t:
            rec['truth'] = {'emotion': t['emotion'], 'stance': t['stance']}
            rec['truth_source'] = 'user_note'
            stats['from_note'] += 1
        else:
            rec['needs'] = (t or {}).get('needs', 'RELABEL_UNRESOLVED')
            rec['truth_source'] = 'unresolved'
            stats['needs'] += 1
    else:
        rec['truth_source'] = 'unjudged'
        stats['unjudged'] += 1
    out.append(rec)

usable = [r for r in out if 'truth' in r]
needs = [r for r in out if 'needs' in r]

with open(f'{BASE}/ground-truth-v1.jsonl', 'w') as f:
    for r in usable:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
with open(f'{BASE}/label-system-gaps.json', 'w') as f:
    json.dump({'description': '仲裁暴露的标签体系缺口清单(不全是模型错误,是标签装不下)',
               'items': needs}, f, ensure_ascii=False, indent=1)

print(f'真值集: {len(usable)} 条 → ground-truth-v1.jsonl')
print(f'  来源分布: K3确认 {stats["k3"]}, 学生确认 {stats["student"]}, 双有效 {stats["both"]}, 用户备注提取 {stats["from_note"]}')
print(f'标签体系缺口: {len(needs)} 条 → label-system-gaps.json')
for r in needs:
    print(f"  [{r['needs']:<24}] {r['text'][:40]}")
print(f'未判: {stats["unjudged"]} 条')
