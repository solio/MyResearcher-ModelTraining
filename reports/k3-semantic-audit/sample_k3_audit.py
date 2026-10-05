#!/usr/bin/env python3
"""K3 分歧审计三层抽样(fixed-seed=42)。

层1 极值日恐惧层(60): HIGH散户组 rolling_extreme 信号日的帖子
层2 学生高置信层(30): emotion_primary 概率>0.9 的帖子(检验高置信=正确?)
层3 随机基线层(30): 全量随机
"""
import json, random
from collections import defaultdict

BASE = '/Users/mac/Documents/trae_projects/MyResearcher/pipeline-runs'
SRC = f'{BASE}/date-selection-20261004/refresh/refresh-v3/source'
OUT = f'{BASE}/contrarian-bottleneck-20261005'
random.seed(42)

proxy = json.load(open(f'{OUT}/retail-proxy-stats.json'))
order = sorted(proxy.items(), key=lambda kv: -kv[1]['turnover'])
tiers = {c: ('HIGH' if i < 7 else ('MID' if i < 14 else 'LOW')) for i, (c, _) in enumerate(order)}

# HIGH 组 extreme 信号日集合
extreme_days = set()
with open(f'{BASE}/date-selection-20261004/evaluation/all-method-opportunities.jsonl') as f:
    for line in f:
        c = json.loads(line)
        if c['method'] == 'reference/rolling_extreme' and c.get('signal') and tiers.get(c['stock_code']) == 'HIGH':
            extreme_days.add((c['stock_code'], c['entry_date']))
print(f'HIGH 组 extreme 信号日: {len(extreme_days)} 个')

# metadata: sample_id -> (stock, entry_date)
meta = {}
with open(f'{SRC}/sampled-post-metadata.jsonl') as f:
    for line in f:
        m = json.loads(line)
        meta[m['sample_id']] = (m['stock_code'], m.get('entry_date'))

# 预测(只保留情绪/立场/操作三头 + 文本)
preds = {}
with open(f'{SRC}/m12-predictions.jsonl') as f:
    for line in f:
        p = json.loads(line)
        emo = p['predictions']['emotion_primary']
        st = p['predictions']['stance']
        act = p['predictions']['action_tendency']
        preds[p['sample_id']] = {
            'sample_id': p['sample_id'],
            'stock_code': p['stock_code'],
            'stock_name': p['stock_name'],
            'text': p['model_text'],
            'student_emotion': emo['prediction'],
            'student_emotion_prob': emo['probabilities'][emo['prediction']],
            'student_stance': st['prediction'],
            'student_stance_prob': st['probabilities'][st['prediction']],
            'student_action': act['prediction'],
            'student_action_prob': act['probabilities'][act['prediction']],
        }

sids = [s for s in preds if s in meta]
print(f'可关联样本总数: {len(sids)}')

# 层1: 极值日
layer1_pool = [s for s in sids if (meta[s][0], meta[s][1]) in extreme_days]
print(f'层1 候选(极值日帖子): {len(layer1_pool)}')
layer1 = random.sample(layer1_pool, min(60, len(layer1_pool)))

# 层2: 学生情绪高置信(排除层1)
rest = [s for s in sids if s not in set(layer1)]
layer2_pool = [s for s in rest if preds[s]['student_emotion_prob'] > 0.9]
print(f'层2 候选(情绪概率>0.9): {len(layer2_pool)}')
layer2 = random.sample(layer2_pool, min(30, len(layer2_pool)))

# 层3: 随机(排除前两层)
rest2 = [s for s in rest if s not in set(layer2)]
layer3 = random.sample(rest2, 30)

with open(f'{OUT}/k3-audit-sample.jsonl', 'w') as f:
    for layer, group in (('extreme_day', layer1), ('student_confident', layer2), ('random_baseline', layer3)):
        for s in group:
            r = dict(preds[s])
            r['layer'] = layer
            r['entry_date'] = meta[s][1]
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'输出 {OUT}/k3-audit-sample.jsonl: {len(layer1)}+{len(layer2)}+{len(layer3)} = {len(layer1)+len(layer2)+len(layer3)} 条')
