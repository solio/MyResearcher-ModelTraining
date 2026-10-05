#!/usr/bin/env python3
"""K3 盲标结果与学生模型对比,产出仲裁包。

K3 标签维度: emotion(对齐学生标签体系) / stance / sarcasm(项目七头没有的增量维度)。
盲标纪律: 打标时不看学生标签,只读文本+股票名+层。
"""
import json

# (idx, emotion, stance, sarcasm, note)
K3 = [
    (0,  'NONE_EXPLICIT', 'BEAR', False, '冷静判断'),
    (1,  'NONE_EXPLICIT', 'NEUTRAL', False, '歧义:猛无方向,极值日疑指跌'),
    (2,  'NONE_EXPLICIT', 'NEUTRAL', False, '事件前观望'),
    (3,  'FRUSTRATION', 'BEAR', False, '恨铁不成钢'),
    (4,  'FRUSTRATION', 'BEAR', False, '怨主力'),
    (5,  'HOPE', 'BULL', False, '打气'),
    (6,  'CALM', 'BEAR', False, '逃顶庆幸'),
    (7,  'NONE_EXPLICIT', 'BEAR', False, '邀功+看空规律'),
    (8,  'FRUSTRATION', 'BEAR', False, '恨铁不成钢'),
    (9,  'NONE_EXPLICIT', 'BULL', False, '加仓宣言'),
    (10, 'HOPE', 'BULL', False, '盼反转'),
    (11, 'FRUSTRATION', 'BEAR', False, '水下一天无奈'),
    (12, 'NONE_EXPLICIT', 'BULL', False, '越跌越买'),
    (13, 'NONE_EXPLICIT', 'BEAR', False, '洗盘黑话看戏'),
    (14, 'REGRET', 'BULL', False, '卖飞懊恼'),
    (15, 'NONE_EXPLICIT', 'BEAR', False, '看跌预测'),
    (16, 'HOPE', 'BULL', False, '涨价逻辑期待'),
    (17, 'NONE_EXPLICIT', 'BEAR', False, '盼跌买入型'),
    (18, 'NONE_EXPLICIT', 'BEAR', False, '估值判断'),
    (19, 'FRUSTRATION', 'BEAR', False, '无奈'),
    (20, 'NONE_EXPLICIT', 'NEUTRAL', False, '冷静经验谈'),
    (21, 'FRUSTRATION', 'NEUTRAL', True, '自嘲:一买就跌'),
    (22, 'NONE_EXPLICIT', 'NEUTRAL', False, '消息陈述'),
    (23, 'ANXIETY', 'BEAR', False, '血雨腥风预期'),
    (24, 'ANGER', 'BEAR', True, '飞舞=废物谐音骂'),
    (25, 'FRUSTRATION', 'BEAR', False, '融资盘激增无奈'),
    (26, 'REGRET', 'BEAR', False, '割肉'),
    (27, 'FRUSTRATION', 'BEAR', False, '腰斩震惊'),
    (28, 'NONE_EXPLICIT', 'BEAR', False, '看空警告'),
    (29, 'FRUSTRATION', 'BEAR', False, '恨铁不成钢'),
    (30, 'FRUSTRATION', 'BEAR', True, '你应该很强的=你怎么这么弱'),
    (31, 'NONE_EXPLICIT', 'BEAR', False, '洗盘看戏'),
    (32, 'NONE_EXPLICIT', 'BEAR', False, '技术分析谨慎'),
    (33, 'NONE_EXPLICIT', 'BULL', False, '买入陈述'),
    (34, 'ANXIETY', 'BULL', False, '自我怀疑但仍信'),
    (35, 'ANGER', 'BEAR', False, '割肉永别'),
    (36, 'ANGER', 'BEAR', False, '死了一样'),
    (37, 'FRUSTRATION', 'BEAR', False, '要死不活'),
    (38, 'NONE_EXPLICIT', 'NEUTRAL', False, '两字报成本,信息不足'),
    (39, 'FRUSTRATION', 'BEAR', False, '跟跌预期'),
    (40, 'ANGER', 'BEAR', False, '蔑称+看空预测'),
    (41, 'NONE_EXPLICIT', 'NEUTRAL', False, '洗盘+支撑分析'),
    (42, 'REGRET', 'BEAR', False, '亏死了割肉'),
    (43, 'NONE_EXPLICIT', 'BULL', False, '纳入指数陈述'),
    (44, 'ANXIETY', 'BEAR', False, '难顶担忧'),
    (45, 'NONE_EXPLICIT', 'BULL', False, '盼跌进场'),
    (46, 'FRUSTRATION', 'BEAR', True, '反讽:以后不要盈利了(讽市场扭曲)'),
    (47, 'ANXIETY', 'BEAR', False, '62的本心寒'),
    (48, 'NONE_EXPLICIT', 'NEUTRAL', False, '疑问句'),
    (49, 'NONE_EXPLICIT', 'NEUTRAL', True, '辛苦了=轻度调侃主力'),
    (50, 'HOPE', 'BULL', False, '长期看好'),
    (51, 'HOPE', 'BULL', False, '花开富贵'),
    (52, 'NONE_EXPLICIT', 'BEAR', False, '产能过剩分析'),
    (53, 'FRUSTRATION', 'BEAR', False, '比暴雷还狠'),
    (54, 'ANGER', 'BEAR', False, '唱多出货'),
    (55, 'ANGER', 'BEAR', False, '揭露对倒出货'),
    (56, 'NONE_EXPLICIT', 'BULL', True, '假恐惧真买入:吓的=反话'),
    (57, 'FRUSTRATION', 'BEAR', False, '一付s相'),
    (58, 'ANXIETY', 'BEAR', False, '慢慢熬'),
    (59, 'FRUSTRATION', 'BEAR', False, '5连阴难以置信'),
    (60, 'ANGER', 'BEAR', True, '拉几=垃圾谐音'),
    (61, 'FRUSTRATION', 'BEAR', False, '天天跌'),
    (62, 'ANGER', 'BEAR', False, '垃圾'),
    (63, 'ANGER', 'BEAR', True, '晚期没救了+讽编造利好'),
    (64, 'CALM', 'NEUTRAL', False, '抗跌庆幸(省略句)'),
    (65, 'FRUSTRATION', 'BEAR', False, '干不过五粮液'),
    (66, 'NONE_EXPLICIT', 'NEUTRAL', False, '客观复盘'),
    (67, 'NONE_EXPLICIT', 'NEUTRAL', False, '回购推断'),
    (68, 'NONE_EXPLICIT', 'NEUTRAL', False, '债券信息转述'),
    (69, 'ANXIETY', 'BEAR', False, '警惕回调'),
    (70, 'ANGER', 'BEAR', False, '辱骂'),
    (71, 'FRUSTRATION', 'BEAR', False, '老登股清仓'),
    (72, 'ANGER', 'BEAR', False, '配合出货'),
    (73, 'NONE_EXPLICIT', 'BULL', False, '大宗折价后上涨数据'),
    (74, 'NONE_EXPLICIT', 'NEUTRAL', False, '联动判断'),
    (75, 'FRUSTRATION', 'BEAR', False, '没戏了'),
    (76, 'CALM', 'NEUTRAL', False, '谈定=淡定谐音'),
    (77, 'NONE_EXPLICIT', 'BULL', False, '订单公告转述'),
    (78, 'FRUSTRATION', 'BEAR', False, '认命'),
    (79, 'NONE_EXPLICIT', 'BEAR', False, '看空预测'),
    (80, 'NONE_EXPLICIT', 'BULL', False, '建仓区'),
    (81, 'ANGER', 'BEAR', False, '辱骂销户'),
    (82, 'ANGER', 'BEAR', True, '真牛逼从不套人=经典反讽'),
    (83, 'ANGER', 'BEAR', False, '垃圾'),
    (84, 'ANXIETY', 'NEUTRAL', False, '洗盘还是出货之问'),
    (85, 'HOPE', 'BULL', False, '150希望'),
    (86, 'NONE_EXPLICIT', 'NEUTRAL', False, '仅代码空帖'),
    (87, 'FRUSTRATION', 'NEUTRAL', False, '挣钱不易'),
    (88, 'HOPE', 'BULL', False, '相信川子'),
    (89, 'ANGER', 'BEAR', False, 'md有病不玩了'),
    (90, 'EXCITEMENT', 'BULL', False, '崛起加仓'),
    (91, 'HOPE', 'BULL', False, '盼龙头带头'),
    (92, 'EXCITEMENT', 'BULL', False, '邀功:信我吃肉'),
    (93, 'FRUSTRATION', 'BEAR', False, '年线死叉'),
    (94, 'FRUSTRATION', 'BEAR', True, '挺好=反话(质押增做空券源)'),
    (95, 'ANXIETY', 'BEAR', False, '怕跌停'),
    (96, 'ANXIETY', 'NEUTRAL', False, '不强则弃的条件式'),
    (97, 'NONE_EXPLICIT', 'NEUTRAL', False, '口号'),
    (98, 'REGRET', 'BEAR', False, '亏12点'),
    (99, 'FRUSTRATION', 'BEAR', True, '你还混什么=讥讽'),
    (100, 'FRUSTRATION', 'BEAR', False, '爬着不动'),
    (101, 'FRUSTRATION', 'BEAR', False, '又完'),
    (102, 'NONE_EXPLICIT', 'BULL', False, '大基金入驻'),
    (103, 'FRUSTRATION', 'NEUTRAL', False, '盼涨焦躁'),
    (104, 'NONE_EXPLICIT', 'NEUTRAL', False, '合作建议'),
    (105, 'HOPE', 'BULL', False, '盼冲69'),
    (106, 'ANGER', 'BEAR', True, '鲁迅吃人梗:史书两个字'),
    (107, 'NONE_EXPLICIT', 'BEAR', True, '主力小乖乖=揶揄+盼补缺'),
    (108, 'NONE_EXPLICIT', 'NEUTRAL', False, '歧义:疑装死躺平'),
    (109, 'HOPE', 'BULL', False, '诗句明志'),
    (110, 'FRUSTRATION', 'BEAR', False, '怨地域'),
    (111, 'EXCITEMENT', 'BULL', False, '百元俱乐部'),
    (112, 'FRUSTRATION', 'BEAR', False, '没人玩'),
    (113, 'FRUSTRATION', 'BEAR', False, '质问公司'),
    (114, 'NONE_EXPLICIT', 'NEUTRAL', False, '缩量陈述'),
    (115, 'FRUSTRATION', 'BEAR', False, '瘫着不动'),
    (116, 'FRUSTRATION', 'BEAR', False, '被碾压'),
    (117, 'NONE_EXPLICIT', 'BULL', False, '涨价函利好'),
    (118, 'NONE_EXPLICIT', 'BEAR', False, '看空立帖为证'),
    (119, 'FRUSTRATION', 'BEAR', False, '废股'),
]

OUT = '/Users/mac/Documents/trae_projects/MyResearcher/pipeline-runs/contrarian-bottleneck-20261005'
rows = [json.loads(l) for l in open(f'{OUT}/k3-audit-sample.jsonl')]
assert len(rows) == 120 and len(K3) == 120

merged = []
for (idx, emo, sta, sar, note) in K3:
    r = dict(rows[idx])
    assert idx == rows.index(rows[idx])  # idx 对齐
    r['k3_emotion'] = emo
    r['k3_stance'] = sta
    r['k3_sarcasm'] = sar
    r['k3_note'] = note
    r['agree_emotion'] = (emo == r['student_emotion'])
    r['agree_stance'] = (sta == r['student_stance'])
    merged.append(r)

with open(f'{OUT}/k3-audit-merged.jsonl', 'w') as f:
    for r in merged:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

# ---- 统计 ----
from collections import Counter
def agree_stats(sub, name):
    n = len(sub)
    ae = sum(1 for r in sub if r['agree_emotion'])
    ast = sum(1 for r in sub if r['agree_stance'])
    sar = sum(1 for r in sub if r['k3_sarcasm'])
    print(f'{name:<22} n={n:>3}  情绪一致 {ae}/{n} ({ae/n*100:.0f}%)  立场一致 {ast}/{n} ({ast/n*100:.0f}%)  K3判反讽 {sar}')

print('===== K3 vs 学生模型一致率 =====')
for layer in ['extreme_day', 'student_confident', 'random_baseline']:
    agree_stats([r for r in merged if r['layer'] == layer], layer)
agree_stats(merged, 'ALL')

# 反讽子集的一致率(核心检验)
sar_rows = [r for r in merged if r['k3_sarcasm']]
print(f'\n===== 反讽子集(n={len(sar_rows)})=====')
for r in sar_rows:
    mark_e = 'OK ' if r['agree_emotion'] else 'DIFF'
    print(f"  [{mark_e}] {r['text'][:38]:<40} 学生:{r['student_emotion']}/{r['student_stance']}  K3:{r['k3_emotion']}/{r['k3_stance']}")
ae = sum(1 for r in sar_rows if r['agree_emotion'])
ast = sum(1 for r in sar_rows if r['agree_stance'])
print(f'反讽子集: 情绪一致 {ae}/{len(sar_rows)}, 立场一致 {ast}/{len(sar_rows)}')

# 分歧仲裁包
diff = [r for r in merged if not r['agree_emotion'] or not r['agree_stance']]
with open(f'{OUT}/k3-arbitration-pack.jsonl', 'w') as f:
    for r in diff:
        f.write(json.dumps({
            'sample_id': r['sample_id'], 'layer': r['layer'],
            'stock_code': r['stock_code'], 'stock_name': r['stock_name'], 'entry_date': r['entry_date'],
            'text': r['text'],
            'student': {'emotion': r['student_emotion'], 'emotion_prob': round(r['student_emotion_prob'], 3),
                        'stance': r['student_stance'], 'stance_prob': round(r['student_stance_prob'], 3)},
            'k3': {'emotion': r['k3_emotion'], 'stance': r['k3_stance'], 'sarcasm': r['k3_sarcasm'], 'note': r['k3_note']},
            'arbitration': None,
        }, ensure_ascii=False) + '\n')
print(f'\n仲裁包: {OUT}/k3-arbitration-pack.jsonl 共 {len(diff)} 条分歧')
