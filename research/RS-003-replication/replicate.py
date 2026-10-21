#!/usr/bin/env python3
"""replicate.py - P1-4: RS-003 两遍复现(对账用)。

(a) 原文档口径: 信号 = dens_bear 处于该股**全期**前 10%(原文档未写明是否只看过去, 按全期分位复刻);
    对照 = dens_bear <= 该股中位数; 主指标 r10(10个交易日), 辅助 r5。
(b) 只看过去口径: 分位数用扩张窗口(只用信号日之前的数据), 至少 60 个交易日热身。
两遍都报: A 段(<=2026-05-31) / B 段(2026-06-01~2026-09-30) 的信号均值/对照均值/超额,
过去5日下跌组内对照, 发帖人分散/集中对比, 逐股方向一致数。
"""
import csv, gzip, json, math, re, hashlib
from collections import defaultdict
from datetime import datetime

CFG = 'research/config/data-sources.yaml'
txt = open(CFG).read()
PANEL = re.search(r'^panel_output_dir: (.+)$', txt, re.M).group(1).strip() + '/panel.csv.gz'
A_END, B_END = '2026-05-31', '2026-09-30'

rows = []
with gzip.open(PANEL, 'rt') as f:
    for r in csv.DictReader(f):
        if r['r10'] == '' or r['past5'] == '': continue
        for k in ['r5', 'r10', 'past5', 'dispersion', 'dens_bear', 'n_posts']:
            r[k] = float(r[k])
        rows.append(r)

by_stock = defaultdict(list)
for r in rows: by_stock[r['stock_code']].append(r)
for s in by_stock: by_stock[s].sort(key=lambda x: x['entry_day'])

def qtile(vals, q):
    v = sorted(vals)
    return v[min(len(v) - 1, int(len(v) * q))]

def compute(which):
    """which='a' 全期分位; which='b' 扩张窗口分位(热身60交易日). 返回每股信号/对照收益列表."""
    out = {}
    for stock, rs in by_stock.items():
        dens = [x['dens_bear'] for x in rs]
        if which == 'a':
            thr = qtile(dens, 0.9)
            med = qtile(dens, 0.5)
            sig = [x for x in rs if x['dens_bear'] >= thr and x['n_posts'] >= 5]
            ctl = [x for x in rs if x['dens_bear'] <= med and x['n_posts'] >= 5]
        else:
            sig, ctl = [], []
            for i, x in enumerate(rs):
                if i < 60: continue  # 热身
                hist = dens[:i]  # 只用之前
                if len(hist) < 60: continue
                thr = qtile(hist, 0.9)
                med = qtile(hist, 0.5)
                if x['dens_bear'] >= thr and x['n_posts'] >= 5: sig.append(x)
                elif x['dens_bear'] <= med and x['n_posts'] >= 5: ctl.append(x)
        out[stock] = {'sig': sig, 'ctl': ctl}
    return out

def agg(res, seg, metric='r10', sub=None):
    """seg='A'/'B'; sub: None/'past5_down'/'disp_high'/'disp_low'"""
    sigs, ctls = [], []
    lo, hi = ('0000-01-01', A_END) if seg == 'A' else ('2026-06-01', B_END)
    for stock, v in res.items():
        for grp, acc in [('sig', sigs), ('ctl', ctls)]:
            for x in v[grp]:
                if not (lo <= x['entry_day'] <= hi): continue
                if sub == 'past5_down' and x['past5'] >= 0: continue
                if sub == 'disp_high' and x['dispersion'] < 0.6: continue
                if sub == 'disp_low' and x['dispersion'] >= 0.6: continue
                acc.append(x[metric])
    if not sigs or not ctls: return None
    return {'sig_mean': sum(sigs) / len(sigs) * 100, 'ctl_mean': sum(ctls) / len(ctls) * 100,
            'excess': (sum(sigs) / len(sigs) - sum(ctls) / len(ctls)) * 100,
            'n_sig': len(sigs), 'n_ctl': len(ctls)}

def per_stock_dir(res, metric='r10'):
    pos = neg = 0
    detail = {}
    for stock, v in res.items():
        if len(v['sig']) < 3 or len(v['ctl']) < 3: continue
        e = (sum(x[metric] for x in v['sig']) / len(v['sig']) - sum(x[metric] for x in v['ctl']) / len(v['ctl'])) * 100
        detail[stock] = round(e, 2)
        if e > 0: pos += 1
        else: neg += 1
    return pos, neg, detail

result = {'built_at': datetime.now().isoformat(timespec='seconds'), 'panel_sha256': hashlib.sha256(open(PANEL, 'rb').read()).hexdigest()[:16]}
for which in ['a', 'b']:
    res = compute(which)
    blk = {}
    for seg in ['A', 'B']:
        blk[seg] = {'main': agg(res, seg), 'past5_down': agg(res, seg, sub='past5_down'),
                    'disp_high': agg(res, seg, sub='disp_high'), 'disp_low': agg(res, seg, sub='disp_low')}
    blk['per_stock'] = per_stock_dir(res)
    # 样本量
    blk['n_sig_total'] = sum(len(v['sig']) for v in res.values())
    blk['n_ctl_total'] = sum(len(v['ctl']) for v in res.values())
    result[which] = blk

OUT = 'research/RS-003-replication'
import os
os.makedirs(OUT, exist_ok=True)
with open(OUT + '/result.json', 'w') as f:
    json.dump(result, f, ensure_ascii=False, indent=1)
print(json.dumps(result, ensure_ascii=False, indent=1))
