#!/usr/bin/env python3
"""replicate.py - P1-4b: RS-003 全组合复现(修复后)。

组合矩阵: 分位口径(a=全期 / b=扩张窗口热身60) x 入场口径(main=次日开盘 / alt=当日15:00前帖->当日收盘)
          x 去重(10日簇首 / 不去重)。共 8 套, 全部写进 result.json 与 run-log.md。
主指标: r10(主口径) / r10c(副口径); 辅助: r5/r5c。
判定条款(P1 预写死): (a)主口径 与原文档每数差<=0.2pp -> 对账通过; (b) 任一段超额<=0 -> RS-003 review。
"""
import csv, gzip, json, math, re, hashlib, os
from collections import defaultdict
from datetime import datetime

txt = open('research/config/data-sources.yaml').read()
PANEL = re.search(r'^panel_output_dir: (.+)$', txt, re.M).group(1).strip() + '/panel.csv.gz'
A_END, B_END = '2026-05-31', '2026-09-30'

rows = []
with gzip.open(PANEL, 'rt') as f:
    for r in csv.DictReader(f):
        rows.append(r)
by_stock = defaultdict(list)
for r in rows: by_stock[r['stock_code']].append(r)
for s in by_stock: by_stock[s].sort(key=lambda x: x['sig_day'])

def qtile(vals, q):
    v = sorted(vals)
    return v[min(len(v) - 1, int(len(v) * q))]

def fnum(x):
    return float(x) if x not in ('', None) else None

def select(stock, rs, mode, dens_key, posts_key):
    """返回 (sig, ctl) 两组行; mode='a'全期 / 'b'扩张窗口"""
    if mode == 'a':
        dens = [fnum(x[dens_key]) for x in rs]
        dens = [x for x in dens if x is not None]
        if not dens: return [], []
        thr, med = qtile(dens, 0.9), qtile(dens, 0.5)
        sig = [x for x in rs if fnum(x[dens_key]) is not None and fnum(x[dens_key]) >= thr and fnum(x[posts_key]) is not None and fnum(x[posts_key]) >= 5]
        ctl = [x for x in rs if fnum(x[dens_key]) is not None and fnum(x[dens_key]) <= med and fnum(x[posts_key]) is not None and fnum(x[posts_key]) >= 5]
        return sig, ctl
    sig, ctl = [], []
    for i, x in enumerate(rs):
        if i < 60: continue
        hist = [fnum(y[dens_key]) for y in rs[:i]]
        hist = [y for y in hist if y is not None]
        if len(hist) < 60: continue
        thr, med = qtile(hist, 0.9), qtile(hist, 0.5)
        dv, pv = fnum(x[dens_key]), fnum(x[posts_key])
        if dv is None or pv is None or pv < 5: continue
        if dv >= thr: sig.append(x)
        elif dv <= med: ctl.append(x)
    return sig, ctl

def dedup(sig, ctl, tds_map, stock):
    def dd(lst):
        out = []
        last = None
        for x in sorted(lst, key=lambda z: z['sig_day']):
            d = x['sig_day']
            ds = tds_map[stock]
            if d not in ds:
                # 非交易日: 用其主口径入场日定位簇间隔
                ref = x['entry_day']
            else:
                ref = d
            if last is None or ds.index(ref) - ds.index(last) > 10:
                out.append(x); last = ref
        return out
    return dd(sig), dd(ctl)

def tds_of():
    txt2 = txt
    bars_p = re.search(r'  bars_ab:\n(?:    .*\n)*?    path: (.+)$', txt2, re.M).group(1).strip()
    tm = defaultdict(set)
    for r in csv.DictReader(open(bars_p)):
        if r['trade_date'] <= B_END: tm[r['instrument_code']].add(r['trade_date'])
    return {c: sorted(v) for c, v in tm.items()}

TDS = tds_of()

def agg(sigs, ctls, ret_key, seg, sub=None, past5_key='past5'):
    lo, hi = ('0000', A_END) if seg == 'A' else ('2026-06-01', B_END)
    def ok(x):
        if not (lo <= x['sig_day'] <= hi): return False
        p = fnum(x[past5_key])
        if sub == 'past5_down' and (p is None or p >= 0): return False
        if sub == 'disp_high' and fnum(x['dispersion']) < 0.6: return False
        if sub == 'disp_low' and fnum(x['dispersion']) >= 0.6: return False
        return True
    S = [fnum(x[ret_key]) for x in sigs if ok(x) and fnum(x[ret_key]) is not None]
    C = [fnum(x[ret_key]) for x in ctls if ok(x) and fnum(x[ret_key]) is not None]
    if not S or not C: return None
    return {'sig_mean': round(sum(S) / len(S) * 100, 3), 'ctl_mean': round(sum(C) / len(C) * 100, 3),
            'excess': round((sum(S) / len(S) - sum(C) / len(C)) * 100, 3), 'n_sig': len(S), 'n_ctl': len(C)}

def per_stock(sig_by_stock, ctl_by_stock, ret_key):
    pos = neg = 0; detail = {}
    for stock in sig_by_stock:
        S = [fnum(x[ret_key]) for x in sig_by_stock[stock] if fnum(x[ret_key]) is not None]
        C = [fnum(x[ret_key]) for x in ctl_by_stock[stock] if fnum(x[ret_key]) is not None]
        if len(S) < 3 or len(C) < 3: continue
        e = (sum(S) / len(S) - sum(C) / len(C)) * 100
        detail[stock] = round(e, 2)
        if e > 0: pos += 1
        else: neg += 1
    return {'pos': pos, 'neg': neg, 'detail_pp': detail}

combos = []
for mode in ['a', 'b']:
    for conv in ['main', 'alt']:
        for dd in [False, True]:
            combos.append((mode, conv, dd))

result = {'built_at': datetime.now().isoformat(timespec='seconds'),
          'panel_sha256': hashlib.sha256(open(PANEL, 'rb').read()).hexdigest(),
          'note': '8 套组合全量; 判定见 run-log.md'}
for mode, conv, dd in combos:
    dens_key = 'dens_bear' if conv == 'main' else 'dens_bear_pre15'
    posts_key = 'n_posts' if conv == 'main' else 'n_posts_pre15'
    ret_key = 'r10' if conv == 'main' else 'r10c'
    past5_key = 'past5' if conv == 'main' else 'past5c'
    sigs, ctls = [], []
    sig_by_stock, ctl_by_stock = defaultdict(list), defaultdict(list)
    for stock, rs in by_stock.items():
        if conv == 'alt': rs = [x for x in rs if x['alt_ok'] == '1']
        sig, ctl = select(stock, rs, mode, dens_key, posts_key)
        if dd: sig, ctl = dedup(sig, ctl, TDS, stock)
        sigs += sig; ctls += ctl
        sig_by_stock[stock] = sig; ctl_by_stock[stock] = ctl
    blk = {}
    for seg in ['A', 'B']:
        blk[seg] = {'main': agg(sigs, ctls, ret_key, seg, past5_key=past5_key),
                    'past5_down': agg(sigs, ctls, ret_key, seg, 'past5_down', past5_key),
                    'disp_high': agg(sigs, ctls, ret_key, seg, 'disp_high', past5_key),
                    'disp_low': agg(sigs, ctls, ret_key, seg, 'disp_low', past5_key)}
    blk['per_stock'] = per_stock(sig_by_stock, ctl_by_stock, ret_key)
    blk['n_sig'] = len(sigs); blk['n_ctl'] = len(ctls)
    result[f'{mode}_{conv}_{"dedup" if dd else "raw"}'] = blk

os.makedirs('research/RS-003-replication', exist_ok=True)
with open('research/RS-003-replication/result.json', 'w') as f:
    json.dump(result, f, ensure_ascii=False, indent=1)
for k in result:
    if '_' not in k or k in ('built_at', 'panel_sha256'): continue
    v = result[k]
    if 'A' not in v: continue
    print(f"[{k}] A: {v['A']['main']} B: {v['B']['main']} 逐股 {v['per_stock']['pos']}正/{v['per_stock']['neg']}负 n={v['n_sig']}/{v['n_ctl']}")
