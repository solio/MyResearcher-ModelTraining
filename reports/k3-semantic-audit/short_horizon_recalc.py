#!/usr/bin/env python3
"""短持有期重算:判定"恐惧极值→反弹"是否在 1/2/3/5 日尺度成立。

输入(全部 frozen,自洽):
- 信号: date-selection-20261004/evaluation/all-method-opportunities.jsonl
- 行情: date-selection-20261004/market-source/daily-bars-qfq.csv (2025-07-30~2026-09-30)

收益口径: 入场日 open -> 第 H 个交易日 close(H=9 已 100% 复现原 10 日 gross_return,70/70)。
净收益 = 毛收益 - 0.001(对齐原报告 0.1% 费用)。
双条件: 信号净均值 > 0.001 且 > 同期情绪平净均值。
"""
import csv, json
from collections import defaultdict

BASE = '/Users/mac/Documents/trae_projects/MyResearcher/pipeline-runs'
OPP = f'{BASE}/date-selection-20261004/evaluation/all-method-opportunities.jsonl'
BARS = f'{BASE}/date-selection-20261004/market-source/daily-bars-qfq.csv'
OUT = f'{BASE}/contrarian-bottleneck-20261005'

FEE = 0.001
HS = [0, 1, 2, 4, 9]
H_LABEL = {0: '1d(当日open→close)', 1: '2d', 2: '3d', 4: '5d', 9: '10d(原口径)'}
METHODS = ['reference/rolling_extreme', 'reference/cooling', 'reference/union',
           'reference/B', 'reference/H', 'reference/HC']

# ---------- 行情 ----------
bars = defaultdict(dict)
with open(BARS) as f:
    for r in csv.DictReader(f):
        bars[r['instrument_code']][r['trade_date']] = r
dates_by_code = {c: sorted(bars[c].keys()) for c in bars}

def calc_ret(code, entry, H):
    ds = dates_by_code.get(code, [])
    if entry not in bars.get(code, {}):
        return None
    i = ds.index(entry)
    if i + H >= len(ds):
        return None
    p0 = float(bars[code][entry]['open'])
    p1 = float(bars[code][ds[i + H]]['close'])
    return p1 / p0 - 1

def exit_date(code, entry, H):
    ds = dates_by_code.get(code, [])
    i = ds.index(entry)
    return ds[i + H] if i + H < len(ds) else None

# ---------- 载入机会行 ----------
sig_rows = defaultdict(list)   # method -> rows(signal=true)
flat_rows = defaultdict(list)  # method -> rows(emotion_flat=true)
stock_name = {}
with open(OPP) as f:
    for line in f:
        c = json.loads(line)
        m = c['method']
        if m not in METHODS:
            continue
        stock_name[c['stock_code']] = c['stock_name']
        if c.get('signal'):
            sig_rows[m].append(c)
        if c.get('emotion_flat'):
            flat_rows[m].append(c)

# ---------- 校验 H=9 复现 ----------
chk_tot = chk_ok = 0
for m in METHODS:
    for c in sig_rows[m]:
        if c['outcome_status'] != 'COMPLETE':
            continue
        r = calc_ret(c['stock_code'], c['entry_date'], 9)
        chk_tot += 1
        if r is not None and abs(r - c['gross_return']) < 5e-7:
            chk_ok += 1
print(f'[校验] H=9 复现原 gross_return: {chk_ok}/{chk_tot} ({chk_ok/chk_tot*100:.1f}%)')

# ---------- 主计算 ----------
def nonoverlap_count(rows, H):
    """按股票内时间序贪心去重:持有窗口(至退出日)内的信号不另计。"""
    n = 0
    last_exit_idx = defaultdict(lambda: -1)
    ordered = sorted(rows, key=lambda c: (c['stock_code'], c['entry_date']))
    for c in ordered:
        ds = dates_by_code.get(c['stock_code'], [])
        if c['entry_date'] not in ds:
            continue
        i = ds.index(c['entry_date'])
        if i > last_exit_idx[c['stock_code']]:
            n += 1
            last_exit_idx[c['stock_code']] = i + H
    return n

results = []  # (method, stock, H, sig_n, sig_net_mean, flat_n, flat_net_mean, nonoverlap)
for m in METHODS:
    # 按股票分组
    sig_by_stock = defaultdict(list)
    for c in sig_rows[m]:
        sig_by_stock[c['stock_code']].append(c)
    flat_by_stock = defaultdict(list)
    for c in flat_rows[m]:
        flat_by_stock[c['stock_code']].append(c)
    stocks = sorted(set(sig_by_stock) | set(flat_by_stock))
    for s in stocks:
        for H in HS:
            srets = [r for r in (calc_ret(s, c['entry_date'], H) for c in sig_by_stock.get(s, [])) if r is not None]
            frets = [r for r in (calc_ret(s, c['entry_date'], H) for c in flat_by_stock.get(s, [])) if r is not None]
            sig_net = (sum(r - FEE for r in srets) / len(srets)) if srets else None
            flat_net = (sum(r - FEE for r in frets) / len(frets)) if frets else None
            results.append({
                'method': m, 'stock': s, 'stock_name': stock_name.get(s, ''), 'H': H,
                'sig_n': len(srets), 'sig_net_mean': sig_net,
                'flat_n': len(frets), 'flat_net_mean': flat_net,
                'nonoverlap': nonoverlap_count(sig_by_stock.get(s, []), H) if srets else 0,
                'dual_pass': (sig_net is not None and flat_net is not None
                              and sig_net > FEE and sig_net > flat_net),
            })

with open(f'{OUT}/short-horizon-per-stock.jsonl', 'w') as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

# ---------- 汇总:逐股先平均再等权 ----------
print('\n========== 汇总(信号净均值,逐股等权) ==========')
hdr = f"{'method':<28}{'H':<4}{'期限':<22}{'信号n':>6}{'净均值%':>9}{'情绪平n':>8}{'平净%':>8}{'双条件':>7}{'非重叠':>6}"
print(hdr)
summary_rows = []
for m in METHODS:
    for H in HS:
        cells = [r for r in results if r['method'] == m and r['H'] == H and r['sig_net_mean'] is not None]
        if not cells:
            continue
        sig_eq = sum(c['sig_net_mean'] for c in cells) / len(cells)
        fcells = [c for c in cells if c['flat_net_mean'] is not None]
        flat_eq = sum(c['flat_net_mean'] for c in fcells) / len(fcells) if fcells else None
        dual = sum(1 for c in cells if c['dual_pass'])
        nover = sum(c['nonoverlap'] for c in cells)
        n_sig = sum(c['sig_n'] for c in cells)
        n_flat = sum(c['flat_n'] for c in fcells)
        summary_rows.append({'method': m, 'H': H, 'stocks': len(cells), 'n_sig': n_sig,
                             'sig_net_mean_eq': sig_eq, 'flat_net_mean_eq': flat_eq,
                             'dual_pass_stocks': dual, 'nonoverlap': nover})
        flat_s = f'{flat_eq*100:>7.2f}' if flat_eq is not None else '    n/a'
        print(f'{m:<28}{H:<4}{H_LABEL[H]:<22}{n_sig:>6}{sig_eq*100:>8.2f}%{n_flat:>8}{flat_s:>8}{dual:>4}/{len(cells):<3}{nover:>6}')

with open(f'{OUT}/short-horizon-summary.json', 'w') as f:
    json.dump({'fee': FEE, 'horizons': H_LABEL, 'check_h9': f'{chk_ok}/{chk_tot}',
               'rows': summary_rows}, f, ensure_ascii=False, indent=2)
print(f'\n输出: {OUT}/short-horizon-per-stock.jsonl, short-horizon-summary.json')
