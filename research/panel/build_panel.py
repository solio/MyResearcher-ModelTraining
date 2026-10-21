#!/usr/bin/env python3
"""build_panel.py - P1-3: 建立「股票 x 交易日」面板。

输出(本地, 不进 git): <panel_output_dir>/panel.csv.gz
每行 = 一只股的一个入场日:
  stock_code, entry_day, n_posts, n_authors, dispersion, max_author_share,
  dens_bear, dens_sell, dens_buy, dens_excite,
  open, r5, r10, mdd10, bench_r5, bench_r10, past5
口径:
  - 帖子只用标题; 归一化沿用 anchors/gen_k3_experiment.py 的 normalize()
  - 入场日 = 帖子日期之后该股下一个交易日(周末/节假日归下一交易日)
  - r5 = 入场日开盘 -> 第4个交易日收盘; r10 = 入场日开盘 -> 第9个交易日收盘
  - mdd10 = 入场日开盘 -> 其后10个交易日内最低价的最大跌幅(无 low 退用收盘, 记 run-log)
  - bench_* = 沪深300 同窗口收益; past5 = 入场日收盘/5个交易日前收盘-1
  - 2026-09-30 之后的帖子/行情不入面板(留给 C 段); 无入场日的帖子丢弃(计数入 run-log)
"""
import csv, gzip, json, re, sqlite3, hashlib
from collections import defaultdict
from datetime import datetime

CFG_PATH = 'research/config/data-sources.yaml'

def load_cfg():
    txt = open(CFG_PATH).read()
    out = {'panel_output_dir': re.search(r'^panel_output_dir: (.+)$', txt, re.M).group(1).strip()}
    for key in ['posts', 'bars_ab', 'benchmark_ab']:
        m = re.search(r'  ' + key + r':\n(?:    .*\n)*?    path: (.+)$', txt, re.M)
        out[key] = m.group(1).strip()
    return out

CFG = load_cfg()
PANEL_OUT = CFG['panel_output_dir'] + '/panel.csv.gz'
CUTOFF = '2026-09-30'

def load_re(p):
    txt = open(p, encoding='utf-8').read()
    parts = [l for l in txt.split('\n') if l.strip() and not l.startswith('#')]
    return re.compile('|'.join('|'.join(parts).split('|')))

ANCH = 'research/anchors/'
BEARW = load_re(ANCH + 'bearw.v1.re')
SELL = load_re(ANCH + 'declare_sell.v1.re')
EXCITE = load_re(ANCH + 'excite.v1.re')
BUY = re.compile(r'满仓|全仓|梭哈|打满|加到|继续加|建仓')

def normalize(t):
    if not t: return ''
    t = re.sub(r'[\s~～·・、,，.。:：;；"\'\[\]()（）<>《》\-_=+*|\\/]+', '', t)
    KEEP = {'!', '！', '哈'}
    return re.sub(r'(.)\1{2,}', lambda m: m.group(0) if m.group(1) in KEEP else m.group(1), t)

bars = defaultdict(dict)
tds = {}
with open(CFG['bars_ab']) as f:
    for r in csv.DictReader(f):
        if r['trade_date'] > CUTOFF: continue
        bars[r['instrument_code']][r['trade_date']] = {
            'o': float(r['open']), 'h': float(r['high']),
            'l': float(r['low']), 'c': float(r['close'])}
for c in bars: tds[c] = sorted(bars[c])

bench = {}
with open(CFG['benchmark_ab']) as f:
    for r in csv.DictReader(f):
        if r['trade_date'] > CUTOFF: continue
        bench[r['trade_date']] = float(r['close'])

def entry_day(stock, day):
    ds = tds.get(stock, [])
    if not ds or day < ds[0]: return None  # 早于行情起点的帖子丢弃(否则全堆到首日)
    for d in ds:
        if d >= day: return d
    return None

cells = defaultdict(lambda: {'n': 0, 'au': defaultdict(int), 'bear': 0, 'sell': 0, 'buy': 0, 'excite': 0})
dropped = {'no_entry_day': 0, 'after_cutoff': 0, 'pre_bars': 0}
conn = sqlite3.connect('file:' + CFG['posts'] + '?mode=ro', uri=True)
for title, stock, pub, aid in conn.execute("SELECT title, stock_code, published_at, author_id FROM posts"):
    if not pub: continue
    day = pub[:10]
    if day > CUTOFF:
        dropped['after_cutoff'] += 1; continue
    ed = entry_day(stock, day)
    if not ed:
        dropped['no_entry_day'] += 1
        if tds.get(stock) and day < tds[stock][0]: dropped['pre_bars'] += 1
        continue
    c = cells[(stock, ed)]
    c['n'] += 1
    c['au'][aid] += 1
    nt = normalize(title)
    if BEARW.search(nt): c['bear'] += 1
    if SELL.search(nt): c['sell'] += 1
    if BUY.search(nt): c['buy'] += 1
    if EXCITE.search(nt): c['excite'] += 1
conn.close()

def bench_ret(d0, d1):
    if d0 in bench and d1 in bench: return bench[d1] / bench[d0] - 1
    return None

rows, no_low, unpriced = [], 0, 0
for (stock, ed), c in sorted(cells.items()):
    ds = tds[stock]
    if ed not in bars[stock]: continue
    i = ds.index(ed)
    b = bars[stock]
    o = b[ed]['o']
    r5 = b[ds[i + 4]]['c'] / o - 1 if i + 4 < len(ds) else None
    r10 = b[ds[i + 9]]['c'] / o - 1 if i + 9 < len(ds) else None
    lows = [b[ds[j]]['l'] for j in range(i, min(i + 10, len(ds))) if ds[j] in b]
    if lows:
        mdd10 = min(lows) / o - 1
    else:
        cls = [b[ds[j]]['c'] for j in range(i, min(i + 10, len(ds)))]
        mdd10 = min(cls) / o - 1 if cls else None
        no_low += 1
    past5 = b[ed]['c'] / b[ds[i - 5]]['c'] - 1 if i >= 5 else None
    br5 = bench_ret(ed, ds[i + 4]) if i + 4 < len(ds) else None
    br10 = bench_ret(ed, ds[i + 9]) if i + 9 < len(ds) else None
    rows.append({
        'stock_code': stock, 'entry_day': ed,
        'n_posts': c['n'], 'n_authors': len(c['au']),
        'dispersion': round(len(c['au']) / c['n'], 4),
        'max_author_share': round(max(c['au'].values()) / c['n'], 4),
        'dens_bear': round(c['bear'] / c['n'], 4),
        'dens_sell': round(c['sell'] / c['n'], 4),
        'dens_buy': round(c['buy'] / c['n'], 4),
        'dens_excite': round(c['excite'] / c['n'], 4),
        'open': o,
        'r5': round(r5, 5) if r5 is not None else '',
        'r10': round(r10, 5) if r10 is not None else '',
        'mdd10': round(mdd10, 5) if mdd10 is not None else '',
        'bench_r5': round(br5, 5) if br5 is not None else '',
        'bench_r10': round(br10, 5) if br10 is not None else '',
        'past5': round(past5, 5) if past5 is not None else '',
    })
    if r5 is None or r10 is None: unpriced += 1

with gzip.open(PANEL_OUT, 'wt', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

log = {
    'built_at': datetime.now().isoformat(timespec='seconds'),
    'code_commit': None,
    'panel': PANEL_OUT,
    'panel_rows': len(rows),
    'panel_sha256': hashlib.sha256(open(PANEL_OUT, 'rb').read()).hexdigest(),
    'posts_in_panel': sum(c['n'] for c in cells.values()),
    'dropped': dropped,
    'rows_without_full_pricing': unpriced,
    'rows_mdd_used_close_fallback': no_low,
    'cutoff': CUTOFF,
    'wordlists': {'bear': 'bearw.v1.re', 'sell': 'declare_sell.v1.re',
                  'buy': 'DECLARE_BUY(from anchors/gen_k3_experiment.py)', 'excite': 'excite.v1.re'},
    'config': CFG_PATH,
}
with open(CFG['panel_output_dir'] + '/build-panel-log.json', 'w') as f:
    json.dump(log, f, ensure_ascii=False, indent=1)
print(json.dumps(log, ensure_ascii=False, indent=1))
