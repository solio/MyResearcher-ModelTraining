#!/usr/bin/env python3
"""build_panel.py - P1-3b: 面板生成器(修复时间泄漏)。

主口径(默认): 帖子日 t(周末/节假日不顺延, t 就是自然日) -> 入场 = t 之后(严格大于)下一交易日开盘。
  理由: t 当天的帖子是当天行情的观众反应, 不能用来在 t 开盘入场(信息在开盘后才存在)。
副口径(预先声明的对照): t 是交易日且帖子在 15:00 前发布 -> t 日收盘入场(密度只算 15:00 前的帖子)。
  t 非交易日时副口径不存在(那些日子的帖子只能走主口径)。
past5(新定义): 入场日前一交易日收盘 / 再往前 5 个交易日收盘 - 1。主口径下即「信号日收盘/其前5日收盘」,
  绝不许用入场日当天的任何价格。

输出列(每行 = 股 x 信号日):
  stock_code, sig_day, entry_day(主口径入场日), open,
  r5, r10, mdd10  (主口径: 开盘 -> +4/+9 交易日收盘; mdd=开盘至后10日最低价)
  alt_ok(副口径是否可用), alt_close(t收盘),
  r5c, r10c, mdd10c (副口径: t收盘 -> +5/+10 交易日收盘; mdd=t收盘至 t+1..t+10 最低价)
  bench_r5, bench_r10 (主口径同窗口沪深300), past5,
  n_posts, n_authors, dispersion, max_author_share,
  dens_bear, dens_sell, dens_buy, dens_excite (全量帖, 主口径用)
  n_posts_pre15, dens_bear_pre15, dens_sell_pre15, dens_buy_pre15, dens_excite_pre15 (15:00 前帖子, 副口径用)
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

def next_td(stock, day):
    """主口径: 严格大于 day 的下一交易日"""
    for d in tds.get(stock, []):
        if d > day: return d
    return None

def alt_td(stock, day):
    """副口径: day 本身是交易日才可当日收盘入场"""
    ds = tds.get(stock, [])
    return day if day in bars.get(stock, {}) else None

# 聚合: 主键 (stock, sig_day); 副口径帖子单独计数
cells = defaultdict(lambda: {'n': 0, 'au': defaultdict(int), 'bear': 0, 'sell': 0, 'buy': 0, 'excite': 0,
                             'n15': 0, 'bear15': 0, 'sell15': 0, 'buy15': 0, 'excite15': 0})
dropped = {'pre_bars': 0, 'after_cutoff': 0}
conn = sqlite3.connect('file:' + CFG['posts'] + '?mode=ro', uri=True)
for title, stock, pub, aid in conn.execute("SELECT title, stock_code, published_at, author_id FROM posts"):
    if not pub or stock not in bars: continue
    day = pub[:10]
    if day > CUTOFF:
        dropped['after_cutoff'] += 1; continue
    if day < tds[stock][0]:
        dropped['pre_bars'] += 1; continue
    c = cells[(stock, day)]
    c['n'] += 1
    c['au'][aid] += 1
    nt = normalize(title)
    h1 = BEARW.search(nt) is not None
    h2 = SELL.search(nt) is not None
    h3 = BUY.search(nt) is not None
    h4 = EXCITE.search(nt) is not None
    if h1: c['bear'] += 1
    if h2: c['sell'] += 1
    if h3: c['buy'] += 1
    if h4: c['excite'] += 1
    hh = int(pub[11:13]) if len(pub) >= 13 else 99
    if hh < 15:
        c['n15'] += 1
        if h1: c['bear15'] += 1
        if h2: c['sell15'] += 1
        if h3: c['buy15'] += 1
        if h4: c['excite15'] += 1
conn.close()

def bench_ret(d0, d1):
    if d0 in bench and d1 in bench: return bench[d1] / bench[d0] - 1
    return None

rows, unpriced = [], 0
for (stock, sig_day), c in sorted(cells.items()):
    ds = tds[stock]
    ed = next_td(stock, sig_day)          # 主口径入场日
    if not ed: continue
    i = ds.index(ed)
    b = bars[stock]
    o = b[ed]['o']
    r5 = b[ds[i + 4]]['c'] / o - 1 if i + 4 < len(ds) else None
    r10 = b[ds[i + 9]]['c'] / o - 1 if i + 9 < len(ds) else None
    lows = [b[ds[j]]['l'] for j in range(i, min(i + 10, len(ds)))]
    mdd10 = min(lows) / o - 1 if lows else None
    # past5 新定义: 入场日前一交易日收盘 / 再往前5日收盘
    past5 = b[ds[i - 1]]['c'] / b[ds[i - 6]]['c'] - 1 if i >= 6 else None
    br5 = bench_ret(ed, ds[i + 4]) if i + 4 < len(ds) else None
    br10 = bench_ret(ed, ds[i + 9]) if i + 9 < len(ds) else None
    # 副口径
    ad = alt_td(stock, sig_day)
    if ad and c['n15'] >= 1:
        j = ds.index(ad)
        ac = b[ad]['c']
        r5c = b[ds[j + 5]]['c'] / ac - 1 if j + 5 < len(ds) else None
        r10c = b[ds[j + 10]]['c'] / ac - 1 if j + 10 < len(ds) else None
        lows2 = [b[ds[k]]['l'] for k in range(j + 1, min(j + 11, len(ds)))]
        mdd10c = min(lows2) / ac - 1 if lows2 else None
        past5c = b[ds[j - 1]]['c'] / b[ds[j - 6]]['c'] - 1 if j >= 6 else None
        alt_ok = 1
    else:
        ad, ac, r5c, r10c, mdd10c, past5c, alt_ok = '', '', None, None, None, None, 0
    def den(x, base): return round(x / base, 4) if base else ''
    rows.append({
        'stock_code': stock, 'sig_day': sig_day, 'entry_day': ed, 'open': o,
        'r5': round(r5, 5) if r5 is not None else '',
        'r10': round(r10, 5) if r10 is not None else '',
        'mdd10': round(mdd10, 5) if mdd10 is not None else '',
        'alt_ok': alt_ok,
        'r5c': round(r5c, 5) if r5c is not None else '',
        'r10c': round(r10c, 5) if r10c is not None else '',
        'mdd10c': round(mdd10c, 5) if mdd10c is not None else '',
        'bench_r5': round(br5, 5) if br5 is not None else '',
        'bench_r10': round(br10, 5) if br10 is not None else '',
        'past5': round(past5, 5) if past5 is not None else '',
        'past5c': round(past5c, 5) if past5c is not None else '',
        'n_posts': c['n'], 'n_authors': len(c['au']),
        'dispersion': round(len(c['au']) / c['n'], 4),
        'max_author_share': round(max(c['au'].values()) / c['n'], 4),
        'dens_bear': den(c['bear'], c['n']), 'dens_sell': den(c['sell'], c['n']),
        'dens_buy': den(c['buy'], c['n']), 'dens_excite': den(c['excite'], c['n']),
        'n_posts_pre15': c['n15'],
        'dens_bear_pre15': den(c['bear15'], c['n15']),
        'dens_sell_pre15': den(c['sell15'], c['n15']),
        'dens_buy_pre15': den(c['buy15'], c['n15']),
        'dens_excite_pre15': den(c['excite15'], c['n15']),
    })
    if r5 is None or r10 is None: unpriced += 1

with gzip.open(PANEL_OUT, 'wt', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

log = {
    'built_at': datetime.now().isoformat(timespec='seconds'),
    'panel': PANEL_OUT,
    'panel_rows': len(rows),
    'panel_sha256': hashlib.sha256(open(PANEL_OUT, 'rb').read()).hexdigest(),
    'posts_in_panel': sum(c['n'] for c in cells.values()),
    'dropped': dropped,
    'rows_without_full_pricing': unpriced,
    'rows_alt_ok': sum(1 for r in rows if r['alt_ok']),
    'cutoff': CUTOFF,
    'conventions': {
        'main': 'sig_day t -> entry = next trading day (d>t) open; returns open->close(+4/+9); mdd open->min low (t+0..t+9)',
        'alt': 'sig_day t is trading day & posts before 15:00 -> entry t close; returns close->close(+5/+10); mdd t close->min low(t+1..t+10)',
        'past5': 'close[entry_day-1td] / close[entry_day-6td] - 1 (no entry-day price)',
    },
    'wordlists': {'bear': 'bearw.v1.re', 'sell': 'declare_sell.v1.re',
                  'buy': 'DECLARE_BUY(from anchors/gen_k3_experiment.py)', 'excite': 'excite.v1.re'},
}
with open(CFG['panel_output_dir'] + '/build-panel-log.json', 'w') as f:
    json.dump(log, f, ensure_ascii=False, indent=1)
print(json.dumps(log, ensure_ascii=False, indent=1))
