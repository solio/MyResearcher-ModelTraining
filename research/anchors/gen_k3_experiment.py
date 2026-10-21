#!/usr/bin/env python3
"""生成 K3 规则分诊实验数据包(k3-triage-v1)。

从 collector.db 全库按(股×日)聚合规则分诊密度,输出与 A5.1 快照同构的
19 列数据,供 Dashboard 多实验切换展示。本脚本不训练、不调用模型、不改旧数据。
"""
import csv, json, re, sqlite3, math
from collections import defaultdict
from datetime import datetime, timezone

COLLECTOR = '/Users/mac/Documents/trae_projects/MyResearcher/MyResearcher-DataCollector/data/collector.db'
BARS = '/Users/mac/Documents/trae_projects/MyResearcher/pipeline-runs/date-selection-20261004/market-source/daily-bars-qfq.csv'
OUT = '/Users/mac/Documents/trae_projects/MyResearcher/worktrees/dashboard-k3-experiments/.local/emotion-explorer/k3-triage-v1.2.json'

# ---- 规则(与 triage-rules v3.2 + 句式 v4.1 + 归一化 一致) ----
DECLARE_SELL = re.compile(r'割肉|清仓|销户|我卖了|已卖|已割|不玩了|跑路|离场|空仓')
DECLARE_BUY = re.compile(r'满仓|全仓|梭哈|打满|加到|继续加|建仓')
EXCITE = re.compile(r'[!！]{2,}|垃圾|拉几|拉机|辣几|垃鸡|辣鸡|烂股|烂票|烂的一|废物|机狗|大狗|真狗|t狗|T狗|苟|猪力|逆子|坑爹|砸中|韭菜|九菜|杂毛|渣渣|股渣|真渣|是渣|恶庄|耍流氓|害人|飞舞|有病|犯病|妈的|md(?![a-zA-Z])|操你妈|滚|天理|完蛋|完了$|去求|欲哭无泪|血雨腥风|快跑|快逃|逃命|大逃亡|熊来了|大跌开始|崩了|亏死|气死|吓死|烦死|闷死|死了都要|救命|吓尿|疯了|破防|绷不住|卧槽|卧玛|我擦|草$|恶心|什么玩意|十三点|抄家|吼哑|哈{3,}|是人是鬼|坚决看好|绝不|不放松|一定要|磨死|套人|套死你|气死你|吓死你|接盘|傻逼|尼玛|王八蛋|脑残|白痴|智障|畜生|杂种|蠢猪|笨屄|贱人|贱货|去死|骗子|爆了$|爆雷|恐慌|凉凉|神经病|扯淡|歇菜|完犊子|提鞋|意音|意淫|lj|LJ|烂到没边|套死你|太差劲|鸡肋')
PER_STOCK_NICK = {
    '601012': re.compile(r'拉基|聋龟|聋鸡|绿鸡|绿狗'),   # 隆基专属黑称,只在隆基生效
}
QIANG = re.compile(r'[!！]|必涨|必跌|恒盘|立帖为证|[?？]|难道|凭什么|为什么|咋|.{0,8}吗$|.{0,8}么$|.{0,8}呢$|谁怕谁|有没有|^(快|赶紧|赶快|别|大家|兄弟们|各位)|(?:快|赶紧|赶快).{0,6}(?:加|买|卖|跑|逃|抄|上|下|进|出)|(?:不要|别).{0,4}(?:卖|割|追|买|碰|信)|记住|太.{1,3}了|真是.{1,4}|棒棒哒|漂亮|厉害[啊呀]|密电|恭喜|佩服|服气|牛[啊呀]|666|呵呵|多正确|早说过|正如我|我说过|被套|套牢|解套|回本|抱团|吐血|后悔|懊恼|悲催|无语|失望|难受|心累|心塞|哭|泪|心疼|受伤|亏[了得哭]|套[了得]|惨|坑|吐了|错过|进早|悔恨|没放弃|追的你|被监控|又被|我不信|不信$|真弱|真垃|学渣|干嘛呢|干啥|啊你$|兄弟|恶心人|气人|可怜|可惜|加油$|磨磨唧唧|墨迹|没格局|真狠|埋人|那么烂|这么烂|真烂|庆幸|该.{0,6}了吧$|来救你|有救了|救你')
BEARW = re.compile(r'跌|空|砸|崩|跑|卖|割|退市|完|破')

def normalize(t):
    if not t: return ''
    t = re.sub(r'[\s~～·・、,，.。:：;；"\'\[\]()（）<>《》\-_=+*|\\/]+', '', t)
    KEEP = {'!', '！', '哈'}
    return re.sub(r'(.)\1{2,}', lambda m: m.group(0) if m.group(1) in KEEP else m.group(1), t)

STOCK_NAMES = {}
with open('/Users/mac/Documents/trae_projects/MyResearcher/pipeline-runs/date-selection-20261004/evaluation/all-method-opportunities.jsonl') as f:
    for line in f:
        d = json.loads(line)
        STOCK_NAMES[d['stock_code']] = d['stock_name']

# ---- 行情 ----
bars = defaultdict(dict)
with open(BARS) as f:
    for r in csv.DictReader(f):
        bars[r['instrument_code']][r['trade_date']] = float(r['close'])

# ---- 聚合(股×日) ----
trade_days = {code: sorted(bars[code].keys()) for code in bars}
def entry_day(stock, day):
    ds = trade_days.get(stock, [])
    for d in ds:
        if d >= day: return d
    return None

cells = defaultdict(lambda: {'n':0,'authors':set(),'ji':0,'qiang':0,'sell':0,'buy':0,'bear':0,'future':False})
conn = sqlite3.connect(f'file:{COLLECTOR}?mode=ro', uri=True)
for title, stock, pub, aid in conn.execute("SELECT title, stock_code, published_at, author_id FROM posts"):
    if not pub: continue
    day = pub[:10]
    ed = entry_day(stock, day)
    if not ed: continue
    key = (stock, ed)
    c = cells[key]
    c['n'] += 1
    c['authors'].add(aid)
    if ed > day: c['future'] = True
    t = normalize(title)
    nick = PER_STOCK_NICK.get(stock)
    if EXCITE.search(t) or (nick and nick.search(t)): c['ji'] += 1
    elif QIANG.search(t): c['qiang'] += 1
    if DECLARE_SELL.search(t): c['sell'] += 1
    if DECLARE_BUY.search(t): c['buy'] += 1
    if BEARW.search(t): c['bear'] += 1
conn.close()

# ---- 按股组装 19 列 ----
stocks = sorted({k[0] for k in cells})
series = []
for stock in stocks:
    days = sorted({k[1] for k in cells if k[0] == stock})
    ji_hist = []
    rows = []
    for day in days:
        c = cells[(stock, day)]
        n = c['n']
        ji = c['ji']/n
        qiang = c['qiang']/n
        sell = c['sell']/n
        buy = c['buy']/n
        bear = c['bear']/n
        calm = max(0.0, 1.0 - ji - qiang)
        small = n < 5
        # 前20日(不含当日)激动密度 Q25/Q75
        hist = [x for x in ji_hist[-20:] if x is not None]
        q25 = q75 = None
        if len(hist) >= 5:
            s = sorted(hist)
            q25 = s[len(s)//4]
            q75 = s[(3*len(s))//4]
        ji_hist.append(None if small else ji)
        price = bars.get(stock, {}).get(day)
        rows.append([day, (None if small else round(ji,6)), q25, q75, None,
                     round(bear,6), round(qiang,6), round(sell,6), round(calm,6),
                     n, len(c['authors']), None, None, None, None, None, False, price, c['future']])
    series.append({'c': stock, 'n': STOCK_NAMES.get(stock, stock), 'p': rows})

dates = [r[0] for s in series for r in s['p']]
document = {
    'metadata': {
        'source_file': COLLECTOR + ' (rules: triage v3.2 + patterns v4.1 + normalize)',
        'imported_at': datetime.now(timezone.utc).isoformat(),
        'snapshot_date': '2026-10-09',
        'model_version': 'rule-triage-v4.3+normalize+按股黑称+交易日映射+小样本置空 (no model, deterministic rules)',
        'stock_count': len(series),
        'point_count': len(dates),
        'entry_start': min(dates),
        'entry_end': max(dates),
        'scoring': '主曲线(情绪净分数位)=激动密度(骂词/恐慌词/急呼规则命中占比);下面板:恐惧=看跌词占比,平静CALM位=强调密度(叹号/反问/被套句式),沮丧位=卖出宣言占比,未明确位=平静密度;观察帖数=当日发帖数,采样帖数=独立作者数;权重=按帖(W0);情绪带=激动密度前20日Q25/Q75',
    },
    'payload': {
        's': series,
        'as_of': '2026-10-09',
        'publication_max': None,
        'manual': {'confirmed': {}, 'pending': {}, 'replaced': {}},
        'price': {'adjustment': 'qfq', 'field': 'close', 'unit': '元', 'source': BARS,
                  'last_trade_date': max(d for b in bars.values() for d in b) if bars else None,
                  'join': ['instrument_code', 'trade_date'], 'missing': 'null; no forward fill'},
    },
}
with open(OUT, 'w') as f:
    json.dump(document, f, ensure_ascii=False)
print(f'输出 {OUT}')
print(f'股票 {len(series)} 只, 点 {len(dates)}, 日期 {min(dates)} ~ {max(dates)}')
for s in series[:5]:
    print(f"  {s['c']} {s['n']}: {len(s['p'])} 天")
