"""Read-only assessment of same-stock, same-date posts absent from the rule-hit export."""
from pathlib import Path
import ast,bisect,collections,csv,hashlib,json,sqlite3,sys
ROOT=Path('/Users/mac/Documents/trae_projects/MyResearcher')
K3=ROOT/'worktrees/dashboard-k3-experiments'
T=Path(__file__).resolve().parent.parent
source=[json.loads(l) for l in (T/'private/source.jsonl').read_text().splitlines()]
days={(r['stock_code'],r['day']) for r in source};stocks=sorted({x[0] for x in days});ref=collections.Counter((r['stock_code'],r['published_at'],r['title']) for r in source)
source_days={(r['stock_code'],r['day']):r['day_posts'] for r in source}
calendar=collections.defaultdict(set)
for name in ['pipeline-runs/market-data-v0-training-db-20261002/daily-bars-qfq.csv','pipeline-runs/market-data-v0-20260916/daily-bars-qfq.csv']:
 with (ROOT/name).open() as f:
  for r in csv.DictReader(f):
   if r['instrument_code'] in stocks and ('20260916' not in name or r['trade_date']>'2026-09-30'):calendar[r['instrument_code']].add(r['trade_date'])
calendar={c:sorted(v) for c,v in calendar.items()}
# Import the pure normalization function only; never execute K3 report code.
tree=ast.parse((K3/'gen_per_stock_panel.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='normalize')
import re
ns={'re':re};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<frozen-k3-normalize>','exec'),ns);normalize=ns['normalize']
sys.path.insert(0,str(K3));from wordlists.wordlist_loader import _load,load_nick
bear,_,sha=_load('bearw.v2.re');nick={s:load_nick(s) for s in stocks}
c=sqlite3.connect('file:'+str(ROOT/'MyResearcher-DataCollector/data/collector.db')+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
cells=collections.defaultdict(list);present=collections.Counter()
for r0 in c.execute('select source,source_item_id,stock_code,title,published_at from posts where stock_code in ('+','.join('?'*len(stocks))+')',stocks):
 r=dict(r0);code=r['stock_code'];pub=r['published_at'];i=bisect.bisect_left(calendar[code],pub[:10])
 if i>=len(calendar[code]):continue
 day=calendar[code][i]
 if (code,day) not in days:continue
 title=r['title'] or '';nt=normalize(title);b=bool(bear.search(nt));n=bool(nick[code] and nick[code].search(nt))
 r.update(day=day,rule_hit=bool(b or n),bearw_v2_hit=b,nick_hit=n)
 cells[(code,day)].append(r);present[(code,pub,title)]+=1
rows=[]
for key in sorted(days):
 rs=cells[key];rows.append({'stock_code':key[0],'day':key[1],'previous_day_posts':source_days[key],'current_day_posts':len(rs),'current_hits':sum(r['rule_hit'] for r in rs),'unhit_posts':sum(not r['rule_hit'] for r in rs),'unhit_nonempty_titles':sum(not r['rule_hit'] and bool((r['title'] or '').strip()) for r in rs)})
missing=ref-present
out={'date_groups':len(days),'stocks':stocks,'source_export_rows':len(source),'current_source_posts_in_same_days':sum(len(v) for v in cells.values()),'current_rule_hits':sum(r['rule_hit'] for v in cells.values() for r in v),'current_unhit':sum(not r['rule_hit'] for v in cells.values() for r in v),'original_export_rows_unmatched_in_db':sum(missing.values()),'day_count_mismatches':sum(r['current_day_posts']!=r['previous_day_posts'] for r in rows),'days':rows,'unhit_max_title_length':max([len(r['title'] or '') for v in cells.values() for r in v if not r['rule_hit']] or [0])}
(T/'resume-20261013/next-batch-assessment.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
(T/'resume-20261013/same-days-source-private.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for k in sorted(cells) for r in cells[k]))
print(json.dumps({k:v for k,v in out.items() if k!='days'},ensure_ascii=False));print(json.dumps(rows,ensure_ascii=False))
