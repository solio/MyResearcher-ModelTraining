"""Freeze supplied K3 sample. No GPU, semantic annotation or remote connection."""
from pathlib import Path
from collections import Counter, defaultdict
import ast
import hashlib
import json
import re
import shutil

D=Path(__file__).resolve().parent
PROJECT=D.parents[4]
K=PROJECT/'worktrees/dashboard-k3-experiments'
EXPORT=K/'exports'
if (D/'results/main/raw.jsonl').exists():
 raise SystemExit('Inference receipts exist: refusing to replace frozen inputs or reset status.')
def digest(b): return hashlib.sha256(b).hexdigest()
def write(name,obj):
 p=D/name;p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def jsonl(name,rows): (D/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))

source=EXPORT/'gpt-compare-sample-20261012.jsonl'
meta=EXPORT/'gpt-compare-sample-20261012-meta.json'
rows=[json.loads(x) for x in source.read_text().splitlines() if x.strip()]
assert len(rows)==388
shutil.copy2(source,D/'private/source.jsonl');shutil.copy2(meta,D/'private/source-meta.json')
models,sidecar=[],[]
days=defaultdict(list)
for i,r in enumerate(rows,1):
 assert set(r['rule_hits'])=={'bearw','declare_sell','excite','declare_buy'}
 assert all(type(x) is bool for x in r['rule_hits'].values())
 assert isinstance(r['title'],str) and r['title'].strip()
 identity=json.dumps([r['stock_code'],r['published_at'],r['title']],ensure_ascii=False,separators=(',',':'))
 ident='K3-'+digest(identity.encode())[:20]
 models.append({'id':ident,'text':r['title'],'stock_code':r['stock_code'],'stock_name':r['stock_name'],'published_at':r['published_at']})
 sidecar.append({'id':ident,'source_row':i,'source_identity_sha256':digest(identity.encode()),**r,
                 'title_length':len(r['title']),'possible_title_cutoff':len(r['title'])==40})
 days[r['stock_code'],r['day']].append(r)
assert len({r['id'] for r in models})==len(models)
assert all(len(rr)==1 for rr in days.values())
write('data/model-inputs.json',{'rows':models})
jsonl('private/evaluation-sidecar.jsonl',sidecar)

schema={'version':'k3-title-semantics-v1','status':'exploratory comparison; not approved production ontology',
 'core_candidates':[
 {'id':'bearish_expression','label':'看跌判断／对下跌的负面评价','hypothesis':'作者认可地表达对所指价格或前景的负向判断，或对下跌、弱势的负面评价。','notes':'单纯提到涨跌、客观引用历史数字、转述别人看法或否认下跌不自动成立；辱骂但无价格或前景含义不自动算看跌。'},
 {'id':'sell_declaration','label':'作者自己的退出／卖出表达','hypothesis':'作者表达自己的卖出、减仓、清仓、退出的操作陈述或明确意向。','notes':'已发生、计划和条件须分别保留。建议他人卖、描述他人卖或否认卖出不等于作者自己的肯定宣言；不要求是全部清仓。'},
 {'id':'high_arousal_expression','label':'明显强烈的情绪／表达','hypothesis':'作者使用明显强烈的情绪表达或修辞，如强烈愤怒、辱骂、恐慌、亢奋、激动。','notes':'可为正面或负面；具体情绪另记。不能仅凭负面观点、黑称、普通讽刺、标点或看涨就自动判强烈。'},
 {'id':'buy_declaration','label':'作者自己的买入／加仓表达','hypothesis':'作者表达自己的买入、加仓、重仓参与的操作陈述或明确意向。','notes':'已发生、计划和条件须分别保留。建议他人买、描述他人买、否认买入或仅提及持仓不自动等于作者的肯定买入宣言。'}],
 'scores':'Intensity and support are separate uncalibrated estimates. Unknown is never zero/neutral. No final scalar sentiment or return weights.',
 'training_allowed':False,'memory_update_allowed':False}
write('runner/meaning_schema.json',schema)

# Audit frozen exported booleans mechanically. These values never enter model inputs.
def norm(t):
 t=re.sub(r'[\s~～·・、,，.。:：;；"\'\[\]()（）<>《》\-_=+*|\\/]+','',t)
 return re.sub(r'(.)\1{2,}',lambda m:m.group(0) if m.group(1) in {'!','！','哈'} else m.group(1),t)
patterns={}
for key in ('bearw','declare_sell','excite'):
 p=K/'wordlists'/f'{key}.v1.re'
 shutil.copy2(p,D/'private'/p.name)
 patterns[key]=re.compile('|'.join(l.strip() for l in p.read_text().splitlines() if l.strip() and not l.startswith('#')))
tree=ast.parse((K/'gen_k3_experiment.py').read_text())
for node in tree.body:
 if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DECLARE_BUY' for t in node.targets):
  patterns['declare_buy']=re.compile(ast.literal_eval(node.value.args[0]))
assert len(patterns)==4
write('private/rule-source.json',{'patterns':{k:v.pattern for k,v in patterns.items()},
 'buy_source':'gen_k3_experiment.py:DECLARE_BUY; export meta did not assign a buy-list version',
 'buy_source_sha256':digest((K/'gen_k3_experiment.py').read_bytes()),
 'notice':'Exported rule booleans remain authoritative for this comparison even if reproduction differs.'})
mismatches=[]
for r in sidecar:
 for key,pat in patterns.items():
  reproduced=bool(pat.search(norm(r['title'])))
  if reproduced!=r['rule_hits'][key]:mismatches.append({'id':r['id'],'rule':key,'exported':r['rule_hits'][key],'reproduced':reproduced})
previous=json.loads((D.parent/'bf16-400-v1/data/stage2-inputs.json').read_text())['rows']
previous_keys={(r['stock_code'],r['published_at']) for r in previous}
overlap=[r['id'] for r in models if (r['stock_code'],r['published_at']) in previous_keys]
audit={'rows':len(rows),'stocks':dict(Counter(r['stock_name'] for r in rows)),
 'tiers':dict(Counter(r['day_context']['tier'] for r in rows)),
 'date_min':min(r['day'] for r in rows),'date_max':max(r['day'] for r in rows),
 'stock_days':len(days),'sampled_per_stock_day':1,'full_days_sampled':0,
 'day_posts_min':min(r['day_context']['day_posts'] for r in rows),'day_posts_max':max(r['day_context']['day_posts'] for r in rows),
 'length_40_count':sum(len(r['title'])==40 for r in rows),'title_cutoff_notice':'40-character cap may be truncated; no automatic completion or replacement with body.',
 'duplicate_titles':len(rows)-len({r['title'] for r in rows}),
 'rule_positive_counts':{k:sum(r['rule_hits'][k] for r in rows) for k in patterns},
 'rule_reproduction_mismatches':mismatches,'prior400_same_stock_timestamp_ids':overlap,
 'sampling_limit':'Provided stratified export lacks selection probabilities and full sampling-frame code; freeze seed/meta without claiming regenerated sampling or corpus prevalence.',
 'missing_provenance':['source_item_id/url','buy rule version in export meta','day_z historical window/ddof and calendar/trading-day convention','inclusion probabilities'],
 'day_extreme_vote':None,'day_extreme_status':'not_estimable_one_sampled_post_per_stock_day'}
write('data-audit.json',audit)
plan={'status':'prepared_not_run','task':'K3 title rule versus BF16 semantic comparison',
 'model':{'repo':'Qwen/Qwen3.8-27B','revision':'1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0','precision':'BF16'},
 'generation':{'temperature':0,'top_p':1,'thinking':'off','max_tokens':2048,'concurrency':8,'timeout':240},
 'inputs':{'n':388,'text':'original title only; unchanged','date_min':audit['date_min'],'date_max':audit['date_max']},
 'comparison':{'bearw':'bearish_expression','declare_sell':'sell_declaration','excite':'high_arousal_expression','declare_buy':'buy_declaration'},
 'blindness':'Inference package contains only titles and stock/time metadata. Rule outputs/day context/tiers/z/private sidecar stay local. Prompt author inspected export structure and examples; not a claim of author blindness. All 388 retained, including 2 possible prior400 stock-timestamp matches, reported separately.',
 'interpretation':'2x2 tables measure rule/model agreement, not accuracy. Model-only positives are review candidates, not proven rule misses. Rule-only positives are also review candidates. Neither predicts future prices here.',
 'decision':'Review both disagreement directions and a reproducible sample of agreements. Only if additional correct observations and removed misreads justify cost, prepare full-day/history aggregation using fixed K3 downstream rules. No model day vote from one title; no automatic whole-corpus expansion.',
 'daily_followup':'Need complete matching daily title populations, stock-specific historical baselines, exact K3 day mapping/extreme rule, posting availability/entry time and fixed forward-return evaluation. Rule-day tier is a sampling stratum, not psychological truth.',
 'scores':'Save scoped observations and separate intensity/support; no arbitrary scalar emotion, no confidence threshold that discards low-score meanings.',
 'operation':'Resume by identical hashes; save every raw receipt before parsed output; malformed rows continue; deadline required; no automatic paid renewal/download or prior80 experiment.',
 'training_allowed':False,'memory_update_allowed':False,
 'source_sha256':{source.name:digest(source.read_bytes()),meta.name:digest(meta.read_bytes())}}
write('plan.json',plan)
write('session-status.json',{'state':'prepared_not_run','expected':388,'attempted':0,'notice':'Input and software prepared locally. Actual node not connected.'})
print(json.dumps({'rows':len(rows),'stock_days':len(days),'rule_mismatches':len(mismatches),'prior400_possible_overlap':len(overlap),'titles_at40':audit['length_40_count']},ensure_ascii=False))

if __name__=='__main__': pass
