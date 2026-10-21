"""Join local-only rule sidecar AFTER inference. Agreement is not accuracy."""
import argparse
from collections import Counter
from pathlib import Path
import html
import hashlib
import json
import random

D=Path(__file__).resolve().parent
def read_rows(p):
 if not p.exists():return []
 rows=[]
 for line in p.read_text().splitlines():
  if line.strip(): rows.append(json.loads(line))
 return rows
def export(out,results):
 frozen=D/'evaluation-manifest.json'
 if frozen.exists():
  for name,h in json.loads(frozen.read_text())['files_sha256'].items():
   if hashlib.sha256((D/name).read_bytes()).hexdigest()!=h:raise ValueError('Frozen evaluation file changed: '+name)
 out.mkdir(parents=True,exist_ok=True)
 side=read_rows(D/'private/evaluation-sidecar.jsonl')
 preds=read_rows(results/'predictions.jsonl');byid={r['id']:r for r in preds}
 if len(byid)!=len(preds):raise ValueError('Duplicate predictions')
 if set(byid)-{r['id'] for r in side}:raise ValueError('Foreign prediction ID')
 plan=json.loads((D/'plan.json').read_text());mapping=plan['comparison']
 audit=json.loads((D/'data-audit.json').read_text())
 matrices={k:Counter() for k in mapping};tiers={};returned=[];candidates=[]
 for src in side:
  p=byid.get(src['id'],{});valid=p.get('status')=='ok'
  obs=p.get('observations',[]) if valid else None
  sem={}
  for rule,signal in mapping.items():
   j=p.get('judgments',{}).get(signal,{}) if valid else {}
   decision=j.get('decision')
   value=True if decision=='supported' else False if decision=='not_supported' else None
   sem[rule]={'value':value,'signal':signal,'decision':decision or 'unavailable', 'claims':j.get('claims',[]) if valid else None}
   if p:
    key=f'rule_{str(src["rule_hits"][rule]).lower()}__model_{str(value).lower() if value is not None else "unknown"}'
    matrices[rule][key]+=1
    tiers.setdefault(src['day_context']['tier'],{}).setdefault(rule,Counter())[key]+=1
   if valid and value is not None and value!=src['rule_hits'][rule]:
    candidates.append({'id':src['id'],'signal':rule,'direction':'model_only' if value else 'rule_only','review_status':'unreviewed'})
  categories={kind:[o['meaning'] for o in (obs or []) if o['kind']==kind and o['state']=='expressed' and (o['subject']=='self' or kind=='style' and o['subject']=='text')] if valid else None for kind in ['emotion','stance','action','style']}
  r={**src,'inference_status':p.get('status','pending'),'error':p.get('error'),
     'emotions':categories['emotion'],'stances':categories['stance'],'actions':categories['action'],'styles':categories['style'],
     'confidence':None,'confidence_note':'No fabricated overall probability. Per-observation and scoped-claim support values are uncalibrated model self-estimates.',
     'projection_scope':'Top-level emotion/stance/action lists include explicit author-self observations only; others/unknown/negation remain in observations. An empty list never means calm.',
     'observations':obs,'semantic_comparison':sem,
     'day_extreme_vote':None,'day_extreme_status':'not_estimable_one_sampled_post_per_stock_day',
     'possible_prior400_overlap':src['id'] in audit['prior400_same_stock_timestamp_ids'],
     'model_manifest':'results/main/manifest.json','semantic_review_status':'unreviewed'}
  returned.append(r)
 (out/'k3-return.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in returned))
 summary={'expected':len(side),'attempted':len(preds),'structure_valid':sum(p.get('status')=='ok' for p in preds),
 'pending':len(side)-len(preds),'all_attempted':len(side)==len(preds),
 'rule_model_agreement_counts':matrices,'by_sampling_tier':tiers,
 'unreviewed_disagreement_claims':len(candidates),'possible_prior400_overlap_n':len(audit['prior400_same_stock_timestamp_ids']),
 'notice':'Counts are agreement/disagreement, not TP/FP, accuracy or improved stock-return forecasts. Model-only observations are candidates for review, not established rule misses. Unknown/format failure/pending never become calm or negative.',
 'day_extreme_vote_available':False,'scores_calibrated':False}
 # Reproducible independent agreement audit, not only handpicked disagreements.
 disagree_ids={r['id'] for r in candidates}
 agreements=[r['id'] for r in returned if r['inference_status']=='ok' and r['id'] not in disagree_ids
             and all(s['value'] is not None for s in r['semantic_comparison'].values())]
 random.Random(20261012).shuffle(agreements)
 queue={'disagreements':candidates,'agreement_audit_ids':agreements[:min(40,len(agreements))],
        'all_unavailable_or_uncertain_ids':[r['id'] for r in returned if r['inference_status'] not in ('pending',) and
            (r['inference_status']!='ok' or any(x['value'] is None for x in r['semantic_comparison'].values()))],
        'notice':'Review against title/evidence first; scores and labels are not gold. Review every disagreement plus sampled agreements; do not calculate population accuracy from this selective review.'}
 for name,obj in [('summary.json',summary),('review-queue.json',queue)]:
  (out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
 payload=json.dumps({'rows':returned,'summary':summary},ensure_ascii=False).replace('<','\\u003c')
 page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>K3 × 27B BF16 对照</title><style>body{font:18px/1.7 system-ui;background:#f2f5f8;color:#192e42;margin:0}main{max-width:1050px;margin:auto;padding:25px 20px}article,.box{background:white;border:1px solid #d9e1e8;border-radius:12px;padding:20px;margin:18px 0}h1{font-size:30px}h2{font-size:21px}.post{font-size:24px;white-space:pre-wrap;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:14px}input,select,button{font:inherit;padding:8px;max-width:100%;box-sizing:border-box}a{color:#1965a7}.muted{color:#587086}.toolbar{display:flex;gap:10px;flex-wrap:wrap}.obs{border-top:1px solid #dde5ed;padding-top:12px;margin-top:12px}</style><main><h1>K3规则 × 27B BF16：同一批标题</h1><div class="box"><b id="progress"></b><p>处理进度不是通过率；规则与模型一致不是正确率。仅对照标题，规则答案和当日z值没有提供给模型。</p><p>388个股票日期各抽1帖，不能据此判断全天极端。日级投票统一留空。每项保留证据、理由与强度；支持度未经校准。</p><a href="../plan.json">试验说明</a> · <a href="k3-return.jsonl">回传JSONL</a> · <a href="summary.json">对照计数</a></div><div class="toolbar"><input id="q" placeholder="搜索股票、标题、编号"><select id="mode"><option value="all">全部</option><option value="disagree">规则与模型有分歧</option><option value="done">已返回</option><option value="error">输出异常</option></select><button id="prev">上一页</button><span id="page"></span><button id="next">下一页</button></div><div id="rows"></div><script>const data=PAYLOAD;let page=0;const labels={bearw:'看跌/下跌负评',declare_sell:'自己卖出表达',excite:'强烈表达（正负均可）',declare_buy:'自己买入表达'};const q=document.querySelector('#q'),mode=document.querySelector('#mode');const add=(el,tag,text,cls)=>{const e=document.createElement(tag);e.textContent=text;if(cls)e.className=cls;el.append(e);return e};function render(){const s=data.summary;document.querySelector('#progress').textContent=`已返回 ${s.attempted} / ${s.expected}；结构有效 ${s.structure_valid}；语义质量待复核`;const rows=data.rows.filter(r=>(!q.value||[r.stock_name,r.title,r.id].join(' ').includes(q.value))&&(mode.value==='all'||mode.value==='done'&&r.inference_status!=='pending'||mode.value==='error'&&r.inference_status==='error'||mode.value==='disagree'&&Object.entries(r.semantic_comparison).some(([k,v])=>v.value!==null&&v.value!==r.rule_hits[k])));page=Math.min(page,Math.max(0,Math.ceil(rows.length/10)-1));document.querySelector('#page').textContent=`${page+1} / ${Math.max(1,Math.ceil(rows.length/10))}（${rows.length}帖）`;const root=document.querySelector('#rows');root.replaceChildren();for(const r of rows.slice(page*10,page*10+10)){const a=add(root,'article','');add(a,'h2',r.stock_name+' · '+r.day+' · '+r.id);add(a,'p',r.title,'post');add(a,'p','状态：'+r.inference_status+(r.error?' · '+r.error:''));for(const[k,v]of Object.entries(r.semantic_comparison)){add(a,'p',labels[k]+'：规则 '+(r.rule_hits[k]?'命中':'未命中')+'；模型 '+(v.value===null?'待定/未返回':v.value?'明确表达':'未明确表达'));for(const c of v.claims||[])add(a,'p',`「${c.quote}」 ${c.reason}（${c.subject} / ${c.target} / ${c.time}；支持度 ${c.support}）`,'muted')}for(const o of r.observations||[]){const d=add(a,'div','','obs');add(d,'b',o.meaning+' · '+o.kind+' · '+o.state);add(d,'p','「'+o.quote+'」 '+o.reason);add(d,'p',`主体 ${o.subject}；对象 ${o.target}；时间 ${o.time}；强度 ${o.intensity??'不适用'}；支持度 ${o.support??'未知'}`,'muted')}if(r.possible_title_cutoff)add(a,'p','标题长度40字，可能不完整；没有补正文。','muted');add(a,'p','抽样层：'+r.day_context.tier+'；当天共 '+r.day_context.day_posts+' 帖，本次1帖；日级判断：证据不足。','muted')}}q.oninput=mode.onchange=()=>{page=0;render()};document.querySelector('#prev').onclick=()=>{page=Math.max(0,page-1);render()};document.querySelector('#next').onclick=()=>{page++;render()};render();</script></main></html>'''.replace('PAYLOAD',payload)
 (out/'review.html').write_text(page)
 return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=D/'results/main');p.add_argument('--out',type=Path,default=D/'exports');a=p.parse_args()
 print(json.dumps(export(a.out,a.results),ensure_ascii=False))
