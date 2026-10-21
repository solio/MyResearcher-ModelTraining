"""Build paired review without exposing per-item A/B identity to the reviewer."""
from pathlib import Path
import hashlib,json,random
D=Path(__file__).resolve().parent
def main():
 out=D/'exports';out.mkdir(exist_ok=True)
 rows=json.loads((D/'heldout-inputs.json').read_text())['rows'];refs={r['id']:r for r in json.loads((D/'reference-pre-output.json').read_text())['rows']}
 keypath=D/'blind-key.json'
 if not keypath.exists():
  rng=random.Random(2026101211);keys={r['id']:rng.sample(['A','B'],2) for r in rows};keypath.write_text(json.dumps(keys,indent=2)+'\n')
 keys=json.loads(keypath.read_text());p=D/'results/paired/predictions.jsonl'
 preds=[json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
 by={r['task_id']:r for r in preds}
 if len(by)!=len(preds):raise ValueError('Duplicate task ID in paired results')
 pairs=[]
 for row in rows:
  pair={'id':row['id'],'stock_name':row['stock_name'],'published_at':row['published_at'],'text':row['text'],'reference':refs[row['id']],'versions':{}}
  for shown,arm in zip(['X','Y'],keys[row['id']]):
   result=by.get(row['id']+'::'+arm)
   pair['versions'][shown]={k:result.get(k) for k in ['status','error','checks','observations','finish_reason']} if result else {'status':'pending'}
  pairs.append(pair)
 summary={'expected_requests':160,'attempted_requests':len(preds),'format_valid':sum(r['status']=='ok' for r in preds),'paired_samples':sum(all(r['id']+'::'+a in by for a in ['A','B']) for r in rows),'expected_samples':80,'quality_status':'unreviewed','notice':'More observations or valid JSON is not an A/B win. X/Y are randomly assigned per post; evaluate against original text and provisional pre-output reference.'}
 for n,x in [('summary.json',summary),('blind-pairs.json',pairs)]:
  temp=out/(n+'.tmp');temp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');temp.replace(out/n)
 page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>80条原计划：配对复核</title><style>body{font:18px/1.7 system-ui;margin:0;background:#f3f6fa;color:#1b3048}main{max-width:1350px;margin:auto;padding:25px 20px}article,.card{background:white;border:1px solid #dbe2e9;border-radius:10px;padding:18px;margin:18px 0}.post{font-size:23px;white-space:pre-wrap}.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px}.version{min-width:0;background:#f1f6fa;padding:15px;border-radius:8px}.meaning{border-top:1px solid #dce4eb;margin-top:12px;padding-top:10px}h1{font-size:29px}h2{font-size:23px}p{overflow-wrap:anywhere}input,button{font:inherit;padding:8px;max-width:100%;box-sizing:border-box}.toolbar{display:flex;flex-wrap:wrap;gap:10px}.muted{color:#596e83}a{color:#1463a8}@media(max-width:800px){.pair{grid-template-columns:1fr}}</style><main><h1>原计划：80条新帖，两版配对复核</h1><div class="card"><b id="status"></b><p>两版看同一原文，模型与生成参数相同。先完整观察是否更好，要对原文逐项复核；输出更多不能自动算赢。</p><p>X/Y在每条帖子内随机对应版本，暂不显示A/B名称。参照在推理前写好，属于助手初稿，不是真人金标准。</p><a href="../../../k3-bf16-ready.html">返回串联进度</a></div><div class="toolbar"><input id="q" placeholder="搜索股票、原文、编号"><button id="prev">上一页</button><span id="page"></span><button id="next">下一页</button></div><div id="rows"></div><script>let rows=[],page=0;const q=document.querySelector('#q');const add=(p,t,s,c)=>{const e=document.createElement(t);e.textContent=s;if(c)e.className=c;p.append(e);return e};function show(){const list=rows.filter(r=>[r.id,r.stock_name,r.text].join(' ').includes(q.value));page=Math.min(page,Math.max(0,Math.ceil(list.length/5)-1));document.querySelector('#page').textContent=`${page+1} / ${Math.max(1,Math.ceil(list.length/5))}`;const root=document.querySelector('#rows');root.replaceChildren();for(const r of list.slice(page*5,page*5+5)){const a=add(root,'article','');add(a,'h2',r.stock_name+' · '+r.id+' · '+r.published_at.slice(0,10));add(a,'p',r.text,'post');const ref=add(a,'details','');add(ref,'summary','推理前写的参照（可有争议）');add(ref,'p','应检查：'+r.reference.clear_meanings_to_check.join('；'));add(ref,'p','边界：'+r.reference.boundaries_or_unresolved.join('；'));const grid=add(a,'div','','pair');for(const[name,v]of Object.entries(r.versions)){const col=add(grid,'div','','version');add(col,'h2','版本 '+name+' · '+v.status);if(v.error)add(col,'p',v.error);for(const[k,x]of Object.entries(v.checks||{})){for(const c of x.claims||[]){const d=add(col,'div','','meaning');add(d,'b',k+' · '+c.state);add(d,'p','「'+c.quote+'」 '+c.reason);add(d,'p',`${c.subject} / ${c.target} / ${c.time}；强度 ${c.intensity??'不适用'}；支持度 ${c.support??'未知'}`,'muted')}}for(const o of v.observations||[]){const d=add(col,'div','','meaning');add(d,'b',o.meaning);add(d,'p','「'+o.quote+'」 '+o.reason);add(d,'p',`${o.subject} / ${o.target} / ${o.time}；强度 ${o.intensity??'不适用'}；支持度 ${o.support??'未知'}`,'muted')}}}}async function refresh(){try{const [s,r]=await Promise.all([fetch('summary.json',{cache:'no-store'}).then(x=>x.json()),fetch('blind-pairs.json',{cache:'no-store'}).then(x=>x.json())]);rows=r;document.querySelector('#status').textContent=`已返回 ${s.attempted_requests}/160 次请求；两版都返回的帖子 ${s.paired_samples}/80；尚未作出语义优劣判定。`;show()}catch(e){document.querySelector('#status').textContent='等待结果同步'}}q.oninput=()=>{page=0;show()};document.querySelector('#prev').onclick=()=>{page=Math.max(0,page-1);show()};document.querySelector('#next').onclick=()=>{page++;show()};refresh();setInterval(refresh,15000);</script></main></html>'''
 (out/'review.html').write_text(page)
 print(json.dumps(summary))
if __name__=='__main__':main()
