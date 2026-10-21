const $=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const root='trials/complete-extraction-v1/results/semantic/';
let baseline=null,predictions={},raws={},lastText=null,opened=new Set(),notes={};
const names={self:'作者',other_retail:'其他散户',main_force:'主力',other:'他人',text:'文本话题',unknown:'未知',completed:'已发生',planned:'计划',current:'当前',past:'过去',conditional:'有条件',general:'一般表达',expressed:'表达成立',explicitly_negated:'明确否认',ambiguous:'无法确认',not_expressed:'未表达'};
const positives=p=>Object.entries(p?.judgments||{}).filter(([k,v])=>v.decision==='supported').map(([k])=>k);
function statement(c){return `<div class="quote">“${esc(c.quote||'无原文引用')}”</div><div>${esc(c.reason||'原输出未提供理由')}</div><div class="scope">${esc(names[c.state]||c.state||'')} · 主体：${esc(names[c.subject]||c.subject||'未提供')} · 对象：${esc(c.target||'未提供')} · 时间：${esc(names[c.time]||c.time||'未提供')}${c.time_text?'（'+esc(c.time_text)+'）':''}<br>表达强度：${c.intensity??'未给出'} · 原文支持度：${c.support??'未给出'}</div>`}
function panel(p,raw,ident,side){
 if(!p)return '<p class="muted">尚未返回，不计作未表达。</p>';
 let out='';
 if(p.status!=='ok'){
  out+=`<p class="error">输出异常：${esc(p.error||'结构未通过')}</p>`;
  try{const source=JSON.parse(raw);if(source.checks){out+='<p class="muted">下方直接展示异常原始回答中的表达，未计入结构有效结果，也未修改原文。</p>';for(const [k,v] of Object.entries(source.checks)){if(Array.isArray(v.claims))out+=`<div class="claim"><strong>${esc(baseline.labels[k]||k)}</strong>${v.claims.map(statement).join('')}</div>`}for(const c of source.observations||[])out+=`<div class="claim"><strong>额外观察：${esc(c.meaning)}</strong>${statement(c)}</div>`}}catch(e){}
 }
 else{
  const yes=positives(p);out+=yes.length?yes.map(k=>{const v=p.judgments[k],claims=(v.claims?.length?v.claims:[v]).filter(c=>c.state==='expressed'||!c.state);return `<div class="claim"><strong>${esc(baseline.labels[k]||k)}</strong>${claims.map(statement).join('')}</div>`}).join(''):'<p class="muted">没有肯定的候选判断。否定、不确定或额外观察请展开查看。</p>';
  const other=Object.entries(p.judgments||{}).filter(([k,v])=>v.state!=='not_expressed'&&v.decision!=='supported');
  out+=`<details data-key="${ident}-${side}-other"><summary>否定／不确定与额外观察</summary>${other.map(([k,v])=>`<div class="claim"><strong>${esc(baseline.labels[k]||k)}</strong>${(v.claims?.length?v.claims:[v]).map(statement).join('')}</div>`).join('')}${(p.observations||[]).map(c=>`<div class="claim"><strong>${esc(c.meaning)}</strong>${statement(c)}</div>`).join('')||'<p class="muted">没有额外观察。</p>'}</details>`;
 }
 out+=`<details data-key="${ident}-${side}-raw"><summary>查看原始输出</summary><pre class="raw">${esc(raw||'尚未同步原始输出')}</pre></details>`;
 return out;
}
function render(){
 if(!baseline)return;
 document.querySelectorAll('details[data-key]').forEach(e=>e.open?opened.add(e.dataset.key):opened.delete(e.dataset.key));
 const values=Object.values(predictions),valid=values.filter(r=>r.status==='ok').length;
 const reasons=values.reduce((n,p)=>n+Object.values(p.judgments||{}).flatMap(v=>v.claims||[]).filter(c=>c.state==='expressed'&&c.reason).length,0);
 $('summary').innerHTML=`<strong>新版已返回 ${values.length} / 80 条，结构有效 ${valid} 条，异常 ${values.length-valid} 条。</strong><br>已保存 ${reasons} 项附理由的肯定记录。原版 80 条仍保持原样。`;
 const q=$('query').value.trim().toLowerCase(),filter=$('filter').value,source=$('source').value;
 const chosen=baseline.rows.filter(r=>{const i=r.input,n=predictions[i.id],a=positives(r.prediction).sort(),b=positives(n).sort();if(q&&!`${i.id} ${i.stock_name} ${i.text}`.toLowerCase().includes(q))return false;if(source==='real'&&!i.id.startsWith('S1R')||source==='synthetic'&&!i.id.startsWith('S1C'))return false;if(filter==='done'&&!n)return false;if(filter==='error'&&(!n||n.status==='ok'))return false;if(filter==='changed'&&(!n||n.status!=='ok'||r.prediction?.status!=='ok'||JSON.stringify(a)===JSON.stringify(b)))return false;return true});
 $('visible').textContent=`显示 ${chosen.length} 条`;
 $('rows').innerHTML=chosen.map(r=>{const i=r.input,n=predictions[i.id],a=positives(r.prediction),b=positives(n),added=b.filter(k=>!a.includes(k)),removed=a.filter(k=>!b.includes(k));const diff=n&&n.status==='ok'&&r.prediction.status==='ok'?`新版新增：${added.map(k=>baseline.labels[k]).join('、')||'无'}；新版不再肯定：${removed.map(k=>baseline.labels[k]).join('、')||'无'}。这些是变化，不自动判定改善或退化。`:'尚不具备双方结构有效的对照结果。';const raw=raws[i.id]?.response?.choices?.[0]?.message?.content;return `<article class="card" id="${i.id}"><div class="muted">${esc(i.id)} · ${esc(i.stock_name||'人工对照')} · ${esc(i.published_at||'人工编写对照句')}</div><div class="post">${esc(i.text)}</div><p class="diff">${esc(diff)}</p><div class="pair"><section class="prediction"><h3>原版</h3>${panel(r.prediction,r.raw_content,i.id,'old')}</section><section class="prediction"><h3>新版</h3>${panel(n,raw,i.id,'new')}</section></div><details data-key="${i.id}-reference"><summary>暂定参照与之前的复核说明</summary><p>参照肯定项：${positives(r.reference).map(k=>esc(baseline.labels[k])).join('、')||'无'}</p><p>${esc(r.review_note)}</p><p class="muted">以上是之前的助手参照与复核意见，未因新版输出而修改。</p></details></article>`}).join('');
 document.querySelectorAll('details[data-key]').forEach(e=>e.open=opened.has(e.dataset.key));
 showNotes();
}
function showNotes(){document.querySelectorAll('.new-review-note').forEach(e=>e.remove());for(const [id,note] of Object.entries(notes)){const card=document.getElementById(id);if(card){const p=document.createElement('p');p.className='hint new-review-note';p.textContent='本轮复核（助手意见）：'+note;card.appendChild(p)}}}
async function update(){try{
 if(!baseline){const r=await fetch('results/quality-review.json',{cache:'no-store'});if(!r.ok)throw Error('原版记录尚不可读');baseline=await r.json()}
 const nr=await fetch('results/semantic-review-notes.json',{cache:'no-store'});if(nr.ok)notes=(await nr.json()).notes||{};
 const r=await fetch(root+'predictions.jsonl',{cache:'no-store'});if(!r.ok)throw Error('新版结果尚未同步到本地');const text=await r.text();
 if(text!==lastText){lastText=text;predictions=Object.fromEntries(text.trim().split('\n').filter(Boolean).map(x=>{const d=JSON.parse(x);return[d.id,d]}));const rr=await fetch(root+'raw.jsonl',{cache:'no-store'});if(rr.ok)raws=Object.fromEntries((await rr.text()).trim().split('\n').filter(Boolean).map(x=>{const d=JSON.parse(x);return[d.id,d]}));render()}
 showNotes();
 $('read-error').textContent='';
}catch(e){$('read-error').textContent=e.message+'；保留上次显示。'}}
for(const id of ['query','filter','source'])$(id).addEventListener(id==='query'?'input':'change',render);
update();setInterval(update,15000);
