const $=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const root='trials/bf16-400-v1/';
let inputs=[],labels={},preds={},raws={},notes={},last='',page=0;
const names={self:'作者',other_retail:'其他散户',main_force:'主力',other:'他人',text:'文本话题',unknown:'未知',completed:'已发生',planned:'计划',current:'当前',past:'过去',conditional:'有条件',general:'一般表达',expressed:'表达成立',explicitly_negated:'明确否认',ambiguous:'无法确认'};
const yes=p=>Object.entries(p?.judgments||{}).filter(([k,v])=>v.decision==='supported');
function claim(c){return `<div class="quote">“${esc(c.quote)}”</div><div class="reason">${esc(c.reason||'未给出理由')}</div><div class="scope">${esc(names[c.state]||c.state||'额外观察')} · 主体：${esc(names[c.subject]||c.subject)} · 对象：${esc(c.target)} · 时间：${esc(names[c.time]||c.time)} ${esc(c.time_text||'')}<br>表达强度：${esc(c.intensity??'不适用／未给出')} · 原文支持度：${esc(c.support??'未给出')}</div>`}
function content(p,raw){
 if(!p)return '<p class="muted">尚未处理，不计作未表达。</p>';
 let s='';
 if(p.status!=='ok'){
  s+=`<p class="error">输出异常：${esc(p.error)}</p><p class="muted">原始回答保留在下方，不当作语义正确结果或无情绪。</p>`;
  try{const r=JSON.parse(raw);for(const [k,v] of Object.entries(r.checks||{}))if(v.claims?.length)s+=`<div class="claim"><strong>${esc(labels[k]||k)}</strong>${v.claims.map(claim).join('')}</div>`;for(const c of r.observations||[])s+=`<div class="claim"><strong>额外观察：${esc(c.meaning)}</strong>${claim(c)}</div>`}catch(e){}
 }else{
  const positive=yes(p);s+=positive.length?positive.map(([k,v])=>`<div class="claim"><strong>${esc(labels[k]||k)}</strong>${(v.claims||[]).map(claim).join('')}</div>`).join(''):'<p class="muted">没有肯定的固定候选判断。下方仍可能有额外观察；这不等于情绪中性。</p>';
  const other=Object.entries(p.judgments||{}).filter(([k,v])=>v.state!=='not_expressed'&&v.decision!=='supported');
  s+=(p.observations||[]).map(c=>`<div class="claim"><strong>额外观察：${esc(c.meaning)}</strong>${claim(c)}</div>`).join('');
  s+=`<details><summary>否定／待定</summary>${other.map(([k,v])=>`<div class="claim"><strong>${esc(labels[k]||k)}</strong>${(v.claims||[]).map(claim).join('')}</div>`).join('')||'<p>没有单独记录的否定或待定项。</p>'}</details>`;
 }
 return s+`<details><summary>模型原始 JSON</summary><pre>${esc(raw||'原始回答尚未同步')}</pre></details>`;
}
function render(){
 const all=Object.values(preds),valid=all.filter(p=>p.status==='ok');
 $('summary').textContent=`已处理 ${all.length} / 400 条；结构有效 ${valid.length} 条；输出异常 ${all.length-valid.length} 条。`;
 const q=$('query').value.trim().toLowerCase(),f=$('filter').value,st=$('stock').value;
 const chosen=inputs.filter(i=>{const p=preds[i.id];if(q&&!`${i.id} ${i.text} ${i.stock_name}`.toLowerCase().includes(q))return false;if(st!=='all'&&i.stock_code!==st)return false;if(f==='all')return true;if(!p)return false;if(f==='done')return true;if(f==='error')return p.status!=='ok';if(p.status!=='ok')return false;const y=yes(p);if(f==='positive')return y.length>0;if(f==='emotion')return y.some(([k])=>k.startsWith('emo_'));if(f==='none')return y.length===0;if(f==='uncertain')return Object.values(p.judgments||{}).some(v=>(v.claims||[]).some(c=>c.state==='ambiguous'));return true});
 const totalPages=Math.max(1,Math.ceil(chosen.length/20));page=Math.min(page,totalPages-1);
 $('count').textContent=`符合条件 ${chosen.length} 条`;$('page-number').textContent=`${page+1} / ${totalPages}`;$('prev').disabled=page===0;$('next').disabled=page===totalPages-1;
 $('rows').innerHTML=chosen.slice(page*20,page*20+20).map(i=>{const raw=raws[i.id]?.response?.choices?.[0]?.message?.content;return `<article class="card" id="${esc(i.id)}"><p class="muted">${esc(i.id)} · ${esc(i.stock_name)} · ${esc(i.published_at)}</p><div class="post">${esc(i.text)}</div>${notes[i.id]?`<p class="note">助手复核意见：${esc(typeof notes[i.id]==='string'?notes[i.id]:notes[i.id].note)}</p>`:''}${content(preds[i.id],raw)}</article>`}).join('');
}
async function get(path){const r=await fetch(root+path,{cache:'no-store'});if(!r.ok)return null;return r}
async function update(){try{
 if(!inputs.length){inputs=(await (await get('data/stage2-inputs.json')).json()).rows;const schema=await (await get('frozen/meaning_schema.json')).json();labels=Object.fromEntries(schema.core_candidates.map(x=>[x.id,x.label]));const stocks=[...new Map(inputs.map(r=>[r.stock_code,r.stock_name])).entries()];$('stock').innerHTML='<option value="all">全部股票</option>'+stocks.map(([code,name])=>`<option value="${esc(code)}">${esc(name)}</option>`).join('')}
 const nr=await get('review-notes.json');let notesChanged=false;if(nr){const n=(await nr.json()).notes||{};notesChanged=JSON.stringify(n)!==JSON.stringify(notes);notes=n;}
 const r=await get('results/natural/predictions.jsonl');const text=r?await r.text():'';
 if(text!==last||notesChanged||!$('rows').children.length){last=text;preds=Object.fromEntries(text.trim().split('\n').filter(Boolean).map(x=>{const p=JSON.parse(x);return[p.id,p]}));const rr=await get('results/natural/raw.jsonl');if(rr)raws=Object.fromEntries((await rr.text()).trim().split('\n').filter(Boolean).map(x=>{const p=JSON.parse(x);return[p.id,p]}));render()}
 $('read-error').textContent='';
 }catch(e){$('read-error').textContent='本地备份暂不可读，保留已有显示：'+e.message}}
for(const id of ['query','stock','filter'])$(id).addEventListener(id==='query'?'input':'change',()=>{page=0;render()});
$('prev').onclick=()=>{page--;render();window.scrollTo({top:0})};$('next').onclick=()=>{page++;render();window.scrollTo({top:0})};
const initialId=new URLSearchParams(location.search).get('id');if(initialId){$('query').value=initialId;$('filter').value='all';}
update();setInterval(update,15000);
