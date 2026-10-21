(() => {
  'use strict';
  const base='trials/k3-full-days-analysis-v1/';
  const $=id=>document.getElementById(id);
  const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const n=x=>Number(x).toLocaleString('zh-CN');
  const pct=x=>(100*x).toFixed(1)+'%';
  const labels={agree:'该项判断有依据',disagree:'明确不同意',context_needed:'边界或上下文不足'};
  let days=[],reviews=[],page=0;
  async function read(name){const r=await fetch(base+name,{cache:'no-store'});if(!r.ok)throw Error(name);return r.json()}
  function renderDays(){
    const filtered=days.filter(x=>!$('analysis-stock').value||x.stock_code===$('analysis-stock').value);
    $('analysis-day-rows').innerHTML=filtered.map(x=>{const c=x.model_checks.bearish_expression;const span=c.unknown?`${pct(c.density_lower)}–${pct(c.density_upper)}`:pct(c.density_lower);return `<tr><td>${esc(x.stock_name)}<small>${esc(x.day)}</small></td><td>${x.posts}</td><td>${x.rule_positive}<small>${pct(x.rule_density)}</small></td><td>${c.positive}${c.unknown?` + ${c.unknown} 未知`:''}<small>${span}</small></td><td>${c.target_field_accounting.explicit_stock_string||0}<small>仅按模型对象字段</small></td><td>${c.rule_unhit_and_model_positive}</td></tr>`}).join('');
  }
  function renderReviews(){
    const assessment=$('analysis-review-filter').value;
    const filtered=reviews.filter(x=>!assessment||x.assessment===assessment);
    const max=Math.max(0,Math.ceil(filtered.length/5)-1);page=Math.min(page,max);
    $('analysis-review-count').textContent=`共 ${filtered.length} 条 · 第 ${page+1} / ${max+1} 页`;
    $('analysis-prev').disabled=page===0;$('analysis-next').disabled=page===max;
    $('analysis-review-cards').innerHTML=filtered.slice(page*5,page*5+5).map(x=>{
      const check=x.model_bearish;const positive=check.decision==='positive';
      const why=check.claims?.length?check.claims.map(c=>`“${c.quote}” → ${c.reason}（支持度 ${c.support??'未提供'}；对象 ${c.target}）`).join('；'):'该检查项返回 not_expressed，或只记录其他主体。';
      return `<article class="analysis-example"><div class="statusline"><small>${esc(x.id)} · ${esc(x.stock)} · ${x.cell.startsWith('rule_hit__')?'规则命中':'规则未命中'}</small><span class="pill ${x.assessment==='disagree'?'warn':''}">${labels[x.assessment]}</span></div><p><strong>${esc(x.title)}</strong></p><p>模型统计项：<b>${positive?'存在作者负面判断':'未认定作者负面判断'}</b></p><p class="queue-note">${esc(why)}</p><p><b>本轮复核：</b>${esc(x.review_reason)}</p>${x.additional_issue?`<p class="queue-warn">其他问题：${esc(x.additional_issue)}</p>`:''}<details><summary>查看模型完整原始输出</summary><pre>${esc(JSON.stringify(x.original_output,null,2))}</pre></details></article>`;
    }).join('');
  }
  async function load(){
    try{
      const [summary,d,r,review]=await Promise.all([read('analysis-summary.json'),read('day-summary.json'),read('reviewed-examples.json'),read('review-summary.json')]);
      days=d;reviews=r;
      $('analysis-loading').classList.add('hidden');$('analysis-content').classList.remove('hidden');
      $('analysis-integrity').textContent=`两批 ${n(summary.received)} 条全部保存，缺项 ${summary.missing}；覆盖 ${summary.stock_date_cells} 个股票日期组合，6 条旧回执已核对原输入和原回答。`;
      $('analysis-comparison-rows').innerHTML=Object.values(summary.comparison).map(x=>`<tr><td>${esc(x.label)}</td><td>${n(x.rule_hit.positive)} / 1,808<small>${pct(x.rule_hit.positive/1808)}；未知 ${x.rule_hit.unknown}</small></td><td>${n(x.rule_unhit.positive)} / 2,594<small>${pct(x.rule_unhit.positive/2594)}；未知 ${x.rule_unhit.unknown}</small></td></tr>`).join('');
      const counts=review.assessment_counts;
      $('analysis-review-summary').textContent=`已逐条复核 80 条：${counts.agree} 条该项判断有依据，${counts.disagree} 条明确不同意，${counts.context_needed} 条存在边界或需要上下文。另记录 11 条涉及其他观察、动作或对象的问题，可能与上述分类重叠。`;
      const stocks=[...new Map(days.map(x=>[x.stock_code,x.stock_name])).entries()];
      $('analysis-stock').innerHTML='<option value="">全部 5 股</option>'+stocks.map(([code,name])=>`<option value="${esc(code)}">${esc(name)}</option>`).join('');
      $('analysis-stock').addEventListener('change',renderDays);
      $('analysis-review-filter').addEventListener('change',()=>{page=0;renderReviews()});
      $('analysis-prev').addEventListener('click',()=>{page--;renderReviews()});
      $('analysis-next').addEventListener('click',()=>{page++;renderReviews()});
      renderDays();renderReviews();
    }catch(e){$('analysis-loading').textContent='分析数据暂未读取到，请刷新页面重试；原始回答下载不受影响。'}
  }
  load();
})();
