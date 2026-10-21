"""Read-only audit of final attempts; no relabeling, inference, or training."""
import html
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / 'qwen38-staged-20261010/data'
sys.path.insert(0, str(ROOT / 'deployment/runner-tolerant'))
from evaluate import evaluate

def read(path):
    return json.loads(path.read_text())

inputs = read(DATA / 'stage1-inputs.json')['rows']
references = read(DATA / 'stage1-reference.json')['rows']
schema = read(ROOT / 'deployment/runner-tolerant/meaning_schema.json')
labels = {x['id']: x['label'] for x in schema['core_candidates']}
predictions, raw, batches = {}, {}, {}
for batch in ['stage1', 'stage1-structured', 'stage1-continuation', 'stage1-recovery']:
    for line in (ROOT / 'results' / batch / 'predictions.jsonl').read_text().splitlines():
        item = json.loads(line)
        predictions[item['id']] = item
        batches[item['id']] = batch
    for line in (ROOT / 'results' / batch / 'raw.jsonl').read_text().splitlines():
        item = json.loads(line)
        raw[item['id']] = item['response']['choices'][0]['message'].get('content', '') if item.get('response', {}).get('choices') else ''

evaluation = evaluate(inputs, references, list(predictions.values()), schema)
assert evaluation['totals'] == read(ROOT / 'results/combined-evaluation.json')['totals']
groups = {}
for prefix in ['S1R', 'S1C']:
    e = evaluate([x for x in inputs if x['id'].startswith(prefix)],
                 [x for x in references if x['id'].startswith(prefix)],
                 [x for x in predictions.values() if x['id'].startswith(prefix)], schema)
    groups[prefix] = {k: e[k] for k in ['input_rows', 'valid_prediction_rows', 'totals', 'per_group']}

# Current assistant review notes, NOT a replacement for the provisional references.
notes = {
 'S1R001': '已卖出识别正确，但漏了“全卖了”中的清仓和 229.4 价位；一句话可以同时表达多项。',
 'S1R002': '后悔识别正确，漏了“反弹”这一走势话题。',
 'S1R006': '已买入正确。“主力给点力”是否算资金话题，旧参照本身标为不确定，不能算已证明的误判。',
 'S1R007': '明确错判：“不要加仓”不等于自己准备清仓；还漏了“25以下”的价位信息。清仓支持度却给到 0.85。',
 'S1R010': '参照不记后悔，模型记了后悔。全句含卖后上涨及自嘲，存在解释空间；单独引用“亏了本金一半”不足以证明后悔。',
 'S1R013': '清仓正确，但漏了清仓同时包含已卖出的事实。',
 'S1R015': '资金话题正确；换手率、量能是否归入价格／技术走势，是定义边界问题，不能简单把参照当真理。',
 'S1R016': '资金正确；对作者是否看空，参照本身不确定。',
 'S1R019': '理解基本正确：说明中明确写了“自述已完成卖出操作”，也正确否定后悔；但正式已卖出字段漏掉了。',
 'S1R020': '希望止跌正确，价格／走势字段漏掉。',
 'S1R023': '说明中正确读出了希望价格稳定；正式价格字段漏掉，并自行给“止跌”加了隐含上涨条件，未遵守本轮定义。',
 'S1R024': '后悔正确；错误地以“表达后悔而非陈述买入”为理由排除“进了这个”的已买入意思。两者可以同时成立。',
 'S1R025': '识别做T计划和价位，没有把尚待尝试的计划当已卖出。',
 'S1R029': '大仓位卖了但仍有保留，不能记作全部清仓。输出忽略后半句；清仓支持度给到 0.90。原帖“户”字原样保留。',
 'S1R031': '正式失望字段的主体是作者，正确；补充说明却把同一失望的主体写成 other，自相矛盾。仅看标签对照不会发现。另有大量“没提到”被错写成“明确否认”。',
 'S1R032': '清仓和已卖出正确；“一直亏”算不算价格话题涉及个人盈亏与股价描述的边界。',
 'S1R033': '识别清仓计划正确；仅凭“给机会吗”推成价格话题证据偏弱。',
 'S1R035': '已买入和走势正确，没有因为“吓唬人”便认定作者自己恐惧。',
 'S1R036': '已卖出正确；个人亏损比例与股价涨跌话题的边界未统一。',
 'S1R037': '已买入正确；希望慢跌以继续买入，不等于希望止跌。后者被错判，支持度仍为 0.90。',
 'S1R038': '模型读出了好业绩、股价下跌和困惑。参照把赞同业绩算看多，模型把看多收窄为看好未来股价；是口径不一致，不能直接当作理解失败。',
 'S1R039': '清仓、已卖出与参照一致；但说明把“留下的”解释成未持有者，还推测羡慕，含义错误。标签一致不代表整条正确。',
 'S1R043': '失望和下跌走势正确；负面评价是否额外算看空存在边界争议，应区分当前状态与未来判断。',
 'S1R045': '兴奋正确。“一枝独秀”是否足以推导看多和价格走强需要统一口径，不能把这两项都视作无争议错误。',
 'S1R046': '怕踏空已被识别为 FOMO；参照要求再记普通恐惧造成漏项。另有明确错判：希望价格下来好上车，被反向记成希望止跌。',
 'S1R047': '已识别 FOMO 和价位；所谓恐惧漏判是参照同时要求 FOMO＋恐惧。',
 'S1R049': '原始输出识别了后悔、卖出北京君正、买入／加仓所指股票，但买入和卖出又同时放进否定列表，结构自相矛盾。不能当合格结果，也不能说完全读不懂。',
 'S1C001': '最简单的已买入句，原始输出同时在肯定和否定列表放入已买入；否定栏的理由实际上也说已完成买入。这是输出一致性错误。',
 'S1C002': '没有把明确否认买入当成已买入。',
 'S1C003': '没有把明天准备买入当成已买入；计划在说明里。',
 'S1C004': '说明中正确记录他人买入，没有归给作者；但固定“其他散户”字段漏掉。“他”身份本身也依赖本对照句的预设。',
 'S1C005': '卖出一半正确记为已卖出，没有记成清仓。',
 'S1C007': '计划卖出放在说明里，未当成已卖出。',
 'S1C008': '说明中读出了老王卖出，未当作者卖出；固定其他散户字段漏掉，说明里的对象与行为主体也有混用。',
 'S1C010': '清仓计划与尚未卖出区分正确。',
 'S1C011': '条件清仓在说明中读对了，却被正式字段排除；本轮清仓定义明明允许条件计划。另漏价位。',
 'S1C013': '识别做T，却擅自要求买入／卖出必须改变净持仓，错误排除了实际发生的买卖。多项含义不应互斥。',
 'S1C014': '识别逆T表达，但以优先归做T为由排除买入／卖出；多项含义可同时成立。',
 'S1C015': '做T计划识别正确，没有当成已交易。',
 'S1C017': '正确否定作者害怕，漏掉下跌的话题；说明里保留了不害怕。',
 'S1C018': '正确区分别人害怕和作者不害怕，说明完整；固定其他散户字段漏掉。0/3 不等于完全读不懂他人情绪。',
 'S1C020': '正确否定后悔，卖早了只进入说明，没有进入已卖出字段。',
 'S1C022': '失望正确；亏损是否算价格话题是参照与预测的边界分歧。',
 'S1C024': '怕错过上涨机会识别为 FOMO，未同时记普通恐惧，形成重复定义引起的漏项。',
 'S1C025': '看多及上涨预测正确；笼统“公司前景”是否足够算基本面，需要统一口径。',
 'S1C027': '陈述涨跌只记价格，没有自动推成作者看空。',
 'S1C029': '经营话题正确，没有自动推成看多。',
}
clear = {'S1R007', 'S1R029', 'S1R037', 'S1R046'}
cases = defaultdict(list)
for c in evaluation['cases']:
    cases[c['id']].append(c)
refs = {x['id']: x for x in references}
rows = []
for item in inputs:
    ident = item['id']
    rows.append({'input': item, 'prediction': predictions[ident], 'reference': refs[ident],
                 'differences': cases[ident], 'batch': batches[ident], 'raw_content': raw[ident],
                 'review_note': notes.get(ident, '本轮固定字段的肯定／否定与暂定参照一致；这不是对全部对象、时态、分数或说明的正确性认证。'),
                 'highlight_clear_error': ident in clear})
payload = {'created_at': datetime.now(timezone.utc).isoformat(), 'reviewer': 'current_assistant_not_human_gold',
           'notice': '固定标签指标未覆盖所有主体、对象、时态和补充说明；分数未经校准。没有改写原参照或原输出。',
           'evaluation': evaluation, 'sample_groups': groups, 'labels': labels, 'rows': rows}
(ROOT / 'results/quality-review.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')

page = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>80 条逐条质量复核</title><style>
:root{color-scheme:light}body{margin:0;background:#f2f5f8;color:#1e293b;font:18px/1.75 system-ui,sans-serif}main{max-width:1120px;margin:auto;padding:32px 24px}h1{font-size:32px;margin:0}h2{font-size:23px}p{margin:12px 0}section,article{background:white;border:1px solid #d7dfe8;border-radius:12px;padding:22px;margin:18px 0}a{color:#175bab}.muted{color:#64748b;font-size:16px}.warn{background:#fff4e9;border-color:#efc090}.bad{color:#a62323}.good{color:#177050}.chips{display:flex;gap:8px;flex-wrap:wrap}.chip{background:#edf2f8;padding:4px 10px;border-radius:6px;font-size:16px}blockquote{font-size:22px;margin:14px 0;padding:12px 18px;background:#f6f8fb;border-left:4px solid #6581aa;white-space:pre-wrap;overflow-wrap:anywhere}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:10px;border-bottom:1px solid #e2e8f0;vertical-align:top}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:15px/1.6 ui-monospace,monospace;max-height:500px;overflow:auto}select,input{font:17px system-ui;padding:10px;border:1px solid #cbd5e1;border-radius:6px;max-width:100%;box-sizing:border-box}nav{position:sticky;top:0;background:#f2f5f8f5;padding:14px 0;display:flex;gap:10px;flex-wrap:wrap;z-index:1}.columns{display:grid;grid-template-columns:1fr 1fr;gap:18px}.quote{color:#475569}summary{cursor:pointer;color:#175bab}.note{background:#fff7df;padding:12px;border-radius:6px}li{margin:6px 0}@media(max-width:760px){.columns{grid-template-columns:1fr}main{padding:20px 12px}table{font-size:15px}}
</style><main><a href="progress.html">返回下载与运行进度</a><h1>这 80 条，质量到底怎么样？</h1>
<p class="muted">读取最终 80 次逐条结果，同时看原文、固定字段、补充说明和原始输出。原参照与结果均保留。</p>
<section><h2>明确情绪识别有可用表现；完整结构化提取还不可靠。</h2><p>78 条输出结构有效，2 条包含自相矛盾字段。<strong>78 不是答对条数。</strong>本批是 50 条定向挑选的真实短帖＋30 条人工编写的对照句，参照由助手编写，尚非用户确认的标准答案。</p>
<table><thead><tr><th>范围</th><th>模型报出的肯定项中，与参照相符</th><th>参照要求的肯定项中，模型找到</th></tr></thead><tbody id="metrics"></tbody></table>
<p class="muted">一帖可能有多项。指标只统计 78 条结构有效结果；不包含 2 条异常，也不评价所有主体、对象、时态、补充说明和分数。运行中增加过 JSON 约束，本表是故障诊断汇总，不是统一配置的正式测评。</p></section>
<section class="warn"><h2>不能只看总分的四件事</h2><ol><li><strong>3 项“漏掉恐惧”均已识别成 FOMO。</strong>参照要求恐惧与 FOMO 同时成立，模型只给 FOMO。这说明重叠定义执行不一致，不能说它不认识怕踏空。</li><li><strong>其他散户 0/3，是固定字段漏填。</strong>三句的说明里都记录了他人的动作／情绪；理解与可供程序使用的输出不是一回事。</li><li><strong>存在明显错误。</strong>不要加仓→清仓，留有少量仓位→清仓，想慢跌／下来上车→希望止跌。</li><li><strong>高分也会错。</strong>上述错误的原文支持度达到 0.85～0.90。表达强度与支持度都保留了，但尚未验证为可靠的连续测量，更不是已拟合的业务权重。</li></ol><p>字段全部与参照一致的帖子，也可能在说明里判断错人、猜错对象。因此下面不把“参照一致”改称“整帖通过”。</p></section>
<nav><select id="source"><option value="all">全部 80 条</option><option value="real">真实短帖 50 条</option><option value="synthetic">对照句 30 条</option></select><select id="status"><option value="all">全部情况</option><option value="clear">重点：明确错判</option><option value="difference">与参照存在分歧</option><option value="error">输出自相矛盾 2 条</option><option value="same">固定标签与参照一致</option></select><input id="query" placeholder="搜索原帖、编号或复核说明" aria-label="搜索"><span id="count"></span></nav><div id="rows"></div>
<p class="muted">本页是当前助手复核，不是新标注金标准。未修改标签定义、原参照、模型权重或训练数据。下一步应在固定这批输入与模型的条件下修复多项提取、否定状态和输出一致性，另用未参与修改的新样本核验。</p></main>
<script id="audit-data" type="application/json">__DATA__</script><script>
const d=JSON.parse(document.getElementById('audit-data').textContent), esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const names={self:'作者',other_retail:'其他散户',main_force:'主力',other:'其他主体',text:'话题',unknown:'不明',completed:'已完成',planned:'计划',current:'当前',past:'过去',conditional:'条件',general:'一般陈述'};
const frac=(a,b)=>`${a} / ${b}（${b?(100*a/b).toFixed(1)+'%':'无样本'}）`;
const mm=[['全部含义（混合样本）',d.evaluation.totals],['仅情绪项（混合样本）',d.evaluation.per_group.emotion],['真实短帖：全部含义',d.sample_groups.S1R.totals],['真实短帖：仅情绪项',d.sample_groups.S1R.per_group.emotion]];
document.getElementById('metrics').innerHTML=mm.map(([n,m])=>`<tr><td>${esc(n)}</td><td>${frac(m.TP,m.TP+m.FP)}</td><td>${frac(m.TP,m.TP+m.FN)}</td></tr>`).join('');
function positives(j){let a=Object.entries(j||{}).filter(([k,v])=>v.decision==='supported');return a.length?'<ul>'+a.map(([k,v])=>`<li><strong>${esc(d.labels[k]||k)}</strong>；${esc(names[v.subject]||v.subject)} / ${esc(names[v.time]||v.time)} / 对象：${esc(v.target)}<br><span class="quote">“${esc(v.quote)}”</span><br><span class="muted">${v.reason?"判断理由："+esc(v.reason):"这版只有原文引用，没有输出判断理由。"}</span>${v.support!=null?`<br><span class="muted">原文支持度 ${esc(v.support)}；表达强度 ${esc(v.intensity??'不适用')}（均未校准）</span>`:''}</li>`).join('')+'</ul>':'<p class="muted">无肯定项</p>'}
function render(){const source=document.getElementById('source').value,status=document.getElementById('status').value,q=document.getElementById('query').value.toLowerCase();let rs=d.rows.filter(r=>{const real=r.input.id.startsWith('S1R'),ok=r.prediction.status==='ok',diff=r.differences.length>0;return(source==='all'||(source==='real')===real)&&(status==='all'||status==='clear'&&r.highlight_clear_error||status==='difference'&&diff||status==='error'&&!ok||status==='same'&&ok&&!diff)&&(!q||[r.input.id,r.input.text,r.review_note].join(' ').toLowerCase().includes(q))});document.getElementById('count').textContent=`显示 ${rs.length} 条`;
document.getElementById('rows').innerHTML=rs.map(r=>{const i=r.input,p=r.prediction,ok=p.status==='ok';const diffs=r.differences.map(x=>`${d.labels[x.meaning]}：${({FP:'模型有／参照无',FN:'参照有／固定字段无',unknown_supported:'模型有／参照不确定'})[x.type]||x.type}`);return `<article id="${esc(i.id)}"><div class="chips"><span class="chip">${esc(i.id)}</span><span class="chip">${i.id.startsWith('S1R')?'真实短帖':'人工对照句'}</span>${i.stock_name?`<span class="chip">${esc(i.stock_name)}</span>`:''}<span class="chip ${!ok||r.highlight_clear_error?'bad':''}">${!ok?'输出自相矛盾':r.differences.length?'与参照有分歧':'固定标签与参照一致'}</span></div><p class="muted">${esc(i.published_at||'')} · ${esc(r.batch)}</p><blockquote>${esc(i.text)}</blockquote><p class="note">复核意见：${esc(r.review_note)}</p>${diffs.length?'<p>'+diffs.map(esc).join('；')+'</p>':''}<div class="columns"><div><strong>模型的正式肯定字段</strong>${ok?positives(p.judgments):'<p class="bad">因肯定与否定重复而未接受；完整原始输出见下方。</p>'}</div><div><strong>原暂定参照的肯定字段</strong>${positives(r.reference.judgments)}</div></div>${(p.observations||[]).length?'<details open><summary>模型补充说明（评测标签分数没有统计这些）</summary><ul>'+p.observations.map(o=>`<li>${esc(o.meaning)}；主体：${esc(names[o.subject]||o.subject)}；对象：${esc(o.target)}；${esc(names[o.time]||o.time)}<br>证据：“${esc(o.quote)}”</li>`).join('')+'</ul></details>':''}<details><summary>完整原始输出，包括否定项、理由与分数</summary><pre>${esc(r.raw_content)}</pre></details><details><summary>完整原暂定参照（未经用户确认）</summary><pre>${esc(JSON.stringify(r.reference.judgments,null,2))}</pre></details></article>`}).join('');}
for(const id of ['source','status','query'])document.getElementById(id).addEventListener(id==='query'?'input':'change',render);render();
</script></html>'''
encoded = json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
(ROOT / 'quality-review.html').write_text(page.replace('__DATA__', encoded))
print(json.dumps({'rows': len(rows), 'valid': evaluation['valid_prediction_rows'], 'output': str(ROOT / 'quality-review.html')}, ensure_ascii=False))
