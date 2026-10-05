#!/usr/bin/env python3
"""把 87 条分歧仲裁包渲染成可交互 HTML(单文件,light 主题,键盘仲裁,可导出结果)。"""
import json

OUT = '/Users/mac/Documents/trae_projects/MyResearcher/pipeline-runs/contrarian-bottleneck-20261005'

rows = [json.loads(l) for l in open(f'{OUT}/k3-arbitration-pack.jsonl')]

def sp(s): return 'BEAR' if s == 'BEAR' else ('BULL' if s == 'BULL' else 'OTH')

for i, r in enumerate(rows):
    r['idx'] = i + 1
    r['hard'] = {sp(r['student']['stance']), sp(r['k3']['stance'])} == {'BULL', 'BEAR'}
    r['emotion_diff'] = r['student']['emotion'] != r['k3']['emotion']

# 排序: 方向硬分歧优先,然后极值层,然后其余
layer_order = {'extreme_day': 0, 'student_confident': 1, 'random_baseline': 2}
rows.sort(key=lambda r: (not r['hard'], layer_order[r['layer']], r['idx']))
for i, r in enumerate(rows):
    r['display_order'] = i + 1

data_js = json.dumps(rows, ensure_ascii=False)

HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>K3 分歧仲裁 · 逆向信号语义审计</title>
<style>
  :root{
    --bg:#f6f7f9; --card:#ffffff; --line:#e3e6ea; --ink:#1c2128; --ink2:#57606a; --ink3:#8b949e;
    --k3:#6b4ee6; --stu:#0b7285; --ok:#1a7f37; --warn:#b45309; --hard:#cf222e;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
    font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Helvetica Neue",sans-serif;
    font-size:15px;line-height:1.65}
  header{position:sticky;top:0;z-index:10;background:rgba(255,255,255,.96);
    backdrop-filter:blur(8px);border-bottom:1px solid var(--line);padding:14px 20px 10px}
  .wrap{max-width:920px;margin:0 auto}
  h1{font-size:18px;margin:0 0 4px}
  .sub{color:var(--ink2);font-size:13px}
  .sub b{color:var(--ink)}
  .bar{display:flex;align-items:center;gap:12px;margin-top:10px;flex-wrap:wrap}
  .prog{flex:1;min-width:160px;height:8px;background:#e9ecf1;border-radius:4px;overflow:hidden}
  .prog>i{display:block;height:100%;background:var(--ok);width:0%;transition:width .2s}
  .chip{padding:4px 10px;border:1px solid var(--line);border-radius:999px;background:#fff;
    cursor:pointer;font-size:13px;color:var(--ink2);user-select:none}
  .chip.on{background:var(--ink);color:#fff;border-color:var(--ink)}
  .btn{padding:6px 14px;border:1px solid var(--line);border-radius:8px;background:#fff;
    cursor:pointer;font-size:13px;color:var(--ink)}
  .btn.primary{background:var(--ok);color:#fff;border-color:var(--ok)}
  main{max-width:920px;margin:0 auto;padding:18px 20px 120px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:12px;
    padding:16px 18px;margin-bottom:14px;box-shadow:0 1px 2px rgba(0,0,0,.03)}
  .card.hard{border-left:4px solid var(--hard)}
  .card.done{opacity:.62}
  .meta{display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:12px;color:var(--ink3);margin-bottom:10px}
  .tag{padding:2px 8px;border-radius:6px;font-size:12px;background:#eef1f5;color:var(--ink2)}
  .tag.hard{background:#fde8e8;color:var(--hard);font-weight:600}
  .tag.sar{background:#f3e8fd;color:var(--k3)}
  .tag.ext{background:#fff3d6;color:var(--warn)}
  .tag.stu{background:#e0f2f5;color:var(--stu)}
  .tag.ran{background:#eef1f5;color:var(--ink2)}
  .text{font-size:17px;line-height:1.7;margin:6px 0 14px;padding-left:12px;border-left:3px solid #d8dde4}
  .vs{display:grid;grid-template-columns:1fr 1fr;gap:12px}
  .side{border:1px solid var(--line);border-radius:10px;padding:10px 12px;font-size:13.5px}
  .side .who{font-weight:700;font-size:13px;margin-bottom:6px;display:flex;justify-content:space-between}
  .side.stu .who{color:var(--stu)} .side.k3 .who{color:var(--k3)}
  .kv{margin:3px 0;color:var(--ink2)}
  .kv b{color:var(--ink)}
  .reason{color:var(--ink2);font-size:12.5px;margin-top:6px;border-top:1px dashed var(--line);padding-top:6px}
  .verdict{display:flex;gap:8px;margin-top:12px;flex-wrap:wrap;align-items:center}
  .vbtn{padding:7px 14px;border-radius:8px;border:1px solid var(--line);background:#fff;cursor:pointer;font-size:13.5px}
  .vbtn:hover{border-color:#b9c0c9}
  .vbtn.k3:hover{background:#f3e8fd;border-color:var(--k3);color:var(--k3)}
  .vbtn.stu:hover{background:#e0f2f5;border-color:var(--stu);color:var(--stu)}
  .vbtn.both:hover{background:#eef1f5}
  .vbtn.none:hover{background:#fff3d6;border-color:var(--warn);color:var(--warn)}
  .vbtn.sel{outline:2px solid var(--ink);outline-offset:1px;font-weight:600}
  .key{display:inline-block;min-width:18px;padding:0 5px;border:1px solid var(--line);
    border-radius:4px;background:#f6f7f9;font-size:11px;color:var(--ink3);margin-right:5px}
  .note{flex:1;min-width:180px}
  .note input{width:100%;padding:6px 10px;border:1px solid var(--line);border-radius:8px;font-size:13px;
    font-family:inherit}
  .ved{font-size:13px;color:var(--ink2);margin-top:8px}
  .ved b{color:var(--ok)}
  footer{position:fixed;bottom:0;left:0;right:0;background:rgba(255,255,255,.97);
    border-top:1px solid var(--line);padding:10px 20px;z-index:10}
  .fwrap{max-width:920px;margin:0 auto;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
  .stat{font-size:13px;color:var(--ink2);flex:1}
  .hint{font-size:12px;color:var(--ink3)}
  @media(max-width:640px){.vs{grid-template-columns:1fr}}
</style>
</head>
<body>
<header><div class="wrap">
  <h1>K3 分歧仲裁 · 逆向信号语义审计</h1>
  <div class="sub">本页只含<b>双方有分歧</b>的条目（两边判定完全一致的已排除，如 K3 判的 13 条反讽中 1 条两边一致故未收录）。
  共 <b id="total"></b> 条，其中 <b id="hardN"></b> 条方向硬分歧（BULL↔BEAR，对信号影响最直接）已排在最前。
  判定标准：<b>谁对就给谁</b>；若属口径差异（如 ANGER vs FRUSTRATION 细分之争）选「都对」；都错选「都不对」。
  判定结果自动存本机浏览器，判完点右下角导出，把 JSON 交回即可。</div>
  <div class="bar">
    <div class="prog"><i id="prog"></i></div>
    <span class="sub" id="progTxt"></span>
  </div>
  <div class="bar">
    <span class="chip on" data-f="all">全部</span>
    <span class="chip" data-f="hard">方向硬分歧(<span id="cHard"></span>)</span>
    <span class="chip" data-f="undone">未判</span>
    <span class="chip" data-f="extreme_day">极值层</span>
    <span class="chip" data-f="sar">K3判反讽</span>
  </div>
</div></header>

<main id="list"></main>

<footer><div class="fwrap">
  <div class="stat" id="footStat"></div>
  <span class="hint">快捷键：<span class="key">1</span>K3对 <span class="key">2</span>学生对 <span class="key">3</span>都对 <span class="key">4</span>都不对 <span class="key">←</span><span class="key">→</span>翻页</span>
  <button class="btn" id="reset">清除全部</button>
  <button class="btn primary" id="export">导出仲裁结果 JSON</button>
</div></footer>

<script>
const DATA = __DATA__;
const LS = 'k3-arbitration-v1';
let state = JSON.parse(localStorage.getItem(LS) || '{}');   // sample_id -> {verdict, note}
let filter = 'all';
let cursor = 0;

const LAYER_NAME = {extreme_day:'极值日', student_confident:'学生高置信', random_baseline:'随机基线'};

function save(){ localStorage.setItem(LS, JSON.stringify(state)); }

function visibleRows(){
  if(filter==='all') return DATA;
  if(filter==='hard') return DATA.filter(r=>r.hard);
  if(filter==='undone') return DATA.filter(r=>!state[r.sample_id]);
  if(filter==='sar') return DATA.filter(r=>r.k3.sarcasm);
  return DATA.filter(r=>r.layer===filter);
}

function render(){
  const rows = visibleRows();
  const list = document.getElementById('list');
  list.innerHTML = rows.map(r=>{
    const v = state[r.sample_id];
    const tags = [
      '<span class="tag">'+r.stock_name+' '+r.stock_code+'</span>',
      '<span class="tag">'+r.entry_date+'</span>',
      '<span class="tag '+(r.layer==='extreme_day'?'ext':'stu')+'">'+LAYER_NAME[r.layer]+'</span>'
    ];
    if(r.hard) tags.push('<span class="tag hard">方向硬分歧</span>');
    if(r.k3.sarcasm) tags.push('<span class="tag sar">K3判反讽</span>');
    if(r.emotion_diff && !r.hard) tags.push('<span class="tag">仅情绪分歧</span>');
    const sel = k => v && v.verdict===k ? ' sel' : '';
    return `<div class="card ${r.hard?'hard':''} ${v?'done':''}" data-id="${r.sample_id}">
      <div class="meta">${tags.join('')}<span style="margin-left:auto">#${r.display_order}</span></div>
      <div class="text">${esc(r.text)}</div>
      <div class="vs">
        <div class="side stu">
          <div class="who"><span>学生模型 A5.1</span></div>
          <div class="kv">情绪：<b>${r.student.emotion}</b> <span style="color:#8b949e">(p=${r.student.emotion_prob})</span></div>
          <div class="kv">立场：<b>${r.student.stance}</b> <span style="color:#8b949e">(p=${r.student.stance_prob})</span></div>
        </div>
        <div class="side k3">
          <div class="who"><span>K3 盲标</span></div>
          <div class="kv">情绪：<b>${r.k3.emotion}</b></div>
          <div class="kv">立场：<b>${r.k3.stance}</b> ${r.k3.sarcasm?'<span class="tag sar">反讽</span>':''}</div>
          <div class="reason">理由：${esc(r.k3.note)}</div>
        </div>
      </div>
      <div class="verdict">
        <button class="vbtn k3${sel('k3')}" data-v="k3"><span class="key">1</span>K3 对</button>
        <button class="vbtn stu${sel('student')}" data-v="student"><span class="key">2</span>学生对</button>
        <button class="vbtn both${sel('both')}" data-v="both"><span class="key">3</span>都对(口径差异)</button>
        <button class="vbtn none${sel('none')}" data-v="none"><span class="key">4</span>都不对</button>
        <span class="note"><input placeholder="备注（可选，如正确答案）" value="${v&&v.note?esc(v.note):''}"></span>
      </div>
      ${v?`<div class="ved">已判：<b>${{k3:'K3 对',student:'学生 对',both:'都对',none:'都不对'}[v.verdict]}</b></div>`:''}
    </div>`;
  }).join('') || '<div class="card">该筛选下没有条目。</div>';

  // 进度
  const done = DATA.filter(r=>state[r.sample_id]).length;
  document.getElementById('prog').style.width = (done/DATA.length*100)+'%';
  document.getElementById('progTxt').textContent = done+' / '+DATA.length;
  const hardDone = DATA.filter(r=>r.hard && state[r.sample_id]).length;
  document.getElementById('footStat').innerHTML =
    `已判 <b>${done}</b>/${DATA.length}　硬分歧 <b>${hardDone}</b>/${document.getElementById('cHard').textContent}`;

  list.querySelectorAll('.vbtn').forEach(b=>{
    b.onclick = e=>{
      const card = e.target.closest('.card');
      setVerdict(card.dataset.id, b.dataset.v);
    };
  });
  list.querySelectorAll('.note input').forEach(inp=>{
    inp.oninput = e=>{
      const id = e.target.closest('.card').dataset.id;
      state[id] = state[id] || {};
      state[id].note = e.target.value;
      save();
    };
  });
}

function esc(s){ return String(s).replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

function setVerdict(id, v){
  state[id] = Object.assign(state[id]||{}, {verdict:v});
  save();
  render();
  // 自动跳到下一个未判(在"未判"筛选下尤其顺手)
  const rows = visibleRows();
  const i = rows.findIndex(r=>r.sample_id===id);
  if(typeof nativeFocus === 'function') nativeFocus(id);
}

document.addEventListener('keydown', e=>{
  if(e.target.tagName==='INPUT') return;
  const rows = visibleRows();
  if(!rows.length) return;
  const keymap = {'1':'k3','2':'student','3':'both','4':'none'};
  if(keymap[e.key]){
    if(cursor>=rows.length) cursor = 0;
    setVerdict(rows[cursor].sample_id, keymap[e.key]);
    cursor = Math.min(cursor+1, rows.length-1);
    scrollToRow(rows[cursor]);
    e.preventDefault();
  }
  if(e.key==='ArrowRight'){ cursor=Math.min(cursor+1,rows.length-1); scrollToRow(rows[cursor]); }
  if(e.key==='ArrowLeft'){ cursor=Math.max(cursor-1,0); scrollToRow(rows[cursor]); }
});

function scrollToRow(r){
  const el = document.querySelector(`.card[data-id="${r.sample_id}"]`);
  if(el && typeof el.scrollIntoView === 'function'){ el.scrollIntoView({behavior:'smooth', block:'center'}); }
}

document.querySelectorAll('.chip').forEach(c=>{
  c.onclick = ()=>{
    document.querySelectorAll('.chip').forEach(x=>x.classList.remove('on'));
    c.classList.add('on');
    filter = c.dataset.f; cursor = 0;
    render();
  };
});

document.getElementById('reset').onclick = ()=>{
  if(confirm('清除全部已判结果？此操作不可撤销。')){ state={}; save(); render(); }
};

document.getElementById('export').onclick = ()=>{
  const out = {
    exported_at: new Date().toISOString(),
    total: DATA.length,
    verdicts: DATA.map(r=>({
      display_order: r.display_order, sample_id: r.sample_id, layer: r.layer,
      hard: r.hard, stock_code: r.stock_code, stock_name: r.stock_name, entry_date: r.entry_date,
      text: r.text,
      student: r.student, k3: r.k3,
      verdict: state[r.sample_id] ? state[r.sample_id].verdict : null,
      note: state[r.sample_id] ? (state[r.sample_id].note||'') : ''
    }))
  };
  const blob = new Blob([JSON.stringify(out, null, 1)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'arbitration-result.json';
  a.click();
};

document.getElementById('total').textContent = DATA.length;
document.getElementById('hardN').textContent = DATA.filter(r=>r.hard).length;
document.getElementById('cHard').textContent = DATA.filter(r=>r.hard).length;
render();
</script>
</body>
</html>
"""

html = HTML.replace('__DATA__', data_js)
with open(f'{OUT}/arbitration.html', 'w') as f:
    f.write(html)
print(f'输出 {OUT}/arbitration.html  (数据 {len(rows)} 条, 硬分歧 {sum(1 for r in rows if r["hard"])} 条)')
