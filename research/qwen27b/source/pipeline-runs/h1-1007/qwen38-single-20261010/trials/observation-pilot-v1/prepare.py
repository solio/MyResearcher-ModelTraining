"""Prepare a local, unrun prompt comparison. Never connects to a GPU node."""
from pathlib import Path
from collections import defaultdict, Counter
import difflib
import gzip
import hashlib
import html
import json
import random
import re

D = Path(__file__).resolve().parent
R = D.parents[1]
PROJECT = R.parents[2]
PREV = R / 'trials/bf16-400-v1'
STAGED = R.parent / 'qwen38-staged-20261010/data'

def sha(b):
    return hashlib.sha256(b).hexdigest()

def text_hash(s):
    return sha(re.sub(r'\s+', '', s).encode())

def save(name, obj):
    (D / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

old = set()
textkeys = {'text', 'title', 'content', 'clean_text', 'raw_text', 'post_text', 'body', 'full_text', 'input_text'}
def visit(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in textkeys and isinstance(v, str) and v.strip():
                old.add(text_hash(v))
            elif isinstance(v, (dict, list)):
                visit(v)
    elif isinstance(obj, list):
        for v in obj:
            visit(v)

prior_manifest = json.loads((PREV / 'data/sampling-manifest.json').read_text())
paths = {PROJECT / p['path'] for p in prior_manifest['prior_files']}
paths.update([PREV / 'data/stage2-inputs.json', STAGED / 'stage1-inputs.json',
              STAGED / 'retry-reserve-inputs.json', STAGED / 'candidates-stage1.json'])
prior_files = []
for p in sorted(paths):
    raw = p.read_bytes()
    if p.suffix == '.jsonl':
        for line in raw.decode().splitlines():
            if line.strip():
                visit(json.loads(line))
    else:
        visit(json.loads(raw))
    prior_files.append({'path': str(p.relative_to(PROJECT)), 'sha256': sha(raw)})

source_path = PREV / 'prepared-data/source-snapshot-20261009.json.gz'
with gzip.open(source_path, 'rt') as f:
    source = json.load(f)
population = defaultdict(list)
seen = set()
excluded = Counter()
for r in sorted(source, key=lambda x: (x['published_at'], x['stock_code'], str(x['source_item_id']))):
    t = r['text']
    h = text_hash(t)
    if not t.strip() or len(t) > 200:
        excluded['empty_or_over_200_chars'] += 1
        continue
    if h in old:
        excluded['previously_discovered_experimental_text'] += 1
        continue
    key = (r['stock_code'], h)
    if key in seen:
        excluded['same_text_within_stock'] += 1
        continue
    seen.add(key)
    population[r['stock_code']].append(r)

rng = random.Random(2026101201)
rows, coverage = [], []
for stock, candidates in sorted(population.items()):
    assert len(candidates) >= 4
    dates = []
    for q in range(4):
        group = candidates[len(candidates)*q//4:len(candidates)*(q+1)//4]
        selected = dict(rng.choice(group))
        selected.update(id=f'O1H{len(rows)+1:03}', source_kind='heldout_natural_short_text',
                        sampling_stratum=f'{stock}:chronological_record_quartile_{q+1}')
        rows.append(selected)
        dates.append({'stratum': q+1, 'eligible_n': len(group),
                      'date_min': group[0]['published_at'], 'date_max': group[-1]['published_at'],
                      'selected_date': selected['published_at']})
    coverage.append({'stock_code': stock, 'stock_name': candidates[0]['stock_name'],
                     'eligible_n': len(candidates), 'n': 4, 'strata': dates})
assert len(rows) == 80 and len(population) == 20
assert not {text_hash(r['text']) for r in rows} & old
save('heldout-inputs.json', {'status': 'unrun; no reference authored; no outputs', 'rows': rows})
save('sampling-manifest.json', {
    'seed': 2026101201, 'source_n': len(source), 'source_gzip_sha256': sha(source_path.read_bytes()),
    'sample_n': len(rows), 'stock_n': len(population), 'per_stock': 4,
    'date_min': min(r['published_at'] for r in rows), 'date_max': max(r['published_at'] for r in rows),
    'selection': 'Per stock: deduplicate <=200-character text across its available history; sort chronologically; split eligible records into four equal-count groups; uniformly draw one per group. No keyword, sentiment, score or model-output filter. Raw text unchanged.',
    'limits': 'Stock-balanced diagnostic sample, not corpus prevalence. Exclusion covers recorded discoverable experimental text files only; not all historical training or pretraining. Four observations per stock cannot establish per-stock accuracy.',
    'heldout_handling': 'Builder reads mechanically; texts are not displayed to the prompt author or used as examples. Reference reading must occur after both prompt hashes are frozen and before seeing outputs.',
    'prior_files': prior_files, 'excluded': dict(excluded), 'coverage': coverage})

baseline = (PREV / 'runner/prompt-semantic.txt').read_text()
candidate = baseline
replacements = [
    ('先理解整句，再逐一检查下面17项。', '先理解整句，先在observations中完整列出原文可确认的各项含义，再用这些观察逐一检查下面17项。'),
    ('根字段为 id、checks、observations。', '根字段为 id、observations、checks，按此顺序生成。'),
    ('observations 是额外含义列表', 'observations 是完整含义列表，保留原文独立表达的含义；含义不因已经属于17项而被省略，不将同一证据拆成重复观察'),
    ('已有17项中的明确含义必须填进对应checks，不能只写到observations；真正额外的含义、其他主体的情绪在此保留。',
     '已有17项中的明确含义同时填进对应checks；observations保留完整观察，checks是按原有定义对同一观察所作的分类。原文明确否认某含义时，在meaning中明确写出否认；不确定某关系时明确写出缺少的信息，不能作为已确认事实。其他主体的情绪也在observations保留。'),
    ('输出前核对：17项是否全部明确作答；', '输出前核对：observations是否保留各项可确认的含义，checks是否与这些观察一致；17项是否全部明确作答；')]
for old_string, new_string in replacements:
    assert candidate.count(old_string) == 1
    candidate = candidate.replace(old_string, new_string)
(D / 'prompt-a.txt').write_text(baseline)
(D / 'prompt-b.txt').write_text(candidate)
(D / 'prompt-diff.txt').write_text(''.join(difflib.unified_diff(baseline.splitlines(True), candidate.splitlines(True), fromfile='A/current', tofile='B/observations-first')))
(D / 'meaning_schema.json').write_bytes((PREV / 'frozen/meaning_schema.json').read_bytes())
prev_plan = json.loads((PREV / 'plan.json').read_text())
plan = {
    'status': 'locally_prepared_not_run', 'approved_ontology': False,
    'training_allowed': False, 'memory_update_allowed': False,
    'question': 'Does generating complete observations before the existing categories retain more clear meaning without adding unsupported interpretations?',
    'model': {k: prev_plan['model'][k] for k in ('repo', 'revision')},
    'precision': 'BF16', 'tensor_parallel_size': 2,
    'generation': {k: prev_plan['generation'][k] for k in ('temperature', 'top_p', 'max_tokens', 'thinking')},
    'concurrency': 8, 'n_inputs': 80, 'n_requests': 160,
    'controlled_change': 'Prompt organization: complete observations first, original 17-category checks second. Definitions, original input text/context, scores, decoding and model fixed. Increased duplicate output/token cost is part of this intervention, not hidden.',
    'schedule': 'Interleave A/B across identical inputs, alternate which arm goes first. Save prompt/model/input hashes and raw outputs for every request. Format errors never halt remaining rows; infrastructure failures require diagnosis.',
    'reference': 'Before outputs are exposed, read the held-out posts against frozen definitions and record clear atomic meanings, evidence and unresolved relations. Assistant provisional reference, not human gold. Prompt edits after reference reading invalidate this holdout.',
    'comparison': 'Blind A/B identity during paired review. Per post compare retained clear meanings, invented meanings, subject/target/negation/completion reversals, extra-observation coverage, ambiguous/omitted content, token use and time. Report wins/losses/ties and denominators; format success, confidence and label agreement are not accuracy.',
    'decision': 'Only expand if paired examples show reproducible improvement in clear-meaning retention without a worsening of unsupported interpretations or severe relation reversals, at acceptable measured cost. If evidence is mixed, do not claim a win; retain current version and diagnose the observed loss pattern. No automatic all-corpus labeling or training from these outputs.',
    'smooth_score_work': 'Separate local representation demonstration, not another A/B prompt change. Preserve polarity, expression intensity, action extent, evidence support and missingness separately. Exact extent numbers only when explicit; qualitative magnitude stays qualitative until empirically calibrated. No final scalar emotion or return weights assigned.',
    'expected_time': 'Previous 400 BF16 requests took 2853.86 seconds. 160 at that throughput ~19 minutes inference only; candidate output is longer, so this is not a completion promise. Startup, review and failures extra.',
    'execution': 'No remote connection or paid work launched. Previous two-hour allocation expired; next session requires an available authorized compute window.',
    'remaining_before_execution': ['Author the blind provisional reference with frozen prompts', 'Wire A/B scheduling and observation-complete reports to the existing runner; align structured JSON property order with the B observations-first prompt without changing allowed fields or semantic validation. Existing response_schema currently lists checks before observations; prompt text alone is not sufficient to assume generation order.', 'Verify actual available node and paid deadline']}
plan['files_sha256'] = {n: sha((D/n).read_bytes()) for n in ['heldout-inputs.json', 'sampling-manifest.json', 'prompt-a.txt', 'prompt-b.txt', 'meaning_schema.json']}
save('plan.json', plan)

examples = [
    ('S2N145', '今天狠心减持了一半，这股实在令人烦心，落后创业板不知道多少了。看着糟心',
     '分别保留：作者自述减持、已发生、数量为原持仓的一半；烦心／糟心的负面感受。',
     '明确“一半”可记为卖出比例0.5，单位是操作前的持仓；“狠心”不能替代仓位比例，也不能凭空推成后悔。'),
    ('S2N266', '周师，请问感觉跌到位了吗？51买了半仓，现在想补又有点怕@妖股之源',
     '分别保留：买入的过去陈述；当时半仓；现在有补仓想法；有点怕；向他人询问是否跌到位。',
     '“半仓”是仓位水平，不是卖出比例；“有点怕”是弱表达，现阶段不声称它精确等于0.23。想补仓与恐惧可以同时成立。'),
    ('S2N245', '买电力',
     '保留涉及买入、对象为电力；原文不足以确定谁买、是建议还是已成交。',
     '不确定的关系留空，仍能保留明确的话题和动作方向。不能将其算成“已经买入”，也不能把空值当平静。')]
cards = ''.join(f'<article><a href="bf16-review.html?id={i}">{i} · 查看上一轮原输出</a><blockquote>{html.escape(t)}</blockquote><p><b>希望保留什么：</b>{html.escape(o)}</p><p><b>分数怎样处理：</b>{html.escape(s)}</p></article>' for i,t,o,s in examples)
manifest = json.loads((D / 'sampling-manifest.json').read_text())
page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>下一步：完整观察与平滑分</title><style>
body{margin:0;background:#f3f5f8;color:#192638;font:17px/1.75 system-ui,sans-serif}main{max-width:1000px;margin:auto;padding:30px 22px 70px}h1{font-size:32px;line-height:1.35}h2{font-size:23px;margin-top:30px}article,.box{background:white;border:1px solid #d9e2eb;border-radius:12px;padding:20px;margin:18px 0}a{color:#175ea8}blockquote{margin:12px 0;padding:12px 18px;border-left:4px solid #639bc9;background:#f1f6fb}.status{display:inline-block;background:#e4edf7;padding:4px 12px;border-radius:8px;font-weight:600}li{margin:12px 0}table{border-collapse:collapse;width:100%;font-size:16px}th,td{text-align:left;vertical-align:top;padding:12px;border-bottom:1px solid #d9e2eb}small{color:#52657b}.tablewrap{overflow:auto}@media(max-width:600px){h1{font-size:27px}main{padding:18px 14px}article{padding:14px}}
</style><main><span class="status">本地候选方案已准备 · 尚未运行 · 效果未证实</span><h1>下一步：让读出的信息进入统计，再检验改法是否有效</h1>
<p>上一轮已经能读出不少有用含义，但部分信息只落在额外观察中；如果后续只统计固定标签，就会丢掉这些信息。下一轮检验一个具体改法：<b>先完整列观察，再填原有标签。</b>原提示词本来就要求多项提取，这次是在检验生成顺序和信息组织是否有帮助，不能预先宣称解决了问题。</p>
<h2>用三条已经看过的真实帖子，把方向讲清楚</h2><p><small>以下是设计示例，由助手整理；不是新模型已经给出的结果，也不是你批准的标签定义。未把这些例子加入候选提示词。</small></p>''' + cards + '''
<h2>你说的“平滑分”，这一轮怎样往前走</h2><div class="box"><p><b>让“清仓”和“卖掉大部分”共享卖出方向，同时保留程度与剩余持仓。</b>相邻标签改变，不应把共同观察全部抹掉。原文写明“一半”才有依据记录0.5；“大部分”先保留定性程度和原话，不能擅自赋0.9。</p><p>恐惧程度、看多／看空方向、卖出程度、模型对证据的把握分别保存。当前模型输出的intensity和support都不足以充当最终情绪值。连续分的具体映射需要后续用可排序的实例校准；不同日期和上下文怎样拟合结果，也还没有验证。</p><p><b>这次只准备表示方式，不改A/B中的评分定义。</b>这样能分清完整提取改法的效果。若将来一个邻近标签变化仍使某日统计从高恐惧跳成零恐惧，就能定位是映射或聚合在放大误差，而不是继续追着每个标签边界修补。</p></div>
<h2>下一轮怎样对照</h2><div class="tablewrap"><table><tr><th>项目</th><th>固定做法</th></tr><tr><td>输入</td><td>新80条，20只股票各4条；按每只股票自己的可用短帖历史分组取样。无情绪关键词、模型高分筛选。</td></tr><tr><td>A</td><td>本轮400条使用的原提示词。</td></tr><tr><td>B</td><td>先生成完整观察，再归入原17项；模型、定义、分数含义、上下文和生成参数相同。</td></tr><tr><td>怎样评</td><td>先写逐帖明确含义参照，再看打乱A/B身份的输出。比较保留了多少明确含义、增添了多少原文没有的意思，单独看谁／哪只股票／否定／计划与完成是否弄反，并记录费用和耗时。</td></tr><tr><td>怎样决定</td><td>清晰含义保留改善且附会及关系颠倒未恶化，才考虑扩大；混合结果不宣称胜出。80条仅用于探索改法，不足以证明全量可用。原参照仍是助手判断，不冒充人工真值。</td></tr></table></div>
<p>这80条的实际日期为 <b>DATE_MIN 至 DATE_MAX</b>。新旧版本都跑这80条，共160次请求。按上一轮速度约19分钟纯推理；新版输出更长，启动和复核另计，实际耗时需实测。</p>
<h2>现在完成到哪了</h2><ul><li>已经准备：固定的新80条、取样清单、A/B提示词、文件指纹与比较说明。</li><li>尚未完成：盲读参照、A/B调度及报告接入验证、实际GPU推理。这里没有新的通过率。</li><li>上次两小时运行窗口已经结束。本地准备没有启动付费任务，也没有申请续费。</li></ul>
<p><a href="trials/observation-pilot-v1/plan.json">查看完整试验说明</a> · <a href="trials/observation-pilot-v1/prompt-diff.txt">查看提示词差异</a> · <a href="bf16-result.html">返回上一轮结果</a></p></main></html>'''
page = page.replace('DATE_MIN', manifest['date_min'][:10]).replace('DATE_MAX', manifest['date_max'][:10])
(R / 'next-observation-pilot.html').write_text(page)
print(json.dumps({'status': plan['status'], 'sample_n':len(rows), 'stocks':len(population),
                  'date_min':manifest['date_min'], 'date_max':manifest['date_max'],
                  'excluded_hashes':len(old), 'prior_files':len(prior_files),
                  'cross_stock_duplicate_selected_texts':len(rows)-len({text_hash(r['text']) for r in rows}),
                  'page':str(R / 'next-observation-pilot.html')}, ensure_ascii=False))
