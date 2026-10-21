#!/usr/bin/env python3
"""Readable closeout with measured counts; no invented quality pass rate."""
import collections
import html
import json
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]


def read(name, default=None):
    p = BASE / name
    if not p.exists():
        return default
    if p.suffix == '.jsonl':
        return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
    return json.loads(p.read_text())


def main():
    rows = read('results/natural/predictions.jsonl', [])
    runtime = read('results/natural/runtime.json', {})
    notes = read('review-notes.json', {}).get('notes', {})
    audit = read('observation-audit-summary.json', {})
    scores = read('score-sensitivity.json', {})
    compare = read('control-comparison.json', {})
    plan = read('plan.json')
    valid = sum(r['status'] == 'ok' for r in rows)
    errors = [{'id': r['id'], 'error': r.get('error')} for r in rows if r['status'] != 'ok']
    overview = {'updated_at_epoch': time.time(), 'natural_attempted': len(rows), 'natural_expected': 400,
                'natural_structure_valid': valid, 'natural_output_errors': errors,
                'assistant_reviewed': len(notes), 'human_gold_available': False,
                'natural_runtime': runtime, 'score_status': 'uncalibrated model estimates; no production aggregate fitted',
                'old80_comparison_available': bool(compare), 'source_rows_indexed': 387905}
    (BASE / 'closeout.json').write_text(json.dumps(overview, ensure_ascii=False, indent=2))
    esc = html.escape
    def e(x):
        return esc(str(x))
    def card(body):
        return '<section class="card">' + body + '</section>'
    content = '<h1>BF16 本轮结果</h1><p><a href="progress.html">任务进度</a> · <a href="bf16-review.html">逐条原文、模型理由与复核意见</a>'
    if compare:
        content += ' · <a href="bf16-comparison.html">同一80条的 INT4 / BF16 对照</a>'
    content += '</p>'
    sentence = '400条新帖均已尝试，原始输出已经保存。' if len(rows) == 400 else f'推理仍在进行，已保存 {len(rows)} / 400 条。'
    content += card(f'<h2>{sentence}</h2><p>结构有效 <strong>{valid}</strong> 条；输出格式异常 <strong>{len(errors)}</strong> 条。助手已逐条阅读 <strong>{len(notes)}</strong> 条的原文、输出和理由。<strong>这些数字不是语义正确率。</strong>400条没有独立人工金标准，复核意见可查看、可反驳，不改模型原始输出。</p>')
    wall = runtime.get('session_wall_seconds')
    detail = f'400条耗时 {wall/60:.1f} 分钟，实测平均 {3600*len(rows)/wall:.0f} 条/小时。' if wall else '完整耗时在队列结束后记录。'
    content += card('<h2>本轮实际做了什么</h2><p>使用两张 A100 40GB 分担一个27B BF16模型，持续批处理8条并发请求。提示词、17项候选含义和输出定义保持冻结；这是推理验证，不是重新训练权重。</p><p>' + detail + '</p><p>样本来自387,905条历史快照：20只股票各20条，日期跨2025年7月至2026年10月。只限制短文本，不按情绪关键词挑选，也没有只使用7—9月。每股均衡抽样不等于原语料占比，不能直接估计全市场情绪比例。</p><p>先跑8条旧样本检查服务；400条后补跑旧样本剩余72条。格式错误写入原始记录并继续队列，逐条存盘及定期备份，避免单条异常停工。</p>')
    cases = [
        ('有用的完整提取', 'S2N266', '51买了半仓，现在想补又有点怕', '已经买入、半仓、现在担忧、计划补仓、向人求助可以同时保留。'),
        ('不应压成单一正负', 'S2N249', '自己还是很幸运的。没想到这票还能这么跌', '庆幸与下跌意外同时成立。行情下跌不等于作者只有负面心情。'),
        ('优先修关系错误', 'S2N245', '买电力', '尚未说明成交，模型却说已买入。这会制造交易事件，需要比相邻标签更优先处理。'),
        ('高分仍会误读', 'S2N258', '都不信你不会涨', '标签为看多，理由却解释为不会上涨；支持度仍给0.95。不能仅靠门槛保证正确。'),
        ('必须读完整句', 'S2N328', '回来了一切都回来了，熟悉的感觉回来了，板块拉它下砸', '模型根据前半句判兴奋，忽略后半句的反讽或无奈；这是方向性误读。'),
        ('不丢掉额外观察', 'S2N313', '哈哈，永赢加仓，笑死了', '机构加仓在额外观察里，固定资金栏却为空。下游只看标签会丢失已识别的信息。'),
        ('相邻标签不应毁掉整帖', 'S2N213', '昨天割了许继加仓进来了，能不能让我开心下', '卖出许继、买入本股的关系可用，卖出份额未知。清仓推断过强，不应连带废弃全部观察。'),
    ]
    body = '<h2>先看这些原文，就能理解结果的边界</h2>'
    existing = {r['id'] for r in rows}
    for title, ident, quote, explanation in cases:
        if ident in existing:
            body += f'<article><h3>{e(title)} · <a href="bf16-review.html?id={ident}">{ident} 原文与输出</a></h3><blockquote>{e(quote)}</blockquote><p>{e(explanation)}</p></article>'
    body += '<p>以上是实际阅读后定位的例子，不是总体错误率，也不把所有语气边界都算成模型失败。</p>'
    content += card(body)
    body = '<h2>平滑分数：本轮验证到哪一步</h2><p>目前保存了“表达强度”和“文本支持度”，但支持度未经校准，强度也不是卖出比例。这轮没有把两者偷换成最终情绪分数。</p><p>将支持度门槛从0.9提高到0.95，会丢掉一部分观察，仍无法排除上面的误读。下表仅看保留量，不代表精确率提升，也没有实际删除数据。</p><div class="scroll"><table><tr><th>观察类别</th><th>原始明确表达条数</th><th>支持度≥0.9剩余</th><th>≥0.95剩余</th><th>≥1.0剩余</th></tr>'
    names = {'action': '动作', 'emotion': '情绪', 'stance': '看法', 'topic': '话题', 'wish': '愿望', 'discussion_target': '讨论对象'}
    for key, d in scores.get('groups', {}).items():
        sweep = {x['threshold']: x['retained_claims'] for x in d['support_threshold_sweep']}
        body += f'<tr><td>{e(names.get(key,key))}</td><td>{d["expressed_claims"]}</td><td>{sweep.get(.9)}</td><td>{sweep.get(.95)}</td><td>{sweep.get(1.)}</td></tr>'
    body += '</table></div><p>计数单位为“观察”，一帖可有多项；只统计原始结构有效帖的固定候选栏，额外观察另存，不能把该表当作者情绪分布。</p><p>真正需要保留的是同一对象、同一时段下的行为方向、程度和不确定性。例如“卖了大部分但还留一点”应保留卖出、大部分、仍持有，不因清仓/减仓边界变化突然变成相反信号。这个程度表示和后续拟合尚未完成；本轮数据为它保留了证据，不能声称已经有可靠隐藏分。</p>'
    action_scores = scores.get('groups', {}).get('action', {})
    if action_scores:
        body += f'<p><strong>实际分数的区分度也有限：</strong>{action_scores.get("expressed_claims",0)} 个动作观察中，{action_scores.get("intensity_histogram",{}).get("0.8",0)} 个强度为0.8。有小数不代表已经能区分少量卖出、大幅减仓和全部退出。</p>'
    content += card(body)
    body = '<h2>CPU、带宽与结果保存</h2><p>全历史387,905条数据已上传并建立可按股票、日期检索的索引；压缩文件与原始数据哈希通过，SQLite完整性检查通过。索引已回传本地并与远端SHA-256核对一致。CPU任务降低优先级，传输限速。</p><p>语料检查找到26,784条含HTML标记、19,162条含HTML实体、4,177条疑似重复标题前缀。这些集合可能重叠，不等于同等数量的坏帖；本轮400条里有9条含标记或实体。先记录，未在推理中途改变原文。</p>'
    body += f'<p>逐项检查额外保住 {audit.get("partial_positive_claims",0)} 个出自格式异常帖、证据与结构单项检查通过的明确观察；这不证明语义正确。原整行失败仍记失败，缺失项保留未知，未补零。</p>'
    body += '<p><a href="trials/bf16-400-v1/closeout.json">运行汇总</a> · <a href="trials/bf16-400-v1/review-notes.json">全部助手复核意见</a> · <a href="trials/bf16-400-v1/score-sensitivity.json">分数敏感性明细</a> · <a href="trials/bf16-400-v1/source-quality-audit.json">语料检查</a></p>'
    content += card(body)
    if compare:
        counts = compare['change_counts']
        content += card(f'<h2>同帖对照已完成</h2><p>旧版 INT4 结构有效 {compare["coverage"]["INT4"]["valid"]}/80，BF16 {compare["coverage"]["BF16"]["valid"]}/80。双方共同有效 {compare["matched_valid"]} 条。相对冻结的内部参照，{counts.get("toward_reference",0)} 个固定栏判断变得一致，{counts.get("away_from_reference",0)} 个变得不一致，{counts.get("unresolved",0)} 个变化无法由该参照判定。</p><p><strong>这不是2项变好、9项变坏的业务判决。</strong>例如别涨也别跌的愿望在BF16额外观察里完整保留，只是没有再填希望止跌栏。另一些变化是边界推论；已经逐项连同额外观察复核，不能把标签变化等同信息丢失。当前对照未显示稳定的整体改善。</p><p>内部参照并非人工金标准，且精度切换伴随双卡部署和并发变化。<a href="bf16-comparison.html">展开查看真实帖与构造句分别的分子、分母和每个变化</a>。</p>')
    content += card('<h2>接下来应依据什么继续</h2><p>先使用本轮原文、理由和同帖对照定位会改变方向的关系错误：谁在表达、针对谁、已经发生还是计划、愿望还是预测。对于清仓/大幅减仓等邻近含义，保留共有的卖出证据及程度，不以精确标签全对作为整帖可用的前提。</p><p>固定标签之外的信息也需要进入底层观察；先保留未确定项，之后再对明确观察和程度表示做可检验的拟合。目前既没有证据宣布这条路失败，也不能凭格式完成率或模型自报高分扩大到全量自动标注。</p>')
    page = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BF16 本轮结果</title><link rel="stylesheet" href="control.css"><style>main{max-width:1100px}body{font-size:18px;line-height:1.8}h1{font-size:32px}h2{font-size:24px}h3{font-size:20px}article{border-top:1px solid #dae1eb;padding:12px 0}blockquote{font-size:22px;border-left:4px solid #4077bb;padding-left:15px;margin:15px 0}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse}td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd;white-space:nowrap}.card{overflow-wrap:anywhere}</style></head><body><main>' + content + '</main></body></html>'
    (ROOT / 'bf16-result.html').write_text(page)
    print(json.dumps({k: overview[k] for k in ['natural_attempted','natural_structure_valid','assistant_reviewed','old80_comparison_available']}))


if __name__ == '__main__':
    main()
