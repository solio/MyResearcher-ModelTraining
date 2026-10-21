#!/usr/bin/env python3
"""Build an offline, Chinese pilot report without inventing absent results.

Usage: python3 report/build_report.py [--run runs/stage1-v1] [--status status.json]
Optional status JSON: {"current_step": "...", "detail": "...", "next_action": "...",
"updated_at": "...", "blockers": ["..."]}. Status prose cannot override the result gate.
Only reads selected run artifacts; never reads raw model reasoning or sends network calls.
"""
import argparse
import collections
import datetime as dt
import hashlib
import html
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
SUBJECT = {"self": "作者本人", "other_retail": "其他散户", "main_force": "主力",
           "other": "其他对象", "text": "文本话题", "unknown": "对象未说明"}
TIME = {"completed": "已完成", "planned": "计划／未来", "current": "当前",
        "past": "过去", "conditional": "满足条件时", "general": "一般陈述", "unknown": "时间未说明"}
DECISION = {"Y": "明确表达", "N": "未明确表达／否定", "?": "参考有歧义"}
STATE = {"same": "与参考一致", "different": "存在差异", "disputed": "需要裁定",
         "pending": "尚未运行", "error": "运行异常"}


def esc(value):
    return html.escape(str(value if value is not None else ""), quote=True)


def read_json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def read_rows(path):
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    try:
        if path.suffix == ".jsonl":
            result = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            obj = json.loads(text)
            result = obj.get("rows", obj.get("predictions")) if isinstance(obj, dict) else obj
    except json.JSONDecodeError:
        # A partially copied final line is an input error, not an absent prediction.
        result = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(result, list):
        raise ValueError(f"Not a row list: {path}")
    ids = [row["id"] for row in result]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs: {path}")
    return result


def decision(item):
    value = item.get("decision") if isinstance(item, dict) else item
    if value in ("Y", "supported", True):
        return "Y"
    if value in ("N", "not_supported", False):
        return "N"
    return "?"


def score(value):
    if isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value):
        return f"{value:.2f}"
    return "—"


def pct(value):
    return f"{100 * value:.1f}%" if value is not None else "—"


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def chip(text, kind=""):
    return f'<span class="chip {esc(kind)}">{esc(text)}</span>'


def quote(text):
    return f'<q>{esc(text)}</q>' if text else '<span class="muted">没有引文</span>'


def collect(base, run):
    inputs = read_rows(base / "data/stage1-inputs.json")
    refs = {x["id"]: x for x in read_rows(base / "data/stage1-reference.json")}
    schema = read_json(base / "runner/meaning_schema.json")
    if not inputs or not refs or not schema:
        raise ValueError("Stage1 inputs, internal reference and meaning schema are required")
    if set(refs) != {x["id"] for x in inputs}:
        raise ValueError("Input/reference ID sets differ")
    candidates = schema["core_candidates"]
    names = {x["id"]: x["label"] for x in candidates}
    predictions = {x["id"]: x for x in read_rows(run / "predictions.jsonl")}
    unexpected = set(predictions) - set(refs)
    if unexpected:
        raise ValueError(f"Predictions contain foreign IDs: {sorted(unexpected)}")
    runtime = read_json(run / "runtime.json", {})
    evaluation = read_json(run / "evaluation.json")
    manifest = read_json(run / "manifest.json", {})
    integrity = []
    if predictions and not manifest:
        integrity.append("结果目录缺少运行清单，不能确认输入和模型版本。")
    if manifest:
        canonical = json.dumps(inputs, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if manifest.get("input_sha256") != hashlib.sha256(canonical.encode()).hexdigest():
            integrity.append("运行输入哈希与本页样本不一致。")
        if manifest.get("semantics", {}).get("schema") != schema:
            integrity.append("运行时的含义定义与本页定义不一致。")
        if any(p.get("run_hash") != manifest.get("run_hash") for p in predictions.values()):
            integrity.append("存在不属于当前运行版本的结果。")
    if evaluation:
        if not manifest or evaluation.get("run_hash") != manifest.get("run_hash"):
            integrity.append("评估文件与当前运行清单不一致。")
        if evaluation.get("prediction_rows") != len(predictions):
            integrity.append("评估文件与当前结果条数不一致，可能是旧评估。")
        canonical_ref = json.dumps(list(refs.values()), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if evaluation.get("reference_sha256") != hashlib.sha256(canonical_ref.encode()).hexdigest():
            integrity.append("评估使用的参考与当前内部参考不一致。")
    counts = {key: collections.Counter(tp=0, fp=0, fn=0, reference_positive=0,
                                      unknown=0, abstain=0, checked=0) for key in names}
    cases = []
    for row in inputs:
        ident = row["id"]
        ref_row = refs[ident]
        if ref_row.get("text") != row["text"]:
            raise ValueError(f"Input/reference text mismatch: {ident}")
        reference = ref_row.get("judgments", {})
        prediction = predictions.get(ident)
        valid = bool(prediction and prediction.get("status") == "ok" and isinstance(prediction.get("judgments"), dict))
        model = prediction.get("judgments", {}) if valid else {}
        for key, item in model.items():
            if decision(item) == "Y" and (not item.get("quote") or item["quote"] not in row["text"]):
                valid = False
                integrity.append(f"{ident} 的证据没有逐字来自原文。")
        extra, missed, matched, uncertain = [], [], [], []
        if valid:
            for key in names:
                gold = decision(reference.get(key))
                item = model.get(key, {})
                actual = decision(item)
                c = counts[key]
                c["checked"] += 1
                c["reference_positive"] += gold == "Y"
                c["abstain"] += actual == "?"
                if gold == "?":
                    if actual == "Y":
                        uncertain.append(key)
                        c["unknown"] += 1
                elif actual == "Y":
                    (matched if gold == "Y" else extra).append(key)
                    c["tp" if gold == "Y" else "fp"] += 1
                elif gold == "Y":
                    missed.append(key)
                    c["fn"] += 1
            state = "different" if extra or missed else "disputed" if uncertain else "same"
        else:
            state = "error" if prediction else "pending"
        critical = []
        if valid:
            for condition in ref_row.get("critical", []):
                key = condition.get("id", condition.get("meaning_id"))
                expected = decision(condition.get("expected"))
                if decision(model.get(key, {})) != expected:
                    critical.append({"label": names.get(key, key), "reason": condition.get("reason", "")})
            if critical:
                state = "different"
        cases.append({"row": row, "reference": reference, "prediction": prediction, "model": model,
                      "valid": valid, "state": state, "extra": extra, "missed": missed,
                      "matched": matched, "uncertain": uncertain, "critical": critical})
    return {"cases": cases, "counts": counts, "names": names, "candidates": candidates,
            "runtime": runtime, "evaluation": evaluation, "manifest": manifest,
            "integrity": list(dict.fromkeys(integrity)), "predictions": predictions}


def render_case(case, names):
    row, reference, pred, model = case["row"], case["reference"], case["prediction"], case["model"]
    source = "真实帖子" if row.get("source_kind") == "real_directed_easy" else "人工编写对照"
    metadata = " · ".join(x for x in [source, row.get("stock_name"), (row.get("published_at") or "")[:10], row["id"]] if x)
    expected = [names[k] for k, v in reference.items() if decision(v) == "Y"]
    found = [names.get(k, k) for k, v in model.items() if decision(v) == "Y"] if case["valid"] else []
    summary = []
    for field, label in [("extra", "额外判断"), ("missed", "漏掉"), ("uncertain", "待裁定")]:
        if case[field]:
            summary.append(f'{label}：' + "、".join(names[x] for x in case[field]))
    if case["state"] == "pending":
        summary = ["结果尚未产生；下面仅展示已准备的内部参考。"]
    elif case["state"] == "error":
        summary = ["没有可评价的有效结果，不能据此说模型没有识别出情绪。"]
    elif not summary:
        summary = ["本帖的明确含义与内部参考一致。"]
    relevant = [k for k in names if decision(reference.get(k)) in ("Y", "?") or
                (case["valid"] and (decision(model.get(k)) != "N" or model.get(k, {}).get("state") == "explicitly_negated"))]
    lines = []
    for key in relevant:
        ref = reference.get(key, {})
        item = model.get(key, {})
        actual = decision(item)
        gold = decision(ref)
        if not case["valid"]:
            result = "尚无有效判断"
        elif actual == "Y":
            result = "明确表达"
        elif actual == "?":
            result = "模型不确定"
        elif item.get("state") == "explicitly_negated":
            result = "原文明确否定"
        else:
            result = "未提取该含义"
        mark = "diff" if key in case["extra"] + case["missed"] else ""
        evidence = quote(item.get("quote")) if case["valid"] else "—"
        subject_time = " · ".join(x for x in [SUBJECT.get(item.get("subject")), item.get("target") if item.get("target") not in (None, "unknown") else None, TIME.get(item.get("time"))] if x)
        explanation = item.get("reason", "")
        lines.append(f'<tr class="{mark}"><td><strong>{esc(names[key])}</strong></td>'
                     f'<td>{esc(DECISION[gold])}<div class="minor">{quote(ref.get("quote"))}</div>'
                     f'<div class="minor">{esc(ref.get("reason", ""))}</div></td>'
                     f'<td>{esc(result)}<div class="minor">{esc(subject_time)}</div></td>'
                     f'<td class="num">{score(item.get("support")) if case["valid"] else "—"}</td>'
                     f'<td class="num">{score(item.get("intensity")) if case["valid"] else "—"}</td>'
                     f'<td>{evidence}<div class="minor">{esc(explanation)}</div></td></tr>')
    observations = []
    if case["valid"]:
        for item in pred.get("observations", []):
            observations.append(f'<li><strong>{esc(item.get("meaning"))}</strong>：{quote(item.get("quote"))}'
                                f'；支持分 {score(item.get("support"))}；表达强度 {score(item.get("intensity"))}'
                                f' <span class="minor">{esc(SUBJECT.get(item.get("subject"), "对象未说明"))} · '
                                f'{esc(TIME.get(item.get("time"), "时间未说明"))}</span></li>')
    more = ('<h3>列表以外的观察</h3><ul>' + "".join(observations) + '</ul>') if observations else ''
    critical = ''.join(f'<p class="alert">关键关系差异：{esc(x["label"])} · {esc(x["reason"])}</p>' for x in case["critical"])
    error = f'<p class="alert">运行记录：{esc(pred.get("error", "结果格式或证据未通过检查"))}</p>' if pred and not case["valid"] else ''
    data_search = " ".join([row["text"], metadata, " ".join(expected), " ".join(found)])
    return f'''<article class="case" data-state="{esc(case['state'])}" data-source="{esc(source)}" data-search="{esc(data_search)}">
      <div class="case-head"><span class="meta">{esc(metadata)}</span>{chip(STATE[case['state']],case['state'])}</div>
      <p class="original">{esc(row['text'])}</p>
      <div class="meaning-line"><b>模型提取</b><span>{''.join(chip(x) for x in found) if case['valid'] and found else ('没有提取出明确含义' if case['valid'] else '尚无有效结果')}</span></div>
      <div class="meaning-line"><b>内部参考</b><span>{''.join(chip(x) for x in expected) if expected else '没有明确的正例含义'}</span></div>
      <p class="difference">{esc('；'.join(summary))}</p>{critical}{error}
      <details><summary>查看逐项判断、两个分数和原文证据</summary><div class="table-wrap"><table>
      <thead><tr><th>含义</th><th>内部参考</th><th>模型判断</th><th>支持分</th><th>表达强度</th><th>模型证据</th></tr></thead>
      <tbody>{''.join(lines) or '<tr><td colspan="6">本帖没有正例或待裁定项。</td></tr>'}</tbody></table></div>
      <p class="minor">未在表中展开的含义，在参考和模型中均未明确表达；这不代表作者现实里没有这些情绪。</p>{more}</details>
    </article>'''


CSS = '''
:root{color-scheme:light;--ink:#173034;--muted:#617175;--line:#d8e2df;--paper:#f3f6f3;--good:#136c53;--warn:#9b4c20}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:17px/1.65 system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif}
main{max-width:1260px;margin:auto;padding:34px 28px 70px}h1{font-size:32px;line-height:1.3;margin:0 0 14px}h2{font-size:24px;margin:32px 0 12px}h3{font-size:18px}p{margin:10px 0}.muted,.meta,.minor{color:var(--muted)}.minor{font-size:14px;line-height:1.5;margin-top:4px}.hero,.panel,.case{background:#fff;border:1px solid var(--line);border-radius:15px;padding:24px;margin-bottom:18px}.hero{border-top:5px solid var(--good)}.lead{font-size:21px}.status{font-weight:700}.notice{padding:15px 18px;background:#edf4f1;border-radius:10px}.alert{color:#91391c;background:#fff0e8;padding:9px 13px;border-radius:8px}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{padding:17px;background:#f3f7f5;border-radius:11px}.metric strong{font-size:31px;display:block}.metric span{font-size:15px}.metric small{display:block;color:var(--muted)}.subline{display:flex;flex-wrap:wrap;gap:12px;margin:12px 0}.chip{font-size:14px;display:inline-block;padding:3px 10px;background:#edf1ef;border-radius:20px;margin:2px 4px 2px 0}.chip.same{color:#176d50;background:#e4f4ea}.chip.different,.chip.error{color:#a23e17;background:#ffeadf}.chip.disputed{background:#fff3ce;color:#795400}.chip.pending{background:#eef0f4;color:#5c6877}.case-head{display:flex;justify-content:space-between;gap:12px;align-items:center}.meta{font-size:14px}.original{font-size:21px;white-space:pre-wrap;overflow-wrap:anywhere;margin:16px 0}.meaning-line{display:grid;grid-template-columns:95px 1fr;gap:8px;margin:8px 0}.meaning-line b{font-size:15px;padding-top:4px}.difference{font-size:15px;color:#685747}details{margin-top:12px}summary{cursor:pointer;color:#205c51;font-size:15px;padding:6px 0}table{border-collapse:collapse;width:100%;font-size:15px}th,td{text-align:left;padding:12px 10px;border-bottom:1px solid var(--line);vertical-align:top}th{font-weight:600;background:#f1f5f3;white-space:nowrap}.table-wrap{overflow:auto;margin-top:12px}.num{font-variant-numeric:tabular-nums;white-space:nowrap}.diff{background:#fff8f2}q{quotes:'“' '”';overflow-wrap:anywhere}.toolbar{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:12px 0 18px}input,select,button{font:inherit;padding:10px 12px;border:1px solid #bbcfc5;border-radius:8px;background:#fff;color:var(--ink)}input{flex:1;min-width:240px}button{cursor:pointer;font-size:15px}button:hover{background:#edf4f1}.case[hidden]{display:none}footer{margin-top:26px;font-size:14px;color:var(--muted);overflow-wrap:anywhere}.small-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.small-grid ul{margin:8px 0;padding-left:22px}.nowrap{white-space:nowrap}@media(max-width:760px){main{padding:22px 14px}.hero,.panel,.case{padding:18px}.metrics{grid-template-columns:1fr 1fr}.small-grid{grid-template-columns:1fr}h1{font-size:27px}.original{font-size:20px}.case-head{align-items:flex-start}.meaning-line{grid-template-columns:80px 1fr}}@media print{body{background:white}main{max-width:none}.toolbar{display:none}.case{break-inside:avoid}.hero,.panel,.case{border-radius:0}}
'''


def build(base, run, status):
    data = collect(base, run)
    cases, names = data["cases"], data["names"]
    state_counts = collections.Counter(x["state"] for x in cases)
    valid = sum(x["valid"] for x in cases)
    completed = len(data["predictions"])
    total = len(cases)
    metrics = sum(data["counts"].values(), collections.Counter())
    evaluation = data["evaluation"]
    full = valid == total and not data["integrity"]
    gate = evaluation.get("gate", {}) if evaluation else {}
    plan = read_json(base / "deployment/plan.json", {})
    if data["integrity"]:
        headline, continuation = "结果一致性需要检查", "暂不扩大：先确认本页样本、模型结果和评估属于同一次运行。"
    elif not completed:
        headline, continuation = "样本已准备，尚未产生模型结果", "下一步先跑 80 条基础样本；目前无法评价识别质量。"
    elif not full:
        headline = f"首轮已收到 {completed} / {total} 条结果，其中 {valid} 条可评价"
        continuation = "先完成或处理本轮异常；未完成样本不记成模型漏判，不据此扩大批次。"
    elif not evaluation:
        headline, continuation = "80 条结果已到齐，等待正式评估", "页面已列出与内部参考的差异；正式评估前不进入下一批。"
    elif gate.get("passed") is True:
        headline, continuation = "80 条基础检验通过，可以继续下一阶段", "可以开始 400 条新的自然抽样，检查真实分布下的识别效果。现在仍不足以全量处理。"
    else:
        headline, continuation = "80 条已评估，本轮暂不扩大", "先看下面的额外判断、漏判、关键关系错误和待裁定项，区分模型问题与参考定义边界。"
    current = status.get("current_step") or status.get("phase") or headline
    detail = status.get("detail") or status.get("summary", "")
    next_action = status.get("next_action") or status.get("next", "")
    if isinstance(next_action, list):
        next_action = " → ".join(str(x) for x in next_action)
    blockers = status.get("blockers", [])
    if isinstance(blockers, str):
        blockers = [blockers]
    stage_note = f'<p class="muted">工作记录：{esc(current)}</p>' if current != headline else ''
    if detail:
        stage_note += f'<p>{esc(detail)}</p>'
    if next_action:
        stage_note += f'<p>正在执行／下一步：{esc(next_action)}</p>'
    if status.get("updated_at"):
        stage_note += f'<p class="minor">工作记录更新时间：{esc(status["updated_at"])}</p>'
    recorded = []
    for key, label in [("completed", "已完成"), ("issues", "技术记录")]:
        items = status.get(key, [])
        if isinstance(items, str):
            items = [items]
        recorded.extend(f'<li><strong>{label}：</strong>{esc(item)}</li>' for item in items)
    if recorded:
        stage_note += '<details><summary>查看准备工作与已记录的问题</summary><ul>' + ''.join(recorded) + '</ul></details>'
    alerts = ''.join(f'<p class="alert">{esc(s)}</p>' for s in data["integrity"] + blockers)
    if full and evaluation and gate.get("passed") is not True:
        totals = evaluation.get("totals", {})
        emotion = evaluation.get("per_group", {}).get("emotion", {})
        problems = []
        p_threshold, r_threshold = gate.get("precision_threshold", .95), gate.get("recall_threshold", .90)
        for group, metrics_set in [("全部明确含义", totals), ("六类情绪", emotion)]:
            for key, threshold, label in [("precision", p_threshold, "提取中与参考一致的比例"),
                                          ("recall", r_threshold, "参考明确含义被找回的比例")]:
                value = metrics_set.get(key)
                if value is None or value < threshold:
                    problems.append(f'{group}：{label}为 {pct(value)}，本轮继续条件为至少 {pct(threshold)}。')
        if evaluation.get("critical_failures"):
            problems.append(f'有 {len(evaluation["critical_failures"])} 项关键的否定、主体或交易时态判断未通过。')
        if totals.get("unknown_supported"):
            problems.append(f'有 {totals["unknown_supported"]} 项模型提取落在参考存在歧义的位置，需先由执行助手结合原文裁定。')
        if evaluation.get("integrity_errors"):
            problems.append("有输出格式或原文证据检查未通过，先处理运行问题。")
        alerts += ''.join(f'<p class="alert">{esc(s)}</p>' for s in problems)
    critical_n = len(evaluation.get("critical_failures", [])) if evaluation and full else sum(len(x["critical"]) for x in cases)
    cards = [(f"{valid} / {total}", "有效结果", "未运行和运行异常另列"),
             (str(metrics["tp"]) if valid else "—", "识别一致的含义", "与内部参考一致的正例数"),
             (str(metrics["fp"]) if valid else "—", "额外判断的含义", "模型有，参考没有；需看原文"),
             (str(metrics["fn"]) if valid else "—", "漏掉的明确含义", "参考有，模型未确认")]
    metric_html = ''.join(f'<div class="metric"><strong>{esc(n)}</strong><span>{esc(label)}</span><small>{esc(note)}</small></div>' for n, label, note in cards)
    aggregate = ''
    if valid:
        aggregate = (f'<p>在 {valid} 条有效结果中，{state_counts["same"]} 条与内部参考一致，'
                     f'{state_counts["different"]} 条有明确差异，{state_counts["disputed"]} 条仅涉及参考歧义。'
                     f'关键的否定、主体或交易时态差异：{critical_n} 项。</p>'
                     f'<p class="minor">按“含义”计：模型提取内容中与参考一致的比例 {pct(ratio(metrics["tp"], metrics["tp"]+metrics["fp"]))}；'
                     f'参考明确含义中被识别到的比例 {pct(ratio(metrics["tp"], metrics["tp"]+metrics["fn"]))}。'
                     '这些是内部参考比对结果，不是已证明的真实准确率；同一帖子可有多个含义。</p>')
    else:
        aggregate = '<p>识别一致、额外判断和漏判均尚无数据；完成推理后再计算识别比例。</p>'
    per_class = []
    for candidate in data["candidates"]:
        key, label = candidate["id"], candidate["label"]
        c = data["counts"][key]
        denom = c["reference_positive"]
        present = c["checked"] > 0
        per_class.append(f'<tr><td>{esc(label)}</td><td>{denom if present else "—"}</td><td>{c["tp"] if present else "—"}</td>'
                         f'<td>{c["fp"] if present else "—"}</td><td>{c["fn"] if present else "—"}</td>'
                         f'<td>{pct(ratio(c["tp"], c["tp"]+c["fp"])) if present else "—"}</td>'
                         f'<td>{pct(ratio(c["tp"], c["tp"]+c["fn"])) if present else "—"}</td>'
                         f'<td>{"正例不足 5 条，慎看百分比" if present and denom < 5 else ""}</td></tr>')
    kinds = collections.Counter(x["row"].get("source_kind") for x in cases)
    real = [x["row"] for x in cases if x["row"].get("source_kind") == "real_directed_easy"]
    dates = sorted(x["published_at"][:10] for x in real if x.get("published_at"))
    months = len({x[:7] for x in dates})
    date_note = f'{dates[0]} 至 {dates[-1]}，覆盖 {months} 个月' if dates else '日期未记录'
    run_model = data["manifest"].get("semantics", {}).get("model")
    model_note = ('已运行模型：' + run_model) if run_model else ('计划使用：' + plan.get("model", "Qwen 27B") + '；尚无运行记录')
    runtime = data["runtime"]
    elapsed = runtime.get("session_wall_seconds")
    elapsed_note = f'{elapsed / 60:.1f} 分钟' if isinstance(elapsed, (int, float)) else '尚未记录'
    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    case_html = ''.join(render_case(x, names) for x in cases)
    state_options = ''.join(f'<option value="{k}">{v}（{state_counts[k]}）</option>' for k, v in STATE.items())
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>情绪识别首轮试跑 · 实际进展与逐帖结果</title><style>{CSS}</style></head><body><main>
    <section class="hero"><p class="meta">情绪识别 · Qwen 27B 首轮试跑</p><h1>{esc(headline)}</h1>
    <p class="lead status">{esc(continuation)}</p>{stage_note}{alerts}
    <div class="subline">{chip('50 条真实短帖 + 30 条简单对照')}{chip('这次只判断明确意思')}{chip('尚未用于训练或股票预测')}</div></section>
    <section class="panel"><h2 style="margin-top:0">实际结果</h2><div class="metrics">{metric_html}</div>{aggregate}
    <p class="notice">这里的“参考”由助手先读原文、再独立审阅形成，<strong>不是你已经认可的金标准</strong>。额外判断不一定就是模型错；每一项都保留原文供检查。最初容易引发争议的题已在推理前替换或标为不确定。</p>
    <details><summary>展开 17 种明确含义的分项结果</summary><div class="table-wrap"><table><thead><tr><th>含义</th><th>参考正例</th><th>识别一致</th><th>额外判断</th><th>漏掉</th><th>提取中一致</th><th>参考被找回</th><th>样本限制</th></tr></thead><tbody>{''.join(per_class)}</tbody></table></div>
    <p class="minor">分项表只统计有效结果；“—”表示尚无可计算分母。真实帖子与人工对照不是自然分布，不能据此宣布全量效果。</p></details></section>
    <section class="panel"><div class="small-grid"><div><h2 style="margin-top:0">三个东西分开看</h2><ul><li><strong>明确含义：</strong>模型认为原文说了什么，例如已卖出、恐惧。</li><li><strong>支持分：</strong>模型估计原文支持该理解的程度。</li><li><strong>表达强度：</strong>该意思表达得有多强；不适合打强度时留空。</li></ul><p class="minor">两个分数都没有经过校准，<strong>0.90 不是 90% 正确概率</strong>。本轮没有把分数压成一个情绪总分。</p></div>
    <div><h2 style="margin-top:0">这一轮用了什么</h2><p>{kinds.get('real_directed_easy',0)} 条刻意挑选的明确真实帖子，{kinds.get('synthetic_control',0)} 条人工编写对照。</p><p>真实帖子：{esc(date_note)}，{len({x.get('stock_code') for x in real})} 只股票。</p><p>{esc(model_note)}<br>本轮推理程序耗时：{esc(elapsed_note)}</p><p class="minor">耗时不等于租赁总时长或账单。下一阶段为新的 400 条自然抽样，本页不冒充其结果。</p></div></div></section>
    <h2>逐条看原文与识别结果</h2><p>默认先完整展示原文和提取的意思；需要时展开证据与分数。没有把长篇推理过程塞进报告。</p>
    <div class="toolbar"><input id="search" type="search" placeholder="搜索股票、原文或明确含义" aria-label="搜索股票、原文或明确含义">
    <select id="state" aria-label="按结果筛选"><option value="">全部状态</option>{state_options}</select>
    <select id="source" aria-label="按来源筛选"><option value="">全部来源</option><option>真实帖子</option><option>人工编写对照</option></select>
    <button id="expand" type="button">展开当前证据</button></div><p id="visible" class="minor">共 {total} 条</p><div id="cases">{case_html}</div>
    <footer>生成时间：{esc(now)}。这是本地离线快照，结果更新后重新生成即可。<br>结果目录：{esc(run)}<br>规则版本：{esc(read_json(base/'runner/meaning_schema.json').get('version'))}。没有读取或展示模型的长篇推理。</footer>
    </main><script>
    const boxes=[...document.querySelectorAll('.case')];
    const search=document.getElementById('search'),state=document.getElementById('state'),source=document.getElementById('source');
    function filter(){{let n=0;const text=search.value.trim().toLowerCase();for(const box of boxes){{box.hidden=!!((text&&!box.dataset.search.toLowerCase().includes(text))||(state.value&&box.dataset.state!==state.value)||(source.value&&box.dataset.source!==source.value));if(!box.hidden)n++;}}document.getElementById('visible').textContent=`当前显示 ${{n}} / ${{boxes.length}} 条`;}}
    for(const control of [search,state,source])control.addEventListener('input',filter);
    document.getElementById('expand').addEventListener('click',()=>{{const shown=boxes.filter(x=>!x.hidden);const open=shown.some(x=>!x.querySelector('details').open);for(const box of shown)box.querySelector('details').open=open;document.getElementById('expand').textContent=open?'收起当前证据':'展开当前证据';}});
    </script></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--run", type=Path, default=None, help="Relative to base, or absolute")
    parser.add_argument("--status", type=Path, help="Optional JSON progress prose; cannot override actual gate")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    base = args.base.resolve()
    run = args.run or Path("runs/stage1-v1")
    run = run if run.is_absolute() else base / run
    # Plan remains separate from measured results. More specific status files override
    # general state; explicit --status wins. None can override the computed gate.
    plan = read_json(base / "deployment/plan.json", {})
    status = {k: plan[k] for k in ("current_step", "detail", "next_action", "updated_at", "blockers") if k in plan}
    status_paths = [base / "deployment/status.json", base / "status.json", base / "report/status.json"]
    if args.status:
        status_paths.append(args.status if args.status.is_absolute() else base / args.status)
    for status_path in status_paths:
        status.update(read_json(status_path, {}))
    out = args.out or base / "report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    output = build(base, run.resolve(), status)
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(output, encoding="utf-8")
    temporary.replace(out)
    print(json.dumps({"report": str(out.resolve()), "run": str(run.resolve()),
                      "status_sources": [str(p) for p in status_paths if p.exists()], "bytes": len(output.encode())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
