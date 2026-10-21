#!/usr/bin/env python3
"""把 result.json / status.json 注入 study-card-template.html，生成可双击打开的单文件卡片。

用法:
    python3 build_card.py <result.json 或 status.json> <输出.html>

会做两类检查:
    错误(退出码 1，不生成文件): 缺字段、状态值非法、一句话结论太长或含术语等。
    提醒(照常生成): 不影响生成，但建议修改。
只用 Python 标准库。
"""
import json
import re
import sys
from pathlib import Path

TEMPLATE = Path(__file__).with_name("study-card-template.html")

STATUSES = {"holds", "weak", "fails", "insufficient", "retracted", "pending", "review"}
HALVES = {"buy", "avoid", "method"}

# 一句话结论、交易含义、数字解释里不许出现的术语（读者要能直接看懂）
JARGON = [
    (r"spearman", "Spearman"),
    (r"(?<![a-z])p\s*[=<>值]", "p 值"),
    (r"p值", "p 值"),
    (r"(?<![a-z])pp(?![a-z])", "pp（请写“个百分点”）"),
    (r"z\s*值|z-?score", "z 值"),
    (r"bootstrap", "bootstrap"),
    (r"wilson", "Wilson"),
    (r"brier", "Brier"),
    (r"(?<![a-z])ic(?![a-z])", "IC"),
    (r"(?<![a-z])auc(?![a-z])", "AUC"),
    (r"stance_index|a5\.1|m12", "模型内部名"),
]

STUDY_REQUIRED = ["study_id", "half", "title", "question", "status", "headline",
                  "for_my_trading", "charts", "gates", "numbers", "limits",
                  "method", "reproduce", "updated_at"]
BOARD_REQUIRED = ["title", "items", "updated_at"]
ITEM_REQUIRED = ["study_id", "half", "question", "status", "one_line"]


def jargon_in(text):
    low = text.lower()
    return [name for pat, name in JARGON if re.search(pat, low)]


def check_study(d, errors, warns):
    for k in STUDY_REQUIRED:
        if k not in d:
            errors.append(f"缺字段: {k}")
    if d.get("status") not in STATUSES:
        errors.append(f"status 必须是 {sorted(STATUSES)} 之一，现在是 {d.get('status')!r}")
    if d.get("half") not in HALVES:
        errors.append(f"half 必须是 {sorted(HALVES)} 之一")
    h = d.get("headline", "")
    if len(h) > 60:
        errors.append(f"headline 太长（{len(h)} 字，上限 60）")
    for j in jargon_in(h):
        errors.append(f"headline 含术语: {j}")
    trade = d.get("for_my_trading", [])
    if len(trade) > 3:
        errors.append("for_my_trading 最多 3 条")
    if d.get("status") not in {"pending"}:
        if not trade:
            errors.append("for_my_trading 不能为空（待测状态除外）")
        elif not re.search(r"不要|别|不适合", trade[-1]):
            errors.append("for_my_trading 最后一条必须写“什么情况下不要用”（含“不要/别/不适合”）")
    for i, t in enumerate(trade):
        for j in jargon_in(t):
            errors.append(f"for_my_trading 第 {i+1} 条含术语: {j}")
    nums = d.get("numbers", [])
    if len(nums) > 6:
        errors.append("numbers 最多 6 个")
    for i, n in enumerate(nums):
        if not n.get("meaning"):
            errors.append(f"numbers 第 {i+1} 个缺 meaning（这个数字的意思）")
        for j in jargon_in(n.get("meaning", "")):
            warns.append(f"numbers 第 {i+1} 个的 meaning 含术语: {j}")
    for i, c in enumerate(d.get("charts", [])):
        if not c.get("title"):
            errors.append(f"charts 第 {i+1} 张缺 title")
        if not c.get("unit"):
            warns.append(f"charts 第 {i+1} 张没写 unit（如 %）")
        for g in c.get("groups", []):
            for b in g.get("bars", []):
                if not isinstance(b.get("value"), (int, float)):
                    errors.append(f"charts 第 {i+1} 张有柱子缺数值: {b.get('name')}")
    gates = d.get("gates", [])
    if d.get("status") in {"holds", "weak", "fails"} and len(gates) < 4:
        errors.append("已下结论的研究必须列出四关")
    for g in gates:
        if g.get("pass") not in (True, False, None):
            errors.append(f"gate {g.get('name')} 的 pass 只能是 true/false/null")
        if not g.get("plain"):
            warns.append(f"gate {g.get('name')} 缺白话问句 plain")
    if d.get("status") in {"holds", "weak"} and any(g.get("pass") is not True for g in gates):
        errors.append("状态是成立/弱成立，但四关并非全部通过")
    ex = d.get("examples", [])
    if not ex and d.get("status") not in {"pending", "insufficient"}:
        warns.append("没有真实帖子例子（examples），建议放 3–5 条")
    if len(ex) > 5:
        warns.append("examples 超过 5 条")
    if not d.get("limits"):
        warns.append("limits 为空——几乎不可能没有局限")


def check_board(d, errors, warns):
    for k in BOARD_REQUIRED:
        if k not in d:
            errors.append(f"缺字段: {k}")
    for i, it in enumerate(d.get("items", [])):
        for k in ITEM_REQUIRED:
            if k not in it:
                errors.append(f"items 第 {i+1} 条缺字段: {k}")
        if it.get("status") not in STATUSES:
            errors.append(f"items 第 {i+1} 条 status 非法: {it.get('status')!r}")
        if len(it.get("one_line", "")) > 60:
            warns.append(f"items 第 {i+1} 条 one_line 超过 60 字")
        for j in jargon_in(it.get("one_line", "")):
            errors.append(f"items 第 {i+1} 条 one_line 含术语: {j}")


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    d = json.loads(src.read_text(encoding="utf-8"))
    errors, warns = [], []
    if d.get("kind") == "board":
        check_board(d, errors, warns)
    else:
        d.setdefault("kind", "study")
        check_study(d, errors, warns)
    for w in warns:
        print("提醒:", w)
    if errors:
        for e in errors:
            print("错误:", e)
        print(f"未生成 {out}（{len(errors)} 个错误）")
        sys.exit(1)
    tpl = TEMPLATE.read_text(encoding="utf-8")
    payload = json.dumps(d, ensure_ascii=False, indent=1).replace("</", "<\\/")
    pat = re.compile(r'(<script id="study-data" type="application/json">)(.*?)(</script>)', re.S)
    if not pat.search(tpl):
        print("错误: 模板里找不到 study-data 标签")
        sys.exit(1)
    html = pat.sub(lambda m: m.group(1) + "\n" + payload + "\n" + m.group(3), tpl, count=1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"已生成 {out}")


if __name__ == "__main__":
    main()
