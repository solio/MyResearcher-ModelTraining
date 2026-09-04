"""Generate and validate the fixed M3 synthetic challenge input set.

This is an input-only diagnostic fixture.  It contains no seven-head labels,
predictions, thresholds, model artifacts, or adjudication decisions.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence


FIELDS = (
    "sample_id",
    "stock_code",
    "stock_name",
    "model_text",
    "expected_scope",
    "challenge_category",
    "construction_note",
)

EXPECTED_SCOPE_COUNTS = {
    "IN_DOMAIN_CONTROL": 20,
    "CLEAR_OOD": 20,
    "SHIFT_CHALLENGE": 60,
}

SHIFT_CATEGORIES = (
    "NEW_ENTITY_INDUSTRY",
    "NEW_TIME_EVENT",
    "PLATFORM_SLANG",
    "TRADITIONAL_CHINESE",
    "TYPO_HOMOPHONE",
    "MISSING_CONTEXT_LINK_EMOJI_OVERLOAD",
)

EXPECTED_CATEGORY_COUNTS = {
    "IN_DOMAIN_CONTROL": 20,
    "CLEAR_OOD_NOISE": 20,
    **{category: 10 for category in SHIFT_CATEGORIES},
}


def _record(
    sample_id: str,
    stock_code: str,
    stock_name: str,
    model_text: str,
    expected_scope: str,
    challenge_category: str,
    construction_note: str,
) -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "stock_code": stock_code,
        "stock_name": stock_name,
        "model_text": model_text,
        "expected_scope": expected_scope,
        "challenge_category": challenge_category,
        "construction_note": construction_note,
    }


def _control_records() -> list[dict[str, str]]:
    texts = [
        "公司订单保持增长，短线回调后我会继续观察，不追高。",
        "今天成交量温和放大，先看支撑位能否守住。",
        "季度利润改善但估值不低，仓位按计划持有。",
        "板块整体走弱，这只股票相对抗跌，暂时不做交易。",
        "公告披露新产能按期投产，后续关注现金流。",
        "早盘冲高回落，价格仍在前期平台附近震荡。",
        "我没有新的买卖计划，只记录今天的量价变化。",
        "同行订单释放较快，目标公司的竞争位置还要继续跟踪。",
        "政策预期带动板块反弹，但消息兑现仍需验证。",
        "持仓成本较高，先等待财报和管理层说明。",
        "这次回撤没有改变中期判断，按纪律保持小仓位。",
        "公司毛利率连续两个季度改善，基本面信号偏正面。",
        "北向资金流入放缓，短线情绪可能继续波动。",
        "如果不能站稳前高，我只观察，不把条件当成已执行动作。",
        "市场普跌时它跌幅较小，是否持续还要看成交结构。",
        "公告没有超出预期，股价反应偏中性。",
        "行业价格战仍在，利润弹性需要更多证据。",
        "今天没有操作，继续记录仓位和盘面变化。",
        "估值回到历史中位附近，风险收益暂时比较均衡。",
        "公司产品更新已发布，后续看客户采用速度。",
    ]
    return [
        _record(
            f"M3SYN-IN-{index:03d}",
            f"600{index:03d}",
            f"合成证券{index:03d}",
            text,
            "IN_DOMAIN_CONTROL",
            "IN_DOMAIN_CONTROL",
            "合成中文证券讨论，保留完整句式和常见市场上下文；不包含人工标签。",
        )
        for index, text in enumerate(texts, start=1)
    ]


def _clear_ood_records() -> list[dict[str, str]]:
    texts = [
        "A kettle sings softly beside the blue window.",
        "{\\x00\\x01\\x02} 010101 ??? []",
        "今日は株式市場ではなく、電車の時刻表についてです。",
        "lorem ipsum dolor sit amet, consectetur adipiscing elit",
        "🐙🛰️🧩🎈🎈🎈",
        "SELECT * FROM planets WHERE ring = true;",
        "the quick brown fox jumps over 13 silent boxes",
        "%%%%% ***** ///// <binary payload omitted>",
        "Recipe: whisk flour, water, and salt for ten minutes.",
        "नमस्ते दुनिया — unrelated language sample",
        "[SYSTEM LOG] cpu=91% mem=4G status=unknown",
        "A short poem about rain on a railway platform.",
        "01001000 01100101 01101100 01101100 01101111",
        "print(\"hello world\") # unrelated code fragment",
        "🧪🧪🧪 no textual context 🧪🧪🧪",
        "The museum closes at five; bring a valid ticket.",
        "ZXCV-QWER-ASDF-1234 noise token stream",
        "Здравствуй, мир; это не证券讨论。",
        "<svg><path d=\"M0 0\"/></svg>",
        "A weather station reports calm winds over the ocean.",
    ]
    return [
        _record(
            f"M3SYN-OOD-{index:03d}",
            f"OOD{index:03d}",
            "Synthetic unrelated input",
            text,
            "CLEAR_OOD",
            "CLEAR_OOD_NOISE",
            "明显无关内容或非语言噪声，用于输入管线的 CLEAR_OOD 趋势诊断；不是真实 OOD 判定。",
        )
        for index, text in enumerate(texts, start=1)
    ]


def _shift_templates() -> Mapping[str, Sequence[tuple[str, str, str]]]:
    return {
        "NEW_ENTITY_INDUSTRY": [
            ("991001", "星河储能", "星河储能的固态电池产线刚公布，产业链估值该怎么看？"),
            ("991002", "云际机器人", "云际机器人切入仓储场景，订单兑现节奏还需要哪些证据？"),
            ("991003", "深海算力", "深海算力的液冷方案听起来新，但商业化路径仍然模糊。"),
            ("991004", "绿洲合成生物", "绿洲合成生物的中试数据很新，同行比较应看哪些指标？"),
            ("991005", "北辰低空", "北辰低空把无人机服务拓到园区，收入确认会不会太快？"),
            ("991006", "墨子量子芯", "墨子量子芯属于新概念实体，暂时找不到足够的可比公司。"),
            ("991007", "海岳碳纤维", "海岳碳纤维扩产消息出来后，价格和需求哪个更关键？"),
            ("991008", "远岫氢港", "远岫氢港的港口制氢项目是否能形成稳定现金流？"),
            ("991009", "极昼智能矿山", "极昼智能矿山进入新行业，客户集中度风险怎么观察？"),
            ("991010", "沧澜卫星", "沧澜卫星的新通信业务刚起步，订单与估值还没有历史参照。"),
        ],
        "NEW_TIME_EVENT": [
            ("992001", "合成证券A", "2037年火星交易窗口开启前，这个板块的资金会怎样迁移？"),
            ("992002", "合成证券B", "下个世纪的量子清算日临近，今天的走势还有参考意义吗？"),
            ("992003", "合成证券C", "午夜临时公告后的第七码盘，成交量信号该如何理解？"),
            ("992004", "合成证券D", "在尚未发生的下一次超级台风事件里，保险链条会受什么影响？"),
            ("992005", "合成证券E", "这次虚构的年度重力节会不会改变新能源需求预期？"),
            ("992006", "合成证券F", "跨越三十年的远期交付合同，短线估值能否提前定价？"),
            ("992007", "合成证券G", "明天之后的第二个季度再融资窗口，条件是否仍然有效？"),
            ("992008", "合成证券H", "未来十年后的监管试点尚未落地，当前价格该如何解读？"),
            ("992009", "合成证券I", "突发的虚拟交易所停摆事件，会怎样影响传统券商？"),
            ("992010", "合成证券J", "非公开的下一时区开盘数据被提及，但时间锚点并不清楚。"),
        ],
        "PLATFORM_SLANG": [
            ("993001", "合成证券K", "这票今天直接起飞，主力yyds，后面是不是要继续冲？"),
            ("993002", "合成证券L", "开盘就是核按钮，老师们怎么看这个情绪拐点？"),
            ("993003", "合成证券M", "已经被按在地板摩擦，躺平还是等反包？"),
            ("993004", "合成证券N", "量化小作文刷屏了，感觉要开香槟但又不敢确认。"),
            ("993005", "合成证券O", "这波属于情绪接力还是大面预定，懂的都懂。"),
            ("993006", "合成证券P", "老师带带我，水下捞鱼的黑话到底指什么？"),
            ("993007", "合成证券Q", "今天又被市场教育，先苟住再说，别问为什么。"),
            ("993008", "合成证券R", "这是不是传说中的龙头首阴，评论区给个信号。"),
            ("993009", "合成证券S", "封板yyds但盘口像在演戏，平台语气和真实意图对不上。"),
            ("993010", "合成证券T", "不懂就问，这个‘破冰’梗在新平台上到底怎么用？"),
        ],
        "TRADITIONAL_CHINESE": [
            ("994001", "合成證券甲", "公司訂單回暖但估值偏高，暫時觀察量價變化。"),
            ("994002", "合成證券乙", "公告提到新產能，後續要看現金流能否跟上。"),
            ("994003", "合成證券丙", "大盤震盪之下，這檔股票相對抗跌是否能延續？"),
            ("994004", "合成證券丁", "我沒有新增交易計畫，只記錄今天的成交結構。"),
            ("994005", "合成證券戊", "如果無法站穩前高，就先等待，不把條件當成已執行。"),
            ("994006", "合成證券己", "政策預期帶動反彈，但消息落地還需要更多證據。"),
            ("994007", "合成證券庚", "毛利率連續改善，基本面訊號偏正面但仍要覆核。"),
            ("994008", "合成證券辛", "市場普跌時跌幅較小，短線情緒仍可能反覆。"),
            ("994009", "合成證券壬", "估值回到歷史中位附近，風險收益暫時均衡。"),
            ("994010", "合成證券癸", "新產品剛發佈，客戶採用速度尚未明朗。"),
        ],
        "TYPO_HOMOPHONE": [
            ("995001", "合成证券甲", "今天涨停板像长亭板，成交量的意思还看得懂吗？"),
            ("995002", "合成证券乙", "这只票要是再跌，我就先观望，不想变成韭菜鱼。"),
            ("995003", "合成证券丙", "业绩暴增还是爆增，标题写错会不会影响判断？"),
            ("995004", "合成证券丁", "主力是不是在洗盘，还是在洗碗，语音输入有点乱。"),
            ("995005", "合成证券戊", "低吸和低息只差一个字，当前讨论到底指哪一个？"),
            ("995006", "合成证券己", "看多写成看豆，评论里还在讨论豆粕，语义发生偏移。"),
            ("995007", "合成证券庚", "这波回撤我先拿住，‘拿住’误写成‘辣住’了。"),
            ("995008", "合成证券辛", "板块轮动太快，同行相对强弱的错别字需要人工确认。"),
            ("995009", "合成证券壬", "想说反弹却打成反糖，短句没有更多上下文。"),
            ("995010", "合成证券癸", "公告里‘增持’输入成‘蒸池’，实体和动作都可能被误读。"),
        ],
        "MISSING_CONTEXT_LINK_EMOJI_OVERLOAD": [
            ("996001", "合成证券甲", "这个可以吗？"),
            ("996002", "合成证券乙", "https://example.invalid/post/996002 🚀🚀🚀"),
            ("996003", "合成证券丙", "？？？😅😅😅 看不懂了"),
            ("996004", "合成证券丁", "按那个来，懂的自然懂。"),
            ("996005", "合成证券戊", "https://example.invalid/a … [图片] [图片]"),
            ("996006", "合成证券己", "它又这样了🙃，没有上下文无法判断。"),
            ("996007", "合成证券庚", "前面说的那件事，继续？"),
            ("996008", "合成证券辛", "📈📉📈📉📈📉"),
            ("996009", "合成证券壬", "看链接里的内容再说，正文为空 https://example.invalid/empty"),
            ("996010", "合成证券癸", "那就这样吧……🤷‍♀️ 具体对象和时间都缺失。"),
        ],
    }


def _shift_records() -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for category, rows in _shift_templates().items():
        for index, (stock_code, stock_name, text) in enumerate(rows, start=1):
            records.append(
                _record(
                    f"M3SYN-SHIFT-{category}-{index:02d}",
                    stock_code,
                    stock_name,
                    text,
                    "SHIFT_CHALLENGE",
                    category,
                    f"仅构造 {category} 的分布变化压力，不提供七头标签，不代表正式 OOD 或 Gold。",
                )
            )
    return records


def build_records() -> list[dict[str, str]]:
    """Return the fixed records in a stable category/within-category order."""

    records = _control_records() + _clear_ood_records() + _shift_records()
    validate_records(records)
    return records


def validate_records(records: Iterable[Mapping[str, object]]) -> None:
    rows = list(records)
    if len(rows) != 100:
        raise ValueError(f"expected 100 records, found {len(rows)}")
    sample_ids = [row.get("sample_id") for row in rows]
    if any(not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids):
        raise ValueError("every record requires a non-empty string sample_id")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_id values must be unique")
    for row in rows:
        if set(row) != set(FIELDS):
            raise ValueError(f"record fields mismatch: {sorted(row)}")
        if any(not isinstance(row[field], str) for field in FIELDS):
            raise ValueError("all challenge fields must be strings")
        if not row["model_text"]:
            raise ValueError(f"empty model_text for {row['sample_id']}")
        if row["expected_scope"] not in EXPECTED_SCOPE_COUNTS:
            raise ValueError(f"unexpected expected_scope: {row['expected_scope']}")
        if row["challenge_category"] not in EXPECTED_CATEGORY_COUNTS:
            raise ValueError(f"unexpected challenge_category: {row['challenge_category']}")
    scope_counts = Counter(row["expected_scope"] for row in rows)
    category_counts = Counter(row["challenge_category"] for row in rows)
    if dict(scope_counts) != EXPECTED_SCOPE_COUNTS:
        raise ValueError(f"scope counts mismatch: {dict(scope_counts)}")
    if dict(category_counts) != EXPECTED_CATEGORY_COUNTS:
        raise ValueError(f"category counts mismatch: {dict(category_counts)}")
    for row in rows:
        expected_scope = row["expected_scope"]
        category = row["challenge_category"]
        if expected_scope == "SHIFT_CHALLENGE" and category not in SHIFT_CATEGORIES:
            raise ValueError("SHIFT_CHALLENGE rows must use one of the six shift categories")
        if expected_scope != "SHIFT_CHALLENGE" and category not in {
            "IN_DOMAIN_CONTROL",
            "CLEAR_OOD_NOISE",
        }:
            raise ValueError("control/OOD rows use their dedicated non-shift category")


def write_jsonl(records: Sequence[Mapping[str, str]], output_path: Path) -> None:
    validate_records(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({field: row[field] for field in FIELDS}, ensure_ascii=False, separators=(",", ":"))
        for row in records
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, str]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return rows


def _default_output() -> Path:
    return Path(__file__).resolve().parents[1] / "tests/fixtures/m3_synthetic_challenge_v1.jsonl"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=_default_output())
    parser.add_argument("--check", action="store_true", help="validate an existing JSONL without rewriting it")
    args = parser.parse_args(argv)
    if args.check:
        records = load_jsonl(args.output)
        validate_records(records)
        print(json.dumps({"path": str(args.output), "records": len(records), "status": "PASS"}, ensure_ascii=False))
        return 0
    records = build_records()
    write_jsonl(records, args.output)
    print(json.dumps({"path": str(args.output), "records": len(records), "status": "WRITTEN"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
