#!/usr/bin/env python3
"""Prepare a small, answer-free v0.3 auxiliary annotation pilot.

This module deliberately stops at task preparation.  It reads the finalized
Human Train100 identities and the source rows for those same identities, then
emits rows containing text and a sparse list of auxiliary heads to annotate.
No answer, model probability, prediction, or label is copied into the task
payload.  The command never imports the model runtime and does not contact the
labeling application.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HUMAN_TRAIN = REPO_ROOT / "runs" / "feedback-loop-v3-finalized-20260910" / "human-train-100-seven-head.jsonl"
DEFAULT_SOURCE_TRAIN = (
    REPO_ROOT / "runs" / "feedback-loop-v1-20260909" / "selection-train.jsonl",
    REPO_ROOT / "runs" / "feedback-loop-v2-20260909" / "selection-train.jsonl",
    REPO_ROOT / "runs" / "feedback-loop-v3-20260910" / "selection-train.jsonl",
)
DEFAULT_OUTPUT_DIR = Path("/Users/mac/Documents/trae_projects/MyResearcher/human-review-artifacts/m6b-v03-auxiliary-pilot-20260911")
DEFAULT_BATCH_ID = "m6b-v03-auxiliary-pilot-20260911"

AUXILIARY_FACTORS = (
    "expectation_phase",
    "arousal_level",
    "desired_price_direction",
    "market_scope",
    "discourse_focus",
)
LEGACY_HEADS = (
    "target_mode",
    "stance",
    "emotion_primary",
    "emotion_target",
    "action_tendency",
    "context_dependency",
    "reasoning_tags",
)
SOURCE_NAMES = ("v1", "v2", "v3")

# These are only applicability cues.  They select which question is useful;
# they never select an answer.  The lists intentionally favour explicit cues
# over broad assumptions so that an inapplicable factor is not added merely to
# reach a quota.  ``market_scope`` and ``arousal_level`` additionally require
# the corresponding finalized top-level Human Train answer (see
# ``_applicable_factors``).
EXPECTATION_CUES = re.compile(
    r"明天|后天|下周|下个月|未来|将|会|预计|打算|希望|如果|何时|本周|年后|之后|目前|现在|今天|已经|已然|刚刚|刚|本月|今年|昨天|上周|曾经|早就|起初|一开始|正式|发布|等待|等着|时候|之前|以后"
)
AROUSAL_CUES = re.compile(
    r"愤怒|生气|气死|焦虑|担心|害怕|恐惧|兴奋|激动|开心|高兴|期待|后悔|遗憾|崩溃|绝望|无语|烦|郁闷|心寒|痛苦|失望|爽|哈哈|哭|难受|蠢|骂|作孽|活该|死的心|心气|不相信|不应该"
)
# These cues describe a price/result direction.  Author actions such as 买入、
# 卖出、清仓、加仓 and 接回 are deliberately absent; they do not by
# themselves establish a desired or observed price direction.
PRICE_DIRECTION_CUES = re.compile(
    r"涨|跌|上行|下行|上涨|下跌|涨停|跌停|反弹|回调|拉升|下杀|突破|跌破|震荡|复苏|起飞|低位|高位|涨幅|跌幅|股价|价格|跑赢|跑不过|领先|落后|破位|回本|大涨|大跌|强势|走强|走弱|偏强|偏弱|红盘|绿盘|红了|绿了"
)
MARKET_SCOPE_CUES = re.compile(
    r"大盘|市场|行业|板块|资金|机构|外资|基金|宏观|指数|A股|沪深|北向|流动性|赛道|周期|散户|主力|光伏|组件|硅片|价格|中报|政策"
)
# Sentence-final particles (吧/呢/啊/呀) are intentionally absent.  A
# discourse task needs a participant, quoted claim, or narrative cue rather
# than tone alone.
DISCOURSE_FOCUS_CUES = re.compile(
    r"我|本人|持仓|仓位|买入|卖出|看好|认为|感觉|觉得|有人|大家|你|他|她|散户|股民|论坛|评论|新闻|消息|公告|大盘|市场|行业|板块|公司|股票|主力|机构|兄弟们|各位|我们|骂|说|告诉|问|为什么|怎么|到底"
)
FACTOR_CUES = {
    "expectation_phase": EXPECTATION_CUES,
    "arousal_level": AROUSAL_CUES,
    "desired_price_direction": PRICE_DIRECTION_CUES,
    "market_scope": MARKET_SCOPE_CUES,
    "discourse_focus": DISCOURSE_FOCUS_CUES,
}
LOW_INFORMATION = {"……", "...", "．．．", "又了", "无", "暂无", "略", "？", "?"}
SEMANTIC_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]")
TRUNCATED_SAMPLE_IDS = {
    "fl1-a00e3349a491d1784806cc76713f74ae68ffe13f9dc4f8e27c681acc89ac2170",
}


class PilotPreparationError(ValueError):
    """A deterministic input or selection contract violation."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise PilotPreparationError(f"JSONL missing: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PilotPreparationError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise PilotPreparationError(f"JSONL row is not an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _is_low_information(text: Any) -> bool:
    if not isinstance(text, str) or not text.strip():
        return True
    compact = re.sub(r"\s+", "", text)
    return compact in LOW_INFORMATION or SEMANTIC_RE.search(text) is None


def _looks_truncated(sample_id: str, text: str, text_length: Any) -> bool:
    """Reject known 40-character collector cuts without guessing missing text."""

    if sample_id in TRUNCATED_SAMPLE_IDS:
        return True
    try:
        length = int(text_length)
    except (TypeError, ValueError):
        length = len(text)
    # The old collector cap is visible as an unpunctuated 39/40-character
    # prefix.  A complete sentence at shorter lengths remains eligible.
    if length >= 40:
        return True
    return length >= 39 and not re.search(r"[。！？!?…．.]$", text)


def _length_bucket(text: str, value: Any = None) -> str:
    try:
        length = int(value) if value is not None else len(text)
    except (TypeError, ValueError):
        length = len(text)
    if length < 10:
        return "SHORT"
    if length <= 25:
        return "MEDIUM"
    return "LONG"


def _stable_key(sample_id: str) -> str:
    return hashlib.sha256(f"{DEFAULT_BATCH_ID}|{sample_id}".encode("utf-8")).hexdigest()


def _numeric_selection_score(row: Mapping[str, Any]) -> float:
    """Read existing uncertainty ordering without copying it to the payload."""

    value = row.get("aggregate_uncertainty")
    if isinstance(value, (int, float)):
        return float(value)
    value = row.get("selection_score")
    if isinstance(value, (int, float)):
        return float(value)
    # V1/V2 coverage rows carry a dictionary score.  They remain eligible but
    # sort after rows with a numeric uncertainty score.
    return 0.0


def _source_name(path: Path) -> str:
    name = path.parent.name
    match = re.match(r"feedback-loop-(v[123])-", name)
    if not match:
        raise PilotPreparationError(f"unexpected source path (only V1/V2/V3 selection-train is allowed): {path}")
    return match.group(1)


def load_human_train(path: Path = DEFAULT_HUMAN_TRAIN) -> dict[str, dict[str, Any]]:
    """Load only identities/metadata from the finalized seven-head pool."""

    rows = _read_jsonl(path)
    if len(rows) != 700:
        raise PilotPreparationError(f"expected 700 finalized Human Train head rows, got {len(rows)}")
    by_sample: dict[str, dict[str, Any]] = {}
    heads_by_sample: dict[str, set[str]] = {}
    for row in rows:
        sample_id = row.get("sample_id")
        head = row.get("head")
        if not isinstance(sample_id, str) or not sample_id:
            raise PilotPreparationError("Human Train100 row lacks sample_id")
        if head not in LEGACY_HEADS:
            raise PilotPreparationError(f"unexpected Human Train100 head: {head!r}")
        if row.get("schema_version") != "semantic-schema-calibrated-v0.2.1":
            raise PilotPreparationError(f"Human Train100 row has unexpected schema_version: {sample_id}")
        if "auxiliary_factors" in row:
            raise PilotPreparationError(f"Human Train100 row must not carry auxiliary factors: {sample_id}")
        if not row.get("is_final"):
            raise PilotPreparationError(f"Human Train100 row is not final: {sample_id}")
        heads_by_sample.setdefault(sample_id, set()).add(str(head))
        sample = by_sample.setdefault(
            sample_id,
            {"sample_id": sample_id, "metadata": row.get("metadata", {}), "answers": {}},
        )
        if head in sample["answers"]:
            raise PilotPreparationError(f"duplicate Human Train100 head: {sample_id}/{head}")
        # Answers are retained in memory solely for factor applicability.  The
        # task writer never serializes this field.
        sample["answers"][str(head)] = row.get("answer")
    if len(by_sample) != 100:
        raise PilotPreparationError(f"expected exactly 100 Human Train samples, got {len(by_sample)}")
    if any(heads != set(LEGACY_HEADS) for heads in heads_by_sample.values()):
        raise PilotPreparationError("Human Train100 must contain exactly seven final heads per sample")
    return by_sample


def load_source_rows(paths: Sequence[Path] = DEFAULT_SOURCE_TRAIN) -> dict[str, dict[str, Any]]:
    """Read source text rows for the same 100 IDs, never labels or predictions."""

    source_rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        if path.name != "selection-train.jsonl":
            raise PilotPreparationError(f"only selection-train.jsonl is allowed as a source: {path}")
        source = _source_name(path)
        for row in _read_jsonl(path):
            sample_id = row.get("sample_id")
            text = row.get("text") or row.get("model_text")
            if not isinstance(sample_id, str) or not sample_id:
                raise PilotPreparationError(f"source row lacks sample_id: {path}")
            if row.get("frozen_role") != "HUMAN_TRAIN":
                raise PilotPreparationError(f"source row is not HUMAN_TRAIN: {sample_id}")
            if not isinstance(text, str) or not text.strip():
                raise PilotPreparationError(f"source row lacks text: {sample_id}")
            if not isinstance(row.get("exact_content_group"), str) or not row["exact_content_group"]:
                raise PilotPreparationError(f"source row lacks exact_content_group: {sample_id}")
            reasons = row.get("sampling_reasons", [])
            if not isinstance(reasons, list) or not all(isinstance(reason, str) for reason in reasons):
                raise PilotPreparationError(f"source row has invalid sampling_reasons: {sample_id}")
            if sample_id in source_rows:
                raise PilotPreparationError(f"duplicate source sample_id: {sample_id}")
            source_rows[sample_id] = {
                "sample_id": sample_id,
                "text": text,
                "observation_id": row.get("observation_id"),
                "stock_code": row.get("stock_code"),
                "stock_name": row.get("stock_name"),
                "exact_content_group": row["exact_content_group"],
                "sampling_reasons": sorted(set(reasons)),
                "text_length": row.get("text_length", len(text)),
                "source_batch": source,
                "selection_score": row.get("selection_score"),
                "aggregate_uncertainty": row.get("aggregate_uncertainty"),
            }
    return source_rows


def _applicable_factors(text: str, answers: Mapping[str, Any]) -> tuple[str, ...]:
    """Return questions justified by text plus finalized top-level context.

    ``answers`` is never copied into the task row.  It is only used to avoid
    asking factors that are structurally inapplicable: market scope is a
    refinement of ``emotion_target=MARKET`` and arousal requires both an
    explicit non-sentinel emotion and textual affect evidence.
    """

    factors: list[str] = []
    if FACTOR_CUES["expectation_phase"].search(text):
        factors.append("expectation_phase")
    emotion = answers.get("emotion_primary")
    if emotion not in {None, "NONE_EXPLICIT", "UNKNOWN"} and FACTOR_CUES["arousal_level"].search(text):
        factors.append("arousal_level")
    if FACTOR_CUES["desired_price_direction"].search(text):
        factors.append("desired_price_direction")
    if answers.get("emotion_target") == "MARKET" and FACTOR_CUES["market_scope"].search(text):
        factors.append("market_scope")
    if FACTOR_CUES["discourse_focus"].search(text):
        factors.append("discourse_focus")
    return tuple(factors)


def _ranked_candidates(
    source_rows: Mapping[str, Mapping[str, Any]],
    human_train: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in source_rows.values():
        text = row["text"]
        if _is_low_information(text) or _looks_truncated(str(row["sample_id"]), text, row.get("text_length")):
            continue
        factors = _applicable_factors(text, human_train[str(row["sample_id"])].get("answers", {}))
        if not factors:
            continue
        candidate = dict(row)
        candidate["heads"] = factors
        candidate["length_bucket"] = _length_bucket(text, row.get("text_length"))
        candidates.append(candidate)
    return sorted(
        candidates,
        key=lambda row: (-_numeric_selection_score(row), _stable_key(str(row["sample_id"])), str(row["sample_id"])),
    )


def select_pilot_samples(
    human_train: Mapping[str, Mapping[str, Any]],
    source_rows: Mapping[str, Mapping[str, Any]],
    *,
    target_samples: int = 20,
) -> list[dict[str, Any]]:
    """Select distinct groups with factor, length and source coverage."""

    if set(human_train) != set(source_rows):
        missing = sorted(set(human_train) - set(source_rows))
        extra = sorted(set(source_rows) - set(human_train))
        raise PilotPreparationError(f"Human Train/source identity mismatch; missing={missing[:3]}, extra={extra[:3]}")
    ranked = _ranked_candidates(source_rows, human_train)
    if len(ranked) < target_samples:
        raise PilotPreparationError(f"only {len(ranked)} eligible Human Train rows; need {target_samples}")

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_groups: set[str] = set()

    def add_first(predicate: Any) -> None:
        if len(selected) >= target_samples:
            return
        for candidate in ranked:
            sample_id = str(candidate["sample_id"])
            group = str(candidate.get("exact_content_group") or sample_id)
            if sample_id in selected_ids or group in selected_groups or not predicate(candidate):
                continue
            selected.append(candidate)
            selected_ids.add(sample_id)
            selected_groups.add(group)
            return

    # First guarantee that every auxiliary factor has at least one textually
    # applicable task, then preserve all three length buckets and each source
    # generation.  The remaining slots are confidence/uncertainty-ranked.
    for factor in AUXILIARY_FACTORS:
        add_first(lambda row, factor=factor: factor in row["heads"])
    for bucket in ("SHORT", "MEDIUM", "LONG"):
        add_first(lambda row, bucket=bucket: row["length_bucket"] == bucket)
    for source in SOURCE_NAMES:
        add_first(lambda row, source=source: row["source_batch"] == source)
        add_first(lambda row, source=source: row["source_batch"] == source)
    for candidate in ranked:
        if len(selected) >= target_samples:
            break
        add_first(lambda row, candidate=candidate: row["sample_id"] == candidate["sample_id"])

    if len(selected) != target_samples:
        raise PilotPreparationError(f"selection produced {len(selected)} rows, expected {target_samples}")
    assignment_count = sum(len(row["heads"]) for row in selected)
    if assignment_count > 50:
        raise PilotPreparationError(f"selection produced {assignment_count} assignments; limit is 50")
    if {factor for row in selected for factor in row["heads"]} != set(AUXILIARY_FACTORS):
        raise PilotPreparationError("all five auxiliary factors must be covered")
    if len({str(row.get("exact_content_group") or row["sample_id"]) for row in selected}) != len(selected):
        raise PilotPreparationError("selected exact_content_group values must be distinct")
    return selected


def _task_row(row: Mapping[str, Any], *, batch_id: str) -> dict[str, Any]:
    """Build an answer-free labeling payload row."""

    metadata = {
        "observation_id": row.get("observation_id"),
        "stock_code": row.get("stock_code"),
        "stock_name": row.get("stock_name"),
        "exact_content_group": row.get("exact_content_group"),
        "sampling_reasons": list(row.get("sampling_reasons", [])),
        "length_bucket": row.get("length_bucket"),
        "text_length": row.get("text_length"),
        "source_batch": row.get("source_batch"),
        "batch_id": batch_id,
        "schema_version": "semantic-schema-candidate-v0.3",
    }
    # ``title`` follows the existing feed convention: no separate title exists
    # in the selection source, so the canonical stock name is retained here.
    return {
        "sample_id": row["sample_id"],
        "text": row["text"],
        "title": row.get("stock_name") or "",
        "metadata": metadata,
        "heads": list(row["heads"]),
    }


def build_pilot_package(
    human_train: Mapping[str, Mapping[str, Any]],
    source_rows: Mapping[str, Mapping[str, Any]],
    *,
    batch_id: str = DEFAULT_BATCH_ID,
    target_samples: int = 20,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = select_pilot_samples(human_train, source_rows, target_samples=target_samples)
    tasks = [_task_row(row, batch_id=batch_id) for row in selected]
    factor_counts = Counter(head for row in tasks for head in row["heads"])
    truncated_ids = sorted(
        str(row["sample_id"])
        for row in source_rows.values()
        if _looks_truncated(str(row["sample_id"]), str(row["text"]), row.get("text_length"))
    )
    market_rows = [row for row in selected if "market_scope" in row["heads"]]
    market_parent_ok = all(
        human_train[str(row["sample_id"])].get("answers", {}).get("emotion_target") == "MARKET"
        for row in market_rows
    )
    summary = {
        "batch_id": batch_id,
        "schema_version": "semantic-schema-candidate-v0.3",
        "purpose": "sparse auxiliary annotation pilot; answer-free preparation only",
        "sample_count": len(tasks),
        "assignment_count": sum(len(row["heads"]) for row in tasks),
        "factor_task_counts": {factor: factor_counts[factor] for factor in AUXILIARY_FACTORS},
        "market_scope_parent_head": "emotion_target",
        "market_scope_parent_validation": market_parent_ok,
        "source_distribution": dict(sorted(Counter(row["metadata"]["source_batch"] for row in tasks).items())),
        "length_distribution": dict(sorted(Counter(row["metadata"]["length_bucket"] for row in tasks).items())),
        "distinct_exact_content_groups": len({row["metadata"]["exact_content_group"] for row in tasks}),
        "excluded_truncated_count": len(truncated_ids),
        "excluded_truncated_sample_ids": truncated_ids,
        "answers_included": False,
        "model_predictions_included": False,
        "human_train_source": str(DEFAULT_HUMAN_TRAIN),
        "allowed_source_batches": [str(path) for path in DEFAULT_SOURCE_TRAIN],
        "labeling_app_imported": False,
        "import_prerequisite": "Owner/app operator must review this answer-free JSONL and explicitly import it; no dispatch was performed by this command.",
    }
    return tasks, summary


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def write_package(output_dir: Path, tasks: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "auxiliary-pilot-tasks.jsonl", tasks)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def prepare(
    *,
    human_train_path: Path = DEFAULT_HUMAN_TRAIN,
    source_paths: Sequence[Path] = DEFAULT_SOURCE_TRAIN,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    batch_id: str = DEFAULT_BATCH_ID,
) -> dict[str, Any]:
    human_train = load_human_train(human_train_path)
    source_rows = load_source_rows(source_paths)
    tasks, summary = build_pilot_package(human_train, source_rows, batch_id=batch_id)
    write_package(output_dir, tasks, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-train", type=Path, default=DEFAULT_HUMAN_TRAIN)
    parser.add_argument("--source-train", type=Path, action="append", dest="source_paths")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-id", default=DEFAULT_BATCH_ID)
    args = parser.parse_args(argv)
    source_paths = tuple(args.source_paths) if args.source_paths else DEFAULT_SOURCE_TRAIN
    try:
        summary = prepare(
            human_train_path=args.human_train,
            source_paths=source_paths,
            output_dir=args.output_dir,
            batch_id=args.batch_id,
        )
    except PilotPreparationError as exc:
        parser.error(str(exc))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
