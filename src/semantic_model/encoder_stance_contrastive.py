"""One isolated Stance-only candidate for future-view vs past-regret separation.

The current SEM7 A5 runtime is read-only control.  This module trains only the
two Stance specialist blocks and its classifier, starting from A5's Stance
state.  Paired auxiliary examples are explicitly marked as synthetic support,
never as user annotation.  A text-only review file is mandatory and must be
frozen before this command can fit or read any A5/candidate predictions.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import random
import re
import sqlite3
import sys
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from . import encoder_action_contrastive as action
from . import encoder_m1 as m1
from . import encoder_m12_human_first_cv as m12
from . import encoder_m5_human100_source_balanced as m5
from . import encoder_m2_1_softlabel as m21
from . import encoder_m2_selected_runtime as selected
from . import encoder_semantic_active_runtime as active
from .errors import ContractError
from .schema import LabelSchema


HEAD = "stance"
STAGE_ID = "SEM7-STANCE-ONLY-PAST-REGRET-CONTRAST-V1"
PROJECT_ROOT = m12.PROJECT_ROOT
REPO_ROOT = m12.REPO_ROOT
DEFAULT_OUTPUT = PROJECT_ROOT / "model-artifacts" / "semantic-stance-only-contrastive-v1-20260929"
DEFAULT_COLLECTOR_DB = PROJECT_ROOT / "MyResearcher-DataCollector" / "data" / "collector.db"
PUBLIC_POOL = PROJECT_ROOT / "pipeline-runs" / "market-extreme-unenriched-20260928" / "source" / "m12-inference-input.jsonl"
PUBLIC_AUDIT = PROJECT_ROOT / "pipeline-runs" / "semantic-seven-head-audit-20260929"
RECENT_ROOT = PROJECT_ROOT / "pipeline-runs" / "recent-chain-20260915"
SEMANTIC_REPAIR_ROOT = PROJECT_ROOT / "pipeline-runs" / "semantic-repair-label-20260928"
MAX_EPOCHS = m12.MAX_EPOCHS
BATCH_SIZE = m12.BATCH_SIZE
BASE_SEED = m12.BASE_SEED_BY_HEAD[HEAD]
BLOCK_LR = m12.BLOCK_LR
HEAD_LR = m12.HEAD_LR
WEIGHT_DECAY = m12.WEIGHT_DECAY
GRADIENT_CLIP = m12.GRADIENT_CLIP
AUXILIARY_WEIGHT = 0.65
EVAL_SOURCE_CUTOFF = "2026-09-28T03:00:00+08:00"
CANNOT_DETERMINE = "无法判定"

# Each two-row pair changes whether a directional claim concerns the future or
# only a completed buy/sell timing regret.  It is a fixed support set, not a
# generated user label.  NEUTRAL is used only because each regret sentence
# explicitly withholds a present/future outlook.
PAIRED_AUXILIARY: tuple[tuple[str, str, str, str], ...] = (
    ("我判断这只股票后面还有上涨空间，接下来走势偏强。", "BULL", "之前卖掉后它继续上涨，我只是后悔卖早了，没有表达接下来方向。", "NEUTRAL"),
    ("对这只股票后市看多，预计后续价格会走高。", "BULL", "卖出以后股价才涨起来，现在只是在可惜过去卖得太早。", "NEUTRAL"),
    ("我觉得后面会反弹，未来价格有望向上。", "BULL", "早些时候卖出了，后来错过上涨，只是在懊恼那次过去的卖出决定。", "NEUTRAL"),
    ("我看空这只股票后面的走势，判断之后还会下跌。", "BEAR", "之前买入后价格下跌，我只是后悔买得太早，没有表达未来看法。", "NEUTRAL"),
    ("我认为后市偏空，股价接下来可能继续走弱。", "BEAR", "买入之后很快跌了，现在只是在后悔过去进场太早。", "NEUTRAL"),
    ("我判断这轮反弹结束了，后面价格还要往下走。", "BEAR", "早些时候买在较高位置，后来下跌，我只后悔当时买入时机。", "NEUTRAL"),
)

PROMOTION_RULE: Mapping[str, Any] = {
    "name": "text_review_directional_improvement_v1",
    "requires_reviewed_future_bull": 5,
    "requires_reviewed_future_bear": 5,
    "requires_reviewed_past_regret_only": 3,
    "candidate_target_errors_must_fall_by_at_least": 2,
    "components_must_not_worsen": ["BULL_to_BEAR", "BEAR_to_BULL", "past_regret_to_directional"],
    "clear_future_direction_accuracy_must_not_decrease": True,
    "market_outcomes_allowed": False,
}


def _require(ok: bool, code: str, message: str, **details: Any) -> None:
    if not ok:
        raise ContractError(code, message, **details)


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    _require(path.is_file(), "STANCE_INPUT_MISSING", "required JSONL is missing", path=str(path))
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _norm(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"[\W_]+", "", value, flags=re.UNICODE)


class _PlainHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style"}:
            self.skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style"} and self.skip:
            self.skip -= 1

    def handle_data(self, data: str) -> None:
        if not self.skip and data.strip():
            self.parts.append(data)


def _plain_text(value: str | None) -> str:
    if not value:
        return ""
    parser = _PlainHTML()
    parser.feed(value)
    return re.sub(r"\s+", " ", html.unescape(" ".join(parser.parts))).strip()


def _training_records(schema: LabelSchema) -> tuple[list[m1.M1Record], dict[str, Any]]:
    # This is the same accepted 147-sample Human Train population already used
    # by the current research line.  Existing answers are copied verbatim.
    records, audit = action.prior._legacy_train()
    assignments = action.prior._read_jsonl(SEMANTIC_REPAIR_ROOT / "train-assignments.jsonl")
    finals = action.prior._read_jsonl(SEMANTIC_REPAIR_ROOT / "train-final.jsonl")
    by_id = {str(row.get("sample_id")): row for row in assignments}
    final_by_id = {str(row.get("sample_id")): row for row in finals}
    _require(len(by_id) == len(assignments) and len(final_by_id) == len(finals), "STANCE_SPARSE_DUPLICATE", "sparse Stance train rows contain duplicate sample IDs")
    stance_ids = {sid for sid, row in by_id.items() if row.get("heads") == [HEAD]}
    _require(stance_ids <= set(final_by_id), "STANCE_SPARSE_NOT_FINAL", "a Stance assignment has no final answer", missing=sorted(stance_ids - set(final_by_id)))
    additions: list[m1.M1Record] = []
    for sid in sorted(stance_ids):
        task, result = by_id[sid], final_by_id[sid]
        _require(result.get("head") == HEAD and result.get("is_final") is True and result.get("disposition") is None, "STANCE_SPARSE_INVALID_FINAL", "Stance train assignment is not a usable final answer", sample_id=sid)
        _require(result.get("schema_version") == "semantic-schema-calibrated-v0.2.1", "STANCE_SPARSE_SCHEMA", "sparse Stance labels use a different schema", sample_id=sid)
        answer = result.get("answer")
        _require(answer in schema.class_order[HEAD], "STANCE_SPARSE_LABEL", "sparse Stance label is outside taxonomy", sample_id=sid, answer=answer)
        metadata = task.get("metadata", {})
        _require(isinstance(task.get("text"), str) and task["text"].strip(), "STANCE_SPARSE_TEXT", "sparse Stance row lacks its exact source text", sample_id=sid)
        additions.append(m1.M1Record(sid, str(metadata.get("stock_code", "")), str(metadata.get("stock_name", "")), str(task["text"]), {HEAD: answer}, {HEAD: 1.0}))
    _require(len(additions) == 3, "STANCE_SPARSE_COUNT", "expected the three existing sparse Stance Train answers", observed=len(additions))
    human_ids = {row.sample_id for row in records}
    human_text = {_norm(row.model_text) for row in records}
    _require(not human_ids.intersection(row.sample_id for row in additions), "STANCE_TRAIN_ID_OVERLAP", "sparse Stance rows overlap existing Human Train IDs")
    _require(not human_text.intersection(_norm(row.model_text) for row in additions), "STANCE_TRAIN_TEXT_OVERLAP", "sparse Stance text duplicates existing Human Train text")
    combined = [*records, *additions]
    _require(len(combined) == 150 and len({row.sample_id for row in combined}) == 150, "STANCE_TRAIN_COUNT", "expected 150 distinct existing human training records", observed=len(combined))
    _require(all(row.label.get(HEAD) in schema.class_order[HEAD] for row in combined), "STANCE_TRAIN_LABEL", "a preserved Human Train Stance answer is outside the frozen taxonomy")
    return combined, {"human_train147": audit, "sparse_stance_train": len(additions), "sparse_stance_sample_ids": [row.sample_id for row in additions], "preserved_stance_distribution": dict(Counter(str(row.label[HEAD]) for row in combined))}


def _legacy_checks(schema: LabelSchema) -> dict[str, list[m1.M1Record]]:
    check20, _ = m5.load_human_check_pool(REPO_ROOT / "runs" / "feedback-loop-v3-finalized-20260910")
    check19, _ = m12.load_recent_feedback_records(
        RECENT_ROOT / "human" / "recent-check19-valid-final.jsonl",
        RECENT_ROOT / "feed" / "feed-input.jsonl",
        role="historical_recent_check",
        expected_samples=19,
    )
    check8 = _sparse_stance_records("check", schema)
    return {"human_check20": check20, "recent_check19": check19, "semantic_repair_stance_check": check8}


def _sparse_stance_records(role: str, schema: LabelSchema) -> list[m1.M1Record]:
    assignments = action.prior._read_jsonl(SEMANTIC_REPAIR_ROOT / f"{role}-assignments.jsonl")
    finals = action.prior._read_jsonl(SEMANTIC_REPAIR_ROOT / f"{role}-final.jsonl")
    indexed = {str(row.get("sample_id")): row for row in assignments}
    final = {str(row.get("sample_id")): row for row in finals}
    selected_ids = {sid for sid, row in indexed.items() if row.get("heads") == [HEAD]}
    result: list[m1.M1Record] = []
    for sid in sorted(selected_ids):
        _require(sid in final, "STANCE_CHECK_NOT_FINAL", "a Stance Check task is not final", sample_id=sid)
        task, answer_row = indexed[sid], final[sid]
        answer = answer_row.get("answer")
        _require(answer_row.get("head") == HEAD and answer_row.get("is_final") is True and answer_row.get("disposition") is None and answer_row.get("schema_version") == "semantic-schema-calibrated-v0.2.1", "STANCE_CHECK_INVALID", "Stance Check row is not a usable final answer", sample_id=sid)
        _require(answer in schema.class_order[HEAD], "STANCE_CHECK_LABEL", "Stance Check answer is outside taxonomy", sample_id=sid, answer=answer)
        metadata = task.get("metadata", {})
        result.append(m1.M1Record(sid, str(metadata.get("stock_code", "")), str(metadata.get("stock_name", "")), str(task["text"]), {HEAD: answer}, {HEAD: 1.0}))
    return result


def _public_exposure(schema: LabelSchema) -> tuple[set[str], set[str], dict[str, Any]]:
    ids: set[str] = set()
    texts: set[str] = set()
    evidence: dict[str, Any] = {"frozen_pre_a5_source": str(PUBLIC_POOL), "prior_review_audit": str(PUBLIC_AUDIT)}
    if PUBLIC_POOL.is_file():
        for row in _read_jsonl(PUBLIC_POOL):
            if row.get("sample_id"):
                ids.add(str(row["sample_id"]))
            if isinstance(row.get("model_text"), str):
                texts.add(_norm(str(row["model_text"])))
    if PUBLIC_AUDIT.is_dir():
        for path in PUBLIC_AUDIT.glob("*.jsonl"):
            for row in _read_jsonl(path):
                sid = row.get("sample_id")
                if sid:
                    ids.add(str(sid))
                text = row.get("model_text", row.get("text"))
                if isinstance(text, str):
                    texts.add(_norm(text))
    # The historical A5 blind review's 56 rows were drawn from PUBLIC_POOL;
    # excluding that entire source is intentionally stronger than relying on
    # its incomplete published sample-ID list.
    evidence["excluded_historical_source_rows"] = "all rows/texts from the pre-A5 frozen input pool; includes published 56-post A5 blind set"
    return ids, texts, evidence


def _bucket(text: str) -> str | None:
    value = text.lower()
    if re.search(r"卖飞|卖早|买早|买错|卖错|后悔.{0,8}(卖|买)|早知道.{0,8}(买|卖)|卖掉就涨|卖了就涨|买了就跌|踏空|割肉走了", value):
        return "past_regret_only"
    if re.search(r"看好|看多|看涨|后市偏多|后市.{0,6}看好|未来.{0,10}(上涨|走强|看多)|明天.{0,8}(涨|反弹)|后面.{0,8}(涨|走强)|接下来.{0,8}(涨|走强)|会涨停|要涨停|涨停.{0,8}(明天|后面|将)|后面会涨|后市看涨", value):
        return "future_bull"
    if re.search(r"看空|看跌|后市偏空|未来.{0,10}(下跌|走弱|看空)|明天.{0,8}(跌|继续下探)|后面.{0,8}(下跌|走弱)|接下来.{0,8}(下跌|走弱)|继续跌|还要跌|跌到退市|持续下跌|下跌趋势|阴跌到", value):
        return "future_bear"
    return None


def _source_rows(db_path: Path) -> list[dict[str, Any]]:
    _require(db_path.is_file(), "STANCE_COLLECTOR_DB_MISSING", "Collector database is missing", path=str(db_path))
    connection = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    try:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT source, source_item_id, stock_code, title, content, published_at "
            "FROM posts WHERE published_at >= ? AND published_at < ? "
            "ORDER BY published_at, source, source_item_id",
            (EVAL_SOURCE_CUTOFF, "2026-09-29T00:00:00+08:00"),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def prepare_evaluation_texts(*, output_dir: str | Path = DEFAULT_OUTPUT, db_path: str | Path = DEFAULT_COLLECTOR_DB) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    _require(not output.exists(), "STANCE_EVAL_OUTPUT_EXISTS", "evaluation batch is immutable and output directory already exists", path=str(output))
    project = m1.ProjectConfig.load(m12.DEFAULT_CONFIG)
    schema, _weak, _dev = m1.load_m1_partitions(project)
    training, training_audit = _training_records(schema)
    checks = _legacy_checks(schema)
    known_ids = {row.sample_id for row in training}
    known_texts = {_norm(row.model_text) for row in training}
    for set_name, records in checks.items():
        known_ids.update(row.sample_id for row in records)
        known_texts.update(_norm(row.model_text) for row in records)
    exposed_ids, exposed_texts, exposure_audit = _public_exposure(schema)
    known_ids |= exposed_ids
    known_texts |= exposed_texts
    source_path = Path(db_path).resolve()
    source_digest = _sha256(source_path)
    raw = _source_rows(source_path)
    dedup: dict[str, dict[str, Any]] = {}
    exclusions = Counter()
    for row in raw:
        text = _plain_text(str(row.get("content") or "")) or _plain_text(str(row.get("title") or ""))
        norm = _norm(text)
        if not text or len(norm) < 6:
            exclusions["blank_or_low_information"] += 1
            continue
        if str(row.get("source_item_id")) in known_ids:
            exclusions["known_sample_id"] += 1
            continue
        if norm in known_texts:
            exclusions["known_training_or_public_text"] += 1
            continue
        bucket = _bucket(text)
        if bucket is None:
            exclusions["not_in_predeclared_text_strata"] += 1
            continue
        group_key = norm
        if group_key in dedup:
            exclusions["duplicate_exact_content"] += 1
            continue
        sid = "stanceeval-" + hashlib.sha256(f"{row['source']}\n{row['source_item_id']}".encode()).hexdigest()
        dedup[group_key] = {
            "sample_id": sid,
            "stock_code": str(row.get("stock_code", "")),
            "published_at": str(row.get("published_at", "")),
            "text": text,
            "selection_stratum": bucket,
            "source": str(row.get("source", "")),
            "source_item_id": str(row.get("source_item_id", "")),
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
    quotas = {"future_bull": 10, "future_bear": 10, "past_regret_only": 8}
    selected_rows: list[dict[str, Any]] = []
    for bucket, quota in quotas.items():
        candidates = [row for row in dedup.values() if row["selection_stratum"] == bucket]
        candidates.sort(key=lambda row: hashlib.sha256(f"stance-eval-20260929-v1|{bucket}|{row['sample_id']}".encode()).hexdigest())
        # Spread each topical stratum across different stocks; fill remaining
        # slots only after one pass through the per-stock cap.
        chosen: list[dict[str, Any]] = []
        used_per_stock: Counter[str] = Counter()
        for row in candidates:
            if used_per_stock[row["stock_code"]] >= 3:
                continue
            chosen.append(row)
            used_per_stock[row["stock_code"]] += 1
            if len(chosen) == quota:
                break
        selected_rows.extend(chosen)
    _require(len(selected_rows) >= 12, "STANCE_EVAL_POOL_TOO_SMALL", "fewer than twelve eligible fresh posts remained after exclusions", observed=len(selected_rows), available={key: sum(row["selection_stratum"] == key for row in dedup.values()) for key in quotas})
    _require(len({row["sample_id"] for row in selected_rows}) == len(selected_rows), "STANCE_EVAL_DUPLICATE_ID", "blind sample IDs are not unique")
    input_path = output / "evaluation-text-only.jsonl"
    lock_path = output / "selection-lock.json"
    _write_jsonl(input_path, selected_rows)
    manifest = {
        "status": "TEXT_SET_FROZEN_REVIEW_PENDING",
        "selection_timestamp": datetime.now(timezone.utc).isoformat(),
        "raw_source": str(source_path),
        "raw_source_sha256": source_digest,
        "public_pool_source_cutoff": "2026-09-28T02:53:20+08:00",
        "minimum_published_at": EVAL_SOURCE_CUTOFF,
        "selection_scope": "Collector posts published after the frozen pre-A5 pool snapshot; predictions not read",
        "predeclared_lexical_strata_and_quotas": quotas,
        "selected_by_stratum": dict(Counter(row["selection_stratum"] for row in selected_rows)),
        "selected_count": len(selected_rows),
        "selected_sample_ids": [row["sample_id"] for row in selected_rows],
        "training_records": {"count": len(training), "stance_distribution": dict(Counter(str(row.label[HEAD]) for row in training)), "audit": training_audit},
        "prior_exposure_exclusion": exposure_audit,
        "excluded_counts": dict(exclusions),
        "text_only_path": str(input_path),
        "text_only_sha256": _sha256(input_path),
        "no_model_predictions_or_outcomes_read": True,
    }
    _dump(lock_path, manifest)
    output.mkdir(parents=True, exist_ok=True)
    return manifest


def _auxiliary_records(schema: LabelSchema, human: Sequence[m1.M1Record]) -> list[m1.M1Record]:
    contexts = sorted({(row.stock_code, row.stock_name) for row in human if row.stock_code and row.stock_name})
    _require(len(contexts) >= 2, "STANCE_AUX_CONTEXTS", "not enough existing stock contexts for paired auxiliary data")
    human_text = {_norm(row.model_text) for row in human}
    result: list[m1.M1Record] = []
    seen: set[str] = set()
    for pair_index, (future_text, future_label, regret_text, regret_label) in enumerate(PAIRED_AUXILIARY):
        for side, text, label in (("future", future_text, future_label), ("past_regret", regret_text, regret_label)):
            normalized = _norm(text)
            _require(normalized not in human_text and normalized not in seen, "STANCE_AUX_TEXT_OVERLAP", "paired auxiliary example overlaps Human Train or another auxiliary text", pair=pair_index, side=side)
            _require(label in schema.class_order[HEAD], "STANCE_AUX_LABEL_INVALID", "paired auxiliary label is outside taxonomy", label=label)
            stock_code, stock_name = contexts[(pair_index + (0 if side == "future" else 1)) % len(contexts)]
            sample_id = "stance-pair-" + hashlib.sha256(f"{pair_index}|{side}|{normalized}".encode()).hexdigest()
            result.append(m1.M1Record(sample_id, stock_code, stock_name, text, {HEAD: label}, {HEAD: AUXILIARY_WEIGHT}))
            seen.add(normalized)
    return result


def _batch(torch: Any, tokenizer: Any, rows: Sequence[m1.M1Record], config: Mapping[str, Any], device: Any) -> dict[str, Any]:
    encoded = [m1.build_input_ids(tokenizer, row, config) for row in rows]
    pad = tokenizer.pad_token_id if isinstance(tokenizer.pad_token_id, int) else 0
    width = max(map(len, encoded))
    order = list(m12_label_order(config, HEAD))
    return {
        "input_ids": torch.tensor([values + [pad] * (width - len(values)) for values in encoded], dtype=torch.long, device=device),
        "attention_mask": torch.tensor([[1] * len(values) + [0] * (width - len(values)) for values in encoded], dtype=torch.long, device=device),
        "labels": torch.tensor([order.index(str(row.label[HEAD])) for row in rows], dtype=torch.long, device=device),
    }


def m12_label_order(config: Mapping[str, Any], head: str) -> Sequence[str]:
    order = config.get("class_order", {}).get(head)
    if not isinstance(order, list):
        raise ContractError("STANCE_CLASS_ORDER_MISSING", "frozen class order is not in runtime config", head=head)
    return order


def _build_model(torch: Any, AutoModel: Any, snapshot: Path, schema: LabelSchema, config: Mapping[str, Any], base_runtime: Any, device: Any) -> Any:
    model = m12.make_model(torch, AutoModel, snapshot, schema, float(config["head_dropout"]))
    model.specialist_blocks[HEAD].load_state_dict(base_runtime.m12_model.specialist_blocks[HEAD].state_dict(), strict=True)
    model.heads[HEAD].load_state_dict(base_runtime.m12_model.heads[HEAD].state_dict(), strict=True)
    model.set_active_head(HEAD)
    expected_trainable = {
        name for name, _parameter in model.named_parameters()
        if name.startswith(f"specialist_blocks.{HEAD}.") or name.startswith(f"heads.{HEAD}.")
    }
    actual_trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    _require(actual_trainable == expected_trainable and expected_trainable, "STANCE_TRAINABLE_SCOPE", "only the two Stance specialist blocks and Stance output layer may train")
    return model.to(device)


def _probabilities(torch: Any, model: Any, tokenizer: Any, records: Sequence[m1.M1Record], config: Mapping[str, Any], device: Any) -> np.ndarray:
    model.eval()
    result: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(records), BATCH_SIZE):
            batch = _batch(torch, tokenizer, records[start : start + BATCH_SIZE], config, device)
            logits = model(batch["input_ids"], batch["attention_mask"])[HEAD]
            _require(bool(torch.isfinite(logits).all().item()), "STANCE_NONFINITE_LOGITS", "Stance logits contain non-finite values")
            result.append(torch.softmax(logits, dim=-1).detach().cpu().numpy())
    return np.concatenate(result, axis=0) if result else np.empty((0, len(m12_label_order(config, HEAD))), dtype=np.float32)


def _fit_curve(*, torch: Any, AutoModel: Any, tokenizer: Any, snapshot: Path, schema: LabelSchema, config: Mapping[str, Any], base_runtime: Any, human_train: Sequence[m1.M1Record], auxiliary: Sequence[m1.M1Record], validation: Sequence[m1.M1Record], seed: int, device: Any, started: float) -> list[np.ndarray]:
    m12._set_seed(torch, seed)
    model = _build_model(torch, AutoModel, snapshot, schema, config, base_runtime, device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    blocks = list(model.specialist_blocks[HEAD].parameters())
    head_params = list(model.heads[HEAD].parameters())
    optimizer = torch.optim.AdamW([{"params": blocks, "lr": BLOCK_LR}, {"params": head_params, "lr": HEAD_LR}], weight_decay=WEIGHT_DECAY)
    balance = m12._balance(torch, human_train, schema, HEAD, device)
    curve: list[np.ndarray] = []
    for epoch in range(1, MAX_EPOCHS + 1):
        _require(time.monotonic() - started < m12.WALL_TIME_SECONDS, "STANCE_TRAIN_TIME_LIMIT", "Stance-only fit exceeded the M12 two-hour bound")
        ordered = [*human_train, *auxiliary]
        random.Random(seed * 10000 + epoch).shuffle(ordered)
        model.train(True)
        for offset in range(0, len(ordered), BATCH_SIZE):
            rows = ordered[offset : offset + BATCH_SIZE]
            batch = _batch(torch, tokenizer, rows, config, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch["attention_mask"])[HEAD]
            _require(bool(torch.isfinite(logits).all().item()), "STANCE_NONFINITE_LOGITS", "Stance training logits contain non-finite values")
            raw = torch.nn.functional.cross_entropy(logits, batch["labels"], weight=balance, reduction="none")
            sample_weight = torch.tensor([float(row.weights.get(HEAD, 1.0)) for row in rows], dtype=torch.float32, device=device)
            loss = (raw * sample_weight).sum() / sample_weight.sum().clamp_min(1e-12)
            _require(bool(torch.isfinite(loss).item()), "STANCE_NONFINITE_LOSS", "Stance training loss is non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, GRADIENT_CLIP)
            optimizer.step()
        if validation:
            curve.append(_probabilities(torch, model, tokenizer, validation, config, device))
    del model
    return curve


def _fit_final(*, torch: Any, AutoModel: Any, tokenizer: Any, snapshot: Path, schema: LabelSchema, config: Mapping[str, Any], base_runtime: Any, human_train: Sequence[m1.M1Record], auxiliary: Sequence[m1.M1Record], epochs: int, seed: int, device: Any, started: float) -> tuple[Any, dict[str, Any]]:
    m12._set_seed(torch, seed)
    model = _build_model(torch, AutoModel, snapshot, schema, config, base_runtime, device)
    blocks = list(model.specialist_blocks[HEAD].parameters())
    head_params = list(model.heads[HEAD].parameters())
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW([{"params": blocks, "lr": BLOCK_LR}, {"params": head_params, "lr": HEAD_LR}], weight_decay=WEIGHT_DECAY)
    balance = m12._balance(torch, human_train, schema, HEAD, device)
    losses: list[float] = []
    updates = 0
    for epoch in range(1, epochs + 1):
        _require(time.monotonic() - started < m12.WALL_TIME_SECONDS, "STANCE_TRAIN_TIME_LIMIT", "Stance-only fit exceeded the M12 two-hour bound")
        ordered = [*human_train, *auxiliary]
        random.Random(seed * 10000 + epoch).shuffle(ordered)
        model.train(True)
        for offset in range(0, len(ordered), BATCH_SIZE):
            rows = ordered[offset : offset + BATCH_SIZE]
            batch = _batch(torch, tokenizer, rows, config, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch["attention_mask"])[HEAD]
            raw = torch.nn.functional.cross_entropy(logits, batch["labels"], weight=balance, reduction="none")
            sample_weight = torch.tensor([float(row.weights.get(HEAD, 1.0)) for row in rows], dtype=torch.float32, device=device)
            loss = (raw * sample_weight).sum() / sample_weight.sum().clamp_min(1e-12)
            _require(bool(torch.isfinite(loss).item()) and bool(torch.isfinite(logits).all().item()), "STANCE_NONFINITE", "Stance loss or logits are non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, GRADIENT_CLIP)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            updates += 1
    return model, {"epochs": epochs, "optimizer_updates": updates, "mean_batch_loss": float(np.mean(losses)), "sample_count": len(human_train), "paired_auxiliary_count": len(auxiliary)}


def _load_review(path: Path, locked_rows: Sequence[Mapping[str, Any]], schema: LabelSchema) -> tuple[list[dict[str, Any]], str]:
    rows = _read_jsonl(path)
    wanted = {str(row["sample_id"]): row for row in locked_rows}
    _require(len(wanted) == len(locked_rows), "STANCE_REVIEW_LOCK_DUPLICATE", "frozen text sample IDs are duplicated")
    found = {str(row.get("sample_id")): row for row in rows}
    _require(len(found) == len(rows) and set(found) == set(wanted), "STANCE_REVIEW_COVERAGE", "text-only review must contain exactly one row per frozen sample", expected=len(wanted), observed=len(rows))
    allowed = set(schema.class_order[HEAD]) | {CANNOT_DETERMINE}
    for sid, row in found.items():
        _require(row.get("text_sha256") == wanted[sid].get("text_sha256"), "STANCE_REVIEW_TEXT_MISMATCH", "review row is not bound to the frozen raw text", sample_id=sid)
        label = row.get("review_label")
        _require(label in allowed, "STANCE_REVIEW_LABEL_INVALID", "review label must be a Stance class or 无法判定", sample_id=sid, label=label)
        _require(isinstance(row.get("evidence"), str) and row["evidence"].strip(), "STANCE_REVIEW_EVIDENCE_MISSING", "each review row requires text-only evidence or a reason for inability to decide", sample_id=sid)
        _require(row.get("reviewer_mode") == "text_only_before_predictions", "STANCE_REVIEW_ORDER_INVALID", "review must record text-only-before-predictions mode", sample_id=sid)
    order = [str(row["sample_id"]) for row in locked_rows]
    return [found[sid] for sid in order], _sha256(path)


def _eval_predictions(probabilities: np.ndarray, labels: Sequence[str], schema: LabelSchema) -> dict[str, Any]:
    order = list(schema.class_order[HEAD])
    predicted = [order[int(index)] for index in np.asarray(probabilities).argmax(axis=1)]
    truth = list(labels)
    precision, recall, f1, support = precision_recall_fscore_support(truth, predicted, labels=order, zero_division=0)
    return {
        "count": len(truth),
        "accuracy": float(np.mean(np.asarray(truth) == np.asarray(predicted))) if truth else None,
        "macro_f1": float(np.mean(f1)) if truth else None,
        "supported_macro_f1": float(np.mean([f1[index] for index, count in enumerate(support) if count > 0])) if any(support) else None,
        "confusion_matrix": confusion_matrix(truth, predicted, labels=order).tolist(),
        "class_order": order,
        "per_class": {label: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])} for i, label in enumerate(order)},
        "predicted_distribution": dict(Counter(predicted)),
    }


def _directional_counts(review: Sequence[Mapping[str, Any]], predicted: Sequence[str]) -> dict[str, Any]:
    counts = Counter()
    future_total = future_correct = 0
    for row, guess in zip(review, predicted):
        label = str(row["review_label"])
        group = str(row.get("probe_type", row.get("selection_stratum", "")))
        if label in ("BULL", "BEAR") and group in ("future_bull", "future_bear"):
            future_total += 1
            future_correct += int(guess == label)
        if group == "future_bull" and label == "BULL" and guess == "BEAR":
            counts["BULL_to_BEAR"] += 1
        if group == "future_bear" and label == "BEAR" and guess == "BULL":
            counts["BEAR_to_BULL"] += 1
        if group == "past_regret_only" and label == "NEUTRAL" and guess in ("BULL", "BEAR"):
            counts["past_regret_to_directional"] += 1
    return {
        "BULL_to_BEAR": int(counts["BULL_to_BEAR"]),
        "BEAR_to_BULL": int(counts["BEAR_to_BULL"]),
        "past_regret_to_directional": int(counts["past_regret_to_directional"]),
        "target_error_total": int(sum(counts.values())),
        "future_direction_accuracy": (future_correct / future_total) if future_total else None,
        "future_direction_count": future_total,
    }


def _merge_stance_rows(base_rows: Sequence[Mapping[str, Any]], probabilities: np.ndarray, schema: LabelSchema, model_version: str) -> list[dict[str, Any]]:
    order = list(schema.class_order[HEAD])
    _require(len(base_rows) == len(probabilities), "STANCE_MERGE_COUNT", "candidate stance rows differ from A5 row count")
    merged: list[dict[str, Any]] = []
    for index, row in enumerate(base_rows):
        source_predictions = row.get("predictions")
        _require(isinstance(source_predictions, Mapping) and set(source_predictions) == set(selected.HEADS), "STANCE_MERGE_HEAD_SET", "A5 row does not satisfy the strict seven-head contract", sample_id=row.get("sample_id"))
        vector = np.asarray(probabilities[index], dtype=float)
        _require(vector.shape == (len(order),) and np.isfinite(vector).all() and np.isclose(vector.sum(), 1.0, atol=1e-6), "STANCE_MERGE_PROBABILITY", "candidate Stance probabilities are invalid", sample_id=row.get("sample_id"))
        updated = copy.deepcopy(dict(row))
        updated["predictions"][HEAD] = {"prediction": order[int(vector.argmax())], "probabilities": {label: float(vector[i]) for i, label in enumerate(order)}}
        updated["model_version"] = model_version
        # The raw seven-head map changes at Stance only; all other output
        # payloads including conditioned_outputs remain a deep-equal copy.
        for untouched in set(selected.HEADS) - {HEAD}:
            _require(updated["predictions"][untouched] == source_predictions[untouched], "STANCE_UNTRAINED_HEAD_CHANGED", "an untrained head changed during candidate merge", head=untouched, sample_id=row.get("sample_id"))
        return_row = updated
        merged.append(return_row)
    return merged


def _candidate_probabilities(torch: Any, model: Any, tokenizer: Any, records: Sequence[selected.InferenceRecord], config: Mapping[str, Any], device: Any) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(records), BATCH_SIZE):
            current = records[start : start + BATCH_SIZE]
            batch = selected.build_inference_batch(torch, tokenizer, current, config, device)
            logits = model(batch["input_ids"], batch["attention_mask"])[HEAD]
            _require(bool(torch.isfinite(logits).all().item()), "STANCE_CANDIDATE_NONFINITE", "candidate Stance logits are non-finite")
            chunks.append(torch.softmax(logits, dim=-1).detach().cpu().numpy())
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, len(m12_label_order(config, HEAD))), dtype=np.float32)


def _as_inference(records: Sequence[m1.M1Record]) -> list[selected.InferenceRecord]:
    return [selected.InferenceRecord(row.sample_id, row.stock_code, row.stock_name, row.model_text) for row in records]


def _labelled_eval_records(review_rows: Sequence[Mapping[str, Any]], schema: LabelSchema) -> list[m1.M1Record]:
    result = []
    for row in review_rows:
        label = row["review_label"]
        if label == CANNOT_DETERMINE:
            continue
        result.append(m1.M1Record(str(row["sample_id"]), str(row.get("stock_code", "")), str(row.get("stock_name", "")), str(row["text"]), {HEAD: str(label)}, {HEAD: 1.0}))
    return result


def _load_saved_blind_evaluation(output: Path, review: Sequence[Mapping[str, Any]], records: Sequence[m1.M1Record], schema: LabelSchema) -> tuple[dict[str, Any] | None, list[dict[str, Any]]] | None:
    evaluation_path = output / "evaluation.json"
    predictions_path = output / "blind-evaluation-predictions.jsonl"
    if not evaluation_path.exists() and not predictions_path.exists():
        return None
    _require(predictions_path.is_file(), "STANCE_BLIND_EVALUATION_PARTIAL", "evaluation summary exists without its frozen paired predictions; refusing to rerun prediction")
    rows = _read_jsonl(predictions_path)
    expected_ids = [row.sample_id for row in records]
    _require([str(row.get("sample_id")) for row in rows] == expected_ids, "STANCE_BLIND_EVALUATION_ORDER", "saved blind predictions do not match the frozen reviewed sample order")
    reviewed = {str(row["sample_id"]): row for row in review}
    order = list(schema.class_order[HEAD])
    for row in rows:
        sid = str(row["sample_id"])
        source = reviewed[sid]
        _require(row.get("text") == source.get("text") and row.get("review_label") == source.get("review_label") and row.get("evidence") == source.get("evidence"), "STANCE_BLIND_EVALUATION_REVIEW_BINDING", "saved blind prediction row no longer matches its text-only review", sample_id=sid)
        for prefix in ("a5", "candidate"):
            _require(row.get(f"{prefix}_prediction") in order, "STANCE_BLIND_EVALUATION_LABEL", "saved blind prediction has a label outside the frozen Stance class order", sample_id=sid, source=prefix)
            probabilities = row.get(f"{prefix}_probabilities")
            _require(isinstance(probabilities, Mapping) and set(probabilities) == set(order), "STANCE_BLIND_EVALUATION_PROBABILITIES", "saved blind prediction has an invalid probability map", sample_id=sid, source=prefix)
            vector = np.asarray([float(probabilities[label]) for label in order], dtype=float)
            _require(np.isfinite(vector).all() and np.isclose(vector.sum(), 1.0, atol=1e-6), "STANCE_BLIND_EVALUATION_PROBABILITIES", "saved blind probabilities are non-finite or unnormalized", sample_id=sid, source=prefix)
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8")) if evaluation_path.is_file() else None
    if evaluation is not None:
        _require(evaluation.get("blind_review_was_frozen_before_training_and_predictions") is True and evaluation.get("market_outcomes_read") is False, "STANCE_BLIND_EVALUATION_STATUS", "saved evaluation lacks the required blind-review/no-market-outcomes declaration")
    return evaluation, rows


def _check_metrics(runtime: Any, candidate_model: Any, records: Sequence[m1.M1Record], schema: LabelSchema, device: Any) -> dict[str, Any]:
    if not records:
        return {"count": 0, "status": "NO_STANCE_CHECK_ROWS"}
    baseline_rows = runtime.predict_records(_as_inference(records))
    probabilities = _candidate_probabilities(runtime.torch, candidate_model, runtime.legacy_runtime.tokenizer, _as_inference(records), runtime.config, device)
    candidate_rows = _merge_stance_rows(baseline_rows, probabilities, schema, STAGE_ID)
    labels = [str(row.label[HEAD]) for row in records]
    order = list(schema.class_order[HEAD])
    before = np.asarray([[float(row["predictions"][HEAD]["probabilities"][label]) for label in order] for row in baseline_rows])
    after = probabilities
    before_metric, after_metric = _eval_predictions(before, labels, schema), _eval_predictions(after, labels, schema)
    baseline_labels = [row["predictions"][HEAD]["prediction"] for row in baseline_rows]
    candidate_labels = [row["predictions"][HEAD]["prediction"] for row in candidate_rows]
    return {
        "count": len(records),
        "a5": before_metric,
        "candidate": after_metric,
        "prediction_changes": sum(before != after for before, after in zip(baseline_labels, candidate_labels)),
        "rows": [
            {"sample_id": row.sample_id, "review_label": row.label[HEAD], "a5": before, "candidate": after}
            for row, before, after in zip(records, baseline_labels, candidate_labels)
        ],
        "untrained_head_invariance": "PASS",
    }


def _public_distribution(runtime: Any, candidate_model: Any, input_path: Path, schema: LabelSchema, device: Any, output_dir: Path) -> dict[str, Any]:
    records = selected.load_unlabelled_jsonl(input_path)
    a5_counts: Counter[str] = Counter()
    candidate_counts: Counter[str] = Counter()
    changed = 0
    checked = 0
    equality: Counter[str] = Counter()
    example_records: list[dict[str, Any]] = []
    for start in range(0, len(records), 128):
        chunk = records[start : start + 128]
        a5_rows = runtime.predict_records(chunk)
        candidate_probs = _candidate_probabilities(runtime.torch, candidate_model, runtime.legacy_runtime.tokenizer, chunk, runtime.config, device)
        candidate_rows = _merge_stance_rows(a5_rows, candidate_probs, schema, STAGE_ID)
        for base, cand in zip(a5_rows, candidate_rows):
            before = str(base["predictions"][HEAD]["prediction"])
            after = str(cand["predictions"][HEAD]["prediction"])
            a5_counts[before] += 1
            candidate_counts[after] += 1
            changed += before != after
            checked += 1
            for head in selected.HEADS:
                equality[head] += base["predictions"][head] == cand["predictions"][head]
            equality["conditioned_outputs"] += base.get("conditioned_outputs") == cand.get("conditioned_outputs")
            if len(example_records) < 100 and before != after:
                example_records.append({"sample_id": base["sample_id"], "a5": before, "candidate": after, "stance_probabilities": cand["predictions"][HEAD]["probabilities"]})
    expected_untrained = len(records) * (len(selected.HEADS) - 1)
    _require(checked == len(records) and all(equality[head] == len(records) for head in set(selected.HEADS) - {HEAD}), "STANCE_FULL_POOL_INVARIANCE", "one or more untrained head predictions changed")
    _write_jsonl(output_dir / "full-pool-stance-changes-sample.jsonl", example_records)
    return {
        "source": str(input_path),
        "sample_count": len(records),
        "published_pool_rows": checked,
        "a5_stance_distribution": dict(a5_counts),
        "candidate_stance_distribution": dict(candidate_counts),
        "stance_argmax_change_count": changed,
        "stance_argmax_change_rate": changed / checked if checked else 0.0,
        "per_head_equal_rows": dict(equality),
        "untouched_head_row_checks_expected": expected_untrained,
        "all_untrained_head_rows_identical": all(equality[head] == len(records) for head in set(selected.HEADS) - {HEAD}),
        "no_market_return_or_price_data_read": True,
    }


def run(*, output_dir: str | Path = DEFAULT_OUTPUT, public_pool: str | Path = PUBLIC_POOL, device_name: str | None = None) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    text_path = output / "evaluation-text-only.jsonl"
    lock_path = output / "selection-lock.json"
    review_path = output / "evaluation-text-review.jsonl"
    _require(output.is_dir() and text_path.is_file() and lock_path.is_file(), "STANCE_EVAL_SET_NOT_FROZEN", "run prepare-eval first; frozen text-only set is missing", output=str(output))
    _require(review_path.is_file(), "STANCE_REVIEW_REQUIRED", "text-only reviewer labels must be written before model prediction or fit", path=str(review_path))
    checkpoint = output / "stance-checkpoint.pt"
    resume_postfit = checkpoint.exists()
    if resume_postfit:
        _require(not (output / "evaluation.json").exists() or (output / "blind-evaluation-predictions.jsonl").is_file(), "STANCE_POSTFIT_PARTIAL_EVALUATION", "evaluation summary exists without its paired predictions; refusing to duplicate inference")
        _require(not (output / "summary.json").exists(), "STANCE_POSTFIT_ALREADY_FINALIZED", "completed artifact already has a final summary; refusing to rerun")
    started = time.monotonic()
    project = m1.ProjectConfig.load(m12.DEFAULT_CONFIG)
    schema, _weak, _dev = m1.load_m1_partitions(project)
    locked_text = _read_jsonl(text_path)
    review, review_sha = _load_review(review_path, locked_text, schema)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    _require(lock.get("text_only_sha256") == _sha256(text_path), "STANCE_EVAL_LOCK_MISMATCH", "text-only set changed after selection lock")
    human, human_audit = _training_records(schema)
    auxiliary = _auxiliary_records(schema, human)
    eval_ids = {str(row["sample_id"]) for row in locked_text}
    eval_norms = {_norm(str(row["text"])) for row in locked_text}
    train_ids = {row.sample_id for row in human}
    train_norms = {_norm(row.model_text) for row in human}
    _require(not eval_ids.intersection(train_ids) and not eval_norms.intersection(train_norms), "STANCE_EVAL_TRAIN_LEAKAGE", "blind evaluation overlaps the training set")
    _np, torch, AutoModel, AutoTokenizer = m21._load_runtime()
    snapshot, snapshot_identity = m21._snapshot(m12.DEFAULT_CACHE)
    _require(snapshot.is_dir(), "STANCE_SNAPSHOT_MISSING", "fixed local LERT snapshot is not present", snapshot=str(snapshot))
    baseline = active.SemanticActiveInferenceRuntime.from_artifacts(version=active.VERSION_A5)
    if device_name is None:
        device, actual_device, _mps = m1._choose_device(torch)
    else:
        device = torch.device(device_name)
        actual_device = str(device)
    baseline.to(device)
    tokenizer = baseline.legacy_runtime.tokenizer
    config = baseline.config
    order = list(schema.class_order[HEAD])
    if resume_postfit:
        # A prior invocation completed the one fixed fit and persisted this
        # checkpoint, then failed while writing metadata. Resume only the
        # post-fit evidence path; never fit a second candidate.
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        _require(saved.get("stage_id") == STAGE_ID and saved.get("head") == HEAD and saved.get("base_model_version") == active.VERSION_A5, "STANCE_RECOVERY_CHECKPOINT_IDENTITY", "saved checkpoint is not this exact isolated Stance candidate")
        _require(saved.get("human_train_count") == len(human) and saved.get("paired_auxiliary_count") == len(auxiliary) and saved.get("base_seed") == BASE_SEED, "STANCE_RECOVERY_CHECKPOINT_CONFIG", "saved checkpoint does not match the frozen training inputs")
        best_epoch = int(saved.get("selected_epoch", 0))
        _require(1 <= best_epoch <= MAX_EPOCHS, "STANCE_RECOVERY_CHECKPOINT_EPOCH", "saved checkpoint has an invalid selected epoch", selected_epoch=best_epoch)
        candidate_model = _build_model(torch, AutoModel, snapshot, schema, config, baseline, device)
        candidate_model.specialist_blocks[HEAD].load_state_dict(saved["specialist_blocks_state_dict"], strict=True)
        candidate_model.heads[HEAD].load_state_dict(saved["head_state_dict"], strict=True)
        oof_metric = {"status": "five-fold epoch selection completed before serialization error; detailed metrics were not persisted and were not recomputed"}
        oof_curve = []
        oof_selected_metrics = None
        final_training = {"status": "fixed final fit completed and recovered from its saved checkpoint; loss details were not persisted"}
    else:
        cv = np.full((MAX_EPOCHS, len(human), len(order)), np.nan, dtype=np.float32)
        for fold_index, (fit_indices, val_indices) in enumerate(m12.fold_indices(human, HEAD, folds=m12.N_FOLDS, seed=BASE_SEED)):
            fit_rows = [human[index] for index in fit_indices]
            val_rows = [human[index] for index in val_indices]
            curve = _fit_curve(torch=torch, AutoModel=AutoModel, tokenizer=tokenizer, snapshot=snapshot, schema=schema, config=config, base_runtime=baseline, human_train=fit_rows, auxiliary=auxiliary, validation=val_rows, seed=BASE_SEED + fold_index, device=device, started=started)
            for epoch, values in enumerate(curve):
                cv[epoch, val_indices, :] = values
            _require(bool(np.isfinite(cv[:, val_indices, :]).all()), "STANCE_CV_FOLD_NONFINITE", "OOF fold produced incomplete or non-finite probabilities", fold=fold_index)
            print(json.dumps({"event": "stance_oof_fold_complete", "fold": fold_index + 1, "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
        _require(bool(np.isfinite(cv).all()), "STANCE_OOF_INCOMPLETE", "Stance OOF predictions are incomplete")
        best_epoch, oof_metric, _thresholds, oof_curve = m12._select_epoch(cv, human, schema, HEAD)
        oof_selected_metrics = m12._metrics(cv[best_epoch - 1], human, schema, HEAD)
        candidate_model, final_training = _fit_final(torch=torch, AutoModel=AutoModel, tokenizer=tokenizer, snapshot=snapshot, schema=schema, config=config, base_runtime=baseline, human_train=human, auxiliary=auxiliary, epochs=best_epoch, seed=BASE_SEED + 100, device=device, started=started)
        torch.save({
            "stage_id": STAGE_ID,
            "head": HEAD,
            "base_model_version": active.VERSION_A5,
            "human_train_count": len(human),
            "paired_auxiliary_count": len(auxiliary),
            "selected_epoch": best_epoch,
            "base_seed": BASE_SEED,
            "specialist_blocks_state_dict": {key: value.detach().cpu() for key, value in candidate_model.specialist_blocks[HEAD].state_dict().items()},
            "head_state_dict": {key: value.detach().cpu() for key, value in candidate_model.heads[HEAD].state_dict().items()},
        }, checkpoint)
    config_json = {
        "stage_id": STAGE_ID,
        "status": "CANDIDATE_TRAINED_EVALUATION_PENDING",
        "control": active.VERSION_A5,
        "control_action_artifact": str(active.DEFAULT_A5_ARTIFACT),
        "control_m12_artifact": str(m12.DEFAULT_PRIOR_M12_ARTIFACT),
        "control_stance_head": "M12 HumanTrain100 checkpoint, read-only",
        "model_id": m12.MODEL_ID,
        "revision": m12.REVISION,
        "snapshot_identity": snapshot_identity,
        "device": actual_device,
        "human_train": {"count": len(human), "input_source": "existing HumanTrain147 plus 3 final sparse Stance Train answers", "distribution": dict(Counter(str(row.label[HEAD]) for row in human)), "audit": human_audit},
        "paired_auxiliary": {"count": len(auxiliary), "weight": AUXILIARY_WEIGHT, "source_label": "synthetic paired support; not user annotation", "records": [{"sample_id": row.sample_id, "pair_kind": row.sample_id, "label": row.label[HEAD], "weight": row.weights[HEAD], "text": row.model_text} for row in auxiliary]},
        "training": {"max_epochs": MAX_EPOCHS, "selected_epoch": best_epoch, "epoch_selection": "5-fold human-only OOF Stance macro-F1; auxiliary pairs are fit-only", "seed": BASE_SEED, "batch_size": BATCH_SIZE, "optimizer": "AdamW", "block_learning_rate": BLOCK_LR, "head_learning_rate": HEAD_LR, "weight_decay": WEIGHT_DECAY, "gradient_clip": GRADIENT_CLIP, "class_balance": "M12 square-root inverse frequency calculated on each human fit fold only", "trainable_scope": [f"specialist_blocks.{HEAD}", f"heads.{HEAD}"], "frozen_scope": ["Encoder", "all six non-Stance specialist blocks and heads"], "postfit_recovery_without_refit": resume_postfit},
        "evaluation_review": {"path": str(review_path), "sha256": review_sha, "text_set_sha256": lock["text_only_sha256"], "selection_lock_sha256": _sha256(lock_path), "review_mode": "text_only_before_predictions", "labels_are": "provisional independent text review, not owner Human labels or Gold", "promotion_rule": dict(PROMOTION_RULE)},
        "forbidden_for_fit_or_selection": ["market outcomes", "Human Check", "blind evaluation posts", "Test", "Gold", "OOD", "price/model probability searches"],
        "no_other_heads_trained": True,
    }
    _dump(output / "training-config.json", config_json)
    _dump(output / "oof.json", {"selected_epoch": best_epoch, "selection_metrics": oof_metric, "curve": oof_curve, "human_oof_at_selected_epoch": oof_selected_metrics, "metrics_persisted_before_recovery": not resume_postfit})
    _write_jsonl(output / "paired-auxiliary-train.jsonl", [{"sample_id": row.sample_id, "text": row.model_text, "stance": row.label[HEAD], "weight": row.weights[HEAD], "source": "synthetic_paired_auxiliary_not_user_annotation"} for row in auxiliary])
    _write_jsonl(output / "human-train-stance-snapshot.jsonl", [{"sample_id": row.sample_id, "text": row.model_text, "answer_preserved_verbatim": row.label[HEAD], "weight": row.weights.get(HEAD, 1.0), "source": "existing_human_training_record"} for row in human])
    # Only now, after checkpoint and training config are frozen, do we read
    # existing Human Check labels and expose A5/candidate predictions. If a
    # prior post-fit invocation already wrote the paired blind predictions,
    # validate and reuse them rather than opening that set a second time.
    eval_labelled_records = _labelled_eval_records(review, schema)
    eval_inference = _as_inference(eval_labelled_records)
    label_by_id = {str(row["sample_id"]): row for row in review if row["review_label"] != CANNOT_DETERMINE}
    labels = [str(label_by_id[row.sample_id]["review_label"]) for row in eval_labelled_records]
    by_id = {str(row["sample_id"]): row for row in review}
    saved_eval = _load_saved_blind_evaluation(output, review, eval_labelled_records, schema)
    if saved_eval is None:
        check_sets = _legacy_checks(schema)
        check_results = {name: _check_metrics(baseline, candidate_model, rows, schema, device) for name, rows in check_sets.items()}
        a5_eval_rows = baseline.predict_records(eval_inference)
        candidate_probs = _candidate_probabilities(torch, candidate_model, tokenizer, eval_inference, config, device)
        candidate_eval_rows = _merge_stance_rows(a5_eval_rows, candidate_probs, schema, STAGE_ID)
        a5_eval_probs = np.asarray([[float(row["predictions"][HEAD]["probabilities"][label]) for label in order] for row in a5_eval_rows])
        a5_preds = [str(row["predictions"][HEAD]["prediction"]) for row in a5_eval_rows]
        cand_preds = [str(row["predictions"][HEAD]["prediction"]) for row in candidate_eval_rows]
        eval_metrics = {"a5": _eval_predictions(a5_eval_probs, labels, schema), "candidate": _eval_predictions(candidate_probs, labels, schema)}
        eval_rows_output = []
        for base_row, cand_row in zip(a5_eval_rows, candidate_eval_rows):
            sid = str(base_row["sample_id"])
            review_row = by_id[sid]
            eval_rows_output.append({"sample_id": sid, "text": review_row["text"], "review_label": review_row["review_label"], "probe_type": review_row.get("probe_type", review_row.get("selection_stratum")), "evidence": review_row["evidence"], "a5_prediction": base_row["predictions"][HEAD]["prediction"], "candidate_prediction": cand_row["predictions"][HEAD]["prediction"], "a5_probabilities": base_row["predictions"][HEAD]["probabilities"], "candidate_probabilities": cand_row["predictions"][HEAD]["probabilities"]})
        _write_jsonl(output / "blind-evaluation-predictions.jsonl", eval_rows_output)
    else:
        saved_evaluation, eval_rows_output = saved_eval
        if saved_evaluation is None:
            check_sets = _legacy_checks(schema)
            check_results = {name: _check_metrics(baseline, candidate_model, rows, schema, device) for name, rows in check_sets.items()}
        else:
            check_results = saved_evaluation["human_checks_opened_after_candidate_freeze"]
        a5_eval_probs = np.asarray([[float(row["a5_probabilities"][label]) for label in order] for row in eval_rows_output], dtype=np.float32)
        candidate_probs = np.asarray([[float(row["candidate_probabilities"][label]) for label in order] for row in eval_rows_output], dtype=np.float32)
        eval_metrics = {"a5": _eval_predictions(a5_eval_probs, labels, schema), "candidate": _eval_predictions(candidate_probs, labels, schema)}
        a5_preds = [str(row["a5_prediction"]) for row in eval_rows_output]
        cand_preds = [str(row["candidate_prediction"]) for row in eval_rows_output]
    direction_review = [row for row in review if row["review_label"] != CANNOT_DETERMINE]
    direction_review_by_id = {str(row["sample_id"]): row for row in direction_review}
    a5_errors = _directional_counts([direction_review_by_id[row.sample_id] for row in eval_labelled_records], a5_preds)
    candidate_errors = _directional_counts([direction_review_by_id[row.sample_id] for row in eval_labelled_records], cand_preds)
    adequate = {
        "future_bull": sum(row["selection_stratum"] == "future_bull" and row["review_label"] == "BULL" for row in review),
        "future_bear": sum(row["selection_stratum"] == "future_bear" and row["review_label"] == "BEAR" for row in review),
        "past_regret_only": sum(row["selection_stratum"] == "past_regret_only" and row["review_label"] == "NEUTRAL" for row in review),
    }
    components = ["BULL_to_BEAR", "BEAR_to_BULL", "past_regret_to_directional"]
    clear_improvement = (
        adequate["future_bull"] >= int(PROMOTION_RULE["requires_reviewed_future_bull"])
        and adequate["future_bear"] >= int(PROMOTION_RULE["requires_reviewed_future_bear"])
        and adequate["past_regret_only"] >= int(PROMOTION_RULE["requires_reviewed_past_regret_only"])
        and a5_errors["target_error_total"] - candidate_errors["target_error_total"] >= int(PROMOTION_RULE["candidate_target_errors_must_fall_by_at_least"])
        and all(candidate_errors[key] <= a5_errors[key] for key in components)
        and (candidate_errors["future_direction_accuracy"] or 0.0) >= (a5_errors["future_direction_accuracy"] or 0.0)
    )
    # Text-only review and checkpoint already existed before either prediction
    # was opened; save exact order and sample-level paired evidence for audit.
    control_review_path = output / "a5-control-review-predictions.jsonl"
    if not control_review_path.exists():
        _write_jsonl(control_review_path, [{"sample_id": row["sample_id"], "prediction": row["a5_prediction"], "probabilities": row["a5_probabilities"]} for row in eval_rows_output])
    _dump(output / "evaluation.json", {"review_set_count": len(review), "labelled_count": len(eval_labelled_records), "cannot_determine_count": len(review) - len(eval_labelled_records), "review_set_adequacy": adequate, "a5": eval_metrics["a5"], "candidate": eval_metrics["candidate"], "directional_error_mechanisms": {"a5": a5_errors, "candidate": candidate_errors, "delta_target_error_count": candidate_errors["target_error_total"] - a5_errors["target_error_total"]}, "clear_directional_improvement": clear_improvement, "promotion_rule": dict(PROMOTION_RULE), "human_checks_opened_after_candidate_freeze": check_results, "blind_review_was_frozen_before_training_and_predictions": True, "market_outcomes_read": False})
    distribution_path = output / "full-prediction-distribution.json"
    if distribution_path.is_file():
        full_distribution = json.loads(distribution_path.read_text(encoding="utf-8"))
    else:
        full_distribution = _public_distribution(baseline, candidate_model, Path(public_pool).resolve(), schema, device, output)
        _dump(distribution_path, full_distribution)
    # The full prediction pass has completed. CPU reload is verification only;
    # no input data or model choice is changed after viewing held-out results.
    cpu_model = _build_model(torch, AutoModel, snapshot, schema, config, baseline, torch.device("cpu"))
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cpu_model.specialist_blocks[HEAD].load_state_dict(saved["specialist_blocks_state_dict"], strict=True)
    cpu_model.heads[HEAD].load_state_dict(saved["head_state_dict"], strict=True)
    cpu_probs = _candidate_probabilities(torch, cpu_model, tokenizer, eval_inference, config, torch.device("cpu"))
    _require(np.allclose(cpu_probs, candidate_probs, rtol=0.0, atol=1e-6), "STANCE_CPU_RELOAD_MISMATCH", "CPU checkpoint reload differs on the frozen diagnostic evaluation set")
    if not resume_postfit:
        torch.save(saved, output / "stance-checkpoint.pt")
    status = "NEXT_RESEARCH_VERSION__NOT_USER_HUMAN_ACCEPTED" if clear_improvement else "REJECTED_KEEP_A5"
    summary = {
        "stage_id": STAGE_ID,
        "status": status,
        "control": active.VERSION_A5,
        "candidate_version": "SEM7-20260929-S1" if clear_improvement else "SEM7-20260929-S1-REJECTED",
        "candidate_changes_only": [HEAD],
        "human_train_count": len(human),
        "paired_auxiliary_count": len(auxiliary),
        "selected_epoch": best_epoch,
        "device": actual_device,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "postfit_recovered_without_refit": resume_postfit,
        "training_rerun": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "cpu_reload": "PASS",
        "blind_text_review": {"sample_count": len(review), "labelled_count": len(eval_labelled_records), "reviewer_role": "provisional text-only review; not user human acceptance"},
        "a5_direction_errors": a5_errors,
        "candidate_direction_errors": candidate_errors,
        "directional_improvement": clear_improvement,
        "human_check_metrics": {name: {"count": value.get("count"), "a5_macro_f1": value.get("a5", {}).get("macro_f1"), "candidate_macro_f1": value.get("candidate", {}).get("macro_f1"), "a5_accuracy": value.get("a5", {}).get("accuracy"), "candidate_accuracy": value.get("candidate", {}).get("accuracy")} for name, value in check_results.items()},
        "full_pool_count": full_distribution["sample_count"],
        "full_pool_stance_changes": full_distribution["stance_argmax_change_count"],
        "all_untrained_heads_unchanged": full_distribution["all_untrained_head_rows_identical"],
        "market_outcomes_used": False,
        "user_human_acceptance": False,
    }
    _dump(output / "summary.json", summary)
    report = _render_report(summary, json.loads((output / "evaluation.json").read_text(encoding="utf-8")), full_distribution, final_training)
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    return summary


def _render_report(summary: Mapping[str, Any], evaluation: Mapping[str, Any], full: Mapping[str, Any], final: Mapping[str, Any]) -> str:
    fit_loss = final.get("mean_batch_loss")
    fit_loss_text = f"{float(fit_loss):.5f}" if fit_loss is not None else "not retained by the original post-fit serialization failure; candidate was not refit"
    lines = [
        "# Stance-only paired temporal contrast diagnostic",
        "",
        f"- Status: `{summary['status']}`",
        f"- Frozen control: `{summary['control']}` (untouched)",
        f"- Candidate changes only: `{HEAD}`; Action and the other five heads are not trained.",
        f"- Human Train: {summary['human_train_count']} existing samples; auxiliary pairs: {summary['paired_auxiliary_count']} explicitly synthetic support rows.",
        f"- Text-only real-post review: {evaluation['review_set_count']} rows, {evaluation['labelled_count']} explicit provisional labels, {evaluation['cannot_determine_count']} unable-to-decide.",
        "- Review answers are provisional researcher text judgments, not user acceptance or Gold.",
        f"- Selected epoch: {summary['selected_epoch']}; final mean batch loss {fit_loss_text}; device `{summary['device']}`; this process elapsed {summary['elapsed_seconds']:.1f}s.",
        "",
        "## Blind real-post evaluation",
        "",
        "| metric | A5 | isolated candidate |",
        "|---|---:|---:|",
        f"| accuracy | {evaluation['a5']['accuracy']:.4f} | {evaluation['candidate']['accuracy']:.4f} |",
        f"| macro-F1 | {evaluation['a5']['macro_f1']:.4f} | {evaluation['candidate']['macro_f1']:.4f} |",
        "",
        "| targeted failure mode | A5 | candidate |",
        "|---|---:|---:|",
    ]
    for key, label in (("BULL_to_BEAR", "explicit future bullish → bearish"), ("BEAR_to_BULL", "explicit future bearish → bullish"), ("past_regret_to_directional", "past-only buy/sell regret → future directional stance")):
        lines.append(f"| {label} | {evaluation['directional_error_mechanisms']['a5'][key]} | {evaluation['directional_error_mechanisms']['candidate'][key]} |")
    lines += [
        f"| future BULL/BEAR exact-label accuracy | {evaluation['directional_error_mechanisms']['a5']['future_direction_accuracy']!s} | {evaluation['directional_error_mechanisms']['candidate']['future_direction_accuracy']!s} |",
        "",
        f"Clear improvement rule: `{evaluation['clear_directional_improvement']}`. The rule was frozen before either model prediction was opened; market returns were not used.",
        "",
        "## Existing Human Check (descriptive, reused)",
        "",
        "These historical sets are not independent acceptance data and did not choose the epoch.",
        "",
        "| set | n | A5 accuracy / macro-F1 | candidate accuracy / macro-F1 |",
        "|---|---:|---:|---:|",
    ]
    for name, result in evaluation["human_checks_opened_after_candidate_freeze"].items():
        if result.get("count", 0):
            before, after = result["a5"], result["candidate"]
            lines.append(f"| {name} | {result['count']} | {before['accuracy']:.4f} / {before['macro_f1']:.4f} | {after['accuracy']:.4f} / {after['macro_f1']:.4f} |")
    lines += [
        "",
        "## Full frozen-pool distribution",
        "",
        f"Source rows: {full['sample_count']}; Stance argmax changed {full['stance_argmax_change_count']} ({full['stance_argmax_change_rate']:.2%}).",
        f"A5 distribution: `{json.dumps(full['a5_stance_distribution'], ensure_ascii=False, sort_keys=True)}`",
        f"Candidate distribution: `{json.dumps(full['candidate_stance_distribution'], ensure_ascii=False, sort_keys=True)}`",
        f"Untrained outputs byte-value equal across every row: `{full['all_untrained_head_rows_identical']}`.",
        "",
        "## Decision",
        "",
    ]
    if summary["status"] == "NEXT_RESEARCH_VERSION__NOT_USER_HUMAN_ACCEPTED":
        lines.append("The candidate is preserved as a reversible next research version only. It has **not** received user human acceptance and does not replace A5's default status.")
    else:
        lines.append("No sufficiently clear directional improvement was demonstrated. Keep A5 unchanged; the observed remaining mechanism is documented below. A next bounded method is structured supervision separating (a) future polarity and scope from (b) past transaction outcome/regret as distinct targets, with stance loss masked when a post contains only retrospective regret. Do not relabel old answers or infer stance from market returns.")
    lines += ["", "No Action or other non-Stance head was trained. No market outcome was read.", ""]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-eval", help="freeze unseen Collector texts without reading model predictions")
    prepare.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    prepare.add_argument("--collector-db", type=Path, default=DEFAULT_COLLECTOR_DB)
    run_parser = sub.add_parser("run", help="fit one isolated Stance candidate and evaluate after the text review is frozen")
    run_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    run_parser.add_argument("--public-pool", type=Path, default=PUBLIC_POOL)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-eval":
            result = prepare_evaluation_texts(output_dir=args.output_dir, db_path=args.collector_db)
        else:
            result = run(output_dir=args.output_dir, public_pool=args.public_pool)
    except (ContractError, OSError, RuntimeError, sqlite3.Error) as exc:
        print(json.dumps({"status": "BLOCKED", "error": str(exc), "code": getattr(exc, "code", None)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
