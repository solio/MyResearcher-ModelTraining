from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .errors import ContractError


SCHEMA_VERSION = "semantic-schema-calibrated-v0.2.1"
V1_HEADS = (
    "target_mode",
    "stance",
    "emotion_primary",
    "emotion_target",
    "action_tendency",
    "reasoning_tags",
    "context_dependency",
)
SINGLE_LABEL_HEADS = tuple(head for head in V1_HEADS if head != "reasoning_tags")
FROZEN_CLASS_ORDER: dict[str, list[str]] = {
    "target_mode": ["ON_TARGET", "CROSS_TARGET", "MARKET_GENERAL", "UNKNOWN"],
    "stance": ["BULL", "BEAR", "NEUTRAL", "MIXED", "UNKNOWN"],
    "emotion_primary": [
        "FEAR",
        "ANXIETY",
        "ANGER",
        "FRUSTRATION",
        "REGRET",
        "HOPE",
        "EXCITEMENT",
        "FOMO",
        "CALM",
        "NONE_EXPLICIT",
        "UNKNOWN",
    ],
    "emotion_target": [
        "PRICE",
        "POSITION",
        "COMPANY",
        "MARKET",
        "OTHER",
        "NOT_APPLICABLE",
        "UNKNOWN",
    ],
    "action_tendency": [
        "BUY",
        "ADD",
        "HOLD",
        "DO_T",
        "REDUCE",
        "SELL",
        "WATCH",
        "NO_ACTION_SIGNAL",
        "UNKNOWN",
    ],
    "reasoning_tags": [
        "FUNDAMENTAL",
        "VALUATION",
        "TECHNICAL_PRICE",
        "FLOW_POSITIONING",
        "NEWS_EVENT",
        "RUMOR",
        "SOCIAL_PROOF",
        "MACRO_POLICY",
        "THEME_NARRATIVE",
        "NO_REASON_GIVEN",
        "RELATIVE_PERFORMANCE",
        "CROSS_STOCK_REFERENCE",
        "SARCASM_IRONY",
        "WORDPLAY",
        "UNKNOWN",
    ],
    "context_dependency": [
        "SELF_CONTAINED",
        "PARTIAL_CONTEXT",
        "EXTERNAL_CONTEXT_REQUIRED",
        "UNKNOWN",
    ],
}
CONFIDENCE_ORDER = ["HIGH", "MEDIUM", "LOW"]


@dataclass(frozen=True)
class LabelSchema:
    path: Path
    schema_version: str
    class_order: Mapping[str, list[str]]
    raw: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "LabelSchema":
        schema_path = Path(path).resolve()
        try:
            raw = json.loads(schema_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ContractError("SCHEMA_NOT_FOUND", str(schema_path)) from exc
        except json.JSONDecodeError as exc:
            raise ContractError("SCHEMA_INVALID", str(exc), path=str(schema_path)) from exc
        if raw.get("schema_version") != SCHEMA_VERSION:
            raise ContractError(
                "SCHEMA_VERSION_MISMATCH",
                "frozen schema file has an unexpected version",
                observed=raw.get("schema_version"),
                expected=SCHEMA_VERSION,
            )
        heads = raw.get("v1_prediction_heads")
        if not isinstance(heads, dict):
            raise ContractError("SCHEMA_INVALID", "v1_prediction_heads must be an object")
        actual = {
            head: heads.get(head, {}).get("class_order")
            for head in V1_HEADS
        }
        if actual != FROZEN_CLASS_ORDER:
            raise ContractError(
                "SCHEMA_CLASS_ORDER_MISMATCH",
                "class order differs from the frozen regression contract",
                actual=actual,
                expected=FROZEN_CLASS_ORDER,
            )
        return cls(
            path=schema_path,
            schema_version=SCHEMA_VERSION,
            class_order=actual,
            raw=raw,
        )

    def encode_reasoning_tags(self, tags: list[str]) -> list[int]:
        if not isinstance(tags, list):
            raise ContractError("MULTI_LABEL_REQUIRED", "reasoning_tags must be a list")
        tag_set = set(tags)
        if len(tag_set) != len(tags):
            raise ContractError("DUPLICATE_REASONING_TAG", "reasoning_tags must be unique")
        unknown = tag_set - set(self.class_order["reasoning_tags"])
        if unknown:
            raise ContractError(
                "UNKNOWN_LABEL_VALUE", "unknown reasoning tag", values=sorted(unknown)
            )
        return [int(label in tag_set) for label in self.class_order["reasoning_tags"]]

