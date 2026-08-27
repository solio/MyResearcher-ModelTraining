from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data import read_json
from .errors import ContractError
from .hashes import content_addressed_id, verify_content_addressed_id


@dataclass(frozen=True)
class PreprocessingContract:
    contract_id: str
    contract_version: str
    status: str
    include_board_context: bool
    board_marker: str
    text_marker: str
    separator: str
    normalize_unicode: str
    strip_outer_whitespace: bool
    exact_template: str | None
    feature_stack_order: tuple[str, ...]
    char_tfidf: Mapping[str, Any]
    word_tfidf: Mapping[str, Any]
    expected_feature_counts: Mapping[str, int]
    raw: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "PreprocessingContract":
        if not isinstance(raw, Mapping):
            raise ContractError(
                "PREPROCESSING_CONTRACT_INVALID", "contract must be an object"
            )
        native = raw.get("schema_version") == "semantic-preprocessing-contract-v0.3.5"
        if native:
            contract_id = content_addressed_id(raw)
        else:
            contract_id = verify_content_addressed_id(
                raw, id_key="preprocessing_contract_id"
            )
        version = raw.get("schema_version") if native else raw.get("contract_version")
        if not isinstance(version, str) or not version:
            raise ContractError(
                "PREPROCESSING_CONTRACT_INVALID", "contract_version is required"
            )
        normalize_unicode = "NONE" if native else raw.get("normalize_unicode", "NFC")
        if normalize_unicode not in {"NFC", "NFKC", "NFD", "NFKD", "NONE"}:
            raise ContractError(
                "PREPROCESSING_CONTRACT_INVALID",
                "unsupported Unicode normalization",
                value=normalize_unicode,
            )
        if native:
            normalize_text = raw.get("normalize_text")
            if not isinstance(normalize_text, Mapping):
                raise ContractError(
                    "PREPROCESSING_CONTRACT_INVALID",
                    "native normalize_text must be an object",
                )
            exact_template = normalize_text.get("exact_template")
            if exact_template != "[股票]{stock_code} {stock_name} [帖子]{model_text}":
                raise ContractError(
                    "PREPROCESSING_CONTRACT_INVALID",
                    "unexpected v0.3.5 input template",
                    observed=exact_template,
                )
            no_normalization = {
                "normalizations_applied": [],
                "lowercase": False,
                "unicode_normalization": None,
                "whitespace_collapse": False,
                "url_masking": False,
                "emoji_removal": False,
                "traditional_simplified_conversion": False,
                "truncation": None,
            }
            observed_normalization = {
                key: normalize_text.get(key) for key in no_normalization
            }
            if observed_normalization != no_normalization:
                raise ContractError(
                    "PREPROCESSING_CONTRACT_INVALID",
                    "v0.3.5 forbids implicit text normalization",
                    observed=observed_normalization,
                )
            feature_stack = raw.get("feature_stack_order")
            char_tfidf = raw.get("char_tfidf")
            word_tfidf = raw.get("word_tfidf")
            feature_counts = raw.get("expected_fitted_feature_counts")
            if feature_stack != ["char_tfidf", "word_tfidf"] or not all(
                isinstance(value, Mapping)
                for value in (char_tfidf, word_tfidf, feature_counts)
            ):
                raise ContractError(
                    "PREPROCESSING_CONTRACT_INVALID",
                    "native feature preparation contract is incomplete",
                )
        else:
            exact_template = None
            feature_stack = ["word_tfidf", "char_tfidf"]
            char_tfidf = {}
            word_tfidf = {}
            feature_counts = {}
        return cls(
            contract_id=contract_id,
            contract_version=version,
            status=str(raw.get("status", "FROZEN" if native else "UNKNOWN")),
            include_board_context=True if native else bool(raw.get("include_board_context", True)),
            board_marker=str(raw.get("board_marker", "[BOARD]")),
            text_marker=str(raw.get("text_marker", "[TEXT]")),
            separator=str(raw.get("separator", "\n")),
            normalize_unicode=str(normalize_unicode),
            strip_outer_whitespace=False if native else bool(raw.get("strip_outer_whitespace", True)),
            exact_template=str(exact_template) if exact_template is not None else None,
            feature_stack_order=tuple(str(value) for value in feature_stack),
            char_tfidf=dict(char_tfidf),
            word_tfidf=dict(word_tfidf),
            expected_feature_counts={
                str(key): int(value) for key, value in feature_counts.items()
            },
            raw=raw,
        )

    @classmethod
    def load(cls, path: str | Path) -> "PreprocessingContract":
        raw = read_json(path)
        if not isinstance(raw, Mapping):
            raise ContractError(
                "PREPROCESSING_CONTRACT_INVALID", "contract must be an object"
            )
        return cls.from_mapping(raw)


def _normalize(value: str, contract: PreprocessingContract) -> str:
    result = value.strip() if contract.strip_outer_whitespace else value
    if contract.normalize_unicode != "NONE":
        result = unicodedata.normalize(contract.normalize_unicode, result)
    return result


def build_model_input(
    record: Mapping[str, Any], contract: PreprocessingContract
) -> str:
    """The only text-construction implementation used by every pipeline stage."""

    model_text = record.get("model_text", record.get("text"))
    if not isinstance(model_text, str) or not model_text:
        raise ContractError(
            "CANONICAL_MODEL_TEXT_INVALID",
            "model_text must be a non-empty string",
            sample_id=record.get("sample_id"),
        )
    if contract.exact_template is not None:
        stock_code = record.get("stock_code", "")
        stock_name = record.get("stock_name", "")
        if not isinstance(stock_code, str) or not isinstance(stock_name, str):
            raise ContractError(
                "BOARD_CONTEXT_INVALID",
                "native template requires string stock_code and stock_name",
                sample_id=record.get("sample_id"),
            )
        return contract.exact_template.format(
            stock_code=stock_code,
            stock_name=stock_name,
            model_text=model_text,
        )
    normalized_text = _normalize(model_text, contract)
    text_part = f"{contract.text_marker} {normalized_text}".rstrip()
    if not contract.include_board_context:
        return text_part
    board_context = record.get("board_context")
    if not isinstance(board_context, str) or not board_context.strip():
        stock_code = record.get("stock_code")
        stock_name = record.get("stock_name")
        if not isinstance(stock_code, str) or not isinstance(stock_name, str):
            raise ContractError(
                "BOARD_CONTEXT_INVALID",
                "board context requires provided board_context or stock_code/stock_name",
                sample_id=record.get("sample_id"),
            )
        board_context = f"{stock_code} {stock_name}"
    normalized_board = _normalize(board_context, contract)
    board_part = f"{contract.board_marker} {normalized_board}".rstrip()
    return contract.separator.join((board_part, text_part))


def build_model_inputs(
    records: Sequence[Mapping[str, Any]], contract: PreprocessingContract
) -> list[str]:
    return [build_model_input(record, contract) for record in records]
