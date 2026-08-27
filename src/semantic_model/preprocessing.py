from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data import read_json
from .errors import ContractError
from .hashes import verify_content_addressed_id


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
    raw: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "PreprocessingContract":
        if not isinstance(raw, Mapping):
            raise ContractError(
                "PREPROCESSING_CONTRACT_INVALID", "contract must be an object"
            )
        contract_id = verify_content_addressed_id(
            raw, id_key="preprocessing_contract_id"
        )
        version = raw.get("contract_version")
        if not isinstance(version, str) or not version:
            raise ContractError(
                "PREPROCESSING_CONTRACT_INVALID", "contract_version is required"
            )
        normalize_unicode = raw.get("normalize_unicode", "NFC")
        if normalize_unicode not in {"NFC", "NFKC", "NFD", "NFKD", "NONE"}:
            raise ContractError(
                "PREPROCESSING_CONTRACT_INVALID",
                "unsupported Unicode normalization",
                value=normalize_unicode,
            )
        return cls(
            contract_id=contract_id,
            contract_version=version,
            status=str(raw.get("status", "UNKNOWN")),
            include_board_context=bool(raw.get("include_board_context", True)),
            board_marker=str(raw.get("board_marker", "[BOARD]")),
            text_marker=str(raw.get("text_marker", "[TEXT]")),
            separator=str(raw.get("separator", "\n")),
            normalize_unicode=str(normalize_unicode),
            strip_outer_whitespace=bool(raw.get("strip_outer_whitespace", True)),
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

    model_text = record.get("model_text")
    if not isinstance(model_text, str) or not model_text:
        raise ContractError(
            "CANONICAL_MODEL_TEXT_INVALID",
            "model_text must be a non-empty string",
            sample_id=record.get("sample_id"),
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

