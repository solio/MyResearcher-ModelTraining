from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .errors import ContractError


class DataRole(str, Enum):
    CANONICAL_INPUT = "CANONICAL_INPUT"
    FROZEN_TEACHER_LABEL = "FROZEN_TEACHER_LABEL"
    WEAK_LABEL = "WEAK_LABEL"
    TEACHER_CANDIDATE = "TEACHER_CANDIDATE"
    GOLD_CANDIDATE = "GOLD_CANDIDATE"
    REVIEWED_LABEL = "REVIEWED_LABEL"
    HUMAN_GOLD = "HUMAN_GOLD"
    ANCHOR = "ANCHOR"
    MODEL_PREDICTION = "MODEL_PREDICTION"
    CHALLENGE = "CHALLENGE"
    QUARANTINE = "QUARANTINE"
    EMBARGO = "EMBARGO"


_PERMISSIONS = {
    "train_label": {DataRole.WEAK_LABEL},
    "evaluation": {DataRole.HUMAN_GOLD, DataRole.ANCHOR, DataRole.CHALLENGE},
    "annotation": {
        DataRole.FROZEN_TEACHER_LABEL,
        DataRole.WEAK_LABEL,
        DataRole.TEACHER_CANDIDATE,
        DataRole.GOLD_CANDIDATE,
        DataRole.REVIEWED_LABEL,
        DataRole.HUMAN_GOLD,
        DataRole.ANCHOR,
        DataRole.CHALLENGE,
    },
}


def assert_role_permission(role: DataRole, *, purpose: str) -> None:
    if role == DataRole.MODEL_PREDICTION and purpose == "annotation":
        raise ContractError(
            "PREDICTION_IS_NOT_ANNOTATION", "model prediction cannot become annotation"
        )
    if role not in _PERMISSIONS.get(purpose, set()):
        raise ContractError(
            "DATA_ROLE_PERMISSION_DENIED",
            f"{role.value} is not allowed for {purpose}",
            role=role.value,
            purpose=purpose,
        )


def read_json(path: str | Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ContractError("ARTIFACT_NOT_FOUND", str(path)) from exc
    except json.JSONDecodeError as exc:
        raise ContractError("INVALID_JSON", str(exc), path=str(path)) from exc


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        handle = Path(path).open("r", encoding="utf-8-sig")
    except FileNotFoundError as exc:
        raise ContractError("ARTIFACT_NOT_FOUND", str(path)) from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(
                    "INVALID_JSONL", str(exc), path=str(path), line=line_number
                ) from exc
            if not isinstance(record, dict):
                raise ContractError(
                    "INVALID_JSONL_RECORD",
                    "JSONL records must be objects",
                    path=str(path),
                    line=line_number,
                )
            records.append(record)
    return records


def index_by_sample_id(
    records: Iterable[Mapping[str, Any]], *, role: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row_number, record in enumerate(records, 1):
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ContractError(
                "MISSING_SAMPLE_ID", f"{role} record has no sample_id", row=row_number
            )
        if sample_id in result:
            raise ContractError(
                "DUPLICATE_SAMPLE_ID", f"duplicate {role} sample_id", sample_id=sample_id
            )
        result[sample_id] = record
    return result


@dataclass(frozen=True)
class JoinedRecord:
    input: Mapping[str, Any]
    label: Mapping[str, Any]


def join_inputs_and_labels(
    inputs: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    *,
    require_complete: bool = True,
) -> list[JoinedRecord]:
    input_index = index_by_sample_id(inputs, role="input")
    label_index = index_by_sample_id(labels, role="label")
    extra = sorted(set(label_index) - set(input_index))
    if extra:
        raise ContractError(
            "LABEL_SAMPLE_ID_NOT_FOUND",
            "label sample_id is absent from canonical input",
            sample_ids=extra,
        )
    missing = sorted(set(input_index) - set(label_index))
    if require_complete and missing:
        raise ContractError(
            "MISSING_LABEL_SAMPLE_ID",
            "canonical input is missing a required label",
            sample_ids=missing,
        )
    for sample_id, label in label_index.items():
        canonical = input_index[sample_id]
        if "published_at" in label and not isinstance(label["published_at"], str):
            raise ContractError(
                "NON_CANONICAL_LABEL_TIMESTAMP",
                "label published_at must be a canonical string and never defines split time",
                sample_id=sample_id,
                value=label["published_at"],
            )
        mismatches = {
            field: {"canonical": canonical.get(field), "label": label.get(field)}
            for field in ("stock_code", "stock_name", "published_at")
            if field in label and label.get(field) != canonical.get(field)
        }
        if mismatches:
            raise ContractError(
                "CANONICAL_METADATA_MISMATCH",
                "repeated label metadata disagrees with canonical input",
                sample_id=sample_id,
                mismatches=mismatches,
            )
    return [
        JoinedRecord(input=input_index[sample_id], label=label_index[sample_id])
        for sample_id in input_index
        if sample_id in label_index
    ]

