from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from .errors import ContractError
from .schema import SCHEMA_VERSION

if TYPE_CHECKING:
    from .schema import SemanticSchemaCandidate


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
    auxiliary: AuxiliaryFactorLabels | None = None


@dataclass(frozen=True)
class AuxiliaryFactorLabels:
    """Optional v0.3 factor values paired with explicit training masks.

    ``values[name] is None`` and ``loss_mask[name] == 0`` mean that the factor
    was not annotated or was not applicable.  A literal ``"UNKNOWN"`` (or a
    list containing it for a multi-label factor) is a real annotated answer and
    therefore has mask ``1``.
    """

    values: Mapping[str, Any]
    loss_mask: Mapping[str, int]

    def __post_init__(self) -> None:
        if set(self.values) != set(self.loss_mask):
            raise ContractError(
                "AUXILIARY_LOSS_MASK_INVALID",
                "values and loss_mask must contain the same factors",
            )
        for factor, mask in self.loss_mask.items():
            if mask not in (0, 1):
                raise ContractError(
                    "AUXILIARY_LOSS_MASK_INVALID",
                    "loss_mask values must be 0 or 1",
                    factor=factor,
                    mask=mask,
                )
            if mask == 0 and self.values[factor] is not None:
                raise ContractError(
                    "AUXILIARY_LOSS_MASK_INVALID",
                    "an unmasked factor must have no value",
                    factor=factor,
                )

    def is_annotated(self, factor: str) -> bool:
        """Whether ``factor`` has an answer that may contribute to loss."""

        try:
            return bool(self.loss_mask[factor])
        except KeyError as exc:
            raise ContractError("UNKNOWN_AUXILIARY_FACTOR", "factor is not present", factor=factor) from exc


@dataclass(frozen=True)
class AuxiliaryAssignment:
    """One row in the labeling app's sparse auxiliary export."""

    sample_id: str
    head: str
    answer: Any
    is_final: bool
    schema_version: str


def parse_auxiliary_assignment(
    assignment: Mapping[str, Any], schema: "SemanticSchemaCandidate"
) -> AuxiliaryAssignment:
    """Parse one app assignment without requiring or copying seven-head labels."""

    sample_id, answer, is_final = schema.validate_auxiliary_assignment(assignment)
    return AuxiliaryAssignment(
        sample_id=sample_id,
        head=str(assignment["head"]),
        answer=answer,
        is_final=is_final,
        schema_version=str(assignment["schema_version"]),
    )


load_auxiliary_assignment = parse_auxiliary_assignment


@dataclass(frozen=True)
class SemanticTrainingView:
    """Internal view joining one legacy seven-head row with sparse factors."""

    sample_id: str
    legacy_label: Mapping[str, Any]
    auxiliary: AuxiliaryFactorLabels
    source: Mapping[str, Any]

    @property
    def label(self) -> Mapping[str, Any]:
        """Compatibility alias for callers that use the legacy label name."""

        return self.legacy_label


def parse_optional_auxiliary_factors(
    record: Mapping[str, Any], schema: "SemanticSchemaCandidate"
) -> AuxiliaryFactorLabels:
    """Parse optional factors for a v0.3 or legacy v0.2.1 label row.

    The candidate schema owns the allowed values and cardinalities; this data
    layer only packages its result for loaders and training code.  Legacy rows
    with no ``auxiliary_factors`` field therefore remain valid and produce five
    zero masks rather than fabricated ``UNKNOWN`` labels.
    """

    values, masks = schema.parse_auxiliary_factors(record)
    return AuxiliaryFactorLabels(values=dict(values), loss_mask=dict(masks))


# Name reads naturally at call sites that load a label row from JSONL.
load_optional_auxiliary_factors = parse_optional_auxiliary_factors


def merge_legacy_labels_and_auxiliary_assignments(
    legacy_labels: Sequence[Mapping[str, Any]],
    assignments: Sequence[Mapping[str, Any]],
    schema: "SemanticSchemaCandidate",
) -> list[SemanticTrainingView]:
    """Merge app-style final assignments into a v0.3 internal training view.

    Legacy rows remain the authoritative v0.2.1 seven-head labels.  Only a
    unique final assignment for one ``sample_id``/factor can set a factor mask
    to one; unfinished assignments are retained only in source counts and do
    not contribute to loss.  No auxiliary answer is promoted to a legacy head.
    """

    legacy_index = index_by_sample_id(legacy_labels, role="legacy_label")
    for sample_id, label in legacy_index.items():
        if label.get("schema_version") != schema.base_schema_version:
            raise ContractError(
                "LEGACY_LABEL_SCHEMA_INVALID",
                "merge requires a full frozen v0.2.1 seven-head label",
                sample_id=sample_id,
                observed=label.get("schema_version"),
            )
        if "auxiliary_factors" in label:
            raise ContractError(
                "V021_AUXILIARY_FORBIDDEN",
                "a legacy v0.2.1 row cannot carry auxiliary factors",
                sample_id=sample_id,
            )
        # Run the complete legacy semantic checks before combining sources.
        schema.validate_record(label)

    final_by_pair: dict[tuple[str, str], Mapping[str, Any]] = {}
    assignment_sources: dict[str, list[Any]] = {}
    unfinished_count = 0
    for assignment in assignments:
        parsed_assignment = parse_auxiliary_assignment(assignment, schema)
        sample_id, is_final = parsed_assignment.sample_id, parsed_assignment.is_final
        if sample_id not in legacy_index:
            raise ContractError(
                "AUXILIARY_SAMPLE_ID_NOT_FOUND",
                "assignment sample_id is absent from legacy labels",
                sample_id=sample_id,
            )
        if assignment.get("source") is not None:
            assignment_sources.setdefault(sample_id, []).append(assignment.get("source"))
        if not is_final:
            unfinished_count += 1
            continue
        head = parsed_assignment.head
        pair = (sample_id, head)
        if pair in final_by_pair:
            raise ContractError(
                "DUPLICATE_FINAL_AUXILIARY_ASSIGNMENT",
                "more than one final assignment exists for sample_id/head",
                sample_id=sample_id,
                head=head,
            )
        final_by_pair[pair] = assignment

    result: list[SemanticTrainingView] = []
    for sample_id, legacy_label in legacy_index.items():
        values: dict[str, Any] = {factor: None for factor in schema.auxiliary_factors}
        masks: dict[str, int] = {factor: 0 for factor in schema.auxiliary_factors}
        for factor in schema.auxiliary_factors:
            assignment = final_by_pair.get((sample_id, factor))
            if assignment is not None:
                answer = assignment["answer"]
                values[factor] = list(answer) if isinstance(answer, list) else answer
                masks[factor] = 1
        result.append(
            SemanticTrainingView(
                sample_id=sample_id,
                legacy_label=legacy_label,
                auxiliary=AuxiliaryFactorLabels(values=values, loss_mask=masks),
                source={
                    "legacy": {
                        "schema_version": schema.base_schema_version,
                        "record_type": "full_seven_head_label",
                        "source": legacy_label.get("source"),
                    },
                    "auxiliary": {
                        "schema_version": schema.schema_version,
                        "record_type": "sparse_app_assignment",
                        "final_assignment_count": sum(
                            int(key[0] == sample_id) for key in final_by_pair
                        ),
                        "unfinished_assignment_count": unfinished_count,
                        "sources": list(assignment_sources.get(sample_id, [])),
                    },
                },
            )
        )
    return result


# A descriptive alias for future loaders; both names use the same merge logic.
merge_auxiliary_assignments = merge_legacy_labels_and_auxiliary_assignments
load_training_view = merge_legacy_labels_and_auxiliary_assignments


def join_inputs_and_labels(
    inputs: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    *,
    require_complete: bool = True,
    auxiliary_schema: "SemanticSchemaCandidate | None" = None,
) -> list[JoinedRecord]:
    """Join by ``sample_id`` and optionally parse v0.3 sparse factors.

    The default remains byte-for-byte compatible with the v0.2.1 callers.  A
    caller that explicitly supplies ``auxiliary_schema`` gets an
    :class:`AuxiliaryFactorLabels` value on each joined row; omission of that
    argument does not change existing train/evaluation loaders.
    """
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
        if label.get("schema_version") == SCHEMA_VERSION and "auxiliary_factors" in label:
            raise ContractError(
                "V021_AUXILIARY_FORBIDDEN",
                "a frozen v0.2.1 seven-head record cannot carry auxiliary factors",
                sample_id=sample_id,
            )
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
    result: list[JoinedRecord] = []
    for sample_id in input_index:
        if sample_id not in label_index:
            continue
        label = label_index[sample_id]
        auxiliary = (
            parse_optional_auxiliary_factors(label, auxiliary_schema)
            if auxiliary_schema is not None
            else None
        )
        result.append(
            JoinedRecord(
                input=input_index[sample_id],
                label=label,
                auxiliary=auxiliary,
            )
        )
    return result
