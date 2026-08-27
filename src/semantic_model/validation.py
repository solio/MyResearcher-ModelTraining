from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .errors import ContractError
from .schema import CONFIDENCE_ORDER, SCHEMA_VERSION, SINGLE_LABEL_HEADS, LabelSchema

__all__ = [
    "ContractError",
    "validate_evidence_dependencies",
    "validate_label_record",
]


def validate_label_record(
    label: Mapping[str, Any],
    schema: LabelSchema,
    *,
    allow_anchor_reasoning_sentinel_combinations: bool = False,
) -> None:
    if label.get("schema_version") != SCHEMA_VERSION:
        raise ContractError(
            "SCHEMA_VERSION_MISMATCH",
            "label does not use the exact frozen schema version",
            sample_id=label.get("sample_id"),
            observed=label.get("schema_version"),
            expected=SCHEMA_VERSION,
        )
    for head in SINGLE_LABEL_HEADS:
        value = label.get(head)
        if value not in schema.class_order[head]:
            raise ContractError(
                "UNKNOWN_LABEL_VALUE",
                f"invalid value for {head}",
                sample_id=label.get("sample_id"),
                value=value,
            )
    tags = label.get("reasoning_tags")
    if not isinstance(tags, list):
        raise ContractError(
            "MULTI_LABEL_REQUIRED",
            "reasoning_tags must be a list",
            sample_id=label.get("sample_id"),
        )
    if not tags:
        raise ContractError(
            "REASONING_TAGS_EMPTY",
            "reasoning_tags must contain at least one label",
            sample_id=label.get("sample_id"),
        )
    schema.encode_reasoning_tags(tags)
    for exclusive_tag in ("NO_REASON_GIVEN", "UNKNOWN"):
        if (
            not allow_anchor_reasoning_sentinel_combinations
            and exclusive_tag in tags
            and len(tags) != 1
        ):
            raise ContractError(
                "LABEL_DEPENDENCY_VIOLATION",
                f"{exclusive_tag} must be the only reasoning tag",
                sample_id=label.get("sample_id"),
            )
    confidence = label.get("label_confidence")
    if confidence not in CONFIDENCE_ORDER:
        raise ContractError(
            "INVALID_LABEL_CONFIDENCE",
            "label_confidence is required for weighting/QA",
            sample_id=label.get("sample_id"),
            value=confidence,
        )
    if (
        label.get("emotion_primary") == "NONE_EXPLICIT"
        and label.get("emotion_target") != "NOT_APPLICABLE"
    ):
        raise ContractError(
            "LABEL_DEPENDENCY_VIOLATION",
            "NONE_EXPLICIT requires NOT_APPLICABLE emotion_target",
            sample_id=label.get("sample_id"),
        )
    if (
        label.get("emotion_target") == "NOT_APPLICABLE"
        and label.get("emotion_primary") != "NONE_EXPLICIT"
    ):
        raise ContractError(
            "LABEL_DEPENDENCY_VIOLATION",
            "NOT_APPLICABLE requires NONE_EXPLICIT emotion_primary",
            sample_id=label.get("sample_id"),
        )


def _evidence_list(evidence: Mapping[str, Any], field: str) -> list[str]:
    value = evidence.get(field, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ContractError(
            "EVIDENCE_SHAPE_INVALID", f"evidence_spans.{field} must be a string list"
        )
    return value


def _all_spans(evidence: Mapping[str, Any]) -> Iterable[tuple[str, str]]:
    for field, value in evidence.items():
        if field == "reasoning_tags":
            if not isinstance(value, Mapping):
                raise ContractError(
                    "EVIDENCE_SHAPE_INVALID",
                    "evidence_spans.reasoning_tags must be an object of string lists",
                )
            for tag, spans in value.items():
                if not isinstance(spans, list) or not all(
                    isinstance(span, str) for span in spans
                ):
                    raise ContractError(
                        "EVIDENCE_SHAPE_INVALID",
                        "reasoning evidence must be string lists",
                        tag=tag,
                    )
                for span in spans:
                    yield f"reasoning_tags.{tag}", span
        else:
            if not isinstance(value, list) or not all(
                isinstance(span, str) for span in value
            ):
                raise ContractError(
                    "EVIDENCE_SHAPE_INVALID",
                    f"evidence_spans.{field} must be a string list",
                )
            for span in value:
                yield field, span


def validate_evidence_dependencies(
    canonical_input: Mapping[str, Any], label: Mapping[str, Any]
) -> None:
    evidence = label.get("evidence_spans")
    if isinstance(evidence, list):
        _validate_native_evidence(canonical_input, label, evidence)
        return
    if not isinstance(evidence, Mapping):
        raise ContractError(
            "EVIDENCE_SHAPE_INVALID",
            "evidence_spans must be an object",
            sample_id=label.get("sample_id"),
        )
    sentinel_rules = {
        "target_mode": {"UNKNOWN"},
        "stance": {"UNKNOWN"},
        "emotion_primary": {"NONE_EXPLICIT", "UNKNOWN"},
        "emotion_target": {"NOT_APPLICABLE", "UNKNOWN"},
        "action_tendency": {"NO_ACTION_SIGNAL", "UNKNOWN"},
        "context_dependency": {"UNKNOWN"},
    }
    for field, forbidden_labels in sentinel_rules.items():
        spans = _evidence_list(evidence, field)
        if label.get(field) in forbidden_labels and spans:
            raise ContractError(
                "EVIDENCE_DEPENDENCY_VIOLATION",
                f"{field}={label.get(field)} forbids {field} evidence",
                sample_id=label.get("sample_id"),
                field=field,
            )
    reasoning_evidence = evidence.get("reasoning_tags", {})
    if not isinstance(reasoning_evidence, Mapping):
        raise ContractError(
            "EVIDENCE_SHAPE_INVALID", "reasoning_tags evidence must be an object"
        )
    for sentinel in ("NO_REASON_GIVEN", "UNKNOWN"):
        if sentinel in label.get("reasoning_tags", []) and reasoning_evidence.get(sentinel):
            raise ContractError(
                "EVIDENCE_DEPENDENCY_VIOLATION",
                f"{sentinel} forbids reasoning Evidence",
                sample_id=label.get("sample_id"),
                field="reasoning_tags",
            )
    model_text = canonical_input.get("model_text")
    if not isinstance(model_text, str):
        raise ContractError(
            "CANONICAL_MODEL_TEXT_INVALID",
            "canonical model_text must be a string",
            sample_id=canonical_input.get("sample_id"),
        )
    for field, span in _all_spans(evidence):
        if span and span not in model_text:
            raise ContractError(
                "EVIDENCE_NOT_SUBSTRING",
                "Evidence must be a substring of canonical model_text",
                sample_id=label.get("sample_id"),
                field=field,
                span=span,
            )


def _validate_native_evidence(
    canonical_input: Mapping[str, Any],
    label: Mapping[str, Any],
    evidence: list[Any],
) -> None:
    """Validate the frozen upstream v0.3.4 Evidence object representation."""

    model_text = canonical_input.get("model_text")
    if not isinstance(model_text, str):
        raise ContractError(
            "CANONICAL_MODEL_TEXT_INVALID",
            "canonical model_text must be a string",
            sample_id=canonical_input.get("sample_id"),
        )
    forbidden = {"UNKNOWN", "NONE_EXPLICIT", "NO_ACTION_SIGNAL", "NO_REASON_GIVEN"}
    valid_fields = {*SINGLE_LABEL_HEADS, "reasoning_tags"}
    for index, item in enumerate(evidence):
        if not isinstance(item, Mapping) or set(item) != {"field", "label", "span"}:
            raise ContractError(
                "EVIDENCE_SHAPE_INVALID",
                "native Evidence items require exactly field, label, and span",
                sample_id=label.get("sample_id"),
                index=index,
            )
        field = item.get("field")
        evidence_label = item.get("label")
        span = item.get("span")
        if field not in valid_fields or not all(
            isinstance(value, str) and value for value in (evidence_label, span)
        ):
            raise ContractError(
                "EVIDENCE_SHAPE_INVALID",
                "native Evidence values must be non-empty strings",
                sample_id=label.get("sample_id"),
                index=index,
            )
        if evidence_label in forbidden:
            raise ContractError(
                "EVIDENCE_DEPENDENCY_VIOLATION",
                "sentinel labels forbid Evidence objects",
                sample_id=label.get("sample_id"),
                field=field,
                label=evidence_label,
            )
        expected = label.get(str(field))
        matches_label = (
            evidence_label in expected
            if field == "reasoning_tags" and isinstance(expected, list)
            else evidence_label == expected
        )
        if not matches_label:
            raise ContractError(
                "EVIDENCE_DEPENDENCY_VIOLATION",
                "Evidence label disagrees with its semantic field",
                sample_id=label.get("sample_id"),
                field=field,
                label=evidence_label,
            )
        if span not in model_text:
            raise ContractError(
                "EVIDENCE_NOT_SUBSTRING",
                "Evidence must be a substring of canonical model_text",
                sample_id=label.get("sample_id"),
                field=field,
                span=span,
            )
