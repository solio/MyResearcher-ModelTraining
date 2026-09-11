from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

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

# v0.2.1 remains the frozen prediction contract.  These constants describe the
# additive, optional v0.3 proposal and intentionally do not alter the frozen
# class order above.
V03_SCHEMA_VERSION = "semantic-schema-candidate-v0.3"
V03_BASE_SCHEMA_VERSION = SCHEMA_VERSION
V03_AUXILIARY_CLASS_ORDER: dict[str, list[str]] = {
    "expectation_phase": ["ANTICIPATED", "ONGOING_OR_REALIZED", "UNKNOWN"],
    "arousal_level": ["LOW", "MEDIUM", "HIGH", "UNKNOWN"],
    "desired_price_direction": [
        "UP",
        "DOWN",
        "RANGE_OR_STABLE",
        "NON_DIRECTIONAL",
        "UNKNOWN",
    ],
    "market_scope": [
        "TRADING_FLOW",
        "INDUSTRY_PRODUCT_REGION",
        "BROAD_MARKET",
        "UNKNOWN",
    ],
    "discourse_focus": [
        "ASSET_OR_COMPANY",
        "AUTHOR_VIEWPOINT",
        "AUTHOR_ACTION_OR_POSITION",
        "OTHER_PERSON_OR_AUDIENCE",
        "CLAIM_OR_NARRATIVE",
        "MARKET_OR_INDUSTRY",
        "UNKNOWN",
    ],
}
V03_AUXILIARY_TASK_TYPES = {
    "expectation_phase": "single_label",
    "arousal_level": "ordered_single_label",
    "desired_price_direction": "single_label",
    "market_scope": "multi_label",
    "discourse_focus": "multi_label",
}
V03_AUXILIARY_FACTORS = tuple(V03_AUXILIARY_CLASS_ORDER)
V03_METADATA_FIELDS = (
    "name_zh",
    "definition",
    "select_when",
    "not_select_when",
    "confusable_with",
    "examples",
    "logical_owner_head",
    "applicable_when",
)
V03_AROUSAL_ORDERED_VALUES = ["LOW", "MEDIUM", "HIGH"]
ACTION_PARENT_CLASS_ORDER = [
    "INCREASE_EXPOSURE",
    "MAINTAIN_EXPOSURE",
    "ACTIVE_REBALANCE",
    "DECREASE_EXPOSURE",
    "NO_EXECUTION",
    "UNKNOWN",
]
ACTION_PARENT_CHILDREN: dict[str, list[str]] = {
    "INCREASE_EXPOSURE": ["BUY", "ADD"],
    "MAINTAIN_EXPOSURE": ["HOLD"],
    "ACTIVE_REBALANCE": ["DO_T"],
    "DECREASE_EXPOSURE": ["REDUCE", "SELL"],
    "NO_EXECUTION": ["WATCH", "NO_ACTION_SIGNAL"],
    "UNKNOWN": ["UNKNOWN"],
}
_ACTION_CHILD_TO_PARENT = {
    child: parent
    for parent, children in ACTION_PARENT_CHILDREN.items()
    for child in children
}


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
        compact_fields = raw.get("fields")
        if isinstance(heads, dict):
            actual = {
                head: heads.get(head, {}).get("class_order")
                for head in V1_HEADS
            }
        elif isinstance(compact_fields, dict):
            actual = {head: compact_fields.get(head) for head in V1_HEADS}
        else:
            raise ContractError(
                "SCHEMA_INVALID",
                "schema requires v1_prediction_heads or compact fields",
            )
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


@dataclass(frozen=True)
class AuxiliaryFactorSpec:
    """Machine-readable definition of one optional v0.3 factor."""

    name: str
    task_type: str
    class_order: tuple[str, ...]
    optional: bool = True
    allow_multiple: bool = False
    unknown_is_exclusive: bool = False


@dataclass(frozen=True)
class SemanticSchemaCandidate:
    """The additive v0.3 schema candidate.

    Loading this candidate never changes :class:`LabelSchema` or any v0.2.1
    record.  It only supplies validation and loss-mask semantics for optional
    auxiliary factors.
    """

    path: Path
    schema_version: str
    base_schema_version: str
    class_order: Mapping[str, list[str]]
    auxiliary_factors: Mapping[str, AuxiliaryFactorSpec]
    action_parent_class_order: tuple[str, ...]
    action_parent_children: Mapping[str, tuple[str, ...]]
    raw: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "SemanticSchemaCandidate":
        schema_path = Path(path).resolve()
        try:
            raw = json.loads(schema_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ContractError("SCHEMA_NOT_FOUND", str(schema_path)) from exc
        except json.JSONDecodeError as exc:
            raise ContractError("SCHEMA_INVALID", str(exc), path=str(schema_path)) from exc
        if not isinstance(raw, Mapping):
            raise ContractError("SCHEMA_INVALID", "candidate schema must be an object")
        if raw.get("schema_version") != V03_SCHEMA_VERSION:
            raise ContractError(
                "SCHEMA_VERSION_MISMATCH",
                "candidate schema has an unexpected version",
                observed=raw.get("schema_version"),
                expected=V03_SCHEMA_VERSION,
            )
        if raw.get("base_schema_version") != V03_BASE_SCHEMA_VERSION:
            raise ContractError(
                "SCHEMA_BASE_VERSION_MISMATCH",
                "candidate must explicitly extend frozen v0.2.1",
                observed=raw.get("base_schema_version"),
                expected=V03_BASE_SCHEMA_VERSION,
            )

        heads = raw.get("v1_prediction_heads")
        if not isinstance(heads, Mapping) or set(heads) != set(V1_HEADS):
            raise ContractError(
                "SCHEMA_HEAD_SET_MISMATCH",
                "candidate must preserve exactly the seven v0.2.1 heads",
                observed=sorted(heads) if isinstance(heads, Mapping) else None,
            )
        actual: dict[str, list[str]] = {}
        for head in V1_HEADS:
            definition = heads.get(head)
            if not isinstance(definition, Mapping):
                raise ContractError("SCHEMA_INVALID", "head definition must be an object", head=head)
            order = definition.get("class_order")
            if not isinstance(order, list) or not all(isinstance(item, str) for item in order):
                raise ContractError("SCHEMA_INVALID", "head class_order must be a string list", head=head)
            actual[head] = list(order)
            if definition.get("task_type") not in {"single_label", "multi_label"}:
                raise ContractError("SCHEMA_INVALID", "invalid v0.2.1 head task_type", head=head)
        if actual != FROZEN_CLASS_ORDER:
            raise ContractError(
                "SCHEMA_CLASS_ORDER_MISMATCH",
                "candidate changed the frozen v0.2.1 class order",
                actual=actual,
                expected=FROZEN_CLASS_ORDER,
            )

        raw_aux = raw.get("auxiliary_factors")
        if not isinstance(raw_aux, Mapping) or set(raw_aux) != set(V03_AUXILIARY_FACTORS):
            raise ContractError(
                "AUXILIARY_FACTOR_SET_MISMATCH",
                "candidate must define exactly the five optional auxiliary factors",
                observed=sorted(raw_aux) if isinstance(raw_aux, Mapping) else None,
            )
        auxiliary: dict[str, AuxiliaryFactorSpec] = {}
        for name in V03_AUXILIARY_FACTORS:
            definition = raw_aux.get(name)
            if not isinstance(definition, Mapping):
                raise ContractError("SCHEMA_INVALID", "auxiliary factor must be an object", factor=name)
            expected_task = V03_AUXILIARY_TASK_TYPES[name]
            order = definition.get("class_order")
            if definition.get("task_type") != expected_task or order != V03_AUXILIARY_CLASS_ORDER[name]:
                raise ContractError(
                    "AUXILIARY_FACTOR_DEFINITION_MISMATCH",
                    "auxiliary factor definition differs from the v0.3 contract",
                    factor=name,
                    expected_task_type=expected_task,
                    expected_class_order=V03_AUXILIARY_CLASS_ORDER[name],
                )
            if definition.get("optional") is not True:
                raise ContractError("AUXILIARY_FACTOR_REQUIRED", "v0.3 auxiliary factors are optional", factor=name)
            for field in V03_METADATA_FIELDS:
                if field not in definition:
                    raise ContractError("AUXILIARY_METADATA_INVALID", "factor metadata is incomplete", factor=name, field=field)
            if not isinstance(definition["name_zh"], str) or not isinstance(definition["definition"], str) or not isinstance(definition["select_when"], str) or not isinstance(definition["not_select_when"], str) or not isinstance(definition["logical_owner_head"], str) or not isinstance(definition["applicable_when"], str):
                raise ContractError("AUXILIARY_METADATA_INVALID", "factor metadata text fields must be strings", factor=name)
            if definition["logical_owner_head"] not in V1_HEADS:
                raise ContractError("AUXILIARY_METADATA_INVALID", "logical owner must be an existing top-level head", factor=name)
            if not isinstance(definition["confusable_with"], list) or not isinstance(definition["examples"], list):
                raise ContractError("AUXILIARY_METADATA_INVALID", "factor metadata lists are invalid", factor=name)
            label_metadata = definition.get("label_metadata")
            if not isinstance(label_metadata, Mapping) or set(label_metadata) != set(order):
                raise ContractError("AUXILIARY_METADATA_INVALID", "metadata is required for every auxiliary label", factor=name)
            for label, metadata in label_metadata.items():
                if not isinstance(metadata, Mapping) or any(field not in metadata for field in V03_METADATA_FIELDS):
                    raise ContractError("AUXILIARY_METADATA_INVALID", "label metadata is incomplete", factor=name, label=label)
            if name == "arousal_level" and definition.get("ordered_values") != V03_AROUSAL_ORDERED_VALUES:
                raise ContractError("AUXILIARY_ORDER_INVALID", "arousal ordered values must exclude UNKNOWN", factor=name)
            if definition.get("missing_semantics") != "unannotated_or_not_applicable" or definition.get("unknown_semantics") != "factor_present_but_not_determinable":
                raise ContractError("AUXILIARY_SEMANTICS_INVALID", "missing and UNKNOWN semantics are not explicit", factor=name)
            if definition.get("missing_loss_mask") != 0 or definition.get("unknown_loss_mask") != 1:
                raise ContractError("AUXILIARY_LOSS_MASK_INVALID", "missing/UNKNOWN masks must be 0/1", factor=name)
            multiple = expected_task == "multi_label"
            if bool(definition.get("allow_multiple", False)) != multiple:
                raise ContractError("AUXILIARY_FACTOR_CARDINALITY_INVALID", "multi-label cardinality mismatch", factor=name)
            if multiple and definition.get("unknown_is_exclusive") is not True:
                raise ContractError("AUXILIARY_UNKNOWN_POLICY_INVALID", "UNKNOWN must be exclusive for multi-label factors", factor=name)
            auxiliary[name] = AuxiliaryFactorSpec(
                name=name,
                task_type=expected_task,
                class_order=tuple(order),
                allow_multiple=multiple,
                unknown_is_exclusive=bool(definition.get("unknown_is_exclusive", False)),
            )

        hierarchy = raw.get("action_hierarchy")
        if not isinstance(hierarchy, Mapping) or hierarchy.get("parent_class_order") != ACTION_PARENT_CLASS_ORDER:
            raise ContractError("ACTION_HIERARCHY_INVALID", "action parent class order is not frozen")
        raw_children = hierarchy.get("children")
        if not isinstance(raw_children, Mapping) or set(raw_children) != set(ACTION_PARENT_CHILDREN):
            raise ContractError("ACTION_HIERARCHY_INVALID", "action parent children are incomplete")
        children: dict[str, tuple[str, ...]] = {}
        for parent in ACTION_PARENT_CLASS_ORDER:
            value = raw_children.get(parent)
            if value != ACTION_PARENT_CHILDREN[parent]:
                raise ContractError(
                    "ACTION_HIERARCHY_INVALID",
                    "action child mapping differs from the frozen contract",
                    parent=parent,
                )
            children[parent] = tuple(value)
        return cls(
            path=schema_path,
            schema_version=V03_SCHEMA_VERSION,
            base_schema_version=V03_BASE_SCHEMA_VERSION,
            class_order=actual,
            auxiliary_factors=auxiliary,
            action_parent_class_order=tuple(ACTION_PARENT_CLASS_ORDER),
            action_parent_children=children,
            raw=raw,
        )

    def parse_auxiliary_factors(self, record: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
        """Return ``(values, loss_mask)`` for a sparse v0.3 or legacy row.

        An omitted factor is represented as ``None`` with mask ``0``.  An
        explicit ``UNKNOWN`` is retained and receives mask ``1``.  ``null`` is
        rejected so missing and UNKNOWN cannot silently collapse into one state.
        """

        if not isinstance(record, Mapping):
            raise ContractError("RECORD_INVALID", "semantic record must be an object")
        if record.get("schema_version") == self.base_schema_version and "auxiliary_factors" in record:
            raise ContractError(
                "V021_AUXILIARY_FORBIDDEN",
                "a frozen v0.2.1 seven-head record cannot carry auxiliary factors",
                sample_id=record.get("sample_id"),
            )
        container = record.get("auxiliary_factors", {})
        if container is None or not isinstance(container, Mapping):
            raise ContractError("AUXILIARY_FACTORS_INVALID", "auxiliary_factors must be an object when present")
        extra = set(container) - set(self.auxiliary_factors)
        if extra:
            raise ContractError("UNKNOWN_AUXILIARY_FACTOR", "record contains an undeclared auxiliary factor", factors=sorted(extra))
        values: dict[str, Any] = {}
        masks: dict[str, int] = {}
        for name, spec in self.auxiliary_factors.items():
            if name not in container:
                values[name] = None
                masks[name] = 0
                continue
            value = container[name]
            if value is None:
                raise ContractError("AUXILIARY_FACTOR_NULL", "omit an unannotated factor instead of using null", factor=name)
            if spec.allow_multiple:
                if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
                    raise ContractError("AUXILIARY_MULTI_LABEL_INVALID", "multi-label factor requires a non-empty string list", factor=name)
                if len(set(value)) != len(value):
                    raise ContractError("AUXILIARY_DUPLICATE_LABEL", "multi-label factor contains duplicates", factor=name)
                invalid = sorted(set(value) - set(spec.class_order))
                if invalid:
                    raise ContractError("UNKNOWN_LABEL_VALUE", "auxiliary factor contains an unknown value", factor=name, values=invalid)
                if spec.unknown_is_exclusive and "UNKNOWN" in value and len(value) != 1:
                    raise ContractError("AUXILIARY_UNKNOWN_COMBINATION", "UNKNOWN cannot be combined with a concrete value", factor=name)
                values[name] = list(value)
            else:
                if not isinstance(value, str) or value not in spec.class_order:
                    raise ContractError("UNKNOWN_LABEL_VALUE", "auxiliary factor contains an unknown value", factor=name, value=value)
                values[name] = value
            masks[name] = 1
        return values, masks

    def validate_auxiliary_assignment(self, assignment: Mapping[str, Any]) -> tuple[str, Any, bool]:
        """Validate one app-exported ``sample_id/head/answer`` assignment.

        App exports are intentionally one factor per row and do not copy the
        seven legacy heads.  Unfinished rows are shape-checked but their answer
        is ignored by the merge loader, so an absent/in-progress answer cannot
        contribute to loss.
        """

        required = {"sample_id", "head", "answer", "is_final", "schema_version"}
        if not isinstance(assignment, Mapping) or not required.issubset(assignment):
            raise ContractError(
                "AUXILIARY_ASSIGNMENT_SHAPE_INVALID",
                "assignment requires sample_id, head, answer, is_final, schema_version",
            )
        legacy_fields = set(assignment) & set(V1_HEADS)
        if legacy_fields:
            raise ContractError(
                "AUXILIARY_ASSIGNMENT_LEGACY_HEAD_FORBIDDEN",
                "sparse assignment must not carry copied v0.2.1 heads",
                fields=sorted(legacy_fields),
            )
        sample_id = assignment.get("sample_id")
        head = assignment.get("head")
        is_final = assignment.get("is_final")
        if not isinstance(sample_id, str) or not sample_id:
            raise ContractError("MISSING_SAMPLE_ID", "auxiliary assignment requires a non-empty sample_id")
        if not isinstance(head, str) or head not in self.auxiliary_factors:
            raise ContractError(
                "AUXILIARY_ASSIGNMENT_HEAD_INVALID",
                "assignment head must be one of the five auxiliary factors",
                head=head,
            )
        if assignment.get("schema_version") != self.schema_version:
            raise ContractError(
                "SCHEMA_VERSION_MISMATCH",
                "auxiliary assignment must use candidate v0.3",
                observed=assignment.get("schema_version"),
                expected=self.schema_version,
            )
        if not isinstance(is_final, bool):
            raise ContractError("AUXILIARY_ASSIGNMENT_FINAL_INVALID", "is_final must be boolean")
        answer = assignment.get("answer")
        if is_final:
            # Reuse the factor cardinality/value validator.  It also rejects
            # duplicate multi-label answers and undeclared labels.
            self.parse_auxiliary_factors(
                {"schema_version": self.schema_version, "auxiliary_factors": {head: answer}}
            )
        return sample_id, answer, is_final

    def encode_auxiliary_factors(
        self, record: Mapping[str, Any]
    ) -> tuple[dict[str, list[int]], dict[str, int]]:
        """Encode sparse factors as one-/multi-hot vectors plus factor masks."""

        values, masks = self.parse_auxiliary_factors(record)
        encoded: dict[str, list[int]] = {}
        for name, spec in self.auxiliary_factors.items():
            value = values[name]
            if value is None:
                encoded[name] = [0] * len(spec.class_order)
            elif spec.allow_multiple:
                selected = set(value)
                encoded[name] = [int(label in selected) for label in spec.class_order]
            else:
                encoded[name] = [int(label == value) for label in spec.class_order]
        return encoded, masks

    def validate_record(
        self,
        record: Mapping[str, Any],
        *,
        allow_anchor_reasoning_sentinel_combinations: bool = False,
    ) -> tuple[dict[str, Any], dict[str, int]]:
        """Validate preserved v0.2.1 heads and parse optional v0.3 factors."""

        if record.get("schema_version") not in {self.schema_version, self.base_schema_version}:
            raise ContractError(
                "SCHEMA_VERSION_MISMATCH",
                "record must use candidate v0.3 or frozen v0.2.1",
                observed=record.get("schema_version"),
            )
        for head in SINGLE_LABEL_HEADS:
            if record.get(head) not in self.class_order[head]:
                raise ContractError("UNKNOWN_LABEL_VALUE", "invalid preserved head value", head=head, value=record.get(head))
        tags = record.get("reasoning_tags")
        if (
            not isinstance(tags, list)
            or not tags
            or not all(isinstance(tag, str) for tag in tags)
            or len(set(tags)) != len(tags)
        ):
            raise ContractError("MULTI_LABEL_REQUIRED", "reasoning_tags must be a non-empty unique list")
        invalid = sorted(set(tags) - set(self.class_order["reasoning_tags"]))
        if invalid:
            raise ContractError("UNKNOWN_LABEL_VALUE", "invalid reasoning tag", values=invalid)
        if (
            not allow_anchor_reasoning_sentinel_combinations
            and any(sentinel in tags for sentinel in ("NO_REASON_GIVEN", "UNKNOWN"))
            and len(tags) != 1
        ):
            raise ContractError("LABEL_DEPENDENCY_VIOLATION", "reasoning sentinel must be exclusive")
        confidence = record.get("label_confidence")
        if confidence not in CONFIDENCE_ORDER:
            raise ContractError(
                "INVALID_LABEL_CONFIDENCE",
                "label_confidence is required for weighting/QA",
                sample_id=record.get("sample_id"),
                value=confidence,
            )
        if record.get("emotion_primary") == "NONE_EXPLICIT" and record.get("emotion_target") != "NOT_APPLICABLE":
            raise ContractError("LABEL_DEPENDENCY_VIOLATION", "NONE_EXPLICIT requires NOT_APPLICABLE emotion_target", sample_id=record.get("sample_id"))
        if record.get("emotion_target") == "NOT_APPLICABLE" and record.get("emotion_primary") != "NONE_EXPLICIT":
            raise ContractError("LABEL_DEPENDENCY_VIOLATION", "NOT_APPLICABLE requires NONE_EXPLICIT emotion_primary", sample_id=record.get("sample_id"))
        return self.parse_auxiliary_factors(record)


# Short loader alias keeps call sites explicit and makes the candidate usable
# without changing the frozen ``LabelSchema.load`` API.
load_candidate_schema = SemanticSchemaCandidate.load


def action_parent_from_label(label: str) -> str:
    """Map one frozen action child label to its v0.3 parent."""

    try:
        return _ACTION_CHILD_TO_PARENT[label]
    except KeyError as exc:
        raise ContractError("UNKNOWN_LABEL_VALUE", "invalid action_tendency child", value=label) from exc


def action_parent_probabilities(
    probabilities: Mapping[str, float] | Sequence[float],
    *,
    child_order: Sequence[str] | None = None,
) -> dict[str, float]:
    """Aggregate child probabilities into the six action parent classes.

    Summation is intentionally used instead of renormalizing: if child
    probabilities sum to one, parent probabilities also sum to one and retain
    the exact mass represented by each frozen v0.2.1 child.
    """

    order = list(child_order or FROZEN_CLASS_ORDER["action_tendency"])
    if order != FROZEN_CLASS_ORDER["action_tendency"]:
        raise ContractError("ACTION_CLASS_ORDER_MISMATCH", "action probabilities require frozen child order")
    if isinstance(probabilities, Mapping):
        if set(probabilities) != set(order):
            raise ContractError("ACTION_PROBABILITY_SHAPE_INVALID", "mapping must contain every action child")
        values = [probabilities[label] for label in order]
    else:
        try:
            values = list(probabilities)
        except TypeError as exc:
            raise ContractError("ACTION_PROBABILITY_SHAPE_INVALID", "probabilities must be a mapping or sequence") from exc
        if len(values) != len(order):
            raise ContractError("ACTION_PROBABILITY_SHAPE_INVALID", "probability vector length differs from action class order")
    numeric: dict[str, float] = {}
    for label, value in zip(order, values, strict=True):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
            raise ContractError("ACTION_PROBABILITY_INVALID", "action probability must be finite and non-negative", label=label)
        numeric[label] = float(value)
    return {
        parent: sum(numeric[child] for child in children)
        for parent, children in ACTION_PARENT_CHILDREN.items()
    }


def action_parent_labels(labels: Sequence[str]) -> list[str]:
    """Map a sequence of child labels for parent-level metrics."""

    return [action_parent_from_label(label) for label in labels]


def action_parent_metric_labels(
    truth: Sequence[str], prediction: Sequence[str]
) -> tuple[list[str], list[str]]:
    """Map truth and prediction together before computing parent metrics."""

    if len(truth) != len(prediction):
        raise ContractError(
            "ACTION_METRIC_LENGTH_MISMATCH",
            "truth and prediction must contain the same number of rows",
            truth=len(truth),
            prediction=len(prediction),
        )
    return action_parent_labels(truth), action_parent_labels(prediction)


# Explicit aliases make the intended aggregation API easy to find while
# retaining one implementation and one frozen child-to-parent mapping.
aggregate_action_parent_probabilities = action_parent_probabilities
map_action_parent_labels = action_parent_labels
