from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .data import index_by_sample_id
from .errors import ContractError
from .schema import V1_HEADS


def validate_field_weights(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_ids: set[str],
    quarantine_ids: set[str] | None = None,
) -> dict[str, dict[str, float]]:
    quarantine_ids = quarantine_ids or set()
    indexed = index_by_sample_id(records, role="field-weight")
    if set(indexed) != expected_ids:
        raise ContractError(
            "FIELD_WEIGHT_CONTRACT_VIOLATION",
            "field-weight identities must exactly match expected identities",
            missing=sorted(expected_ids - set(indexed)),
            extra=sorted(set(indexed) - expected_ids),
        )
    result: dict[str, dict[str, float]] = {}
    for sample_id, record in indexed.items():
        if "sample_weight" in record:
            raise ContractError(
                "FIELD_WEIGHT_CONTRACT_VIOLATION",
                "a global sample_weight cannot replace per-head weights",
                sample_id=sample_id,
            )
        weights = record.get("weights")
        if not isinstance(weights, Mapping) or set(weights) != set(V1_HEADS):
            raise ContractError(
                "FIELD_WEIGHT_CONTRACT_VIOLATION",
                "weights must contain exactly the seven V1 heads",
                sample_id=sample_id,
                observed=sorted(weights) if isinstance(weights, Mapping) else None,
            )
        normalized: dict[str, float] = {}
        for head in V1_HEADS:
            value = weights[head]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ContractError(
                    "FIELD_WEIGHT_CONTRACT_VIOLATION",
                    "head weight must be numeric",
                    sample_id=sample_id,
                    head=head,
                )
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0:
                raise ContractError(
                    "FIELD_WEIGHT_CONTRACT_VIOLATION",
                    "head weight must be finite and non-negative",
                    sample_id=sample_id,
                    head=head,
                    value=numeric,
                )
            normalized[head] = numeric
        if sample_id in quarantine_ids and any(normalized.values()):
            raise ContractError(
                "FIELD_WEIGHT_CONTRACT_VIOLATION",
                "quarantine identities must have zero weight for every head",
                sample_id=sample_id,
            )
        result[sample_id] = normalized
    return result

