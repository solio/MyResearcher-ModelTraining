from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import Any

from .errors import ContractError


SPLITS = ("train", "dev", "test", "embargo")


def validate_split_manifest(
    manifest: Mapping[str, Any],
    *,
    trainable_ids: set[str],
    quarantine_ids: set[str],
    expected_counts: Mapping[str, int],
    anchor_ids: set[str] | None = None,
    gold_ids: set[str] | None = None,
    canonical_inputs: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, list[str]]:
    assignments = manifest.get("assignments")
    if not isinstance(assignments, list):
        raise ContractError("SPLIT_MANIFEST_INVALID", "assignments must be a list")
    seen: dict[str, str] = {}
    by_split: dict[str, list[str]] = {name: [] for name in SPLITS}
    group_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    if canonical_inputs is not None and manifest.get("time_source") != "canonical_input.published_at":
        raise ContractError(
            "SPLIT_TIME_SOURCE_INVALID",
            "split manifest must declare canonical_input.published_at as its time source",
            observed=manifest.get("time_source"),
        )
    for row_number, assignment in enumerate(assignments, 1):
        if not isinstance(assignment, Mapping):
            raise ContractError(
                "SPLIT_MANIFEST_INVALID", "assignment must be an object", row=row_number
            )
        sample_id = assignment.get("sample_id")
        split = assignment.get("split")
        if not isinstance(sample_id, str) or not sample_id:
            raise ContractError("SPLIT_MANIFEST_INVALID", "sample_id is required")
        if split not in SPLITS:
            raise ContractError(
                "SPLIT_MANIFEST_INVALID", "unknown split", sample_id=sample_id, split=split
            )
        missing_group_fields = [
            field for field in ("duplicate_cluster_id", "event_group") if field not in assignment
        ]
        if missing_group_fields:
            raise ContractError(
                "SPLIT_GROUP_KEY_MISSING",
                "split assignment must preserve duplicate and event group keys",
                sample_id=sample_id,
                fields=missing_group_fields,
            )
        if canonical_inputs is not None:
            canonical = canonical_inputs.get(sample_id)
            if canonical is None:
                raise ContractError(
                    "SPLIT_SAMPLE_ID_NOT_FOUND",
                    "split identity is absent from canonical input",
                    sample_id=sample_id,
                )
            repeated_time = assignment.get("published_at")
            if repeated_time is not None and repeated_time != canonical.get("published_at"):
                raise ContractError(
                    "CANONICAL_METADATA_MISMATCH",
                    "split repeated time disagrees with canonical input",
                    sample_id=sample_id,
                    canonical=canonical.get("published_at"),
                    split_manifest=repeated_time,
                )
        if sample_id in seen:
            raise ContractError(
                "SPLIT_IDENTITY_LEAKAGE",
                "sample_id appears more than once in split manifest",
                sample_id=sample_id,
                splits=[seen[sample_id], split],
            )
        seen[sample_id] = split
        by_split[split].append(sample_id)
        for field in ("duplicate_cluster_id", "echo_group_id", "event_group"):
            group = assignment.get(field)
            if group not in (None, ""):
                group_splits[(field, str(group))].add(split)
    overlap = sorted(set(seen) & quarantine_ids)
    if overlap:
        raise ContractError(
            "QUARANTINE_SPLIT_LEAKAGE",
            "quarantine identity appears in a trainable/embargo split",
            sample_ids=overlap,
        )
    if set(seen) != trainable_ids:
        raise ContractError(
            "SPLIT_COVERAGE_MISMATCH",
            "split identities must exactly match trainable identities",
            missing=sorted(trainable_ids - set(seen)),
            extra=sorted(set(seen) - trainable_ids),
        )
    actual_counts = Counter(seen.values())
    normalized_expected = {name: int(expected_counts.get(name, -1)) for name in SPLITS}
    normalized_actual = {name: actual_counts.get(name, 0) for name in SPLITS}
    if normalized_actual != normalized_expected:
        raise ContractError(
            "SPLIT_COUNT_MISMATCH",
            "split counts differ from the frozen contract",
            actual=normalized_actual,
            expected=normalized_expected,
        )
    leaks = {
        f"{field}:{group}": sorted(splits)
        for (field, group), splits in group_splits.items()
        if len(splits) > 1
    }
    if leaks:
        raise ContractError(
            "SPLIT_GROUP_LEAKAGE",
            "duplicate/event group crosses split boundaries",
            groups=leaks,
        )
    anchor_ids = anchor_ids or set()
    gold_ids = gold_ids or set()
    cross_role = (set(seen) & anchor_ids) | (set(seen) & gold_ids) | (
        anchor_ids & gold_ids
    )
    if cross_role:
        raise ContractError(
            "SPLIT_IDENTITY_LEAKAGE",
            "split, Anchor, and Gold identities must be disjoint",
            sample_ids=sorted(cross_role),
        )
    return by_split
