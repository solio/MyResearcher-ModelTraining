import pytest

from semantic_model.split import validate_split_manifest
from semantic_model.validation import ContractError


def records(counts):
    result = []
    index = 0
    for split, count in counts.items():
        for _ in range(count):
            index += 1
            result.append(
                {
                    "sample_id": f"P{index:04d}",
                    "split": split,
                    "duplicate_cluster_id": f"D{index:04d}",
                    "event_group": f"E{index:04d}",
                }
            )
    return result


def test_exact_split_counts_pass():
    assignments = records({"train": 4, "dev": 2, "test": 2, "embargo": 1})
    validate_split_manifest(
        {"assignments": assignments},
        trainable_ids={row["sample_id"] for row in assignments},
        quarantine_ids=set(),
        expected_counts={"train": 4, "dev": 2, "test": 2, "embargo": 1},
    )


def test_frozen_v0_3_5_counts_total_2979():
    expected = {"train": 1822, "dev": 448, "test": 467, "embargo": 242}
    assignments = records(expected)
    assert len(assignments) == 2979
    validate_split_manifest(
        {"assignments": assignments},
        trainable_ids={row["sample_id"] for row in assignments},
        quarantine_ids=set(),
        expected_counts=expected,
    )



def test_wrong_split_count_blocks():
    assignments = records({"train": 3, "dev": 2, "test": 2, "embargo": 1})
    with pytest.raises(ContractError, match="SPLIT_COUNT_MISMATCH"):
        validate_split_manifest(
            {"assignments": assignments},
            trainable_ids={row["sample_id"] for row in assignments},
            quarantine_ids=set(),
            expected_counts={"train": 4, "dev": 2, "test": 2, "embargo": 1},
        )


def test_duplicate_assignment_blocks():
    assignments = records({"train": 1, "dev": 1, "test": 1, "embargo": 1})
    assignments.append({**assignments[0], "split": "dev"})
    with pytest.raises(ContractError, match="SPLIT_IDENTITY_LEAKAGE"):
        validate_split_manifest(
            {"assignments": assignments},
            trainable_ids={row["sample_id"] for row in assignments},
            quarantine_ids=set(),
            expected_counts={"train": 1, "dev": 2, "test": 1, "embargo": 1},
        )


def test_quarantine_cannot_enter_split():
    assignments = records({"train": 1, "dev": 1, "test": 1, "embargo": 1})
    with pytest.raises(ContractError, match="QUARANTINE_SPLIT_LEAKAGE"):
        validate_split_manifest(
            {"assignments": assignments},
            trainable_ids={row["sample_id"] for row in assignments},
            quarantine_ids={assignments[0]["sample_id"]},
            expected_counts={"train": 1, "dev": 1, "test": 1, "embargo": 1},
        )


def test_duplicate_or_event_group_cannot_cross_splits():
    assignments = records({"train": 1, "dev": 1, "test": 1, "embargo": 1})
    assignments[1]["duplicate_cluster_id"] = assignments[0]["duplicate_cluster_id"]
    with pytest.raises(ContractError, match="SPLIT_GROUP_LEAKAGE"):
        validate_split_manifest(
            {"assignments": assignments},
            trainable_ids={row["sample_id"] for row in assignments},
            quarantine_ids=set(),
            expected_counts={"train": 1, "dev": 1, "test": 1, "embargo": 1},
        )


def test_anchor_and_gold_must_be_disjoint():
    assignments = records({"train": 1, "dev": 1, "test": 1, "embargo": 1})
    with pytest.raises(ContractError, match="SPLIT_IDENTITY_LEAKAGE"):
        validate_split_manifest(
            {"assignments": assignments},
            trainable_ids={row["sample_id"] for row in assignments},
            quarantine_ids=set(),
            expected_counts={"train": 1, "dev": 1, "test": 1, "embargo": 1},
            anchor_ids={assignments[2]["sample_id"]},
            gold_ids=set(),
        )


def test_missing_group_keys_fail_closed():
    assignment = {
        "sample_id": "P0001",
        "split": "train",
        "duplicate_cluster_id": "D0001",
    }
    with pytest.raises(ContractError, match="SPLIT_GROUP_KEY_MISSING"):
        validate_split_manifest(
            {"assignments": [assignment]},
            trainable_ids={"P0001"},
            quarantine_ids=set(),
            expected_counts={"train": 1, "dev": 0, "test": 0, "embargo": 0},
        )


def test_split_declares_canonical_time_source():
    assignment = records({"train": 1})[0]
    canonical = {
        assignment["sample_id"]: {
            "sample_id": assignment["sample_id"],
            "published_at": "2026-07-13T16:29:31+08:00",
        }
    }
    with pytest.raises(ContractError, match="SPLIT_TIME_SOURCE_INVALID"):
        validate_split_manifest(
            {"assignments": [assignment]},
            trainable_ids={assignment["sample_id"]},
            quarantine_ids=set(),
            expected_counts={"train": 1, "dev": 0, "test": 0, "embargo": 0},
            canonical_inputs=canonical,
        )
