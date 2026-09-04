import hashlib
import json
from pathlib import Path

from tools.m3_synthetic_challenge_v1 import (
    EXPECTED_CATEGORY_COUNTS,
    EXPECTED_SCOPE_COUNTS,
    FIELDS,
    build_records,
    load_jsonl,
    validate_records,
    write_jsonl,
)


FIXTURE = Path(__file__).parent / "fixtures" / "m3_synthetic_challenge_v1.jsonl"


def test_generated_records_have_fixed_shape_and_counts():
    records = build_records()
    validate_records(records)
    assert len(records) == 100
    assert len({record["sample_id"] for record in records}) == 100
    assert {field for record in records for field in record} == set(FIELDS)

    scope_counts = {scope: sum(row["expected_scope"] == scope for row in records) for scope in EXPECTED_SCOPE_COUNTS}
    category_counts = {
        category: sum(row["challenge_category"] == category for row in records)
        for category in EXPECTED_CATEGORY_COUNTS
    }
    assert scope_counts == EXPECTED_SCOPE_COUNTS
    assert category_counts == EXPECTED_CATEGORY_COUNTS


def test_serialization_is_deterministic_and_contains_no_head_labels(tmp_path: Path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    records = build_records()
    write_jsonl(records, first)
    write_jsonl(records, second)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest(first) == digest(second)
    assert load_jsonl(first) == load_jsonl(second)
    forbidden_fields = {
        "target_mode",
        "stance",
        "emotion_primary",
        "emotion_target",
        "action_tendency",
        "context_dependency",
        "reasoning_tags",
        "weak_label",
        "prediction",
    }
    assert not forbidden_fields.intersection(records[0])


def test_versioned_fixture_matches_generator():
    records = load_jsonl(FIXTURE)
    validate_records(records)
    assert records == build_records()
    # Ensure the file is JSONL rather than a wrapped array/object artifact.
    assert all(isinstance(json.loads(line), dict) for line in FIXTURE.read_text(encoding="utf-8").splitlines())
