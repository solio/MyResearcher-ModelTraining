import json
from pathlib import Path


PATH = Path(__file__).parent / "fixtures" / "challenge_cases.jsonl"


def test_challenge_fixture_freezes_key_semantic_boundaries():
    cases = [
        json.loads(line)
        for line in PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(cases) == 10
    boundaries = {case["boundary"] for case in cases}
    assert {
        "negation_is_not_reversal",
        "wish_is_not_action",
        "conditional_is_not_completed_action",
        "other_people_action_is_not_author_action",
        "stance_action_independence",
        "unknown_is_not_neutral",
        "none_explicit_is_not_calm",
        "calm_requires_explicit_evidence",
    } <= boundaries
    for case in cases:
        forbidden = case.get("forbidden", {})
        expected = case.get("expected", {})
        for field, values in forbidden.items():
            assert expected.get(field) not in values
