from __future__ import annotations

import json
from pathlib import Path

import torch

from semantic_model.auxiliary_heads import AuxiliaryHeadModel
from semantic_model.encoder_m10c_head_specific_diagnostic import (
    _content_address,
    _discourse_prediction,
    _market_prediction,
    compose_head_specific,
    fresh_shadow_predictions,
    untouched_head_differences,
)
from semantic_model.schema import SemanticSchemaCandidate


SCHEMA = SemanticSchemaCandidate.load("schema/semantic-schema-candidate-v0.3.json")


def _model(width: int = 4) -> AuxiliaryHeadModel:
    return AuxiliaryHeadModel(SCHEMA, width)


def test_only_discourse_is_replaced_and_four_other_heads_are_identical():
    torch.manual_seed(7)
    parent = _model()
    discourse = _model()
    before = {name: value.detach().clone() for name, value in parent.state_dict().items()}
    combined = compose_head_specific(parent, discourse)
    after = {name: value.detach().clone() for name, value in combined.state_dict().items()}
    differences = untouched_head_differences(before, after)
    for factor in ("expectation_phase", "arousal_level", "desired_price_direction", "market_scope"):
        assert differences[factor] == []
    # The source discourse is independently initialized, so this replacement
    # is observable without asserting a particular random value.
    assert all(torch.equal(before[name], after[name]) for name in before if not name.startswith("auxiliary_heads.discourse_focus."))


def test_discourse_policy_is_unknown_exclusive_and_fallbacks_empty():
    order = SCHEMA.auxiliary_factors["discourse_focus"].class_order
    # All non-UNKNOWN logits are below .5; fallback picks the highest one.
    result = _discourse_prediction(torch.tensor([-3.0, -2.0, -4.0, -5.0, -1.0, -6.0, -7.0]), order)
    assert len(result["prediction"]) == 1
    assert result["prediction"] == ["CLAIM_OR_NARRATIVE"]
    # UNKNOWN above threshold is exclusive even when another class is high.
    unknown = _discourse_prediction(torch.tensor([4.0, 4.0, -1.0, -1.0, -1.0, -1.0, 5.0]), order)
    assert unknown["prediction"] == ["UNKNOWN"]


def test_market_policy_keeps_empty_output_allowed():
    order = SCHEMA.auxiliary_factors["market_scope"].class_order
    result = _market_prediction(torch.full((len(order),), -8.0), order)
    assert result["prediction"] == []
    assert result["fallback"] is False


def test_fresh_shadow_processes_every_row_with_fake_runtime():
    width = 4
    model = _model(width)
    representations = {
        "emotion_target": torch.ones(2, width),
        "emotion_primary": torch.ones(2, width),
        "target_mode": torch.ones(2, width),
    }

    class FakeModel:
        def eval(self):
            return self

        def __call__(self, input_ids, attention_mask):
            return {"owner_representations": representations}

    class FakeRuntime:
        model = FakeModel()
        schema = SCHEMA

        def build_inputs(self, records, device):
            return {"input_ids": torch.ones(len(records), 2, dtype=torch.long), "attention_mask": torch.ones(len(records), 2, dtype=torch.long)}

    rows = [
        {"sample_id": "s1", "model_text": "主力资金流入市场", "stock_code": "1", "stock_name": "A"},
        {"sample_id": "s2", "model_text": "期待明天上涨", "stock_code": "2", "stock_name": "B"},
    ]
    result = fresh_shadow_predictions(FakeRuntime(), model, rows)
    assert len(result) == len(rows)
    assert [row["sample_id"] for row in result] == ["s1", "s2"]


def test_manifest_content_address_is_deterministic():
    files = {"a": {"bytes": 1, "sha256": "0" * 64}}
    assert _content_address(files) == _content_address(json.loads(json.dumps(files)))
