from __future__ import annotations

from pathlib import Path

import pytest
import torch

from semantic_model.auxiliary_heads import (
    AuxiliaryHeadModel,
    build_auxiliary_targets,
    compute_auxiliary_losses,
    logical_owner_head,
    merge_and_build_auxiliary_targets,
    total_auxiliary_loss,
)
from semantic_model.data import merge_legacy_labels_and_auxiliary_assignments
from semantic_model.schema import SemanticSchemaCandidate, V03_AUXILIARY_FACTORS


SCHEMA_PATH = Path(__file__).parents[1] / "schema" / "semantic-schema-candidate-v0.3.json"


@pytest.fixture
def candidate_schema() -> SemanticSchemaCandidate:
    return SemanticSchemaCandidate.load(SCHEMA_PATH)


def _legacy(valid_label: dict, sample_id: str) -> dict:
    row = dict(valid_label)
    row["sample_id"] = sample_id
    return row


def _assignments() -> list[dict]:
    version = "semantic-schema-candidate-v0.3"
    return [
        {
            "sample_id": "P001",
            "head": "expectation_phase",
            "answer": "ANTICIPATED",
            "is_final": True,
            "schema_version": version,
        },
        {
            "sample_id": "P001",
            "head": "arousal_level",
            "answer": "UNKNOWN",
            "is_final": True,
            "schema_version": version,
        },
        {
            "sample_id": "P001",
            "head": "market_scope",
            "answer": ["TRADING_FLOW", "BROAD_MARKET"],
            "is_final": True,
            "schema_version": version,
        },
        {
            "sample_id": "P001",
            "head": "discourse_focus",
            "answer": ["ASSET_OR_COMPANY", "AUTHOR_VIEWPOINT"],
            "is_final": True,
            "schema_version": version,
        },
        {
            "sample_id": "P002",
            "head": "expectation_phase",
            "answer": None,
            "is_final": False,
            "schema_version": version,
        },
        {
            "sample_id": "P002",
            "head": "market_scope",
            "answer": ["UNKNOWN"],
            "is_final": True,
            "schema_version": version,
        },
        {
            "sample_id": "P002",
            "head": "discourse_focus",
            "answer": ["UNKNOWN"],
            "is_final": True,
            "schema_version": version,
        },
    ]


def test_auxiliary_heads_use_logical_owner_representations_and_are_independent(candidate_schema):
    model = AuxiliaryHeadModel(
        candidate_schema,
        {"emotion_primary": 4, "emotion_target": 5, "target_mode": 6},
    )
    assert {factor: logical_owner_head(candidate_schema, factor) for factor in V03_AUXILIARY_FACTORS} == {
        "expectation_phase": "emotion_primary",
        "arousal_level": "emotion_primary",
        "desired_price_direction": "emotion_primary",
        "market_scope": "emotion_target",
        "discourse_focus": "target_mode",
    }
    assert set(model.heads) == set(V03_AUXILIARY_FACTORS)
    assert len({id(layer) for layer in model.heads.values()}) == 5
    batch_size = 2
    representations = {
        "emotion_primary": torch.randn(batch_size, 4),
        "emotion_target": torch.randn(batch_size, 5),
        "target_mode": torch.randn(batch_size, 6),
    }
    logits = model(representations)
    assert set(logits) == set(V03_AUXILIARY_FACTORS)
    for factor, values in logits.items():
        assert values.shape == (batch_size, len(candidate_schema.auxiliary_factors[factor].class_order))
        assert torch.isfinite(values).all()


def test_merge_targets_masks_missing_unfinished_and_unknown_without_fabrication(candidate_schema, valid_label):
    legacy = [_legacy(valid_label, "P001"), _legacy(valid_label, "P002"), _legacy(valid_label, "P003")]
    views = merge_legacy_labels_and_auxiliary_assignments(legacy, _assignments(), candidate_schema)
    batch = build_auxiliary_targets(views, candidate_schema)
    assert batch.sample_ids == ("P001", "P002", "P003")
    # Explicit UNKNOWN is a real target and is therefore unmasked.
    expectation_unknown_index = candidate_schema.auxiliary_factors["arousal_level"].class_order.index("UNKNOWN")
    assert batch.targets["arousal_level"][0].item() == expectation_unknown_index
    assert batch.loss_masks["arousal_level"].tolist() == [1.0, 0.0, 0.0]
    # The unfinished expectation assignment is ignored, not converted to UNKNOWN.
    assert batch.loss_masks["expectation_phase"].tolist() == [1.0, 0.0, 0.0]
    assert batch.loss_masks["desired_price_direction"].tolist() == [0.0, 0.0, 0.0]
    assert batch.targets["market_scope"][0].tolist() == [1.0, 0.0, 1.0, 0.0]
    assert batch.targets["market_scope"][1].tolist() == [0.0, 0.0, 0.0, 1.0]
    assert batch.loss_masks["market_scope"].tolist() == [1.0, 1.0, 0.0]


def test_masked_losses_forward_backward_and_checkpoint_reload(candidate_schema, valid_label, tmp_path):
    legacy = [_legacy(valid_label, "P001"), _legacy(valid_label, "P002"), _legacy(valid_label, "P003")]
    batch = merge_and_build_auxiliary_targets(legacy, _assignments(), candidate_schema)
    model = AuxiliaryHeadModel(candidate_schema, {"emotion_primary": 4, "emotion_target": 5, "target_mode": 6})
    representations = {
        "emotion_primary": torch.randn(3, 4),
        "emotion_target": torch.randn(3, 5),
        "target_mode": torch.randn(3, 6),
    }
    logits = model(representations)
    losses = compute_auxiliary_losses(logits, batch, candidate_schema)
    assert set(losses) == set(V03_AUXILIARY_FACTORS)
    assert all(torch.isfinite(value) for value in losses.values())
    total_auxiliary_loss(losses).backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    # The entirely absent desired-price factor is an exact differentiable zero.
    assert losses["desired_price_direction"].item() == 0.0
    for parameter in model.heads["desired_price_direction"].parameters():
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad).item() == 0

    checkpoint = tmp_path / "auxiliary-heads.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = AuxiliaryHeadModel(candidate_schema, {"emotion_primary": 4, "emotion_target": 5, "target_mode": 6})
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    with torch.no_grad():
        original_outputs = model(representations)
        restored_outputs = restored(representations)
    for factor in V03_AUXILIARY_FACTORS:
        assert torch.equal(original_outputs[factor], restored_outputs[factor])
        assert torch.isfinite(restored_outputs[factor]).all()


def test_all_factors_absent_are_zero_safe_and_legacy_v021_is_unchanged(candidate_schema, valid_label):
    legacy = [_legacy(valid_label, "P001")]
    before = dict(legacy[0])
    views = merge_legacy_labels_and_auxiliary_assignments(legacy, [], candidate_schema)
    assert legacy[0] == before
    assert "auxiliary_factors" not in legacy[0]
    batch = build_auxiliary_targets(views, candidate_schema)
    model = AuxiliaryHeadModel(candidate_schema, 3)
    logits = model({
        "emotion_primary": torch.randn(1, 3),
        "emotion_target": torch.randn(1, 3),
        "target_mode": torch.randn(1, 3),
    })
    losses = compute_auxiliary_losses(logits, batch, candidate_schema)
    assert all(value.item() == 0.0 for value in losses.values())
    total_auxiliary_loss(losses).backward()
    assert all(parameter.grad is not None and torch.count_nonzero(parameter.grad).item() == 0 for parameter in model.parameters())
