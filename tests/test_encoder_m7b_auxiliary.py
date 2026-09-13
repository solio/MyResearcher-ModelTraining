"""Synthetic tests for the finite M7B auxiliary diagnostic runner."""

from __future__ import annotations

import torch

from semantic_model.auxiliary_heads import AuxiliaryFactorLabels, AuxiliaryHeadModel, build_auxiliary_targets
from semantic_model.data import SemanticTrainingView
from semantic_model.encoder_m7b_auxiliary import deterministic_folds, run_oof, _clone_state, _fit_cached_heads
from semantic_model.schema import SemanticSchemaCandidate, V03_AUXILIARY_FACTORS


SCHEMA_PATH = __import__("pathlib").Path(__file__).parents[1] / "schema" / "semantic-schema-candidate-v0.3.json"


def _views(schema: SemanticSchemaCandidate, n: int = 9) -> list[SemanticTrainingView]:
    rows: list[SemanticTrainingView] = []
    for index in range(n):
        values = {
            "expectation_phase": "ANTICIPATED" if index % 2 else "ONGOING_OR_REALIZED",
            "arousal_level": "HIGH" if index % 2 else "LOW",
            "desired_price_direction": "UP" if index % 2 else "DOWN",
            "market_scope": ["BROAD_MARKET"] if index % 2 else ["TRADING_FLOW", "BROAD_MARKET"],
            "discourse_focus": ["ASSET_OR_COMPANY"] if index % 2 else ["AUTHOR_VIEWPOINT"],
        }
        rows.append(
            SemanticTrainingView(
                sample_id=f"S{index:02d}",
                legacy_label={},
                auxiliary=AuxiliaryFactorLabels(values=values, loss_mask={factor: 1 for factor in V03_AUXILIARY_FACTORS}),
                source={"synthetic": True},
            )
        )
    return rows


def test_deterministic_folds_are_sample_grouped_and_complete():
    ids = ["S03", "S01", "S04", "S02", "S00", "S05"]
    first = deterministic_folds(ids)
    second = deterministic_folds(list(reversed(ids)))
    assert first == second
    assert set(first) == set(ids)
    assert set(first.values()) == {0, 1, 2}


def test_m7b_oof_reports_all_factors_and_rare_diagnostic_marker():
    schema = SemanticSchemaCandidate.load(SCHEMA_PATH)
    views = _views(schema)
    batch = build_auxiliary_targets(views, schema)
    torch.manual_seed(20260913)
    template = AuxiliaryHeadModel(schema, 4)
    reps = {
        "emotion_primary": torch.randn(len(views), 4),
        "emotion_target": torch.randn(len(views), 4),
        "target_mode": torch.randn(len(views), 4),
    }
    result = run_oof(reps, batch, [view.sample_id for view in views], schema, initial_state=_clone_state(template.state_dict()))
    assert set(result["aggregate"]) == set(V03_AUXILIARY_FACTORS)
    assert result["aggregate"]["arousal_level"]["diagnostic_class"] == "LOW_SAMPLE_DIAGNOSTIC"
    assert result["aggregate"]["market_scope"]["diagnostic_class"] == "LOW_SAMPLE_DIAGNOSTIC"
    assert result["aggregate"]["expectation_phase"]["metrics"]["n"] == len(views)
    assert result["aggregate"]["market_scope"]["metrics"]["micro_f1"] is not None
    assert all(len(indices) > 0 for indices in ([i for i, sid in enumerate(result["fold_assignment"]) if result["fold_assignment"][sid] == fold] for fold in range(3)))


def test_m7b_final_fit_updates_only_auxiliary_model_and_reloads_state():
    schema = SemanticSchemaCandidate.load(SCHEMA_PATH)
    views = _views(schema, n=3)
    batch = build_auxiliary_targets(views, schema)
    reps = {"emotion_primary": torch.randn(3, 4), "emotion_target": torch.randn(3, 4), "target_mode": torch.randn(3, 4)}
    torch.manual_seed(20260913)
    template = AuxiliaryHeadModel(schema, 4)
    initial = _clone_state(template.state_dict())
    fitted, history = _fit_cached_heads(initial, reps, batch, schema, [0, 1, 2], epochs=2)
    assert len(history) == 2 and history[-1] <= history[0]
    assert any(not torch.equal(initial[name], fitted.state_dict()[name]) for name in initial)
    reloaded = AuxiliaryHeadModel(schema, 4)
    reloaded.load_state_dict(fitted.state_dict())
    with torch.no_grad():
        first = fitted(reps)
        second = reloaded(reps)
    assert all(torch.equal(first[factor], second[factor]) for factor in V03_AUXILIARY_FACTORS)
