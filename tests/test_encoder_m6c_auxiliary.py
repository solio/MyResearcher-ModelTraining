from __future__ import annotations

from pathlib import Path

import pytest

from semantic_model.data import merge_legacy_labels_and_auxiliary_assignments
from semantic_model.encoder_m6c_auxiliary import (
    AuxiliaryTrainingRuntime,
    adapt_human_train100_rows,
    load_auxiliary_training_views,
)
from semantic_model.errors import ContractError
from semantic_model.schema import SemanticSchemaCandidate, V1_HEADS


SCHEMA_PATH = Path(__file__).parents[1] / "schema" / "semantic-schema-candidate-v0.3.json"
ARTIFACT_PATH = Path(__file__).parents[2] / "model-artifacts" / "m2-selected-candidate-frozen-for-m3-20260904"
CACHE_PATH = Path(__file__).parents[1] / ".encoder-artifacts" / "hf-cache-lert-small"
BASE_VERSION = "semantic-schema-calibrated-v0.2.1"
AUX_VERSION = "semantic-schema-candidate-v0.3"


@pytest.fixture
def candidate_schema() -> SemanticSchemaCandidate:
    return SemanticSchemaCandidate.load(SCHEMA_PATH)


def _flat_rows(valid_label: dict, sample_ids: tuple[str, ...] = ("P001", "P002")) -> list[dict]:
    rows: list[dict] = []
    answers = {
        "target_mode": "ON_TARGET",
        "stance": "BULL",
        "emotion_primary": "NONE_EXPLICIT",
        "emotion_target": "NOT_APPLICABLE",
        "action_tendency": "HOLD",
        "context_dependency": "SELF_CONTAINED",
        "reasoning_tags": ["FUNDAMENTAL"],
    }
    for index, sample_id in enumerate(sample_ids):
        for head in V1_HEADS:
            rows.append(
                {
                    "sample_id": sample_id,
                    "head": head,
                    "answer": list(answers[head]) if isinstance(answers[head], list) else answers[head],
                    "is_final": True,
                    "schema_version": BASE_VERSION,
                    "metadata": {
                        "stock_code": f"60101{index}",
                        "stock_name": "合成样本",
                        "model_text": f"这是第 {index} 条可判断的测试文本。",
                    },
                }
            )
    return rows


def _assignments(sample_ids: tuple[str, ...] = ("P001", "P002")) -> list[dict]:
    answers = {
        "expectation_phase": "ANTICIPATED",
        "arousal_level": "UNKNOWN",
        "desired_price_direction": "UP",
        "market_scope": ["TRADING_FLOW", "BROAD_MARKET"],
        "discourse_focus": ["ASSET_OR_COMPANY", "AUTHOR_VIEWPOINT"],
    }
    rows: list[dict] = []
    for sample_id in sample_ids:
        for head, answer in answers.items():
            rows.append({"sample_id": sample_id, "head": head, "answer": answer, "is_final": True, "schema_version": AUX_VERSION, "source": "synthetic"})
    return rows


def test_flat_human_export_adapts_to_complete_legacy_rows_and_merges_sparse_assignments(candidate_schema, valid_label, tmp_path):
    rows = _flat_rows(valid_label)
    legacy = adapt_human_train100_rows(rows, schema=candidate_schema, expected_samples=2)
    assert len(legacy) == 2
    assert all(set(row) >= {"sample_id", "schema_version", *V1_HEADS} for row in legacy)
    assert all("auxiliary_factors" not in row for row in legacy)
    views = merge_legacy_labels_and_auxiliary_assignments(legacy, _assignments(), candidate_schema)
    assert len(views) == 2
    assert all(view.auxiliary.loss_mask[factor] == 1 for view in views for factor in candidate_schema.auxiliary_factors)

    human_path = tmp_path / "human-train.jsonl"
    assignment_path = tmp_path / "assignments.jsonl"
    human_path.write_text("\n".join(__import__("json").dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    assignment_path.write_text("\n".join(__import__("json").dumps(row, ensure_ascii=False) for row in _assignments()) + "\n", encoding="utf-8")
    loaded = load_auxiliary_training_views(human_path, assignment_path, schema=candidate_schema, expected_samples=2)
    assert [view.sample_id for view in loaded] == ["P001", "P002"]


def test_unfinished_sparse_assignment_is_masked_and_flat_duplicate_or_missing_head_fails(candidate_schema, valid_label):
    rows = _flat_rows(valid_label)
    assignments = _assignments()
    assignments[0] = {**assignments[0], "is_final": False, "answer": None}
    legacy = adapt_human_train100_rows(rows, schema=candidate_schema, expected_samples=2)
    views = merge_legacy_labels_and_auxiliary_assignments(legacy, assignments, candidate_schema)
    assert views[0].auxiliary.loss_mask["expectation_phase"] == 0
    assert views[1].auxiliary.loss_mask["expectation_phase"] == 1
    assert views[0].auxiliary.values["expectation_phase"] is None

    with pytest.raises(ContractError, match="M6C_HUMAN_HEAD_COVERAGE_INVALID"):
        adapt_human_train100_rows(rows[:-1], schema=candidate_schema, expected_samples=2)
    with pytest.raises(ContractError, match="M6C_HUMAN_DUPLICATE"):
        adapt_human_train100_rows([*rows, rows[0]], schema=candidate_schema, expected_samples=2)


@pytest.mark.real_data
@pytest.mark.skipif(not ARTIFACT_PATH.is_dir() or not CACHE_PATH.is_dir(), reason="selected LERT artifact/cache unavailable")
def test_two_sample_real_encoder_forward_backward_checkpoint_reload_preserves_legacy_outputs(candidate_schema, valid_label, tmp_path):
    """Real local snapshot smoke; no Train/Dev/Test rows are read."""

    from semantic_model import encoder_m1 as m1

    legacy = []
    for sample_id in ("P001", "P002"):
        row = dict(valid_label)
        row["sample_id"] = sample_id
        legacy.append(row)
    views = merge_legacy_labels_and_auxiliary_assignments(legacy, _assignments(), candidate_schema)
    records = [
        m1.M1Record(sample_id="P001", stock_code="601012", stock_name="隆基绿能", model_text="订单增长但估值仍需观察。", label=legacy[0], weights={head: 1.0 for head in V1_HEADS}),
        m1.M1Record(sample_id="P002", stock_code="000001", stock_name="平安银行", model_text="如果回调到支撑位，我会继续持有。", label=legacy[1], weights={head: 1.0 for head in V1_HEADS}),
    ]
    runtime = AuxiliaryTrainingRuntime.from_selected_artifact(ARTIFACT_PATH, cache_dir=CACHE_PATH)
    result = runtime.run_two_sample_smoke(records, views, tmp_path / "m6c-auxiliary-smoke.pt")
    assert result["device"] == "cpu"
    assert result["training_epochs"] == 0
    assert result["optimizer_steps"] == 1
    assert result["legacy_seven_head_logits_unchanged"] is True
    assert result["legacy_seven_head_eval_unchanged"] is True
    assert result["frozen_base_parameters_unchanged"] is True
    assert result["auxiliary_parameters_updated"] is True
    assert result["checkpoint_reload_equal"] is True
    assert result["auxiliary_outputs_finite"] is True
    assert set(result["auxiliary_output_shapes"]) == {"expectation_phase", "arousal_level", "desired_price_direction", "market_scope", "discourse_focus"}
