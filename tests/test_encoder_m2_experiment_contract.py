from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "manifests" / "encoder-m2-experiment-contract-v1.json"
M1_CONTRACT_PATH = ROOT / "manifests" / "encoder-experiment-contract-v1.json"


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _m1_contract() -> dict:
    return json.loads(M1_CONTRACT_PATH.read_text(encoding="utf-8"))


def test_m2_machine_contract_parses_and_freezes_new_lineage():
    contract = _contract()

    assert contract["manifest_schema_version"] == "myresearcher.encoder-m2-experiment-contract.v1"
    assert contract["status"] == "M2_EXPERIMENT_AND_SELECTION_CONTRACT_FROZEN_PENDING_OWNER_EXECUTION_AUTHORIZATION"
    assert contract["new_model_lineage"]["lineage_id"] == "myresearcher-encoder-m2-rbt3-quality-v1"
    assert contract["new_model_lineage"]["parent_evidence_is_read_only"] is True
    assert contract["new_model_lineage"]["m1_artifact_is_not_overwritable"] is True
    assert contract["recommended_first_execution"]["recommendation"].startswith("Reuse")
    model = contract["recommended_first_execution"]["recommended_model"]
    assert model["model_id"] == "hfl/rbt3"
    assert model["revision"] == "0aa0527ff4170f29e1dfd3eb6ef60dc67e1bf75c"
    assert model["trust_remote_code"] is False


def test_m2_contract_has_complete_fixed_gradient_and_seed_reporting_rules():
    contract = _contract()
    stages = contract["minimal_experiment_gradient"]

    assert [stage["stage_id"] for stage in stages] == [
        "M2-S1-FROZEN-SHARED-SEVEN-HEAD-CONTROL",
        "M2-S2-PARTIAL-LAST-ONE-SHARED-SEVEN-HEAD",
        "M2-S3-FROZEN-SINGLE-TASK-HEAD-CONTROL",
    ]
    assert stages[0]["encoder_state"] == "FROZEN"
    assert stages[0]["unfrozen_transformer_blocks"] == 0
    assert stages[1]["encoder_state"] == "PARTIAL_UNFREEZE"
    assert stages[1]["unfrozen_transformer_blocks"] == 1
    assert stages[2]["encoder_state"] == "FROZEN"
    assert all(stage["seeds"] == [35, 71, 107] for stage in stages)
    assert {stage["run_units"] for stage in stages} == {3, "3_PER_TRIGGERED_HEAD_MAXIMUM_21"}
    aggregation = contract["seed_aggregation_and_selection"]
    assert aggregation["fixed_seeds"] == [35, 71, 107]
    assert aggregation["all_seeds_required"] is True
    assert set(aggregation["aggregate_statistics_per_metric"]) == {
        "mean",
        "sample_standard_deviation",
        "minimum_worst_seed",
        "maximum",
        "per_seed_values",
    }
    common = contract["frozen_input_and_common_training_configuration"]
    assert common["early_stopping"]["max_epochs"] == 12
    assert common["early_stopping"]["patience_epochs"] == 3
    assert common["gradient_controls"]["gradient_clipping_max_norm"] == 1.0


def test_m2_contract_freezes_data_roles_metrics_and_no_single_aggregate_selection():
    contract = _contract()
    roles = contract["data_role_and_seal"]

    assert roles["train"] == {
        "rows": 1822,
        "role": "ONLY_FIT_POPULATION_WITH_FROZEN_SAMPLE_ID_X_HEAD_WEIGHTS",
        "allowed_for_fit_after_authorization": True,
    }
    assert roles["dev"]["rows"] == 448
    for sealed_name in ("test", "anchor50", "gold", "ood"):
        assert roles[sealed_name]["sealed"] is True
    metrics = contract["dev_metrics_and_no_regression"]
    assert set(metrics["primary_metric_by_head"]) == {
        "target_mode",
        "stance",
        "emotion_primary",
        "emotion_target",
        "action_tendency",
        "context_dependency",
        "reasoning_tags",
    }
    assert len(metrics["critical_boundary_proxies"]) == 7
    assert set(metrics["required_secondary_metrics"]) == {
        "six_single_label_heads",
        "reasoning_tags",
    }
    assert metrics["stage_no_regression_gate"]["maximum_mean_primary_metric_drop"] == 0.01
    assert metrics["stage_no_regression_gate"]["maximum_worst_seed_primary_metric_drop"] == 0.03
    assert metrics["stage_no_regression_gate"]["maximum_critical_label_f1_drop_when_dev_support_at_least_20"] == 0.05
    assert "single aggregate" in contract["seed_aggregation_and_selection"]["selection_rule"]


def test_m2_contract_is_fail_closed_and_does_not_extend_m1_authorization():
    contract = _contract()
    authorization = contract["m2_execution_authorization"]

    assert authorization["authorization_granted"] is False
    assert authorization["training_allowed"] is False
    assert authorization["execution_state"] == "FAIL_CLOSED_PENDING_OWNER_AUTHORIZATION"
    assert authorization["m1_d023_authorization_reused_for_m2"] is False
    assert _m1_contract()["current_owner_authorization"]["first_run"]["seeds"] == 1
    assert contract["recommended_first_execution"]["recommended_model"]["new_download_allowed_by_this_contract"] is False
    prohibited = contract["prohibitions"]
    assert set(prohibited) == {
        "model_download",
        "dependency_install",
        "encoder_training",
        "classical_training",
        "test_evaluation",
        "anchor_evaluation",
        "gold_evaluation_or_creation",
        "ood_evaluation_or_creation",
        "llm_cloud_or_external_api",
        "production_inference_49054",
        "merge_unreviewed_m2_dev_disagreement_branch",
    }
    assert all(value is False for value in prohibited.values())
    assert contract["data_role_and_seal"]["test"]["allowed_for_selection"] is False
    assert contract["data_role_and_seal"]["gold"]["creation_allowed"] is False
    assert contract["data_role_and_seal"]["ood"]["evaluation_allowed"] is False
    assert contract["data_role_and_seal"]["production_inference_49054"]["allowed"] is False
