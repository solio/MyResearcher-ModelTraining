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


def _classical_metrics(contract: dict) -> dict:
    return contract["immutable_controls"]["classical_v0_3_5_control"][
        "frozen_dev_metrics"
    ]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _candidate_at_classical_plus(
    contract: dict, deltas: dict[str, float] | None = None
) -> dict:
    """Synthetic M2 result used only to prove the frozen contract predicates."""

    deltas = deltas or {}
    classical = _classical_metrics(contract)
    reports: dict[str, dict[str, dict[str, float | int | str]]] = {}
    for head, values in classical["scalar_heads"].items():
        reports[head] = {
            label: {
                "f1": details["f1"] + 0.02,
                "support": details["support"],
                "status": "PASS"
                if details["support"] >= 20
                else "NOT_EVALUABLE_FOR_NUMERICAL_NO_REGRESSION",
            }
            for label, details in values["per_class"].items()
        }
    reports["reasoning_tags"] = {
        label: {
            "f1": details["f1"] + 0.02,
            "support": details["support"],
            "status": "PASS"
            if details["support"] >= 20
            else "NOT_EVALUABLE_FOR_NUMERICAL_NO_REGRESSION",
        }
        for label, details in classical["reasoning_tags"]["per_label"].items()
    }
    primary = {
        head: [values["primary_macro_f1"] + deltas.get(head, 0.02)] * 3
        for head, values in classical["scalar_heads"].items()
    }
    primary["reasoning_tags"] = [
        classical["reasoning_tags"]["primary_macro_f1"]
        + deltas.get("reasoning_tags", 0.02)
    ] * 3
    reasoning = classical["reasoning_tags"]
    return {
        "stage_progression_gate_passed": True,
        "eligible_final_stage": "M2-S2-PARTIAL-LAST-ONE-SHARED-SEVEN-HEAD",
        "primary_macro_f1_by_head": primary,
        "reasoning_secondary_by_metric": {
            "micro_f1": [reasoning["micro_f1"] + 0.02] * 3,
            "exact_set_accuracy": [reasoning["exact_set_accuracy"] + 0.02] * 3,
        },
        "critical_label_reports": reports,
    }


def _final_classical_gate_failures(contract: dict, candidate: dict) -> list[str]:
    """Small test-only evaluator proving the explicit final selection gate."""

    gate = contract["selection_gate_types"]["final_m2_candidate_selection_exit_gate"]
    if not candidate["stage_progression_gate_passed"]:
        return ["STAGE_PROGRESSION_GATE_FAILED"]
    if not candidate["eligible_final_stage"].startswith("M2-S2-"):
        return ["INELIGIBLE_FINAL_STAGE"]

    classical = _classical_metrics(contract)
    primary_gate = gate["per_head_primary_macro_f1"]
    critical_gate = gate["critical_label_gate"]
    failures: list[str] = []
    improvements = 0
    scalar = classical["scalar_heads"]
    primary_baselines = {
        **{head: values["primary_macro_f1"] for head, values in scalar.items()},
        "reasoning_tags": classical["reasoning_tags"]["primary_macro_f1"],
    }
    for head, baseline in primary_baselines.items():
        values = candidate["primary_macro_f1_by_head"][head]
        if _mean(values) < baseline - primary_gate["maximum_mean_drop_below_classical"]:
            failures.append(f"CLASSICAL_MEAN_PRIMARY_REGRESSION:{head}")
        if min(values) < baseline - primary_gate["maximum_worst_seed_drop_below_classical"]:
            failures.append(f"CLASSICAL_WORST_SEED_PRIMARY_REGRESSION:{head}")
        if _mean(values) >= baseline + primary_gate["minimum_mean_improvement_over_classical"]:
            improvements += 1
    if improvements < primary_gate["minimum_heads_with_mean_improvement_at_least"]:
        failures.append("CLASSICAL_MINIMUM_HEAD_IMPROVEMENTS_NOT_MET")

    per_label_sources = {
        **{head: values["per_class"] for head, values in scalar.items()},
        "reasoning_tags": classical["reasoning_tags"]["per_label"],
    }
    for head, labels in per_label_sources.items():
        for label, baseline in labels.items():
            observed = candidate["critical_label_reports"][head][label]
            if baseline["support"] < critical_gate["support_threshold"]:
                if observed["status"] != "NOT_EVALUABLE_FOR_NUMERICAL_NO_REGRESSION":
                    failures.append(f"LOW_SUPPORT_FALSE_PASS:{head}:{label}")
            elif observed["f1"] < baseline["f1"] - critical_gate["maximum_f1_drop_below_classical"]:
                failures.append(f"CLASSICAL_CRITICAL_LABEL_REGRESSION:{head}:{label}")

    secondary_gate = gate["reasoning_secondary_gate"]
    for metric in ("micro_f1", "exact_set_accuracy"):
        values = candidate["reasoning_secondary_by_metric"][metric]
        baseline = classical["reasoning_tags"][metric]
        if _mean(values) < baseline - secondary_gate["maximum_mean_drop_below_classical_for_micro_and_exact_set"]:
            failures.append(f"CLASSICAL_REASONING_MEAN_REGRESSION:{metric}")
        if min(values) < baseline - secondary_gate["maximum_worst_seed_drop_below_classical_for_micro_and_exact_set"]:
            failures.append(f"CLASSICAL_REASONING_WORST_SEED_REGRESSION:{metric}")
    return failures


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


def test_classical_dev_reference_identities_and_all_primary_metrics_are_frozen():
    contract = _contract()
    control = contract["immutable_controls"]["classical_v0_3_5_control"]
    reference = control["frozen_dev_reference"]

    assert reference["reference_package_content_id"] == "828944580b96d872241a6619bdb8f60dae2cd7067a0cc6741b418f1e6a7bdc85"
    assert reference["dev_reference_predictions_relative_path"] == "predictions/dev_reference_predictions_v0.3.5.jsonl"
    assert reference["dev_reference_predictions_sha256"] == "833ae4139a99f088986b8b551ae1bc42017844a6215fdf738075faa3ce1174c5"
    assert reference["accepted_recomputed_metrics_relative_path"] == "metrics/recomputed_from_original_model_v0.3.5.json"
    assert reference["accepted_recomputed_metrics_sha256"] == "5e6d9fd186d39e3891733dcf35f00898c1373a187d6dc77aa5720b4ac6779595"
    assert reference["verification"]["independent_dev_recomputation_matches_accepted_metrics_exactly"] is True
    metrics = _classical_metrics(contract)
    assert {
        **{head: value["primary_macro_f1"] for head, value in metrics["scalar_heads"].items()},
        "reasoning_tags": metrics["reasoning_tags"]["primary_macro_f1"],
    } == {
        "target_mode": 0.32587545082033675,
        "stance": 0.3090808276936797,
        "emotion_primary": 0.20679286503516198,
        "emotion_target": 0.27263592379286355,
        "action_tendency": 0.18546185442009192,
        "context_dependency": 0.4705763666581384,
        "reasoning_tags": 0.41025605014132455,
    }
    assert metrics["reasoning_tags"]["micro_f1"] == 0.48633879781420764
    assert metrics["reasoning_tags"]["exact_set_accuracy"] == 0.12723214285714285
    assert all(value["per_class"] for value in metrics["scalar_heads"].values())
    assert metrics["reasoning_tags"]["per_label"]


def test_candidate_that_only_passes_s1_stage_gate_but_is_below_classical_is_rejected():
    contract = _contract()
    all_heads = set(_classical_metrics(contract)["scalar_heads"]) | {"reasoning_tags"}
    candidate = _candidate_at_classical_plus(
        contract,
        {head: -0.02 for head in all_heads},
    )

    failures = _final_classical_gate_failures(contract, candidate)
    assert candidate["stage_progression_gate_passed"] is True
    assert any(item.startswith("CLASSICAL_MEAN_PRIMARY_REGRESSION") for item in failures)


def test_aggregate_improvement_cannot_hide_a_single_classical_head_regression():
    contract = _contract()
    candidate = _candidate_at_classical_plus(contract, {"action_tendency": -0.02})

    assert _mean([_mean(values) for values in candidate["primary_macro_f1_by_head"].values()]) > _mean(
        [
            value["primary_macro_f1"]
            for value in _classical_metrics(contract)["scalar_heads"].values()
        ]
        + [_classical_metrics(contract)["reasoning_tags"]["primary_macro_f1"]]
    )
    failures = _final_classical_gate_failures(contract, candidate)
    assert "CLASSICAL_MEAN_PRIMARY_REGRESSION:action_tendency" in failures


def test_candidate_with_fewer_than_four_classical_head_improvements_is_rejected():
    contract = _contract()
    improving = {"target_mode": 0.02, "stance": 0.02, "reasoning_tags": 0.02}
    all_heads = set(_classical_metrics(contract)["scalar_heads"]) | {"reasoning_tags"}
    candidate = _candidate_at_classical_plus(
        contract, {head: improving.get(head, 0.0) for head in all_heads}
    )

    failures = _final_classical_gate_failures(contract, candidate)
    assert failures == ["CLASSICAL_MINIMUM_HEAD_IMPROVEMENTS_NOT_MET"]


def test_low_support_critical_label_cannot_be_represented_as_a_pass():
    contract = _contract()
    candidate = _candidate_at_classical_plus(contract)
    candidate["critical_label_reports"]["target_mode"]["MARKET_GENERAL"]["status"] = "PASS"

    failures = _final_classical_gate_failures(contract, candidate)
    assert "LOW_SUPPORT_FALSE_PASS:target_mode:MARKET_GENERAL" in failures


def test_s1_completion_may_support_s2_authorization_but_can_never_be_selected():
    contract = _contract()
    stage_gate = contract["selection_gate_types"]["stage_progression_gate"]["S1_frozen_control"]

    assert stage_gate["classical_win_required"] is False
    assert stage_gate["allowed_output"] == "MAY_REQUEST_S2_OWNER_AUTHORIZATION"
    assert stage_gate["forbidden_output"] == "M2_SELECTED_CANDIDATE"
    candidate = _candidate_at_classical_plus(contract)
    candidate["eligible_final_stage"] = "M2-S1-FROZEN-SHARED-SEVEN-HEAD-CONTROL"
    assert _final_classical_gate_failures(contract, candidate) == ["INELIGIBLE_FINAL_STAGE"]
