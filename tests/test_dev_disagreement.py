from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from semantic_model import dev_disagreement as analysis
from semantic_model.errors import ContractError


def class_order() -> dict[str, list[str]]:
    return {
        **{head: ["A", "B"] for head in analysis.SINGLE_LABEL_HEADS},
        "reasoning_tags": ["R1", "R2"],
    }


def scalar_prediction(label: str, *, confidence: float = 0.95) -> dict:
    probabilities = (
        [{"label": "A", "probability": confidence}, {"label": "B", "probability": 1.0 - confidence}]
        if label == "A"
        else [{"label": "A", "probability": 1.0 - confidence}, {"label": "B", "probability": confidence}]
    )
    return {
        "prediction": label,
        "ordered_probabilities": probabilities,
        "confidence": confidence,
    }


def reasoning_prediction(labels: list[str], *, confidence: float = 0.95) -> dict:
    return {
        "predicted_labels": labels,
        "ordered_probabilities": [
            {"label": "R1", "probability": confidence if "R1" in labels else 1.0 - confidence},
            {"label": "R2", "probability": confidence if "R2" in labels else 1.0 - confidence},
        ],
        "threshold_outcomes": {"R1": "R1" in labels, "R2": "R2" in labels},
        "confidence": confidence,
    }


def prediction(label: str, reasoning: list[str], *, confidence: float = 0.95) -> dict:
    return {
        **{head: scalar_prediction(label, confidence=confidence) for head in analysis.SINGLE_LABEL_HEADS},
        "reasoning_tags": reasoning_prediction(reasoning, confidence=confidence),
    }


def fake_populations() -> tuple[list[dict], list[dict], list[dict]]:
    records: list[dict] = []
    classical: list[dict] = []
    encoder: list[dict] = []
    for index in range(analysis.DEV_ROWS):
        records.append(
            {
                "sample_id": f"dev-{index:03d}",
                "weak_label": {
                    **{head: "A" for head in analysis.SINGLE_LABEL_HEADS},
                    "reasoning_tags": ["R1"],
                },
            }
        )
        if index == 0:
            classical.append(prediction("A", ["R1"]))
            encoder.append(prediction("B", ["R2"]))
        elif index == 1:
            classical.append(prediction("B", ["R2"], confidence=0.70))
            encoder.append(prediction("A", ["R1"], confidence=0.70))
        else:
            classical.append(prediction("A", ["R1"]))
            encoder.append(prediction("A", ["R1"]))
    return records, classical, encoder


def _trusted_run(root: Path) -> analysis.VerifiedClassicalRun:
    return analysis.VerifiedClassicalRun(
        root=root,
        run_manifest={"run_id": root.name, "run_manifest_id": f"manifest-{root.name}"},
        model_manifest={"model_manifest_id": f"model-{root.name}"},
        thresholds={},
        preprocessing=None,  # Selection tests do not invoke the model path.
    )


def test_selected_canonical_reader_does_not_decode_unselected_payloads(tmp_path):
    source = tmp_path / "canonical.jsonl"
    source.write_text(
        '\n'.join(
            [
                '{"sample_id":"dev-1","model_text":"selected"}',
                '{"sample_id":"test-1","model_text":',
            ]
        )
        + '\n',
        encoding="utf-8",
    )
    selected = analysis._read_selected_canonical_inputs(source, {"dev-1"})
    assert selected == {"dev-1": {"sample_id": "dev-1", "model_text": "selected"}}


def test_analysis_module_never_uses_the_m1_train_dev_loader_or_test_labels():
    source = inspect.getsource(analysis)
    assert "load_m1_partitions" not in source
    assert "split_labels_test" not in source
    assert "split_labels_train" not in source


def test_classical_selector_requires_a_single_manifest_verified_candidate(monkeypatch, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()

    def candidate(path, **_kwargs):
        return _trusted_run(path) if path == first else None

    monkeypatch.setattr(analysis, "_verify_run_candidate", candidate)
    selected = analysis.select_unique_trusted_classical_run(
        tmp_path,
        expected_data_id="data",
        expected_reference_id="reference",
        expected_schema_version="schema",
    )
    assert selected.root == first


def test_classical_selector_fails_closed_when_multiple_candidates_match(monkeypatch, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    monkeypatch.setattr(analysis, "_verify_run_candidate", lambda path, **_kwargs: _trusted_run(path))
    with pytest.raises(ContractError, match="BLOCKED_CLASSICAL_TRUSTED_RUN_AMBIGUOUS"):
        analysis.select_unique_trusted_classical_run(
            tmp_path,
            expected_data_id="data",
            expected_reference_id="reference",
            expected_schema_version="schema",
        )


def test_reasoning_prediction_preserves_threshold_outcomes_and_fallback():
    result = analysis._reasoning_prediction(
        ["R1", "R2"],
        [0.10, 0.20],
        {"R1": 0.50, "R2": 0.50},
        ensure_at_least_one=True,
    )
    assert result["predicted_labels"] == ["R2"]
    assert result["threshold_outcomes"] == {"R1": False, "R2": True}
    assert result["ordered_probabilities"][1]["label"] == "R2"


def test_per_sample_and_aggregate_outputs_call_out_weak_label_scope():
    records, classical, encoder = fake_populations()
    rows = analysis.build_per_sample_analysis(
        records,
        classical,
        encoder,
        high_confidence_threshold=0.80,
    )
    assert set(rows[0]["disagreement_heads"]) == set(analysis.V1_HEADS)
    assert set(rows[0]["high_confidence_disagreement_heads"]) == set(analysis.V1_HEADS)
    assert rows[0]["heads"]["reasoning_tags"]["weak_label"] == ["R1"]
    assert rows[0]["heads"]["reasoning_tags"]["classical"]["threshold_outcomes"] == {"R1": True, "R2": False}
    summary = analysis.aggregate_analysis(
        rows,
        class_order(),
        high_confidence_threshold=0.80,
        review_queue_size=10,
    )
    stance = summary["heads"]["stance"]
    assert stance["disagreement_count"] == 2
    assert stance["classical_only_matches_weak_label"] == 1
    assert stance["encoder_only_matches_weak_label"] == 1
    assert stance["high_confidence_disagreements"] == 1
    assert summary["review_queue"][0]["sample_id"] == "dev-000"
    assert "not Gold" in summary["weak_label_interpretation"]


def test_content_addressed_output_is_idempotent_and_excludes_observation(tmp_path):
    records, classical, encoder = fake_populations()
    rows = analysis.build_per_sample_analysis(records, classical, encoder, high_confidence_threshold=0.80)
    summary = analysis.aggregate_analysis(rows, class_order(), high_confidence_threshold=0.80, review_queue_size=5)
    identity = {
        "scope": summary["scope"],
        "encoder_content_address": "e" * 64,
        "encoder_checkpoint_sha256": "c" * 64,
        "encoder_snapshot_pytorch_model_sha256": "s" * 64,
        "encoder_model_id": "local-encoder",
        "encoder_revision": "fixed-revision",
        "classical_run_id": "run-id",
        "classical_run_manifest_id": "r" * 64,
        "classical_model_manifest_id": "m" * 64,
        "classical_model_sha256": "q" * 64,
        "classical_status": "COMPARABLE_DIAGNOSTIC_RUN_ONLY",
        "data_package_content_id": "d" * 64,
        "reference_package_content_id": "f" * 64,
        "schema_version": "schema-v1",
        "dev_weak_labels_sha256": "l" * 64,
        "analysis_module_sha256": "a" * 64,
        "confidence_threshold": 0.80,
        "review_queue_size": 5,
        "config_sha256": "z" * 64,
    }
    first_dir, first_id = analysis.write_analysis_output(tmp_path, rows=rows, summary=summary, identity=identity)
    second_dir, second_id = analysis.write_analysis_output(tmp_path, rows=rows, summary=summary, identity=identity)
    manifest = analysis._read_json(first_dir / "content-addressed-manifest.json", code="TEST")
    assert first_id == second_id
    assert first_dir == second_dir
    assert manifest["analysis_content_address"] == first_id
    assert manifest["identity"]["dev_row_count"] == analysis.DEV_ROWS
    assert (first_dir / "per-sample-analysis.jsonl").is_file()
    assert (first_dir / "aggregate-report.json").is_file()
