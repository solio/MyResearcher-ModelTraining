from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

import pytest

from semantic_model import dev_disagreement as analysis
from semantic_model.config import ProjectConfig
from semantic_model.encoder_m1 import build_input_ids
from semantic_model.errors import ContractError
from semantic_model.hashes import sha256_file
from semantic_model.schema import LabelSchema


PROJECT_ROOT = Path(__file__).parents[1]


def _find_real_dev_config(
    *,
    explicit_config: str | None,
    fallback_config: Path,
) -> Path | None:
    """Return a config only when both permitted local Dev inputs are present."""

    candidates = [Path(explicit_config).expanduser()] if explicit_config else [fallback_config]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            config = ProjectConfig.load(candidate)
            canonical_inputs = config.data_path("canonical_inputs")
            dev_labels = config.data_path("split_labels_dev")
        except ContractError:
            continue
        if canonical_inputs.is_file() and dev_labels.is_file():
            return candidate.resolve()
    return None


REAL_CONFIG_PATH = _find_real_dev_config(
    explicit_config=os.environ.get("MYRESEARCHER_M2_DEV_CONFIG"),
    fallback_config=PROJECT_ROOT / "configs/baseline_v0.3.5.yaml",
)


class CharacterTokenizer:
    """Minimal deterministic tokenizer for exact input-builder parity tests."""

    cls_token_id = 101
    sep_token_id = 102

    def __call__(self, value, **_kwargs):
        return {"input_ids": [1000 + (ord(character) % 997) for character in value]}


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


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _analysis_input_fixture(tmp_path: Path) -> tuple[Path, analysis.VerifiedEncoderArtifact, Path, Path]:
    """Create only the manifest, canonical bytes, and Dev-label bytes allowed here."""

    project = tmp_path / "project"
    package = project / "immutable-package"
    canonical = package / "data/canonical.jsonl"
    dev_labels = package / "splits/labels/dev.jsonl"
    canonical.parent.mkdir(parents=True)
    dev_labels.parent.mkdir(parents=True)
    canonical.write_text('{"sample_id":"dev-1","model_text":"only-dev"}\n', encoding="utf-8")
    dev_labels.write_text('{"sample_id":"dev-1"}\n', encoding="utf-8")
    package_manifest = package / "CONTENT_MANIFEST.json"
    manifest = {
        "manifest_schema_version": "content-addressed-package-manifest-v1",
        "files": [
            {
                "path": "data/canonical.jsonl",
                "size_bytes": canonical.stat().st_size,
                "sha256": sha256_file(canonical),
            },
            {
                "path": "splits/labels/dev.jsonl",
                "size_bytes": dev_labels.stat().st_size,
                "sha256": sha256_file(dev_labels),
            },
        ],
    }
    _write_json(package_manifest, manifest)
    package_hash = sha256_file(package_manifest)
    (package / "CONTENT_MANIFEST.sha256").write_text(
        f"{package_hash}  CONTENT_MANIFEST.json\n",
        encoding="utf-8",
    )
    config_path = project / "analysis-config.yaml"
    _write_json(
        config_path,
        {
            "config_schema_version": "myresearcher.semantic-baseline-config.v1",
            "project_root": ".",
            "data": {
                "root": "immutable-package",
                "package_manifest": "CONTENT_MANIFEST.json",
                "package_manifest_sha256": "CONTENT_MANIFEST.sha256",
                "canonical_inputs": "data/canonical.jsonl",
                "split_labels_dev": "splits/labels/dev.jsonl",
                "expected_package_manifest_sha256": package_hash,
            },
        },
    )
    artifact = analysis.VerifiedEncoderArtifact(
        root=tmp_path / "accepted-artifact",
        cache_root=tmp_path / "cache",
        manifest={
            "provenance": {
                "config_sha256": sha256_file(config_path),
                "data_package_content_id": package_hash,
            }
        },
        training_config={},
        class_order=class_order(),
        snapshot_sha256="s" * 64,
    )
    return config_path, artifact, canonical, dev_labels


def _analysis_output(tmp_path: Path) -> tuple[Path, str, list[dict], dict, dict]:
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
        "package_manifest_sha256": "d" * 64,
        "canonical_inputs_sha256": "i" * 64,
        "dev_weak_labels_sha256": "l" * 64,
        "analysis_module_sha256": "a" * 64,
        "confidence_threshold": 0.80,
        "review_queue_size": 5,
        "config_sha256": "z" * 64,
    }
    output_dir, analysis_id = analysis.write_analysis_output(
        tmp_path,
        rows=rows,
        summary=summary,
        identity=identity,
    )
    return output_dir, analysis_id, rows, summary, identity


def _temporary_analysis_directories(root: Path) -> list[Path]:
    return sorted(path for path in root.glob(".tmp-m2-dev-*") if path.is_dir())


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


def test_config_binding_mismatch_blocks_before_any_model_load(monkeypatch, tmp_path):
    config_path, artifact, _canonical, _dev_labels = _analysis_input_fixture(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["analysis_test_nonce"] = "changes-bytes-without-changing-data-paths"
    _write_json(config_path, payload)
    invoked: list[str] = []
    monkeypatch.setattr(analysis, "verify_encoder_artifact", lambda *_args, **_kwargs: artifact)
    monkeypatch.setattr(analysis, "_predict_classical", lambda *_args, **_kwargs: invoked.append("classical"))
    monkeypatch.setattr(analysis, "_predict_encoder", lambda *_args, **_kwargs: invoked.append("encoder"))
    with pytest.raises(ContractError) as failure:
        analysis.run_dev_disagreement_analysis(
            config_path=config_path,
            encoder_artifact="unused-after-monkeypatch",
            classical_run_catalog=tmp_path / "runs",
            output_root=tmp_path / "output",
        )
    assert failure.value.code == "M2_ANALYSIS_CONFIG_BINDING_MISMATCH"
    assert invoked == []


@pytest.mark.parametrize("input_name", ["canonical", "dev_labels"])
def test_package_manifest_binding_blocks_changed_canonical_or_dev_bytes(tmp_path, input_name):
    config_path, artifact, canonical, dev_labels = _analysis_input_fixture(tmp_path)
    target = canonical if input_name == "canonical" else dev_labels
    target.write_text(target.read_text(encoding="utf-8") + "tamper", encoding="utf-8")
    with pytest.raises(ContractError) as failure:
        analysis.verify_analysis_input_binding(config_path, artifact)
    assert failure.value.code in {"M2_DATA_PACKAGE_SIZE_MISMATCH", "M2_DATA_PACKAGE_HASH_MISMATCH"}


def test_analysis_uses_shared_m1_build_input_ids_without_a_second_contract(monkeypatch):
    record = {
        "sample_id": "dev-1",
        "stock_code": "600001",
        "stock_name": "示例",
        "model_text": "只读分析",
        "weak_label": {**{head: "A" for head in analysis.SINGLE_LABEL_HEADS}, "reasoning_tags": ["R1"]},
    }
    called: list[object] = []

    def shared_builder(_tokenizer, m1_record, _config):
        called.append(m1_record)
        return [101, 102]

    monkeypatch.setattr(analysis, "build_input_ids", shared_builder)
    assert analysis._shared_encoder_input_ids(CharacterTokenizer(), record, {"max_length": 8}) == [101, 102]
    assert len(called) == 1
    assert called[0].sample_id == "dev-1"
    source = inspect.getsource(analysis)
    assert "def _encoder_input_ids" not in source
    assert "return build_input_ids(tokenizer, _as_m1_record(record), config)" in source


def test_real_dev_parity_discovery_skips_portably_without_a_local_package(tmp_path):
    assert _find_real_dev_config(
        explicit_config=None,
        fallback_config=tmp_path / "configs/baseline_v0.3.5.yaml",
    ) is None


@pytest.mark.real_data
@pytest.mark.skipif(REAL_CONFIG_PATH is None, reason="local immutable Dev package unavailable")
def test_all_current_dev_records_use_the_exact_m1_builder():
    assert REAL_CONFIG_PATH is not None
    config = ProjectConfig.load(REAL_CONFIG_PATH)
    schema = LabelSchema.load(config.repo_path("schema_path"))
    _config, records, _labels_sha = analysis.load_dev_records(config, schema.class_order)
    frozen_config = {
        "stock_code_token_cap": 8,
        "stock_name_token_cap": 16,
        "max_length": 256,
        "truncation": "HEAD_TAIL",
    }
    tokenizer = CharacterTokenizer()
    assert len(records) == analysis.DEV_ROWS
    for record in records:
        assert analysis._shared_encoder_input_ids(tokenizer, record, frozen_config) == build_input_ids(
            tokenizer,
            analysis._as_m1_record(record),
            frozen_config,
        )


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


def test_classical_selector_fails_closed_when_no_candidate_matches(monkeypatch, tmp_path):
    (tmp_path / "only-run").mkdir()
    monkeypatch.setattr(analysis, "_verify_run_candidate", lambda *_args, **_kwargs: None)
    with pytest.raises(ContractError, match="BLOCKED_CLASSICAL_TRUSTED_RUN_NOT_FOUND"):
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
    assert (first_dir / "summary.md").is_file()
    assert _temporary_analysis_directories(tmp_path) == []


def test_write_exception_cleans_its_temporary_directory(monkeypatch, tmp_path):
    records, classical, encoder = fake_populations()
    rows = analysis.build_per_sample_analysis(records, classical, encoder, high_confidence_threshold=0.80)
    summary = analysis.aggregate_analysis(rows, class_order(), high_confidence_threshold=0.80, review_queue_size=5)
    monkeypatch.setattr(analysis, "_write_jsonl", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("synthetic write failure")))
    with pytest.raises(RuntimeError, match="synthetic write failure"):
        analysis.write_analysis_output(tmp_path, rows=rows, summary=summary, identity={"scope": summary["scope"]})
    assert _temporary_analysis_directories(tmp_path) == []


@pytest.mark.parametrize(
    ("filename", "mutation"),
    [
        ("per-sample-analysis.jsonl", "replace"),
        ("per-sample-analysis.jsonl", "truncate"),
        ("per-sample-analysis.jsonl", "delete"),
        ("aggregate-report.json", "replace"),
        ("aggregate-report.json", "truncate"),
        ("aggregate-report.json", "delete"),
    ],
)
def test_existing_analysis_payload_tampering_fails_closed(tmp_path, filename, mutation):
    output_dir, _analysis_id, rows, summary, identity = _analysis_output(tmp_path)
    target = output_dir / filename
    if mutation == "replace":
        target.write_text('{"tampered":true}\n', encoding="utf-8")
    elif mutation == "truncate":
        target.write_text("", encoding="utf-8")
    else:
        target.unlink()
    with pytest.raises(ContractError) as failure:
        analysis.write_analysis_output(tmp_path, rows=rows, summary=summary, identity=identity)
    assert failure.value.code == "M2_ANALYSIS_OUTPUT_TAMPERED"
    assert _temporary_analysis_directories(tmp_path) == []


@pytest.mark.parametrize("mutation", ["replace", "truncate", "delete"])
def test_existing_analysis_summary_tampering_fails_closed(tmp_path, mutation):
    output_dir, _analysis_id, rows, summary, identity = _analysis_output(tmp_path)
    summary_path = output_dir / "summary.md"
    if mutation == "replace":
        summary_path.write_text("replacement\n", encoding="utf-8")
    elif mutation == "truncate":
        summary_path.write_text("", encoding="utf-8")
    else:
        summary_path.unlink()
    with pytest.raises(ContractError) as failure:
        analysis.write_analysis_output(tmp_path, rows=rows, summary=summary, identity=identity)
    assert failure.value.code == "M2_ANALYSIS_OUTPUT_TAMPERED"
    assert _temporary_analysis_directories(tmp_path) == []


@pytest.mark.parametrize("mutation", ["identity", "address"])
def test_existing_analysis_manifest_identity_or_address_tampering_fails_closed(tmp_path, mutation):
    output_dir, _analysis_id, rows, summary, identity = _analysis_output(tmp_path)
    manifest_path = output_dir / "content-addressed-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "identity":
        manifest["identity"]["scope"] = "tampered"
    else:
        manifest["analysis_content_address"] = "0" * 64
    _write_json(manifest_path, manifest)
    with pytest.raises(ContractError) as failure:
        analysis.write_analysis_output(tmp_path, rows=rows, summary=summary, identity=identity)
    assert failure.value.code == "M2_ANALYSIS_OUTPUT_TAMPERED"
    assert _temporary_analysis_directories(tmp_path) == []
