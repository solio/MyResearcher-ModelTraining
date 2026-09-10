from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import feedback_adapter_v1 as adapter
from tools import feedback_pool_finalize as finalize


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _selection_row(sample_id: str, group: str, text: str = "这是一条可供人工判断的语义文本") -> dict:
    return {
        "sample_id": sample_id,
        "exact_content_group": group,
        "model_text": text,
        "text": text,
        "stock_code": "601012",
        "stock_name": "synthetic",
    }


def _answer(head: str):
    return [adapter.CLASS_ORDER[head][0]] if head == "reasoning_tags" else adapter.CLASS_ORDER[head][0]


def _label_row(sample_id: str, head: str, batch: str = "synthetic") -> dict:
    return {
        "answer": _answer(head),
        "batch": batch,
        "disposition": "FINAL",
        "head": head,
        "is_final": True,
        "sample_id": sample_id,
        "schema_version": finalize.SCHEMA_VERSION,
        "snapshot_role": "HUMAN_FEED_REVISION",
        "updated_at": "2026-09-10T00:00:00Z",
    }


def _label_rows(sample_ids: list[str]) -> list[dict]:
    return [_label_row(sample_id, head) for sample_id in sample_ids for head in adapter.HEADS]


def _write_round(run_dir: Path, prefix: str, train_count: int, check_count: int, low_check: int = 0) -> tuple[list[str], list[str]]:
    train_ids = [f"{prefix}-train-{index:02d}" for index in range(train_count)]
    check_ids = [f"{prefix}-check-{index:02d}" for index in range(check_count)]
    train_rows = [_selection_row(sample_id, f"group-{sample_id}") for sample_id in train_ids]
    check_rows = []
    for index, sample_id in enumerate(check_ids):
        text = "……" if index < low_check else "这是一条可供人工判断的语义文本"
        check_rows.append(_selection_row(sample_id, f"group-{sample_id}", text))
    _write_jsonl(run_dir / "selection-train.jsonl", train_rows)
    _write_jsonl(run_dir / "selection-check.jsonl", check_rows)
    _write_jsonl(run_dir / "human-labels-revisable.jsonl", _label_rows(train_ids + check_ids))
    return train_ids, check_ids


def _fixture(tmp_path: Path) -> dict:
    v1_train, v1_check = _write_round(tmp_path / "v1", "v1", 10, 4, low_check=2)
    v2_train, v2_check = _write_round(tmp_path / "v2", "v2", 10, 4)
    v3_train, v3_check = _write_round(tmp_path / "v3", "v3", 80, 14)
    status_values = {
        finalize.TRAIN_BATCH: {"batch": finalize.TRAIN_BATCH, "total": 560, "done": 560, "finals": 560, "complete": True},
        finalize.CHECK_BATCH: {"batch": finalize.CHECK_BATCH, "total": 98, "done": 98, "finals": 98, "complete": True},
    }
    train_path = tmp_path / "fl3-train-labels.jsonl"
    check_path = tmp_path / "fl3-check-labels.jsonl"
    _write_jsonl(train_path, _label_rows(v3_train))
    _write_jsonl(check_path, _label_rows(v3_check))
    return {
        "v1_run": tmp_path / "v1",
        "v2_run": tmp_path / "v2",
        "v3_run": tmp_path / "v3",
        "train_labels_path": train_path,
        "check_labels_path": check_path,
        "status_values": status_values,
        "v3_train": v3_train,
    }


def _finalize(tmp_path: Path, fixture: dict, **overrides):
    args = dict(
        output_dir=tmp_path / "output",
        v1_run=fixture["v1_run"],
        v2_run=fixture["v2_run"],
        v3_run=fixture["v3_run"],
        train_labels_path=fixture["train_labels_path"],
        check_labels_path=fixture["check_labels_path"],
        status_values=fixture["status_values"],
    )
    args.update(overrides)
    return finalize.finalize(**args)


def test_incomplete_batches_fail_without_creating_partial_output(tmp_path):
    output = tmp_path / "not-created"
    incomplete = {
        finalize.TRAIN_BATCH: {"total": 560, "done": 0, "finals": 0, "complete": False},
        finalize.CHECK_BATCH: {"total": 98, "done": 0, "finals": 0, "complete": False},
    }
    with pytest.raises(finalize.FinalizeError, match="not both 100% final"):
        finalize.finalize(output_dir=output, status_values=incomplete)
    assert not output.exists()


def test_missing_head_is_rejected(tmp_path):
    fixture = _fixture(tmp_path)
    rows = json.loads("[" + ",".join(fixture["train_labels_path"].read_text(encoding="utf-8").splitlines()) + "]")
    rows.pop()
    bad = tmp_path / "missing-head.jsonl"
    _write_jsonl(bad, rows)
    with pytest.raises(finalize.FinalizeError, match="expected 560 rows"):
        _finalize(tmp_path, fixture, train_labels_path=bad)


def test_duplicate_sample_head_is_rejected(tmp_path):
    fixture = _fixture(tmp_path)
    rows = [json.loads(line) for line in fixture["train_labels_path"].read_text(encoding="utf-8").splitlines()]
    rows[-1] = dict(rows[0])
    bad = tmp_path / "duplicate.jsonl"
    _write_jsonl(bad, rows)
    with pytest.raises(finalize.FinalizeError, match="duplicate sample/head"):
        _finalize(tmp_path, fixture, train_labels_path=bad)


def test_invalid_taxonomy_is_rejected(tmp_path):
    fixture = _fixture(tmp_path)
    rows = [json.loads(line) for line in fixture["train_labels_path"].read_text(encoding="utf-8").splitlines()]
    rows[0]["answer"] = "NOT_IN_TAXONOMY"
    bad = tmp_path / "invalid-label.jsonl"
    _write_jsonl(bad, rows)
    with pytest.raises(finalize.FinalizeError, match="outside the frozen taxonomy"):
        _finalize(tmp_path, fixture, train_labels_path=bad)


def test_success_merges_exact_train100_check20_and_historical_low_information(tmp_path):
    fixture = _fixture(tmp_path)
    summary = _finalize(tmp_path, fixture)
    output = tmp_path / "output"
    assert summary["status"] == "FINALIZED_REVISABLE_HUMAN_LABELS"
    assert summary["labels_are_revisable_not_gold"] is True
    assert summary["training_authorized_by_this_tool"] is False
    assert summary["counts"]["human_train_100_samples"] == 100
    assert summary["counts"]["human_check_20_semantic_samples"] == 20
    assert summary["counts"]["excluded_historical_low_information_check_samples"] == 2
    train_rows = [json.loads(line) for line in (output / "human-train-100-seven-head.jsonl").read_text(encoding="utf-8").splitlines()]
    check_rows = [json.loads(line) for line in (output / "human-check-20-semantic-seven-head.jsonl").read_text(encoding="utf-8").splitlines()]
    excluded_rows = [json.loads(line) for line in (output / "excluded-low-information-check.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(train_rows) == 700
    assert len(check_rows) == 140
    assert len(excluded_rows) == 14
    assert {row["sample_id"] for row in train_rows}.isdisjoint({row["sample_id"] for row in check_rows})
    assert all("prediction" not in row and "probabilities" not in row for row in train_rows + check_rows + excluded_rows)
    assert len({(row["sample_id"], row["head"]) for row in train_rows}) == 700
    assert len({(row["sample_id"], row["head"]) for row in check_rows}) == 140
