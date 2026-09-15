from __future__ import annotations

import json

import pytest

from semantic_model import encoder_terminal_macbert_base as terminal
from semantic_model.schema import LabelSchema


def _schema() -> LabelSchema:
    return LabelSchema.load("schema/semantic-schema-calibrated-v0.2.1.json")


def test_macbert_identity_and_fixed_training_contract() -> None:
    config = terminal._config(_schema())
    assert terminal.MODEL_ID == "hfl/chinese-macbert-base"
    assert len(terminal.REVISION) == 40
    assert config["initialization_source"].endswith("NO_WARM_START")
    assert config["max_length"] == 256
    assert config["batch_size"] == 16
    assert config["stopping"]["max_epochs"] == 24
    assert config["optimizer"]["block_learning_rate"] == 1e-5
    assert config["optimizer"]["head_learning_rate"] == 5e-4
    assert config["reasoning_thresholds"] == {}


def test_trainable_scope_only_selected_block_and_head() -> None:
    import torch

    class Dummy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Linear(2, 2)
            self.specialist_blocks = torch.nn.ModuleDict({"target_mode": torch.nn.Linear(2, 2)})
            self.heads = torch.nn.ModuleDict({"target_mode": torch.nn.Linear(2, 2)})

    model = Dummy()
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in [*model.specialist_blocks["target_mode"].parameters(), *model.heads["target_mode"].parameters()]:
        parameter.requires_grad = True
    identity = terminal.validate_trainable_parameters(model, "target_mode")
    assert identity["encoder_frozen"] is True
    assert identity["active_head"] == "target_mode"


def test_gate_requires_all_five_terminal_conditions() -> None:
    current = {head: {"accuracy": 0.2, "macro_f1": 0.2} for head in terminal.V1_HEADS}
    current["reasoning_tags"] = {"accuracy": 0.2, "exact_set_accuracy": 0.2, "macro_f1": 0.2}
    treatment = {head: {"accuracy": 0.3, "macro_f1": 0.3} for head in terminal.V1_HEADS}
    treatment["reasoning_tags"] = {"accuracy": 0.3, "exact_set_accuracy": 0.3, "macro_f1": 0.3}
    gate = terminal._check_gate(current, treatment, 86, 100)
    assert set(gate["checks"]) == {
        "mean_macro_f1",
        "raw_or_reasoning_strictly_improved_at_least_5_of_7",
        "no_head_drop_over_0.05",
        "strictly_above_majority_at_least_5_of_7",
        "human48_at_least_98_of_240",
    }
    assert gate["passed"] is False  # majority baselines are intentionally strict


def test_human48_excluded_partial_rows_are_not_treated_as_samples(tmp_path) -> None:
    schema = _schema()
    rows = []
    for sample in ("ok-a", "ok-b"):
        for head in terminal.HUMAN48_HEADS:
            rows.append({"sample_id": sample, "head": head, "answer": schema.class_order[head][0], "is_final": True})
    rows.append({"sample_id": "P87937c4e6bfab288", "head": "target_mode", "answer": "ON_TARGET", "is_final": True})
    export = tmp_path / "h48.jsonl"
    export.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    input_rows = [{"sample_id": sample, "stock_code": "000001", "stock_name": "示例", "model_text": "足够长的人工评估文本"} for sample in ("ok-a", "ok-b")]
    inputs = tmp_path / "input.jsonl"
    inputs.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in input_rows) + "\n", encoding="utf-8")
    with pytest.raises(Exception, match="HUMAN48_COVERAGE_INVALID"):
        terminal._load_human48(export, inputs, schema)
