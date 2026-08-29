"""Bounded M1 frozen-Encoder diagnostic run for the owner-pinned RBT3 artifact.

This entry point deliberately avoids the Classical preparation path because that
path loads all trainable labels.  M1 must consume labels from the frozen Train
and Dev split files only and must never produce a Test metric or use Test data
for configuration selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .config import ProjectConfig
from .errors import ContractError
from .hashes import content_addressed_id, sha256_file
from .schema import SINGLE_LABEL_HEADS, V1_HEADS, LabelSchema


MODEL_ID = "hfl/rbt3"
REVISION = "0aa0527ff4170f29e1dfd3eb6ef60dc67e1bf75c"
LICENSE = "Apache-2.0"
RUN_SCHEMA_VERSION = "myresearcher.encoder-m1-diagnostic-run.v1"
INPUT_BUILDER_VERSION = "encoder-input-builder-v1"
DEFAULT_SEED = 35
MAX_DISK_GIB = 10
WALL_TIME_LIMIT_SECONDS = 2 * 60 * 60
SAMPLE_ID_RE = re.compile(r'"sample_id"\s*:\s*"([^"]+)"')


@dataclass(frozen=True)
class M1Record:
    sample_id: str
    stock_code: str
    stock_name: str
    model_text: str
    label: Mapping[str, Any]
    weights: Mapping[str, float]


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _jsonl_append(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError("INVALID_JSONL", str(exc), path=str(path), line=line_number) from exc
            if not isinstance(row, dict):
                raise ContractError("INVALID_JSONL_RECORD", "record must be an object", path=str(path), line=line_number)
            rows.append(row)
    return rows


def _read_selected_jsonl(path: Path, sample_ids: set[str]) -> dict[str, dict[str, Any]]:
    """Parse only selected records from a mixed JSONL file.

    The canonical-input and weight files contain Test rows.  The cheap regular
    expression is used to identify a row first; JSON decoding (and therefore
    inspection of text/weights) occurs only for a Train or Dev sample ID.
    """

    selected: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            match = SAMPLE_ID_RE.search(line)
            if match is None or match.group(1) not in sample_ids:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError("INVALID_JSONL", str(exc), path=str(path), line=line_number) from exc
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or sample_id not in sample_ids:
                raise ContractError("M1_SELECTED_ROW_ID_INVALID", "selected row has invalid sample_id", path=str(path), line=line_number)
            if sample_id in selected:
                raise ContractError("DUPLICATE_SAMPLE_ID", "duplicate selected sample_id", sample_id=sample_id, path=str(path))
            selected[sample_id] = row
    missing = sorted(sample_ids - set(selected))
    if missing:
        raise ContractError("M1_SELECTED_ROWS_MISSING", "selected rows missing from immutable source", sample_ids=missing[:10], missing_count=len(missing))
    return selected


def _normalise(value: str) -> str:
    return unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))


def _record_from_rows(
    sample_id: str,
    inputs: Mapping[str, Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
    weights: Mapping[str, Mapping[str, Any]],
) -> M1Record:
    input_row, label_row, weight_row = inputs[sample_id], labels[sample_id], weights[sample_id]
    for field in ("stock_code", "stock_name"):
        input_value, label_value = input_row.get(field), label_row.get(field)
        if label_value is not None and input_value != label_value:
            raise ContractError("CANONICAL_METADATA_MISMATCH", "split label metadata differs from canonical input", sample_id=sample_id, field=field)
    text = input_row.get("model_text")
    if not isinstance(text, str) or not text.strip():
        raise ContractError("M1_INVALID_MODEL_TEXT", "model_text must be a nonblank string", sample_id=sample_id)
    raw_weights = weight_row.get("weights")
    if not isinstance(raw_weights, Mapping):
        raise ContractError("M1_FIELD_WEIGHTS_INVALID", "weights object is required", sample_id=sample_id)
    parsed_weights: dict[str, float] = {}
    for head in V1_HEADS:
        value = raw_weights.get(head)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
            raise ContractError("M1_FIELD_WEIGHTS_INVALID", "invalid per-head field weight", sample_id=sample_id, head=head)
        parsed_weights[head] = float(value)
    return M1Record(
        sample_id=sample_id,
        stock_code=_normalise(str(input_row.get("stock_code") or "")),
        stock_name=_normalise(str(input_row.get("stock_name") or "")),
        model_text=_normalise(text),
        label=label_row,
        weights=parsed_weights,
    )


def load_m1_partitions(config: ProjectConfig) -> tuple[LabelSchema, list[M1Record], list[M1Record]]:
    """Load only Train/Dev labels and their matching canonical inputs/weights."""

    schema = LabelSchema.load(config.repo_path("schema_path"))
    train_rows = _read_jsonl(config.data_path("split_labels_train"))
    dev_rows = _read_jsonl(config.data_path("split_labels_dev"))
    train_ids = [row.get("sample_id") for row in train_rows]
    dev_ids = [row.get("sample_id") for row in dev_rows]
    if not all(isinstance(sample_id, str) and sample_id for sample_id in [*train_ids, *dev_ids]):
        raise ContractError("M1_SPLIT_LABEL_ID_INVALID", "Train/Dev split labels require sample IDs")
    train_id_set, dev_id_set = set(train_ids), set(dev_ids)
    if len(train_id_set) != 1822 or len(dev_id_set) != 448 or train_id_set & dev_id_set:
        raise ContractError("M1_SPLIT_CONTRACT_INVALID", "expected disjoint Train 1822 and Dev 448", train=len(train_id_set), dev=len(dev_id_set), overlap=len(train_id_set & dev_id_set))
    selected_ids = train_id_set | dev_id_set
    labels = {str(row["sample_id"]): row for row in [*train_rows, *dev_rows]}
    if len(labels) != len(selected_ids):
        raise ContractError("DUPLICATE_SAMPLE_ID", "duplicate Train/Dev split label identity")
    inputs = _read_selected_jsonl(config.data_path("canonical_inputs"), selected_ids)
    weights = _read_selected_jsonl(config.data_path("field_weights"), selected_ids)
    train = [_record_from_rows(sample_id, inputs, labels, weights) for sample_id in train_ids]
    dev = [_record_from_rows(sample_id, inputs, labels, weights) for sample_id in dev_ids]
    return schema, train, dev


def _summary(values: Sequence[int]) -> dict[str, float | int | None]:
    if not values:
        return {key: None for key in ("min", "mean", "p50", "p90", "p95", "p99", "max")}
    ordered = sorted(values)

    def percentile(p: int) -> int:
        return ordered[max(0, math.ceil(p * len(ordered) / 100) - 1)]

    return {"min": min(ordered), "mean": round(sum(ordered) / len(ordered), 3), "p50": percentile(50), "p90": percentile(90), "p95": percentile(95), "p99": percentile(99), "max": max(ordered)}


def _token_components(tokenizer: Any, record: M1Record) -> tuple[list[int], list[int], list[int]]:
    def encode(value: str) -> list[int]:
        result = tokenizer(value, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)
        ids = result.get("input_ids")
        if not isinstance(ids, list) or not all(isinstance(item, int) for item in ids):
            raise ContractError("M1_TOKENIZER_OUTPUT_INVALID", "tokenizer did not return a list of IDs", sample_id=record.sample_id)
        return ids
    return encode(record.stock_code), encode(record.stock_name), encode(record.model_text)


def audit_and_freeze_configuration(tokenizer: Any, train: Sequence[M1Record], dev: Sequence[M1Record]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Use Train+Dev inputs only to produce the immutable pre-fit configuration."""

    component_lengths: dict[str, list[int]] = {"stock_code": [], "stock_name": [], "model_text": [], "assembled_before_caps": []}
    by_population: dict[str, list[int]] = {"Train": [], "Dev": [], "TrainPlusDev": []}
    records: list[tuple[str, M1Record]] = [("Train", record) for record in train] + [("Dev", record) for record in dev]
    for population, record in records:
        code_ids, name_ids, text_ids = _token_components(tokenizer, record)
        total = len(code_ids) + len(name_ids) + len(text_ids) + 4
        component_lengths["stock_code"].append(len(code_ids))
        component_lengths["stock_name"].append(len(name_ids))
        component_lengths["model_text"].append(len(text_ids))
        component_lengths["assembled_before_caps"].append(total)
        by_population[population].append(total)
        by_population["TrainPlusDev"].append(total)
    candidates = [128, 256, 384]
    coverage = {
        str(candidate): {
            "records": len(by_population["TrainPlusDev"]),
            "covered_records": sum(length <= candidate for length in by_population["TrainPlusDev"]),
            "coverage": round(sum(length <= candidate for length in by_population["TrainPlusDev"]) / len(by_population["TrainPlusDev"]), 6),
            "truncated_records": sum(length > candidate for length in by_population["TrainPlusDev"]),
            "truncation_rate": round(sum(length > candidate for length in by_population["TrainPlusDev"]) / len(by_population["TrainPlusDev"]), 6),
        }
        for candidate in candidates
    }
    eligible = [candidate for candidate in candidates if coverage[str(candidate)]["coverage"] >= 0.99]
    max_length = eligible[0] if eligible else candidates[-1]
    config = {
        "configuration_schema_version": "myresearcher.encoder-m1-pretrain-config.v1",
        "configuration_frozen_before_fit": True,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "trust_remote_code": False,
        "input_builder_version": INPUT_BUILDER_VERSION,
        "normalization": "NFC_WITH_CRLF_CR_TO_LF_AND_OTHER_WHITESPACE_PRESERVED",
        "special_token_budget": 4,
        "stock_code_token_cap": 8,
        "stock_name_token_cap": 16,
        "max_length": max_length,
        "max_length_selection_rule": "SMALLEST_128_256_384_WITH_TRAIN_PLUS_DEV_COVERAGE_AT_LEAST_0_99_ELSE_384",
        "truncation": "HEAD_TAIL",
        "padding": "DYNAMIC_RIGHT_PADDING_TO_BATCH_LONGEST_WITH_ATTENTION_MASK",
        "batch_size": 16,
        "seed": DEFAULT_SEED,
        "optimizer": {"name": "AdamW", "learning_rate": 0.0005, "weight_decay": 0.01},
        "stopping": {"max_epochs": 12, "early_stopping_metric": "mean_per_head_macro_f1_with_reasoning_macro_f1", "patience": 3, "wall_time_limit_seconds": WALL_TIME_LIMIT_SECONDS},
        "head_dropout": 0.1,
        "reasoning_probability_threshold": 0.5,
        "encoder_state": "FROZEN",
        "trainable_heads": 7,
        "fit_population": "Train_1822_only",
        "dev_role": "early_stopping_and_diagnostic_only",
        "test_role": "not_loaded_not_used",
    }
    audit = {
        "audit_schema_version": "myresearcher.encoder-m1-tokenizer-audit.v1",
        "model_id": MODEL_ID,
        "required_revision": REVISION,
        "population_scope": ["Train", "Dev"],
        "forbidden_configuration_inputs": ["Test_labels", "Test_metrics"],
        "records": {"Train": len(train), "Dev": len(dev), "TrainPlusDev": len(train) + len(dev)},
        "lengths_with_four_special_tokens": {name: _summary(values) for name, values in by_population.items()},
        "component_token_lengths": {name: _summary(values) for name, values in component_lengths.items()},
        "candidate_max_length_coverage": coverage,
        "frozen_pretrain_configuration": config,
    }
    return audit, config


def build_input_ids(tokenizer: Any, record: M1Record, config: Mapping[str, Any]) -> list[int]:
    code_ids, name_ids, text_ids = _token_components(tokenizer, record)
    code_ids = code_ids[: int(config["stock_code_token_cap"])]
    name_ids = name_ids[: int(config["stock_name_token_cap"])]
    special = 4
    remaining = int(config["max_length"]) - special - len(code_ids) - len(name_ids)
    if remaining < 0:
        raise ContractError("M1_SEGMENT_CAPS_EXCEED_MAX_LENGTH", "configured segment caps leave no body budget")
    if len(text_ids) > remaining:
        if config["truncation"] != "HEAD_TAIL":
            raise ContractError("M1_TRUNCATION_POLICY_INVALID", "M1 accepts only the frozen HEAD_TAIL policy")
        leading = math.ceil(remaining / 2)
        text_ids = text_ids[:leading] + text_ids[-(remaining - leading):] if remaining - leading else text_ids[:leading]
    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    if not isinstance(cls_id, int) or not isinstance(sep_id, int):
        raise ContractError("M1_SPECIAL_TOKEN_IDS_INVALID", "tokenizer must provide CLS and SEP IDs")
    assembled = [cls_id, *code_ids, sep_id, *name_ids, sep_id, *text_ids, sep_id]
    if len(assembled) > int(config["max_length"]):
        raise ContractError("M1_INPUT_LENGTH_EXCEEDED", "input builder exceeded frozen max length", sample_id=record.sample_id)
    return assembled


def _choose_device(torch: Any) -> tuple[Any, str, bool]:
    mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if mps_available:
        return torch.device("mps"), "mps", True
    return torch.device("cpu"), "cpu", False


def _as_batch(torch: Any, tokenizer: Any, records: Sequence[M1Record], config: Mapping[str, Any], device: Any) -> dict[str, Any]:
    rows = [build_input_ids(tokenizer, record, config) for record in records]
    pad_id = tokenizer.pad_token_id
    if not isinstance(pad_id, int):
        pad_id = 0
    max_len = max(map(len, rows))
    input_ids = torch.tensor([row + [pad_id] * (max_len - len(row)) for row in rows], dtype=torch.long, device=device)
    attention_mask = torch.tensor([[1] * len(row) + [0] * (max_len - len(row)) for row in rows], dtype=torch.long, device=device)
    labels: dict[str, Any] = {}
    weights: dict[str, Any] = {}
    for head in SINGLE_LABEL_HEADS:
        labels[head] = torch.tensor([config["class_order"][head].index(str(record.label[head])) for record in records], dtype=torch.long, device=device)
        weights[head] = torch.tensor([record.weights[head] for record in records], dtype=torch.float32, device=device)
    labels["reasoning_tags"] = torch.tensor([[int(tag in record.label["reasoning_tags"]) for tag in config["class_order"]["reasoning_tags"]] for record in records], dtype=torch.float32, device=device)
    weights["reasoning_tags"] = torch.tensor([record.weights["reasoning_tags"] for record in records], dtype=torch.float32, device=device)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "weights": weights}


def _make_model(torch: Any, AutoModel: Any, snapshot_path: Path, schema: LabelSchema, dropout: float) -> Any:
    nn = torch.nn

    class FrozenEncoderSevenHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = AutoModel.from_pretrained(str(snapshot_path), local_files_only=True, trust_remote_code=False)
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
            self.encoder.eval()
            hidden_size = int(self.encoder.config.hidden_size)
            self.dropout = nn.Dropout(dropout)
            self.heads = nn.ModuleDict({head: nn.Linear(hidden_size, len(schema.class_order[head])) for head in V1_HEADS})

        def train(self, mode: bool = True) -> Any:
            super().train(mode)
            self.encoder.eval()
            return self

        def forward(self, input_ids: Any, attention_mask: Any) -> Mapping[str, Any]:
            with torch.no_grad():
                encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                representation = encoded.last_hidden_state[:, 0, :]
            representation = self.dropout(representation)
            return {head: classifier(representation) for head, classifier in self.heads.items()}

    return FrozenEncoderSevenHead()


def _weighted_loss(torch: Any, logits: Mapping[str, Any], batch: Mapping[str, Any]) -> Any:
    losses: list[Any] = []
    functional = torch.nn.functional
    for head in SINGLE_LABEL_HEADS:
        raw = functional.cross_entropy(logits[head], batch["labels"][head], reduction="none")
        weight = batch["weights"][head]
        losses.append((raw * weight).sum() / weight.sum().clamp_min(1e-12))
    raw_reasoning = functional.binary_cross_entropy_with_logits(logits["reasoning_tags"], batch["labels"]["reasoning_tags"], reduction="none").mean(dim=1)
    reasoning_weight = batch["weights"]["reasoning_tags"]
    losses.append((raw_reasoning * reasoning_weight).sum() / reasoning_weight.sum().clamp_min(1e-12))
    return torch.stack(losses).mean()


def _scalar_metrics(truth: Sequence[int], predicted: Sequence[int], class_order: Sequence[str]) -> dict[str, Any]:
    rows = []
    correct = sum(left == right for left, right in zip(truth, predicted, strict=True))
    for index, name in enumerate(class_order):
        tp = sum(left == index and right == index for left, right in zip(truth, predicted, strict=True))
        fp = sum(left != index and right == index for left, right in zip(truth, predicted, strict=True))
        fn = sum(left == index and right != index for left, right in zip(truth, predicted, strict=True))
        support = sum(left == index for left in truth)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append((name, support, precision, recall, f1))
    return {
        "accuracy": round(correct / len(truth), 6) if truth else 0.0,
        "macro_f1": round(sum(row[4] for row in rows) / len(rows), 6) if rows else 0.0,
        "per_class": {name: {"support": support, "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6)} for name, support, precision, recall, f1 in rows},
    }


def _reasoning_metrics(truth: Sequence[Sequence[int]], predicted: Sequence[Sequence[int]], class_order: Sequence[str]) -> dict[str, Any]:
    per_label: dict[str, Any] = {}
    all_tp = all_fp = all_fn = 0
    for index, name in enumerate(class_order):
        tp = sum(row[index] == 1 and predicted_row[index] == 1 for row, predicted_row in zip(truth, predicted, strict=True))
        fp = sum(row[index] == 0 and predicted_row[index] == 1 for row, predicted_row in zip(truth, predicted, strict=True))
        fn = sum(row[index] == 1 and predicted_row[index] == 0 for row, predicted_row in zip(truth, predicted, strict=True))
        support = sum(row[index] == 1 for row in truth)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[name] = {"support": support, "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6)}
        all_tp += tp
        all_fp += fp
        all_fn += fn
    micro_precision = all_tp / (all_tp + all_fp) if all_tp + all_fp else 0.0
    micro_recall = all_tp / (all_tp + all_fn) if all_tp + all_fn else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if micro_precision + micro_recall else 0.0
    return {"micro_f1": round(micro_f1, 6), "macro_f1": round(sum(value["f1"] for value in per_label.values()) / len(per_label), 6), "exact_set_accuracy": round(sum(left == right for left, right in zip(truth, predicted, strict=True)) / len(truth), 6) if truth else 0.0, "per_label": per_label}


def diagnostic_metrics(torch: Any, model: Any, tokenizer: Any, records: Sequence[M1Record], config: Mapping[str, Any], device: Any) -> dict[str, Any]:
    model.eval()
    actual: dict[str, list[Any]] = {head: [] for head in V1_HEADS}
    predicted: dict[str, list[Any]] = {head: [] for head in V1_HEADS}
    with torch.no_grad():
        for offset in range(0, len(records), int(config["batch_size"])):
            batch_records = records[offset:offset + int(config["batch_size"])]
            batch = _as_batch(torch, tokenizer, batch_records, config, device)
            logits = model(batch["input_ids"], batch["attention_mask"])
            for head in SINGLE_LABEL_HEADS:
                actual[head].extend(batch["labels"][head].detach().cpu().tolist())
                predicted[head].extend(logits[head].argmax(dim=1).detach().cpu().tolist())
            actual["reasoning_tags"].extend(batch["labels"]["reasoning_tags"].detach().cpu().int().tolist())
            predicted["reasoning_tags"].extend((torch.sigmoid(logits["reasoning_tags"]) >= float(config["reasoning_probability_threshold"])).detach().cpu().int().tolist())
    result = {head: _scalar_metrics(actual[head], predicted[head], config["class_order"][head]) for head in SINGLE_LABEL_HEADS}
    result["reasoning_tags"] = _reasoning_metrics(actual["reasoning_tags"], predicted["reasoning_tags"], config["class_order"]["reasoning_tags"])
    result["diagnostic_score"] = round(sum(float(result[head]["macro_f1"]) for head in V1_HEADS) / len(V1_HEADS), 6)
    return result


def _hash_tree(root: Path, *, exclude: Iterable[str] = ()) -> list[dict[str, Any]]:
    exclusions = set(exclude)
    rows: list[dict[str, Any]] = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in exclusions:
            continue
        rows.append({"path": relative, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return rows


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def assemble_official_download_parts(
    parts_dir: str | Path,
    destination: str | Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
) -> dict[str, Any]:
    """Atomically assemble verified HTTP range parts for a fixed official file.

    This supports a constrained-network retry without accepting a mirror or a
    floating reference.  The caller supplies only parts retrieved from the
    exact official resolve URL; the byte count and published SHA-256 must match
    before the destination becomes visible to the training entry point.
    """

    source = Path(parts_dir)
    target = Path(destination)
    parts = sorted(path for path in source.iterdir() if path.is_file() and path.name.startswith("part-"))
    if not parts:
        raise ContractError("M1_RANGE_PARTS_MISSING", "no official download parts were found", parts_dir=str(source))
    temporary = target.with_name(f"{target.name}.assembling")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("xb") as output:
        for part in parts:
            with part.open("rb") as input_handle:
                shutil.copyfileobj(input_handle, output, length=1024 * 1024)
    observed_bytes = temporary.stat().st_size
    observed_sha256 = sha256_file(temporary)
    if observed_bytes != expected_bytes or observed_sha256 != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise ContractError("M1_RANGE_ASSEMBLY_HASH_MISMATCH", "assembled official file does not match pinned size/SHA-256", observed_bytes=observed_bytes, expected_bytes=expected_bytes, observed_sha256=observed_sha256, expected_sha256=expected_sha256)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(target)
    return {"path": str(target), "bytes": observed_bytes, "sha256": observed_sha256, "parts": len(parts)}


def _dependency_version(name: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def _git_commit(worktree: Path) -> str | None:
    result = subprocess.run(["git", "-C", str(worktree), "rev-parse", "HEAD"], check=False, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def _write_content_manifest(output_dir: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    files = _hash_tree(output_dir, exclude={"content-addressed-manifest.json"})
    manifest = {**payload, "files": files}
    manifest["content_address"] = content_addressed_id(manifest, omit_keys={"content_address"})
    _json_dump(output_dir / "content-addressed-manifest.json", manifest)
    return manifest


def run_m1(config_path: str | Path, output_dir: str | Path, cache_dir: str | Path) -> dict[str, Any]:
    """Retrieve the exact artifact, audit Train/Dev tokens, and run one diagnostic fit."""

    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    config = ProjectConfig.load(config_path)
    root = Path(output_dir).resolve()
    cache = Path(cache_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise ContractError("M1_OUTPUT_ALREADY_EXISTS", "refuse to merge a new run into an existing artifact directory", output_dir=str(root))
    root.mkdir(parents=True, exist_ok=True)
    before_artifact_bytes = _directory_size(root)
    if before_artifact_bytes:
        raise ContractError("M1_OUTPUT_NOT_EMPTY", "M1 artifact directory must start empty")
    start = time.monotonic()
    initial_free = shutil.disk_usage(root).free
    if _directory_size(cache) + _directory_size(root) > MAX_DISK_GIB * 1024**3:
        raise ContractError("M1_DISK_LIMIT_EXCEEDED", "existing Encoder cache/artifacts exceed the 10 GiB M1 limit")
    random.seed(DEFAULT_SEED)
    np.random.seed(DEFAULT_SEED)
    torch.manual_seed(DEFAULT_SEED)
    if hasattr(torch, "mps"):
        torch.mps.manual_seed(DEFAULT_SEED)

    local_exact_snapshot = cache / "official-snapshot" / REVISION
    required_snapshot_files = {
        "config.json", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "vocab.txt", "added_tokens.json", "README.md",
    }
    if local_exact_snapshot.is_dir() and all((local_exact_snapshot / name).is_file() for name in required_snapshot_files):
        snapshot = local_exact_snapshot.resolve()
    else:
        snapshot = Path(snapshot_download(
            repo_id=MODEL_ID,
            revision=REVISION,
            cache_dir=str(cache),
            allow_patterns=["config.json", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "added_tokens.json", "README.md"],
        )).resolve()
    if snapshot.name != REVISION:
        raise ContractError("M1_RESOLVED_REVISION_MISMATCH", "downloaded snapshot directory does not match required revision", observed=snapshot.name, expected=REVISION)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True)
    model_probe = AutoModel.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
    model_probe.eval()
    del model_probe
    artifact_files = _hash_tree(snapshot)
    tokenizer_files = [row for row in artifact_files if row["path"] in {"tokenizer.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json", "config.json"}]
    if not tokenizer_files:
        raise ContractError("M1_TOKENIZER_ARTIFACT_MISSING", "no tokenizer/config artifacts found in exact snapshot")

    schema, train, dev = load_m1_partitions(config)
    tokenizer_audit, frozen_config = audit_and_freeze_configuration(tokenizer, train, dev)
    frozen_config = {**frozen_config, "class_order": {head: list(schema.class_order[head]) for head in V1_HEADS}}
    tokenizer_audit["frozen_pretrain_configuration"] = frozen_config
    _json_dump(root / "tokenizer-audit.json", tokenizer_audit)
    _json_dump(root / "training-config.json", frozen_config)
    tokenizer.save_pretrained(str(root / "tokenizer"))
    _json_dump(root / "class-order.json", {"schema_version": schema.schema_version, "class_order": frozen_config["class_order"]})

    device, device_name, mps_available = _choose_device(torch)
    model = _make_model(torch, AutoModel, snapshot, schema, float(frozen_config["head_dropout"])).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    encoder_trainable = sum(parameter.numel() for parameter in model.encoder.parameters() if parameter.requires_grad)
    if encoder_trainable != 0 or len(model.heads) != 7:
        raise ContractError("M1_FREEZE_OR_HEAD_CONTRACT_VIOLATION", "Encoder must be frozen with exactly seven heads", encoder_trainable_parameters=encoder_trainable, heads=len(model.heads))
    optimizer = torch.optim.AdamW(trainable, lr=float(frozen_config["optimizer"]["learning_rate"]), weight_decay=float(frozen_config["optimizer"]["weight_decay"]))
    environment = {
        "python": {"version": platform.python_version(), "executable": sys.executable, "implementation": platform.python_implementation()},
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine(), "logical_cpu_count": os.cpu_count()},
        "packages": {name: _dependency_version(name) for name in ("torch", "transformers", "tokenizers", "numpy", "huggingface-hub", "safetensors")},
        "torch": {"version": torch.__version__, "mps_available": mps_available, "selected_device": device_name, "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads()},
        "limits": {"additional_disk_gib": MAX_DISK_GIB, "single_run_wall_time_seconds": WALL_TIME_LIMIT_SECONDS},
        "baseline_environment_mutation": "NOT_PERFORMED",
    }
    _json_dump(root / "environment.json", environment)
    log_path = root / "training-log.jsonl"
    best_score, best_epoch, stale_epochs = -1.0, 0, 0
    wall_time_exceeded = False
    for epoch in range(1, int(frozen_config["stopping"]["max_epochs"]) + 1):
        if time.monotonic() - start > WALL_TIME_LIMIT_SECONDS:
            wall_time_exceeded = True
            break
        epoch_start = time.monotonic()
        model.train()
        shuffled = list(train)
        random.Random(DEFAULT_SEED + epoch).shuffle(shuffled)
        losses: list[float] = []
        for offset in range(0, len(shuffled), int(frozen_config["batch_size"])):
            if time.monotonic() - start > WALL_TIME_LIMIT_SECONDS:
                wall_time_exceeded = True
                break
            batch = _as_batch(torch, tokenizer, shuffled[offset:offset + int(frozen_config["batch_size"])], frozen_config, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = _weighted_loss(torch, logits, batch)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        dev_metrics = diagnostic_metrics(torch, model, tokenizer, dev, frozen_config, device)
        score = float(dev_metrics["diagnostic_score"])
        improved = score > best_score
        if improved:
            best_score, best_epoch, stale_epochs = score, epoch, 0
            torch.save({"run_schema_version": RUN_SCHEMA_VERSION, "model_id": MODEL_ID, "revision": REVISION, "frozen_config": frozen_config, "heads_state_dict": {key: value.detach().cpu() for key, value in model.heads.state_dict().items()}}, root / "heads-checkpoint.pt")
        else:
            stale_epochs += 1
        _jsonl_append(log_path, {"epoch": epoch, "train_weighted_loss": round(sum(losses) / len(losses), 6) if losses else None, "dev_diagnostic_score": score, "dev_macro_f1_by_head": {head: dev_metrics[head]["macro_f1"] for head in V1_HEADS}, "epoch_seconds": round(time.monotonic() - epoch_start, 3), "elapsed_seconds": round(time.monotonic() - start, 3), "improved": improved, "stale_epochs": stale_epochs, "wall_time_exceeded": wall_time_exceeded})
        if wall_time_exceeded or stale_epochs >= int(frozen_config["stopping"]["patience"]):
            break
    if not (root / "heads-checkpoint.pt").is_file():
        raise ContractError("M1_NO_CHECKPOINT", "no diagnostic checkpoint was created")
    checkpoint = torch.load(root / "heads-checkpoint.pt", map_location="cpu", weights_only=True)
    best_model = _make_model(torch, AutoModel, snapshot, schema, float(frozen_config["head_dropout"])).to(device)
    best_model.heads.load_state_dict(checkpoint["heads_state_dict"])
    train_metrics = diagnostic_metrics(torch, best_model, tokenizer, train, frozen_config, device)
    dev_metrics = diagnostic_metrics(torch, best_model, tokenizer, dev, frozen_config, device)

    cpu_model = _make_model(torch, AutoModel, snapshot, schema, float(frozen_config["head_dropout"])).to(torch.device("cpu"))
    cpu_model.heads.load_state_dict(checkpoint["heads_state_dict"])
    cpu_model.eval()
    cpu_batch = _as_batch(torch, tokenizer, [dev[0]], frozen_config, torch.device("cpu"))
    with torch.no_grad():
        cpu_outputs = cpu_model(cpu_batch["input_ids"], cpu_batch["attention_mask"])
    smoke = {"required": True, "device": "cpu", "sample_id": dev[0].sample_id, "all_logits_finite": all(bool(torch.isfinite(value).all().item()) for value in cpu_outputs.values()), "output_shapes": {head: list(value.shape) for head, value in cpu_outputs.items()}}
    if not smoke["all_logits_finite"]:
        raise ContractError("M1_CPU_RELOAD_SMOKE_FAILED", "CPU reload produced non-finite logits")
    _json_dump(root / "cpu-reload-inference-smoke.json", smoke)

    elapsed = time.monotonic() - start
    metrics = {"metric_scope": "WEAK_LABEL_DIAGNOSTIC_ONLY_NOT_GOLD_NOT_PRODUCTION", "model_id": MODEL_ID, "revision": REVISION, "best_epoch": best_epoch, "early_stopping_score": best_score, "wall_time_exceeded": wall_time_exceeded, "train": train_metrics, "dev": dev_metrics, "sample_counts": {"train": len(train), "dev": len(dev)}, "class_distributions": {"train": _class_distribution(train, schema), "dev": _class_distribution(dev, schema)}}
    _json_dump(root / "diagnostic-metrics.json", metrics)
    artifact_size = _directory_size(root)
    cache_size = _directory_size(cache)
    total_encoder_disk = artifact_size + cache_size
    if total_encoder_disk > MAX_DISK_GIB * 1024**3:
        raise ContractError("M1_DISK_LIMIT_EXCEEDED", "M1 cache plus artifacts exceeded 10 GiB", cache_bytes=cache_size, artifact_bytes=artifact_size)
    execution = {"status": "M1_DIAGNOSTIC_TRAINING_COMPLETED" if not wall_time_exceeded else "M1_DIAGNOSTIC_STOPPED_AT_WALL_TIME_LIMIT", "actual_device": device_name, "mps_available": mps_available, "elapsed_seconds": round(elapsed, 3), "best_epoch": best_epoch, "cache_bytes": cache_size, "artifact_bytes": artifact_size, "total_encoder_disk_bytes": total_encoder_disk, "initial_free_disk_bytes": initial_free, "final_free_disk_bytes": shutil.disk_usage(root).free, "model_files": artifact_files, "tokenizer_files": tokenizer_files, "resolved_revision": snapshot.name, "license": LICENSE, "trust_remote_code": False, "cpu_reload_inference_smoke": smoke, "code_commit": _git_commit(Path(__file__).resolve().parents[2])}
    _json_dump(root / "execution-summary.json", execution)
    manifest = _write_content_manifest(root, {"manifest_schema_version": "myresearcher.encoder-m1-artifact-manifest.v1", "diagnostic_only": True, "model_id": MODEL_ID, "resolved_revision": snapshot.name, "license": LICENSE, "artifact_root": str(root), "cache_root": str(cache), "execution_summary_sha256": sha256_file(root / "execution-summary.json"), "training_config_sha256": sha256_file(root / "training-config.json"), "tokenizer_audit_sha256": sha256_file(root / "tokenizer-audit.json"), "metrics_sha256": sha256_file(root / "diagnostic-metrics.json"), "checkpoint_sha256": sha256_file(root / "heads-checkpoint.pt"), "tokenizer_tree_sha256": content_addressed_id({"files": _hash_tree(root / "tokenizer")})})
    return {"output_dir": str(root), "cache_dir": str(cache), "content_address": manifest["content_address"], "execution": execution, "metrics": metrics}


def _class_distribution(records: Sequence[M1Record], schema: LabelSchema) -> dict[str, Any]:
    return {"scalar": {head: dict(sorted(Counter(str(record.label[head]) for record in records).items())) for head in SINGLE_LABEL_HEADS}, "reasoning_tags": {tag: sum(tag in record.label["reasoning_tags"] for record in records) for tag in schema.class_order["reasoning_tags"]}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the owner-authorized M1 frozen RBT3 seven-head diagnostic loop")
    parser.add_argument("--config", required=True, help="Existing immutable baseline config; only Train/Dev M1 paths are read")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    args = parser.parse_args(argv)
    try:
        result = run_m1(args.config, args.output_dir, args.cache_dir)
    except ContractError as exc:
        sys.stderr.write(json.dumps({"status": "M1_DIAGNOSTIC_FAILED", "error": exc.as_dict()}, ensure_ascii=False) + "\n")
        return 2
    sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
