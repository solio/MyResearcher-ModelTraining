"""Terminal, one-shot MacBERT-base seven-head supervision experiment.

This module intentionally mirrors the M5 source-balanced objective while using
the owner-pinned MacBERT snapshot as the *original* initialization.  It is a
diagnostic experiment only: the selected LERT artifact remains untouched and
the result can never promote itself to production.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import platform
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from . import encoder_m1 as m1
from . import encoder_m2_1_softlabel as m21
from . import encoder_m2_selected_runtime as selected
from . import encoder_m5_human100_source_balanced as m5
from .errors import ContractError
from .hashes import sha256_file
from .schema import SINGLE_LABEL_HEADS, V1_HEADS, LabelSchema


MODEL_ID = "hfl/chinese-macbert-base"
REVISION = "a986e004d2a7f2a1c2f5a3edef4e20604a974ed1"
LICENSE = "Apache-2.0"
SEED = 35
MAX_LENGTH = 256
BATCH_SIZE = 16
MAX_EPOCHS = 24
PATIENCE = 3
BLOCK_LR = 1e-5
HEAD_LR = 5e-4
WEIGHT_DECAY = 0.01
GRADIENT_CLIP = 1.0
WALL_TIME_SECONDS = 2 * 60 * 60
MAX_DISK_GIB = 10
WEAK_SOURCE_MASS = 0.5
HUMAN_SOURCE_MASS = 0.5
RUN_SCHEMA_VERSION = "myresearcher.encoder-terminal-macbert-base-seven-head-run.v1"
STAGE_ID = "TERMINAL-MACBERT-BASE-SEVEN-HEAD-SEED35"
BASELINE_CONTENT_ADDRESS = "8f73bc437602ce25d6e8ba0c7fd787676a4b566077b1765f3510739ae85928cf"
PROJECT_ROOT = Path(__file__).resolve().parents[2].parent
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "baseline_v0.3.5.yaml"
DEFAULT_CACHE = REPO_ROOT / ".encoder-artifacts" / "hf-cache-macbert-base"
DEFAULT_PARENT = PROJECT_ROOT / "model-artifacts" / "m2-selected-candidate-frozen-for-m3-20260904"
DEFAULT_POOL = REPO_ROOT / "runs" / "feedback-loop-v3-finalized-20260910"
DEFAULT_HUMAN48 = Path("/tmp/human48.jsonl")
DEFAULT_HUMAN48_INPUT = REPO_ROOT / "runs" / "m3c-shadow500-20260907" / "shadow500-input.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "model-artifacts" / "terminal-macbert-base-seven-head-20260915"
MACBERT_REQUIRED_FILES = ("config.json", "pytorch_model.bin", "added_tokens.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt")
HUMAN48_HEADS = ("target_mode", "stance", "emotion_primary", "emotion_target", "action_tendency")
HUMAN48_EXCLUDED_IDS = {"P87937c4e6bfab288", "Pb8d1f534480f5653"}


def _require(ok: bool, code: str, message: str, **details: Any) -> None:
    if not ok:
        raise ContractError(code, message, **details)


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _read_jsonl(path: Path, code: str) -> list[dict[str, Any]]:
    _require(path.is_file(), code, "JSONL input is missing", path=str(path))
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(code, "invalid JSONL row", path=str(path), line=line_number) from exc
        _require(isinstance(value, dict), code, "JSONL rows must be objects", path=str(path), line=line_number)
        rows.append(value)
    return rows


def snapshot(cache_dir: Path) -> tuple[Path, dict[str, Any]]:
    root = cache_dir.resolve() / "official-snapshot" / REVISION
    _require(root.is_dir(), "MACBERT_CACHE_MISSING", "fixed MacBERT snapshot directory is missing", path=str(root))
    missing = [name for name in MACBERT_REQUIRED_FILES if not (root / name).is_file()]
    _require(not missing, "MACBERT_CACHE_FILES_MISSING", "fixed snapshot is missing required files", missing=missing)
    files = [{"path": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in sorted(root.iterdir()) if path.is_file()]
    return root, {"model_id": MODEL_ID, "revision": REVISION, "license": LICENSE, "files": files, "required_files": list(MACBERT_REQUIRED_FILES), "content_address": m1.content_addressed_id({"model_id": MODEL_ID, "revision": REVISION, "files": files})}


def _config(schema: LabelSchema) -> dict[str, Any]:
    return {
        "stage_id": STAGE_ID,
        "run_schema_version": RUN_SCHEMA_VERSION,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "license": LICENSE,
        "local_files_only": True,
        "trust_remote_code": False,
        "initialization_source": "OFFICIAL_FIXED_MACBERT_ORIGINAL_WEIGHTS_NO_WARM_START",
        "shared_components": ["tokenizer", "input_builder", "embeddings", "transformer_blocks_0_through_10"],
        "per_head_components": "independent_transformer_block_11_and_output_head",
        "max_length": MAX_LENGTH,
        "truncation": "HEAD_TAIL",
        "padding": "DYNAMIC_RIGHT_PADDING_TO_BATCH_LONGEST_WITH_ATTENTION_MASK",
        "token_type_ids": "NOT_EMITTED",
        "batch_size": BATCH_SIZE,
        "head_dropout": 0.1,
        "stock_code_token_cap": 8,
        "stock_name_token_cap": 16,
        "class_order": {head: list(schema.class_order[head]) for head in V1_HEADS},
        "optimizer": {"name": "AdamW", "block_learning_rate": BLOCK_LR, "head_learning_rate": HEAD_LR, "weight_decay": WEIGHT_DECAY, "betas": (0.9, 0.999), "epsilon": 1e-8},
        "stopping": {"max_epochs": MAX_EPOCHS, "patience": PATIENCE, "early_stopping_metric": "per_head_dev_macro_f1"},
        "gradient_clipping_max_norm": GRADIENT_CLIP,
        "reasoning_thresholds": {},
        "fit_population": "Train_1822_plus_HumanTrain100_source_balanced",
        "dev_role": "early_stopping_and_weak_label_regression_observation_only",
        "human_check_role": "post_checkpoint_freeze_diagnostic_only",
        "sealed_roles_not_loaded": ["Test", "Anchor", "Gold", "OOD", "LLM", "production"],
        "selected_candidate": False,
    }


def _set_seed(torch: Any, seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.mps.manual_seed(seed)
    except Exception:
        pass


def _locate_last_block(encoder: Any) -> Any:
    layers = getattr(getattr(encoder, "encoder", None), "layer", None)
    _require(layers is not None and len(layers) >= 12, "MACBERT_ARCHITECTURE_INVALID", "MacBERT must expose at least twelve transformer blocks")
    return layers[-1]


def make_model(torch: Any, AutoModel: Any, snapshot_path: Path, schema: LabelSchema, dropout: float = 0.1) -> Any:
    nn = torch.nn

    class MacBertSpecialistModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = AutoModel.from_pretrained(str(snapshot_path), local_files_only=True, trust_remote_code=False)
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
            self.encoder.eval()
            original_last = _locate_last_block(self.encoder)
            hidden_size = int(self.encoder.config.hidden_size)
            self.dropout = nn.Dropout(dropout)
            self.specialist_blocks = nn.ModuleDict({head: copy.deepcopy(original_last) for head in V1_HEADS})
            self.heads = nn.ModuleDict({head: nn.Linear(hidden_size, len(schema.class_order[head])) for head in V1_HEADS})
            self.active_head: str | None = None
            for block in self.specialist_blocks.values():
                for parameter in block.parameters():
                    parameter.requires_grad = False
            for classifier in self.heads.values():
                for parameter in classifier.parameters():
                    parameter.requires_grad = False

        def set_active_head(self, head: str | None) -> None:
            _require(head is None or head in V1_HEADS, "MACBERT_HEAD_INVALID", "unknown head", head=head)
            self.active_head = head
            for name, parameter in self.named_parameters():
                parameter.requires_grad = bool(head and (name.startswith(f"specialist_blocks.{head}.") or name.startswith(f"heads.{head}.")))

        def train(self, mode: bool = True) -> Any:
            super().train(mode)
            self.encoder.eval()
            return self

        def forward(self, input_ids: Any, attention_mask: Any) -> Mapping[str, Any]:
            captured: dict[str, Any] = {}

            def bypass(_module: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any], _output: Any) -> tuple[Any, ...]:
                captured["args"], captured["kwargs"] = args, kwargs
                return (args[0],)

            base_last = _locate_last_block(self.encoder)
            hook = base_last.register_forward_hook(bypass, with_kwargs=True)
            try:
                self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            finally:
                hook.remove()
            _require("args" in captured, "MACBERT_FORWARD_INVALID", "last-block input was not captured")
            heads = (self.active_head,) if self.active_head else V1_HEADS
            result: dict[str, Any] = {}
            for head in heads:
                block_output = self.specialist_blocks[head](*captured["args"], **captured["kwargs"])
                hidden = block_output[0] if isinstance(block_output, tuple) else block_output
                result[head] = self.heads[head](self.dropout(hidden[:, 0, :]))
            return result

    return MacBertSpecialistModel()


def validate_trainable_parameters(model: Any, head: str | None) -> dict[str, Any]:
    actual = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    expected = {name for name, _parameter in model.named_parameters() if head and (name.startswith(f"specialist_blocks.{head}.") or name.startswith(f"heads.{head}."))}
    _require(actual == expected, "MACBERT_TRAINABLE_SCOPE_INVALID", "only the selected final block and head may train", head=head, unexpected=sorted(actual - expected), missing=sorted(expected - actual))
    _require(not any(name.startswith("encoder.") for name in actual), "MACBERT_ENCODER_UNFROZEN", "MacBERT encoder must remain frozen")
    return {"active_head": head, "trainable_parameters": sorted(actual), "encoder_frozen": True}


def _batch(torch: Any, tokenizer: Any, records: Sequence[m1.M1Record], config: Mapping[str, Any], device: Any) -> dict[str, Any]:
    rows = [m1.build_input_ids(tokenizer, record, config) for record in records]
    pad = tokenizer.pad_token_id if isinstance(tokenizer.pad_token_id, int) else 0
    max_len = max(map(len, rows))
    input_ids = torch.tensor([row + [pad] * (max_len - len(row)) for row in rows], dtype=torch.long, device=device)
    attention = torch.tensor([[1] * len(row) + [0] * (max_len - len(row)) for row in rows], dtype=torch.long, device=device)
    labels: dict[str, Any] = {}; weights: dict[str, Any] = {}
    for head in SINGLE_LABEL_HEADS:
        labels[head] = torch.tensor([config["class_order"][head].index(str(record.label[head])) for record in records], dtype=torch.long, device=device)
        weights[head] = torch.tensor([record.weights[head] for record in records], dtype=torch.float32, device=device)
    labels["reasoning_tags"] = torch.tensor([[int(tag in record.label["reasoning_tags"]) for tag in config["class_order"]["reasoning_tags"]] for record in records], dtype=torch.float32, device=device)
    weights["reasoning_tags"] = torch.tensor([record.weights["reasoning_tags"] for record in records], dtype=torch.float32, device=device)
    return {"input_ids": input_ids, "attention_mask": attention, "labels": labels, "weights": weights}


def _loss(torch: Any, model: Any, tokenizer: Any, records: Sequence[m1.M1Record], config: Mapping[str, Any], device: Any, head: str) -> tuple[Any, float]:
    batch = _batch(torch, tokenizer, records, config, device)
    logits = model(batch["input_ids"], batch["attention_mask"])[head]
    _require(bool(torch.isfinite(logits).all().item()), "MACBERT_NONFINITE_LOGITS", "non-finite training logits", head=head)
    if head in SINGLE_LABEL_HEADS:
        raw = torch.nn.functional.cross_entropy(logits, batch["labels"][head], reduction="none")
    else:
        raw = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch["labels"][head], reduction="none").mean(dim=1)
    weight = batch["weights"][head]
    loss = (raw * weight).sum() / weight.sum().clamp_min(1e-12)
    _require(bool(torch.isfinite(loss).item()), "MACBERT_NONFINITE_LOSS", "non-finite loss", head=head)
    return loss, float(weight.detach().sum().cpu())


def _metric(torch: Any, model: Any, tokenizer: Any, records: Sequence[m1.M1Record], config: Mapping[str, Any], device: Any, head: str, thresholds: Mapping[str, float]) -> tuple[dict[str, Any], np.ndarray]:
    model.eval(); values: list[np.ndarray] = []
    with torch.no_grad():
        for offset in range(0, len(records), BATCH_SIZE):
            batch = _batch(torch, tokenizer, records[offset:offset + BATCH_SIZE], config, device)
            logits = model(batch["input_ids"], batch["attention_mask"])[head]
            _require(bool(torch.isfinite(logits).all().item()), "MACBERT_NONFINITE_EVAL_LOGITS", "non-finite evaluation logits", head=head)
            values.append((torch.softmax(logits, dim=-1) if head in SINGLE_LABEL_HEADS else torch.sigmoid(logits)).detach().cpu().numpy())
    probs = np.concatenate(values, axis=0)
    if head in SINGLE_LABEL_HEADS:
        order = list(config["class_order"][head]); truth = [order.index(str(record.label[head])) for record in records]; pred = probs.argmax(axis=1)
        precision, recall, f1, support = precision_recall_fscore_support(truth, pred, labels=list(range(len(order))), zero_division=0)
        return ({"accuracy": float(np.mean(np.asarray(truth) == pred)), "macro_f1": float(np.mean(f1)), "per_class": {label: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])} for i, label in enumerate(order)}, "confusion_matrix": confusion_matrix(truth, pred, labels=list(range(len(order)))).tolist()}, probs)
    order = list(config["class_order"][head]); truth = np.asarray([[int(tag in record.label["reasoning_tags"]) for tag in order] for record in records]); pred = np.asarray([[int(value >= float(thresholds.get(label, 0.5))) for value, label in zip(row, order, strict=True)] for row in probs]); per_label: dict[str, Any] = {}; tp_total = fp_total = fn_total = 0
    for i, label in enumerate(order):
        tp = int(((truth[:, i] == 1) & (pred[:, i] == 1)).sum()); fp = int(((truth[:, i] == 0) & (pred[:, i] == 1)).sum()); fn = int(((truth[:, i] == 1) & (pred[:, i] == 0)).sum()); support = int(truth[:, i].sum()); p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0; f = 2 * p * r / (p + r) if p + r else 0.0; per_label[label] = {"precision": p, "recall": r, "f1": f, "support": support}; tp_total += tp; fp_total += fp; fn_total += fn
    mp = tp_total / (tp_total + fp_total) if tp_total + fp_total else 0.0; mr = tp_total / (tp_total + fn_total) if tp_total + fn_total else 0.0; micro = 2 * mp * mr / (mp + mr) if mp + mr else 0.0
    return ({"macro_f1": float(np.mean([item["f1"] for item in per_label.values()])), "micro_f1": float(micro), "exact_set_accuracy": float(np.mean(np.all(truth == pred, axis=1))), "per_label": per_label}, probs)


def _model_arrays(torch: Any, model: Any, tokenizer: Any, records: Sequence[m1.M1Record], config: Mapping[str, Any], device: Any, thresholds: Mapping[str, float]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    model.set_active_head(None); model.eval(); arrays: dict[str, list[np.ndarray]] = {head: [] for head in V1_HEADS}
    with torch.no_grad():
        for offset in range(0, len(records), BATCH_SIZE):
            batch = _batch(torch, tokenizer, records[offset:offset + BATCH_SIZE], config, device); outputs = model(batch["input_ids"], batch["attention_mask"])
            for head in SINGLE_LABEL_HEADS: arrays[head].append(torch.softmax(outputs[head], dim=-1).detach().cpu().numpy())
            arrays["reasoning_tags"].append(torch.sigmoid(outputs["reasoning_tags"]).detach().cpu().numpy())
    values = {head: np.concatenate(chunks, axis=0) for head, chunks in arrays.items()}
    metrics: dict[str, Any] = {}
    for head in V1_HEADS:
        model.set_active_head(head); metrics[head] = _metric(torch, model, tokenizer, records, config, device, head, thresholds)[0]
    model.set_active_head(None)
    return values, metrics


def _current_baseline(runtime: Any, records: Sequence[m1.M1Record], schema: LabelSchema) -> dict[str, np.ndarray]:
    inputs = [selected.InferenceRecord(r.sample_id, r.stock_code, r.stock_name, r.model_text) for r in records]
    rows = runtime.predict_records(inputs)
    result: dict[str, list[np.ndarray]] = {head: [] for head in V1_HEADS}
    for row in rows:
        for head in SINGLE_LABEL_HEADS: result[head].append(np.asarray([float(row["predictions"][head]["probabilities"][label]) for label in schema.class_order[head]], dtype=float))
        result["reasoning_tags"].append(np.asarray([float(row["predictions"]["reasoning_tags"]["probabilities"][label]) for label in schema.class_order["reasoning_tags"]], dtype=float))
    return {head: np.asarray(values) for head, values in result.items()}


def _metrics_from_arrays(arrays: Mapping[str, np.ndarray], records: Sequence[m1.M1Record], schema: LabelSchema, thresholds: Mapping[str, float]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for head in SINGLE_LABEL_HEADS:
        order = list(schema.class_order[head]); truth = [order.index(str(r.label[head])) for r in records]; pred = arrays[head].argmax(axis=1); p, r, f, s = precision_recall_fscore_support(truth, pred, labels=list(range(len(order))), zero_division=0); result[head] = {"accuracy": float(np.mean(np.asarray(truth) == pred)), "macro_f1": float(np.mean(f)), "per_class": {label: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])} for i, label in enumerate(order)}, "confusion_matrix": confusion_matrix(truth, pred, labels=list(range(len(order)))).tolist()}
    order = list(schema.class_order["reasoning_tags"]); truth = np.asarray([[int(tag in rec.label["reasoning_tags"]) for tag in order] for rec in records]); pred = np.asarray([[int(value >= float(thresholds.get(label, 0.5))) for value, label in zip(row, order, strict=True)] for row in arrays["reasoning_tags"]]); per: dict[str, Any] = {}; tp = fp = fn = 0
    for i, label in enumerate(order):
        t = int(((truth[:, i] == 1) & (pred[:, i] == 1)).sum()); fpos = int(((truth[:, i] == 0) & (pred[:, i] == 1)).sum()); fneg = int(((truth[:, i] == 1) & (pred[:, i] == 0)).sum()); sup = int(truth[:, i].sum()); pp = t / (t + fpos) if t + fpos else 0.0; rr = t / (t + fneg) if t + fneg else 0.0; ff = 2 * pp * rr / (pp + rr) if pp + rr else 0.0; per[label] = {"precision": pp, "recall": rr, "f1": ff, "support": sup}; tp += t; fp += fpos; fn += fneg
    mp = tp / (tp + fp) if tp + fp else 0.0; mr = tp / (tp + fn) if tp + fn else 0.0; result["reasoning_tags"] = {"macro_f1": float(np.mean([v["f1"] for v in per.values()])), "micro_f1": float(2 * mp * mr / (mp + mr) if mp + mr else 0.0), "exact_set_accuracy": float(np.mean(np.all(truth == pred, axis=1))), "per_label": per}
    return result


def _load_human48(path: Path, input_path: Path, schema: LabelSchema) -> list[m1.M1Record]:
    rows = _read_jsonl(path, "HUMAN48_EXPORT_MISSING")
    by_sample: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("is_final") is not True: continue
        sample, head = row.get("sample_id"), row.get("head")
        if isinstance(sample, str) and sample not in HUMAN48_EXCLUDED_IDS and head in HUMAN48_HEADS:
            _require((sample, head) not in {(s, h) for s, values in by_sample.items() for h in values}, "HUMAN48_DUPLICATE", "duplicate Human48 sample/head", sample_id=sample, head=head)
            by_sample.setdefault(sample, {})[head] = row.get("answer")
    _require(len(by_sample) == 48 and all(set(values) == set(HUMAN48_HEADS) for values in by_sample.values()), "HUMAN48_COVERAGE_INVALID", "Human48 must contain 48 samples and five final scalar heads", samples=len(by_sample))
    inputs = {str(r.get("sample_id")): r for r in _read_jsonl(input_path, "HUMAN48_INPUT_MISSING")}
    records: list[m1.M1Record] = []
    for sample in sorted(by_sample):
        _require(sample in inputs, "HUMAN48_INPUT_ROW_MISSING", "Human48 sample is absent from immutable input", sample_id=sample)
        source = inputs[sample]; labels = {head: by_sample[sample][head] for head in HUMAN48_HEADS}; labels["context_dependency"] = schema.class_order["context_dependency"][0]; labels["reasoning_tags"] = [schema.class_order["reasoning_tags"][0]]
        for head in HUMAN48_HEADS: _require(labels[head] in schema.class_order[head], "HUMAN48_LABEL_INVALID", "Human48 label is outside taxonomy", sample_id=sample, head=head)
        records.append(m1.M1Record(sample, str(source.get("stock_code", "")), str(source.get("stock_name", "")), str(source.get("model_text", "")), labels, {head: 1.0 for head in V1_HEADS}))
    return records


def _parent_thresholds(parent_artifact: Path, torch: Any, schema: LabelSchema) -> dict[str, float]:
    bundle_path = parent_artifact / "unified-runtime-bundle.pt"
    try: bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    except TypeError: bundle = torch.load(bundle_path, map_location="cpu")
    values = bundle.get("reasoning_thresholds", {}) if isinstance(bundle, Mapping) else {}
    return {label: float(values.get(label, 0.5)) for label in schema.class_order["reasoning_tags"]}


def _train_head(*, runtime: tuple[Any, Any, Any, Any], output: Path, snapshot_path: Path, schema: LabelSchema, config: Mapping[str, Any], weak: Sequence[m1.M1Record], human: Sequence[m1.M1Record], dev: Sequence[m1.M1Record], head: str, started: float, thresholds: Mapping[str, float]) -> dict[str, Any]:
    _np, torch, AutoModel, AutoTokenizer = runtime; _set_seed(torch, SEED); device, device_name, mps = m1._choose_device(torch)
    folder = output / head / f"seed-{SEED}"; folder.mkdir(parents=True, exist_ok=False); tokenizer = AutoTokenizer.from_pretrained(str(snapshot_path), local_files_only=True, trust_remote_code=False, use_fast=True); model = make_model(torch, AutoModel, snapshot_path, schema, float(config["head_dropout"])).to(device); model.set_active_head(head); parameter_identity = validate_trainable_parameters(model, head); block_params = list(model.specialist_blocks[head].parameters()); head_params = list(model.heads[head].parameters()); trainable = [*block_params, *head_params]; optimizer = torch.optim.AdamW([{"params": block_params, "lr": BLOCK_LR}, {"params": head_params, "lr": HEAD_LR}], weight_decay=WEIGHT_DECAY)
    best, best_epoch, stale = -math.inf, 0, 0; log_path = folder / "training-log.jsonl"; weak_total = sum(float(record.weights[head]) for record in weak); human_total = sum(float(record.weights[head]) for record in human)
    for epoch in range(1, MAX_EPOCHS + 1):
        _limits(started, output, f"before_epoch_{head}"); model.train(True); source_losses: dict[str, float] = {}; optimizer.zero_grad(set_to_none=True)
        for source_name, source, mass, total in (("weak", weak, WEAK_SOURCE_MASS, weak_total), ("human", human, HUMAN_SOURCE_MASS, human_total)):
            ordered = list(source); random.Random(SEED + epoch + (10000 if source_name == "human" else 0)).shuffle(ordered); losses: list[float] = []
            weighted_sum = 0.0
            for offset in range(0, len(ordered), BATCH_SIZE):
                _limits(started, output, f"during_{source_name}_{head}"); loss, weight_sum = _loss(torch, model, tokenizer, ordered[offset:offset + BATCH_SIZE], config, device, head); (loss * (mass * weight_sum / total)).backward(); weighted_sum += float(loss.detach().cpu()) * weight_sum
            source_losses[source_name] = weighted_sum / max(float(total), 1e-12)
        torch.nn.utils.clip_grad_norm_(trainable, GRADIENT_CLIP)
        for parameter in trainable:
            if parameter.grad is not None: _require(bool(torch.isfinite(parameter.grad).all().item()), "MACBERT_NONFINITE_GRADIENT", "non-finite gradient", head=head)
        optimizer.step(); model.set_active_head(head); dev_metric, _ = _metric(torch, model, tokenizer, dev, config, device, head, thresholds); score = float(dev_metric["macro_f1"]); improved = score > best
        if improved:
            best, best_epoch, stale = score, epoch, 0; torch.save({"run_schema_version": RUN_SCHEMA_VERSION, "stage_id": STAGE_ID, "head": head, "seed": SEED, "model_id": MODEL_ID, "revision": REVISION, "specialist_block_state_dict": {k: v.detach().cpu() for k, v in model.specialist_blocks[head].state_dict().items()}, "head_state_dict": {k: v.detach().cpu() for k, v in model.heads[head].state_dict().items()}, "parameter_identity": parameter_identity}, folder / "macbert-specialist-checkpoint.pt")
        else: stale += 1
        with log_path.open("a", encoding="utf-8") as handle: handle.write(json.dumps({"epoch": epoch, "head": head, "weak_mean_loss": source_losses["weak"], "human_mean_loss": source_losses["human"], "objective": WEAK_SOURCE_MASS * source_losses["weak"] + HUMAN_SOURCE_MASS * source_losses["human"], "dev_macro_f1": score, "improved": improved}, sort_keys=True) + "\n")
        if stale >= PATIENCE: break
    checkpoint_path = folder / "macbert-specialist-checkpoint.pt"; _require(checkpoint_path.is_file(), "MACBERT_CHECKPOINT_MISSING", "best checkpoint missing", head=head); checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True); model.specialist_blocks[head].load_state_dict(checkpoint["specialist_block_state_dict"]); model.heads[head].load_state_dict(checkpoint["head_state_dict"]); train_metric, _ = _metric(torch, model, tokenizer, [*weak, *human], config, device, head, thresholds); dev_metric, _ = _metric(torch, model, tokenizer, dev, config, device, head, thresholds); cpu_model = make_model(torch, AutoModel, snapshot_path, schema, float(config["head_dropout"])); cpu_model.set_active_head(head); cpu_model.specialist_blocks[head].load_state_dict(checkpoint["specialist_block_state_dict"]); cpu_model.heads[head].load_state_dict(checkpoint["head_state_dict"]); cpu_model.eval(); cpu_batch = _batch(torch, tokenizer, [dev[0]], config, torch.device("cpu"));
    with torch.no_grad(): cpu_logits = cpu_model(cpu_batch["input_ids"], cpu_batch["attention_mask"])[head]
    cpu_smoke = {"device": "cpu", "finite": bool(torch.isfinite(cpu_logits).all().item()), "output_shape": list(cpu_logits.shape), "sample_id": dev[0].sample_id}; _require(cpu_smoke["finite"], "MACBERT_CPU_RELOAD_FAILED", "CPU reload emitted non-finite output", head=head); resource = {"head": head, "seed": SEED, "actual_device": device_name, "mps_available": mps, "elapsed_seconds": round(time.monotonic() - started, 3), "cpu_reload_result": cpu_smoke, "python": {"implementation": platform.python_implementation(), "version": platform.python_version()}, "logical_cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads()}; _dump(folder / "seed-metrics.json", {"head": head, "seed": SEED, "best_epoch": best_epoch, "train": train_metric, "dev": dev_metric, "cpu_reload": cpu_smoke}); _dump(folder / "resource-log.json", resource); checkpoint_sha = sha256_file(checkpoint_path); del model, cpu_model; gc.collect(); return {"head": head, "seed": SEED, "best_epoch": best_epoch, "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": checkpoint_sha, "train": train_metric, "dev": dev_metric, "cpu_reload": cpu_smoke, "resource": resource}


def _limits(started: float, output: Path, phase: str) -> None:
    _require(time.monotonic() - started <= WALL_TIME_SECONDS, "MACBERT_WALL_TIME_LIMIT_EXCEEDED", "two-hour wall clock limit exceeded", phase=phase)
    _require(m1._directory_size(output) <= MAX_DISK_GIB * 1024**3, "MACBERT_DISK_LIMIT_EXCEEDED", "ten GiB output limit exceeded", phase=phase)


def _human48_metrics(arrays: Mapping[str, np.ndarray], records: Sequence[m1.M1Record], schema: LabelSchema) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for head in HUMAN48_HEADS:
        order = list(schema.class_order[head])
        truth = [order.index(str(record.label[head])) for record in records]
        predicted = np.asarray(arrays[head]).argmax(axis=1)
        precision, recall, f1, support = precision_recall_fscore_support(truth, predicted, labels=list(range(len(order))), zero_division=0)
        result[head] = {"accuracy": float(np.mean(np.asarray(truth) == predicted)), "macro_f1": float(np.mean(f1)), "per_class": {label: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])} for i, label in enumerate(order)}, "confusion_matrix": confusion_matrix(truth, predicted, labels=list(range(len(order)))).tolist()}
    return result


def _check_gate(current: Mapping[str, Any], treatment: Mapping[str, Any], human48_current: int, human48_treatment: int) -> dict[str, Any]:
    heads = list(V1_HEADS); baseline_acc = {"target_mode": .75, "stance": .30, "emotion_primary": .25, "emotion_target": .40, "action_tendency": .20, "context_dependency": .60, "reasoning_tags": .20}; majority = {"target_mode": .80, "stance": .65, "emotion_primary": .25, "emotion_target": .35, "action_tendency": .10, "context_dependency": .85, "reasoning_tags": .25}
    raw_keys = {head: ("exact_set_accuracy" if head == "reasoning_tags" else "accuracy") for head in heads}; deltas = {head: float(treatment[head][raw_keys[head]] - current[head][raw_keys[head]]) for head in heads}; gate = {"mean_macro_f1": float(np.mean([treatment[h]["macro_f1"] if h != "reasoning_tags" else treatment[h]["macro_f1"] for h in heads])) >= .2870685697, "raw_or_reasoning_strictly_improved_at_least_5_of_7": sum(deltas[h] > 0 for h in heads) >= 5, "no_head_drop_over_0.05": all(deltas[h] >= -0.05 for h in heads), "strictly_above_majority_at_least_5_of_7": sum(float(treatment[h][raw_keys[h]]) > majority[h] for h in heads) >= 5, "human48_at_least_98_of_240": human48_treatment >= 98}
    return {"checks": gate, "passed": bool(all(gate.values())), "current_accuracy_baseline": baseline_acc, "majority_baseline": majority, "raw_deltas": deltas, "human48_current": human48_current, "human48_treatment": human48_treatment}


def run_terminal(*, config_path: str | Path = DEFAULT_CONFIG, cache_dir: str | Path = DEFAULT_CACHE, parent_artifact: str | Path = DEFAULT_PARENT, pool_dir: str | Path = DEFAULT_POOL, human48_jsonl: str | Path = DEFAULT_HUMAN48, human48_input: str | Path = DEFAULT_HUMAN48_INPUT, output_dir: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = Path(output_dir).resolve(); _require(not output.exists(), "MACBERT_OUTPUT_ALREADY_EXISTS", "output directory must be new", path=str(output)); started = time.monotonic(); project = m1.ProjectConfig.load(config_path); schema, weak, dev = m1.load_m1_partitions(project); human, human_audit = m5.load_human_train_pool(pool_dir); _require(len(weak) == 1822 and len(dev) == 448 and len(human) == 100, "MACBERT_DATA_SCOPE_INVALID", "expected Train 1822, Dev 448, Human Train 100")
    runtime = m1._load_runtime_dependencies(); _np, torch, AutoModel, AutoTokenizer = runtime; snapshot_path, snapshot_identity = snapshot(Path(cache_dir)); config = _config(schema); parent_thresholds = _parent_thresholds(Path(parent_artifact).resolve(), torch, schema); config["reasoning_thresholds"] = parent_thresholds; output.mkdir(parents=True, exist_ok=False); _dump(output / "training-config.json", config); _dump(output / "provenance.json", {"stage_id": STAGE_ID, "model_identity": {"model_id": MODEL_ID, "revision": REVISION, "license": LICENSE}, "cache_snapshot": snapshot_identity, "data_scope": {"weak_train": len(weak), "human_train": len(human), "dev": len(dev), "human_check": 20, "human48": 48}, "parent_selected_artifact_content_address": BASELINE_CONTENT_ADDRESS, "human_train_audit": human_audit, "forbidden_inputs": ["Test", "Anchor", "Gold", "OOD", "LLM", "production"]})
    baseline_runtime = selected.UnifiedCPUInferenceRuntime.from_artifact(parent_artifact, config_path=config_path, cache_dir=REPO_ROOT / ".encoder-artifacts" / "hf-cache-lert-small"); results: list[dict[str, Any]] = []
    for head in V1_HEADS:
        results.append(_train_head(runtime=runtime, output=output, snapshot_path=snapshot_path, schema=schema, config=config, weak=weak, human=human, dev=dev, head=head, started=started, thresholds=parent_thresholds))
    _require(len(results) == len(V1_HEADS), "MACBERT_CHECKPOINT_COVERAGE_INVALID", "all seven checkpoints must be frozen")
    # Check20 and Human48 are opened only after every checkpoint is frozen.
    human_check, check_audit = m5.load_human_check_pool(pool_dir); human48 = _load_human48(Path(human48_jsonl), Path(human48_input), schema)
    treatment = make_model(torch, AutoModel, snapshot_path, schema, float(config["head_dropout"])); treatment.set_active_head(None)
    for item in results:
        checkpoint = torch.load(item["checkpoint_path"], map_location="cpu", weights_only=True); treatment.specialist_blocks[item["head"]].load_state_dict(checkpoint["specialist_block_state_dict"]); treatment.heads[item["head"]].load_state_dict(checkpoint["head_state_dict"])
    unified_bundle_path = output / "macbert-unified-seven-head-bundle.pt"
    torch.save({"run_schema_version": RUN_SCHEMA_VERSION, "stage_id": STAGE_ID, "status": "TERMINAL_DIAGNOSTIC_ONLY", "selected_candidate": False, "production_approval": False, "model_identity": {"model_id": MODEL_ID, "revision": REVISION, "license": LICENSE}, "seed": SEED, "specialist_block_state_dict": {head: {k: v.detach().cpu() for k, v in treatment.specialist_blocks[head].state_dict().items()} for head in V1_HEADS}, "head_state_dict": {head: {k: v.detach().cpu() for k, v in treatment.heads[head].state_dict().items()} for head in V1_HEADS}, "reasoning_thresholds": parent_thresholds}, unified_bundle_path)
    treatment = treatment.to(torch.device("cpu")); tokenizer = AutoTokenizer.from_pretrained(str(snapshot_path), local_files_only=True, trust_remote_code=False, use_fast=True)
    check_records = human_check; baseline_check = _metrics_from_arrays(_current_baseline(baseline_runtime, check_records, schema), check_records, schema, parent_thresholds); treatment_check_arrays, treatment_check = _model_arrays(torch, treatment, tokenizer, check_records, config, torch.device("cpu"), parent_thresholds)
    baseline_dev = _metrics_from_arrays(_current_baseline(baseline_runtime, dev, schema), dev, schema, parent_thresholds); treatment_dev_arrays, treatment_dev = _model_arrays(torch, treatment, tokenizer, dev, config, torch.device("cpu"), parent_thresholds)
    baseline_h48_arrays = _current_baseline(baseline_runtime, human48, schema); treatment_h48_arrays, _ = _model_arrays(torch, treatment, tokenizer, human48, config, torch.device("cpu"), parent_thresholds); baseline_h48_metrics = _human48_metrics(baseline_h48_arrays, human48, schema); treatment_h48_metrics = _human48_metrics(treatment_h48_arrays, human48, schema); baseline_h48_total = int(sum(round(item["accuracy"] * len(human48)) for item in baseline_h48_metrics.values())); treatment_h48_total = int(sum(round(item["accuracy"] * len(human48)) for item in treatment_h48_metrics.values())); _require(baseline_h48_total == 86, "MACBERT_BASELINE_HUMAN48_MISMATCH", "current runtime does not reproduce frozen 86/240 baseline", observed=baseline_h48_total)
    gate = _check_gate(baseline_check, treatment_check, baseline_h48_total, treatment_h48_total); status = "TERMINAL_BASE_ENCODER_PASS_DIAGNOSTIC" if gate["passed"] else "TERMINAL_BASE_ENCODER_NO_GO"; _dump(output / "human-check20-before-after.json", {"current": baseline_check, "macbert": treatment_check, "gate": gate}); _dump(output / "dev448-before-after.json", {"current": baseline_dev, "macbert": treatment_dev}); _dump(output / "human48-before-after.json", {"current": baseline_h48_metrics, "macbert": treatment_h48_metrics, "current_total": baseline_h48_total, "macbert_total": treatment_h48_total}); _dump(output / "stage-aggregate.json", {"stage_id": STAGE_ID, "status": status, "selected_candidate": False, "production_approval": False, "gate": gate, "actual_device": sorted({item["resource"]["actual_device"] for item in results}), "elapsed_seconds": round(time.monotonic() - started, 3)})
    lines = ["# Terminal MacBERT-base seven-head diagnostic", "", f"Status: **{status}**", "", f"Model: `{MODEL_ID}@{REVISION}` ({LICENSE}); initialization is official original weights with no warm-start.", "", "| head | current Check20 accuracy/exact | MacBERT accuracy/exact | Macro-F1 |", "|---|---:|---:|---:|"]
    for head in V1_HEADS: lines.append(f"| {head} | {baseline_check[head].get('accuracy', baseline_check[head].get('exact_set_accuracy', 0.0)):.6f} | {treatment_check[head].get('accuracy', treatment_check[head].get('exact_set_accuracy', 0.0)):.6f} | {treatment_check[head]['macro_f1']:.6f} |")
    lines += ["", f"Human48 current/MacBERT: {baseline_h48_total}/240 → {treatment_h48_total}/240.", "", "The five gates are recorded in human-check20-before-after.json.  This run is diagnostic and never grants production approval.", "", "No Test, Anchor, Gold, OOD, LLM, or production data were read."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8"); manifest = m1._write_content_manifest(output, {"manifest_schema_version": "myresearcher.terminal-macbert-base-artifact-manifest.v1", "stage_id": STAGE_ID, "status": status, "selected_candidate": False, "production_approval": False, "model_identity": {"model_id": MODEL_ID, "revision": REVISION, "license": LICENSE}, "cache_snapshot": snapshot_identity, "parent_selected_artifact_content_address": BASELINE_CONTENT_ADDRESS, "seed": SEED, "train_rows": 1822, "human_train_rows": 100, "dev_rows": 448, "human_check_rows": 20, "human48_rows": 48, "checkpoint_sha256": {item["head"]: item["checkpoint_sha256"] for item in results}, "gate": gate, "forbidden_inputs": ["Test", "Anchor", "Gold", "OOD", "LLM", "production"]}); _limits(started, output, "after_manifest"); return {"status": status, "output_dir": str(output), "content_address": manifest["content_address"], "results": results, "human_check20": {"current": baseline_check, "macbert": treatment_check}, "human48": {"current_total": baseline_h48_total, "macbert_total": treatment_h48_total}, "gate": gate}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One-shot terminal MacBERT-base seven-head diagnostic")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG)); parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE)); parser.add_argument("--parent-artifact", default=str(DEFAULT_PARENT)); parser.add_argument("--pool-dir", default=str(DEFAULT_POOL)); parser.add_argument("--human48-jsonl", default=str(DEFAULT_HUMAN48)); parser.add_argument("--human48-input", default=str(DEFAULT_HUMAN48_INPUT)); parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT)); args = parser.parse_args(argv)
    try:
        result = run_terminal(config_path=args.config, cache_dir=args.cache_dir, parent_artifact=args.parent_artifact, pool_dir=args.pool_dir, human48_jsonl=args.human48_jsonl, human48_input=args.human48_input, output_dir=args.output_dir); print(json.dumps(result, ensure_ascii=False, indent=2, default=str)); return 0
    except ContractError as exc:
        print(json.dumps({"status": "TERMINAL_BASE_ENCODER_BLOCKED_OR_REJECTED", "blocker_code": exc.code, "details": exc.details, "training_invoked": False}, ensure_ascii=False, indent=2), file=sys.stderr); return 2


if __name__ == "__main__":
    raise SystemExit(main())
