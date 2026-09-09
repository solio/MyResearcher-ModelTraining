"""M4 one-shot Human100 supervision correction experiment.

The experiment is intentionally narrow: it warm-starts the five selected LERT
specialists from the immutable M2 candidate, appends each reviewed Human100
row once with weight 1.0, and freezes each checkpoint before a single
post-training Human48 diagnostic.  Context and reasoning are carried through
unchanged from the selected runtime and are never fit in this stage.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import random
import shutil
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from . import encoder_m1 as m1
from . import encoder_m2_1_softlabel as m21
from . import encoder_m2_lert_final as lert
from . import encoder_m2_selected_runtime as selected
from .errors import ContractError
from .hashes import content_addressed_id, sha256_file
from .schema import LabelSchema, SINGLE_LABEL_HEADS, V1_HEADS


STAGE_ID = "M4-HUMAN100-SUPERVISION-FIVE-HEAD-SEED35"
RUN_SCHEMA_VERSION = "myresearcher.encoder-m4-human-supervision-run.v1"
MODEL_ID = "hfl/chinese-lert-small"
REVISION = "69e3e69ba258be5b301b26937e5b55a076c90460"
LICENSE = "Apache-2.0"
SEED = 35
TRAINED_HEADS = ("target_mode", "stance", "emotion_primary", "emotion_target", "action_tendency")
REUSED_HEADS = ("context_dependency", "reasoning_tags")
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
BASELINE_CONTENT_ADDRESS = "8f73bc437602ce25d6e8ba0c7fd787676a4b566077b1765f3510739ae85928cf"
EXPECTED_HUMAN48_CORRECT = 86
MIN_HUMAN48_TREATMENT_CORRECT = 96
MIN_HEADS_IMPROVED = 3
MAX_HEAD_DECREASE = 2

REPO = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO.parent
DEFAULT_CACHE = REPO / ".encoder-artifacts/hf-cache-lert-small"
DEFAULT_ARTIFACT = PROJECT_ROOT / "model-artifacts/m2-selected-candidate-frozen-for-m3-20260904"
DEFAULT_OUTPUT = PROJECT_ROOT / "model-artifacts/m4-human-supervision-20260908"
DEFAULT_WORKBOOK = PROJECT_ROOT / "human-review-artifacts/m3b-gold100-20260902/m3b-gold100-review-pack.xlsx"
DEFAULT_REVIEW_EXPORT = Path("/tmp/m3c-shadow500-review50-20260908-all.jsonl")
DEFAULT_SHADOW_INPUT = REPO / "runs/m3c-shadow500-20260907/shadow500-input.jsonl"
CONTRACT_PATH = REPO / "manifests/encoder-m2-experiment-contract-v1.json"
CONFIG_PATH = REPO / "configs/baseline_v0.3.5.yaml"
SCHEMA_PATH = REPO / "schema/semantic-schema-calibrated-v0.2.1.json"
EXCLUDED_REVIEW_IDS = {"P87937c4e6bfab288", "Pb8d1f534480f5653"}

NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _require(ok: bool, code: str, message: str, **details: Any) -> None:
    if not ok:
        raise ContractError(code, message, **details)


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _column_letters(value: str) -> str:
    return "".join(character for character in value if character.isalpha())


def read_human100(path: str | Path, schema: LabelSchema) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Read the supplied workbook without changing it or using reasoning fields."""

    workbook = Path(path).resolve()
    with ZipFile(workbook) as archive:
        shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        shared = ["".join(text.text or "" for text in item.findall(".//a:t", NS)) for item in shared_root.findall("a:si", NS)]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet2.xml"))
        rows: list[dict[str, str]] = []
        for row in sheet.findall(".//a:sheetData/a:row", NS):
            if int(row.attrib["r"]) < 3:
                continue
            values: dict[str, str] = {}
            for cell in row.findall("a:c", NS):
                node = cell.find("a:v", NS)
                if node is None:
                    value = ""
                elif cell.attrib.get("t") == "s":
                    index = int(node.text or "0")
                    value = shared[index] if 0 <= index < len(shared) else ""
                elif cell.attrib.get("t") == "inlineStr":
                    value = "".join(text.text or "" for text in cell.findall(".//a:t", NS))
                else:
                    value = node.text or ""
                values[_column_letters(cell.attrib["r"])] = value
            rows.append(values)
    _require(rows and rows[0].get("A") == "sample_id", "M4_HUMAN100_HEADER_INVALID", "workbook sheet header is invalid")
    columns = sorted(rows[0], key=lambda value: (len(value), value))
    headers = [rows[0][column] for column in columns]
    records = [{headers[index]: row.get(column, "") for index, column in enumerate(columns)} for row in rows[1:]]
    _require(len(records) == 100, "M4_HUMAN100_ROW_COUNT_INVALID", "Human workbook must contain exactly 100 rows", observed=len(records))
    required = {"sample_id", "stock_code", "stock_name", "model_text", "human_context_dependency", "human_reasoning_tags", *(f"human_{head}" for head in TRAINED_HEADS)}
    _require(required <= set(headers), "M4_HUMAN100_COLUMNS_INVALID", "Human workbook is missing required columns", missing=sorted(required - set(headers)))
    _require(len({row["sample_id"] for row in records}) == 100 and all(row["sample_id"].strip() and row["model_text"].strip() for row in records), "M4_HUMAN100_ID_OR_TEXT_INVALID", "Human rows require unique IDs and nonblank model_text")
    _require(all(not row["human_reasoning_tags"].strip() for row in records), "M4_HUMAN100_REASONING_NOT_ALLOWED", "human_reasoning_tags must be empty and ignored")
    for row in records:
        for head in TRAINED_HEADS:
            _require(row[f"human_{head}"] in schema.class_order[head], "M4_HUMAN100_LABEL_INVALID", "Human label is outside frozen class order", sample_id=row["sample_id"], head=head, label=row[f"human_{head}"])
    audit = {"path": str(workbook), "sha256": sha256_file(workbook), "rows": len(records), "training_heads": list(TRAINED_HEADS), "human_context_dependency": "ignored; source field is a primary reasoning tag", "human_reasoning_tags": "ignored_and_empty", "shared_string_empty_values": "empty strings remain empty"}
    return records, audit


def read_review_export(path: str | Path, shadow_input: str | Path, schema: LabelSchema) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read the full shared export and join only 48 eligible samples by sample_id."""

    export = Path(path).resolve()
    raw_rows = []
    for line_number, line in enumerate(export.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError("M4_REVIEW_EXPORT_INVALID", "review export contains invalid JSON", line=line_number) from exc
        _require(isinstance(row, Mapping), "M4_REVIEW_EXPORT_INVALID", "review export rows must be objects", line=line_number)
        raw_rows.append(dict(row))
    _require(len(raw_rows) == 350, "M4_REVIEW_EXPORT_INCOMPLETE", "full 50x7 shared export is required", rows=len(raw_rows))
    _require({row.get("batch") for row in raw_rows} == {"m3c-shadow500-review50-20260908"}, "M4_REVIEW_EXPORT_BATCH_INVALID", "unexpected review batch")
    expected_heads = set(V1_HEADS)
    by_sample: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in raw_rows:
        sample_id, head = row.get("sample_id"), row.get("head")
        _require(isinstance(sample_id, str) and sample_id and head in expected_heads, "M4_REVIEW_EXPORT_ROW_INVALID", "review row identity is invalid")
        # The two owner-declared EXCLUDE_SEMANTIC samples may retain their
        # terminal ``无法判断`` disposition and therefore are not final.  They
        # are excluded as whole samples below; every eligible row must be
        # final.
        if sample_id not in EXCLUDED_REVIEW_IDS:
            _require(row.get("is_final") is True, "M4_REVIEW_EXPORT_NOT_FINAL", "all eligible review rows must be final", sample_id=sample_id, head=head)
        _require(sample_id not in by_sample or head not in by_sample[sample_id], "M4_REVIEW_EXPORT_DUPLICATE", "duplicate sample/head row", sample_id=sample_id, head=head)
        by_sample.setdefault(sample_id, {})[head] = row
    _require(len(by_sample) == 50 and all(set(heads) == expected_heads for heads in by_sample.values()), "M4_REVIEW_EXPORT_COVERAGE_INVALID", "export must cover 50 samples and seven heads")
    inputs: dict[str, Mapping[str, Any]] = {}
    for line in Path(shadow_input).resolve().read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            inputs[str(item["sample_id"])] = item
    eligible: list[dict[str, Any]] = []
    for sample_id, heads in by_sample.items():
        if sample_id in EXCLUDED_REVIEW_IDS:
            continue
        _require(sample_id in inputs, "M4_REVIEW_INPUT_MISSING", "eligible review sample is missing from immutable Shadow input", sample_id=sample_id)
        values: dict[str, Any] = {"sample_id": sample_id, "stock_code": str(inputs[sample_id].get("stock_code", "")), "stock_name": str(inputs[sample_id].get("stock_name", "")), "model_text": str(inputs[sample_id].get("model_text", ""))}
        _require(values["model_text"].strip(), "M4_REVIEW_INPUT_TEXT_EMPTY", "review input text is empty", sample_id=sample_id)
        for head in TRAINED_HEADS:
            answer = heads[head].get("answer")
            _require(isinstance(answer, str) and answer in schema.class_order[head], "M4_REVIEW_LABEL_INVALID", "review answer is invalid", sample_id=sample_id, head=head, answer=answer)
            _require(heads[head].get("disposition") in (None, ""), "M4_REVIEW_DISPOSITION_NONEMPTY", "eligible sample has a non-final disposition", sample_id=sample_id, head=head)
            values[f"human_{head}"] = answer
        # The exported context/reasoning answers are intentionally never read.
        eligible.append(values)
    _require(len(eligible) == 48, "M4_REVIEW_ELIGIBLE_COUNT_INVALID", "exactly 48 non-excluded review samples are required", observed=len(eligible))
    audit = {"path": str(export), "sha256": sha256_file(export), "rows": len(raw_rows), "sample_count": 50, "eligible_count": len(eligible), "excluded_sample_ids": sorted(EXCLUDED_REVIEW_IDS), "joined_shadow_input_sha256": sha256_file(Path(shadow_input).resolve()), "training_or_selection_use": False, "heads_read": list(TRAINED_HEADS), "context_and_reasoning_fields": "not read"}
    return eligible, audit


def human_record(row: Mapping[str, Any], schema: LabelSchema) -> m1.M1Record:
    """Create an M1-shaped record while keeping untrained fields inert."""

    labels: dict[str, Any] = {head: str(row[f"human_{head}"]) for head in TRAINED_HEADS}
    labels["context_dependency"] = schema.class_order["context_dependency"][0]
    labels["reasoning_tags"] = [schema.class_order["reasoning_tags"][0]]
    weights = {head: 1.0 for head in V1_HEADS}
    return m1.M1Record(sample_id=str(row["sample_id"]), stock_code=str(row.get("stock_code", "")), stock_name=str(row.get("stock_name", "")), model_text=str(row["model_text"]), label=labels, weights=weights)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "M4_JSON_INVALID", "JSON object required", path=str(path))
    return value


def strict_metrics(probabilities: np.ndarray, truth: Sequence[str], order: Sequence[str]) -> dict[str, Any]:
    predicted = [order[int(index)] for index in np.argmax(probabilities, axis=1)]
    precision, recall, f1, support = precision_recall_fscore_support(truth, predicted, labels=list(order), zero_division=0)
    supported = [index for index, count in enumerate(support) if count > 0]
    return {"accuracy": float(np.mean(np.asarray(predicted) == np.asarray(truth))), "macro_f1_supported": float(np.mean(f1[supported])) if supported else 0.0, "macro_f1_schema_all": float(np.mean(f1)), "per_class": {label: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])} for i, label in enumerate(order)}, "confusion_matrix": confusion_matrix(truth, predicted, labels=list(order)).tolist(), "class_order": list(order), "predictions": predicted}


def _probability_rows(predictions: Sequence[Mapping[str, Any]], head: str, order: Sequence[str]) -> np.ndarray:
    return np.asarray([[float(row["predictions"][head]["probabilities"][label]) for label in order] for row in predictions], dtype=float)


def evaluate_five(probabilities: Mapping[str, np.ndarray], rows: Sequence[Mapping[str, Any]], schema: LabelSchema) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for head in TRAINED_HEADS:
        truth = [str(row[f"human_{head}"]) for row in rows]
        metrics = strict_metrics(probabilities[head], truth, schema.class_order[head])
        metrics["correct_count"] = int(round(metrics["accuracy"] * len(rows)))
        result[head] = metrics
    result["total_correct"] = int(sum(item["correct_count"] for item in result.values()))
    return result


def human_gate(baseline: Mapping[str, Any], treatment: Mapping[str, Any]) -> dict[str, Any]:
    baseline_counts = {head: int(baseline[head]["correct_count"]) for head in TRAINED_HEADS}
    treatment_counts = {head: int(treatment[head]["correct_count"]) for head in TRAINED_HEADS}
    deltas = {head: treatment_counts[head] - baseline_counts[head] for head in TRAINED_HEADS}
    checks = {"treatment_total_at_least_96_of_240": sum(treatment_counts.values()) >= MIN_HUMAN48_TREATMENT_CORRECT, "at_least_3_of_5_heads_improve": sum(delta > 0 for delta in deltas.values()) >= MIN_HEADS_IMPROVED, "no_head_decreases_more_than_2": all(delta >= -MAX_HEAD_DECREASE for delta in deltas.values()), "baseline_total_is_86_of_240": sum(baseline_counts.values()) == EXPECTED_HUMAN48_CORRECT}
    return {"passed": bool(all(checks.values())), "checks": checks, "baseline_correct": baseline_counts, "treatment_correct": treatment_counts, "correct_count_delta": deltas, "baseline_total": sum(baseline_counts.values()), "treatment_total": sum(treatment_counts.values()), "scope": "Human48 post-freeze diagnostic only"}


def _set_seed(torch: Any, np_module: Any, seed: int) -> None:
    random.seed(seed)
    np_module.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.mps.manual_seed(seed)
    except Exception:
        pass


def _limits(started: float, root: Path, phase: str) -> None:
    _require(time.monotonic() - started <= WALL_TIME_SECONDS, "M4_WALL_TIME_LIMIT_EXCEEDED", "M4 two-hour limit exceeded", phase=phase)
    _require(m1._directory_size(root) <= MAX_DISK_GIB * 1024**3, "M4_DISK_LIMIT_EXCEEDED", "M4 ten GiB output limit exceeded", phase=phase)


def _load_bundle(path: Path, torch: Any) -> Mapping[str, Any]:
    try:
        bundle = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        bundle = torch.load(path, map_location="cpu")
    _require(isinstance(bundle, Mapping), "M4_BASELINE_BUNDLE_INVALID", "selected bundle must be a mapping")
    identity = bundle.get("model_identity")
    _require(isinstance(identity, Mapping) and identity.get("model_id") == MODEL_ID and identity.get("revision") == REVISION and identity.get("license") == LICENSE, "M4_BASELINE_MODEL_IDENTITY_INVALID", "selected bundle model identity changed")
    _require(bundle.get("selected_candidate") is True, "M4_BASELINE_NOT_SELECTED", "baseline artifact is not the frozen selected candidate")
    for key in ("lert_specialist_block_state_dict", "lert_heads_state_dict"):
        _require(set(bundle.get(key, {})) == set(selected.LERT_HEADS), "M4_BASELINE_BUNDLE_INCOMPLETE", "selected bundle is missing a LERT head", key=key)
    return bundle


def _load_model_for_head(torch: Any, AutoModel: Any, snapshot: Path, schema: LabelSchema, config: Mapping[str, Any], bundle: Mapping[str, Any], head: str, device: Any) -> Any:
    model = lert._make_model(torch, AutoModel, snapshot, schema, float(config["head_dropout"]))
    model.specialist_blocks[head].load_state_dict(bundle["lert_specialist_block_state_dict"][head])
    model.heads[head].load_state_dict(bundle["lert_heads_state_dict"][head])
    model.set_active_head(head)
    return model.to(device)


def _predict_head(torch: Any, model: Any, tokenizer: Any, records: Sequence[m1.M1Record], config: Mapping[str, Any], head: str, device: Any) -> np.ndarray:
    output: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(records), BATCH_SIZE):
            batch = m1._as_batch(torch, tokenizer, records[offset : offset + BATCH_SIZE], config, device)
            logits = model(batch["input_ids"], batch["attention_mask"])[head]
            _require(bool(torch.isfinite(logits).all().item()), "M4_NONFINITE_LOGITS", "non-finite logits during prediction", head=head)
            output.append(torch.softmax(logits, dim=-1).detach().cpu().numpy())
    return np.concatenate(output, axis=0) if output else np.empty((0, len(config["class_order"][head])))


def _checkpoint_for_head(folder: Path, torch: Any, model: Any, head: str, seed: int, config: Mapping[str, Any], provenance: Mapping[str, Any]) -> None:
    torch.save({"run_schema_version": RUN_SCHEMA_VERSION, "stage_id": STAGE_ID, "head": head, "seed": seed, "frozen_config": dict(config), "provenance": dict(provenance), "specialist_block_state_dict": {k: v.detach().cpu() for k, v in model.specialist_blocks[head].state_dict().items()}, "head_state_dict": {k: v.detach().cpu() for k, v in model.heads[head].state_dict().items()}}, folder / "human-supervision-checkpoint.pt")


def _cpu_reload_smoke(torch: Any, AutoModel: Any, snapshot: Path, schema: LabelSchema, config: Mapping[str, Any], checkpoint: Mapping[str, Any], head: str, tokenizer: Any, record: m1.M1Record) -> dict[str, Any]:
    model = lert._make_model(torch, AutoModel, snapshot, schema, float(config["head_dropout"]))
    model.specialist_blocks[head].load_state_dict(checkpoint["specialist_block_state_dict"])
    model.heads[head].load_state_dict(checkpoint["head_state_dict"])
    model = model.to(torch.device("cpu")); model.eval()
    batch = m1._as_batch(torch, tokenizer, [record], config, torch.device("cpu"))
    with torch.no_grad():
        logits = model(batch["input_ids"], batch["attention_mask"])[head]
    finite = bool(torch.isfinite(logits).all().item())
    _require(finite, "M4_CPU_RELOAD_FAILED", "CPU reload output is non-finite", head=head)
    return {"required": True, "device": "cpu", "head": head, "sample_id": record.sample_id, "finite": finite, "output_shape": list(logits.shape)}


def _train_head(*, runtime: tuple[Any, Any, Any, Any], root: Path, snapshot: Path, schema: LabelSchema, combined_train: Sequence[m1.M1Record], dev: Sequence[m1.M1Record], config: Mapping[str, Any], bundle: Mapping[str, Any], provenance: Mapping[str, Any], head: str, started: float, cache_dir: Path) -> dict[str, Any]:
    np_module, torch, AutoModel, AutoTokenizer = runtime
    seed_started = time.monotonic()
    folder = root / head / f"seed-{SEED}"
    folder.mkdir(parents=True, exist_ok=False)
    _set_seed(torch, np_module, SEED)
    device, device_name, mps_available = m1._choose_device(torch)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True)
    model = _load_model_for_head(torch, AutoModel, snapshot, schema, config, bundle, head, device)
    expected = {name for name, parameter in model.named_parameters() if name.startswith(f"specialist_blocks.{head}.") or name.startswith(f"heads.{head}.")}
    actual = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    _require(actual == expected and expected, "M4_TRAINABLE_SCOPE_INVALID", "only selected block and head may train", head=head)
    block_params = list(model.specialist_blocks[head].parameters()); head_params = list(model.heads[head].parameters()); trainable = [*block_params, *head_params]
    optimizer = torch.optim.AdamW([{"params": block_params, "lr": BLOCK_LR}, {"params": head_params, "lr": HEAD_LR}], weight_decay=WEIGHT_DECAY, betas=tuple(config["optimizer"]["betas"]), eps=float(config["optimizer"]["epsilon"]))
    class_weights = None
    if head == "action_tendency":
        counts = Counter(str(item.label[head]) for item in combined_train[: len(combined_train) - 100])
        # M2.2's Train-only balancing remains frozen; Human100 is not used to derive weights.
        train_count = len(combined_train) - 100
        class_weights = {label: min(4.0, max(0.5, math.sqrt(train_count / (len(schema.class_order[head]) * counts.get(label, 0))))) if counts.get(label, 0) else 4.0 for label in schema.class_order[head]}
        class_weight_tensor = torch.tensor([class_weights[label] for label in schema.class_order[head]], dtype=torch.float32, device=device)
    best_score, best_epoch, stale = -math.inf, 0, 0
    log = folder / "training-log.jsonl"
    for epoch in range(1, MAX_EPOCHS + 1):
        _limits(started, root, f"before_epoch_{head}")
        model.train(True)
        shuffled = list(combined_train); random.Random(SEED + epoch).shuffle(shuffled); losses: list[float] = []
        for offset in range(0, len(shuffled), BATCH_SIZE):
            _limits(started, root, f"during_epoch_{head}")
            batch_records = shuffled[offset : offset + BATCH_SIZE]
            batch = m1._as_batch(torch, tokenizer, batch_records, config, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch["attention_mask"])[head]
            _require(bool(torch.isfinite(logits).all().item()), "M4_NONFINITE_LOGITS", "non-finite training logits", head=head)
            raw = torch.nn.functional.cross_entropy(logits, batch["labels"][head], reduction="none")
            if class_weights is not None:
                raw = raw * class_weight_tensor[batch["labels"][head]]
            weights = batch["weights"][head]
            loss = (raw * weights).sum() / weights.sum().clamp_min(1e-12)
            _require(bool(torch.isfinite(loss).item()), "M4_NONFINITE_LOSS", "non-finite training loss", head=head)
            loss.backward()
            for parameter in trainable:
                if parameter.grad is not None:
                    _require(bool(torch.isfinite(parameter.grad).all().item()), "M4_NONFINITE_GRADIENT", "non-finite gradient", head=head)
            torch.nn.utils.clip_grad_norm_(trainable, GRADIENT_CLIP)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        dev_prob = _predict_head(torch, model, tokenizer, dev, config, head, device)
        dev_truth = [str(item.label[head]) for item in dev]
        score = float(strict_metrics(dev_prob, dev_truth, schema.class_order[head])["macro_f1_schema_all"])
        improved = score > best_score
        if improved:
            best_score, best_epoch, stale = score, epoch, 0
            _checkpoint_for_head(folder, torch, model, head, SEED, config, provenance)
        else:
            stale += 1
        m1._jsonl_append(log, {"head": head, "seed": SEED, "epoch": epoch, "train_weighted_loss": float(sum(losses) / len(losses)), "dev_macro_f1_schema_all": score, "improved": improved, "stale_epochs": stale})
        if stale >= PATIENCE:
            break
    checkpoint_path = folder / "human-supervision-checkpoint.pt"
    _require(checkpoint_path.is_file(), "M4_CHECKPOINT_MISSING", "best checkpoint was not created", head=head)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    best_model = _load_model_for_head(torch, AutoModel, snapshot, schema, config, bundle, head, device)
    best_model.specialist_blocks[head].load_state_dict(checkpoint["specialist_block_state_dict"]); best_model.heads[head].load_state_dict(checkpoint["head_state_dict"])
    train_prob = _predict_head(torch, best_model, tokenizer, combined_train, config, head, device); dev_prob = _predict_head(torch, best_model, tokenizer, dev, config, head, device)
    train_metric = strict_metrics(train_prob, [str(item.label[head]) for item in combined_train], schema.class_order[head]); dev_metric = strict_metrics(dev_prob, [str(item.label[head]) for item in dev], schema.class_order[head])
    smoke = _cpu_reload_smoke(torch, AutoModel, snapshot, schema, config, checkpoint, head, tokenizer, dev[0])
    _dump(folder / "seed-metrics.json", {"head": head, "seed": SEED, "best_epoch": best_epoch, "early_stopping_metric": f"{head}.macro_f1_schema_all", "train_population": "Train_1822_plus_Human100_once", "dev_role": "early_stopping_and_diagnostic_only", "train": train_metric, "dev": dev_metric, "cpu_reload_inference_smoke": smoke})
    resource = {"head": head, "seed": SEED, "actual_device": device_name, "mps_available": mps_available, "elapsed_seconds": round(time.monotonic() - seed_started, 3), "cache_bytes": m1._directory_size(cache_dir), "artifact_bytes": m1._directory_size(folder), "checkpoint_bytes": checkpoint_path.stat().st_size, "cpu_reload_result": smoke, "python": {"implementation": platform.python_implementation(), "version": platform.python_version()}, "logical_cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(), "class_weights": class_weights}
    _dump(folder / "resource-log.json", resource)
    return {"head": head, "seed": SEED, "checkpoint_sha256": sha256_file(checkpoint_path), "metrics": {"train": train_metric, "dev": dev_metric}, "resource": resource, "checkpoint_path": str(checkpoint_path)}


def _build_treatment_bundle(parent: Mapping[str, Any], results: Sequence[Mapping[str, Any]], torch: Any, output: Path) -> dict[str, Any]:
    bundle = copy.deepcopy(dict(parent))
    for result in results:
        checkpoint = torch.load(result["checkpoint_path"], map_location="cpu", weights_only=True)
        head = result["head"]
        bundle["lert_specialist_block_state_dict"][head] = checkpoint["specialist_block_state_dict"]
        bundle["lert_heads_state_dict"][head] = checkpoint["head_state_dict"]
    bundle.update({"stage": STAGE_ID, "status": "HUMAN_SUPERVISION_TREATMENT_DIAGNOSTIC", "selected_candidate": False, "parent_selected_artifact_content_address": BASELINE_CONTENT_ADDRESS, "trained_heads": list(TRAINED_HEADS), "reused_heads": list(REUSED_HEADS)})
    path = output / "unified-treatment-runtime-bundle.pt"
    torch.save(bundle, path)
    return {"path": str(path), "sha256": sha256_file(path), "trained_heads": list(TRAINED_HEADS), "reused_heads": list(REUSED_HEADS)}


def load_treatment_runtime(*, baseline_artifact: str | Path, treatment_artifact: str | Path, config_path: str | Path = CONFIG_PATH, cache_dir: str | Path = DEFAULT_CACHE) -> Any:
    """Load the seven-head treatment runtime without changing either artifact.

    The existing selected runtime owns input construction and Classical
    fallback mapping.  We reuse that implementation and replace only the five
    in-memory LERT specialist states from the new immutable treatment bundle;
    context and reasoning (including the Classical SARCASM_IRONY fallback)
    therefore remain byte-identical to the selected candidate.
    """

    np_module, torch, _, _ = m1._load_runtime_dependencies()
    del np_module
    treatment = Path(treatment_artifact).resolve()
    bundle_path = treatment / "unified-treatment-runtime-bundle.pt"
    _require(bundle_path.is_file(), "M4_TREATMENT_BUNDLE_MISSING", "treatment runtime bundle is missing")
    try:
        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    except TypeError:
        bundle = torch.load(bundle_path, map_location="cpu")
    _require(isinstance(bundle, Mapping) and bundle.get("status") == "HUMAN_SUPERVISION_TREATMENT_DIAGNOSTIC", "M4_TREATMENT_BUNDLE_INVALID", "treatment bundle status is invalid")
    runtime = selected.UnifiedCPUInferenceRuntime.from_artifact(baseline_artifact, config_path=config_path, cache_dir=cache_dir)
    for head in TRAINED_HEADS:
        runtime.model.specialist_blocks[head].load_state_dict(bundle["lert_specialist_block_state_dict"][head])
        runtime.model.heads[head].load_state_dict(bundle["lert_heads_state_dict"][head])
    runtime.model.eval()
    return runtime


def run_m4(config_path: str | Path = CONFIG_PATH, *, cache_dir: str | Path = DEFAULT_CACHE, baseline_artifact: str | Path = DEFAULT_ARTIFACT, workbook: str | Path = DEFAULT_WORKBOOK, review_export: str | Path = DEFAULT_REVIEW_EXPORT, shadow_input: str | Path = DEFAULT_SHADOW_INPUT, output_dir: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    _require(not output.exists(), "M4_OUTPUT_ALREADY_EXISTS", "M4 output directory must be new", path=str(output))
    started = time.monotonic()
    project = m1.ProjectConfig.load(config_path)
    schema, train, dev = m1.load_m1_partitions(project)
    _require(len(train) == 1822 and len(dev) == 448, "M4_TRAIN_DEV_SCOPE_INVALID", "expected Train 1822 and Dev 448")
    human_rows, human_audit = read_human100(workbook, schema)
    review_rows, review_audit = read_review_export(review_export, shadow_input, schema)
    _require(not ({row["sample_id"] for row in human_rows} & {item.sample_id for item in train + dev}), "M4_HUMAN_SPLIT_OVERLAP", "Human100 overlaps Train/Dev")
    human_records = [human_record(row, schema) for row in human_rows]
    combined = [*train, *human_records]
    runtime = m1._load_runtime_dependencies()
    np_module, torch, AutoModel, AutoTokenizer = runtime
    bundle_path = Path(baseline_artifact).resolve() / "unified-runtime-bundle.pt"
    selected._verify_manifest(Path(baseline_artifact).resolve(), expected_content_address=BASELINE_CONTENT_ADDRESS)
    parent = _load_bundle(bundle_path, torch)
    snapshot = m21._snapshot(Path(cache_dir).resolve())[0]
    frozen = lert._contract_requirements(CONTRACT_PATH)
    config = lert._config(frozen["contract"], schema.class_order)
    config.update({"stage_id": STAGE_ID, "run_schema_version": RUN_SCHEMA_VERSION, "seed": SEED, "train_population": "Train_1822_plus_Human100_once", "human_supervision_weight": 1.0, "human_oversampling": False, "human_training_heads": list(TRAINED_HEADS), "reused_heads": list(REUSED_HEADS), "dev_role": "early_stopping_and_diagnostic_only", "review48_role": "single_post_freeze_diagnostic_only", "selected_candidate": False, "action_class_weights_population": "Train_1822_only"})
    output.mkdir(parents=True, exist_ok=False)
    classical_destination = output / "classical-fallback" / "original_model_v0.3.5.joblib"
    classical_destination.parent.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(baseline_artifact).resolve() / "classical-fallback" / "original_model_v0.3.5.joblib", classical_destination)
    _dump(output / "training-config.json", config)
    provenance = {"stage_id": STAGE_ID, "model_identity": {"model_id": MODEL_ID, "revision": REVISION, "license": LICENSE}, "parent_selected_artifact": {"path": str(Path(baseline_artifact).resolve()), "content_address": BASELINE_CONTENT_ADDRESS, "manifest_sha256": sha256_file(Path(baseline_artifact).resolve() / "content-addressed-manifest.json")}, "cache_snapshot": m21._snapshot(Path(cache_dir).resolve())[1], "human100": human_audit, "review48": review_audit, "data_scope": {"train": 1822, "dev": 448, "human100_fit": 100, "human48_diagnostic": 48}, "forbidden_inputs": ["Test", "Anchor", "OOD", "LLM", "cloud", "production"]}
    _dump(output / "provenance.json", provenance)
    results: list[dict[str, Any]] = []
    for head in TRAINED_HEADS:
        results.append(_train_head(runtime=runtime, root=output, snapshot=snapshot, schema=schema, combined_train=combined, dev=dev, config=config, bundle=parent, provenance=provenance, head=head, started=started, cache_dir=Path(cache_dir).resolve()))
    # Human48 is deliberately evaluated only after all five checkpoints are
    # frozen.  No model prediction is made on this diagnostic population
    # during fit or checkpoint selection.
    selected_runtime = selected.UnifiedCPUInferenceRuntime.from_artifact(baseline_artifact, config_path=config_path, cache_dir=cache_dir)
    review_records = [selected.InferenceRecord(sample_id=row["sample_id"], stock_code=row["stock_code"], stock_name=row["stock_name"], model_text=row["model_text"]) for row in review_rows]
    baseline_predictions = selected_runtime.predict_records(review_records)
    baseline_prob = {head: _probability_rows(baseline_predictions, head, schema.class_order[head]) for head in TRAINED_HEADS}
    baseline_metrics = evaluate_five(baseline_prob, review_rows, schema)
    _require(baseline_metrics["total_correct"] == EXPECTED_HUMAN48_CORRECT, "M4_BASELINE_HUMAN48_MISMATCH", "frozen baseline does not reproduce expected 86/240 Human48 metric", observed=baseline_metrics["total_correct"])
    _dump(output / "human48-baseline-metrics.json", baseline_metrics)
    treatment_prob: dict[str, np.ndarray] = {}
    for result in results:
        head = result["head"]
        checkpoint = torch.load(result["checkpoint_path"], map_location="cpu", weights_only=True)
        model = _load_model_for_head(torch, AutoModel, snapshot, schema, config, parent, head, torch.device("cpu"))
        model.specialist_blocks[head].load_state_dict(checkpoint["specialist_block_state_dict"]); model.heads[head].load_state_dict(checkpoint["head_state_dict"])
        tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True)
        treatment_prob[head] = _predict_head(torch, model, tokenizer, [human_record(row, schema) for row in review_rows], config, head, torch.device("cpu"))
    treatment_metrics = evaluate_five(treatment_prob, review_rows, schema)
    gate = human_gate(baseline_metrics, treatment_metrics)
    _dump(output / "human48-treatment-metrics.json", treatment_metrics)
    _dump(output / "human48-gate.json", gate)
    bundle = _build_treatment_bundle(parent, results, torch, output)
    _dump(output / "unified-treatment-bundle-evidence.json", bundle)
    treatment_runtime = load_treatment_runtime(baseline_artifact=baseline_artifact, treatment_artifact=output, config_path=config_path, cache_dir=cache_dir)
    smoke_rows = treatment_runtime.predict_records([review_records[0]])
    smoke = smoke_rows[0]["predictions"]
    smoke_shapes = {head: len(values["probabilities"]) for head, values in smoke.items()}
    _require(set(smoke_shapes) == set(V1_HEADS) and all(size > 0 for size in smoke_shapes.values()), "M4_UNIFIED_RUNTIME_SMOKE_FAILED", "treatment runtime did not emit all seven heads")
    _dump(output / "unified-runtime-smoke.json", {"required": True, "device": "cpu", "all_seven_heads": True, "output_shapes": smoke_shapes, "finite": True, "reused_heads": {head: "unchanged_from_selected_candidate" for head in REUSED_HEADS}})
    final_status = "HUMAN_SUPERVISION_POSITIVE_DIAGNOSTIC" if gate["passed"] else "HUMAN_SUPERVISION_NO_GAIN_DIAGNOSTIC"
    reused_consistency = {"context_dependency": "REUSED_FROM_SELECTED_CANDIDATE_UNCHANGED", "reasoning_tags": "REUSED_FROM_SELECTED_CANDIDATE_UNCHANGED", "training_labels_read": False, "improvement_gate_included": False}
    aggregate = {"stage_id": STAGE_ID, "seed": SEED, "trained_heads": list(TRAINED_HEADS), "reused_heads": list(REUSED_HEADS), "all_five_checkpoints_frozen": True, "baseline_human48_correct": EXPECTED_HUMAN48_CORRECT, "treatment_human48_correct": treatment_metrics["total_correct"], "gate": gate, "reused_head_consistency": reused_consistency, "status": final_status, "selected_candidate": False, "production_approval": False}
    _dump(output / "stage-aggregate.json", aggregate)
    # Write the human-readable report before hashing the evidence tree.  It
    # refers to the adjacent manifest rather than embedding a self-referential
    # content address.
    report = _render_report(final_status, aggregate, baseline_metrics, treatment_metrics, results, output, "see content-addressed-manifest.json")
    (output / "report.md").write_text(report, encoding="utf-8")
    manifest = m1._write_content_manifest(output, {"manifest_schema_version": "myresearcher.encoder-m4-human-supervision-artifact-manifest.v1", "stage_id": STAGE_ID, "status": final_status, "selected_candidate": False, "production_approval": False, "model_identity": provenance["model_identity"], "parent_selected_artifact_content_address": BASELINE_CONTENT_ADDRESS, "train_rows": 1822, "dev_rows": 448, "human100_fit_rows": 100, "human48_diagnostic_rows": 48, "human100_weight": 1.0, "human_oversampling": False, "trained_heads": list(TRAINED_HEADS), "reused_heads": list(REUSED_HEADS), "checkpoint_sha256": {item["head"]: item["checkpoint_sha256"] for item in results}, "human48_gate_sha256": sha256_file(output / "human48-gate.json"), "unified_runtime_smoke_sha256": sha256_file(output / "unified-runtime-smoke.json"), "reused_head_consistency": reused_consistency, "forbidden_inputs": provenance["forbidden_inputs"]})
    _limits(started, output, "after_final_manifest")
    return {"status": final_status, "output_dir": str(output), "content_address": manifest["content_address"], "aggregate": aggregate, "baseline": baseline_metrics, "treatment": treatment_metrics, "results": results}


def _render_report(status: str, aggregate: Mapping[str, Any], baseline: Mapping[str, Any], treatment: Mapping[str, Any], results: Sequence[Mapping[str, Any]], output: Path, content_address: str) -> str:
    lines = ["# M4 Human100 supervision correction diagnostic", "", f"Status: **{status}**", "", "This is one seed-35 treatment. Human100 was appended once with weight 1.0 to five heads; context_dependency and reasoning_tags were reused unchanged. Human48 was read only after all five checkpoints were frozen and was not used for fitting, early stopping, threshold, or checkpoint selection.", "", f"Artifact: `{output}`", f"Content address: `{content_address}`", "", "| head | baseline correct / 48 | treatment correct / 48 | delta |", "|---|---:|---:|---:|"]
    for head in TRAINED_HEADS:
        delta = treatment[head]["correct_count"] - baseline[head]["correct_count"]
        lines.append(f"| {head} | {baseline[head]['correct_count']} | {treatment[head]['correct_count']} | {delta:+d} |")
    lines += [f"| **total / 240** | **{baseline['total_correct']}** | **{treatment['total_correct']}** | **{treatment['total_correct'] - baseline['total_correct']:+d}** |", "", f"Gate: **{'PASS' if aggregate['gate']['passed'] else 'FAIL'}**", "", "No Test, Anchor, OOD, LLM, cloud, or production data were read. A pass is diagnostic only and does not alter the frozen seven-head candidate.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the one-shot M4 Human100 supervision correction diagnostic")
    parser.add_argument("--config", default=str(CONFIG_PATH)); parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE)); parser.add_argument("--baseline-artifact", default=str(DEFAULT_ARTIFACT)); parser.add_argument("--workbook", default=str(DEFAULT_WORKBOOK)); parser.add_argument("--review-export", required=True); parser.add_argument("--shadow-input", default=str(DEFAULT_SHADOW_INPUT)); parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    try:
        result = run_m4(args.config, cache_dir=args.cache_dir, baseline_artifact=args.baseline_artifact, workbook=args.workbook, review_export=args.review_export, shadow_input=args.shadow_input, output_dir=args.output_dir)
    except ContractError as exc:
        print(json.dumps({"status": "M4_BLOCKED_OR_REJECTED", "blocker_code": exc.code, "details": exc.details, "training_invoked": False}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
