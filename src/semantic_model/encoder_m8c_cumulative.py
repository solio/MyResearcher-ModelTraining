"""M8C cumulative auxiliary training and fresh-pool shadow diagnostics.

This module joins the already-final M7B assignments with the newly-final M8B
assignments, fits the five v0.3 auxiliary heads once from a deterministic
random initialization, and then runs a read-only shadow pass over the
DataClean round-004 fresh pool.  The selected seven-head runtime and encoder
remain frozen; this artifact is diagnostic/shadow evidence only.

The code deliberately does not read Test, Gold, OOD or reference predictions.
Fresh-pool eligibility is based only on the explicit text cues shared with the
M7A selector.  In particular, it does not apply M8A's historical ``length >=
39`` exclusion to round-004 content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .auxiliary_heads import (
    AuxiliaryHeadModel,
    AuxiliaryTargetBatch,
    build_auxiliary_targets,
)
from .encoder_m6c_auxiliary import (
    DEFAULT_CANDIDATE_SCHEMA,
    AuxiliaryTrainingRuntime,
    load_auxiliary_task_views,
)
from .encoder_m7b_auxiliary import (
    _clone_state,
    _fit_cached_heads,
    deterministic_folds,
    metric_for_factor,
    read_jsonl,
    run_oof,
)
from .errors import ContractError
from .schema import SemanticSchemaCandidate, V03_AUXILIARY_FACTORS

try:
    from tools.feedback_loop_m7a import _applicable_factors
except ModuleNotFoundError:  # pragma: no cover - direct module invocation
    from ..tools.feedback_loop_m7a import _applicable_factors  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT.parent
DEFAULT_M7B_ARTIFACT = PROJECT_ROOT / "model-artifacts" / "m7b-v03-auxiliary-diagnostic-20260913"
DEFAULT_M7B_TASKS = DEFAULT_M7B_ARTIFACT / "tasks-snapshot.jsonl"
DEFAULT_M7B_ASSIGNMENTS = DEFAULT_M7B_ARTIFACT / "assignments-used-92.jsonl"
DEFAULT_M7B_CHECKPOINT = DEFAULT_M7B_ARTIFACT / "auxiliary-checkpoint.pt"
DEFAULT_M8B_TASKS = PROJECT_ROOT / "human-review-artifacts" / "m8b-v03-shadow-review32-20260914" / "auxiliary-tasks.jsonl"
DEFAULT_M8B_ASSIGNMENTS = REPO_ROOT / "runs" / "m8c-m8b-assignments-pulled-20260914.jsonl"
DEFAULT_FRESH_POOL = PROJECT_ROOT / "MyResearcher-DataClean" / "artifacts" / "round-004" / "fresh-pool-2000.jsonl"
DEFAULT_ARTIFACT = PROJECT_ROOT / "model-artifacts" / "m8c-v03-cumulative-auxiliary-20260914"
DEFAULT_SELECTED_ARTIFACT = PROJECT_ROOT / "model-artifacts" / "m2-selected-candidate-frozen-for-m3-20260904"
DEFAULT_CACHE = REPO_ROOT / ".encoder-artifacts" / "hf-cache-lert-small"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "baseline_v0.3.5.yaml"
SEED = 20260913
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 0.0
M8B_COUNTS = {
    "expectation_phase": 6,
    "arousal_level": 6,
    "desired_price_direction": 6,
    "market_scope": 6,
    "discourse_focus": 8,
}
EXPECTED_CUMULATIVE_ASSIGNMENTS = 124
EXPECTED_CUMULATIVE_MASKS = {
    "expectation_phase": 30,
    "arousal_level": 13,
    "desired_price_direction": 28,
    "market_scope": 14,
    "discourse_focus": 39,
}


def _require(ok: bool, code: str, message: str, **details: Any) -> None:
    if not ok:
        raise ContractError(code, message, **details)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def merge_task_rows(task_rows_by_source: Sequence[Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    """Union task declarations for duplicate sample IDs without duplicating rows."""

    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for source_rows in task_rows_by_source:
        for row in source_rows:
            _require(isinstance(row, Mapping), "M8C_TASK_INVALID", "task row must be an object")
            sample_id = row.get("sample_id")
            _require(isinstance(sample_id, str) and sample_id, "M8C_TASK_SAMPLE_ID_INVALID", "task requires a non-empty sample_id")
            text = row.get("text")
            _require(isinstance(text, str) and text.strip(), "M8C_TASK_TEXT_INVALID", "task requires non-empty text", sample_id=sample_id)
            heads = row.get("heads")
            _require(isinstance(heads, list) and heads and all(isinstance(head, str) for head in heads), "M8C_TASK_HEADS_INVALID", "task heads must be non-empty strings", sample_id=sample_id)
            if sample_id not in merged:
                copy = dict(row)
                copy["heads"] = list(dict.fromkeys(heads))
                copy["metadata"] = dict(row.get("metadata") or {})
                merged[sample_id] = copy
                order.append(sample_id)
                continue
            existing = merged[sample_id]
            for field in ("text", "title"):
                if row.get(field) not in (None, "") and existing.get(field) not in (None, ""):
                    _require(row.get(field) == existing.get(field), "M8C_TASK_IDENTITY_CONFLICT", "duplicate task sample has conflicting text/title", sample_id=sample_id, field=field)
            existing_metadata = existing.setdefault("metadata", {})
            incoming_metadata = row.get("metadata") or {}
            _require(isinstance(incoming_metadata, Mapping), "M8C_TASK_METADATA_INVALID", "task metadata must be an object", sample_id=sample_id)
            for field in ("exact_content_group", "stock_code", "stock_name", "observation_id"):
                left = existing_metadata.get(field)
                right = incoming_metadata.get(field)
                if left not in (None, "") and right not in (None, ""):
                    _require(left == right, "M8C_TASK_IDENTITY_CONFLICT", "duplicate task sample has conflicting metadata", sample_id=sample_id, field=field)
            for key, value in incoming_metadata.items():
                existing_metadata.setdefault(key, value)
            existing["heads"] = list(dict.fromkeys([*existing.get("heads", []), *heads]))
    result = [merged[sample_id] for sample_id in order]
    _require(len({row["sample_id"] for row in result}) == len(result), "M8C_TASK_SAMPLE_ID_DUPLICATE", "merged task rows are not unique")
    return result


def merge_cumulative_inputs(
    old_tasks: Sequence[Mapping[str, Any]],
    old_assignments: Sequence[Mapping[str, Any]],
    new_tasks: Sequence[Mapping[str, Any]],
    new_assignments: Sequence[Mapping[str, Any]],
    schema: SemanticSchemaCandidate,
) -> tuple[list[Any], list[Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Load old/new sparse assignments with explicit pair-level accounting."""

    _require(len(old_assignments) == 92, "M8C_OLD_ASSIGNMENT_COUNT_INVALID", "M7B valid assignments must contain 92 rows", observed=len(old_assignments))
    _require(len(new_assignments) == 32, "M8C_NEW_ASSIGNMENT_COUNT_INVALID", "M8B final assignments must contain 32 rows", observed=len(new_assignments))
    assignments = [dict(row) for row in [*old_assignments, *new_assignments]]
    _require(len(assignments) == EXPECTED_CUMULATIVE_ASSIGNMENTS, "M8C_ASSIGNMENT_COUNT_INVALID", "cumulative assignments must contain 124 rows", observed=len(assignments))
    pair_counts = Counter((row.get("sample_id"), row.get("head")) for row in assignments)
    duplicate_pairs = [pair for pair, count in pair_counts.items() if count != 1]
    _require(not duplicate_pairs, "M8C_ASSIGNMENT_PAIR_DUPLICATE", "cumulative assignments contain duplicate sample/head pairs", pairs=duplicate_pairs)
    merged_tasks = merge_task_rows((old_tasks, new_tasks))
    records, views = load_auxiliary_task_views(merged_tasks, assignments, schema=schema)
    target_batch = build_auxiliary_targets(views, schema)
    masks = {factor: int(target_batch.loss_masks[factor].sum().item()) for factor in V03_AUXILIARY_FACTORS}
    _require(masks == EXPECTED_CUMULATIVE_MASKS, "M8C_CUMULATIVE_MASKS_INVALID", "cumulative sparse masks differ from expected counts", observed=masks, expected=EXPECTED_CUMULATIVE_MASKS)
    return records, views, merged_tasks, assignments


def restrict_batch_to_pairs(
    batch: AuxiliaryTargetBatch,
    sample_ids: Sequence[str],
    allowed_pairs: set[tuple[str, str]],
) -> AuxiliaryTargetBatch:
    """Select rows and zero masks for pairs outside a named diagnostic subset."""

    index_by_id = {sample_id: index for index, sample_id in enumerate(batch.sample_ids)}
    _require(len(index_by_id) == len(batch.sample_ids), "M8C_SAMPLE_ID_DUPLICATE", "target batch sample IDs must be unique")
    indices = [index_by_id[sample_id] for sample_id in sample_ids]
    index_tensor = torch.tensor(indices, dtype=torch.long)
    targets = {factor: values.index_select(0, index_tensor) for factor, values in batch.targets.items()}
    masks: dict[str, Tensor] = {}
    for factor, values in batch.loss_masks.items():
        selected = values.index_select(0, index_tensor).clone()
        for row_index, sample_id in enumerate(sample_ids):
            if (sample_id, factor) not in allowed_pairs:
                selected[row_index] = 0.0
        masks[factor] = selected
    return AuxiliaryTargetBatch(targets=targets, loss_masks=masks, sample_ids=tuple(sample_ids))


def _single_prediction(logits: Tensor, class_order: Sequence[str]) -> dict[str, Any]:
    probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy().astype(float)
    values = probabilities.tolist()
    return {
        "probabilities": {label: float(probabilities[index]) for index, label in enumerate(class_order)},
        "prediction": [class_order[int(np.argmax(probabilities))]],
        "confidence": float(np.max(probabilities)),
        "normalized_entropy": _normalized_entropy(values),
    }


def _multi_prediction(logits: Tensor, class_order: Sequence[str]) -> dict[str, Any]:
    probabilities = torch.sigmoid(logits).detach().cpu().numpy().astype(float)
    values = probabilities.tolist()
    labels = [label for index, label in enumerate(class_order) if probabilities[index] >= 0.5]
    return {
        "probabilities": {label: float(probabilities[index]) for index, label in enumerate(class_order)},
        "prediction": labels,
        "confidence": float(np.max(probabilities)) if len(probabilities) else 0.0,
        "normalized_entropy": _multi_entropy(values),
        "threshold": 0.5,
    }


def _normalized_entropy(values: Sequence[float]) -> float:
    probabilities = np.asarray(values, dtype=float)
    if probabilities.size == 0 or not np.isfinite(probabilities).all():
        return 1.0
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    probabilities = probabilities / probabilities.sum()
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    return entropy / math.log(max(2, probabilities.size))


def _multi_entropy(values: Sequence[float]) -> float:
    probabilities = np.clip(np.asarray(values, dtype=float), 1e-12, 1.0 - 1e-12)
    if probabilities.size == 0:
        return 1.0
    entropy = -(probabilities * np.log(probabilities) + (1.0 - probabilities) * np.log(1.0 - probabilities))
    return float(np.mean(entropy) / math.log(2.0))


def summarize_fresh_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize auxiliary predictions without treating empty multi-label output as a class."""

    summary: dict[str, Any] = {"sample_count": len(rows), "factor_summary": {}}
    for factor in V03_AUXILIARY_FACTORS:
        entries = [row.get("auxiliary", {}).get(factor) for row in rows if factor in row.get("applicable_factors", []) and factor in row.get("auxiliary", {})]
        spec_multi = factor in {"market_scope", "discourse_focus"}
        distribution: Counter[str] = Counter()
        empty_count = 0
        entropies: list[float] = []
        confidences: list[float] = []
        for entry in entries:
            prediction = list(entry.get("prediction", []))
            if spec_multi:
                if not prediction:
                    empty_count += 1
                distribution.update(prediction)
            elif prediction:
                distribution[prediction[0]] += 1
            entropies.append(float(entry.get("normalized_entropy", 1.0)))
            confidences.append(float(entry.get("confidence", 0.0)))
        denominator = len(entries) if spec_multi else sum(distribution.values())
        shares = {label: count / denominator for label, count in distribution.items()} if denominator else {}
        max_share = max(shares.values(), default=0.0)
        summary["factor_summary"][factor] = {
            "applicable_sample_count": len(entries),
            "prediction_distribution": dict(sorted(distribution.items())),
            "empty_prediction_count": empty_count if spec_multi else None,
            "empty_prediction_rate": (empty_count / len(entries)) if spec_multi and entries else None,
            "max_category_or_label_share": max_share,
            "single_category_collapse": bool(max_share > 0.90),
            "mean_confidence": float(np.mean(confidences)) if confidences else None,
            "mean_normalized_entropy": float(np.mean(entropies)) if entropies else None,
            "high_uncertainty_count": int(sum(value >= 0.75 for value in entropies)),
        }
    return summary


def _fresh_input_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = _jsonl(path)
    _require(len(rows) == 2000, "M8C_FRESH_POOL_COUNT_INVALID", "fresh pool must contain 2000 rows", observed=len(rows))
    sample_ids = [row.get("sample_id") for row in rows]
    _require(all(isinstance(value, str) and value for value in sample_ids), "M8C_FRESH_SAMPLE_ID_INVALID", "fresh rows require non-empty sample IDs")
    _require(len(set(sample_ids)) == len(sample_ids), "M8C_FRESH_SAMPLE_ID_DUPLICATE", "fresh pool sample IDs must be unique")
    for row in rows:
        text = row.get("model_text") or row.get("content")
        _require(isinstance(text, str) and text.strip(), "M8C_FRESH_TEXT_INVALID", "fresh pool model_text must be non-empty", sample_id=row.get("sample_id"))
    return rows


def _fresh_record(row: Mapping[str, Any]) -> Any:
    from .encoder_m2_selected_runtime import InferenceRecord

    text = str(row.get("model_text") or row.get("content") or "")
    return InferenceRecord(
        sample_id=str(row["sample_id"]),
        stock_code=str(row.get("stock_code") or ""),
        stock_name=str(row.get("stock_name") or ""),
        model_text=text,
        metadata={
            "text_length": row.get("text_length", len(text)),
            "text_source": row.get("text_source"),
            "exact_content_group": row.get("exact_content_group"),
        },
    )


def fresh_shadow_predictions(
    runtime: AuxiliaryTrainingRuntime,
    auxiliary_model: AuxiliaryHeadModel,
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Run the frozen runtime once per batch and attach only applicable aux outputs."""

    records = [_fresh_record(row) for row in rows]
    output: list[dict[str, Any]] = []
    auxiliary_model.eval()
    runtime.model.eval()
    for offset in range(0, len(records), 16):
        batch_records = records[offset : offset + 16]
        batch = runtime.build_inputs(batch_records, "cpu")
        with torch.no_grad():
            captured = runtime.model(batch["input_ids"], batch["attention_mask"])
            logits_by_factor = auxiliary_model(captured["owner_representations"])
        for index, (record, row) in enumerate(zip(batch_records, rows[offset : offset + 16])):
            text = record.model_text
            applicable = list(_applicable_factors(text))
            result: dict[str, Any] = {
                "sample_id": record.sample_id,
                "model_text": text,
                "stock_code": record.stock_code,
                "stock_name": record.stock_name,
                "text_length": row.get("text_length", len(text)),
                "text_source": row.get("text_source"),
                "exact_content_group": row.get("exact_content_group"),
                "applicable_factors": applicable,
                "auxiliary": {},
            }
            for factor in applicable:
                spec = runtime.schema.auxiliary_factors[factor]
                logits = logits_by_factor[factor][index]
                _require(bool(torch.isfinite(logits).all().item()), "M8C_FRESH_NONFINITE_OUTPUT", "fresh auxiliary logits are non-finite", factor=factor, sample_id=record.sample_id)
                if spec.allow_multiple:
                    result["auxiliary"][factor] = _multi_prediction(logits, spec.class_order)
                else:
                    result["auxiliary"][factor] = _single_prediction(logits, spec.class_order)
            output.append(result)
    _require(len(output) == len(rows), "M8C_FRESH_OUTPUT_COUNT_INVALID", "fresh shadow output count differs from input", inputs=len(rows), outputs=len(output))
    return output


def _load_checkpoint(path: str | Path, schema: SemanticSchemaCandidate, hidden_size: int) -> AuxiliaryHeadModel:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    _require(isinstance(payload, Mapping) and isinstance(payload.get("auxiliary_state_dict"), Mapping), "M8C_CHECKPOINT_INVALID", "auxiliary checkpoint payload is invalid", path=str(path))
    model = AuxiliaryHeadModel(schema, hidden_size)
    model.load_state_dict(payload["auxiliary_state_dict"])
    model.eval()
    return model


def _clone_batch(batch: AuxiliaryTargetBatch) -> AuxiliaryTargetBatch:
    return AuxiliaryTargetBatch(
        targets={factor: values.clone() for factor, values in batch.targets.items()},
        loss_masks={factor: values.clone() for factor, values in batch.loss_masks.items()},
        sample_ids=tuple(batch.sample_ids),
    )


def run_m8c(
    *,
    old_tasks_path: str | Path = DEFAULT_M7B_TASKS,
    old_assignments_path: str | Path = DEFAULT_M7B_ASSIGNMENTS,
    new_tasks_path: str | Path = DEFAULT_M8B_TASKS,
    new_assignments_path: str | Path = DEFAULT_M8B_ASSIGNMENTS,
    old_checkpoint_path: str | Path = DEFAULT_M7B_CHECKPOINT,
    fresh_pool_path: str | Path = DEFAULT_FRESH_POOL,
    output_dir: str | Path = DEFAULT_ARTIFACT,
    selected_artifact: str | Path = DEFAULT_SELECTED_ARTIFACT,
    cache_dir: str | Path = DEFAULT_CACHE,
    config_path: str | Path = DEFAULT_CONFIG,
    schema_path: str | Path = DEFAULT_CANDIDATE_SCHEMA,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    _require(not output.exists(), "M8C_OUTPUT_EXISTS", "M8C output directory must be new", path=str(output))
    schema = SemanticSchemaCandidate.load(schema_path)
    old_tasks = _jsonl(old_tasks_path)
    old_assignments = _jsonl(old_assignments_path)
    new_tasks = _jsonl(new_tasks_path)
    new_assignments = _jsonl(new_assignments_path)
    new_head_counts = Counter(row.get("head") for row in new_assignments)
    _require(dict(new_head_counts) == M8B_COUNTS, "M8C_M8B_HEAD_COUNTS_INVALID", "M8B final head counts differ", observed=dict(new_head_counts), expected=M8B_COUNTS)
    _require(all(row.get("is_final") is True for row in new_assignments), "M8C_M8B_ASSIGNMENT_NOT_FINAL", "M8B pull must contain only final assignments")
    _require(all(row.get("disposition") in (None, "") for row in new_assignments), "M8C_M8B_DISPOSITION_PRESENT", "M8B final assignments must not carry a disposition")
    _require(len({(row.get("sample_id"), row.get("head")) for row in new_assignments}) == len(new_assignments), "M8C_M8B_PAIR_DUPLICATE", "M8B assignments must have unique sample/head pairs")
    records, views, merged_tasks, assignments = merge_cumulative_inputs(old_tasks, old_assignments, new_tasks, new_assignments, schema)
    target_batch = build_auxiliary_targets(views, schema, device="cpu")
    output.mkdir(parents=True, exist_ok=False)
    # Preserve both source snapshots verbatim as local evidence.
    write_jsonl(output / "m7b-tasks-snapshot.jsonl", old_tasks)
    write_jsonl(output / "m8b-tasks-snapshot.jsonl", new_tasks)
    write_jsonl(output / "assignments-cumulative-124.jsonl", assignments)
    write_jsonl(output / "tasks-cumulative-93.jsonl", merged_tasks)

    runtime = AuxiliaryTrainingRuntime.from_selected_artifact(selected_artifact, config_path=config_path, cache_dir=cache_dir)
    batch_inputs = runtime.build_inputs(records, "cpu")
    with torch.no_grad():
        captured = runtime.model(batch_inputs["input_ids"], batch_inputs["attention_mask"])
    representations = {owner: values.detach().cpu() for owner, values in captured["owner_representations"].items()}
    hidden_size = int(next(iter(representations.values())).shape[-1])
    old_model = _load_checkpoint(old_checkpoint_path, schema, hidden_size)
    new_ids = [row["sample_id"] for row in new_tasks]
    new_pairs = {(row.get("sample_id"), row.get("head")) for row in new_assignments}
    full_batch = build_auxiliary_targets(views, schema, device="cpu")
    new_batch = restrict_batch_to_pairs(full_batch, new_ids, new_pairs)
    new_indices = [full_batch.sample_ids.index(sample_id) for sample_id in new_ids]
    new_reps = {owner: values.index_select(0, torch.tensor(new_indices, dtype=torch.long)) for owner, values in representations.items()}
    with torch.no_grad():
        old_new_logits = old_model(new_reps)
    pre_update = {
        factor: metric_for_factor(factor, old_new_logits[factor], new_batch, schema)
        for factor in V03_AUXILIARY_FACTORS
    }

    torch.manual_seed(SEED)
    template = AuxiliaryHeadModel(schema, hidden_size)
    initial_state = _clone_state(template.state_dict())
    started = time.monotonic()
    oof = run_oof(representations, target_batch, [record.sample_id for record in records], schema, initial_state=initial_state)
    final_model, final_history = _fit_cached_heads(initial_state, representations, target_batch, schema, list(range(len(records))), epochs=EPOCHS)
    with torch.no_grad():
        final_logits = final_model(representations)
    final_model.eval()
    final_metrics = {factor: metric_for_factor(factor, final_logits[factor], target_batch, schema) for factor in V03_AUXILIARY_FACTORS}
    fresh_rows = _fresh_input_rows(fresh_pool_path)
    fresh_predictions = fresh_shadow_predictions(runtime, final_model, fresh_rows)
    fresh_summary = summarize_fresh_rows(fresh_predictions)
    elapsed = time.monotonic() - started
    checkpoint = output / "auxiliary-checkpoint.pt"
    torch.save(
        {
            "component": "m8c_v03_cumulative_auxiliary_diagnostic",
            "schema_version": schema.schema_version,
            "seed": SEED,
            "epochs": EPOCHS,
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
            "sample_ids": [record.sample_id for record in records],
            "assignment_count": len(assignments),
            "auxiliary_state_dict": final_model.state_dict(),
        },
        checkpoint,
    )
    reloaded = _load_checkpoint(checkpoint, schema, hidden_size)
    with torch.no_grad():
        reload_logits = reloaded(representations)
    reload_equal = all(torch.equal(final_logits[factor], reload_logits[factor]) for factor in V03_AUXILIARY_FACTORS)
    _require(reload_equal, "M8C_CHECKPOINT_RELOAD_MISMATCH", "reloaded cumulative checkpoint differs from final outputs")
    summary = {
        "component": "M8C_V03_CUMULATIVE_AUXILIARY_DIAGNOSTIC",
        "status": "DIAGNOSTIC_SHADOW_ONLY",
        "selected_candidate": False,
        "training_scope": "five auxiliary heads only; frozen selected LERT and seven legacy heads",
        "old_assignment_count": len(old_assignments),
        "new_m8b_assignment_count": len(new_assignments),
        "cumulative_assignment_count": len(assignments),
        "cumulative_sample_count": len(records),
        "cumulative_factor_masks": {factor: int(target_batch.loss_masks[factor].sum().item()) for factor in V03_AUXILIARY_FACTORS},
        "m8b_factor_counts": dict(new_head_counts),
        "seed": SEED,
        "epochs": EPOCHS,
        "optimizer": {"name": "AdamW", "learning_rate": LR, "weight_decay": WEIGHT_DECAY},
        "device": "cpu",
        "elapsed_seconds": elapsed,
        "representations_computed_once": True,
        "pre_update_m8b_metrics": pre_update,
        "oof": oof,
        "final_full_data_metrics": final_metrics,
        "final_losses": {factor: float(value.detach().cpu()) for factor, value in __import__("semantic_model.auxiliary_heads", fromlist=["compute_auxiliary_losses"]).compute_auxiliary_losses(final_logits, target_batch, schema).items()},
        "final_loss_start": float(final_history[0]),
        "final_loss_end": float(final_history[-1]),
        "checkpoint": str(checkpoint),
        "checkpoint_reload_equal": reload_equal,
        "fresh_pool_path": str(Path(fresh_pool_path).resolve()),
        "fresh_pool_sha256": sha256_file(fresh_pool_path),
        "fresh_pool_input_rows": len(fresh_rows),
        "fresh_rows_text_length_ge_39_processed": sum(1 for row in fresh_rows if int(row.get("text_length", len(str(row.get("model_text") or "")))) >= 39),
        "fresh_shadow_predictions": str(output / "fresh-2000-shadow-predictions.jsonl"),
        "fresh_summary": fresh_summary,
        "forbidden_data_access": {"test": False, "gold": False, "ood": False, "reference_predictions": False, "llm": False},
    }
    write_jsonl(output / "fresh-2000-shadow-predictions.jsonl", fresh_predictions)
    (output / "pre-update-m8b-metrics.json").write_text(json.dumps(pre_update, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (output / "oof-metrics.json").write_text(json.dumps(oof, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (output / "aggregate-metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    report = render_report(summary)
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    files = [path for path in output.iterdir() if path.is_file() and path.name != "manifest.json"]
    manifest = {
        "artifact": "m8c-v03-cumulative-auxiliary-20260914",
        "status": "DIAGNOSTIC_SHADOW_ONLY",
        "files": {path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(files)},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# M8C cumulative auxiliary diagnostic and fresh-2000 shadow",
        "",
        "This is one diagnostic run. The selected LERT encoder and all seven legacy heads stayed frozen; the five auxiliary heads were reinitialized once and trained on the cumulative sparse assignments. No candidate promotion or production claim is made.",
        "",
        f"- Assignments: old M7B {summary['old_assignment_count']} + final M8B {summary['new_m8b_assignment_count']} = {summary['cumulative_assignment_count']}; merged samples {summary['cumulative_sample_count']}",
        f"- Masks: `{json.dumps(summary['cumulative_factor_masks'], ensure_ascii=False, sort_keys=True)}`",
        f"- Device/time: {summary['device']} / {summary['elapsed_seconds']:.2f}s; seed {summary['seed']}; epochs {summary['epochs']}; AdamW lr={summary['optimizer']['learning_rate']} wd={summary['optimizer']['weight_decay']}",
        "",
        "## M8B pre-update (old M7B checkpoint on the 32 new assignments)",
        "",
    ]
    for factor, metrics in summary["pre_update_m8b_metrics"].items():
        lines.append(f"- `{factor}`: `{json.dumps(metrics, ensure_ascii=False, sort_keys=True)}`")
    lines.extend(["", "## Cumulative 3-fold OOF", "", "OOF metrics and fold baselines are descriptive only; no one-hot log-loss or all-negative baseline is treated as success evidence.", ""])
    for factor, item in summary["oof"]["aggregate"].items():
        lines.append(f"- `{factor}` ({item['diagnostic_class']}, n={item['label_count']}): metrics `{json.dumps(item['metrics'], ensure_ascii=False, sort_keys=True)}`; baseline `{json.dumps(item['baseline'], ensure_ascii=False, sort_keys=True)}`")
    lines.extend(["", "## Fresh pool shadow", "", f"- Input rows inferred: {summary['fresh_pool_input_rows']}; rows with text_length >=39 were processed (not globally excluded): {summary['fresh_rows_text_length_ge_39_processed']}"])
    for factor, item in summary["fresh_summary"]["factor_summary"].items():
        lines.append(f"- `{factor}`: applicable={item['applicable_sample_count']}, distribution={json.dumps(item['prediction_distribution'], ensure_ascii=False, sort_keys=True)}, empty={item['empty_prediction_count']}, empty_rate={item['empty_prediction_rate']}, high_uncertainty={item['high_uncertainty_count']}, collapse={item['single_category_collapse']}")
    lines.extend(["", "The fresh pass uses explicit text applicability only. M8A's length>=39 exclusion is not applied. Test, Gold, OOD, reference predictions and LLM inputs were not read. The output remains diagnostic/shadow evidence and does not create a labeling batch.", ""])
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-tasks", type=Path, default=DEFAULT_M7B_TASKS)
    parser.add_argument("--old-assignments", type=Path, default=DEFAULT_M7B_ASSIGNMENTS)
    parser.add_argument("--new-tasks", type=Path, default=DEFAULT_M8B_TASKS)
    parser.add_argument("--new-assignments", type=Path, default=DEFAULT_M8B_ASSIGNMENTS)
    parser.add_argument("--old-checkpoint", type=Path, default=DEFAULT_M7B_CHECKPOINT)
    parser.add_argument("--fresh-pool", type=Path, default=DEFAULT_FRESH_POOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--selected-artifact", type=Path, default=DEFAULT_SELECTED_ARTIFACT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--schema", type=Path, default=DEFAULT_CANDIDATE_SCHEMA)
    args = parser.parse_args(argv)
    result = run_m8c(
        old_tasks_path=args.old_tasks,
        old_assignments_path=args.old_assignments,
        new_tasks_path=args.new_tasks,
        new_assignments_path=args.new_assignments,
        old_checkpoint_path=args.old_checkpoint,
        fresh_pool_path=args.fresh_pool,
        output_dir=args.output_dir,
        selected_artifact=args.selected_artifact,
        cache_dir=args.cache_dir,
        config_path=args.config,
        schema_path=args.schema,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "merge_task_rows",
    "merge_cumulative_inputs",
    "restrict_batch_to_pairs",
    "summarize_fresh_rows",
    "fresh_shadow_predictions",
    "run_m8c",
]
