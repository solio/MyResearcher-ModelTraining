"""M9B cumulative auxiliary effect gate.

This is a single, finite diagnostic over the already-final M9A feedback.  It
reuses the frozen LERT representation path and the five linear v0.3 heads
from M8C, reinitializes those heads once, evaluates a deterministic
sample-grouped three-fold OOF, fits one 50-epoch diagnostic checkpoint, and
reruns the existing fresh-2000 shadow pass.  The selected seven-head model,
encoder, data schema and thresholds are never modified.

Only M8C's 124 valid assignments plus the 50 final M9A assignments are read.
The one known incomplete historical pair (``这公司真，`` /
``discourse_focus``) is masked if it is present; its database record is never
changed.  Test, Gold, OOD, Anchor, Human100/48, LLM and production inputs are
outside this runner by construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .auxiliary_heads import AuxiliaryHeadModel, AuxiliaryTargetBatch, build_auxiliary_targets
from .encoder_m6c_auxiliary import (
    DEFAULT_CANDIDATE_SCHEMA,
    AuxiliaryTrainingRuntime,
    load_auxiliary_task_views,
)
from .encoder_m7b_auxiliary import (
    _clone_state,
    _fit_cached_heads,
    _majority_baseline,
    _slice_batch,
    _slice_representations,
    deterministic_folds,
    metric_for_factor,
    read_jsonl,
)
from .encoder_m8c_cumulative import (
    _fresh_input_rows,
    _load_checkpoint,
    _normalized_entropy,
    _multi_entropy,
    _single_prediction,
    _multi_prediction,
    fresh_shadow_predictions,
    merge_task_rows,
    summarize_fresh_rows,
)
from .errors import ContractError
from .schema import SemanticSchemaCandidate, V03_AUXILIARY_FACTORS


SEED = 20260913
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 0.0
INCOMPLETE_SAMPLE_ID = "fl1-2c7192dc027b67c511861e2b927e27bfa2ed8132eed411729d067ea24fd85564"
INCOMPLETE_HEAD = "discourse_focus"
EXPECTED_ASSIGNMENTS = 174
EXPECTED_SAMPLES = 143
EXPECTED_MASKS = {
    "expectation_phase": 36,
    "arousal_level": 23,
    "desired_price_direction": 36,
    "market_scope": 24,
    "discourse_focus": 55,
}
M9A_COUNTS = {
    "expectation_phase": 6,
    "arousal_level": 10,
    "desired_price_direction": 8,
    "market_scope": 10,
    "discourse_focus": 16,
}

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT.parent
DEFAULT_M8C_ARTIFACT = PROJECT_ROOT / "model-artifacts" / "m8c-v03-cumulative-auxiliary-20260914"
DEFAULT_M8C_TASKS = DEFAULT_M8C_ARTIFACT / "tasks-cumulative-93.jsonl"
DEFAULT_M8C_ASSIGNMENTS = DEFAULT_M8C_ARTIFACT / "assignments-cumulative-124.jsonl"
DEFAULT_M8C_CHECKPOINT = DEFAULT_M8C_ARTIFACT / "auxiliary-checkpoint.pt"
DEFAULT_M8C_FRESH_PREDICTIONS = DEFAULT_M8C_ARTIFACT / "fresh-2000-shadow-predictions.jsonl"
DEFAULT_M8C_AGGREGATE = DEFAULT_M8C_ARTIFACT / "aggregate-metrics.json"
DEFAULT_M9A_TASKS = PROJECT_ROOT / "human-review-artifacts" / "m9a-v03-fresh-review50-20260914" / "auxiliary-tasks.jsonl"
DEFAULT_M9A_ASSIGNMENTS = REPO_ROOT / "runs" / "m9a-v03-fresh-review50-20260914-assignments-pulled.jsonl"
DEFAULT_FRESH_POOL = PROJECT_ROOT / "MyResearcher-DataClean" / "artifacts" / "round-004" / "fresh-pool-2000.jsonl"
DEFAULT_SELECTED_ARTIFACT = PROJECT_ROOT / "model-artifacts" / "m2-selected-candidate-frozen-for-m3-20260904"
DEFAULT_CACHE = REPO_ROOT / ".encoder-artifacts" / "hf-cache-lert-small"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "baseline_v0.3.5.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "model-artifacts" / "m9b-v03-cumulative174-effect-gate-20260914"


def _require(ok: bool, code: str, message: str, **details: Any) -> None:
    if not ok:
        raise ContractError(code, message, **details)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def validate_m9a_assignments(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate the frozen M9A pull before any model/runtime work."""

    _require(len(rows) == 50, "M9B_M9A_ASSIGNMENT_COUNT_INVALID", "M9A pull must contain exactly 50 rows", observed=len(rows))
    pairs = [(row.get("sample_id"), row.get("head")) for row in rows]
    _require(all(isinstance(sample_id, str) and sample_id for sample_id, _ in pairs), "M9B_M9A_SAMPLE_ID_INVALID", "M9A rows require sample_id")
    _require(all(isinstance(head, str) and head in V03_AUXILIARY_FACTORS for _, head in pairs), "M9B_M9A_HEAD_INVALID", "M9A rows require known auxiliary heads")
    _require(len(set(pairs)) == 50, "M9B_M9A_PAIR_DUPLICATE", "M9A sample/head pairs must be unique")
    _require(all(row.get("is_final") is True for row in rows), "M9B_M9A_NOT_FINAL", "M9A pull must contain only final assignments")
    _require(all(row.get("disposition") in (None, "") for row in rows), "M9B_M9A_DISPOSITION_PRESENT", "M9A assignments must have no disposition")
    counts = Counter(head for _, head in pairs)
    _require(dict(counts) == M9A_COUNTS, "M9B_M9A_HEAD_COUNTS_INVALID", "M9A per-head counts differ from the frozen batch", observed=dict(counts), expected=M9A_COUNTS)
    return {"assignment_count": len(rows), "sample_count": len({sample_id for sample_id, _ in pairs}), "head_counts": dict(counts), "pairs": pairs}


def combine_m9b_inputs(
    old_tasks: Sequence[Mapping[str, Any]],
    old_assignments: Sequence[Mapping[str, Any]],
    new_tasks: Sequence[Mapping[str, Any]],
    new_assignments: Sequence[Mapping[str, Any]],
    schema: SemanticSchemaCandidate,
) -> tuple[list[Any], list[Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Merge M8C and M9A while masking the known incomplete historical pair."""

    _require(len(old_assignments) == 124, "M9B_OLD_ASSIGNMENT_COUNT_INVALID", "M8C must provide 124 valid assignments", observed=len(old_assignments))
    _require(len(new_assignments) == 50, "M9B_NEW_ASSIGNMENT_COUNT_INVALID", "M9A must provide 50 final assignments", observed=len(new_assignments))
    raw = [dict(row) for row in [*old_assignments, *new_assignments]]
    pair_counts = Counter((row.get("sample_id"), row.get("head")) for row in raw)
    duplicates = [pair for pair, count in pair_counts.items() if count != 1]
    _require(not duplicates, "M9B_ASSIGNMENT_PAIR_DUPLICATE", "cumulative assignments contain duplicate pairs", pairs=duplicates)
    effective = [row for row in raw if not (row.get("sample_id") == INCOMPLETE_SAMPLE_ID and row.get("head") == INCOMPLETE_HEAD)]
    _require(len(effective) == EXPECTED_ASSIGNMENTS, "M9B_ASSIGNMENT_COUNT_INVALID", "effective cumulative assignment count differs", observed=len(effective), expected=EXPECTED_ASSIGNMENTS)
    merged_tasks = merge_task_rows((old_tasks, new_tasks))
    _require(len(merged_tasks) == EXPECTED_SAMPLES, "M9B_SAMPLE_COUNT_INVALID", "merged task samples differ from the frozen 143", observed=len(merged_tasks), expected=EXPECTED_SAMPLES)
    records, views = load_auxiliary_task_views(merged_tasks, effective, schema=schema)
    target_batch = build_auxiliary_targets(views, schema, device="cpu")
    masks = {factor: int(target_batch.loss_masks[factor].sum().item()) for factor in V03_AUXILIARY_FACTORS}
    _require(masks == EXPECTED_MASKS, "M9B_MASK_COUNTS_INVALID", "effective masks differ from the frozen M9B contract", observed=masks, expected=EXPECTED_MASKS)
    return records, views, merged_tasks, effective, {"raw_assignment_count": len(raw), "effective_assignment_count": len(effective), "masked_incomplete_pair": {"sample_id": INCOMPLETE_SAMPLE_ID, "head": INCOMPLETE_HEAD, "present_in_raw": len(raw) != len(effective)}}


def _prediction_diagnostics(factor: str, logits: Tensor, batch: AuxiliaryTargetBatch, schema: SemanticSchemaCandidate) -> dict[str, Any]:
    """Return OOF cardinality/empty-output diagnostics in addition to metrics."""

    valid = batch.loss_masks[factor] > 0
    if not bool(valid.any().item()):
        return {"n": 0, "prediction_cardinality_distribution": {}, "mean_prediction_cardinality": None, "empty_prediction_count": None, "empty_prediction_rate": None}
    values = logits[valid].detach().cpu()
    spec = schema.auxiliary_factors[factor]
    if spec.allow_multiple:
        predictions = (torch.sigmoid(values).numpy() >= 0.5).astype(int)
        cardinality = predictions.sum(axis=1).astype(int)
        counts = Counter(int(value) for value in cardinality)
        empty = int((cardinality == 0).sum())
        return {"n": int(len(cardinality)), "prediction_cardinality_distribution": {str(key): value for key, value in sorted(counts.items())}, "mean_prediction_cardinality": float(cardinality.mean()), "empty_prediction_count": empty, "empty_prediction_rate": float(empty / len(cardinality))}
    predictions = torch.softmax(values, dim=-1).argmax(dim=-1).numpy().astype(int)
    counts = Counter(int(value) for value in predictions)
    return {"n": int(len(predictions)), "prediction_cardinality_distribution": {"1": int(len(predictions))}, "mean_prediction_cardinality": 1.0, "empty_prediction_count": None, "empty_prediction_rate": None, "prediction_class_index_distribution": {str(key): value for key, value in sorted(counts.items())}}


def run_oof_detailed(
    representations: Mapping[str, Tensor],
    batch: AuxiliaryTargetBatch,
    sample_ids: Sequence[str],
    schema: SemanticSchemaCandidate,
    *,
    initial_state: Mapping[str, Tensor],
) -> dict[str, Any]:
    """One deterministic three-fold OOF pass retaining logits for diagnostics."""

    folds = deterministic_folds(sample_ids)
    oof_logits = {factor: torch.zeros((len(sample_ids), len(schema.auxiliary_factors[factor].class_order)), dtype=torch.float32) for factor in V03_AUXILIARY_FACTORS}
    baseline_logits = {factor: torch.zeros_like(oof_logits[factor]) for factor in V03_AUXILIARY_FACTORS}
    fold_reports: list[dict[str, Any]] = []
    for fold in range(3):
        train_indices = [index for index, sample_id in enumerate(sample_ids) if folds[sample_id] != fold]
        valid_indices = [index for index, sample_id in enumerate(sample_ids) if folds[sample_id] == fold]
        model, history = _fit_cached_heads(initial_state, representations, batch, schema, train_indices, epochs=EPOCHS)
        valid_reps = _slice_representations(representations, valid_indices)
        with torch.no_grad():
            fitted = model(valid_reps)
        valid_batch = _slice_batch(batch, valid_indices)
        train_batch = _slice_batch(batch, train_indices)
        fold_report: dict[str, Any] = {"fold": fold, "train_samples": len(train_indices), "valid_samples": len(valid_indices), "final_loss": history[-1]}
        for factor in V03_AUXILIARY_FACTORS:
            oof_logits[factor][valid_indices] = fitted[factor].detach().cpu()
            baseline = _majority_baseline(factor, train_batch, valid_batch, schema)
            spec = schema.auxiliary_factors[factor]
            if spec.allow_multiple:
                baseline_logits[factor][valid_indices] = -20.0
            else:
                valid_train = train_batch.loss_masks[factor] > 0
                counts = torch.bincount(train_batch.targets[factor][valid_train], minlength=len(spec.class_order)) if int(valid_train.sum()) else torch.ones(len(spec.class_order), dtype=torch.long)
                baseline_class = int(torch.argmax(counts).item())
                baseline_logits[factor][valid_indices, baseline_class] = 20.0
            fold_report[factor] = {"metrics": metric_for_factor(factor, fitted[factor], valid_batch, schema), "baseline": baseline}
        fold_reports.append(fold_report)
    aggregate: dict[str, Any] = {}
    for factor in V03_AUXILIARY_FACTORS:
        aggregate[factor] = {
            "metrics": metric_for_factor(factor, oof_logits[factor], batch, schema),
            "baseline": metric_for_factor(factor, baseline_logits[factor], batch, schema),
            "prediction_diagnostics": _prediction_diagnostics(factor, oof_logits[factor], batch, schema),
            "diagnostic_class": "LOW_SAMPLE_DIAGNOSTIC" if int(batch.loss_masks[factor].sum().item()) < 30 else "STANDARD_DIAGNOSTIC",
            "label_count": int(batch.loss_masks[factor].sum().item()),
        }
    return {"fold_assignment": folds, "folds": fold_reports, "aggregate": aggregate}


def active_shadow_rows(rows: Sequence[Mapping[str, Any]], excluded_pairs: set[tuple[str, str]]) -> list[dict[str, Any]]:
    """Remove only M9A-covered factor outputs while retaining all other rows."""

    result: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row.get("sample_id"))
        factors = [factor for factor in row.get("applicable_factors", []) if (sample_id, factor) not in excluded_pairs]
        copy = dict(row)
        copy["applicable_factors"] = factors
        copy["auxiliary"] = {factor: value for factor, value in (row.get("auxiliary") or {}).items() if factor in factors}
        result.append(copy)
    return result


def discourse_verdict(oof_metrics: Mapping[str, Any]) -> str:
    """Apply the fixed M9B discourse gate without threshold changes."""

    metrics = oof_metrics.get("metrics", {})
    diagnostics = oof_metrics.get("prediction_diagnostics", {})
    micro = metrics.get("micro_f1")
    empty_rate = diagnostics.get("empty_prediction_rate")
    if isinstance(micro, (int, float)) and isinstance(empty_rate, (int, float)) and micro > 0.15 and empty_rate < 0.80:
        return "WEAK_SIGNAL_CONTINUE"
    return "CURRENT_LINEAR_HEAD_FAILED_REDESIGN_REQUIRED"


def _head_conclusion(factor: str, item: Mapping[str, Any], verdict: str) -> str:
    if factor == "discourse_focus":
        return "REDESIGN" if verdict != "WEAK_SIGNAL_CONTINUE" else "KEEP"
    if factor in {"arousal_level", "market_scope"}:
        return "COLLECT_LATER"
    metrics = item.get("metrics", {})
    baseline = item.get("baseline", {})
    if factor in {"expectation_phase", "desired_price_direction"} and metrics.get("macro_f1") is not None and baseline.get("macro_f1") is not None and metrics["macro_f1"] > baseline["macro_f1"]:
        return "KEEP"
    return "COLLECT_LATER"


def _oof_comparison(current: Mapping[str, Any], previous: Mapping[str, Any]) -> dict[str, Any]:
    """Compare OOF metrics while retaining both sample sizes explicitly."""

    result: dict[str, Any] = {}
    for factor in V03_AUXILIARY_FACTORS:
        now = current.get("aggregate", {}).get(factor, {})
        old = previous.get("aggregate", {}).get(factor, {})
        now_metrics = dict(now.get("metrics", {}))
        old_metrics = dict(old.get("metrics", {}))
        common = sorted(set(now_metrics) & set(old_metrics))
        delta = {
            key: float(now_metrics[key]) - float(old_metrics[key])
            for key in common
            if key != "n" and isinstance(now_metrics[key], (int, float)) and isinstance(old_metrics[key], (int, float))
        }
        result[factor] = {
            "m8c_sample_count": old_metrics.get("n"),
            "m9b_sample_count": now_metrics.get("n"),
            "m8c_metrics": old_metrics,
            "m9b_metrics": now_metrics,
            "delta_m9b_minus_m8c": delta,
            "sample_scale_warning": "descriptive_only_sample_sizes_differ",
        }
    return result


def run_m9b(
    *,
    m8c_tasks_path: str | Path = DEFAULT_M8C_TASKS,
    m8c_assignments_path: str | Path = DEFAULT_M8C_ASSIGNMENTS,
    m8c_checkpoint_path: str | Path = DEFAULT_M8C_CHECKPOINT,
    m8c_fresh_predictions_path: str | Path = DEFAULT_M8C_FRESH_PREDICTIONS,
    m8c_aggregate_path: str | Path = DEFAULT_M8C_AGGREGATE,
    m9a_tasks_path: str | Path = DEFAULT_M9A_TASKS,
    m9a_assignments_path: str | Path = DEFAULT_M9A_ASSIGNMENTS,
    fresh_pool_path: str | Path = DEFAULT_FRESH_POOL,
    output_dir: str | Path = DEFAULT_OUTPUT,
    selected_artifact: str | Path = DEFAULT_SELECTED_ARTIFACT,
    cache_dir: str | Path = DEFAULT_CACHE,
    config_path: str | Path = DEFAULT_CONFIG,
    schema_path: str | Path = DEFAULT_CANDIDATE_SCHEMA,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    _require(not output.exists(), "M9B_OUTPUT_EXISTS", "M9B output directory must be new", path=str(output))
    schema = SemanticSchemaCandidate.load(schema_path)
    old_tasks = read_jsonl(m8c_tasks_path)
    old_assignments = read_jsonl(m8c_assignments_path)
    new_tasks = read_jsonl(m9a_tasks_path)
    new_assignments = read_jsonl(m9a_assignments_path)
    m9a_validation = validate_m9a_assignments(new_assignments)
    records, views, merged_tasks, effective_assignments, input_accounting = combine_m9b_inputs(old_tasks, old_assignments, new_tasks, new_assignments, schema)
    target_batch = build_auxiliary_targets(views, schema, device="cpu")
    output.mkdir(parents=True, exist_ok=False)
    _write_jsonl(output / "m8c-tasks-snapshot.jsonl", old_tasks)
    _write_jsonl(output / "m9a-tasks-snapshot.jsonl", new_tasks)
    _write_jsonl(output / "tasks-cumulative-143.jsonl", merged_tasks)
    _write_jsonl(output / "assignments-cumulative-174.jsonl", effective_assignments)
    _write_jsonl(output / "m9a-assignments-pulled-final.jsonl", new_assignments)

    runtime = AuxiliaryTrainingRuntime.from_selected_artifact(selected_artifact, config_path=config_path, cache_dir=cache_dir)
    batch_inputs = runtime.build_inputs(records, "cpu")
    with torch.no_grad():
        captured = runtime.model(batch_inputs["input_ids"], batch_inputs["attention_mask"])
    representations = {owner: values.detach().cpu() for owner, values in captured["owner_representations"].items()}
    hidden_size = int(next(iter(representations.values())).shape[-1])
    old_model = _load_checkpoint(m8c_checkpoint_path, schema, hidden_size)
    m9a_ids = [row["sample_id"] for row in new_tasks]
    m9a_pairs = {(row.get("sample_id"), row.get("head")) for row in new_assignments}
    id_to_index = {sample_id: index for index, sample_id in enumerate([record.sample_id for record in records])}
    m9a_indices = [id_to_index[sample_id] for sample_id in m9a_ids]
    m9a_batch = AuxiliaryTargetBatch(
        targets={factor: target_batch.targets[factor].index_select(0, torch.tensor(m9a_indices, dtype=torch.long)) for factor in V03_AUXILIARY_FACTORS},
        loss_masks={factor: target_batch.loss_masks[factor].index_select(0, torch.tensor(m9a_indices, dtype=torch.long)) for factor in V03_AUXILIARY_FACTORS},
        sample_ids=tuple(m9a_ids),
    )
    m9a_reps = {owner: values.index_select(0, torch.tensor(m9a_indices, dtype=torch.long)) for owner, values in representations.items()}
    with torch.no_grad():
        old_m9a_logits = old_model(m9a_reps)
    pre_update = {factor: metric_for_factor(factor, old_m9a_logits[factor], m9a_batch, schema) for factor in V03_AUXILIARY_FACTORS}

    torch.manual_seed(SEED)
    template = AuxiliaryHeadModel(schema, hidden_size)
    initial_state = _clone_state(template.state_dict())
    started = time.monotonic()
    oof = run_oof_detailed(representations, target_batch, [record.sample_id for record in records], schema, initial_state=initial_state)
    final_model, final_history = _fit_cached_heads(initial_state, representations, target_batch, schema, list(range(len(records))), epochs=EPOCHS)
    with torch.no_grad():
        final_logits = final_model(representations)
    final_losses = __import__("semantic_model.auxiliary_heads", fromlist=["compute_auxiliary_losses"]).compute_auxiliary_losses(final_logits, target_batch, schema)
    final_total = sum(final_losses.values())
    _require(bool(torch.isfinite(final_total).item()), "M9B_FINAL_NONFINITE_LOSS", "final auxiliary loss is non-finite")
    checkpoint = output / "auxiliary-checkpoint.pt"
    torch.save({"component": "m9b_v03_cumulative174_effect_gate", "schema_version": schema.schema_version, "seed": SEED, "epochs": EPOCHS, "learning_rate": LR, "weight_decay": WEIGHT_DECAY, "sample_ids": [record.sample_id for record in records], "assignment_count": len(effective_assignments), "auxiliary_state_dict": final_model.state_dict()}, checkpoint)
    reload_model = _load_checkpoint(checkpoint, schema, hidden_size)
    with torch.no_grad():
        reload_logits = reload_model(representations)
    reload_equal = all(torch.equal(final_logits[factor], reload_logits[factor]) for factor in V03_AUXILIARY_FACTORS)
    _require(reload_equal, "M9B_CHECKPOINT_RELOAD_MISMATCH", "reloaded auxiliary outputs differ from final outputs")

    fresh_rows = _fresh_input_rows(fresh_pool_path)
    fresh_after = fresh_shadow_predictions(runtime, final_model, fresh_rows)
    fresh_before = read_jsonl(m8c_fresh_predictions_path)
    _require(len(fresh_before) == len(fresh_after) == 2000, "M9B_FRESH_COUNT_INVALID", "M8C and M9B fresh shadows must both contain 2000 rows", before=len(fresh_before), after=len(fresh_after))
    fresh_before_summary = summarize_fresh_rows(fresh_before)
    fresh_after_summary = summarize_fresh_rows(fresh_after)
    fresh_active_summary = summarize_fresh_rows(active_shadow_rows(fresh_after, m9a_pairs))
    m8c_aggregate = json.loads(Path(m8c_aggregate_path).read_text(encoding="utf-8"))
    elapsed = time.monotonic() - started
    discourse = oof["aggregate"]["discourse_focus"]
    discourse_gate = discourse_verdict(discourse)
    conclusions = {factor: _head_conclusion(factor, oof["aggregate"][factor], discourse_gate) for factor in V03_AUXILIARY_FACTORS}
    summary: dict[str, Any] = {
        "component": "M9B_V03_CUMULATIVE174_EFFECT_GATE",
        "status": "DIAGNOSTIC_ONLY",
        "selected_candidate": False,
        "m9a_validation": {key: value for key, value in m9a_validation.items() if key != "pairs"},
        "input_accounting": input_accounting,
        "cumulative_assignment_count": len(effective_assignments),
        "cumulative_sample_count": len(records),
        "cumulative_factor_masks": {factor: int(target_batch.loss_masks[factor].sum().item()) for factor in V03_AUXILIARY_FACTORS},
        "seed": SEED,
        "epochs": EPOCHS,
        "optimizer": {"name": "AdamW", "learning_rate": LR, "weight_decay": WEIGHT_DECAY},
        "device": "cpu",
        "elapsed_seconds": elapsed,
        "representations_computed_once": True,
        "pre_update_m9a_metrics": pre_update,
        "oof": oof,
        "m8c_oof_comparison": _oof_comparison(oof, m8c_aggregate.get("oof", {})),
        "final_losses": {factor: float(value.detach().cpu()) for factor, value in final_losses.items()},
        "final_total_loss": float(final_total.detach().cpu()),
        "final_loss_start": float(final_history[0]),
        "final_loss_end": float(final_history[-1]),
        "checkpoint": str(checkpoint),
        "checkpoint_reload_equal": reload_equal,
        "fresh_pool_path": str(Path(fresh_pool_path).resolve()),
        "fresh_pool_sha256": sha256_file(fresh_pool_path),
        "fresh_pool_input_rows": len(fresh_rows),
        "fresh_shadow_predictions": str(output / "fresh-2000-shadow-predictions.jsonl"),
        "fresh_before_m8c_summary": fresh_before_summary,
        "fresh_after_m9b_summary": fresh_after_summary,
        "fresh_active_after_m9b_summary_excluding_m9a_pairs": fresh_active_summary,
        "discourse_verdict": discourse_gate,
        "head_conclusions": conclusions,
        "forbidden_data_access": {"test": False, "gold": False, "ood": False, "anchor": False, "human100": False, "human48": False, "llm": False},
    }
    _write_jsonl(output / "fresh-2000-shadow-predictions.jsonl", fresh_after)
    (output / "pre-update-m9a-metrics.json").write_text(json.dumps(pre_update, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (output / "oof-metrics.json").write_text(json.dumps(oof, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (output / "aggregate-metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    report_lines = [
        "# M9B cumulative 174 auxiliary effect gate",
        "",
        "This is one diagnostic run. The selected LERT seven-head model and encoder stayed frozen; only the five v0.3 auxiliary linear heads were reinitialized once and fitted on the cumulative sparse assignments. No production promotion is made.",
        "",
        f"- Assignments/samples: {len(effective_assignments)} / {len(records)}; masks: `{json.dumps(summary['cumulative_factor_masks'], ensure_ascii=False, sort_keys=True)}`",
        f"- Device/time: {summary['device']} / {elapsed:.2f}s; seed={SEED}; epochs={EPOCHS}; AdamW lr={LR} wd={WEIGHT_DECAY}",
        f"- M9A pull: 50/50 final, no disposition, unique pairs; per-head `{json.dumps(M9A_COUNTS, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## M9A pre-update (M8C checkpoint)",
        "",
    ]
    for factor, metrics in pre_update.items():
        report_lines.append(f"- `{factor}`: `{json.dumps(metrics, ensure_ascii=False, sort_keys=True)}`")
    report_lines.extend(["", "## Cumulative 3-fold OOF", "", "OOF metrics are descriptive. Training-set fit and all-negative/majority baselines are not treated as success evidence.", "", "M8C comparison is descriptive only because the sample sizes differ (M8C 124 assignments/93 task samples versus M9B 174 assignments/143 task samples)."])
    for factor, item in oof["aggregate"].items():
        report_lines.append(f"- `{factor}` ({item['diagnostic_class']}, n={item['label_count']}): metrics `{json.dumps(item['metrics'], ensure_ascii=False, sort_keys=True)}`; baseline `{json.dumps(item['baseline'], ensure_ascii=False, sort_keys=True)}`; prediction diagnostics `{json.dumps(item['prediction_diagnostics'], ensure_ascii=False, sort_keys=True)}`")
    for factor, item in summary["m8c_oof_comparison"].items():
        report_lines.append(f"- OOF delta `{factor}` (M9B n={item['m9b_sample_count']} vs M8C n={item['m8c_sample_count']}): `{json.dumps(item['delta_m9b_minus_m8c'], ensure_ascii=False, sort_keys=True)}`")
    report_lines.extend(["", "## Discourse gate", "", f"`{discourse_gate}` — fixed rule: OOF micro-F1 > 0.15 and empty-prediction rate < 80%; no threshold adjustment was used.", ""])
    report_lines.extend(["## Fresh-2000 shadow", "", "The same 2,000-row fresh pool was processed after fitting. Long CONTENT rows were retained; applicability remains explicit-text-only. The active summary removes only M9A-covered sample/head pairs, not whole rows.", ""])
    for factor in V03_AUXILIARY_FACTORS:
        before = fresh_before_summary["factor_summary"][factor]
        after = fresh_after_summary["factor_summary"][factor]
        active = fresh_active_summary["factor_summary"][factor]
        report_lines.append(f"- `{factor}`: M8C empty={before.get('empty_prediction_rate')}; M9B all empty={after.get('empty_prediction_rate')}; M9B active empty={active.get('empty_prediction_rate')}; M9B all distribution={json.dumps(after.get('prediction_distribution', {}), ensure_ascii=False, sort_keys=True)}")
    report_lines.extend(["", "## Head conclusions", "", *[f"- `{factor}`: **{conclusions[factor]}**" for factor in V03_AUXILIARY_FACTORS], "", "This artifact remains diagnostic/shadow-only. It does not replace the selected candidate and does not create a new annotation batch. Test, Gold, OOD, Anchor, Human100/48, LLM and production data were not read.", ""])
    (output / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    files = [path for path in output.iterdir() if path.is_file() and path.name != "manifest.json"]
    manifest = {"artifact": "m9b-v03-cumulative174-effect-gate-20260914", "status": "DIAGNOSTIC_ONLY", "files": {path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(files)}}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m8c-tasks", type=Path, default=DEFAULT_M8C_TASKS)
    parser.add_argument("--m8c-assignments", type=Path, default=DEFAULT_M8C_ASSIGNMENTS)
    parser.add_argument("--m8c-checkpoint", type=Path, default=DEFAULT_M8C_CHECKPOINT)
    parser.add_argument("--m8c-fresh-predictions", type=Path, default=DEFAULT_M8C_FRESH_PREDICTIONS)
    parser.add_argument("--m8c-aggregate", type=Path, default=DEFAULT_M8C_AGGREGATE)
    parser.add_argument("--m9a-tasks", type=Path, default=DEFAULT_M9A_TASKS)
    parser.add_argument("--m9a-assignments", type=Path, default=DEFAULT_M9A_ASSIGNMENTS)
    parser.add_argument("--fresh-pool", type=Path, default=DEFAULT_FRESH_POOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--selected-artifact", type=Path, default=DEFAULT_SELECTED_ARTIFACT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--schema", type=Path, default=DEFAULT_CANDIDATE_SCHEMA)
    args = parser.parse_args(argv)
    summary = run_m9b(
        m8c_tasks_path=args.m8c_tasks,
        m8c_assignments_path=args.m8c_assignments,
        m8c_checkpoint_path=args.m8c_checkpoint,
        m8c_fresh_predictions_path=args.m8c_fresh_predictions,
        m8c_aggregate_path=args.m8c_aggregate,
        m9a_tasks_path=args.m9a_tasks,
        m9a_assignments_path=args.m9a_assignments,
        fresh_pool_path=args.fresh_pool,
        output_dir=args.output_dir,
        selected_artifact=args.selected_artifact,
        cache_dir=args.cache_dir,
        config_path=args.config,
        schema_path=args.schema,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
