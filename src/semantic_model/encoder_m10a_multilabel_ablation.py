"""M10A: fixed multi-label empty-prediction mechanism ablation.

The M9B representation and sample-grouped folds are reused exactly.  Only the
two v0.3 multi-label auxiliary factors are evaluated here:
``market_scope`` and ``discourse_focus``.  The selected LERT encoder, the
legacy seven heads, all single-label auxiliary heads, labels and thresholds
remain untouched.

Four policies are frozen before the run:

``A`` ordinary masked BCE with the fixed 0.5 threshold;
``B`` fold-local ``negative / positive`` ``pos_weight`` BCE;
``C`` A plus a one-label argmax fallback when thresholding is empty;
``D`` B plus the same fallback.

UNKNOWN is exclusive for every policy.  This module is diagnostic only.  A
mechanism can be selected only when discourse improves both micro- and
macro-F1 (without becoming an AUTHOR_VIEWPOINT-only output) and market does
not regress on any of its three aggregate metrics.  Otherwise the result is a
frozen-representation or label-formulation redesign signal and no fresh
shadow or replacement checkpoint is produced.
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
from torch import Tensor, nn

from .auxiliary_heads import AuxiliaryHeadModel, AuxiliaryTargetBatch, build_auxiliary_targets
from .encoder_m6c_auxiliary import AuxiliaryTrainingRuntime, DEFAULT_CANDIDATE_SCHEMA, load_auxiliary_task_views
from .encoder_m8c_cumulative import _fresh_input_rows, summarize_fresh_rows
from .encoder_m9b_effect_gate import (
    DEFAULT_CACHE,
    DEFAULT_CONFIG,
    DEFAULT_FRESH_POOL,
    DEFAULT_SELECTED_ARTIFACT,
    EXPECTED_ASSIGNMENTS,
    EXPECTED_MASKS,
    EXPECTED_SAMPLES,
    _clone_state,
    _slice_representations,
    read_jsonl,
    sha256_file,
)
from .errors import ContractError
from .schema import SemanticSchemaCandidate


SEED = 20260913
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 0.0
THRESHOLD = 0.5
MULTI_FACTORS = ("market_scope", "discourse_focus")
POLICIES = ("A_BCE_THRESHOLD", "B_POS_WEIGHT_BCE", "C_BCE_TOP1_FALLBACK", "D_POS_WEIGHT_BCE_TOP1_FALLBACK")
M9B_ARTIFACT = Path(__file__).resolve().parents[3] / "model-artifacts" / "m9b-v03-cumulative174-effect-gate-20260914"
DEFAULT_TASKS = M9B_ARTIFACT / "tasks-cumulative-143.jsonl"
DEFAULT_ASSIGNMENTS = M9B_ARTIFACT / "assignments-cumulative-174.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[3] / "model-artifacts" / "m10a-v03-multilabel-mechanism-ablation-20260914"


def _require(ok: bool, code: str, message: str, **details: Any) -> None:
    if not ok:
        raise ContractError(code, message, **details)


def _factor_model(schema: SemanticSchemaCandidate, hidden_size: int, factor: str, state: Mapping[str, Tensor]) -> AuxiliaryHeadModel:
    model = AuxiliaryHeadModel(schema, hidden_size)
    model.load_state_dict(state)
    # Only this factor's linear layer is ever placed in an optimizer.  The
    # other four layers are retained solely so the shared state format stays
    # compatible with M8C/M9B.
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(f"auxiliary_heads.{factor}.")
    model.eval()
    return model


def _masked_bce_loss(logits: Tensor, target: Tensor, mask: Tensor, *, pos_weight: Tensor | None) -> Tensor:
    mask = mask.to(device=logits.device, dtype=logits.dtype)
    observed = mask.sum()
    if observed.item() == 0:
        return logits.sum() * 0.0
    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    per_row = criterion(logits, target.to(device=logits.device, dtype=logits.dtype)).mean(dim=1)
    value = (per_row * mask).sum() / observed
    _require(bool(torch.isfinite(value).item()), "M10A_NONFINITE_LOSS", "multi-label loss is non-finite")
    return value


def fold_pos_weight(target: Tensor, mask: Tensor, train_indices: Sequence[int]) -> tuple[Tensor, dict[str, Any]]:
    """Compute standard negative/positive weights from the training fold only.

    A class with no positive training examples has no finite standard ratio;
    its weight is recorded as ``1.0`` and remains uninformative rather than
    manufacturing an infinite gradient.
    """

    indices = torch.tensor(list(train_indices), dtype=torch.long, device=target.device)
    fold_mask = mask.index_select(0, indices) > 0
    values = target.index_select(0, indices)[fold_mask]
    classes = target.shape[-1]
    if values.numel() == 0:
        weights = torch.ones(classes, dtype=torch.float32, device=target.device)
        return weights, {"positive": [0] * classes, "negative": [0] * classes, "zero_positive_weight_one": list(range(classes))}
    positives = values.sum(dim=0).to(dtype=torch.float32)
    negatives = values.shape[0] - positives
    weights = torch.ones_like(positives)
    positive = positives > 0
    weights[positive] = negatives[positive] / positives[positive]
    _require(bool(torch.isfinite(weights).all().item()), "M10A_POS_WEIGHT_NONFINITE", "fold-local pos_weight is non-finite")
    return weights, {
        "positive": [int(value) for value in positives.tolist()],
        "negative": [int(value) for value in negatives.tolist()],
        "zero_positive_weight_one": [int(index) for index, value in enumerate(positive.tolist()) if not value],
    }


def fit_factor(
    initial_state: Mapping[str, Tensor],
    representations: Mapping[str, Tensor],
    batch: AuxiliaryTargetBatch,
    schema: SemanticSchemaCandidate,
    factor: str,
    train_indices: Sequence[int],
    *,
    use_pos_weight: bool,
    epochs: int = EPOCHS,
) -> tuple[AuxiliaryHeadModel, list[float], dict[str, Any]]:
    hidden_size = int(next(iter(representations.values())).shape[-1])
    model = _factor_model(schema, hidden_size, factor, initial_state)
    owner = model.owner_head_by_factor[factor]
    head = model.auxiliary_heads[factor]
    train_reps = representations[owner].index_select(0, torch.tensor(list(train_indices), dtype=torch.long))
    train_target = batch.targets[factor].index_select(0, torch.tensor(list(train_indices), dtype=torch.long))
    train_mask = batch.loss_masks[factor].index_select(0, torch.tensor(list(train_indices), dtype=torch.long))
    pos_weight = None
    pos_weight_report: dict[str, Any] | None = None
    if use_pos_weight:
        pos_weight, pos_weight_report = fold_pos_weight(batch.targets[factor], batch.loss_masks[factor], train_indices)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    history: list[float] = []
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = head(train_reps)
        loss = _masked_bce_loss(logits, train_target, train_mask, pos_weight=pos_weight)
        _require(bool(torch.isfinite(loss).item()), "M10A_NONFINITE_LOSS", "multi-label loss is non-finite", factor=factor)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))
    model.eval()
    return model, history, {"pos_weight": pos_weight.detach().cpu().tolist() if pos_weight is not None else None, "pos_weight_counts": pos_weight_report}


def apply_multi_label_policy(logits: Tensor, class_order: Sequence[str], *, fallback: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return sigmoid probabilities and UNKNOWN-exclusive binary predictions."""

    probabilities = torch.sigmoid(logits).detach().cpu().numpy().astype(float)
    _require(bool(np.isfinite(probabilities).all()), "M10A_NONFINITE_OUTPUT", "multi-label probabilities are non-finite")
    predictions = probabilities >= THRESHOLD
    unknown_index = class_order.index("UNKNOWN") if "UNKNOWN" in class_order else None
    for row_index in range(len(predictions)):
        if unknown_index is not None and predictions[row_index, unknown_index]:
            predictions[row_index, :] = False
            predictions[row_index, unknown_index] = True
        elif unknown_index is not None:
            predictions[row_index, unknown_index] = False
        if fallback and not predictions[row_index].any():
            predictions[row_index, int(np.argmax(probabilities[row_index]))] = True
            # The argmax may be UNKNOWN; in either case the exclusivity rule
            # is satisfied because this row has exactly one selected label.
    return probabilities, predictions.astype(np.int8)


def multilabel_metrics(
    logits: Tensor,
    target: Tensor,
    mask: Tensor,
    class_order: Sequence[str],
    *,
    fallback: bool,
) -> dict[str, Any]:
    valid = mask > 0
    values = logits[valid]
    truth = target[valid].detach().cpu().numpy().astype(np.int8)
    _, pred = apply_multi_label_policy(values, class_order, fallback=fallback)
    n = int(len(truth))
    if n == 0:
        return {"n": 0, "micro_f1": None, "macro_f1": None, "exact_match": None, "per_label": {}, "empty_prediction_count": 0, "empty_prediction_rate": 0.0, "mean_prediction_cardinality": 0.0, "mean_true_cardinality": 0.0}
    per_label: dict[str, dict[str, float | int]] = {}
    tp_total = fp_total = fn_total = 0.0
    for index, label in enumerate(class_order):
        tp = float(np.sum((pred[:, index] == 1) & (truth[:, index] == 1)))
        fp = float(np.sum((pred[:, index] == 1) & (truth[:, index] == 0)))
        fn = float(np.sum((pred[:, index] == 0) & (truth[:, index] == 1)))
        support = int(truth[:, index].sum())
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        f1 = 0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        tp_total += tp
        fp_total += fp
        fn_total += fn
    micro = 0.0 if 2 * tp_total + fp_total + fn_total == 0 else 2 * tp_total / (2 * tp_total + fp_total + fn_total)
    empty = int((pred.sum(axis=1) == 0).sum())
    return {
        "n": n,
        "micro_f1": micro,
        "macro_f1": float(np.mean([float(value["f1"]) for value in per_label.values()])),
        "exact_match": float(np.all(pred == truth, axis=1).mean()),
        "per_label": per_label,
        "empty_prediction_count": empty,
        "empty_prediction_rate": float(empty / n),
        "mean_prediction_cardinality": float(pred.sum(axis=1).mean()),
        "mean_true_cardinality": float(truth.sum(axis=1).mean()),
        "predicted_label_counts": {label: int(pred[:, index].sum()) for index, label in enumerate(class_order)},
    }


def run_policy_oof(
    policy: str,
    representations: Mapping[str, Tensor],
    batch: AuxiliaryTargetBatch,
    sample_ids: Sequence[str],
    schema: SemanticSchemaCandidate,
    *,
    initial_state: Mapping[str, Tensor],
    epochs: int = EPOCHS,
) -> dict[str, Any]:
    _require(policy in POLICIES, "M10A_POLICY_INVALID", "unknown frozen mechanism policy", policy=policy)
    use_pos_weight = policy in {"B_POS_WEIGHT_BCE", "D_POS_WEIGHT_BCE_TOP1_FALLBACK"}
    fallback = policy in {"C_BCE_TOP1_FALLBACK", "D_POS_WEIGHT_BCE_TOP1_FALLBACK"}
    ordered = sorted(sample_ids)
    _require(len(ordered) == len(set(ordered)), "M10A_SAMPLE_ID_DUPLICATE", "OOF sample IDs must be unique")
    folds = {sample_id: index % 3 for index, sample_id in enumerate(ordered)}
    logits_by_factor = {factor: torch.zeros((len(sample_ids), len(schema.auxiliary_factors[factor].class_order)), dtype=torch.float32) for factor in MULTI_FACTORS}
    fold_reports: list[dict[str, Any]] = []
    for fold in range(3):
        train_indices = [index for index, sample_id in enumerate(sample_ids) if folds[sample_id] != fold]
        valid_indices = [index for index, sample_id in enumerate(sample_ids) if folds[sample_id] == fold]
        fold_report: dict[str, Any] = {"fold": fold, "train_samples": len(train_indices), "valid_samples": len(valid_indices), "factors": {}}
        for factor in MULTI_FACTORS:
            model, history, weight_report = fit_factor(initial_state, representations, batch, schema, factor, train_indices, use_pos_weight=use_pos_weight, epochs=epochs)
            owner = model.owner_head_by_factor[factor]
            valid_reps = representations[owner].index_select(0, torch.tensor(valid_indices, dtype=torch.long))
            with torch.no_grad():
                logits_by_factor[factor][valid_indices] = model.auxiliary_heads[factor](valid_reps).detach().cpu()
            valid_target = batch.targets[factor].index_select(0, torch.tensor(valid_indices, dtype=torch.long))
            valid_mask = batch.loss_masks[factor].index_select(0, torch.tensor(valid_indices, dtype=torch.long))
            fold_report["factors"][factor] = {
                "metrics": multilabel_metrics(logits_by_factor[factor][valid_indices], valid_target, valid_mask, schema.auxiliary_factors[factor].class_order, fallback=fallback),
                "training_loss_start": history[0] if history else 0.0,
                "training_loss_end": history[-1] if history else 0.0,
                **weight_report,
            }
        fold_reports.append(fold_report)
    aggregate = {
        factor: multilabel_metrics(logits_by_factor[factor], batch.targets[factor], batch.loss_masks[factor], schema.auxiliary_factors[factor].class_order, fallback=fallback)
        for factor in MULTI_FACTORS
    }
    return {"policy": policy, "use_pos_weight": use_pos_weight, "fallback": fallback, "fold_assignment": folds, "folds": fold_reports, "aggregate": aggregate, "oof_logits": logits_by_factor}


def select_policy(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the frozen, conservative selection rule to A/B/C/D results."""

    baseline = results["A_BCE_THRESHOLD"]["aggregate"]
    candidates: list[dict[str, Any]] = []
    discourse_order = ["ASSET_OR_COMPANY", "AUTHOR_VIEWPOINT", "AUTHOR_ACTION_OR_POSITION", "OTHER_PERSON_OR_AUDIENCE", "CLAIM_OR_NARRATIVE", "MARKET_OR_INDUSTRY", "UNKNOWN"]
    for policy in POLICIES[1:]:
        discourse = results[policy]["aggregate"]["discourse_focus"]
        market = results[policy]["aggregate"]["market_scope"]
        base_discourse = baseline["discourse_focus"]
        base_market = baseline["market_scope"]
        non_author = [label for label in discourse_order if label != "AUTHOR_VIEWPOINT" and float(discourse["per_label"].get(label, {}).get("f1", 0.0)) > 0.0]
        discourse_better = float(discourse["micro_f1"]) > float(base_discourse["micro_f1"]) and float(discourse["macro_f1"]) > float(base_discourse["macro_f1"])
        market_ok = all(float(market[key]) >= float(base_market[key]) for key in ("micro_f1", "macro_f1", "exact_match"))
        candidate = {
            "policy": policy,
            "discourse_micro_improved": discourse_better and float(discourse["micro_f1"]) > float(base_discourse["micro_f1"]),
            "discourse_macro_improved": discourse_better and float(discourse["macro_f1"]) > float(base_discourse["macro_f1"]),
            "discourse_non_author_f1_labels": non_author,
            "market_no_regression": market_ok,
            "eligible": bool(discourse_better and bool(non_author) and market_ok),
        }
        candidates.append(candidate)
    eligible = [candidate["policy"] for candidate in candidates if candidate["eligible"]]
    return {
        "baseline_policy": "A_BCE_THRESHOLD",
        "candidates": candidates,
        "selected_policy": eligible[0] if len(eligible) == 1 else None,
        "verdict": "DIAGNOSTIC_MECHANISM_SELECTED" if len(eligible) == 1 else "FROZEN_REPRESENTATION_OR_LABEL_FORMULATION_REDESIGN_REQUIRED",
        "selection_rule": "candidate requires discourse micro-F1 and macro-F1 both strictly above A, at least one non-AUTHOR_VIEWPOINT label with positive F1, and market micro/macro/exact each not below A",
    }


def _fresh_shadow_two_factors(runtime: AuxiliaryTrainingRuntime, models: Mapping[str, AuxiliaryHeadModel], rows: Sequence[Mapping[str, Any]], *, fallback: bool) -> list[dict[str, Any]]:
    from .encoder_m8c_cumulative import _fresh_record, _multi_prediction

    records = [_fresh_record(row) for row in rows]
    output: list[dict[str, Any]] = []
    for model in models.values():
        model.eval()
    runtime.model.eval()
    for offset in range(0, len(records), 16):
        current_records = records[offset : offset + 16]
        current_rows = rows[offset : offset + 16]
        inputs = runtime.build_inputs(current_records, "cpu")
        with torch.no_grad():
            captured = runtime.model(inputs["input_ids"], inputs["attention_mask"])
            logits = {factor: models[factor].auxiliary_heads[factor](captured["owner_representations"][models[factor].owner_head_by_factor[factor]]) for factor in MULTI_FACTORS}
        for index, (record, source) in enumerate(zip(current_records, current_rows)):
            applicable = [factor for factor in MULTI_FACTORS if factor in source.get("applicable_factors", [])]
            row: dict[str, Any] = {"sample_id": record.sample_id, "model_text": record.model_text, "stock_code": record.stock_code, "stock_name": record.stock_name, "text_length": source.get("text_length", len(record.model_text)), "text_source": source.get("text_source"), "exact_content_group": source.get("exact_content_group"), "applicable_factors": applicable, "auxiliary": {}}
            for factor in applicable:
                spec = runtime.schema.auxiliary_factors[factor]
                _, binary = apply_multi_label_policy(logits[factor][index : index + 1], spec.class_order, fallback=fallback)
                probabilities = torch.sigmoid(logits[factor][index]).detach().cpu().numpy().astype(float)
                row["auxiliary"][factor] = {"probabilities": {label: float(probabilities[col]) for col, label in enumerate(spec.class_order)}, "prediction": [label for col, label in enumerate(spec.class_order) if binary[0, col]], "threshold": THRESHOLD}
            output.append(row)
    return output


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def run_m10a(
    *,
    tasks_path: str | Path = DEFAULT_TASKS,
    assignments_path: str | Path = DEFAULT_ASSIGNMENTS,
    output_dir: str | Path = DEFAULT_OUTPUT,
    selected_artifact: str | Path = DEFAULT_SELECTED_ARTIFACT,
    cache_dir: str | Path = DEFAULT_CACHE,
    config_path: str | Path = DEFAULT_CONFIG,
    schema_path: str | Path = DEFAULT_CANDIDATE_SCHEMA,
    fresh_pool_path: str | Path = DEFAULT_FRESH_POOL,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    _require(not output.exists(), "M10A_OUTPUT_EXISTS", "M10A output directory must be new", path=str(output))
    schema = SemanticSchemaCandidate.load(schema_path)
    tasks = read_jsonl(tasks_path)
    assignments = read_jsonl(assignments_path)
    _require(len(tasks) == EXPECTED_SAMPLES, "M10A_TASK_COUNT_INVALID", "M9B cumulative tasks must contain 143 samples", observed=len(tasks))
    _require(len(assignments) == EXPECTED_ASSIGNMENTS, "M10A_ASSIGNMENT_COUNT_INVALID", "M9B cumulative assignments must contain 174 rows", observed=len(assignments))
    records, views = load_auxiliary_task_views(tasks, assignments, schema=schema)
    batch = build_auxiliary_targets(views, schema, device="cpu")
    masks = {factor: int(batch.loss_masks[factor].sum().item()) for factor in MULTI_FACTORS}
    _require(masks == {factor: EXPECTED_MASKS[factor] for factor in MULTI_FACTORS}, "M10A_MASK_COUNT_INVALID", "M9B multi-label masks differ", observed=masks)
    runtime = AuxiliaryTrainingRuntime.from_selected_artifact(selected_artifact, config_path=config_path, cache_dir=cache_dir)
    inputs = runtime.build_inputs(records, "cpu")
    with torch.no_grad():
        captured = runtime.model(inputs["input_ids"], inputs["attention_mask"])
    representations = {owner: values.detach().cpu() for owner, values in captured["owner_representations"].items()}
    hidden_size = int(next(iter(representations.values())).shape[-1])
    torch.manual_seed(SEED)
    initial_template = AuxiliaryHeadModel(schema, hidden_size)
    initial_state = _clone_state(initial_template.state_dict())
    started = time.monotonic()
    raw_results: dict[str, Any] = {}
    for policy in POLICIES:
        result = run_policy_oof(policy, representations, batch, [record.sample_id for record in records], schema, initial_state=initial_state)
        raw_results[policy] = result
    selection = select_policy(raw_results)
    elapsed = time.monotonic() - started
    output.mkdir(parents=True, exist_ok=False)
    result_for_json: dict[str, Any] = {}
    for policy, result in raw_results.items():
        result_for_json[policy] = {key: value for key, value in result.items() if key != "oof_logits"}
    summary: dict[str, Any] = {
        "component": "M10A_V03_MULTILABEL_MECHANISM_ABLATION",
        "status": "DIAGNOSTIC_ONLY",
        "selected_candidate": False,
        "seed": SEED,
        "epochs": EPOCHS,
        "optimizer": {"name": "AdamW", "learning_rate": LR, "weight_decay": WEIGHT_DECAY},
        "threshold": THRESHOLD,
        "factors": list(MULTI_FACTORS),
        "cumulative_assignment_count": len(assignments),
        "cumulative_sample_count": len(records),
        "multi_label_masks": masks,
        "device": "cpu",
        "elapsed_seconds": elapsed,
        "same_sample_grouped_folds": True,
        "policies": result_for_json,
        "selection": selection,
        "forbidden_data_access": {"test": False, "gold": False, "ood": False, "anchor": False, "llm": False},
    }
    _write_json(output / "oof-metrics.json", summary)
    _write_json(output / "selection.json", selection)
    _write_jsonl(output / "tasks-snapshot.jsonl", tasks)
    _write_jsonl(output / "assignments-snapshot.jsonl", assignments)
    # No full-data checkpoint or fresh shadow is emitted when the fixed rule
    # finds no unambiguously better mechanism.  If a future data set produces a
    # unique winner, the runner can materialize that policy explicitly; this
    # run must not manufacture a pseudo-improvement artifact.
    if selection["selected_policy"] is not None:
        winner = selection["selected_policy"]
        use_pos = raw_results[winner]["use_pos_weight"]
        fallback = raw_results[winner]["fallback"]
        models: dict[str, AuxiliaryHeadModel] = {}
        for factor in MULTI_FACTORS:
            models[factor], _, _ = fit_factor(initial_state, representations, batch, schema, factor, list(range(len(records))), use_pos_weight=use_pos)
        fresh_rows = _fresh_input_rows(fresh_pool_path)
        shadow = _fresh_shadow_two_factors(runtime, models, fresh_rows, fallback=fallback)
        _write_jsonl(output / "fresh-2000-shadow-predictions.jsonl", shadow)
        _write_json(output / "fresh-shadow-summary.json", summarize_fresh_rows(shadow))
    report_lines = [
        "# M10A multi-label mechanism ablation",
        "",
        "Only `market_scope` and `discourse_focus` were evaluated. The selected LERT encoder, legacy seven heads, other auxiliary heads, labels and threshold were not changed.",
        "",
        f"- Cumulative input: {len(assignments)} assignments / {len(records)} samples; multi-label masks: `{json.dumps(masks, ensure_ascii=False, sort_keys=True)}`",
        f"- Device/time: CPU / {elapsed:.2f}s; seed={SEED}; epochs={EPOCHS}; AdamW lr={LR}; wd={WEIGHT_DECAY}; threshold={THRESHOLD}",
        "- A/B/C/D use the same deterministic sample-grouped three folds and the same random initialization. Fold-local `pos_weight` uses training labels only.",
        "",
        "## Aggregate OOF comparison",
        "",
        "A = ordinary BCE; B = fold-local pos_weight BCE; C = A plus top-1 fallback for empty threshold output; D = B plus that fallback. UNKNOWN is exclusive in every policy.",
        "",
    ]
    for policy in POLICIES:
        aggregate = raw_results[policy]["aggregate"]
        report_lines.append(f"- `{policy}`: market `{json.dumps(aggregate['market_scope'], ensure_ascii=False, sort_keys=True)}`; discourse `{json.dumps(aggregate['discourse_focus'], ensure_ascii=False, sort_keys=True)}`")
    report_lines.extend(["", "## Selection verdict", "", f"- `{selection['verdict']}`", f"- Selected policy: `{selection['selected_policy']}`", f"- Rule: {selection['selection_rule']}", "- A mechanism is not accepted merely because empty predictions become zero; discourse must improve both micro- and macro-F1 without becoming AUTHOR_VIEWPOINT-only, and market must not regress.", "", "No production model or candidate promotion is produced. Test, Gold, OOD, Anchor and LLM inputs were not read.", ""])
    (output / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    files = [path for path in output.iterdir() if path.is_file() and path.name != "manifest.json"]
    manifest = {"artifact": "m10a-v03-multilabel-mechanism-ablation-20260914", "status": "DIAGNOSTIC_ONLY", "files": {path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(files)}}
    _write_json(output / "manifest.json", manifest)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--selected-artifact", type=Path, default=DEFAULT_SELECTED_ARTIFACT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--schema", type=Path, default=DEFAULT_CANDIDATE_SCHEMA)
    parser.add_argument("--fresh-pool", type=Path, default=DEFAULT_FRESH_POOL)
    args = parser.parse_args(argv)
    result = run_m10a(tasks_path=args.tasks, assignments_path=args.assignments, output_dir=args.output_dir, selected_artifact=args.selected_artifact, cache_dir=args.cache_dir, config_path=args.config, schema_path=args.schema, fresh_pool_path=args.fresh_pool)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
