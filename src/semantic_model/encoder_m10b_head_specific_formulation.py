"""M10B head-specific multi-label formulation diagnostic.

M10A showed that one policy should not be forced to serve both multi-label
heads.  This runner therefore evaluates only two frozen choices:

* ``market_scope``: ordinary BCE/0.5 (M10A A) versus four-class softmax with
  ordinary cross entropy;
* ``discourse_focus``: M10A D (fold-local ``pos_weight`` BCE, threshold 0.5,
  UNKNOWN exclusivity and top-1 fallback) versus its ordinary-BCE A reference.

The M9B 143-sample/174-assignment data, deterministic three folds, seed,
initialization, optimizer, learning rate and 50 epochs are shared.  No other
auxiliary or legacy head is trained.  A market softmax winner is allowed to
form a diagnostic head-specific checkpoint; discourse's decision is always
independent of the market result.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .auxiliary_heads import AuxiliaryHeadModel, AuxiliaryTargetBatch, build_auxiliary_targets
from .encoder_m10a_multilabel_ablation import (
    EPOCHS,
    LR,
    MULTI_FACTORS,
    THRESHOLD,
    WEIGHT_DECAY,
    apply_multi_label_policy,
    fit_factor,
    multilabel_metrics,
)
from .encoder_m6c_auxiliary import AuxiliaryTrainingRuntime, DEFAULT_CANDIDATE_SCHEMA, load_auxiliary_task_views
from .encoder_m8c_cumulative import (
    _fresh_input_rows,
    _fresh_record,
    _multi_entropy,
    _single_prediction,
    _load_checkpoint,
    summarize_fresh_rows,
)
from .encoder_m9b_effect_gate import (
    DEFAULT_CACHE,
    DEFAULT_CONFIG,
    DEFAULT_FRESH_POOL,
    DEFAULT_SELECTED_ARTIFACT,
    EXPECTED_ASSIGNMENTS,
    EXPECTED_MASKS,
    EXPECTED_SAMPLES,
    _clone_state,
    read_jsonl,
    sha256_file,
)
from .errors import ContractError
from .schema import SemanticSchemaCandidate


SEED = 20260913
MARKET = "market_scope"
DISCOURSE = "discourse_focus"
M9B_ARTIFACT = Path(__file__).resolve().parents[3] / "model-artifacts" / "m9b-v03-cumulative174-effect-gate-20260914"
M10A_ARTIFACT = Path(__file__).resolve().parents[3] / "model-artifacts" / "m10a-v03-multilabel-mechanism-ablation-20260914"
DEFAULT_M9B_TASKS = M9B_ARTIFACT / "tasks-cumulative-143.jsonl"
DEFAULT_M9B_ASSIGNMENTS = M9B_ARTIFACT / "assignments-cumulative-174.jsonl"
DEFAULT_M9B_CHECKPOINT = M9B_ARTIFACT / "auxiliary-checkpoint.pt"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[3] / "model-artifacts" / "m10b-v03-head-specific-formulation-20260914"


def _require(ok: bool, code: str, message: str, **details: Any) -> None:
    if not ok:
        raise ContractError(code, message, **details)


def _slice_indices(values: Tensor, indices: Sequence[int]) -> Tensor:
    return values.index_select(0, torch.tensor(list(indices), dtype=torch.long, device=values.device))


def _folds(sample_ids: Sequence[str]) -> dict[str, int]:
    ordered = sorted(sample_ids)
    _require(len(ordered) == len(set(ordered)), "M10B_SAMPLE_ID_DUPLICATE", "sample IDs must be unique for OOF")
    return {sample_id: index % 3 for index, sample_id in enumerate(ordered)}


def run_bce_oof(
    factor: str,
    representations: Mapping[str, Tensor],
    batch: AuxiliaryTargetBatch,
    sample_ids: Sequence[str],
    schema: SemanticSchemaCandidate,
    *,
    initial_state: Mapping[str, Tensor],
    use_pos_weight: bool,
    fallback: bool,
) -> dict[str, Any]:
    """Run one multi-label policy over the fixed sample-grouped folds."""

    _require(factor in MULTI_FACTORS, "M10B_FACTOR_INVALID", "M10B BCE OOF supports only the two multi-label factors", factor=factor)
    fold_assignment = _folds(sample_ids)
    classes = len(schema.auxiliary_factors[factor].class_order)
    oof_logits = torch.zeros((len(sample_ids), classes), dtype=torch.float32)
    fold_reports: list[dict[str, Any]] = []
    for fold in range(3):
        train_indices = [index for index, sample_id in enumerate(sample_ids) if fold_assignment[sample_id] != fold]
        valid_indices = [index for index, sample_id in enumerate(sample_ids) if fold_assignment[sample_id] == fold]
        model, history, weight_report = fit_factor(initial_state, representations, batch, schema, factor, train_indices, use_pos_weight=use_pos_weight, epochs=EPOCHS)
        owner = model.owner_head_by_factor[factor]
        valid_representations = representations[owner].index_select(0, torch.tensor(valid_indices, dtype=torch.long))
        with torch.no_grad():
            oof_logits[valid_indices] = model.auxiliary_heads[factor](valid_representations).detach().cpu()
        valid_target = _slice_indices(batch.targets[factor], valid_indices)
        valid_mask = _slice_indices(batch.loss_masks[factor], valid_indices)
        fold_reports.append({
            "fold": fold,
            "train_samples": len(train_indices),
            "valid_samples": len(valid_indices),
            "metrics": multilabel_metrics(oof_logits[valid_indices], valid_target, valid_mask, schema.auxiliary_factors[factor].class_order, fallback=fallback),
            "training_loss_start": history[0] if history else 0.0,
            "training_loss_end": history[-1] if history else 0.0,
            **weight_report,
        })
    aggregate = multilabel_metrics(oof_logits, batch.targets[factor], batch.loss_masks[factor], schema.auxiliary_factors[factor].class_order, fallback=fallback)
    return {"factor": factor, "use_pos_weight": use_pos_weight, "fallback": fallback, "fold_assignment": fold_assignment, "folds": fold_reports, "aggregate": aggregate, "oof_logits": oof_logits}


def _softmax_metrics(logits: Tensor, target: Tensor, mask: Tensor, class_order: Sequence[str]) -> dict[str, Any]:
    valid = mask > 0
    values = logits[valid]
    target_values = target[valid]
    n = int(values.shape[0])
    if n == 0:
        return {"n": 0, "accuracy": None, "macro_f1": None, "macro_f1_supported": None, "exact_match": None, "micro_f1": None, "per_class": {}, "confusion_matrix": [], "mean_confidence": None}
    _require(bool(torch.all(target_values.sum(dim=1) == 1).item()), "M10B_MARKET_CARDINALITY_INVALID", "market_scope softmax requires exactly one target class per observed row")
    probabilities = torch.softmax(values, dim=-1).detach().cpu().numpy().astype(float)
    truth = target_values.argmax(dim=-1).detach().cpu().numpy().astype(int)
    prediction = probabilities.argmax(axis=1).astype(int)
    confusion = np.zeros((len(class_order), len(class_order)), dtype=int)
    per_class: dict[str, dict[str, float | int]] = {}
    tp_total = fp_total = fn_total = 0.0
    for index, label in enumerate(class_order):
        confusion[index, :] = np.bincount(prediction[truth == index], minlength=len(class_order))
        tp = float(np.sum((prediction == index) & (truth == index)))
        fp = float(np.sum((prediction == index) & (truth != index)))
        fn = float(np.sum((prediction != index) & (truth == index)))
        support = int(np.sum(truth == index))
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        f1 = 0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        tp_total += tp
        fp_total += fp
        fn_total += fn
    micro = 0.0 if 2 * tp_total + fp_total + fn_total == 0 else 2 * tp_total / (2 * tp_total + fp_total + fn_total)
    supported = [float(item["f1"]) for item in per_class.values() if int(item["support"]) > 0]
    return {
        "n": n,
        "accuracy": float(np.mean(prediction == truth)),
        "macro_f1": float(np.mean([float(item["f1"]) for item in per_class.values()])),
        "macro_f1_supported": float(np.mean(supported)) if supported else 0.0,
        "exact_match": float(np.mean(prediction == truth)),
        "micro_f1": micro,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "mean_confidence": float(probabilities.max(axis=1).mean()),
        "prediction_distribution": {label: int(np.sum(prediction == index)) for index, label in enumerate(class_order)},
        "true_distribution": {label: int(np.sum(truth == index)) for index, label in enumerate(class_order)},
    }


def fit_market_softmax(
    initial_state: Mapping[str, Tensor],
    representations: Mapping[str, Tensor],
    batch: AuxiliaryTargetBatch,
    schema: SemanticSchemaCandidate,
    train_indices: Sequence[int],
    *,
    epochs: int = EPOCHS,
) -> tuple[AuxiliaryHeadModel, list[float]]:
    hidden_size = int(next(iter(representations.values())).shape[-1])
    model = AuxiliaryHeadModel(schema, hidden_size)
    model.load_state_dict(initial_state)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("auxiliary_heads.market_scope.")
    owner = model.owner_head_by_factor[MARKET]
    indices = torch.tensor(list(train_indices), dtype=torch.long)
    train_representations = representations[owner].index_select(0, indices)
    target_multi = batch.targets[MARKET].index_select(0, indices)
    mask = batch.loss_masks[MARKET].index_select(0, indices)
    observed = mask > 0
    _require(bool(torch.all(target_multi[observed].sum(dim=1) == 1).item()), "M10B_MARKET_CARDINALITY_INVALID", "market_scope softmax requires one class per observed target")
    target_class = target_multi.argmax(dim=-1)
    optimizer = torch.optim.AdamW(model.auxiliary_heads[MARKET].parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(reduction="none")
    history: list[float] = []
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model.auxiliary_heads[MARKET](train_representations)
        per_row = criterion(logits, target_class)
        observed_float = mask.to(dtype=logits.dtype)
        denominator = observed_float.sum()
        loss = logits.sum() * 0.0 if denominator.item() == 0 else (per_row * observed_float).sum() / denominator
        _require(bool(torch.isfinite(loss).item()), "M10B_NONFINITE_LOSS", "market softmax loss is non-finite")
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))
    model.eval()
    return model, history


def run_market_softmax_oof(
    representations: Mapping[str, Tensor],
    batch: AuxiliaryTargetBatch,
    sample_ids: Sequence[str],
    schema: SemanticSchemaCandidate,
    *,
    initial_state: Mapping[str, Tensor],
) -> dict[str, Any]:
    fold_assignment = _folds(sample_ids)
    classes = len(schema.auxiliary_factors[MARKET].class_order)
    logits = torch.zeros((len(sample_ids), classes), dtype=torch.float32)
    fold_reports: list[dict[str, Any]] = []
    for fold in range(3):
        train_indices = [index for index, sample_id in enumerate(sample_ids) if fold_assignment[sample_id] != fold]
        valid_indices = [index for index, sample_id in enumerate(sample_ids) if fold_assignment[sample_id] == fold]
        model, history = fit_market_softmax(initial_state, representations, batch, schema, train_indices)
        owner = model.owner_head_by_factor[MARKET]
        valid_representations = representations[owner].index_select(0, torch.tensor(valid_indices, dtype=torch.long))
        with torch.no_grad():
            logits[valid_indices] = model.auxiliary_heads[MARKET](valid_representations).detach().cpu()
        fold_reports.append({
            "fold": fold,
            "train_samples": len(train_indices),
            "valid_samples": len(valid_indices),
            "metrics": _softmax_metrics(logits[valid_indices], _slice_indices(batch.targets[MARKET], valid_indices), _slice_indices(batch.loss_masks[MARKET], valid_indices), schema.auxiliary_factors[MARKET].class_order),
            "training_loss_start": history[0] if history else 0.0,
            "training_loss_end": history[-1] if history else 0.0,
        })
    aggregate = _softmax_metrics(logits, batch.targets[MARKET], batch.loss_masks[MARKET], schema.auxiliary_factors[MARKET].class_order)
    return {"factor": MARKET, "formulation": "four_class_softmax_cross_entropy", "fold_assignment": fold_assignment, "folds": fold_reports, "aggregate": aggregate, "oof_logits": logits}


def market_softmax_passes(ordinary_bce: Mapping[str, Any], softmax: Mapping[str, Any]) -> bool:
    """The frozen market gate: accuracy and macro-F1 must both beat A."""

    a = ordinary_bce["aggregate"]
    s = softmax["aggregate"]
    # A is a multi-label report, so its strict accuracy is the exact-set
    # metric.  This is the .375 reference frozen by M10A.
    baseline_accuracy = a.get("accuracy", a.get("exact_match"))
    return float(s["accuracy"]) > float(baseline_accuracy) and float(s["macro_f1"]) > float(a["macro_f1"])


def discourse_policy_passes(ordinary_bce: Mapping[str, Any], weighted_fallback: Mapping[str, Any]) -> bool:
    """D is independently adoptable when both F1 metrics improve."""

    a = ordinary_bce["aggregate"]
    d = weighted_fallback["aggregate"]
    if not (float(d["micro_f1"]) > float(a["micro_f1"]) and float(d["macro_f1"]) > float(a["macro_f1"])):
        return False
    per_label = d.get("per_label", {})
    non_author = [label for label, values in per_label.items() if label not in {"AUTHOR_VIEWPOINT", "UNKNOWN"} and float(values.get("f1", 0.0)) > 0.0]
    return bool(non_author)


def _head_specific_shadow(
    runtime: AuxiliaryTrainingRuntime,
    model: AuxiliaryHeadModel,
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    from .encoder_m8c_cumulative import _applicable_factors

    records = [_fresh_record(row) for row in rows]
    output: list[dict[str, Any]] = []
    runtime.model.eval()
    model.eval()
    for offset in range(0, len(records), 16):
        current_records = records[offset : offset + 16]
        current_rows = rows[offset : offset + 16]
        inputs = runtime.build_inputs(current_records, "cpu")
        with torch.no_grad():
            captured = runtime.model(inputs["input_ids"], inputs["attention_mask"])
            logits = {factor: model.auxiliary_heads[factor](captured["owner_representations"][model.owner_head_by_factor[factor]]) for factor in MULTI_FACTORS}
            logits.update({factor: model.auxiliary_heads[factor](captured["owner_representations"][model.owner_head_by_factor[factor]]) for factor in ("expectation_phase", "arousal_level", "desired_price_direction")})
        for index, (record, source) in enumerate(zip(current_records, current_rows)):
            applicable = list(_applicable_factors(record.model_text))
            row: dict[str, Any] = {"sample_id": record.sample_id, "model_text": record.model_text, "stock_code": record.stock_code, "stock_name": record.stock_name, "text_length": source.get("text_length", len(record.model_text)), "text_source": source.get("text_source"), "exact_content_group": source.get("exact_content_group"), "applicable_factors": applicable, "auxiliary": {}}
            for factor in applicable:
                spec = runtime.schema.auxiliary_factors[factor]
                vector = logits[factor][index]
                if factor == DISCOURSE:
                    probabilities, binary = apply_multi_label_policy(vector.unsqueeze(0), spec.class_order, fallback=True)
                    probabilities = probabilities[0]
                    prediction = [label for column, label in enumerate(spec.class_order) if binary[0, column]]
                    entry = {"probabilities": {label: float(probabilities[column]) for column, label in enumerate(spec.class_order)}, "prediction": prediction, "confidence": float(probabilities.max()), "normalized_entropy": _multi_entropy(probabilities.tolist()), "threshold": THRESHOLD}
                elif factor == MARKET:
                    probabilities = torch.softmax(vector, dim=-1).detach().cpu().numpy().astype(float)
                    prediction = [spec.class_order[int(np.argmax(probabilities))]]
                    entry = {"probabilities": {label: float(probabilities[column]) for column, label in enumerate(spec.class_order)}, "prediction": prediction, "confidence": float(probabilities.max()), "normalized_entropy": float(-np.sum(np.clip(probabilities, 1e-12, 1.0) * np.log(np.clip(probabilities, 1e-12, 1.0))) / np.log(max(2, len(probabilities))))}
                else:
                    entry = _single_prediction(vector, spec.class_order)
                row["auxiliary"][factor] = entry
            output.append(row)
    _require(len(output) == len(rows), "M10B_FRESH_OUTPUT_COUNT_INVALID", "head-specific shadow output count differs from input")
    return output


def _jsonable_policy(result: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "oof_logits"}


def run_m10b(
    *,
    tasks_path: str | Path = DEFAULT_M9B_TASKS,
    assignments_path: str | Path = DEFAULT_M9B_ASSIGNMENTS,
    m9b_checkpoint_path: str | Path = DEFAULT_M9B_CHECKPOINT,
    m10a_metrics_path: str | Path = M10A_ARTIFACT / "oof-metrics.json",
    output_dir: str | Path = DEFAULT_OUTPUT,
    selected_artifact: str | Path = DEFAULT_SELECTED_ARTIFACT,
    cache_dir: str | Path = DEFAULT_CACHE,
    config_path: str | Path = DEFAULT_CONFIG,
    schema_path: str | Path = DEFAULT_CANDIDATE_SCHEMA,
    fresh_pool_path: str | Path = DEFAULT_FRESH_POOL,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    _require(not output.exists(), "M10B_OUTPUT_EXISTS", "M10B output directory must be new", path=str(output))
    schema = SemanticSchemaCandidate.load(schema_path)
    tasks = read_jsonl(tasks_path)
    assignments = read_jsonl(assignments_path)
    _require(len(tasks) == EXPECTED_SAMPLES, "M10B_TASK_COUNT_INVALID", "M9B tasks must contain 143 samples", observed=len(tasks))
    _require(len(assignments) == EXPECTED_ASSIGNMENTS, "M10B_ASSIGNMENT_COUNT_INVALID", "M9B assignments must contain 174 rows", observed=len(assignments))
    records, views = load_auxiliary_task_views(tasks, assignments, schema=schema)
    batch = build_auxiliary_targets(views, schema, device="cpu")
    observed_masks = {factor: int(batch.loss_masks[factor].sum().item()) for factor in MULTI_FACTORS}
    _require(observed_masks == {factor: EXPECTED_MASKS[factor] for factor in MULTI_FACTORS}, "M10B_MASK_COUNT_INVALID", "M9B multi-label masks differ", observed=observed_masks)
    runtime = AuxiliaryTrainingRuntime.from_selected_artifact(selected_artifact, config_path=config_path, cache_dir=cache_dir)
    model_inputs = runtime.build_inputs(records, "cpu")
    with torch.no_grad():
        captured = runtime.model(model_inputs["input_ids"], model_inputs["attention_mask"])
    representations = {owner: values.detach().cpu() for owner, values in captured["owner_representations"].items()}
    hidden_size = int(next(iter(representations.values())).shape[-1])
    torch.manual_seed(SEED)
    template = AuxiliaryHeadModel(schema, hidden_size)
    initial_state = _clone_state(template.state_dict())
    sample_ids = [record.sample_id for record in records]
    started = time.monotonic()
    market_a = run_bce_oof(MARKET, representations, batch, sample_ids, schema, initial_state=initial_state, use_pos_weight=False, fallback=False)
    market_softmax = run_market_softmax_oof(representations, batch, sample_ids, schema, initial_state=initial_state)
    discourse_a = run_bce_oof(DISCOURSE, representations, batch, sample_ids, schema, initial_state=initial_state, use_pos_weight=False, fallback=False)
    discourse_d = run_bce_oof(DISCOURSE, representations, batch, sample_ids, schema, initial_state=initial_state, use_pos_weight=True, fallback=True)
    elapsed = time.monotonic() - started
    market_pass = market_softmax_passes(market_a, market_softmax)
    discourse_pass = discourse_policy_passes(discourse_a, discourse_d)
    m10a_reference = json.loads(Path(m10a_metrics_path).read_text(encoding="utf-8")) if Path(m10a_metrics_path).is_file() else None
    policy: dict[str, Any] = {
        "market_scope": {"baseline": "M10A_A_BCE_THRESHOLD", "candidate": "FOUR_CLASS_SOFTMAX_CROSS_ENTROPY", "candidate_pass": market_pass, "gate": "softmax accuracy and macro_f1 must both strictly exceed A", "selected_formulation": "four_class_softmax_cross_entropy" if market_pass else "M10A_A_BCE_THRESHOLD"},
        "discourse_focus": {"baseline": "M10A_A_BCE_THRESHOLD", "candidate": "M10A_D_POS_WEIGHT_BCE_TOP1_FALLBACK", "candidate_pass": discourse_pass, "independent_of_market": True, "selected_formulation": "M10A_D_POS_WEIGHT_BCE_TOP1_FALLBACK" if discourse_pass else "M10A_A_BCE_THRESHOLD"},
    }
    summary: dict[str, Any] = {
        "component": "M10B_V03_HEAD_SPECIFIC_FORMULATION",
        "status": "DIAGNOSTIC_ONLY",
        "selected_candidate": False,
        "seed": SEED,
        "epochs": EPOCHS,
        "optimizer": {"name": "AdamW", "learning_rate": LR, "weight_decay": WEIGHT_DECAY},
        "threshold": THRESHOLD,
        "cumulative_assignment_count": len(assignments),
        "cumulative_sample_count": len(records),
        "multi_label_masks": observed_masks,
        "device": "cpu",
        "elapsed_seconds": elapsed,
        "same_sample_grouped_folds": True,
        "market_scope": {"ordinary_bce": _jsonable_policy(market_a), "softmax": _jsonable_policy(market_softmax)},
        "discourse_focus": {"ordinary_bce": _jsonable_policy(discourse_a), "pos_weight_fallback": _jsonable_policy(discourse_d)},
        "m10a_reference": m10a_reference,
        "policy": policy,
        "head_specific_policy": {"expectation_phase": "M9B checkpoint unchanged", "arousal_level": "M9B checkpoint unchanged", "desired_price_direction": "M9B checkpoint unchanged", "market_scope": policy[MARKET]["selected_formulation"], "discourse_focus": policy[DISCOURSE]["selected_formulation"]},
        "forbidden_data_access": {"test": False, "gold": False, "ood": False, "anchor": False, "llm": False},
    }
    output.mkdir(parents=True, exist_ok=False)
    _json_write = lambda path, value: path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    _json_write(output / "oof-metrics.json", summary)
    _json_write(output / "head-specific-policy.json", policy)
    output.joinpath("tasks-snapshot.jsonl").write_text("".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in tasks), encoding="utf-8")
    output.joinpath("assignments-snapshot.jsonl").write_text("".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in assignments), encoding="utf-8")
    checkpoint_path: Path | None = None
    fresh_summary: dict[str, Any] | None = None
    if market_pass:
        parent = _load_checkpoint(m9b_checkpoint_path, schema, hidden_size)
        market_full, _ = fit_market_softmax(initial_state, representations, batch, schema, list(range(len(records))))
        discourse_full, _, _ = fit_factor(initial_state, representations, batch, schema, DISCOURSE, list(range(len(records))), use_pos_weight=True, epochs=EPOCHS)
        head_specific = AuxiliaryHeadModel(schema, hidden_size)
        head_specific.load_state_dict(parent.state_dict())
        head_specific.auxiliary_heads[MARKET].load_state_dict(market_full.auxiliary_heads[MARKET].state_dict())
        head_specific.auxiliary_heads[DISCOURSE].load_state_dict(discourse_full.auxiliary_heads[DISCOURSE].state_dict())
        checkpoint_path = output / "head-specific-auxiliary-checkpoint.pt"
        torch.save({"component": "m10b_v03_head_specific_formulation", "seed": SEED, "epochs": EPOCHS, "market_formulation": "four_class_softmax_cross_entropy", "discourse_formulation": "pos_weight_bce_threshold_top1_fallback", "parent_m9b_checkpoint": str(m9b_checkpoint_path), "auxiliary_state_dict": head_specific.state_dict()}, checkpoint_path)
        reloaded = _load_checkpoint(checkpoint_path, schema, hidden_size)
        with torch.no_grad():
            before_logits = head_specific({owner: value for owner, value in representations.items()})
            after_logits = reloaded({owner: value for owner, value in representations.items()})
        _require(all(torch.equal(before_logits[factor], after_logits[factor]) for factor in before_logits), "M10B_CHECKPOINT_RELOAD_MISMATCH", "head-specific checkpoint reload differs")
        fresh_rows = _fresh_input_rows(fresh_pool_path)
        fresh_predictions = _head_specific_shadow(runtime, reloaded, fresh_rows)
        fresh_summary = summarize_fresh_rows(fresh_predictions)
        output.joinpath("fresh-2000-shadow-predictions.jsonl").write_text("".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in fresh_predictions), encoding="utf-8")
        _json_write(output / "fresh-shadow-summary.json", fresh_summary)
        summary["checkpoint"] = str(checkpoint_path)
        summary["checkpoint_reload_equal"] = True
        summary["fresh_shadow_summary"] = fresh_summary
    else:
        summary["checkpoint"] = None
        summary["checkpoint_reload_equal"] = None
        summary["fresh_shadow_summary"] = None
    _json_write(output / "oof-metrics.json", summary)
    report_lines = [
        "# M10B head-specific formulation diagnostic",
        "",
        "Only `market_scope` formulation and `discourse_focus` D were evaluated. The encoder, legacy seven heads and the other three auxiliary factors stayed unchanged.",
        "",
        f"- Input: {len(assignments)} assignments / {len(records)} samples; masks `{json.dumps(observed_masks, ensure_ascii=False, sort_keys=True)}`",
        f"- Device/time: CPU / {elapsed:.2f}s; seed={SEED}; epochs={EPOCHS}; AdamW lr={LR}; wd={WEIGHT_DECAY}; threshold={THRESHOLD}",
        "- All policies use the same deterministic sample-grouped three folds and the same initialization.",
        "",
        "## Market scope",
        "",
        f"- A ordinary BCE: `{json.dumps(market_a['aggregate'], ensure_ascii=False, sort_keys=True)}`",
        f"- Four-class softmax: `{json.dumps(market_softmax['aggregate'], ensure_ascii=False, sort_keys=True)}`",
        f"- Gate: `{'PASS' if market_pass else 'FORMULATION_UNRESOLVED'}`; accuracy and macro-F1 must both strictly exceed A.",
        "",
        "## Discourse focus",
        "",
        f"- A ordinary BCE: `{json.dumps(discourse_a['aggregate'], ensure_ascii=False, sort_keys=True)}`",
        f"- D pos-weight BCE + top-1 fallback: `{json.dumps(discourse_d['aggregate'], ensure_ascii=False, sort_keys=True)}`",
        f"- Independent D diagnostic policy: `{'ADOPT_DIAGNOSTIC' if discourse_pass else 'RETAIN_A'}`; market metrics do not veto this decision.",
        f"- D mean predicted cardinality={discourse_d['aggregate']['mean_prediction_cardinality']:.6f}; mean true cardinality={discourse_d['aggregate']['mean_true_cardinality']:.6f}; over-prediction risk is retained in the evidence.",
        "",
        "## Head-specific policy and outcome",
        "",
        *[f"- `{head}`: `{value}`" for head, value in summary["head_specific_policy"].items()],
        f"- Market softmax checkpoint/fresh shadow: `{'generated' if market_pass else 'not_generated_because_market_gate_failed'}`",
        "- This remains diagnostic only; no production model or candidate promotion is created.",
        "",
    ]
    output.joinpath("REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    files = [path for path in output.iterdir() if path.is_file() and path.name != "manifest.json"]
    _json_write(output / "manifest.json", {"artifact": "m10b-v03-head-specific-formulation-20260914", "status": "DIAGNOSTIC_ONLY", "files": {path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(files)}})
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_M9B_TASKS)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_M9B_ASSIGNMENTS)
    parser.add_argument("--m9b-checkpoint", type=Path, default=DEFAULT_M9B_CHECKPOINT)
    parser.add_argument("--m10a-metrics", type=Path, default=M10A_ARTIFACT / "oof-metrics.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--selected-artifact", type=Path, default=DEFAULT_SELECTED_ARTIFACT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--schema", type=Path, default=DEFAULT_CANDIDATE_SCHEMA)
    parser.add_argument("--fresh-pool", type=Path, default=DEFAULT_FRESH_POOL)
    args = parser.parse_args(argv)
    result = run_m10b(tasks_path=args.tasks, assignments_path=args.assignments, m9b_checkpoint_path=args.m9b_checkpoint, m10a_metrics_path=args.m10a_metrics, output_dir=args.output_dir, selected_artifact=args.selected_artifact, cache_dir=args.cache_dir, config_path=args.config, schema_path=args.schema, fresh_pool_path=args.fresh_pool)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
