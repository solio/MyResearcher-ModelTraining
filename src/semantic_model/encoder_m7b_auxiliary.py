"""M7B: one deterministic five-factor auxiliary diagnostic.

This runner is intentionally small and finite.  It consumes answer-free task
JSONL plus final sparse assignments from the labeling app, captures the frozen
selected-candidate LERT representations once, and trains only the five v0.3
auxiliary linear heads.  Three sample-id folds are evaluated out of fold, then
one 50-epoch full-data diagnostic checkpoint is written.  The seven legacy
heads and all encoder parameters stay frozen and are never part of the
optimizer.

The module does not read Test/Gold/OOD data, does not download a model, and
does not implement an epoch runner for any legacy head.  The single known
incomplete assignment is represented by an explicit in-memory mask and is
never changed in the labeling database.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .auxiliary_heads import (
    AuxiliaryHeadModel,
    AuxiliaryTargetBatch,
    build_auxiliary_targets,
    compute_auxiliary_losses,
    total_auxiliary_loss,
)
from .encoder_m6c_auxiliary import (
    DEFAULT_CANDIDATE_SCHEMA,
    AuxiliaryTrainingRuntime,
    load_auxiliary_task_views,
)
from .errors import ContractError
from .schema import SemanticSchemaCandidate, V03_AUXILIARY_FACTORS


SEED = 20260913
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 0.0
EXPECTED_MASKS = {
    "expectation_phase": 24,
    "arousal_level": 7,
    "desired_price_direction": 22,
    "market_scope": 8,
    "discourse_focus": 31,
}
INCOMPLETE_SAMPLE_ID = "fl1-2c7192dc027b67c511861e2b927e27bfa2ed8132eed411729d067ea24fd85564"
INCOMPLETE_HEAD = "discourse_focus"


def _require(ok: bool, code: str, message: str, **details: Any) -> None:
    if not ok:
        raise ContractError(code, message, **details)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8-sig").splitlines()
    except FileNotFoundError as exc:
        raise ContractError("M7B_INPUT_NOT_FOUND", "JSONL input is missing", path=str(source)) from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError("M7B_JSONL_INVALID", "JSONL row is invalid", path=str(source), line=line_number) from exc
        _require(isinstance(value, dict), "M7B_JSONL_INVALID", "JSONL row must be an object", path=str(source), line=line_number)
        rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def combine_inputs(
    task_paths: Sequence[str | Path],
    assignment_paths: Sequence[str | Path],
    *,
    schema: SemanticSchemaCandidate,
) -> tuple[list[Any], list[Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Combine the three final batches and mask only the declared bad pair."""

    _require(len(task_paths) == len(assignment_paths) == 3, "M7B_BATCH_INPUT_COUNT_INVALID", "M7B requires M6B, M7A and supplement task/assignment pairs")
    tasks: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    for task_path, assignment_path in zip(task_paths, assignment_paths):
        tasks.extend(read_jsonl(task_path))
        assignments.extend(read_jsonl(assignment_path))
    sample_ids = [row.get("sample_id") for row in tasks]
    _require(len(tasks) == 64 and len(set(sample_ids)) == 64, "M7B_SAMPLE_COUNT_INVALID", "combined tasks must contain 64 unique samples", observed=len(tasks), unique=len(set(sample_ids)))
    _require(len(assignments) == 93, "M7B_ASSIGNMENT_COUNT_INVALID", "raw final assignments must contain 93 rows", observed=len(assignments))
    filtered: list[dict[str, Any]] = []
    masked = 0
    for row in assignments:
        if row.get("sample_id") == INCOMPLETE_SAMPLE_ID and row.get("head") == INCOMPLETE_HEAD:
            masked += 1
            continue
        filtered.append(row)
    _require(masked == 1, "M7B_INCOMPLETE_PAIR_INVALID", "the declared incomplete pair was not found exactly once", observed=masked)
    _require(len(filtered) == 92, "M7B_VALID_ASSIGNMENT_COUNT_INVALID", "filtered assignments must contain 92 rows", observed=len(filtered))
    records, views = load_auxiliary_task_views(tasks, filtered, schema=schema)
    _require(len(records) == len(views) == 64, "M7B_VIEW_COUNT_INVALID", "merged M7B views must contain 64 samples", records=len(records), views=len(views))
    target_batch = build_auxiliary_targets(views, schema)
    masks = {factor: int(target_batch.loss_masks[factor].sum().item()) for factor in V03_AUXILIARY_FACTORS}
    _require(masks == EXPECTED_MASKS, "M7B_MASK_COUNTS_INVALID", "auxiliary masks differ from the frozen M7B input contract", observed=masks, expected=EXPECTED_MASKS)
    return records, views, assignments, filtered


def deterministic_folds(sample_ids: Sequence[str], n_splits: int = 3) -> dict[str, int]:
    _require(n_splits == 3, "M7B_FOLD_COUNT_INVALID", "M7B requires exactly three folds")
    ordered = sorted(sample_ids)
    _require(len(ordered) == len(set(ordered)), "M7B_SAMPLE_ID_DUPLICATE", "sample IDs must be unique before folding")
    return {sample_id: index % n_splits for index, sample_id in enumerate(ordered)}


def _slice_batch(batch: AuxiliaryTargetBatch, indices: Sequence[int]) -> AuxiliaryTargetBatch:
    index_tensor = torch.tensor(list(indices), dtype=torch.long, device=next(iter(batch.targets.values())).device)
    return AuxiliaryTargetBatch(
        targets={factor: values.index_select(0, index_tensor) for factor, values in batch.targets.items()},
        loss_masks={factor: values.index_select(0, index_tensor) for factor, values in batch.loss_masks.items()},
        sample_ids=tuple(batch.sample_ids[index] for index in indices),
    )


def _slice_representations(representations: Mapping[str, Tensor], indices: Sequence[int]) -> dict[str, Tensor]:
    index_tensor = torch.tensor(list(indices), dtype=torch.long)
    return {owner: values.index_select(0, index_tensor) for owner, values in representations.items()}


def _clone_state(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def _f1(tp: float, fp: float, fn: float) -> float:
    denominator = 2.0 * tp + fp + fn
    return 0.0 if denominator == 0.0 else 2.0 * tp / denominator


def _single_metrics(logits: Tensor, targets: Tensor, mask: Tensor, class_order: Sequence[str]) -> dict[str, Any]:
    valid = mask > 0
    n = int(valid.sum().item())
    if n == 0:
        return {"n": 0, "accuracy": None, "macro_f1": None, "log_loss": None, "per_class_f1": {label: 0.0 for label in class_order}}
    probs = torch.softmax(logits[valid], dim=-1).detach().cpu().numpy()
    truth = targets[valid].detach().cpu().numpy().astype(int)
    pred = probs.argmax(axis=1)
    per_class: dict[str, float] = {}
    for index, label in enumerate(class_order):
        tp = float(np.sum((pred == index) & (truth == index)))
        fp = float(np.sum((pred == index) & (truth != index)))
        fn = float(np.sum((pred != index) & (truth == index)))
        per_class[label] = _f1(tp, fp, fn)
    true_prob = np.clip(probs[np.arange(n), truth], 1e-12, 1.0)
    return {
        "n": n,
        "accuracy": float(np.mean(pred == truth)),
        "macro_f1": float(np.mean(list(per_class.values()))),
        "macro_f1_supported": float(np.mean([value for index, value in enumerate(per_class.values()) if int(np.sum(truth == index)) > 0])) if any(np.sum(truth == index) > 0 for index in range(len(class_order))) else 0.0,
        "log_loss": float(-np.log(true_prob).mean()),
        "per_class_f1": per_class,
        "class_support": {label: int(np.sum(truth == index)) for index, label in enumerate(class_order)},
    }


def _multi_metrics(logits: Tensor, targets: Tensor, mask: Tensor, class_order: Sequence[str]) -> dict[str, Any]:
    valid = mask > 0
    n = int(valid.sum().item())
    if n == 0:
        return {"n": 0, "micro_f1": None, "macro_f1": None, "exact_match": None, "per_label_f1": {label: 0.0 for label in class_order}}
    truth = targets[valid].detach().cpu().numpy().astype(int)
    pred = (torch.sigmoid(logits[valid]).detach().cpu().numpy() >= 0.5).astype(int)
    per_label: dict[str, float] = {}
    tp_total = fp_total = fn_total = 0.0
    for index, label in enumerate(class_order):
        tp = float(np.sum((pred[:, index] == 1) & (truth[:, index] == 1)))
        fp = float(np.sum((pred[:, index] == 1) & (truth[:, index] == 0)))
        fn = float(np.sum((pred[:, index] == 0) & (truth[:, index] == 1)))
        tp_total += tp
        fp_total += fp
        fn_total += fn
        per_label[label] = _f1(tp, fp, fn)
    return {
        "n": n,
        "micro_f1": _f1(tp_total, fp_total, fn_total),
        "macro_f1": float(np.mean(list(per_label.values()))),
        "exact_match": float(np.all(pred == truth, axis=1).mean()),
        "per_label_f1": per_label,
        "label_support": {label: int(truth[:, index].sum()) for index, label in enumerate(class_order)},
    }


def metric_for_factor(factor: str, logits: Tensor, batch: AuxiliaryTargetBatch, schema: SemanticSchemaCandidate) -> dict[str, Any]:
    spec = schema.auxiliary_factors[factor]
    if spec.allow_multiple:
        return _multi_metrics(logits, batch.targets[factor], batch.loss_masks[factor], spec.class_order)
    return _single_metrics(logits, batch.targets[factor], batch.loss_masks[factor], spec.class_order)


def _majority_baseline(factor: str, train_batch: AuxiliaryTargetBatch, valid_batch: AuxiliaryTargetBatch, schema: SemanticSchemaCandidate) -> dict[str, Any]:
    spec = schema.auxiliary_factors[factor]
    if spec.allow_multiple:
        logits = torch.full_like(valid_batch.targets[factor], -20.0)
        return _multi_metrics(logits, valid_batch.targets[factor], valid_batch.loss_masks[factor], spec.class_order)
    train_valid = train_batch.loss_masks[factor] > 0
    if int(train_valid.sum().item()) == 0:
        majority = 0
    else:
        counts = torch.bincount(train_batch.targets[factor][train_valid], minlength=len(spec.class_order))
        majority = int(torch.argmax(counts).item())
    logits = torch.full((len(valid_batch.sample_ids), len(spec.class_order)), -20.0, dtype=torch.float32)
    logits[:, majority] = 20.0
    return _single_metrics(logits, valid_batch.targets[factor], valid_batch.loss_masks[factor], spec.class_order)


def _fit_cached_heads(
    initial_state: Mapping[str, Tensor],
    representations: Mapping[str, Tensor],
    batch: AuxiliaryTargetBatch,
    schema: SemanticSchemaCandidate,
    train_indices: Sequence[int],
    *,
    epochs: int = EPOCHS,
) -> tuple[AuxiliaryHeadModel, list[float]]:
    hidden_size = next(iter(representations.values())).shape[-1]
    model = AuxiliaryHeadModel(schema, int(hidden_size))
    model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_reps = _slice_representations(representations, train_indices)
    train_batch = _slice_batch(batch, train_indices)
    history: list[float] = []
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_reps)
        losses = compute_auxiliary_losses(logits, train_batch, schema)
        total = total_auxiliary_loss(losses)
        _require(bool(torch.isfinite(total).item()), "M7B_NONFINITE_LOSS", "auxiliary loss became non-finite")
        total.backward()
        optimizer.step()
        history.append(float(total.detach().cpu()))
    model.eval()
    return model, history


def run_oof(
    representations: Mapping[str, Tensor],
    batch: AuxiliaryTargetBatch,
    sample_ids: Sequence[str],
    schema: SemanticSchemaCandidate,
    *,
    initial_state: Mapping[str, Tensor],
) -> dict[str, Any]:
    folds = deterministic_folds(sample_ids)
    fold_reports: list[dict[str, Any]] = []
    oof_logits: dict[str, Tensor] = {
        factor: torch.zeros((len(sample_ids), len(schema.auxiliary_factors[factor].class_order)), dtype=torch.float32)
        for factor in V03_AUXILIARY_FACTORS
    }
    baseline_logits: dict[str, Tensor] = {
        factor: torch.zeros_like(oof_logits[factor]) for factor in V03_AUXILIARY_FACTORS
    }
    for fold in range(3):
        train_indices = [index for index, sample_id in enumerate(sample_ids) if folds[sample_id] != fold]
        valid_indices = [index for index, sample_id in enumerate(sample_ids) if folds[sample_id] == fold]
        model, history = _fit_cached_heads(initial_state, representations, batch, schema, train_indices)
        with torch.no_grad():
            fitted = model(_slice_representations(representations, valid_indices))
        valid_batch = _slice_batch(batch, valid_indices)
        train_batch = _slice_batch(batch, train_indices)
        fold_report: dict[str, Any] = {"fold": fold, "train_samples": len(train_indices), "valid_samples": len(valid_indices), "final_loss": history[-1]}
        for factor in V03_AUXILIARY_FACTORS:
            oof_logits[factor][valid_indices] = fitted[factor].detach().cpu()
            with torch.no_grad():
                baseline = _majority_baseline(factor, train_batch, valid_batch, schema)
                spec = schema.auxiliary_factors[factor]
                if spec.allow_multiple:
                    baseline_logits[factor][valid_indices] = -20.0
                else:
                    # Recover the baseline class from the metric support by
                    # constructing it again; this avoids storing labels in the
                    # OOF artifact itself.
                    train_valid = train_batch.loss_masks[factor] > 0
                    counts = torch.bincount(train_batch.targets[factor][train_valid], minlength=len(spec.class_order)) if int(train_valid.sum()) else torch.ones(len(spec.class_order), dtype=torch.long)
                    majority = int(torch.argmax(counts).item())
                    baseline_logits[factor][valid_indices, majority] = 20.0
            fold_report[factor] = {
                "metrics": metric_for_factor(factor, fitted[factor], valid_batch, schema),
                "baseline": baseline,
            }
        fold_reports.append(fold_report)
    aggregate: dict[str, Any] = {}
    for factor in V03_AUXILIARY_FACTORS:
        aggregate[factor] = {
            "metrics": metric_for_factor(factor, oof_logits[factor], batch, schema),
            "baseline": metric_for_factor(factor, baseline_logits[factor], batch, schema),
            "diagnostic_class": "LOW_SAMPLE_DIAGNOSTIC" if int(batch.loss_masks[factor].sum().item()) < 10 else "STANDARD_DIAGNOSTIC",
            "label_count": int(batch.loss_masks[factor].sum().item()),
        }
    return {"fold_assignment": folds, "folds": fold_reports, "aggregate": aggregate}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tensor_fingerprint(tensors: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(tensors):
        digest.update(name.encode("utf-8"))
        digest.update(tensors[name].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def run_m7b(
    *,
    task_paths: Sequence[str | Path],
    assignment_paths: Sequence[str | Path],
    output_dir: str | Path,
    artifact_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    config_path: str | Path | None = None,
    schema_path: str | Path = DEFAULT_CANDIDATE_SCHEMA,
) -> dict[str, Any]:
    """Run the finite M7B OOF + full-data auxiliary diagnostic."""

    output = Path(output_dir).resolve()
    _require(not output.exists(), "M7B_OUTPUT_EXISTS", "M7B output directory must be new and immutable", path=str(output))
    output.mkdir(parents=True, exist_ok=False)
    schema = SemanticSchemaCandidate.load(schema_path)
    records, views, raw_assignments, filtered_assignments = combine_inputs(task_paths, assignment_paths, schema=schema)
    # Preserve answer-free tasks and both raw/filtered assignment snapshots as
    # evidence.  The raw snapshot is not fed to the loader after the one
    # explicitly masked pair is removed.
    all_tasks: list[dict[str, Any]] = []
    for path in task_paths:
        all_tasks.extend(read_jsonl(path))
    write_jsonl(output / "tasks-snapshot.jsonl", all_tasks)
    write_jsonl(output / "assignments-pulled-final.jsonl", raw_assignments)
    write_jsonl(output / "assignments-used-92.jsonl", filtered_assignments)

    runtime_kwargs: dict[str, Any] = {}
    if artifact_dir is not None:
        runtime_kwargs["artifact_dir"] = artifact_dir
    if cache_dir is not None:
        runtime_kwargs["cache_dir"] = cache_dir
    if config_path is not None:
        runtime_kwargs["config_path"] = config_path
    runtime = AuxiliaryTrainingRuntime.from_selected_artifact(**runtime_kwargs)
    # One frozen representation pass for all 64 records.  The original model
    # remains in eval mode and its parameters are never handed to an optimizer.
    batch_inputs = runtime.build_inputs(records, "cpu")
    base_state_before = _clone_state(runtime.model.base_model.state_dict())
    with torch.no_grad():
        captured_before = runtime.model(batch_inputs["input_ids"], batch_inputs["attention_mask"])
    representations = {owner: values.detach().cpu() for owner, values in captured_before["owner_representations"].items()}
    legacy_before = {head: values.detach().cpu().clone() for head, values in captured_before["legacy_logits"].items()}
    target_batch = build_auxiliary_targets(views, schema, device="cpu")
    torch.manual_seed(SEED)
    template = AuxiliaryHeadModel(schema, int(next(iter(representations.values())).shape[-1]))
    initial_state = _clone_state(template.state_dict())
    started = time.monotonic()
    oof = run_oof(representations, target_batch, [record.sample_id for record in records], schema, initial_state=initial_state)
    final_model, final_history = _fit_cached_heads(initial_state, representations, target_batch, schema, list(range(len(records))))
    final_logits: dict[str, Tensor]
    with torch.no_grad():
        final_logits = final_model(representations)
    final_losses = compute_auxiliary_losses(final_logits, target_batch, schema)
    final_total = total_auxiliary_loss(final_losses)
    _require(bool(torch.isfinite(final_total).item()), "M7B_FINAL_NONFINITE_LOSS", "final auxiliary loss is non-finite")
    checkpoint = output / "auxiliary-checkpoint.pt"
    torch.save({
        "component": "m7b_v03_auxiliary_diagnostic",
        "schema_version": schema.schema_version,
        "seed": SEED,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "sample_ids": [record.sample_id for record in records],
        "auxiliary_state_dict": final_model.state_dict(),
    }, checkpoint)
    reload_model = AuxiliaryHeadModel(schema, int(next(iter(representations.values())).shape[-1]))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    _require(isinstance(payload, Mapping), "M7B_CHECKPOINT_INVALID", "saved checkpoint is not a mapping")
    reload_model.load_state_dict(payload["auxiliary_state_dict"])
    reload_model.eval()
    with torch.no_grad():
        reload_logits = reload_model(representations)
    reload_equal = all(torch.equal(final_logits[factor], reload_logits[factor]) for factor in V03_AUXILIARY_FACTORS)
    _require(reload_equal, "M7B_CHECKPOINT_RELOAD_MISMATCH", "reloaded auxiliary outputs differ from saved outputs")
    # Re-run the frozen base after auxiliary fitting and compare both tensor
    # state and legacy logits exactly.  Only the standalone auxiliary model was
    # optimized, so this should remain byte-identical.
    with torch.no_grad():
        captured_after = runtime.model(batch_inputs["input_ids"], batch_inputs["attention_mask"])
    base_after = runtime.model.base_model.state_dict()
    base_unchanged = all(torch.equal(value, base_after[name].detach().cpu()) for name, value in base_state_before.items())
    legacy_unchanged = all(torch.equal(legacy_before[head], captured_after["legacy_logits"][head].detach().cpu()) for head in legacy_before)
    _require(base_unchanged and legacy_unchanged, "M7B_LEGACY_CHANGED", "frozen encoder or seven-head logits changed")
    elapsed = time.monotonic() - started
    masks = {factor: int(target_batch.loss_masks[factor].sum().item()) for factor in V03_AUXILIARY_FACTORS}
    summary = {
        "component": "M7B_V03_AUXILIARY_DIAGNOSTIC",
        "status": "DIAGNOSTIC_ONLY",
        "selected_candidate": False,
        "training_scope": "five auxiliary heads only",
        "sample_count": len(records),
        "raw_assignment_count": len(raw_assignments),
        "valid_assignment_count": len(filtered_assignments),
        "masked_pair": {"sample_id": INCOMPLETE_SAMPLE_ID, "head": INCOMPLETE_HEAD, "reason": "known incomplete text pair; database unchanged"},
        "factor_masks": masks,
        "seed": SEED,
        "epochs": EPOCHS,
        "optimizer": {"name": "AdamW", "learning_rate": LR, "weight_decay": WEIGHT_DECAY},
        "device": "cpu",
        "elapsed_seconds": elapsed,
        "representations_computed_once": True,
        "base_parameter_tensors": len(base_state_before),
        "base_parameters_unchanged": base_unchanged,
        "legacy_seven_head_logits_unchanged": legacy_unchanged,
        "oof": oof,
        "final_losses": {factor: float(value.detach().cpu()) for factor, value in final_losses.items()},
        "final_total_loss": float(final_total.detach().cpu()),
        "final_loss_start": float(final_history[0]),
        "final_loss_end": float(final_history[-1]),
        "checkpoint": str(checkpoint),
        "checkpoint_reload_equal": reload_equal,
        "auxiliary_outputs_finite": bool(all(torch.isfinite(values).all().item() for values in final_logits.values())),
        "auxiliary_parameter_fingerprint_initial": _tensor_fingerprint(initial_state),
        "auxiliary_parameter_fingerprint_final": _tensor_fingerprint(final_model.state_dict()),
    }
    (output / "aggregate-metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    report = _render_report(summary)
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    files = [path for path in output.iterdir() if path.is_file()]
    manifest = {
        "artifact": "m7b-v03-auxiliary-diagnostic-20260913",
        "status": "DIAGNOSTIC_ONLY",
        "files": {path.name: {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted(files) if path.name != "manifest.json"},
    }
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    (output / "manifest.json").write_bytes(manifest_bytes)
    return summary


def _render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# M7B v0.3 auxiliary diagnostic",
        "",
        "This is a finite diagnostic of five sparse auxiliary heads. The LERT encoder, seven legacy specialist blocks and seven legacy outputs were frozen; no selected candidate or production status is produced.",
        "",
        f"- Samples: {summary['sample_count']}; raw final assignments: {summary['raw_assignment_count']}; valid assignments: {summary['valid_assignment_count']}",
        f"- Masks: {json.dumps(summary['factor_masks'], ensure_ascii=False, sort_keys=True)}",
        f"- Seed/epochs/optimizer: {summary['seed']} / {summary['epochs']} / AdamW lr={summary['optimizer']['learning_rate']} wd={summary['optimizer']['weight_decay']}",
        f"- Device/time: {summary['device']} / {summary['elapsed_seconds']:.2f}s",
        f"- Frozen base tensors unchanged: {summary['base_parameters_unchanged']} ({summary['base_parameter_tensors']}); legacy logits unchanged: {summary['legacy_seven_head_logits_unchanged']}",
        f"- Checkpoint reload equal and finite: {summary['checkpoint_reload_equal']} / {summary['auxiliary_outputs_finite']}",
        "",
        "## OOF aggregate",
        "",
        "| Factor | N | OOF metrics | Baseline | Classification |",
        "|---|---:|---|---|---|",
    ]
    for factor in V03_AUXILIARY_FACTORS:
        item = summary["oof"]["aggregate"][factor]
        lines.append(f"| `{factor}` | {item['label_count']} | `{json.dumps(item['metrics'], ensure_ascii=False, sort_keys=True)}` | `{json.dumps(item['baseline'], ensure_ascii=False, sort_keys=True)}` | {item['diagnostic_class']} |")
    lines.extend([
        "",
        "## Final full-data fit",
        "",
        f"Per-factor losses: `{json.dumps(summary['final_losses'], ensure_ascii=False, sort_keys=True)}`; total `{summary['final_total_loss']:.8f}` (start `{summary['final_loss_start']:.8f}`, end `{summary['final_loss_end']:.8f}`).",
        "",
        "The known `这公司真，` / `discourse_focus` pair was masked in memory only; the labeling database was not changed. arousal_level and market_scope remain LOW_SAMPLE_DIAGNOSTIC and are not treated as quality conclusions. Human Check, Test, Gold and OOD data were not read.",
        "",
    ])
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m6b-tasks", type=Path, required=True)
    parser.add_argument("--m6b-assignments", type=Path, required=True)
    parser.add_argument("--m7a-tasks", type=Path, required=True)
    parser.add_argument("--m7a-assignments", type=Path, required=True)
    parser.add_argument("--supplement-tasks", type=Path, required=True)
    parser.add_argument("--supplement-assignments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--schema", type=Path, default=DEFAULT_CANDIDATE_SCHEMA)
    args = parser.parse_args(argv)
    result = run_m7b(
        task_paths=(args.m6b_tasks, args.m7a_tasks, args.supplement_tasks),
        assignment_paths=(args.m6b_assignments, args.m7a_assignments, args.supplement_assignments),
        output_dir=args.output_dir,
        artifact_dir=args.artifact_dir,
        cache_dir=args.cache_dir,
        config_path=args.config,
        schema_path=args.schema,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "combine_inputs",
    "deterministic_folds",
    "metric_for_factor",
    "run_m7b",
]
