"""M10C unified head-specific auxiliary diagnostic freeze.

M10B established that the two multi-label factors need independent policies:
``market_scope`` remains the ordinary M9B/M10A-A BCE head, while
``discourse_focus`` uses the accepted D policy (full-data ``pos_weight`` BCE,
0.5 threshold, UNKNOWN exclusivity and top-1 fallback).  This module freezes
that decision into one reloadable auxiliary checkpoint.  The selected LERT
encoder and all legacy seven heads are read-only; no production candidate is
created.
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

from .auxiliary_heads import AuxiliaryHeadModel, build_auxiliary_targets
from .encoder_m10a_multilabel_ablation import EPOCHS, LR, THRESHOLD, WEIGHT_DECAY, apply_multi_label_policy, fit_factor
from .encoder_m6c_auxiliary import AuxiliaryTrainingRuntime, DEFAULT_CANDIDATE_SCHEMA, load_auxiliary_task_views
from .encoder_m8c_cumulative import _applicable_factors, _fresh_input_rows, _fresh_record, _load_checkpoint, _multi_entropy, _single_prediction, summarize_fresh_rows
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
from .schema import SemanticSchemaCandidate, V03_AUXILIARY_FACTORS


SEED = 20260913
MARKET = "market_scope"
DISCOURSE = "discourse_focus"
SINGLE_FACTORS = ("expectation_phase", "arousal_level", "desired_price_direction")
M9B_ARTIFACT = Path(__file__).resolve().parents[3] / "model-artifacts" / "m9b-v03-cumulative174-effect-gate-20260914"
DEFAULT_TASKS = M9B_ARTIFACT / "tasks-cumulative-143.jsonl"
DEFAULT_ASSIGNMENTS = M9B_ARTIFACT / "assignments-cumulative-174.jsonl"
DEFAULT_PARENT_CHECKPOINT = M9B_ARTIFACT / "auxiliary-checkpoint.pt"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[3] / "model-artifacts" / "m10c-v03-head-specific-diagnostic-freeze-20260914"


def _require(ok: bool, code: str, message: str, **details: Any) -> None:
    if not ok:
        raise ContractError(code, message, **details)


def _clone_state_dict(model: AuxiliaryHeadModel) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def compose_head_specific(parent: AuxiliaryHeadModel, discourse: AuxiliaryHeadModel) -> AuxiliaryHeadModel:
    """Copy M9B and replace only the discourse linear layer."""

    model = AuxiliaryHeadModel(parent.schema, parent.auxiliary_heads[MARKET].in_features)
    model.load_state_dict(parent.state_dict())
    model.auxiliary_heads[DISCOURSE].load_state_dict(discourse.auxiliary_heads[DISCOURSE].state_dict())
    return model.eval()


def untouched_head_differences(before: Mapping[str, Tensor], after: Mapping[str, Tensor]) -> dict[str, list[str]]:
    """Return state entries changed for each auxiliary factor."""

    result: dict[str, list[str]] = {}
    for factor in V03_AUXILIARY_FACTORS:
        prefix = f"auxiliary_heads.{factor}."
        result[factor] = [name for name in before if name.startswith(prefix) and not torch.equal(before[name], after[name])]
    return result


def _market_prediction(logits: Tensor, class_order: Sequence[str]) -> dict[str, Any]:
    probabilities, binary = apply_multi_label_policy(logits.unsqueeze(0), class_order, fallback=False)
    values = probabilities[0]
    selected = [label for i, label in enumerate(class_order) if binary[0, i]]
    return {
        "probabilities": {label: float(values[i]) for i, label in enumerate(class_order)},
        "prediction": selected,
        "confidence": float(np.max(values)),
        "normalized_entropy": _multi_entropy(values.tolist()),
        "threshold": THRESHOLD,
        "fallback": False,
    }


def _discourse_prediction(logits: Tensor, class_order: Sequence[str]) -> dict[str, Any]:
    probabilities, binary = apply_multi_label_policy(logits.unsqueeze(0), class_order, fallback=True)
    values = probabilities[0]
    selected = [label for i, label in enumerate(class_order) if binary[0, i]]
    return {
        "probabilities": {label: float(values[i]) for i, label in enumerate(class_order)},
        "prediction": selected,
        "confidence": float(np.max(values)),
        "normalized_entropy": _multi_entropy(values.tolist()),
        "threshold": THRESHOLD,
        "fallback": True,
    }


def fresh_shadow_predictions(
    runtime: AuxiliaryTrainingRuntime,
    model: AuxiliaryHeadModel,
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Run one CPU shadow pass and apply the frozen per-head policies."""

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
            logits = model(captured["owner_representations"])
        for index, (record, source) in enumerate(zip(current_records, current_rows)):
            applicable = list(_applicable_factors(record.model_text))
            row: dict[str, Any] = {
                "sample_id": record.sample_id,
                "model_text": record.model_text,
                "stock_code": record.stock_code,
                "stock_name": record.stock_name,
                "text_length": source.get("text_length", len(record.model_text)),
                "text_source": source.get("text_source"),
                "exact_content_group": source.get("exact_content_group"),
                "applicable_factors": applicable,
                "auxiliary": {},
            }
            for factor in applicable:
                if factor not in logits:
                    continue
                spec = runtime.schema.auxiliary_factors[factor]
                vector = logits[factor][index]
                _require(bool(torch.isfinite(vector).all().item()), "M10C_FRESH_NONFINITE_OUTPUT", "fresh auxiliary logits are non-finite", factor=factor, sample_id=record.sample_id)
                if factor == MARKET:
                    row["auxiliary"][factor] = _market_prediction(vector, spec.class_order)
                elif factor == DISCOURSE:
                    row["auxiliary"][factor] = _discourse_prediction(vector, spec.class_order)
                else:
                    row["auxiliary"][factor] = _single_prediction(vector, spec.class_order)
            output.append(row)
    _require(len(output) == len(rows), "M10C_FRESH_OUTPUT_COUNT_INVALID", "fresh shadow output count differs from input", inputs=len(rows), outputs=len(output))
    return output


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def _content_address(files: Mapping[str, Mapping[str, Any]]) -> str:
    encoded = json.dumps({"files": files}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_m10c(
    *,
    tasks_path: str | Path = DEFAULT_TASKS,
    assignments_path: str | Path = DEFAULT_ASSIGNMENTS,
    parent_checkpoint_path: str | Path = DEFAULT_PARENT_CHECKPOINT,
    output_dir: str | Path = DEFAULT_OUTPUT,
    selected_artifact: str | Path = DEFAULT_SELECTED_ARTIFACT,
    cache_dir: str | Path = DEFAULT_CACHE,
    config_path: str | Path = DEFAULT_CONFIG,
    schema_path: str | Path = DEFAULT_CANDIDATE_SCHEMA,
    fresh_pool_path: str | Path = DEFAULT_FRESH_POOL,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    _require(not output.exists(), "M10C_OUTPUT_EXISTS", "M10C output directory must be new", path=str(output))
    schema = SemanticSchemaCandidate.load(schema_path)
    tasks = read_jsonl(tasks_path)
    assignments = read_jsonl(assignments_path)
    _require(len(tasks) == EXPECTED_SAMPLES, "M10C_TASK_COUNT_INVALID", "M9B tasks must contain 143 samples", observed=len(tasks))
    _require(len(assignments) == EXPECTED_ASSIGNMENTS, "M10C_ASSIGNMENT_COUNT_INVALID", "M9B assignments must contain 174 rows", observed=len(assignments))
    records, views = load_auxiliary_task_views(tasks, assignments, schema=schema)
    batch = build_auxiliary_targets(views, schema, device="cpu")
    masks = {factor: int(batch.loss_masks[factor].sum().item()) for factor in V03_AUXILIARY_FACTORS}
    expected = {factor: EXPECTED_MASKS[factor] for factor in V03_AUXILIARY_FACTORS}
    _require(masks == expected, "M10C_MASK_COUNT_INVALID", "M9B masks differ from frozen counts", observed=masks, expected=expected)

    runtime = AuxiliaryTrainingRuntime.from_selected_artifact(selected_artifact, config_path=config_path, cache_dir=cache_dir)
    inputs = runtime.build_inputs(records, "cpu")
    with torch.no_grad():
        captured = runtime.model(inputs["input_ids"], inputs["attention_mask"])
    representations = {owner: values.detach().cpu() for owner, values in captured["owner_representations"].items()}
    hidden_size = int(next(iter(representations.values())).shape[-1])

    parent = _load_checkpoint(parent_checkpoint_path, schema, hidden_size)
    parent_state = _clone_state_dict(parent)
    torch.manual_seed(SEED)
    template = AuxiliaryHeadModel(schema, hidden_size)
    initial_state = _clone_state(template.state_dict())
    all_indices = list(range(len(records)))
    started = time.monotonic()
    discourse_model, discourse_history, discourse_weight_report = fit_factor(
        initial_state,
        representations,
        batch,
        schema,
        DISCOURSE,
        all_indices,
        use_pos_weight=True,
        epochs=EPOCHS,
    )
    head_specific = compose_head_specific(parent, discourse_model)
    elapsed = time.monotonic() - started
    differences = untouched_head_differences(parent_state, _clone_state_dict(head_specific))
    _require(not any(differences[factor] for factor in ("expectation_phase", "arousal_level", "desired_price_direction", MARKET)), "M10C_UNTOUCHED_HEAD_CHANGED", "non-discourse auxiliary head changed", differences=differences)
    _require(bool(differences[DISCOURSE]), "M10C_DISCOURSE_NOT_CHANGED", "discourse head did not change after D training")

    output.mkdir(parents=True, exist_ok=False)
    parent_hash = sha256_file(parent_checkpoint_path)
    checkpoint = output / "head-specific-auxiliary-checkpoint.pt"
    checkpoint_payload = {
        "component": "m10c_v03_head_specific_diagnostic_freeze",
        "schema_version": schema.schema_version,
        "seed": SEED,
        "epochs": EPOCHS,
        "optimizer": {"name": "AdamW", "learning_rate": LR, "weight_decay": WEIGHT_DECAY},
        "parent_artifact": str(M9B_ARTIFACT),
        "parent_checkpoint": str(Path(parent_checkpoint_path).resolve()),
        "parent_checkpoint_sha256": parent_hash,
        "assignment_count": len(assignments),
        "sample_count": len(records),
        "factor_masks": masks,
        "head_policies": {
            "expectation_phase": {"loss": "parent M9B state unchanged", "inference": "softmax argmax"},
            "arousal_level": {"loss": "parent M9B state unchanged", "inference": "softmax argmax"},
            "desired_price_direction": {"loss": "parent M9B state unchanged", "inference": "softmax argmax"},
            MARKET: {"loss": "ordinary BCE", "inference": "sigmoid threshold 0.5; UNKNOWN exclusive; empty allowed", "formulation_status": "FORMULATION_UNRESOLVED"},
            DISCOURSE: {"loss": "full-data pos_weight BCE", "inference": "sigmoid threshold 0.5; UNKNOWN exclusive; empty top-1 fallback", "training_assignments": masks[DISCOURSE]},
        },
        "untouched_head_differences": differences,
        "auxiliary_state_dict": head_specific.state_dict(),
    }
    torch.save(checkpoint_payload, checkpoint)
    reloaded = _load_checkpoint(checkpoint, schema, hidden_size)
    with torch.no_grad():
        before_reload = head_specific(representations)
        after_reload = reloaded(representations)
    reload_equal = all(torch.equal(before_reload[factor], after_reload[factor]) for factor in V03_AUXILIARY_FACTORS)
    _require(reload_equal, "M10C_CHECKPOINT_RELOAD_MISMATCH", "reloaded checkpoint logits differ")

    fresh_rows = _fresh_input_rows(fresh_pool_path)
    fresh_predictions = fresh_shadow_predictions(runtime, reloaded, fresh_rows)
    fresh_summary = summarize_fresh_rows(fresh_predictions)
    fresh_path = output / "fresh-2000-shadow-predictions.jsonl"
    _write_jsonl(fresh_path, fresh_predictions)

    policy = {
        "component": "M10C_V03_HEAD_SPECIFIC_DIAGNOSTIC_FREEZE",
        "status": "DIAGNOSTIC_ONLY",
        "selected_candidate": False,
        "parent": {"artifact": str(M9B_ARTIFACT), "checkpoint": str(Path(parent_checkpoint_path).resolve()), "sha256": parent_hash},
        "seed": SEED,
        "epochs": EPOCHS,
        "optimizer": {"name": "AdamW", "learning_rate": LR, "weight_decay": WEIGHT_DECAY},
        "factor_masks": masks,
        "heads": checkpoint_payload["head_policies"],
        "discourse_training_loss_start": discourse_history[0] if discourse_history else None,
        "discourse_training_loss_end": discourse_history[-1] if discourse_history else None,
        "untouched_heads_tensor_identical": True,
        "forbidden_data_access": {"test": False, "gold": False, "ood": False, "anchor": False, "llm": False},
    }
    summary = {
        "component": "M10C_V03_HEAD_SPECIFIC_DIAGNOSTIC_FREEZE",
        "status": "DIAGNOSTIC_ONLY",
        "selected_candidate": False,
        "assignment_count": len(assignments),
        "sample_count": len(records),
        "factor_masks": masks,
        "seed": SEED,
        "epochs": EPOCHS,
        "optimizer": {"name": "AdamW", "learning_rate": LR, "weight_decay": WEIGHT_DECAY},
        "device": "cpu",
        "elapsed_seconds": elapsed,
        "parent_checkpoint_sha256": parent_hash,
        "untouched_heads_tensor_identical": True,
        "discourse_head_changed": True,
        "checkpoint_reload_equal": reload_equal,
        "fresh_pool_rows": len(fresh_rows),
        "fresh_summary": fresh_summary,
        "forbidden_data_access": {"test": False, "gold": False, "ood": False, "anchor": False, "llm": False},
    }
    _write_json(output / "policy.json", policy)
    _write_json(output / "summary.json", summary)
    report = "\n".join(
        [
            "# M10C head-specific diagnostic freeze",
            "",
            "This closes the M10 mechanism phase. The M9B auxiliary checkpoint is the parent; only `discourse_focus` was replaced with the independently accepted D policy. The encoder, legacy seven heads and the other four auxiliary heads remain frozen and tensor-identical.",
            "",
            f"- Parent checkpoint SHA-256: `{parent_hash}`",
            f"- Data: {len(assignments)} assignments / {len(records)} samples; masks `{json.dumps(masks, ensure_ascii=False, sort_keys=True)}`",
            f"- Device/time: CPU / {elapsed:.2f}s; seed={SEED}; epochs={EPOCHS}; AdamW lr={LR}; wd={WEIGHT_DECAY}",
            "- `expectation_phase`, `arousal_level`, and `desired_price_direction`: M9B unchanged.",
            "- `market_scope`: ordinary BCE + sigmoid 0.5, empty output allowed; `FORMULATION_UNRESOLVED` remains recorded.",
            "- `discourse_focus`: full-data pos-weight BCE + sigmoid 0.5, UNKNOWN exclusive, top-1 fallback on empty output.",
            "",
            "## Fresh-2000 shadow",
            "",
            f"All {len(fresh_rows)} fresh rows were processed once on CPU. Market and discourse distributions are recorded in `summary.json`; discourse fallback means its empty rate is expected to be zero, while market retains the ordinary-BCE empty rate.",
            "",
            "This is diagnostic evidence only (`selected_candidate=false`, no production approval). M10 mechanism experiments are closed; no further variants are planned.",
            "",
        ]
    )
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    files = [path for path in output.iterdir() if path.is_file() and path.name != "manifest.json"]
    file_meta = {path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(files)}
    manifest = {
        "artifact": "m10c-v03-head-specific-diagnostic-freeze-20260914",
        "status": "DIAGNOSTIC_ONLY",
        "selected_candidate": False,
        "content_address": _content_address(file_meta),
        "files": file_meta,
    }
    _write_json(output / "manifest.json", manifest)
    return {**summary, "policy": policy, "checkpoint": str(checkpoint), "fresh_predictions": str(fresh_path), "content_address": manifest["content_address"]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--parent-checkpoint", type=Path, default=DEFAULT_PARENT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--selected-artifact", type=Path, default=DEFAULT_SELECTED_ARTIFACT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--schema", type=Path, default=DEFAULT_CANDIDATE_SCHEMA)
    parser.add_argument("--fresh-pool", type=Path, default=DEFAULT_FRESH_POOL)
    args = parser.parse_args(argv)
    result = run_m10c(
        tasks_path=args.tasks,
        assignments_path=args.assignments,
        parent_checkpoint_path=args.parent_checkpoint,
        output_dir=args.output_dir,
        selected_artifact=args.selected_artifact,
        cache_dir=args.cache_dir,
        config_path=args.config,
        schema_path=args.schema,
        fresh_pool_path=args.fresh_pool,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
