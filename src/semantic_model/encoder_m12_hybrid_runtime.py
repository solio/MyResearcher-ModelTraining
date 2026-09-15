"""Inference-only hybrid runtime for the two winning M12 heads.

The frozen selected seven-head runtime remains the source of truth for
``target_mode``, ``emotion_primary``, ``action_tendency``,
``reasoning_tags``, and ``context_dependency``.  M12 HumanTrain100
checkpoints replace only ``stance`` and ``emotion_target``.  The output keeps
the selected runtime's row-by-row ``predictions`` contract so it can be used
as a feedback-loop prediction snapshot without any annotation or training
side effects.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import encoder_m12_human_first_cv as m12
from . import encoder_m2_1_softlabel as m21
from . import encoder_m2_selected_runtime as selected
from .errors import ContractError


STAGE_ID = "M12-HUMAN-FIRST-LERT-HYBRID-INFERENCE"
RUN_SCHEMA_VERSION = "myresearcher.encoder-m12-hybrid-runtime.v1"
REPLACED_HEADS = ("stance", "emotion_target")
PASSTHROUGH_HEADS = (
    "target_mode",
    "emotion_primary",
    "action_tendency",
    "reasoning_tags",
    "context_dependency",
)
DEFAULT_M12_ARTIFACT = Path(__file__).resolve().parents[3] / "model-artifacts" / "m12-human-first-lert-oof-20260915"


def _require(ok: bool, code: str, message: str, **details: Any) -> None:
    if not ok:
        raise ContractError(code, message, **details)


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractError("M12_HYBRID_JSON_MISSING", "required M12 JSON is missing", path=str(path)) from exc
    except json.JSONDecodeError as exc:
        raise ContractError("M12_HYBRID_JSON_INVALID", "required M12 JSON is invalid", path=str(path)) from exc
    _require(isinstance(value, Mapping), "M12_HYBRID_JSON_INVALID", "required M12 JSON must be an object", path=str(path))
    return value


def _load_checkpoint(torch: Any, artifact: Path, head: str) -> Mapping[str, Any]:
    path = artifact / head / "human-first-checkpoint.pt"
    _require(path.is_file(), "M12_HYBRID_CHECKPOINT_MISSING", "M12 head checkpoint is missing", head=head, path=str(path))
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    _require(isinstance(checkpoint, Mapping), "M12_HYBRID_CHECKPOINT_INVALID", "M12 checkpoint must be a mapping", head=head)
    _require(checkpoint.get("stage_id") == m12.STAGE_ID, "M12_HYBRID_CHECKPOINT_INVALID", "checkpoint stage differs from M12", head=head)
    _require(checkpoint.get("head") == head, "M12_HYBRID_CHECKPOINT_INVALID", "checkpoint head does not match requested head", head=head)
    _require(isinstance(checkpoint.get("specialist_blocks_state_dict"), Mapping), "M12_HYBRID_CHECKPOINT_INVALID", "M12 checkpoint lacks the two-block state", head=head)
    _require(isinstance(checkpoint.get("head_state_dict"), Mapping), "M12_HYBRID_CHECKPOINT_INVALID", "M12 checkpoint lacks the output-head state", head=head)
    return checkpoint


def _validate_m12_artifact(artifact: Path, schema: Any) -> Mapping[str, Any]:
    config = _read_json(artifact / "training-config.json")
    _require(config.get("stage_id") == m12.STAGE_ID, "M12_HYBRID_ARTIFACT_INVALID", "artifact is not the requested M12 run")
    _require(config.get("model_id") == selected.MODEL_ID and config.get("revision") == selected.REVISION, "M12_HYBRID_MODEL_MISMATCH", "M12 model identity differs from selected runtime")
    class_order = config.get("class_order")
    _require(isinstance(class_order, Mapping), "M12_HYBRID_ARTIFACT_INVALID", "M12 class order is missing")
    for head in REPLACED_HEADS:
        _require(list(class_order.get(head, [])) == list(schema.class_order[head]), "M12_HYBRID_CLASS_ORDER_MISMATCH", "M12 class order differs from selected runtime", head=head)
    return config


def _candidate_probabilities(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    records: Sequence[selected.InferenceRecord],
    config: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Run the M12 model and return probabilities only for the two replacements."""

    values: dict[str, list[np.ndarray]] = {head: [] for head in REPLACED_HEADS}
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(records), 16):
            current = records[offset : offset + 16]
            batch = selected.build_inference_batch(torch, tokenizer, current, config, torch.device("cpu"))
            outputs = model(batch["input_ids"], batch["attention_mask"])
            for head in REPLACED_HEADS:
                logits = outputs[head]
                _require(bool(torch.isfinite(logits).all().item()), "M12_HYBRID_NONFINITE_LOGITS", "M12 logits are non-finite", head=head)
                values[head].append(torch.softmax(logits, dim=-1).detach().cpu().numpy())
    return {head: np.concatenate(chunks, axis=0) for head, chunks in values.items()}


def merge_replaced_heads(
    legacy_rows: Sequence[Mapping[str, Any]],
    candidate_probabilities: Mapping[str, np.ndarray],
    schema: Any,
) -> list[dict[str, Any]]:
    """Replace exactly two scalar heads while preserving the selected payload."""

    _require(set(candidate_probabilities) == set(REPLACED_HEADS), "M12_HYBRID_HEAD_SET_INVALID", "candidate probabilities must contain exactly the two replacement heads")
    result: list[dict[str, Any]] = []
    for row_index, legacy_row in enumerate(legacy_rows):
        _require(isinstance(legacy_row, Mapping), "M12_HYBRID_LEGACY_ROW_INVALID", "selected runtime row must be an object", row=row_index)
        legacy_predictions = legacy_row.get("predictions")
        _require(isinstance(legacy_predictions, Mapping), "M12_HYBRID_LEGACY_ROW_INVALID", "selected runtime row lacks predictions", row=row_index)
        _require(set(legacy_predictions) == set(selected.HEADS), "M12_HYBRID_LEGACY_HEAD_SET_INVALID", "selected runtime row is not a complete seven-head payload", row=row_index)
        predictions = dict(legacy_predictions)
        for head in REPLACED_HEADS:
            order = list(schema.class_order[head])
            values = np.asarray(candidate_probabilities[head])
            _require(values.ndim == 2 and values.shape[0] == len(legacy_rows) and values.shape[1] == len(order), "M12_HYBRID_PROBABILITY_SHAPE_INVALID", "M12 probability shape does not match the schema", head=head, row=row_index)
            vector = values[row_index].astype(float, copy=False)
            _require(bool(np.isfinite(vector).all()), "M12_HYBRID_NONFINITE_PROBABILITY", "M12 probability is non-finite", head=head, row=row_index)
            _require(np.isclose(float(vector.sum()), 1.0, rtol=1e-6, atol=1e-6), "M12_HYBRID_PROBABILITY_NOT_NORMALIZED", "M12 scalar probabilities do not sum to one", head=head, row=row_index, total=float(vector.sum()))
            predictions[head] = {
                "probabilities": {label: float(vector[index]) for index, label in enumerate(order)},
                "prediction": order[int(np.argmax(vector))],
            }
        merged = dict(legacy_row)
        merged["predictions"] = predictions
        result.append(merged)
    return result


class M12HybridInferenceRuntime:
    """Selected seven-head runtime with the two winning Human-first heads."""

    def __init__(self, *, selected_runtime: Any, m12_model: Any, tokenizer: Any, torch: Any, config: Mapping[str, Any], artifact: Path, artifact_config: Mapping[str, Any]) -> None:
        self.selected_runtime = selected_runtime
        self.m12_model = m12_model
        self.tokenizer = tokenizer
        self.torch = torch
        self.config = config
        self.artifact = artifact
        self.artifact_config = artifact_config
        self.schema = selected_runtime.schema

    @classmethod
    def from_artifacts(
        cls,
        *,
        m12_artifact: str | Path = DEFAULT_M12_ARTIFACT,
        selected_artifact: str | Path = selected.DEFAULT_SELECTED_ARTIFACT,
        config_path: str | Path = selected.CONFIG_PATH,
        cache_dir: str | Path = selected.DEFAULT_CACHE,
    ) -> "M12HybridInferenceRuntime":
        selected_runtime = selected.UnifiedCPUInferenceRuntime.from_artifact(
            selected_artifact,
            config_path=config_path,
            cache_dir=cache_dir,
        )
        artifact = Path(m12_artifact).resolve()
        artifact_config = _validate_m12_artifact(artifact, selected_runtime.schema)
        _np, torch, AutoModel, AutoTokenizer = m21._load_runtime()
        snapshot, _identity = m21._snapshot(Path(cache_dir).resolve())
        model = m12.make_model(
            torch,
            AutoModel,
            snapshot,
            selected_runtime.schema,
            float(selected_runtime.config["head_dropout"]),
        )
        for head in REPLACED_HEADS:
            checkpoint = _load_checkpoint(torch, artifact, head)
            model.specialist_blocks[head].load_state_dict(checkpoint["specialist_blocks_state_dict"])
            model.heads[head].load_state_dict(checkpoint["head_state_dict"])
        model.set_active_head(None)
        model = model.to(torch.device("cpu"))
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True)
        return cls(
            selected_runtime=selected_runtime,
            m12_model=model,
            tokenizer=tokenizer,
            torch=torch,
            config=selected_runtime.config,
            artifact=artifact,
            artifact_config=artifact_config,
        )

    def predict_records(self, records: Sequence[selected.InferenceRecord]) -> list[dict[str, Any]]:
        if not records:
            return []
        legacy_rows = self.selected_runtime.predict_records(records)
        candidate = _candidate_probabilities(
            torch=self.torch,
            model=self.m12_model,
            tokenizer=self.tokenizer,
            records=records,
            config=self.config,
        )
        rows = merge_replaced_heads(legacy_rows, candidate, self.schema)
        _require([row.get("sample_id") for row in rows] == [record.sample_id for record in records], "M12_HYBRID_SAMPLE_ORDER_INVALID", "hybrid output order differs from input")
        return rows


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def run_hybrid(
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    m12_artifact: str | Path = DEFAULT_M12_ARTIFACT,
    selected_artifact: str | Path = selected.DEFAULT_SELECTED_ARTIFACT,
    config_path: str | Path = selected.CONFIG_PATH,
    cache_dir: str | Path = selected.DEFAULT_CACHE,
) -> dict[str, Any]:
    records = selected.load_unlabelled_jsonl(input_jsonl)
    runtime = M12HybridInferenceRuntime.from_artifacts(
        m12_artifact=m12_artifact,
        selected_artifact=selected_artifact,
        config_path=config_path,
        cache_dir=cache_dir,
    )
    rows = runtime.predict_records(records)
    output = Path(output_jsonl).resolve()
    write_jsonl(output, rows)
    return {
        "status": "PREDICTIONS_WRITTEN",
        "stage_id": STAGE_ID,
        "input_records": len(records),
        "output_records": len(rows),
        "output_jsonl": str(output),
        "replaced_heads": list(REPLACED_HEADS),
        "passthrough_heads": list(PASSTHROUGH_HEADS),
        "selected_artifact": str(Path(selected_artifact).resolve()),
        "m12_artifact": str(runtime.artifact),
        "training_invoked": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True, help="model-ready JSONL with sample_id, stock_code, stock_name, model_text")
    parser.add_argument("--output-jsonl", type=Path, required=True, help="seven-head prediction snapshot JSONL")
    parser.add_argument("--m12-artifact", type=Path, default=DEFAULT_M12_ARTIFACT)
    parser.add_argument("--selected-artifact", type=Path, default=selected.DEFAULT_SELECTED_ARTIFACT)
    parser.add_argument("--config", type=Path, default=selected.CONFIG_PATH)
    parser.add_argument("--cache-dir", type=Path, default=selected.DEFAULT_CACHE)
    args = parser.parse_args(argv)
    try:
        result = run_hybrid(
            args.input_jsonl,
            args.output_jsonl,
            m12_artifact=args.m12_artifact,
            selected_artifact=args.selected_artifact,
            config_path=args.config,
            cache_dir=args.cache_dir,
        )
    except (ContractError, OSError, RuntimeError) as exc:
        print(json.dumps({"status": "M12_HYBRID_BLOCKED", "error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the CLI smoke
    raise SystemExit(main())
