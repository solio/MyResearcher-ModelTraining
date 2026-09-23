"""Inference runtime promoting M16b stance and emotion-target heads.

The existing M12 hybrid remains the source for five heads.  M16b replaces
``stance`` and ``emotion_target``.  Both improve Recent Check19 and
HumanCheck20 materially, keep a positive net correct count across all three
independent evaluations, and improve mean macro-F1 across those evaluations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import encoder_m1 as m1
from . import encoder_m2_1_softlabel as m21
from . import encoder_m2_selected_runtime as selected
from . import encoder_m12_hybrid_runtime as parent
from . import encoder_m15_shared_multitask_lert as m15
from . import encoder_m16_weak_pretrained_multitask_lert as m16
from .output_conditioning import add_conditioned_outputs
from .errors import ContractError


STAGE_ID = "M16B-STANCE-EMOTION-TARGET-HYBRID-INFERENCE"
RUN_SCHEMA_VERSION = "myresearcher.encoder-m16b-two-head-hybrid-runtime.v1"
REPLACED_HEADS = ("stance", "emotion_target")
PASSTHROUGH_HEADS = tuple(head for head in selected.HEADS if head not in REPLACED_HEADS)
DEFAULT_M16_ARTIFACT = m16.DEFAULT_OUTPUT


def _require(ok: bool, code: str, message: str, **details: Any) -> None:
    if not ok:
        raise ContractError(code, message, **details)


def _load_checkpoint(torch: Any, artifact: Path, schema: Any) -> Mapping[str, Any]:
    path = artifact / "weak-pretrained-human-corrected-checkpoint.pt"
    _require(path.is_file(), "M16_HYBRID_CHECKPOINT_MISSING", "M16 final checkpoint is missing", path=str(path))
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    _require(isinstance(checkpoint, Mapping) and checkpoint.get("stage_id") == m16.STAGE_ID, "M16_HYBRID_CHECKPOINT_INVALID", "checkpoint is not the frozen M16 run")
    _require(isinstance(checkpoint.get("model_state_dict"), Mapping), "M16_HYBRID_CHECKPOINT_INVALID", "M16 checkpoint lacks model state")
    class_order = checkpoint.get("class_order")
    _require(isinstance(class_order, Mapping), "M16_HYBRID_CHECKPOINT_INVALID", "M16 checkpoint lacks class order")
    for head in selected.HEADS:
        _require(list(class_order.get(head, [])) == list(schema.class_order[head]), "M16_HYBRID_CLASS_ORDER_MISMATCH", "M16 class order differs from parent runtime", head=head)
    return checkpoint


def merge_heads(parent_rows: Sequence[Mapping[str, Any]], probabilities: Mapping[str, np.ndarray], schema: Any) -> list[dict[str, Any]]:
    _require(set(probabilities) == set(REPLACED_HEADS), "M16_HYBRID_HEAD_SET_INVALID", "M16 probabilities must contain exactly the promoted heads")
    prepared: dict[str, np.ndarray] = {}
    for head in REPLACED_HEADS:
        order = list(schema.class_order[head])
        values = np.asarray(probabilities[head], dtype=float)
        _require(values.ndim == 2 and values.shape == (len(parent_rows), len(order)), "M16_HYBRID_PROBABILITY_SHAPE_INVALID", "M16 probability shape differs from schema", head=head, observed=list(values.shape))
        _require(bool(np.isfinite(values).all()), "M16_HYBRID_NONFINITE_PROBABILITY", "M16 probabilities are non-finite", head=head)
        _require(bool(np.allclose(values.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6)), "M16_HYBRID_PROBABILITY_NOT_NORMALIZED", "M16 probabilities do not sum to one", head=head)
        prepared[head] = values
    result: list[dict[str, Any]] = []
    for row_index, parent_row in enumerate(parent_rows):
        predictions = parent_row.get("predictions")
        _require(isinstance(predictions, Mapping) and set(predictions) == set(selected.HEADS), "M16_HYBRID_PARENT_ROW_INVALID", "parent row is not a strict seven-head prediction", row=row_index)
        merged_predictions = dict(predictions)
        for head in REPLACED_HEADS:
            order = list(schema.class_order[head])
            vector = prepared[head][row_index]
            merged_predictions[head] = {
                "probabilities": {label: float(vector[index]) for index, label in enumerate(order)},
                "prediction": order[int(np.argmax(vector))],
            }
        merged = dict(parent_row)
        merged["predictions"] = merged_predictions
        result.append(add_conditioned_outputs(merged))
    return result


class M16HybridInferenceRuntime:
    """M12 parent predictions with the M16 stance head."""

    def __init__(self, *, parent_runtime: Any, model: Any, tokenizer: Any, torch: Any, artifact: Path) -> None:
        self.parent_runtime = parent_runtime
        self.model = model
        self.tokenizer = tokenizer
        self.torch = torch
        self.artifact = artifact
        self.schema = parent_runtime.schema

    @classmethod
    def from_artifacts(
        cls,
        *,
        m16_artifact: str | Path = DEFAULT_M16_ARTIFACT,
        m12_artifact: str | Path = parent.DEFAULT_M12_ARTIFACT,
        selected_artifact: str | Path = selected.DEFAULT_SELECTED_ARTIFACT,
        config_path: str | Path = selected.CONFIG_PATH,
        cache_dir: str | Path = selected.DEFAULT_CACHE,
    ) -> "M16HybridInferenceRuntime":
        parent_runtime = parent.M12HybridInferenceRuntime.from_artifacts(
            m12_artifact=m12_artifact,
            selected_artifact=selected_artifact,
            config_path=config_path,
            cache_dir=cache_dir,
        )
        _np, torch, AutoModel, AutoTokenizer = m1._load_runtime_dependencies()
        snapshot, _identity = m21._snapshot(Path(cache_dir).resolve())
        artifact = Path(m16_artifact).resolve()
        checkpoint = _load_checkpoint(torch, artifact, parent_runtime.schema)
        model = m15.make_model(torch, AutoModel, snapshot, parent_runtime.schema)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True)
        return cls(parent_runtime=parent_runtime, model=model, tokenizer=tokenizer, torch=torch, artifact=artifact)

    def predict_records(self, records: Sequence[selected.InferenceRecord]) -> list[dict[str, Any]]:
        if not records:
            return []
        parent_rows = self.parent_runtime.predict_records(records)
        arrays = m15._predict(self.torch, self.model, self.tokenizer, records, self.torch.device("cpu"))
        rows = merge_heads(parent_rows, {head: arrays[head] for head in REPLACED_HEADS}, self.schema)
        _require([row.get("sample_id") for row in rows] == [record.sample_id for record in records], "M16_HYBRID_SAMPLE_ORDER_INVALID", "M16 hybrid output order differs from input")
        return rows


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def run_hybrid(
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    m16_artifact: str | Path = DEFAULT_M16_ARTIFACT,
    m12_artifact: str | Path = parent.DEFAULT_M12_ARTIFACT,
    selected_artifact: str | Path = selected.DEFAULT_SELECTED_ARTIFACT,
    config_path: str | Path = selected.CONFIG_PATH,
    cache_dir: str | Path = selected.DEFAULT_CACHE,
) -> dict[str, Any]:
    records = selected.load_unlabelled_jsonl(input_jsonl)
    runtime = M16HybridInferenceRuntime.from_artifacts(
        m16_artifact=m16_artifact,
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
        "m16_artifact": str(runtime.artifact),
        "training_invoked": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--m16-artifact", type=Path, default=DEFAULT_M16_ARTIFACT)
    parser.add_argument("--m12-artifact", type=Path, default=parent.DEFAULT_M12_ARTIFACT)
    parser.add_argument("--selected-artifact", type=Path, default=selected.DEFAULT_SELECTED_ARTIFACT)
    parser.add_argument("--config", type=Path, default=selected.CONFIG_PATH)
    parser.add_argument("--cache-dir", type=Path, default=selected.DEFAULT_CACHE)
    args = parser.parse_args(argv)
    try:
        result = run_hybrid(
            args.input_jsonl,
            args.output_jsonl,
            m16_artifact=args.m16_artifact,
            m12_artifact=args.m12_artifact,
            selected_artifact=args.selected_artifact,
            config_path=args.config,
            cache_dir=args.cache_dir,
        )
    except (ContractError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "M16_HYBRID_BLOCKED", "error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
