"""Run the fixed 100-row M3 Challenge V1 through the frozen CPU candidate.

This is a one-shot input trend diagnostic.  It consumes only the Challenge
JSONL and the selected local runtime; it does not load any labels or other
evaluation set and cannot train or alter the candidate.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from semantic_model import encoder_m1 as m1
from semantic_model import encoder_m2_selected_runtime as runtime_module
from semantic_model.hashes import sha256_file


DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "tests/fixtures/m3_synthetic_challenge_v1.jsonl"
DEFAULT_ARTIFACT = Path(__file__).resolve().parents[2] / "model-artifacts" / "m2-selected-candidate-frozen-for-m3-20260904"
DEFAULT_CACHE = Path(__file__).resolve().parents[1] / ".encoder-artifacts" / "hf-cache-lert-small"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "runs" / "m3a-challenge-v1-20260904"
SCALAR_HEADS = runtime_module.SCALAR_HEADS


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _unk_ratio(tokenizer: Any, record: Any, config: Mapping[str, Any]) -> float:
    ids = m1.build_input_ids(tokenizer, record, config)
    unk_id = tokenizer.unk_token_id
    if not isinstance(unk_id, int) or not ids:
        return 0.0
    return float(sum(value == unk_id for value in ids) / len(ids))


def _confidence_row(prediction: Mapping[str, Any]) -> dict[str, Any]:
    maxima: dict[str, float] = {}
    for head in SCALAR_HEADS:
        values = prediction["predictions"][head]["probabilities"]
        maxima[head] = float(max(values.values()))
    mean_max = float(np.mean(list(maxima.values())))
    return {"scalar_head_max_probability": maxima, "six_head_mean_max_probability": mean_max, "ood_score_one_minus_mean_max_probability": 1.0 - mean_max}


def enrich_predictions(records: Sequence[Any], predictions: Sequence[Mapping[str, Any]], tokenizer: Any, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    if len(records) != len(predictions):
        raise ValueError("prediction count does not match input count")
    enriched: list[dict[str, Any]] = []
    for record, prediction in zip(records, predictions, strict=True):
        row = dict(prediction)
        diagnostics = _confidence_row(row)
        diagnostics["tokenizer_unk_ratio"] = _unk_ratio(tokenizer, record, config)
        row["diagnostics"] = diagnostics
        enriched.append(row)
    return enriched


def _group_rows(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if not isinstance(value, str):
            raise ValueError(f"challenge row has no {field}")
        groups[value].append(row)
    return dict(sorted(groups.items()))


def confidence_groups(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, group in _group_rows(rows, field).items():
        head_means = {head: float(np.mean([row["diagnostics"]["scalar_head_max_probability"][head] for row in group])) for head in SCALAR_HEADS}
        mean_values = [row["diagnostics"]["six_head_mean_max_probability"] for row in group]
        result[name] = {
            "count": len(group),
            "scalar_head_max_probability_mean": head_means,
            "six_head_mean_max_probability_mean": float(np.mean(mean_values)),
            "six_head_mean_max_probability_min": float(np.min(mean_values)),
            "six_head_mean_max_probability_max": float(np.max(mean_values)),
            "tokenizer_unk_ratio_mean": float(np.mean([row["diagnostics"]["tokenizer_unk_ratio"] for row in group])),
        }
    return result


def clear_ood_confidence(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    clear = [row for row in rows if row.get("expected_scope") == "CLEAR_OOD"]
    any_high = [any(row["diagnostics"]["scalar_head_max_probability"][head] >= 0.90 for head in SCALAR_HEADS) for row in clear]
    mean_high = [row["diagnostics"]["six_head_mean_max_probability"] >= 0.80 for row in clear]
    both_high = [left and right for left, right in zip(any_high, mean_high, strict=True)]
    denominator = len(clear) or 1
    return {
        "count": len(clear),
        "any_scalar_head_at_least_0_90_count": int(sum(any_high)),
        "any_scalar_head_at_least_0_90_ratio": float(sum(any_high) / denominator),
        "six_head_mean_at_least_0_80_count": int(sum(mean_high)),
        "six_head_mean_at_least_0_80_ratio": float(sum(mean_high) / denominator),
        "both_conditions_count": int(sum(both_high)),
        "both_conditions_ratio": float(sum(both_high) / denominator),
    }


def ood_auc(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if row.get("expected_scope") in {"IN_DOMAIN_CONTROL", "CLEAR_OOD"}]
    labels = [int(row["expected_scope"] == "CLEAR_OOD") for row in selected]
    confidence_score = [row["diagnostics"]["ood_score_one_minus_mean_max_probability"] for row in selected]
    unk_score = [row["diagnostics"]["tokenizer_unk_ratio"] for row in selected]
    return {
        "scope": "IN_DOMAIN_CONTROL_vs_CLEAR_OOD_only",
        "rows": len(selected),
        "positive_class": "CLEAR_OOD",
        "score_one_minus_six_head_mean_max_probability": float(roc_auc_score(labels, confidence_score)),
        "score_tokenizer_unk_ratio": float(roc_auc_score(labels, unk_score)),
    }


def shift_description(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    shift = [row for row in rows if row.get("expected_scope") == "SHIFT_CHALLENGE"]
    result: dict[str, Any] = {}
    for category, group in _group_rows(shift, "challenge_category").items():
        result[category] = {
            "count": len(group),
            "six_head_mean_max_probability_mean": float(np.mean([row["diagnostics"]["six_head_mean_max_probability"] for row in group])),
            "tokenizer_unk_ratio_mean": float(np.mean([row["diagnostics"]["tokenizer_unk_ratio"] for row in group])),
            "participates_in_pass_fail": False,
        }
    return result


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    auc = ood_auc(rows)
    status = "M3_CHALLENGE_SEPARABLE_TREND" if max(auc["score_one_minus_six_head_mean_max_probability"], auc["score_tokenizer_unk_ratio"]) >= 0.75 else "M3_CHALLENGE_NO_SEPARABLE_TREND"
    return {
        "status": status,
        "rows": len(rows),
        "scope_counts": {name: len(group) for name, group in _group_rows(rows, "expected_scope").items()},
        "expected_scope_confidence": confidence_groups(rows, "expected_scope"),
        "challenge_category_confidence": confidence_groups(rows, "challenge_category"),
        "clear_ood_high_confidence": clear_ood_confidence(rows),
        "ood_roc_auc": auc,
        "shift_challenge_description": shift_description(rows),
        "selection_or_threshold_tuning_performed": False,
        "production_ood_claim": False,
    }


def run(*, input_path: str | Path = DEFAULT_INPUT, artifact_dir: str | Path = DEFAULT_ARTIFACT, cache_dir: str | Path = DEFAULT_CACHE, output_dir: str | Path = DEFAULT_OUTPUT, config_path: str | Path = runtime_module.CONFIG_PATH) -> dict[str, Any]:
    input_file = Path(input_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing immutable challenge output: {output}")
    records = runtime_module.load_unlabelled_jsonl(input_file)
    if len(records) != 100:
        raise RuntimeError(f"expected 100 challenge inputs, found {len(records)}")
    runtime = runtime_module.UnifiedCPUInferenceRuntime.from_artifact(artifact_dir, config_path=config_path, cache_dir=cache_dir)
    predictions = runtime.predict_records(records)
    rows = enrich_predictions(records, predictions, runtime.tokenizer, runtime.config)
    summary = summarize(rows)
    output.mkdir(parents=True, exist_ok=False)
    prediction_path = output / "challenge-predictions.jsonl"
    prediction_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    _write_json(output / "challenge-summary.json", summary)
    manifest = m1._write_content_manifest(output, {
        "manifest_schema_version": "myresearcher.m3a-challenge-v1-diagnostic-manifest.v1",
        "status": summary["status"],
        "input_sha256": sha256_file(input_file),
        "selected_artifact_content_address": runtime.manifest["content_address"],
        "selected_candidate": True,
        "production_approval": False,
        "rows": 100,
        "forbidden_inputs_not_read": ["Test", "Anchor", "Human100", "Gold", "OOD_real", "production", "LLM", "cloud"],
    })
    return {"status": summary["status"], "rows": len(rows), "output_dir": str(output), "content_address": manifest["content_address"], "summary": summary}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=runtime_module.CONFIG_PATH)
    args = parser.parse_args(argv)
    print(json.dumps(run(input_path=args.input_jsonl, artifact_dir=args.artifact_dir, cache_dir=args.cache_dir, output_dir=args.output_dir, config_path=args.config), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
