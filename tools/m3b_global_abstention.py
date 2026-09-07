"""Evaluate a fixed input-level abstention score on Dev and Challenge V1.

This is a diagnostic-only, no-training pass.  The threshold is selected once
from the Dev score distribution with NumPy's ``method='lower'`` percentile and
then applied unchanged to the existing input-only Challenge predictions.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from semantic_model import encoder_m1 as m1
from semantic_model import encoder_m2_selected_runtime as selected
from semantic_model.hashes import sha256_file


DEFAULT_ARTIFACT = Path(__file__).resolve().parents[2] / "model-artifacts" / "m2-selected-candidate-frozen-for-m3-20260904"
DEFAULT_CACHE = Path(__file__).resolve().parents[1] / ".encoder-artifacts" / "hf-cache-lert-small"
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "baseline_v0.3.5.yaml"
DEFAULT_CHALLENGE_PREDICTIONS = Path(__file__).resolve().parents[1] / "runs" / "m3a-challenge-v1-20260904" / "challenge-predictions.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "runs" / "m3b-global-abstention-20260907"
SCALAR_HEADS = selected.SCALAR_HEADS


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def load_prediction_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid prediction JSONL at line {line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"prediction row {line_number} is not an object")
        rows.append(value)
    return rows


def six_head_mean_max_probability(row: Mapping[str, Any]) -> float:
    maxima = []
    predictions = row.get("predictions")
    if not isinstance(predictions, Mapping):
        raise ValueError("prediction row has no predictions object")
    for head in SCALAR_HEADS:
        values = predictions[head].get("probabilities")
        if not isinstance(values, Mapping) or not values:
            raise ValueError(f"prediction row has no probabilities for {head}")
        maxima.append(max(float(value) for value in values.values()))
    score = float(np.mean(maxima))
    if not np.isfinite(score):
        raise ValueError("input-level score is non-finite")
    return score


def freeze_dev_threshold(scores: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(scores, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Dev scores must be non-empty and finite")
    threshold = float(np.percentile(values, 5, method="lower"))
    abstain = values < threshold
    return {
        "score": "six_head_mean_max_probability",
        "percentile": 5,
        "method": "lower",
        "threshold": threshold,
        "dev_rows": int(values.size),
        "dev_abstain_count": int(abstain.sum()),
        "dev_false_rejection_rate": float(abstain.mean()),
        "dev_coverage": float((~abstain).mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def _head_metrics(truth: Sequence[str], predictions: Sequence[str], order: Sequence[str]) -> dict[str, Any]:
    return {
        "rows": len(truth),
        "accuracy": float(accuracy_score(truth, predictions)) if truth else None,
        "macro_f1": float(f1_score(truth, predictions, labels=list(order), average="macro", zero_division=0)) if truth else None,
        "class_order": list(order),
    }


def dev_head_metrics(records: Sequence[m1.M1Record], prediction_rows: Sequence[Mapping[str, Any]], schema: Any, threshold: float) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for head in SCALAR_HEADS:
        truth = [str(record.label[head]) for record in records]
        predicted = [str(row["predictions"][head]["prediction"]) for row in prediction_rows]
        keep = [six_head_mean_max_probability(row) >= threshold for row in prediction_rows]
        kept_truth = [value for value, include in zip(truth, keep, strict=True) if include]
        kept_predicted = [value for value, include in zip(predicted, keep, strict=True) if include]
        result[head] = {
            "abstain_before": _head_metrics(truth, predicted, schema.class_order[head]),
            "abstain_after": _head_metrics(kept_truth, kept_predicted, schema.class_order[head]),
            "abstain_count": int(sum(not include for include in keep)),
            "coverage": float(sum(keep) / len(keep)),
        }
    return result


def apply_challenge_threshold(rows: Sequence[Mapping[str, Any]], threshold: float) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        score = six_head_mean_max_probability(row)
        copy = dict(row)
        copy["abstention"] = {"score": score, "threshold": float(threshold), "abstain": bool(score < threshold), "coverage": bool(score >= threshold)}
        result.append(copy)
    return result


def challenge_metrics(rows: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    applied = apply_challenge_threshold(rows, threshold)
    clear = [row for row in applied if row.get("expected_scope") == "CLEAR_OOD"]
    control = [row for row in applied if row.get("expected_scope") == "IN_DOMAIN_CONTROL"]
    clear_recall = float(sum(row["abstention"]["abstain"] for row in clear) / len(clear)) if clear else 0.0
    control_false_rejection = float(sum(row["abstention"]["abstain"] for row in control) / len(control)) if control else 0.0
    shifts: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in applied:
        if row.get("expected_scope") == "SHIFT_CHALLENGE":
            shifts[str(row.get("challenge_category"))].append(row)
    shift_rates = {category: {"rows": len(group), "abstain_count": int(sum(row["abstention"]["abstain"] for row in group)), "abstention_rate": float(sum(row["abstention"]["abstain"] for row in group) / len(group))} for category, group in sorted(shifts.items())}
    released_clear = [row for row in clear if not row["abstention"]["abstain"]]
    highest_released = None
    if released_clear:
        row = max(released_clear, key=lambda item: item["abstention"]["score"])
        highest_released = {"sample_id": row.get("sample_id"), "score": float(row["abstention"]["score"]), "threshold": float(threshold)}
    return {
        "threshold": float(threshold),
        "clear_ood": {"rows": len(clear), "abstain_count": int(sum(row["abstention"]["abstain"] for row in clear)), "recall": clear_recall},
        "in_domain_control": {"rows": len(control), "abstain_count": int(sum(row["abstention"]["abstain"] for row in control)), "false_rejection_rate": control_false_rejection},
        "shift_challenge": shift_rates,
        "highest_confidence_released_clear_ood": highest_released,
        "applied_rows": applied,
    }


def run(*, artifact_dir: str | Path = DEFAULT_ARTIFACT, cache_dir: str | Path = DEFAULT_CACHE, config_path: str | Path = DEFAULT_CONFIG, challenge_predictions: str | Path = DEFAULT_CHALLENGE_PREDICTIONS, output_dir: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing abstention output: {output}")
    runtime = selected.UnifiedCPUInferenceRuntime.from_artifact(artifact_dir, config_path=config_path, cache_dir=cache_dir)
    project = m1.ProjectConfig.load(config_path)
    schema, _train, dev = m1.load_m1_partitions(project)
    dev_predictions = runtime.predict_records(dev)
    if len(dev_predictions) != 448:
        raise RuntimeError(f"expected 448 Dev predictions, found {len(dev_predictions)}")
    dev_scores = [six_head_mean_max_probability(row) for row in dev_predictions]
    threshold_info = freeze_dev_threshold(dev_scores)
    challenge_rows = load_prediction_jsonl(challenge_predictions)
    if len(challenge_rows) != 100:
        raise RuntimeError(f"expected 100 existing Challenge predictions, found {len(challenge_rows)}")
    challenge = challenge_metrics(challenge_rows, threshold_info["threshold"])
    dev_metrics = dev_head_metrics(dev, dev_predictions, schema, threshold_info["threshold"])
    status = "M3_GLOBAL_ABSTENTION_FEASIBLE_DIAGNOSTIC" if threshold_info["dev_false_rejection_rate"] <= 0.05 and challenge["clear_ood"]["recall"] >= 0.80 else "M3_GLOBAL_ABSTENTION_INSUFFICIENT_DIAGNOSTIC"
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "dev-abstention-metrics.json", {"threshold": threshold_info, "heads": dev_metrics, "rows": len(dev_predictions)})
    _write_json(output / "challenge-abstention-metrics.json", {key: value for key, value in challenge.items() if key != "applied_rows"})
    _write_json(output / "abstention-aggregate.json", {
        "schema_version": "myresearcher.m3b-global-abstention-diagnostic.v1",
        "status": status,
        "selected_artifact_content_address": runtime.manifest["content_address"],
        "dev": {"rows": 448, "threshold": threshold_info, "heads": dev_metrics},
        "challenge": {"rows": 100, "clear_ood_recall": challenge["clear_ood"]["recall"], "synthetic_control_false_rejection_rate": challenge["in_domain_control"]["false_rejection_rate"], "shift_abstention_rates": challenge["shift_challenge"], "highest_confidence_released_clear_ood": challenge["highest_confidence_released_clear_ood"]},
        "selection_or_threshold_tuning_on_challenge": False,
        "forbidden_inputs_not_read": ["Test", "Anchor", "Human100", "real_production_data", "Gold", "OOD_real"],
    })
    manifest = m1._write_content_manifest(output, {"manifest_schema_version": "myresearcher.m3b-global-abstention-manifest.v1", "status": status, "selected_artifact_content_address": runtime.manifest["content_address"], "challenge_input_predictions_sha256": sha256_file(challenge_predictions), "rows": {"dev": 448, "challenge": 100}, "forbidden_inputs_not_read": ["Test", "Anchor", "Human100", "real_production_data", "Gold", "OOD_real", "LLM", "cloud"]})
    return {"status": status, "output_dir": str(output), "content_address": manifest["content_address"], "threshold": threshold_info, "dev_heads": dev_metrics, "challenge": {key: value for key, value in challenge.items() if key != "applied_rows"}}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--challenge-predictions", type=Path, default=DEFAULT_CHALLENGE_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(json.dumps(run(artifact_dir=args.artifact_dir, cache_dir=args.cache_dir, config_path=args.config, challenge_predictions=args.challenge_predictions, output_dir=args.output_dir), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
