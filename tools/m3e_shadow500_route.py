"""Route the existing M3-C Shadow500 results without another inference pass.

This module consumes only the saved Shadow500 input/prediction JSONL files.  It
is an input-quality router, not an OOD detector: it makes no model call and
does not change probabilities, thresholds, weights, or predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


THRESHOLD = 0.5243767648935318
ROUTES = (
    "INPUT_EXCLUDED",
    "LOW_CONFIDENCE_ABSTAIN",
    "REVIEW_RECOMMENDED",
    "MODEL_OUTPUT",
)
EXPECTED_COUNTS = {
    "INPUT_EXCLUDED": 8,
    "LOW_CONFIDENCE_ABSTAIN": 27,
    "REVIEW_RECOMMENDED": 158,
    "MODEL_OUTPUT": 307,
}
INPUT_FIELDS = ("sample_id", "stock_code", "stock_name", "model_text")
RISK_FIELDS = ("text_quality_tier", "model_eligibility", "quality_flags")
DEFAULT_RUN = Path(__file__).resolve().parents[1] / "runs" / "m3c-shadow500-20260907"
DEFAULT_INPUT = DEFAULT_RUN / "shadow500-input.jsonl"
DEFAULT_PREDICTIONS = DEFAULT_RUN / "shadow500-predictions.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "runs" / "m3e-shadow500-routing-20260907"


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row {line_number} is not an object: {path}")
        rows.append(value)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def route(model_eligibility: str, score: float, *, threshold: float = THRESHOLD) -> str:
    """Apply the frozen mutually exclusive routing precedence."""

    eligibility = str(model_eligibility)
    if eligibility == "EXCLUDE_SEMANTIC":
        return "INPUT_EXCLUDED"
    if float(score) < float(threshold):
        return "LOW_CONFIDENCE_ABSTAIN"
    if eligibility == "CHALLENGE_REVIEW":
        return "REVIEW_RECOMMENDED"
    if eligibility == "ELIGIBLE":
        return "MODEL_OUTPUT"
    raise ValueError(f"unsupported model_eligibility: {eligibility}")


def route_prediction_row(row: Mapping[str, Any], *, threshold: float = THRESHOLD) -> dict[str, Any]:
    """Add a route and suppress excluded predictions from downstream output."""

    sample_id = row.get("sample_id")
    eligibility = row.get("model_eligibility")
    score = row.get("six_head_mean_max_probability")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("prediction row requires a non-empty sample_id")
    if not isinstance(eligibility, str):
        raise ValueError(f"prediction row {sample_id} has no model_eligibility")
    try:
        numeric_score = float(score)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"prediction row {sample_id} has an invalid confidence score") from exc
    route_name = route(eligibility, numeric_score, threshold=threshold)
    output = {
        "sample_id": sample_id,
        "stock_code": row.get("stock_code", ""),
        "stock_name": row.get("stock_name", ""),
        "text_quality_tier": row.get("text_quality_tier", ""),
        "model_eligibility": eligibility,
        "quality_flags": row.get("quality_flags", ""),
        "route": route_name,
        "six_head_mean_max_probability": numeric_score,
        "tokenizer_unk_ratio": float(row.get("tokenizer_unk_ratio", 0.0)),
        "threshold": float(threshold),
        "abstain": bool(row.get("abstain", numeric_score < threshold)),
        "downstream_action": {
            "INPUT_EXCLUDED": "drop_before_normal_seven_head_output",
            "LOW_CONFIDENCE_ABSTAIN": "abstain_and_do_not_emit_as_confident_output",
            "REVIEW_RECOMMENDED": "retain_prediction_with_review_recommendation",
            "MODEL_OUTPUT": "emit_normal_model_output",
        }[route_name],
        "prediction_available": route_name != "INPUT_EXCLUDED",
    }
    if route_name != "INPUT_EXCLUDED":
        if "predictions" not in row:
            raise ValueError(f"prediction row {sample_id} is missing predictions")
        output["predictions"] = row["predictions"]
    return output


def route_rows(rows: Sequence[Mapping[str, Any]], *, threshold: float = THRESHOLD) -> list[dict[str, Any]]:
    result = [route_prediction_row(row, threshold=threshold) for row in rows]
    ids = [row["sample_id"] for row in result]
    if len(ids) != len(set(ids)):
        raise ValueError("routed prediction sample IDs are not unique")
    return result


def _validate_input_prediction_binding(inputs: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]) -> None:
    input_ids = [row.get("sample_id") for row in inputs]
    prediction_ids = [row.get("sample_id") for row in predictions]
    if len(input_ids) != len(set(input_ids)) or len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError("Shadow500 input/prediction sample IDs must be unique")
    if input_ids != prediction_ids:
        raise ValueError("Shadow500 input and prediction order/identity do not match")
    forbidden = {"author_id", "author_name", "url", "label", "labels", "split", "role"}
    for row in inputs:
        if forbidden.intersection(row):
            raise ValueError("Shadow500 input contains a forbidden author/label/split field")


def run(
    *,
    input_path: str | Path = DEFAULT_INPUT,
    predictions_path: str | Path = DEFAULT_PREDICTIONS,
    output_dir: str | Path = DEFAULT_OUTPUT,
    threshold: float = THRESHOLD,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite immutable routing output: {output}")
    inputs = _read_jsonl(input_path)
    predictions = _read_jsonl(predictions_path)
    if len(inputs) != 500 or len(predictions) != 500:
        raise ValueError(f"expected 500 Shadow500 inputs and predictions, got {len(inputs)} and {len(predictions)}")
    _validate_input_prediction_binding(inputs, predictions)
    routed = route_rows(predictions, threshold=threshold)
    counts = Counter(row["route"] for row in routed)
    if set(counts) != set(ROUTES) or sum(counts.values()) != 500:
        raise ValueError(f"routing is not exhaustive: {dict(counts)}")
    output.mkdir(parents=True, exist_ok=False)
    route_path = output / "shadow500-routed.jsonl"
    summary_path = output / "routing-summary.json"
    _write_jsonl(route_path, routed)
    summary = {
        "status": "M3_SHADOW500_ROUTING_COMPLETED_DIAGNOSTIC",
        "diagnostic_only": True,
        "semantic_ood_claim": False,
        "selected_candidate": False,
        "production_approval": False,
        "source": {
            "input_path": str(Path(input_path).resolve()),
            "input_sha256": _sha256(input_path),
            "predictions_path": str(Path(predictions_path).resolve()),
            "predictions_sha256": _sha256(predictions_path),
            "rows": 500,
            "second_model_inference": False,
        },
        "threshold": {
            "score": "six_head_mean_max_probability",
            "value": float(threshold),
            "rule": "score < threshold -> LOW_CONFIDENCE_ABSTAIN after EXCLUDE_SEMANTIC precedence",
            "source": "frozen M3-C/M3-B threshold; not recalibrated",
        },
        "route_precedence": [
            "model_eligibility=EXCLUDE_SEMANTIC -> INPUT_EXCLUDED",
            "otherwise score < 0.5243767648935318 -> LOW_CONFIDENCE_ABSTAIN",
            "otherwise model_eligibility=CHALLENGE_REVIEW -> REVIEW_RECOMMENDED",
            "otherwise model_eligibility=ELIGIBLE -> MODEL_OUTPUT",
        ],
        "counts": {route_name: int(counts[route_name]) for route_name in ROUTES},
        "expected_counts": EXPECTED_COUNTS,
        "counts_match_expected": {route_name: int(counts[route_name]) == value for route_name, value in EXPECTED_COUNTS.items()},
        "input_excluded": {
            "count": int(counts["INPUT_EXCLUDED"]),
            "predictions_suppressed": True,
            "normal_seven_head_downstream_output": False,
            "not_remapped_to_unknown": True,
        },
        "route_path": route_path.name,
        "forbidden_inputs_not_read": ["labels", "Test", "Anchor", "Human100", "Gold", "OOD", "reference_predictions", "LLM", "cloud", "production_data"],
    }
    _write_json(summary_path, summary)
    summary["route_sha256"] = _sha256(route_path)
    summary["summary_path"] = summary_path.name
    # The summary is intentionally observational rather than self-hashed.
    _write_json(summary_path, summary)
    return {
        "status": summary["status"],
        "rows": len(routed),
        "counts": summary["counts"],
        "output_dir": str(output),
        "route_sha256": summary["route_sha256"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    args = parser.parse_args(argv)
    print(json.dumps(run(input_path=args.input, predictions_path=args.predictions, output_dir=args.output_dir, threshold=args.threshold), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
