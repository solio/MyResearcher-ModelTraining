"""M8A read-only shadow inference for the five v0.3 auxiliary heads.

The M7B checkpoint is used only as a diagnostic shadow layer.  The selected
seven-head runtime remains the authoritative output and is never replaced.
Applicability is decided from explicit text cues (the frozen M7A rules), while
the final-assignment coverage file suppresses already covered sample/head
pairs.  No label, Test, Gold, OOD or production input is read here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .auxiliary_heads import AuxiliaryHeadModel
from .encoder_m2_selected_runtime import InferenceRecord, UnifiedCPUInferenceRuntime
from .encoder_m6c_auxiliary import DEFAULT_CANDIDATE_SCHEMA
from .encoder_m7b_auxiliary import read_jsonl
from .schema import SemanticSchemaCandidate, V03_AUXILIARY_FACTORS

try:
    from tools.feedback_loop_m7a import _applicable_factors
    from tools.prepare_v03_auxiliary_pilot import _is_low_information, _looks_truncated
except ModuleNotFoundError:  # pragma: no cover - direct module invocation
    from ..tools.feedback_loop_m7a import _applicable_factors  # type: ignore
    from ..tools.prepare_v03_auxiliary_pilot import _is_low_information, _looks_truncated  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT.parent
DEFAULT_FEED = REPO_ROOT / "runs" / "feedback-loop-v3-20260910" / "feed-input.jsonl"
DEFAULT_COVERAGE = PROJECT_ROOT / "human-review-artifacts" / "annotation-coverage-20260913" / "final-assignment-coverage.jsonl"
DEFAULT_M7B_CHECKPOINT = PROJECT_ROOT / "model-artifacts" / "m7b-v03-auxiliary-diagnostic-20260913" / "auxiliary-checkpoint.pt"
DEFAULT_SELECTED_ARTIFACT = PROJECT_ROOT / "model-artifacts" / "m2-selected-candidate-frozen-for-m3-20260904"
DEFAULT_CACHE = REPO_ROOT / ".encoder-artifacts" / "hf-cache-lert-small"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "baseline_v0.3.5.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "model-artifacts" / "m8a-v03-auxiliary-shadow-20260914"
M7B_CHECKPOINT_SHA256 = ""
MULTI_FACTORS = {"market_scope", "discourse_focus"}
AUXILIARY_THRESHOLD = 0.5
HIGH_UNCERTAINTY_ENTROPY = 0.75


def _require(ok: bool, code: str, message: str, **details: Any) -> None:
    if not ok:
        raise ValueError(f"{code}: {message}; {details}" if details else f"{code}: {message}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_coverage(path: str | Path) -> set[tuple[str, str]]:
    """Load final sample/head coverage without reading any answer values."""

    rows = read_jsonl(path)
    result: set[tuple[str, str]] = set()
    for row in rows:
        if row.get("is_final") is not True:
            continue
        sample_id, head = row.get("sample_id"), row.get("head")
        if isinstance(sample_id, str) and isinstance(head, str) and head in V03_AUXILIARY_FACTORS:
            result.add((sample_id, head))
    return result


def prepare_shadow_rows(
    feed_rows: Sequence[Mapping[str, Any]],
    covered_pairs: set[tuple[str, str]],
) -> tuple[list[InferenceRecord], list[dict[str, Any]]]:
    """Filter duplicate/low-information/truncated inputs and freeze applicability."""

    ids = [row.get("sample_id") for row in feed_rows]
    _require(len(ids) == len(set(ids)), "M8A_FEED_SAMPLE_ID_DUPLICATE", "feed sample_id values must be unique")
    group_counts = Counter(row.get("exact_content_group") for row in feed_rows)
    prepared: list[InferenceRecord] = []
    metadata_rows: list[dict[str, Any]] = []
    for row in feed_rows:
        sample_id = row.get("sample_id")
        text = row.get("model_text") or row.get("text")
        group = row.get("exact_content_group")
        if not isinstance(sample_id, str) or not sample_id or not isinstance(text, str) or not text.strip():
            continue
        if _is_low_information(text):
            continue
        text_length = row.get("text_length", len(text))
        try:
            length = int(text_length)
        except (TypeError, ValueError):
            length = len(text)
        if not isinstance(group, str) or not group or group_counts[group] > 1:
            continue
        if length >= 39 or _looks_truncated(sample_id, text, text_length):
            continue
        if text.strip().endswith(("，", "、", "：", ":", "和", "与", "或", "如果", "因为", "但是", "所以", "而且", "然后", "以及", "最近", "其中", "之后", "以后", "时候")):
            continue
        applicable = tuple(_applicable_factors(text))
        uncovered = tuple(factor for factor in applicable if (sample_id, factor) not in covered_pairs)
        if not uncovered:
            continue
        metadata = {
            "sample_id": sample_id,
            "model_text": text,
            "stock_code": row.get("stock_code"),
            "stock_name": row.get("stock_name"),
            "exact_content_group": group,
            "observation_id": row.get("observation_id"),
            "sampling_reasons": list(row.get("sampling_reasons") or []),
            "text_length": length,
            "applicable_factors": list(applicable),
            "uncovered_factors": list(uncovered),
        }
        prepared.append(
            InferenceRecord(
                sample_id=sample_id,
                stock_code=str(row.get("stock_code") or ""),
                stock_name=str(row.get("stock_name") or ""),
                model_text=text,
                metadata={"exact_content_group": group},
            )
        )
        metadata_rows.append(metadata)
    _require(len(prepared) == len(metadata_rows), "M8A_PREPARED_ID_MISMATCH", "prepared records and metadata differ")
    return prepared, metadata_rows


def _normalized_entropy(probabilities: Sequence[float]) -> float:
    values = np.asarray(probabilities, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        return 1.0
    values = np.clip(values, 1e-12, 1.0)
    values = values / values.sum()
    entropy = float(-(values * np.log(values)).sum())
    return entropy / math.log(max(2, values.size))


def _multi_entropy(probabilities: Sequence[float]) -> float:
    values = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0 - 1e-12)
    entropy = -(values * np.log(values) + (1.0 - values) * np.log(1.0 - values))
    return float(np.mean(entropy) / math.log(2.0)) if values.size else 1.0


def auxiliary_prediction_rows(
    records: Sequence[InferenceRecord],
    metadata_rows: Sequence[Mapping[str, Any]],
    owner_representations: Mapping[str, torch.Tensor],
    auxiliary_model: AuxiliaryHeadModel,
    runtime: UnifiedCPUInferenceRuntime,
) -> list[dict[str, Any]]:
    """Create per-input shadow rows from cached frozen representations."""

    _require(len(records) == len(metadata_rows), "M8A_RECORD_METADATA_MISMATCH", "record and metadata counts differ")
    with torch.no_grad():
        auxiliary_logits = auxiliary_model(owner_representations)
    legacy_rows = runtime.predict_records(records)
    result: list[dict[str, Any]] = []
    for index, (record, metadata) in enumerate(zip(records, metadata_rows)):
        row: dict[str, Any] = {
            "sample_id": record.sample_id,
            "model_text": record.model_text,
            "stock_code": record.stock_code,
            "stock_name": record.stock_name,
            "exact_content_group": metadata.get("exact_content_group"),
            "applicable_factors": list(metadata.get("applicable_factors", [])),
            "uncovered_factors": list(metadata.get("uncovered_factors", [])),
            "legacy_predictions": legacy_rows[index].get("predictions", {}),
            "auxiliary": {},
        }
        for factor in metadata.get("uncovered_factors", []):
            spec = auxiliary_model.schema.auxiliary_factors[factor]
            logits = auxiliary_logits[factor][index].detach().cpu()
            if not torch.isfinite(logits).all():
                raise ValueError(f"M8A_AUXILIARY_NONFINITE_OUTPUT: {factor}/{record.sample_id}")
            if factor in MULTI_FACTORS:
                probabilities = torch.sigmoid(logits).numpy().astype(float).tolist()
                labels = [label for label, probability in zip(spec.class_order, probabilities) if probability >= AUXILIARY_THRESHOLD]
                entropy = _multi_entropy(probabilities)
                confidence = float(max(probabilities)) if probabilities else 0.0
            else:
                probabilities = torch.softmax(logits, dim=-1).numpy().astype(float).tolist()
                predicted_index = int(np.argmax(probabilities))
                labels = [spec.class_order[predicted_index]]
                entropy = _normalized_entropy(probabilities)
                confidence = float(max(probabilities)) if probabilities else 0.0
            row["auxiliary"][factor] = {
                "logits": [float(value) for value in logits.numpy().tolist()],
                "probabilities": {label: float(probability) for label, probability in zip(spec.class_order, probabilities)},
                "prediction": labels,
                "confidence": confidence,
                "normalized_entropy": entropy,
                "threshold": AUXILIARY_THRESHOLD if factor in MULTI_FACTORS else None,
                "high_uncertainty": bool(entropy >= HIGH_UNCERTAINTY_ENTROPY),
            }
        result.append(row)
    return result


def summarize_shadow(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "sample_count": len(rows),
        "factor_summary": {},
        "high_uncertainty_union_count": 0,
        "criteria": {
            "applicability": "explicit text cues only; legacy predictions never veto",
            "multi_label_threshold": AUXILIARY_THRESHOLD,
            "high_uncertainty": f"normalized_entropy >= {HIGH_UNCERTAINTY_ENTROPY}",
            "collapse": "max predicted category/label share > 0.90",
        },
    }
    for factor in V03_AUXILIARY_FACTORS:
        entries = [row["auxiliary"][factor] for row in rows if factor in row.get("auxiliary", {})]
        predictions = [entry["prediction"] for entry in entries]
        distribution: Counter[str] = Counter()
        for prediction in predictions:
            if factor in MULTI_FACTORS:
                distribution.update(prediction)
            elif prediction:
                distribution[prediction[0]] += 1
        denominator = len(predictions) if factor not in MULTI_FACTORS else sum(distribution.values())
        shares = {label: (count / denominator if denominator else 0.0) for label, count in distribution.items()}
        max_share = max(shares.values(), default=0.0)
        uncertainties = [float(entry["normalized_entropy"]) for entry in entries]
        confidences = [float(entry["confidence"]) for entry in entries]
        summary["factor_summary"][factor] = {
            "applicable_uncovered_sample_count": len(entries),
            "prediction_distribution": dict(sorted(distribution.items())),
            "category_or_label_share": dict(sorted(shares.items())),
            "max_category_or_label_share": max_share,
            "single_category_collapse": bool(max_share > 0.90),
            "mean_confidence": float(np.mean(confidences)) if confidences else None,
            "mean_normalized_entropy": float(np.mean(uncertainties)) if uncertainties else None,
            "high_uncertainty_count": int(sum(value >= HIGH_UNCERTAINTY_ENTROPY for value in uncertainties)),
            "high_uncertainty_rate": float(sum(value >= HIGH_UNCERTAINTY_ENTROPY for value in uncertainties) / len(uncertainties)) if uncertainties else None,
        }
    summary["high_uncertainty_union_count"] = sum(
        1 for row in rows if any(entry.get("high_uncertainty") for entry in row.get("auxiliary", {}).values())
    )
    return summary


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def run_shadow(
    *,
    feed_path: str | Path = DEFAULT_FEED,
    coverage_path: str | Path = DEFAULT_COVERAGE,
    checkpoint_path: str | Path = DEFAULT_M7B_CHECKPOINT,
    selected_artifact: str | Path = DEFAULT_SELECTED_ARTIFACT,
    cache_dir: str | Path = DEFAULT_CACHE,
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT,
    schema_path: str | Path = DEFAULT_CANDIDATE_SCHEMA,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    _require(not output.exists(), "M8A_OUTPUT_EXISTS", "shadow output must be a new immutable directory", path=str(output))
    checkpoint = Path(checkpoint_path).resolve()
    _require(checkpoint.is_file(), "M8A_CHECKPOINT_MISSING", "M7B auxiliary checkpoint is missing", path=str(checkpoint))
    schema = SemanticSchemaCandidate.load(schema_path)
    feed_rows = read_jsonl(feed_path)
    coverage = load_coverage(coverage_path)
    records, metadata_rows = prepare_shadow_rows(feed_rows, coverage)
    _require(records, "M8A_NO_ELIGIBLE_INPUTS", "no uncovered eligible feed inputs remain")
    # AuxiliaryTrainingRuntime wraps the same selected runtime and exposes the
    # owner representations while preserving the original seven-head model.
    from .encoder_m6c_auxiliary import AuxiliaryTrainingRuntime
    auxiliary_runtime = AuxiliaryTrainingRuntime.from_selected_artifact(selected_artifact, config_path=config_path, cache_dir=cache_dir)
    runtime = auxiliary_runtime.selected_runtime
    model_inputs = __import__("semantic_model.encoder_m2_selected_runtime", fromlist=["build_inference_batch"]).build_inference_batch(
        runtime.torch, runtime.tokenizer, records, runtime.config, runtime.torch.device("cpu")
    )
    # LertAuxiliaryTrainingModel captures all three logical-owner pooled
    # representations in one frozen forward.  Its random auxiliary module is
    # ignored; the saved M7B state is loaded into a separate read-only head.
    with auxiliary_runtime.torch.no_grad():
        captured = auxiliary_runtime.model(model_inputs["input_ids"], model_inputs["attention_mask"])
    owner_representations = {owner: value.detach().cpu() for owner, value in captured["owner_representations"].items()}
    hidden_size = int(next(iter(owner_representations.values())).shape[-1])
    auxiliary_model = AuxiliaryHeadModel(schema, hidden_size)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    _require(isinstance(payload, Mapping) and isinstance(payload.get("auxiliary_state_dict"), Mapping), "M8A_CHECKPOINT_INVALID", "M7B checkpoint payload is invalid")
    auxiliary_model.load_state_dict(payload["auxiliary_state_dict"])
    auxiliary_model.eval()
    rows = auxiliary_prediction_rows(records, metadata_rows, owner_representations, auxiliary_model, runtime)
    summary = summarize_shadow(rows)
    summary.update({
        "status": "M8A_AUXILIARY_SHADOW_DIAGNOSTIC_ONLY",
        "selected_candidate": False,
        "feed_path": str(Path(feed_path).resolve()),
        "feed_rows": len(feed_rows),
        "final_assignment_coverage_path": str(Path(coverage_path).resolve()),
        "covered_auxiliary_pairs": len(coverage),
        "m7b_checkpoint": str(checkpoint),
        "m7b_checkpoint_sha256": sha256_file(checkpoint),
        "selected_artifact": str(Path(selected_artifact).resolve()),
        "snapshot": str((Path(cache_dir).resolve() / "official-snapshot" / "69e3e69ba258be5b301b26937e5b55a076c90460")),
        "model_id": "hfl/chinese-lert-small",
        "model_revision": "69e3e69ba258be5b301b26937e5b55a076c90460",
        "device": "cpu",
        "m7b_reference": "OOF signal was sparse diagnostic evidence only; this checkpoint is shadow/active-sampling evidence, not an effective model.",
    })
    output.mkdir(parents=True, exist_ok=False)
    (output / "shadow-predictions.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    _write_json(output / "shadow-summary.json", summary)
    report_lines = [
        "# M8A auxiliary shadow diagnostic",
        "",
        "The M7B five-factor checkpoint is attached read-only to the frozen selected LERT runtime. It is not an effective model, selected candidate, or production output.",
        "",
        f"- Feed rows read: {len(feed_rows)}; eligible shadow samples: {len(rows)}",
        f"- Final auxiliary pairs excluded: {len(coverage)}; device: CPU",
        f"- M7B checkpoint SHA-256: `{summary['m7b_checkpoint_sha256']}`",
        "",
        "| Factor | Uncovered applicable samples | Mean confidence | Mean normalized entropy | High uncertainty | Max share | Collapse |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for factor in V03_AUXILIARY_FACTORS:
        item = summary["factor_summary"][factor]
        report_lines.append(f"| `{factor}` | {item['applicable_uncovered_sample_count']} | {item['mean_confidence'] if item['mean_confidence'] is not None else 'n/a'} | {item['mean_normalized_entropy'] if item['mean_normalized_entropy'] is not None else 'n/a'} | {item['high_uncertainty_count']} | {item['max_category_or_label_share']:.4f} | {item['single_category_collapse']} |")
    report_lines.extend([
        "",
        f"Rows with at least one high-uncertainty auxiliary factor: {summary['high_uncertainty_union_count']}",
        "",
        "Applicability uses explicit text cues only: legacy seven-head predictions do not veto `arousal_level` or `market_scope`; bare `市场` is not sufficient; action words alone do not trigger `desired_price_direction`. Multi-label thresholds are fixed at 0.5 and were not searched.",
        "",
        "No annotation batch was created. Human Check, Test, Gold, OOD, LLM and production data were not read.",
        "",
    ])
    (output / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    files = [path for path in output.iterdir() if path.is_file()]
    manifest = {"artifact": "m8a-v03-auxiliary-shadow-20260914", "files": {path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(files)}}
    _write_json(output / "manifest.json", manifest)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feed", type=Path, default=DEFAULT_FEED)
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_M7B_CHECKPOINT)
    parser.add_argument("--selected-artifact", type=Path, default=DEFAULT_SELECTED_ARTIFACT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--schema", type=Path, default=DEFAULT_CANDIDATE_SCHEMA)
    args = parser.parse_args(argv)
    result = run_shadow(feed_path=args.feed, coverage_path=args.coverage, checkpoint_path=args.checkpoint, selected_artifact=args.selected_artifact, cache_dir=args.cache_dir, config_path=args.config, output_dir=args.output_dir, schema_path=args.schema)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["load_coverage", "prepare_shadow_rows", "auxiliary_prediction_rows", "summarize_shadow", "run_shadow"]
