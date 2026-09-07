"""Run the one-shot, input-only M3-C Shadow500 diagnostic.

The canonical pool is scanned only to select a deterministic 500-row sample.
The pilot file contributes sample IDs only; no labels, split roles, or other
evaluation data are read.  The selected candidate is used for one CPU
inference pass and the output is a diagnostic, never Gold/OOD or production
evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from semantic_model import encoder_m1 as m1
from semantic_model import encoder_m2_selected_runtime as selected
from semantic_model.hashes import sha256_file


POOL_SHA256 = "ee4adae632c3e5d22ede6d5fd072817fdb1ffc83b0b1cfbaac7ab1c61ba4100c"
POOL_ROWS = 49_054
PILOT_ROWS = 3_000
SHADOW_ROWS = 500
SELECTION_PREFIX = "m3c-shadow500-v1|"
THRESHOLD = 0.5243767648935318
THRESHOLD_METHOD = "Dev fifth percentile, method=lower; strict score < threshold"
RISK_FIELDS = ("text_quality_tier", "model_eligibility", "quality_flags")
SAFE_INPUT_FIELDS = ("sample_id", "stock_code", "stock_name", "model_text")
DEFAULT_POOL = Path(
    "/Users/mac/Documents/trae_projects/MyResearcher/produce-docs/"
    "MyResearcher_Semantic_Sampling_Local_Pipeline_Formal_v0.1/formal_run/clean_posts.csv"
)
DEFAULT_PILOT = DEFAULT_POOL.with_name("semantic_pilot_inputs.jsonl")
DEFAULT_ARTIFACT = Path(
    "/Users/mac/Documents/trae_projects/MyResearcher/model-artifacts/"
    "m2-selected-candidate-frozen-for-m3-20260904"
)
DEFAULT_CACHE = Path(__file__).resolve().parents[1] / ".encoder-artifacts" / "hf-cache-lert-small"
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "baseline_v0.3.5.yaml"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "runs" / "m3c-shadow500-20260907"


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


def _sha256_selection(sample_id: str) -> str:
    return hashlib.sha256((SELECTION_PREFIX + sample_id).encode("utf-8")).hexdigest()


def read_pilot_ids(path: str | Path) -> set[str]:
    """Read only the pilot ``sample_id`` field, never labels or split roles."""

    ids: set[str] = set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping) or not isinstance(value.get("sample_id"), str) or not value["sample_id"]:
            raise ValueError(f"pilot row {line_number} has no valid sample_id")
        sample_id = value["sample_id"]
        if sample_id in ids:
            raise ValueError(f"duplicate pilot sample_id: {sample_id}")
        ids.add(sample_id)
    if len(ids) != PILOT_ROWS:
        raise ValueError(f"expected {PILOT_ROWS} unique pilot IDs, found {len(ids)}")
    return ids


def _safe_input(row: Mapping[str, Any]) -> dict[str, Any]:
    result = {field: row.get(field, "") for field in SAFE_INPUT_FIELDS}
    if not all(isinstance(result[field], str) and result[field] for field in SAFE_INPUT_FIELDS):
        raise ValueError("canonical row lacks a required non-empty input field")
    result.update({field: row.get(field, "") for field in RISK_FIELDS})
    return result


def read_canonical_pool(path: str | Path, pilot_ids: set[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read canonical inputs and risk metadata, excluding only pilot IDs."""

    source = Path(path).resolve()
    observed_sha = sha256_file(source)
    if observed_sha != POOL_SHA256:
        raise ValueError(f"canonical CSV SHA-256 mismatch: {observed_sha}")
    rows: list[dict[str, Any]] = []
    all_ids: set[str] = set()
    stock_codes: set[str] = set()
    total = 0
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = set(SAFE_INPUT_FIELDS) | set(RISK_FIELDS)
        if not required.issubset(set(reader.fieldnames or ())):
            raise ValueError("canonical CSV is missing required input/risk fields")
        for row in reader:
            total += 1
            sample_id = row.get("sample_id", "")
            if not sample_id or sample_id in all_ids:
                raise ValueError("canonical CSV sample_id is empty or duplicated")
            all_ids.add(sample_id)
            stock_codes.add(row.get("stock_code", ""))
            if sample_id in pilot_ids:
                continue
            if not (row.get("model_text") or "").strip():
                continue
            rows.append(_safe_input(row))
    if total != POOL_ROWS:
        raise ValueError(f"expected {POOL_ROWS} canonical rows, found {total}")
    if len(all_ids) != POOL_ROWS or len(stock_codes) != 16:
        raise ValueError("canonical identity or stock count does not match contract")
    return rows, {
        "path": str(source),
        "sha256": observed_sha,
        "rows": total,
        "unique_sample_ids": len(all_ids),
        "stock_count": len(stock_codes),
        "stock_codes": sorted(stock_codes),
        "pilot_ids_excluded": len(pilot_ids),
    }


def select_shadow_rows(rows: Sequence[Mapping[str, Any]], *, quotas: Mapping[str, int] | None = None) -> list[dict[str, Any]]:
    """Select 32/31 rows per stock using the pinned hash ordering."""

    by_stock: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stock[str(row["stock_code"])].append(row)
    codes = sorted(by_stock)
    if len(codes) != 16:
        raise ValueError(f"expected 16 stock codes after pilot exclusion, found {len(codes)}")
    expected = {code: (32 if index < 4 else 31) for index, code in enumerate(codes)}
    if quotas is not None:
        expected = dict(quotas)
    selected_rows: list[dict[str, Any]] = []
    for code in codes:
        quota = int(expected[code])
        ordered = sorted(
            by_stock[code],
            key=lambda row: (_sha256_selection(str(row["sample_id"])), str(row["sample_id"])),
        )
        if len(ordered) < quota:
            raise ValueError(f"not enough eligible rows for {code}: {len(ordered)} < {quota}")
        selected_rows.extend(dict(row) for row in ordered[:quota])
    if quotas is None and len(selected_rows) != SHADOW_ROWS:
        raise ValueError(f"expected {SHADOW_ROWS} selected rows, found {len(selected_rows)}")
    if len({row["sample_id"] for row in selected_rows}) != len(selected_rows):
        raise ValueError("Shadow500 sample IDs are not unique")
    return selected_rows


def _normalised_group(value: Any) -> str:
    if value is None:
        return "(missing)"
    text = str(value).strip()
    return text if text else "(empty)"


def _risk_flagged(row: Mapping[str, Any]) -> bool:
    flags = _normalised_group(row.get("quality_flags"))
    eligibility = str(row.get("model_eligibility", "")).strip().upper()
    return flags not in {"(empty)", "[]", "NULL", "NONE"} or eligibility not in {"ELIGIBLE"}


def _short_or_missing_context(row: Mapping[str, Any]) -> bool:
    text = str(row.get("model_text", "")).strip()
    tier = str(row.get("text_quality_tier", "")).strip().upper()
    return len(text) <= 20 or tier in {"C_SHORT_TEXT", "D_EMPTY_OR_TEMPLATE"}


def confidence_from_prediction(row: Mapping[str, Any]) -> tuple[dict[str, float], float]:
    predictions = row.get("predictions")
    if not isinstance(predictions, Mapping):
        raise ValueError("prediction has no predictions object")
    maxima: dict[str, float] = {}
    for head in selected.SCALAR_HEADS:
        values = predictions[head].get("probabilities")
        if not isinstance(values, Mapping) or not values:
            raise ValueError(f"prediction has no probabilities for {head}")
        maxima[head] = float(max(float(value) for value in values.values()))
    score = float(np.mean(list(maxima.values())))
    if not np.isfinite(score):
        raise ValueError("six-head confidence is non-finite")
    return maxima, score


def unk_ratio(tokenizer: Any, record: Any, config: Mapping[str, Any]) -> float:
    ids = m1.build_input_ids(tokenizer, record, config)
    unknown = tokenizer.unk_token_id
    if not isinstance(unknown, int) or not ids:
        return 0.0
    return float(sum(value == unknown for value in ids) / len(ids))


def _group_summary(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_normalised_group(row.get(field))].append(row)
    result: dict[str, Any] = {}
    for name, group in sorted(groups.items()):
        scores = [float(row["six_head_mean_max_probability"]) for row in group]
        unks = [float(row["tokenizer_unk_ratio"]) for row in group]
        abstain = [bool(row["abstain"]) for row in group]
        result[name] = {
            "rows": len(group),
            "abstain_count": int(sum(abstain)),
            "abstention_rate": float(np.mean(abstain)),
            "coverage": float(1.0 - np.mean(abstain)),
            "mean_six_head_max_probability": float(np.mean(scores)),
            "mean_tokenizer_unk_ratio": float(np.mean(unks)),
        }
    return result


def confidence_quantiles(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    values = np.asarray([float(row["six_head_mean_max_probability"]) for row in rows], dtype=float)
    return {f"p{percentile}": float(np.percentile(values, percentile)) for percentile in (0, 5, 25, 50, 75, 95, 100)}


def choose_review_candidates(rows: Sequence[Mapping[str, Any]], *, limit: int = 50) -> list[dict[str, Any]]:
    """Select at most 50 candidates in the frozen bucket order."""

    buckets: list[tuple[str, list[Mapping[str, Any]], int, Any]] = [
        (
            "LOW_CONFIDENCE",
            [row for row in rows if bool(row["abstain"]) or float(row["six_head_mean_max_probability"]) < THRESHOLD],
            20,
            lambda row: (float(row["six_head_mean_max_probability"]), str(row["sample_id"])),
        ),
        (
            "HIGHEST_UNK_RATIO",
            list(rows),
            10,
            lambda row: (-float(row["tokenizer_unk_ratio"]), float(row["six_head_mean_max_probability"]), str(row["sample_id"])),
        ),
        (
            "RISK_FLAGGED_HIGH_CONFIDENCE",
            [row for row in rows if _risk_flagged(row) and not bool(row["abstain"])],
            10,
            lambda row: (-float(row["six_head_mean_max_probability"]), str(row["sample_id"])),
        ),
        (
            "SHORT_OR_MISSING_CONTEXT_RELEASED",
            [row for row in rows if _short_or_missing_context(row) and not bool(row["abstain"])],
            10,
            lambda row: (-float(row["six_head_mean_max_probability"]), str(row["sample_id"])),
        ),
    ]
    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name, candidates, quota, key in buckets:
        for row in sorted(candidates, key=key):
            sample_id = str(row["sample_id"])
            if sample_id in seen:
                continue
            copy = {
                field: row[field]
                for field in (*SAFE_INPUT_FIELDS, *RISK_FIELDS)
                if field in row
            }
            copy.update(
                {
                    "review_bucket": name,
                    "six_head_mean_max_probability": float(row["six_head_mean_max_probability"]),
                    "tokenizer_unk_ratio": float(row["tokenizer_unk_ratio"]),
                    "abstain": bool(row["abstain"]),
                }
            )
            chosen.append(copy)
            seen.add(sample_id)
            if len([item for item in chosen if item["review_bucket"] == name]) >= quota or len(chosen) >= limit:
                break
        if len(chosen) >= limit:
            break
    return chosen[:limit]


def _prediction_row(record: Any, prediction: Mapping[str, Any], tokenizer: Any, config: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    maxima, score = confidence_from_prediction(prediction)
    ratio = unk_ratio(tokenizer, record, config)
    row = {
        "sample_id": str(source["sample_id"]),
        "stock_code": str(source["stock_code"]),
        "stock_name": str(source["stock_name"]),
        "text_quality_tier": source.get("text_quality_tier", ""),
        "model_eligibility": source.get("model_eligibility", ""),
        "quality_flags": source.get("quality_flags", ""),
        "predictions": prediction.get("predictions"),
        "scalar_head_max_probability": maxima,
        "six_head_mean_max_probability": score,
        "tokenizer_unk_ratio": ratio,
        "threshold": THRESHOLD,
        "abstain": bool(score < THRESHOLD),
    }
    if not math.isfinite(ratio):
        raise ValueError("tokenizer UNK ratio is non-finite")
    return row


def run(
    *,
    pool_path: str | Path = DEFAULT_POOL,
    pilot_path: str | Path = DEFAULT_PILOT,
    artifact_dir: str | Path = DEFAULT_ARTIFACT,
    cache_dir: str | Path = DEFAULT_CACHE,
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing Shadow500 output: {output}")
    pilot_ids = read_pilot_ids(pilot_path)
    eligible, pool_info = read_canonical_pool(pool_path, pilot_ids)
    selected_rows = select_shadow_rows(eligible)
    if any(str(row["sample_id"]) in pilot_ids for row in selected_rows):
        raise RuntimeError("Shadow500 selection intersects semantic pilot IDs")
    # Write input evidence before model loading.  It contains only inputs and
    # the three explicit risk metadata fields; author/URL/label fields cannot
    # enter this projection.
    input_rows = [
        {field: row[field] for field in (*SAFE_INPUT_FIELDS, *RISK_FIELDS)}
        for row in selected_rows
    ]
    runtime = selected.UnifiedCPUInferenceRuntime.from_artifact(
        artifact_dir, config_path=config_path, cache_dir=cache_dir
    )
    records = [
        selected.InferenceRecord(
            sample_id=str(row["sample_id"]),
            stock_code=str(row["stock_code"]),
            stock_name=str(row["stock_name"]),
            model_text=str(row["model_text"]),
        )
        for row in selected_rows
    ]
    predictions = runtime.predict_records(records)
    if len(predictions) != SHADOW_ROWS:
        raise RuntimeError(f"expected {SHADOW_ROWS} predictions, found {len(predictions)}")
    prediction_rows = [
        _prediction_row(record, prediction, runtime.tokenizer, runtime.config, source)
        for record, prediction, source in zip(records, predictions, selected_rows, strict=True)
    ]
    candidates = choose_review_candidates(prediction_rows)
    summary = {
        "status": "M3_SHADOW500_COMPLETED_DIAGNOSTIC",
        "diagnostic_only": True,
        "selected_candidate": False,
        "production_output": False,
        "pool": pool_info,
        "pilot": {
            "path": str(Path(pilot_path).resolve()),
            "unique_sample_ids": len(pilot_ids),
            "rows_excluded": len(pilot_ids),
            "labels_read": False,
            "split_roles_read": False,
        },
        "selection": {
            "prefix": SELECTION_PREFIX,
            "stock_codes_sorted": sorted({str(row["stock_code"]) for row in selected_rows}),
            "quota_by_stock": {code: (32 if index < 4 else 31) for index, code in enumerate(sorted({str(row["stock_code"]) for row in selected_rows}))},
            "rows": len(selected_rows),
            "nonempty_model_text_only": True,
            "pilot_excluded": True,
        },
        "inference": {
            "rows": len(prediction_rows),
            "device": "cpu",
            "selected_artifact_content_address": runtime.manifest["content_address"],
            "model_id": runtime.snapshot_identity.get("model_id", "hfl/chinese-lert-small"),
            "revision": runtime.snapshot_identity.get("revision", "69e3e69ba258be5b301b26937e5b55a076c90460"),
            "training": False,
            "download": False,
        },
        "threshold": {
            "score": "six_head_mean_max_probability",
            "value": THRESHOLD,
            "rule": "abstain when score < threshold",
            "method": THRESHOLD_METHOD,
            "selected_from": "frozen M3-B Dev distribution; not recomputed on Shadow500",
        },
        "confidence_quantiles": confidence_quantiles(prediction_rows),
        "abstention": {
            "abstain_count": int(sum(bool(row["abstain"]) for row in prediction_rows)),
            "abstention_rate": float(np.mean([bool(row["abstain"]) for row in prediction_rows])),
            "coverage": float(np.mean([not bool(row["abstain"]) for row in prediction_rows])),
        },
        "groups": {field: _group_summary(prediction_rows, field) for field in ("stock_code", "text_quality_tier", "model_eligibility", "quality_flags")},
        "review_candidates": {
            "rows": len(candidates),
            "path": "review-candidates50.jsonl",
            "status": "REVIEW_CANDIDATES_ONLY_NOT_GOLD_OR_OOD",
            "bucket_order": ["LOW_CONFIDENCE", "HIGHEST_UNK_RATIO", "RISK_FLAGGED_HIGH_CONFIDENCE", "SHORT_OR_MISSING_CONTEXT_RELEASED"],
            "rules": {
                "low_confidence_quota": 20,
                "highest_unk_ratio_quota": 10,
                "risk_flagged_high_confidence_quota": 10,
                "short_or_missing_context_released_quota": 10,
                "short_definition": "model_text length <= 20 OR text_quality_tier C_SHORT_TEXT/D_EMPTY_OR_TEMPLATE",
                "risk_definition": "non-empty quality_flags or model_eligibility != ELIGIBLE",
                "deduplicate_by": "sample_id",
            },
        },
        "forbidden_inputs_not_read": ["labels", "split_roles", "Test", "Anchor", "Human100", "Gold", "OOD", "reference_predictions", "LLM", "cloud", "production_data"],
    }
    output.mkdir(parents=True, exist_ok=False)
    _write_jsonl(output / "shadow500-input.jsonl", input_rows)
    _write_jsonl(output / "shadow500-predictions.jsonl", prediction_rows)
    _write_jsonl(output / "review-candidates50.jsonl", candidates)
    _write_json(output / "shadow-summary.json", summary)
    manifest = m1._write_content_manifest(
        output,
        {
            "manifest_schema_version": "myresearcher.m3c-shadow500-diagnostic-manifest.v1",
            "status": summary["status"],
            "input_pool_sha256": POOL_SHA256,
            "pilot_sample_id_count": len(pilot_ids),
            "rows_selected": SHADOW_ROWS,
            "selected_artifact_content_address": runtime.manifest["content_address"],
            "threshold": THRESHOLD,
            "forbidden_inputs_not_read": summary["forbidden_inputs_not_read"],
        },
    )
    return {"status": summary["status"], "rows": SHADOW_ROWS, "pilot_excluded": len(pilot_ids), "abstention_rate": summary["abstention"]["abstention_rate"], "review_candidates": len(candidates), "output_dir": str(output), "content_address": manifest["content_address"]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(json.dumps(run(pool_path=args.pool, pilot_path=args.pilot, artifact_dir=args.artifact_dir, cache_dir=args.cache_dir, config_path=args.config, output_dir=args.output_dir), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
