#!/usr/bin/env python3
"""Feedback Loop V1 Phase A: DataClean pilot -> prediction snapshot -> labels.

The command is intentionally small and inference-only.  ``prepare`` reads the
300-row DataClean pilot, projects input-safe records, runs the already frozen
seven-head runtime, and freezes a deterministic 14-row Train/Check selection.
``dispatch`` sends only text and safe metadata to the existing labeling app;
model predictions never enter those task files.  ``pull`` converts app rows
into a revisable local snapshot, while ``status`` reports both remote batches.

No function in this module fits a model, reads a label split, or opens Test,
Anchor, Gold, OOD, or production data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT_PATH = Path("/Users/mac/Documents/trae_projects/MyResearcher/MyResearcher-DataClean/artifacts/round-003/pilot-300.jsonl")
TARGET_MAPPING_PATH = Path("/Users/mac/Documents/trae_projects/MyResearcher/MyResearcher-DataCollector/config/targets.short-term.json")
CANDIDATE_ARTIFACT = Path("/Users/mac/Documents/trae_projects/MyResearcher/model-artifacts/m2-selected-candidate-frozen-for-m3-20260904")
CACHE_DIR = REPO_ROOT / ".encoder-artifacts" / "hf-cache-lert-small"
RUN_DIR = REPO_ROOT / "runs" / "feedback-loop-v1-20260909"
LABELING_APP_ROOT = Path("/Users/mac/Documents/trae_projects/MyResearcher/MyResearcher-Labeling-app")
LABELING_TOOL = LABELING_APP_ROOT / "tools" / "gpt_tasks.py"
LABELING_CONFIG = LABELING_APP_ROOT / "data" / "config-gpt.json"
LABELING_APP_PYTHON = LABELING_APP_ROOT / ".venv" / "bin" / "python"
TRAIN_BATCH = "fl1-train-20260909"
CHECK_BATCH = "fl1-check-20260909"
HEADS = (
    "target_mode",
    "stance",
    "emotion_primary",
    "emotion_target",
    "action_tendency",
    "context_dependency",
    "reasoning_tags",
)
SCALAR_HEADS = HEADS[:-1]
SELECTION_VERSION = "feedback-loop-v1-selection-20260909-v1"
FORBIDDEN_INPUT_FIELDS = {
    "author_id", "author_name", "url", "labels", "label", "weights",
    "target_mode", "stance", "emotion_primary", "emotion_target",
    "action_tendency", "context_dependency", "reasoning_tags",
}
STOCK_CODE_RE = re.compile(r"^\d{6}$")


class FeedbackLoopError(ValueError):
    """A deterministic, user-actionable Phase A validation failure."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def content_address(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json({k: v for k, v in value.items() if k != "content_address"})).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FeedbackLoopError(f"JSONL missing: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FeedbackLoopError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise FeedbackLoopError(f"JSONL row is not an object at {path}:{line_number}")
            rows.append(row)
    return rows


def load_stock_mapping(path: Path = TARGET_MAPPING_PATH) -> dict[str, str]:
    """Read stock names from DataCollector's existing target mapping only."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FeedbackLoopError(f"target mapping unavailable: {path}") from exc
    names = raw.get("stock_names") if isinstance(raw, dict) else None
    if not isinstance(names, dict):
        raise FeedbackLoopError("target mapping must contain stock_names")
    result = {str(code): str(name) for code, name in names.items() if str(code) and isinstance(name, str) and name.strip()}
    if not result:
        raise FeedbackLoopError("target mapping contains no stock names")
    return result


def _scope_codes(row: Mapping[str, Any]) -> list[str]:
    lineage = row.get("collector_lineage")
    scopes = lineage.get("scopes") if isinstance(lineage, Mapping) else None
    if not isinstance(scopes, list) or not scopes:
        raise FeedbackLoopError(f"observation {row.get('observation_id')} has no collector scope")
    codes: set[str] = set()
    for scope in scopes:
        if not isinstance(scope, Mapping):
            continue
        key = scope.get("scope_key")
        if isinstance(key, str) and key.startswith("stock:"):
            codes.add(key.split(":", 1)[1])
        requested = scope.get("requested_bar_code")
        if requested is not None:
            codes.add(str(requested))
    if len(codes) != 1 or not STOCK_CODE_RE.fullmatch(next(iter(codes), "")):
        raise FeedbackLoopError(f"observation {row.get('observation_id')} has ambiguous stock scope: {sorted(codes)}")
    return sorted(codes)


def project_pilot_row(row: Mapping[str, Any], stock_mapping: Mapping[str, str]) -> dict[str, Any]:
    observation_id = row.get("observation_id")
    content = row.get("content")
    if not isinstance(observation_id, str) or not observation_id.strip() or not isinstance(content, str) or not content.strip():
        raise FeedbackLoopError("pilot rows require non-empty observation_id and content")
    if not isinstance(row.get("exact_content_group"), str) or not row["exact_content_group"]:
        raise FeedbackLoopError(f"observation {observation_id} lacks exact_content_group")
    reasons = row.get("sampling_reasons", [])
    if not isinstance(reasons, list) or not all(isinstance(item, str) for item in reasons):
        raise FeedbackLoopError(f"observation {observation_id} has invalid sampling_reasons")
    code = _scope_codes(row)[0]
    if code not in stock_mapping:
        raise FeedbackLoopError(f"stock {code} is absent from target mapping")
    sample_id = f"fl1-{observation_id.strip()}"
    # This projection deliberately omits author identity, URLs, labels, and
    # every other source field not required by the labeling workflow.
    return {
        "sample_id": sample_id,
        "observation_id": observation_id.strip(),
        "stock_code": code,
        "stock_name": stock_mapping[code],
        "model_text": content,
        "text": content,
        "exact_content_group": row["exact_content_group"],
        "exact_content_key": row.get("exact_content_key"),
        "exact_content_count": row.get("exact_content_count"),
        "sampling_reasons": sorted(set(reasons)),
        "text_length": row.get("text_length", len(content)),
    }


def load_pilot(path: Path = PILOT_PATH, stock_mapping: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    if len(rows) != 300:
        raise FeedbackLoopError(f"expected 300 pilot rows, got {len(rows)}")
    stock_mapping = load_stock_mapping() if stock_mapping is None else stock_mapping
    projected = [project_pilot_row(row, stock_mapping) for row in rows]
    ids = [row["sample_id"] for row in projected]
    if len(set(ids)) != len(ids):
        raise FeedbackLoopError("pilot observation_id/sample_id is not unique")
    return projected


def _length_bucket(value: Any) -> str:
    try:
        length = int(value)
    except (TypeError, ValueError):
        length = 0
    if length < 10:
        return "SHORT"
    if length <= 25:
        return "MEDIUM"
    return "LONG"


def _stable_hash(sample_id: str, purpose: str = SELECTION_VERSION) -> str:
    return hashlib.sha256(f"{purpose}|{sample_id}".encode("utf-8")).hexdigest()


def _snapshot_uncertainty(snapshot: Mapping[str, Any]) -> float:
    value = snapshot.get("six_head_mean_max_probability")
    if not isinstance(value, (int, float)):
        raise FeedbackLoopError(f"prediction snapshot lacks six-head confidence for {snapshot.get('sample_id')}")
    return 1.0 - float(value)


def select_samples(feed_rows: Sequence[Mapping[str, Any]], prediction_rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Freeze 7 uncertainty/boundary + 7 coverage rows, each group-distinct."""

    by_id = {str(row.get("sample_id")): row for row in prediction_rows}
    if len(by_id) != len(feed_rows):
        raise FeedbackLoopError("prediction snapshot does not cover the feed exactly")
    groups: set[str] = set()
    boundary: list[dict[str, Any]] = []
    ranked = sorted(feed_rows, key=lambda row: (-_snapshot_uncertainty(by_id[str(row["sample_id"])]), _stable_hash(str(row["sample_id"])), str(row["sample_id"])))
    for row in ranked:
        group = str(row["exact_content_group"])
        if group in groups:
            continue
        selected = dict(row)
        selected["selection_stratum"] = "MODEL_UNCERTAINTY_BOUNDARY"
        selected["selection_score"] = round(_snapshot_uncertainty(by_id[str(row["sample_id"])]), 12)
        boundary.append(selected)
        groups.add(group)
        if len(boundary) == 7:
            break
    if len(boundary) != 7:
        raise FeedbackLoopError("pilot has fewer than seven distinct exact-content groups for boundary stratum")

    covered_lengths = {_length_bucket(row.get("text_length")) for row in boundary}
    covered_reasons = {reason for row in boundary for reason in row.get("sampling_reasons", [])}
    coverage: list[dict[str, Any]] = []
    remaining = [row for row in feed_rows if str(row["exact_content_group"]) not in groups]
    while remaining and len(coverage) < 7:
        def score(row: Mapping[str, Any]) -> tuple[int, int, int, str, str]:
            length = _length_bucket(row.get("text_length"))
            reasons = set(row.get("sampling_reasons", []))
            new_reasons = len(reasons - covered_reasons)
            new_length = int(length not in covered_lengths)
            # Prefer novelty, then rows covering multiple declared reasons.
            return (new_reasons, new_length, len(reasons), _stable_hash(str(row["sample_id"])), str(row["sample_id"]))
        chosen = max(remaining, key=score)
        selected = dict(chosen)
        length = _length_bucket(chosen.get("text_length"))
        reasons = set(chosen.get("sampling_reasons", []))
        selected["selection_stratum"] = "LENGTH_AND_SAMPLING_REASON_COVERAGE"
        selected["selection_score"] = {"new_reasons": len(reasons - covered_reasons), "new_length_bucket": int(length not in covered_lengths), "length_bucket": length}
        coverage.append(selected)
        groups.add(str(chosen["exact_content_group"]))
        covered_lengths.add(length)
        covered_reasons.update(reasons)
        remaining = [row for row in remaining if str(row["exact_content_group"]) not in groups]
    if len(coverage) != 7:
        raise FeedbackLoopError("pilot has fewer than seven additional exact-content groups for coverage stratum")

    # Five + five in Train, two + two in Check keeps both declared strata in
    # each frozen split while guaranteeing text/group disjointness.
    selected = boundary[:5] + coverage[:5] + boundary[5:] + coverage[5:]
    train = [dict(row, frozen_role="HUMAN_TRAIN") for row in selected[:10]]
    check = [dict(row, frozen_role="HUMAN_CHECK") for row in selected[10:]]
    if set(row["exact_content_group"] for row in train) & set(row["exact_content_group"] for row in check):
        raise FeedbackLoopError("Train/Check exact-content groups overlap")
    return train, check


def _prediction_uncertainty(prediction: Mapping[str, Any]) -> tuple[float, dict[str, float]]:
    outputs = prediction.get("predictions")
    if not isinstance(outputs, Mapping):
        raise FeedbackLoopError("runtime prediction has no predictions object")
    maxima: dict[str, float] = {}
    for head in SCALAR_HEADS:
        probs = outputs.get(head, {}).get("probabilities") if isinstance(outputs.get(head), Mapping) else None
        if not isinstance(probs, Mapping) or not probs:
            raise FeedbackLoopError(f"runtime prediction missing probabilities for {head}")
        values = [float(value) for value in probs.values()]
        if not all(value == value and abs(value) != float("inf") for value in values):
            raise FeedbackLoopError(f"runtime prediction has non-finite probabilities for {head}")
        maxima[head] = max(values)
    mean_max = sum(maxima.values()) / len(maxima)
    reasoning_probs = outputs.get("reasoning_tags", {}).get("probabilities", {}) if isinstance(outputs.get("reasoning_tags"), Mapping) else {}
    reasoning_uncertainty = sum(1.0 - abs(2.0 * float(value) - 1.0) for value in reasoning_probs.values()) / len(reasoning_probs) if reasoning_probs else 0.0
    return 1.0 - mean_max, {"six_head_mean_max_probability": mean_max, "reasoning_mean_uncertainty": reasoning_uncertainty}


def _load_runtime():
    # Keep heavy Torch/Transformers imports behind prepare.  Unit tests can use
    # synthetic predictions without importing either package.
    if str(REPO_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "src"))
    from semantic_model.encoder_m2_selected_runtime import UnifiedCPUInferenceRuntime, InferenceRecord
    return UnifiedCPUInferenceRuntime, InferenceRecord


def prepare(*, pilot_path: Path = PILOT_PATH, output_dir: Path = RUN_DIR, artifact_dir: Path = CANDIDATE_ARTIFACT, cache_dir: Path = CACHE_DIR) -> dict[str, Any]:
    if output_dir.exists():
        raise FeedbackLoopError(f"refusing to overwrite existing Phase A run: {output_dir}")
    mapping = load_stock_mapping()
    feed = load_pilot(pilot_path, mapping)
    Runtime, InferenceRecord = _load_runtime()
    runtime = Runtime.from_artifact(artifact_dir, cache_dir=cache_dir)
    records = [InferenceRecord(sample_id=row["sample_id"], stock_code=row["stock_code"], stock_name=row["stock_name"], model_text=row["model_text"], metadata={"exact_content_group": row["exact_content_group"]}) for row in feed]
    predictions = runtime.predict_records(records)
    if len(predictions) != len(feed):
        raise FeedbackLoopError("runtime returned an unexpected prediction count")
    pred_by_id = {str(row["sample_id"]): row for row in predictions}
    snapshot_rows: list[dict[str, Any]] = []
    for row in feed:
        prediction = pred_by_id.get(row["sample_id"])
        if prediction is None:
            raise FeedbackLoopError(f"runtime omitted {row['sample_id']}")
        uncertainty, details = _prediction_uncertainty(prediction)
        snapshot_rows.append({**row, "predictions": prediction["predictions"], "uncertainty_score": uncertainty, **details})
    train, check = select_samples(feed, snapshot_rows)
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_jsonl(output_dir / "feed-input.jsonl", feed)
    _write_jsonl(output_dir / "prediction-snapshot.jsonl", snapshot_rows)
    _write_jsonl(output_dir / "selection-train.jsonl", train)
    _write_jsonl(output_dir / "selection-check.jsonl", check)
    # The app parser accepts only sample_id/text/title/metadata.  Predictions
    # are intentionally absent from these two files.
    _write_jsonl(output_dir / "labeling-train.jsonl", ({"sample_id": row["sample_id"], "text": row["model_text"], "title": row["stock_name"], "metadata": {"observation_id": row["observation_id"], "stock_code": row["stock_code"], "stock_name": row["stock_name"], "exact_content_group": row["exact_content_group"], "sampling_reasons": row["sampling_reasons"], "text_length": row["text_length"], "selection_stratum": row["selection_stratum"], "frozen_role": row["frozen_role"]}} for row in train))
    _write_jsonl(output_dir / "labeling-check.jsonl", ({"sample_id": row["sample_id"], "text": row["model_text"], "title": row["stock_name"], "metadata": {"observation_id": row["observation_id"], "stock_code": row["stock_code"], "stock_name": row["stock_name"], "exact_content_group": row["exact_content_group"], "sampling_reasons": row["sampling_reasons"], "text_length": row["text_length"], "selection_stratum": row["selection_stratum"], "frozen_role": row["frozen_role"]}} for row in check))
    manifest: dict[str, Any] = {
        "schema_version": "myresearcher.feedback-loop-v1-phase-a.v1",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": "PREPARED_BEFORE_HUMAN_LABELS",
        "pilot": {"path": str(pilot_path), "sha256": sha256_file(pilot_path), "rows": len(feed), "source": "DataClean_round_003_pilot_300"},
        "target_mapping": {"path": str(TARGET_MAPPING_PATH), "sha256": sha256_file(TARGET_MAPPING_PATH), "stock_codes": sorted({row["stock_code"] for row in feed})},
        "candidate": {"artifact_path": str(artifact_dir), "content_address": "8f73bc437602ce25d6e8ba0c7fd787676a4b566077b1765f3510739ae85928cf"},
        "prediction_snapshot": {"path": "prediction-snapshot.jsonl", "sent_to_labeling_app": False, "rows": len(snapshot_rows), "runtime_device": "cpu", "candidate_runtime": "UnifiedCPUInferenceRuntime"},
        "selection": {"version": SELECTION_VERSION, "rule": "7 model-uncertainty boundary + 7 length/sampling_reasons coverage; distinct exact_content_group; 5+5 Train then 2+2 Check", "train_rows": 10, "check_rows": 4, "train_check_text_overlap": False, "sample_ids": [row["sample_id"] for row in train + check]},
        "labeling": {"heads": list(HEADS), "train_batch": TRAIN_BATCH, "train_tasks": 70, "check_batch": CHECK_BATCH, "check_tasks": 28, "predictions_excluded_from_task_payload": True},
        "forbidden": ["Test", "Anchor", "Gold", "OOD", "LLM", "cloud", "production_inference", "training"],
        "file_sha256": {},
    }
    for filename in ("feed-input.jsonl", "prediction-snapshot.jsonl", "selection-train.jsonl", "selection-check.jsonl", "labeling-train.jsonl", "labeling-check.jsonl"):
        manifest["file_sha256"][filename] = sha256_file(output_dir / filename)
    manifest["content_address"] = content_address(manifest)
    _write_json(output_dir / "prepare-manifest.json", manifest)
    return manifest


def _app_python() -> str:
    return str(LABELING_APP_PYTHON if LABELING_APP_PYTHON.is_file() else sys.executable)


def _run_app(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    command = [_app_python(), str(LABELING_TOOL), *args, "--config", str(LABELING_CONFIG)]
    return subprocess.run(command, cwd=str(LABELING_APP_ROOT), text=True, capture_output=True, check=False)


def dispatch(*, run_dir: Path = RUN_DIR) -> dict[str, Any]:
    manifest = json.loads((run_dir / "prepare-manifest.json").read_text(encoding="utf-8"))
    heads = ",".join(HEADS)
    results = []
    for batch, filename, samples, tasks in ((TRAIN_BATCH, "labeling-train.jsonl", 10, 70), (CHECK_BATCH, "labeling-check.jsonl", 4, 28)):
        # A second invocation must not create duplicate assignments.  Existing
        # batches are accepted only when their immutable task count is exact.
        existing = _run_app(["status", "--batch", batch])
        if existing.returncode == 0:
            try:
                existing_status = json.loads(existing.stdout)
            except json.JSONDecodeError as exc:
                raise FeedbackLoopError(f"existing status for {batch} is invalid JSON") from exc
            if int(existing_status.get("total", -1)) != tasks:
                raise FeedbackLoopError(f"existing batch {batch} has unexpected task count")
            results.append({"batch": batch, "samples": samples, "heads": len(HEADS), "tasks": tasks, "exit": 0, "already_present": True})
            continue
        completed = _run_app(["add", "--batch", batch, "--file", str(run_dir / filename), "--heads", heads])
        if completed.returncode != 0:
            raise FeedbackLoopError(f"labeling dispatch failed for {batch}: exit={completed.returncode}")
        results.append({"batch": batch, "samples": samples, "heads": len(HEADS), "tasks": tasks, "exit": completed.returncode})
    payload = {"status": "DISPATCHED", "batches": results, "prediction_payload_sent": False, "prepare_content_address": manifest.get("content_address")}
    _write_json(run_dir / "dispatch.json", payload)
    return payload


def status(*, run_dir: Path = RUN_DIR) -> dict[str, Any]:
    results = []
    for batch in (TRAIN_BATCH, CHECK_BATCH):
        completed = _run_app(["status", "--batch", batch])
        if completed.returncode != 0:
            raise FeedbackLoopError(f"labeling status failed for {batch}: exit={completed.returncode}")
        try:
            value = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise FeedbackLoopError(f"labeling status returned invalid JSON for {batch}") from exc
        results.append(value)
    payload = {"status": "VISIBLE", "batches": results}
    _write_json(run_dir / "labeling-status.json", payload)
    return payload


def merge_pull_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Merge latest final answer per sample/head while preserving revisions."""

    grouped: dict[tuple[str, str], list[tuple[int, Mapping[str, Any]]]] = {}
    for index, row in enumerate(rows):
        sample_id, head = row.get("sample_id"), row.get("head")
        if not isinstance(sample_id, str) or not sample_id or head not in HEADS:
            raise FeedbackLoopError("pull rows require a valid sample_id and schema head")
        grouped.setdefault((sample_id, str(head)), []).append((index, row))
    merged: list[dict[str, Any]] = []
    for (sample_id, head), candidates in sorted(grouped.items(), key=lambda item: (item[0][0], HEADS.index(item[0][1]))):
        # Prefer an explicit final answer over a later in-progress heartbeat;
        # if no final exists yet, retain the newest provisional disposition so
        # the local snapshot remains useful for resuming review.
        finals = [item for item in candidates if item[1].get("is_final")]
        source_index, chosen = max(finals or candidates, key=lambda item: (str(item[1].get("updated_at") or ""), item[0]))
        del source_index
        merged.append({
            "sample_id": sample_id,
            "head": head,
            "answer": chosen.get("answer"),
            "disposition": chosen.get("disposition"),
            "updated_at": chosen.get("updated_at"),
            "batch": chosen.get("batch"),
            "schema_version": chosen.get("schema_version"),
            "is_final": bool(chosen.get("is_final")),
            "snapshot_role": "HUMAN_FEED_REVISION",
        })
    return merged


def pull(*, input_jsonl: Path | None = None, batches: Sequence[str] = (), output: Path) -> dict[str, Any]:
    temporary_paths: list[Path] = []
    try:
        if input_jsonl is not None:
            rows = _read_jsonl(input_jsonl)
        else:
            if not batches:
                batches = (TRAIN_BATCH, CHECK_BATCH)
            rows = []
            for batch in batches:
                handle = tempfile.NamedTemporaryFile(prefix="fl1-pull-", suffix=".jsonl", delete=False)
                handle.close()
                temp = Path(handle.name)
                temporary_paths.append(temp)
                completed = _run_app(["pull", "--format", "jsonl", "--batch", batch, "--out", str(temp)])
                if completed.returncode != 0:
                    raise FeedbackLoopError(f"labeling pull failed for {batch}: exit={completed.returncode}")
                rows.extend(_read_jsonl(temp))
        merged = merge_pull_rows(rows)
        _write_jsonl(output, merged)
        return {"status": "PULLED_REVISABLE_SNAPSHOT", "input_rows": len(rows), "merged_rows": len(merged), "output": str(output), "training_started": False}
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_prepare = sub.add_parser("prepare")
    p_prepare.add_argument("--pilot", type=Path, default=PILOT_PATH)
    p_prepare.add_argument("--output-dir", type=Path, default=RUN_DIR)
    p_prepare.add_argument("--artifact-dir", type=Path, default=CANDIDATE_ARTIFACT)
    p_prepare.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    p_dispatch = sub.add_parser("dispatch")
    p_dispatch.add_argument("--run-dir", type=Path, default=RUN_DIR)
    p_status = sub.add_parser("status")
    p_status.add_argument("--run-dir", type=Path, default=RUN_DIR)
    p_pull = sub.add_parser("pull")
    p_pull.add_argument("--input-jsonl", type=Path, default=None, help="synthetic/local app export; skips remote pull")
    p_pull.add_argument("--batch", action="append", default=[])
    p_pull.add_argument("--output-jsonl", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(pilot_path=args.pilot, output_dir=args.output_dir, artifact_dir=args.artifact_dir, cache_dir=args.cache_dir)
        elif args.command == "dispatch":
            result = dispatch(run_dir=args.run_dir)
        elif args.command == "status":
            result = status(run_dir=args.run_dir)
        else:
            result = pull(input_jsonl=args.input_jsonl, batches=args.batch, output=args.output_jsonl)
    except (FeedbackLoopError, OSError, RuntimeError) as exc:
        print(f"[feedback-loop-v1] FAIL-CLOSED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
