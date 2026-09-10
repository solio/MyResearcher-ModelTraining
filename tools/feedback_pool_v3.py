#!/usr/bin/env python3
"""Prepare the one-time 100-Train/20-semantic-Check feedback pool.

This module only consumes the frozen 300-row V1 input and prediction snapshot.
It never imports the model runtime and never trains, calibrates, or reads any
label/evaluation split.  V1/V2's 28 frozen sample identities are excluded,
then 80 Human Train and 14 Human Check rows are selected with deterministic
seven-head coverage and uncertainty rules.  Labeling payloads deliberately
contain text and safe metadata only; model predictions remain in the local
pre-label snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tools.feedback_loop_v1 import (
    FeedbackLoopError,
    HEADS,
    REPO_ROOT,
    SCALAR_HEADS,
    _prediction_uncertainty,
    _read_jsonl,
    _stable_hash,
    _write_json,
    _write_jsonl,
    is_low_information_input,
    sha256_file,
)


SOURCE_RUN_DIR = REPO_ROOT / "runs" / "feedback-loop-v1-20260909"
OUTPUT_RUN_DIR = REPO_ROOT / "runs" / "feedback-loop-v3-20260910"
TRAIN_BATCH = "fl3-train-100-target"
CHECK_BATCH = "fl3-check-20-semantic-target"
TRAIN_COUNT = 80
CHECK_COUNT = 14
POOL_VERSION = "feedback-loop-v3-seven-head-pool-20260910-v1"
EXPECTED_SOURCE_ROWS = 300

# The caps are selection balancing constraints, not label assumptions.  They
# prevent a single predicted class from consuming the pool while leaving
# enough candidates for every class in this snapshot.  Explicit BEAR and
# SELF_CONTAINED caps address the two known V1/V2 skews.
STANCE_BEAR_CAP = 28
CONTEXT_SELF_CAP = 56


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


def _safe_text(row: Mapping[str, Any]) -> str:
    text = row.get("model_text", row.get("text"))
    if not isinstance(text, str) or not text.strip():
        raise FeedbackLoopError(f"incomplete input text for {row.get('sample_id')}")
    if is_low_information_input(text):
        raise FeedbackLoopError(f"low-information input reached eligible set: {row.get('sample_id')}")
    return text


def is_known_incomplete(row: Mapping[str, Any]) -> bool:
    """Return true for structurally incomplete or non-semantic source rows."""

    text = row.get("model_text", row.get("text"))
    if not isinstance(text, str) or not text.strip() or is_low_information_input(text):
        return True
    stripped = text.strip()
    # The pilot contains a URL-only capture and a few rows truncated at a
    # separator.  Neither can support a complete semantic judgement.  This is
    # intentionally narrower than the retired short-text rule: meaningful
    # terse inputs (``抄底``, ``看好`` and similar) remain eligible.
    if re.match(r"^(?:https?://|www\.)", stripped, flags=re.IGNORECASE):
        return True
    if stripped.endswith(("，", ",", "：", ":", "；", ";")):
        return True
    if not isinstance(row.get("sample_id"), str) or not row["sample_id"].strip():
        return True
    if not isinstance(row.get("exact_content_group"), str) or not row["exact_content_group"].strip():
        return True
    if not isinstance(row.get("stock_code"), str) or not row["stock_code"].strip():
        return True
    if not isinstance(row.get("stock_name"), str) or not row["stock_name"].strip():
        return True
    try:
        length = int(row.get("text_length", len(text)))
    except (TypeError, ValueError):
        return True
    return length <= 0


def _source_selection_ids(source_dir: Path) -> set[str]:
    """Read only frozen V1/V2 selection identities; no labels are opened."""

    result: set[str] = set()
    for path in (
        source_dir / "selection-train.jsonl",
        source_dir / "selection-check.jsonl",
        REPO_ROOT / "runs" / "feedback-loop-v2-20260909" / "selection-train.jsonl",
        REPO_ROOT / "runs" / "feedback-loop-v2-20260909" / "selection-check.jsonl",
    ):
        if not path.is_file():
            continue
        rows = _read_jsonl(path)
        ids = [row.get("sample_id") for row in rows]
        if any(not isinstance(value, str) or not value for value in ids):
            raise FeedbackLoopError(f"selection file contains invalid sample_id: {path}")
        result.update(ids)
    return result


def _refresh_snapshot(feed_rows: Sequence[Mapping[str, Any]], prediction_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(row.get("sample_id")): row for row in prediction_rows}
    feed_ids = [str(row.get("sample_id")) for row in feed_rows]
    if len(feed_rows) != EXPECTED_SOURCE_ROWS or len(prediction_rows) != EXPECTED_SOURCE_ROWS:
        raise FeedbackLoopError("V3 source must contain exactly the original 300 feed and prediction rows")
    if len(set(feed_ids)) != len(feed_ids) or set(feed_ids) != set(by_id):
        raise FeedbackLoopError("prediction snapshot does not cover the feed exactly")
    refreshed: list[dict[str, Any]] = []
    for row in feed_rows:
        sample_id = str(row["sample_id"])
        prediction = by_id[sample_id]
        wrapped = prediction if isinstance(prediction.get("predictions"), Mapping) else {"predictions": prediction.get("predictions")}
        uncertainty, details = _prediction_uncertainty(wrapped)
        outputs = wrapped.get("predictions")
        if not isinstance(outputs, Mapping):
            raise FeedbackLoopError(f"missing predictions for {sample_id}")
        refreshed.append({**row, "predictions": outputs, "uncertainty_score": uncertainty, **details})
    return refreshed


def _feature_tokens(row: Mapping[str, Any]) -> set[str]:
    outputs = row.get("predictions")
    if not isinstance(outputs, Mapping):
        raise FeedbackLoopError(f"selection row has no predictions: {row.get('sample_id')}")
    tokens = {f"length:{_length_bucket(row.get('text_length'))}"}
    tokens.update(f"reason:{value}" for value in row.get("sampling_reasons", []) if isinstance(value, str))
    for head in SCALAR_HEADS:
        output = outputs.get(head)
        if not isinstance(output, Mapping) or not isinstance(output.get("prediction"), str):
            raise FeedbackLoopError(f"missing scalar prediction for {head}/{row.get('sample_id')}")
        tokens.add(f"{head}:{output['prediction']}")
    reasoning = outputs.get("reasoning_tags")
    if not isinstance(reasoning, Mapping):
        raise FeedbackLoopError(f"missing reasoning prediction for {row.get('sample_id')}")
    labels = reasoning.get("predicted_labels", reasoning.get("prediction", []))
    if not isinstance(labels, list):
        raise FeedbackLoopError(f"invalid reasoning labels for {row.get('sample_id')}")
    if labels:
        tokens.update(f"reasoning:{label}" for label in labels if isinstance(label, str))
    else:
        tokens.add("reasoning:__none__")
    return tokens


def _scalar_classes(row: Mapping[str, Any]) -> dict[str, str]:
    outputs = row["predictions"]
    return {head: str(outputs[head]["prediction"]) for head in SCALAR_HEADS}


def _dominant_allowed(row: Mapping[str, Any], counts: Mapping[str, Counter[str]], selected_total: int) -> bool:
    classes = _scalar_classes(row)
    stance = classes["stance"]
    context = classes["context_dependency"]
    if stance == "BEAR" and counts["stance"][stance] >= STANCE_BEAR_CAP:
        return False
    if context == "SELF_CONTAINED" and counts["context_dependency"][context] >= CONTEXT_SELF_CAP:
        return False
    del selected_total
    return True


def _length_targets(count: int) -> dict[str, int]:
    if count == CHECK_COUNT:
        return {"SHORT": 4, "MEDIUM": 6, "LONG": 4}
    # Deliberately retain meaningful short text while preserving medium/long
    # context coverage in the larger Train pool.
    return {"SHORT": 24, "MEDIUM": 32, "LONG": 24}


def _select_role(
    candidates: Sequence[Mapping[str, Any]],
    count: int,
    role: str,
    *,
    selected_groups: set[str],
    selected_ids: set[str],
    scalar_counts: dict[str, Counter[str]],
    seen_tokens: set[str],
) -> list[dict[str, Any]]:
    targets = _length_targets(count)
    length_counts: Counter[str] = Counter()
    chosen: list[dict[str, Any]] = []
    boundary_target = count // 2
    phases = (("MODEL_UNCERTAINTY_BOUNDARY", boundary_target), ("SEVEN_HEAD_COVERAGE", count - boundary_target))

    for phase, _ in phases:
        while len(chosen) < (boundary_target if phase == "MODEL_UNCERTAINTY_BOUNDARY" else count):
            available = [
                row for row in candidates
                if str(row["sample_id"]) not in selected_ids
                and str(row["exact_content_group"]) not in selected_groups
                and _dominant_allowed(row, scalar_counts, len(chosen))
            ]
            if not available:
                raise FeedbackLoopError(f"unable to select {count} distinct eligible {role} rows")
            remaining_slots = count - len(chosen)
            deficits = {bucket: max(0, target - length_counts[bucket]) for bucket, target in targets.items()}
            # Reserve buckets only when every remaining slot is needed to meet
            # the declared quotas.  Otherwise the score below gives deficient
            # buckets a bonus without accidentally filtering all candidates.
            if sum(deficits.values()) >= remaining_slots:
                available = [row for row in available if deficits.get(_length_bucket(row.get("text_length")), 0) > 0]
            if not available:
                # A caller may provide a synthetic or unusually skewed source
                # where the exact length quota cannot be met.  Keep selection
                # fail-closed on identity/capacity while allowing the report to
                # expose the observed bucket mix rather than raising an opaque
                # ``max()`` error.
                available = [
                    row for row in candidates
                    if str(row["sample_id"]) not in selected_ids
                    and str(row["exact_content_group"]) not in selected_groups
                    and _dominant_allowed(row, scalar_counts, len(chosen))
                ]
            if not available:
                raise FeedbackLoopError(f"unable to select {count} distinct eligible {role} rows")

            def score(row: Mapping[str, Any]) -> tuple[float, str, str]:
                tokens = _feature_tokens(row)
                unseen = tokens - seen_tokens
                scalar_novelty = sum(token.startswith(tuple(f"{head}:" for head in SCALAR_HEADS)) for token in unseen)
                reasoning_novelty = sum(token.startswith("reasoning:") for token in unseen)
                length = _length_bucket(row.get("text_length"))
                deficit_bonus = max(0, targets[length] - length_counts[length])
                uncertainty = float(row["aggregate_uncertainty"])
                # Rarity rewards low-frequency predicted classes without using
                # any human labels.
                rarity = sum(1.0 / max(1, sum(1 for candidate in candidates if _scalar_classes(candidate)[head] == value)) for head, value in _scalar_classes(row).items())
                if phase == "MODEL_UNCERTAINTY_BOUNDARY":
                    value = 16.0 * uncertainty + 2.0 * len(unseen) + 1.5 * deficit_bonus + rarity
                else:
                    value = 12.0 * scalar_novelty + 4.0 * reasoning_novelty + 2.0 * len(unseen) + 1.5 * deficit_bonus + 2.0 * uncertainty + 3.0 * rarity
                return (value, _stable_hash(str(row["sample_id"]), f"{POOL_VERSION}|{role}|{phase}"), str(row["sample_id"]))

            row = max(available, key=score)
            selected = dict(row)
            selected["frozen_role"] = role
            selected["selection_stratum"] = phase
            selected["selection_score"] = round(score(row)[0], 12)
            selected["length_bucket"] = _length_bucket(row.get("text_length"))
            selected["predicted_classes"] = _scalar_classes(row)
            selected["predicted_reasoning_labels"] = list(row["predictions"]["reasoning_tags"].get("predicted_labels", row["predictions"]["reasoning_tags"].get("prediction", [])))
            chosen.append(selected)
            selected_ids.add(str(row["sample_id"]))
            selected_groups.add(str(row["exact_content_group"]))
            length_counts[selected["length_bucket"]] += 1
            for head, value in selected["predicted_classes"].items():
                scalar_counts[head][value] += 1
            seen_tokens.update(_feature_tokens(row))
    if len(chosen) != count:
        raise FeedbackLoopError(f"selected {len(chosen)} {role} rows, expected {count}")
    return chosen


def _coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scalar: dict[str, dict[str, int]] = {}
    for head in SCALAR_HEADS:
        scalar[head] = dict(sorted(Counter(str(row["predictions"][head]["prediction"]) for row in rows).items()))
    reasoning = Counter(label for row in rows for label in row["predictions"]["reasoning_tags"].get("predicted_labels", row["predictions"]["reasoning_tags"].get("prediction", [])))
    return {
        "rows": len(rows),
        "sample_ids": [str(row["sample_id"]) for row in rows],
        "exact_content_groups": len({str(row["exact_content_group"]) for row in rows}),
        "length_buckets": dict(sorted(Counter(str(row["length_bucket"]) for row in rows).items())),
        "sampling_reasons": dict(sorted(Counter(reason for row in rows for reason in row.get("sampling_reasons", [])).items())),
        "scalar_predicted_classes": scalar,
        "reasoning_predicted_labels": dict(sorted(reasoning.items())),
        "stance_bear": scalar.get("stance", {}).get("BEAR", 0),
        "context_self_contained": scalar.get("context_dependency", {}).get("SELF_CONTAINED", 0),
    }


def build_labeling_payload(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return app-safe rows; no prediction, probability, label, or weight fields."""

    payload: list[dict[str, Any]] = []
    for row in rows:
        payload.append({
            "sample_id": str(row["sample_id"]),
            "text": _safe_text(row),
            "title": str(row["stock_name"]),
            "metadata": {
                "observation_id": str(row["observation_id"]),
                "stock_code": str(row["stock_code"]),
                "stock_name": str(row["stock_name"]),
                "exact_content_group": str(row["exact_content_group"]),
                "sampling_reasons": sorted(set(str(v) for v in row.get("sampling_reasons", []))),
                "text_length": int(row.get("text_length", len(row["model_text"]))),
                "length_bucket": str(row["length_bucket"]),
                "frozen_role": str(row["frozen_role"]),
            },
        })
    return payload


def select_pool(
    feed_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    *,
    exclude_sample_ids: Iterable[str] = (),
    train_count: int = TRAIN_COUNT,
    check_count: int = CHECK_COUNT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Select disjoint role pools from an already-saved seven-head snapshot."""

    if train_count != TRAIN_COUNT or check_count != CHECK_COUNT:
        raise FeedbackLoopError("V3 pool sizes are frozen at 80 Train and 14 Check")
    snapshot = _refresh_snapshot(feed_rows, prediction_rows)
    excluded = {str(value) for value in exclude_sample_ids}
    eligible: list[dict[str, Any]] = []
    filter_counts = Counter()
    seen_ids: set[str] = set()
    for row in snapshot:
        sample_id = str(row["sample_id"])
        if sample_id in seen_ids:
            raise FeedbackLoopError(f"duplicate feed sample_id: {sample_id}")
        seen_ids.add(sample_id)
        if sample_id in excluded:
            filter_counts["prior_v1_v2_sample_ids"] += 1
            continue
        if is_low_information_input(row.get("model_text", row.get("text"))):
            filter_counts["low_information"] += 1
            continue
        if is_known_incomplete(row):
            filter_counts["known_incomplete"] += 1
            continue
        eligible.append(row)
    if len({str(row["exact_content_group"]) for row in eligible}) < train_count + check_count:
        raise FeedbackLoopError("fewer than 94 distinct eligible exact-content groups remain")

    selected_groups: set[str] = set()
    selected_ids: set[str] = set()
    scalar_counts = {head: Counter() for head in SCALAR_HEADS}
    seen_tokens: set[str] = set()
    # Check is reserved first so its small semantic audit has guaranteed
    # length/coverage representation; Train then fills the remaining pool.
    check = _select_role(eligible, check_count, "HUMAN_CHECK", selected_groups=selected_groups, selected_ids=selected_ids, scalar_counts=scalar_counts, seen_tokens=seen_tokens)
    train = _select_role(eligible, train_count, "HUMAN_TRAIN", selected_groups=selected_groups, selected_ids=selected_ids, scalar_counts=scalar_counts, seen_tokens=seen_tokens)
    train_ids = {str(row["sample_id"]) for row in train}
    check_ids = {str(row["sample_id"]) for row in check}
    if train_ids & check_ids or {str(row["exact_content_group"]) for row in train} & {str(row["exact_content_group"]) for row in check}:
        raise FeedbackLoopError("Train and Check pools overlap")
    combined = train + check
    summary = {
        "selection_version": POOL_VERSION,
        "algorithm": "deterministic two-phase seven-head uncertainty/coverage greedy; Check reserved before Train",
        "filter_counts": dict(sorted(filter_counts.items())),
        "excluded_sample_ids": sorted(excluded),
        "train": _coverage(train),
        "check": _coverage(check),
        "combined": _coverage(combined),
        "train_check_sample_id_disjoint": True,
        "train_check_exact_content_group_disjoint": True,
        "model_predictions_used_for_selection": True,
        "human_labels_used_for_selection": False,
    }
    return train, check, summary


def _app_python() -> str:
    app_root = Path("/Users/mac/Documents/trae_projects/MyResearcher/MyResearcher-Labeling-app")
    candidate = app_root / ".venv" / "bin" / "python"
    return str(candidate if candidate.is_file() else sys.executable)


def _run_app(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    app_root = Path("/Users/mac/Documents/trae_projects/MyResearcher/MyResearcher-Labeling-app")
    tool = app_root / "tools" / "gpt_tasks.py"
    config = app_root / "data" / "config-gpt.json"
    command = [_app_python(), str(tool), *args, "--config", str(config)]
    return subprocess.run(command, cwd=str(app_root), text=True, capture_output=True, check=False)


def _parse_status(completed: subprocess.CompletedProcess[str], batch: str) -> dict[str, Any] | None:
    if completed.returncode != 0:
        return None
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise FeedbackLoopError(f"invalid status JSON for {batch}") from exc
    return value if isinstance(value, dict) else None


def prepare(*, source_run_dir: Path = SOURCE_RUN_DIR, output_dir: Path = OUTPUT_RUN_DIR, train_batch: str = TRAIN_BATCH, check_batch: str = CHECK_BATCH) -> dict[str, Any]:
    if output_dir.exists():
        raise FeedbackLoopError(f"refusing to overwrite existing V3 run: {output_dir}")
    feed = _read_jsonl(source_run_dir / "feed-input.jsonl")
    predictions = _read_jsonl(source_run_dir / "prediction-snapshot.jsonl")
    excluded = _source_selection_ids(source_run_dir)
    train, check, summary = select_pool(feed, predictions, exclude_sample_ids=excluded)
    output_dir.mkdir(parents=True, exist_ok=False)
    snapshot = _refresh_snapshot(feed, predictions)
    _write_jsonl(output_dir / "feed-input.jsonl", feed)
    _write_jsonl(output_dir / "prediction-snapshot.jsonl", snapshot)
    _write_jsonl(output_dir / "selection-train.jsonl", train)
    _write_jsonl(output_dir / "selection-check.jsonl", check)
    _write_jsonl(output_dir / "labeling-train.jsonl", build_labeling_payload(train))
    _write_jsonl(output_dir / "labeling-check.jsonl", build_labeling_payload(check))
    manifest: dict[str, Any] = {
        "schema_version": "myresearcher.feedback-loop-v3-pool.v1",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": "PREPARED_BEFORE_HUMAN_LABELS",
        "source": {
            "run_dir": str(source_run_dir),
            "feed_file": "feed-input.jsonl",
            "prediction_snapshot_file": "prediction-snapshot.jsonl",
            "rows": len(feed),
            "predictions_reused_without_model_inference": True,
            "feed_sha256": sha256_file(source_run_dir / "feed-input.jsonl"),
            "prediction_snapshot_sha256": sha256_file(source_run_dir / "prediction-snapshot.jsonl"),
        },
        "selection": summary,
        "labeling": {
            "heads": list(HEADS),
            "train_batch": train_batch,
            "train_samples": len(train),
            "train_tasks": len(train) * len(HEADS),
            "check_batch": check_batch,
            "check_samples": len(check),
            "check_tasks": len(check) * len(HEADS),
            "predictions_in_labeling_payload": False,
        },
        "forbidden": ["adapter", "training", "Test", "Anchor", "Gold", "OOD", "Human48", "Human100", "LLM", "production_inference"],
        "file_sha256": {},
    }
    for filename in ("feed-input.jsonl", "prediction-snapshot.jsonl", "selection-train.jsonl", "selection-check.jsonl", "labeling-train.jsonl", "labeling-check.jsonl"):
        manifest["file_sha256"][filename] = sha256_file(output_dir / filename)
    manifest["content_address"] = hashlib.sha256(json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    _write_json(output_dir / "prepare-manifest.json", manifest)
    _write_json(output_dir / "selection-summary.json", summary)
    return manifest


def dispatch(*, run_dir: Path = OUTPUT_RUN_DIR) -> dict[str, Any]:
    manifest = json.loads((run_dir / "prepare-manifest.json").read_text(encoding="utf-8"))
    labeling = manifest["labeling"]
    heads = ",".join(HEADS)
    results: list[dict[str, Any]] = []
    specs = ((labeling["train_batch"], "labeling-train.jsonl", labeling["train_tasks"], labeling["train_samples"]), (labeling["check_batch"], "labeling-check.jsonl", labeling["check_tasks"], labeling["check_samples"]))
    for batch, filename, expected_tasks, expected_samples in specs:
        existing = _parse_status(_run_app(["status", "--batch", batch]), batch)
        if existing is not None:
            if int(existing.get("total", -1)) != int(expected_tasks):
                raise FeedbackLoopError(f"existing {batch} has unexpected task count")
            results.append({"batch": batch, "samples": expected_samples, "heads": len(HEADS), "tasks": expected_tasks, "already_present": True})
            continue
        completed = _run_app(["add", "--batch", batch, "--file", str(run_dir / filename), "--heads", heads])
        if completed.returncode != 0:
            raise FeedbackLoopError(f"dispatch failed for {batch}: {completed.stderr.strip()}")
        results.append({"batch": batch, "samples": expected_samples, "heads": len(HEADS), "tasks": expected_tasks, "already_present": False})
    payload = {"status": "DISPATCHED", "batches": results, "prediction_payload_sent": False, "prepare_content_address": manifest.get("content_address")}
    _write_json(run_dir / "dispatch.json", payload)
    return payload


def status(*, run_dir: Path = OUTPUT_RUN_DIR) -> dict[str, Any]:
    manifest = json.loads((run_dir / "prepare-manifest.json").read_text(encoding="utf-8"))
    values = []
    for batch in (manifest["labeling"]["train_batch"], manifest["labeling"]["check_batch"]):
        completed = _run_app(["status", "--batch", batch])
        if completed.returncode != 0:
            raise FeedbackLoopError(f"status failed for {batch}: {completed.stderr.strip()}")
        value = _parse_status(completed, batch)
        if value is None:
            raise FeedbackLoopError(f"status returned no JSON for {batch}")
        values.append(value)
    payload = {"status": "VISIBLE", "batches": values}
    _write_json(run_dir / "labeling-status.json", payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_prepare = sub.add_parser("prepare")
    p_prepare.add_argument("--source-run-dir", type=Path, default=SOURCE_RUN_DIR)
    p_prepare.add_argument("--output-dir", type=Path, default=OUTPUT_RUN_DIR)
    p_prepare.add_argument("--train-batch", default=TRAIN_BATCH)
    p_prepare.add_argument("--check-batch", default=CHECK_BATCH)
    p_dispatch = sub.add_parser("dispatch")
    p_dispatch.add_argument("--run-dir", type=Path, default=OUTPUT_RUN_DIR)
    p_status = sub.add_parser("status")
    p_status.add_argument("--run-dir", type=Path, default=OUTPUT_RUN_DIR)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(source_run_dir=args.source_run_dir, output_dir=args.output_dir, train_batch=args.train_batch, check_batch=args.check_batch)
        elif args.command == "dispatch":
            result = dispatch(run_dir=args.run_dir)
        else:
            result = status(run_dir=args.run_dir)
    except (FeedbackLoopError, OSError, RuntimeError, KeyError, ValueError) as exc:
        print(f"[feedback-pool-v3] FAIL-CLOSED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
