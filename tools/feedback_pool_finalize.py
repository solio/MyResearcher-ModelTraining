#!/usr/bin/env python3
"""Finalize the FL3 human pool after both remote batches are complete.

The command is deliberately a *post-labeling* operation.  ``status`` only
queries the existing Labeling App.  ``finalize`` first requires 560/560 and
98/98 final assignments, then validates and merges the immutable V1/V2 local
snapshots with the two FL3 batches into exactly 100 Train and 20 semantic
Check samples.  Human answers remain revisable labels; this tool never calls
them Gold and never runs an adapter or model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tools import feedback_adapter_v1 as adapter


REPO_ROOT = Path(__file__).resolve().parents[1]
V1_RUN = REPO_ROOT / "runs" / "feedback-loop-v1-20260909"
V2_RUN = REPO_ROOT / "runs" / "feedback-loop-v2-20260909"
V3_RUN = REPO_ROOT / "runs" / "feedback-loop-v3-20260910"
DEFAULT_OUTPUT = REPO_ROOT / "runs" / "feedback-loop-v3-finalized-20260910"
TRAIN_BATCH = "fl3-train-100-target"
CHECK_BATCH = "fl3-check-20-semantic-target"
EXPECTED_TASKS = {TRAIN_BATCH: 80 * len(adapter.HEADS), CHECK_BATCH: 14 * len(adapter.HEADS)}
SCHEMA_VERSION = "semantic-schema-calibrated-v0.2.1"
FORBIDDEN_PREDICTION_KEYS = {
    "prediction", "predictions", "probability", "probabilities", "model_prediction",
    "model_predictions", "model_output", "weights", "weight", "logits",
}


class FinalizeError(ValueError):
    """Actionable fail-closed validation error."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FinalizeError(f"missing JSONL: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FinalizeError(f"invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise FinalizeError(f"JSONL row is not an object at {path}:{line_number}")
        rows.append(value)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _contains_forbidden_prediction_key(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).lower() in FORBIDDEN_PREDICTION_KEYS:
                return str(key)
            found = _contains_forbidden_prediction_key(nested)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _contains_forbidden_prediction_key(nested)
            if found is not None:
                return found
    return None


def _selection(path: Path, expected_rows: int, role: str) -> tuple[list[dict[str, Any]], set[str]]:
    rows = _read_jsonl(path)
    if len(rows) != expected_rows:
        raise FinalizeError(f"{role} selection expected {expected_rows} rows, got {len(rows)}")
    ids = [row.get("sample_id") for row in rows]
    if any(not isinstance(sample_id, str) or not sample_id.strip() for sample_id in ids):
        raise FinalizeError(f"{role} selection contains an invalid sample_id")
    if len(set(ids)) != expected_rows:
        raise FinalizeError(f"{role} selection contains duplicate sample_ids")
    groups = [row.get("exact_content_group") for row in rows]
    if any(not isinstance(group, str) or not group.strip() for group in groups):
        raise FinalizeError(f"{role} selection contains a missing exact_content_group")
    return rows, set(ids)


def _validate_answer(head: str, answer: Any, sample_id: str) -> None:
    if head == "reasoning_tags":
        if not isinstance(answer, list):
            raise FinalizeError(f"{sample_id}/{head} answer must be a list")
        if len(set(answer)) != len(answer):
            raise FinalizeError(f"{sample_id}/{head} answer contains duplicate tags")
        if any(not isinstance(tag, str) or tag not in adapter.CLASS_ORDER[head] for tag in answer):
            raise FinalizeError(f"{sample_id}/{head} answer contains an unknown tag")
        return
    if not isinstance(answer, str) or answer not in adapter.CLASS_ORDER[head]:
        raise FinalizeError(f"{sample_id}/{head} answer is outside the frozen taxonomy: {answer!r}")


def _validate_labels(
    path: Path,
    selection_ids: set[str],
    expected_samples: int,
    role: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = _read_jsonl(path)
    expected_rows = expected_samples * len(adapter.HEADS)
    if len(rows) != expected_rows:
        raise FinalizeError(f"{role} labels expected {expected_rows} rows, got {len(rows)}")
    seen: set[tuple[str, str]] = set()
    by_sample: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = row.get("sample_id")
        head = row.get("head")
        if not isinstance(sample_id, str) or sample_id not in selection_ids:
            raise FinalizeError(f"{role} labels contain an unexpected sample_id: {sample_id!r}")
        if head not in adapter.HEADS:
            raise FinalizeError(f"{role} labels contain an unknown head: {head!r}")
        key = (sample_id, str(head))
        if key in seen:
            raise FinalizeError(f"{role} labels contain duplicate sample/head: {key}")
        seen.add(key)
        if row.get("is_final") is not True:
            raise FinalizeError(f"{role} has a non-final answer for {key}")
        if row.get("schema_version") != SCHEMA_VERSION:
            raise FinalizeError(f"{role} has an unexpected schema_version for {key}")
        forbidden = _contains_forbidden_prediction_key(row)
        if forbidden is not None:
            raise FinalizeError(f"{role} contains model prediction field {forbidden!r}")
        _validate_answer(str(head), row.get("answer"), sample_id)
        by_sample.setdefault(sample_id, {})[str(head)] = dict(row)
    if set(by_sample) != selection_ids:
        missing = sorted(selection_ids - set(by_sample))
        extra = sorted(set(by_sample) - selection_ids)
        raise FinalizeError(f"{role} sample coverage mismatch; missing={missing[:1]} extra={extra[:1]}")
    for sample_id, values in by_sample.items():
        if set(values) != set(adapter.HEADS):
            raise FinalizeError(f"{role} sample {sample_id} does not have exactly seven heads")
    return rows, by_sample


def _is_low_information(selection_row: Mapping[str, Any]) -> bool:
    return adapter._is_low_information_input(selection_row.get("model_text", selection_row.get("text", "")))


def _round_inputs(run_dir: Path, round_name: str, train_count: int, check_count: int) -> dict[str, Any]:
    train_selection, train_ids = _selection(run_dir / "selection-train.jsonl", train_count, f"{round_name} Train")
    check_selection, check_ids = _selection(run_dir / "selection-check.jsonl", check_count, f"{round_name} Check")
    labels, by_sample = _validate_labels(run_dir / "human-labels-revisable.jsonl", train_ids | check_ids, train_count + check_count, f"{round_name} labels")
    check_low_ids = {str(row["sample_id"]) for row in check_selection if _is_low_information(row)}
    return {
        "round": round_name,
        "train_selection": train_selection,
        "check_selection": check_selection,
        "train_ids": train_ids,
        "check_ids": check_ids,
        "check_low_ids": check_low_ids,
        "labels": labels,
        "labels_by_sample": by_sample,
    }


def _app_python() -> str:
    app_root = Path("/Users/mac/Documents/trae_projects/MyResearcher/MyResearcher-Labeling-app")
    candidate = app_root / ".venv" / "bin" / "python"
    return str(candidate if candidate.is_file() else sys.executable)


def _run_app(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    app_root = Path("/Users/mac/Documents/trae_projects/MyResearcher/MyResearcher-Labeling-app")
    command = [_app_python(), str(app_root / "tools" / "gpt_tasks.py"), *args, "--config", str(app_root / "data" / "config-gpt.json")]
    return subprocess.run(command, cwd=str(app_root), text=True, capture_output=True, check=False)


def _remote_status(batch: str) -> dict[str, Any]:
    completed = _run_app(["status", "--batch", batch])
    if completed.returncode != 0:
        raise FinalizeError(f"status failed for {batch}: {completed.stderr.strip()}")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise FinalizeError(f"status returned invalid JSON for {batch}") from exc
    if not isinstance(value, dict):
        raise FinalizeError(f"status returned a non-object for {batch}")
    expected = EXPECTED_TASKS[batch]
    if int(value.get("total", -1)) != expected:
        raise FinalizeError(f"{batch} has total={value.get('total')}, expected {expected}")
    return value


def status() -> dict[str, Any]:
    """Read both FL3 batch statuses without modifying the labeling app."""

    values = [_remote_status(TRAIN_BATCH), _remote_status(CHECK_BATCH)]
    return {"status": "VISIBLE", "batches": values}


def _all_final(status_values: Mapping[str, Mapping[str, Any]]) -> bool:
    for batch, expected in EXPECTED_TASKS.items():
        value = status_values.get(batch)
        if not isinstance(value, Mapping):
            return False
        if int(value.get("total", -1)) != expected:
            return False
        if int(value.get("done", -1)) != expected or int(value.get("finals", -1)) != expected or value.get("complete") is not True:
            return False
    return True


def _pull_batch(batch: str) -> list[dict[str, Any]]:
    handle = tempfile.NamedTemporaryFile(prefix="feedback-fl3-finalize-", suffix=".jsonl", delete=False)
    handle.close()
    path = Path(handle.name)
    try:
        completed = _run_app(["pull", "--format", "jsonl", "--batch", batch, "--final-only", "--out", str(path)])
        if completed.returncode != 0:
            raise FinalizeError(f"pull failed for {batch}: {completed.stderr.strip()}")
        return _read_jsonl(path)
    finally:
        path.unlink(missing_ok=True)


def _sorted_label_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted((dict(row) for row in rows), key=lambda row: (str(row.get("sample_id")), adapter.HEADS.index(str(row.get("head")))))


def finalize(
    *,
    output_dir: Path = DEFAULT_OUTPUT,
    v1_run: Path = V1_RUN,
    v2_run: Path = V2_RUN,
    v3_run: Path = V3_RUN,
    train_labels_path: Path | None = None,
    check_labels_path: Path | None = None,
    status_values: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate remote completion and emit exactly Train100/Check20."""

    if output_dir.exists():
        raise FinalizeError(f"refusing to overwrite finalized output: {output_dir}")
    if status_values is None:
        live = status()
        status_values = {str(value["batch"]): value for value in live["batches"]}
    if not _all_final(status_values):
        raise FinalizeError("FL3 batches are not both 100% final; no finalized training/check files were generated")

    v1 = _round_inputs(v1_run, "V1", 10, 4)
    v2 = _round_inputs(v2_run, "V2", 10, 4)
    train_selection, train_ids = _selection(v3_run / "selection-train.jsonl", 80, "FL3 Train")
    check_selection, check_ids = _selection(v3_run / "selection-check.jsonl", 14, "FL3 Check")
    if train_labels_path is None:
        train_labels = _pull_batch(TRAIN_BATCH)
    else:
        train_labels = _read_jsonl(train_labels_path)
    if check_labels_path is None:
        check_labels = _pull_batch(CHECK_BATCH)
    else:
        check_labels = _read_jsonl(check_labels_path)
    fl3_train_by_sample_rows = _validate_labels_from_rows(train_labels, train_ids, 80, "FL3 Train labels")
    fl3_check_by_sample_rows = _validate_labels_from_rows(check_labels, check_ids, 14, "FL3 Check labels")

    train_sets = [v1["train_ids"], v2["train_ids"], train_ids]
    check_sets = [v1["check_ids"], v2["check_ids"], check_ids]
    all_train_ids = set().union(*train_sets)
    all_check_ids = set().union(*check_sets)
    if len(all_train_ids) != 100:
        raise FinalizeError(f"merged Train expected 100 samples, got {len(all_train_ids)}")
    if len(all_train_ids & all_check_ids):
        raise FinalizeError("Train and Check sample_id sets overlap")
    v1_low = v1["check_low_ids"]
    v2_low = v2["check_low_ids"]
    low_ids = v1_low | v2_low
    semantic_check_ids = all_check_ids - low_ids
    if len(low_ids) != 2 or len(semantic_check_ids) != 20:
        raise FinalizeError(f"expected two historical low-information checks and 20 semantic checks, got {len(low_ids)} and {len(semantic_check_ids)}")
    if len(v1["check_ids"] - v1_low) != 2 or len(v2["check_ids"] - v2_low) != 4:
        raise FinalizeError("V1/V2 semantic Check populations are not the expected 2 and 4")
    if any(_is_low_information(row) for row in check_selection):
        raise FinalizeError("FL3 Check selection contains low-information input")

    train_rows = _sorted_label_rows([
        *[row for row in v1["labels"] if str(row["sample_id"]) in v1["train_ids"]],
        *[row for row in v2["labels"] if str(row["sample_id"]) in v2["train_ids"]],
        *fl3_train_by_sample_rows["rows"],
    ])
    check_rows = _sorted_label_rows([
        *[row for row in v1["labels"] if str(row["sample_id"]) not in v1_low and str(row["sample_id"]) in v1["check_ids"]],
        *[row for row in v2["labels"] if str(row["sample_id"]) not in v2_low and str(row["sample_id"]) in v2["check_ids"]],
        *fl3_check_by_sample_rows["rows"],
    ])
    excluded_rows = _sorted_label_rows([
        *[row for row in v1["labels"] if str(row["sample_id"]) in v1_low],
        *[row for row in v2["labels"] if str(row["sample_id"]) in v2_low],
    ])
    if len(train_rows) != 100 * len(adapter.HEADS) or len(check_rows) != 20 * len(adapter.HEADS) or len(excluded_rows) != 2 * len(adapter.HEADS):
        raise FinalizeError("final row counts do not equal 700/140/14")
    # Re-run the complete output-level key check after filtering the two
    # historical low-information Check samples.
    _validate_labels_from_rows(train_rows, all_train_ids, 100, "merged Train")
    _validate_labels_from_rows(check_rows, semantic_check_ids, 20, "merged semantic Check")
    _validate_labels_from_rows(excluded_rows, low_ids, 2, "excluded historical Check")

    output_dir.mkdir(parents=True, exist_ok=False)
    _write_jsonl(output_dir / "human-train-100-seven-head.jsonl", train_rows)
    _write_jsonl(output_dir / "human-check-20-semantic-seven-head.jsonl", check_rows)
    _write_jsonl(output_dir / "excluded-low-information-check.jsonl", excluded_rows)
    summary: dict[str, Any] = {
        "schema_version": "myresearcher.feedback-loop-v3-finalized-revisable-labels.v1",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": "FINALIZED_REVISABLE_HUMAN_LABELS",
        "labels_are_revisable_not_gold": True,
        "training_authorized_by_this_tool": False,
        "source_batches": {TRAIN_BATCH: dict(status_values[TRAIN_BATCH]), CHECK_BATCH: dict(status_values[CHECK_BATCH])},
        "counts": {
            "v1_train_samples": 10,
            "v2_train_samples": 10,
            "fl3_train_samples": 80,
            "human_train_100_samples": 100,
            "v1_semantic_check_samples": len(v1["check_ids"] - v1_low),
            "v2_semantic_check_samples": len(v2["check_ids"] - v2_low),
            "fl3_semantic_check_samples": 14,
            "human_check_20_semantic_samples": 20,
            "excluded_historical_low_information_check_samples": 2,
            "heads_per_sample": len(adapter.HEADS),
        },
        "sample_ids": {
            "train": sorted(all_train_ids),
            "check_semantic": sorted(semantic_check_ids),
            "excluded_low_information_check": sorted(low_ids),
        },
        "disjoint": True,
        "model_predictions_in_human_label_files": False,
        "forbidden_reads": ["Test", "Anchor", "Gold", "OOD", "Human48", "Human100", "LLM", "production_data"],
        "file_sha256": {},
    }
    for filename in ("human-train-100-seven-head.jsonl", "human-check-20-semantic-seven-head.jsonl", "excluded-low-information-check.jsonl"):
        summary["file_sha256"][filename] = _sha256(output_dir / filename)
    summary["content_address"] = hashlib.sha256(_canonical(summary)).hexdigest()
    _write_json(output_dir / "summary.json", summary)
    return summary


def _validate_labels_from_rows(rows: Sequence[Mapping[str, Any]], selection_ids: set[str], expected_samples: int, role: str) -> dict[str, Any]:
    """Validate an in-memory pull or merged output using the same strict rules."""

    expected_rows = expected_samples * len(adapter.HEADS)
    if len(rows) != expected_rows:
        raise FinalizeError(f"{role} expected {expected_rows} rows, got {len(rows)}")
    seen: set[tuple[str, str]] = set()
    by_sample: dict[str, dict[str, Any]] = {}
    for original in rows:
        row = dict(original)
        sample_id = row.get("sample_id")
        head = row.get("head")
        if not isinstance(sample_id, str) or sample_id not in selection_ids:
            raise FinalizeError(f"{role} contains an unexpected sample_id: {sample_id!r}")
        if head not in adapter.HEADS:
            raise FinalizeError(f"{role} contains an unknown head: {head!r}")
        key = (sample_id, str(head))
        if key in seen:
            raise FinalizeError(f"{role} contains duplicate sample/head: {key}")
        seen.add(key)
        if row.get("is_final") is not True:
            raise FinalizeError(f"{role} contains a non-final answer for {key}")
        if row.get("schema_version") != SCHEMA_VERSION:
            raise FinalizeError(f"{role} contains an unexpected schema_version for {key}")
        forbidden = _contains_forbidden_prediction_key(row)
        if forbidden is not None:
            raise FinalizeError(f"{role} contains model prediction field {forbidden!r}")
        _validate_answer(str(head), row.get("answer"), sample_id)
        by_sample.setdefault(sample_id, {})[str(head)] = row
    if set(by_sample) != selection_ids or any(set(values) != set(adapter.HEADS) for values in by_sample.values()):
        raise FinalizeError(f"{role} does not provide exactly seven heads for every selected sample")
    return {"rows": [dict(row) for row in rows], "by_sample": by_sample}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="read both FL3 batch statuses without modifying the app")
    p_finalize = sub.add_parser("finalize", help="merge V1/V2/FL3 only after both FL3 batches are fully final")
    p_finalize.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p_finalize.add_argument("--v1-run", type=Path, default=V1_RUN)
    p_finalize.add_argument("--v2-run", type=Path, default=V2_RUN)
    p_finalize.add_argument("--v3-run", type=Path, default=V3_RUN)
    args = parser.parse_args(argv)
    try:
        result = status() if args.command == "status" else finalize(output_dir=args.output_dir, v1_run=args.v1_run, v2_run=args.v2_run, v3_run=args.v3_run)
    except (FinalizeError, OSError, RuntimeError, ValueError) as exc:
        print(f"[feedback-pool-finalize] FAIL-CLOSED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
