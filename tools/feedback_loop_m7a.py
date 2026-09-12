#!/usr/bin/env python3
"""Prepare and dispatch the M7A sparse auxiliary annotation batch.

The preparation side consumes only the already-frozen 300-row feed and its
prediction snapshot.  It selects questions, never answers: task payloads are
limited to sample identity, text, metadata and applicable v0.3 auxiliary
heads.  ``dispatch`` is a thin call to the existing labeling-app importer and
``status`` is read-only.  No model runtime is imported and no inference is
performed here.
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
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:  # ``python -m tools...`` from the repo root
    from tools.prepare_v03_auxiliary_pilot import (
        AROUSAL_CUES,
        DISCOURSE_FOCUS_CUES,
        EXPECTATION_CUES,
        MARKET_SCOPE_CUES,
        PRICE_DIRECTION_CUES,
        _is_low_information,
        _looks_truncated,
    )
except ModuleNotFoundError:  # direct ``python tools/feedback_loop_m7a.py``
    from prepare_v03_auxiliary_pilot import (  # type: ignore[no-redef]
        AROUSAL_CUES,
        DISCOURSE_FOCUS_CUES,
        EXPECTATION_CUES,
        MARKET_SCOPE_CUES,
        PRICE_DIRECTION_CUES,
        _is_low_information,
        _looks_truncated,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent
SOURCE_RUN_DIR = REPO_ROOT / "runs" / "feedback-loop-v1-20260909"
DEFAULT_FEED = SOURCE_RUN_DIR / "feed-input.jsonl"
DEFAULT_PREDICTIONS = SOURCE_RUN_DIR / "prediction-snapshot.jsonl"
DEFAULT_DATACLEAN_PILOT = PROJECT_ROOT / "MyResearcher-DataClean" / "artifacts" / "round-003" / "pilot-300.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "human-review-artifacts" / "m7a-v03-aux50-20260913"
DEFAULT_BATCH = "m7a-v03-aux50-20260913"
SUPPLEMENT_OUTPUT_DIR = PROJECT_ROOT / "human-review-artifacts" / "m7a-v03-rare-head-supplement-20260913"
SUPPLEMENT_BATCH = "m7a-v03-rare-head-supplement-20260913"
DEFAULT_APP_ROOT = PROJECT_ROOT / "MyResearcher-Labeling-app"
DEFAULT_APP_CONFIG = DEFAULT_APP_ROOT / "data" / "config-gpt.json"
DEFAULT_APP_PYTHON = DEFAULT_APP_ROOT / ".venv" / "bin" / "python"
DEFAULT_TASKS = DEFAULT_OUTPUT_DIR / "auxiliary-tasks.jsonl"
SUPPLEMENT_TASKS = SUPPLEMENT_OUTPUT_DIR / "auxiliary-tasks.jsonl"
FACTORS = (
    "expectation_phase",
    "arousal_level",
    "desired_price_direction",
    "market_scope",
    "discourse_focus",
)
# The frozen pilot vocabulary is intentionally conservative.  These additional
# sector words are explicit market/industry references in the collector feed,
# not answers; they only make a market-scope question applicable when the
# model's existing emotion_target prediction is MARKET.
M7A_MARKET_SCOPE_CUES = re.compile(
    r"大盘|指数|光伏板块|主力|散户|资金|机构|外资|基金|行业|板块|产品|地区|"
    r"交易|盘面|科技|芯片|新能源|消费|医药|金融|地产|汽车|银行"
)
V1_SELECTIONS = (
    REPO_ROOT / "runs" / "feedback-loop-v1-20260909" / "selection-train.jsonl",
    REPO_ROOT / "runs" / "feedback-loop-v1-20260909" / "selection-check.jsonl",
    REPO_ROOT / "runs" / "feedback-loop-v2-20260909" / "selection-train.jsonl",
    REPO_ROOT / "runs" / "feedback-loop-v2-20260909" / "selection-check.jsonl",
    REPO_ROOT / "runs" / "feedback-loop-v3-20260910" / "selection-train.jsonl",
    REPO_ROOT / "runs" / "feedback-loop-v3-20260910" / "selection-check.jsonl",
)
HUMAN_TRAIN = REPO_ROOT / "runs" / "feedback-loop-v3-finalized-20260910" / "human-train-100-seven-head.jsonl"
HUMAN_CHECK = REPO_ROOT / "runs" / "feedback-loop-v3-finalized-20260910" / "human-check-20-semantic-seven-head.jsonl"
M6B_TASKS = PROJECT_ROOT / "human-review-artifacts" / "m6b-v03-auxiliary-pilot-20260911" / "auxiliary-pilot-tasks.jsonl"
M6B_ANNOTATIONS = PROJECT_ROOT / "human-review-artifacts" / "m6b-v03-auxiliary-pilot-20260911" / "auxiliary-pilot-annotations.jsonl"


class M7APreparationError(ValueError):
    """Deterministic feed/selection contract violation."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise M7APreparationError(f"JSONL missing: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise M7APreparationError(f"invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise M7APreparationError(f"JSONL row is not an object at {path}:{line_number}")
        rows.append(row)
    return rows


def _stable_key(sample_id: str) -> str:
    return hashlib.sha256(f"m7a-v03-aux50-20260913|{sample_id}".encode("utf-8")).hexdigest()


def _normalized_entropy(probabilities: Mapping[str, Any]) -> float:
    values = [float(value) for value in probabilities.values()]
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        return 1.0
    total = sum(values)
    if total <= 0:
        return 1.0
    values = [value / total for value in values if value > 0]
    entropy = -sum(value * math.log(value) for value in values)
    return entropy / math.log(max(2, len(probabilities)))


def _aggregate_uncertainty(prediction_row: Mapping[str, Any]) -> float:
    predictions = prediction_row.get("predictions", {})
    scalar = []
    for head in ("target_mode", "stance", "emotion_primary", "emotion_target", "action_tendency", "context_dependency"):
        values = predictions.get(head, {}).get("probabilities", {})
        if isinstance(values, Mapping):
            scalar.append(_normalized_entropy(values))
    reasoning = prediction_row.get("reasoning_mean_uncertainty")
    reasoning_value = float(reasoning) if isinstance(reasoning, (int, float)) and math.isfinite(float(reasoning)) else 1.0
    return (sum(scalar) / len(scalar) if scalar else 1.0) * 0.7 + reasoning_value * 0.3


def _load_prior_identity(extra_paths: Sequence[Path] = ()) -> tuple[set[str], set[str]]:
    """Return prior IDs and exact-content groups without reading labels as answers."""

    ids: set[str] = set()
    groups: set[str] = set()
    for path in (*V1_SELECTIONS, HUMAN_TRAIN, HUMAN_CHECK, M6B_TASKS, M6B_ANNOTATIONS, *extra_paths):
        if not path.is_file():
            continue
        for row in _read_jsonl(path):
            sample_id = row.get("sample_id")
            if isinstance(sample_id, str) and sample_id:
                ids.add(sample_id)
            group = row.get("exact_content_group")
            if not isinstance(group, str):
                group = (row.get("metadata") or {}).get("exact_content_group") if isinstance(row.get("metadata"), Mapping) else None
            if isinstance(group, str) and group:
                groups.add(group)
    return ids, groups


def _validate_dataclean_pilot(path: Path, feed_rows: Sequence[Mapping[str, Any]]) -> str:
    """Bind the frozen feed to the read-only DataClean pilot without copying PII."""

    rows = _read_jsonl(path)
    if len(rows) != 300:
        raise M7APreparationError(f"DataClean round-003 pilot must contain 300 rows, got {len(rows)}")
    by_observation = {row.get("observation_id"): row for row in rows}
    if len(by_observation) != len(rows):
        raise M7APreparationError("DataClean pilot observation_id values must be unique")
    for feed in feed_rows:
        observation_id = feed.get("observation_id")
        source = by_observation.get(observation_id)
        if source is None or source.get("content") != (feed.get("text") or feed.get("model_text")):
            raise M7APreparationError(f"feed/DataClean pilot mismatch for observation_id={observation_id}")
        if source.get("exact_content_group") != feed.get("exact_content_group"):
            raise M7APreparationError(f"feed/DataClean content group mismatch for observation_id={observation_id}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _applicable_factors(text: str, prediction_row: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    """Choose applicable questions from explicit text cues only.

    ``prediction_row`` remains an optional compatibility argument for callers
    that use the frozen snapshot to rank uncertainty.  Its legacy emotion
    predictions are deliberately not a gate for either rare factor.
    """

    factors: list[str] = []
    if EXPECTATION_CUES.search(text):
        factors.append("expectation_phase")
    if AROUSAL_CUES.search(text):
        factors.append("arousal_level")
    if PRICE_DIRECTION_CUES.search(text):
        factors.append("desired_price_direction")
    if M7A_MARKET_SCOPE_CUES.search(text):
        factors.append("market_scope")
    if DISCOURSE_FOCUS_CUES.search(text):
        factors.append("discourse_focus")
    return tuple(factors)


def _looks_unfinished(text: str) -> bool:
    """Reject obvious collector cuts while retaining valid short sentences."""

    compact = text.strip()
    if not compact:
        return True
    return compact.endswith(("，", "、", "：", ":", "和", "与", "或", "如果", "因为", "但是", "所以", "而且", "然后", "以及", "最近", "其中", "之后", "以后", "时候"))


def _ensure_output_writable(output_dir: Path) -> None:
    """Permit regeneration only before a batch has been imported."""

    if not output_dir.exists() or not any(output_dir.iterdir()):
        return
    summary_path = output_dir / "summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        raise M7APreparationError(f"existing output is not a valid preparation package: {output_dir}")
    if isinstance(summary, Mapping) and summary.get("labeling_app_imported"):
        raise M7APreparationError(f"imported M7A output is immutable; refusing to overwrite: {output_dir}")


def load_feed_and_predictions(feed_path: Path = DEFAULT_FEED, prediction_path: Path = DEFAULT_PREDICTIONS, *, require_300: bool = True) -> list[dict[str, Any]]:
    feed = _read_jsonl(feed_path)
    prediction_rows = _read_jsonl(prediction_path)
    prediction_index = {row.get("sample_id"): row for row in prediction_rows}
    if require_300 and (len(feed) != 300 or len(prediction_index) != 300):
        raise M7APreparationError(f"expected 300 feed/prediction rows, got feed={len(feed)} predictions={len(prediction_index)}")
    if len(prediction_index) != len(prediction_rows):
        raise M7APreparationError("prediction snapshot contains duplicate sample_id")
    candidates: list[dict[str, Any]] = []
    for row in feed:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or sample_id not in prediction_index:
            raise M7APreparationError(f"feed/prediction identity mismatch: {sample_id!r}")
        text = row.get("text") or row.get("model_text")
        if not isinstance(text, str) or _is_low_information(text):
            continue
        length_value = row.get("text_length", len(text))
        try:
            length = int(length_value)
        except (TypeError, ValueError):
            length = len(text)
        # M7A excludes the old collector's 39/40-character boundary cuts.
        if length >= 39 or _looks_truncated(sample_id, text, length_value) or _looks_unfinished(text):
            continue
        group = row.get("exact_content_group")
        if not isinstance(group, str) or not group:
            continue
        prediction = prediction_index[sample_id]
        factors = _applicable_factors(text, prediction)
        if not factors:
            continue
        candidates.append({
            "sample_id": sample_id,
            "text": text,
            "title": row.get("stock_name") or "",
            "stock_code": row.get("stock_code"),
            "stock_name": row.get("stock_name"),
            "exact_content_group": group,
            "observation_id": row.get("observation_id"),
            "sampling_reasons": sorted(set(row.get("sampling_reasons") or [])),
            "text_length": length,
            "length_bucket": "SHORT" if length < 10 else "MEDIUM" if length <= 25 else "LONG",
            "factors": factors,
            "aggregate_uncertainty": _aggregate_uncertainty(prediction),
        })
    return candidates


def select_tasks(candidates: Sequence[Mapping[str, Any]], *, target_assignments: int = 50, max_heads_per_sample: int = 2) -> list[dict[str, Any]]:
    prior_ids, prior_groups = _load_prior_identity()
    filtered = [row for row in candidates if row["sample_id"] not in prior_ids and row["exact_content_group"] not in prior_groups]
    # Stable uncertainty order, then deterministic hash.  A first pass over
    # length buckets guarantees short, medium and long semantic coverage.
    ranked = sorted(filtered, key=lambda row: (-float(row["aggregate_uncertainty"]), _stable_key(str(row["sample_id"])), str(row["sample_id"])))
    ordered: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for bucket in ("SHORT", "MEDIUM", "LONG"):
        for row in ranked:
            if row["length_bucket"] == bucket and row["sample_id"] not in seen:
                ordered.append(row); seen.add(str(row["sample_id"])); break
    ordered.extend(row for row in ranked if row["sample_id"] not in seen)

    # Existing M6B counts are used only as a coverage prior.  No answer is
    # loaded into, or emitted from, the task payload.
    current_counts = Counter({"expectation_phase": 12, "arousal_level": 4, "desired_price_direction": 7, "market_scope": 1, "discourse_focus": 10})
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_groups: set[str] = set()
    assignment_count = 0
    for row in ordered:
        if assignment_count >= target_assignments:
            break
        sample_id = str(row["sample_id"])
        group = str(row["exact_content_group"])
        if sample_id in selected_ids or group in selected_groups:
            continue
        applicable = sorted(row["factors"], key=lambda factor: (current_counts[factor], _stable_key(sample_id + "|" + factor), factor))
        heads = applicable[:max_heads_per_sample]
        if not heads:
            continue
        if assignment_count + len(heads) > target_assignments:
            heads = heads[: target_assignments - assignment_count]
        if not heads:
            break
        item = dict(row); item["heads"] = heads
        selected.append(item)
        selected_ids.add(sample_id); selected_groups.add(group)
        assignment_count += len(heads)
        for factor in heads:
            current_counts[factor] += 1
    if assignment_count == 0:
        raise M7APreparationError("no eligible M7A assignments remain after prior-ID/group and input-quality filters")
    if {factor for row in selected for factor in row["heads"]} != set(f for f in FACTORS if any(f in row["factors"] for row in filtered)):
        # This catches an accidental omission of an applicable under-covered
        # factor, while allowing a genuinely unavailable factor to stay empty.
        raise M7APreparationError("selection did not cover every factor with an eligible source cue")
    return selected


def select_rare_head_supplement(
    candidates: Sequence[Mapping[str, Any]],
    *,
    target_market_scope: int = 11,
    target_arousal_level: int = 2,
    current_task_path: Path = DEFAULT_TASKS,
) -> list[dict[str, Any]]:
    """Select only the two under-covered factors for the M7A supplement.

    The requested numbers are upper bounds.  If the frozen feed has fewer
    applicable rows after all prior-ID/group and input-quality exclusions, the
    real count is emitted without filler.  A row may receive both rare heads,
    but no other factor is ever added.
    """

    prior_ids, prior_groups = _load_prior_identity((current_task_path,))
    eligible = [
        row for row in candidates
        if row.get("sample_id") not in prior_ids
        and row.get("exact_content_group") not in prior_groups
        and set(row.get("factors", ())) & {"market_scope", "arousal_level"}
    ]
    ranked = sorted(
        eligible,
        key=lambda row: (-float(row.get("aggregate_uncertainty", 0.0)), _stable_key(str(row["sample_id"])), str(row["sample_id"])),
    )
    selected: dict[str, dict[str, Any]] = {}
    selected_groups: set[str] = set()
    counts = Counter()
    for factor, limit in (("market_scope", target_market_scope), ("arousal_level", target_arousal_level)):
        for row in ranked:
            if counts[factor] >= limit or factor not in row.get("factors", ()):
                continue
            sample_id = str(row["sample_id"])
            group = str(row["exact_content_group"])
            existing = selected.get(sample_id)
            if existing is None and group in selected_groups:
                continue
            if existing is None:
                existing = dict(row)
                existing["heads"] = []
                selected[sample_id] = existing
                selected_groups.add(group)
            if factor not in existing["heads"] and len(existing["heads"]) < 2:
                existing["heads"].append(factor)
                counts[factor] += 1
    result = list(selected.values())
    if not result:
        raise M7APreparationError("no eligible rare-head supplement assignments remain")
    if any(set(row["heads"]) - {"market_scope", "arousal_level"} or len(row["heads"]) > 2 for row in result):
        raise M7APreparationError("rare-head supplement contains an invalid head assignment")
    return result


def _task_payload(row: Mapping[str, Any], batch_id: str) -> dict[str, Any]:
    metadata = {
        "batch_id": batch_id,
        "observation_id": row.get("observation_id"),
        "stock_code": row.get("stock_code"),
        "stock_name": row.get("stock_name"),
        "exact_content_group": row.get("exact_content_group"),
        "sampling_reasons": list(row.get("sampling_reasons") or []),
        "length_bucket": row.get("length_bucket"),
        "text_length": row.get("text_length"),
        "selection_basis": "frozen seven-head uncertainty plus length/reason coverage; applicability cues only",
        "schema_version": "semantic-schema-candidate-v0.3",
    }
    return {"sample_id": row["sample_id"], "text": row["text"], "title": row.get("title") or "", "metadata": metadata, "heads": list(row["heads"])}


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def prepare(*, feed_path: Path = DEFAULT_FEED, prediction_path: Path = DEFAULT_PREDICTIONS, dataclean_pilot_path: Path = DEFAULT_DATACLEAN_PILOT, output_dir: Path = DEFAULT_OUTPUT_DIR, batch_id: str = DEFAULT_BATCH) -> dict[str, Any]:
    _ensure_output_writable(output_dir)
    feed_rows = _read_jsonl(feed_path)
    dataclean_sha256 = _validate_dataclean_pilot(dataclean_pilot_path, feed_rows)
    candidates = load_feed_and_predictions(feed_path, prediction_path)
    selected = select_tasks(candidates)
    tasks = [_task_payload(row, batch_id) for row in selected]
    factor_counts = Counter(head for row in tasks for head in row["heads"])
    summary = {
        "batch_id": batch_id,
        "schema_version": "semantic-schema-candidate-v0.3",
        "source_feed": str(feed_path),
        "source_prediction_snapshot": str(prediction_path),
        "source_dataclean_pilot": str(dataclean_pilot_path),
        "source_dataclean_pilot_sha256": dataclean_sha256,
        "sample_count": len(tasks),
        "assignment_count": sum(len(row["heads"]) for row in tasks),
        "factor_assignment_counts": {factor: factor_counts[factor] for factor in FACTORS},
        "distinct_exact_content_groups": len({row["metadata"]["exact_content_group"] for row in tasks}),
        "max_heads_per_sample": max(len(row["heads"]) for row in tasks),
        "answers_included": False,
        "model_predictions_included": False,
        "human_labels_used_for_selection": False,
        "excluded_prior_ids_and_groups": True,
        "selection_is_deterministic": True,
        "labeling_app_imported": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "auxiliary-tasks.jsonl", tasks)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def prepare_rare_supplement(
    *,
    feed_path: Path = DEFAULT_FEED,
    prediction_path: Path = DEFAULT_PREDICTIONS,
    dataclean_pilot_path: Path = DEFAULT_DATACLEAN_PILOT,
    output_dir: Path = SUPPLEMENT_OUTPUT_DIR,
    batch_id: str = SUPPLEMENT_BATCH,
) -> dict[str, Any]:
    """Prepare the rare-head supplement without touching the original M7A run."""

    _ensure_output_writable(output_dir)
    feed_rows = _read_jsonl(feed_path)
    dataclean_sha256 = _validate_dataclean_pilot(dataclean_pilot_path, feed_rows)
    candidates = load_feed_and_predictions(feed_path, prediction_path)
    selected = select_rare_head_supplement(candidates)
    tasks = [_task_payload(row, batch_id) for row in selected]
    factor_counts = Counter(head for row in tasks for head in row["heads"])
    summary = {
        "batch_id": batch_id,
        "schema_version": "semantic-schema-candidate-v0.3",
        "source_feed": str(feed_path),
        "source_prediction_snapshot": str(prediction_path),
        "source_dataclean_pilot": str(dataclean_pilot_path),
        "source_dataclean_pilot_sha256": dataclean_sha256,
        "sample_count": len(tasks),
        "assignment_count": sum(len(row["heads"]) for row in tasks),
        "factor_assignment_counts": {factor: factor_counts[factor] for factor in FACTORS},
        "allowed_heads": ["market_scope", "arousal_level"],
        "distinct_exact_content_groups": len({row["metadata"]["exact_content_group"] for row in tasks}),
        "max_heads_per_sample": max(len(row["heads"]) for row in tasks),
        "answers_included": False,
        "model_predictions_included": False,
        "human_labels_used_for_selection": False,
        "excluded_prior_ids_and_groups": True,
        "selection_is_deterministic": True,
        "labeling_app_imported": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "auxiliary-tasks.jsonl", tasks)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _run_app(args: Sequence[str], *, app_root: Path = DEFAULT_APP_ROOT, python: Path = DEFAULT_APP_PYTHON) -> subprocess.CompletedProcess[str]:
    command = [str(python), str(app_root / "tools" / "gpt_tasks.py"), *args]
    return subprocess.run(command, cwd=app_root, text=True, capture_output=True, check=False)


def dispatch(*, tasks_path: Path = DEFAULT_TASKS, batch_id: str = DEFAULT_BATCH, app_config: Path = DEFAULT_APP_CONFIG, output_dir: Path = DEFAULT_OUTPUT_DIR) -> str:
    if not tasks_path.is_file():
        raise M7APreparationError(f"prepared task file is missing: {tasks_path}")
    status = _run_app(["status", "--batch", batch_id, "--config", str(app_config)])
    if status.returncode == 0:
        raise M7APreparationError(f"batch already exists; refusing duplicate dispatch: {batch_id}")
    result = _run_app(["add", "--assignment-file", str(tasks_path), "--batch", batch_id, "--schema-version", "semantic-schema-candidate-v0.3", "--config", str(app_config)])
    if result.returncode != 0:
        raise M7APreparationError(f"labeling-app dispatch failed: {result.stderr.strip() or result.stdout.strip()}")
    return result.stdout.strip()


def status(*, batch_id: str = DEFAULT_BATCH, app_config: Path = DEFAULT_APP_CONFIG, output_dir: Path = DEFAULT_OUTPUT_DIR) -> str:
    result = _run_app(["status", "--batch", batch_id, "--config", str(app_config)])
    if result.returncode != 0:
        raise M7APreparationError(f"labeling-app status failed: {result.stderr.strip() or result.stdout.strip()}")
    output = result.stdout.strip()
    # Keep the ignored preparation summary truthful after the read-only status
    # check; this does not alter task payloads or the labeling database.
    summary_path = output_dir / "summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        status_payload = json.loads(output)
        if isinstance(summary, dict) and status_payload.get("batch") == batch_id:
            summary["labeling_app_imported"] = True
            summary["labeling_app_status"] = status_payload
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "dispatch", "status", "prepare-supplement", "dispatch-supplement", "status-supplement"):
        sub.add_parser(name)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result: Any = prepare()
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        elif args.command == "dispatch":
            print(dispatch())
        elif args.command == "status":
            print(status())
        elif args.command == "prepare-supplement":
            result = prepare_rare_supplement()
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        elif args.command == "dispatch-supplement":
            print(dispatch(tasks_path=SUPPLEMENT_TASKS, batch_id=SUPPLEMENT_BATCH, output_dir=SUPPLEMENT_OUTPUT_DIR))
        else:
            print(status(batch_id=SUPPLEMENT_BATCH, output_dir=SUPPLEMENT_OUTPUT_DIR))
    except M7APreparationError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
