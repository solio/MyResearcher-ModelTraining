"""M14b: identity-stripped, source-balanced human character TF-IDF heads.

M13 showed that a frozen generic CLS embedding did not generalise across the
three independent human checks.  M14 changes the representation rather than
continuing to tune that experiment: it learns character n-grams directly from
the human-labelled forum text.  This is intended to capture short A-share
phrases, abbreviations and wordplay.  Explicit ticker wrappers, six-digit stock
codes and the focal stock name are removed before vectorisation so the cohort's
stock mix cannot act as a label shortcut.

The experiment is deliberately bounded.  Two pre-registered character ranges
and two regularisation strengths are selected only by five-fold OOF predictions
on Human147.  The old and recent label sources receive equal total fit weight.
Recent Check19, Human Check20 and Human48 are loaded only after every head and
reasoning threshold has been frozen.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from . import encoder_m1 as m1
from . import encoder_m5_human100_source_balanced as m5
from . import encoder_m12_human_first_cv as m12
from . import encoder_m12_hybrid_runtime as hybrid
from . import encoder_m13_frozen_lert_linear as m13
from . import encoder_terminal_macbert_base as terminal
from .errors import ContractError
from .schema import SINGLE_LABEL_HEADS, V1_HEADS, LabelSchema


STAGE_ID = "M14B-IDENTITY-STRIPPED-HUMAN-CHAR-TFIDF-HUMAN147"
SEED = 35
C_GRID = (0.3, 3.0)
REPRESENTATIONS: tuple[tuple[str, tuple[int, int]], ...] = (
    ("char_1_4", (1, 4)),
    ("char_2_5", (2, 5)),
)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "baseline_v0.3.5.yaml"
DEFAULT_CACHE = REPO_ROOT / ".encoder-artifacts" / "hf-cache-lert-small"
DEFAULT_SELECTED_ARTIFACT = PROJECT_ROOT / "model-artifacts" / "m2-selected-candidate-frozen-for-m3-20260904"
DEFAULT_POOL = REPO_ROOT / "runs" / "feedback-loop-v3-finalized-20260910"
DEFAULT_RECENT_FEED = PROJECT_ROOT / "pipeline-runs" / "recent-chain-20260915" / "feed" / "feed-input.jsonl"
DEFAULT_RECENT_TRAIN = PROJECT_ROOT / "pipeline-runs" / "recent-chain-20260915" / "human" / "recent-train50-final.jsonl"
DEFAULT_RECENT_CHECK = PROJECT_ROOT / "pipeline-runs" / "recent-chain-20260915" / "human" / "recent-check19-valid-final.jsonl"
DEFAULT_HUMAN48 = Path("/tmp/human48.jsonl")
DEFAULT_HUMAN48_INPUT = REPO_ROOT / "runs" / "m3c-shadow500-20260907" / "shadow500-input.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "model-artifacts" / "m14b-identity-stripped-char-tfidf-human147-20260916"

TICKER_WRAPPER_RE = re.compile(r"\$[^$\n]{1,40}\((?:SH|SZ|BJ)?\d{6}\)\$")
EXCHANGE_CODE_RE = re.compile(r"(?<![A-Za-z0-9])(?:SH|SZ|BJ)\d{6}(?!\d)", re.IGNORECASE)
SIX_DIGIT_CODE_RE = re.compile(r"(?<!\d)\d{6}(?!\d)")
WHITESPACE_RE = re.compile(r"\s+")


def _require(ok: bool, code: str, message: str, **details: Any) -> None:
    if not ok:
        raise ContractError(code, message, **details)


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _texts(records: Sequence[m1.M1Record]) -> list[str]:
    """Remove focal identity from post text without inserting a cohort marker."""

    values: list[str] = []
    for record in records:
        text = TICKER_WRAPPER_RE.sub(" ", record.model_text)
        text = EXCHANGE_CODE_RE.sub(" ", text)
        text = SIX_DIGIT_CODE_RE.sub(" ", text)
        if record.stock_name.strip():
            text = re.sub(re.escape(record.stock_name.strip()), " ", text, flags=re.IGNORECASE)
        values.append(WHITESPACE_RE.sub(" ", text).strip())
    _require(len(values) == len(records), "M14_TEXT_COUNT_INVALID", "identity-stripped text count differs from records")
    return values


def _targets(records: Sequence[m1.M1Record], schema: LabelSchema, head: str) -> np.ndarray:
    order = list(schema.class_order[head])
    if head in SINGLE_LABEL_HEADS:
        return np.asarray([[int(str(record.label[head]) == label) for label in order] for record in records], dtype=int)
    return np.asarray([[int(label in record.label[head]) for label in order] for record in records], dtype=int)


def source_balanced_weights(records: Sequence[m1.M1Record], recent_ids: set[str]) -> np.ndarray:
    """Give base and recent human sources equal total mass within a fit fold."""

    recent = np.asarray([record.sample_id in recent_ids for record in records], dtype=bool)
    recent_count = int(recent.sum())
    base_count = int(len(records) - recent_count)
    _require(base_count > 0 and recent_count > 0, "M14_SOURCE_MISSING", "each fit fold must contain both human sources", base=base_count, recent=recent_count)
    result = np.empty(len(records), dtype=float)
    result[~recent] = len(records) / (2.0 * base_count)
    result[recent] = len(records) / (2.0 * recent_count)
    return result


def make_vectorizer(representation: str) -> TfidfVectorizer:
    ranges = dict(REPRESENTATIONS)
    _require(representation in ranges, "M14_REPRESENTATION_INVALID", "unknown M14 representation", representation=representation)
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=ranges[representation],
        min_df=2,
        max_features=30_000,
        lowercase=False,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float64,
    )


def _sqrt_balanced_weights(y: np.ndarray) -> dict[int, float] | None:
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        return None
    return {
        0: math.sqrt(len(y) / (2.0 * negatives)),
        1: math.sqrt(len(y) / (2.0 * positives)),
    }


def fit_binary_models(
    features: Any,
    targets: np.ndarray,
    c_value: float,
    sample_weight: np.ndarray,
) -> list[dict[str, Any]]:
    _require(features.shape[0] == len(targets) == len(sample_weight), "M14_FIT_SHAPE_INVALID", "features, targets and sample weights must align")
    result: list[dict[str, Any]] = []
    for column in range(targets.shape[1]):
        y = targets[:, column]
        class_weight = _sqrt_balanced_weights(y)
        if class_weight is None:
            result.append({"constant": float(y[0]) if len(y) else 0.0, "estimator": None})
            continue
        estimator = LogisticRegression(
            C=float(c_value),
            solver="liblinear",
            class_weight=class_weight,
            max_iter=2000,
            tol=1e-6,
            random_state=SEED,
        )
        estimator.fit(features, y, sample_weight=sample_weight)
        result.append({"constant": None, "estimator": estimator})
    return result


def predict_binary_models(models: Sequence[Mapping[str, Any]], features: Any) -> np.ndarray:
    columns: list[np.ndarray] = []
    for item in models:
        estimator = item.get("estimator")
        if estimator is None:
            columns.append(np.full(features.shape[0], float(item.get("constant", 0.0)), dtype=float))
            continue
        classes = [int(value) for value in estimator.classes_]
        _require(1 in classes, "M14_POSITIVE_CLASS_MISSING", "binary estimator has no positive class")
        columns.append(np.asarray(estimator.predict_proba(features)[:, classes.index(1)], dtype=float))
    result = np.stack(columns, axis=1)
    _require(bool(np.isfinite(result).all()), "M14_NONFINITE_PROBABILITY", "linear model produced non-finite probabilities")
    return result


def _normalise_scalar(values: np.ndarray) -> np.ndarray:
    return m13.normalise_scalar_probabilities(values)


def _candidate_key(metric: Mapping[str, Any]) -> tuple[float, float]:
    primary = float(metric.get("accuracy", metric.get("micro_f1", 0.0)))
    return float(metric["macro_f1"]), primary


def select_head_model(
    records: Sequence[m1.M1Record],
    schema: LabelSchema,
    head: str,
    recent_ids: set[str],
) -> tuple[dict[str, Any], dict[str, float], dict[str, Any]]:
    targets = _targets(records, schema, head)
    texts = _texts(records)
    splits = m12.fold_indices(records, head)
    trials: list[dict[str, Any]] = []
    best_key: tuple[float, float] | None = None
    selected_representation = REPRESENTATIONS[0][0]
    selected_c = C_GRID[0]
    selected_thresholds: dict[str, float] = {}
    selected_metric: dict[str, Any] = {}

    for representation, _ngram_range in REPRESENTATIONS:
        for c_value in C_GRID:
            oof = np.zeros_like(targets, dtype=float)
            for train_indices, validation_indices in splits:
                train_records = [records[int(index)] for index in train_indices]
                validation_texts = [texts[int(index)] for index in validation_indices]
                vectorizer = make_vectorizer(representation)
                train_features = vectorizer.fit_transform([texts[int(index)] for index in train_indices])
                validation_features = vectorizer.transform(validation_texts)
                weights = source_balanced_weights(train_records, recent_ids)
                models = fit_binary_models(train_features, targets[train_indices], c_value, weights)
                oof[validation_indices] = predict_binary_models(models, validation_features)
            if head in SINGLE_LABEL_HEADS:
                oof = _normalise_scalar(oof)
                thresholds: dict[str, float] = {}
            else:
                thresholds = m12.calibrate_reasoning_thresholds(oof, records, schema.class_order[head])
            metric = m12._metrics(oof, records, schema, head, thresholds if head == "reasoning_tags" else None)
            trials.append({
                "representation": representation,
                "C": c_value,
                "macro_f1": float(metric["macro_f1"]),
                "primary": float(metric.get("accuracy", metric.get("micro_f1", 0.0))),
                "thresholds": thresholds,
            })
            key = _candidate_key(metric)
            if best_key is None or key > best_key:
                best_key = key
                selected_representation = representation
                selected_c = c_value
                selected_thresholds = thresholds
                selected_metric = metric

    final_vectorizer = make_vectorizer(selected_representation)
    final_features = final_vectorizer.fit_transform(texts)
    final_weights = source_balanced_weights(records, recent_ids)
    final_models = fit_binary_models(final_features, targets, selected_c, final_weights)
    model = {
        "representation": selected_representation,
        "C": selected_c,
        "vectorizer": final_vectorizer,
        "models": final_models,
    }
    report = {
        "selected_representation": selected_representation,
        "selected_C": selected_c,
        "oof_metric": selected_metric,
        "trials": trials,
        "class_support": {label: int(targets[:, index].sum()) for index, label in enumerate(schema.class_order[head])},
        "feature_count": int(final_features.shape[1]),
    }
    return model, selected_thresholds, report


def predict_head(model: Mapping[str, Any], records: Sequence[m1.M1Record], head: str) -> np.ndarray:
    features = model["vectorizer"].transform(_texts(records))
    values = predict_binary_models(model["models"], features)
    return _normalise_scalar(values) if head in SINGLE_LABEL_HEADS else values


def _metric_table(
    arrays: Mapping[str, np.ndarray],
    records: Sequence[m1.M1Record],
    schema: LabelSchema,
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    return {
        head: m12._metrics(np.asarray(arrays[head]), records, schema, head, thresholds if head == "reasoning_tags" else None)
        for head in V1_HEADS
    }


def run_m14(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    cache_dir: str | Path = DEFAULT_CACHE,
    selected_artifact: str | Path = DEFAULT_SELECTED_ARTIFACT,
    pool_dir: str | Path = DEFAULT_POOL,
    recent_feed_jsonl: str | Path = DEFAULT_RECENT_FEED,
    recent_train_labels: str | Path = DEFAULT_RECENT_TRAIN,
    recent_check_labels: str | Path = DEFAULT_RECENT_CHECK,
    human48_jsonl: str | Path = DEFAULT_HUMAN48,
    human48_input: str | Path = DEFAULT_HUMAN48_INPUT,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    _require(not output.exists(), "M14_OUTPUT_EXISTS", "output directory must be new", path=str(output))
    output.mkdir(parents=True)
    started = time.monotonic()
    project = m1.ProjectConfig.load(config_path)
    schema, _weak, _dev = m1.load_m1_partitions(project)

    base_train, _ = m5.load_human_train_pool(pool_dir)
    base_train, base_audit = m12.exclude_dependency_conflicts(base_train, role="base_human_train")
    recent_train, recent_audit = m12.load_recent_feedback_records(
        recent_train_labels,
        recent_feed_jsonl,
        role="recent_train",
        exclude_dependency_conflicts=True,
    )
    train = [*base_train, *recent_train]
    recent_ids = {record.sample_id for record in recent_train}
    _require(len(train) == 147, "M14_TRAIN_COUNT_INVALID", "M14 requires the frozen Human147 fit population", observed=len(train))
    _require(len({record.sample_id for record in train}) == len(train), "M14_TRAIN_DUPLICATE", "M14 training sample IDs must be unique")

    head_models: dict[str, Any] = {}
    reasoning_thresholds: dict[str, float] = {}
    cv_report: dict[str, Any] = {}
    for head in V1_HEADS:
        model, thresholds, report = select_head_model(train, schema, head, recent_ids)
        head_models[head] = model
        if head == "reasoning_tags":
            reasoning_thresholds = thresholds
        cv_report[head] = report
    joblib.dump(
        {
            "stage_id": STAGE_ID,
            "feature": "identity_stripped_model_text_character_tfidf",
            "source_weighting": "equal_total_mass_base98_recent49",
            "class_order": {head: list(schema.class_order[head]) for head in V1_HEADS},
            "head_models": head_models,
            "reasoning_thresholds": reasoning_thresholds,
        },
        output / "linear-heads.joblib",
    )
    _dump(output / "cross-validation.json", cv_report)
    _dump(output / "reasoning-thresholds.json", reasoning_thresholds)

    # Independent labels are not opened until model family, C and thresholds freeze.
    recent_check, recent_check_audit = m12.load_recent_feedback_records(
        recent_check_labels,
        recent_feed_jsonl,
        role="cumulative_recent_check",
    )
    _require(len(recent_check) == 19, "M14_RECENT_CHECK_COUNT_INVALID", "M14 requires cumulative recent Check19", observed=len(recent_check))
    human_check, _ = m5.load_human_check_pool(pool_dir)
    human48 = terminal._load_human48(Path(human48_jsonl), Path(human48_input), schema)
    train_ids = {record.sample_id for record in train}
    _require(not train_ids.intersection(record.sample_id for record in recent_check), "M14_RECENT_CHECK_OVERLAP", "recent Check overlaps M14 fit population")

    evaluation_sets = {"recent_check19": recent_check, "human_check20": human_check, "human48": human48}
    candidate_tables: dict[str, Any] = {}
    current_tables: dict[str, Any] = {}
    current_runtime = hybrid.M12HybridInferenceRuntime.from_artifacts(
        selected_artifact=selected_artifact,
        config_path=config_path,
        cache_dir=cache_dir,
    )
    _np, torch, _AutoModel, _AutoTokenizer = m1._load_runtime_dependencies()
    parent_bundle = torch.load(Path(selected_artifact).resolve() / "unified-runtime-bundle.pt", map_location="cpu", weights_only=False)
    current_thresholds = {
        label: float(parent_bundle.get("reasoning_thresholds", {}).get(label, 0.5))
        for label in schema.class_order["reasoning_tags"]
    }
    for name, records in evaluation_sets.items():
        candidate_arrays = {head: predict_head(head_models[head], records, head) for head in V1_HEADS}
        candidate_all = _metric_table(candidate_arrays, records, schema, reasoning_thresholds)
        current_all = _metric_table(m12._baseline_probabilities(current_runtime, records, schema), records, schema, current_thresholds)
        if name == "human48":
            candidate_tables[name] = {head: candidate_all[head] for head in terminal.HUMAN48_HEADS}
            current_tables[name] = {head: current_all[head] for head in terminal.HUMAN48_HEADS}
        else:
            candidate_tables[name] = candidate_all
            current_tables[name] = current_all

    decisions = m13.promotion_decisions(
        current_tables["recent_check19"],
        candidate_tables["recent_check19"],
        current_tables["human_check20"],
        candidate_tables["human_check20"],
        current_tables["human48"],
        candidate_tables["human48"],
    )
    promoted = [head for head, item in decisions.items() if item["promote"]]
    _dump(output / "metrics.json", {"current": current_tables, "candidate": candidate_tables, "promotion": decisions})
    summary = {
        "stage_id": STAGE_ID,
        "status": "PROMOTE_PER_HEAD" if promoted else "STOP_NO_STABLE_GAIN",
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "train_samples": len(train),
        "base_train_samples": len(base_train),
        "recent_train_samples": len(recent_train),
        "recent_check_samples": len(recent_check),
        "human_check_samples": len(human_check),
        "human48_samples": len(human48),
        "base_audit": base_audit,
        "recent_train_audit": recent_audit,
        "recent_check_audit": recent_check_audit,
        "feature": "identity_stripped_model_text_character_tfidf",
        "source_weighting": "equal_total_mass_base98_recent49",
        "selected": {
            head: {"representation": head_models[head]["representation"], "C": head_models[head]["C"]}
            for head in V1_HEADS
        },
        "promoted_heads": promoted,
        "new_manual_labels_required": False,
        "deployment_changed": False,
    }
    _dump(output / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--selected-artifact", type=Path, default=DEFAULT_SELECTED_ARTIFACT)
    parser.add_argument("--pool-dir", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--recent-feed-jsonl", type=Path, default=DEFAULT_RECENT_FEED)
    parser.add_argument("--recent-train-labels", type=Path, default=DEFAULT_RECENT_TRAIN)
    parser.add_argument("--recent-check-labels", type=Path, default=DEFAULT_RECENT_CHECK)
    parser.add_argument("--human48-jsonl", type=Path, default=DEFAULT_HUMAN48)
    parser.add_argument("--human48-input", type=Path, default=DEFAULT_HUMAN48_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        result = run_m14(
            config_path=args.config,
            cache_dir=args.cache_dir,
            selected_artifact=args.selected_artifact,
            pool_dir=args.pool_dir,
            recent_feed_jsonl=args.recent_feed_jsonl,
            recent_train_labels=args.recent_train_labels,
            recent_check_labels=args.recent_check_labels,
            human48_jsonl=args.human48_jsonl,
            human48_input=args.human48_input,
            output_dir=args.output_dir,
        )
    except (ContractError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "M14_BLOCKED", "error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
