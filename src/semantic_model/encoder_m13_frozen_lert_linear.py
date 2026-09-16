"""M13: frozen LERT embeddings with human-only regularized linear heads.

This experiment deliberately changes the learning mechanism after the M12
incremental fine-tuning loop stopped converging.  The official pinned LERT
encoder is frozen, every record is represented by its L2-normalized CLS
embedding, and each schema class receives a deterministic one-vs-rest
logistic classifier.  Regularization is selected only by five-fold OOF
predictions on Human147.  Recent Check19, Human Check20 and Human48 are opened
only after all final linear models are frozen.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from . import encoder_m1 as m1
from . import encoder_m2_1_softlabel as m21
from . import encoder_m2_selected_runtime as selected
from . import encoder_m5_human100_source_balanced as m5
from . import encoder_m12_human_first_cv as m12
from . import encoder_m12_hybrid_runtime as hybrid
from . import encoder_terminal_macbert_base as terminal
from .errors import ContractError
from .schema import SINGLE_LABEL_HEADS, V1_HEADS, LabelSchema


STAGE_ID = "M13-FROZEN-LERT-LINEAR-HUMAN147"
MODEL_ID = m12.MODEL_ID
REVISION = m12.REVISION
SEED = 35
BATCH_SIZE = 16
C_GRID = (0.03, 0.3, 3.0)
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
DEFAULT_OUTPUT = PROJECT_ROOT / "model-artifacts" / "m13-frozen-lert-linear-human147-20260916"


def _require(ok: bool, code: str, message: str, **details: Any) -> None:
    if not ok:
        raise ContractError(code, message, **details)


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _targets(records: Sequence[m1.M1Record], schema: LabelSchema, head: str) -> np.ndarray:
    order = list(schema.class_order[head])
    if head in SINGLE_LABEL_HEADS:
        return np.asarray([[int(str(record.label[head]) == label) for label in order] for record in records], dtype=int)
    return np.asarray([[int(label in record.label[head]) for label in order] for record in records], dtype=int)


def normalise_scalar_probabilities(values: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(values, dtype=float)
    _require(probabilities.ndim == 2 and probabilities.shape[1] > 1, "M13_SCALAR_SHAPE_INVALID", "scalar probability matrix must be two-dimensional")
    probabilities = np.clip(probabilities, 0.0, 1.0)
    totals = probabilities.sum(axis=1, keepdims=True)
    empty = totals[:, 0] <= 0.0
    if bool(empty.any()):
        probabilities[empty] = 1.0 / probabilities.shape[1]
        totals = probabilities.sum(axis=1, keepdims=True)
    return probabilities / totals


def _sqrt_balanced_weights(y: np.ndarray) -> dict[int, float] | None:
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        return None
    return {
        0: math.sqrt(len(y) / (2.0 * negatives)),
        1: math.sqrt(len(y) / (2.0 * positives)),
    }


def fit_binary_models(features: np.ndarray, targets: np.ndarray, c_value: float) -> list[dict[str, Any]]:
    _require(features.ndim == 2 and targets.ndim == 2 and len(features) == len(targets), "M13_FIT_SHAPE_INVALID", "features and targets must align")
    result: list[dict[str, Any]] = []
    for column in range(targets.shape[1]):
        y = targets[:, column]
        weights = _sqrt_balanced_weights(y)
        if weights is None:
            result.append({"constant": float(y[0]) if len(y) else 0.0, "estimator": None})
            continue
        estimator = LogisticRegression(
            C=float(c_value),
            solver="liblinear",
            class_weight=weights,
            max_iter=2000,
            tol=1e-6,
            random_state=SEED,
        )
        estimator.fit(features, y)
        result.append({"constant": None, "estimator": estimator})
    return result


def predict_binary_models(models: Sequence[Mapping[str, Any]], features: np.ndarray) -> np.ndarray:
    columns: list[np.ndarray] = []
    for item in models:
        estimator = item.get("estimator")
        if estimator is None:
            columns.append(np.full(len(features), float(item.get("constant", 0.0)), dtype=float))
            continue
        classes = [int(value) for value in estimator.classes_]
        _require(1 in classes, "M13_POSITIVE_CLASS_MISSING", "binary estimator has no positive class")
        columns.append(np.asarray(estimator.predict_proba(features)[:, classes.index(1)], dtype=float))
    result = np.stack(columns, axis=1)
    _require(bool(np.isfinite(result).all()), "M13_NONFINITE_PROBABILITY", "linear model produced non-finite probabilities")
    return result


def _embedding_config(schema: LabelSchema) -> dict[str, Any]:
    value = m12._config(schema)
    value["batch_size"] = BATCH_SIZE
    return value


def extract_embeddings(
    runtime: tuple[Any, Any, Any, Any],
    snapshot: Path,
    records: Sequence[m1.M1Record],
    config: Mapping[str, Any],
    device: Any,
) -> np.ndarray:
    _np, torch, AutoModel, AutoTokenizer = runtime
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True)
    model = AutoModel.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False).to(device)
    model.eval()
    values: list[np.ndarray] = []
    with torch.no_grad():
        for offset in range(0, len(records), BATCH_SIZE):
            batch = m1._as_batch(torch, tokenizer, records[offset : offset + BATCH_SIZE], config, device)
            hidden = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state[:, 0, :]
            hidden = torch.nn.functional.normalize(hidden, p=2, dim=1)
            values.append(hidden.detach().cpu().numpy().astype(np.float64))
    del model
    gc.collect()
    try:
        torch.mps.empty_cache()
    except Exception:
        pass
    result = np.concatenate(values, axis=0)
    _require(result.shape[0] == len(records) and bool(np.isfinite(result).all()), "M13_EMBEDDING_INVALID", "frozen LERT embeddings are incomplete or non-finite")
    return result


def select_head_model(
    features: np.ndarray,
    records: Sequence[m1.M1Record],
    schema: LabelSchema,
    head: str,
) -> tuple[float, list[dict[str, Any]], dict[str, float], dict[str, Any]]:
    targets = _targets(records, schema, head)
    splits = m12.fold_indices(records, head)
    trials: list[dict[str, Any]] = []
    best_key: tuple[float, float] | None = None
    selected_c = C_GRID[0]
    selected_oof = np.zeros_like(targets, dtype=float)
    selected_thresholds: dict[str, float] = {}
    selected_metric: dict[str, Any] = {}
    for c_value in C_GRID:
        oof = np.zeros_like(targets, dtype=float)
        for train_indices, validation_indices in splits:
            models = fit_binary_models(features[train_indices], targets[train_indices], c_value)
            oof[validation_indices] = predict_binary_models(models, features[validation_indices])
        if head in SINGLE_LABEL_HEADS:
            oof = normalise_scalar_probabilities(oof)
            thresholds: dict[str, float] = {}
        else:
            thresholds = m12.calibrate_reasoning_thresholds(oof, records, schema.class_order[head])
        metric = m12._metrics(oof, records, schema, head, thresholds if head == "reasoning_tags" else None)
        primary = float(metric.get("accuracy", metric.get("micro_f1", 0.0)))
        key = (float(metric["macro_f1"]), primary)
        trials.append({"C": c_value, "macro_f1": float(metric["macro_f1"]), "primary": primary, "thresholds": thresholds})
        if best_key is None or key > best_key:
            best_key = key
            selected_c = c_value
            selected_oof = oof.copy()
            selected_thresholds = thresholds
            selected_metric = metric
    final_models = fit_binary_models(features, targets, selected_c)
    report = {
        "selected_C": selected_c,
        "oof_metric": selected_metric,
        "trials": trials,
        "class_support": {label: int(targets[:, index].sum()) for index, label in enumerate(schema.class_order[head])},
        "oof_probability_shape": list(selected_oof.shape),
    }
    return selected_c, final_models, selected_thresholds, report


def predict_head(models: Sequence[Mapping[str, Any]], features: np.ndarray, head: str) -> np.ndarray:
    values = predict_binary_models(models, features)
    return normalise_scalar_probabilities(values) if head in SINGLE_LABEL_HEADS else values


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


def _compact_primary(table: Mapping[str, Mapping[str, Any]], head: str) -> float:
    return float(table[head].get("accuracy", table[head].get("micro_f1", 0.0)))


def promotion_decisions(
    current_recent: Mapping[str, Mapping[str, Any]],
    candidate_recent: Mapping[str, Mapping[str, Any]],
    current_check20: Mapping[str, Mapping[str, Any]],
    candidate_check20: Mapping[str, Mapping[str, Any]],
    current_h48: Mapping[str, Mapping[str, Any]],
    candidate_h48: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for head in V1_HEADS:
        recent_primary_delta = _compact_primary(candidate_recent, head) - _compact_primary(current_recent, head)
        recent_macro_delta = float(candidate_recent[head]["macro_f1"] - current_recent[head]["macro_f1"])
        check_primary_delta = _compact_primary(candidate_check20, head) - _compact_primary(current_check20, head)
        check_macro_delta = float(candidate_check20[head]["macro_f1"] - current_check20[head]["macro_f1"])
        checks = {
            "recent_primary_gain_at_least_two_of_nineteen": recent_primary_delta >= (2.0 / 19.0) - 1e-12 if head != "reasoning_tags" else recent_primary_delta >= 0.03,
            "recent_macro_non_regression": recent_macro_delta >= -1e-12,
            "human_check20_primary_non_regression": check_primary_delta >= -1e-12,
            "human_check20_macro_drop_at_most_0_02": check_macro_delta >= -0.02 - 1e-12,
        }
        human48_delta: dict[str, float] | None = None
        if head in terminal.HUMAN48_HEADS:
            human48_delta = {
                "primary": _compact_primary(candidate_h48, head) - _compact_primary(current_h48, head),
                "macro_f1": float(candidate_h48[head]["macro_f1"] - current_h48[head]["macro_f1"]),
            }
            checks.update({
                "human48_primary_non_regression": human48_delta["primary"] >= -1e-12,
                "human48_macro_drop_at_most_0_02": human48_delta["macro_f1"] >= -0.02 - 1e-12,
            })
        result[head] = {
            "promote": all(checks.values()),
            "checks": checks,
            "recent_delta": {"primary": recent_primary_delta, "macro_f1": recent_macro_delta},
            "human_check20_delta": {"primary": check_primary_delta, "macro_f1": check_macro_delta},
            "human48_delta": human48_delta,
        }
    return result


def run_m13(
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
    _require(not output.exists(), "M13_OUTPUT_EXISTS", "output directory must be new", path=str(output))
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
    _require(len(train) == 147, "M13_TRAIN_COUNT_INVALID", "M13 requires the frozen Human147 fit population", observed=len(train))
    _require(len({record.sample_id for record in train}) == len(train), "M13_TRAIN_DUPLICATE", "M13 training sample IDs must be unique")

    runtime = m1._load_runtime_dependencies()
    _np, torch, AutoModel, _AutoTokenizer = runtime
    snapshot, snapshot_identity = m21._snapshot(Path(cache_dir).resolve())
    config = _embedding_config(schema)
    device, device_name, mps_available = m1._choose_device(torch)
    train_features = extract_embeddings(runtime, snapshot, train, config, device)

    head_models: dict[str, Any] = {}
    head_thresholds: dict[str, float] = {}
    cv_report: dict[str, Any] = {}
    for head in V1_HEADS:
        selected_c, models, thresholds, report = select_head_model(train_features, train, schema, head)
        head_models[head] = {"C": selected_c, "models": models}
        if head == "reasoning_tags":
            head_thresholds = thresholds
        cv_report[head] = report
    joblib.dump(
        {
            "stage_id": STAGE_ID,
            "model_id": MODEL_ID,
            "revision": REVISION,
            "feature": "official_frozen_lert_cls_l2_normalized",
            "class_order": {head: list(schema.class_order[head]) for head in V1_HEADS},
            "head_models": head_models,
            "reasoning_thresholds": head_thresholds,
        },
        output / "linear-heads.joblib",
    )
    _dump(output / "cross-validation.json", cv_report)
    _dump(output / "reasoning-thresholds.json", head_thresholds)

    # Evaluation inputs are deliberately loaded only after final models exist.
    recent_check, recent_check_audit = m12.load_recent_feedback_records(
        recent_check_labels,
        recent_feed_jsonl,
        role="cumulative_recent_check",
    )
    _require(len(recent_check) == 19, "M13_RECENT_CHECK_COUNT_INVALID", "M13 requires cumulative recent Check19", observed=len(recent_check))
    human_check, _ = m5.load_human_check_pool(pool_dir)
    human48 = terminal._load_human48(Path(human48_jsonl), Path(human48_input), schema)
    train_ids = {record.sample_id for record in train}
    _require(not train_ids.intersection(record.sample_id for record in recent_check), "M13_RECENT_CHECK_OVERLAP", "recent Check overlaps M13 fit population")

    evaluation_sets = {"recent_check19": recent_check, "human_check20": human_check, "human48": human48}
    candidate_tables: dict[str, Any] = {}
    current_tables: dict[str, Any] = {}
    current_runtime = hybrid.M12HybridInferenceRuntime.from_artifacts(
        selected_artifact=selected_artifact,
        config_path=config_path,
        cache_dir=cache_dir,
    )
    parent_bundle = torch.load(Path(selected_artifact).resolve() / "unified-runtime-bundle.pt", map_location="cpu", weights_only=False)
    current_thresholds = {label: float(parent_bundle.get("reasoning_thresholds", {}).get(label, 0.5)) for label in schema.class_order["reasoning_tags"]}
    for name, records in evaluation_sets.items():
        features = extract_embeddings(runtime, snapshot, records, config, device)
        arrays = {head: predict_head(head_models[head]["models"], features, head) for head in V1_HEADS}
        candidate_all = _metric_table(arrays, records, schema, head_thresholds)
        current_all = _metric_table(m12._baseline_probabilities(current_runtime, records, schema), records, schema, current_thresholds)
        if name == "human48":
            candidate_tables[name] = {head: candidate_all[head] for head in terminal.HUMAN48_HEADS}
            current_tables[name] = {head: current_all[head] for head in terminal.HUMAN48_HEADS}
        else:
            candidate_tables[name] = candidate_all
            current_tables[name] = current_all

    decisions = promotion_decisions(
        current_tables["recent_check19"],
        candidate_tables["recent_check19"],
        current_tables["human_check20"],
        candidate_tables["human_check20"],
        current_tables["human48"],
        candidate_tables["human48"],
    )
    promoted = [head for head, item in decisions.items() if item["promote"]]
    metrics = {"current": current_tables, "candidate": candidate_tables, "promotion": decisions}
    _dump(output / "metrics.json", metrics)
    summary = {
        "stage_id": STAGE_ID,
        "status": "PROMOTE_PER_HEAD" if promoted else "STOP_NO_STABLE_GAIN",
        "actual_device": device_name,
        "mps_available": mps_available,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "train_samples": len(train),
        "recent_check_samples": len(recent_check),
        "human_check_samples": len(human_check),
        "human48_samples": len(human48),
        "base_audit": base_audit,
        "recent_train_audit": recent_audit,
        "recent_check_audit": recent_check_audit,
        "snapshot_identity": snapshot_identity,
        "feature": "official_frozen_lert_cls_l2_normalized",
        "selected_C": {head: head_models[head]["C"] for head in V1_HEADS},
        "promoted_heads": promoted,
        "same_m12_loop_reopened": False,
        "new_manual_labels_required": False,
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
        result = run_m13(
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
        print(json.dumps({"status": "M13_BLOCKED", "error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
