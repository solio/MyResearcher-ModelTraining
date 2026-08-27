from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from .calibrate import (
    calibrate_multilabel_thresholds,
    calibrate_single_label_threshold,
)
from .errors import ContractError
from .models.classical import ClassicalMultiHeadModel
from .schema import SINGLE_LABEL_HEADS


def _confidence_summary(confidence: np.ndarray) -> dict[str, float]:
    if len(confidence) == 0:
        return {key: 0.0 for key in ("min", "p25", "median", "p75", "max", "mean")}
    return {
        "min": float(np.min(confidence)),
        "p25": float(np.quantile(confidence, 0.25)),
        "median": float(np.quantile(confidence, 0.5)),
        "p75": float(np.quantile(confidence, 0.75)),
        "max": float(np.max(confidence)),
        "mean": float(np.mean(confidence)),
    }


def _expected_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, *, bins: int = 10
) -> float:
    if len(confidence) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        upper_inclusive = index == bins - 1
        selected = (confidence >= edges[index]) & (
            confidence <= edges[index + 1]
            if upper_inclusive
            else confidence < edges[index + 1]
        )
        if selected.any():
            result += float(selected.mean()) * abs(
                float(correct[selected].mean()) - float(confidence[selected].mean())
            )
    return result


def calibrate_model_thresholds(
    model: ClassicalMultiHeadModel,
    texts: Sequence[str],
    labels: Sequence[Mapping[str, Any]],
    *,
    minimum_coverage: float,
    method: str = "dev-threshold-v0.1",
) -> dict[str, Any]:
    probabilities = model.predict_probabilities(texts)
    single_thresholds: dict[str, float] = {}
    for head in SINGLE_LABEL_HEADS:
        order = model.class_order[head]
        true_indices = [order.index(str(label[head])) for label in labels]
        single_thresholds[head] = calibrate_single_label_threshold(
            probabilities[head], true_indices, minimum_coverage=minimum_coverage
        )
    reasoning_order = model.class_order["reasoning_tags"]
    true_matrix = np.asarray(
        [
            [int(tag in label["reasoning_tags"]) for tag in reasoning_order]
            for label in labels
        ],
        dtype=int,
    )
    reasoning_thresholds = calibrate_multilabel_thresholds(
        probabilities["reasoning_tags"], true_matrix, method=method
    )
    return {
        "calibration_version": method,
        "minimum_coverage": float(minimum_coverage),
        "ensure_at_least_one_reasoning_tag": method == "reference-v0.3.5",
        "single_label": single_thresholds,
        "reasoning_tags": dict(zip(reasoning_order, reasoning_thresholds, strict=True)),
    }


def evaluate_model(
    model: ClassicalMultiHeadModel,
    texts: Sequence[str],
    labels: Sequence[Mapping[str, Any]],
    sample_ids: Sequence[str],
    thresholds: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not (len(texts) == len(labels) == len(sample_ids)):
        raise ContractError("EVALUATION_INPUT_INVALID", "evaluation inputs must align")
    probabilities = model.predict_probabilities(texts)
    metrics: dict[str, Any] = {}
    errors: list[dict[str, Any]] = []
    for head in SINGLE_LABEL_HEADS:
        order = model.class_order[head]
        true = [str(label[head]) for label in labels]
        predicted_indices = probabilities[head].argmax(axis=1)
        predicted = [order[int(index)] for index in predicted_indices]
        precision, recall, f1, support = precision_recall_fscore_support(
            true,
            predicted,
            labels=order,
            zero_division=0,
        )
        confidence = probabilities[head].max(axis=1)
        true_indices = np.asarray([order.index(value) for value in true], dtype=int)
        selected_true_probabilities = probabilities[head][
            np.arange(len(true_indices)), true_indices
        ]
        threshold = float(thresholds["single_label"][head])
        abstained = confidence < threshold
        correct = np.asarray([left == right for left, right in zip(true, predicted)])
        metrics[head] = {
            "task_type": "single_label",
            "loss": float(
                -np.mean(np.log(np.clip(selected_true_probabilities, 1e-15, 1.0)))
            ),
            "macro_f1": float(f1_score(true, predicted, labels=order, average="macro", zero_division=0)),
            "per_class": {
                label: {
                    "support": int(support[index]),
                    "precision": float(precision[index]),
                    "recall": float(recall[index]),
                    "f1": float(f1[index]),
                }
                for index, label in enumerate(order)
            },
            "confusion_matrix": confusion_matrix(true, predicted, labels=order).tolist(),
            "class_order": order,
            "confidence": _confidence_summary(confidence),
            "calibration": {
                "ece_10_bin": _expected_calibration_error(confidence, correct)
            },
            "abstention": {
                "threshold": threshold,
                "count": int(abstained.sum()),
                "coverage": float((~abstained).mean()) if len(abstained) else 0.0,
            },
        }
        for index, (truth, prediction) in enumerate(zip(true, predicted)):
            if truth != prediction or bool(abstained[index]):
                errors.append(
                    {
                        "sample_id": sample_ids[index],
                        "head": head,
                        "truth": truth,
                        "prediction": prediction,
                        "confidence": float(confidence[index]),
                        "abstained": bool(abstained[index]),
                        "probabilities": {
                            label: float(probabilities[head][index, class_index])
                            for class_index, label in enumerate(order)
                        },
                    }
                )

    reasoning_order = model.class_order["reasoning_tags"]
    truth_matrix = np.asarray(
        [
            [int(tag in label["reasoning_tags"]) for tag in reasoning_order]
            for label in labels
        ],
        dtype=int,
    )
    threshold_array = np.asarray(
        [float(thresholds["reasoning_tags"][tag]) for tag in reasoning_order]
    )
    predicted_matrix = (
        probabilities["reasoning_tags"] >= threshold_array.reshape(1, -1)
    ).astype(int)
    if thresholds.get("ensure_at_least_one_reasoning_tag"):
        empty_rows = np.where(predicted_matrix.sum(axis=1) == 0)[0]
        if len(empty_rows):
            predicted_matrix[
                empty_rows,
                np.argmax(probabilities["reasoning_tags"][empty_rows], axis=1),
            ] = 1
    per_class = {}
    binary_confusions = {}
    for index, tag in enumerate(reasoning_order):
        precision, recall, f1, support = precision_recall_fscore_support(
            truth_matrix[:, index],
            predicted_matrix[:, index],
            labels=[0, 1],
            zero_division=0,
        )
        per_class[tag] = {
            "support": int(support[1]),
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
            "threshold": float(threshold_array[index]),
        }
        binary_confusions[tag] = confusion_matrix(
            truth_matrix[:, index], predicted_matrix[:, index], labels=[0, 1]
        ).tolist()
    clipped = np.clip(probabilities["reasoning_tags"], 1e-15, 1 - 1e-15)
    binary_cross_entropy = -np.mean(
        truth_matrix * np.log(clipped) + (1 - truth_matrix) * np.log(1 - clipped)
    )
    metrics["reasoning_tags"] = {
        "task_type": "multi_label",
        "loss": float(binary_cross_entropy),
        "micro_f1": float(
            f1_score(truth_matrix, predicted_matrix, average="micro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(truth_matrix, predicted_matrix, average="macro", zero_division=0)
        ),
        "per_class": per_class,
        "confusion_matrices": binary_confusions,
        "class_order": reasoning_order,
        "confidence": _confidence_summary(probabilities["reasoning_tags"].ravel()),
        "abstention": {
            "rows_with_no_selected_tag": int((predicted_matrix.sum(axis=1) == 0).sum())
        },
    }
    for row_index, sample_id in enumerate(sample_ids):
        truth_tags = {
            reasoning_order[index]
            for index, value in enumerate(truth_matrix[row_index])
            if value
        }
        predicted_tags = {
            reasoning_order[index]
            for index, value in enumerate(predicted_matrix[row_index])
            if value
        }
        if truth_tags != predicted_tags:
            errors.append(
                {
                    "sample_id": sample_id,
                    "head": "reasoning_tags",
                    "truth": sorted(truth_tags),
                    "prediction": sorted(predicted_tags),
                    "abstained": not predicted_tags,
                    "probabilities": {
                        tag: float(probabilities["reasoning_tags"][row_index, index])
                        for index, tag in enumerate(reasoning_order)
                    },
                }
            )
    return metrics, errors
