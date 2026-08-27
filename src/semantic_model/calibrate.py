from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .errors import ContractError


def calibrate_single_label_threshold(
    probabilities: np.ndarray,
    true_indices: Sequence[int],
    *,
    minimum_coverage: float = 0.8,
) -> float:
    if probabilities.ndim != 2 or len(probabilities) != len(true_indices):
        raise ContractError(
            "CALIBRATION_INPUT_INVALID", "probability/label dimensions do not agree"
        )
    if not 0 < minimum_coverage <= 1:
        raise ContractError(
            "CALIBRATION_INPUT_INVALID", "minimum_coverage must be in (0, 1]"
        )
    if len(true_indices) == 0:
        return 1.0
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    truth = np.asarray(true_indices, dtype=int)
    candidates = sorted({0.0, 1.0, *confidence.tolist()})
    best = (float("-inf"), 0.0)
    for threshold in candidates:
        covered = confidence >= threshold
        coverage = float(covered.mean())
        if coverage < minimum_coverage or not covered.any():
            continue
        accuracy = float((predicted[covered] == truth[covered]).mean())
        candidate = (accuracy, float(threshold))
        if candidate > best:
            best = candidate
    return best[1] if best[0] != float("-inf") else 0.0


def _binary_f1(truth: np.ndarray, predicted: np.ndarray) -> float:
    true_positive = int(np.logical_and(truth == 1, predicted == 1).sum())
    false_positive = int(np.logical_and(truth == 0, predicted == 1).sum())
    false_negative = int(np.logical_and(truth == 1, predicted == 0).sum())
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2 * true_positive / denominator


def calibrate_multilabel_thresholds(
    probabilities: np.ndarray, true_matrix: np.ndarray
) -> list[float]:
    if probabilities.ndim != 2 or probabilities.shape != true_matrix.shape:
        raise ContractError(
            "CALIBRATION_INPUT_INVALID", "multi-label dimensions do not agree"
        )
    thresholds: list[float] = []
    for column in range(probabilities.shape[1]):
        truth = true_matrix[:, column].astype(int)
        scores = probabilities[:, column]
        if truth.sum() == 0:
            thresholds.append(1.0)
            continue
        candidates = sorted({0.0, 0.5, 1.0, *scores.tolist()})
        best = (float("-inf"), 0.5)
        for threshold in candidates:
            predicted = (scores >= threshold).astype(int)
            candidate = (_binary_f1(truth, predicted), float(threshold))
            if candidate > best:
                best = candidate
        thresholds.append(best[1])
    return thresholds


def single_label_decisions(
    probabilities: np.ndarray, class_order: Sequence[str], threshold: float
) -> list[dict[str, object]]:
    decisions = []
    for row in probabilities:
        index = int(np.argmax(row))
        confidence = float(row[index])
        decisions.append(
            {
                "label": class_order[index],
                "confidence": confidence,
                "abstained": confidence < threshold,
            }
        )
    return decisions

