import numpy as np

from semantic_model.calibrate import (
    calibrate_multilabel_thresholds,
    calibrate_single_label_threshold,
    single_label_decisions,
)


def test_low_confidence_is_explicit_abstention_not_label_rewrite():
    probabilities = np.asarray([[0.35, 0.40, 0.25]])
    decisions = single_label_decisions(
        probabilities, ["BEAR", "NEUTRAL", "UNKNOWN"], threshold=0.5
    )
    assert decisions == [
        {"label": "NEUTRAL", "confidence": 0.4, "abstained": True}
    ]
    # NEUTRAL remains the raw argmax; abstention is an independent decision bit.


def test_dev_calibration_respects_minimum_coverage():
    probabilities = np.asarray([[0.9, 0.1], [0.55, 0.45], [0.4, 0.6]])
    threshold = calibrate_single_label_threshold(
        probabilities, [0, 0, 1], minimum_coverage=2 / 3
    )
    assert 0.0 <= threshold <= 1.0
    assert float((probabilities.max(axis=1) >= threshold).mean()) >= 2 / 3


def test_absent_reasoning_class_calibrates_to_no_false_positive():
    probabilities = np.asarray([[0.1, 0.8], [0.2, 0.7]])
    truth = np.asarray([[0, 1], [0, 1]])
    thresholds = calibrate_multilabel_thresholds(probabilities, truth)
    assert thresholds[0] == 1.0

