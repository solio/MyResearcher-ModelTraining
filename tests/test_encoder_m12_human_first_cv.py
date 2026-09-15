from __future__ import annotations

import numpy as np

from semantic_model import encoder_m1 as m1
from semantic_model import encoder_m12_human_first_cv as m12


def _record(index: int, label: str, reasoning: list[str] | None = None) -> m1.M1Record:
    labels = {
        "target_mode": label,
        "reasoning_tags": reasoning or ["NO_REASON_GIVEN"],
    }
    return m1.M1Record(str(index), "000001", "示例", f"文本{index}", labels, {head: 1.0 for head in m12.V1_HEADS})


def test_fold_indices_cover_every_sample_once() -> None:
    records = [_record(index, "ON_TARGET" if index % 3 else "CROSS_TARGET") for index in range(100)]
    folds = m12.fold_indices(records, "target_mode")
    assert len(folds) == 5
    assert sorted(index for _train, validation in folds for index in validation) == list(range(100))
    assert all(not (set(train) & set(validation)) for train, validation in folds)


def test_reasoning_threshold_calibration_is_model_specific() -> None:
    records = [
        _record(0, "ON_TARGET", ["NO_REASON_GIVEN"]),
        _record(1, "ON_TARGET", ["TECHNICAL_PRICE"]),
        _record(2, "ON_TARGET", ["TECHNICAL_PRICE"]),
        _record(3, "ON_TARGET", ["NO_REASON_GIVEN"]),
    ]
    order = ["NO_REASON_GIVEN", "TECHNICAL_PRICE"]
    probabilities = np.asarray([[0.8, 0.2], [0.3, 0.7], [0.2, 0.8], [0.7, 0.3]])
    thresholds = m12.calibrate_reasoning_thresholds(probabilities, records, order)
    assert set(thresholds) == set(order)
    assert all(0.1 <= value <= 0.9 for value in thresholds.values())
