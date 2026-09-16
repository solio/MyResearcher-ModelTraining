from __future__ import annotations

import numpy as np

from semantic_model import encoder_m1 as m1
from semantic_model import encoder_m14_human_char_tfidf as m14


def _record(sample_id: str, text: str) -> m1.M1Record:
    return m1.M1Record(
        sample_id=sample_id,
        stock_code="601012",
        stock_name="不应进入特征",
        model_text=text,
        label={},
        weights={},
    )


def test_text_features_exclude_stock_identity() -> None:
    left = _record("base-1", "不应进入特征高开低走")
    right = m1.M1Record("recent-1", "000001", "另一只股票", "$另一只股票(SZ000001)$ 高开低走 000001", {}, {})

    assert m14._texts([left, right]) == ["高开低走", "高开低走"]


def test_source_balanced_weights_give_sources_equal_mass() -> None:
    records = [_record("base-1", "甲乙"), _record("base-2", "甲丙"), _record("recent-1", "乙丙")]

    observed = m14.source_balanced_weights(records, {"recent-1"})

    assert np.isclose(observed[:2].sum(), observed[2:].sum())
    assert np.isclose(observed.mean(), 1.0)


def test_tfidf_binary_models_support_constant_column() -> None:
    texts = ["高开低走", "低开高走", "今天看涨", "今天看跌"]
    vectorizer = m14.make_vectorizer("char_1_4")
    features = vectorizer.fit_transform(texts)
    targets = np.asarray([[0, 1], [0, 1], [1, 1], [1, 1]])

    models = m14.fit_binary_models(features, targets, 0.3, np.ones(4))
    observed = m14.predict_binary_models(models, features)

    assert observed.shape == (4, 2)
    assert np.allclose(observed[:, 1], 1.0)
    assert observed[0, 0] < observed[2, 0]
