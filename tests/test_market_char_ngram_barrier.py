"""Focused leakage and sparse-feature tests for the one-off TF-IDF trial."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from tools import market_char_ngram_barrier as trial


def test_h10_positive_is_endpoint_not_first_barrier():
    calendar = [f"d{i:02d}" for i in range(11)]
    qfq = {("000001", day): {"open": 100.0, "close": 100.0} for day in calendar}
    qfq[("000001", "d00")]["close"] = 94.0
    qfq[("000001", "d09")]["close"] = 101.0
    row = {"stock_code": "000001", "entry_date": "d00", "outcome_date": "d09"}
    assert trial.h10_positive_label(row, calendar, qfq) == 1
    qfq[("000001", "d10")]["close"] = 1.0
    assert trial.h10_positive_label(row, calendar, qfq) == 1
    row["outcome_date"] = "d10"
    with pytest.raises(ValueError, match="horizon"):
        trial.h10_positive_label(row, calendar, qfq)


def test_full_daily_titles_and_train_only_tfidf_vocabulary():
    bags = {("a", "d0"): ["价格大涨", "情绪沸腾"],
            ("a", "d1"): ["价格大跌", "情绪恐慌"]}
    docs = trial.daily_documents(bags)
    assert all(title in docs[("a", "d0")] for title in bags[("a", "d0")])
    # The real configuration's min_df=5 is inappropriate for a two-document
    # unit fixture; temporarily use its fixed helper on replicated train docs.
    fit = {("a", f"d{i}"): "价格大涨 情绪沸腾" for i in range(5)}
    fit[("a", "future")] = "未来独有词"
    vectorizer = trial.fit_daily_tfidf(set(fit) - {("a", "future")}, fit)
    assert vectorizer.transform(["未来独有词"]).nnz == 0
    assert vectorizer.transform(["价格大涨"]).nnz > 0


def test_history_sparse_n20_uses_only_pre_entry_daily_documents():
    calendar = [f"d{i:02d}" for i in range(21)]
    daily = {("a", day): sparse.csr_matrix([[1.0, float(i % 2)]], dtype=np.float32)
             for i, day in enumerate(calendar)}
    row = {"stock_code": "a", "entry_date": "d19"}
    original = trial.history_matrix([row], calendar, daily, 2)
    assert original.shape == (1, 4)
    assert sparse.issparse(original)
    daily[("a", "d20")] = sparse.csr_matrix([[999.0, 999.0]], dtype=np.float32)
    changed = trial.history_matrix([row], calendar, daily, 2)
    assert np.allclose(original.toarray(), changed.toarray())


def test_same_stock_shift_stays_within_each_stock():
    rows = [{"stock_code": code, "entry_date": f"d{i:02d}"}
            for code in ("a", "b") for i in range(12)]
    shifted = trial.shifted_indices(rows, shift=3)
    assert len(shifted) == len(rows)
    for index, source_index in enumerate(shifted):
        assert rows[index]["stock_code"] == rows[source_index]["stock_code"]
        assert index != source_index


def test_sparse_offset_fit_keeps_market_offset_fixed():
    x = sparse.csr_matrix([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    y = np.asarray([1, 0, 1, 0])
    offset = np.zeros(4)
    beta, status = trial.sparse_offset_fit(x, y, offset)
    assert beta[0] > 0 and beta[1] < 0
    assert status["iterations"] > 0
    assert np.array_equal(offset, np.zeros(4))
