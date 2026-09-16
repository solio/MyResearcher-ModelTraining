from __future__ import annotations

from semantic_model import encoder_m1 as m1
from semantic_model import encoder_m15_shared_multitask_lert as m15


def _record(index: int, *, recent: bool, stance: str) -> m1.M1Record:
    return m1.M1Record(
        sample_id=f"{'recent' if recent else 'base'}-{index}",
        stock_code="601012",
        stock_name="隆基绿能",
        model_text="高开低走",
        label={"stance": stance},
        weights={},
    )


def test_multitask_folds_cover_every_row_once() -> None:
    stances = ["BULL", "BEAR", "NEUTRAL", "MIXED", "UNKNOWN"]
    records = [
        _record(index, recent=index >= 15, stance=stances[index % len(stances)])
        for index in range(25)
    ]
    recent_ids = {record.sample_id for record in records if record.sample_id.startswith("recent-")}

    splits = m15.multitask_fold_indices(records, recent_ids)

    assert len(splits) == 5
    assert sorted(index for _train, validation in splits for index in validation) == list(range(25))
    assert all(set(train).isdisjoint(validation) for train, validation in splits)


def test_m15_configuration_is_bounded() -> None:
    assert m15.MAX_EPOCHS == 6
    assert m15.N_FOLDS == 5
    assert m15.TRAINABLE_TOP_LAYERS == 4
    assert m15.ENCODER_LR == 2e-5
    assert m15.HEAD_LR == 2e-4


def test_source_weight_map_is_fixed_before_minibatching() -> None:
    records = [_record(0, recent=False, stance="BULL"), _record(1, recent=False, stance="BEAR"), _record(2, recent=True, stance="BEAR")]
    weights = m15.source_weight_map(records, {"recent-2"})

    assert weights["base-0"] + weights["base-1"] == weights["recent-2"]
