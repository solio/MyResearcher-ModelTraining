from __future__ import annotations

from semantic_model import encoder_m16_weak_pretrained_multitask_lert as m16


def test_m16_has_one_bounded_pretrain_and_correction_schedule() -> None:
    assert m16.WEAK_EPOCHS == 2
    assert m16.HUMAN_MAX_EPOCHS == 4
    assert "WEAK1821" in m16.STAGE_ID


def test_m16_reuses_the_shared_multitask_architecture() -> None:
    assert m16.m15.TRAINABLE_TOP_LAYERS == 4
    assert m16.m15.BATCH_SIZE == 8
    assert m16.m15.ENCODER_LR == 2e-5
    assert m16.m15.HEAD_LR == 2e-4
