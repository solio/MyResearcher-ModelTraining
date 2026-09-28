from pathlib import Path

from semantic_model import encoder_action_contrastive as contrast
from semantic_model.encoder_m1 import M1Record
from semantic_model.schema import LabelSchema


ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_examples_are_separate_from_real_train_and_probe() -> None:
    schema = LabelSchema.load(ROOT / "schema" / "semantic-schema-calibrated-v0.2.1.json")
    human = [M1Record(f"human-{i}", f"60{i:04d}", f"股票{i}", f"真实帖子{i}", {contrast.HEAD: "WATCH"}, {contrast.HEAD: 1.0}) for i in range(10)]
    train, probes = contrast._examples(schema, human)
    assert len(train) == 2 * 2 * len(contrast.TRAIN_PAIRS)
    assert len(probes) == len(contrast.HELD_OUT_PROBES)
    assert len({row.sample_id for row in train}) == len(train)
    assert {row.label[contrast.HEAD] for row in train} <= set(schema.class_order[contrast.HEAD])
    assert {contrast.prior._normalized(row.model_text) for row in train}.isdisjoint(
        contrast.prior._normalized(row.model_text) for row in probes
    )
    assert all(row.weights[contrast.HEAD] == contrast.SYNTHETIC_WEIGHT for row in train)


def test_three_way_direction_keeps_nearby_classes_together() -> None:
    assert contrast._three_way("BUY") == contrast._three_way("ADD")
    assert contrast._three_way("REDUCE") == contrast._three_way("SELL")
    assert contrast._three_way("BUY") != contrast._three_way("SELL")
    assert contrast._three_way("HOLD") == contrast._three_way("WATCH")


def test_finite_natural_silver_selection_is_distinct_from_human_gold() -> None:
    schema = LabelSchema.load(ROOT / "schema" / "semantic-schema-calibrated-v0.2.1.json")
    prefixes = [prefix for prefix, _label in (*contrast.SILVER_SELECTION, *contrast.ROLE_NEGATIVE_SILVER)]
    probe_prefixes = [prefix for prefix, _label in contrast.NATURAL_ROLE_PROBES]
    assert len(prefixes) == len(set(prefixes))
    assert len(probe_prefixes) == len(set(probe_prefixes))
    assert set(prefixes).isdisjoint(probe_prefixes)
    assert all(label in schema.class_order[contrast.HEAD] for _prefix, label in (*contrast.SILVER_SELECTION, *contrast.ROLE_NEGATIVE_SILVER, *contrast.NATURAL_ROLE_PROBES))
    assert contrast.SILVER_WEIGHT < 1.0


def test_evidence_reweight_preserves_owner_answer_and_text() -> None:
    prefix, label, _reason = contrast.LOW_EVIDENCE_TRAIN[0]
    rows = [M1Record(prefix + "-full", "600000", "股票", "原文不改", {contrast.HEAD: label}, {contrast.HEAD: 1.0})]
    # Use a local one-row selection to exercise the transform without
    # requiring the full historical Human Train pool in the unit test.
    original = contrast.LOW_EVIDENCE_TRAIN
    try:
        contrast.LOW_EVIDENCE_TRAIN = original[:1]
        weighted, audit = contrast._evidence_weighted_human(rows)
    finally:
        contrast.LOW_EVIDENCE_TRAIN = original
    assert weighted[0].label == rows[0].label
    assert weighted[0].model_text == rows[0].model_text
    assert weighted[0].weights[contrast.HEAD] == contrast.LOW_EVIDENCE_WEIGHT
    assert audit[0]["answer_preserved"] == label
