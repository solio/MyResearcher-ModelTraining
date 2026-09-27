import json

from tools import market_extreme_refresh as refresh


def test_input_key_preserves_the_actual_model_inputs():
    row = {"sample_id": "old", "stock_code": "300054", "stock_name": "鼎龙股份", "model_text": "先看一下"}
    assert refresh.input_key(row) == refresh.input_key(dict(row, sample_id="new"))
    assert refresh.input_key(row) != refresh.input_key(dict(row, model_text="先卖一下"))
    assert refresh.input_key(row) != refresh.input_key(dict(row, stock_code="601012"))
    assert refresh.input_key(row) != refresh.input_key(dict(row, stock_name="不同名字"))


def test_identical_input_reuses_probabilities_without_loading_a_model(tmp_path, monkeypatch):
    row = {"sample_id": "new", "stock_code": "300054", "stock_name": "鼎龙股份", "model_text": "先看一下"}
    donor = {"sample_id": "old", "predictions": {"stance": {"prediction": "NEUTRAL", "probabilities": {"NEUTRAL": 1.0}}}}
    monkeypatch.setattr(refresh, "existing_probabilities", lambda *args: {refresh.input_key(row): donor})
    audit = refresh.refreshed_predictions([row], tmp_path, "cpu")
    assert audit == {"reused_exact_inputs": 1, "new_inference_posts": 0, "device": None}
    result = json.loads((tmp_path / "m12-predictions.jsonl").read_text())
    assert result["sample_id"] == "new"
    assert result["predictions"] == donor["predictions"]
    assert donor["sample_id"] == "old"
