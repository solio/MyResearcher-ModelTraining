from semantic_model import prepare, train


BLOCKED = {
    "status": "BLOCKED_MISSING_CANONICAL_ARTIFACTS",
    "training_allowed": False,
    "blocker_codes": ["BLOCKED_MISSING_CANONICAL_ARTIFACTS"],
}


def test_prepare_stops_at_audit_without_creating_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "run_audit", lambda _: (BLOCKED, 2))
    result, exit_code = prepare.run_prepare(tmp_path / "missing.yaml")
    assert exit_code == 2
    assert result == BLOCKED
    assert list(tmp_path.iterdir()) == []


def test_train_stops_at_audit_without_creating_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "run_audit", lambda _: (BLOCKED, 2))
    result, exit_code = train.run_train(tmp_path / "missing.yaml")
    assert exit_code == 2
    assert result == BLOCKED
    assert list(tmp_path.iterdir()) == []

