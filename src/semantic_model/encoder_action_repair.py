"""One bounded Action candidate from existing human labels and sparse owner review.

This does not alter the selected/M12 artifacts.  Epoch selection uses only
human Train out-of-fold predictions; the later owner Check is opened after the
candidate checkpoint has been saved.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import encoder_m1 as m1
from . import encoder_m2_1_softlabel as m21
from . import encoder_m2_lert_final as lert
from . import encoder_m2_selected_runtime as selected
from . import encoder_m5_human100_source_balanced as m5
from . import encoder_m12_human_first_cv as m12
from .errors import ContractError
from .schema import LabelSchema


HEAD = "action_tendency"
STAGE_ID = "SEMANTIC-ACTION-REPAIR-CANDIDATE"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = Path(__file__).resolve().parents[2]
LABEL_ROOT = PROJECT_ROOT / "pipeline-runs" / "semantic-repair-label-20260928"
RECENT_ROOT = PROJECT_ROOT / "pipeline-runs" / "recent-chain-20260915"
DEFAULT_OUTPUT = PROJECT_ROOT / "model-artifacts" / "semantic-action-repair-candidate-20260929"
MAX_EPOCHS = 12
FOLDS = 5
BATCH_SIZE = 16
SEED = 35
BLOCK_LR = 1e-5
HEAD_LR = 2e-4
WALL_SECONDS = 2 * 60 * 60

# Do not change the owner's answers. These three answers conflict with the
# frozen author-action definition and are reported verbatim, not used as Gold.
TRAIN_EXCLUSIONS = {
    "msv1-34c24af500c4e0837a3a31f77c8b5d6b19c0a71bc2fda6c34a36260716969a80": "friend_sold_but_author_holds",
    "msv1-4bedecd90a22e1073e3bb224dbfe61b79bb25e2490d37b96248179dbbc1c3d6b": "other_investors_hold_not_author",
}
CHECK_EXCLUSIONS = {
    "msv1-4707710bf345fb916df0f07910ecfb24a5e217f6a2774644e5cb624149d0c1e8": "advice_to_others_not_author_hold",
}


def _require(ok: bool, code: str, message: str, **details: Any) -> None:
    if not ok:
        raise ContractError(code, message, **details)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    _require(path.is_file(), "ACTION_REPAIR_INPUT_MISSING", "required input is missing", path=str(path))
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sparse_records(role: str, schema: LabelSchema) -> tuple[list[m1.M1Record], list[dict[str, Any]]]:
    root = LABEL_ROOT
    assignments = _read_jsonl(root / f"{role}-assignments.jsonl")
    finals = _read_jsonl(root / f"{role}-final.jsonl")
    indexed: dict[str, dict[str, Any]] = {}
    for item in assignments:
        sid = item.get("sample_id")
        _require(isinstance(sid, str) and sid not in indexed, "ACTION_REPAIR_ASSIGNMENT_ID", "duplicate or invalid assignment id", sample_id=sid)
        indexed[sid] = item
    action_ids = {sid for sid, item in indexed.items() if item.get("heads") == [HEAD]}
    observed: dict[str, dict[str, Any]] = {}
    for row in finals:
        sid = row.get("sample_id")
        _require(sid in indexed and sid not in observed, "ACTION_REPAIR_FINAL_ID", "unknown or duplicate final id", sample_id=sid)
        _require(row.get("head") == indexed[sid].get("heads", [None])[0], "ACTION_REPAIR_FINAL_HEAD", "final head differs from assignment", sample_id=sid)
        _require(row.get("is_final") is True and row.get("disposition") is None, "ACTION_REPAIR_NOT_FINAL", "assignment lacks a usable final answer", sample_id=sid)
        _require(row.get("schema_version") == "semantic-schema-calibrated-v0.2.1", "ACTION_REPAIR_SCHEMA", "label schema differs", sample_id=sid)
        _require(row.get("metadata", {}).get("source_item_id") == indexed[sid].get("metadata", {}).get("source_item_id"), "ACTION_REPAIR_IDENTITY", "source identity differs", sample_id=sid)
        observed[sid] = row
    _require(set(indexed) == set(observed), "ACTION_REPAIR_FINAL_COVERAGE", "not all assignments have final answers", role=role)
    exclusions = TRAIN_EXCLUSIONS if role == "train" else CHECK_EXCLUSIONS
    _require(set(exclusions) <= action_ids, "ACTION_REPAIR_EXCLUSION_INVALID", "exclusion is not an Action assignment", role=role)
    records: list[m1.M1Record] = []
    audit: list[dict[str, Any]] = []
    for sid in sorted(action_ids):
        item, row = indexed[sid], observed[sid]
        answer = row.get("answer")
        _require(answer in schema.class_order[HEAD], "ACTION_REPAIR_ANSWER_INVALID", "answer is outside Action taxonomy", sample_id=sid, answer=answer)
        if sid in exclusions:
            audit.append({"sample_id": sid, "answer_preserved": answer, "reason": exclusions[sid]})
            continue
        metadata = item["metadata"]
        records.append(m1.M1Record(sid, str(metadata["stock_code"]), str(metadata["stock_name"]), str(item["text"]), {HEAD: answer}, {HEAD: 1.0}))
    return records, audit


def _legacy_train() -> tuple[list[m1.M1Record], dict[str, Any]]:
    base, _ = m5.load_human_train_pool(REPO_ROOT / "runs" / "feedback-loop-v3-finalized-20260910")
    base, base_audit = m12.exclude_dependency_conflicts(base, role="base_human_train")
    recent, recent_audit = m12.load_recent_feedback_records(
        RECENT_ROOT / "human" / "recent-train50-final.jsonl",
        RECENT_ROOT / "feed" / "feed-input.jsonl",
        role="recent_train",
        expected_samples=50,
        exclude_dependency_conflicts=True,
    )
    result = [*base, *recent]
    _require(len(result) == 147 and len({r.sample_id for r in result}) == 147, "ACTION_REPAIR_LEGACY_TRAIN", "expected 147 distinct existing human Train records", observed=len(result))
    return result, {"base": base_audit, "recent": recent_audit}


def _normalized(text: str) -> str:
    return re.sub(r"[^\w]+", "", text, flags=re.UNICODE).lower()


def _validate_train(records: Sequence[m1.M1Record], new_records: Sequence[m1.M1Record]) -> None:
    ids = [r.sample_id for r in records]
    _require(len(ids) == len(set(ids)), "ACTION_REPAIR_TRAIN_ID_OVERLAP", "duplicate training sample id")
    text_groups: dict[str, list[str]] = {}
    for r in records:
        text_groups.setdefault(_normalized(r.model_text), []).append(r.sample_id)
    mixed = {key: sids for key, sids in text_groups.items() if len(sids) > 1 and any(s in {r.sample_id for r in new_records} for s in sids)}
    _require(not mixed, "ACTION_REPAIR_NEW_TEXT_OVERLAP", "new training text duplicates an existing training text", duplicates=mixed)


def _batch(torch: Any, tokenizer: Any, records: Sequence[m1.M1Record], config: Mapping[str, Any], schema: LabelSchema, device: Any) -> dict[str, Any]:
    rows = [m1.build_input_ids(tokenizer, record, config) for record in records]
    pad = tokenizer.pad_token_id if isinstance(tokenizer.pad_token_id, int) else 0
    width = max(map(len, rows))
    order = list(schema.class_order[HEAD])
    return {
        "input_ids": torch.tensor([row + [pad] * (width - len(row)) for row in rows], dtype=torch.long, device=device),
        "attention_mask": torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in rows], dtype=torch.long, device=device),
        "labels": torch.tensor([order.index(str(r.label[HEAD])) for r in records], dtype=torch.long, device=device),
    }


def _make_model(torch: Any, AutoModel: Any, snapshot: Path, schema: LabelSchema, config: Mapping[str, Any], parent: Mapping[str, Any], device: Any) -> Any:
    model = lert._make_model(torch, AutoModel, snapshot, schema, float(config["head_dropout"]))
    model.specialist_blocks[HEAD].load_state_dict(parent["lert_specialist_block_state_dict"][HEAD])
    model.heads[HEAD].load_state_dict(parent["lert_heads_state_dict"][HEAD])
    model.set_active_head(HEAD)
    return model.to(device)


def _predict(torch: Any, model: Any, tokenizer: Any, records: Sequence[m1.M1Record], config: Mapping[str, Any], schema: LabelSchema, device: Any) -> np.ndarray:
    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(records), BATCH_SIZE):
            batch = _batch(torch, tokenizer, records[start:start + BATCH_SIZE], config, schema, device)
            logits = model(batch["input_ids"], batch["attention_mask"])[HEAD]
            pieces.append(torch.softmax(logits, dim=-1).detach().cpu().numpy())
    return np.concatenate(pieces)


def _fit(torch: Any, AutoModel: Any, tokenizer: Any, snapshot: Path, schema: LabelSchema, config: Mapping[str, Any], parent: Mapping[str, Any], records: Sequence[m1.M1Record], validation: Sequence[m1.M1Record], *, epochs: int, seed: int, device: Any, started: float) -> tuple[list[np.ndarray], Any]:
    m12._set_seed(torch, seed)
    model = _make_model(torch, AutoModel, snapshot, schema, config, parent, device)
    block = list(model.specialist_blocks[HEAD].parameters())
    classifier = list(model.heads[HEAD].parameters())
    optimizer = torch.optim.AdamW([{"params": block, "lr": BLOCK_LR}, {"params": classifier, "lr": HEAD_LR}], weight_decay=0.01)
    balance = m12._balance(torch, records, schema, HEAD, device)
    curve = []
    for epoch in range(1, epochs + 1):
        _require(time.monotonic() - started < WALL_SECONDS, "ACTION_REPAIR_TIME_LIMIT", "training exceeded two hours")
        shuffled = list(records)
        random.Random(seed + epoch).shuffle(shuffled)
        model.train(True)
        for start in range(0, len(shuffled), BATCH_SIZE):
            batch = _batch(torch, tokenizer, shuffled[start:start + BATCH_SIZE], config, schema, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch["attention_mask"])[HEAD]
            loss = torch.nn.functional.cross_entropy(logits, batch["labels"], weight=balance)
            _require(bool(torch.isfinite(loss).item()), "ACTION_REPAIR_NONFINITE_LOSS", "non-finite Action loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*block, *classifier], 1.0)
            optimizer.step()
        if validation:
            curve.append(_predict(torch, model, tokenizer, validation, config, schema, device))
    return curve, model


def _baseline_check(path: Path, records: Sequence[m1.M1Record], schema: LabelSchema) -> np.ndarray:
    wanted = {r.sample_id for r in records}
    found = {}
    for row in _read_jsonl(path):
        sid = row.get("sample_id")
        if sid in wanted:
            found[sid] = row["predictions"][HEAD]["probabilities"]
    _require(set(found) == wanted, "ACTION_REPAIR_BASELINE_MISSING", "baseline snapshot lacks Check predictions", missing=sorted(wanted - set(found)))
    return np.asarray([[float(found[r.sample_id][label]) for label in schema.class_order[HEAD]] for r in records], dtype=float)


def run(*, output_dir: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    _require(not output.exists(), "ACTION_REPAIR_OUTPUT_EXISTS", "candidate output already exists", path=str(output))
    project = m1.ProjectConfig.load(m12.DEFAULT_CONFIG)
    schema, _weak, _dev = m1.load_m1_partitions(project)
    legacy, legacy_audit = _legacy_train()
    additions, train_exclusions = _sparse_records("train", schema)
    train = [*legacy, *additions]
    _validate_train(train, additions)
    _require(len(additions) == 12 and len(train) == 159, "ACTION_REPAIR_TRAIN_COUNT", "unexpected Action fit population", observed=len(train))
    started = time.monotonic()
    _np, torch, AutoModel, AutoTokenizer = m21._load_runtime()
    snapshot, identity = m21._snapshot(m12.DEFAULT_CACHE)
    parent_path = m12.DEFAULT_SELECTED_ARTIFACT / "unified-runtime-bundle.pt"
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    _require(HEAD in parent["lert_specialist_block_state_dict"] and HEAD in parent["lert_heads_state_dict"], "ACTION_REPAIR_PARENT_MISSING", "selected Action weights missing")
    frozen = lert._contract_requirements(selected.CONTRACT_PATH)
    config = lert._config(frozen["contract"], schema.class_order)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True)
    device, device_name, _mps = m1._choose_device(torch)
    oof = np.full((MAX_EPOCHS, len(train), len(schema.class_order[HEAD])), np.nan, dtype=np.float32)
    for fold, (fit_indices, val_indices) in enumerate(m12.fold_indices(train, HEAD, folds=FOLDS, seed=SEED)):
        fit = [train[i] for i in fit_indices]
        val = [train[i] for i in val_indices]
        curve, model = _fit(torch, AutoModel, tokenizer, snapshot, schema, config, parent, fit, val, epochs=MAX_EPOCHS, seed=SEED + fold, device=device, started=started)
        for epoch, probabilities in enumerate(curve):
            oof[epoch, val_indices, :] = probabilities
        del model
        gc.collect()
        if device_name == "mps":
            torch.mps.empty_cache()
        print(json.dumps({"event": "fold_complete", "fold": fold + 1, "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
    _require(bool(np.isfinite(oof).all()), "ACTION_REPAIR_OOF_INCOMPLETE", "incomplete out-of-fold predictions")
    epoch, oof_metric, _thresholds, curve = m12._select_epoch(oof, train, schema, HEAD)
    _unused_curve, model = _fit(torch, AutoModel, tokenizer, snapshot, schema, config, parent, train, [], epochs=epoch, seed=SEED + 100, device=device, started=started)
    output.mkdir(parents=True)
    torch.save({"stage_id": STAGE_ID, "head": HEAD, "selected_epoch": epoch, "parent_artifact": str(m12.DEFAULT_SELECTED_ARTIFACT), "specialist_block_state_dict": {k: v.detach().cpu() for k, v in model.specialist_blocks[HEAD].state_dict().items()}, "head_state_dict": {k: v.detach().cpu() for k, v in model.heads[HEAD].state_dict().items()}}, output / "action-checkpoint.pt")
    m12._dump(output / "training-config.json", {"stage_id": STAGE_ID, "head": HEAD, "fit_population": len(train), "legacy_human": len(legacy), "new_human": len(additions), "train_exclusions": train_exclusions, "legacy_audit": legacy_audit, "class_counts": dict(Counter(str(r.label[HEAD]) for r in train)), "folds": FOLDS, "max_epochs": MAX_EPOCHS, "selected_epoch": epoch, "device": device_name, "model_id": m12.MODEL_ID, "revision": m12.REVISION, "source_snapshot": str(snapshot), "source_identity": identity, "parent_artifact": str(m12.DEFAULT_SELECTED_ARTIFACT), "not_used_for_fit_or_epoch_selection": ["new owner Check8", "weak Dev", "Test", "Anchor", "Gold", "OOD", "market prices"]})
    m12._dump(output / "cross-validation.json", {"oof_metric": oof_metric, "curve": curve})
    # The new owner Check is not loaded until the candidate weights are frozen.
    check, check_exclusions = _sparse_records("check", schema)
    _require(len(check) == 5, "ACTION_REPAIR_CHECK_COUNT", "unexpected Action Check population", observed=len(check))
    train_ids = {r.sample_id for r in train}
    _require(not train_ids.intersection(r.sample_id for r in check), "ACTION_REPAIR_CHECK_ID_OVERLAP", "Train and Check overlap")
    train_text = {_normalized(r.model_text) for r in train}
    _require(not train_text.intersection(_normalized(r.model_text) for r in check), "ACTION_REPAIR_CHECK_TEXT_OVERLAP", "Train and Check share normalized text")
    model.eval()
    candidate = _predict(torch, model, tokenizer, check, config, schema, device)
    baseline = _baseline_check(PROJECT_ROOT / "pipeline-runs" / "market-extreme-unenriched-20260928" / "source" / "m12-predictions.jsonl", check, schema)
    candidate_metrics = m12._metrics(candidate, check, schema, HEAD)
    baseline_metrics = m12._metrics(baseline, check, schema, HEAD)
    order = list(schema.class_order[HEAD])
    rows = [{"sample_id": r.sample_id, "text": r.model_text, "human": r.label[HEAD], "current": order[int(np.argmax(baseline[i]))], "candidate": order[int(np.argmax(candidate[i]))], "current_p_human": float(baseline[i, order.index(r.label[HEAD])]), "candidate_p_human": float(candidate[i, order.index(r.label[HEAD])])} for i, r in enumerate(check)]
    result = {"status": "CANDIDATE_TRAINED_NOT_PROMOTED", "head": HEAD, "fit_samples": len(train), "new_fit_samples": len(additions), "check_samples": len(check), "check_exclusions": check_exclusions, "selected_epoch": epoch, "oof_macro_f1": oof_metric["macro_f1"], "current_check": baseline_metrics, "candidate_check": candidate_metrics, "check_rows": rows, "elapsed_seconds": round(time.monotonic() - started, 2)}
    m12._dump(output / "summary.json", result)
    print(json.dumps({"event": "complete", "output": str(output), "status": result["status"], "check_current_accuracy": baseline_metrics["accuracy"], "check_candidate_accuracy": candidate_metrics["accuracy"]}, ensure_ascii=False), flush=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train one isolated Action repair candidate")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        run(output_dir=args.output_dir)
    except ContractError as exc:
        print(json.dumps({"status": "BLOCKED", "code": exc.code, "details": exc.details}, ensure_ascii=False), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
