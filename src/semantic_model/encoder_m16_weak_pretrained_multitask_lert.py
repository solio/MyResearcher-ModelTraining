"""M16: weak-pretrained shared LERT followed by Human147 correction.

M15 proved the shared multi-task training path works, but Human147 alone made
several heads collapse to their majority class.  M16 keeps exactly that model
and optimisation mechanism, first giving it two fixed epochs over the existing
Train1822 weak-label corpus to learn all taxonomy boundaries.  It then performs
four Human147 OOF correction epochs and freezes one globally selected epoch.

Weak Dev448 and every human Check remain unused for fit or selection.  No new
manual labels and no hyper-parameter grid are introduced.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import encoder_m1 as m1
from . import encoder_m2_1_softlabel as m21
from . import encoder_m5_human100_source_balanced as m5
from . import encoder_m12_human_first_cv as m12
from . import encoder_m12_hybrid_runtime as hybrid
from . import encoder_m14_human_char_tfidf as m14
from . import encoder_m15_shared_multitask_lert as m15
from . import encoder_terminal_macbert_base as terminal
from .errors import ContractError
from .schema import V1_HEADS


STAGE_ID = "M16B-WEAK1821-DEDUPED-SHARED-MULTITASK-LERT-HUMAN147"
RUN_SCHEMA_VERSION = "myresearcher.encoder-m16b-weak-deduped-multitask.v1"
WEAK_EPOCHS = 2
HUMAN_MAX_EPOCHS = 4
DEFAULT_OUTPUT = m15.PROJECT_ROOT / "model-artifacts" / "m16b-weak1821-deduped-multitask-human147-20260916"


def _require(ok: bool, code: str, message: str, **details: Any) -> None:
    if not ok:
        raise ContractError(code, message, **details)


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _optimizer(torch: Any, model: Any) -> tuple[Any, list[Any], list[Any]]:
    encoder_parameters = [parameter for name, parameter in model.named_parameters() if name.startswith("encoder.") and parameter.requires_grad]
    head_parameters = [parameter for name, parameter in model.named_parameters() if name.startswith("heads.")]
    optimizer = torch.optim.AdamW(
        [{"params": encoder_parameters, "lr": m15.ENCODER_LR}, {"params": head_parameters, "lr": m15.HEAD_LR}],
        weight_decay=m15.WEIGHT_DECAY,
    )
    return optimizer, encoder_parameters, head_parameters


def pretrain_weak(
    *,
    runtime: tuple[Any, Any, Any, Any],
    snapshot: Path,
    schema: Any,
    records: Sequence[m1.M1Record],
    output: Path,
    device: Any,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    _np, torch, AutoModel, AutoTokenizer = runtime
    m15._set_seed(torch, m15.SEED + 200)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True)
    model = m15.make_model(torch, AutoModel, snapshot, schema).to(device)
    identity = m15.trainable_parameter_identity(model)
    optimizer, encoder_parameters, head_parameters = _optimizer(torch, model)
    balances = m15._class_balance(torch, records, schema, device)
    epoch_losses: list[float] = []
    updates = 0
    for epoch in range(1, WEAK_EPOCHS + 1):
        ordered = list(records)
        random.Random(m15.SEED + 20000 + epoch).shuffle(ordered)
        losses: list[float] = []
        model.train(True)
        for offset in range(0, len(ordered), m15.BATCH_SIZE):
            current = ordered[offset : offset + m15.BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            batch = m15._token_batch(torch, tokenizer, current, device)
            outputs = model(batch["input_ids"], batch["attention_mask"])
            weights = torch.ones(len(current), dtype=torch.float32, device=device)
            loss, _head_losses = m15._batch_loss(torch, outputs, current, schema, balances, weights, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*encoder_parameters, *head_parameters], m15.GRADIENT_CLIP)
            optimizer.step()
            updates += 1
            losses.append(float(loss.detach().cpu()))
        epoch_losses.append(float(np.mean(losses)))
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    checkpoint = output / "weak1822-pretrain-checkpoint.pt"
    torch.save(
        {
            "run_schema_version": RUN_SCHEMA_VERSION,
            "stage_id": STAGE_ID,
            "weak_epochs": WEAK_EPOCHS,
            "weak_samples": len(records),
            "optimizer_updates": updates,
            "model_state_dict": state,
            "parameter_identity": identity,
            "identity_stripping": True,
        },
        checkpoint,
    )
    del model
    gc.collect()
    try:
        torch.mps.empty_cache()
    except Exception:
        pass
    return state, {"checkpoint": str(checkpoint), "epochs": WEAK_EPOCHS, "samples": len(records), "optimizer_updates": updates, "epoch_mean_losses": epoch_losses}


def human_curve(
    *,
    runtime: tuple[Any, Any, Any, Any],
    snapshot: Path,
    schema: Any,
    initial_state: Mapping[str, Any],
    train_records: Sequence[m1.M1Record],
    validation_records: Sequence[m1.M1Record],
    recent_ids: set[str],
    fold: int,
    device: Any,
) -> list[dict[str, np.ndarray]]:
    _np, torch, AutoModel, AutoTokenizer = runtime
    m15._set_seed(torch, m15.SEED + 300 + fold)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True)
    model = m15.make_model(torch, AutoModel, snapshot, schema)
    model.load_state_dict(initial_state)
    model = model.to(device)
    optimizer, encoder_parameters, head_parameters = _optimizer(torch, model)
    balances = m15._class_balance(torch, train_records, schema, device)
    weights_by_id = m15.source_weight_map(train_records, recent_ids)
    curve: list[dict[str, np.ndarray]] = []
    for epoch in range(1, HUMAN_MAX_EPOCHS + 1):
        ordered = list(train_records)
        random.Random(m15.SEED + fold * 1000 + epoch).shuffle(ordered)
        model.train(True)
        for offset in range(0, len(ordered), m15.BATCH_SIZE):
            current = ordered[offset : offset + m15.BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            batch = m15._token_batch(torch, tokenizer, current, device)
            outputs = model(batch["input_ids"], batch["attention_mask"])
            weights = m15._source_weights(torch, current, weights_by_id, device)
            loss, _head_losses = m15._batch_loss(torch, outputs, current, schema, balances, weights, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*encoder_parameters, *head_parameters], m15.GRADIENT_CLIP)
            optimizer.step()
        curve.append(m15._predict(torch, model, tokenizer, validation_records, device))
    del model
    gc.collect()
    try:
        torch.mps.empty_cache()
    except Exception:
        pass
    return curve


def train_final_human(
    *,
    runtime: tuple[Any, Any, Any, Any],
    snapshot: Path,
    schema: Any,
    initial_state: Mapping[str, Any],
    records: Sequence[m1.M1Record],
    recent_ids: set[str],
    epochs: int,
    output: Path,
    device: Any,
) -> dict[str, Any]:
    _np, torch, AutoModel, AutoTokenizer = runtime
    m15._set_seed(torch, m15.SEED + 400)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True)
    model = m15.make_model(torch, AutoModel, snapshot, schema)
    model.load_state_dict(initial_state)
    model = model.to(device)
    optimizer, encoder_parameters, head_parameters = _optimizer(torch, model)
    balances = m15._class_balance(torch, records, schema, device)
    weights_by_id = m15.source_weight_map(records, recent_ids)
    epoch_losses: list[float] = []
    updates = 0
    for epoch in range(1, epochs + 1):
        ordered = list(records)
        random.Random(m15.SEED + 40000 + epoch).shuffle(ordered)
        model.train(True)
        losses: list[float] = []
        for offset in range(0, len(ordered), m15.BATCH_SIZE):
            current = ordered[offset : offset + m15.BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            batch = m15._token_batch(torch, tokenizer, current, device)
            outputs = model(batch["input_ids"], batch["attention_mask"])
            weights = m15._source_weights(torch, current, weights_by_id, device)
            loss, _head_losses = m15._batch_loss(torch, outputs, current, schema, balances, weights, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*encoder_parameters, *head_parameters], m15.GRADIENT_CLIP)
            optimizer.step()
            updates += 1
            losses.append(float(loss.detach().cpu()))
        epoch_losses.append(float(np.mean(losses)))
    checkpoint = output / "weak-pretrained-human-corrected-checkpoint.pt"
    torch.save(
        {
            "run_schema_version": RUN_SCHEMA_VERSION,
            "stage_id": STAGE_ID,
            "weak_epochs": WEAK_EPOCHS,
            "human_epochs": epochs,
            "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "class_order": {head: list(schema.class_order[head]) for head in V1_HEADS},
            "identity_stripping": True,
        },
        checkpoint,
    )
    del model
    gc.collect()
    try:
        torch.mps.empty_cache()
    except Exception:
        pass
    return {"checkpoint": str(checkpoint), "epochs": epochs, "optimizer_updates": updates, "epoch_mean_losses": epoch_losses}


def _load_final(torch: Any, AutoModel: Any, snapshot: Path, schema: Any, output: Path) -> Any:
    checkpoint = torch.load(output / "weak-pretrained-human-corrected-checkpoint.pt", map_location="cpu", weights_only=True)
    _require(checkpoint.get("stage_id") == STAGE_ID, "M16_CHECKPOINT_INVALID", "M16 checkpoint stage mismatch")
    model = m15.make_model(torch, AutoModel, snapshot, schema)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def run_m16(
    *,
    config_path: str | Path = m15.DEFAULT_CONFIG,
    cache_dir: str | Path = m15.DEFAULT_CACHE,
    selected_artifact: str | Path = m15.DEFAULT_SELECTED_ARTIFACT,
    pool_dir: str | Path = m15.DEFAULT_POOL,
    recent_feed_jsonl: str | Path = m15.DEFAULT_RECENT_FEED,
    recent_train_labels: str | Path = m15.DEFAULT_RECENT_TRAIN,
    recent_check_labels: str | Path = m15.DEFAULT_RECENT_CHECK,
    human48_jsonl: str | Path = m15.DEFAULT_HUMAN48,
    human48_input: str | Path = m15.DEFAULT_HUMAN48_INPUT,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    _require(not output.exists(), "M16_OUTPUT_EXISTS", "output directory must be new", path=str(output))
    output.mkdir(parents=True)
    started = time.monotonic()
    project = m1.ProjectConfig.load(config_path)
    schema, weak_train, weak_dev = m1.load_m1_partitions(project)
    base_train, _ = m5.load_human_train_pool(pool_dir)
    base_train, base_audit = m12.exclude_dependency_conflicts(base_train, role="base_human_train")
    recent_train, recent_audit = m12.load_recent_feedback_records(recent_train_labels, recent_feed_jsonl, role="recent_train", exclude_dependency_conflicts=True)
    human_train = [*base_train, *recent_train]
    recent_ids = {record.sample_id for record in recent_train}
    _require(len(weak_train) == 1822 and len(weak_dev) == 448, "M16_WEAK_SPLIT_INVALID", "M16 requires frozen Train1822/Dev448")
    _require(len(human_train) == 147, "M16_HUMAN_TRAIN_INVALID", "M16 requires Human147")
    _require(not ({record.sample_id for record in weak_train} & {record.sample_id for record in human_train}), "M16_WEAK_HUMAN_ID_OVERLAP", "weak and human sample IDs overlap")
    human_texts = set(m14._texts(human_train))
    weak_texts = m14._texts(weak_train)
    weak_overlap = [record for record, text in zip(weak_train, weak_texts) if text in human_texts]
    weak_fit = [record for record, text in zip(weak_train, weak_texts) if text not in human_texts]
    _require(len(weak_overlap) == 1 and len(weak_fit) == 1821, "M16_WEAK_HUMAN_TEXT_OVERLAP_INVALID", "M16 expects exactly the known conflicting exact-text weak duplicate", overlap=[record.sample_id for record in weak_overlap])

    runtime = m1._load_runtime_dependencies()
    _np, torch, AutoModel, AutoTokenizer = runtime
    snapshot, snapshot_identity = m21._snapshot(Path(cache_dir).resolve())
    device, device_name, mps_available = m1._choose_device(torch)
    initial_state, weak_report = pretrain_weak(runtime=runtime, snapshot=snapshot, schema=schema, records=weak_fit, output=output, device=device)

    splits = m15.multitask_fold_indices(human_train, recent_ids)
    oof_by_epoch = [{head: np.zeros((len(human_train), len(schema.class_order[head])), dtype=float) for head in V1_HEADS} for _ in range(HUMAN_MAX_EPOCHS)]
    fold_sizes: list[dict[str, int]] = []
    for fold, (training_indices, validation_indices) in enumerate(splits):
        fit = [human_train[index] for index in training_indices]
        validation = [human_train[index] for index in validation_indices]
        curves = human_curve(
            runtime=runtime,
            snapshot=snapshot,
            schema=schema,
            initial_state=initial_state,
            train_records=fit,
            validation_records=validation,
            recent_ids=recent_ids,
            fold=fold,
            device=device,
        )
        for epoch in range(HUMAN_MAX_EPOCHS):
            for head in V1_HEADS:
                oof_by_epoch[epoch][head][validation_indices] = curves[epoch][head]
        fold_sizes.append({"fold": fold, "train": len(fit), "validation": len(validation)})
    selected_epoch, reasoning_thresholds, oof_metrics, epoch_curve = m15.select_global_epoch(oof_by_epoch, human_train, schema)
    _dump(output / "oof-selection.json", {"selected_epoch": selected_epoch, "epoch_curve": epoch_curve, "selected_metrics": oof_metrics, "fold_sizes": fold_sizes})
    _dump(output / "reasoning-thresholds.json", reasoning_thresholds)
    final_training = train_final_human(
        runtime=runtime,
        snapshot=snapshot,
        schema=schema,
        initial_state=initial_state,
        records=human_train,
        recent_ids=recent_ids,
        epochs=selected_epoch,
        output=output,
        device=device,
    )

    # Independent checks are loaded only after weak pretraining and human correction freeze.
    recent_check, recent_check_audit = m12.load_recent_feedback_records(recent_check_labels, recent_feed_jsonl, role="cumulative_recent_check")
    human_check, _ = m5.load_human_check_pool(pool_dir)
    human48 = terminal._load_human48(Path(human48_jsonl), Path(human48_input), schema)
    final_model = _load_final(torch, AutoModel, snapshot, schema, output).to(torch.device("cpu"))
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True)
    current_runtime = hybrid.M12HybridInferenceRuntime.from_artifacts(selected_artifact=selected_artifact, config_path=config_path, cache_dir=cache_dir)
    parent_bundle = torch.load(Path(selected_artifact).resolve() / "unified-runtime-bundle.pt", map_location="cpu", weights_only=False)
    current_thresholds = {label: float(parent_bundle.get("reasoning_thresholds", {}).get(label, 0.5)) for label in schema.class_order["reasoning_tags"]}
    candidate_tables: dict[str, Any] = {}
    current_tables: dict[str, Any] = {}
    for name, records in {"recent_check19": recent_check, "human_check20": human_check, "human48": human48}.items():
        candidate_all = m15._metric_table(m15._predict(torch, final_model, tokenizer, records, torch.device("cpu")), records, schema, reasoning_thresholds)
        current_all = m15._metric_table(m12._baseline_probabilities(current_runtime, records, schema), records, schema, current_thresholds)
        if name == "human48":
            candidate_tables[name] = {head: candidate_all[head] for head in terminal.HUMAN48_HEADS}
            current_tables[name] = {head: current_all[head] for head in terminal.HUMAN48_HEADS}
        else:
            candidate_tables[name] = candidate_all
            current_tables[name] = current_all
    _dump(output / "metrics.json", {"current": current_tables, "candidate": candidate_tables})
    summary = {
        "stage_id": STAGE_ID,
        "status": "EVALUATED_AWAITING_SEMANTIC_DECISION",
        "actual_device": device_name,
        "mps_available": mps_available,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "weak_train_samples_received": len(weak_train),
        "weak_train_samples_used": len(weak_fit),
        "weak_exact_text_overlap_excluded": [record.sample_id for record in weak_overlap],
        "weak_dev_samples_not_used": len(weak_dev),
        "human_train_samples": len(human_train),
        "recent_check_samples": len(recent_check),
        "human_check_samples": len(human_check),
        "human48_samples": len(human48),
        "weak_epochs": WEAK_EPOCHS,
        "selected_human_epoch": selected_epoch,
        "weak_pretraining": weak_report,
        "human_correction": final_training,
        "snapshot_identity": snapshot_identity,
        "base_audit": base_audit,
        "recent_train_audit": recent_audit,
        "recent_check_audit": recent_check_audit,
        "identity_stripping": True,
        "new_manual_labels_required": False,
        "deployment_changed": False,
    }
    _dump(output / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=m15.DEFAULT_CONFIG)
    parser.add_argument("--cache-dir", type=Path, default=m15.DEFAULT_CACHE)
    parser.add_argument("--selected-artifact", type=Path, default=m15.DEFAULT_SELECTED_ARTIFACT)
    parser.add_argument("--pool-dir", type=Path, default=m15.DEFAULT_POOL)
    parser.add_argument("--recent-feed-jsonl", type=Path, default=m15.DEFAULT_RECENT_FEED)
    parser.add_argument("--recent-train-labels", type=Path, default=m15.DEFAULT_RECENT_TRAIN)
    parser.add_argument("--recent-check-labels", type=Path, default=m15.DEFAULT_RECENT_CHECK)
    parser.add_argument("--human48-jsonl", type=Path, default=m15.DEFAULT_HUMAN48)
    parser.add_argument("--human48-input", type=Path, default=m15.DEFAULT_HUMAN48_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        result = run_m16(
            config_path=args.config,
            cache_dir=args.cache_dir,
            selected_artifact=args.selected_artifact,
            pool_dir=args.pool_dir,
            recent_feed_jsonl=args.recent_feed_jsonl,
            recent_train_labels=args.recent_train_labels,
            recent_check_labels=args.recent_check_labels,
            human48_jsonl=args.human48_jsonl,
            human48_input=args.human48_input,
            output_dir=args.output_dir,
        )
    except (ContractError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "M16_BLOCKED", "error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
