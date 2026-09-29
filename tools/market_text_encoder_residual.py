"""One train-only-motivated residual-text contrast, not a new independent test.

The previous fixed Encoder run learned an almost constant text offset.  This
contrast freezes its market-only base and gives the raw-title last Encoder
block a bounded, stronger optimizer.  We explicitly remove the train-mean
text offset at evaluation, and test a same-stock time-shift placebo.  All
diagnostic dates were already seen by the research process; never promote.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import market_text_encoder as study


def run(args: argparse.Namespace) -> dict:
    import torch
    from transformers import AutoTokenizer

    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    source = Path(args.parent_study).resolve()
    parent = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    if parent["configuration"]["epochs"] != 2 or parent["evidence_level"] != study.EVIDENCE:
        raise ValueError("requires the fixed first raw-title run")
    paths = {name: Path(value["path"]) for name, value in parent["input_identity"].items()}
    for name, path in paths.items():
        if study.file_hash(path) != parent["input_identity"][name]["sha256"]:
            raise ValueError(f"parent input changed: {name}")
    config = parent["configuration"]
    rows, titles, calendar, qfq = study.build_text_rows(paths["metadata"], paths["qfq"],
                                                         paths["raw"], paths["benchmark"],
                                                         window=config["window"], horizon=config["horizon"])
    fit, diagnostic = study.split(rows, config["fit_end"], config["diagnostic_start"])
    mean = np.asarray(parent["scaler"]["market_mean"], dtype=np.float32)
    std = np.asarray(parent["scaler"]["market_std"], dtype=np.float32)
    return_mean = float(parent["scaler"]["return_mean"])
    return_std = float(parent["scaler"]["return_std"])
    for row in (*fit, *diagnostic):
        row["market_scaled"] = ((np.asarray(row["market"], dtype=np.float32) - mean) / std).tolist()
        row["target_scaled"] = (row["return"] - return_mean) / return_std
    threshold, groups = study.training_up_events(fit, calendar, qfq, fit_end=config["fit_end"])
    if len(groups) != config["up_event_groups"] or abs(threshold - config["up_event_threshold"]) > 1e-12:
        raise ValueError("training event identity changed")

    started = time.monotonic()
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(study.BASE_MODEL), local_files_only=True,
                                              trust_remote_code=False, use_fast=True)
    tokens = study.TitleTokenCache(tokenizer, titles, max_length=config["max_length"])
    baseline = parent["market_baseline"]
    model = study.make_model(study.BASE_MODEL, len(mean), baseline_beta=baseline["beta"],
                             baseline_intercept=baseline["intercept"], device=device)
    model.market_beta.requires_grad_(False)
    model.market_intercept.requires_grad_(False)
    model.eval()
    original_weight = model.backbone.encoder.layer[-1].output.dense.weight.detach().clone()
    encoder_parameters = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    head_parameters = [*model.attention.parameters(), *model.text_head.parameters()]
    optimizer = torch.optim.AdamW([
        {"params": encoder_parameters, "lr": 2e-4, "weight_decay": 0.01},
        {"params": head_parameters, "lr": 1e-3, "weight_decay": 0.01},
    ])
    rng = random.Random(seed)
    training_log = []
    for epoch in range(3):
        order = list(range(len(fit)))
        rng.shuffle(order)
        losses = []
        for position, index in enumerate(order, start=1):
            if time.monotonic() - started > args.max_seconds:
                raise TimeoutError("residual contrast exceeded its fixed wall-clock budget")
            row = fit[index]
            optimizer.zero_grad()
            prediction = model.predict_rows([row], tokens)
            target = torch.tensor([row["target_scaled"]], dtype=prediction.dtype, device=device)
            loss = torch.nn.functional.smooth_l1_loss(prediction, target, beta=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*encoder_parameters, *head_parameters], 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            if position % 256 == 0:
                print(f"residual epoch={epoch + 1}/3 rows={position}/{len(fit)} elapsed_s={time.monotonic()-started:.1f}", flush=True)
        events = list(groups)
        rng.shuffle(events)
        for group in events:
            candidates = [fit[index] for index in group]
            optimizer.zero_grad()
            scores = model.predict_rows(candidates, tokens)
            targets = torch.tensor([row["target_scaled"] for row in candidates], dtype=scores.dtype, device=device)
            regression = torch.nn.functional.smooth_l1_loss(scores, targets, beta=1.0)
            ranking = study.pairwise_rank_loss(torch, scores, [row["return"] for row in candidates])
            loss = regression + config["ranking_loss_weight"] * ranking
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*encoder_parameters, *head_parameters], 1.0)
            optimizer.step()
        training_log.append({"epoch": epoch + 1, "all_date_mean_loss": float(np.mean(losses)),
                             "up_event_steps": len(events)})
        print(f"residual epoch={epoch + 1} complete loss={training_log[-1]['all_date_mean_loss']:.6f} elapsed_s={time.monotonic()-started:.1f}", flush=True)

    last_layer_delta = float(torch.linalg.vector_norm(
        model.backbone.encoder.layer[-1].output.dense.weight.detach() - original_weight))
    if not math.isfinite(last_layer_delta) or last_layer_delta <= 0:
        raise ValueError("final Encoder block did not update")

    def score_rows(batch_rows):
        output_scores = []
        with torch.no_grad():
            for i, row in enumerate(batch_rows, 1):
                full = float(model.predict_rows([row], tokens)[0].cpu())
                market = float(np.dot(np.asarray(row["market_scaled"], dtype=np.float32),
                                      np.asarray(baseline["beta"], dtype=np.float32)) + baseline["intercept"])
                output_scores.append((full, market, full - market))
                if i % 256 == 0:
                    print(f"residual scoring={i}/{len(batch_rows)} elapsed_s={time.monotonic()-started:.1f}", flush=True)
        return output_scores

    fit_scores = score_rows(fit)
    diagnostic_scores = score_rows(diagnostic)
    text_train_mean = float(np.mean([item[2] for item in fit_scores]))
    # A constant text offset does not count as post-content information.
    predictions = []
    for row, (full, market, text) in zip(diagnostic, diagnostic_scores):
        predictions.append({"stock_code": row["stock_code"], "entry_date": row["entry_date"],
                            "outcome_date": row["outcome_date"], "actual_horizon_return": row["return"],
                            "predicted_return__market_count": market * return_std + return_mean,
                            "predicted_return__raw_title_encoder": full * return_std + return_mean,
                            "predicted_return__centered_text": (market + text - text_train_mean) * return_std + return_mean,
                            "text_component_scaled": text, "evidence_level": study.EVIDENCE})
    # Fixed same-stock 15-date cyclic shift, as in the first model's placebo.
    by_stock: dict[str, list[dict]] = defaultdict(list)
    for row in diagnostic:
        by_stock[row["stock_code"]].append(row)
    shifted = {}
    for code, items in by_stock.items():
        items.sort(key=lambda row: row["entry_date"])
        offset = len(items) // 2
        for i, row in enumerate(items):
            donor = items[(i + offset) % len(items)]
            shifted[(code, row["entry_date"])] = {**row, "post_ids": donor["post_ids"], "ages": donor["ages"]}
    with torch.no_grad():
        for row, prediction in zip(diagnostic, predictions):
            moved = shifted[(row["stock_code"], row["entry_date"])]
            full = float(model.predict_rows([moved], tokens)[0].cpu())
            prediction["predicted_return__shifted_title"] = full * return_std + return_mean
    metrics = {name: study._metrics(predictions, f"predicted_return__{name}")
               for name in ("market_count", "raw_title_encoder", "centered_text", "shifted_title")}
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "residual-title-encoder-state.pt"
    torch.save(model.state_dict(), checkpoint)
    result = {"evidence_level": study.EVIDENCE,
              "purpose": "single isolation of prior near-constant text branch; not independent test",
              "parent_study_summary_sha256": study.file_hash(source / "summary.json"),
              "script_sha256": study.file_hash(Path(__file__)),
              "checkpoint_sha256": study.file_hash(checkpoint),
              "device": str(device), "elapsed_seconds": time.monotonic() - started,
              "configuration": {"epochs": 3, "last_block_lr": 2e-4, "text_head_lr": 1e-3,
                                "market_branch_frozen": True, "ranking_loss_weight": config["ranking_loss_weight"],
                                "text_field": "title", "fit_end": config["fit_end"],
                                "diagnostic_start": config["diagnostic_start"], "seed": seed},
              "fit_rows": len(fit), "diagnostic_rows": len(diagnostic),
              "last_layer_delta_l2": last_layer_delta,
              "text_component_fit_mean_scaled": text_train_mean,
              "text_component_fit_std_scaled": float(np.std([item[2] for item in fit_scores])),
              "text_component_diagnostic_std_scaled": float(np.std([item[2] for item in diagnostic_scores])),
              "training_log": training_log, "metrics": metrics,
              "event_count": len(study._event_diagnostic(predictions, calendar, qfq, threshold,
                                                         fit_end=config["fit_end"]))}
    study._write_json(output / "summary.json", result)
    study._write_jsonl(output / "diagnostic-predictions.jsonl", predictions)
    print(json.dumps({"output_dir": str(output), "metrics": metrics,
                      "text_std": result["text_component_diagnostic_std_scaled"],
                      "elapsed_s": result["elapsed_seconds"]}, sort_keys=True), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-study", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=7200.0)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
