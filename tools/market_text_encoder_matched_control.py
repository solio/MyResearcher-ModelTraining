"""Matched no-text control for the fixed raw-title Encoder experiment.

Starts from its exact pre-fit market-only coefficients, then repeats the two
all-date epochs and every train-only UP-event ranking step without any text.
The output answers whether a difference is due to text or the event schedule.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import market_text_encoder as study


def run(study_dir: Path) -> dict:
    import torch

    study_dir = study_dir.resolve()
    destination = study_dir / "matched-control.json"
    if destination.exists():
        raise FileExistsError(destination)
    summary = json.loads((study_dir / "summary.json").read_text(encoding="utf-8"))
    config = summary["configuration"]
    if summary["evidence_level"] != study.EVIDENCE or config["epochs"] != 2:
        raise ValueError("this control is bound to the completed fixed two-epoch retrospective study")
    paths = {name: Path(value["path"]) for name, value in summary["input_identity"].items()}
    for name, path in paths.items():
        if study.file_hash(path) != summary["input_identity"][name]["sha256"]:
            raise ValueError(f"input changed: {name}")
    rows, _, calendar, qfq = study.build_text_rows(paths["metadata"], paths["qfq"], paths["raw"],
                                                   paths["benchmark"], window=config["window"],
                                                   horizon=config["horizon"])
    fit, diagnostic = study.split(rows, config["fit_end"], config["diagnostic_start"])
    scaler = summary["scaler"]
    mean = np.asarray(scaler["market_mean"], dtype=np.float32)
    std = np.asarray(scaler["market_std"], dtype=np.float32)
    return_mean, return_std = float(scaler["return_mean"]), float(scaler["return_std"])
    for row in (*fit, *diagnostic):
        row["market_scaled"] = ((np.asarray(row["market"], dtype=np.float32) - mean) / std).tolist()
        row["target_scaled"] = (row["return"] - return_mean) / return_std
    threshold, event_groups = study.training_up_events(fit, calendar, qfq, fit_end=config["fit_end"])
    if len(event_groups) != config["up_event_groups"] or abs(threshold - config["up_event_threshold"]) > 1e-12:
        raise ValueError("train-only event set changed")
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    baseline = summary["market_baseline"]
    beta = torch.nn.Parameter(torch.tensor(baseline["beta"], dtype=torch.float32))
    intercept = torch.nn.Parameter(torch.tensor(baseline["intercept"], dtype=torch.float32))
    optimizer = torch.optim.AdamW([{"params": [beta, intercept], "lr": 1e-4, "weight_decay": 0.01}])
    rng = random.Random(seed)
    log = []
    for epoch in range(config["epochs"]):
        order = list(range(len(fit)))
        rng.shuffle(order)
        losses = []
        for index in order:
            row = fit[index]
            optimizer.zero_grad()
            market = torch.tensor(row["market_scaled"], dtype=torch.float32)
            target = torch.tensor(row["target_scaled"], dtype=torch.float32)
            prediction = market @ beta + intercept
            loss = torch.nn.functional.smooth_l1_loss(prediction, target, beta=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([beta, intercept], 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        events = list(event_groups)
        rng.shuffle(events)
        for group in events:
            candidates = [fit[index] for index in group]
            optimizer.zero_grad()
            market = torch.tensor([row["market_scaled"] for row in candidates], dtype=torch.float32)
            target = torch.tensor([row["target_scaled"] for row in candidates], dtype=torch.float32)
            scores = market @ beta + intercept
            regression = torch.nn.functional.smooth_l1_loss(scores, target, beta=1.0)
            ranking = study.pairwise_rank_loss(torch, scores, [row["return"] for row in candidates])
            loss = regression + config["ranking_loss_weight"] * ranking
            loss.backward()
            torch.nn.utils.clip_grad_norm_([beta, intercept], 1.0)
            optimizer.step()
        log.append({"epoch": epoch + 1, "all_date_mean_loss": float(np.mean(losses)),
                    "up_event_steps": len(events)})
    original = [json.loads(line) for line in (study_dir / "diagnostic-predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(original) != len(diagnostic):
        raise ValueError("diagnostic identity count changed")
    output = []
    with torch.no_grad():
        for row, parent in zip(diagnostic, original):
            if (row["stock_code"], row["entry_date"]) != (parent["stock_code"], parent["entry_date"]):
                raise ValueError("diagnostic identity/order changed")
            if abs(row["return"] - parent["actual_horizon_return"]) > 1e-9:
                raise ValueError("diagnostic target changed")
            score = torch.tensor(row["market_scaled"], dtype=torch.float32) @ beta + intercept
            value = float(score) * return_std + return_mean
            output.append({**parent, "predicted_return__matched_market_count": value})
    metrics = study._metrics(output, "predicted_return__matched_market_count")
    result = {"evidence_level": study.EVIDENCE, "parent_summary_sha256": study.file_hash(study_dir / "summary.json"),
              "script_sha256": study.file_hash(Path(__file__)), "schedule": "same two all-date epochs and train-only UP-event steps as raw-title Encoder",
              "ranking_loss_weight": config["ranking_loss_weight"], "up_event_groups": len(event_groups),
              "fit_rows": len(fit), "diagnostic_rows": len(diagnostic),
              "beta": beta.detach().numpy().tolist(), "intercept": float(intercept.detach()),
              "training_log": log, "metrics": metrics,
              "comparison": {"encoder_minus_matched_market_rmse": summary["metrics"]["raw_title_encoder"]["rmse"] - metrics["rmse"],
                             "encoder_minus_matched_market_mae": summary["metrics"]["raw_title_encoder"]["mae"] - metrics["mae"]}}
    study._write_json(destination, result)
    study._write_jsonl(study_dir / "matched-control-predictions.jsonl", output)
    if not all(math.isfinite(float(value)) for value in result["comparison"].values()):
        raise ValueError("non-finite matched comparison")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.study_dir)
    print(json.dumps({"metrics": result["metrics"], "comparison": result["comparison"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
