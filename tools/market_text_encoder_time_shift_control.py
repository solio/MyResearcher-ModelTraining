"""One fixed negative control: shift title bags by half the diagnostic span within each stock.

The prices, targets, market features, stock identities and trained weights are
unchanged.  Only which historical titles align with each diagnostic date moves.
No training or hyperparameter search is performed.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import market_text_encoder as study


def run(study_dir: Path) -> dict:
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    root = study_dir.resolve()
    result_path = root / "time-shift-control.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    config = summary["configuration"]
    paths = {name: Path(value["path"]) for name, value in summary["input_identity"].items()}
    rows, titles, _, _ = study.build_text_rows(paths["metadata"], paths["qfq"], paths["raw"],
                                                paths["benchmark"], window=config["window"],
                                                horizon=config["horizon"])
    _, diagnostic = study.split(rows, config["fit_end"], config["diagnostic_start"])
    scaler = summary["scaler"]
    mean = np.asarray(scaler["market_mean"], dtype=np.float32)
    std = np.asarray(scaler["market_std"], dtype=np.float32)
    for row in diagnostic:
        row["market_scaled"] = ((np.asarray(row["market"], dtype=np.float32) - mean) / std).tolist()
    by_stock: dict[str, list[dict]] = defaultdict(list)
    for row in diagnostic:
        by_stock[row["stock_code"]].append(row)
    shifted = {}
    for code, items in by_stock.items():
        items.sort(key=lambda row: row["entry_date"])
        offset = len(items) // 2
        if offset < config["window"]:
            raise ValueError(f"insufficient dates for nonoverlapping title shift: {code}")
        for index, row in enumerate(items):
            donor = items[(index + offset) % len(items)]
            shifted[(code, row["entry_date"])] = {**row,
                "post_ids": donor["post_ids"], "ages": donor["ages"]}
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(study.BASE_MODEL), local_files_only=True,
                                              trust_remote_code=False, use_fast=True)
    token_cache = study.TitleTokenCache(tokenizer, titles, max_length=config["max_length"])
    baseline = summary["market_baseline"]
    model = study.make_model(study.BASE_MODEL, len(mean), baseline_beta=baseline["beta"],
                             baseline_intercept=baseline["intercept"], device=device)
    model.load_state_dict(torch.load(root / "raw-title-encoder-state.pt", map_location=device, weights_only=True))
    model.eval()
    originals = [json.loads(line) for line in (root / "diagnostic-predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    output = []
    with torch.no_grad():
        for row, original in zip(diagnostic, originals):
            if (row["stock_code"], row["entry_date"]) != (original["stock_code"], original["entry_date"]):
                raise ValueError("diagnostic row identity changed")
            moved = shifted[(row["stock_code"], row["entry_date"])]
            score = float(model.predict_rows([moved], token_cache)[0].cpu())
            shifted_return = score * scaler["return_std"] + scaler["return_mean"]
            output.append({"stock_code": row["stock_code"], "entry_date": row["entry_date"],
                           "actual_horizon_return": row["return"],
                           "predicted_return__unaltered": original["predicted_return__raw_title_encoder"],
                           "predicted_return__time_shifted": shifted_return})
    unaltered = study._metrics(output, "predicted_return__unaltered")
    shifted_metrics = study._metrics(output, "predicted_return__time_shifted")
    differences = [abs(row["predicted_return__unaltered"] - row["predicted_return__time_shifted"])
                   for row in output]
    result = {"evidence_level": study.EVIDENCE,
              "control": "same-stock title bags cyclically shifted by half of each stock's diagnostic dates",
              "rows": len(output), "script_sha256": study.file_hash(Path(__file__)),
              "unaltered_metrics": unaltered, "time_shifted_metrics": shifted_metrics,
              "mean_absolute_prediction_change": float(np.mean(differences)),
              "max_absolute_prediction_change": max(differences)}
    study._write_json(result_path, result)
    study._write_jsonl(root / "time-shift-control-predictions.jsonl", output)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    result = run(parser.parse_args().study_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
