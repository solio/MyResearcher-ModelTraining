from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .data import read_json
from .errors import ContractError
from .hashes import content_addressed_id, sha256_file, verify_content_addressed_id


def inspect_run(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir).resolve()
    run_manifest = read_json(path / "run_manifest.json")
    model_manifest = read_json(path / "model_manifest.json")
    metrics = read_json(path / "metrics.json")
    comparison = read_json(path / "baseline_comparison.json")
    if not isinstance(run_manifest, Mapping) or not isinstance(model_manifest, Mapping):
        raise ContractError("RUN_ARTIFACT_INVALID", "run/model manifest must be objects")
    expected_run_manifest_id = content_addressed_id(
        run_manifest,
        omit_keys={"run_manifest_id", "elapsed_seconds", "environment"},
    )
    if run_manifest.get("run_manifest_id") != expected_run_manifest_id:
        raise ContractError(
            "CONTENT_ADDRESS_MISMATCH", "run_manifest_id does not match content"
        )
    verify_content_addressed_id(model_manifest, id_key="model_manifest_id")
    checks = {
        "model.joblib": model_manifest.get("model_sha256"),
        "thresholds.json": model_manifest.get("thresholds_sha256"),
        "schema.json": model_manifest.get("schema_sha256"),
        "preprocessing_contract.json": model_manifest.get(
            "preprocessing_contract_sha256"
        ),
        "inference-output.schema.json": model_manifest.get(
            "inference_schema_sha256"
        ),
        "metrics.json": run_manifest.get("metrics_sha256"),
        "errors.jsonl": run_manifest.get("errors_sha256"),
        "baseline_comparison.json": run_manifest.get(
            "baseline_comparison_sha256"
        ),
        "model_manifest.json": run_manifest.get("model_manifest_sha256"),
        "training_diagnostics.json": run_manifest.get(
            "training_diagnostics_sha256"
        ),
    }
    for filename, expected_hash in checks.items():
        artifact_path = path / filename
        if not artifact_path.is_file() or sha256_file(artifact_path) != expected_hash:
            raise ContractError(
                "RUN_ARTIFACT_HASH_MISMATCH", "run artifact hash mismatch", file=filename
            )
    return {
        "status": run_manifest["status"],
        "capability_maturity": run_manifest["capability_maturity"],
        "run_id": run_manifest["run_id"],
        "run_manifest_id": run_manifest["run_manifest_id"],
        "baseline_v0_3_5_reproduced": run_manifest["status"]
        == "BASELINE_V0_3_5_REPRODUCED_DIAGNOSTIC_ONLY",
        "production_approved": False,
        "metrics": metrics,
        "baseline_comparison": comparison,
        "verified_artifacts": sorted(checks),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify and read immutable run metrics")
    parser.add_argument("--run", required=True)
    args = parser.parse_args(argv)
    try:
        result = inspect_run(args.run)
        exit_code = 0
    except ContractError as exc:
        result = {
            "status": "BLOCKED_EVALUATION_CONTRACT_ERROR",
            "blocker_codes": [exc.code],
            "error": exc.as_dict(),
        }
        exit_code = 3
    sys.stdout.write(
        f"{json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)}\n"
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
