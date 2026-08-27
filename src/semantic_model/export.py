from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from .data import read_json
from .errors import ContractError
from .evaluate import inspect_run
from .hashes import content_addressed_id, sha256_file


EXPORT_FILES = (
    "model.joblib",
    "model_manifest.json",
    "thresholds.json",
    "schema.json",
    "preprocessing_contract.json",
    "inference-output.schema.json",
    "training_diagnostics.json",
    "MODEL_CARD.md",
)


def export_run(run_dir: str | Path) -> dict[str, Any]:
    source = Path(run_dir).resolve()
    verified = inspect_run(source)
    run_manifest = read_json(source / "run_manifest.json")
    model_manifest = read_json(source / "model_manifest.json")
    identity = {
        "export_schema_version": "myresearcher.classical-export.v1",
        "run_id": verified["run_id"],
        "run_manifest_id": verified["run_manifest_id"],
        "model_manifest_id": model_manifest["model_manifest_id"],
        "model_version": model_manifest["model_version"],
        "schema_version": model_manifest["schema_version"],
        "preprocessing_contract_id": model_manifest["preprocessing_contract_id"],
    }
    export_id = content_addressed_id(identity)
    export_root = source.parent / "exports"
    final_dir = export_root / export_id
    if final_dir.exists():
        manifest = read_json(final_dir / "export_manifest.json")
        if manifest.get("export_id") != export_id:
            raise ContractError("IMMUTABLE_EXPORT_CONFLICT", "export identity mismatch")
        return {
            "status": "EXISTING_IMMUTABLE_EXPORT",
            "export_id": export_id,
            "model_dir": str(final_dir),
        }
    export_root.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=".tmp-export-", dir=export_root))
    try:
        for filename in EXPORT_FILES:
            source_path = source / filename
            if not source_path.is_file():
                raise ContractError(
                    "RUN_ARTIFACT_MISSING", "export source file is missing", file=filename
                )
            shutil.copyfile(source_path, temp_dir / filename)
        artifact_hashes = {
            filename: sha256_file(temp_dir / filename) for filename in EXPORT_FILES
        }
        manifest = {
            **identity,
            "export_id": export_id,
            "status": run_manifest["status"],
            "artifacts": artifact_hashes,
        }
        manifest["export_manifest_id"] = content_addressed_id(
            manifest, omit_keys={"export_manifest_id"}
        )
        (temp_dir / "export_manifest.json").write_text(
            f"{json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )
        os.replace(temp_dir, final_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return {"status": "EXPORTED", "export_id": export_id, "model_dir": str(final_dir)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export verified local CPU model bundle")
    parser.add_argument("--run", required=True)
    args = parser.parse_args(argv)
    try:
        result = export_run(args.run)
        exit_code = 0
    except ContractError as exc:
        result = {
            "status": "BLOCKED_EXPORT_CONTRACT_ERROR",
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
