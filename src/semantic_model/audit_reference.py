from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from .audit_data import audit_config
from .config import ProjectConfig
from .errors import ContractError
from .hashes import content_addressed_id, without_local_paths
from .reference_package import audit_reference_archive


def build_reference_audit_result(
    archive: Mapping[str, Any], data_audit: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the stable reference-package audit from already validated inputs.

    The strict exact-environment preflight consumes this shared result so a
    successful comparable-package audit cannot be mistaken for environment
    authorization, while its historical audit identity remains unchanged.
    """

    reference = data_audit.get("validation_summary", {}).get("reference_package", {})
    if not isinstance(reference, Mapping) or not reference.get("available"):
        raise ContractError(
            "REFERENCE_PACKAGE_NOT_FOUND",
            "verified archive is not installed at baseline_reference.root",
        )
    result: dict[str, Any] = {
        "reference_audit_schema_version": "semantic-baseline-reference-audit-v0.3.5",
        "status": (
            "REFERENCE_PACKAGE_VALIDATED_EXACT_ENVIRONMENT"
            if reference.get("exact_reproduction_environment_match")
            else "REFERENCE_PACKAGE_VALIDATED_COMPARABLE_ENVIRONMENT_ONLY"
        ),
        "archive": dict(archive),
        "reference_package": dict(reference),
        "binding_data_audit_id": data_audit["audit_id"],
        "training_allowed": data_audit["training_allowed"],
        "exact_reproduction_authorized_for_current_environment": reference.get(
            "exact_reproduction_authorized_for_current_environment", False
        ),
        "production_approval": False,
        "blocker_codes": reference.get("reproduction_blocker_codes", []),
    }
    result["reference_audit_id"] = content_addressed_id(
        without_local_paths(result), omit_keys={"reference_audit_id"}
    )
    return result


def run_reference_audit(
    config_path: str | Path, archive_path: str | Path
) -> tuple[dict[str, Any], int]:
    try:
        config = ProjectConfig.load(config_path)
        archive = audit_reference_archive(config, archive_path)
        data_audit = audit_config(config)
        return build_reference_audit_result(archive, data_audit), 0
    except ContractError as exc:
        result = {
            "reference_audit_schema_version": "semantic-baseline-reference-audit-v0.3.5",
            "status": "BLOCKED_INVALID_REFERENCE_PACKAGE",
            "training_allowed": False,
            "exact_reproduction_authorized_for_current_environment": False,
            "production_approval": False,
            "blocker_codes": [exc.code],
            "error": exc.as_dict(),
        }
        result["reference_audit_id"] = content_addressed_id(
            without_local_paths(result), omit_keys={"reference_audit_id"}
        )
        return result, 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only audit of the immutable v0.3.5 baseline reference ZIP"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", help="optional JSON audit report path")
    args = parser.parse_args(argv)
    result, exit_code = run_reference_audit(args.config, args.archive)
    serialized = f"{json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)}\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
    sys.stdout.write(serialized)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
