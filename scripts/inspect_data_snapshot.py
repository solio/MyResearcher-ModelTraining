#!/usr/bin/env python3
"""Read-only probe for the upstream semantic sampling snapshot.

This script records facts used during repository bootstrap. The production
audit gate lives in ``semantic_model.audit_data`` and must remain the training
entry point.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            records.append(value)
    return records


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def record_stats(path: Path, records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    sample_ids = [row.get("sample_id") for row in rows]
    present_ids = [value for value in sample_ids if isinstance(value, str) and value]
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": len(rows),
        "sample_id_unique": len(set(present_ids)),
        "sample_id_missing": len(rows) - len(present_ids),
        "sample_id_duplicate_rows": len(present_ids) - len(set(present_ids)),
        "schema_versions": sorted(
            {
                str(row["schema_version"])
                for row in rows
                if row.get("schema_version") not in (None, "")
            }
        ),
        "stock_count": len(
            {str(row["stock_code"]) for row in rows if row.get("stock_code")}
        ),
        "sources": sorted(
            {str(row["source"]) for row in rows if row.get("source")}
        ),
    }


def compare_teacher_metadata(
    records: list[dict[str, Any]], canonical: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    fields = ("stock_code", "stock_name", "published_at")
    unknown_ids = []
    mismatch_counts: Counter[str] = Counter()
    numeric_published_at = 0
    for record in records:
        sample_id = record.get("sample_id")
        canonical_record = canonical.get(sample_id)
        if canonical_record is None:
            unknown_ids.append(sample_id)
            continue
        for field in fields:
            if field == "published_at" and isinstance(record.get(field), (int, float)):
                numeric_published_at += 1
            if record.get(field) != canonical_record.get(field):
                mismatch_counts[field] += 1
    return {
        "unknown_sample_ids": len(unknown_ids),
        "metadata_mismatch_counts": dict(sorted(mismatch_counts.items())),
        "numeric_published_at_rows": numeric_published_at,
    }


def sqlite_stats(path: Path) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        ]
        return {
            "path": str(path),
            "sha256": sha256_file(path),
            "tables": {
                table: connection.execute(
                    f'SELECT COUNT(*) FROM "{table}"'  # noqa: S608 - table is catalog-derived
                ).fetchone()[0]
                for table in tables
            },
        }
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.data_root.resolve()

    inputs_path = root / "semantic_pilot_inputs.jsonl"
    inputs = read_jsonl(inputs_path)
    canonical = {record["sample_id"]: record for record in inputs}
    artifacts: dict[str, Any] = {
        "semantic_pilot_inputs": record_stats(inputs_path, inputs),
    }

    for name in ("semantic_pilot.csv", "clean_posts.csv", "gold_candidates.csv"):
        path = root / name
        artifacts[name] = record_stats(path, read_csv(path))

    teacher_files = {
        "teacher_A_400_v0.2": root / "teacherA/teacher_A_gold_400_v0.2.jsonl",
        "teacher_B_400_v0.2": root / "teacherB/teacher_B_gold_400_v0.2.jsonl",
        "teacher_A_blind100_v0.2.2": root / "teacherA/blind_100_labels_v0.2.2.jsonl",
        "teacher_B_blind100_v0.2.2": root / "teacherB/blind_100_labels_v0.2.2.jsonl",
    }
    for logical_name, path in teacher_files.items():
        records = read_jsonl(path)
        artifacts[logical_name] = {
            **record_stats(path, records),
            **compare_teacher_metadata(records, canonical),
        }

    for name in (
        "run_manifest.json",
        "semantic_schema_candidate_v0.1.json",
        "MyResearcher_Fresh_Blind100_A_Validation_v0.2.3.xlsx",
        "MyResearcher_Human_Gold_Calibrated_v0.2.xlsx",
        "human/MyResearcher_Human_Gold_Review_100_v0.1.xlsx",
    ):
        path = root / name
        artifacts[name] = {"path": str(path), "sha256": sha256_file(path)}

    database_path = root / "semantic_stage_v0_1.db"
    artifacts["semantic_stage_v0_1.db"] = sqlite_stats(database_path)

    payload = {
        "artifact_schema_version": "myresearcher.local-source-probe.v1",
        "data_root": str(root),
        "read_only": True,
        "artifacts": artifacts,
    }
    serialized = f"{json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)}\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
