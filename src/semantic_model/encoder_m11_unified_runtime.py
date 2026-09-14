"""Offline unified 7+5 prediction runtime for the M11 feedback loop.

The legacy seven-head runtime and the M10C auxiliary checkpoint are loaded
read-only and exposed through one JSONL command.  Auxiliary factors are
attached only when the existing explicit text applicability rules apply; the
model output is never used as an annotation-task answer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .auxiliary_heads import AuxiliaryHeadModel
from .encoder_m10c_head_specific_diagnostic import (
    DISCOURSE,
    MARKET,
    _discourse_prediction,
    _market_prediction,
)
from .encoder_m2_selected_runtime import (
    DEFAULT_CACHE,
    CONFIG_PATH,
    DEFAULT_SELECTED_ARTIFACT,
    InferenceRecord,
    UnifiedCPUInferenceRuntime,
    load_unlabelled_jsonl,
)
from .encoder_m8c_cumulative import _applicable_factors, _single_prediction
from .errors import ContractError
from .hashes import sha256_file


M10C_ARTIFACT = Path(__file__).resolve().parents[3] / "model-artifacts" / "m10c-v03-head-specific-diagnostic-freeze-20260914"
DEFAULT_AUXILIARY_ARTIFACT = M10C_ARTIFACT
DEFAULT_OUTPUT = Path(__file__).resolve().parents[3] / "model-artifacts" / "m11-v03-unified-7plus5-offline-runtime-20260915"
M10C_CHECKPOINT_NAME = "head-specific-auxiliary-checkpoint.pt"
M10C_POLICY_VERSION = "M10C_V03_HEAD_SPECIFIC_DIAGNOSTIC_FREEZE"


def _require(ok: bool, code: str, message: str, **details: Any) -> None:
    if not ok:
        raise ContractError(code, message, **details)


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractError("M11_JSON_NOT_FOUND", "required JSON is missing", path=str(path)) from exc
    except json.JSONDecodeError as exc:
        raise ContractError("M11_JSON_INVALID", "required JSON is invalid", path=str(path)) from exc
    _require(isinstance(value, Mapping), "M11_JSON_INVALID", "required JSON must be an object", path=str(path))
    return value


def verify_m10c_artifact(artifact_dir: str | Path) -> Mapping[str, Any]:
    """Verify the ignored M10C files before loading its state dict."""

    root = Path(artifact_dir).resolve()
    manifest = _read_json(root / "manifest.json")
    files = manifest.get("files")
    _require(isinstance(files, Mapping) and files, "M11_AUXILIARY_MANIFEST_INVALID", "M10C manifest files are required")
    for name, metadata in files.items():
        _require(isinstance(name, str) and name and Path(name).name == name, "M11_AUXILIARY_MANIFEST_INVALID", "M10C manifest contains an unsafe path", path=name)
        _require(isinstance(metadata, Mapping), "M11_AUXILIARY_MANIFEST_INVALID", "M10C file metadata must be an object", path=name)
        path = root / name
        _require(path.is_file(), "M11_AUXILIARY_FILE_MISSING", "M10C file is missing", path=name)
        _require(int(metadata.get("bytes", -1)) == path.stat().st_size and metadata.get("sha256") == sha256_file(path), "M11_AUXILIARY_FILE_MISMATCH", "M10C file hash differs", path=name)
    _require(manifest.get("status") == "DIAGNOSTIC_ONLY" and manifest.get("selected_candidate") is False, "M11_AUXILIARY_STATUS_INVALID", "M10C must remain diagnostic-only")
    return manifest


def load_m10c_model(artifact_dir: str | Path, schema: Any, hidden_size: int) -> tuple[AuxiliaryHeadModel, Mapping[str, Any], Path]:
    root = Path(artifact_dir).resolve()
    manifest = verify_m10c_artifact(root)
    checkpoint = root / M10C_CHECKPOINT_NAME
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    _require(isinstance(payload, Mapping) and isinstance(payload.get("auxiliary_state_dict"), Mapping), "M11_AUXILIARY_CHECKPOINT_INVALID", "M10C checkpoint does not contain an auxiliary state dict")
    _require(payload.get("component") == "m10c_v03_head_specific_diagnostic_freeze", "M11_AUXILIARY_CHECKPOINT_INVALID", "unexpected M10C checkpoint component")
    model = AuxiliaryHeadModel(schema, hidden_size)
    model.load_state_dict(payload["auxiliary_state_dict"])
    model.eval()
    policy = _read_json(root / "policy.json")
    _require(policy.get("component") == M10C_POLICY_VERSION, "M11_AUXILIARY_POLICY_INVALID", "M10C policy version differs")
    return model, manifest, checkpoint


class Unified7Plus5OfflineRuntime:
    """One CPU entry point that returns legacy and applicable auxiliary heads."""

    def __init__(self, legacy_runtime: UnifiedCPUInferenceRuntime, auxiliary_runtime: Any, auxiliary_model: AuxiliaryHeadModel, *, legacy_checkpoint: Path, auxiliary_checkpoint: Path, auxiliary_manifest: Mapping[str, Any]) -> None:
        self.legacy_runtime = legacy_runtime
        self.auxiliary_runtime = auxiliary_runtime
        self.auxiliary_model = auxiliary_model
        self.legacy_checkpoint = legacy_checkpoint
        self.auxiliary_checkpoint = auxiliary_checkpoint
        self.auxiliary_manifest = auxiliary_manifest

    @classmethod
    def from_artifacts(
        cls,
        *,
        selected_artifact: str | Path = DEFAULT_SELECTED_ARTIFACT,
        auxiliary_artifact: str | Path = DEFAULT_AUXILIARY_ARTIFACT,
        config_path: str | Path = CONFIG_PATH,
        cache_dir: str | Path = DEFAULT_CACHE,
    ) -> "Unified7Plus5OfflineRuntime":
        legacy = UnifiedCPUInferenceRuntime.from_artifact(selected_artifact, config_path=config_path, cache_dir=cache_dir)
        from .encoder_m6c_auxiliary import AuxiliaryTrainingRuntime

        attached = AuxiliaryTrainingRuntime.from_selected_artifact(selected_artifact, config_path=config_path, cache_dir=cache_dir)
        schema = attached.schema
        # A single real representation determines the auxiliary input width;
        # no labels or data split are read to discover it.
        sample = InferenceRecord(sample_id="__m11_width_probe__", stock_code="probe", stock_name="probe", model_text="width probe")
        probe_inputs = attached.build_inputs([sample], "cpu")
        with torch.no_grad():
            captured = attached.model(probe_inputs["input_ids"], probe_inputs["attention_mask"])
        hidden_size = int(next(iter(captured["owner_representations"].values())).shape[-1])
        auxiliary, manifest, checkpoint = load_m10c_model(auxiliary_artifact, schema, hidden_size)
        legacy_bundle = Path(selected_artifact).resolve() / "unified-runtime-bundle.pt"
        return cls(legacy, attached, auxiliary, legacy_checkpoint=legacy_bundle, auxiliary_checkpoint=checkpoint, auxiliary_manifest=manifest)

    def _bundle_metadata(self) -> dict[str, Any]:
        return {
            "legacy_selected_artifact": str(self.legacy_runtime.artifact_dir),
            "legacy_checkpoint": str(self.legacy_checkpoint),
            "legacy_checkpoint_sha256": sha256_file(self.legacy_checkpoint),
            "legacy_artifact_content_address": self.legacy_runtime.manifest.get("content_address"),
            "auxiliary_artifact": str(self.auxiliary_checkpoint.parent),
            "auxiliary_checkpoint": str(self.auxiliary_checkpoint),
            "auxiliary_checkpoint_sha256": sha256_file(self.auxiliary_checkpoint),
            "auxiliary_artifact_content_address": self.auxiliary_manifest.get("content_address"),
            "schema_version": "semantic-schema-candidate-v0.3",
            "legacy_schema_version": self.legacy_runtime.schema.schema_version,
            "auxiliary_policy_version": M10C_POLICY_VERSION,
            "legacy_stage": self.legacy_runtime.manifest.get("status"),
        }

    def predict_records(self, records: Sequence[InferenceRecord]) -> list[dict[str, Any]]:
        if not records:
            return []
        legacy_rows = self.legacy_runtime.predict_records(records)
        attached = self.auxiliary_runtime
        attached.model.eval()
        self.auxiliary_model.eval()
        aux_by_id: dict[str, dict[str, Any]] = {}
        for offset in range(0, len(records), 16):
            current = records[offset : offset + 16]
            inputs = attached.build_inputs(current, "cpu")
            with torch.no_grad():
                captured = attached.model(inputs["input_ids"], inputs["attention_mask"])
                logits = self.auxiliary_model(captured["owner_representations"])
            for row_index, record in enumerate(current):
                factors = {}
                for factor in _applicable_factors(record.model_text):
                    spec = attached.schema.auxiliary_factors[factor]
                    vector = logits[factor][row_index]
                    _require(bool(torch.isfinite(vector).all().item()), "M11_AUXILIARY_NONFINITE_OUTPUT", "auxiliary logits are non-finite", factor=factor, sample_id=record.sample_id)
                    if factor == MARKET:
                        factors[factor] = _market_prediction(vector, spec.class_order)
                    elif factor == DISCOURSE:
                        factors[factor] = _discourse_prediction(vector, spec.class_order)
                    else:
                        factors[factor] = _single_prediction(vector, spec.class_order)
                aux_by_id[record.sample_id] = factors
        bundle = self._bundle_metadata()
        output: list[dict[str, Any]] = []
        for legacy_row, record in zip(legacy_rows, records):
            output.append({
                "sample_id": record.sample_id,
                "stock_code": record.stock_code,
                "stock_name": record.stock_name,
                "model_text": record.model_text,
                "legacy_heads": legacy_row["predictions"],
                "auxiliary_factors": aux_by_id[record.sample_id],
                "model_bundle": bundle,
            })
        _require([row["sample_id"] for row in output] == [record.sample_id for record in records], "M11_SAMPLE_ORDER_INVALID", "prediction order differs from input")
        return output


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _artifact_manifest(root: Path, *, bundle: Mapping[str, Any], input_rows: int, output_rows: int, auxiliary_consistent: bool) -> dict[str, Any]:
    files = [path for path in root.iterdir() if path.is_file() and path.name != "manifest.json"]
    file_meta = {path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(files)}
    canonical = json.dumps({"files": file_meta}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "artifact": "m11-v03-unified-7plus5-offline-runtime-20260915",
        "status": "DIAGNOSTIC_ONLY",
        "production_approval": False,
        "input_records": input_rows,
        "output_records": output_rows,
        "auxiliary_matches_m10c_fresh_shadow": auxiliary_consistent,
        "content_address": hashlib.sha256(canonical).hexdigest(),
        "model_bundle": dict(bundle),
        "files": file_meta,
    }


def run_m11(
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    selected_artifact: str | Path = DEFAULT_SELECTED_ARTIFACT,
    auxiliary_artifact: str | Path = DEFAULT_AUXILIARY_ARTIFACT,
    config_path: str | Path = CONFIG_PATH,
    cache_dir: str | Path = DEFAULT_CACHE,
    artifact_dir: str | Path | None = None,
    m10c_shadow_path: str | Path | None = None,
) -> dict[str, Any]:
    records = load_unlabelled_jsonl(input_jsonl)
    runtime = Unified7Plus5OfflineRuntime.from_artifacts(selected_artifact=selected_artifact, auxiliary_artifact=auxiliary_artifact, config_path=config_path, cache_dir=cache_dir)
    rows = runtime.predict_records(records)
    write_jsonl(Path(output_jsonl), rows)
    bundle = rows[0]["model_bundle"] if rows else runtime._bundle_metadata()
    auxiliary_consistent: bool | None = None
    if m10c_shadow_path is not None:
        reference = [json.loads(line) for line in Path(m10c_shadow_path).read_text(encoding="utf-8").splitlines() if line.strip()]
        _require(len(reference) == len(rows), "M11_M10C_REFERENCE_COUNT_INVALID", "M10C fresh shadow row count differs", reference=len(reference), observed=len(rows))
        auxiliary_consistent = all(left.get("sample_id") == right.get("sample_id") and left.get("auxiliary", {}) == right.get("auxiliary_factors", {}) for left, right in zip(reference, rows))
        _require(auxiliary_consistent, "M11_M10C_AUXILIARY_MISMATCH", "unified auxiliary outputs differ from M10C fresh shadow")
    result = {"input_records": len(records), "output_records": len(rows), "output_jsonl": str(Path(output_jsonl).resolve()), "model_bundle": bundle, "auxiliary_matches_m10c_fresh_shadow": auxiliary_consistent}
    if artifact_dir is not None:
        root = Path(artifact_dir).resolve()
        _require(not root.exists(), "M11_OUTPUT_EXISTS", "M11 artifact directory must be new", path=str(root))
        root.mkdir(parents=True, exist_ok=False)
        artifact_predictions = root / "unified-predictions.jsonl"
        write_jsonl(artifact_predictions, rows)
        _write_json(root / "bundle-manifest.json", bundle)
        summary = {"component": "M11_V03_UNIFIED_7PLUS5_OFFLINE_RUNTIME", "status": "DIAGNOSTIC_ONLY", "selected_candidate": False, "input_records": len(records), "output_records": len(rows), "sample_id_order_preserved": True, "legacy_heads_complete": True, "auxiliary_matches_m10c_fresh_shadow": auxiliary_consistent, "forbidden_data_access": {"test": False, "gold": False, "ood": False, "anchor": False, "llm": False}}
        _write_json(root / "summary.json", summary)
        report = "\n".join([
            "# M11 unified 7+5 offline runtime",
            "",
            "This artifact wires the frozen seven legacy heads and the M10C five-factor diagnostic checkpoint into one offline JSONL entry point. It is a local feedback-loop snapshot only; it is not production approval.",
            "",
            f"- Processed `{len(records)}` input rows once on CPU; output order and sample IDs are preserved.",
            "- `legacy_heads` contains the complete frozen seven-head output.",
            "- `auxiliary_factors` contains only factors allowed by the existing explicit applicability rules.",
            "- M10C policies are unchanged: softmax argmax for the three single-label factors; ordinary BCE/sigmoid 0.5 with empty output allowed for market; D policy with UNKNOWN exclusivity and top-1 fallback for discourse.",
            f"- Auxiliary equality against the M10C fresh shadow: `{auxiliary_consistent}`.",
            "- No annotation payload is produced; predictions remain local snapshots.",
            "",
            "M11 only completes offline feedback-loop integration and does not approve a production model.",
            "",
        ])
        (root / "REPORT.md").write_text(report, encoding="utf-8")
        manifest = _artifact_manifest(root, bundle=bundle, input_rows=len(records), output_rows=len(rows), auxiliary_consistent=bool(auxiliary_consistent))
        _write_json(root / "manifest.json", manifest)
        result["artifact_dir"] = str(root)
        result["artifact_content_address"] = manifest["content_address"]
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--selected-artifact", type=Path, default=DEFAULT_SELECTED_ARTIFACT)
    parser.add_argument("--auxiliary-artifact", type=Path, default=DEFAULT_AUXILIARY_ARTIFACT)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--artifact-dir", type=Path, default=None, help="optional immutable artifact directory")
    parser.add_argument("--m10c-shadow", type=Path, default=None, help="optional M10C fresh shadow JSONL for exact auxiliary comparison")
    args = parser.parse_args(argv)
    print(json.dumps(run_m11(args.input_jsonl, args.output_jsonl, selected_artifact=args.selected_artifact, auxiliary_artifact=args.auxiliary_artifact, config_path=args.config, cache_dir=args.cache_dir, artifact_dir=args.artifact_dir, m10c_shadow_path=args.m10c_shadow), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
