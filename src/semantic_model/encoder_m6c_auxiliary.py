"""M6C: connect sparse v0.3 auxiliary heads to the existing LERT runtime.

This module is a training-path adapter, not a training job.  It reuses the
accepted selected-candidate runtime and its local fixed snapshot/checkpoint,
captures the pooled outputs of the three logical-owner specialists, and sends
those representations to :class:`semantic_model.auxiliary_heads.AuxiliaryHeadModel`.
Only the five new auxiliary output layers are trainable.  The frozen encoder,
the existing specialist blocks, and the existing seven logical outputs are
never handed to an optimizer.

The data side accepts the flat labeling-app export used by Human Train100 and
converts it to complete seven-head legacy rows before calling the canonical
``merge_legacy_labels_and_auxiliary_assignments`` loader.  Sparse assignments
remain one-factor-per-row; unfinished rows are masked out by the shared data
layer and explicit ``UNKNOWN`` remains a real masked-in answer.

No model download, Test/Gold/OOD access, or epoch loop is implemented here.
The optional pilot smoke performs one auxiliary optimizer step solely to
prove that the wiring is real and that the frozen base outputs do not change.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, Mapping, Sequence

import torch
from torch import Tensor, nn

from . import encoder_m2_selected_runtime as selected
from .auxiliary_heads import (
    AuxiliaryHeadModel,
    AuxiliaryTargetBatch,
    build_auxiliary_targets,
    compute_auxiliary_losses,
    total_auxiliary_loss,
)
from .data import AuxiliaryFactorLabels, SemanticTrainingView, merge_legacy_labels_and_auxiliary_assignments
from .errors import ContractError
from .schema import SemanticSchemaCandidate, V1_HEADS, V03_AUXILIARY_FACTORS


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT.parent
DEFAULT_SELECTED_ARTIFACT = PROJECT_ROOT / "model-artifacts" / "m2-selected-candidate-frozen-for-m3-20260904"
DEFAULT_CACHE = REPO_ROOT / ".encoder-artifacts" / "hf-cache-lert-small"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "baseline_v0.3.5.yaml"
DEFAULT_CANDIDATE_SCHEMA = REPO_ROOT / "schema" / "semantic-schema-candidate-v0.3.json"
DEFAULT_HUMAN_TRAIN = REPO_ROOT / "runs" / "feedback-loop-v3-finalized-20260910" / "human-train-100-seven-head.jsonl"
DEFAULT_SELECTION_FILES = (
    REPO_ROOT / "runs" / "feedback-loop-v1-20260909" / "selection-train.jsonl",
    REPO_ROOT / "runs" / "feedback-loop-v2-20260909" / "selection-train.jsonl",
    REPO_ROOT / "runs" / "feedback-loop-v3-20260910" / "selection-train.jsonl",
)
BASE_SCHEMA_VERSION = "semantic-schema-calibrated-v0.2.1"
CANDIDATE_SCHEMA_VERSION = "semantic-schema-candidate-v0.3"


def _require(ok: bool, code: str, message: str, **details: Any) -> None:
    if not ok:
        raise ContractError(code, message, **details)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8-sig").splitlines()
    except FileNotFoundError as exc:
        raise ContractError("M6C_HUMAN_LABELS_NOT_FOUND", "Human label JSONL is missing", path=str(source)) from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError("M6C_JSONL_INVALID", "JSONL row is invalid", path=str(source), line=line_number) from exc
        _require(isinstance(value, dict), "M6C_JSONL_INVALID", "JSONL row must be an object", path=str(source), line=line_number)
        rows.append(value)
    return rows


def _selection_index(paths: Sequence[str | Path] = DEFAULT_SELECTION_FILES) -> dict[str, dict[str, Any]]:
    """Load read-only selection metadata to recover text for app exports."""

    index: dict[str, dict[str, Any]] = {}
    for path in paths:
        source = Path(path)
        if not source.is_file():
            continue
        for row in _read_jsonl(source):
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                continue
            if sample_id in index and index[sample_id] != row:
                raise ContractError("M6C_SELECTION_ID_CONFLICT", "selection metadata conflicts across rounds", sample_id=sample_id)
            index[sample_id] = row
    return index


def _answer_from_row(row: Mapping[str, Any], *, head: str) -> Any:
    answer = row.get("answer")
    if head == "reasoning_tags":
        _require(isinstance(answer, list) and len(answer) == len(set(answer)), "M6C_HUMAN_REASONING_INVALID", "reasoning_tags answer must be a unique list", sample_id=row.get("sample_id"))
    else:
        _require(isinstance(answer, str), "M6C_HUMAN_SCALAR_INVALID", "single-label answer must be a string", head=head, sample_id=row.get("sample_id"))
    return answer


def _normalize_sample_ids(sample_ids: Collection[str] | None) -> tuple[str, ...] | None:
    """Normalize an optional ordered subset of Human Train sample IDs."""

    if sample_ids is None:
        return None
    ordered = tuple(sample_ids)
    _require(bool(ordered), "M6C_SAMPLE_ID_SET_EMPTY", "selected sample_id set must not be empty")
    _require(all(isinstance(sample_id, str) and sample_id for sample_id in ordered), "M6C_SAMPLE_ID_INVALID", "selected sample IDs must be non-empty strings")
    _require(len(set(ordered)) == len(ordered), "M6C_SAMPLE_ID_DUPLICATE", "selected sample IDs must be unique")
    return ordered


def adapt_human_train100_rows(
    rows_or_path: Sequence[Mapping[str, Any]] | str | Path,
    *,
    schema: SemanticSchemaCandidate,
    expected_samples: int | None = 100,
    sample_ids: Collection[str] | None = None,
) -> list[dict[str, Any]]:
    """Adapt 700 app rows into complete v0.2.1 seven-head label records.

    The app export is intentionally flat (one ``sample_id/head`` per row),
    while the canonical merge loader consumes one complete seven-head legacy
    record.  This adapter validates the frozen taxonomy and semantic
    dependencies through ``schema.validate_record``; it never copies an
    auxiliary answer into a legacy head and never adds model predictions.
    """

    rows = list(rows_or_path) if not isinstance(rows_or_path, (str, Path)) else _read_jsonl(rows_or_path)
    requested_ids = _normalize_sample_ids(sample_ids)
    requested_set = set(requested_ids or ())
    # Establish presence from the raw export before validating any selected
    # rows.  Unrelated rows are deliberately not adapted or schema-checked.
    observed_ids = {
        row.get("sample_id")
        for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("sample_id"), str) and row.get("sample_id")
    }
    if requested_ids is not None:
        missing = [sample_id for sample_id in requested_ids if sample_id not in observed_ids]
        _require(not missing, "M6C_HUMAN_SAMPLE_ID_NOT_FOUND", "selected assignment sample_id is absent from Human Train export", sample_ids=missing)
    grouped: dict[str, dict[str, Mapping[str, Any]]] = {}
    metadata: dict[str, Mapping[str, Any]] = {}
    for row_number, row in enumerate(rows, 1):
        if requested_ids is not None and (not isinstance(row, Mapping) or row.get("sample_id") not in requested_set):
            continue
        _require(isinstance(row, Mapping), "M6C_HUMAN_LABEL_INVALID", "Human label row must be an object", row=row_number)
        sample_id, head = row.get("sample_id"), row.get("head")
        _require(isinstance(sample_id, str) and sample_id, "M6C_HUMAN_SAMPLE_ID_INVALID", "Human label requires sample_id", row=row_number)
        _require(row.get("schema_version") == BASE_SCHEMA_VERSION, "M6C_HUMAN_SCHEMA_INVALID", "Human Train rows must use frozen v0.2.1 schema", sample_id=sample_id)
        _require(row.get("is_final") is True, "M6C_HUMAN_LABEL_NOT_FINAL", "Human Train rows must be final", sample_id=sample_id, head=head)
        _require(isinstance(head, str) and head in V1_HEADS, "M6C_HUMAN_HEAD_INVALID", "Human Train row has an unknown legacy head", sample_id=sample_id, head=head)
        sample_group = grouped.setdefault(sample_id, {})
        _require(head not in sample_group, "M6C_HUMAN_DUPLICATE", "duplicate sample/head in Human Train export", sample_id=sample_id, head=head)
        sample_group[head] = row
        row_metadata = row.get("metadata", {})
        _require(isinstance(row_metadata, Mapping), "M6C_HUMAN_METADATA_INVALID", "Human label metadata must be an object", sample_id=sample_id)
        if sample_id in metadata and metadata[sample_id] != row_metadata:
            # Timestamps and app progress fields are allowed to differ; the
            # identity fields must be stable across the seven rows.
            for field in ("stock_code", "stock_name", "model_text", "text"):
                _require(metadata[sample_id].get(field) == row_metadata.get(field), "M6C_HUMAN_METADATA_CONFLICT", "sample metadata differs across head rows", sample_id=sample_id, field=field)
        metadata[sample_id] = row_metadata

    effective_expected = expected_samples if requested_ids is None or expected_samples != 100 else None
    if effective_expected is not None:
        _require(len(grouped) == effective_expected, "M6C_HUMAN_SAMPLE_COUNT_INVALID", "Human Train sample count differs from the expected pool", expected=effective_expected, observed=len(grouped))
    legacy_rows: list[dict[str, Any]] = []
    output_ids = requested_ids if requested_ids is not None else tuple(sorted(grouped))
    for sample_id in output_ids:
        sample_heads = grouped[sample_id]
        _require(set(sample_heads) == set(V1_HEADS), "M6C_HUMAN_HEAD_COVERAGE_INVALID", "each Human Train sample must have exactly seven heads", sample_id=sample_id, observed=sorted(sample_heads))
        row_metadata = metadata[sample_id]
        legacy: dict[str, Any] = {
            "sample_id": sample_id,
            "schema_version": BASE_SCHEMA_VERSION,
            "target_mode": _answer_from_row(sample_heads["target_mode"], head="target_mode"),
            "stance": _answer_from_row(sample_heads["stance"], head="stance"),
            "emotion_primary": _answer_from_row(sample_heads["emotion_primary"], head="emotion_primary"),
            "emotion_target": _answer_from_row(sample_heads["emotion_target"], head="emotion_target"),
            "action_tendency": _answer_from_row(sample_heads["action_tendency"], head="action_tendency"),
            "reasoning_tags": _answer_from_row(sample_heads["reasoning_tags"], head="reasoning_tags"),
            "context_dependency": _answer_from_row(sample_heads["context_dependency"], head="context_dependency"),
            # The app's flat export does not carry evidence spans.  The
            # frozen validator only requires confidence for the v0.2.1 view;
            # preserve an exported value when available and use a neutral
            # adapter default otherwise.  This is metadata, never a model
            # prediction or an auxiliary answer.
            "label_confidence": next((sample_heads[head].get("label_confidence") for head in V1_HEADS if sample_heads[head].get("label_confidence") in {"HIGH", "MEDIUM", "LOW"}), "MEDIUM"),
            "evidence_spans": {},
        }
        for field in ("stock_code", "stock_name", "published_at"):
            value = row_metadata.get(field)
            if value is not None:
                legacy[field] = value
        try:
            schema.validate_record(legacy)
        except ContractError as exc:
            # Preserve the canonical validator/code while making the
            # assignment-scoped failure actionable for the selected sample.
            details = dict(exc.details)
            details.setdefault("sample_id", sample_id)
            raise ContractError(exc.code, exc.message, **details) from exc
        legacy_rows.append(legacy)
    return legacy_rows


def human_train100_records(
    rows_or_path: Sequence[Mapping[str, Any]] | str | Path,
    *,
    schema: SemanticSchemaCandidate,
    selection_files: Sequence[str | Path] = DEFAULT_SELECTION_FILES,
    expected_samples: int | None = 100,
    sample_ids: Collection[str] | None = None,
) -> list[Any]:
    """Return ``M1Record``-compatible rows with text recovered read-only.

    Finalized Human Train100 rows normally keep text in the immutable
    selection JSONL rather than repeating it 700 times.  A future export may
    include ``model_text`` directly; both shapes are accepted.
    """

    from . import encoder_m1 as m1

    raw_rows = list(rows_or_path) if not isinstance(rows_or_path, (str, Path)) else _read_jsonl(rows_or_path)
    ordered_ids = _normalize_sample_ids(sample_ids)
    legacy_rows = adapt_human_train100_rows(raw_rows, schema=schema, expected_samples=expected_samples, sample_ids=ordered_ids)
    selection = _selection_index(selection_files)
    ordered_set = set(ordered_ids or ())
    flat_by_sample: dict[str, Mapping[str, Any]] = {}
    for row in raw_rows:
        if isinstance(row, Mapping) and (ordered_ids is None or row.get("sample_id") in ordered_set):
            flat_by_sample.setdefault(str(row["sample_id"]), row)
    result: list[m1.M1Record] = []
    for legacy in legacy_rows:
        sample_id = str(legacy["sample_id"])
        source = dict(selection.get(sample_id, {}))
        source.update({key: flat_by_sample[sample_id].get(key) for key in ("model_text", "text", "stock_code", "stock_name") if flat_by_sample[sample_id].get(key) is not None})
        metadata = flat_by_sample[sample_id].get("metadata", {})
        if isinstance(metadata, Mapping):
            source.update({key: metadata.get(key) for key in ("model_text", "text", "stock_code", "stock_name") if metadata.get(key) is not None})
        text = source.get("model_text", source.get("text"))
        _require(isinstance(text, str) and text.strip(), "M6C_HUMAN_TEXT_MISSING", "Human Train sample text is missing from export/selection metadata", sample_id=sample_id)
        result.append(
            m1.M1Record(
                sample_id=sample_id,
                stock_code=str(source.get("stock_code") or ""),
                stock_name=str(source.get("stock_name") or ""),
                model_text=text,
                label={head: legacy[head] for head in V1_HEADS},
                weights={head: 1.0 for head in V1_HEADS},
            )
        )
    return result


def load_auxiliary_training_views(
    human_train_jsonl: Sequence[Mapping[str, Any]] | str | Path,
    assignments_jsonl: Sequence[Mapping[str, Any]] | str | Path,
    *,
    schema: SemanticSchemaCandidate,
    expected_samples: int | None = 100,
) -> list[SemanticTrainingView]:
    """Load only the Human Train subset named by sparse assignments.

    Reading assignments first is intentional: unrelated legacy rows (including
    historical anomalies) never enter this pilot's validation path.
    """

    assignments = list(assignments_jsonl) if not isinstance(assignments_jsonl, (str, Path)) else _read_jsonl(assignments_jsonl)
    _require(assignments, "M6C_AUXILIARY_ASSIGNMENTS_EMPTY", "sparse assignment export is empty")
    ordered_ids: list[str] = []
    seen_ids: set[str] = set()
    for row in assignments:
        _require(isinstance(row, Mapping), "M6C_JSONL_INVALID", "sparse assignment row must be an object")
        sample_id = row.get("sample_id")
        _require(isinstance(sample_id, str) and sample_id, "M6C_HUMAN_SAMPLE_ID_INVALID", "sparse assignment requires sample_id")
        if sample_id not in seen_ids:
            seen_ids.add(sample_id)
            ordered_ids.append(sample_id)
    # A default expected_samples=100 means "the complete pool".  For an
    # assignment-scoped pilot it must not re-impose that full-pool count.
    subset_expected = None if expected_samples == 100 else expected_samples
    legacy = adapt_human_train100_rows(human_train_jsonl, schema=schema, expected_samples=subset_expected, sample_ids=ordered_ids)
    # This is the canonical merge implementation; no second merge semantics
    # are introduced in the runner.
    return merge_legacy_labels_and_auxiliary_assignments(legacy, assignments, schema)


_TASK_ALLOWED_FIELDS = {"sample_id", "text", "title", "heads", "metadata"}
_TASK_FORBIDDEN_FIELDS = {
    "answer", "answers", "label", "labels", "weights", "prediction",
    "predictions", "probability", "probabilities", "suggested_label",
}
_SKIPPED_DISPOSITIONS = {
    "SKIP", "SKIPPED", "LOW_INFORMATION_INPUT", "INPUT_EXCLUDED", "REJECTED",
}


def load_auxiliary_task_views(
    tasks_jsonl: Sequence[Mapping[str, Any]] | str | Path,
    assignments_jsonl: Sequence[Mapping[str, Any]] | str | Path,
    *,
    schema: SemanticSchemaCandidate,
) -> tuple[list[Any], list[SemanticTrainingView]]:
    """Load answer-free tasks plus sparse app assignments without legacy labels.

    This is the v0.3 path for new collector/feed samples.  A task declares the
    sample and the factors that are applicable; only a final assignment for a
    declared pair contributes a masked target.  The returned training view
    deliberately carries an empty ``legacy_label`` mapping rather than a
    fabricated seven-head label.  Existing Human Train100 callers continue to
    use :func:`load_auxiliary_training_views` unchanged.
    """

    task_rows = list(tasks_jsonl) if not isinstance(tasks_jsonl, (str, Path)) else _read_jsonl(tasks_jsonl)
    _require(task_rows, "M7A_TASKS_EMPTY", "auxiliary task JSONL is empty")
    task_by_id: dict[str, Mapping[str, Any]] = {}
    task_order: list[str] = []
    declared_heads: dict[str, tuple[str, ...]] = {}
    records: list[Any] = []
    for row_number, row in enumerate(task_rows, 1):
        _require(isinstance(row, Mapping), "M7A_TASK_SHAPE_INVALID", "task row must be an object", row=row_number)
        unknown = set(row) - _TASK_ALLOWED_FIELDS
        _require(not unknown, "M7A_TASK_PREDICTION_FORBIDDEN", "task contains fields outside the answer-free shape", fields=sorted(unknown), row=row_number)
        forbidden = set(row) & _TASK_FORBIDDEN_FIELDS
        _require(not forbidden, "M7A_TASK_PREDICTION_FORBIDDEN", "task payload must not contain labels or model predictions", fields=sorted(forbidden), row=row_number)
        sample_id = row.get("sample_id")
        text = row.get("text")
        _require(isinstance(sample_id, str) and sample_id.strip(), "M7A_TASK_SAMPLE_ID_INVALID", "task requires a non-empty sample_id", row=row_number)
        _require(sample_id not in task_by_id, "M7A_TASK_SAMPLE_ID_DUPLICATE", "task sample_id must be unique", sample_id=sample_id)
        _require(isinstance(text, str) and text.strip(), "M7A_TASK_TEXT_INVALID", "task requires non-empty text", sample_id=sample_id)
        heads = row.get("heads")
        _require(isinstance(heads, list) and heads, "M7A_TASK_HEADS_INVALID", "task heads must be a non-empty list", sample_id=sample_id)
        _require(all(isinstance(head, str) and head in V03_AUXILIARY_FACTORS for head in heads), "M7A_TASK_HEAD_INVALID", "task declares an unknown auxiliary factor", sample_id=sample_id)
        _require(len(set(heads)) == len(heads), "M7A_TASK_HEAD_DUPLICATE", "task heads must be unique", sample_id=sample_id)
        metadata = row.get("metadata", {})
        _require(isinstance(metadata, Mapping), "M7A_TASK_METADATA_INVALID", "task metadata must be an object", sample_id=sample_id)
        metadata_forbidden = {str(key) for key in metadata} & _TASK_FORBIDDEN_FIELDS
        _require(not metadata_forbidden, "M7A_TASK_PREDICTION_FORBIDDEN", "task metadata must not contain labels or model predictions", fields=sorted(metadata_forbidden), sample_id=sample_id)
        title = row.get("title", "")
        _require(title is None or isinstance(title, str), "M7A_TASK_TITLE_INVALID", "task title must be a string when present", sample_id=sample_id)
        task_copy = dict(row)
        task_copy["metadata"] = dict(metadata)
        task_by_id[sample_id] = task_copy
        task_order.append(sample_id)
        declared_heads[sample_id] = tuple(heads)
        records.append(
            selected.InferenceRecord(
                sample_id=sample_id,
                stock_code=str(metadata.get("stock_code") or ""),
                stock_name=str(metadata.get("stock_name") or title or ""),
                model_text=text,
                metadata=dict(metadata),
            )
        )

    assignment_rows = list(assignments_jsonl) if not isinstance(assignments_jsonl, (str, Path)) else _read_jsonl(assignments_jsonl)
    final_by_pair: dict[tuple[str, str], Mapping[str, Any]] = {}
    assignment_count: dict[str, int] = {sample_id: 0 for sample_id in task_order}
    for row_number, assignment in enumerate(assignment_rows, 1):
        _require(isinstance(assignment, Mapping), "M7A_ASSIGNMENT_SHAPE_INVALID", "assignment row must be an object", row=row_number)
        sample_id = assignment.get("sample_id")
        head = assignment.get("head")
        _require(isinstance(sample_id, str) and sample_id in task_by_id, "M7A_TASK_SAMPLE_ID_NOT_FOUND", "assignment sample_id was not declared by a task", sample_id=sample_id, row=row_number)
        _require(isinstance(head, str) and head in V03_AUXILIARY_FACTORS, "M7A_ASSIGNMENT_HEAD_INVALID", "assignment head must be an auxiliary factor", head=head, row=row_number)
        _require(head in declared_heads[sample_id], "M7A_TASK_HEAD_NOT_DECLARED", "assignment head was not declared by its task", sample_id=sample_id, head=head)
        is_final = assignment.get("is_final")
        _require(isinstance(is_final, bool), "M7A_ASSIGNMENT_FINAL_INVALID", "assignment is_final must be boolean", row=row_number)
        # The shared validator enforces schema version, factor cardinality and
        # the no-legacy-head rule.  For unfinished rows, answers are ignored
        # by design, so ``null`` is accepted without inventing UNKNOWN.
        skipped = str(assignment.get("disposition") or "").upper() in _SKIPPED_DISPOSITIONS
        if is_final and assignment.get("answer") is not None and not skipped:
            schema.validate_auxiliary_assignment(assignment)
        else:
            shape = dict(assignment)
            if not is_final or assignment.get("answer") is None or skipped:
                # The shared validator intentionally rejects null answers on a
                # final row.  Treat a final no-answer/skip as an unmasked
                # terminal state while still validating its shape and schema.
                shape["is_final"] = False
                shape["answer"] = None
            # This still checks sample/head/schema/is_final and forbids legacy
            # fields; a final no-answer or skipped row is intentionally masked.
            schema.validate_auxiliary_assignment(shape)
        pair = (sample_id, head)
        if is_final:
            _require(pair not in final_by_pair, "M7A_DUPLICATE_FINAL_ASSIGNMENT", "duplicate final assignment for sample_id/head", sample_id=sample_id, head=head)
            final_by_pair[pair] = assignment
        assignment_count[sample_id] += 1

    views: list[SemanticTrainingView] = []
    for sample_id in task_order:
        values = {factor: None for factor in V03_AUXILIARY_FACTORS}
        masks = {factor: 0 for factor in V03_AUXILIARY_FACTORS}
        for factor in declared_heads[sample_id]:
            assignment = final_by_pair.get((sample_id, factor))
            if assignment is None:
                continue
            disposition = str(assignment.get("disposition") or "").upper()
            answer = assignment.get("answer")
            if answer is None or disposition in _SKIPPED_DISPOSITIONS:
                continue
            values[factor] = list(answer) if isinstance(answer, list) else answer
            masks[factor] = 1
        task = task_by_id[sample_id]
        views.append(
            SemanticTrainingView(
                sample_id=sample_id,
                legacy_label={},
                auxiliary=AuxiliaryFactorLabels(values=values, loss_mask=masks),
                source={
                    "task": {
                        "record_type": "sparse_answer_free_task",
                        "sample_id": sample_id,
                        "text": task["text"],
                        "title": task.get("title", ""),
                        "metadata": dict(task.get("metadata", {})),
                        "declared_heads": list(declared_heads[sample_id]),
                    },
                    "auxiliary": {
                        "record_type": "sparse_app_assignment",
                        "assignment_count": assignment_count[sample_id],
                        "final_assignment_count": sum(int(key[0] == sample_id) for key in final_by_pair),
                    },
                },
            )
        )
    return records, views


class LertAuxiliaryTrainingModel(nn.Module):
    """Frozen selected LERT model plus five trainable auxiliary layers."""

    def __init__(self, base_model: nn.Module, schema: SemanticSchemaCandidate) -> None:
        super().__init__()
        self.base_model = base_model
        self.schema = schema
        for parameter in self.base_model.parameters():
            parameter.requires_grad = False
        self.base_model.eval()
        hidden_size = int(self.base_model.encoder.config.hidden_size)
        self.auxiliary_heads = AuxiliaryHeadModel(schema, hidden_size)
        self.logical_owner_heads = dict(self.auxiliary_heads.owner_head_by_factor)

    def train(self, mode: bool = True) -> "LertAuxiliaryTrainingModel":
        super().train(mode)
        # Never put the loaded encoder or old specialist heads in train mode;
        # only the new auxiliary module follows the caller's mode.
        self.base_model.eval()
        self.auxiliary_heads.train(mode)
        return self

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> dict[str, Any]:
        captured: dict[str, Tensor] = {}
        hooks: list[Any] = []
        owners = set(self.logical_owner_heads.values())
        for owner in owners:
            _require(owner in self.base_model.specialist_blocks, "M6C_OWNER_SPECIALIST_MISSING", "selected LERT runtime lacks a logical-owner specialist", owner=owner)
            block = self.base_model.specialist_blocks[owner]

            def capture(_module: Any, _args: tuple[Any, ...], output: Any, *, owner_name: str = owner) -> None:
                hidden = output[0] if isinstance(output, tuple) else output
                _require(isinstance(hidden, Tensor) and hidden.ndim == 3, "M6C_OWNER_REPRESENTATION_INVALID", "specialist block did not emit a sequence representation", owner=owner_name)
                captured[owner_name] = hidden[:, 0, :]

            hooks.append(block.register_forward_hook(capture))
        try:
            legacy_logits = self.base_model(input_ids, attention_mask)
        finally:
            for hook in hooks:
                hook.remove()
        _require(set(captured) == owners, "M6C_OWNER_REPRESENTATION_MISSING", "not every logical-owner representation was captured", expected=sorted(owners), observed=sorted(captured))
        owner_representations = {owner: captured[owner] for owner in owners}
        auxiliary_logits = self.auxiliary_heads(owner_representations)
        return {
            "legacy_logits": legacy_logits,
            "auxiliary_logits": auxiliary_logits,
            "owner_representations": owner_representations,
        }

    def trainable_parameter_report(self) -> dict[str, Any]:
        trainable = [name for name, parameter in self.named_parameters() if parameter.requires_grad]
        frozen_base = [name for name, parameter in self.base_model.named_parameters() if not parameter.requires_grad]
        unexpected_base = [name for name, parameter in self.base_model.named_parameters() if parameter.requires_grad]
        _require(not unexpected_base, "M6C_BASE_PARAMETER_UNFROZEN", "an encoder or legacy specialist parameter is trainable", parameters=unexpected_base)
        _require(trainable and all(name.startswith("auxiliary_heads.") for name in trainable), "M6C_AUXILIARY_PARAMETER_SCOPE_INVALID", "only auxiliary parameters may be trainable", parameters=trainable)
        return {"trainable_parameters": sorted(trainable), "frozen_base_parameter_count": len(frozen_base), "base_trainable_parameters": unexpected_base}


@dataclass
class AuxiliaryTrainingRuntime:
    """Loaded selected runtime and its M6C auxiliary training attachment."""

    selected_runtime: Any
    model: LertAuxiliaryTrainingModel
    schema: SemanticSchemaCandidate
    snapshot: Path
    artifact_dir: Path

    @property
    def torch(self) -> Any:
        return self.selected_runtime.torch

    @property
    def tokenizer(self) -> Any:
        return self.selected_runtime.tokenizer

    @property
    def config(self) -> Mapping[str, Any]:
        return self.selected_runtime.config

    @classmethod
    def from_selected_artifact(
        cls,
        artifact_dir: str | Path = DEFAULT_SELECTED_ARTIFACT,
        *,
        config_path: str | Path = DEFAULT_CONFIG,
        cache_dir: str | Path = DEFAULT_CACHE,
    ) -> "AuxiliaryTrainingRuntime":
        selected_runtime = selected.UnifiedCPUInferenceRuntime.from_artifact(
            Path(artifact_dir), config_path=config_path, cache_dir=cache_dir, verify_manifest=True
        )
        schema = SemanticSchemaCandidate.load(DEFAULT_CANDIDATE_SCHEMA)
        snapshot = Path(cache_dir).resolve() / "official-snapshot" / selected.REVISION
        model = LertAuxiliaryTrainingModel(selected_runtime.model, schema)
        model.eval()
        model.trainable_parameter_report()
        return cls(selected_runtime=selected_runtime, model=model, schema=schema, snapshot=snapshot, artifact_dir=Path(artifact_dir).resolve())

    def build_inputs(self, records: Sequence[Any], device: str | torch.device = "cpu") -> dict[str, Tensor]:
        return selected.build_inference_batch(self.torch, self.tokenizer, records, self.config, self.torch.device(device))

    def load_views(self, human_train_jsonl: str | Path, assignments_jsonl: str | Path, *, expected_samples: int | None = 100) -> list[SemanticTrainingView]:
        return load_auxiliary_training_views(human_train_jsonl, assignments_jsonl, schema=self.schema, expected_samples=expected_samples)

    def load_task_views(self, tasks_jsonl: str | Path, assignments_jsonl: str | Path) -> tuple[list[Any], list[SemanticTrainingView]]:
        """Load new-sample answer-free tasks without requiring Human Train100."""

        return load_auxiliary_task_views(tasks_jsonl, assignments_jsonl, schema=self.schema)

    def run_auxiliary_pilot_smoke(
        self,
        records: Sequence[Any],
        views: Sequence[SemanticTrainingView],
        checkpoint_path: str | Path,
    ) -> dict[str, Any]:
        """Run one real-model auxiliary update for an arbitrary non-empty pilot.

        This is deliberately one mechanical optimizer step (never an epoch
        loop); it proves that real encoder representations reach all five
        masked auxiliary heads while the legacy seven-head path stays frozen.
        """

        _require(len(records) >= 1 and len(records) == len(views), "M6C_SMOKE_SAMPLE_COUNT_INVALID", "auxiliary pilot smoke requires N>=1 records and matching views", records=len(records), views=len(views))
        _require([record.sample_id for record in records] == [view.sample_id for view in views], "M6C_SMOKE_ID_MISMATCH", "records and merged views must share sample order")
        self.model.eval()
        batch = self.build_inputs(records, "cpu")
        before_full = self.selected_runtime.predict_records(records)
        with torch.no_grad():
            before_reference = self.model(batch["input_ids"], batch["attention_mask"])
        # Re-run outside no_grad for the auxiliary graph.  The base model is
        # frozen, but its outputs still need to remain connected to the new
        # output layers for this one optimizer-step smoke.
        before = self.model(batch["input_ids"], batch["attention_mask"])
        target_batch: AuxiliaryTargetBatch = build_auxiliary_targets(views, self.schema, device="cpu")
        losses = compute_auxiliary_losses(before["auxiliary_logits"], target_batch, self.schema)
        total = total_auxiliary_loss(losses)
        before_base_state = {name: parameter.detach().cpu().clone() for name, parameter in self.model.base_model.named_parameters()}
        before_aux_state = {name: parameter.detach().cpu().clone() for name, parameter in self.model.auxiliary_heads.state_dict().items()}
        optimizer = torch.optim.AdamW(self.model.auxiliary_heads.parameters(), lr=1e-3, weight_decay=0.0)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        self.model.eval()
        with torch.no_grad():
            after = self.model(batch["input_ids"], batch["attention_mask"])
        after_full = self.selected_runtime.predict_records(records)
        base_logits_unchanged = all(torch.equal(before_reference["legacy_logits"][head], after["legacy_logits"][head]) for head in before_reference["legacy_logits"])
        full_logits_unchanged = before_full == after_full
        base_state_unchanged = all(torch.equal(value, self.model.base_model.state_dict()[name].detach().cpu()) for name, value in before_base_state.items())
        aux_updated = any(not torch.equal(value, self.model.auxiliary_heads.state_dict()[name].detach().cpu()) for name, value in before_aux_state.items())
        _require(base_logits_unchanged and full_logits_unchanged and base_state_unchanged, "M6C_LEGACY_OUTPUT_CHANGED", "frozen legacy seven-head outputs changed during auxiliary update")
        _require(aux_updated, "M6C_AUXILIARY_NOT_UPDATED", "auxiliary optimizer step changed no auxiliary parameter")
        for factor, values in after["auxiliary_logits"].items():
            _require(values.shape[0] == len(records) and values.shape[1] == len(self.schema.auxiliary_factors[factor].class_order), "M6C_AUXILIARY_OUTPUT_SHAPE_INVALID", "auxiliary output shape is inconsistent with schema", factor=factor, shape=tuple(values.shape))
            _require(bool(torch.isfinite(values).all().item()), "M6C_AUXILIARY_NONFINITE_OUTPUT", "auxiliary output is non-finite", factor=factor)
        checkpoint = Path(checkpoint_path).resolve()
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"schema_version": self.schema.schema_version, "component": "m6c_auxiliary_heads", "auxiliary_state_dict": self.model.auxiliary_heads.state_dict()}, checkpoint)
        reloaded_heads = AuxiliaryHeadModel(self.schema, int(self.model.base_model.encoder.config.hidden_size))
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        _require(isinstance(payload, Mapping) and isinstance(payload.get("auxiliary_state_dict"), Mapping), "M6C_CHECKPOINT_INVALID", "auxiliary checkpoint payload is invalid")
        reloaded_heads.load_state_dict(payload["auxiliary_state_dict"])
        with torch.no_grad():
            reloaded_outputs = reloaded_heads(after["owner_representations"])
        reload_equal = all(torch.equal(after["auxiliary_logits"][factor], reloaded_outputs[factor]) for factor in V03_AUXILIARY_FACTORS)
        _require(reload_equal, "M6C_CHECKPOINT_RELOAD_MISMATCH", "reloaded auxiliary checkpoint output differs")
        return {
            "samples": [record.sample_id for record in records],
            "snapshot": str(self.snapshot),
            "selected_artifact": str(self.artifact_dir),
            "trainable": self.model.trainable_parameter_report(),
            "legacy_seven_head_logits_unchanged": base_logits_unchanged,
            "legacy_seven_head_eval_unchanged": full_logits_unchanged,
            "frozen_base_parameters_unchanged": base_state_unchanged,
            "auxiliary_parameters_updated": aux_updated,
            "auxiliary_output_shapes": {factor: list(values.shape) for factor, values in after["auxiliary_logits"].items()},
            "auxiliary_outputs_finite": True,
            "losses": {factor: float(value.detach().cpu()) for factor, value in losses.items()},
            "total_loss": float(total.detach().cpu()),
            "checkpoint": str(checkpoint),
            "checkpoint_reload_equal": reload_equal,
            "device": "cpu",
            "training_epochs": 0,
            "optimizer_steps": 1,
        }

    # Backwards-compatible name retained for the original two-row test and
    # callers; it now accepts any N>=1 pilot.
    def run_two_sample_smoke(
        self,
        records: Sequence[Any],
        views: Sequence[SemanticTrainingView],
        checkpoint_path: str | Path,
    ) -> dict[str, Any]:
        return self.run_auxiliary_pilot_smoke(records, views, checkpoint_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_SELECTED_ARTIFACT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--human-train-jsonl", type=Path, default=None)
    parser.add_argument("--tasks-jsonl", type=Path, default=None, help="answer-free task JSONL; use with --aux-assignments-jsonl for new samples")
    parser.add_argument("--aux-assignments-jsonl", type=Path, default=None, help="future gpt_tasks.py pull JSONL; one sparse auxiliary assignment per row")
    parser.add_argument("--smoke-checkpoint", type=Path, default=None, help="checkpoint path for one N>=1 auxiliary pilot optimizer-step smoke")
    args = parser.parse_args(argv)
    schema = SemanticSchemaCandidate.load(DEFAULT_CANDIDATE_SCHEMA)
    result: dict[str, Any] = {"component": "m6c_auxiliary_training_path", "schema_version": schema.schema_version}
    _require(not (args.tasks_jsonl is not None and args.human_train_jsonl is not None), "M7A_INPUT_MODE_CONFLICT", "tasks-jsonl mode cannot be combined with Human Train JSONL")
    if args.smoke_checkpoint is not None:
        _require(args.aux_assignments_jsonl is not None and (args.human_train_jsonl is not None or args.tasks_jsonl is not None), "M6C_SMOKE_INPUTS_REQUIRED", "smoke requires either Human Train or answer-free tasks plus sparse assignments")
        runtime = AuxiliaryTrainingRuntime.from_selected_artifact(args.artifact_dir, config_path=args.config, cache_dir=args.cache_dir)
        if args.tasks_jsonl is not None:
            records, views = runtime.load_task_views(args.tasks_jsonl, args.aux_assignments_jsonl)
        else:
            views = runtime.load_views(args.human_train_jsonl, args.aux_assignments_jsonl)
            records = human_train100_records(args.human_train_jsonl, schema=runtime.schema, sample_ids=[view.sample_id for view in views])
        smoke = runtime.run_auxiliary_pilot_smoke(records, views, args.smoke_checkpoint)
        result["smoke"] = smoke
    elif args.tasks_jsonl is not None and args.aux_assignments_jsonl is not None:
        records, views = load_auxiliary_task_views(args.tasks_jsonl, args.aux_assignments_jsonl, schema=schema)
        target_batch = build_auxiliary_targets(views, schema)
        result["merged_samples"] = len(views)
        result["factor_masks"] = {factor: int(target_batch.loss_masks[factor].sum().item()) for factor in V03_AUXILIARY_FACTORS}
        result["legacy_labels_required"] = False
        result["input_records"] = len(records)
    elif args.human_train_jsonl is not None and args.aux_assignments_jsonl is not None:
        views = load_auxiliary_training_views(args.human_train_jsonl, args.aux_assignments_jsonl, schema=schema)
        target_batch = build_auxiliary_targets(views, schema)
        result["merged_samples"] = len(views)
        result["factor_masks"] = {factor: int(target_batch.loss_masks[factor].sum().item()) for factor in V03_AUXILIARY_FACTORS}
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


__all__ = [
    "AuxiliaryTrainingRuntime",
    "LertAuxiliaryTrainingModel",
    "adapt_human_train100_rows",
    "human_train100_records",
    "load_auxiliary_training_views",
    "load_auxiliary_task_views",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
