"""Minimal masked training path for the optional v0.3 auxiliary factors.

The frozen v0.2.1 seven-head model remains the source of its own outputs.  This
module adds five *independent* output layers which consume the representation
owned by the corresponding legacy head (``logical_owner_head`` in the v0.3
schema).  It deliberately does not construct an encoder, download a model, or
run an epoch: callers can use it as the small, testable loss component that a
future training runner will attach to an existing representation pipeline.

Missing and unfinished assignments are represented by a zero loss mask.  The
placeholder target used for a masked single-label row is never evaluated and
therefore is not an implicit ``UNKNOWN`` label.  A literal ``UNKNOWN`` answer
is encoded normally and receives mask one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .data import AuxiliaryFactorLabels, SemanticTrainingView, merge_legacy_labels_and_auxiliary_assignments
from .errors import ContractError
from .schema import SemanticSchemaCandidate, V03_AUXILIARY_FACTORS


@dataclass(frozen=True)
class AuxiliaryTargetBatch:
    """Tensor targets and masks for a batch of merged training views.

    ``targets`` contains class indices for single-label factors and multi-hot
    tensors for multi-label factors.  For an unmasked row the tensor value is a
    harmless zero placeholder (or an all-zero vector); ``loss_masks`` is the
    authoritative indication that the row contributes to training.
    """

    targets: Mapping[str, Tensor]
    loss_masks: Mapping[str, Tensor]
    sample_ids: tuple[str, ...]

    @property
    def masks(self) -> Mapping[str, Tensor]:
        """Short alias used by loss callers."""

        return self.loss_masks

    @property
    def loss_mask(self) -> Mapping[str, Tensor]:
        """Singular alias matching the schema/data terminology."""

        return self.loss_masks

    def __iter__(self) -> Iterator[Mapping[str, Tensor]]:
        """Allow ``targets, masks = build_auxiliary_targets(...)`` callers."""

        yield self.targets
        yield self.loss_masks

    def __getitem__(self, index: int) -> Mapping[str, Tensor]:
        if index == 0:
            return self.targets
        if index == 1:
            return self.loss_masks
        raise IndexError(index)


def logical_owner_head(schema: SemanticSchemaCandidate, factor: str) -> str:
    """Return the frozen top-level representation owner for ``factor``."""

    if factor not in schema.auxiliary_factors:
        raise ContractError("UNKNOWN_AUXILIARY_FACTOR", "factor is not present in candidate schema", factor=factor)
    try:
        owner = schema.raw["auxiliary_factors"][factor]["logical_owner_head"]
    except (KeyError, TypeError) as exc:
        raise ContractError("AUXILIARY_OWNER_MISSING", "logical_owner_head is missing from schema", factor=factor) from exc
    if not isinstance(owner, str) or not owner:
        raise ContractError("AUXILIARY_OWNER_INVALID", "logical_owner_head must be a non-empty head name", factor=factor)
    if owner not in schema.class_order:
        raise ContractError("AUXILIARY_OWNER_INVALID", "logical_owner_head is not a preserved v0.2.1 head", factor=factor, owner=owner)
    return owner


def _factor_labels(auxiliary: AuxiliaryFactorLabels, factor: str) -> tuple[Any, int]:
    try:
        return auxiliary.values[factor], int(auxiliary.loss_mask[factor])
    except KeyError as exc:
        raise ContractError("AUXILIARY_BATCH_FACTOR_MISSING", "merged view does not contain every auxiliary factor", factor=factor) from exc


def build_auxiliary_targets(
    views: Sequence[SemanticTrainingView],
    schema: SemanticSchemaCandidate,
    *,
    device: torch.device | str | None = None,
) -> AuxiliaryTargetBatch:
    """Encode merged sparse assignments into tensors and explicit loss masks.

    ``views`` should come from :func:`merge_legacy_labels_and_auxiliary_assignments`.
    The function does not infer a missing factor as ``UNKNOWN`` and does not
    mutate the legacy labels.
    """

    if not isinstance(views, Sequence):
        raise ContractError("AUXILIARY_BATCH_INVALID", "views must be a sequence of merged training views")
    sample_ids: list[str] = []
    for view in views:
        if not isinstance(view, SemanticTrainingView):
            raise ContractError("AUXILIARY_BATCH_INVALID", "views must contain SemanticTrainingView values")
        sample_ids.append(view.sample_id)

    targets: dict[str, Tensor] = {}
    masks: dict[str, Tensor] = {}
    target_device = torch.device(device) if device is not None else None
    for factor in V03_AUXILIARY_FACTORS:
        spec = schema.auxiliary_factors[factor]
        encoded: list[Any] = []
        factor_masks: list[int] = []
        for view in views:
            value, mask = _factor_labels(view.auxiliary, factor)
            if mask not in (0, 1):
                raise ContractError("AUXILIARY_LOSS_MASK_INVALID", "loss mask must be 0 or 1", factor=factor, mask=mask)
            if mask == 0:
                # Zero is only a masked placeholder; it is never interpreted
                # as a class and cannot fabricate UNKNOWN.
                encoded.append([0] * len(spec.class_order) if spec.allow_multiple else 0)
            elif spec.allow_multiple:
                if not isinstance(value, list):
                    raise ContractError("AUXILIARY_TARGET_INVALID", "multi-label value must be a list", factor=factor)
                selected = set(value)
                encoded.append([int(label in selected) for label in spec.class_order])
            else:
                if not isinstance(value, str) or value not in spec.class_order:
                    raise ContractError("AUXILIARY_TARGET_INVALID", "single-label value is outside the schema", factor=factor, value=value)
                encoded.append(spec.class_order.index(value))
            factor_masks.append(mask)

        if spec.allow_multiple:
            targets[factor] = torch.tensor(encoded, dtype=torch.float32, device=target_device).reshape(len(views), len(spec.class_order))
        else:
            targets[factor] = torch.tensor(encoded, dtype=torch.long, device=target_device)
        masks[factor] = torch.tensor(factor_masks, dtype=torch.float32, device=target_device)
    return AuxiliaryTargetBatch(targets=targets, loss_masks=masks, sample_ids=tuple(sample_ids))


# Explicit name for callers that prefer the data-layer wording.
encode_auxiliary_targets = build_auxiliary_targets
build_auxiliary_batch = build_auxiliary_targets


class AuxiliaryHeadModel(nn.Module):
    """Five independent factor classifiers over logical-owner representations.

    ``owner_dims`` may be one integer (all owners use the same pooled width) or
    a mapping keyed by the owner heads named by the candidate schema.  Multiple
    factors may have the same owner, but each factor always receives its own
    ``nn.Linear`` parameters and its own gradients.
    """

    def __init__(
        self,
        schema: SemanticSchemaCandidate,
        owner_dims: Mapping[str, int] | int,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.schema = schema
        self.owner_head_by_factor = {
            factor: logical_owner_head(schema, factor) for factor in V03_AUXILIARY_FACTORS
        }
        if isinstance(owner_dims, int):
            if owner_dims <= 0:
                raise ContractError("AUXILIARY_OWNER_DIM_INVALID", "owner dimension must be positive")
            dimensions = {owner: owner_dims for owner in self.owner_head_by_factor.values()}
        elif isinstance(owner_dims, Mapping):
            dimensions = {}
            for owner in self.owner_head_by_factor.values():
                value = owner_dims.get(owner)
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ContractError("AUXILIARY_OWNER_DIM_INVALID", "every logical owner requires a positive integer dimension", owner=owner)
                dimensions[owner] = value
        else:
            raise ContractError("AUXILIARY_OWNER_DIM_INVALID", "owner_dims must be an integer or mapping")
        if not isinstance(dropout, (int, float)) or not 0.0 <= float(dropout) < 1.0:
            raise ContractError("AUXILIARY_DROPOUT_INVALID", "dropout must be in [0, 1)")
        self.dropout = nn.Dropout(float(dropout))
        self.auxiliary_heads = nn.ModuleDict({
            factor: nn.Linear(dimensions[owner], len(schema.auxiliary_factors[factor].class_order))
            for factor, owner in self.owner_head_by_factor.items()
        })

    @property
    def heads(self) -> nn.ModuleDict:
        """Compatibility alias; the returned modules are still independent."""

        return self.auxiliary_heads

    def forward(self, owner_representations: Mapping[str, Tensor]) -> dict[str, Tensor]:
        if not isinstance(owner_representations, Mapping):
            raise ContractError("AUXILIARY_REPRESENTATIONS_INVALID", "owner representations must be a mapping")
        result: dict[str, Tensor] = {}
        for factor, owner in self.owner_head_by_factor.items():
            representation = owner_representations.get(owner)
            if representation is None:
                raise ContractError("AUXILIARY_OWNER_REPRESENTATION_MISSING", "logical owner representation is missing", factor=factor, owner=owner)
            if not isinstance(representation, Tensor):
                raise ContractError("AUXILIARY_REPRESENTATION_INVALID", "owner representation must be a torch tensor", owner=owner)
            if representation.ndim == 3:
                representation = representation[:, 0, :]
            if representation.ndim != 2:
                raise ContractError("AUXILIARY_REPRESENTATION_SHAPE_INVALID", "owner representation must have shape [batch, hidden]", owner=owner, shape=tuple(representation.shape))
            expected_width = self.auxiliary_heads[factor].in_features
            if representation.shape[-1] != expected_width:
                raise ContractError("AUXILIARY_REPRESENTATION_SHAPE_INVALID", "owner representation width differs from the schema head", factor=factor, expected=expected_width, observed=int(representation.shape[-1]))
            if not torch.isfinite(representation).all():
                raise ContractError("AUXILIARY_NONFINITE_REPRESENTATION", "owner representation contains NaN or Inf", owner=owner)
            result[factor] = self.auxiliary_heads[factor](self.dropout(representation))
        return result


# Natural short alias for callers that describe the component as "heads".
AuxiliaryHeads = AuxiliaryHeadModel


def masked_auxiliary_loss(
    logits: Tensor,
    target: Tensor,
    loss_mask: Tensor,
    *,
    task_type: str,
) -> Tensor:
    """Compute a zero-safe normalized loss for one sparse factor.

    An all-zero mask returns a differentiable exact zero (``logits.sum()*0``),
    so an optimizer can safely include an absent factor without NaNs or fake
    labels.  A non-zero mask averages only over annotated rows, not over the
    full batch.
    """

    if not isinstance(logits, Tensor) or logits.ndim != 2:
        raise ContractError("AUXILIARY_LOGITS_SHAPE_INVALID", "logits must have shape [batch, classes]")
    if not isinstance(target, Tensor) or not isinstance(loss_mask, Tensor):
        raise ContractError("AUXILIARY_TARGET_SHAPE_INVALID", "target and loss_mask must be tensors")
    if target.shape[0] != logits.shape[0] or loss_mask.ndim != 1 or loss_mask.shape[0] != logits.shape[0]:
        raise ContractError("AUXILIARY_TARGET_SHAPE_INVALID", "target/mask batch dimension does not match logits")
    if task_type not in {"single_label", "ordered_single_label", "multi_label"}:
        raise ContractError("AUXILIARY_TASK_TYPE_INVALID", "unknown auxiliary task type", task_type=task_type)
    mask = loss_mask.to(device=logits.device, dtype=logits.dtype)
    if not torch.isfinite(logits).all() or not torch.isfinite(mask).all() or bool(((mask < 0) | (mask > 1)).any()):
        raise ContractError("AUXILIARY_NONFINITE_LOSS_INPUT", "logits and masks must be finite and masks must be in [0, 1]")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ContractError("AUXILIARY_LOSS_MASK_INVALID", "loss masks must contain only 0 or 1")
    if task_type in {"single_label", "ordered_single_label"}:
        if target.ndim != 1:
            raise ContractError("AUXILIARY_TARGET_SHAPE_INVALID", "single-label targets must have shape [batch]")
        per_row = F.cross_entropy(logits, target.to(device=logits.device, dtype=torch.long), reduction="none")
    else:
        if target.ndim != 2 or target.shape != logits.shape:
            raise ContractError("AUXILIARY_TARGET_SHAPE_INVALID", "multi-label targets must match logits shape")
        per_row = F.binary_cross_entropy_with_logits(
            input=logits,
            target=target.to(device=logits.device, dtype=logits.dtype),
            reduction="none",
        ).mean(dim=1)
    observed = mask.sum()
    if observed.item() == 0:
        return logits.sum() * 0.0
    loss = (per_row * mask).sum() / observed
    if not torch.isfinite(loss):
        raise ContractError("AUXILIARY_NONFINITE_LOSS", "masked auxiliary loss is NaN or Inf")
    return loss


def compute_auxiliary_losses(
    logits_by_factor: Mapping[str, Tensor],
    batch: AuxiliaryTargetBatch,
    schema: SemanticSchemaCandidate,
) -> dict[str, Tensor]:
    """Compute one zero-safe loss per factor, preserving factor identity."""

    if set(logits_by_factor) != set(V03_AUXILIARY_FACTORS):
        raise ContractError("AUXILIARY_LOGITS_FACTOR_SET_INVALID", "logits must contain exactly the five auxiliary factors")
    if set(batch.targets) != set(V03_AUXILIARY_FACTORS) or set(batch.loss_masks) != set(V03_AUXILIARY_FACTORS):
        raise ContractError("AUXILIARY_BATCH_FACTOR_SET_INVALID", "targets and masks must contain exactly the five auxiliary factors")
    return {
        factor: masked_auxiliary_loss(
            logits_by_factor[factor],
            batch.targets[factor],
            batch.loss_masks[factor],
            task_type=schema.auxiliary_factors[factor].task_type,
        )
        for factor in V03_AUXILIARY_FACTORS
    }


compute_masked_auxiliary_losses = compute_auxiliary_losses


def total_auxiliary_loss(losses: Mapping[str, Tensor]) -> Tensor:
    """Sum per-factor losses without dropping an all-missing zero factor."""

    if set(losses) != set(V03_AUXILIARY_FACTORS):
        raise ContractError("AUXILIARY_LOSS_FACTOR_SET_INVALID", "loss mapping must contain exactly the five factors")
    values = list(losses.values())
    if not values:
        return torch.tensor(0.0)
    return torch.stack(values).sum()


def merge_and_build_auxiliary_targets(
    legacy_labels: Sequence[Mapping[str, Any]],
    assignments: Sequence[Mapping[str, Any]],
    schema: SemanticSchemaCandidate,
    *,
    device: torch.device | str | None = None,
) -> AuxiliaryTargetBatch:
    """Convenience wrapper that reuses the canonical merge loader exactly."""

    views = merge_legacy_labels_and_auxiliary_assignments(legacy_labels, assignments, schema)
    return build_auxiliary_targets(views, schema, device=device)


__all__ = [
    "AuxiliaryHeadModel",
    "AuxiliaryHeads",
    "AuxiliaryTargetBatch",
    "build_auxiliary_targets",
    "build_auxiliary_batch",
    "compute_auxiliary_losses",
    "compute_masked_auxiliary_losses",
    "encode_auxiliary_targets",
    "logical_owner_head",
    "masked_auxiliary_loss",
    "merge_and_build_auxiliary_targets",
    "total_auxiliary_loss",
]
