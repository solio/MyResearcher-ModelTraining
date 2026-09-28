"""Isolated Action experiment with explicit author/negation/extent contrasts.

Synthetic rows are training examples, never Human Gold.  Epoch selection uses
only out-of-fold real Human Train predictions; all historical Human Check sets
are opened after the final checkpoint has been written and are diagnostic only.
No runtime routing or existing model artifact is changed by this module.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import encoder_action_repair as prior
from . import encoder_m1 as m1
from . import encoder_m2_1_softlabel as m21
from . import encoder_m2_lert_final as lert
from . import encoder_m2_selected_runtime as selected
from . import encoder_m5_human100_source_balanced as m5
from . import encoder_m12_human_first_cv as m12
from .errors import ContractError
from .schema import LabelSchema


HEAD = prior.HEAD
STAGE_ID = "ACTION-AUTHOR-NEGATION-CONTRAST-V1"
DEFAULT_OUTPUT = prior.PROJECT_ROOT / "model-artifacts" / "semantic-action-contrast-v1-20260929"
TWO_BLOCK_OUTPUT = prior.PROJECT_ROOT / "model-artifacts" / "semantic-action-contrast-two-block-v1-20260929"
SILVER_OUTPUT = prior.PROJECT_ROOT / "model-artifacts" / "semantic-action-contrast-natural-silver-v1-20260929"
EVIDENCE_WEIGHT_OUTPUT = prior.PROJECT_ROOT / "model-artifacts" / "semantic-action-contrast-evidence-weight-v1-20260929"
ROLE_NEGATIVE_OUTPUT = prior.PROJECT_ROOT / "model-artifacts" / "semantic-action-contrast-role-negative-v1-20260929"
NATURAL_SOURCE = prior.PROJECT_ROOT / "pipeline-runs" / "market-extreme-unenriched-20260928" / "source" / "m12-inference-input.jsonl"
SYNTHETIC_WEIGHT = 0.65
SILVER_WEIGHT = 0.75
LOW_EVIDENCE_WEIGHT = 0.2
MAX_EPOCHS = 12
BATCH_SIZE = 16
SEED = 35

# The alternatives change the semantic relation, not merely the stock or one
# token.  The held-out probes below use different surface forms.  None of
# these labels are promoted to human annotation or used for final acceptance.
TRAIN_PAIRS: tuple[tuple[str, str, str, str], ...] = (
    ("我今天新开了一笔仓位。", "BUY", "我今天没开新仓，打算再观察。", "WATCH"),
    ("原来空仓，我刚买进一笔。", "BUY", "原来空仓，我还没买进，先等等。", "WATCH"),
    ("我已经持有，今天再添了一点。", "ADD", "我已经持有，今天没再添，原仓继续拿着。", "HOLD"),
    ("我有底仓，刚又补了一单。", "ADD", "我有底仓，但这次没补，照旧持有。", "HOLD"),
    ("我把手里的股票都卖掉了。", "SELL", "我没卖手里的股票，继续持有。", "HOLD"),
    ("我决定把全部持仓退出。", "SELL", "我还没决定是否退出，先观望。", "WATCH"),
    ("我卖出了三成仓位，剩下继续拿。", "REDUCE", "我卖出了全部仓位，已经空仓。", "SELL"),
    ("我打算降低两成仓位。", "REDUCE", "我打算把仓位完全清掉。", "SELL"),
    ("我只是卖掉一小部分，底仓还在。", "REDUCE", "我已经全部卖光，没有底仓了。", "SELL"),
    ("我今天买了第一笔。", "BUY", "隔壁朋友今天买了，我自己只是看着。", "NO_ACTION_SIGNAL"),
    ("我又加了一笔仓。", "ADD", "同事又加了一笔仓，我自己的操作没提。", "NO_ACTION_SIGNAL"),
    ("我今天卖掉了全部持仓。", "SELL", "朋友今天卖掉了全部持仓，我还在拿着。", "HOLD"),
    ("我今天减掉一小部分仓位。", "REDUCE", "别人今天减掉一小部分仓位，我并未说自己的操作。", "NO_ACTION_SIGNAL"),
    ("我准备买入这只票。", "BUY", "你们可以买入这只票，我没说自己的打算。", "NO_ACTION_SIGNAL"),
    ("我准备卖出全部持仓。", "SELL", "你们可以卖出，我没有表达自己的交易打算。", "NO_ACTION_SIGNAL"),
    ("我选择继续持有，一点也不出。", "HOLD", "希望价格继续涨，但我没谈买卖。", "NO_ACTION_SIGNAL"),
    ("我不割肉，继续拿着。", "HOLD", "我已经割肉退出，手里没有这票。", "SELL"),
    ("我不追着买，先等回调。", "WATCH", "我已经追进去，新建了仓位。", "BUY"),
    ("没有突破之前，我暂不入场。", "WATCH", "突破以后我已经入场买了一笔。", "BUY"),
    ("我早盘卖了一笔，尾盘接回，做了个T。", "DO_T", "我早盘卖了一笔，剩下的持仓减少了。", "REDUCE"),
    ("我昨天做T，卖出后又接回来。", "DO_T", "我昨天卖掉了所有仓位，没有接回。", "SELL"),
    ("我买之前是空仓，现在已经持有。", "BUY", "我买之前就有仓，现在只是继续加码。", "ADD"),
    ("他催我买，我没有跟，先看走势。", "WATCH", "他催我买，我跟着买入第一笔。", "BUY"),
    ("看空归看空，我仍拿着不卖。", "HOLD", "看空之后，我把仓位全部卖掉。", "SELL"),
)

HELD_OUT_PROBES: tuple[tuple[str, str], ...] = (
    ("我尚未下单，想再看看。", "WATCH"),
    ("老王加仓了，我只是在旁边看。", "NO_ACTION_SIGNAL"),
    ("我把六成仓降到了三成。", "REDUCE"),
    ("这票我手上的份额全出了。", "SELL"),
    ("我没有出货，仍旧拿着。", "HOLD"),
    ("我只盼着它涨，没说要交易。", "NO_ACTION_SIGNAL"),
    ("今天我才建了第一笔仓。", "BUY"),
    ("底仓还在，我又添了一些。", "ADD"),
    ("我上午抛出，下午又接回做差价。", "DO_T"),
    ("是别人全卖了，不是我卖的；我继续持有。", "HOLD"),
    ("我没跟着买，现在仍然观望。", "WATCH"),
    ("我打算一股不留地退出。", "SELL"),
)

# Finite expert-reviewed, explicit-author-action examples from the already
# frozen natural-post input snapshot.  These are SILVER training suggestions,
# not owner labels or Gold.  Short sample-id prefixes resolve to unique full
# IDs; the loader fails on any missing, ambiguous, or Human Train/Check overlap.
# No old or new model prediction was used to assign these labels.
SILVER_SELECTION: tuple[tuple[str, str], ...] = (
    ("msv1-e5a03439", "BUY"), ("msv1-01714aff", "BUY"),
    ("msv1-99a2d1b8", "BUY"), ("msv1-7febfc28", "BUY"),
    ("msv1-f21664aa", "BUY"), ("msv1-743e1069", "BUY"),
    ("msv1-220db5d1", "BUY"), ("msv1-b5f795e7", "BUY"),
    ("msv1-9ab73b6c", "BUY"), ("msv1-812e746e", "BUY"),
    ("msv1-1b8916f2", "ADD"), ("msv1-294017ad", "ADD"),
    ("msv1-682f6949", "ADD"), ("msv1-06e49212", "ADD"),
    ("msv1-30b6eaf8", "ADD"), ("msv1-ff9039f6", "ADD"),
    ("msv1-1dbfbcca", "ADD"), ("msv1-eeb2931b", "ADD"),
    ("msv1-4e44ffce", "ADD"), ("msv1-56caf7cc", "ADD"),
    ("msv1-e0079eed", "ADD"),
    ("msv1-bf1bf9ed", "REDUCE"), ("msv1-54b7092e", "REDUCE"),
    ("msv1-5a9f8fb3", "REDUCE"), ("msv1-cf23ac84", "REDUCE"),
    ("msv1-c5e18226", "REDUCE"),
    ("msv1-0b3144e8", "REDUCE"), ("msv1-3cc5f949", "REDUCE"),
    ("msv1-43007565", "REDUCE"), ("msv1-30890d21", "REDUCE"),
    ("msv1-77caed7c", "REDUCE"), ("msv1-cbd674d8", "REDUCE"),
    ("msv1-93aaae12", "REDUCE"), ("msv1-04208216", "REDUCE"),
    ("msv1-8855815d", "REDUCE"),
    ("msv1-f0b22880", "SELL"), ("msv1-976e0c7c", "SELL"),
    ("msv1-24c9c541", "SELL"), ("msv1-5f0c2dbb", "SELL"),
    ("msv1-8aaf03f2", "SELL"), ("msv1-5cfa291c", "SELL"),
    ("msv1-46458252", "SELL"), ("msv1-2e728690", "SELL"),
    ("msv1-0708467b", "SELL"), ("msv1-b56d77ad", "SELL"),
    ("msv1-ccf7313d", "SELL"), ("msv1-ba832c99", "SELL"),
    ("msv1-58d8f9b8", "HOLD"),
    ("msv1-207d83e6", "HOLD"), ("msv1-d75e3ecc", "HOLD"),
    ("msv1-a2e0d7af", "HOLD"), ("msv1-160fc175", "HOLD"),
    ("msv1-83be32d8", "HOLD"), ("msv1-e8f1b7de", "HOLD"),
    ("msv1-9bfe3b20", "HOLD"), ("msv1-173c33d0", "HOLD"),
    ("msv1-307e7280", "HOLD"),
    ("msv1-9ea00b8a", "WATCH"), ("msv1-62570f97", "WATCH"),
    ("msv1-28a90acb", "WATCH"), ("msv1-24df6f6d", "WATCH"),
    ("msv1-bc50f0fe", "WATCH"), ("msv1-81001587", "WATCH"),
    ("msv1-94dfda79", "WATCH"), ("msv1-48db1fdc", "WATCH"),
)

# A separate, finite natural-language intervention for the next experiment:
# real posts mentioning another actor's trade/advice, or an explicit author
# refusal/wait.  None are pulled from Human Train/Check and none are Gold.
ROLE_NEGATIVE_SILVER: tuple[tuple[str, str], ...] = (
    ("msv1-91350a75", "NO_ACTION_SIGNAL"),
    ("msv1-b7191af3", "NO_ACTION_SIGNAL"),
    ("msv1-7e5875a9", "NO_ACTION_SIGNAL"),
    ("msv1-ee19c560", "NO_ACTION_SIGNAL"),
    ("msv1-933944b1", "NO_ACTION_SIGNAL"),
    ("msv1-cbcdf9ac", "NO_ACTION_SIGNAL"),
    ("msv1-7ba11101", "NO_ACTION_SIGNAL"),
    ("msv1-fa18e636", "NO_ACTION_SIGNAL"),
    ("msv1-ec776156", "NO_ACTION_SIGNAL"),
    ("msv1-c5980d5c", "NO_ACTION_SIGNAL"),
    ("msv1-3ea028dd", "NO_ACTION_SIGNAL"),
    ("msv1-5b751c01", "NO_ACTION_SIGNAL"),
    ("msv1-a31cf4b2", "NO_ACTION_SIGNAL"),
    ("msv1-ac7aed1d", "HOLD"),
    ("msv1-728af06e", "WATCH"),
    ("msv1-4d5108a0", "WATCH"),
    ("msv1-9ce6444b", "WATCH"),
    ("msv1-3ed51aa1", "WATCH"),
)

# Fresh, naturally occurring, expert-provisional check examples.  These are
# selected by reading text only before the role-negative experiment, not from
# model predictions or outcomes, and never enter fitting/epoch selection.
NATURAL_ROLE_PROBES: tuple[tuple[str, str], ...] = (
    ("msv1-5b62b598", "NO_ACTION_SIGNAL"),
    ("msv1-536e624e", "NO_ACTION_SIGNAL"),
    ("msv1-c219ac6c", "NO_ACTION_SIGNAL"),
    ("msv1-c3b4d0a8", "NO_ACTION_SIGNAL"),
    ("msv1-18a9ae3f", "WATCH"),
    ("msv1-9a1a60ef", "WATCH"),
    ("msv1-ab7e55f1", "HOLD"),
    ("msv1-ced045be", "HOLD"),
    ("msv1-f45fee9c", "HOLD"),
    ("msv1-4c099bb4", "BUY"),
    ("msv1-7b2de33c", "BUY"),
    ("msv1-0ad5a51c", "ADD"),
    ("msv1-a2e3255c", "REDUCE"),
    ("msv1-0dd1e4f0", "SELL"),
)

# These human answers are preserved verbatim.  For this experiment alone the
# Action loss gets a lower weight where the text lacks evidence for the stated
# *author's* action.  This is an expert training-weight judgment, not a label
# correction or Gold adjudication.  The full original rows remain in the OOF
# report; they are omitted only from epoch selection for this variant.
LOW_EVIDENCE_TRAIN: tuple[tuple[str, str, str], ...] = (
    ("fl1-2e1ff54a", "BUY", "no_author_buy"),
    ("fl1-43e3eb04", "BUY", "author_question_not_buy"),
    ("fl1-ed6dc2c5", "BUY", "strategy_not_author_buy"),
    ("fl1-09b3ce8c", "ADD", "no_author_add_evidence"),
    ("fl1-1df03236", "ADD", "no_author_add_evidence"),
    ("fl1-2084b95a", "ADD", "market_comment_not_author_add"),
    ("fl1-992398ce", "ADD", "generic_buy_advice_not_author_add"),
    ("fl1-bbf2d130", "ADD", "stance_not_author_add"),
    ("fl1-c5441a12", "ADD", "wish_not_author_add"),
    ("fresh-76815619", "ADD", "fundamental_comment_not_author_add"),
    ("msv1-a0c22261", "ADD", "position_not_stated_for_add"),
    ("fresh-1cf2a241", "REDUCE", "other_stock_performance_not_author_reduce"),
    ("fresh-4d69bcc3", "REDUCE", "outlook_not_author_reduce"),
    ("fresh-a7be0dbe", "REDUCE", "other_stock_comparison_not_author_reduce"),
    ("fresh-f52da825", "REDUCE", "other_actor_selling_not_author_reduce"),
    ("fl1-107a2ad0", "SELL", "risk_question_not_author_sell"),
    ("fl1-40537649", "SELL", "frustration_not_author_sell"),
    ("fl1-836342c1", "SELL", "counterfactual_not_author_sell"),
    ("fl1-88e23756", "SELL", "historical_analogy_not_author_sell"),
    ("fl1-ad07a9dc", "SELL", "advice_to_others_not_author_sell"),
    ("fl1-b9e84f30", "SELL", "imperative_without_author_sell"),
    ("fl1-f3f67770", "SELL", "performance_comparison_not_author_sell"),
    ("fl1-fe2ba824", "SELL", "external_prediction_not_author_sell"),
    ("fl1-ffbf6981", "SELL", "wish_not_author_sell"),
    ("fresh-24f7f118", "SELL", "historical_hypothetical_not_author_sell"),
    ("fresh-2b0e48ce", "SELL", "advice_to_others_not_author_sell"),
    ("fresh-68aadd20", "SELL", "advice_to_others_not_author_sell"),
    ("fresh-a735e5cf", "SELL", "fundamental_comment_not_author_sell"),
    ("fresh-d9989895", "SELL", "price_outlook_not_author_sell"),
    ("fresh-ee067c86", "SELL", "other_people_positions_not_author_sell"),
)


def _require(ok: bool, code: str, message: str, **details: Any) -> None:
    if not ok:
        raise ContractError(code, message, **details)


def _examples(schema: LabelSchema, human: Sequence[m1.M1Record]) -> tuple[list[m1.M1Record], list[m1.M1Record]]:
    contexts = sorted({(row.stock_code, row.stock_name) for row in human})
    _require(len(contexts) >= 8, "ACTION_CONTRAST_CONTEXTS", "not enough real-stock contexts for separated synthetic probes")
    human_text = {prior._normalized(row.model_text) for row in human}
    train: list[m1.M1Record] = []
    seen_text: set[str] = set()
    for pair in TRAIN_PAIRS:
        for index in (0, 2):
            text, label = pair[index], pair[index + 1]
            _require(label in schema.class_order[HEAD], "ACTION_CONTRAST_LABEL", "synthetic label is outside the Action taxonomy", label=label)
            norm = prior._normalized(text)
            _require(norm not in human_text and norm not in seen_text, "ACTION_CONTRAST_TEXT_OVERLAP", "synthetic text overlaps human Train or another synthetic text", text=text)
            seen_text.add(norm)
            for context in contexts[:2]:
                code, name = context
                sid = "synthetic-action-" + hashlib.sha256(f"{code}\n{norm}".encode()).hexdigest()
                train.append(m1.M1Record(sid, code, name, text, {HEAD: label}, {HEAD: SYNTHETIC_WEIGHT}))
    probe: list[m1.M1Record] = []
    for index, (text, label) in enumerate(HELD_OUT_PROBES):
        norm = prior._normalized(text)
        _require(label in schema.class_order[HEAD] and norm not in seen_text and norm not in human_text, "ACTION_CONTRAST_PROBE_OVERLAP", "probe overlaps training text or has invalid label", text=text)
        code, name = contexts[2 + index % (len(contexts) - 2)]
        probe.append(m1.M1Record(f"synthetic-probe-{index:02d}", code, name, text, {HEAD: label}, {HEAD: 1.0}))
    return train, probe


def _silver_records(schema: LabelSchema, human: Sequence[m1.M1Record], synthetic: Sequence[m1.M1Record], *, source: Path = NATURAL_SOURCE, include_role_negatives: bool = False) -> list[m1.M1Record]:
    selection = (*SILVER_SELECTION, *ROLE_NEGATIVE_SILVER) if include_role_negatives else SILVER_SELECTION
    _require(len({prefix for prefix, _label in selection}) == len(selection), "ACTION_SILVER_DUPLICATE_PREFIX", "silver selection contains duplicate prefixes")
    rows = prior._read_jsonl(source)
    train_ids = {record.sample_id for record in human}
    train_text = {prior._normalized(record.model_text) for record in [*human, *synthetic]}
    check_assignments = prior._read_jsonl(prior.LABEL_ROOT / "check-assignments.jsonl")
    check_ids = {str(row["sample_id"]) for row in check_assignments}
    check_text = {prior._normalized(str(row["text"])) for row in check_assignments}
    result: list[m1.M1Record] = []
    for prefix, label in selection:
        matches = [row for row in rows if str(row.get("sample_id", "")).startswith(prefix)]
        _require(len(matches) == 1 and label in schema.class_order[HEAD], "ACTION_SILVER_ID_OR_LABEL", "silver ID prefix is not unique or label is invalid", prefix=prefix, count=len(matches))
        row = matches[0]
        sid = str(row["sample_id"])
        norm = prior._normalized(str(row["model_text"]))
        _require(sid not in train_ids and sid not in check_ids and norm not in train_text and norm not in check_text, "ACTION_SILVER_TRAIN_CHECK_OVERLAP", "silver record overlaps Human Train/Check or synthetic text", sample_id=sid)
        train_ids.add(sid)
        train_text.add(norm)
        result.append(m1.M1Record(sid, str(row["stock_code"]), str(row["stock_name"]), str(row["model_text"]), {HEAD: label}, {HEAD: SILVER_WEIGHT}))
    return result


def _natural_role_probes(schema: LabelSchema, training: Sequence[m1.M1Record], *, source: Path = NATURAL_SOURCE) -> list[m1.M1Record]:
    rows = prior._read_jsonl(source)
    train_ids = {row.sample_id for row in training}
    train_text = {prior._normalized(row.model_text) for row in training}
    result: list[m1.M1Record] = []
    for prefix, label in NATURAL_ROLE_PROBES:
        matches = [row for row in rows if str(row.get("sample_id", "")).startswith(prefix)]
        _require(len(matches) == 1 and label in schema.class_order[HEAD], "ACTION_NATURAL_PROBE_ID", "natural probe prefix is not unique or label is invalid", prefix=prefix, count=len(matches))
        row = matches[0]
        sid, norm = str(row["sample_id"]), prior._normalized(str(row["model_text"]))
        _require(sid not in train_ids and norm not in train_text, "ACTION_NATURAL_PROBE_LEAKAGE", "natural probe overlaps training", sample_id=sid)
        result.append(m1.M1Record(sid, str(row["stock_code"]), str(row["stock_name"]), str(row["model_text"]), {HEAD: label}, {HEAD: 1.0}))
    return result


def _evidence_weighted_human(human: Sequence[m1.M1Record]) -> tuple[list[m1.M1Record], list[dict[str, Any]]]:
    changes: dict[str, dict[str, Any]] = {}
    for prefix, expected, reason in LOW_EVIDENCE_TRAIN:
        matches = [row for row in human if row.sample_id.startswith(prefix)]
        _require(len(matches) == 1 and matches[0].label[HEAD] == expected, "ACTION_LOW_EVIDENCE_ID", "evidence-weight prefix is missing, ambiguous, or has a different answer", prefix=prefix, count=len(matches))
        row = matches[0]
        _require(row.sample_id not in changes, "ACTION_LOW_EVIDENCE_DUPLICATE", "human row would be downweighted twice", sample_id=row.sample_id)
        changes[row.sample_id] = {"sample_id": row.sample_id, "text": row.model_text, "answer_preserved": expected, "reason": reason, "original_weight": float(row.weights[HEAD]), "experimental_weight": LOW_EVIDENCE_WEIGHT}
    weighted = [m1.M1Record(row.sample_id, row.stock_code, row.stock_name, row.model_text, row.label, {**row.weights, HEAD: LOW_EVIDENCE_WEIGHT}) if row.sample_id in changes else row for row in human]
    return weighted, list(changes.values())


def _fit(
    torch: Any,
    AutoModel: Any,
    tokenizer: Any,
    snapshot: Path,
    schema: LabelSchema,
    config: Mapping[str, Any],
    parent: Mapping[str, Any],
    human_fit: Sequence[m1.M1Record],
    synthetic: Sequence[m1.M1Record],
    silver: Sequence[m1.M1Record],
    validation: Sequence[m1.M1Record],
    *,
    epochs: int,
    seed: int,
    device: Any,
    started: float,
    two_block: bool = False,
) -> tuple[list[np.ndarray], Any]:
    m12._set_seed(torch, seed)
    model = _make_model(torch, AutoModel, snapshot, schema, config, parent, device, two_block=two_block)
    block = list(model.specialist_blocks[HEAD].parameters())
    classifier = list(model.heads[HEAD].parameters())
    optimizer = torch.optim.AdamW([{"params": block, "lr": prior.BLOCK_LR}, {"params": classifier, "lr": prior.HEAD_LR}], weight_decay=0.01)
    balance = m12._balance(torch, human_fit, schema, HEAD, device)
    curve: list[np.ndarray] = []
    for epoch in range(1, epochs + 1):
        _require(time.monotonic() - started < prior.WALL_SECONDS, "ACTION_CONTRAST_TIME_LIMIT", "training exceeded two hours")
        shuffled = [*human_fit, *synthetic, *silver]
        random.Random(seed + epoch).shuffle(shuffled)
        model.train(True)
        for start in range(0, len(shuffled), BATCH_SIZE):
            records = shuffled[start:start + BATCH_SIZE]
            batch = prior._batch(torch, tokenizer, records, config, schema, device)
            weights = torch.tensor([float(row.weights[HEAD]) for row in records], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch["attention_mask"])[HEAD]
            raw = torch.nn.functional.cross_entropy(logits, batch["labels"], weight=balance, reduction="none")
            loss = (raw * weights).sum() / weights.sum().clamp_min(1e-12)
            _require(bool(torch.isfinite(loss).item()), "ACTION_CONTRAST_NONFINITE_LOSS", "non-finite Action loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*block, *classifier], 1.0)
            optimizer.step()
        if validation:
            curve.append(prior._predict(torch, model, tokenizer, validation, config, schema, device))
    return curve, model


def _make_model(
    torch: Any, AutoModel: Any, snapshot: Path, schema: LabelSchema,
    config: Mapping[str, Any], parent: Mapping[str, Any], device: Any, *, two_block: bool,
) -> Any:
    if not two_block:
        return prior._make_model(torch, AutoModel, snapshot, schema, config, parent, device)
    # Block 10 starts from official LERT.  Block 11 and the classifier start
    # from the frozen selected Action model.  This changes trainable capacity,
    # not the input data, supervision, loss, or class mapping.
    model = m12.make_model(torch, AutoModel, snapshot, schema, float(config["head_dropout"]))
    model.specialist_blocks[HEAD][1].load_state_dict(parent["lert_specialist_block_state_dict"][HEAD])
    model.heads[HEAD].load_state_dict(parent["lert_heads_state_dict"][HEAD])
    model.set_active_head(HEAD)
    return model.to(device)


def _three_way(label: str) -> str:
    if label in ("BUY", "ADD"):
        return "BUY_SIDE"
    if label in ("REDUCE", "SELL"):
        return "SELL_SIDE"
    return "OTHER"


def _eval_set(
    torch: Any, current: Any, candidate: Any, tokenizer: Any, records: Sequence[m1.M1Record],
    config: Mapping[str, Any], schema: LabelSchema, device: Any,
) -> dict[str, Any]:
    order = list(schema.class_order[HEAD])
    before = prior._predict(torch, current, tokenizer, records, config, schema, device)
    after = prior._predict(torch, candidate, tokenizer, records, config, schema, device)
    before_labels = [order[int(index)] for index in before.argmax(axis=1)]
    after_labels = [order[int(index)] for index in after.argmax(axis=1)]
    truth = [str(row.label[HEAD]) for row in records]
    rows = [{"sample_id": row.sample_id, "text": row.model_text, "human_or_synthetic": truth[i], "current": before_labels[i], "candidate": after_labels[i]} for i, row in enumerate(records)]
    return {
        "count": len(records),
        "current": m12._metrics(before, records, schema, HEAD),
        "candidate": m12._metrics(after, records, schema, HEAD),
        "current_three_way_correct": sum(_three_way(truth[i]) == _three_way(before_labels[i]) for i in range(len(records))),
        "candidate_three_way_correct": sum(_three_way(truth[i]) == _three_way(after_labels[i]) for i in range(len(records))),
        "current_opposite_sign": sum(_three_way(truth[i]) in ("BUY_SIDE", "SELL_SIDE") and _three_way(before_labels[i]) in ("BUY_SIDE", "SELL_SIDE") and _three_way(truth[i]) != _three_way(before_labels[i]) for i in range(len(records))),
        "candidate_opposite_sign": sum(_three_way(truth[i]) in ("BUY_SIDE", "SELL_SIDE") and _three_way(after_labels[i]) in ("BUY_SIDE", "SELL_SIDE") and _three_way(truth[i]) != _three_way(after_labels[i]) for i in range(len(records))),
        "rows": rows,
    }


def run(*, output_dir: str | Path = DEFAULT_OUTPUT, two_block: bool = False, natural_silver: bool = False, human_evidence_weights: bool = False, role_negative_silver: bool = False) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    _require(not output.exists(), "ACTION_CONTRAST_OUTPUT_EXISTS", "candidate output already exists", path=str(output))
    project = m1.ProjectConfig.load(m12.DEFAULT_CONFIG)
    schema, _weak, _dev = m1.load_m1_partitions(project)
    legacy, legacy_audit = prior._legacy_train()
    additions, train_exclusions = prior._sparse_records("train", schema)
    human = [*legacy, *additions]
    prior._validate_train(human, additions)
    _require(len(human) == 159, "ACTION_CONTRAST_HUMAN_COUNT", "unexpected Human Train count", observed=len(human))
    evidence_audit: list[dict[str, Any]] = []
    if human_evidence_weights:
        human, evidence_audit = _evidence_weighted_human(human)
    synthetic, probes = _examples(schema, human)
    _require(not role_negative_silver or natural_silver, "ACTION_ROLE_NEGATIVE_REQUIRES_SILVER", "role-negative silver is an extension of the natural silver experiment")
    silver = _silver_records(schema, human, synthetic, include_role_negatives=role_negative_silver) if natural_silver else []
    natural_probes = _natural_role_probes(schema, [*human, *synthetic, *silver]) if role_negative_silver else []
    started = time.monotonic()
    _np, torch, AutoModel, AutoTokenizer = m21._load_runtime()
    snapshot, identity = m21._snapshot(m12.DEFAULT_CACHE)
    parent = torch.load(m12.DEFAULT_SELECTED_ARTIFACT / "unified-runtime-bundle.pt", map_location="cpu", weights_only=False)
    frozen = lert._contract_requirements(selected.CONTRACT_PATH)
    config = lert._config(frozen["contract"], schema.class_order)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True)
    device, device_name, _mps = m1._choose_device(torch)
    oof = np.full((MAX_EPOCHS, len(human), len(schema.class_order[HEAD])), np.nan, dtype=np.float32)
    for fold, (fit_indices, val_indices) in enumerate(m12.fold_indices(human, HEAD, folds=5, seed=SEED)):
        fit = [human[index] for index in fit_indices]
        val = [human[index] for index in val_indices]
        curve, model = _fit(torch, AutoModel, tokenizer, snapshot, schema, config, parent, fit, synthetic, silver, val, epochs=MAX_EPOCHS, seed=SEED + fold, device=device, started=started, two_block=two_block)
        for epoch, probabilities in enumerate(curve):
            oof[epoch, val_indices, :] = probabilities
        del model
        gc.collect()
        if device_name == "mps":
            torch.mps.empty_cache()
        print(json.dumps({"event": "fold_complete", "fold": fold + 1, "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
    _require(bool(np.isfinite(oof).all()), "ACTION_CONTRAST_OOF_INCOMPLETE", "incomplete Human Train OOF predictions")
    selection_indices = [index for index, row in enumerate(human) if float(row.weights[HEAD]) > LOW_EVIDENCE_WEIGHT] if human_evidence_weights else list(range(len(human)))
    selection_records = [human[index] for index in selection_indices]
    epoch, oof_metric, _thresholds, curve = m12._select_epoch(oof[:, selection_indices, :], selection_records, schema, HEAD)
    all_human_oof = m12._metrics(oof[epoch - 1], human, schema, HEAD)
    _unused_curve, model = _fit(torch, AutoModel, tokenizer, snapshot, schema, config, parent, human, synthetic, silver, [], epochs=epoch, seed=SEED + 100, device=device, started=started, two_block=two_block)
    output.mkdir(parents=True)
    checkpoint = output / "action-checkpoint.pt"
    stage_id = STAGE_ID + ("-TWO-BLOCK" if two_block else "-ONE-BLOCK") + ("-NATURAL-SILVER" if natural_silver else "") + ("-EVIDENCE-WEIGHT" if human_evidence_weights else "") + ("-ROLE-NEGATIVE" if role_negative_silver else "")
    torch.save({"stage_id": stage_id, "head": HEAD, "selected_epoch": epoch, "two_block": two_block, "parent_artifact": str(m12.DEFAULT_SELECTED_ARTIFACT), "specialist_block_state_dict": {k: v.detach().cpu() for k, v in model.specialist_blocks[HEAD].state_dict().items()}, "head_state_dict": {k: v.detach().cpu() for k, v in model.heads[HEAD].state_dict().items()}}, checkpoint)
    m12._dump(output / "training-config.json", {"stage_id": stage_id, "two_block": two_block, "model_id": m12.MODEL_ID, "revision": m12.REVISION, "snapshot_identity": identity, "device": device_name, "human_train_count": len(human), "human_evidence_weight_count": len(evidence_audit), "human_evidence_weight_audit": evidence_audit, "synthetic_train_count": len(synthetic), "synthetic_weight": SYNTHETIC_WEIGHT, "natural_silver_count": len(silver), "natural_silver_weight": SILVER_WEIGHT, "natural_silver_source": str(NATURAL_SOURCE) if silver else None, "natural_silver_selection": [{"sample_id": row.sample_id, "label": row.label[HEAD], "text": row.model_text} for row in silver], "role_negative_silver": role_negative_silver, "natural_role_probes": [{"sample_id": row.sample_id, "label": row.label[HEAD], "text": row.model_text} for row in natural_probes], "human_class_counts": dict(Counter(str(row.label[HEAD]) for row in human)), "synthetic_class_counts": dict(Counter(str(row.label[HEAD]) for row in synthetic)), "silver_class_counts": dict(Counter(str(row.label[HEAD]) for row in silver)), "human_train_exclusions": train_exclusions, "legacy_audit": legacy_audit, "max_epochs": MAX_EPOCHS, "selected_epoch": epoch, "epoch_selection": "five-fold OOF high-evidence Human Train macro-F1" if human_evidence_weights else "five-fold OOF all Human Train macro-F1; synthetic probes and every Human Check excluded", "train_pairs": TRAIN_PAIRS, "held_out_probes": HELD_OUT_PROBES})
    m12._dump(output / "cross-validation.json", {"selection_human_oof_macro_f1": oof_metric["macro_f1"], "all_human_oof_macro_f1": all_human_oof["macro_f1"], "selection_count": len(selection_records), "curve": curve})
    # The current selected weights are loaded independently.  Human Check
    # answers are first accessed only after the candidate checkpoint exists.
    current = prior._make_model(torch, AutoModel, snapshot, schema, config, parent, device)
    check5, check_exclusions = prior._sparse_records("check", schema)
    check20, _ = m5.load_human_check_pool(prior.REPO_ROOT / "runs" / "feedback-loop-v3-finalized-20260910")
    recent19, _ = m12.load_recent_feedback_records(prior.RECENT_ROOT / "human" / "recent-check19-valid-final.jsonl", prior.RECENT_ROOT / "feed" / "feed-input.jsonl", role="historical_recent_check", expected_samples=19)
    sets = {"owner_recent_check5": check5, "historical_human_check20": check20, "historical_recent_check19": recent19, "synthetic_held_out_probes": probes}
    if natural_probes:
        sets["expert_provisional_natural_role14"] = natural_probes
    evaluations = {name: _eval_set(torch, current, model, tokenizer, records, config, schema, device) for name, records in sets.items()}
    del model, current
    gc.collect()
    if device_name == "mps":
        torch.mps.empty_cache()
    reloaded = _make_model(torch, AutoModel, snapshot, schema, config, parent, torch.device("cpu"), two_block=two_block)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    reloaded.specialist_blocks[HEAD].load_state_dict(saved["specialist_block_state_dict"])
    reloaded.heads[HEAD].load_state_dict(saved["head_state_dict"])
    cpu_probs = prior._predict(torch, reloaded, tokenizer, probes[:1], config, schema, torch.device("cpu"))
    _require(bool(np.isfinite(cpu_probs).all()) and bool(np.allclose(cpu_probs.sum(axis=1), 1.0, atol=1e-6)), "ACTION_CONTRAST_CPU_RELOAD", "CPU reload inference failed")
    m12._dump(output / "evaluation.json", {"evaluation_roles": {"owner_recent_check5": "human held-out from this fit, already exposed in previous candidate evaluation", "historical_human_check20": "human historical reused diagnostic, not independent selection", "historical_recent_check19": "human historical reused diagnostic, not independent selection", "synthetic_held_out_probes": "synthetic different phrasings, not Human Gold", "expert_provisional_natural_role14": "real posts with text-only expert labels frozen before this candidate fit; diagnostic silver check, not owner Gold"}, "check_exclusions": check_exclusions, "sets": evaluations})
    result = {"stage_id": stage_id, "two_block": two_block, "status": "CANDIDATE_TRAINED_NOT_PROMOTED", "human_train": len(human), "human_evidence_downweighted": len(evidence_audit), "synthetic_train": len(synthetic), "natural_silver_train": len(silver), "selected_epoch": epoch, "selection_human_oof_macro_f1": oof_metric["macro_f1"], "all_human_oof_macro_f1": all_human_oof["macro_f1"], "elapsed_seconds": round(time.monotonic() - started, 2), "cpu_reload": "PASS", "set_summary": {name: {"count": value["count"], "current_accuracy": value["current"]["accuracy"], "candidate_accuracy": value["candidate"]["accuracy"], "current_three_way_correct": value["current_three_way_correct"], "candidate_three_way_correct": value["candidate_three_way_correct"], "current_opposite_sign": value["current_opposite_sign"], "candidate_opposite_sign": value["candidate_opposite_sign"]} for name, value in evaluations.items()}}
    m12._dump(output / "summary.json", result)
    print(json.dumps({"event": "complete", "output": str(output), **result}, ensure_ascii=False), flush=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train one isolated Action contrastive candidate")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--two-block", action="store_true", help="train original LERT block 10 plus selected Action block 11")
    parser.add_argument("--natural-silver", action="store_true", help="also train on finite expert-reviewed natural-post silver examples")
    parser.add_argument("--human-evidence-weights", action="store_true", help="reduce training weight of explicit Action-contract conflicts without changing owner answers")
    parser.add_argument("--role-negative-silver", action="store_true", help="extend silver training with real other-actor and negated-action examples")
    args = parser.parse_args(argv)
    try:
        run(output_dir=args.output_dir, two_block=args.two_block, natural_silver=args.natural_silver, human_evidence_weights=args.human_evidence_weights, role_negative_silver=args.role_negative_silver)
    except ContractError as exc:
        print(json.dumps({"status": "BLOCKED", "code": exc.code, "details": exc.details}, ensure_ascii=False), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
