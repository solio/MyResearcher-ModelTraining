# AGENTS.md

This file is the highest-level repository constraint for every human or AI
executor working in `MyResearcher-ModelTraining`.

## Current scope

The active delivery is **Milestone 1 — reproducible training engineering and a
diagnostic TF-IDF baseline**. The repository measures observable post-level
`belief–emotion–action state`. It does not predict prices, recommend trades, or
produce the seven downstream group states.

Milestone 1 explicitly excludes Encoder training/downloads, LoRA, services,
GUIs, production inference over 49,054 posts, aggregation, author research,
external fact augmentation, crawlers, and trading strategy work.

## Sources of truth

- The archived implementation draft is
  `docs/specs/semantic-student-training-and-acceptance-v0.1.md`; its provenance
  is recorded next to it.
- The frozen label source is the versioned
  `schema/semantic-schema-calibrated-v0.2.1.json`, exported once from the
  workbook named in its `source` object. Runtime code never parses the workbook.
- `docs/milestone-1-acceptance.md` freezes acceptance and negative cases before
  implementation.
- `manifests/local-data-inventory.json` records the observed local snapshot.
- Large upstream data are read-only and remain outside Git.

## Hard boundaries

1. Join labels to canonical inputs only by `sample_id`. Missing, duplicate,
   extra, or conflicting identities fail closed. Never fuzzy-match text or
   infer identity from stock/time.
2. Canonical input, teacher candidate/weak label, reviewed label, adjudicated
   Human Gold, Anchor, prediction, challenge, quarantine, and embargo are
   distinct roles. Prediction never becomes annotation; agreement/confidence
   never automatically promotes teacher output to Gold.
3. Repeated `stock_code`, `stock_name`, and `published_at` in a label file must
   agree with canonical input. Canonical time always comes from canonical input;
   Excel serial timestamps in teacher files never define a split.
4. Schema versions never mix silently. Any migration is explicit, versioned,
   fail-closed, and emits a per-field report.
5. One `build_model_input(record, contract)` implementation is shared by
   prepare, train, evaluate, export/inference. A provisional contract cannot be
   reported as the v0.3.5 contract.
6. A frozen split manifest is mandatory. Never synthesize or randomly rebuild
   the missing 1,822/448/467/242 split. Quarantine, duplicate/echo groups, event
   boundaries, and cross-set identity leakage are fatal.
7. Weights are `sample_id × prediction_head`, versioned in configuration/data.
   A single global sample weight is insufficient.
8. Evidence dependencies gate training. Violations enter an explicit
   quarantine and are never downgraded to warnings.
9. Abstention is per head and versioned with class order, probabilities,
   calibration, and threshold. Abstention never maps to `NEUTRAL`,
   `NONE_EXPLICIT`, or `NO_ACTION_SIGNAL`.
10. Unit/synthetic tests prove at most `TESTED`. Only a reproducible run on real
    frozen inputs, labels, split, and reviewed evidence can be
    `DATA_VALIDATED`; domain acceptance is `EXPERT_ACCEPTED`.
11. All critical conclusions carry `CONFIRMED`, `PROVISIONAL`, `HYPOTHESIS`, or
    `REJECTED`. If one root cause survives two local repair rounds, stop adding
    compatibility branches and return to the owning contract/architecture.
12. Never modify `formal_run`, `MyResearcher-DataClean`, or
    `MyResearcher-DataCollector`. Never commit CSV/DB/XLSX/model weights or
    checkpoints to ordinary Git history.

## Semantic invariants

- `UNKNOWN` is not `NEUTRAL`.
- `NONE_EXPLICIT` is not `CALM`.
- `NO_ACTION_SIGNAL` is not `WATCH`.
- Bearish is not automatically fear; bullish is not automatically buy; buy is
  not automatically excitement.
- Wish, conditional action, other people's action, and advice to others are not
  the author's completed action.
- “不割肉” is not `SELL`; “不追高” is not `BUY`.
- Refusal/abstention is not an ordinary taxonomy class.

## Git and execution

- Work on a feature branch; never force-push or overwrite `main`.
- Persist specifications, decisions, hashes, manifests, commands, and blocker
  codes in Git rather than relying on conversations.
- CPU execution is mandatory. CI uses synthetic fixtures and never downloads a
  large model.
- A real training entry point must call the audit gate first. Missing canonical
  artifacts produce `BLOCKED_MISSING_CANONICAL_ARTIFACTS`, a non-zero exit, and
  no fabricated label, split, run, or model artifact.

