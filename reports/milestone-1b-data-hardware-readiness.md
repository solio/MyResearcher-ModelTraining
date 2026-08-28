# Milestone 1B — data and hardware readiness audit

Status: MILESTONE_1B_SELECTION_CONTRACT_READY_FOR_OWNER_APPROVAL
Audit date: 2026-08-28
Evidence vocabulary: CONFIRMED, PROVISIONAL, HYPOTHESIS, BLOCKED

This is a read-only readiness report. It neither downloads a model/tokenizer nor
installs an Encoder dependency, trains an Encoder, invokes a generative LLM,
creates Gold, changes v0.3.5, or runs 49,054 production rows.

## Method and reproducible command

The semantic_model.audit_encoder_readiness CLI is network-independent and
imports neither torch nor transformers. It does not duplicate package or role
validation. Before calculating distributions it calls the only canonical gate,
semantic_model.audit_data.run_audit, and fails closed unless that gate returns
an eligible real-data/reference-bound state, a non-empty canonical audit ID,
and raw canonical training_allowed=true. It then compares the canonical return
directly to the static data and reference content pins and the reference-to-data
binding. It reads only JSON/JSONL and system metadata and emits sorted JSON to
stdout; it never writes a data, model, run, or training artifact.

The real audit used the project runtime while reading the original immutable
package location from the baseline configuration:

    PYTHONPATH=src /Users/mac/Documents/trae_projects/MyResearcher/MyResearcher-ModelTraining/.venv/bin/python \
      -m semantic_model.audit_encoder_readiness \
      --config /Users/mac/Documents/trae_projects/MyResearcher/MyResearcher-ModelTraining/configs/baseline_v0.3.5.yaml

Canonical command result: exit code 0; status
READY_FOR_COMPARABLE_DIAGNOSTIC_RUN; canonical audit ID
5fab05d633c509122bb8bbddd95b5d79f8d76a660b284f5cb20120df2865e414.
It reports raw training_allowed=true, data content ID
cf7a10f25d951d79607cfd80b70751f11415c2772274e275e6ee1b57f32f470b,
reference content ID
828944580b96d872241a6619bdb8f60dae2cd7067a0cc6741b418f1e6a7bdc85,
and their verified reference binding.

Readiness command result: exit code 0; planning-only status
MILESTONE_1B_SELECTION_CONTRACT_READY_FOR_OWNER_APPROVAL; readiness audit ID
f8e740feb75ebeedc43bbf3653ddf821d56653a7afcb953043c824fb9f89bfb5.
It propagates the canonical status/ID/pins verbatim but keeps
selection_or_training_authorized=false.

From this worktree, the equivalent relative readonly command is:

    PYTHONPATH=src ../../MyResearcher-ModelTraining/.venv/bin/python -m semantic_model.audit_encoder_readiness --config ../../MyResearcher-ModelTraining/configs/baseline_v0.3.5.yaml

The ID covers the stable package, role, contract, and runtime facts. Free disk
is intentionally reported as a time-sensitive observation but excluded from
the reproducibility identity, so ordinary filesystem activity does not change
the audit ID.

## 1. Immutable package and role readiness

| Check | Result | Fact level |
| --- | --- | --- |
| Canonical immutable data package | 28/28 payloads, paths, byte sizes, and SHA-256 values verified | CONFIRMED |
| Data content ID | cf7a10f25d951d79607cfd80b70751f11415c2772274e275e6ee1b57f32f470b | CONFIRMED |
| Baseline-reference package | 17/17 payloads, paths, byte sizes, and SHA-256 values verified | CONFIRMED |
| Reference content ID | 828944580b96d872241a6619bdb8f60dae2cd7067a0cc6741b418f1e6a7bdc85 | CONFIRMED |
| Schema | semantic-schema-calibrated-v0.2.1; ordered seven-head labels validated | CONFIRMED |
| Canonical/weak label relation | 3,000 inputs = frozen/repaired labels; quarantine 21; trainable weak labels 2,979 | CONFIRMED |
| Field weights | all 20,853 sample_id × prediction_head entries validated | CONFIRMED |
| Anchor isolation | Anchor50 has zero Teacher3000 overlap | CONFIRMED |
| New independent adjudicated Gold | no versioned artifact found | BLOCKED |
| Versioned OOD set | no artifact found | BLOCKED |

The training-role interpretation is strict:

- The 2,979 rows are weighted weak labels, not Gold.
- Dev and Test retain their frozen weak-label diagnostic roles.
- Anchor50 is a fixed diagnostic set, not an unbiased final test.
- Teacher prediction, model agreement, Anchor, expert agreement, and future
  LLM suggestions do not become Human Gold.

## 2. Split, stock, and time readiness

| Population | Rows | Interval | Fact level |
| --- | ---: | --- | --- |
| Train | 1,822 | calendar date <= 2026-07-30 | CONFIRMED |
| Embargo-1 | 131 | 2026-07-31 | CONFIRMED |
| Dev | 448 | 2026-08-01..2026-08-06 | CONFIRMED |
| Embargo-2 | 111 | 2026-08-07 | CONFIRMED |
| Test | 467 | calendar date >= 2026-08-08 | CONFIRMED |
| Embargo total | 242 | two frozen boundary dates | CONFIRMED |
| Trainable weak-label total | 2,979 | 2026-07-09 through 2026-08-15 | CONFIRMED |
| Canonical input total | 3,000 | includes 21 quarantined rows | CONFIRMED |
| Anchor50 | 50 | separate fixed diagnostic data role | CONFIRMED |

The 16-stock trainable distribution is balanced at 184–188 rows per stock:

| Stock | Rows | Stock | Rows |
| --- | ---: | --- | ---: |
| 002028 思源电气 | 188 | 002463 沪电股份 | 186 |
| 002648 卫星化学 | 187 | 002891 中宠股份 | 187 |
| 300054 鼎龙股份 | 186 | 300487 蓝晓科技 | 185 |
| 300666 江丰电子 | 185 | 600312 平高电气 | 187 |
| 601012 隆基绿能 | 186 | 601888 中国中免 | 187 |
| 603039 泛微网络 | 186 | 603179 新泉股份 | 186 |
| 603806 福斯特 | 184 | 603997 继峰股份 | 186 |
| 605020 永和股份 | 186 | 688676 金盘科技 | 187 |

The CLI's stable JSON contains the exact per-calendar-date counts for every
Train/Dev/Test/Embargo population. The highest observed train dates are
2026-07-30 (149), 2026-07-10 (142), and 2026-07-31 is correctly embargoed
(131); low-volume calendar dates remain part of the frozen chronology and are
not resampled.

## 3. Training-class support

All counts here are Train weak-label support, not Gold support. They are usable
for experiment planning but cannot formally establish production acceptance.

### Scalar heads

| Head | Ordered class support |
| --- | --- |
| target_mode | ON_TARGET 1,333; CROSS_TARGET 320; MARKET_GENERAL 97; UNKNOWN 72 |
| stance | BULL 271; BEAR 513; NEUTRAL 457; MIXED 18; UNKNOWN 563 |
| emotion_primary | FEAR 113; ANXIETY 198; ANGER 200; FRUSTRATION 383; REGRET 71; HOPE 140; EXCITEMENT 166; FOMO 3; CALM 158; NONE_EXPLICIT 365; UNKNOWN 25 |
| emotion_target | PRICE 611; POSITION 375; COMPANY 208; MARKET 118; OTHER 114; NOT_APPLICABLE 365; UNKNOWN 31 |
| action_tendency | BUY 85; ADD 25; HOLD 138; DO_T 28; REDUCE 23; SELL 121; WATCH 156; NO_ACTION_SIGNAL 1,240; UNKNOWN 6 |
| context_dependency | SELF_CONTAINED 1,314; PARTIAL_CONTEXT 428; EXTERNAL_CONTEXT_REQUIRED 80; UNKNOWN 0 |

### Reasoning tags

| Tag | Positive | Negative |
| --- | ---: | ---: |
| FUNDAMENTAL | 191 | 1,631 |
| VALUATION | 59 | 1,763 |
| TECHNICAL_PRICE | 695 | 1,127 |
| FLOW_POSITIONING | 327 | 1,495 |
| NEWS_EVENT | 139 | 1,683 |
| RUMOR | 19 | 1,803 |
| SOCIAL_PROOF | 148 | 1,674 |
| MACRO_POLICY | 90 | 1,732 |
| THEME_NARRATIVE | 198 | 1,624 |
| NO_REASON_GIVEN | 316 | 1,506 |
| RELATIVE_PERFORMANCE | 210 | 1,612 |
| CROSS_STOCK_REFERENCE | 177 | 1,645 |
| SARCASM_IRONY | 134 | 1,688 |
| WORDPLAY | 138 | 1,684 |
| UNKNOWN | 62 | 1,760 |

The report flags—but does not turn into a new acceptance threshold—the current
reporting floor of 20 weak-label Train examples. Below it are stance MIXED
(18), emotion-primary FOMO (3), action-tendency UNKNOWN (6), context-dependency
UNKNOWN (0), and Reasoning RUMOR (19).

Consequently, no class has formal independent-Gold acceptance evidence today.
The listed five classes are additionally unsuitable for a strong per-class
claim from weak-label Train evidence alone. In Anchor50, several classes also
have zero or tiny support (for example action DO_T 0, emotion CALM 0/FOMO 0,
and context UNKNOWN 0), reinforcing that Anchor50 cannot close this gap.

## 4. Field-weight readiness

The immutable confidence distribution is HIGH 1,947, MEDIUM 699, LOW 333 for
each prediction head. It is not a global sample-weight replacement. Drift rules
make the effective total differ by head:

| Prediction head | Effective weight total | Interpretation |
| --- | ---: | --- |
| target_mode | 2,600.994 | confidence-weighted baseline |
| stance | 2,507.745 | stance-specific drift reductions preserved |
| emotion_primary | 1,823.560 | emotion-specific drift reductions preserved |
| emotion_target | 1,823.560 | emotion-specific drift reductions preserved |
| action_tendency | 2,244.873 | action-specific drift reductions preserved |
| reasoning_tags | 2,600.994 | confidence-weighted baseline |
| context_dependency | 2,600.994 | confidence-weighted baseline |

The audit JSON retains, for every head, the exact numeric-weight frequency and
the effective HIGH/MEDIUM/LOW subtotal. Any Encoder implementation must retain
this field-level multiplication separately from head-loss and class weights.

## 5. Raw input text readiness

The text facts apply to all 3,000 canonical inputs, deliberately including the
21 quarantined rows. They are pre-tokenizer measurements.

| Measure | Character count | UTF-8 byte count |
| --- | ---: | ---: |
| Minimum | 1 | 1 |
| Median (p50) | 13 | 37 |
| Mean | 24.111 | 63.525 |
| p90 | 40 | 111 |
| p95 | 115 | 284 |
| p99 | 181 | 441 |
| Maximum | 472 | 1,264 |

| Observation | Count | Fact level / method |
| --- | ---: | --- |
| non-string model_text | 0 | CONFIRMED direct type inspection |
| empty string | 0 | CONFIRMED |
| whitespace-only text | 0 | CONFIRMED |
| URL marker (http://, https://, or www.) | 3 | CONFIRMED for the deterministic substring rule |
| Han + ASCII-Latin mixture | 350 | CONFIRMED for the stated code-point rule |
| emoji-codepoint-range observation | 0 | CONFIRMED for fixed Unicode ranges, not a semantic emoji claim |
| conservative Traditional-character indicator | 0 | CONFIRMED for the stated indicator set, not a full script classifier |
| control/format Unicode code point | 0 | CONFIRMED for Unicode categories Cc/Cf |

No token length, tokenizer vocabulary coverage, unknown-token rate, or
truncation rate is claimed. They are
BLOCKED_TOKENIZER_ARTIFACT_NOT_AVAILABLE until an owner-approved tokenizer
retrieval occurs.

## 6. Anchor, Gold, OOD, and challenge readiness

Anchor50 has exactly 11 HUMAN_CONFIRMED and 39 EXPERT_PREADJUDICATED records.
Its expert-confidence distribution is HIGH 28, MEDIUM 13, LOW 9. This is a
useful fixed diagnostic/reference set but not a separate untouched final Gold
set.

| Evidence | Verified count | State |
| --- | ---: | --- |
| Separate independent adjudicated Gold | 0 | BLOCKED_NO_SEPARATE_VERSIONED_INDEPENDENT_GOLD_ARTIFACT |
| Versioned OOD examples | 0 | BLOCKED_NO_VERSIONED_OOD_ARTIFACT |
| Anchor50 HUMAN_CONFIRMED subset | 11 | diagnostic Anchor subset, not independent final Gold |
| Anchor50 EXPERT_PREADJUDICATED subset | 39 | diagnostic Anchor subset, not Human Gold |

The recommended first challenge/Gold sampling order is: author action versus
others' action or advice; negation and conditional action; wishes versus
completed action; UNKNOWN/NEUTRAL; CALM/NONE_EXPLICIT; WATCH/NO_ACTION_SIGNAL;
BUY/ADD/REDUCE/SELL and FOMO; sarcasm/wordplay/new slang/context dependency;
multi-label Reasoning; and new stock/sector/time/platform/character-perturbation
OOD. This is a HYPOTHESIS for sampling priority, not a new label or Gold
decision.

## 7. Hardware and runtime readiness

The official project runtime used for the audit is CPython 3.12.13 at the
baseline repository's .venv path.

| Capability | Observation | Readiness conclusion |
| --- | --- | --- |
| OS / CPU architecture | macOS 26.5.2, Darwin arm64, 12 logical CPUs | CONFIRMED hardware fact |
| Memory | 32.0 GiB | CONFIRMED |
| Free disk at audit | 711.347 GiB | CONFIRMED snapshot, not a reserved budget |
| Project Python | CPython 3.12.13 | CONFIRMED |
| torch, transformers, datasets, peft in project .venv | all absent | CONFIRMED; no installation was performed |
| Apple Silicon / MPS | hardware may support | PROVISIONAL; training runtime not verified because torch is absent |
| CUDA | no nvidia-smi; no Linux GPU environment evidence | BLOCKED |
| Ambient shell Anaconda runtime | separate CPython 3.13.9 environment has pre-existing torch 2.10.0, transformers 4.57.6, datasets 4.5.0, PEFT 0.18.1 | CONFIRMED observation; it was not used for this contract and does not grant project authorization |

The next decision is not merely “can this laptop run it?” The owner must select
the supported execution policy: CPU-only, MPS plus mandatory CPU inference,
Linux CUDA, or more than one target. Batch size, latency, throughput, training
duration, and model-size limits are BLOCKED pending the approved artifact and
runtime benchmark.

## 8. Readiness conclusion

**Data:** CONFIRMED ready for governed Encoder experiments as weighted weak
labels, with immutable package identity and all required data-role/split/weight
checks passing. It is BLOCKED for formal Encoder acceptance because there is no
independent adjudicated Gold or OOD artifact.

**Hardware:** CONFIRMED as an Apple-Silicon 32 GiB local planning target with
mandatory future CPU inference. MPS is only a possible hardware path, not a
verified project training runtime; current Linux CUDA evidence is absent.

**Final Milestone 1B state:**
MILESTONE_1B_SELECTION_CONTRACT_READY_FOR_OWNER_APPROVAL.

Required owner decisions are candidate/revision/license; download permission;
PyTorch/Transformers permission; CPU/MPS/CUDA policy; resource/model-size/
latency budget; post-tokenizer max length; Gold/OOD quantity; LLM-review
permission and budget; and the next-stage training resources.

## 9. Engineering verification

| Command | Result | Evidence level |
| --- | --- | --- |
| `.venv/bin/python -m compileall -q src tests` | Exit 0 | `CONFIRMED` |
| .venv/bin/python -m pytest -q tests/test_encoder_readiness.py | 13 passed | CONFIRMED |
| `.venv/bin/python -m pip check` | `No broken requirements found.` | `CONFIRMED` |
| `git diff --check` | Exit 0 | `CONFIRMED` |
| .venv/bin/python -m pytest -q | Exit 0; full suite passed | CONFIRMED |
| .venv/bin/python -m semantic_model.audit_data --config ABSOLUTE_SOURCE_CONFIG | Exit 0; canonical READY_FOR_COMPARABLE_DIAGNOSTIC_RUN; audit ID 5fab05d…e414 | CONFIRMED |
| .venv/bin/python -m semantic_model.audit_encoder_readiness --config ABSOLUTE_SOURCE_CONFIG | Exit 0; planning status; audit ID f8e740…bfb5 | CONFIRMED |

The full suite is now a prerequisite satisfied by this review-fix, not an
ignored import gap. The isolated source-integrity cherry-pick adds the tracked
frozen Classical package; it does not reconstruct, mutate, execute, or
unpickle the original reference model.
