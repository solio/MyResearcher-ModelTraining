# Milestone 1B — Chinese Encoder selection and acceptance contract

Status: `M1_DIAGNOSTIC_TRAINING_COMPLETED_WEAK_LABEL_DIAGNOSTIC_ONLY`
Contract version: `encoder-experiment-contract-v1`
Date: 2026-08-28
Evidence vocabulary: `CONFIRMED`, `PROVISIONAL`, `HYPOTHESIS`, `BLOCKED`

This records the completed bounded M1 first run. The exact owner-authorized
artifact was retrieved and hashed, the isolated runtime was created, the
Train-plus-Dev tokenizer audit froze configuration before fit, and one frozen
seven-head diagnostic run completed on MPS with mandatory CPU reload/inference
smoke passing. Paid LLM review, new Gold/OOD creation, Test-based selection,
cloud/external APIs, and 49,054-row inference remain out of scope. This status
does **not** mean `ENCODER_ACCEPTED` or `PRODUCTION_READY`.

The machine-readable companion is
[`manifests/encoder-experiment-contract-v1.json`](../manifests/encoder-experiment-contract-v1.json).
The observed data and hardware facts are recorded in
[`reports/milestone-1b-data-hardware-readiness.md`](../reports/milestone-1b-data-hardware-readiness.md).

## 1. Frozen scope, terminology, and roles

`基础模型` means a pretrained discriminative Chinese NLP Encoder in the
BERT/RoBERTa/MacBERT family. It does not default to a generative LLM.

The following ownership boundary is `CONFIRMED` and remains unchanged:

| Layer | Role | Not allowed to become |
| --- | --- | --- |
| Chinese pretrained Encoder | Future normal post-level semantic-model candidate | Gold source or a trading model |
| Frozen TF-IDF + Logistic Regression v0.3.5 | Permanent diagnostic, regression, and disagreement baseline | Production Encoder substitute |
| Generative LLM | Finite offline review, verification, falsification, and difficult-case analysis | Routine per-post classifier or automatic Gold source |
| Human/Expert adjudication | Final Gold truth entry | A model-confidence consequence |

The immutable data lineage is also unchanged: data package
`cf7a10f25d951d79607cfd80b70751f11415c2772274e275e6ee1b57f32f470b`,
baseline-reference package
`828944580b96d872241a6619bdb8f60dae2cd7067a0cc6741b418f1e6a7bdc85`,
and label Schema `semantic-schema-calibrated-v0.2.1`.

`UNKNOWN` is a semantic taxonomy value; it is not OOD. Abstention is a
versioned decision; it is not `UNKNOWN`, `NEUTRAL`, `NONE_EXPLICIT`, or
`NO_ACTION_SIGNAL`.

## 2. Selected M1 artifact — one exact HFL RBT3 revision

M1 has exactly one owner-authorized model and tokenizer identity. It is not a
candidate comparison and no floating `main`, `latest`, mirror, substitute, or
second model may be retrieved for this run.

- **Official ID:** `hfl/rbt3`.
- **Required revision:**
  `0aa0527ff4170f29e1dfd3eb6ef60dc67e1bf75c` (40 hexadecimal characters).
- **License:** Apache-2.0, accepted by the owner.
- **Load policy:** `trust_remote_code=false`; download only from the official
  `hfl/rbt3` repository at the required revision; record the resolved commit,
  every retrieved file, SHA-256, byte size, license evidence, and cache path.
- **Architecture expectation:** a three-layer 768-dimensional, 12-head,
  BERT-compatible Chinese Encoder. The actual loaded configuration and
  parameter count are execution evidence and must be recorded after retrieval.
- **Runtime policy:** create an isolated Encoder runtime; use MPS first when
  PyTorch validates it, otherwise use CPU. CPU checkpoint reload and one local
  inference smoke test are mandatory in either case.
- **Resource policy:** keep the total additional local disk footprint at or
  below 10 GiB and stop the single diagnostic run at two hours. Actual size,
  device, throughput, memory observation, and duration are measured rather
  than estimated.

The artifact and tokenizer were retrieved from the fixed revision, MPS was
validated and used, the isolated runtime was retained, and CPU reload/inference
smoke passed. M2 may compare other models only after M1 exits; it is not a
current M1 prerequisite.

## 3. M2 model selection and stability requirements

No model name and no one-number score may select the future Encoder. These are
M2 selection requirements, not M1 first-run blockers. Every M2
candidate/stage must report all of the following, by seed and by data role:

- Macro-F1 for every scalar head and precision/recall/F1/support for every
  class;
- Reasoning micro-F1, macro-F1, per-label measures, and exact-set/full-label
  behavior;
- critical semantic error rates for author-vs-other action, advice, negation,
  conditional action, wish, `UNKNOWN/NEUTRAL`, `CALM/NONE_EXPLICIT`,
  `WATCH/NO_ACTION_SIGNAL`, BUY/ADD/REDUCE/SELL, FOMO, sarcasm/wordplay, and
  context dependency;
- OOD recall, in-domain false rejection, high-confidence OOD errors,
  selective risk, retained in-domain performance, abstention, and probability
  calibration;
- mean, dispersion, and worst seed from at least three seeds—not the best
  seed alone;
- stock/time/source slices, confusion evidence, and high-confidence errors;
- CPU single-record and batch latency, MPS/GPU throughput where authorized,
  peak memory, serialized size, export complexity, offline reload, license,
  and long-term artifact retention.

The v0.1 numerical thresholds in the archived student specification are
`PROVISIONAL`. They cannot become production gates until the owner freezes
adequate independent adjudicated Gold and OOD evidence. A pooled metric cannot
mask a material head regression, critical semantic regression, calibration
failure, abstention failure, or worst-seed instability.

## 4. Frozen direction for the shared Encoder and seven heads

The future network direction is frozen; it is deliberately not implemented in
this milestone:

```text
one shared Chinese pretrained Encoder
  ├── target_mode         Softmax + CrossEntropyLoss
  ├── stance              Softmax + CrossEntropyLoss
  ├── emotion_primary     Softmax + CrossEntropyLoss
  ├── emotion_target      Softmax + CrossEntropyLoss
  ├── action_tendency     Softmax + CrossEntropyLoss
  ├── context_dependency  Softmax + CrossEntropyLoss
  └── reasoning_tags      15-dimensional Sigmoid + BCEWithLogitsLoss
```

The shared representation is the final-layer `last_hidden_state[:, 0, :]`
(`CLS` representation), rather than a pretrained pooler output. Each task head
is a single affine projection after shared dropout `0.1`; no extra head MLP is
assumed. The frozen Schema provides every class order.

The loss contract keeps three concepts separate:

| Concept | Meaning | Contract |
| --- | --- | --- |
| Sample field weight | Reliability/drift weight for one `sample_id × prediction_head` | Multiply that sample/head loss only; use immutable v0.3.5 expansion |
| Head loss weight | Explicit aggregate weight of one task in a multi-task loss | A separately versioned config scalar; initial candidate is 1.0 per head |
| Class weight | Per-class CE weighting or per-tag BCE `pos_weight` | Disabled unless a Dev ablation justifies it; never substitutes either other weight |

M1 uses only the shared seven-head Encoder. Separate single-task comparators,
their equal-budget shared comparison, and head-by-head negative-transfer
reporting are mandatory M2 controls, not prerequisites to the first run.

Each scalar output retains its full ordered probability vector. Reasoning
retains a 15-label probability vector. Abstention is per head, calibrated only
on Dev, and saved with class order and threshold. Reasoning thresholds are
saved per tag or as an explicit global policy. An OOD or abstained record must
not be forced to contain a Reasoning tag.

## 5. Tokenizer and input contract

No tokenizer was downloaded or loaded in Milestone 1B. Therefore every token
length claim is `BLOCKED`; the report contains raw character and UTF-8 byte
lengths only.

The pre-tokenizer input contract for the next approved audit is:

```text
[CLS] {stock_code_or_empty} [SEP] {stock_name_or_empty} [SEP] {model_text} [SEP]
```

- Only `stock_code`, `stock_name`, and `model_text` may enter. They represent
  declared board context only; no news, price, company fact, external lookup,
  author metadata, or generated context is permitted.
- Use the model's existing `[CLS]` and `[SEP]` tokens. `[STOCK]` and `[POST]`
  are not added, and tokenizer-vocabulary modification is forbidden unless a
  separately approved migration and parity plan exists.
- Normalize Unicode to NFC and `CRLF`/`CR` line endings to `LF`; preserve all
  other whitespace. Preserve URLs, emoji, Traditional Chinese, mixed language,
  and special characters rather than deleting or translating them.
- An absent `stock_code` or `stock_name` is an empty delimited segment plus a
  recorded missing-field flag. A non-string, empty, or whitespace-only
  `model_text` fails closed.
- Dynamic right padding to the batch maximum and the associated attention mask
  are required. Padding side, mask generation, tokenizer revision/hash,
  normalization, template, and all special-token IDs are export fields.
- The completed Train-plus-Dev audit selected `max_length=256`: coverage was
  96.5639% at 128, 99.9119% at 256, and 99.9559% at 384. It used no Test label
  or Test metric. Character lengths were not used as a proxy.
- The frozen pre-fit configuration is code cap 8, name cap 16, HEAD_TAIL
  truncation, dynamic right padding with attention masks, batch size 16, seed
  35, AdamW (`lr=0.0005`, `weight_decay=0.01`), up to 12 epochs, and patience
  3 on the declared Dev diagnostic score. It was not mutated in response to
  Dev diagnostics.

### 5.1 Input-builder decision block — no implicit tokenizer behaviour

The prose template is now a frozen executable tokenizer contract. The versioned
encoder-input-builder-v1 is shared unchanged
by M1 Train, Dev diagnostic, checkpoint reload smoke, and later governed
consumers. It rejects non-string, empty, and whitespace-only model_text;
NFC-normalizes each allowed source field and converts CRLF/CR to LF before
tokenization. It records stock_code_missing and stock_name_missing as input
metadata only; those flags are never model features or token IDs.

The builder uses manual three-segment ID assembly after tokenizing each field
with add_special_tokens=false; it must not use a text_pair shortcut. It
assembles exactly one existing CLS ID and three existing SEP IDs:

    [CLS] code_ids [SEP] name_ids [SEP] model_text_ids [SEP]

Therefore a component call with add_special_tokens=true is forbidden and
duplicated CLS/SEP is mechanically impossible. Special vocabulary additions
remain forbidden. The builder emits only input_ids and a matching right-padded
attention_mask; it does not emit, infer, or silently discard token_type_ids.
This portable decision avoids candidate-specific segment-ID behaviour.

padding_side is right. The special-token budget is exactly four IDs, making
the content budget max_length - 4; an implementation rejects a maximum below
four. Code and name segments are retained before body text. Their explicit
caps and the selected maximum are frozen by the authorized post-download
Train-plus-Dev audit; no character-length proxy, Test label, or Test metric may
supply them.

The only body-truncation candidates are RIGHT (retain the first remaining body
IDs) and HEAD_TAIL (retain ceil(remaining / 2) leading plus
floor(remaining / 2) trailing body IDs, with no inserted special token).
HEAD_TAIL is the deterministic M1 policy because action/conclusion language
often occurs late. Before fitting, the completed audit recorded
`max_length=256`, code/name token caps 8/16, dynamic right padding, batch 16,
seed 35, AdamW, and the stopping condition. The immutable run artifacts show
that no configuration mutation followed fit start.

## 6. M1 first run, M2 selection, and M3 acceptance

### M1_FIRST_RUN — owner-authorized minimal seven-head loop

M1 is authorized to execute exactly one frozen shared Encoder with six single-label heads and one
15-label Reasoning head. It uses one seed, fits Train 1,822, uses Dev only for
early stopping and diagnostics, does not use Test for selection, and never
treats Anchor as Gold. The M1 deliverables are a checkpoint, config, ordered
class contract, artifact hashes, weak-label diagnostic metrics, and reload
inference smoke test. No M1 metric is a production claim.

M1 P0 retains the immutable input builder, Schema, frozen split,
sample_id-by-head weights, artifact identity, license review, and owner
authorization. Missing independent Gold/OOD, three seeds, single-task
comparators, multi-candidate comparison, calibration, or complete resource
benchmarking does not block the first run.

### M2_MODEL_SELECTION_AND_STABILITY

M2 requires three or more seeds; approved Google BERT/MacBERT/RBT3 or other
candidate comparison; single-task versus shared multi-task controls;
partial/full unfreeze; negative-transfer, rare-class, class-imbalance,
tokenizer length/truncation, calibration, and disagreement analysis. The
frozen v0.3.5 Classical baseline remains the regression/disagreement control.

### M3_PRODUCTION_ACCEPTANCE

M3 requires independent adjudicated Gold, a versioned OOD set, abstention,
formal Test unseal, production thresholds, critical semantic-boundary gates,
and an explicit production-candidate acceptance or rejection decision.

### 6.1 Evaluation-role integrity and unseal protocol

Train is the only fitting role. In M1, Dev is limited to early stopping and
diagnostics for the already owner-selected artifact/configuration. In M2, Dev
alone selects candidate identity, freeze stage, architecture, maximum length,
truncation policy, optimizer and other hyperparameters, class/loss policy,
calibration, and thresholds. Test, Anchor, and Embargo never make those
selections.

Test is sealed until candidate, stage, all hyperparameters, seed aggregation,
code commit, and evaluation manifest are frozen. It then has one formal,
recorded unseal; repeated default test peeking is prohibited. Anchor50 remains
a fixed regression/disagreement/challenge diagnostic, is not a selection
dataset, and is not Gold. Each embargo population must be explicitly reserved
for one future temporal-validation purpose (or one separately named use);
neither may become a repeated default validation split.

Independent adjudicated Gold and a versioned OOD set are still absent. That
blocks formal production acceptance regardless of weak-label, Dev, Test, or
Anchor performance.

## 7. Data, Gold, OOD, and LLM review acceptance

The current verified state is:

- Train 1,822, Dev 448, Test 467, Embargo 131 + 111, trainable weak labels
  2,979, canonical inputs 3,000, and Anchor50 50;
- Anchor50 source distribution is exactly 11 `HUMAN_CONFIRMED` and 39
  `EXPERT_PREADJUDICATED`; it is a fixed diagnostic set, not an unbiased final
  Gold test;
- no separately versioned independent adjudicated Gold artifact exists;
- no versioned OOD artifact exists.

`MODEL_PREDICTED`, `LLM_REVIEW_SUGGESTION`, `EXPERT_ACCEPTED`, and
`HUMAN_ADJUDICATED` are distinct provenance values. Teacher output, Anchor,
expert agreement, classical/Encoder agreement, and an LLM suggestion never
automatically become Human Gold.

The minimum recommendation is 100–200 independently adjudicated Gold records.
The stronger 300–500 production-candidate target is a recommendation, not an
owner-approved hard gate. Gold sampling must prioritize author action,
advice/other-person action, negation, conditionals, wishes,
`UNKNOWN/NEUTRAL`, `CALM/NONE_EXPLICIT`, `WATCH/NO_ACTION_SIGNAL`, rare action
classes and FOMO, sarcasm, new slang, context dependence, and multi-label
Reasoning. Gold reused for training can no longer remain an untouched final
acceptance set.

The future OOD set must separately classify new stocks, sectors, time ranges,
slang, other platforms, unrelated text, pure links, pure emoji, mixed language,
Traditional Chinese, OCR/typos, very short/long inputs, missing/image-dependent
context, and character/rule attacks. `OOD != UNKNOWN` and `OOD != low
confidence` are frozen semantics.

Future generative LLM use is limited to a reviewed queue: Classical/Encoder
disagreement, OOD, low confidence, high-risk BUY/SELL, slang, Evidence
verification, falsification, challenge cases, and error patterns. Each call
must preserve provider, model/version, prompt version, generation settings,
input hash, Schema, raw response, parsed candidate, in-text Evidence, both
model predictions, disagreement reason, human disposition, and final
provenance. Its default result is `MODEL_REVIEW_SUGGESTION`.

## 8. Resource and immutable export gate

The completed run used an Apple-Silicon (`arm64`) MPS device in an isolated
CPython 3.12.13 runtime with torch 2.8.0, Transformers 4.57.6, tokenizers
0.22.2, and NumPy 2.5.2. It completed in 79.161 seconds, used a 157,525,306
byte model/tokenizer cache plus 763,544-byte diagnostic artifact directory,
and passed CPU reload/inference with finite logits for all seven outputs. The
Classical project `.venv` and ambient Anaconda runtime were not modified. The
full isolated M1 footprint, including the runtime and retained retry evidence,
remained far below the 10 GiB limit; the two-hour limit was not approached.

Every selected export must include candidate ID, resolved full revision, model
and tokenizer SHA-256 values, license copy, Schema/class order, input contract,
max length, truncation/padding, architecture/dropout, three distinct weight
types, calibration, abstention/OOD policy, data content ID, code commit,
dependency lock, resources, and all seeds. Weights, checkpoints, local runs,
and large datasets remain outside ordinary Git history.

## 9. Owner authorization, tokenizer gate, and deferred work

### A. OWNER_AUTHORIZED_M1_ARTIFACT_RUNTIME_RESOURCE_BUNDLE

The owner authorizes exactly one M1 artifact/runtime/resource bundle:

- model and tokenizer: hfl/rbt3 at revision
  0aa0527ff4170f29e1dfd3eb6ef60dc67e1bf75c;
- Apache-2.0 license accepted;
- official model and tokenizer download authorized;
- isolated Encoder-runtime dependency installation authorized;
- MPS-first training with CPU fallback authorized;
- CPU checkpoint reload/inference verification required;
- frozen Encoder, seven trainable heads, one seed;
- Train 1,822; Dev only for early stopping and diagnostic reporting;
- local additional disk limit: 10 GiB;
- single-run wall-time limit: two hours.

Execution evidence is complete: official artifact/tokenizer download true,
isolated runtime/dependencies true, MPS validation true, tokenizer audit true,
training started/completed true, and CPU checkpoint reload/inference smoke
true. Production approval remains false.

### B. POST_DOWNLOAD_TOKENIZER_AUDIT_PRETRAIN_CONFIGURATION

After the authorized tokenizer was retrieved and hashed, the completed
Train-plus-Dev input-length audit froze max_length, segment caps, truncation,
padding, batching, seed, optimizer, and stopping condition before fit. It used
neither Test labels nor Test metrics. This was an M1 implementation gate, not
an M2/M3 selection exercise; the shared builder, Schema, frozen split,
sample_id-by-head weights, artifact identity, and license record remained M1
P0.

### C. DEFERRED_NOT_REQUIRED_FOR_M1

The following are not M1 prerequisites: Gold, OOD, LLM review, multiple
models, three seeds, single-task comparators, partial/full unfreeze, production
thresholds, formal Test selection, and 49,054-row inference. They remain
governed M2/M3 work. No cloud service, external API, LLM, Gold/OOD creation,
Test-based selection, production approval, or production inference is
authorized by this bundle.

The valid current state is:

    M1_DIAGNOSTIC_TRAINING_COMPLETED_WEAK_LABEL_DIAGNOSTIC_ONLY

It records one bounded execution, not a production result, and does not broaden
the frozen v0.3.5 lineage or permit production work.

## 10. Completed M1 diagnostic evidence

The content-addressed run artifact is
`.encoder-artifacts/m1-rbt3-0aa0527f/`, with content address
`3cd8d53cae5ec7346595163c227b4cef8abfd90c5e63c37578c8fc48dc147685`.
The official fixed-revision `pytorch_model.bin` is 156,380,647 bytes with
SHA-256 `3e04f7477f55dffce2a2fbc4d0ba35068415162a9e92e3d5cc74a49781ba4eb0`.

| Dev weak-label diagnostic | Macro-F1 |
| --- | ---: |
| target_mode | 0.326448 |
| stance | 0.375895 |
| emotion_primary | 0.167174 |
| emotion_target | 0.258074 |
| action_tendency | 0.132341 |
| reasoning_tags | 0.152158 |
| context_dependency | 0.330764 |

These are weak-label diagnostic metrics from the single authorized seed, not
Gold, Test, OOD, model-selection, or production evidence.
