# Milestone 1B — Chinese Encoder selection and acceptance contract

Status: `MILESTONE_1B_SELECTION_CONTRACT_READY_FOR_OWNER_APPROVAL`
Contract version: `encoder-experiment-contract-v1`
Date: 2026-08-28
Evidence vocabulary: `CONFIRMED`, `PROVISIONAL`, `HYPOTHESIS`, `BLOCKED`

This is a planning and readiness contract. It authorizes neither an Encoder
download nor tokenizer retrieval, dependency installation, training, paid LLM
review, new Gold creation, or 49,054-row inference. In particular, this status
does **not** mean `ENCODER_TRAINED`, `ENCODER_ACCEPTED`, or
`PRODUCTION_READY`.

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

## 2. Candidate set — no winner is selected in Milestone 1B

All candidate facts below were read from official model pages, configuration
files, official repositories, licenses, or papers. No candidate artifact was
downloaded. A future approved download must re-resolve the ref, record every
file SHA-256, and reject a mismatch; the values below are a selection shortlist,
not a local artifact attestation.

### 2.1 Classic candidate — Google Chinese BERT base

- **ID and official sources:** `google-bert/bert-base-chinese` at the
  [official Hub revision](https://huggingface.co/google-bert/bert-base-chinese/tree/8f23c25b06e129b6c986331a13d8d025a92cf0ea), the
  [official Google BERT repository](https://github.com/google-research/bert),
  and the [BERT paper](https://arxiv.org/abs/1810.04805).
- **Observed revision:** `8f23c25b06e129b6c986331a13d8d025a92cf0ea`.
  `CONFIRMED` as the observed Hub ref; it must be re-confirmed at the approved
  retrieval time before it can become an artifact lock.
- **Architecture/tokenizer:** BERT, 12 layers, 768 hidden dimensions, 12
  attention heads, 512 max positions, 21,128-token WordPiece vocabulary. The
  official Hub inventory reports 102,882,442 safetensors parameters; the
  downstream load must separately record whether the count is Encoder-only or
  includes a masked-language-model head.
- **License:** Apache-2.0 for the official BERT release and the official Hub
  card. Tokenizer and model license are both recorded as Apache-2.0, subject to
  owner/legal confirmation of the exact retrieved distribution.
- **Weight and deployment:** the official listing shows a roughly 412 MiB
  PyTorch/safetensors weight file plus a small tokenizer vocabulary. Local
  offline deployment is `PROVISIONAL`: it becomes possible only after an
  owner-approved retrieval, license copy, revision lock, hash manifest, and
  tested local reload.
- **Runtime/resource position:** CPU inference is a mandatory future test;
  MPS and CUDA are framework/runtime tests, not claimed properties of this
  un-downloaded model. A 12-layer base model is the quality/reference control;
  its training batch, latency, throughput, and peak memory are `BLOCKED` until
  measured.
- **Known limitations:** generic Chinese pretraining does not establish
  finance-forum robustness, taxonomy accuracy, calibration, OOD handling, or
  action-ownership understanding. The official BERT source notes Chinese
  simplified and traditional coverage, but this does not replace a project OOD
  evaluation.
- **Impersonation risk:** `MEDIUM`. Historical aliases and community mirrors
  exist. Accept only the `google-bert` namespace, the locked revision, official
  license, full file set, and locally produced SHA-256 manifest.

### 2.2 Chinese-enhanced candidate — HFL MacBERT base

- **ID and official sources:** `hfl/chinese-macbert-base` on the
  [HFL official Hub page](https://huggingface.co/hfl/chinese-macbert-base), its
  [official configuration](https://huggingface.co/hfl/chinese-macbert-base/blob/main/config.json), the
  [HFL Chinese-BERT-wwm repository](https://github.com/ymcui/Chinese-BERT-wwm),
  and the [MacBERT paper](https://arxiv.org/abs/2004.13922).
- **Observed revision:** `a986e00`, an official-Hub displayed abbreviated
  commit. This is `PROVISIONAL`: the next authorized task must resolve and pin
  the full commit SHA before any file is accepted.
- **Architecture/tokenizer:** BERT-compatible MacBERT base, 12 layers, hidden
  size 768, 12 attention heads, 512 max positions, and a 21,128-token
  WordPiece/BertTokenizer vocabulary. The parameter scale is approximately
  102–103M; exact loaded Encoder parameter count is `BLOCKED` until the
  approved source files are retrieved and hashed.
- **License:** the official HFL Hub card declares Apache-2.0; the HFL
  [repository license](https://github.com/ymcui/Chinese-BERT-wwm/blob/master/LICENSE)
  is Apache-2.0. Model and tokenizer use must still be owner-reviewed against
  the exact distribution retained in the artifact store.
- **Weight and deployment:** the official listing exposes a roughly 412 MiB
  PyTorch weight file. Offline local deployment is conditional on the same
  retrieval, hash, retention, and reload steps as the classic candidate.
- **Runtime/resource position:** CPU is required to remain a viable inference
  path. MPS and Linux CUDA are `BLOCKED` pending an approved PyTorch runtime
  and benchmark. No batch size or training-duration estimate is presented as a
  measurement.
- **Known limitations:** its pretraining objective is not evidence for this
  project's seven heads, weak-label reliability, special text forms, finance
  slang, abstention, or OOD. No tokenizer-length distribution was fabricated.
- **Impersonation risk:** `MEDIUM`. Only `hfl/chinese-macbert-base` at a full
  resolved revision is eligible; a similarly named community fine-tune is not a
  substitute.

### 2.3 Lightweight local candidate — HFL RBT3

- **ID and official sources:** `hfl/rbt3` on the
  [HFL official Hub page](https://huggingface.co/hfl/rbt3), its
  [official configuration](https://huggingface.co/hfl/rbt3/blob/main/config.json),
  and the [HFL Chinese-BERT-wwm repository](https://github.com/ymcui/Chinese-BERT-wwm).
  The HFL repository describes RBT3 as the three-layer RoBERTa-wwm-ext model.
- **Observed revision:** `0aa0527`, an official-Hub displayed abbreviated
  commit. It is `PROVISIONAL`; an authorized retrieval must lock a full SHA.
- **Architecture/tokenizer:** three layers, 768 hidden dimensions, 12 attention
  heads, 512 max positions, 21,128 WordPiece vocabulary, loaded through the
  BertTokenizer/BertModel-compatible family. The architecture implies roughly
  38–39M parameters; only an approved downstream load may publish an exact
  count.
- **License:** Apache-2.0 on the official Hub card and HFL repository license,
  with the exact downloaded model/tokenizer license retained in the future
  artifact manifest.
- **Weight and deployment:** the official listing shows a roughly 156 MiB
  PyTorch weight file. It is the resource/latency comparator, not a presumed
  accuracy winner. Three layers reduce parameter/storage cost, but a 768-wide
  representation still requires real sequence, batch, and activation-memory
  measurement.
- **Runtime/resource position:** CPU, MPS, and CUDA claims are all deferred to
  the same approved runtime benchmark. Its intended local/offline use is
  conditional on source verification and artifact retention.
- **Known limitations and impersonation risk:** generic Chinese pretraining
  and a shorter network do not validate semantic boundaries or OOD. `MEDIUM`
  fork risk applies: accept only `hfl/rbt3`, a full resolved revision, hashes,
  and an official license record.

## 3. How selection will work

No model name and no one-number score may select the future Encoder. Every
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

Both controls are mandatory: train a separate single-task Encoder comparator
for each head, then train the shared multi-task model with the same data/split
budget. Negative transfer must be reported head by head.

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
- Max sequence length is unresolved pending a real candidate-tokenizer audit.
  The comparison candidates are 128, 256, and 384. The owner chooses only
  after Train/Dev/Test/Anchor/challenge coverage is measured with the selected
  artifact; character lengths must not be used as a proxy.
- Right truncation and text head-tail preservation are both explicit candidate
  policies. The selected policy must be compared, recorded, and byte-for-byte
  identical in train, evaluation, export, and inference.

## 6. Experiment ladder and control groups

All Encoder candidates use frozen Train/Dev/Test roles, schema/class order,
field weights, and at least three seeds. Dev alone selects early stopping,
calibration, and thresholds. Test and Anchor remain diagnostic until the owner
freezes an independent Gold/OOD protocol.

1. **Experiment A — frozen Encoder.** Freeze every Encoder parameter and train
   only the seven heads. It measures pretrained representation value with lower
   overfit risk.
2. **Experiment B — partial unfreeze.** Unfreeze the last 2–4 layers as an
   explicit experiment variable. Encoder LR must be lower than head LR. Use
   early stopping and record instability.
3. **Experiment C — full fine-tuning.** Enter only if A/B evidence and the
   Gold plan support it. Require warmup, weight decay, gradient clipping,
   layer-wise LR decay, dropout, early stopping, and three or more seeds.

The fixed v0.3.5 classical baseline remains mandatory. A new Classical+OOD
baseline may be added in a new version. For each selected candidate, retain
single-task and shared-multi-task controls plus a Classical/Encoder
disagreement report.

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

`CONFIRMED` local facts are a 32 GiB Apple-Silicon (`arm64`) Mac with 12 logical
CPUs and roughly 711 GiB free disk at audit. The project `.venv` is CPython
3.12.13 and has no torch, transformers, datasets, or PEFT installed. MPS is
therefore `hardware may support; runtime training unverified`; no current Linux
CUDA evidence exists. These are not permissions to install or use an Encoder
runtime.

The next authorized milestone must measure—not estimate as fact—CPU batch-1
and batch-N latency, accelerator throughput, peak memory, serialized size, and
offline reload. CPU inference is mandatory; MPS/CUDA may only be acceleration
paths. The owner must set the model-size, batch-size, latency, training-time,
artifact-retention, and cost budgets after this contract is approved.

Every selected export must include candidate ID, resolved full revision, model
and tokenizer SHA-256 values, license copy, Schema/class order, input contract,
max length, truncation/padding, architecture/dropout, three distinct weight
types, calibration, abstention/OOD policy, data content ID, code commit,
dependency lock, resources, and all seeds. Weights, checkpoints, local runs,
and large datasets remain outside ordinary Git history.

## 9. Owner authorization checklist and final state

The owner must decide the following before Milestone 2 can download or train:

1. Candidate(s), exact full revisions, and license acceptance.
2. Whether official weight/tokenizer downloads are authorized.
3. Whether a named project runtime may install/use PyTorch and Transformers.
4. CPU-only, MPS, Linux CUDA, or a required multi-target support policy.
5. Model-size, artifact-retention, training-time, and cost budgets.
6. Maximum sequence length after the tokenizer audit.
7. Gold quantity, adjudication protocol, and OOD size/source policy.
8. Whether LLM review is allowed, with provider/privacy controls and budget.
9. The exact resources and first authorized experiment stage.

Until those decisions are made, the only valid Milestone 1B completion state
is:

```text
MILESTONE_1B_SELECTION_CONTRACT_READY_FOR_OWNER_APPROVAL
```

It does not broaden the frozen v0.3.5 lineage, authorize production, or permit
the 49,054-post workload.
