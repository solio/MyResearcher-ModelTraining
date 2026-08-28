# Architecture Handoff and Model Roadmap

Status: `ACTIVE_HANDOFF_CONTRACT`

Aligned with the project owner: 2026-08-28

Audience: every future human developer, reviewer, and AI agent working in this
repository.

## 1. Why this document exists

Recent owner/developer discussions clarified an important terminology and
architecture issue that must not be lost when work changes hands:

- in this project, **foundation/base NLP model** means a pretrained
  discriminative Encoder such as BERT, RoBERTa, or MacBERT;
- it does **not** mean a generative large language model;
- a BERT-class Encoder is the intended primary production-model direction;
- the current TF-IDF + Logistic Regression implementation is a frozen
  diagnostic and regression baseline, not the final production architecture;
- a frontier generative LLM should be used only a limited number of times for
  offline review, challenge, verification, falsification, disagreement
  analysis, or annotation assistance;
- an LLM is not intended to sit in the normal per-post production inference
  path and its answer never becomes Gold automatically.

Every new executor must read this document before proposing model work. It is a
roadmap and handoff contract, not authorization to expand the active milestone.
The current `AGENTS.md` scope remains controlling: Encoder downloads and
training begin only after the project owner explicitly opens that milestone.

## 2. Executive decisions

These decisions are settled unless the project owner explicitly revises them:

1. **The current classical model is not the final model.** It exists to
   reproduce v0.3.5, validate data learnability, expose label drift, provide a
   cheap regression oracle, and make future Encoder gains measurable.
2. **The future primary semantic student is a pretrained Chinese Encoder.** It
   should share one contextual representation and expose six single-label
   classification heads plus one 15-label Reasoning head.
3. **The classical and Encoder models should coexist.** The Encoder is the
   production candidate; the classical model remains a control, regression
   sentinel, and disagreement source.
4. **A generative LLM is an offline reviewer, not the primary classifier.** Its
   finite calls should focus on difficult, high-risk, low-confidence, OOD, and
   model-disagreement cases.
5. **Human or explicitly governed Expert adjudication remains the truth gate.**
   Neither an Encoder prediction, a classical prediction, confidence, model
   agreement, nor an LLM review automatically promotes a row to Gold.
6. **OOD is not `UNKNOWN`.** Vocabulary coverage, distribution shift, and
   unsupported input require an explicit OOD/abstention layer and reason code.
7. **The frozen v0.3.5 lineage must never be silently improved.** Solver,
   features, thresholds, vocabulary, OOD behavior, or model architecture
   changes create a new version and cannot retain the v0.3.5 reproduction name.
8. **Exact baseline reproduction and Encoder R&D are different workstreams.**
   Exact reproduction protects historical provenance. Encoder R&D targets
   better semantic performance. Once the owner opens the next milestone, they
   may proceed in parallel without changing the frozen baseline.

## 3. Terminology that future executors must use

| Term | Meaning in this repository | Does not mean |
| --- | --- | --- |
| Classical baseline | Character/word TF-IDF plus Logistic Regression | Production semantic model |
| Foundation/base NLP model | Pretrained Encoder such as BERT/RoBERTa/MacBERT | Necessarily a generative LLM |
| Encoder student | Fine-tuned discriminative post-level semantic model | Teacher, reviewer, or Gold source |
| Generative LLM | Limited offline review/verification/falsification assistant | Routine classifier for every post |
| `UNKNOWN` | The text does not support a taxonomy decision for one field | Out-of-distribution input |
| Abstention | A versioned decision not to rely on a head prediction | A normal taxonomy class |
| OOD | The input is outside the validated model/data distribution | Merely a low-frequency class |
| Gold | Human/Expert adjudicated truth under explicit provenance | A model prediction or agreement |
| Reproduced | Same accepted reference environment and every frozen oracle passes | A cross-platform similar run |
| Production candidate | A new version passing the future Gold/OOD/quality gates | The reproduced v0.3.5 artifact |

Do not describe the current artifact as "BERT-free by design forever". The
accurate statement is: **Milestone 1 deliberately implements the classical
baseline first; a BERT-class Encoder is the intended next primary model.**

## 4. Current implementation truth

Snapshot date: 2026-08-28. Future executors must verify the live repository
before relying on this snapshot.

### 4.1 What is implemented

- read-only audit of the canonical immutable data package;
- read-only audit of the immutable baseline-reference package without
  executing supplied code or loading the external `joblib`;
- exact Schema and seven-head class order validation;
- 21-row Evidence quarantine and 2,979-row trainable weak-label view;
- frozen Train 1,822 / Embargo-1 131 / Dev 448 / Embargo-2 111 / Test 467
  split;
- Anchor50 role/provenance checks;
- explicit `sample_id × prediction_head` weights;
- frozen v0.3.5 input construction and TF-IDF feature contract;
- six independent scalar Logistic Regression heads;
- 15 independent binary Reasoning Logistic Regression estimators;
- Dev threshold selection, abstention metadata, evaluation, errors, model
  export, CPU inference, and immutable content-addressed runs;
- estimator convergence diagnostics and coefficient/intercept fingerprint
  comparison;
- full Train/Dev/Test/Anchor50 reference comparison over 2,787 rows;
- tests for archive safety, fail-closed contracts, data leakage, feature order,
  model structure, export, inference, and reproduction-claim gating.

### 4.2 What the current model is

Input contract:

```text
[股票]{stock_code} {stock_name} [帖子]{model_text}
```

Feature representation:

- character TF-IDF: 2–5 grams, 11,945 fitted features;
- word TF-IDF: 1–2 grams, 313 fitted features;
- frozen stack order: character then word;
- total: 12,258 sparse float32 features;
- vectorizers fit on Train 1,822 only.

Estimators:

- six single-label heads: Logistic Regression, SAGA, `C=3.0`,
  `max_iter=2000`, `class_weight=balanced`, seed 35;
- Reasoning: 15 explicit binary Logistic Regression estimators, liblinear,
  `C=3.0`, `max_iter=1600`, `class_weight=balanced`, seed 35;
- every sample uses its frozen per-field weight; one global sample weight is
  forbidden.

The original and current comparable runs share the same convergence shape:

- all six original scalar SAGA estimators reached 2,000 iterations without
  convergence;
- all six comparable local scalar SAGA estimators also reached 2,000 without
  convergence;
- all 15 original and comparable Reasoning liblinear estimators converged.

### 4.3 Current evidence state

- canonical data content ID:
  `cf7a10f25d951d79607cfd80b70751f11415c2772274e275e6ee1b57f32f470b`;
- baseline-reference content ID:
  `828944580b96d872241a6619bdb8f60dae2cd7067a0cc6741b418f1e6a7bdc85`;
- original model SHA-256:
  `4e1dbe0fe1d4d37be728cebe849630ffd75a1fb6d66988bd15112375e6476b5a`;
- historical metrics recomputed from the frozen original predictions have
  maximum absolute difference `0.0`;
- the verified local macOS arm64 / scikit-learn 1.7.2 run is
  `COMPARABLE_DIAGNOSTIC_RUN_ONLY`;
- exact reproduction remains blocked by
  `BLOCKED_REFERENCE_ENVIRONMENT_MISMATCH`;
- no production approval exists.

Exact v0.3.5 reproduction still requires the accepted Linux x86_64 reference
environment, exact labels for all 2,787 rows, metric absolute tolerance
`1e-12`, and probability absolute tolerance `1e-10`. Even a passing exact run
is only `BASELINE_V0_3_5_REPRODUCED_DIAGNOSTIC_ONLY`.

### 4.4 What is not implemented

- no pretrained Encoder has been selected, downloaded, trained, or exported;
- no BERT/RoBERTa/MacBERT tokenizer contract exists yet;
- no shared Encoder multi-head network exists;
- no frozen/partial/full fine-tuning comparison exists;
- no multi-seed Encoder experiment exists;
- no explicit lexical-coverage or embedding-distance OOD detector exists;
- no production-scale, cross-platform, cross-stock, or temporal generalization
  approval exists;
- no sufficient independent adjudicated Gold set exists;
- no generative LLM review queue, provenance schema, or review budget is
  implemented;
- no 49,054-row production inference has been authorized or run.

Do not report "model development complete" from the classical training result.
The accurate statement is: **the data and historical baseline engineering are
established; the intended Encoder primary model has not yet been trained.**

## 5. Target architecture

The intended system separates routine classification, cheap regression,
distribution safety, and expensive review:

```text
canonical post + minimum board context
                |
                v
     input validation and OOD gate
                |
      +---------+----------+
      |                    |
      v                    v
Chinese pretrained     explicit abstain /
Encoder student        review queue
      |
      v
six Softmax heads + one 15-label Sigmoid head
      |
      +--------------------+
                           |
                           v
          classical/Encoder disagreement check
                           |
            +--------------+--------------+
            |                             |
            v                             v
      normal fixed-Schema output     limited LLM review
                                           |
                                           v
                                 human/Expert adjudication
                                           |
                                           v
                              versioned Gold/challenge data
```

### 5.1 Encoder student responsibilities

The Encoder student should be the normal post-level classifier. It must:

- consume only the declared post and minimum board context;
- run locally/offline after its artifacts are installed;
- output the frozen/versioned taxonomy, probabilities, confidence, and
  abstention metadata;
- use one shared contextual Encoder representation;
- use six single-label Softmax heads with cross-entropy loss;
- use one 15-dimensional Sigmoid Reasoning head with BCE-with-logits loss;
- preserve per-sample, per-field weighting;
- support deterministic manifests, immutable export, and CPU inference;
- expose enough intermediate diagnostics for error analysis and OOD gating;
- never invent Evidence, external facts, positions, or recommendations.

### 5.2 Classical baseline responsibilities

The classical baseline remains permanently useful after the Encoder exists:

- reproducibility and data-pipeline smoke testing;
- cheap CI-compatible regression testing;
- label learnability diagnostics;
- coefficient and feature inspection;
- drift comparison;
- a second opinion for model-disagreement sampling;
- detection of Encoder regressions or multi-task interference.

It must not be deleted merely because a stronger Encoder is introduced.

### 5.3 Generative LLM responsibilities

A frontier LLM may be called a limited number of times for:

- low-confidence or OOD samples;
- classical/Encoder disagreement;
- high-risk BUY/SELL and author-action ownership errors;
- `UNKNOWN/NEUTRAL`, `CALM/NONE_EXPLICIT`, and
  `WATCH/NO_ACTION_SIGNAL` boundary review;
- new slang and event-language investigation;
- falsifying a proposed label using explicit text evidence;
- proposing challenge cases;
- summarizing recurring error modes;
- producing an annotation suggestion for human/Expert review.

An LLM must not:

- classify every production post in the normal path;
- be the only source of a Gold label;
- silently rewrite labels or Evidence;
- promote agreement or confidence to truth;
- add live news, prices, or company facts to a post-level semantic decision;
- make a trade recommendation;
- introduce an unrecorded model or prompt-version dependency.

Every LLM review record should capture at least:

- provider and exact model/version;
- prompt/template version;
- generation settings;
- input hash and Schema version;
- original response and parsed candidate;
- cited in-text Evidence;
- classical and Encoder predictions/probabilities;
- disagreement reason;
- reviewer disposition and edits;
- final provenance role and timestamp.

The default LLM output role is `MODEL_REVIEW_SUGGESTION`, not Gold.

## 6. OOD and unseen-language requirements

The current TF-IDF vocabulary is fixed. Unseen word and character n-grams are
ignored at transform time. Character n-grams provide limited overlap-based
generalization but no semantic understanding. A nearly OOV post may still
receive a confident normal-class prediction from intercepts, template/stock
features, or a few spurious matches.

The current single-label abstention threshold is not an OOD detector, and the
historical Reasoning rule that forces at least one tag is unsafe for production
OOD input. `UNKNOWN` remains a semantic class and must not absorb OOD.

A new version, not v0.3.5, should add:

- character and word vocabulary coverage computed on `model_text`, excluding
  constant template tokens;
- matched-feature count, sparse vector norm, and other input-quality signals;
- nearest-Train cosine similarity for the classical representation;
- Encoder-space distance or another validated OOD score;
- explicit `out_of_distribution` and `abstain_reason` fields;
- an OOD path that does not force a Reasoning tag;
- temporal vocabulary/drift monitoring;
- a frozen OOD/challenge set covering new slang, stocks, periods, platforms,
  unrelated content, code/link/emoji-heavy input, traditional Chinese,
  misspellings, and missing-context cases.

Required OOD reporting should include:

- OOD recall;
- in-domain false-rejection rate;
- high-confidence OOD error rate;
- selective risk at declared coverage;
- retained in-domain field metrics;
- probability calibration;
- performance by time, stock, source, and challenge category.

## 7. Encoder selection and experiment plan

No exact Encoder winner is preselected. BERT, Chinese RoBERTa, MacBERT, a
lightweight Chinese Encoder, or another locally deployable candidate may enter
the benchmark. Selection is by evidence, not model reputation.

Before any model download, the next milestone must freeze:

- candidate IDs, revisions, weight hashes, tokenizer hashes, and licenses;
- local/offline artifact policy;
- input template, maximum length, truncation, padding, and special tokens;
- shared-head architecture and loss aggregation;
- freeze/unfreeze stages;
- optimizer, learning-rate schedule, batch size, seeds, and early stopping;
- per-field sample-weight application;
- Dev/Test/Gold/OOD role separation;
- output Schema and abstention/OOD contract;
- resource budgets and inference latency targets;
- experiment manifest and comparison-report format.

### 7.1 Recommended experiment ladder

Run increasingly flexible experiments instead of jumping directly to full
fine-tuning on 1,822 weak-label Train rows:

1. **Frozen Encoder + trained heads** — measures the value of pretrained
   representations with minimal overfitting risk.
2. **Partial unfreeze** — unfreeze the last layers with layer-specific learning
   rates and early stopping.
3. **Full fine-tuning** — only after earlier evidence and a stronger Gold set;
   use warmup, weight decay, gradient clipping, layer-wise learning-rate decay,
   and multiple seeds.
4. **Single-task versus shared multi-task comparison** — detect negative
   transfer between Stance, Emotion, Action, Context, and Reasoning.
5. **Classical versus Encoder disagreement analysis** — route informative
   disagreements to limited LLM and human review.

At least three seeds should be reported for serious Encoder candidates. Report
mean, dispersion, and worst seed rather than selecting one favorable run.

### 7.2 Data readiness

The current weak-label data can support initial experiments but not production
approval. The existing specification calls for at least 100–200 new independent
adjudicated Gold rows covering critical boundaries. A stronger production
target of 300–500 rows should be considered and explicitly approved by the
owner.

Gold collection should prioritize:

- author action versus another person's action or advice;
- negation and conditional action;
- wish versus completed action;
- `UNKNOWN/NEUTRAL`;
- `CALM/NONE_EXPLICIT`;
- `WATCH/NO_ACTION_SIGNAL`;
- rare BUY/ADD/REDUCE/SELL and FOMO cases;
- sarcasm, wordplay, new slang, and missing context;
- multi-label Reasoning combinations;
- cases where classical, Encoder, LLM, and human judgment disagree.

Gold used repeatedly for model decisions cannot remain an untouched final test.
If Gold is promoted into training, create a new version and retain a new unseen
adjudicated set.

### 7.3 Selection evidence

Do not choose an Encoder from a single aggregate score. Compare:

- every scalar head's Macro-F1 and per-class precision/recall/support;
- Reasoning micro-F1, macro-F1, per-label metrics, and exact-set behavior;
- high-risk semantic boundary error rates;
- OOD and abstention performance;
- probability calibration and high-confidence errors;
- seed stability;
- performance by stock, time, and source;
- CPU latency, accelerator throughput, peak memory, model size, and export
  complexity;
- offline reproducibility, license, artifact retention, and dependency risk.

The candidate numerical production thresholds in the archived specification
remain provisional until an adequate independent Gold set is frozen.

## 8. Milestone roadmap

### Milestone 1A — Classical baseline engineering

Current state: substantially complete at
`COMPARABLE_DIAGNOSTIC_RUN_ONLY`.

Remaining bounded gate:

- run the frozen code in the accepted Linux x86_64 reference environment;
- require exact 2,787-row labels;
- require metrics absolute tolerance `1e-12`;
- require probability absolute tolerance `1e-10`;
- if successful, label only
  `BASELINE_V0_3_5_REPRODUCED_DIAGNOSTIC_ONLY`.

This gate never authorizes production.

### Milestone 1B — Encoder selection contract

Current state: planned, not yet authorized or implemented.

Deliverables:

- candidate and license matrix;
- tokenizer/input contract;
- shared multi-head architecture specification;
- training and seed policy;
- Gold/OOD data plan;
- evaluation and resource gates;
- immutable model-artifact and provenance contract;
- explicit authorization to download and train the selected candidates.

### Milestone 2 — Encoder student experiments

Current state: not started.

Deliverables:

- frozen, partial-unfreeze, and justified full-fine-tune runs;
- single-task/multi-task comparison;
- classical/Encoder disagreement dataset;
- field, class, Gold, OOD, calibration, seed, and resource reports;
- limited LLM review workflow and provenance records;
- a selected production candidate or a documented rejection.

### Milestone 3 — Production-candidate acceptance

Current state: not started.

Requires:

- adequate untouched adjudicated Gold;
- critical semantic-boundary error gates;
- OOD and abstention gates;
- temporal, stock, and source robustness evidence;
- model artifact/security/license review;
- CPU or accelerator performance acceptance;
- downstream author-deduplication and group-state evaluation;
- explicit owner approval before any 49,054-row full inference.

## 9. Actions that require explicit new scope

The roadmap does not itself authorize these actions under the current
Milestone 1 scope:

- downloading BERT/RoBERTa/MacBERT or any other model weights;
- adding PyTorch/Transformers production dependencies;
- training an Encoder;
- creating or paying for LLM review jobs;
- sending project data to an external model/API;
- labeling new Gold without an adjudication protocol;
- changing the frozen v0.3.5 model or reference policy;
- running production inference over 49,054 posts.

Before doing any of them, obtain explicit owner authorization, update the
active milestone, freeze its acceptance contract, and record the decision.

## 10. Mandatory handoff checklist

At the beginning of any future model task, the executor must state:

1. which milestone and model lineage it is working on;
2. whether the work changes frozen v0.3.5 or creates a new version;
3. whether it is classical, Encoder, OOD, LLM review, Gold, or production work;
4. what data roles are used for training, Dev, Test, Gold, and OOD;
5. whether external model downloads or APIs are required and authorized;
6. what evidence and acceptance gates will decide success;
7. how the work preserves the classical baseline as a control;
8. how predictions, model reviews, Expert acceptance, and Human Gold remain
   distinct provenance roles.

Before marking a future task complete, the executor must report:

- exact model/artifact IDs and hashes;
- data, Schema, tokenizer, configuration, code, dependency, and environment
  identities;
- per-head and per-class metrics;
- Gold and OOD results;
- seed variability;
- abstention and high-confidence errors;
- resource measurements;
- remaining blockers;
- whether production is explicitly authorized.

## 11. Short handoff summary

If context is limited, preserve at least this statement verbatim:

> The current TF-IDF + Logistic Regression artifact is a frozen diagnostic and
> regression baseline, not the final production model. In this project,
> "foundation/base NLP model" means a pretrained Encoder such as BERT,
> RoBERTa, or MacBERT. A Chinese Encoder with six single-label heads and one
> 15-label Reasoning head is the intended primary model direction once the
> owner explicitly opens that milestone. A generative LLM is reserved for a
> limited offline review, verification, falsification, and disagreement
> workflow; its output never becomes Gold automatically and it is not the
> routine per-post inference model. OOD is separate from `UNKNOWN`, and no
> model may run the 49,054-row production workload without explicit acceptance.
