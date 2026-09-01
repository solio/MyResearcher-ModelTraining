# RBT3 reasoning corrective diagnostic

`semantic_model.encoder_m2_reasoning_corrective` implements
`RBT3_REASONING_CORRECTIVE_V1`, a reasoning-only M2 diagnostic. It uses the
existing Train/Dev loader, `encoder_m1.build_input_ids`, per-sample/per-head
weights, metric implementation, and CPU reload smoke. Preflight reads only
Train (1,822) and Dev (448) and the already-present fixed RBT3 cache.

Each run starts from the original `hfl/rbt3@0aa0527ff4170f29e1dfd3eb6ef60dc67e1bf75c`
weights; no S1/S2/S3 checkpoint is used. The embeddings and first two
Transformer blocks remain frozen. Only the final block (`1e-5`) and the
`reasoning_tags` head (`5e-4`) are optimized with AdamW (`weight_decay=0.01`).
The seven-head module is constructed in the same order as S3 so that a given
seed has the same reasoning-head initialization, then the six non-target heads
are frozen, excluded from loss/optimizer, and omitted from result metrics.

Seeds are fixed at 35, 71, and 107. Input and stopping controls are fixed at
`max_length=256`, `HEAD_TAIL`, batch 16, at most 12 epochs, patience 3, and
gradient clipping 1.0. Dev `reasoning_tags.macro_f1` alone drives early
stopping; Micro-F1, exact-set accuracy, and all 15 per-label F1 values are
diagnostic outputs. S3 is the single-variable last-block comparison; S2 is
descriptive only because its head learning rate and stopping score differ; S1
is the frozen shared baseline.

Example (run only after the implementation commit and with the existing local
cache; the output directory must be new):

```sh
PYTHONPATH=src .encoder-venv/bin/python -m semantic_model.encoder_m2_reasoning_corrective \
  --config configs/baseline_v0.3.5.yaml \
  --cache-dir .encoder-artifacts/hf-cache \
  --output-dir /Users/mac/Documents/trae_projects/MyResearcher/model-artifacts/m2-rbt3-reasoning-corrective-v1-20260901
```

The output is always diagnostic (`selected_candidate=false`). A passing gate
means `PASSED_DIAGNOSTIC_ONLY`; any non-finite value, resource limit, missing
checkpoint, CPU reload failure, or comparison failure is retained as
`REJECTED_DIAGNOSTIC_ONLY`. No Test, Anchor, Gold, OOD, reference prediction,
LLM, cloud, download, or production path is available to this entry point.
