# M2 final specialist runner

`semantic_model.encoder_m2_final_specialist` is the last registered RBT3
candidate stage. It uses the fixed local `hfl/rbt3` snapshot and the existing
Train/Dev loader only. The embeddings and Transformer blocks 0–1 are shared
and frozen; each task owns an independent copy of block 2 and its output head.
Each head/seed run is independent and starts from the original snapshot. No
S1/S2/S3/R1/R2 checkpoint is used as a warm start.

Reasoning is fail-fast. Seeds 35, 71 and 107 train `reasoning_tags` first;
thresholds are selected from Train only on the 0.05–0.95 grid and then frozen
for Dev. The Classical feasibility gate is evaluated before any other head is
started. A failed gate writes `RBT3_M2_CANDIDATE_REJECTED` evidence and stops.
Only a passed reasoning gate permits the six remaining specialist heads.

The final artifact contains one checkpoint per head/seed, Train/Dev metrics,
resource logs, threshold files, a final Classical gate, and a seed-35 unified
bundle. The bundle is checked on CPU against each standalone seed-35 head for
finite, numerically equivalent output. Even a passed gate is
`M2_SELECTED_CANDIDATE_FROZEN_FOR_M3`, never production approval or bulk
inference.

Example command (run only after the implementation/contract commit):

```text
PYTHONPATH=src .encoder-venv/bin/python -m semantic_model.encoder_m2_final_specialist \
  --config configs/baseline_v0.3.5.yaml \
  --cache-dir .encoder-artifacts/hf-cache \
  --output-dir /path/to/new/m2-final-specialist-artifact
```

The fixed contract extension is recorded in
`manifests/encoder-m2-experiment-contract-v1.json` under
`final_specialist_candidate_contract`. Test, Anchor, Gold, OOD, reference
predictions, LLM/cloud calls, production inference, and new corrective stages
remain prohibited.
