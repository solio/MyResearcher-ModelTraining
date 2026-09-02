# M2 LERT-small final specialist candidate

`semantic_model.encoder_m2_lert_final` is the one owner-authorized new-model
attempt after the RBT3 lineage was rejected. It is bound to
`hfl/chinese-lert-small@69e3e69ba258be5b301b26937e5b55a076c90460` under the
Apache-2.0 license. The runner may download only this exact revision into its
separate `.encoder-artifacts/hf-cache-lert-small` cache and uses
`trust_remote_code=false`; dependency versions are not changed.

The architecture shares the LERT tokenizer/input builder, embeddings, and
Transformer blocks 0–10. Each of the seven tasks has an independent copy of
block 11 and its own output head. A run trains one selected block/head at a
time, never warm-starts from RBT3 or any earlier artifact, and the eventual
bundle (only if all gates pass) returns all seven outputs from one input.

Reasoning runs first for seeds 35, 71, and 107. Its 15 thresholds are selected
on Train 1,822 only using the R2 grid and then frozen on Dev 448. The same
Classical Macro/Micro/Exact, stability, and critical-label gates are evaluated.
Failure emits `LERT_SMALL_M2_CANDIDATE_REJECTED` and stops before the six other
heads. Success permits the six independent specialists and the final frozen
Classical gate; only that complete path could emit
`M2_SELECTED_CANDIDATE_FROZEN_FOR_M3`, never production approval.

No Test, Anchor, Gold, OOD, reference predictions, LLM, cloud, or production
inference path is loaded. Every completed run writes its checkpoint, metrics,
thresholds where applicable, resource log, and CPU finite-output reload. New
output directories are immutable and prior RBT3 evidence remains untouched.
