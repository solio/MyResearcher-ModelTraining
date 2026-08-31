# M2-S2 partial-unfreeze runner

`semantic_model.encoder_m2_s2` executes only
`M2-S2-PARTIAL-LAST-ONE-SHARED-SEVEN-HEAD` for the fixed local `hfl/rbt3`
revision. It reuses M1's Train/Dev loader, `build_input_ids`, seven-head class
order, sample-by-head weights, metrics, and CPU reload smoke.

```bash
PYTHONPATH=src .encoder-venv/bin/python -m semantic_model.encoder_m2_s2 \
  --config configs/baseline_v0.3.5.yaml \
  --cache-dir .encoder-artifacts/hf-cache \
  --s1-artifact /Users/mac/Documents/trae_projects/MyResearcher/model-artifacts/m2-s1-first-three-seed-20260831 \
  --output-dir /Users/mac/Documents/trae_projects/MyResearcher/model-artifacts/m2-s2-partial-last-one-three-seed-20260831
```

The runner freezes embeddings and the first two RBT3 Transformer blocks, then
trains only the final block and the seven heads. Heads use learning rate
`3e-4`; the final block uses `1e-5`; both groups use AdamW weight decay `0.01`.
Seeds are 35, 71, and 107, with Train 1,822 for fitting and Dev 448 for early
stopping/diagnostics only. MPS is preferred with CPU fallback, each seed has a
120-minute wall limit and the output has a 10 GiB limit.

After all three seeds, `s2-vs-s1-matching-seed-report.json` compares every
head and critical label to the immutable S1 artifact. It records per-seed,
mean, worst-seed, and standard-deviation deltas. Promotion requires every
stage no-regression rule, no more than two flat/lower heads, and at least two
heads improving by `0.01` in matching-seed mean Macro-F1. A failed report may
identify a future S3 trigger but never starts S3 or selects a model.

The runner does not invoke the canonical audit and never opens Test, Anchor,
Gold, OOD, reference predictions, LLM, cloud, or production paths.
