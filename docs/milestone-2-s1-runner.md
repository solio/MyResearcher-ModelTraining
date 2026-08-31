# M2-S1 direct-owner runner

Status: `DIRECT_OWNER_AUTHORIZED_PENDING_LOCAL_TECHNICAL_PREFLIGHT`

`encoder_m2_s1.py` runs only
`M2-S1-FROZEN-SHARED-SEVEN-HEAD-CONTROL`: frozen `hfl/rbt3` at the fixed
revision, seven shared heads, and seeds 35/71/107. It is diagnostic-only and
never names an M2 selected candidate.

```bash
python -m semantic_model.encoder_m2_s1 \
  --config configs/baseline_v0.3.5.yaml \
  --output-dir /absolute/new-m2-s1-output \
  --cache-dir .encoder-artifacts/hf-cache
```

The direct owner instruction is sufficient. There is no additional execution
identity, branch, remote, worktree, or remote-sync runtime gate.

Before importing Torch/Transformers or loading a model, the runner only:

- loads the immutable Train 1,822 and Dev 448 split labels plus their selected
  canonical inputs and per-head weights;
- validates the frozen schema/class order and Train/Dev counts;
- hashes all eight existing local fixed-revision snapshot files.

It deliberately does not call the full canonical audit and does not open Test,
Anchor, Gold, OOD, the reference package, or reference predictions. After
runtime import it verifies the frozen local package versions, then validates a
new immutable output location and runs MPS-first with CPU fallback.

Each seed writes a checkpoint, Train/Dev weak-label diagnostic metrics,
critical-boundary report, resource log, and CPU reload smoke. Non-finite
values, resource excess, checkpoint/reload failures, runtime exceptions, and
mixed devices fail closed. A failure after output creation records a
content-addressed rejected manifest; no normal aggregate or selected candidate
is emitted.
