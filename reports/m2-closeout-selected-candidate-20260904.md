# M2 closeout — frozen head-selected candidate

Status: `M2_SELECTED_CANDIDATE_FROZEN_FOR_M3` (`selected_candidate=true`,
`production_approval=false`).  This closes the M2 selection work; it does not
authorize production inference.

The immutable selected artifact is
`/Users/mac/Documents/trae_projects/MyResearcher/model-artifacts/`
`m2-selected-candidate-frozen-for-m3-20260904`, content address
`8f73bc437602ce25d6e8ba0c7fd787676a4b566077b1765f3510739ae85928cf`.
Its direct parent is the unchanged M2.2 bundle
`ca2d1399ab252a14d1fb33f4e6ee2581f2253b0d5c63d84ef0ebb5fdad3caf92`.

## Unified sources

- `target_mode`, `stance`, `emotion_primary`, `emotion_target`: LERT seed 35
- `action_tendency`: LERT balanced hard-label seed 35
- `context_dependency`: Classical estimator, mapped by `estimator.classes_` to
  the four Schema classes; absent classes are explicit zero-probability columns
- `reasoning_tags`: LERT's 14 labels; `SARCASM_IRONY` replaced by Classical
  positive probability and threshold `0.45`

The runtime loads `hfl/chinese-lert-small@69e3e69ba258be5b301b26937e5b55a076c90460`
from the existing local snapshot only.  The CPU smoke used the first three Dev
records and returned finite outputs with shapes target 4, stance 5,
emotion_primary 11, emotion_target 7, action 9, context 4, and reasoning 15.
Every scalar probability vector summed to one within the recorded floating-point
tolerance.

## Reproduction command

```text
PYTHONPATH=src .encoder-venv/bin/python -m semantic_model.encoder_m2_selected_runtime \
  --config configs/baseline_v0.3.5.yaml \
  --cache-dir .encoder-artifacts/hf-cache-lert-small \
  --parent-bundle /Users/mac/Documents/trae_projects/MyResearcher/model-artifacts/m2-2-action-balanced-three-seed-20260904/unified-candidate \
  --selected-artifact /Users/mac/Documents/trae_projects/MyResearcher/model-artifacts/m2-selected-candidate-frozen-for-m3-20260904 \
  --smoke
```

No training, download, dependency installation, Test/Anchor/OOD/Human100
access, or production inference was performed.
