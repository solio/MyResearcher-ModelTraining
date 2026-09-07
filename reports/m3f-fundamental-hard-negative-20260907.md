# M3-F FUNDAMENTAL hard-negative diagnostic

Status: `M3F_FUNDAMENTAL_HARD_NEGATIVE_REJECTED`  
Scope: one bounded three-seed Train/Dev experiment; `selected_candidate=false` and no seven-head or production integration.

## Fixed setup

- Initialisation: accepted local LERT specialist checkpoint
  `model-artifacts/m2-lert-final-specialist-seven-head-20260902/reasoning_tags/seed-35/specialist-checkpoint.pt`
- Model cache: local `hfl/chinese-lert-small@69e3e69ba258be5b301b26937e5b55a076c90460` snapshot only; no download.
- Train 1,822 / Dev 448, seeds 35/71/107, MPS-first (all three runs used MPS), CPU reload required.
- Encoder, reasoning specialist block, and the other 14 reasoning rows were frozen. Only the `FUNDAMENTAL` output row was trainable.
- The 48 Train hard negatives are exactly the non-`FUNDAMENTAL` rows containing one of `基金|机构|外资|举牌|定增|股价`; their `FUNDAMENTAL` BCE contribution was multiplied by 3.0. Threshold remained 0.47.

## Old Dev reference and boundary check

The read-only old-model recomputation on Dev gives precision `0.727273`, recall `0.666667`, F1 `0.695652`, support `24`. The fixed boundary-negative set contains 11 Dev rows and the old model has **0 false positives** on that set. Consequently, a requirement to reduce that count in at least two seeds is mathematically unsatisfiable without changing the declared gate.

## Three-seed result

| seed | device | best epoch | Dev precision | Dev recall | Dev F1 | boundary FP | CPU reload | elapsed | checkpoint |
|---:|---|---:|---:|---:|---:|---:|---|---:|---|
| 35 | mps | 1 | 0.727273 | 0.666667 | 0.695652 | 0 | finite | 34.847 s | `reasoning_tags/seed-35/fundamental-hard-negative-checkpoint.pt` |
| 71 | mps | 1 | 0.727273 | 0.666667 | 0.695652 | 0 | finite | 27.253 s | `reasoning_tags/seed-71/fundamental-hard-negative-checkpoint.pt` |
| 107 | mps | 1 | 0.727273 | 0.666667 | 0.695652 | 0 | finite | 24.888 s | `reasoning_tags/seed-107/fundamental-hard-negative-checkpoint.pt` |

Aggregate: precision mean `0.727273` (sample SD `0`), recall mean `0.666667` (sample SD `0`), F1 mean `0.695652` (sample SD `0`), worst-seed F1 `0.695652`. All three seeds retained the same 0 boundary false positives as the old model; all 14 non-target reasoning rows are byte-identical by the recorded frozen-row hash; all CPU reload outputs are finite.

Gate checks:

- mean precision strictly above old: **FAIL** (equal to old);
- mean F1 ≥ `0.675652`: PASS;
- mean recall ≥ `0.616667`: PASS;
- at least 2/3 boundary-FP reductions: **FAIL** (0/3; old count is 0);
- other 14 rows unchanged: PASS;
- CPU reload finite: PASS.

The fixed outcome is therefore `M3F_FUNDAMENTAL_HARD_NEGATIVE_REJECTED`; no replacement candidate was assembled and no second weighting, threshold, model, or training scheme was attempted.

## Artifacts and execution notes

Final rejected diagnostic artifact (new immutable retry directory):
`/Users/mac/Documents/trae_projects/MyResearcher/model-artifacts/m3f-fundamental-hard-negative-three-seed-20260907-retry`  
Content address: `47725d9e0999a73d465be883e3937668fff2ffad6c671c1e93cb7d0059d28443`  
Disk usage: approximately 9.2 MiB (well below 10 GiB).

An earlier partial directory from the checkpoint-loader compatibility failure is preserved, unchanged, at `/Users/mac/Documents/trae_projects/MyResearcher/model-artifacts/m3f-fundamental-hard-negative-three-seed-20260907` with rejected evidence; it is not used as a result artifact. The only code adjustment for the retry was an explicit trusted local `weights_only=False` fallback for the accepted checkpoint's PyTorch metadata, plus permitting a Python 3.12 patch refresh (3.12.14 vs the recorded 3.12.13) while keeping all package versions exact.

No Test, Gold, Human100, OOD, Shadow500, reference predictions, LLM, cloud service, new model download, or dependency installation was used.
