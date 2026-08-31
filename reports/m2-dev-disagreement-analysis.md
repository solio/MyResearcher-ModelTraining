# M2 Dev-only Encoder / Classical Disagreement Analysis

Status: `M2_DEV_DISAGREEMENT_ANALYSIS_COMPLETED_WEAK_LABEL_DIAGNOSTIC_ONLY`

This is a read-only analysis of the frozen **Dev 448** rows. Every use of
“match”, “mismatch”, or “both wrong” in this report means comparison with the
frozen Dev **weak label only**. It is not a Gold, truth, quality-acceptance, or
production judgement. No Test, Gold, OOD, production input, LLM, model
download, or fitting operation was used.

## Immutable inputs and output

| Input/output | Verified identity |
| --- | --- |
| Accepted Encoder artifact | `b898ac50ac45baf56d094719213c4e3e23de10e2018cf825a69a372e748e8e58` |
| Encoder checkpoint | `e64f71a0b323ac0a7a513b6ae4fddf0e6418b4fdb11f337699e5687da1981cd6` |
| Encoder base snapshot (`pytorch_model.bin`) | `3e04f7477f55dffce2a2fbc4d0ba35068415162a9e92e3d5cc74a49781ba4eb0` |
| Encoder model/revision | `hfl/rbt3` / `0aa0527ff4170f29e1dfd3eb6ef60dc67e1bf75c` |
| Classical run | `49f67b0476c4f439f1d867476d5c850316048fe09e859d652f2cf4225b31c4db` |
| Classical run manifest | `784ed578cfc71b98df021e251380a99fda7724907dd42e1b61e8e5b1bd6c8dd2` |
| Classical model | `fe03d685fcfe3158f0b87a358144e5c991dfe16e1f909276a402a583c7b3c11c` |
| Canonical data package | `cf7a10f25d951d79607cfd80b70751f11415c2772274e275e6ee1b57f32f470b` |
| Reference package | `828944580b96d872241a6619bdb8f60dae2cd7067a0cc6741b418f1e6a7bdc85` |
| Dev weak-label file | `c71f4ae3e3a7ac8fed73d93b635dfd71f8132e25df858f599908ac262e02d37e` |
| Analysis output | `3929bb956260cf1d6ce148e41eec0ca428ae626a12f3234b7ae080398ff5a480` |

The Classical catalog contained four local runs. The analyzer accepted exactly
one: the one above. Selection required a valid run manifest content address,
clean recorded source state, canonical data/reference identities matching the
accepted Encoder provenance, `COMPARABLE_DIAGNOSTIC_RUN_ONLY` status, matching
model-manifest identity, and verified hashes for the local model, thresholds,
and preprocessing contract. Zero or multiple candidates cause a fail-closed
blocker; timestamps and “latest” naming are never inputs to selection.

The ignored, reproducible output directory is:

```text
runs/m2-dev-disagreement/3929bb956260cf1d6ce148e41eec0ca428ae626a12f3234b7ae080398ff5a480/
```

It contains the complete per-sample/per-head ordered probabilities, Reasoning
threshold outcomes, aggregate report, review queue, and content-addressed
manifest. The 4.3 MiB per-sample prediction file is intentionally not tracked.

## Head-level Dev weak-label comparison

“High-confidence disagreement” means the two models disagree and the minimum
of their head-level decision confidences is at least `0.80`. For the multi-label
Reasoning head, confidence is the least confident threshold decision across its
15 tags; the Classical threshold policy and Encoder’s frozen `0.5` threshold
are both preserved in the detailed output.

| Head | Disagreement | Classical-only weak-label matches | Encoder-only weak-label matches | Both mismatch weak label | High-confidence disagreements |
| --- | ---: | ---: | ---: | ---: | ---: |
| target_mode | 257 / 448 (57.4%) | 26 | 184 | 84 | 33 |
| stance | 268 / 448 (59.8%) | 50 | 114 | 180 | 1 |
| emotion_primary | 333 / 448 (74.3%) | 52 | 94 | 256 | 0 |
| emotion_target | 274 / 448 (61.2%) | 63 | 96 | 193 | 0 |
| action_tendency | 240 / 448 (53.6%) | 31 | 148 | 104 | 0 |
| context_dependency | 174 / 448 (38.8%) | 27 | 116 | 81 | 13 |
| reasoning_tags (exact label-set) | 413 / 448 (92.2%) | 37 | 36 | 355 | 0 |

Per-class scalar results and per-tag binary Reasoning counts are retained in
the ignored aggregate JSON. In particular, the high exact-set Reasoning
disagreement must not be read as an Encoder quality verdict: the two systems
use different frozen threshold/fallback policies, and the comparison target is
weak labels rather than independently adjudicated truth.

## Review queue

The queue ranks Dev IDs deterministically using the number of disagreeing
heads, two additional points per high-confidence disagreement, and the number
of heads for which neither model matches the weak label. It is an efficient
future manual-review sampler, not an automatic relabeling or Gold-promotion
mechanism.

Top ten IDs:

1. `P1ad863f59bec8a6d` — score 14
2. `Pce324d0d986cbd86` — score 14
3. `P109ae1d707c8bbda` — score 12
4. `P2b421a5836021265` — score 12
5. `P2e7162c064ed145e` — score 12
6. `P432d9406a5d0e5cc` — score 12
7. `P4ab9c535fff1479f` — score 12
8. `P64e3754ea81bb03a` — score 12
9. `P87f01c1e904eefa4` — score 12
10. `P9a5a98423f1f1347` — score 12

## Interpretation and next gate

The result supplies a Dev-only diagnostic baseline for M2: it exposes where
the two frozen systems differ and where their weak-label outcomes diverge. It
does **not** select a model, justify a threshold change, create Gold, unseal
Test, or authorize production. Before this side branch is integrated, rerun
the same content-addressed analysis against the final M1 integration branch’s
accepted artifact/data identities; a changed identity intentionally produces a
new output address rather than silently replacing this artifact.
