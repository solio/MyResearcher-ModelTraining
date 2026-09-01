# M2 quality iteration — S1 control handoff

## S1 technical result

The frozen shared-seven-head control completed for seeds 35, 71, and 107 on
MPS. All seven per-head sample standard deviations passed the `0.05` stability
limit. The only permitted S1 output is `S1_CONTROL_EVIDENCE_ONLY`;
`selected_candidate=false`.

The immutable S1 artifact is
`04a23d76413049e57ff083655f80ad8c3dfc7ed90702a3c9ed66bcfd79f377f6` at
`/Users/mac/Documents/trae_projects/MyResearcher/model-artifacts/m2-s1-first-three-seed-20260831`.
It is read-only and is the matching-seed comparator for S2.

| Head | Dev Macro-F1 mean | Sample standard deviation |
| --- | ---: | ---: |
| `target_mode` | 0.328733 | 0.008003 |
| `stance` | 0.374859 | 0.007566 |
| `emotion_primary` | 0.198078 | 0.035309 |
| `emotion_target` | 0.286061 | 0.023519 |
| `action_tendency` | 0.139849 | 0.009758 |
| `context_dependency` | 0.326668 | 0.004465 |
| `reasoning_tags` | 0.153885 | 0.003345 |

S1 is a frozen control, not a model-selection result. S2 may compare only the
same seven heads, the same Train/Dev roles, the same three seeds, and the same
fixed RBT3 cache. Test, Anchor, Gold, OOD, reference predictions, LLM, cloud,
and production paths remain outside this evidence.

## S2 review and S3 diagnostic closeout

The S2 partial-last-block run was independently reviewed as
`ACCEPT_EVIDENCE_BUT_DO_NOT_PROMOTE`. Its seven-head aggregate improved over
S1, but `emotion_primary:CALM` and `reasoning_tags:NO_REASON_GIVEN` exceeded
the frozen critical-label regression tolerance, so those two heads were the
only predeclared S3 triggers. The S2 artifact remains immutable and is not
rewritten.

S3 then ran the frozen single-task diagnostic for exactly those two heads.
Each head used seeds 35, 71, and 107; the Encoder stayed fully frozen and only
the selected head was trainable. All six runs used MPS and each checkpoint
passed the offline CPU reload/inference smoke. S3 has no seven-head promotion
rule and can never produce a selected candidate.

| Head | Seed | Device | Best epoch | Elapsed seconds | CPU reload |
| --- | ---: | --- | ---: | ---: | --- |
| `emotion_primary` | 35 | MPS | 11 | 81.497 | finite / passed |
| `emotion_primary` | 71 | MPS | 9 | 73.611 | finite / passed |
| `emotion_primary` | 107 | MPS | 11 | 74.947 | finite / passed |
| `reasoning_tags` | 35 | MPS | 12 | 73.056 | finite / passed |
| `reasoning_tags` | 71 | MPS | 12 | 67.779 | finite / passed |
| `reasoning_tags` | 107 | MPS | 12 | 63.160 | finite / passed |

| Head | S3 Dev per-seed Macro-F1 (35 / 71 / 107) | Mean | Std | Worst | S3−S1 per-seed delta |
| --- | --- | ---: | ---: | ---: | --- |
| `emotion_primary` | 0.216395 / 0.201228 / 0.241920 | 0.219848 | 0.020565 | 0.201228 | +0.023340 / +0.035678 / +0.006290 |
| `reasoning_tags` | 0.152158 / 0.152405 / 0.144848 | 0.149804 | 0.004294 | 0.144848 | −0.004690 / −0.002143 / −0.005410 |

For `reasoning_tags`, S3 Micro-F1 was 0.267559 / 0.270627 / 0.291347
(mean 0.276511, std 0.012940, worst 0.267559), and exact-set accuracy was
0.125000 / 0.131696 / 0.138393 (mean 0.131696, std 0.006697, worst 0.125000).

The critical-label report is diagnostic only. `emotion_primary:CALM`
(support 48 per seed) changed by −0.031671 / +0.103448 / +0.010823 and
therefore recovered relative to the S2 failure in two of three seeds, while
`reasoning_tags:NO_REASON_GIVEN` (support 86 per seed) changed by
−0.015605 / 0.000000 / −0.130704 and did not recover in seed 107. S3
stability passed for both triggered heads, but the result is not a promotion
and no further model or stage is started.

The immutable S3 evidence is at
`/Users/mac/Documents/trae_projects/MyResearcher/model-artifacts/m2-s3-frozen-single-task-triggered-heads-20260831`
with content address
`7928614bdda834d0de6e3cc6b8d26bc02a10c821c4564dfa61e0ad419ac8899c`.
