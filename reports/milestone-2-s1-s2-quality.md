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
