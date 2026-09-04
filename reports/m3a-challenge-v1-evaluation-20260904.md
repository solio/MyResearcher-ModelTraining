# M3-A Challenge V1 — frozen candidate trend diagnostic

The frozen selected candidate
`8f73bc437602ce25d6e8ba0c7fd787676a4b566077b1765f3510739ae85928cf` was
loaded on CPU and run once on all 100 input-only Challenge V1 records.  No
labels were supplied to the input builder.  The seven-head predictions and
per-row tokenizer diagnostics are retained outside Git at
`runs/m3a-challenge-v1-20260904/challenge-predictions.jsonl`; the output
directory content address is
`a36212573f22f1028cd0d074627c3c3eb23bc67f11ac7d0054c7980657a6b44d`.

The fixed 40-row AUC comparison uses only `IN_DOMAIN_CONTROL` (20) versus
`CLEAR_OOD` (20).  `SHIFT_CHALLENGE` (60) is descriptive and excluded from
pass/fail:

| Declared score | ROC-AUC (CLEAR_OOD positive) |
|---|---:|
| `1 - six-head mean max probability` | **1.000000** |
| tokenizer `UNK` ratio | **0.600000** |

At the declared `0.90`/`0.80` confidence checks, `CLEAR_OOD` had an any-head
maximum ≥ 0.90 ratio of `0.05`, a six-head mean ≥ 0.80 ratio of `0.00`, and a
joint ratio of `0.00`.  Because at least one declared score reached the fixed
`0.75` criterion, the deterministic result is
`M3_CHALLENGE_SEPARABLE_TREND`.

This is a synthetic trend diagnostic only.  It does not create an OOD gate,
abstention policy, production approval, or any threshold/model-selection
decision.
