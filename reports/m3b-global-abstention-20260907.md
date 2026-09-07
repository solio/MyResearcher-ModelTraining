# M3-B input-level abstention diagnostic

The frozen candidate
`8f73bc437602ce25d6e8ba0c7fd787676a4b566077b1765f3510739ae85928cf` was
run on all 448 Dev records on CPU.  The input-level score is the mean of the
six scalar-head maximum probabilities.  Its threshold was selected once from
Dev with the fifth percentile and `method=lower`; abstention is strictly
`score < threshold`.

- Frozen threshold: **0.5243767648935318**
- Dev false-rejection: **22/448 = 0.049107142857142856**
- Dev coverage: **426/448 = 0.9508928571428571**
- Challenge output: `runs/m3b-global-abstention-20260907/`
- Aggregate content address: `4957a97691277e5846b34ca64bd1b81753dc58695f975baa036272cd411607b1`

## Dev head metrics

Metrics after abstention are computed only on the 426 released rows; no labels
were used to choose the threshold.

| Head | Accuracy before → after | Macro-F1 before → after |
|---|---:|---:|
| `target_mode` | 0.8392857 → 0.8474178 | 0.4661814 → 0.4741519 |
| `stance` | 0.5267857 → 0.5375587 | 0.4040968 → 0.4104384 |
| `emotion_primary` | 0.4107143 → 0.4272300 | 0.2884802 → 0.3028071 |
| `emotion_target` | 0.4888393 → 0.5000000 | 0.3161614 → 0.3225199 |
| `action_tendency` | 0.7165179 → 0.7276995 | 0.3295347 → 0.3300710 |
| `context_dependency` | 0.5915179 → 0.5985915 | 0.3529323 → 0.3595203 |

## Challenge application

- `CLEAR_OOD` recall: **13/20 = 0.65**
- `IN_DOMAIN_CONTROL` false-rejection rate: **0/20 = 0.00**
- Highest-score released `CLEAR_OOD`: `M3SYN-OOD-003`, score
  `0.5904351671536764` (above the frozen threshold)
- `SHIFT_CHALLENGE` abstention rate: `NEW_ENTITY_INDUSTRY` 0/10,
  `NEW_TIME_EVENT` 0/10, `PLATFORM_SLANG` 0/10,
  `TRADITIONAL_CHINESE` 0/10, `TYPO_HOMOPHONE` 0/10,
  `MISSING_CONTEXT_LINK_EMOJI_OVERLOAD` 0/10.  These are descriptive only.

Because Dev false rejection is within 5% but `CLEAR_OOD` recall is below 80%,
the fixed result is **`M3_GLOBAL_ABSTENTION_INSUFFICIENT_DIAGNOSTIC`**.
No threshold was moved after seeing Challenge results, and this synthetic
diagnostic does not establish a production OOD or abstention capability.
