# M2 Dev-only Encoder / Classical Disagreement Analysis

## Status and boundary

Current evidence status: M2_DEV_DISAGREEMENT_ANALYSIS_COMPLETED_WEAK_LABEL_DIAGNOSTIC_ONLY.

This is a read-only analysis of frozen Dev 448 rows. Every match, mismatch, and
both-mismatch count is relative to the frozen Dev weak label only. It is not
Gold, truth, model selection, threshold selection, OOD, or production
judgement. No Test, Gold, OOD, production input, LLM, model download, or fit
was used.

| Evidence | Content address | Status |
| --- | --- | --- |
| Historical Side output | 3929bb956260cf1d6ce148e41eec0ca428ae626a12f3234b7ae080398ff5a480 | REJECTED_M2_PROVENANCE_INCOMPLETE |
| Retained P0 provenance-bound output | 7e5ec0fbf4d51abcb0bc14e6ed71fc7ef1e7412a11331649e2d1cac1087e672f | M2_DEV_DISAGREEMENT_ANALYSIS_COMPLETED_WEAK_LABEL_DIAGNOSTIC_ONLY; retained and superseded by P2 tooling cleanup |
| Current P2 cleanup output | df64f91e4d6fc64cc8c9e429fe908040c8a3916ecf7b8463fb0654d74a233756 | M2_DEV_DISAGREEMENT_ANALYSIS_COMPLETED_WEAK_LABEL_DIAGNOSTIC_ONLY |

The historical `3929...` directory was neither deleted nor overwritten. It is
rejected because it did not bind live config/package/canonical/Dev bytes, clean
implementation commit, critical source set, runtime versions, or
existing-output tamper checks.

The `7e5...` P0 directory is also neither deleted nor overwritten. Its
provenance remains valid. It is superseded only because P2 changes the
implementation identity to clean staging directories and strictly verify
`summary.md`; the resulting address must therefore differ. Its complete
per-sample prediction payload and aggregate report remain byte-for-byte equal
to the current P2 result.

New ignored output directory:

    runs/m2-dev-disagreement/df64f91e4d6fc64cc8c9e429fe908040c8a3916ecf7b8463fb0654d74a233756/

It contains the complete 448-row per-sample/per-head JSONL, aggregate report,
review queue, summary, and content-addressed manifest. The 4.3 MiB prediction
file remains intentionally untracked.

## Immutable data, implementation, and environment binding

The real analysis ran from clean implementation commit
ccb41fe89e3a267037645fbcdcfb25f811a3b4b9 with source_worktree_clean=true.

| Binding | Verified value |
| --- | --- |
| Accepted Encoder artifact | b898ac50ac45baf56d094719213c4e3e23de10e2018cf825a69a372e748e8e58 |
| Encoder checkpoint | e64f71a0b323ac0a7a513b6ae4fddf0e6418b4fdb11f337699e5687da1981cd6 |
| Encoder base snapshot (pytorch_model.bin) | 3e04f7477f55dffce2a2fbc4d0ba35068415162a9e92e3d5cc74a49781ba4eb0 |
| Encoder model / revision | hfl/rbt3 / 0aa0527ff4170f29e1dfd3eb6ef60dc67e1bf75c |
| Config SHA-256 | 92436eff13c4c67a6dec8a3d645c4e40a4ce8c927d6cb522720709d980cff617 |
| Package CONTENT_MANIFEST.json SHA-256 / package ID | cf7a10f25d951d79607cfd80b70751f11415c2772274e275e6ee1b57f32f470b |
| Canonical inputs SHA-256 | 70601459084e6b49b7bab47f42d3533f53918f7618f166b3949e9b109b05f76b |
| Dev weak-label SHA-256 | c71f4ae3e3a7ac8fed73d93b635dfd71f8132e25df858f599908ac262e02d37e |
| Reference package | 828944580b96d872241a6619bdb8f60dae2cd7067a0cc6741b418f1e6a7bdc85 |
| Classical run / manifest / model | 49f67b0476c4f439f1d867476d5c850316048fe09e859d652f2cf4225b31c4db / 784ed578cfc71b98df021e251380a99fda7724907dd42e1b61e8e5b1bd6c8dd2 / fe03d685fcfe3158f0b87a358144e5c991dfe16e1f909276a402a583c7b3c11c |
| Per-sample / aggregate SHA-256 | 428639bcd338ba64ef5af591b2e418bccddeb18767c242990b61e2f25c7e5325 / e287927acd847cd3a94af1d563e605328248b09262e77bace01c1b5bb8148559 |
| Current summary.md SHA-256 | dd48b59231c8338c46795dc5d520f3025cb7a8797dbde352b65788b56fafac82 |

Before either model is loaded, the analyzer requires current config SHA-256 to
equal Encoder provenance; current package-manifest bytes to equal the accepted
package ID; CONTENT_MANIFEST.sha256 to bind that exact manifest path/hash; and
safe-relative canonical_inputs and split_labels_dev paths, sizes, and SHA-256
values to equal the package manifest. No non-Dev label payload is parsed.

Critical source SHA-256 values recorded in the new identity:

    src/semantic_model/dev_disagreement.py  132ed31e748fb4842298d349c59e8df337fa0a32ca787ee90859b52ae44a9041
    src/semantic_model/encoder_m1.py        f04101de746145c813586960228f661341363630e6e7441cd464d5d995e4a19d
    src/semantic_model/preprocessing.py     4248da49bc5c257245ef7706e143107827dbafed1afa6ee76e7441ce6542fb03
    src/semantic_model/models/classical.py  184c24799e2491de3ce645853e80254086c3e238cea2c779b9d629d15c6e9728
    src/semantic_model/config.py            5fbfcd3d33a2ca038875e2aede397d6a5e02920db589b7b2127bac077f891c7b
    src/semantic_model/hashes.py            83932b32bf536bab98d9ce0894ca364288e818c14a42dcd5fde9619a9906d412

Runtime recorded in the same identity: CPython 3.12.13, Torch 2.8.0,
Transformers 4.57.6, NumPy 2.5.2, scikit-learn 1.7.2, and joblib 1.5.2.

Encoder input IDs are now exclusively made by
semantic_model.encoder_m1.build_input_ids via an explicit Dev-record to
M1Record adapter. The analysis module has no alternate special-token,
segment-cap, or HEAD_TAIL implementation.

## Actual Dev weak-label statistics

High-confidence disagreement means disagreement with minimum decision
confidence at least 0.80. Reasoning confidence is the least-confident threshold
decision across 15 tags; detailed output preserves frozen threshold/fallback.

| Head | Disagreement | Classical-only match | Encoder-only match | Both mismatch | High-confidence |
| --- | ---: | ---: | ---: | ---: | ---: |
| target_mode | 257 / 448 (57.4%) | 26 | 184 | 84 | 33 |
| stance | 268 / 448 (59.8%) | 50 | 114 | 180 | 1 |
| emotion_primary | 333 / 448 (74.3%) | 52 | 94 | 256 | 0 |
| emotion_target | 274 / 448 (61.2%) | 63 | 96 | 193 | 0 |
| action_tendency | 240 / 448 (53.6%) | 31 | 148 | 104 | 0 |
| context_dependency | 174 / 448 (38.8%) | 27 | 116 | 81 | 13 |
| reasoning_tags (exact set) | 413 / 448 (92.2%) | 37 | 36 | 355 | 0 |

All seven head-level counts exactly match the old aggregate. This is expected:
the repair changes provenance/tamper gates and uses the shared M1 builder; all
448 current inputs had zero input-ID differences from the previous equivalent
builder. Per-class scalar and per-tag Reasoning counts are in aggregate JSON.

Top ten deterministic future-review IDs are unchanged:

    P1ad863f59bec8a6d 14   Pce324d0d986cbd86 14
    P109ae1d707c8bbda 12   P2b421a5836021265 12
    P2e7162c064ed145e 12   P432d9406a5d0e5cc 12
    P4ab9c535fff1479f 12   P64e3754ea81bb03a 12
    P87f01c1e904eefa4 12   P9a5a98423f1f1347 12

## Reuse and tamper gate

The exact real analysis command ran twice from the clean implementation commit.
The second run recomputed the same address and verified manifest schema/address,
per-sample and aggregate existence and SHA-256, aggregate schema, and 448
unique sample IDs before reuse. It also regenerates `summary.md` deterministically
from the verified aggregate, identity, and analysis address, then requires an
exact byte-for-byte match. This makes missing, replaced, or truncated summaries
fail closed without adding a summary-derived value to the content-address
identity.

Missing, replaced, or truncated payloads, or a changed manifest identity/address,
fail closed. The analyzer never overwrites a corrupted target or returns it as
reusable. A `finally` cleanup removes only its own
`runs/m2-dev-disagreement/.tmp-m2-dev-*` staging directory on success, reuse,
ordinary failure, and unexpected exceptions. Both actual runs ended with zero
such directories; no content-address directory was removed or overwritten.

## Validation record

- Focused provenance/tamper/input-builder suite: 26 passed, including the
  actual 448-record shared-builder parity check, zero/multiple Classical
  candidates, all payload and manifest tamper cases, all three `summary.md`
  tamper forms (replacement, truncation, deletion), repeat-write cleanup, and
  exception-path cleanup.
- The actual-448 test is marked `real_data` and discovers only an explicitly
  configured permitted package or a locally present canonical/Dev pair; it
  safely skips when neither exists. Tests no longer contain a `/Users/mac/...`
  project-path assumption.
- Full scope-safe suite: 163 passed via `pytest -q -m 'not real_data'`.
  Collection contains 167 tests; the four deselected `real_data` tests are
  deliberately excluded because the complete canonical/reference audits would
  open prohibited Test/reference prediction content in this Dev-only task.
- compileall over src and tests passed; pip check passed in both the Classical
  project runtime and isolated Encoder runtime; git diff --check passed.

This evidence does not select a candidate, create Gold, unseal Test, evaluate
OOD, or authorize production.
