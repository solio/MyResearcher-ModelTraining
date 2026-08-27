# Canonical Handoff Status and Remaining Requirement

## Accepted immutable data package

The v0.3.5 semantic data handoff received on 2026-08-27 is accepted for
diagnostic training. The archive is local/ignored and is addressed by:

- ZIP SHA-256:
  `c5ff639954fe71d8bc780175584406c6f5c84998c39d0040fdae830134a95378`;
- `CONTENT_MANIFEST.json` SHA-256:
  `cf7a10f25d951d79607cfd80b70751f11415c2772274e275e6ee1b57f32f470b`;
- 28 payload files / 14,992,255 bytes, all hashes and sizes verified;
- no absolute/traversal/backslash path, duplicate ZIP entry, or symlink.

The accepted package supplies, without reconstruction:

1. 3,000 byte-frozen teacher labels and their 3,000 canonical inputs;
2. a mechanical 3,000-row repair that removes exactly 21 forbidden Evidence
   objects and changes zero semantic labels;
3. the 21-row quarantine subset and 2,979-row repaired trainable view;
4. five disjoint chronological partitions whose union is exactly the 2,979
   trainable IDs;
5. one explicit seven-head field-weight vector per trainable row;
6. Anchor50 labels, decision manifest, and provenance proving 11
   human-confirmed + 39 Expert-preadjudicated rows and zero Teacher3000 overlap;
7. compact frozen Schema, preprocessing contract, weighting contract, audit
   expectations, training-gate provenance, reference code, and baseline metrics.

Package bytes are never normalized or rewritten. The repository accepts both
the package's native Evidence-object array and the historical synthetic-fixture
mapping through separate strict validators. Anchor-only compatibility for three
old `NO_REASON_GIVEN` combinations does not weaken Teacher3000 rules.

## Reference-environment handoff — resolved

The immutable reference handoff received on 2026-08-27 resolves the missing
artifact request. Its ZIP SHA-256 is
`78064a4fe739920491d70ff1888d9233b02b6ac3ac38db8e82080e3549857410`;
its content address is
`828944580b96d872241a6619bdb8f60dae2cd7067a0cc6741b418f1e6a7bdc85`.
All 17 payloads / 11,439,730 bytes validate and bind to the accepted data
package content address `cf7a10f…f470b`.

The handoff supplies:

- the original model, SHA-256
  `4e1dbe0fe1d4d37be728cebe849630ffd75a1fb6d66988bd15112375e6476b5a`;
- Python 3.12.13 / Linux x86_64 / Ubuntu 24.04.3 / AMD EPYC provenance;
- NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0, joblib 1.5.3,
  threadpoolctl 3.6.0, and OpenBLAS 0.3.30/pthreads details;
- full `pip freeze`, NumPy/scikit-learn configuration, and threadpool capture;
- six scalar and 15 Reasoning estimator class orders, `n_iter_`, convergence,
  and coefficient/intercept fingerprints;
- 2,787 per-row Train/Dev/Test/Anchor50 truths, predictions, probabilities,
  normalized-text hashes, and Reasoning thresholds;
- original metrics, metrics recomputed from the frozen model, and an exact
  comparison with maximum absolute difference `0.0`;
- a frozen reproduction policy.

One provenance limitation remains explicit: the environment manifest was
captured retrospectively from the same persistent runtime that holds and loads
the original artifact without warnings. It was not automatically emitted by
the original training command. The frozen model, prediction files, metrics,
and coefficient fingerprints now prevent further dependency guessing.

## Remaining execution requirement

No artifact handoff is missing. Exact reproduction now requires an actual run
in the accepted reference environment. That run must satisfy all of:

1. identical data and reference package content addresses;
2. exact prediction labels on every Train/Dev/Test/Anchor50 row;
3. metrics absolute difference at most `1e-12`;
4. probability absolute difference at most `1e-10`.

Cross-platform or different scikit-learn versions have no authorized exactness
tolerance. The current macOS arm64 / scikit-learn 1.7.2 environment is therefore
`COMPARABLE_DIAGNOSTIC_RUN_ONLY`, with
`BLOCKED_REFERENCE_ENVIRONMENT_MISMATCH`. A passing same-environment run may
only become `BASELINE_V0_3_5_REPRODUCED_DIAGNOSTIC_ONLY`; production approval,
Encoder work within Milestone 1, and 49,054-row production inference remain
forbidden.
