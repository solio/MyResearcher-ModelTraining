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

## Remaining reference-environment handoff

Exact baseline reproduction remains blocked. The data owner/reference-run owner
must provide a content-addressed environment manifest containing at least:

- exact Python implementation/version and platform/CPU architecture;
- exact NumPy, SciPy, scikit-learn, joblib and threadpool/BLAS versions;
- solver convergence warnings and each estimator's `n_iter_`;
- thread/parallelism environment relevant to numerical reproducibility;
- the original model artifact hash or prediction file hashes for Train, Dev,
  Test and Anchor50;
- a declared cross-platform metric tolerance and rationale.

Stable blocker: `BLOCKED_MISSING_REFERENCE_ENVIRONMENT`.

This is not cosmetic. In the pinned local scikit-learn 1.7.2 environment, all
six `saga` scalar heads reach the reference `max_iter=2000` without convergence.
The feature matrix is byte-equivalent at the sparse-value level and Reasoning
Anchor micro-F1 matches exactly, but scalar metrics differ. A controlled
scikit-learn 1.6.1 experiment moved one scalar metric closer without matching
it, so dependency guessing cannot establish provenance.

Until this handoff arrives, diagnostic training/export is allowed, but
`BASELINE_V0_3_5_REPRODUCED`, production approval, Encoder work, and 49,054-row
inference are forbidden.
