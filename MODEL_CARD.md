# Model Card — semantic-student-v0.1.0 diagnostic baseline

## Status

`DATA_PACKAGE_VALIDATED`; exact v0.3.5 reproduction is
`BLOCKED_MISSING_REFERENCE_ENVIRONMENT`.

The canonical weak-label package, quarantine, repaired/trainable views, frozen
split, seven-head weights, preprocessing, Schema, and Anchor50 have passed the
read-only data audit. A real CPU diagnostic model has trained, evaluated,
exported, and completed a two-row inference smoke test. It is not approved as
the historical v0.3.5 reproduction because the reference environment is absent
and the six reference `saga` scalar heads do not converge at `max_iter=2000` in
the pinned local environment.

## Intended use

Diagnostic learning of post-level observable target, stance, explicit emotion,
author action tendency, reasoning tags, and context dependency. The model is
useful for testing data learnability and surfacing label drift. It is not a
price model, investment recommendation, production service, or group-state
model.

## Data

- 3,000 frozen weak teacher labels, not Gold;
- 21 quarantined rows with forbidden Evidence protocol violations;
- 2,979 trainable weak-label rows;
- chronological split with two one-day embargoes: 1,822 / 131 / 448 / 111 /
  467;
- explicit 2,979×7 head-specific weight matrix;
- Anchor50: 11 human-confirmed + 39 Expert-preadjudicated weak-Gold rows,
  excluded from Train/Dev/Test and not an unbiased production test set.

The broader cleaned snapshot contains 49,054 posts from one source
(`eastmoney_guba`) and 16 stocks. No cross-platform generalization is supported.

## Reproduction limitation

The package embeds reference code and metrics but not the environment that
produced them. Local CPython 3.12.13 / NumPy 2.3.3 / SciPy 1.16.2 /
scikit-learn 1.7.2 reproduces the exact 12,258-feature matrix and the Anchor
Reasoning micro-F1, but not all six scalar Anchor Macro-F1 values. Every real
run records the observed environment, comparison deltas, warnings, and per-head
convergence state.

## Release gate

A model may be called `BASELINE_V0_3_5_REPRODUCED` only when the reference
environment is content-addressed and the declared metric tolerance passes.
Neither a synthetic test nor the current diagnostic run satisfies that gate.
No model file is committed to Git, and 49,054-row inference remains forbidden.
