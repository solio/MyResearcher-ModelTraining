# Model Card — semantic-student-v0.1.0 diagnostic baseline

## Status

`DATA_AND_REFERENCE_VALIDATED_COMPARABLE_ONLY`; exact v0.3.5 reproduction on
the current machine is blocked by `BLOCKED_REFERENCE_ENVIRONMENT_MISMATCH`.

The canonical weak-label package, quarantine, repaired/trainable views, frozen
split, seven-head weights, preprocessing, Schema, and Anchor50 have passed the
read-only data audit. A real CPU diagnostic model has trained, evaluated,
exported, and completed a two-row inference smoke test. The immutable reference
package now supplies the original model, environment, 21 estimator diagnostics,
2,787 per-row predictions, probabilities, thresholds, and metric oracle. The
current macOS arm64 / scikit-learn 1.7.2 run is a comparable diagnostic run,
not an exact Linux x86_64 / scikit-learn 1.8.0 reproduction.

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

## Reference finding and reproduction limitation

The original artifact was loaded without warnings in its retrospectively
captured persistent runtime: CPython 3.12.13 / Linux x86_64 / NumPy 2.3.5 /
SciPy 1.17.0 / scikit-learn 1.8.0 / joblib 1.5.3 / OpenBLAS 0.3.30. The
original six scalar SAGA estimators themselves have `n_iter_=max_iter=2000`
and `converged=false`; all 15 Reasoning estimators converged. Recomputed
historical metrics have maximum absolute difference `0.0`.

The environment provenance was captured retrospectively from the same runtime
that holds and loads the original artifact; it was not automatically emitted by
the original fit command. That limitation is preserved. The frozen model,
coefficient/intercept fingerprints, predictions, probabilities, thresholds,
and metrics now provide primary oracles, so dependency versions no longer need
to be guessed.

## Release gate

A model may be called `BASELINE_V0_3_5_REPRODUCED_DIAGNOSTIC_ONLY` only in the
accepted reference environment when every Train/Dev/Test/Anchor50 prediction
label matches, all metrics pass absolute tolerance `1e-12`, and all
probabilities pass absolute tolerance `1e-10`. Cross-platform or different
scikit-learn versions have no authorized exactness tolerance and may only be
`COMPARABLE_DIAGNOSTIC_RUN_ONLY`. Reproduction never authorizes production.
No model file is committed to Git, and 49,054-row inference remains forbidden.
