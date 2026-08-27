# Model Card — semantic-student-v0.1.0 diagnostic baseline

## Status

`BASELINE_HARNESS_PROPOSED`; `BASELINE_V0_3_5_REPRODUCTION_BLOCKED`.

No real model artifact is committed or approved. The visible canonical package
is incomplete, so Milestone 1 must not train on reconstructed labels/splits or
run production inference.

## Intended use

Diagnostic learning of post-level observable target, stance, explicit emotion,
author action tendency, reasoning tags, and context dependency. It is not a
price model, investment recommendation, production service, or group-state
model.

## Data and limitations

The observed upstream cleaned snapshot contains 49,054 posts from one source
(`eastmoney_guba`) and 16 stocks. No cross-platform generalization claim is
supported. The required 3,000 frozen teacher labels, quarantine, split,
field-weight map, Anchor50, and original v0.3.5 baseline contract/report are
missing; see `docs/data-inventory.md`.

## Release gate

A real model card is generated with each immutable run and must record hashes,
config, code/environment, metrics, calibration, abstention, limitations,
status, and blocker codes. A synthetic run is never a production release.

