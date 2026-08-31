# M2-S1 runner and independent owner-authorization receipt

Status: `IMPLEMENTED_FAIL_CLOSED_PENDING_FUTURE_OWNER_RECEIPT`

[`encoder_m2_s1.py`](../src/semantic_model/encoder_m2_s1.py) implements only
`M2-S1-FROZEN-SHARED-SEVEN-HEAD-CONTROL`. It cannot select an M2 candidate,
start S2/S3, unfreeze the Encoder, use a new model, or enter a production path.
It reuses M1's immutable Train/Dev loader, `build_input_ids`, class order,
sample-id-by-head weights, metric implementation, and CPU reload smoke test.

## Invocation

```bash
python -m semantic_model.encoder_m2_s1 \
  --config config/semantic-model.yaml \
  --output-dir /absolute/path/new-m2-s1-immutable-run \
  --cache-dir /path/to/existing-fixed-rbt3-cache
```

There is intentionally no arbitrary receipt argument. The command is non-zero
until the one fixed, tracked receipt path and its independent owner-decision
record are valid. A failed receipt or D-026 gate does not import Torch or
Transformers and does not load a model, inspect the model cache, create an
output directory, or invoke fit.

## Receipt boundary

The tracked [receipt schema](../manifests/owner-decisions/m2-s1-owner-authorization-receipt-schema.json),
[receipt](../manifests/owner-decisions/m2-s1-owner-authorization-receipt.json),
[decision-record schema](../manifests/owner-decisions/m2-s1-owner-decision-record-schema.json),
and [decision record](../manifests/owner-decisions/m2-s1-owner-decision-record.json)
are the only eligible files. Both current records remain
`authorization_granted: false`; the decision record allowlist is empty. A
self-hashed JSON at any other path is not an authorization.

A future valid receipt must be content-addressed and bind exactly:

- frozen M2 contract commit `df12078b90f21c5942f838fb2175b636bc20a5db` and
  contract SHA-256 `80e792fb796e66777a0b7607982aa17823cef8890577fdef7b0b7e7e296179f3`;
- the owner-declared unified branch `feat/m2-s1-runner`, its exact current
  commit, matching `origin/feat/m2-s1-runner`, canonical config SHA-256, and
  one unique output directory;
- the sole S1 stage, `hfl/rbt3`, revision
  `0aa0527ff4170f29e1dfd3eb6ef60dc67e1bf75c`, Apache-2.0, and all three
  model/tokenizer file hashes, the complete eight-file accepted M1 snapshot
  identity, and the accepted M1 runtime environment hash;
- `local_files_only=true`, no download, seeds 35/71/107, Train 1,822 and Dev
  448 roles, MPS-first/CPU-fallback, 120 minutes per seed, and 10 GiB total
  new local disk;
- explicit `false` values for Test, Anchor, Gold, OOD, LLM, cloud/external
  API, production, dependency installation, full unfreeze, and S2/S3.

After the receipt passes, D-026 requires exactly one local branch, exactly one
matching remote branch, exactly one primary worktree, a clean worktree, HEAD
equal to the receipt, and upstream `0/0`. Any failure returns
`BLOCKED_PRE_TRAINING_REPOSITORY_NOT_CONSOLIDATED` before canonical audit,
cache, runtime, or output activity.

Only then does the runner execute canonical audit/data-reference binding and
clean source provenance, hash every one of the eight cache files, import and
verify Python/Torch/Transformers/NumPy/tokenizers runtime versions, and allow a
new receipt-authorized output directory. It rejects cache/data/M1-artifact and
other protected output paths or reuse of the receipt's directory. Each seed
records a critical-boundary report; mixed MPS/CPU seeds produce only
device-stratified rejected evidence. Any ContractError, runtime/OOM error, or
wall/disk failure after output creation emits a content-addressed rejected
manifest and cannot aggregate or select a candidate.
