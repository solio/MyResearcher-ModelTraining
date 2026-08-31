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
  --output-dir artifacts/new-m2-s1-immutable-run \
  --cache-dir /path/to/existing-fixed-rbt3-cache \
  --owner-authorization-receipt /path/to/separately-issued-receipt.json
```

The command is intentionally non-zero until a valid receipt is supplied. A
failed receipt gate does not import Torch or Transformers and does not load a
model, inspect the model cache, create an output directory, or invoke fit.

## Receipt boundary

The tracked [schema](../manifests/encoder-m2-s1-owner-authorization-receipt-schema.json)
and [template](../manifests/encoder-m2-s1-owner-authorization-receipt-template.json)
are not an authorization. The template has `authorization_granted: false`, no
valid expiry, and no valid content address. It must never be copied into an
artifact as evidence of an owner decision.

A future valid receipt must be content-addressed and bind exactly:

- frozen M2 contract commit `0d2f64cf0ce26953e83b17d043da4441f4930dc0` and
  the current contract SHA-256;
- the sole S1 stage, `hfl/rbt3`, revision
  `0aa0527ff4170f29e1dfd3eb6ef60dc67e1bf75c`, Apache-2.0, and all three
  model/tokenizer file hashes;
- `local_files_only=true`, no download, seeds 35/71/107, Train 1,822 and Dev
  448 roles, MPS-first/CPU-fallback, 120 minutes per seed, and 10 GiB total
  new local disk;
- explicit `false` values for Test, Anchor, Gold, OOD, LLM, cloud/external
  API, production, dependency installation, full unfreeze, and S2/S3.

After that receipt passes, the runner executes canonical audit and the
data/reference binding, clean tracked-source provenance, contract/config
identities, and fixed local-cache hashes before the first dynamic Torch or
Transformers import. Each of the three ordered seeds produces its own
checkpoint, metrics, resource log, CPU reload smoke evidence, and immutable
stage manifest. An incomplete seed set never produces an aggregate. S1 can
emit only `MAY_REQUEST_S2_OWNER_AUTHORIZATION` or rejected/blocked evidence;
it always records `selected_candidate: false`.
