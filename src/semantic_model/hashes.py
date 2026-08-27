from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Collection, Mapping

from .errors import ContractError


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def content_addressed_id(
    value: Mapping[str, Any], *, omit_keys: Collection[str] = ()
) -> str:
    identity_payload = {key: item for key, item in value.items() if key not in omit_keys}
    return hashlib.sha256(canonical_json_bytes(identity_payload)).hexdigest()


def verify_content_addressed_id(value: Mapping[str, Any], *, id_key: str) -> str:
    observed = value.get(id_key)
    expected = content_addressed_id(value, omit_keys={id_key})
    if observed != expected:
        raise ContractError(
            "CONTENT_ADDRESS_MISMATCH",
            f"{id_key} does not match canonical manifest content",
            observed=observed,
            expected=expected,
        )
    return expected
