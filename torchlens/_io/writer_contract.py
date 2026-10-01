"""The normative writer contract and its digest (ecosystem MEMO 3.1).

``writer_contract_digest()`` is the NORMATIVE digest over the canonicalized
persistence contract: version constraints, the KEEP/DROP field-policy tables
AND persisted field-NAME sets per record type, nested component spec tables,
manifest/body/codec/component schema IDs, alias rules, and the
writer-identity stamp REQUIREMENTS. Two builds share a digest exactly when
they write the same grammar; two builds whose persisted field sets differ get
different digests even under the SAME ``tlspec_version`` stamp -- the drift
that made released v2.33.0 and v2.34.1 indistinguishable by stamp alone.

Writer identity enters as the set of identity fields a conforming writer
must STAMP (never the concrete version values): baking the release string
into the digest would make every release trivially distinct and destroy the
same-grammar signal the ledger keys on. Concrete writer identity lives in
the compatibility ledger row next to the digest, not inside it.

The contract of record is committed at
``torchlens/schemas/writer_contract_v8.json`` and enforced by
``tests/test_tlspec_envelope_contract.py``: regenerating from the live tree
must reproduce the committed contract byte-for-byte, and a persisted field
name that DISAPPEARS from the live contract must be covered by an alias rule
or the diff test fails (alias-or-fail, MEMO 3.3) -- the per-PR cheap variant
of the forward-load gate (MEMO 3.4/G8).
"""

from __future__ import annotations

import hashlib
import json
from importlib import resources
from typing import Any

from . import MIN_TLSPEC_VERSION, MIN_TORCHLENS_VERSION_TEXT, TLSPEC_VERSION
from ._json import loads_bounded
from .field_registry import registry_snapshot

#: Identity fields a conforming writer stamps on every manifest (the
#: REQUIREMENT is contract; the values are ledger data, deliberately outside
#: the digest -- see module docstring).
WRITER_IDENTITY_FIELDS: tuple[str, ...] = (
    "torchlens_version",
    "python_version",
    "torch_version",
    "platform",
    "created_at",
)

#: Payload codec schema IDs this writer can emit (payload_codec.py).
PAYLOAD_CODEC_IDS: tuple[str, ...] = (
    "torch_safetensors_v1",
    "numpy_safetensors_v1",
)


def _manifest_schema_ids() -> dict[str, str]:
    """Canonical digest per shipped manifest JSON schema.

    The schema files are parsed and re-serialized canonically before hashing
    so formatting-only edits do not read as contract changes.
    """

    ids: dict[str, str] = {}
    schema_dir = resources.files("torchlens") / "schemas"
    for entry in sorted(schema_dir.iterdir(), key=lambda item: item.name):
        if not entry.name.startswith("tlspec_manifest_") or not entry.name.endswith(".json"):
            continue
        parsed = loads_bounded(entry.read_text(encoding="utf-8"))
        ids[entry.name] = hashlib.sha256(_canonical_json(parsed).encode("utf-8")).hexdigest()
    return ids


def _component_schema_versions() -> dict[str, Any]:
    """Versioned component-grammar constants riding inside bundles."""

    from ..runnable import (
        LEGACY_RUNNABLE_TLSPEC_SCHEMA_VERSIONS,
        RUNNABLE_ACTIVATION_PAYLOAD_SCHEMA_VERSION,
        RUNNABLE_CALL_RECIPE_VERSION,
        RUNNABLE_CALLABLE_REF_SCHEMA_VERSION,
        RUNNABLE_INITIALIZER_POLICY_VERSION,
        RUNNABLE_TLSPEC_SCHEMA_VERSION,
        WITNESS_FAMILY_REGISTRY_VERSION,
    )

    return {
        "runnable_descriptor": RUNNABLE_TLSPEC_SCHEMA_VERSION,
        "runnable_call_recipe": RUNNABLE_CALL_RECIPE_VERSION,
        "runnable_callable_ref": RUNNABLE_CALLABLE_REF_SCHEMA_VERSION,
        "runnable_initializer_policy": RUNNABLE_INITIALIZER_POLICY_VERSION,
        "runnable_activation_payload": RUNNABLE_ACTIVATION_PAYLOAD_SCHEMA_VERSION,
        "runnable_legacy_descriptors": sorted(LEGACY_RUNNABLE_TLSPEC_SCHEMA_VERSIONS),
        "witness_family_registry": WITNESS_FAMILY_REGISTRY_VERSION,
        "state_dict_blob_family": "state_dict_v1",
        "nonpersistent_buffer_family": "runnable_nonpersistent_buffer_v1",
    }


def _canonical_json(value: Any) -> str:
    """Serialize ``value`` canonically (sorted keys, tight separators)."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def writer_contract() -> dict[str, Any]:
    """Build the canonicalized writer contract from the live tree."""

    return {
        "contract_schema": "torchlens.writer_contract.v1",
        "tlspec_version": TLSPEC_VERSION,
        "min_tlspec_version": MIN_TLSPEC_VERSION,
        "min_torchlens_version": MIN_TORCHLENS_VERSION_TEXT,
        "writer_identity_fields": list(WRITER_IDENTITY_FIELDS),
        "payload_codec_ids": list(PAYLOAD_CODEC_IDS),
        "manifest_schema_ids": _manifest_schema_ids(),
        "component_schema_versions": _component_schema_versions(),
        **registry_snapshot(),
    }


def writer_contract_digest(contract: dict[str, Any] | None = None) -> str:
    """SHA-256 of the canonicalized writer contract."""

    if contract is None:
        contract = writer_contract()
    return hashlib.sha256(_canonical_json(contract).encode("utf-8")).hexdigest()
