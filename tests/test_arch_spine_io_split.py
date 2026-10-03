"""Move-compat gates for the _io split (C01 item 3) + disposition labels (item 4).

The split sends the IO error vocabulary to ``format_errors`` (logical L0)
and the format contract to ``format_contract`` (L1) with ``_io/__init__``
a declared re-export facade. Two invariants gate it:

- ZERO SPELLING CHANGES: every historical ``torchlens._io`` name resolves
  unchanged (the facade contract).
- ZERO PERSISTED-BYTE CHANGES: pickle-visible identity of every persisted
  type stays ``torchlens._io`` (bundle metadata references types by
  (module, qualname) and the guarded unpickler allowlists exactly those
  paths). The committed genuine-release artifacts in
  ``tests/release_goldens`` rehydrating green is the real pre-move gate;
  this suite pins the mechanism that keeps them green.
"""

from __future__ import annotations

import pickle

import pytest

import torchlens._io as tl_io
from torchlens._io import format_contract, format_errors

_FACADE_SPELLINGS = (
    "TLSPEC_VERSION",
    "MIN_TLSPEC_VERSION",
    "MIN_TORCHLENS_VERSION_TEXT",
    "TorchLensIOError",
    "ArtifactVersionBelowFloorError",
    "ArtifactVersionAboveRuntimeError",
    "ArtifactRuntimeIncompatibleError",
    "UnknownPersistedFieldError",
    "PreReleaseArtifactError",
    "ArtifactSchemaAgeWarning",
    "JaxPayloadLoadHint",
    "PayloadLoadHints",
    "BlobRef",
    "FieldPolicy",
    "below_floor_error",
    "above_ceiling_error",
    "read_tlspec_version",
    "default_fill_state",
    "coerce_container_typed_state",
    "rehydrate_nested",
    "validate_prerelease_state",
    "_LEGACY_THREAD_WARNING_EMITTED",
)

# The load-hint classes (JaxPayloadLoadHint, PayloadLoadHints) are runtime
# options, never persisted, so they keep their real defining module.
_PICKLE_VISIBLE_TYPES = (
    "TorchLensIOError",
    "ArtifactVersionBelowFloorError",
    "ArtifactVersionAboveRuntimeError",
    "ArtifactRuntimeIncompatibleError",
    "UnknownPersistedFieldError",
    "PreReleaseArtifactError",
    "ArtifactSchemaAgeWarning",
    "BlobRef",
    "FieldPolicy",
)


def test_every_historical_spelling_resolves_on_the_facade() -> None:
    for name in _FACADE_SPELLINGS:
        assert hasattr(tl_io, name), f"torchlens._io.{name} vanished in the split"


def test_facade_and_split_modules_serve_the_same_objects() -> None:
    assert tl_io.FieldPolicy is format_contract.FieldPolicy
    assert tl_io.BlobRef is format_contract.BlobRef
    assert tl_io.TLSPEC_VERSION == format_contract.TLSPEC_VERSION
    assert tl_io.TorchLensIOError is format_errors.TorchLensIOError


def test_pickle_visible_identity_stays_the_facade_path() -> None:
    """Persisted-byte stability: __module__ of every persisted type is unchanged."""

    for name in _PICKLE_VISIBLE_TYPES:
        cls = getattr(tl_io, name)
        assert cls.__module__ == "torchlens._io", (
            f"{name}.__module__ is {cls.__module__!r}: the split changed the "
            "pickle-visible identity; bundle metadata and the guarded "
            "unpickler allowlist reference 'torchlens._io' exactly"
        )


def test_field_policy_round_trips_by_the_historical_path() -> None:
    payload = pickle.dumps(tl_io.FieldPolicy.KEEP)
    assert b"torchlens._io" in payload
    assert b"format_contract" not in payload
    assert pickle.loads(payload) is tl_io.FieldPolicy.KEEP


def test_split_modules_declare_their_ruled_layers() -> None:
    assert format_errors.__tl_layer__ == "L0"
    assert format_errors.__tl_vocabulary__ is True
    assert format_contract.__tl_layer__ == "L1"
    assert tl_io.__tl_layer__ == "FACADE"


@pytest.mark.smoke
def test_fastlog_dispositions_are_declared() -> None:
    """Item 4 label rows: storage_disk L2, recover/cleanup L3, types vocabulary."""

    import importlib

    storage_disk = importlib.import_module("torchlens.fastlog.storage_disk")
    recover = importlib.import_module("torchlens.fastlog.recover")
    cleanup = importlib.import_module("torchlens.fastlog.cleanup")
    types = importlib.import_module("torchlens.fastlog.types")
    assert storage_disk.__tl_layer__ == "L2"
    assert recover.__tl_layer__ == "L3"
    assert cleanup.__tl_layer__ == "L3"
    assert types.__tl_layer__ == "L1"
    assert types.__tl_vocabulary__ is True
