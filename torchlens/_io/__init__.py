"""Portable I/O primitives for TorchLens model logs (declared re-export FACADE).

The ``torchlens._io`` package implements TorchLens' portable save/load path:
it scrubs a ``Trace`` into metadata plus tensor blobs, writes directory
bundles backed by ``safetensors``, and rehydrates those bundles into eager or
lazy model logs. Portable bundles are for archival and analysis, not replay:
``validate_forward_pass()`` is unsupported after ``torchlens.load()``,
expert ``lazy=True, materialize_nested=False`` loads must call
``torchlens.io.materialize_nested()`` before re-save, and lazy refs open,
verify, and close blob files per materialization instead of sharing handles.

Split per the persistence four-way ruling (architecture memo 3.2 / s9 item
3): the IO error vocabulary lives in :mod:`torchlens._io.format_errors`
(logical L0), the format CONTRACT and its record-adjacent helpers in
:mod:`torchlens._io.format_contract` (L1, beside ``data_classes/``), and this
``__init__`` is the declared re-export facade -- every historical spelling
resolves here unchanged.
"""

from __future__ import annotations

from .format_contract import (
    MIN_TLSPEC_VERSION,
    MIN_TORCHLENS_VERSION_TEXT,
    TLSPEC_VERSION,
    BlobRef,
    FieldPolicy,
    JaxPayloadLoadHint,
    PayloadLoadHints,
    above_ceiling_error,
    below_floor_error,
    coerce_container_typed_state,
    default_fill_state,
    raise_if_manifest_below_floor,
    raise_if_saved_by_older_major,
    read_tlspec_version,
)
from .format_errors import (
    ArtifactError,
    ArtifactRuntimeIncompatibleError,
    ArtifactTooNewError,
    ArtifactTooOldError,
    PreReleaseArtifactError,
    UnknownPersistedFieldError,
    refuse_bundle_exists,
    refuse_missing_blobs,
)
from .prerelease import validate_prerelease_state

__tl_layer__ = "FACADE"

#: Session-time once-per-process warning latch; tests/conftest.py rebinds it
#: on THIS module, so it must stay defined here (never in the split modules).
_LEGACY_THREAD_WARNING_EMITTED: dict[str, bool] = {"flag": False}

__all__ = [
    "refuse_bundle_exists",
    "refuse_missing_blobs",
    "MIN_TLSPEC_VERSION",
    "MIN_TORCHLENS_VERSION_TEXT",
    "TLSPEC_VERSION",
    "ArtifactRuntimeIncompatibleError",
    "ArtifactTooNewError",
    "ArtifactTooOldError",
    "BlobRef",
    "FieldPolicy",
    "JaxPayloadLoadHint",
    "PayloadLoadHints",
    "PreReleaseArtifactError",
    "ArtifactError",
    "UnknownPersistedFieldError",
    "above_ceiling_error",
    "below_floor_error",
    "raise_if_manifest_below_floor",
    "raise_if_saved_by_older_major",
    "coerce_container_typed_state",
    "default_fill_state",
    "read_tlspec_version",
    "validate_prerelease_state",
]
