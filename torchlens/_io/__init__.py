"""Portable I/O primitives for TorchLens model logs (declared re-export FACADE).

The ``torchlens._io`` package implements TorchLens' portable save/load path:
it scrubs a ``Trace`` into metadata plus tensor blobs, writes directory
bundles backed by ``safetensors``, and rehydrates those bundles into eager or
lazy model logs. Portable bundles are for archival and analysis, not replay:
``validate_forward_pass()`` is unsupported after ``torchlens.load()``,
expert ``lazy=True, materialize_nested=False`` loads must call
``torchlens.io.rehydrate_nested()`` before re-save, and lazy refs open,
verify, and close blob files per materialization instead of sharing handles.

Split per the persistence four-way ruling (architecture memo 3.2 / s9 item
3): the IO error vocabulary lives in :mod:`torchlens._io.format_errors`
(logical L0), the format CONTRACT and its record-adjacent helpers in
:mod:`torchlens._io.format_contract` (L1, beside ``data_classes/``), and this
``__init__`` is the declared re-export facade -- every historical spelling
resolves here unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

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
    read_tlspec_version,
)
from .format_errors import (
    ArtifactRuntimeIncompatibleError,
    ArtifactSchemaAgeWarning,
    ArtifactVersionAboveRuntimeError,
    ArtifactVersionBelowFloorError,
    PreReleaseArtifactError,
    TorchLensIOError,
    UnknownPersistedFieldError,
)
from .prerelease import validate_prerelease_state

__tl_layer__ = "FACADE"

#: Session-time once-per-process warning latch; tests/conftest.py rebinds it
#: on THIS module, so it must stay defined here (never in the split modules).
_LEGACY_THREAD_WARNING_EMITTED: dict[str, bool] = {"flag": False}

__all__ = [
    "MIN_TLSPEC_VERSION",
    "MIN_TORCHLENS_VERSION_TEXT",
    "TLSPEC_VERSION",
    "ArtifactRuntimeIncompatibleError",
    "ArtifactSchemaAgeWarning",
    "ArtifactVersionAboveRuntimeError",
    "ArtifactVersionBelowFloorError",
    "BlobRef",
    "FieldPolicy",
    "JaxPayloadLoadHint",
    "PayloadLoadHints",
    "PreReleaseArtifactError",
    "TorchLensIOError",
    "UnknownPersistedFieldError",
    "above_ceiling_error",
    "below_floor_error",
    "raise_if_manifest_below_floor",
    "coerce_container_typed_state",
    "default_fill_state",
    "read_tlspec_version",
    "rehydrate_nested",
    "validate_prerelease_state",
]


def rehydrate_nested(
    trace: Any,
    *,
    map_location: str | torch.device = "cpu",
    payload_hints: PayloadLoadHints | Mapping[str, Any] | None = None,
) -> None:
    """Replace any remaining nested portable blob refs on a loaded model log.

    This function is a no-op unless the model log was loaded with
    ``lazy=True, materialize_nested=False``. In the default load mode, nested
    tensors are already materialized.

    Parameters
    ----------
    trace:
        Model log loaded from a portable bundle.
    map_location:
        Target device for the materialized nested tensors.
    payload_hints:
        Optional backend payload hints used during materialization.

    Examples
    --------
    >>> import torchlens as tl
    >>> log = tl.load("demo_bundle", lazy=True, materialize_nested=False)
    >>> tl.io.rehydrate_nested(log)
    >>> log.save("demo_bundle_copy")
    """

    from .rehydrate import rehydrate_nested as _rehydrate_nested

    _rehydrate_nested(trace, map_location=map_location, payload_hints=payload_hints)
