"""Portable-I/O error vocabulary (logical L0; physically homed with _io).

The persistence four-way ruling (architecture memo 3.2) sends IO error
CLASSES to L0: they are frozen vocabulary consumed by every stratum that
touches an artifact. They stay physically inside ``torchlens._io`` under
Rule V's single-lower-consumer clause (declared logical layer, printed in
the vocabulary census); the ``torchlens._io`` facade re-exports every
historical spelling unchanged.
"""

from __future__ import annotations

from ..errors._base import CompatibilityError, TorchLensWarning

__tl_layer__ = "L0"
__tl_vocabulary__ = True


class TorchLensIOError(CompatibilityError, RuntimeError):
    """Raised when TorchLens portable bundle state is invalid or unsupported."""


class ArtifactVersionBelowFloorError(TorchLensIOError):
    """Raised when an artifact predates the supported rehydration floor.

    TorchLens loads artifacts written by torchlens ``2.33`` or newer
    (``tlspec_version >= 6``). Older artifacts refuse with this error rather
    than being partially reconstructed; re-save them with a torchlens release
    in the ``2.33``-to-``2.34`` range that can still read them.
    """


class ArtifactVersionAboveRuntimeError(TorchLensIOError):
    """Raised when an artifact declares a schema newer than this runtime's ceiling.

    The artifact is valid; the READER is too old. Distinct from
    :class:`ArtifactVersionBelowFloorError` (artifact too old) and from
    integrity refusals: the remedy is upgrading torchlens to the release that
    wrote the artifact (or newer), never editing the artifact. Every
    above-ceiling site routes through ``above_ceiling_error`` so the
    refusal carries the same stable ``fields["code"]`` everywhere (G3
    consolidation; the bare ``ValueError`` at the validation entry used to
    defeat the load path's typed-refusal pass-through).
    """


class ArtifactRuntimeIncompatibleError(TorchLensIOError):
    """Raised when the RUNTIME cannot serve an otherwise-valid artifact.

    A runtime-axis refusal (torch major-version mismatch, unparseable torch
    version), never a schema-calendar one: the artifact stays valid and loads
    under a runtime with the recorded major version. Distinct class so
    callers and the compatibility ledger can tell runtime policy apart from
    the version window (ecosystem MEMO 3.1/3.2).
    """


class UnknownPersistedFieldError(TorchLensIOError):
    """Raised when incoming persisted state carries fields this reader does not know.

    The persisted-state contract (ecosystem MEMO 3.3): unknown fields refuse
    TYPED, in BOTH writer directions, by default -- a newer writer's additive
    field is captured truth this reader would silently drop (the
    ``dropped_edge_tensor_args`` lesson: the one real unknown field in
    torchlens history was evidence whose loss also disabled a metadata
    invariant), and an older writer's field this reader deleted without an
    alias is a reader regression. ``fields`` carries ``code``
    (``unknown_persisted_field``), ``record_type``, ``unknown_fields``,
    ``declared_tlspec_version``, ``runtime_tlspec_version``, and ``remedy``.
    Inert inventory (see everything without loading anything):
    ``torchlens.io.inspect_state_contract``.
    """


class PreReleaseArtifactError(TorchLensIOError):
    """Raised when a pre-release-marked artifact loads without the switch.

    Sprint-gated fields persist only under the test-only activation switch
    (:mod:`torchlens._io.prerelease`), and every state written under the
    switch carries the pre-release marker. Loading such an artifact as a real
    current-version artifact refuses with this error so switched and real
    writes are never indistinguishable.
    """


class ArtifactSchemaAgeWarning(TorchLensWarning):
    """Warning emitted when a loaded artifact predates the runtime schema.

    The artifact is between the rehydration floor and the current
    ``tlspec_version``: it loads at its own recorded schema, and fields
    introduced by later schema versions are absent rather than default-filled.

    This is deliberately a ``UserWarning`` subclass, not a
    ``DeprecationWarning``. It deprecates no API -- it reports the AGE of one
    artifact -- and ``DeprecationWarning`` is hidden from end users by default,
    which made the advisory effectively invisible at the one moment it matters.
    """


# Pickle-visible identity stays the historical facade path: persisted bundle
# metadata references these classes by (module, qualname), and the guarded
# unpickler allowlists exactly those paths -- the physical split must not
# change a single persisted byte (move-compat law, memo org rule 9).
for _cls in (
    TorchLensIOError,
    ArtifactVersionBelowFloorError,
    ArtifactVersionAboveRuntimeError,
    ArtifactRuntimeIncompatibleError,
    UnknownPersistedFieldError,
    PreReleaseArtifactError,
    ArtifactSchemaAgeWarning,
):
    _cls.__module__ = "torchlens._io"
del _cls
