"""Portable I/O primitives for TorchLens model logs.

The ``torchlens._io`` package implements TorchLens' portable save/load path:
it scrubs a ``Trace`` into metadata plus tensor blobs, writes directory
bundles backed by ``safetensors``, and rehydrates those bundles into eager or
lazy model logs. Portable bundles are for archival and analysis, not replay:
``validate_forward_pass()`` is unsupported after ``torchlens.load()``,
expert ``lazy=True, materialize_nested=False`` loads must call
``torchlens.rehydrate_nested()`` before re-save, and lazy refs open,
verify, and close blob files per materialization instead of sharing handles.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, NamedTuple

import torch

from ..errors._base import CompatibilityError, TorchLensWarning
from .prerelease import validate_prerelease_state

# v6 adds persisted ModuleCall forward-pre-hook provenance value objects.
# v7 adds the persisted capture outcome (`_capture_outcome`, string-only payload).
# v8 is the coordinated feature-sprint activation: every S3 pre-release-gated
# family flips to its persisting policy together (Op.site_key, the L6 edge/
# audit families, L1 grouping + grouping_policy, L8 distributed_scope, L9
# grad_fn_timing_provenance + checkpoint_invocation_witness, the L7a
# structure_only marker, the L3 primitive-op profile + kernel telemetry, the
# S6 Bundle member_relations key, and the S7 episode annotations ledger),
# with their load-validation rows live on real artifacts.
TLSPEC_VERSION = 8
_LEGACY_THREAD_WARNING_EMITTED: dict[str, bool] = {"flag": False}

# Rehydration floor: artifacts older than tlspec_version 6 (first shipped in
# torchlens 2.33) refuse to load instead of being resurrected through legacy
# field-alias ladders. ``MIN_TORCHLENS_VERSION_TEXT`` is the release named in
# refusal messages and matched against parsed manifest ``torchlens_version``.
MIN_TLSPEC_VERSION = 6
MIN_TORCHLENS_VERSION_TEXT = "2.33"


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
    above-ceiling site routes through :func:`above_ceiling_error` so the
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


_BELOW_FLOOR_REMEDY = (
    "Load and re-save the artifact with a torchlens release "
    f">= {MIN_TORCHLENS_VERSION_TEXT} that still reads it."
)


def below_floor_error(
    *,
    observed: str,
    subject: str = "Artifact",
    path: str | None = None,
) -> ArtifactVersionBelowFloorError:
    """Build the typed rehydration-floor refusal with structured fields.

    Every below-floor refusal was hand-copied at its raise site with an empty
    ``fields`` payload, and four of the six omitted the artifact path they held
    in scope (R65). This one constructor gives all of them a stable
    ``fields["code"]``, the ``observed`` version, the floor, a ``remedy``, and
    the ``path`` when the caller has one.

    Parameters
    ----------
    observed:
        Rendered source version (``"tlspec_version=N"`` or a description of an
        unversioned state).
    subject:
        Human-readable subject named in the message (e.g. ``"Bundle manifest"``).
    path:
        Artifact path, when the caller has it in scope.

    Returns
    -------
    ArtifactVersionBelowFloorError
        The typed refusal, ready to raise.
    """

    message = (
        f"{subject} has {observed}, below the supported rehydration floor "
        f"tlspec_version={MIN_TLSPEC_VERSION} (torchlens "
        f"{MIN_TORCHLENS_VERSION_TEXT}). {_BELOW_FLOOR_REMEDY}"
    )
    return ArtifactVersionBelowFloorError(
        message,
        code="artifact_version_below_floor",
        observed=observed,
        floor_tlspec_version=MIN_TLSPEC_VERSION,
        floor_torchlens_version=MIN_TORCHLENS_VERSION_TEXT,
        path=path,
        remedy=_BELOW_FLOOR_REMEDY,
    )


_ABOVE_CEILING_REMEDY = "Upgrade torchlens to the release that wrote this artifact (or newer)."


def above_ceiling_error(
    *,
    observed: int,
    subject: str = "Artifact",
    path: str | None = None,
) -> ArtifactVersionAboveRuntimeError:
    """Build the typed above-runtime-ceiling refusal with structured fields.

    The single constructor for every "artifact is newer than this runtime"
    refusal (G3 ceiling consolidation): the manifest policy gate, the pickle
    state gate, and the validation entry all raise through here, so the code,
    ceiling, and remedy are identical no matter which door the artifact came
    in through.

    Parameters
    ----------
    observed:
        The declared ``tlspec_version`` on the incoming artifact or state.
    subject:
        Human-readable subject named in the message (e.g. ``"Bundle"``).
    path:
        Artifact path, when the caller has one.

    Returns
    -------
    ArtifactVersionAboveRuntimeError
        The typed refusal, ready to raise.
    """

    message = (
        f"{subject} uses tlspec_version={observed}, but this runtime only supports "
        f"up to {TLSPEC_VERSION}. {_ABOVE_CEILING_REMEDY}"
    )
    return ArtifactVersionAboveRuntimeError(
        message,
        code="artifact_version_above_runtime",
        observed=observed,
        ceiling_tlspec_version=TLSPEC_VERSION,
        path=path,
        remedy=_ABOVE_CEILING_REMEDY,
    )


def _raise_below_floor(cls_name: str, version_text: str) -> None:
    """Raise the typed rehydration-floor refusal for one object state.

    Parameters
    ----------
    cls_name:
        Human-readable class name used in the error message.
    version_text:
        Rendered source version (``"tlspec_version=N"`` or a description of
        an unversioned state).
    """

    raise below_floor_error(observed=version_text, subject=f"{cls_name} state")


@dataclass(frozen=True)
class JaxPayloadLoadHint:
    """JAX-specific payload materialization hints.

    Parameters
    ----------
    sharding:
        Explicit JAX ``Sharding`` target. When supplied, materialized JAX
        payloads are placed with this sharding and reconstruction metadata is
        not consulted.
    reconstruct_sharding:
        Whether to reconstruct a saved ``NamedSharding`` from portable
        ``codec_metadata``.
    mesh_axis_names:
        Optional expected mesh axis names for metadata reconstruction. When
        provided, the saved contract must match exactly.
    platform:
        Optional JAX device platform to use for metadata reconstruction, for
        example ``"cpu"``.
    devices:
        Optional explicit local devices to use for metadata reconstruction.
    """

    sharding: Any | None = None
    reconstruct_sharding: bool = False
    mesh_axis_names: tuple[str, ...] | None = None
    platform: str | None = None
    devices: Sequence[Any] | None = None


@dataclass(frozen=True)
class PayloadLoadHints:
    """Backend-specific payload materialization hints.

    Parameters
    ----------
    jax:
        Optional JAX payload hint.
    """

    jax: JaxPayloadLoadHint | None = None


class BlobRef(NamedTuple):
    """Reference to a persisted tensor blob in a portable bundle."""

    blob_id: str
    kind: str


class FieldPolicy(str, Enum):
    """Portable scrub policy for one serialized field."""

    KEEP = "keep"
    BLOB = "blob"
    BLOB_RECURSIVE = "blob_recursive"
    DROP = "drop"
    STRINGIFY = "stringify"
    WEAKREF_STRIP = "weakref_strip"


def read_tlspec_version(state: dict[str, Any], *, cls_name: str, cls: type | None = None) -> int:
    """Validate the serialized I/O format version for one object state.

    Parameters
    ----------
    state:
        Serialized state dict for the object being restored.
    cls_name:
        Human-readable class name used in errors.
    cls:
        The portable record class being restored. When provided, the
        persisted-state contract's known/unknown partition runs here --
        BEFORE any object mutation, identically on all three storage paths
        (``__dict__``-backed, hook-restored columnar, slotted) -- and unknown
        fields refuse typed in both writer directions (MEMO 3.3). The
        partition is scoped to the governed-artifact-load window armed by
        the ``.tlspec`` loaders; plain session pickling stays outside the
        contract because live records legitimately round-trip user-set
        extras the save path refuses to persist.

    Returns
    -------
    int
        The decoded version, always ``MIN_TLSPEC_VERSION`` or newer.

    Raises
    ------
    ArtifactVersionBelowFloorError
        If the state predates the ``tlspec_version >= MIN_TLSPEC_VERSION``
        rehydration floor (including unversioned pre-sprint states).
    UnknownPersistedFieldError
        If ``cls`` is provided, a governed artifact load is in progress, and
        the state carries fields outside the class's declared contract
        (fields + defaults + declared aliases).
    TorchLensIOError
        If the serialized version is newer than this runtime understands or
        is not an integer.
    """

    validate_prerelease_state(state, cls_name=cls_name)
    version = state.pop("tlspec_version", None)
    if version is None:
        _raise_below_floor(cls_name, "no tlspec_version (predates portable I/O versioning)")
    if not isinstance(version, int):
        raise TorchLensIOError(f"{cls_name} pickle state has invalid tlspec_version={version!r}.")
    if version > TLSPEC_VERSION:
        raise above_ceiling_error(observed=version, subject=f"{cls_name} pickle state")
    if version < MIN_TLSPEC_VERSION:
        _raise_below_floor(cls_name, f"tlspec_version={version}")
    if cls is not None:
        from .state_contract import enforce_known_state, governed_load_active

        if governed_load_active():
            enforce_known_state(cls, state, declared_version=version)
    return version


def default_fill_state(state: dict[str, Any], *, defaults: dict[str, Any]) -> None:
    """Populate missing state keys with deep-copied default values.

    Parameters
    ----------
    state:
        Mutable serialized state dict being restored.
    defaults:
        Mapping from field name to the default value that should be injected
        when that field is absent.
    """

    for field_name, default_value in defaults.items():
        if field_name not in state:
            state[field_name] = copy.deepcopy(default_value)


_COERCIBLE_CONTAINER_TYPES = (list, dict, tuple, set, frozenset)


def coerce_container_typed_state(
    state: dict[str, Any],
    defaults: dict[str, Any],
    *,
    exclude: frozenset[str] | set[str] = frozenset(),
) -> None:
    """Coerce present-but-wrong-typed container fields to their declared type.

    ``default_fill_state`` only fills keys that are *absent* from ``state``; by
    design it never inspects the type of a key that is already present, so a
    field restored from an older serialization that used a different
    container type for the same name (e.g. a field that used to default to
    ``{}`` and now defaults to ``[]``) survives unchanged with the wrong type.
    That is silent today and crashes or misbehaves later, whenever code calls
    a list-only method (``.append``) on the restored dict, or vice versa.

    This closes that gap for fields whose declared type is unambiguous: a
    field is only touched here if its entry in ``defaults`` is exactly a
    ``list``, ``dict``, ``tuple``, or ``set`` literal. Fields defaulting to
    ``None``, a scalar, or a non-builtin container (e.g. ``OrderedDict``,
    a custom accessor class) are left untouched, since for those fields the
    "declared type" is not a single unambiguous builtin and a naive coercion
    could silently corrupt a legitimately-varying value (see ``exclude`` for
    the one known case of this: a field typed ``List[int] | str``).

    ``None`` is treated as the "no data was ever recorded" case (e.g. an old
    ``Optional``-typed field that is now a required container): converting it
    to an empty typed container is lossless, since there was never any
    element data to preserve, and mirrors what an absent key would have
    received from ``default_fill_state``.

    A present value that is genuinely incompatible with the declared type
    (``declared_type(current_value)`` raises) is a DIFFERENT case: real
    corruption, not a legacy container-type migration. Per the
    validation-is-a-tripwire principle, this function must never silently
    swallow that into an empty default -- doing so previously turned a loud
    crash into silent, undetectable data loss (e.g. a corrupted
    ``layer_dict_main_keys`` quietly coming back as an empty dict instead of
    surfacing the corruption). Such a field is left in its original,
    detectably-wrong state and this function raises ``TorchLensIOError`` so
    the caller learns about the corruption immediately, instead of shipping a
    Trace that looks valid but silently lost data.

    Parameters
    ----------
    state:
        Mutable serialized state dict being restored. Expected to have
        already been passed through ``default_fill_state`` with the same
        ``defaults`` mapping (so every field in ``defaults`` is present).
    defaults:
        The same defaults mapping passed to ``default_fill_state``.
    exclude:
        Field names to skip even though their default value is a plain
        container, for fields whose real declared type is a union with a
        non-container alternative that must not be coerced away (e.g. a
        legacy sentinel string like ``"all"``).

    Raises
    ------
    TorchLensIOError
        If a field is present with a value that cannot be converted to its
        declared container type (e.g. an ``int`` where a ``dict`` is
        declared). This is corruption, not a legacy-format field -- the
        function refuses to silently discard it.
    """

    for field_name, default_value in defaults.items():
        if field_name in exclude or field_name not in state:
            continue
        declared_type = type(default_value)
        if declared_type not in _COERCIBLE_CONTAINER_TYPES:
            continue
        current_value = state[field_name]
        if isinstance(current_value, declared_type):
            continue
        if current_value is None:
            # Absent-equivalent: field existed in the old schema but was never
            # populated. Lossless to convert to an empty typed container.
            state[field_name] = declared_type()
            continue
        try:
            state[field_name] = declared_type(current_value)
        except (TypeError, ValueError) as exc:
            raise TorchLensIOError(
                f"State field {field_name!r} is present with value of type "
                f"{type(current_value).__name__!r} that cannot be converted to the "
                f"declared container type {declared_type.__name__!r}. This is not a "
                "recognized legacy container-type migration (e.g. a set stored where "
                "a list is now declared) -- it looks like corrupted or tampered "
                "pickle state. Refusing to silently discard the field's contents; "
                "fix the source of the corruption rather than suppressing this error."
            ) from exc


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
    >>> tl.rehydrate_nested(log)
    >>> log.save("demo_bundle_copy")
    """

    from .rehydrate import rehydrate_nested as _rehydrate_nested

    _rehydrate_nested(trace, map_location=map_location, payload_hints=payload_hints)
