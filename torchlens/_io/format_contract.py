"""The tlspec FORMAT CONTRACT: L1 product vocabulary + record-adjacent helpers.

The persistence four-way ruling (architecture memo 3.2): the format contract
and its record-adjacent helpers -- ``TLSPEC_VERSION``, ``FieldPolicy``,
``BlobRef``, payload load hints, version read, fill-state defaults, container
coercion -- live at L1 beside ``data_classes/``. These names define the
portable PRODUCT, so every ``data_classes -> _io`` edge is intra-layer by
construction. ``torchlens._io``'s package ``__init__`` is a declared
re-export FACADE over this module; no public spelling changed in the split
(memo s9 item 3).
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, NamedTuple

from .format_errors import (
    ArtifactVersionAboveRuntimeError,
    ArtifactVersionBelowFloorError,
    TorchLensIOError,
)
from .prerelease import validate_prerelease_state

__tl_layer__ = "L1"

# v6 adds persisted ModuleCall forward-pre-hook provenance value objects.
# v7 adds the persisted capture outcome (`_capture_outcome`, string-only payload).
# v8 is the coordinated feature-sprint activation: every S3 pre-release-gated
# family flips to its persisting policy together (Op.site_key, the L6 edge/
# audit families, L1 grouping + grouping_policy, L8 distributed_scope, L9
# grad_fn_timing_provenance + checkpoint_invocation_witness, the L7a
# structure_only marker, the L3 primitive-op profile + kernel telemetry, the
# S6 Bundle member_relations key, and the S7 episode annotations ledger),
# with their load-validation rows live on real artifacts.
# v9 is the completeness-work coordinated schema write (C07): the audit
# row grammar admits the shipped ACT per-site "source" disclosure, PARAM
# rows/recipes, and the EVENT (intervention_event_v2) row family with its
# optional hash-chain extension; the "sidecar" annotations namespace flips
# from the pre-release registrar to plain persistence with envelope-shape
# load validation; the "health_facts" and "capture_advisories" annotation
# families are admitted as reserved keys; and the entry-dark field slots for
# injection provenance (Op.injection_provenance), source snapshots
# (Trace.source_snapshots + FuncCallLocation.source_file_digest), and the
# structure-only evidence envelope (Trace.structure_evidence) are declared
# with fail-closed load validation so their Phase-3 writers (F01/F30/F33)
# need no further version bump.
# The C07X amendment rides the SAME v9 window (TLSPEC_VERSION stays 9;
# every slot optional/entry-dark): bundle relation grammar v2 (required/
# optional split, successor_of evidence envelope + carry_mode/state_source,
# the graded-claim vocabulary, the 1 MiB per-row evidence budget) with the
# three-leg preserve-and-disclose loader doctrine and Bundle
# preserved-sections carriage; episode ledger grammar v2 (family version 2:
# declared step_output_kind/step_output_from/step_axis, generic step_output
# rows, arithmetic cache_len DELETED for the measured carried-state witness
# slots entry_state_digest/exit_state_digest, capture_digest +
# perturbed/intervention_digest/fire_count coupling slots, optional
# step_join); the Trace.root_entry_point identity fact (written
# unconditionally) with the Op.tl_authored_root marker and Op.episode_step
# stamp -- all with fail-closed load validation so the F40b/F40c/F41/F42/
# F-EPISODE/F-WITNESS writers need no further version bump.
TLSPEC_VERSION = 9

# Rehydration floor: artifacts older than tlspec_version 6 refuse to load
# instead of being resurrected through legacy field-alias ladders. The FIRST
# tlspec-6 writer was released v2.31.0 (measured on genuine wheels, ecosystem
# panel r3 -- the historical "first shipped in torchlens 2.33" comment here
# was false and backed the producer inequality that orphaned lawful
# v2.31.0/v2.32.4 artifacts; producer pair-consistency now lives in the
# governed ledger, ``torchlens._io.compat_ledger``, gate G5).
# ``MIN_TORCHLENS_VERSION_TEXT`` remains the release-family text stamped into
# the writer contract; its VALUE is frozen with the committed v9 contract
# golden (C07 territory) and is no longer matched against any manifest.
MIN_TLSPEC_VERSION = 6
MIN_TORCHLENS_VERSION_TEXT = "2.33"


# Ledger-derived honesty (gate G6): a below-floor remedy must never NAME a
# release without verified read evidence -- the shipped ">= 2.33" remedy told
# genuine v2.16 ModelLog holders to use releases proven unable to read their
# artifacts. Callers with ledger context pass a derived ``remedy=``; this
# default stays honest for unknown pre-floor eras by pointing at the ledger
# instead of asserting a reader.
_BELOW_FLOOR_REMEDY = (
    "Re-save the artifact with a torchlens release verified to read it; "
    "torchlens.ecosystem.compat_window() lists the governed rows and their "
    "verified readers."
)


def below_floor_error(
    *,
    observed: str,
    subject: str = "Artifact",
    path: str | None = None,
    remedy: str | None = None,
    code: str = "artifact_version_below_floor",
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
    remedy:
        Ledger-derived remedy override (gate G6). Callers that can classify
        the artifact pass the verified-reader remedy from
        ``torchlens._io.compat_ledger.bridge_reader_remedy``; the default
        stays honest by pointing at the ledger instead of naming a reader.
    code:
        Always ``"artifact_version_below_floor"``; raise sites pass it
        explicitly so the S-17 census sees the code where the raise happens,
        not buried in this factory (registry-kernel precedent).

    Returns
    -------
    ArtifactVersionBelowFloorError
        The typed refusal, ready to raise.
    """

    effective_remedy = _BELOW_FLOOR_REMEDY if remedy is None else remedy
    message = (
        f"{subject} has {observed}, below the supported rehydration floor "
        f"tlspec_version={MIN_TLSPEC_VERSION} (torchlens "
        f"{MIN_TORCHLENS_VERSION_TEXT}). Remedy: {effective_remedy}"
    )
    return ArtifactVersionBelowFloorError(
        message,
        code=code,
        observed=observed,
        floor_tlspec_version=MIN_TLSPEC_VERSION,
        floor_torchlens_version=MIN_TORCHLENS_VERSION_TEXT,
        path=path,
        remedy=effective_remedy,
    )


def raise_if_manifest_below_floor(raw_version: Any, path: str) -> None:
    """Refuse an integer manifest stamp below the rehydration floor.

    The shared preflight guard: a below-floor artifact also fails the current
    schema, so this runs BEFORE schema validation to refuse with the floor
    named instead of a missing-field error. Non-integer stamps return
    untouched (their own typed checks own that case).

    Parameters
    ----------
    raw_version:
        The manifest's raw ``tlspec_version`` value.
    path:
        Artifact path for the structured fields.

    Raises
    ------
    ArtifactVersionBelowFloorError
        When ``raw_version`` is an int below ``MIN_TLSPEC_VERSION``.
    """

    if isinstance(raw_version, int) and raw_version < MIN_TLSPEC_VERSION:
        raise below_floor_error(
            observed=f"tlspec_version={raw_version}",
            subject="Bundle manifest",
            path=path,
            code="artifact_version_below_floor",
        )


_ABOVE_CEILING_REMEDY = "Upgrade torchlens to the release that wrote this artifact (or newer)."


def above_ceiling_error(
    *,
    observed: int,
    subject: str = "Artifact",
    path: str | None = None,
    code: str = "artifact_version_above_runtime",
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
    code:
        Always ``"artifact_version_above_runtime"``; raise sites pass it
        explicitly so the S-17 census sees the code where the raise happens,
        not buried in this factory (registry-kernel precedent).

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
        code=code,
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

    raise below_floor_error(
        observed=version_text,
        subject=f"{cls_name} state",
        code="artifact_version_below_floor",
    )


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


# Pickle-visible identity stays the historical facade path: bundle metadata
# pickles reference these types by (module, qualname), and the guarded
# unpickler allowlists exactly those paths -- the physical split must not
# change a single persisted byte (move-compat law, memo org rule 9). The
# load-hint classes are runtime options, never persisted, so they keep their
# real defining module (the class-member license oracle reads it).
for _cls in (BlobRef, FieldPolicy):
    _cls.__module__ = "torchlens._io"
del _cls


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
        raise above_ceiling_error(
            observed=version,
            subject=f"{cls_name} pickle state",
            code="artifact_version_above_runtime",
        )
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
