"""Pre-release field registrar: sprint-gated fields under a frozen tlspec.

VERSION DISCIPLINE (persistence-schema seam): no active schema change ships
under the CURRENT ``tlspec_version``. A new portable field therefore lands in
its record's ``FIELD_POLICY`` declared ``FieldPolicy.DROP`` (non-persisting)
and is REGISTERED here. Portability and round-trip exit gates run under the
TEST-ONLY activation switch, which does exactly two things while active:

- each registered field scrubs with its intended PERSISTING policy instead of
  its declared ``DROP`` (one hook in ``scrub._effective_policy``), and
- every Trace state written stamps the pre-release marker
  (:data:`PRERELEASE_STATE_KEY`), unconditionally -- the marker rides ANY
  switch-on write, so no per-field accounting bug can omit it.

Loads validate the marker at the one version chokepoint
(:func:`torchlens._io.read_tlspec_version`) and REFUSE typed
(``PreReleaseArtifactError``) unless the switch is active, so a switched
artifact can never pass as a real current-version artifact. The coordinated
version bump later flips the declared policies to persisting, adds the
load-validation rows, and retires the registrations; this module then holds
an empty registry again.

The pytest guard on activation is a seatbelt against accidental production
activation, not a security boundary -- the marker plus the fail-closed load
refusal is the belt that keeps pre-release artifacts out of circulation.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from . import FieldPolicy

#: State-dict key carrying the pre-release marker on switch-on writes. Lives
#: next to ``tlspec_version`` in the scrubbed Trace envelope and is popped and
#: validated by ``read_tlspec_version`` before any version acceptance.
PRERELEASE_STATE_KEY = "_tlspec_prerelease"

#: Exact marker value; anything else in the payload refuses even under the
#: active switch (fail-closed against tampered or hand-built payloads).
PRERELEASE_MARKER = "torchlens-prerelease-fields-v1"

#: Registered gated fields: owner record class -> field name -> the persisting
#: policy the field will carry after the coordinated version bump. Exact-type
#: keyed (record classes are not subclassed across the persistence surface).
_REGISTRY: dict[type, dict[str, FieldPolicy]] = {}

#: Registered gated ANNOTATIONS keys: sub-keys of the (already-persisting)
#: ``Trace.annotations`` mapping that constitute NEW persistence write paths
#: under the frozen tlspec version. They are scrubbed OUT of every persisted
#: state unless the test-only switch is active (the S7 episode-ledger home
#: rides this row). Value = a short owner/reason string for the inventory.
_ANNOTATIONS_KEY_REGISTRY: dict[str, str] = {}

#: The test-only activation switch. Read directly (module attribute) on the
#: scrub hot path; mutated only by :func:`activate_prerelease_fields`.
_ACTIVE: bool = False


def register_prerelease_field(
    owner: type,
    field_name: str,
    *,
    persisted_policy: FieldPolicy | None = None,
) -> None:
    """Register one sprint-gated field on a record class.

    Parameters
    ----------
    owner:
        Record class whose ``PORTABLE_STATE_SPEC`` declares the field.
    field_name:
        Declared field name. Its declared policy MUST be ``FieldPolicy.DROP``
        -- that is the whole discipline: the field never persists under the
        current tlspec version except beneath the test-only switch.
    persisted_policy:
        Policy the field scrubs with while the switch is active (and will
        carry after the coordinated bump). Defaults to ``FieldPolicy.KEEP``.

    Raises
    ------
    ValueError
        If the field is not declared on ``owner``, is not declared ``DROP``,
        or the requested persisting policy is itself ``DROP``.
    """

    from . import FieldPolicy

    spec = getattr(owner, "PORTABLE_STATE_SPEC", None)
    if not isinstance(spec, dict) or field_name not in spec:
        raise ValueError(
            f"{owner.__name__}.{field_name} is not a declared portable field; "
            "pre-release fields must be declared in the owning record's "
            "FIELD_POLICY before registration."
        )
    if spec[field_name] is not FieldPolicy.DROP:
        raise ValueError(
            f"{owner.__name__}.{field_name} is declared "
            f"{spec[field_name].value!r}, not 'drop'. Pre-release fields must "
            "land non-persisting (FieldPolicy.DROP) under the current tlspec "
            "version; only the coordinated version bump flips them."
        )
    resolved = FieldPolicy.KEEP if persisted_policy is None else FieldPolicy(persisted_policy)
    if resolved is FieldPolicy.DROP:
        raise ValueError(
            f"{owner.__name__}.{field_name}: a DROP persisted_policy is the "
            "declared default already; registration would be a no-op."
        )
    _REGISTRY.setdefault(owner, {})[field_name] = resolved


def unregister_prerelease_field(owner: type, field_name: str) -> None:
    """Remove one registration (tests restore; the bump retires for real)."""

    by_field = _REGISTRY.get(owner)
    if by_field is not None:
        by_field.pop(field_name, None)
        if not by_field:
            del _REGISTRY[owner]


def register_prerelease_annotations_key(key: str, *, owner: str) -> None:
    """Register one sprint-gated ``Trace.annotations`` sub-key.

    ``Trace.annotations`` itself persists (``FieldPolicy.KEEP``), so a NEW
    key inside it is a new persistence write path under the frozen tlspec
    version and must not ride real artifacts. Registered keys are removed
    from every persisted annotations payload unless the test-only switch is
    active; switch-on writes carry the pre-release marker as usual.

    Parameters
    ----------
    key:
        The annotations sub-key (e.g. ``"episode"``).
    owner:
        Short owner/reason string kept in the registrar inventory.
    """

    if not key or not isinstance(key, str):
        raise ValueError("pre-release annotations key must be a non-empty string")
    _ANNOTATIONS_KEY_REGISTRY[key] = owner


def unregister_prerelease_annotations_key(key: str) -> None:
    """Remove one annotations-key registration (tests restore; bump retires)."""

    _ANNOTATIONS_KEY_REGISTRY.pop(key, None)


def gated_annotations_keys() -> frozenset[str]:
    """Return the registered gated ``Trace.annotations`` sub-keys."""

    return frozenset(_ANNOTATIONS_KEY_REGISTRY)


def registered_prerelease_fields() -> dict[str, tuple[str, ...]]:
    """Return the registrar inventory: owner class name -> sorted field names.

    Gated annotations sub-keys appear under the synthetic owner name
    ``"Trace.annotations"`` so the inventory stays one flat mapping.
    """

    inventory = {owner.__name__: tuple(sorted(fields)) for owner, fields in _REGISTRY.items()}
    if _ANNOTATIONS_KEY_REGISTRY:
        inventory["Trace.annotations"] = tuple(sorted(_ANNOTATIONS_KEY_REGISTRY))
    return inventory


def prerelease_fields_active() -> bool:
    """Return whether the test-only activation switch is currently on."""

    return _ACTIVE


@contextmanager
def activate_prerelease_fields() -> Iterator[None]:
    """Activate the switch for one scope. TEST-ONLY: refuses outside pytest.

    Raises
    ------
    RuntimeError
        If not running under pytest (``PYTEST_CURRENT_TEST`` unset). The
        switch exists so portability exit gates can round-trip gated fields;
        it is never a production spelling.
    """

    global _ACTIVE
    if "PYTEST_CURRENT_TEST" not in os.environ:
        raise RuntimeError(
            "activate_prerelease_fields() is a TEST-ONLY switch for schema "
            "exit gates and refuses to run outside pytest. Gated fields ship "
            "for real only at the coordinated tlspec version bump."
        )
    previous = _ACTIVE
    _ACTIVE = True
    try:
        yield
    finally:
        _ACTIVE = previous


def effective_policy(owner_type: type, field_name: str, declared: FieldPolicy) -> FieldPolicy:
    """Return the policy persistence consumers must honor for one field NOW.

    The activation switch flips a registered DROP-declared field to its
    intended persisting policy on EVERY side of the persistence seam: scrub
    (write), rehydration (blob materialization at load), and the nested
    ``BlobRef`` resave guard. The write side alone is not enough -- a
    switched save of a ``BLOB_RECURSIVE``-registered field emits nested
    ``BlobRef`` leaves that only materialize back into tensors if the load
    side resolves the SAME effective policy (prebump acceptance finding,
    2026-08-17: ``Op.edge_substitutions`` loaded as dead ``BlobRef`` objects
    because rehydration consulted only the declared ``DROP``).

    Parameters
    ----------
    owner_type:
        Record class owning the field.
    field_name:
        Declared field name.
    declared:
        The field's declared policy from ``PORTABLE_STATE_SPEC``.

    Returns
    -------
    FieldPolicy
        ``declared``, unless the switch is active and the field is
        registered, in which case the registered persisting policy.
    """

    from . import FieldPolicy

    if declared is not FieldPolicy.DROP or not _ACTIVE:
        return declared
    override = persisted_policy_override(owner_type, field_name)
    return declared if override is None else override


def persisted_policy_override(owner_type: type, field_name: str) -> FieldPolicy | None:
    """Return the switched-on persisting policy for one field, else ``None``.

    Callers gate on the module attribute ``_ACTIVE`` first; this function does
    not re-check it so the off path stays one attribute read plus a branch.
    """

    by_field = _REGISTRY.get(owner_type)
    if by_field is None:
        return None
    return by_field.get(field_name)


def prerelease_marker_payload() -> dict[str, Any]:
    """Return the marker payload stamped into switch-on Trace writes."""

    fields = sorted(
        f"{owner.__name__}.{name}" for owner, by_field in _REGISTRY.items() for name in by_field
    )
    return {"marker": PRERELEASE_MARKER, "fields": fields}


def validate_prerelease_state(state: dict[str, Any], *, cls_name: str) -> None:
    """Pop and validate the pre-release marker from one serialized state.

    Absent marker: a real artifact, no-op. Present marker: refuses typed
    unless the switch is active, and refuses a malformed payload even when it
    is (fail-closed -- a hand-built or tampered marker never loads).

    Raises
    ------
    PreReleaseArtifactError
        If the state carries the marker while the switch is inactive, or the
        marker payload is malformed.
    """

    payload = state.pop(PRERELEASE_STATE_KEY, None)
    if payload is None:
        return
    from . import PreReleaseArtifactError

    if not _ACTIVE:
        raise PreReleaseArtifactError(
            f"{cls_name} state carries the pre-release field marker "
            f"({PRERELEASE_STATE_KEY!r}): it was written with sprint-gated "
            "fields activated and is NOT a supported artifact of this tlspec "
            "version. Re-save it without the test-only activation switch, or "
            "load it under activate_prerelease_fields() in the owning exit-"
            "gate test."
        )
    if not (isinstance(payload, dict) and payload.get("marker") == PRERELEASE_MARKER):
        raise PreReleaseArtifactError(
            f"{cls_name} pre-release marker payload is malformed: {payload!r}."
        )


# ---------------------------------------------------------------------------
# Live sprint-gated registrations (S3 registrar inventory rows). Each row is
# retired at the coordinated tlspec version bump that activates its family.
# The tlspec v8 bump (2026-08-17) retired every feature-megasprint row; the
# tlspec v9 bump (2026-08-27, lane C07) retired the C01 "sidecar" annotations
# row. The registry is empty again until the next sprint gates a new family
# here.
# ---------------------------------------------------------------------------
