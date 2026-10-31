"""Per-sub-key TRAVEL POLICY for ``Trace.annotations`` (lane F40a, foldA D6).

``Trace.annotations`` is classified as inheritable graph state
(``data_classes/_trace_components.py``), so every sub-key rides every
copy-on-write fork verbatim -- and until this module existed NO sub-key had a
travel policy, so a fresh re-execution product (``trace.run(inputs=...)``,
guarded-fast refresh, loaded-sparse run) carried the ORIGINAL capture's
episode ledger and observer values under a ``path_faithfulness=VERIFIED``
report with zero warnings (the foldA round-3 R11/R12 receipts). The defect is
the ABSENCE of a policy, so the fix is a policy REGISTRY, per sub-key, not an
episode-only patch: the question "does this evidence describe THIS product's
execution?" is asked once per key, here.

Pattern-of-record: :mod:`torchlens._io.prerelease`'s
``_ANNOTATIONS_KEY_REGISTRY`` (a per-sub-key registry for the persistence
policy dimension); this is the sibling registry for the TRAVEL dimension.

THE RULE (foldA D6, shed 0): capture-evidence annotations never ride a fresh
execution. ``path_faithfulness=VERIFIED`` may NEVER coexist with foreign step
evidence. The seam is the ONE provider settlement finalizer
(``torchlens._runnable_providers._finalize_provider_run``), which every run
provider -- loaded sparse, live refresh, guarded-fast loaded, guarded-fast
live -- settles through; the seam file takes a one-line policy call, never
episode logic. A plain ``Trace.fork()`` is NOT a fresh execution (the fork's
records still hold the capture's own values), so forks carry these keys
until an engine re-executes; the fork builder cross-references this module
instead of deciding per key.

This lane deliberately mints NO digest and NO persisted field (foldA D7): the
positive "this ledger belongs to this product" claim is F40b's mint and F42's
consumption. This module only stops the ledger riding a fresh execution.

The per-op ``Op.episode_step`` stamps (lane F42, written at settlement) are
evidence of the SAME captured execution as the ledger, so the policy scrubs
them on every fresh-execution product alongside the ledger drop (W051 fix
for audit finding AUD-CODE 1.1: a stamped product whose ledger became a
travel note read as ``capture_kind="plain"`` and could be SAVED but never
LOADED -- the C07X identity-fact gate refuses stamps without a declaration,
correctly; the defect was the surviving stamps, never the gate).

Every spelling here is DOCUMENTED-UNSTABLE pending the naming session;
semantics are pinned by the foldA memo (D6) and the red tests in
``tests/test_episode_truth_travel_policy.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

__all__ = [
    "TRAVEL_CAPTURE_EVIDENCE",
    "AnnotationsTravelPolicy",
    "register_travel_policy",
    "registered_travel_policies",
    "scrub_episode_step_stamps",
    "scrub_fresh_execution_annotations",
    "travel_policy_for",
]

#: The one shipped policy class: the sub-key is EVIDENCE derived from one
#: specific executed forward (per-step ledgers, capture-time observations).
#: It never rides a product presenting a different execution.
TRAVEL_CAPTURE_EVIDENCE = "capture_evidence"

DropMode = Literal["note", "remove"]
"""How a capture-evidence key leaves a fresh-execution product.

``note``
    Replace the payload IN-KEY with the already-load-validated diagnostic
    record shape ``{"quarantined": True, "code": ..., "detail": ...}`` (the
    exact shape ``validate_loaded_episode_annotations`` round-trips), so the
    drop stays visible on the product and survives save/load without minting
    a new persistence write path.
``remove``
    Delete the key. Right for keys whose readers treat absence as "nothing
    observed on this product" (``Trace.logged_values`` returns an empty
    mapping), where an in-key note would masquerade as a value payload.
"""


@dataclass(frozen=True)
class AnnotationsTravelPolicy:
    """One registered ``Trace.annotations`` sub-key travel row.

    Parameters
    ----------
    policy:
        Policy class token; only :data:`TRAVEL_CAPTURE_EVIDENCE` ships.
    drop_mode:
        How the key leaves a fresh-execution product (see :data:`DropMode`).
    note_code:
        Diagnostic code stamped into the in-key note (``note`` mode only).
        A note-payload token, never an exception code: nothing raises it.
    owner:
        Short owner/reason string for the registry inventory.
    """

    policy: str
    drop_mode: DropMode
    note_code: str | None
    owner: str


#: Registered sub-key travel rows. ``sidecar`` is deliberately ABSENT: its
#: travel semantics belong to the C01 sidecar-seam owner, argued in that
#: lane, per key, once (foldA s8).
_TRAVEL_REGISTRY: dict[str, AnnotationsTravelPolicy] = {}


def register_travel_policy(
    key: str,
    *,
    policy: str = TRAVEL_CAPTURE_EVIDENCE,
    drop_mode: DropMode = "remove",
    note_code: str | None = None,
    owner: str,
) -> None:
    """Register one ``Trace.annotations`` sub-key travel row.

    Parameters
    ----------
    key:
        The annotations sub-key (e.g. ``"episode"``).
    policy:
        Policy class token; only :data:`TRAVEL_CAPTURE_EVIDENCE` is accepted
        (an unknown token would silently register a policy nothing applies).
    drop_mode:
        ``"note"`` or ``"remove"`` (see :data:`DropMode`).
    note_code:
        Required with ``drop_mode="note"``; forbidden with ``"remove"``.
    owner:
        Short owner/reason string kept in the registry inventory.

    Raises
    ------
    ValueError
        On an empty key, an unknown policy token, or a ``note_code`` that
        disagrees with the drop mode.
    """

    if not key or not isinstance(key, str):
        raise ValueError("annotations travel-policy key must be a non-empty string")
    if policy != TRAVEL_CAPTURE_EVIDENCE:
        raise ValueError(
            f"unknown annotations travel policy {policy!r}; only "
            f"{TRAVEL_CAPTURE_EVIDENCE!r} ships (foldA D6)"
        )
    if (drop_mode == "note") != (note_code is not None):
        raise ValueError(
            "annotations travel-policy note_code is required with drop_mode='note' "
            "and forbidden with drop_mode='remove'"
        )
    _TRAVEL_REGISTRY[key] = AnnotationsTravelPolicy(
        policy=policy, drop_mode=drop_mode, note_code=note_code, owner=owner
    )


def travel_policy_for(key: str) -> AnnotationsTravelPolicy | None:
    """Return the registered travel row for one sub-key, or ``None``."""

    return _TRAVEL_REGISTRY.get(key)


def registered_travel_policies() -> dict[str, AnnotationsTravelPolicy]:
    """Return a snapshot of the registered sub-key travel rows."""

    return dict(_TRAVEL_REGISTRY)


def _episode_drop_detail(payload: Any) -> str:
    """Build the human-readable in-key note detail for a dropped episode key."""

    episode_id = None
    if isinstance(payload, Mapping):
        header = payload.get("header")
        if isinstance(header, Mapping):
            episode_id = header.get("episode_id")
        declared = payload.get("declared")
        if episode_id is None and isinstance(declared, Mapping):
            episode_id = declared.get("episode_id")
    origin = f" (episode_id={episode_id})" if episode_id else ""
    return (
        f"episode evidence{origin} describes the ORIGINAL captured execution; "
        "this product is a fresh re-execution (run/refresh), so the per-step "
        "ledger was dropped and the per-op episode_step stamps were cleared by "
        "the annotations travel policy. Re-capture with "
        "tl.trace(..., episode=...) to derive step evidence for these inputs."
    )


def scrub_episode_step_stamps(trace: Any) -> int:
    """Clear every ``Op.episode_step`` stamp on ONE fresh-execution product.

    The stamps are per-op evidence of the captured execution (which stepped
    call recorded the op); on a re-executed product they would answer
    ``at_step`` post hoc from stale evidence, and they make the product
    unloadable once the ledger is a travel note (the load gate admits stamps
    only beside an episode declaration). Returns the number of stamps
    cleared; a product with no stamps is a no-op.
    """

    cleared = 0
    for op in getattr(trace, "layer_list", None) or ():
        if getattr(op, "episode_step", None) is None:
            continue
        try:
            op.episode_step = None
        except (AttributeError, TypeError):
            continue
        cleared += 1
    return cleared


def scrub_fresh_execution_annotations(trace: Any) -> tuple[str, ...]:
    """Apply the fresh-execution travel policy to ONE product trace.

    Called from the provider settlement finalizer for every product that
    presents a re-executed forward. Registered capture-evidence keys are
    dropped per their drop mode; every other sub-key travels untouched. The
    per-op ``Op.episode_step`` stamps are cleared on every fresh-execution
    product (:func:`scrub_episode_step_stamps`), whether or not the product
    still carries the ledger key -- the stamps describe the same original
    execution the ledger does.

    Parameters
    ----------
    trace:
        The product trace about to be handed back inside a ``RunResult``
        (a COW fork, or -- on the guarded-fast live path -- the user's live
        trace, whose payloads the run just refreshed in place).

    Returns
    -------
    tuple[str, ...]
        The sub-keys the policy dropped, in registry order (empty when the
        product carried none of them).
    """

    scrub_episode_step_stamps(trace)
    annotations = getattr(trace, "annotations", None)
    if not isinstance(annotations, dict):
        return ()
    dropped: list[str] = []
    for key, row in _TRAVEL_REGISTRY.items():
        if key not in annotations:
            continue
        payload = annotations[key]
        if isinstance(payload, Mapping) and payload.get("quarantined") is True:
            # Already an inert diagnostic record (load quarantine or a prior
            # policy drop): nothing evidential left to drop, and rewriting it
            # would erase the original diagnostic.
            continue
        if row.drop_mode == "note":
            annotations[key] = {
                "quarantined": True,
                "code": row.note_code,
                "detail": _episode_drop_detail(payload),
            }
        else:
            del annotations[key]
        dropped.append(key)
    return tuple(dropped)


# ---------------------------------------------------------------------------
# Registered rows (lane F40a). ``sidecar`` stays with its C01 owner.
# ---------------------------------------------------------------------------

register_travel_policy(
    "episode",
    drop_mode="note",
    # A note-payload diagnostic token riding the episode key's existing
    # quarantine grammar; never raised, so it lives outside the exception
    # code contract.
    note_code="episode_evidence_dropped_fresh_execution",
    owner="F40a: per-step episode evidence of ONE captured execution (foldA R11)",
)

register_travel_policy(
    "logged_values",
    drop_mode="remove",
    owner="F40a: capture-time observer values of ONE captured execution (foldA R12)",
)
