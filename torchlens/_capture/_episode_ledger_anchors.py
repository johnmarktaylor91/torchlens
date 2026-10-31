"""Load-time anchors: does a grammatically valid episode ledger describe THIS product?

Split out of :mod:`._episode_ledger` (W051 lint fix; the ledger module sat
over the 2000-line unledgered-module cap). ONE responsibility: the structural
anchoring a loaded ``annotations["episode"]`` payload must pass AFTER its
grammar parse (audit 2.19: a valid ledger grafted onto any artifact used to
load clean and make the product an "episode"; the key's presence was the only
anchor). Every check returns the quarantine reason as a string, never raises;
the caller (``validate_loaded_episode_annotations``) owns the typed refusals
and the quarantine record.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = [
    "_anchor_loaded_ledger",
    "quarantine_loaded_ledger",
    "reanchor_loaded_episode_evidence",
]

_EVIDENCE_MISMATCH_DETAIL = (
    "episode ledger evidence column does not re-derive from this "
    "product's retained root output; the ledger does not describe "
    "this execution"
)
"""Anchor-4 quarantine reason (shared by the in-``__setstate__`` pass and the
post-blob-attach pass so the two doors spell one refutation)."""


def _recorded_module_passes(trace: Any, address: str) -> frozenset[int]:
    """The 1-based call (pass) indices the product recorded for ``address``.

    Read from the restored OP records (``Op.modules`` spells every enclosing
    module call ``"<address>:<pass>"``): inside ``Trace.__setstate__`` the
    module accessor is not yet attached on ``.tlspec`` loads, while the op
    rows are -- and the op rows are the identity facts the digest binds too.
    Falls back to the module accessor's call records on live traces whose
    ops carry no module entries.
    """

    passes: set[int] = set()
    prefix = f"{address}:"
    for op in getattr(trace, "layer_list", None) or ():
        for entry in getattr(op, "modules", None) or ():
            text = str(entry)
            if text.startswith(prefix) and text[len(prefix) :].isdigit():
                passes.add(int(text[len(prefix) :]))
    if passes:
        return frozenset(passes)
    try:
        calls = trace.modules[address].calls.values()
    except (LookupError, AttributeError, TypeError, ValueError):
        return frozenset()
    return frozenset(
        int(getattr(call, "call_index", index + 1)) for index, call in enumerate(calls)
    )


def _anchor_loaded_ledger(trace: Any, payload: Mapping[str, Any]) -> str | None:
    """Anchor a grammatically valid loaded ledger to ITS product (W051).

    Audit 2.19: a valid ledger grafted onto any artifact loaded clean and
    made the product an "episode" -- the presence of the key was the only
    anchor. Four checks, each returning the quarantine reason on failure:

    1. the header's ``stepped_module`` recorded at least one call on this
       product (read from the restored op records' module entries);
    2. every started row's ``member_call_index`` names one of that module's
       recorded calls, and the started-row count equals the call count;
    3. the persisted ``capture_digest`` equals the digest recomputed from the
       product + this ledger (the content-binding v2 form; a legacy v1
       identity digest is accepted at load and reads pre-binding at
       attestation);
    4. the evidence column re-derives from the product's retained root
       output (skipped, never failed, when the payload is not retained).
    """

    from ._episode_derivation import recompute_capture_digests, rederived_evidence_matches

    header = payload.get("header") or {}
    rows = [row for row in (payload.get("rows") or []) if isinstance(row, Mapping)]
    address = str(header.get("stepped_module"))
    passes = _recorded_module_passes(trace, address)
    if not passes:
        return (
            f"episode ledger header names stepped_module {address!r}, which "
            "recorded no call on this product; the ledger does not describe "
            "this capture"
        )
    started = [row for row in rows if row.get("status") != "absent"]
    if len(started) != len(passes):
        return (
            f"episode ledger carries {len(started)} started rows but the stepped "
            f"module {address!r} recorded {len(passes)} calls on this product"
        )
    for row in started:
        coord_value = row.get("coord")
        coord: Mapping[str, Any] = coord_value if isinstance(coord_value, Mapping) else {}
        index = coord.get("member_call_index")
        if not isinstance(index, int) or isinstance(index, bool) or index not in passes:
            return (
                f"episode ledger row {row.get('episode_step')!r} names "
                f"member_call_index {index!r}, outside the {len(passes)} recorded "
                f"calls of {address!r} on this product"
            )
    persisted = header.get("capture_digest")
    if isinstance(persisted, str) and persisted:
        bound, legacy = recompute_capture_digests(trace, payload)
        if persisted not in (bound, legacy):
            return (
                "episode ledger capture_digest does not match the digest "
                "recomputed from this product and the ledger's own content; "
                "the ledger was rewritten after minting or belongs to another "
                "product"
            )
    try:
        matches = rederived_evidence_matches(trace, payload)
    except Exception as exc:  # noqa: BLE001 - an unreadable payload is not a mismatch
        matches = None
        _ = exc
    if matches is False:
        return _EVIDENCE_MISMATCH_DETAIL
    return None


def quarantine_loaded_ledger(annotations: dict[str, Any], detail: str, *, stacklevel: int) -> None:
    """Replace a loaded episode payload by its ``episode_ledger_incoherent`` record.

    ONE warning, one closed-shape diagnostic record (``quarantined`` / ``code``
    / ``detail``); the product's rows stop being claims and the outcome
    derivation treats the ledger fail-closed. Both load-time anchor passes
    (``validate_loaded_episode_annotations`` inside ``Trace.__setstate__`` and
    :func:`reanchor_loaded_episode_evidence` after blob attach) write through
    this one door.
    """

    import warnings

    from ..errors import TorchLensWarning

    warnings.warn(
        "episode ledger failed load validation and was quarantined "
        f"(episode_ledger_incoherent): {detail}. The product's episode "
        "rows are no longer claims; the outcome derivation treats the ledger "
        "fail-closed.",
        TorchLensWarning,
        stacklevel=stacklevel,
    )
    annotations["episode"] = {
        "quarantined": True,
        "code": "episode_ledger_incoherent",
        "detail": detail,
    }


def reanchor_loaded_episode_evidence(trace: Any) -> None:
    """Re-run anchor 4 AFTER payload attach on a ``.tlspec`` load (W051-CAPT3).

    Inside ``Trace.__setstate__`` the root output is still an unmaterialized
    blob handle, so the fourth anchor of :func:`_anchor_loaded_ledger` had
    nothing to compare and admitted any ledger whose digest and coordinates
    held -- a ledger re-minted over another execution of the SAME program
    (two equal-length prompts) loaded clean and was only refuted on the first
    ``trace.episode_coupling`` read. The bundle loader calls this once the
    payloads are attached, so the LOAD itself refutes the swap: a mismatch
    quarantines through :func:`quarantine_loaded_ledger` exactly like the
    in-``__setstate__`` pass. Anchors 1-3 are not re-run (they hold or the
    payload is already quarantined); nothing here ever admits a payload the
    first pass refused.
    """

    annotations = getattr(trace, "annotations", None)
    if not isinstance(annotations, dict):
        return
    payload = annotations.get("episode")
    if not isinstance(payload, Mapping) or set(payload) != {"header", "rows"}:
        return  # absent, declaration-only, or already quarantined
    from ._episode_derivation import rederived_evidence_matches

    try:
        matches = rederived_evidence_matches(trace, payload)
    except Exception:  # noqa: BLE001 - an unreadable payload is not a mismatch
        return
    if matches is False:
        quarantine_loaded_ledger(annotations, _EVIDENCE_MISMATCH_DETAIL, stacklevel=3)
