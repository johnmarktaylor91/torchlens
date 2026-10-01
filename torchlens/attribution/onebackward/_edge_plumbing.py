"""Edge-grain plumbing bought NOW for the deferred EAP stage (item 7, D16).

The edge-attribution-patching requirement needs a trustworthy map from each
backward grad-INPUT slot to the exact forward consumption it corresponds to.
That correspondence is the genuinely hard problem: on real gpt2, 155 of 594
backward nodes have no direct forward mate, so positional zipping would
produce plausible WRONG edge weights -- exactly the error class the sprint
exists to prevent. This module mints the ``GradInputUseMap`` CONTRACT while
forward provenance and runtime topology coexist (a capture-side seam), with
a fail-closed status per slot: ``exact`` only for provably-unambiguous 1:1
correspondences, ``ambiguous`` when several forward consumptions are
candidates, ``unsupported`` otherwise. **EAP remains honestly blocked after
this build** -- the map is session-time plumbing (no persisted fields; its
persistence is the EDGE-MAP-SCHEMA coordination with the NEXT trace-schema
bump), and edge-grain reads stay out of the v1 surface.

The second seam is origin-namespaced backward-pass stamping: a bare pass
index is ambiguous on arrival, so recorded backward passes can carry
``(origin in {internal, read, user}, target_ids, chunk_id)``. v1 reads run
suppressed (never recorded), so the stamps are session-time and default
``user`` for unstamped recorded passes.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import Any

from ._accessor import read_edge_index
from ._errors import ReadError

__all__ = [
    "GradInputUse",
    "GradInputUseMap",
    "mint_grad_input_use_map",
    "PASS_ORIGINS",
    "PassOriginStamp",
    "stamp_backward_pass_origin",
    "backward_pass_origins",
]

# Closed correspondence-status vocabulary (D16 / section 8 plumbing).
GRAD_INPUT_USE_STATUSES: tuple[str, ...] = ("exact", "ambiguous", "unsupported")

# Closed origin vocabulary for backward-pass stamps.
PASS_ORIGINS: tuple[str, ...] = ("internal", "read", "user")


@dataclass(frozen=True)
class GradInputUse:
    """Correspondence claim for one backward grad-input slot.

    Attributes
    ----------
    status:
        ``'exact'`` (provably one forward consumption), ``'ambiguous'``
        (several candidates -- NEVER silently picked from), or
        ``'unsupported'`` (no derivable forward mate).
    addresses:
        Candidate canonical forward-edge addresses
        ``(child_func_call_id, arg_kind, arg_path)``; exactly one for
        ``exact``, possibly many for ``ambiguous``, empty for
        ``unsupported``.
    """

    status: str
    addresses: tuple[tuple[Any, ...], ...]


@dataclass(frozen=True)
class GradInputUseMap:
    """Session-time backward-slot -> forward-consumption correspondence map.

    Attributes
    ----------
    entries:
        ``(site label, grad_input_slot)`` -> :class:`GradInputUse`.
    unmated_sites:
        Labels of addressable sites whose producing node has NO derivable
        forward mate for any slot -- the census that kills positional
        zipping (155/594 on real gpt2).
    census:
        Status -> count over all entries, for the regression pin.
    """

    entries: dict[tuple[str, int], GradInputUse]
    unmated_sites: tuple[str, ...]
    census: dict[str, int]


def mint_grad_input_use_map(trace: Any) -> GradInputUseMap:
    """Mint the fail-closed correspondence map for a live trace.

    A slot is ``exact`` ONLY when the consuming op has exactly one recorded
    tensor-argument edge and its producing node has exactly one
    differentiable next-function slot -- the provably-unambiguous 1:1 case.
    Multiple candidate forward edges mint ``ambiguous`` (candidates listed,
    never picked from); everything else is ``unsupported``.

    Parameters
    ----------
    trace:
        Live torch trace with edge provenance
        (``intervention_ready``-gated ``trace.edges``).

    Returns
    -------
    GradInputUseMap
        The session-time contract object; never persisted in v1
        (EDGE-MAP-SCHEMA owns its future persistence).

    Raises
    ------
    ReadError
        Code ``read_edge_provenance_unavailable`` when the trace has no edge
        provenance to mint from.
    """

    index = read_edge_index(trace)
    try:
        edge_records = list(trace.edges)
    except Exception as exc:  # noqa: BLE001 - typed re-raise below
        raise ReadError(
            "Edge-use provenance is unavailable on this trace, so the "
            "grad-input correspondence map cannot be minted. Remedy: "
            "capture with CaptureOptions(intervention_ready=True)",
            code="read_edge_provenance_unavailable",
        ) from exc
    # Forward consumptions by CONSUMING op label. Edge records spell labels
    # BARE (layer level), so a multi-pass consumer naturally accumulates the
    # per-pass records under one key and mints AMBIGUOUS -- never a silent
    # positional pick across passes.
    consumptions: dict[str, list[tuple[Any, ...]]] = {}
    for record in edge_records:
        address = (record.child_func_call_id, record.arg_kind, record.arg_path)
        consumptions.setdefault(record.child_label, []).append(address)
    entries: dict[tuple[str, int], GradInputUse] = {}
    unmated: list[str] = []
    census = dict.fromkeys(GRAD_INPUT_USE_STATUSES, 0)
    for label, edge in index.edges.items():
        next_functions = getattr(edge.node, "next_functions", ())
        differentiable_slots = [
            slot for slot, (parent, _) in enumerate(next_functions) if parent is not None
        ]
        addresses = tuple(consumptions.get(label, ()) or consumptions.get(edge.layer_label, ()))
        site_mated = False
        for slot in differentiable_slots:
            if len(addresses) == 1 and len(differentiable_slots) == 1:
                use = GradInputUse(status="exact", addresses=addresses)
            elif len(addresses) > 1:
                use = GradInputUse(status="ambiguous", addresses=addresses)
            else:
                use = GradInputUse(status="unsupported", addresses=())
            entries[(label, slot)] = use
            census[use.status] += 1
            site_mated = site_mated or use.status != "unsupported"
        if differentiable_slots and not site_mated:
            unmated.append(label)
    return GradInputUseMap(entries=entries, unmated_sites=tuple(unmated), census=census)


@dataclass(frozen=True)
class PassOriginStamp:
    """Origin stamp for one recorded backward pass.

    Attributes
    ----------
    origin:
        ``'internal'`` / ``'read'`` / ``'user'``.
    target_ids:
        Read target ids the pass served (empty for user passes).
    chunk_id:
        Engine chunk ordinal, or ``None``.
    """

    origin: str
    target_ids: tuple[str, ...] = ()
    chunk_id: int | None = None


_PASS_STAMPS: weakref.WeakKeyDictionary[Any, dict[int, PassOriginStamp]]
_PASS_STAMPS = weakref.WeakKeyDictionary()


def stamp_backward_pass_origin(
    trace: Any,
    pass_index: int,
    origin: str,
    *,
    target_ids: tuple[str, ...] = (),
    chunk_id: int | None = None,
) -> PassOriginStamp:
    """Stamp one recorded backward pass with its origin namespace.

    Session-time only (no persisted fields in v1); the EDGE stage's recorded
    read passes will stamp ``origin='read'`` with their target ids and chunk
    ordinal so a bare pass index is never ambiguous on arrival.

    Parameters
    ----------
    trace:
        The owning trace.
    pass_index:
        1-based recorded backward pass index.
    origin:
        Member of :data:`PASS_ORIGINS`.
    target_ids:
        Read target ids the pass served.
    chunk_id:
        Engine chunk ordinal.

    Raises
    ------
    ReadError
        Code ``read_pass_origin_invalid`` outside the closed vocabulary or
        on re-stamping an already-stamped pass.
    """

    if origin not in PASS_ORIGINS:
        raise ReadError(
            f"Backward-pass origin {origin!r} is not in the closed "
            f"vocabulary {PASS_ORIGINS}. Remedy: pick a listed origin",
            code="read_pass_origin_invalid",
            origin=origin,
        )
    stamps = _PASS_STAMPS.setdefault(trace, {})
    if pass_index in stamps:
        raise ReadError(
            f"Backward pass {pass_index} is already stamped "
            f"({stamps[pass_index]!r}); stamps are write-once. Remedy: "
            "stamp each recorded pass exactly once",
            code="read_pass_origin_invalid",
            pass_index=pass_index,
        )
    stamp = PassOriginStamp(origin=origin, target_ids=tuple(target_ids), chunk_id=chunk_id)
    stamps[pass_index] = stamp
    return stamp


def backward_pass_origins(trace: Any) -> dict[int, PassOriginStamp]:
    """Return origin stamps for every recorded backward pass of a trace.

    Unstamped recorded passes default to ``origin='user'`` (an ordinary
    ``log_backward``); the default is applied at read time, never stored.
    """

    stamps = dict(_PASS_STAMPS.get(trace, {}))
    recorded = int(getattr(trace, "num_backward_passes", 0) or 0)
    for pass_index in range(1, recorded + 1):
        stamps.setdefault(pass_index, PassOriginStamp(origin="user"))
    return stamps
