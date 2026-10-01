"""Hop rule: grammar proposes, payload identity verifies (mikit D3/D4, wave 0a).

Internal substrate for dataflow walks over captured op graphs. A HOP is one
step upstream across an op that transports a value without changing what it
means: identity/view kinds, eval-mode dropout, dtype casts, and parseable
index subsets (which change shape but carry an explicit position mapping,
never a silent one). The rule has two halves:

- The GRAMMAR (a closed op-kind table) may only PROPOSE a hop.
- PAYLOAD IDENTITY (bitwise ``torch.equal`` between the walk endpoints,
  after replaying recorded casts/index subsets) VERIFIES the proposal and is
  MANDATORY before any tensor claim is made through the walk. Structure-only
  consumers (string facets, plan surfaces) may run on the proposer alone.

Every step carries the op's pass-qualified label and live ``site_key``
(mikit D4: pass-qualified identity is day-one identity, never a user
decision). A shape-changing op never hops silently: it either parses into an
explicit :class:`IndexMapRecord` or the walk stops at it, named.

This module is internal (underscore) and its spellings are
DOCUMENTED-UNSTABLE pending the naming session; ``torchlens.semantic``
recipes are its consumers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import torch

__all__ = [
    "BRANCH_GRAMMAR",
    "VALUE_GRAMMAR",
    "HopRecord",
    "HopRefusal",
    "HopWalk",
    "IndexMapRecord",
    "walk_upstream",
]

#: Ops that reproduce their single tensor parent's value bitwise.
_IDENTITY_FUNCS = frozenset({"contiguous", "clone", "detach", "alias", "identity"})

#: Ops that reproduce the parent's values when the recorded shape is unchanged.
_SHAPE_GATED_IDENTITY_FUNCS = frozenset({"view", "reshape"})

#: Dtype-cast ops; verification replays the recorded result dtype.
_CAST_FUNCS = frozenset({"to", "float", "double", "half", "bfloat16"})

#: Subscript ops; hop only when every recorded index parses (slices/ints).
_INDEX_FUNCS = frozenset({"__getitem__", "getitem"})

_DROPOUT_FUNCS = frozenset({"dropout"})

GrammarName = Literal["value", "branch"]

#: Value-preserving grammar (mikit D3): identity/view kinds, casts, parseable
#: index subsets, and dropout ONLY when the recorded args prove it inert
#: (eval mode or p == 0). Used wherever the walk backs a tensor claim.
VALUE_GRAMMAR: GrammarName = "value"

#: Branch-connectivity grammar: same kinds, but dropout proposes regardless of
#: training mode. Used ONLY for structural branch identification (e.g. "this
#: add consumes the attention branch"), where the returned tensor is the
#: consumer op's own captured output and never crosses the hop.
BRANCH_GRAMMAR: GrammarName = "branch"


@dataclass(frozen=True)
class HopRecord:
    """One verified-or-proposed hop across a value-transporting op."""

    op_label: str
    site_key: str | None
    func_name: str
    kind: Literal["identity", "cast", "index", "dropout"]
    detail: str


@dataclass(frozen=True)
class IndexMapRecord:
    """Explicit position mapping across the walk's index hops (mikit F6).

    ``derivation`` is ``"identity"`` when no index hop changed any extent
    (every recorded subscript kept every position) and ``"slice"`` when the
    kept positions were derived from the recorded slice arguments.
    ``kept_positions_by_dim`` holds, per result dimension, the source
    positions each result index maps to, or ``None`` for an untouched
    dimension. Consumers (DLA's position validation, ``logits_to_keep``
    disclosure) read this record instead of guessing from shapes.
    """

    derivation: Literal["identity", "slice"]
    hop_op_label: str | None
    site_key: str | None
    index_repr: str | None
    source_shape: tuple[int, ...] | None
    result_shape: tuple[int, ...] | None
    kept_positions_by_dim: tuple[tuple[int, ...] | None, ...] | None


@dataclass(frozen=True)
class HopWalk:
    """A completed anchor walk.

    ``verification`` is ``"payload_identity"`` when the endpoint payloads
    replayed bitwise-equal, ``"unavailable"`` when either endpoint payload was
    not saved (tensor claims must refuse), and ``"structure_only"`` when the
    caller asked for the proposer alone.
    """

    anchor: Any
    start_label: str
    hops: tuple[HopRecord, ...]
    verification: Literal["payload_identity", "unavailable", "structure_only"]
    index_map: IndexMapRecord


@dataclass(frozen=True)
class HopRefusal:
    """A walk that stopped without an anchor; ``reason`` names why and where."""

    reason: str
    at_label: str | None = None


def _single_parent_label(op: Any) -> str | None:
    """Return the op's sole tensor-parent label, or ``None`` if not exactly one."""

    parents = tuple(getattr(op, "parents", ()) or ())
    if len(parents) != 1:
        return None
    return str(parents[0])


def _recorded_shape(op: Any) -> tuple[int, ...] | None:
    """Return an op's recorded output shape as a plain int tuple."""

    shape = getattr(op, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(dim) for dim in shape)
    except (TypeError, ValueError):
        return None


def _dropout_is_inert(op: Any) -> bool:
    """Return whether recorded dropout args prove eval mode or p == 0.

    ``F.dropout`` records ``[p, training]`` positionally (and/or keyword).
    Absent or unparseable evidence fails closed: the value grammar then does
    not propose the hop.
    """

    pos = list(getattr(op, "non_tensor_pos_args", ()) or ())
    kwargs = dict(getattr(op, "non_tensor_kwargs", {}) or {})
    p: Any = kwargs.get("p", pos[0] if len(pos) >= 1 else None)
    training: Any = kwargs.get("training", pos[1] if len(pos) >= 2 else None)
    if isinstance(p, (int, float)) and float(p) == 0.0:
        return True
    return training is False


def _parse_index_args(op: Any) -> tuple[Any, ...] | None:
    """Return the recorded subscript as a tuple of slices/ints, or ``None``.

    Only plain slices and ints (basic indexing) are accepted; tensor/bool
    advanced indexing fails closed so a data-dependent gather can never pose
    as a position mapping.
    """

    pos = list(getattr(op, "non_tensor_pos_args", ()) or ())
    if len(pos) != 1:
        return None
    index = pos[0]
    entries = index if isinstance(index, tuple) else (index,)
    parsed: list[Any] = []
    for entry in entries:
        if isinstance(entry, slice):
            for bound in (entry.start, entry.stop, entry.step):
                if bound is not None and not isinstance(bound, int):
                    return None
            parsed.append(entry)
        elif isinstance(entry, int) and not isinstance(entry, bool):
            parsed.append(entry)
        else:
            return None
    return tuple(parsed)


def _propose(op: Any, grammar: GrammarName) -> HopRecord | None:
    """Classify one op under the grammar; ``None`` means no proposal."""

    func_name = str(getattr(op, "func_name", ""))
    if _single_parent_label(op) is None:
        return None
    classified = _classify_hop(op, func_name, grammar)
    if classified is None:
        return None
    kind, detail = classified
    label = str(getattr(op, "label", "<unknown>"))
    return HopRecord(label, getattr(op, "site_key", None), func_name, kind, detail)


HopKind = Literal["identity", "cast", "index", "dropout"]

#: Constant classifications: ops whose kind needs no per-op evidence.
_CONSTANT_HOP_KINDS: dict[str, tuple[HopKind, str]] = {
    **dict.fromkeys(_IDENTITY_FUNCS, ("identity", "identity op")),
    **dict.fromkeys(_CAST_FUNCS, ("cast", "dtype cast")),
}


def _classify_hop(op: Any, func_name: str, grammar: GrammarName) -> tuple[HopKind, str] | None:
    """Return the (kind, detail) classification for one proposable op."""

    constant = _CONSTANT_HOP_KINDS.get(func_name)
    if constant is not None:
        return constant
    if func_name in _SHAPE_GATED_IDENTITY_FUNCS:
        return _classify_shape_gated_view(op)
    if func_name in _INDEX_FUNCS and _parse_index_args(op) is not None:
        return "index", "recorded basic-index subset"
    if func_name in _DROPOUT_FUNCS:
        return _classify_dropout(op, grammar)
    return None


def _classify_dropout(op: Any, grammar: GrammarName) -> tuple[HopKind, str] | None:
    """Classify dropout: branch grammar always; value grammar only when inert."""

    if grammar == BRANCH_GRAMMAR:
        return "dropout", "dropout (branch grammar)"
    if _dropout_is_inert(op):
        return "dropout", "inert (eval-mode) dropout"
    return None


def _classify_shape_gated_view(op: Any) -> tuple[Literal["identity"], str] | None:
    """Classify a view/reshape as identity ONLY when the shape is unchanged."""

    own = _recorded_shape(op)
    trace = getattr(op, "trace", None)
    parent_label = _single_parent_label(op)
    if trace is None or own is None or parent_label is None:
        return None
    try:
        parent = trace.ops[parent_label]
    except (KeyError, ValueError):
        return None
    if own != _recorded_shape(parent):
        return None
    return "identity", "shape-preserving view"


def _payload(op: Any) -> torch.Tensor | None:
    """Return the op's saved output tensor, or ``None`` when unreadable."""

    readable = bool(
        getattr(op, "has_saved_activation", False) or getattr(op, "out_ref", None) is not None
    )
    if not readable:
        return None
    value = getattr(op, "out", None)
    return value if isinstance(value, torch.Tensor) else None


def _kept_positions(
    index_args: tuple[Any, ...], source_shape: tuple[int, ...]
) -> tuple[tuple[int, ...] | None, ...] | None:
    """Per result dim, the source positions kept by a basic-index subscript.

    Integer entries (which drop a dimension) refuse: the mapping would not be
    dimension-aligned with the result. Trailing untouched dims map ``None``.
    """

    per_dim: list[tuple[int, ...] | None] = []
    for dim, entry in enumerate(index_args):
        if dim >= len(source_shape):
            return None
        if isinstance(entry, int):
            return None
        positions = tuple(range(*entry.indices(source_shape[dim])))
        full = positions == tuple(range(source_shape[dim]))
        per_dim.append(None if full else positions)
    per_dim.extend([None] * (len(source_shape) - len(index_args)))
    return tuple(per_dim)


def _build_index_map(hops: tuple[HopRecord, ...], trace: Any) -> IndexMapRecord | HopRefusal:
    """Fold the walk's index hops into one explicit position map.

    Wave-0 scope: at most ONE index hop per walk (the ``logits_to_keep``
    class); two subsets would need slice composition and refuse instead.
    """

    index_hops = [hop for hop in hops if hop.kind == "index"]
    if not index_hops:
        return IndexMapRecord("identity", None, None, None, None, None, None)
    if len(index_hops) > 1:
        return HopRefusal(
            "walk crosses more than one index subset op; composing position maps "
            "is not supported, so the anchor claim would be unverifiable",
            at_label=index_hops[1].op_label,
        )
    hop = index_hops[0]
    try:
        op = trace.ops[hop.op_label]
        parent = trace.ops[str(tuple(op.parents)[0])]
    except (KeyError, IndexError, ValueError):
        return HopRefusal("index hop op records are unavailable", at_label=hop.op_label)
    index_args = _parse_index_args(op)
    source_shape = _recorded_shape(parent)
    result_shape = _recorded_shape(op)
    if index_args is None or source_shape is None:
        return HopRefusal("index hop arguments are unavailable", at_label=hop.op_label)
    kept = _kept_positions(index_args, source_shape)
    if kept is None:
        return HopRefusal(
            "index hop uses dimension-dropping or unparseable indexing; the position "
            "map would not be dimension-aligned",
            at_label=hop.op_label,
        )
    derivation: Literal["identity", "slice"] = (
        "identity" if all(entry is None for entry in kept) else "slice"
    )
    return IndexMapRecord(
        derivation=derivation,
        hop_op_label=hop.op_label,
        site_key=hop.site_key,
        index_repr=repr(index_args),
        source_shape=source_shape,
        result_shape=result_shape,
        kept_positions_by_dim=kept,
    )


def _verify_endpoints(
    trace: Any, start_op: Any, anchor: Any, hops: tuple[HopRecord, ...]
) -> Literal["payload_identity", "unavailable"] | HopRefusal:
    """Replay recorded casts/index subsets from the anchor and compare bitwise."""

    start_value = _payload(start_op)
    anchor_value = _payload(anchor)
    if start_value is None or anchor_value is None:
        return "unavailable"
    value = anchor_value
    for hop in reversed(hops):
        try:
            op = trace.ops[hop.op_label]
        except (KeyError, ValueError):
            return "unavailable"
        if hop.kind == "index":
            index_args = _parse_index_args(op)
            if index_args is None:
                return "unavailable"
            value = value[index_args]
        elif hop.kind == "cast":
            dtype = getattr(op, "dtype", None)
            if not isinstance(dtype, torch.dtype):
                return "unavailable"
            value = value.to(dtype)
    if value.shape != start_value.shape or not torch.equal(value, start_value):
        return HopRefusal(
            "payload identity FAILED between the walk endpoints: the proposed hops do "
            "not reproduce the consumed value, so the anchor claim is rejected",
            at_label=str(getattr(anchor, "label", "<unknown>")),
        )
    return "payload_identity"


def walk_upstream(
    trace: Any,
    start_op: Any,
    is_anchor: Callable[[Any], bool],
    *,
    grammar: GrammarName = VALUE_GRAMMAR,
    structure_only: bool = False,
    max_hops: int = 8,
) -> HopWalk | HopRefusal:
    """Walk upstream from ``start_op`` across proposed hops to an anchor.

    The start op itself is tested first (a zero-hop walk is legal). Each
    subsequent step requires the current op to be a grammar proposal with a
    single tensor parent. On anchor arrival, payload identity is verified
    endpoint-to-endpoint unless ``structure_only`` was requested; a FAILED
    verification refuses the whole walk (never a degraded anchor).
    """

    current = start_op
    hops: list[HopRecord] = []
    seen: set[str] = set()
    for _ in range(max_hops + 1):
        label = str(getattr(current, "label", "<unknown>"))
        if label in seen:
            return HopRefusal("walk revisited an op (cycle guard)", at_label=label)
        seen.add(label)
        if is_anchor(current):
            frozen = tuple(hops)
            index_map = _build_index_map(frozen, trace)
            if isinstance(index_map, HopRefusal):
                return index_map
            if structure_only:
                return HopWalk(current, str(start_op.label), frozen, "structure_only", index_map)
            verification = _verify_endpoints(trace, start_op, current, frozen)
            if isinstance(verification, HopRefusal):
                return verification
            return HopWalk(current, str(start_op.label), frozen, verification, index_map)
        proposal = _propose(current, grammar)
        if proposal is None:
            return HopRefusal(
                f"op {label!r} ({getattr(current, 'func_name', '?')}) is not a "
                "value-transporting hop under the grammar",
                at_label=label,
            )
        parent_label = _single_parent_label(current)
        if parent_label is None:
            return HopRefusal(f"op {label!r} has no single tensor parent", at_label=label)
        try:
            current = trace.ops[parent_label]
        except (KeyError, ValueError):
            return HopRefusal(f"parent op {parent_label!r} is unavailable", at_label=label)
        hops.append(proposal)
    return HopRefusal(
        f"no anchor within {max_hops} hops of {getattr(start_op, 'label', '<unknown>')!r}",
        at_label=str(getattr(current, "label", "<unknown>")),
    )
