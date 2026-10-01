"""The backward additive-tree walk (mikit D2/D3/D4 -- the decomposition engine).

The engine walks BACKWARD from the concrete captured tensor the final norm
consumes (resolved by dataflow, never forward enumeration, never an
architecture registry): each qualifying addition is a SPINE step whose other
operand is the WRITER that wrote into the stream; hops across
value-preserving ops are proposed by the shared op-kind grammar and verified
by payload identity (``torchlens.semantic._hops``); every step carries the
op's pass-qualified label and live site key (day-one identity -- a pass-blind
walk is silently half-wrong on a plain gpt2 forward, measured 13/25).

Membership rules, in order, at one addition ``s = x + y``:

1. An operand CONTINUES the spine iff its hop-normalized producer is itself
   a qualifying addition (same recorded full-hidden shape, two tensor
   parents).
2. If NEITHER qualifies, the addition is the ROOT: both operands are writers
   (token + position embeddings on GPT-2; the panel memo's "two additive
   leaves, one row on rotary models" rule falls out of the graph).
3. If BOTH qualify (a genuine additive join), the spine operand is the one
   whose sub-tree the OTHER operand's ancestry re-enters (a writer consumes
   the state it writes onto); if that test cannot decide, the walk refuses
   typed with the frontier op named -- never a guess.

The walk itself never reads payloads except through the hop rule's verifier;
value claims (writer tensors) are the caller's read, against the retention
refusal machinery.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from ..semantic._hops import VALUE_GRAMMAR, HopRefusal, HopWalk, walk_upstream
from ._errors import refuse

__all__ = ["SpineNode", "SpineWalk", "walk_spine"]

#: Addition spellings that merge two tensors into the stream.
_ADD_FUNCS = frozenset({"__add__", "__radd__", "__iadd__", "add", "add_"})

#: Walk ceiling: real LMs have < 200 spine steps; a runaway graph refuses.
_MAX_SPINE_STEPS = 4096


@dataclass(frozen=True)
class SpineNode:
    """One spine addition, bottom-up ordered fields.

    Parameters
    ----------
    op:
        The captured addition op record (its ``out`` is the spine state).
    writer_labels:
        Pass-qualified labels of the operand ops that WROTE into the stream
        at this step (two at the root, one everywhere else).
    spine_parent_label:
        Pass-qualified label of the upstream spine addition, ``None`` at the
        root.
    """

    op: Any
    writer_labels: tuple[str, ...]
    spine_parent_label: str | None


@dataclass(frozen=True)
class SpineWalk:
    """A completed spine walk, root-first.

    ``nodes`` are in EXECUTION order (root embedding merge first, final
    pre-norm addition last) -- the order whose accumulation replays the
    forward's own add sequence. ``hop_walks`` records every verified hop
    chain keyed by the hop START op label (disclosure, not authority).
    """

    nodes: tuple[SpineNode, ...]
    target_label: str
    hop_walks: dict[str, HopWalk]


def _tensor_parent_labels(op: Any) -> tuple[str, ...]:
    """Return the op's tensor-parent labels in operand order."""

    positions = getattr(op, "parent_arg_positions", None) or {}
    args = positions.get("args", {}) if isinstance(positions, dict) else {}
    if args:
        return tuple(str(args[key]) for key in sorted(args))
    return tuple(str(label) for label in (getattr(op, "parents", ()) or ()))


def _recorded_shape(op: Any) -> tuple[int, ...] | None:
    """Return an op's recorded output shape as an int tuple, or ``None``."""

    shape = getattr(op, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(dim) for dim in shape)
    except (TypeError, ValueError):
        return None


def _add_alpha(op: Any) -> float:
    """Return the recorded ``alpha`` coefficient of an add op (default 1)."""

    kwargs = getattr(op, "non_tensor_kwargs", None) or {}
    alpha = kwargs.get("alpha", 1) if isinstance(kwargs, dict) else 1
    try:
        return float(alpha)
    except (TypeError, ValueError):
        return 1.0


def _is_qualifying_add(op: Any, hidden_shape: tuple[int, ...]) -> bool:
    """Return whether an op is a spine-candidate addition.

    Qualification: an add-family func, exactly two tensor parents, the FULL
    recorded hidden shape (broadcast bias adds fail this), and a unit alpha
    (a scaled merge is a named frontier, not a silent hop).
    """

    if str(getattr(op, "func_name", "")) not in _ADD_FUNCS:
        return False
    if len(_tensor_parent_labels(op)) != 2:
        return False
    if _recorded_shape(op) != hidden_shape:
        return False
    return _add_alpha(op) == 1.0


def _normalize(
    trace: Any,
    op: Any,
    hidden_shape: tuple[int, ...],
    hop_walks: dict[str, HopWalk],
    *,
    structure_only: bool,
) -> Any:
    """Hop-normalize an operand to its deepest value-identical producer.

    Walks the VALUE grammar upstream until either a qualifying addition or
    the deepest value-preserving ancestor is reached. A refused walk (a
    grammar stop is not a refusal) normalizes to the op itself.
    """

    def _is_anchor(candidate: Any) -> bool:
        """Stop at a qualifying addition; the grammar stops everywhere else."""

        return _is_qualifying_add(candidate, hidden_shape)

    walk = walk_upstream(
        trace, op, _is_anchor, grammar=VALUE_GRAMMAR, structure_only=structure_only
    )
    if isinstance(walk, HopRefusal):
        return op
    if walk.hops:
        hop_walks[str(getattr(op, "label", "<unknown>"))] = walk
    return walk.anchor


def _reaches(trace: Any, from_label: str, target_label: str, *, limit: int = 20000) -> bool:
    """Return whether ``target_label`` is an ancestor of ``from_label``.

    Bounded breadth-first search over parent edges; the bound refuses rather
    than silently truncating (a wrong reachability answer would misassign
    spine membership).
    """

    seen: set[str] = set()
    frontier: deque[str] = deque([from_label])
    while frontier:
        if len(seen) > limit:
            refuse(
                code="mi_spine_ambiguous",
                message=f"Ancestry search exceeded {limit} ops while disambiguating a spine join.",
                remedy="decompose a smaller target region, or report the architecture "
                "(this is a certifiable-topology frontier)",
                from_label=from_label,
                target_label=target_label,
            )
        label = frontier.popleft()
        if label in seen:
            continue
        seen.add(label)
        if label == target_label:
            return True
        try:
            op = trace.ops[label]
        except (KeyError, ValueError):
            continue
        frontier.extend(str(parent) for parent in (getattr(op, "parents", ()) or ()))
    return False


def _choose_spine_operand(
    trace: Any,
    add_op: Any,
    operands: tuple[tuple[str, Any], ...],
    candidates: tuple[int, ...],
) -> int:
    """Return the operand index that continues the spine at a two-add join.

    The writer consumes the state it writes onto, so the spine operand is
    the one that is an ancestor of the OTHER operand. Undecidable joins
    refuse typed with the frontier named.
    """

    first, second = operands[0], operands[1]
    first_norm_label = str(getattr(first[1], "label", first[0]))
    second_norm_label = str(getattr(second[1], "label", second[0]))
    first_feeds_second = _reaches(trace, second[0], first_norm_label)
    second_feeds_first = _reaches(trace, first[0], second_norm_label)
    if first_feeds_second and not second_feeds_first:
        return 0
    if second_feeds_first and not first_feeds_second:
        return 1
    refuse(
        code="mi_spine_ambiguous",
        message=f"Both operands of {getattr(add_op, 'label', '<add>')!r} are additive and the "
        "ancestry test cannot decide which continues the residual stream.",
        remedy="this topology is outside the certifiable additive-residual frontier; "
        "report the architecture or decompose from an explicit target=",
        add_label=str(getattr(add_op, "label", "")),
        candidates=[operands[index][0] for index in candidates],
    )
    raise AssertionError("unreachable")


def walk_spine(
    trace: Any,
    target_op: Any,
    *,
    structure_only: bool = False,
) -> SpineWalk:
    """Walk the residual spine backward from ``target_op``.

    Parameters
    ----------
    trace:
        The captured trace.
    target_op:
        The op producing the tensor the final norm consumes (or any explicit
        downstream spine state).
    structure_only:
        Run the hop grammar without payload verification (structure surfaces
        only; tensor claims must not ride a structure-only walk).

    Returns
    -------
    SpineWalk
        Root-first spine nodes with writer labels and hop disclosures.
    """

    hop_walks: dict[str, HopWalk] = {}
    hidden_shape = _recorded_shape(target_op)
    if hidden_shape is None:
        refuse(
            code="mi_target_unresolvable",
            message=f"Target op {getattr(target_op, 'label', '<unknown>')!r} has no recorded shape.",
            remedy="pass a captured op with a recorded output shape as target=",
        )
    start = _normalize(trace, target_op, hidden_shape, hop_walks, structure_only=structure_only)
    if not _is_qualifying_add(start, hidden_shape):
        refuse(
            code="mi_target_unresolvable",
            message=f"No additive residual merge found at or upstream of "
            f"{getattr(target_op, 'label', '<unknown>')!r} "
            f"(deepest producer: {getattr(start, 'label', '<unknown>')!r}, "
            f"func {getattr(start, 'func_name', '?')!r}).",
            remedy="the target must sit on an additive residual stream; pass an explicit "
            "target= op on the stream, or report the architecture frontier",
            target_label=str(getattr(target_op, "label", "")),
        )

    nodes_bottom_up: list[SpineNode] = []
    current = start
    for _ in range(_MAX_SPINE_STEPS):
        operand_labels = _tensor_parent_labels(current)
        operands: list[tuple[str, Any]] = []
        for label in operand_labels:
            try:
                parent = trace.ops[label]
            except (KeyError, ValueError):
                refuse(
                    code="mi_target_unresolvable",
                    message=f"Operand {label!r} of {getattr(current, 'label', '?')!r} has no op record.",
                    remedy="capture with default options and retry",
                    operand=label,
                )
            normalized = _normalize(
                trace, parent, hidden_shape, hop_walks, structure_only=structure_only
            )
            operands.append((label, normalized))
        candidate_indices = tuple(
            index
            for index, (_, normalized) in enumerate(operands)
            if _is_qualifying_add(normalized, hidden_shape)
        )
        if not candidate_indices:
            nodes_bottom_up.append(
                SpineNode(
                    op=current,
                    writer_labels=tuple(label for label, _ in operands),
                    spine_parent_label=None,
                )
            )
            break
        if len(candidate_indices) == 1:
            spine_index = candidate_indices[0]
        else:
            spine_index = _choose_spine_operand(
                trace, current, (operands[0], operands[1]), candidate_indices
            )
        spine_label, spine_op = operands[spine_index]
        writer_label = operands[1 - spine_index][0]
        nodes_bottom_up.append(
            SpineNode(
                op=current,
                writer_labels=(writer_label,),
                spine_parent_label=str(getattr(spine_op, "label", spine_label)),
            )
        )
        current = spine_op
    else:
        refuse(
            code="mi_spine_open",
            message=f"The spine walk exceeded {_MAX_SPINE_STEPS} steps without reaching a root.",
            remedy="this graph is outside the certifiable additive-residual frontier",
        )

    return SpineWalk(
        nodes=tuple(reversed(nodes_bottom_up)),
        target_label=str(getattr(target_op, "label", "")),
        hop_walks=hop_walks,
    )
