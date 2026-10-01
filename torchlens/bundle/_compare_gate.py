"""The Bundle comparison gate: two predicates, operand-scoped, topology-aware.

Split from ``torchlens/bundle/__init__.py`` under the R43 file-size ratchet
(A-GATE lane). The measured defect this module exists to kill (foldB D6): the
old ``_RELATIONSHIP_RANK`` lattice collapsed the model axis and the input axis
into one rank, so an identity relationship was accepted as proof of input
equality and the reachable input check hashed shape/dtype/device only — the
gate refused in NONE of four probe legs, on two independent code paths.

The gate now checks, per operand pair and in order: ORDERING TOPOLOGY
(comparison operands joined by a directed path of ordering relations refuse
regardless of rank), the MODEL-AXIS floor (a projection of the reported
``Relationship`` vocabulary onto model evidence only), and VALUE-LEVEL INPUT
IDENTITY (proven only by positive payload evidence; unproven refuses
fail-closed). The persisted digest carrier is the A-GATE/digest slice (after
C07-X); :func:`_input_value_digest` is that carrier's live derivation.
"""

from __future__ import annotations

import contextlib
import hashlib
import weakref
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from .._capture_state_helpers import _hash_input_tensor_value
from ..intervention.errors import BundleRelationshipError
from ..intervention.types import Relationship

if TYPE_CHECKING:
    from ..data_classes.trace import Trace
    from ._relations import MemberRelationTable

# MODEL-AXIS projection of the reported ``Relationship`` vocabulary (A-GATE,
# foldB D6): this table ranks the MODEL axis only; input identity is a
# separate, value-level predicate (``_input_value_digest``) that no identity
# relationship ever satisfies. ``shared_graph_same_input`` and
# ``shared_graph_diff_input`` carry identical model-axis evidence: their
# input-axis difference is deliberately invisible here.
_MODEL_AXIS_RANK: dict[Relationship, int] = {
    Relationship.UNKNOWN: 0,
    Relationship.DIFF_MODEL: 0,
    Relationship.SHARED_ARCHITECTURE: 1,
    Relationship.SAME_PARAM_SHAPES: 2,
    Relationship.SHARED_GRAPH_DIFFERENT_INPUT: 3,
    Relationship.SHARED_GRAPH_SAME_INPUT: 3,
    Relationship.SAME_MODEL_OBJECT_AT_CAPTURE: 4,
    Relationship.SAME_OBJECT: 5,
}


@dataclass(frozen=True)
class _GateRequirement:
    """Two-predicate requirement one gated Bundle read declares.

    Parameters
    ----------
    model_floor_rank:
        Minimum ``_MODEL_AXIS_RANK`` value every operand pair's derived
        relationship must reach.
    model_floor_label:
        Human-readable name of the floor for refusal messages.
    same_input:
        Whether the operand pair must additionally prove VALUE-LEVEL input
        identity. This predicate is independent of the relationship: an
        identity relationship (``same_object`` / ``same_model_at_capture``)
        is never accepted as proof of input equality (foldB D6 path 1), and
        the shape-level ``input_signature_hash`` is never consulted
        (foldB D6 path 2 — it is value-blind).
    """

    model_floor_rank: int
    model_floor_label: str
    same_input: bool


_GATE_REQUIREMENTS: dict[str, _GateRequirement] = {
    "node": _GateRequirement(
        model_floor_rank=_MODEL_AXIS_RANK[Relationship.SAME_PARAM_SHAPES],
        model_floor_label="same_param_shapes",
        same_input=False,
    ),
    "compare_at": _GateRequirement(
        model_floor_rank=_MODEL_AXIS_RANK[Relationship.SHARED_GRAPH_SAME_INPUT],
        model_floor_label="shared_graph",
        same_input=True,
    ),
    "most_changed": _GateRequirement(
        model_floor_rank=_MODEL_AXIS_RANK[Relationship.SHARED_GRAPH_SAME_INPUT],
        model_floor_label="shared_graph",
        same_input=True,
    ),
    "diff": _GateRequirement(
        model_floor_rank=_MODEL_AXIS_RANK[Relationship.SHARED_GRAPH_SAME_INPUT],
        model_floor_label="shared_graph",
        same_input=True,
    ),
}

#: S6 PAIR-row kinds that order members in time/causality. Members joined by
#: a directed path of these rows are NOT comparison operands (A-GATE item 4:
#: the refusal keys on topology, never on relationship rank). ``alternative_of``
#: is deliberately absent — counterfactual peers stay comparable.
_ORDERING_RELATION_KINDS: frozenset[str] = frozenset({"successor_of", "forked_from", "escalates"})

#: Session cache for the live-derived input value digests, keyed weakly by
#: member Trace (lifetime-safe against id reuse) with the trace's
#: ``_spec_revision`` folded in so a fork edit invalidates the entry.
_INPUT_VALUE_DIGEST_CACHE: weakref.WeakKeyDictionary[Any, tuple[Any, str | None]] = (
    weakref.WeakKeyDictionary()
)


def _input_value_digest(trace: Trace) -> str | None:
    """Return the live-derived value-level input digest for one member.

    The digest folds every input op's retained payload through
    :func:`~torchlens._capture_state_helpers._hash_input_tensor_value`
    (shape + logical dtype + bytes; device- and grad-flag-free). ``None``
    means UNPROVEN — an input payload is unsaved, non-tensor, unreadable,
    or the trace records no input ops — and the comparison gate refuses
    fail-closed rather than guessing. The persisted carrier that survives
    payload-free artifacts is the A-GATE/digest slice (after C07-X); this
    live derivation is the same digest law.

    Parameters
    ----------
    trace:
        Bundle member to digest.

    Returns
    -------
    str | None
        SHA-256 hex digest, or ``None`` when input identity is unprovable.
    """

    revision = getattr(trace, "_spec_revision", None)
    try:
        cached = _INPUT_VALUE_DIGEST_CACHE.get(trace)
    except TypeError:
        cached = None
    if cached is not None and cached[0] == revision:
        return cached[1]
    digest = _derive_input_value_digest(trace)
    with contextlib.suppress(TypeError):
        _INPUT_VALUE_DIGEST_CACHE[trace] = (revision, digest)
    return digest


def _derive_input_value_digest(trace: Trace) -> str | None:
    """Derive the input value digest from retained input-op payloads.

    Returns
    -------
    str | None
        SHA-256 hex digest over the ordered input payloads, or ``None``
        (unproven) when any input payload is unavailable. Every failure mode
        maps to ``None`` deliberately: the caller's refusal is the disclosed
        handling, never a silent pass.
    """

    try:
        input_ops = list(trace.input_ops)
    except Exception:  # noqa: BLE001 - unprovable evidence refuses at the gate
        return None
    if not input_ops:
        return None
    hasher = hashlib.sha256()
    hasher.update(f"input_value_digest_v1:{len(input_ops)}".encode())
    for op in input_ops:
        if not getattr(op, "has_saved_activation", False):
            return None
        try:
            payload = op.out
        except Exception:  # noqa: BLE001 - unprovable evidence refuses at the gate
            return None
        if not isinstance(payload, torch.Tensor):
            return None
        try:
            hasher.update(_hash_input_tensor_value(payload).encode("utf-8"))
        except Exception:  # noqa: BLE001 - unprovable evidence refuses at the gate
            return None
    return hasher.hexdigest()


def ordering_path(relations: MemberRelationTable, start: str, goal: str) -> tuple[str, ...] | None:
    """Return the ordering-relation kinds along a directed path, if any.

    Walks only :data:`_ORDERING_RELATION_KINDS` rows of the S6 member
    relation table, from each row's ``from_member`` to its ``to_member``.
    Sibling members (two rows pointing at one common predecessor) are NOT
    joined: only a directed path orders two members.

    Parameters
    ----------
    relations:
        The Bundle's S6 member-relation table.
    start:
        Candidate origin member name.
    goal:
        Candidate destination member name.

    Returns
    -------
    tuple[str, ...] | None
        Row kinds along the first directed path found, or ``None``.
    """

    adjacency: dict[str, list[tuple[str, str]]] = {}
    for row in relations:
        if row.kind in _ORDERING_RELATION_KINDS:
            adjacency.setdefault(str(row.from_member), []).append((str(row.to_member), row.kind))
    if not adjacency:
        return None
    frontier: list[tuple[str, tuple[str, ...]]] = [(start, ())]
    seen = {start}
    while frontier:
        current, kinds = frontier.pop(0)
        for neighbor, kind in adjacency.get(current, []):
            if neighbor == goal:
                return (*kinds, kind)
            if neighbor not in seen:
                seen.add(neighbor)
                frontier.append((neighbor, (*kinds, kind)))
    return None


def require_comparable(
    members: Mapping[str, Trace],
    relations: MemberRelationTable,
    operation: str,
    pairs: Sequence[tuple[str, str]] | None,
    relationship_of: Callable[[Trace, Trace], Relationship],
) -> None:
    """Run the two-predicate comparison gate over the operand pairs.

    Per pair, three ordered checks (A-GATE, foldB D6/D18):

    1. TOPOLOGY (comparison-grade operations only): operands joined by a
       directed path of ordering relations (``successor_of`` /
       ``forked_from`` / ``escalates``) are not comparison operands,
       regardless of relationship rank.
    2. MODEL FLOOR: the derived relationship's model-axis rank must reach
       the operation's floor.
    3. INPUT IDENTITY (comparison-grade operations only): value-level input
       equality proven from retained input payloads. An identity
       relationship never satisfies this predicate, and absent evidence
       refuses fail-closed (unproven is not proven-same).

    Parameters
    ----------
    members:
        The Bundle's ordered member mapping.
    relations:
        The Bundle's S6 member-relation table.
    operation:
        Gated operation name (a ``_GATE_REQUIREMENTS`` key).
    pairs:
        Operand pairs the operation actually reads. ``None`` means every
        i<j member pair (operations whose operand set is the whole bundle).
        Scoping the gate to the true operand pair is what keeps one foreign
        member from disabling every gated read in a chain.
    relationship_of:
        The Bundle's relationship derivation for one member pair.

    Raises
    ------
    BundleRelationshipError
        With ``fields["code"]`` set to ``bundle_gate_ordering_topology``,
        ``bundle_gate_model_axis_unmet``,
        ``bundle_gate_input_values_differ``, or
        ``bundle_gate_input_identity_unproven``.
    """

    requirement = _GATE_REQUIREMENTS[operation]
    if pairs is None:
        names = list(members)
        pairs = [
            (left_name, right_name)
            for index, left_name in enumerate(names)
            for right_name in names[index + 1 :]
        ]
    for left_name, right_name in pairs:
        left = members[left_name]
        right = members[right_name]
        if requirement.same_input:
            path = ordering_path(relations, left_name, right_name) or ordering_path(
                relations, right_name, left_name
            )
            if path is not None:
                raise BundleRelationshipError(
                    f"Bundle operation {operation!r} refuses: members "
                    f"{left_name!r} and {right_name!r} are joined by an "
                    f"ordering relation ({' -> '.join(path)}). Members "
                    "joined by an ordering relation are not comparison "
                    "operands — the refusal keys on topology, not on "
                    "relationship rank. Remedy: compare peer members "
                    "(for example alternative_of counterfactuals), or "
                    "bundle the captures you want to compare without an "
                    "ordering row between them.",
                    code="bundle_gate_ordering_topology",
                    operation=operation,
                    left_member=left_name,
                    right_member=right_name,
                    ordering_path=list(path),
                )
        relationship = relationship_of(left, right)
        if _MODEL_AXIS_RANK[relationship] < requirement.model_floor_rank:
            raise BundleRelationshipError(
                f"Bundle operation {operation!r} requires model-axis "
                f"evidence of at least {requirement.model_floor_label!r}; "
                f"member pair {left_name}/{right_name} derives "
                f"{relationship.value!r}. Remedy: bundle captures whose "
                "models share the required structure (see "
                "Bundle.relationship), or use per-member reads such as "
                "Bundle.apply for cross-model analysis.",
                code="bundle_gate_model_axis_unmet",
                operation=operation,
                left_member=left_name,
                right_member=right_name,
                relationship=relationship.value,
                required_floor=requirement.model_floor_label,
            )
        if requirement.same_input and left is not right:
            left_digest = _input_value_digest(left)
            right_digest = _input_value_digest(right)
            if left_digest is not None and right_digest is not None:
                if left_digest != right_digest:
                    raise BundleRelationshipError(
                        f"Bundle operation {operation!r} requires "
                        "value-level input identity; member pair "
                        f"{left_name}/{right_name} captured DIFFERENT "
                        "input values (retained input payload digests "
                        "disagree). An identity or shared-graph "
                        "relationship is never accepted as proof of "
                        "input equality. Remedy: compare captures of "
                        "the same input, or use per-member reads such "
                        "as Bundle.apply for cross-input analysis.",
                        code="bundle_gate_input_values_differ",
                        operation=operation,
                        left_member=left_name,
                        right_member=right_name,
                    )
            else:
                unproven = [
                    name
                    for name, digest in (
                        (left_name, left_digest),
                        (right_name, right_digest),
                    )
                    if digest is None
                ]
                raise BundleRelationshipError(
                    f"Bundle operation {operation!r} requires value-level "
                    f"input identity, and member(s) {unproven!r} retain "
                    "no input payloads to prove it (inputs not saved, or "
                    "a loaded artifact without input activations). "
                    "Refusing unproven input identity is deliberate: an "
                    "identity relationship or a shape-level input hash is "
                    "never accepted as proof of input equality. Remedy: "
                    "capture with input payloads retained (the default "
                    "save policy) or save the artifact with activations "
                    "included; a persisted input digest lands with the "
                    "input_digest carrier.",
                    code="bundle_gate_input_identity_unproven",
                    operation=operation,
                    left_member=left_name,
                    right_member=right_name,
                    unproven_members=unproven,
                )
