"""Rank (stacking) channel: annotation -> same-graphviz-rank groups (L5 M3).

``draw(stack_by=...)`` pins nodes sharing an annotation value to one rank
(column under LR, row under TB), producing the classic unrolled-RNN
timestep diagram. General over ANY annotation (field or callable), not just
time; STRICTLY OPT-IN -- plain ``draw()`` never stacks.

AUTO RESOLUTION (``stack_by=True`` / ``"auto"``) is granted ONLY under THE
LOCKSTEP LICENSE (design memo 4.1): take all ops of multi-pass layers in
raw execution order; auto-annotation (``pass_index`` on multi-pass ops
only) activates iff their pass_index sequence is GLOBALLY NON-DECREASING.
When it holds, the trace's multi-pass execution factors into ordered
windows and "same column" asserts exactly "same execution window" -- which
is precisely what the classic figure claims, and all it claims (a layer
absent from window k simply has no node in that column). When it fails --
chained loops, late-resumption interior skips, nested loops whose flat
tally breaks monotonicity -- no derivable ground truth exists and the
request refuses typed (``stack_by_auto_underivable``); same refusal when
the trace has no multi-pass layers at all. Explicit field/callable sources
BYPASS the license entirely (the user asserts their own semantics; the
caption discloses what was used).

v1 has exactly ONE cohort (the license is global); cohort partitioning of
disjoint concurrent loops is a named follow-on, not scope.

NAMING: every spelling here (``stack_by``, the auto form, the two refusal
codes) is DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._encoding import (
    LAYER_SOURCE_ROWS,
    EncodingChannelSpec,
    EncodingState,
    _encoding_error,
    is_record_derived_image_node,
)

if TYPE_CHECKING:  # typing only
    from ..data_classes.trace import Trace

#: Legend/caption disclosure wording (pinned by tests).
STACK_AUTO_DISPLAY = "pass_index (auto)"
NOTE_STACK_CALLABLE = "stack values from user callable"


def _require_stack_spec(state: EncodingState) -> EncodingChannelSpec:
    """Return the active stack spec or fail on an internal phase-order bug."""

    if state.stack_spec is None:
        raise RuntimeError("stack resolution requires an active stack spec")
    return state.stack_spec


def resolve_stack_by(stack_by: Any, vis_mode: str) -> EncodingChannelSpec | None:
    """Validate ``stack_by`` at option validation, before any render work.

    Parameters
    ----------
    stack_by:
        ``None``, ``True``/``"auto"`` (licensed auto resolution), a
        field-name string, or a callable ``node -> hashable``.
    vis_mode:
        Requested render granularity; rolled graphs refuse (per-pass nodes
        do not exist to stack).

    Returns
    -------
    EncodingChannelSpec | None
        Resolved channel spec, or ``None`` when the channel is inactive.

    Raises
    ------
    InvalidArgumentError
        ``stack_by_requires_unrolled`` on rolled graphs;
        ``encoding_source_invalid`` for unknown sources.
    """

    if stack_by is None or stack_by is False:
        return None
    if vis_mode == "rolled":
        raise _encoding_error(
            "stack_by requires the unrolled graph: rolled mode has one "
            "aggregate node per layer, so there are no per-pass nodes to "
            "stack into columns",
            code="stack_by_requires_unrolled",
            remedy="pass vis_mode='unrolled' (the default), or drop stack_by",
            argument="stack_by",
        )
    if stack_by is True:
        return EncodingChannelSpec(
            channel="stack",
            source_kind="auto",
            source="auto",
            display_name=STACK_AUTO_DISPLAY,
        )
    if callable(stack_by) and not isinstance(stack_by, str):
        name = getattr(stack_by, "__name__", type(stack_by).__name__)
        return EncodingChannelSpec(
            channel="stack",
            source_kind="callable",
            source=stack_by,
            display_name=f"callable {name}",
        )
    if isinstance(stack_by, str):
        normalized = stack_by.strip().lower()
        if normalized == "auto":
            return EncodingChannelSpec(
                channel="stack",
                source_kind="auto",
                source="auto",
                display_name=STACK_AUTO_DISPLAY,
            )
        from ..constants import LAYER_PASS_LOG_FIELD_ORDER

        if stack_by in LAYER_SOURCE_ROWS or stack_by in LAYER_PASS_LOG_FIELD_ORDER:
            return EncodingChannelSpec(
                channel="stack",
                source_kind="field",
                source=stack_by,
                display_name=stack_by,
            )
        raise _encoding_error(
            f"stack_by source {stack_by!r} is not a known record field, True/'auto', or a callable",
            code="encoding_source_invalid",
            remedy=(
                "pass stack_by=True for the licensed auto annotation, a "
                "Layer/Op field name, or a callable node -> hashable"
            ),
            argument="stack_by",
        )
    raise _encoding_error(
        f"stack_by must be True/'auto', a field-name string, or a callable; "
        f"received {type(stack_by).__name__}",
        code="encoding_source_invalid",
        remedy="pass True, 'auto', a string field name, or a callable",
        argument="stack_by",
    )


def _multipass_ops_in_raw_order(trace: Trace) -> list[Any]:
    """Return all ops of multi-pass layers, ordered by raw execution index."""

    ops = [
        op for op in getattr(trace, "layer_list", ()) if int(getattr(op, "num_passes", 1) or 1) > 1
    ]
    ops.sort(key=lambda op: int(getattr(op, "raw_index", 0) or 0))
    return ops


def check_lockstep_license(trace: Trace) -> None:
    """Grant or refuse the auto-annotation license (memo 4.1).

    The predicate IS the case list: every globally monotone trace accepts
    (vanilla lockstep AND ragged monotone prefixes -- early exit,
    suffix-skip, monotone nested interleavings); every non-monotone trace
    refuses (chained/sequential loops, late-resumption interior skips,
    nested loops whose flat tally breaks monotonicity).

    Raises
    ------
    InvalidArgumentError
        ``stack_by_auto_underivable`` when no ground truth is derivable.
    """

    ops = _multipass_ops_in_raw_order(trace)
    if not ops:
        raise _encoding_error(
            "stack_by auto-annotation is underivable: the trace has no "
            "multi-pass layers to derive execution windows from",
            code="stack_by_auto_underivable",
            remedy=(
                "pass an explicit annotation (stack_by=<field> or a "
                "callable) naming your own stacking semantics"
            ),
            argument="stack_by",
        )
    previous = 0
    for op in ops:
        pass_index = int(getattr(op, "pass_index", 1) or 1)
        if pass_index < previous:
            raise _encoding_error(
                "stack_by auto-annotation is underivable: the multi-pass "
                "execution order is not globally monotone (pass "
                f"{pass_index} of {getattr(op, 'layer_label', op)!r} runs "
                f"after pass {previous}), so 'same column = same execution "
                "window' has no derivable ground truth (chained or "
                "non-monotone nested loops)",
                code="stack_by_auto_underivable",
                remedy=(
                    "pass an explicit annotation (stack_by=<field> or a "
                    "callable) naming your own stacking semantics"
                ),
                argument="stack_by",
            )
        previous = pass_index


def _stack_value_for_node(state: EncodingState, node: Any) -> Any:
    """Resolve one node's stack annotation value (None = un-annotated)."""

    spec = _require_stack_spec(state)
    if spec.source_kind == "auto":
        if int(getattr(node, "num_passes", 1) or 1) <= 1:
            # Single-pass stem/head ops stay un-annotated so they do not
            # all pin to column 1 (that is how classic diagrams draw stems).
            return None
        return int(getattr(node, "pass_index", 1) or 1)
    if spec.source_kind == "callable":
        try:
            value = spec.source(node)
        except Exception as error:
            raise _encoding_error(
                f"stack_by callable {spec.display_name!r} raised on node "
                f"{getattr(node, 'label', node)!r}: {error}",
                code="encoding_callable_error",
                remedy="fix the callable; return None to leave a node unstacked",
                argument="stack_by",
            ) from error
        state.stack_note(NOTE_STACK_CALLABLE)
        return value
    from ..utils._multipass_access import get_multipass_attr

    return get_multipass_attr(node, spec.source, None, multipass=None)


def compute_stack_groups(state: EncodingState, trace: Trace, universe: Any) -> None:
    """PHASE A: resolve rank groups over the visible-node universe.

    Groups with fewer than two members are dropped (a one-node rank
    constraint is a layout no-op). Group keys sort deterministically by
    repr; membership order follows emission order.
    """

    spec = state.stack_spec
    if spec is None:
        return
    if spec.source_kind == "auto":
        check_lockstep_license(trace)

    from ._render_common import _render_node_name

    groups: dict[Any, list[str]] = {}
    for unit in universe.units:
        emission = unit.emission
        if emission.kind != "raw_op" or emission.node is None:
            continue
        node = emission.node
        if is_record_derived_image_node(trace, node):
            continue
        value = _stack_value_for_node(state, node)
        if value is None:
            continue
        try:
            hash(value)
        except TypeError:
            raise _encoding_error(
                f"stack_by source {spec.display_name!r} produced an unhashable "
                f"{type(value).__name__!r} on node "
                f"{getattr(node, 'label', node)!r}; rank keys must be hashable",
                code="encoding_value_invalid",
                remedy="return a hashable annotation value (int, str, tuple)",
                argument="stack_by",
            ) from None
        node_name = _render_node_name(node, "unrolled")
        groups.setdefault(value, []).append(node_name)

    state.stack_groups = tuple(
        (repr(value), tuple(members))
        for value, members in sorted(groups.items(), key=lambda item: repr(item[0]))
        if len(members) >= 2
    )
