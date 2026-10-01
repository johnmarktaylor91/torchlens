"""Residual accumulation + decomposition (mikit item 8; TLens
``accumulated_resid`` / ``decompose_resid``, exact where theirs is
conventional).

``residual_accumulation`` returns the ACTUAL captured running residual
states, in execution order -- reads from capture, never recomputes.
``residual_decomposition`` returns every literal upstream writer once, with
coefficients, graded per mikit D7, and (strict mode, the default) enforces
the execution-order BITWISE identity: accumulating the literal captured
writers in execution order replays the forward's own fp32 add sequence, so
all partial sums equal the captured spine states under ``torch.equal``
(measured on five model/gauge/implementation combinations; per-dtype degrade
clause D5 applies off fp32/CPU).

Spellings DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

from typing import Any

import torch

from ..semantic._hops import VALUE_GRAMMAR, HopRefusal, walk_upstream
from ._errors import refuse
from ._records import ComponentRow, ComponentStack, Coordinate, StackGrading
from ._walk import SpineNode, SpineWalk, walk_spine

__all__ = ["residual_accumulation", "residual_decomposition"]


def _module_stack(op: Any) -> tuple[str, ...]:
    """Return the op's containing-module addresses, outermost first."""

    stack = []
    for entry in getattr(op, "modules", ()) or ():
        base, sep, tail = str(entry).rpartition(":")
        stack.append(base if sep and tail.isdigit() else str(entry))
    return tuple(stack)


def _innermost_module(op: Any) -> str | None:
    """Return the op's innermost containing module address, if any."""

    stack = _module_stack(op)
    return stack[-1] if stack else None


def _writer_kind(trace: Any, op: Any) -> str:
    """Classify a writer for display from its module chain (metadata only)."""

    for address in reversed(_module_stack(op)):
        try:
            record = trace.modules[address]
        except (KeyError, ValueError):
            continue
        class_name = str(getattr(record, "class_name", "")).lower()
        if "embedding" in class_name:
            return "embedding"
        if "attention" in class_name or "attn" in class_name:
            return "attention"
        if "mlp" in class_name or "feedforward" in class_name or "ffn" in class_name:
            return "mlp"
    return "writer"


def _deepest_value_ancestor(trace: Any, op: Any) -> tuple[Any, str]:
    """Return the deepest value-identical ancestor and its verification.

    Implements the mikit D6 LABEL rule: the writer's label comes from the
    innermost containing module of the deepest ancestor the value grammar
    can reach (post-dropout writers honestly label the dropout in train
    mode, the projection in eval mode).
    """

    def _grammar_stop(candidate: Any) -> bool:
        """Anchor at the first op the grammar does not propose across."""

        from ..semantic import _hops

        return _hops._propose(candidate, VALUE_GRAMMAR) is None

    walk = walk_upstream(trace, op, _grammar_stop, grammar=VALUE_GRAMMAR)
    if isinstance(walk, HopRefusal):
        return op, "walk_refused"
    return walk.anchor, walk.verification


def _resolve_target_op(trace: Any, target: Any) -> Any:
    """Resolve the decomposition target to a captured op record.

    ``"final"`` anchors the tensor the final norm consumes by dataflow (the
    lm-head recipe's own walk); an op label string resolves through the
    trace; an op record passes through.
    """

    if target is None or target == "final":
        from ._anchors import resolve_lm_head

        return resolve_lm_head(trace).target_op
    if isinstance(target, str):
        try:
            return trace.ops[target]
        except (KeyError, ValueError):
            refuse(
                code="mi_target_unresolvable",
                message=f"target= op {target!r} is not in the trace.",
                remedy="pass a pass-qualified captured op label",
                target=target,
            )
    return target


def _saved_out(op: Any) -> torch.Tensor | None:
    """Return an op's saved output payload, or ``None`` when unsaved."""

    out = getattr(op, "out", None)
    return out if isinstance(out, torch.Tensor) else None


def _require_payloads(trace: Any, labels: list[str], *, analysis: str) -> dict[str, torch.Tensor]:
    """Return label -> payload, refusing with the COMPLETE missing-site set."""

    missing: list[dict[str, Any]] = []
    payloads: dict[str, torch.Tensor] = {}
    for label in labels:
        try:
            op = trace.ops[label]
        except (KeyError, ValueError):
            missing.append({"op_label": label, "site_key": None})
            continue
        value = _saved_out(op)
        if value is None:
            missing.append({"op_label": label, "site_key": getattr(op, "site_key", None)})
        else:
            payloads[str(getattr(op, "label", label))] = value
    if missing:
        refuse(
            code="mi_payload_missing",
            message=f"{analysis} needs {len(missing)} unsaved op payload(s): "
            f"{sorted(str(row['op_label']) for row in missing)}.",
            remedy="recapture with tl.mechinterp.retention_plan(trace, analyses=...).predicate "
            "as save= (or layers_to_save='all'); never a partial table",
            missing_sites=missing,
            analysis=analysis,
        )
    return payloads


def _state_rows(trace: Any, walk: SpineWalk, include_mid: bool) -> list[tuple[SpineNode, str]]:
    """Return (node, state_label) pairs, filtered by ``include_mid``.

    A MID state is a spine addition whose innermost containing module also
    contains a LATER spine addition (two merges inside one block: the
    attention add and the block-output add). ``include_mid=False`` keeps only
    each module's last merge -- and never fabricates a mid for parallel
    blocks, because filtering is derived from the captured graph.
    """

    rows = [(node, str(getattr(node.op, "label", ""))) for node in walk.nodes]
    if include_mid:
        return rows
    keep: list[tuple[SpineNode, str]] = []
    for index, (node, label) in enumerate(rows):
        module = _innermost_module(node.op)
        later = (
            _innermost_module(other.op) == module and module is not None
            for other, _ in rows[index + 1 :]
        )
        if module is None or not any(later):
            keep.append((node, label))
    return keep


def _positions_view(value: torch.Tensor, positions: Any) -> torch.Tensor:
    """Slice a row value on the position axis (axis 1), when requested."""

    if positions is None:
        return value
    index = torch.as_tensor(positions, dtype=torch.long, device=value.device)
    return value.index_select(1, index)


def residual_accumulation(
    trace: Any,
    target: Any = "final",
    *,
    include_mid: bool = True,
    positions: Any = None,
) -> ComponentStack:
    """Return the captured running residual states, in execution order.

    Every row is a READ of a captured spine state (embedding sum, each
    proven post-write state, the final pre-norm state) -- never a recompute.
    TLens ``accumulated_resid`` equivalent, exact by construction.

    Parameters
    ----------
    trace:
        A finished torchlens trace of the model forward.
    target:
        ``"final"`` (the tensor the final norm consumes, anchored by
        dataflow), an op label, or an op record.
    include_mid:
        Include intermediate merges (the attention add inside a block).
        Real merges only; nothing is fabricated for parallel blocks.
    positions:
        Optional position indices; rows are views sliced on axis 1.
    """

    target_op = _resolve_target_op(trace, target)
    walk = walk_spine(trace, target_op)
    state_rows = _state_rows(trace, walk, include_mid)
    payloads = _require_payloads(
        trace, [label for _, label in state_rows], analysis="residual_accumulation"
    )
    rows = []
    for node, label in state_rows:
        op = node.op
        value = payloads[label]
        rows.append(
            ComponentRow(
                coordinate=Coordinate(
                    label=_innermost_module(op) or label,
                    op_label=label,
                    site_key=getattr(op, "site_key", None),
                    pass_index=getattr(op, "pass_index", None),
                    kind="residual",
                    module_address=_innermost_module(op),
                    provenance="captured",
                ),
                value=_positions_view(value, positions),
            )
        )
    target_value = _saved_out(target_op)
    return ComponentStack(
        tuple(rows),
        grading="complete",
        target_coordinate=Coordinate(
            label=str(getattr(target_op, "label", "target")),
            op_label=str(getattr(target_op, "label", "")),
            site_key=getattr(target_op, "site_key", None),
            pass_index=getattr(target_op, "pass_index", None),
            kind="residual",
        ),
        target_value=_positions_view(target_value, positions) if target_value is not None else None,
        identity_receipt={
            "check": "captured_reads",
            "result": "read_not_recomputed",
            "n_states": len(rows),
        },
    )


def _writer_row(
    trace: Any,
    node: SpineNode,
    writer_label: str,
    positions: Any,
    payloads: dict[str, torch.Tensor],
) -> ComponentRow:
    """Build one writer row: literal operand value, D6 module label."""

    writer_op = trace.ops[writer_label]
    value = payloads[str(getattr(writer_op, "label", writer_label))]
    ancestor, verification = _deepest_value_ancestor(trace, writer_op)
    label = _innermost_module(ancestor) or _innermost_module(writer_op)
    if label is None:
        label = f"<top-level {getattr(ancestor, 'func_name', 'op')}>"
    return ComponentRow(
        coordinate=Coordinate(
            label=label,
            op_label=str(getattr(writer_op, "label", writer_label)),
            site_key=getattr(writer_op, "site_key", None),
            pass_index=getattr(writer_op, "pass_index", None),
            kind=_writer_kind(trace, ancestor),
            module_address=_innermost_module(ancestor),
            provenance=f"captured/{verification}",
        ),
        value=_positions_view(value, positions),
    )


def _assert_bitwise_identity(
    trace: Any, walk: SpineWalk, strict: bool, payloads: dict[str, torch.Tensor]
) -> dict[str, Any]:
    """Run the execution-order equality gate (mikit D5).

    Accumulates the literal writer operands in execution order and asserts
    every partial sum ``torch.equal`` to its captured spine state. Returns
    the identity receipt; a strict-mode mismatch refuses OPEN.
    """

    total: torch.Tensor | None = None
    checked = 0
    for node in walk.nodes:
        for writer_label in node.writer_labels:
            writer_value = payloads[str(trace.ops[writer_label].label)]
            total = writer_value if total is None else total + writer_value
        state = payloads[str(getattr(node.op, "label", ""))]
        if total is None:
            continue
        if torch.equal(total, state):
            checked += 1
            total = state  # continue from the captured state (identical bytes)
            continue
        if strict:
            residual = float((total.float() - state.float()).abs().max())
            refuse(
                code="mi_spine_open",
                message=f"Execution-order accumulation does not reproduce the captured spine "
                f"state {getattr(node.op, 'label', '?')!r} bitwise "
                f"(max |residual| {residual:.3e}). The stack is OPEN.",
                remedy="this capture's residual topology is outside the certifiable frontier "
                "(or ran off fp32/CPU, where the per-dtype degrade clause applies); "
                "pass strict=False for a diagnostic-only stack with an "
                "unresolved_remainder row",
                state_label=str(getattr(node.op, "label", "")),
                max_abs_residual=residual,
                states_verified=checked,
            )
        return {
            "check": "execution_order_torch_equal",
            "result": "failed",
            "states_verified": checked,
            "failed_at": str(getattr(node.op, "label", "")),
        }
    return {
        "check": "execution_order_torch_equal",
        "result": "bitwise_equal",
        "states_verified": checked,
    }


def residual_decomposition(
    trace: Any,
    target: Any = "final",
    *,
    strict: bool = True,
    positions: Any = None,
) -> ComponentStack:
    """Decompose a residual state into every literal upstream writer, once.

    TLens ``decompose_resid`` equivalent, exact where theirs is
    conventional: writer VALUES are the literal operands that entered each
    spine addition (post-dropout in train mode -- and the identity still
    closes, which TLens's train-mode ``decompose_resid`` cannot claim);
    writer LABELS are module addresses per mikit D6, never op names.

    Parameters
    ----------
    trace:
        A finished torchlens trace.
    target:
        ``"final"``, an op label, or an op record (see
        :func:`residual_accumulation`).
    strict:
        Enforce the execution-order bitwise identity (default). A failing
        strict walk refuses OPEN; ``strict=False`` instead appends an
        ``unresolved_remainder`` row, stamps the stack ``diagnostic_only``,
        and is rejected by DLA, gallery, and launch claims (mikit D7).
    positions:
        Optional position indices; row VIEWS are sliced on axis 1 after the
        identity gate runs on the full tensors.
    """

    target_op = _resolve_target_op(trace, target)
    walk = walk_spine(trace, target_op)
    writer_labels = [label for node in walk.nodes for label in node.writer_labels]
    state_labels = [str(getattr(node.op, "label", "")) for node in walk.nodes]
    payloads = _require_payloads(
        trace, writer_labels + state_labels, analysis="residual_decomposition"
    )
    receipt = _assert_bitwise_identity(trace, walk, strict, payloads)

    rows = [
        _writer_row(trace, node, writer_label, positions, payloads)
        for node in walk.nodes
        for writer_label in node.writer_labels
    ]
    target_value = _saved_out(target_op)
    grading: StackGrading = "complete"
    diagnostic_only = False
    if receipt["result"] != "bitwise_equal":
        final_state = payloads[str(getattr(walk.nodes[-1].op, "label", ""))]
        accumulated = torch.zeros_like(final_state)
        for label in writer_labels:
            accumulated = accumulated + payloads[str(trace.ops[label].label)]
        remainder = final_state - accumulated
        rows.append(
            ComponentRow(
                coordinate=Coordinate(
                    label="unresolved_remainder",
                    kind="unresolved_remainder",
                    provenance="computed",
                ),
                value=_positions_view(remainder, positions),
            )
        )
        grading = "closed_unresolved"
        diagnostic_only = True

    return ComponentStack(
        tuple(rows),
        grading=grading,
        target_coordinate=Coordinate(
            label=str(getattr(target_op, "label", "target")),
            op_label=str(getattr(target_op, "label", "")),
            site_key=getattr(target_op, "site_key", None),
            pass_index=getattr(target_op, "pass_index", None),
            kind="residual",
        ),
        target_value=_positions_view(target_value, positions) if target_value is not None else None,
        identity_receipt=receipt,
        diagnostic_only=diagnostic_only,
    )
