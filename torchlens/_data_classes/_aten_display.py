"""ATen cards over recorded tensor metadata, never retained tensor values."""

from __future__ import annotations

from typing import Any

from .._utils._display_card import inline_names, one_line, render_card
from .._utils._display_keys import display_key
from .._utils._display_numbers import format_exact_count, plural
from .._utils._display_text import tensor_text
from .._utils.fail_open import fail_open


def _fact_text(fact: Any, *, device: bool = True) -> str:
    """Render one tensor fact as dtype, shape and, unless ``device`` is false, device text."""

    return tensor_text(fact.dtype, fact.shape, fact.device if device else None)


def _owners(row: Any) -> str:
    """Name the ops owning this ATen row: display keys, else the first parent row, else status."""

    keys = getattr(row, "_display_owner_keys", ())
    if keys:
        return "in " + inline_names(keys)
    if row.parent_op_refs:
        return "in operation " + str(row.parent_op_refs[0].op_row_index)
    return row.owner_status


def _tensor_flow(row: Any) -> str:
    """Summarize tensor flow: the two shapes for a one-in, one-out call, otherwise the counts."""

    inputs, outputs = row.input_tensor_facts, row.output_tensor_facts
    if len(inputs) == len(outputs) == 1:
        return f"{_fact_text(inputs[0], device=False)} to {_fact_text(outputs[0])}"
    return f"{plural(len(inputs), 'tensor input')}, {plural(len(outputs), 'tensor output')}"


def aten_repr(row: Any) -> str:
    """One-line ATen repr from metadata only; a fixed text replaces it if a read fails."""

    return fail_open(lambda: _aten_identity(row), lambda _error: "AtenOp: metadata unavailable")


def _aten_identity(row: Any) -> str:
    """Build the ATen repr line: operator, overload, any non-returned outcome, optional facts."""

    operator = f"{row.namespace}::{row.operator}"
    if row.overload not in ("", "default"):
        operator += f".{row.overload}"
    facts = [operator]
    if row.outcome != "returned":
        facts.append(f"({row.outcome})")
    # Keep operator identity and outcome; longer tensor flow remains in the card.
    optional = [_tensor_flow(row), _owners(row)]
    if row.view_copy_kind == "view":
        optional.append("view")
    return one_line(f"AtenOp {row.label}", facts, optional)


def _fact_rows(facts: tuple[Any, ...]) -> str:
    """Render up to three tensor facts with stride and grad flag, then the exact remainder."""

    lines = []
    for fact in facts[:3]:
        grad = ", requires grad" if fact.requires_grad else ""
        lines.append(f"{_fact_text(fact)}, stride {fact.stride}{grad}")
    if len(facts) > len(lines):
        lines.append(f"and {len(facts) - len(lines)} more tensors")
    return "; ".join(lines) or "no tensor arguments"


def aten_str(row: Any) -> str:
    """Multi-line ATen card; the one-line repr replaces it if a read fails."""

    return fail_open(lambda: _aten_card(row), lambda _error: aten_repr(row))


def _aten_card(row: Any) -> str:
    """Build the ATen card: schema, sequence, inputs, outputs, FLOPs and module rows."""

    phase = (
        f"{row.capture_phase} pass {row.forward_pass_index}"
        if row.capture_phase == "forward"
        else f"backward epoch {row.backward_epoch_index}"
    )
    module = row.module_call_stack[-1][0] if row.module_call_stack else "self"
    flops = (
        format_exact_count(row.algorithmic_flops)
        if row.algorithmic_flops is not None
        else row.flop_status
    )
    rows = [
        ("schema", row.schema or "not recorded"),
        ("sequence", f"{row.sequence}, slot {row.decomposition_slot}, {_owners(row)}, {phase}"),
        ("inputs", _fact_rows(row.input_tensor_facts)),
        ("outputs", f"{_fact_rows(row.output_tensor_facts)}, {row.view_copy_kind}"),
        ("flops", flops),
        ("module", module),
    ]
    return render_card(
        aten_repr(row),
        rows,
        more=(".input_tensor_facts", ".output_tensor_facts", ".execution_context"),
    )


def bind_owner_keys(trace: Any) -> None:
    """Install display-only keys at capture/restore, outside persisted row fields."""

    profile = trace.__dict__.get("_primitive_op_profile")
    if profile is None:
        return
    ops = trace._layer_list
    for row in profile.primitive_ops:
        row._display_owner_keys = tuple(
            display_key(ops[ref.op_row_index])
            for ref in row.parent_op_refs
            if 0 <= ref.op_row_index < len(ops)
        )
