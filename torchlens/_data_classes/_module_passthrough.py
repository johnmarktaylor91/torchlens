"""Single-output reads of a module call that also returned values it did not compute.

A preview backend records a module call's returned leaves at module exit
(``torchlens.backends._module_returns``). A leaf produced before the call began (an
identity module, a dropout layer in eval mode) is flagged as an output of the call on its
producing op, but it stays out of ``ModuleCall.output_ops``, which holds only ops inside the
call. A single-output read (``ModuleCall.out``, ``Module.out``) must then say so instead of
reporting "0 outputs", and must refuse when the call returned such a value beside its own
output. Torch inserts an identity op inside the call instead, so its calls never reach the
passthrough branch.
"""

from __future__ import annotations

from typing import Any


def passthrough_op_labels(call: Any, trace: Any) -> list[str]:
    """Return the ops a module call returned unchanged from outside its own ops.

    Parameters
    ----------
    call:
        The module call.
    trace:
        Its live trace, or ``None``.

    Returns
    -------
    list[str]
        Op labels in execution order; empty when the call returned none. Torch's
        trace-output pseudo-op copies its producer's module calls without the
        ``is_module_output`` flag, so it never counts.
    """

    if trace is None:
        return []
    inside = set(call.ops)
    return [
        op.label
        for op in trace._layer_list
        if op._module_context_is_module_output
        and call.label in (op._module_context_output_of_module_calls or ())
        and op.label not in inside
    ]


def single_output_refusal(call: Any, trace: Any, count: int) -> str | None:
    """Return why a single-output read of a module call refuses, or ``None``.

    Parameters
    ----------
    call:
        The module call being read.
    trace:
        Its live trace, or ``None``.
    count:
        Number of output ops the read found.

    Returns
    -------
    str | None
        The refusal message, or ``None`` when the call has exactly one output.
    """

    # Returned leaves outside the call only exist when the call returned more leaves
    # than it has output ops (duplicates do the same), so the op scan is skipped otherwise.
    passthrough = (
        passthrough_op_labels(call, trace)
        if count == 0 or len(call.output_paths or ()) > count
        else []
    )
    if count == 1 and not passthrough:
        return None
    if passthrough:
        names = ", ".join(repr(label) for label in passthrough)
        own = f"beside {count} output ops of its own" if count else "and computes no output"
        return (
            f"ModuleCall '{call.label}' returns its input unchanged ({names}) {own}; "
            f"read the returned input with trace.ops[{passthrough[0]!r}]."
        )
    return f"ModuleCall '{call.label}' has {count} outputs; use .outs[i] or resolve .output_ops[i]."
