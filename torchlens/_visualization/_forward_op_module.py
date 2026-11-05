"""Module-call keys for the forward ops that backward renders attach to."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._utils._multipass_access import get_multipass_attr, is_multipass_layer
from .._utils._record_group_table import record_member

if TYPE_CHECKING:
    from .._data_classes.layer import Layer


def _forward_op_is_module_output(op: Layer) -> bool:
    """Resolve ``is_module_output`` for a forward op, aggregate-safe on recurrent Layers.

    ``is_module_output`` is a per-pass field whose access raises the multi-pass
    ``ValueError`` tripwire on a recurrent aggregate ``Layer`` (this is what
    detonated ``draw_combined`` on any recurrent model). Module-output status is a
    static containment property, so -- matching how the sibling module fields
    ``output_of_modules`` / ``modules`` are already stored aggregate-as-first-pass
    on the Layer -- resolve it explicitly from the first captured pass instead of
    leaking the tripwire out of the public combined renderer.

    Parameters
    ----------
    op:
        Forward ``Op`` or aggregate ``Layer`` paired with a grad_fn.

    Returns
    -------
    bool
        Whether the forward op is a module output.
    """

    if is_multipass_layer(op):
        ops = getattr(op, "ops", None)
        if ops is not None:
            first_pass = next(iter(ops.values()), None)
            if first_pass is not None:
                return bool(record_member(first_pass, "is_module_output", False))
        return False
    return bool(get_multipass_attr(op, "is_module_output", False, multipass=False))


def _module_key_for_forward_op(op: Layer) -> str | None:
    """Return the unrolled module cluster key for a forward op.

    Parameters
    ----------
    op:
        Forward operation or layer log associated with a grad_fn_handle.

    Returns
    -------
    str | None
        Module-call key or None for top-level ops.
    """

    output_modules = list(record_member(op, "output_of_modules", []) or [])
    if _forward_op_is_module_output(op) and output_modules:
        output_module = str(output_modules[0])
        output_calls = list(record_member(op, "output_of_module_calls", []) or [])
        for output_call in output_calls:
            if str(output_call).split(":", 1)[0] == output_module:
                return str(output_call)
        return f"{output_module}:1"
    modules = list(getattr(op, "modules", []) or [])
    if not modules:
        return None
    return str(modules[-1])
