"""Which arguments or outputs a captured op logs when they differ from the call's own.

Three ops record something other than their literal ``(args, kwargs)``:

* a **storage-rebinding setter** is recorded as the canonical
  single-argument ``detach(rhs)`` call: the receiver's OLD value has zero dataflow
  into the result (the rebind replaces every value), so only the right-hand side is
  logged, and the live receiver still gets the op's label through the same-object
  in-place propagation in the wrapper;
* ``__deepcopy__`` drops its memo, which is copy-protocol bookkeeping that ends up
  holding the output;
* a **foreign data round trip** (``_foreign_data``) logs the departed tensor(s) as the
  arguments of the ``foreigndata`` op, so the parent edge, saved args and replay follow
  the generic op path while the callable carries the resolution.

A receiver-mutating call that returns ``None`` logs its mutated receiver as the output
(:func:`logged_receiver_output`), so the op is recorded whatever the binding returns.
"""

from __future__ import annotations

from typing import Any

import torch

from ._foreign_data import HostEntry


def logged_args(
    func_name: str,
    args: tuple[Any, ...],
    arg_copies: tuple[Any, ...],
    is_storage_rebinding_setter: bool,
    host_entry: HostEntry | None,
) -> tuple[tuple[Any, ...], tuple[Any, ...]] | None:
    """Return ``(log_args, log_arg_copies)`` when the op logs other than its own arguments.

    ``None`` means the call's own ``(args, kwargs)`` and their copies are logged. Every
    special case logs positional arguments only (its keyword arguments are dropped).

    Parameters
    ----------
    func_name:
        Name of the wrapped function.
    args:
        The call's live positional arguments.
    arg_copies:
        The pre-call copies taken for in-place-safe logging.
    is_storage_rebinding_setter:
        Whether the op rebinds its receiver's storage (``Tensor.data`` setter).
    host_entry:
        The resolved foreign data round trip, when the op is one.
    """

    if host_entry is not None:
        return host_entry.args, host_entry.args
    if is_storage_rebinding_setter and len(args) >= 2:
        return (args[1],), (arg_copies[1],) if len(arg_copies) >= 2 else (args[1],)
    if func_name == "__deepcopy__" and args:
        return (args[0],), tuple(arg_copies[:1]) or (args[0],)
    return None


def logged_receiver_output(
    out: Any,
    args: tuple[Any, ...],
    func_name: str,
    always_receiver: bool,
    is_mutating_property_setter: bool,
) -> Any:
    """Return the output a receiver-mutating call logs: its mutated receiver(s) when needed.

    ``__setitem__``, ``zero_`` and ``__delitem__`` (``always_receiver``) and a mutating
    property setter that returned ``None`` (``t.real = rhs``, round-31 M6) log the receiver
    tensor ``args[0]``. An in-place foreach call (``torch._foreach_add_`` and family) that
    returned ``None`` logs a new list of its receiver members: torch 2.1 binds these
    operators as returning nothing, while later torch returns that same list, so both
    record the same mutation ops. Every other call logs its own output.

    Parameters
    ----------
    out:
        The value the wrapped call returned.
    args:
        Positional arguments of the call.
    func_name:
        Name of the wrapped function.
    always_receiver:
        Whether the call always logs its receiver (``__setitem__``, ``zero_``, ``__delitem__``).
    is_mutating_property_setter:
        Whether the call is a mutating tensor property setter.

    Returns
    -------
    Any
        The output to log; the caller still returns ``out`` itself to user code.
    """

    receiver = args[0] if args else None
    if always_receiver or (
        is_mutating_property_setter and out is None and isinstance(receiver, torch.Tensor)
    ):
        return receiver
    if (
        out is None
        and func_name.startswith("_foreach_")
        and func_name.endswith("_")
        and isinstance(receiver, (list, tuple))
        and all(isinstance(member, torch.Tensor) for member in receiver)
    ):
        return list(receiver)
    return out
