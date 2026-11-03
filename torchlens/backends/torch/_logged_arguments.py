"""Which arguments a captured op logs when they differ from the call's own.

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
"""

from __future__ import annotations

from typing import Any

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
