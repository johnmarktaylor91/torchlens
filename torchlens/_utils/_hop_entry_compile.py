"""Higher-order-operator entry compiles during a forced-eager capture.

``cond``, ``map``, ``while_loop`` and ``scan`` enter through
``torch._higher_order_ops.utils._hop_compile_and_call``, a fullgraph compile
that lifts closure parameters into operands and gives the eager kernels the
structure torch tests; the capture's ``force_eager`` stance skips it. This
module lets that one compile run, and hides helper frames from Dynamo.
The torch-private entry itself is reached only through
:func:`torchlens._utils._torch_compat.swap_hop_entry_compile`; this module is
imported directly so the compat module's eager import closure does not grow.
"""

from __future__ import annotations

import contextlib
import types
from collections.abc import Callable, Iterator
from typing import Any

import torch

from ._torch_compat import swap_hop_entry_compile
from ._torch_symbols import torch_attr

__all__ = ("compile_disabled", "hop_compile_stance_lift_scope")


@contextlib.contextmanager
def hop_compile_stance_lift_scope(
    around: Callable[[], contextlib.AbstractContextManager[Any]], *, enabled: bool
) -> Iterator[bool]:
    """Let torch's higher-order-operator entry compiles run inside a forced-eager scope.

    For the scope, each entry compile runs under the default stance inside
    ``around()``, on a per-scope copy of its target (``_with_fresh_code``).

    Parameters
    ----------
    around:
        Context factory entered for each lifted compile.
    enabled:
        Whether torch has ``set_stance``; ``False`` leaves the entry untouched.

    Yields
    ------
    bool
        Whether the entry was hooked; ``False`` when disabled or when the
        operator utilities were never imported or lack the entry.
    """

    original = swap_hop_entry_compile(None) if enabled else None
    if original is None:
        yield False
        return
    copies: dict[Any, Any] = {}

    def _lifted(fn: Any, args: Any, kwargs: Any = None) -> Any:
        """Run one operator entry compile under the default stance inside ``around()``.

        Parameters
        ----------
        fn:
            Function the entry compiles.
        args:
            Its positional arguments.
        kwargs:
            Its keyword arguments, if any.

        Returns
        -------
        Any
            The compiled call's result.
        """

        target = copies.get(fn)
        if target is None:
            target = copies[fn] = _with_fresh_code(fn)
        with around(), torch.compiler.set_stance("default"):
            return original(target, args, kwargs)

    swap_hop_entry_compile(_lifted)
    try:
        yield True
    finally:
        swap_hop_entry_compile(original)


def _with_fresh_code(fn: Any) -> Any:
    """Return a copy of plain function ``fn`` with its own code object; else ``fn``.

    Dynamo keys its compile cache on the code object, so compiles of the copy
    neither spend the recompile budget of the user's own operator calls nor
    leave capture-time entries behind for them.

    Parameters
    ----------
    fn:
        Callable an operator entry compiles.

    Returns
    -------
    Any
        The copy, sharing globals, defaults and closure, or ``fn`` itself.
    """

    if not isinstance(fn, types.FunctionType):
        return fn
    copy = types.FunctionType(
        fn.__code__.replace(), fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__
    )
    copy.__kwdefaults__ = fn.__kwdefaults__
    return copy


def compile_disabled(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Return ``fn`` behind ``torch.compiler.disable``, or ``fn`` when torch lacks it.

    Parameters
    ----------
    fn:
        Callable whose frames, and the frames it calls, Dynamo must never trace.

    Returns
    -------
    Callable[..., Any]
        The disabled callable.
    """

    compiler = torch_attr("compiler")
    disable = vars(compiler).get("disable") if isinstance(compiler, types.ModuleType) else None
    return disable(fn) if callable(disable) else fn
