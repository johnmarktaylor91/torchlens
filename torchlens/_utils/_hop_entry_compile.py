"""Higher-order-operator entry compiles during a forced-eager capture.

``cond``, ``map``, ``while_loop`` and ``scan`` enter through
``torch._higher_order_ops.utils._hop_compile_and_call``, a fullgraph compile
that lifts closure parameters into operands and gives the eager kernels the
structure torch tests; the capture's ``force_eager`` stance skips it. This
module lets that one compile run, and hides helper frames from Dynamo.

Dynamo does not compile a frame while a dispatch mode is on the stack. For an
entry whose caller names modes to step aside (``map`` under grad during a
witnessed capture), those modes leave the stack for Dynamo's frame evaluation
and are re-entered for the compiled graph's run, through a backend wrapped
around the one the entry's compile environment hands out; the modes still see
every op the graph runs.

The torch-private entry and environment are reached only through
:func:`torchlens._utils._torch_compat.swap_hop_entry_compile` and
:func:`~torchlens._utils._torch_compat.swap_hop_compilation_env`; this module is
imported directly so the compat module's eager import closure does not grow.
"""

from __future__ import annotations

import contextlib
import types
from collections.abc import Callable, Iterator
from typing import Any

import torch

from ._torch_compat import swap_hop_compilation_env, swap_hop_entry_compile
from ._torch_symbols import torch_attr

__all__ = ("compile_disabled", "hop_compile_stance_lift_scope")


_StepsAside = Callable[[Any, Any], tuple[Any, ...]]
"""``(fn, args)`` of one entry compile to the dispatch modes it steps aside, bottom first."""


@contextlib.contextmanager
def hop_compile_stance_lift_scope(
    around: Callable[[], contextlib.AbstractContextManager[Any]],
    *,
    enabled: bool,
    steps_aside: _StepsAside | None = None,
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
    steps_aside:
        Names, per entry compile, the dispatch modes (the whole stack, bottom
        first) that leave the stack for Dynamo and rejoin the compiled run;
        ``None`` or an empty answer leaves the modes in place.

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
    copies: dict[tuple[Any, bool], Any] = {}
    aside: list[tuple[Any, ...]] = []

    def _lifted(fn: Any, args: Any, kwargs: Any = None) -> Any:
        """Run entry compile ``fn(*args, **kwargs)`` under the default stance inside ``around()``.

        Returns the compiled call's result; ``steps_aside(fn, args)`` modes rejoin its run.
        """

        modes = steps_aside(fn, args) if steps_aside is not None else ()
        key = (fn, bool(modes))
        target = copies.get(key)
        if target is None:
            target = copies[key] = _with_fresh_code(fn)
        rejoin = _modes_rejoin_compiled_run(modes, aside) if modes else contextlib.nullcontext()
        with around(), torch.compiler.set_stance("default"), rejoin:
            return original(target, args, kwargs)

    swap_hop_entry_compile(_lifted)
    try:
        yield True
    finally:
        swap_hop_entry_compile(original)


@contextlib.contextmanager
def _modes_rejoin_compiled_run(
    modes: tuple[Any, ...], aside: list[tuple[Any, ...]]
) -> Iterator[None]:
    """Step ``modes`` off the dispatch stack for one entry compile; its graph runs under them.

    Without the compile environment to hand a backend through, nothing changes.
    The compile itself gets its own code copy (the ``copies`` key), so a graph
    compiled with the plain backend never runs where the modes must rejoin.

    Parameters
    ----------
    modes:
        The whole dispatch-mode stack, bottom first.
    aside:
        The scope's stack of stepped-aside modes, read by the compiled run.

    Yields
    ------
    None
        While the modes are off the stack and the wrapped environment is installed.
    """

    environment = swap_hop_compilation_env(None)
    if environment is None:
        yield
        return

    @contextlib.contextmanager
    def _rejoining_environment() -> Iterator[Any]:
        """Enter torch's compile environment and yield its backend, wrapped to rejoin.

        Yields
        ------
        Any
            The wrapped backend.
        """

        with environment() as backend:
            yield _rejoining_backend(backend, aside)

    swap_hop_compilation_env(_rejoining_environment)
    aside.append(modes)
    for mode in reversed(modes):
        mode.__exit__(None, None, None)
    try:
        yield
    finally:
        for mode in modes:
            mode.__enter__()
        aside.pop()
        swap_hop_compilation_env(environment)


def _rejoining_backend(backend: Any, aside: list[tuple[Any, ...]]) -> Callable[..., Any]:
    """Wrap Dynamo backend ``backend`` so each compiled graph runs under the stepped-aside modes.

    Parameters
    ----------
    backend:
        The backend torch's compile environment yielded.
    aside:
        The scope's stack of stepped-aside modes; the innermost entry is re-entered.

    Returns
    -------
    Callable[..., Any]
        A backend with ``backend``'s signature.
    """

    def _compile(gm: Any, example_inputs: Any, **kwargs: Any) -> Callable[..., Any]:
        """Compile ``gm`` with the wrapped backend; return a runner that re-enters the modes.

        Parameters
        ----------
        gm:
            The graph Dynamo captured.
        example_inputs:
            Its example inputs.
        **kwargs:
            Further backend arguments.

        Returns
        -------
        Callable[..., Any]
            The compiled graph's runner.
        """

        compiled = backend(gm, example_inputs, **kwargs)

        def _run(*args: Any, **run_kwargs: Any) -> Any:
            """Run the compiled graph with the innermost stepped-aside modes re-entered.

            Parameters
            ----------
            *args, **run_kwargs:
                The graph's inputs.

            Returns
            -------
            Any
                The graph's outputs.
            """

            with contextlib.ExitStack() as stack:
                for mode in aside[-1] if aside else ():
                    stack.enter_context(mode)
                return compiled(*args, **run_kwargs)

        return _run

    return _compile


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
