"""Higher-order-operator entry compiles during a forced-eager capture.

``cond``, ``map``, ``while_loop`` and ``scan`` enter through
``torch._higher_order_ops.utils._hop_compile_and_call``, a fullgraph compile
that lifts closure parameters into operands and gives the eager kernels the
structure torch tests; the capture's ``force_eager`` stance skips it. This
module lets that one compile run, and hides helper frames from Dynamo.

Dynamo inlines every module a ``cond`` branch calls and lifts its parameters
into operands, so the compiled branch never re-enters TorchLens's module or
function wrappers. The lifted ``cond`` entry's backend therefore swaps each
branch graph for a :class:`_CondBranchCallback`: torch's fake passes and the
backward still use the branch graph, while the one real (dense) run calls the
user's own branch function of the current call, which TorchLens captures.

Dynamo does not compile a frame while a dispatch mode is on the stack. For an
entry whose caller names modes to step aside (a ``cond`` entry, or ``map``
under grad, during a witnessed capture), those modes leave the stack for
Dynamo's frame evaluation and are re-entered for the compiled graph's run,
through a backend wrapped around the one the entry's compile environment hands
out; the modes still see every op the graph runs, the ``cond`` branch's dense
run included.

The torch-private entry and environment are reached only through
:func:`torchlens._utils._torch_compat.swap_hop_entry_compile` and
:func:`~torchlens._utils._torch_compat.swap_hop_compilation_env`; this module is
imported directly so the compat module's eager import closure does not grow.
"""

from __future__ import annotations

import contextlib
import types
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import torch

from ..errors._base import transparent_contextmanager
from ._torch_compat import (
    get_current_dispatch_mode_stack,
    get_tracing_tensor_types,
    swap_hop_compilation_env,
    swap_hop_entry_compile,
)
from ._torch_symbols import torch_attr

__all__ = ("compile_disabled", "hop_compile_stance_lift_scope")


_StepsAside = Callable[[Any, Any], tuple[Any, ...]]
"""``(fn, args)`` of one entry compile to the dispatch modes it steps aside, bottom first."""


@transparent_contextmanager
def hop_compile_stance_lift_scope(
    around: Callable[[bool], contextlib.AbstractContextManager[Any]],
    *,
    enabled: bool,
    steps_aside: _StepsAside | None = None,
) -> Iterator[bool]:
    """Let torch's higher-order-operator entry compiles run inside a forced-eager scope.

    For the scope, each entry compile runs under the default stance inside
    ``around(cond_branches)``, on a per-scope copy of its target
    (``_with_fresh_code``). ``cond_branches`` is ``True`` for a ``cond`` entry
    whose real run will call the user's own branches; that needs torch's
    ``setup_compilation_env`` to hand the backend over, so without it every
    entry runs as before and a branch that calls a module still fails loudly.

    Parameters
    ----------
    around:
        Context factory entered for each lifted compile, given ``cond_branches``.
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
    original_env = swap_hop_compilation_env(None)
    cond_calls: list[_CondCall | None] = []
    lifted = _lifted_entry(
        original,
        around,
        cond_calls,
        cond_branches=original_env is not None,
        steps_aside=steps_aside,
    )
    if original_env is not None:
        swap_hop_compilation_env(_branch_callback_env(original_env, cond_calls))
    swap_hop_entry_compile(lifted)
    try:
        yield True
    finally:
        swap_hop_entry_compile(original)
        if original_env is not None:
            swap_hop_compilation_env(original_env)


def _lifted_entry(
    original: Callable[..., Any],
    around: Callable[[bool], contextlib.AbstractContextManager[Any]],
    cond_calls: list[_CondCall | None],
    *,
    cond_branches: bool,
    steps_aside: _StepsAside | None,
) -> Callable[..., Any]:
    """Return the replacement for torch's operator entry compile in one lift scope.

    Parameters
    ----------
    original:
        torch's entry compile.
    around:
        Context factory entered for each lifted compile, given ``cond_branches``.
    cond_calls:
        The scope's stack of active entries; a ``cond`` entry pushes its call.
    cond_branches:
        Whether a ``cond`` entry's real run may call the user's own branches.
    steps_aside:
        See :func:`hop_compile_stance_lift_scope`.

    Returns
    -------
    Callable[..., Any]
        The lifted entry compile.
    """

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
        call = _cond_call(fn, args, modes) if cond_branches else None
        rejoin = _modes_rejoin_compiled_run(modes, aside) if modes else contextlib.nullcontext()
        cond_calls.append(call)
        try:
            with around(call is not None), torch.compiler.set_stance("default"), rejoin:
                return original(target, args, kwargs)
        finally:
            cond_calls.pop()

    return _lifted


@transparent_contextmanager
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

    @transparent_contextmanager
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

        if getattr(compiled, "_boxed_call", False):
            _run._boxed_call = True  # type: ignore[attr-defined]
        return _run

    return _compile


@dataclass(frozen=True)
class _CondCall:
    """One lifted ``cond`` entry call: the user's own branches and operands.

    Attributes
    ----------
    branches:
        The true and the false branch function, as the user passed them.
    operands:
        The operands, as the user passed them.
    modes:
        The dispatch modes the entry steps aside and its compiled run re-enters.
    """

    branches: tuple[Callable[..., Any], Callable[..., Any]]
    operands: Any
    modes: tuple[Any, ...] = ()


def _cond_operator() -> Any:
    """Return torch's ``cond`` higher-order operator, or ``None`` when absent.

    Returns
    -------
    Any
        ``torch.ops.higher_order.cond`` or ``None``.
    """

    return getattr(getattr(torch.ops, "higher_order", None), "cond", None)


def _cond_call(fn: Any, args: Any, modes: tuple[Any, ...]) -> _CondCall | None:
    """Return the ``cond`` call an entry compile runs, or ``None`` for any other entry.

    The entry is ``cond``'s when ``fn`` calls the ``cond`` operator by a global
    name (torch wraps the operator in a small function for Dynamo); its
    arguments are then ``(pred, true_fn, false_fn, operands)``.

    Parameters
    ----------
    fn:
        Function the entry compiles.
    args:
        Its positional arguments.
    modes:
        The dispatch modes the entry steps aside (``steps_aside``'s answer).

    Returns
    -------
    _CondCall | None
        The user's branches, operands and the modes to step aside, or ``None``
        (also when a dispatch mode other than TorchLens's own is active: Dynamo
        then does not compile, and the entry runs as before).
    """

    cond_op = _cond_operator()
    code = getattr(fn, "__code__", None)
    namespace = getattr(fn, "__globals__", None)
    if cond_op is None or code is None or not isinstance(namespace, dict):
        return None
    if not any(namespace.get(name) is cond_op for name in code.co_names):
        return None
    if not modes and get_current_dispatch_mode_stack():
        return None
    return _CondCall(branches=(args[1], args[2]), operands=args[3], modes=modes)


def _branch_callback_env(
    original_env: Callable[..., Any], cond_calls: list[_CondCall | None]
) -> Callable[..., contextlib.AbstractContextManager[Any]]:
    """Return a ``setup_compilation_env`` whose backend installs branch callbacks for ``cond``.

    Parameters
    ----------
    original_env:
        Torch's ``setup_compilation_env``.
    cond_calls:
        The scope's lifted entry calls, innermost last (``None`` for non-``cond``).

    Returns
    -------
    Callable[..., contextlib.AbstractContextManager[Any]]
        The replacement context factory.
    """

    @transparent_contextmanager
    def _env(*args: Any, **kwargs: Any) -> Iterator[Any]:
        """Yield torch's backend, wrapped when the innermost lifted entry is a ``cond``.

        Parameters
        ----------
        *args:
            Positional arguments for torch's context.
        **kwargs:
            Keyword arguments for torch's context.

        Yields
        ------
        Any
            The backend the entry compiles with.
        """

        with original_env(*args, **kwargs) as backend:
            if cond_calls and cond_calls[-1] is not None:
                yield _with_branch_callbacks(backend, cond_calls)
            else:
                yield backend

    return _env


def _with_branch_callbacks(
    backend: Callable[..., Any], cond_calls: list[_CondCall | None]
) -> Callable[..., Any]:
    """Return ``backend`` preceded by the swap of each ``cond`` branch graph for a callback.

    Parameters
    ----------
    backend:
        Torch's backend for the entry compile.
    cond_calls:
        The scope's lifted entry calls, innermost last.

    Returns
    -------
    Callable[..., Any]
        The wrapping backend.
    """

    def _backend(graph_module: Any, example_inputs: Any, **kwargs: Any) -> Any:
        """Install branch callbacks on ``graph_module``, then compile it with torch's backend.

        Parameters
        ----------
        graph_module:
            The entry's compiled ``torch.fx.GraphModule``.
        example_inputs:
            Its example inputs.

        Returns
        -------
        Any
            Torch's backend's compiled callable.
        """

        cond_op = _cond_operator()
        for node in graph_module.graph.nodes:
            if node.op != "call_function" or node.target is not cond_op:
                continue
            for branch, graph_node in enumerate(node.args[1:3]):
                graph = getattr(graph_module, graph_node.target)
                callback = _CondBranchCallback(graph, branch, cond_calls)
                setattr(graph_module, graph_node.target, callback)
        return backend(graph_module, example_inputs, **kwargs)

    return _backend


class _CondBranchCallback(torch.nn.Module):
    """A ``cond`` branch whose one real run calls the user's own branch function.

    Torch calls a branch three ways: its fake-tensor passes (schema checks and
    the backward graph, which differentiates ``graph``), and the dense kernel's
    one real run. Only the real run, with no dispatch mode active, no tracing
    tensor argument and the current call's operands as its leading arguments,
    calls the user's branch with the user's operands, so the submodules and
    functions it calls run through TorchLens's wrappers exactly once. Every
    other call runs ``graph``, as plain torch does.

    Parameters
    ----------
    graph:
        Dynamo's branch graph, taking the operands then the lifted closure tensors.
    branch:
        ``0`` for the true branch, ``1`` for the false branch.
    cond_calls:
        The scope's lifted entry calls, innermost last.
    """

    def __init__(self, graph: Any, branch: int, cond_calls: list[_CondCall | None]) -> None:
        """Hold the branch graph and where to find the current call."""

        super().__init__()
        self.graph = graph
        self._branch = branch
        self._cond_calls = cond_calls

    def forward(self, *args: Any) -> Any:
        """Run the user's branch on the dense run, else the branch graph.

        Parameters
        ----------
        *args:
            The branch graph's inputs: operand leaves, then lifted tensors.

        Returns
        -------
        Any
            A tuple of the branch's output tensors, as the graph returns.
        """

        call = self._cond_calls[-1] if self._cond_calls else None
        if call is None or not _dense_run_of(call, args):
            return self.graph(*args)
        return tuple(_tensor_leaves(call.branches[self._branch](*call.operands)))


def _dense_run_of(call: _CondCall, args: tuple[Any, ...]) -> bool:
    """Return whether a branch call is the dense kernel's real run of ``call``.

    Parameters
    ----------
    call:
        The innermost lifted ``cond`` call.
    args:
        The branch call's arguments.

    Returns
    -------
    bool
        ``True`` with no dispatch mode active but the call's re-entered own
        modes, no tracing tensor argument, and the call's own operand tensors
        leading ``args``.
    """

    stack = get_current_dispatch_mode_stack() or []
    if any(all(mode is not own for own in call.modes) for mode in stack):
        return False
    tracing_types = get_tracing_tensor_types()
    if tracing_types and any(isinstance(arg, tracing_types) for arg in args):
        return False
    operands = list(_tensor_leaves(call.operands))
    return len(args) >= len(operands) and all(a is b for a, b in zip(args, operands))


def _tensor_leaves(value: Any) -> Iterator[Any]:
    """Yield the leaves of nested tuples, lists and dicts, in pytree order.

    Parameters
    ----------
    value:
        A branch's operands or output.

    Yields
    ------
    Any
        Each leaf, depth first; dict values in insertion order.
    """

    if isinstance(value, dict):
        for item in value.values():
            yield from _tensor_leaves(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensor_leaves(item)
    else:
        yield value


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
