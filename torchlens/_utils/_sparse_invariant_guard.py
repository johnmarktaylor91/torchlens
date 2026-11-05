"""Sparse-tensor invariant checking for every replay of recorded (untrusted) calls.

PyTorch skips its sparse index checks by default: a sparse constructor handed indices
outside the declared shape builds the tensor anyway, and a later op on it reads or
writes out of range. A saved bundle is untrusted input, so every replay or execution
of a recorded call runs with ``torch.sparse.check_sparse_tensor_invariants`` enabled
(:func:`call_with_sparse_invariants`); the previous global setting is restored on exit,
including on exceptions. Ordinary eager capture of a user's own model is unchanged.

The constructors' ``check_invariants`` keyword overrides the global setting per call,
so a recorded call carrying a value other than ``None`` or ``True`` for it is refused
(:func:`invariant_switch_disabled`, :func:`mapping_disables_invariants`), at bundle
parse and again at execution.

Where the context manager is absent (``HAS_SPARSE_INVARIANT_CHECK`` false), replay
refuses any sparse construction instead of running it unvalidated.

Leaf module: imports ``torch``, the standard library, the ``errors`` taxonomy and the
capability flags in ``_torch_compat``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any, TypeVar

import torch

from ..errors._runnable import RunPreconditionError
from . import _torch_compat

__tl_layer__ = "L1"  # reads the _torch_compat capability flags at import

_T = TypeVar("_T")

#: The sparse constructors' per-call override of the global invariant setting.
INVARIANT_KEYWORD = "check_invariants"


def _keeps_invariants(value: Any) -> bool:
    """Return whether a ``check_invariants`` value leaves the checks on (``None`` or ``True``)."""

    return value is None or value is True


def invariant_switch_disabled(
    tensor_arguments: Iterable[Any], literal_arguments: Iterable[Any]
) -> bool:
    """Return whether a parsed call descriptor can switch the sparse invariant checks off.

    Parameters
    ----------
    tensor_arguments:
        The call's tensor leaves (each exposes ``argument_path``).
    literal_arguments:
        The call's literal leaves (each exposes ``argument_path`` and the parsed
        ``value``; a scalar literal exposes ``kind`` and ``value``).

    Returns
    -------
    bool
        ``True`` when a tensor fills the ``check_invariants`` keyword, or a literal
        fills it (or a path under it) with anything but a scalar ``None`` or ``True``.
    """

    def _names_switch(path: tuple[Any, ...]) -> bool:
        """Return whether a path is rooted at the ``check_invariants`` keyword."""
        return len(path) >= 2 and path[0] == "kwargs" and path[1] == INVARIANT_KEYWORD

    if any(_names_switch(argument.argument_path) for argument in tensor_arguments):
        return True
    for argument in literal_arguments:
        path, literal = argument.argument_path, argument.value
        if not _names_switch(path):
            continue
        kind = getattr(literal, "kind", None)
        if len(path) != 2 or kind not in ("none", "bool"):
            return True
        if not _keeps_invariants(getattr(literal, "value", False)):
            return True
    return False


def mapping_disables_invariants(kwargs: Mapping[str, Any] | None) -> bool:
    """Return whether a keyword mapping switches the sparse invariant checks off.

    Parameters
    ----------
    kwargs:
        Keyword mapping of one recorded or reconstructed call.

    Returns
    -------
    bool
        ``True`` when ``check_invariants`` is present with a value other than
        ``None`` or ``True``.
    """

    if not kwargs or INVARIANT_KEYWORD not in kwargs:
        return False
    return not _keeps_invariants(kwargs[INVARIANT_KEYWORD])


def invariant_keyword_refusal_detail(where: str) -> str:
    """Return the shared refusal message for a bundle keyword that disables the checks.

    Parameters
    ----------
    where:
        Short description of the refusing boundary, used in the message.

    Returns
    -------
    str
        One refusal sentence naming the keyword and the rule.
    """

    return (
        f"{where} carries keyword argument {INVARIANT_KEYWORD!r} with a value that turns "
        "off sparse-tensor index validation. Bundle-supplied calls always replay with "
        "sparse invariant checks on, so this keyword may only be None or True."
    )


def is_sparse_construction(func: Callable[..., Any]) -> bool:
    """Return whether a replayed callable may construct a sparse tensor (by name).

    Parameters
    ----------
    func:
        The resolved callable about to run.

    Returns
    -------
    bool
        ``True`` when the callable's name or module mentions ``sparse`` (every
        sparse constructor, conversion, ``*_unsafe`` spelling and the legacy
        ``torch.sparse.*Tensor`` types do).
    """

    names = (getattr(func, attr, None) for attr in ("__name__", "__qualname__", "__module__"))
    return any("sparse" in str(name).lower() for name in names if name)


def call_with_sparse_invariants(
    func: Callable[..., _T], args: Any, kwargs: Mapping[str, Any]
) -> _T:
    """Run one replayed call with sparse invariant checks enabled, then restore them.

    Parameters
    ----------
    func:
        Callable to execute.
    args:
        Positional arguments (any sequence).
    kwargs:
        Keyword arguments.

    Returns
    -------
    _T
        Return value from ``func``.

    Raises
    ------
    RunPreconditionError
        ``call_structure_mismatch`` when the keywords switch the checks off, or when
        the checks are unavailable on this torch and ``func`` may build a sparse tensor.
    """

    if mapping_disables_invariants(kwargs):
        raise RunPreconditionError(
            invariant_keyword_refusal_detail("Replayed call"), code="call_structure_mismatch"
        )
    if not _torch_compat.HAS_SPARSE_INVARIANT_CHECK:
        if is_sparse_construction(func):
            raise RunPreconditionError(
                f"Replayed call {getattr(func, '__name__', func)!r} may construct a sparse "
                "tensor, and this torch cannot validate sparse indices; it is refused.",
                code="call_structure_mismatch",
            )
        return func(*args, **kwargs)
    with torch.sparse.check_sparse_tensor_invariants(enable=True):
        return func(*args, **kwargs)


def refuse_invariant_switch(layer: Any, kwargs: Mapping[str, Any] | None) -> None:
    """Refuse a replay re-execution whose keywords switch sparse invariant checks off.

    The execution-boundary belt beside the pointer-keyword belt: it raises before
    any probe refusal handler can absorb it.

    Parameters
    ----------
    layer:
        The op about to be re-executed.
    kwargs:
        The keyword mapping the re-execution would pass.

    Raises
    ------
    RunPreconditionError
        ``call_structure_mismatch`` when ``check_invariants`` is neither ``None`` nor ``True``.
    """

    if not mapping_disables_invariants(kwargs):
        return
    label = getattr(layer, "layer_label", "<unknown>")
    raise RunPreconditionError(
        invariant_keyword_refusal_detail(f"Replay of op {label!r}"),
        code="call_structure_mismatch",
    )
