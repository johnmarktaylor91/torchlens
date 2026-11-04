"""TorchLens-internal reads during capture: the scalar-read marker and routed reads.

A leaf by design: the tensor-label helpers (``_tl``) and the shared tensor
utilities wrap their own bookkeeping reads in :func:`internal_scalar_read`,
so this marker must import nothing from the completeness package that those
modules feed.

Routed tensor reads
-------------------
Capture's bookkeeping reads tensor metadata tens of thousands of times per trace
(storage identity, element counts, layout flags) under ``pause_logging``. Each such
read used to re-enter the per-op wrapper only to take its logging-off fast path and
call the original: about 41k of the 44k wrapper entries of a GPT-2 generate capture.

:func:`tensor_read` and :func:`tensor_reads` return, for one receiver, callables that
are called with the receiver as their only argument. A callable is the saved original
(the unwrap ledger's entry for the TorchLens wrapper installed on the receiver's class)
only when that wrapper's fast path would be a pure pass-through for this call:

* the logging toggle is off (the wrapper's own fast-path predicate);
* no guarded fast re-run collector is observing wrapped calls;
* the call fingerprint would not count the call (no fingerprint, or a counted pause);
* the receiver's class is exactly ``torch.Tensor`` or ``torch.nn.Parameter``, so no
  subclass override or ``__torch_function__`` decline redispatch can apply;
* the name is in :data:`ROUTED_TENSOR_READS`, whose wrappers carry no fast-path side
  effect (no clone-on-write materialization, device injection, subclass construction or
  detached-activation propagation; ``tests/test_internal_call_fastpath.py`` reads each
  wrapper's closure to prove it).

Otherwise the callable is ``operator.methodcaller(name)``, the exact spelling the call
site used before (``t.untyped_storage()``), so the wrapper runs as it always did. The
class attribute is re-read on every call, so a re-wrap, an unwrap or a peer patch over
the wrapper is honoured immediately. Callers keep their ``pause_logging`` scope: a raw
original call while logging is live is a torch callable outside its wrapper edge.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from operator import methodcaller
from typing import Any

import torch

from ... import _state

__all__ = (
    "ROUTED_TENSOR_READS",
    "_internal_read_state",
    "internal_scalar_read",
    "tensor_read",
    "tensor_reads",
)

_internal_read_state = threading.local()
"""Per-thread depth counter for the explicit TorchLens internal-scalar-read marker."""


class _InternalScalarRead:
    """The context manager :func:`internal_scalar_read` returns.

    A slotted class rather than a ``@contextmanager`` generator: capture enters this
    marker tens of thousands of times per trace, and the generator protocol costs a
    generator, a helper object and a ``StopIteration`` per entry. Same order and
    restores as the generator it replaced: depth up, then the fingerprint hold;
    on exit (also on an exception, never suppressed) the hold is released, then the
    depth restored.
    """

    __slots__ = ("_depth", "_counted")

    def __enter__(self) -> None:
        depth = getattr(_internal_read_state, "depth", 0)
        self._depth = depth
        _internal_read_state.depth = depth + 1
        # An internal read is not a model call: keep it out of the ordered call
        # fingerprint under the same owner-thread rule as ``pause_logging()``.
        self._counted = _state.hold_fingerprint()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        try:
            _state.release_fingerprint(self._counted)
        finally:
            _internal_read_state.depth = self._depth


def internal_scalar_read() -> _InternalScalarRead:
    """Mark a region as a genuine TorchLens capture-internal scalar/comparison read.

    TorchLens itself reads a scalar/bool from a tensor during capture -- extracting a
    scalar-``bool`` op-output value (``ops._log_output_tensor_info``), comparing a
    pre-call input copy against its post-call value for mutation/alias detection
    (``tensor_nanequal`` via ``detect_torch_alias_contract`` and the child-version
    snapshot), and similar bookkeeping reads. Those reads lower to
    ``aten._local_scalar_dense`` (or, for content comparisons, ``aten.equal`` /
    ``aten.allclose`` returning a Python ``bool``) and would otherwise be misrecorded as
    USER host escapes and falsely trip the fail-closed INCOMPLETE gates. This context
    manager marks them EXPLICITLY so the internal-vs-user classifier is an
    allowlist-BY-CONSTRUCTION: an escape is "internal" iff this marker is active, never
    because a stack frame's filename happens to resolve inside the ``torchlens`` package
    (which is spoofable by an ``exec``-compiled user helper carrying a torchlens
    ``co_filename``, and fails for a frameless C-callable). A genuine user escape runs
    with the marker inactive and is always recorded.
    """

    return _InternalScalarRead()


ROUTED_TENSOR_READS: frozenset[str] = frozenset(
    {
        "data_ptr",
        "element_size",
        "is_conj",
        "is_contiguous",
        "is_inference",
        "is_neg",
        "nelement",
        "numel",
        "storage_offset",
        "untyped_storage",
    }
)
"""Zero-argument tensor reads whose wrapper fast path is a pure pass-through."""

_SPELLED: dict[str, Callable[[Any], Any]] = {
    name: methodcaller(name) for name in ROUTED_TENSOR_READS
}
_UNWRAP_LEDGER = _state.wrap_epoch_ledgers()[0]  # append-only; never rebound
_TENSOR = torch.Tensor
_PARAMETER = torch.nn.Parameter


def _route_open(tensor: Any) -> bool:
    """Return whether a routed read on ``tensor`` would pass through its wrapper unlogged.

    Parameters
    ----------
    tensor:
        The receiver of the read.

    Returns
    -------
    bool
        True when the wrapper's fast path would only call the original.
    """

    cls = type(tensor)
    return (
        (cls is _TENSOR or cls is _PARAMETER)
        and not _state._logging_enabled
        and _state._active_fast_run_collector is None
        and (_state._call_fingerprint is None or _state._pause_depth > 0)
    )


def tensor_read(tensor: Any, name: str) -> Callable[[Any], Any]:
    """Return the callable for one routed read of ``tensor`` (call it with ``tensor``).

    Parameters
    ----------
    tensor:
        The receiver of the read.
    name:
        A name in :data:`ROUTED_TENSOR_READS`.

    Returns
    -------
    Callable[[Any], Any]
        The saved original when the wrapper would pass through, else the method call.
    """

    spelled = _SPELLED[name]
    if not _route_open(tensor):
        return spelled
    original = _UNWRAP_LEDGER.get(id(getattr(type(tensor), name, None)))
    return spelled if original is None else original


def tensor_reads(tensor: Any, *names: str) -> tuple[Callable[[Any], Any], ...]:
    """Return the callables for several routed reads of ``tensor``, in ``names`` order.

    Parameters
    ----------
    tensor:
        The receiver of the reads.
    *names:
        Names in :data:`ROUTED_TENSOR_READS`.

    Returns
    -------
    tuple[Callable[[Any], Any], ...]
        One callable per name, each as :func:`tensor_read` would return it.
    """

    if not _route_open(tensor):
        return tuple(_SPELLED[name] for name in names)
    cls = type(tensor)
    resolved = []
    for name in names:
        original = _UNWRAP_LEDGER.get(id(getattr(cls, name, None)))
        resolved.append(_SPELLED[name] if original is None else original)
    return tuple(resolved)
