"""The explicit marker for TorchLens-internal scalar reads during capture.

A leaf by design: the tensor-label helpers (``_tl``) and the shared tensor
utilities wrap their own bookkeeping reads in :func:`internal_scalar_read`,
so this marker must import nothing from the completeness package that those
modules feed.

At import it registers the wrapper pass-through predicate and the unwrap ledger
with :func:`torchlens._utils._torch_symbols.register_internal_reads`, so the
low-level routed tensor reads can reach the saved originals without importing
this layer.
"""

from __future__ import annotations

import threading
from typing import Any

from .. import _state
from ._torch_symbols import register_internal_reads

__all__ = ("internal_scalar_read", "_internal_read_state")

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


register_internal_reads(_state.internal_reads_pass_through, _state.wrap_epoch_ledgers()[0])
