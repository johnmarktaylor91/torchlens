"""The explicit marker for TorchLens-internal scalar reads during capture.

A leaf by design: the tensor-label helpers (``_tl``) and the shared tensor
utilities wrap their own bookkeeping reads in :func:`internal_scalar_read`,
so this marker must import nothing from the completeness package that those
modules feed.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

from .. import _state

__all__ = ("internal_scalar_read", "_internal_read_state")

_internal_read_state = threading.local()
"""Per-thread depth counter for the explicit TorchLens internal-scalar-read marker."""


@contextmanager
def internal_scalar_read() -> Iterator[None]:
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

    depth = getattr(_internal_read_state, "depth", 0)
    _internal_read_state.depth = depth + 1
    # An internal read is not a model call: keep it out of the ordered call
    # fingerprint under the same owner-thread rule as ``pause_logging()``.
    counted = _state.hold_fingerprint()
    try:
        try:
            yield
        finally:
            _state.release_fingerprint(counted)
    finally:
        _internal_read_state.depth = depth
