"""MLX backend for TorchLens.

Eager wrapper capture with live per-op replay validation, static-label
``save=`` filtering, static-label ``intervene=``/``halt=`` dispatch, and
derived gradients. ``mx.compile``/``mx.grad``/``mx.vmap`` traced-transform
entries refuse typed at capture entry (a compiled model attribute ceilings the
capture with ``capture_verified=False``); true backward capture is unsupported
so MLX traces always report ``Trace.autograd.has_backward_pass = False``, and RNG
replay snapshots are currently ``None``.

Array operators (``a + b``), ``mx.array`` methods, the ``mx.*`` and
``mx.fast.*`` function tables and the ``mlx.nn`` layer classes listed in
``wrappers.py`` are all captured. A forward that still reaches an op no
wrapper saw is settled by the backend-local unwrapped policy
(:func:`unwrapped_policy`): ``"flag"`` (default) marks the trace
``capture_verified=False`` with reason ``unwrapped_op_in_capture``;
``"refuse"`` raises ``BackendUnsupportedError`` with that code.
"""

from __future__ import annotations

from .backend import GradOptions, MLXBackend
from .settings import get_unwrapped_policy, set_unwrapped_policy, unwrapped_policy

__all__ = [
    "GradOptions",
    "MLXBackend",
    "get_unwrapped_policy",
    "set_unwrapped_policy",
    "unwrapped_policy",
]
