"""Technical-preview MLX backend for TorchLens.

Eager wrapper capture with live per-op replay validation, static-label
``save=`` filtering, static-label ``intervene=``/``halt=`` dispatch, and
derived gradients. ``mx.compile``/``mx.grad``/``mx.vmap`` traced-transform
entries refuse typed at capture entry (a compiled model attribute ceilings the
capture with ``capture_verified=False``); true backward capture is unsupported
so MLX traces always report ``Trace.autograd.has_backward_pass = False``, and RNG
replay snapshots are currently ``None``.
"""

from __future__ import annotations

from .backend import GradOptions, MLXBackend

__all__ = ["GradOptions", "MLXBackend"]
