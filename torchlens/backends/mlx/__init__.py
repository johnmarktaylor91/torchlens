"""Technical-preview MLX backend for TorchLens.

This backend is a smoke-level implementation for the capture-pipeline
unification sprint. It is not feature-complete: ``mx.compile``-wrapped models
are unsupported, backward capture is unsupported so MLX traces always report
``Trace.has_backward_pass = False``, and RNG replay snapshots are currently
``None``. See plan §9 for the full MLX-readiness checklist.
"""

from __future__ import annotations

from .backend import MLXBackend

__all__ = ["MLXBackend"]
