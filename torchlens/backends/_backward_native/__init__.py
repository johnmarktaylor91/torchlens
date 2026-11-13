"""Native backward capture, one private module per backend.

Each backend's module (``jax.py``, ``tf.py``, ``mlx.py``, ``paddle.py``,
``tinygrad.py``) is owned by that backend's backward lane and defines the hook
functions the capture lanes call:

- ``backward_ready_scope(session)``: context manager around the forward;
- ``retain_backward_state(trace, session)``: called at the end of the capture;
- ``record_backward(trace, loss, **backward_kwargs)``: run one backward pass and
  journal what it observed as backward events (``torchlens._ir.events``), which
  ``torchlens.backends._backward_neutral`` folds into the records;
- optionally ``capture_backward(trace)``: a context manager for backward calls
  the user drives; a module without it refuses that entry point.

The framework is imported inside those functions, never at module import.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType
from typing import Any

from ...errors import RunCapabilityUnavailableError
from .._backward_neutral import NEUTRAL_PAYLOADS, PayloadPolicy, materialize_neutral_backward

__all__ = (
    "NATIVE_BACKWARD_BACKENDS",
    "NativeBackwardSurface",
    "native_backward_module",
    "native_backward_surface",
    "payload_policy",
)

NATIVE_BACKWARD_BACKENDS: frozenset[str] = frozenset({"jax", "mlx", "paddle", "tf", "tinygrad"})
"""Backend names that may own a native backward module in this package."""

_CODE = "run_capability_unavailable"  # RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE
_DETECTION_STAGE = "native_backward_capture"


def native_backward_module(backend: str) -> ModuleType | None:
    """Return the native backward module for ``backend``, or ``None`` when none exists.

    Parameters
    ----------
    backend:
        Registered backend name (``"jax"``, ``"tf"``, ...).

    Returns
    -------
    ModuleType | None
        The imported module, or ``None`` when the backend is not listed or its
        module is not built yet. An import error raised inside an existing
        module propagates, so a broken module is never mistaken for a missing one.

    Examples
    --------
    >>> native_backward_module("torch") is None
    True
    """

    if backend not in NATIVE_BACKWARD_BACKENDS:
        return None
    module_name = f"{__name__}.{backend}"
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            return None
        raise


def payload_policy(module: ModuleType) -> PayloadPolicy:
    """Return the native module's ``PAYLOADS`` policy, else the neutral default."""

    policy = getattr(module, "PAYLOADS", NEUTRAL_PAYLOADS)
    return policy if isinstance(policy, PayloadPolicy) else NEUTRAL_PAYLOADS


@dataclass(frozen=True)
class NativeBackwardSurface:
    """The ``backward_capture`` implementation of a backend with a native module.

    It has the two entry points PyTorch's surface (``backends.torch.backward``)
    has, so ``Trace.backward`` and ``Trace.capture_backward`` dispatch to either
    the same way.
    """

    backend: str
    module: ModuleType

    def trace_backward(self, trace: Any, loss: Any, **backward_kwargs: Any) -> Any:
        """Run one backward pass through the native module and return ``trace``."""

        stream = getattr(trace, "_capture_events", None)
        if stream is not None and int(getattr(stream, "pass_index_base", 0) or 0):
            raise RunCapabilityUnavailableError(
                f"backward() on a forked {self.backend} trace is unavailable: the fork keeps "
                "the backward records it was made with. Call backward on the original trace.",
                code=_CODE,
                detection_stage=_DETECTION_STAGE,
            )
        self.module.record_backward(trace, loss, **backward_kwargs)
        self.sync_projection(trace)
        return trace

    def sync_projection(self, trace: Any) -> None:
        """Fold the trace's journaled backward events into its records when they changed."""

        materialize_neutral_backward(trace, payloads=payload_policy(self.module))

    def recording_backward(self, trace: Any) -> Any:
        """Return the module's context manager for user-driven backward calls."""

        capture = getattr(self.module, "capture_backward", None)
        if capture is None:
            raise RunCapabilityUnavailableError(
                f"capture_backward() is unavailable on {self.backend}: backward calls the user "
                "drives cannot be observed there. Call trace.backward(...) instead.",
                code=_CODE,
                detection_stage=_DETECTION_STAGE,
            )
        return capture(trace)


def native_backward_surface(backend: str) -> NativeBackwardSurface | None:
    """Return the ``backward_capture`` surface for ``backend``, or ``None`` when unbuilt.

    The registry's ``capability_implementations["backward_capture"]`` factory for
    a non-PyTorch backend is ``lambda: native_backward_surface(name)``; ``None``
    makes the gate refuse typed instead of admitting an unbuilt capture.
    """

    module = native_backward_module(backend)
    return None if module is None else NativeBackwardSurface(backend, module)
