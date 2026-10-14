"""The ``host_tensor_source`` capture advisory for default (unwitnessed) captures.

A default capture does not arm the completeness witness, so a tensor built from a
foreign host buffer inside ``forward`` (``torch.from_numpy``, ``torch.from_dlpack``,
``torch.frombuffer``, or ``torch.as_tensor`` on a NumPy-style array) enters the graph
as a source with no recorded dependence on the buffer it came from, and nothing
checks it: ``capture_verified`` stays ``None`` ("not checked"). This module records
one ``capture_advisories`` row per capture that counts those calls, so the shared
artifact says so. It never warns and never changes a capture verdict.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from ... import _state

__all__ = (
    "ADVISORY_HOST_TENSOR_SOURCE",
    "HOST_TENSOR_SOURCE_FUNCS",
    "capture_host_tensor_source_advisory",
)

#: ``Trace.annotations["capture_advisories"]`` kind for host-buffer tensor sources.
ADVISORY_HOST_TENSOR_SOURCE = "host_tensor_source"

#: ``torch`` module attributes observed by the advisory, in report order.
HOST_TENSOR_SOURCE_FUNCS: tuple[str, ...] = ("from_numpy", "from_dlpack", "frombuffer", "as_tensor")

_TORCHLENS_ROOT = Path(__file__).resolve().parents[2]
_TORCH_ROOT = Path(torch.__file__).resolve().parent


class _HostSourceState:
    """Per-capture aggregate of observed host-buffer tensor sources."""

    def __init__(self, trace: Any) -> None:
        self.trace = trace
        self.owner_thread_id = threading.get_ident()
        self.count = 0
        self.first_location: str | None = None
        self.names: list[str] = []


def _user_caller_location(depth: int) -> str | None:
    """Return ``file:line`` of the caller ``depth`` frames up when it is user code.

    Calls made by TorchLens or by torch's own Python are not user host sources.
    """

    try:
        frame = sys._getframe(depth)
    except ValueError:
        return None
    filename = Path(frame.f_code.co_filename).resolve()
    for root in (_TORCHLENS_ROOT, _TORCH_ROOT):
        if filename.is_relative_to(root):
            return None
    return f"{filename}:{frame.f_lineno}"


def _is_foreign_array(data: Any) -> bool:
    """Whether ``torch.as_tensor`` data is a foreign host array (not a tensor or list)."""

    if isinstance(data, torch.Tensor):
        return False
    return any(hasattr(data, attr) for attr in ("__array_interface__", "__array__", "__dlpack__"))


def _make_source_wrapper(original: Any, name: str, state: _HostSourceState) -> Any:
    """Wrap one ``torch`` host-source function to count user-code calls during capture."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Count one host-buffer tensor source, then delegate unchanged."""

        if (
            _state._logging_enabled
            and _state._active_trace is state.trace
            and threading.get_ident() == state.owner_thread_id
            and (name != "as_tensor" or _is_foreign_array(args[0] if args else kwargs.get("data")))
        ):
            location = _user_caller_location(2)
            if location is not None:
                state.count += 1
                if state.first_location is None:
                    state.first_location = location
                if name not in state.names:
                    state.names.append(name)
        return original(*args, **kwargs)

    wrapper.__name__ = getattr(original, "__name__", name)
    wrapper.__qualname__ = getattr(original, "__qualname__", name)
    wrapper.__doc__ = getattr(original, "__doc__", None)
    wrapper.__wrapped__ = original  # type: ignore[attr-defined]
    return wrapper


def _record_advisory(state: _HostSourceState) -> None:
    """Append the per-capture advisory row to the trace's persisted annotations."""

    if not state.count:
        return
    annotations = getattr(state.trace, "annotations", None)
    if not isinstance(annotations, dict):
        return
    rows = annotations.setdefault("capture_advisories", [])
    if not isinstance(rows, list):
        return
    sources = ", ".join(f"torch.{name}" for name in state.names)
    rows.append(
        {
            "kind": ADVISORY_HOST_TENSOR_SOURCE,
            "count": int(state.count),
            "first_location": state.first_location,
            "message": (
                f"tensor(s) built from a host buffer during capture ({sources}); the "
                "dependence on that buffer is not captured and was not checked (the "
                "completeness witness is not armed on the default capture)"
            ),
        }
    )


@contextmanager
def capture_host_tensor_source_advisory(trace: Any) -> Iterator[None]:
    """Observe host-buffer tensor sources for one default capture.

    Parameters
    ----------
    trace:
        Active plain Trace receiving the advisory row.

    Yields
    ------
    None
        The capture runs inside this scoped patch of the ``torch`` module attributes.

    Notes
    -----
    The patches are read-through and restored on every exit path. The advisory row is
    written only when the capture body returns normally; it never warns.
    """

    state = _HostSourceState(trace)
    restores: dict[str, Any] = {}
    try:
        for name in HOST_TENSOR_SOURCE_FUNCS:
            original = getattr(torch, name, None)
            if original is None or not callable(original):
                continue
            setattr(torch, name, _make_source_wrapper(original, name, state))
            restores[name] = original
    except BaseException:
        for name, original in restores.items():
            setattr(torch, name, original)
        raise
    try:
        yield
    finally:
        for name, original in restores.items():
            setattr(torch, name, original)
    _record_advisory(state)
