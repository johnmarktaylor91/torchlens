"""Paddle capture modes that keep payloads out of RAM: structure-only and streaming.

Structure-only capture (``structure_only=True``) runs the real eager forward, so
concrete control flow is taken exactly as in a full capture, but no op keeps a
payload. The trace carries the structure-only marker, so every value-bearing
surface (validation, value exports, runnable saves) refuses through the one
structure-only chokepoint instead of reporting a check it has no witnesses for.

Both modes are honored only once the Paddle backend spec opens the matching
capability gate (``structure_only_capture`` and ``has_streaming``); until then
the options stay with the extra-keyword policy, which refuses them as before.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..registry import get_backend_spec
from . import _retention

MODE_KEY = "_paddle_capture_mode"
"""Trace ``__dict__`` key holding the capture's :class:`PaddleCaptureMode`."""


@dataclass(frozen=True)
class PaddleCaptureMode:
    """Payload handling of one Paddle capture.

    Attributes
    ----------
    structure_only
        Whether no op keeps a payload.
    storage
        The ``storage=`` options payloads stream to, or ``None``.
    retained_selector
        The static-label ``save=`` selector whose unselected payloads leave RAM
        during capture, or ``None``.
    """

    structure_only: bool = False
    storage: Any = None
    retained_selector: Any = None


def pop_capture_mode(
    capture: Any, extra_kwargs: dict[str, Any], retained_selector: object = None
) -> PaddleCaptureMode:
    """Take the payload-mode options the Paddle spec admits out of ``extra_kwargs``.

    Parameters
    ----------
    capture
        The call's ``CaptureOptions``, or ``None``.
    extra_kwargs
        Keyword options not bound by ``capture_trace``; admitted ones are removed.
    retained_selector
        The static-label ``save=`` selector capture-time retention may use.

    Returns
    -------
    PaddleCaptureMode
        The admitted mode; options the spec does not admit stay for the refusal policy.
    """

    capabilities = get_backend_spec("paddle").capabilities
    structure_only = False
    if bool(getattr(capabilities, "structure_only_capture", False)):
        flag = extra_kwargs.pop("structure_only", None)
        structure_only = bool(flag) or bool(getattr(capture, "structure_only", False))
    storage = None
    if bool(getattr(capabilities, "has_streaming", False)):
        storage = extra_kwargs.pop("storage", None)
        extra_kwargs.pop("streaming", None)
    return PaddleCaptureMode(structure_only, storage, None if structure_only else retained_selector)


def start_capture_mode(trace: Any, mode: PaddleCaptureMode) -> None:
    """Arm ``mode`` on the trace under construction.

    Parameters
    ----------
    trace
        The trace under construction.
    mode
        The admitted payload mode.
    """

    trace.__dict__[MODE_KEY] = mode
    trace._options_structure_only = mode.structure_only
    if mode.retained_selector is not None:
        _retention.start_retention(trace, mode.retained_selector)


def saves_payloads(session: Any) -> bool:
    """Whether ops of the capture ``session`` keep payloads (False for structure-only)."""

    mode = session.__dict__.get(MODE_KEY)
    return mode is None or not mode.structure_only
