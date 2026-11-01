"""Per-pass bookkeeping shared by the RUN stage's skeleton and its exit arms.

``_RunState`` is private to the RUN stage: the hand-off records in
``stages.py`` cross stage boundaries; this one never leaves
``run_and_log_inputs_through_model`` and the arm helpers it calls.
"""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from types import TracebackType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .._capture_state_helpers import CompiledCapturePrep
    from ..backends.registry import CaptureBackend
    from .session import CaptureSession


@dataclass(slots=True)
class _RunState:
    """Mutable bookkeeping of one forward pass, shared by the skeleton's helpers.

    Private to this module: the hand-off records in ``stages.py`` cross stage
    boundaries; this one never leaves ``run_and_log_inputs_through_model``.
    """

    backend: CaptureBackend
    model: object
    input_args: Any
    input_kwargs: dict[Any, Any] | None
    seed: int
    postprocess: bool
    reservation_resume: object | None = None
    capture_slot: Any = None
    restore_rng: Callable[[], None] = lambda: None
    input_arg_names: list[str] = field(default_factory=list)
    model_device: object | None = None
    input_tensors: list[Any] | None = None
    capture_session: CaptureSession | None = None
    capture_events: object | None = None
    instrumentation: contextlib.ExitStack | None = None
    compiled_prep: CompiledCapturePrep | None = None
    exc_info: tuple[type[BaseException] | None, BaseException | None, TracebackType | None] = (
        None,
        None,
        None,
    )

    def cleanup_payload(self) -> tuple[Any, Any, tuple[Any, Any]]:
        """Return the payload the backend's session cleanups take."""

        return (self.model, self.input_tensors, (self.input_args, self.input_kwargs))


def _attach_note(exc: BaseException, note: str) -> None:
    """Attach ``note`` to ``exc`` (PEP 678), or warn on Python 3.10 without notes."""

    add_note = getattr(exc, "add_note", None)
    if add_note is not None:
        add_note(note)
    else:  # Python 3.10: no PEP 678 notes -- surface via warning.
        warnings.warn(note, RuntimeWarning, stacklevel=3)
