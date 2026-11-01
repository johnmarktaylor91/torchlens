"""Model instrumentation entry for the WRAP stage.

``instrumented()`` is the one bracket RUN opens around a forward pass. It
composes, in today's order, the calls that install capture machinery on torch
and on the user's model, and on exit removes them in reverse: the per-session
cleanup, the compiled-capture unwind, the observer restore, and the optional
``unwrap_torch()``. Every exit arm restores what the bracket changed, and the
restore runs before any release, so a capture is a pure observer of the model
(``tests/test_capture_pure_observer.py``).

The arms of a forward pass clean their own session today (``cleanup_model_session``,
``cleanup_failed_forward_session``, ``cleanup_stopped_forward_session`` all retire
the label session last). The bracket's exit therefore cleans only when the label
session it opened is still active, so adopting it over the existing arms adds no
second cleanup and leaves no first one out.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..._capture.stages import InstrumentedModel
from ._tl import active_label_session_token

if TYPE_CHECKING:
    from torch import nn

    from ..._runnable_state_live import CaptureObserverHandle
    from .backend import TorchBackend


@dataclass(frozen=True, slots=True)
class WrapPolicy:
    """What the bracket does around RUN's forward besides preparing the model.

    ``carry_state=None`` leaves the pure-observer bracket to the caller (the
    door owns it today); ``False`` snapshots and restores buffers, training
    flags, plain attributes and RNG; ``True`` keeps forward-equivalent state
    and records attribute writes only. ``on_failure`` runs after the observer
    restore when the body raises. ``unwrap_when_done`` calls
    ``unwrap_torch()`` after a completed capture.
    """

    carry_state: bool | None = None
    on_failure: Callable[[], None] | None = None
    unwrap_when_done: bool = False


_DEFAULT_POLICY = WrapPolicy()


def _resolve_cleanup_payload(cleanup_payload: Any) -> Any:
    """Return the payload ``cleanup_model_session`` takes, calling a thunk if given one.

    RUN learns the input tensors only after the compiled-capture preparation
    has entered, so it may hand the bracket a zero-argument callable that
    builds ``(model, input_tensors, (input_args, input_kwargs))`` at exit.
    """

    if callable(cleanup_payload):
        return cleanup_payload()
    return cleanup_payload


def _session_still_open(opened_token: int | None) -> bool:
    """Whether the label session this bracket opened has not been retired yet.

    Every session cleanup path ends with ``end_label_session()``; a retired or
    replaced session means an arm already cleaned the model.
    """

    if opened_token is None:
        return False
    return active_label_session_token() == opened_token


def _cleanup_if_open(
    backend: TorchBackend, trace: Any, cleanup_payload: Any, opened_token: int | None
) -> None:
    """Run the per-session cleanup once, unless an arm already ran it."""

    if _session_still_open(opened_token):
        backend.cleanup_model_session(trace, _resolve_cleanup_payload(cleanup_payload))


def _observer_context(model: nn.Module, policy: WrapPolicy) -> Any:
    """The pure-observer bracket when asked for, else a no-op context."""

    if policy.carry_state is None:
        return nullcontext(None)
    from ..._runnable_state_live import capture_observer_bracket

    return capture_observer_bracket(
        model, carry_state=policy.carry_state, on_failure=policy.on_failure
    )


def _enter_compiled_prep(stack: ExitStack, model: nn.Module) -> Any:
    """Enter the compiled-capture preparation and return its site record."""

    from ..._capture_state_helpers import CompiledCapturePrep, prepare_compiled_capture

    compiled_prep = stack.enter_context(prepare_compiled_capture(model))
    if not isinstance(compiled_prep, CompiledCapturePrep):
        compiled_prep = CompiledCapturePrep(sites=(), force_eager_stance=False)
    return compiled_prep


def _unwrap_after_success() -> None:
    """Remove the torch wrappers once a capture that asked for it has completed."""

    from .wrappers import unwrap_torch

    unwrap_torch()


@contextmanager
def instrumented(
    backend: TorchBackend,
    trace: Any,
    model: nn.Module,
    cleanup_payload: Any,
    policy: WrapPolicy = _DEFAULT_POLICY,
) -> Iterator[InstrumentedModel]:
    """Prepare ``model`` for one capture and restore everything on exit.

    Enter order: observer snapshot (when ``carry_state`` is given), compiled
    capture preparation, one-time model preparation, per-session preparation.
    Exit order: per-session cleanup (unless the arm already ran it), compiled
    capture unwind, observer restore (then ``on_failure`` when the body
    raised), and ``unwrap_torch()`` after a completed capture that asked for it.

    ``cleanup_payload`` is what ``TorchBackend.cleanup_model_session`` takes on
    exit, or a zero-argument callable returning it (RUN knows the input tensors
    only after entering). ``policy`` chooses the observer bracket, the failure
    callback and the unwrap (:class:`WrapPolicy`).
    """

    with ExitStack() as stack:
        observer: CaptureObserverHandle | None = stack.enter_context(
            _observer_context(model, policy)
        )
        compiled_prep = _enter_compiled_prep(stack, model)
        backend.prepare_model_once(model)
        backend.prepare_model_session(trace, model)
        opened_token = active_label_session_token()
        stack.callback(_cleanup_if_open, backend, trace, cleanup_payload, opened_token)
        handle = InstrumentedModel(
            model=model,
            compiled_prep=compiled_prep,
            observer=observer,
            label_session=opened_token,
            unwrap_when_done=policy.unwrap_when_done,
        )
        yield handle
        if observer is not None and observer.trace is None:
            observer.trace = trace
    if policy.unwrap_when_done:
        _unwrap_after_success()
