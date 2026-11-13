"""TensorFlow derived-gradient replay of the program the capture actually ran.

TensorFlow derived gradients come from one auxiliary forward replay under a
``tf.GradientTape``. When the capture carried ``intervene=`` edits or a
``stop_after=`` halt, rerunning the unedited model would differentiate a
different program (and always refuse on output divergence). This module runs
the replay with the same edits wired into the writable wrap layer and the same
halt, so the tape records the edited program and ``loss_fn`` receives the
halted boundary. The finished trace is left as the capture wrote it: the halt
latch the replay sets is put back, and the zero-fire audit is left to the live
capture so its warnings are not repeated.

A callable edit runs once more during the replay; an edit with side effects
sees a second call.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from ._halt import CallbackAbort

if TYPE_CHECKING:
    from .modules import TFModuleTree

_STOP_LATCH = "_stop_requested"
_ABSENT = object()


@contextmanager
def preserved_module_counts(module_tree: TFModuleTree | None) -> Iterator[None]:
    """Clear module call counters for a replay, then restore the capture's values.

    Parameters
    ----------
    module_tree
        Object-module discovery tree, or ``None`` in function-root mode.

    Yields
    ------
    None
        Control while the replay runs with fresh counters.
    """

    if module_tree is None:
        yield
        return
    call_counts = dict(module_tree.call_counts)
    forward_args = dict(module_tree.forward_args_by_call)
    module_tree.call_counts.clear()
    module_tree.forward_args_by_call.clear()
    try:
        yield
    finally:
        module_tree.call_counts.clear()
        module_tree.call_counts.update(call_counts)
        module_tree.forward_args_by_call.clear()
        module_tree.forward_args_by_call.update(forward_args)


def run_tf_gradient_replay(
    session: Any, *, trace: Any, intervene: Any, halt: Any, runtime: Any
) -> Any:
    """Run one payload-free replay session with the capture's edits and halt.

    Call it inside the active ``tf.GradientTape`` context. ``tl.grad`` clauses
    in ``intervene`` become identity edits whose custom gradient applies the
    clause's edit, so the tape propagates the edited gradient upstream.

    Parameters
    ----------
    session
        Payload-free ``TFEagerCaptureSession`` carrying the gradient tap.
    trace
        Finished trace of the capture being differentiated.
    intervene
        The capture's ``intervene=`` value, re-applied to the replay.
    halt
        The capture's ``stop_after=`` value, re-applied to the replay.
    runtime
        The TensorFlow intervention module (``tf.interventions``), passed in so
        this module stays outside the backend package's import cycle.

    Returns
    -------
    TFCaptureResult
        The replay result; with a halt its ``output`` is the halted boundary.
    """

    if intervene is not None:
        intervene = runtime.tf_gradient_replay_intervene(trace, session.tf, intervene)
    if intervene is None and halt is None:
        return session.run()
    wiring = runtime.wire_eager_session(trace, session, intervene, halt)
    try:
        with _restored_stop_latch(trace):
            if wiring.plan is None:
                return session.run()
            with runtime.tf_intervention_wrap(session.tf, wiring.plan, session):
                return session.run()
    except CallbackAbort as abort:
        raise abort.error.with_traceback(abort.error.__traceback__) from None


@contextmanager
def _restored_stop_latch(trace: Any) -> Iterator[None]:
    """Put back the capture's stop-request latch after the replay's halt sets one."""

    latch = trace.__dict__.get(_STOP_LATCH, _ABSENT)
    try:
        yield
    finally:
        if latch is _ABSENT:
            trace.__dict__.pop(_STOP_LATCH, None)
        else:
            trace.__dict__[_STOP_LATCH] = latch


__all__ = ("preserved_module_counts", "run_tf_gradient_replay")
