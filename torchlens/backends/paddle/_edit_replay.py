"""Derived-gradient replay of the effective edited Paddle program.

A Paddle capture computes derived gradients by running the model a second
time under ``paddle.grad``. When the capture applied live edits
(``trace(intervene=...)``) or stopped early (``trace(stop_after=...)``), that
second run must differentiate the program the capture actually executed, not
the unedited model. :func:`run_derived_grad_forward` runs it through a tap
that re-applies every replacing hook at the same wrapped-call ordinal and
output path where it fired during capture, and stops at the same call that
halted the capture, returning that call's (edited) output as the frontier.

The hooks run inside the replayed call, so Paddle differentiates through
them exactly as written: ``tl.scale(2.0)`` doubles the gradient arriving
upstream, and ``tl.zero_ablate()`` disconnects it. The backend's existing
check that the replay output equals the captured output (for a halted
capture, the captured frontier) then confirms the replay ran the same
program; a hook that does not reproduce its value fails that check.

Call ordinals count top-level wrapped calls that return at least one tensor,
the same rule the capture uses when it numbers op calls; nested calls (inside
a wrapped call or a hook) are not counted, as during capture.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ...intervention._grad_site_neutral import GradSiteCall, armed_grad_sites
from ._edit_helpers import _replace_leaves
from ._grad_sites import armed_hooks_by_call
from .wrappers import paddle_tap_observer

EditHooks = tuple[tuple[tuple[Any, ...], Callable[[Any], Any]], ...]
GradHooks = tuple[tuple[tuple[Any, ...], Any, str], ...]


@dataclass(frozen=True)
class EditReplayPlan:
    """The edits and halt point of one Paddle capture, keyed by call ordinal.

    Parameters
    ----------
    edits
        Replacing hooks per wrapped-call ordinal: ``(output path, hook)`` pairs
        in the order they fired.
    halt_call_id
        Ordinal of the call whose output halted the capture, or ``None``.
    grad_hooks
        Armed ``tl.grad`` edits per wrapped-call ordinal: ``(output path, armed
        edit, raw label)``, registered as Paddle gradient hooks on the replayed
        output.
    """

    edits: Mapping[int, EditHooks] = field(default_factory=dict)
    halt_call_id: int | None = None
    grad_hooks: Mapping[int, GradHooks] = field(default_factory=dict)

    @property
    def is_plain(self) -> bool:
        """Whether the capture neither edited, halted nor armed a gradient edit."""

        return not self.edits and self.halt_call_id is None and not self.grad_hooks


def edit_replay_plan(trace: Any) -> EditReplayPlan:
    """Return the edit replay plan recorded by the trace's live intervention runtime.

    Parameters
    ----------
    trace
        Paddle trace still holding ``_paddle_intervention_runtime`` (derived
        gradients are attached before capture cleanup removes it).

    Returns
    -------
    EditReplayPlan
        The recorded plan; plain when the capture had no runtime.
    """

    grad_hooks = armed_hooks_by_call(trace) if armed_grad_sites(trace) else {}
    runtime = getattr(trace, "_paddle_intervention_runtime", None)
    if runtime is None:
        return EditReplayPlan(grad_hooks=grad_hooks)
    return EditReplayPlan(dict(runtime.fired_edits), runtime.halt_call_id, grad_hooks)


class _ReplayHalt(BaseException):
    """Stop the replayed forward at the captured halt call (a BaseException, like StopSignal)."""

    def __init__(self, frontier_output: Any) -> None:
        """Carry the halting call's output as the replay's frontier output."""

        super().__init__("paddle derived-gradient replay reached the captured halt")
        self.frontier_output = frontier_output


class PaddleEditReplayTap:
    """Paddle tap observer that replays the captured edits and halt.

    Parameters
    ----------
    backend
        Active ``PaddleBackend`` (tensor traversal).
    plan
        Edits and halt point to replay.
    inner
        Optional tap observer to chain (the intermediate-gradient observer);
        it sees each top-level call with the edited output.
    """

    def __init__(self, backend: Any, plan: EditReplayPlan, inner: Any | None = None) -> None:
        """Store the plan and start counting calls at zero."""

        self._backend = backend
        self._plan = plan
        self._inner = inner
        self._depth = 0
        self._call_id = 0
        self._grad_calls: dict[tuple[int, str], int] = {}

    def call(
        self,
        original: Callable[..., Any],
        op_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Run one wrapped call, apply the edits recorded at its ordinal, stop at the halt.

        Parameters
        ----------
        original
            Unwrapped Paddle callable.
        op_name
            TorchLens operation name of the wrapper.
        args
            Positional call arguments.
        kwargs
            Keyword call arguments.

        Returns
        -------
        Any
            The call's output with the recorded replacements substituted.
        """

        if self._depth > 0:
            return original(*args, **kwargs)
        call_id = self._call_id + 1

        def edited(*call_args: Any, **call_kwargs: Any) -> Any:
            """Run ``original`` and substitute the hooks recorded at ``call_id``."""

            output = self._apply_edits(call_id, original(*call_args, **call_kwargs))
            self._register_grad_hooks(call_id, output)
            return output

        self._depth += 1
        try:
            if self._inner is None:
                output = edited(*args, **kwargs)
            else:
                output = self._inner.call(edited, op_name, args, kwargs)
        finally:
            self._depth -= 1
        if not self._has_tensors(output):
            return output
        self._call_id = call_id
        if call_id == self._plan.halt_call_id:
            raise _ReplayHalt(output)
        return output

    def _has_tensors(self, output: Any) -> bool:
        """Whether ``output`` holds a Paddle tensor (only those calls get an ordinal)."""

        return any(True for _ in self._backend._iter_tensors_with_paths(output))

    def _register_grad_hooks(self, call_id: int, output: Any) -> None:
        """Register the ``tl.grad`` edits armed at ``call_id`` as gradient hooks."""

        armed = self._plan.grad_hooks.get(call_id, ())
        if not armed:
            return
        leaves = dict(self._backend._iter_tensors_with_paths(output))
        for path, edit, label in armed:
            leaf = leaves.get(path)
            if leaf is not None and not bool(leaf.stop_gradient):
                leaf.register_hook(self._grad_hook(edit, label))

    def _grad_hook(self, edit: Any, label: str) -> Callable[[Any], Any]:
        """Wrap one armed edit as a Paddle gradient hook that counts its calls."""

        def _hook(grad: Any) -> Any:
            """Apply the armed edit to the gradient arriving at the site."""

            key = (id(edit), label)
            call_index = self._grad_calls.get(key, 0)
            self._grad_calls[key] = call_index + 1
            return edit.apply(grad, GradSiteCall(label, 0, call_index, 1))

        return _hook

    def _apply_edits(self, call_id: int, output: Any) -> Any:
        """Substitute the hooks recorded at ``call_id`` into ``output``, in firing order."""

        hooks = self._plan.edits.get(call_id, ())
        if not hooks:
            return output
        leaves = dict(self._backend._iter_tensors_with_paths(output))
        replacements: dict[tuple[Any, ...], Any] = {}
        for path, hook in hooks:
            if path in leaves:
                replacements[path] = hook(replacements.get(path, leaves[path]))
        return _replace_leaves(output, replacements)


def run_derived_grad_forward(
    backend: Any,
    trace: Any,
    observer: Any | None,
    model: Callable[..., Any],
    args: Sequence[Any],
    kwargs: Mapping[Any, Any],
) -> Any:
    """Run the derived-gradient forward over the program the capture executed.

    A plain capture runs the model as before (through ``observer`` when
    intermediate gradients were requested). An edited or halted capture runs
    through :class:`PaddleEditReplayTap` and returns the model output, or the
    halting call's output for a halted capture.

    Parameters
    ----------
    backend
        Active ``PaddleBackend``.
    trace
        Finalized trace still holding its live intervention runtime.
    observer
        Intermediate-gradient tap observer, or ``None``.
    model
        Prepared Paddle callable.
    args
        Positional model inputs.
    kwargs
        Keyword model inputs.

    Returns
    -------
    Any
        Raw output of the replayed program.
    """

    plan = edit_replay_plan(trace)
    if plan.is_plain and getattr(trace, "_paddle_intervention_runtime", None) is None:
        if observer is None:
            return model(*args, **dict(kwargs))
        with paddle_tap_observer(observer):
            return model(*args, **dict(kwargs))
    tap = PaddleEditReplayTap(backend, plan, observer)
    trace._paddle_edit_replay_tap = tap
    try:
        with paddle_tap_observer(tap):
            return model(*args, **dict(kwargs))
    except _ReplayHalt as halt:
        return halt.frontier_output
    finally:
        del trace._paddle_edit_replay_tap


__all__ = (
    "EditReplayPlan",
    "PaddleEditReplayTap",
    "edit_replay_plan",
    "run_derived_grad_forward",
)
