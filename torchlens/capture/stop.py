"""Unified internal stop directives for capture-spine boundaries."""

from __future__ import annotations

import traceback
import weakref
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ..errors import CaptureError
from ..fastlog._halt import HaltSignal
from ..fastlog.exceptions import PredicateError

if TYPE_CHECKING:
    from ..fastlog.options import ForwardErrorMode, RecordingOptions
    from ..fastlog.types import RecordContext

ForwardStopDisposition = Literal["raise", "attach_partial", "return_partial"]


# The torchlens PACKAGE directory (``.../torchlens/torchlens``), derived from this
# file's own location so it identifies the installed package regardless of what the
# surrounding checkout/repo directory is named. Using a path-component check for the
# string ``"torchlens"`` is WRONG: a repo cloned as ``.../torchlens/tests/...`` has a
# ``torchlens`` component and would be mis-classified as library code (which hid the
# real user frame and made find_nan report ``<frozen runpy>`` under some launchers).
_TORCHLENS_PKG_DIR = Path(__file__).resolve().parent.parent


def _is_within(path: Path, root: Path) -> bool:
    """Return whether ``path`` is inside ``root`` (version-portable)."""

    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _is_library_frame(filename: str) -> bool:
    """Return whether a stack frame belongs to library/launcher code, not user code."""

    # Launcher/import-machinery frames (``<frozen runpy>``, ``<frozen importlib._bootstrap>``)
    # are never user code. Keep ``<string>``/``<stdin>`` -- those ARE user code (``-c``/REPL).
    if filename.startswith("<frozen"):
        return True
    try:
        path = Path(filename).resolve()
    except (OSError, ValueError):
        return True
    parts = path.parts
    if "site-packages" in parts or "dist-packages" in parts:
        return True
    if _is_within(path, _TORCHLENS_PKG_DIR):
        return True
    try:
        import torch

        if _is_within(path, Path(torch.__file__).resolve().parent):
            return True
    except Exception:
        pass
    return False


def _live_user_location() -> tuple[str | None, int | None]:
    """Return the nearest non-library stack location.

    Walks the live stack from the innermost frame outward and returns the first frame
    that is genuine user code -- i.e. not inside the torchlens or torch packages, not
    under site/dist-packages, and not a synthetic launcher frame.

    Returns
    -------
    tuple[str | None, int | None]
        User file path and line number, or ``(None, None)`` when no user
        frame can be identified.
    """

    for frame in reversed(traceback.extract_stack()[:-1]):
        if _is_library_frame(frame.filename):
            continue
        return str(frame.filename), frame.lineno
    return None, None


@dataclass(frozen=True, slots=True)
class StopDirective:
    """Compiled stop policy for one active capture.

    Parameters
    ----------
    halt_options
        Predicate options carrying the optional ``halt`` callable.
    raise_on_nan
        Whether op-boundary non-finite tensors abort with ``CaptureError``.
    forward_error_mode
        Failed-forward policy used by fastlog surfaces. Plain trace uses
        ``"raise"``.
    inference_only
        Whether the forward is running under the inference-only public mode.
    inference_only_conflicts
        Explicit backward-related options that conflict with ``inference_only``.
    inference_only_implied
        Whether ``inference_only`` was implied by ``torch.no_grad()`` /
        ``torch.inference_mode()`` at capture entry rather than passed
        explicitly. Only an implied flag is reconciled against the forward's
        actual autograd history (see :func:`cleared_inference_only_directive`).
    """

    halt_options: RecordingOptions | None = None
    raise_on_nan: bool = False
    forward_error_mode: ForwardErrorMode = "raise"
    inference_only: bool = False
    inference_only_conflicts: tuple[str, ...] = ()
    inference_only_implied: bool = False

    @property
    def has_halt(self) -> bool:
        """Return whether a halt predicate is active."""

        return self.halt_options is not None and self.halt_options.halt is not None

    def evaluate_halt(
        self,
        ctx: RecordContext,
        frontier_output: Any | None = None,
    ) -> None:
        """Evaluate the compiled halt predicate for one capture event.

        Parameters
        ----------
        ctx
            Event context supplied to the halt predicate.
        frontier_output
            Live tensor or output structure used as the partial frontier.

        Raises
        ------
        HaltSignal
            If the halt predicate returns ``True``.
        PredicateError
            If the halt predicate returns a non-bool value.
        """

        if self.halt_options is None or self.halt_options.halt is None:
            return
        result = self.halt_options.halt(ctx)
        if not isinstance(result, bool):
            raise PredicateError(
                "halt predicate must return bool. "
                "Remedy: return True or False from the halt predicate.",
                ctx=ctx,
                result=result,
                code="predicate_return_invalid",
            )
        if result:
            # F6 stop-request latch + settlement boundary facts. The boundary
            # label prefers ``raw_label`` so the recorded label resolves
            # through ``trace[...]`` even on the prefix-alias compatibility
            # retry, where ``ctx.label`` is a bare prefix.
            boundary_label = getattr(ctx, "raw_label", None) or ctx.label
            from .. import _state
            from .outcome import StopRequest

            active_trace = _state._active_trace
            if active_trace is not None:
                active_trace.__dict__["_stop_requested"] = StopRequest(
                    kind="halt",
                    reason=ctx.label,
                    boundary_kind=getattr(ctx, "kind", None),
                    boundary_label=boundary_label,
                )
            raise HaltSignal(
                ctx.label,
                frontier_output=frontier_output,
                boundary_kind=getattr(ctx, "kind", None),
                boundary_label=boundary_label,
            )

    def raise_nonfinite(
        self,
        *,
        raw_label: str,
        func_name: str,
        shape: tuple[int, ...],
        dtype: object,
        parents: list[str],
    ) -> None:
        """Raise the structured non-finite capture error for one op boundary.

        Parameters
        ----------
        raw_label
            Raw operation label that produced the non-finite tensor.
        func_name
            Function name for the operation.
        shape
            Tensor shape.
        dtype
            Tensor dtype.
        parents
            Parent raw labels for the operation.

        Raises
        ------
        CaptureError
            Always raised with the historical non-finite payload.
        """

        message = (
            "TorchLens capture stopped at first non-finite tensor: "
            f"op={func_name!r}, layer={raw_label!r}, shape={shape}, dtype={dtype}."
        )
        # Structural marker for settle-time classification (and the F6 latch):
        # the settlement authority reads it to settle ABORTED_NONFINITE instead
        # of generic FAILED, and the boundary checkpoint uses it to detect a
        # swallowed abort. Set via the process-level active-trace slot because
        # this frozen directive has no back-reference by design.
        from .. import _state
        from .outcome import StopRequest

        file_path, line_no = _live_user_location()
        error = CaptureError(
            message,
            file_path=file_path,
            line_no=line_no,
            affected_sites=[raw_label],
            op=func_name,
            layer=raw_label,
            shape=shape,
            dtype=str(dtype),
            parents=parents,
        )
        active_trace = _state._active_trace
        if active_trace is not None:
            # The latch carries the exact exception's identity (weakly):
            # settlement classifies ABORTED_NONFINITE only for THIS error, so
            # an unrelated CaptureError after a swallowed abort keeps its own
            # FAILED diagnostics (R06).
            active_trace.__dict__["_stop_requested"] = StopRequest(
                kind="nonfinite",
                reason=message,
                boundary_kind="op",
                boundary_label=raw_label,
                error_ref=weakref.ref(error),
            )
        raise error

    def forward_disposition(self, exc: BaseException) -> ForwardStopDisposition:
        """Return the compiled failed-forward disposition.

        Parameters
        ----------
        exc
            Original forward exception. It is accepted so future policies can
            inspect exception metadata without changing the call sites.

        Returns
        -------
        ForwardStopDisposition
            Failed-forward policy for the active surface.
        """

        del exc
        return self.forward_error_mode


def stop_directive_for_trace(trace: object) -> StopDirective:
    """Return the compiled stop directive attached to ``trace``.

    Parameters
    ----------
    trace
        Active trace-like capture session.

    Returns
    -------
    StopDirective
        Existing directive or a conservative default compiled from legacy attrs.
    """

    directive = getattr(trace, "_stop_directive", None)
    if isinstance(directive, StopDirective):
        return directive
    halt_options = getattr(trace, "_predicate_save_options", None)
    directive = StopDirective(
        halt_options=halt_options,
        raise_on_nan=bool(getattr(trace, "raise_on_nan", False)),
        forward_error_mode="raise",
        inference_only=bool(getattr(trace, "inference_only", False)),
    )
    setattr(trace, "_stop_directive", directive)
    return directive


def evaluate_halt_stop(
    trace: object,
    ctx: RecordContext,
    options: RecordingOptions,
    frontier_output: Any | None = None,
) -> None:
    """Evaluate the halt portion of the active stop directive.

    Parameters
    ----------
    trace
        Active trace-like capture session.
    ctx
        Event context supplied to the halt predicate.
    options
        Predicate options to compile if the trace has no directive yet.
    frontier_output
        Live tensor or output structure used as the partial frontier.

    Returns
    -------
    None
        Raises when the halt predicate matches.
    """

    directive = stop_directive_for_trace(trace)
    if directive.halt_options is not options:
        directive = StopDirective(
            halt_options=options,
            raise_on_nan=directive.raise_on_nan,
            forward_error_mode=directive.forward_error_mode,
            inference_only=directive.inference_only,
            inference_only_conflicts=directive.inference_only_conflicts,
            inference_only_implied=directive.inference_only_implied,
        )
        setattr(trace, "_stop_directive", directive)
    directive.evaluate_halt(ctx, frontier_output=frontier_output)


def _outputs_carry_autograd_history(outputs: Any) -> bool:
    """Return whether any tensor in a forward's outputs has a ``grad_fn``.

    Parameters
    ----------
    outputs
        The value the user forward returned (any nesting of containers).

    Returns
    -------
    bool
        ``True`` when at least one output tensor carries autograd history.
    """

    import torch

    from .. import _state
    from ..utils.introspection import get_vars_of_type_from_obj

    with _state.pause_logging():
        tensors = get_vars_of_type_from_obj(outputs, torch.Tensor, search_depth=5)
        return any(getattr(t, "grad_fn", None) is not None for t in tensors)


def cleared_inference_only_directive(directive: object, outputs: Any) -> StopDirective | None:
    """Return the directive with ``inference_only`` cleared when the forward kept its graph.

    AGT-2132: a capture entered under ``torch.no_grad()`` (or inference mode)
    implies ``inference_only=True`` because such a forward discards the
    autograd graph. A forward that re-enables grad itself (``with
    torch.enable_grad():``) still builds one, so the premise does not hold:
    the caller then reports ``inference_only=False`` and backward stays
    available. An explicit ``inference_only=True`` is never cleared.

    Parameters
    ----------
    directive
        The capture's stop directive, which records whether the flag was
        implied. Anything else (no directive) means nothing to change.
    outputs
        The value the user forward returned.

    Returns
    -------
    StopDirective | None
        A copy with ``inference_only=False`` when the flag was implied and an
        output carries a ``grad_fn``; otherwise ``None`` (nothing to change).
    """

    if not isinstance(directive, StopDirective):
        return None
    if not (directive.inference_only and directive.inference_only_implied):
        return None
    if not _outputs_carry_autograd_history(outputs):
        return None
    return replace(directive, inference_only=False)
