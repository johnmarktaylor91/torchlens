"""Unified internal stop directives for capture-spine boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from ..errors import CaptureError
from ..fastlog._halt import HaltSignal
from ..fastlog.exceptions import PredicateError

if TYPE_CHECKING:
    from ..fastlog.options import ForwardErrorMode, RecordingOptions
    from ..fastlog.types import RecordContext

ForwardStopDisposition = Literal["raise", "attach_partial", "return_partial"]


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
    """

    halt_options: "RecordingOptions | None" = None
    raise_on_nan: bool = False
    forward_error_mode: "ForwardErrorMode" = "raise"
    inference_only: bool = False
    inference_only_conflicts: tuple[str, ...] = ()

    @property
    def has_halt(self) -> bool:
        """Return whether a halt predicate is active."""

        return self.halt_options is not None and self.halt_options.halt is not None

    def evaluate_halt(
        self,
        ctx: "RecordContext",
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
            raise PredicateError("halt predicate must return bool", ctx=ctx, result=result)
        if result:
            raise HaltSignal(ctx.label, frontier_output=frontier_output)

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
        raise CaptureError(
            message,
            affected_sites=[raw_label],
            op=func_name,
            layer=raw_label,
            shape=shape,
            dtype=str(dtype),
            parents=parents,
        )

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
    ctx: "RecordContext",
    options: "RecordingOptions",
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
        )
        setattr(trace, "_stop_directive", directive)
    directive.evaluate_halt(ctx, frontier_output=frontier_output)
