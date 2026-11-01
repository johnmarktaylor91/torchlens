"""Exit-arm entries for the FINALIZE stage.

Seam entries: each wraps the helper today's arm in
``torchlens._capture.run.run_and_log_inputs_through_model`` calls. The arm
sequences themselves still run inline there; the FINALIZE and RUN lanes move
them behind these names.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .settle import (
    _extract_and_mark_outputs,
    _finalize_halted_trace,
    _scrub_failed_capture_transients,
    _settle_interrupted_halted_arm,
)

if TYPE_CHECKING:
    from ...backends.registry import CaptureBackend
    from ...recording._halt import StopSignal
    from ..stages import ForwardResult


def finalize_completed(
    result: ForwardResult,
    backend: CaptureBackend | None = None,
) -> tuple[list[Any], list[str]]:
    """Extract and mark the outputs of a completed forward pass.

    Parameters
    ----------
    result:
        Forward result on the completed arm.
    backend:
        Active capture backend, or ``None`` for the trace's own.

    Returns
    -------
    tuple[list[Any], list[str]]
        Output tensors and their addresses.
    """
    return _extract_and_mark_outputs(result.trace, result.raw_output, backend)


def finalize_halted(
    result: ForwardResult,
    backend: CaptureBackend,
    halt_exc: StopSignal,
    model: object,
    input_tensors: list[Any],
    postprocess: bool,
) -> Any | None:
    """Finalize a trace that stopped at a halt frontier.

    Parameters
    ----------
    result:
        Forward result on the halted arm.
    backend:
        Active capture backend.
    halt_exc:
        The stop signal that ended the forward pass.
    model:
        The prepared model.
    input_tensors:
        Input tensors of the forward pass.
    postprocess:
        Whether the trace is materialized immediately.

    Returns
    -------
    Any | None
        The halted output, when one was produced.
    """
    return _finalize_halted_trace(
        result.trace, backend, halt_exc, model, input_tensors, postprocess
    )


def finalize_failed(result: ForwardResult) -> None:
    """Scrub the transients a failed forward pass left on its trace.

    Parameters
    ----------
    result:
        Forward result on the failed arm.
    """
    _scrub_failed_capture_transients(result.trace)


def finalize_interrupted(
    result: ForwardResult,
    interrupt_exc: BaseException,
    halt_exc: StopSignal,
) -> None:
    """Stamp an interrupt that escaped the halted arm, never masking it.

    Parameters
    ----------
    result:
        Forward result carrying the trace and capture session.
    interrupt_exc:
        The interrupt in flight.
    halt_exc:
        The stop signal of the halted arm.
    """
    _settle_interrupted_halted_arm(result.trace, result.session, interrupt_exc, halt_exc)
