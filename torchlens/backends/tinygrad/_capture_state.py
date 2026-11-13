"""Per-capture state of the tinygrad backend: call keywords, source lines, edits.

Moved out of ``backend`` so that module stays under its size ceiling. A
capture sets the keyword arguments it calls the model with and a fresh
source-line recorder (``_source_lines``); an edited capture's runs also carry
an edit session (``_edits``). The functions here read that state when the
capture realizes its values and emits its events.
"""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import Any, cast

from ..._ir.events import ModuleFrame, OpEvent
from ._captures import TinygradCaptureSession, TinygradUOpCapture
from ._derived_grads import call_input_leaves
from ._edits import fire_amendment
from ._module_tree import TinygradModuleFrame
from ._realize_once import RealizedCapture, realize_capture
from ._source_lines import SourceLineRecorder

__tl_layer__ = "L2"


def begin_capture(
    backend: Any,
    input_args: object,
    input_kwargs: Any,
    num_context_lines: object,
    save_code_context: object,
) -> list[Any]:
    """Set the per-capture call keywords and source-line recorder; normalize the args.

    Parameters
    ----------
    backend
        The ``TinygradBackend`` starting a capture.
    input_args
        User-supplied positional input args.
    input_kwargs
        User-supplied keyword input args, or ``None``.
    num_context_lines
        Source lines kept on each side of each recorded line.
    save_code_context
        Whether recorded source text is snapshotted at capture.

    Returns
    -------
    list[Any]
        Positional argument list.
    """

    backend._call_kwargs = MappingProxyType(dict(input_kwargs or {}))
    backend._capture_session = TinygradCaptureSession()
    backend._source_lines = SourceLineRecorder(
        type(backend)._run_observed_forward.__code__,
        cast(int, num_context_lines),
        bool(save_code_context),
    )
    return cast(list[Any], backend._normalize_input_args(input_args))


def realize_with_state(
    backend: Any, args: list[Any], outputs: tuple[Any, ...], preexisting_uops: tuple[Any, ...]
) -> RealizedCapture:
    """Plan and realize the capture once, with the call's keywords and any edits.

    Parameters
    ----------
    backend
        The capturing ``TinygradBackend``.
    args
        Normalized positional arguments.
    outputs
        Flat tinygrad output tensors of the forward.
    preexisting_uops
        UOps live tensors held before the forward ran.

    Returns
    -------
    RealizedCapture
        The realized capture; unobserved recorded UOps get source lines.
    """

    session = backend._edit_session
    realized = realize_capture(
        call_input_leaves(args, backend._call_kwargs),
        outputs,
        preexisting_uops,
        backend.is_tensor,
        edit_sites=session.replaced if session is not None else None,
        session=backend._capture_session,
    )
    if backend._source_lines is not None:
        backend._source_lines.fill_unobserved(realized.plan.recorded, realized.plan.edit_sites)
    return realized


def mark_edit(
    backend: Any, trace: Any, event: OpEvent, model_uop: Any, capture: TinygradUOpCapture
) -> TinygradUOpCapture:
    """Mark the event and capture of an op an edit replaced; others pass through.

    Parameters
    ----------
    backend
        The capturing ``TinygradBackend``.
    trace
        Trace receiving events.
    event
        The op's appended event.
    model_uop
        The UOp the model knows the op's value by (an edit's replacement).
    capture
        The op's validation capture.

    Returns
    -------
    TinygradUOpCapture
        ``capture``, carrying the edit when one replaced the op.
    """

    session = backend._edit_session
    edit = session.edit_for(model_uop) if session is not None else None
    if edit is None or session is None:
        return capture
    trace.capture_events.append_amendment(fire_amendment(event, edit, session.selector))
    return replace(capture, edit=edit.apply)


def event_module_frames(
    module_stack: tuple[TinygradModuleFrame, ...],
) -> tuple[tuple[ModuleFrame, ...], tuple[tuple[str, int], ...]]:
    """The event module stack and ``(address, call_index)`` pairs of a tinygrad stack.

    Parameters
    ----------
    module_stack
        Active tinygrad object-module stack of one op.

    Returns
    -------
    tuple[tuple[ModuleFrame, ...], tuple[tuple[str, int], ...]]
        The event's ``module_stack`` and ``modules`` fields.
    """

    frames = tuple(
        ModuleFrame(
            address=frame.address,
            address_normalized=frame.address,
            module_type=frame.module_type,
            call_index=frame.call_index,
            fx_qualpath=None,
            entry_argnames=(),
        )
        for frame in module_stack
    )
    return frames, tuple((frame.address, frame.call_index) for frame in module_stack)
