"""Derived gradients of an edited tinygrad capture.

``trace(intervene=..., grad_options=GradOptions(...))`` differentiates the
program the capture recorded: the forward with the edit applied at the same
Tensor-API calls the edited capture edited. The gradient run goes through the
backend's own derived-gradient path (its output check against the captured
output, its ``.grad`` bracketing and its parameter mirroring) with the model
wrapped so that the edit is re-applied at those call numbers. Input gradients
and intermediate gradients are therefore tinygrad's gradients of the edited
program, never of the unedited model.

An intermediate gradient is attached to an op when exactly one Tensor-API call
returned it; at an edited op it is the gradient at the edited value (the value
the trace stores and downstream ops read), as on PyTorch.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from ..._data_classes.derived_grad import (
    DerivedGradAccessor,
    IntermediateDerivedGradAccessor,
    IntermediateDerivedGradRecord,
    set_intermediate_derived_grads,
)
from ..._ir.refs import DtypeRef
from ._derived_grads import _differentiated_input_leaves, _normalize_tinygrad_input_grad_argnums
from ._uop_graph import _uop_name

if TYPE_CHECKING:
    from ._edits import TinygradEditSession

__tl_layer__ = "L2"


class _EditedGradientRun:
    """Re-apply an edit at fixed call numbers while the gradient forward runs.

    It stands in for the capture observer: the edit session pauses it while an
    edit builds its replacement, so the edit's own calls are not numbered,
    exactly as during the capture.
    """

    def __init__(self, session: TinygradEditSession) -> None:
        """Prepare a run that edits the calls ``session`` names."""

        self.session = session
        self.results: list[Any] = []
        self.paused = False
        self._original: Any = None

    def alias(self, site: Any, replacement: Any) -> None:
        """Nothing to alias: the gradient run records no UOp names."""

        del site, replacement

    def __enter__(self) -> _EditedGradientRun:
        """Install the numbering and editing wrapper on ``Tensor._apply_uop``."""

        from tinygrad import Tensor

        self._original = Tensor._apply_uop
        original = self._original

        def wrapped(tensor: Any, fxn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
            """Number one call and return its (possibly edited) result."""

            result = original(tensor, fxn, *args, **kwargs)
            if self.paused:
                return result
            name = str(getattr(fxn, "__name__", _uop_name(result.uop).lower()))
            returned = self.session.apply(self, result, name)
            self.results.append(returned)
            return returned

        # The capture observer replaces the same tinygrad method the same way.
        Tensor._apply_uop = wrapped  # type: ignore[method-assign]
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Restore ``Tensor._apply_uop``."""

        from tinygrad import Tensor

        Tensor._apply_uop = self._original  # type: ignore[method-assign]


def _edited_model(model: Callable[..., Any], run: _EditedGradientRun) -> Callable[..., Any]:
    """``model`` with ``run`` editing its forward."""

    def edited(*args: Any, **kwargs: Any) -> Any:
        """Run the forward with the edit applied."""

        run.results.clear()
        with run:
            return model(*args, **kwargs)

    return edited


def _call_numbers_by_label(trace: Any, session: TinygradEditSession) -> dict[str, list[int]]:
    """``raw label -> call numbers`` that returned each captured op in the edited capture."""

    numbers: dict[int, list[int]] = {}
    for number, (uop, _name) in enumerate(session.calls):
        numbers.setdefault(id(uop), []).append(number)
    for replacement_id, site_uop in session.replaced.items():
        numbers[replacement_id] = numbers.get(id(site_uop), [])
    return {
        capture.label_raw: numbers.get(id(capture.uop), [])
        for capture in trace.tinygrad_uop_captures
    }


def _intermediate_record(backend: Any, op: Any, grad: Any) -> IntermediateDerivedGradRecord:
    """The intermediate gradient record of one op."""

    copied = backend._realized_copy(grad)
    dtype = str(getattr(copied, "dtype", ""))
    return IntermediateDerivedGradRecord(
        op_label=op.label,
        layer_label=op.layer_label,
        aval=f"Tensor(shape={tuple(getattr(copied, 'shape', ()))}, dtype={dtype})",
        dtype_ref=DtypeRef(backend="tinygrad", name=dtype),
        grad=copied,
        provenance={
            "backend": "tinygrad",
            "kind": "intermediate_derived_gradient",
            "mechanism": "tinygrad_edited_program_backward",
            "status": "exact",
        },
    )


def _intermediate_records(
    backend: Any, trace: Any, by_label: Mapping[str, list[int]], results: Sequence[Any]
) -> dict[str, IntermediateDerivedGradRecord]:
    """Gradient records for saved non-input ops returned by exactly one call."""

    records: dict[str, IntermediateDerivedGradRecord] = {}
    for op in trace._layer_list:
        found = by_label.get(str(op._label_raw), [])
        if not op.has_saved_activation or op.is_input or len(found) != 1:
            continue
        grad = getattr(results[found[0]], "grad", None)
        if grad is not None:
            records[op.label] = _intermediate_record(backend, op, grad)
    return records


def _leaf_options(backend: Any, args: Sequence[Any], grad_options: Any) -> Any:
    """The options of the leaf-gradient run (intermediates come from the same run)."""

    options = dataclasses.replace(grad_options, intermediate_grads=False)
    if not grad_options.intermediate_grads:
        return options
    argnums = _normalize_tinygrad_input_grad_argnums(grad_options.input_grad_argnums, len(args))
    if _differentiated_input_leaves(backend, args, argnums):
        return options
    # Intermediates only: differentiate every input so the backend runs one backward.
    return dataclasses.replace(options, input_grad_argnums=None)


def attach_edited_derived_grads(
    backend: Any,
    trace: Any,
    call: Mapping[str, Any],
    capture_session: TinygradEditSession,
    replay_session: TinygradEditSession,
) -> None:
    """Attach derived gradients of the edited program to an edited capture.

    Parameters
    ----------
    backend
        The ``TinygradBackend`` that produced ``trace``.
    trace
        The edited capture.
    call
        The ``capture_trace`` keyword arguments of the capture; its
        ``grad_options`` is the tinygrad ``GradOptions``.
    capture_session
        The edit session of the edited capture (its call numbers).
    replay_session
        A fresh session that edits the same call numbers with the same edit.

    Raises
    ------
    ValueError
        When the gradient run's output differs from the edited capture's
        output, or the loss is not scalar (the backend's derived-gradient
        checks).
    """

    grad_options = call["grad_options"]
    run = _EditedGradientRun(replay_session)
    args = backend._normalize_input_args(call["input_args"])
    leaf_options = _leaf_options(backend, args, grad_options)
    backend._attach_derived_grads(
        trace=trace,
        model=_edited_model(call["model"], run),
        args=args,
        captured_output=trace._output_slot.value,
        grad_options=leaf_options,
    )
    if leaf_options.input_grad_argnums != grad_options.input_grad_argnums:
        trace._autograd_derived_grads = DerivedGradAccessor({})
    if grad_options.intermediate_grads:
        by_label = _call_numbers_by_label(trace, capture_session)
        records = _intermediate_records(backend, trace, by_label, run.results)
        set_intermediate_derived_grads(trace, IntermediateDerivedGradAccessor(records))
