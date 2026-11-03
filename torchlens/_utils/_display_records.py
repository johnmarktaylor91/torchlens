"""Bounded cards for backward records, source frames and Trace concern views."""

from __future__ import annotations

from typing import Any

from ._display_bounded import short_dataclass_repr
from ._display_card import REPR_COLS, inline_names, one_line, render_card
from ._display_numbers import format_count, format_duration, format_iec_bytes
from ._display_text import tensor_text
from .fail_open import fail_open


def backward_repr(record: Any) -> str:
    """Read identity and counts, without traversing saved gradients."""

    return fail_open(
        lambda: _backward_identity(record),
        lambda _error: f"{type(record).__name__}: metadata unavailable",
    )


def _backward_identity(record: Any) -> str:
    kind = type(record).__name__
    if kind == "GradFn":
        return one_line(
            f"GradFn {record.label}",
            [record.class_name, f"{format_count(len(record.calls))} calls"],
        )
    if kind == "GradFnCall":
        return one_line(
            f"GradFnCall {record.label}", [f"backward pass {record.backward_pass_number}"]
        )
    return one_line(
        f"BackwardPass {record.pass_number}",
        [record.trigger, record.status, f"{format_count(len(record.grad_fn_calls))} calls"],
    )


def backward_str(record: Any) -> str:
    """Keep gradient payloads behind explicit accessors."""

    return fail_open(lambda: _backward_card(record), lambda _error: backward_repr(record))


def _backward_card(record: Any) -> str:
    kind = type(record).__name__
    if kind == "GradFn":
        rows = [
            ("parents", inline_names(record.parents)),
            ("children", inline_names(record.children)),
            ("module", record.module_address or "self"),
        ]
        more = (".calls", ".parents", ".children")
    elif kind == "GradFnCall":
        rows = [("function", record.grad_fn_label), ("ordinal", str(record.ordinal))]
        more = (".grad_inputs", ".grad_outputs", ".fire_record")
    else:
        rows = [
            ("trigger", record.trigger),
            ("implicit", str(record.implicit)),
            ("roots", str(len(record.root_grad_fn_ids))),
        ]
        more = (".grad_fn_calls", ".root_meta", ".engine_flags")
    return render_card(backward_repr(record), rows, more=more)


def frame_repr(frame: Any) -> str:
    return fail_open(
        lambda: one_line("FuncCallLocation", [_frame_location(frame)], [f"in {frame.func_name}"]),
        lambda _error: "FuncCallLocation: metadata unavailable",
    )


def _frame_location(frame: Any) -> str:
    path = str(frame.file).replace("\\", "/")
    location = f"{path}:{frame.line_number}"
    budget = REPR_COLS - len("FuncCallLocation: ")
    if len(location) > budget:
        location = f"{'/'.join(path.rsplit('/', 2)[-2:])}:{frame.line_number}"
    return location if len(location) <= budget else "..." + location[-(budget - 3) :]


def frame_str(frame: Any) -> str:
    return fail_open(
        lambda: render_card(
            frame_repr(frame),
            [
                ("file", str(frame.file)),
                ("line", str(frame.line_number)),
                ("function", str(frame.func_name)),
            ],
            more=(".code_context", ".source_context"),
        ),
        lambda _error: frame_repr(frame),
    )


_GROUP_FIELDS = {
    "TraceCost": ("_cost_func_calls_duration", "_cost_saved_activation_memory"),
    "TraceGraph": ("_graph_is_recurrent", "_graph_is_branching"),
    "TraceAutograd": ("_options_backward_ready",),
    "TraceProvenance": ("_provenance_backend", "_provenance_verified"),
    "TraceSettings": ("_options_mode", "_options_seed"),
    "TraceSource": ("_source_class_name",),
    "TraceHealth": ("_health_completeness_verified", "_health_escape_detector_verified"),
}


def group_repr(group: Any) -> str:
    """Identity and stored scalar leaves only; group properties may run analysis."""

    return fail_open(
        lambda: _group_identity(group),
        lambda _error: f"{type(group).__name__}: metadata unavailable",
    )


def _group_identity(group: Any) -> str:
    trace = group._trace
    name = type(group).__name__
    if name == "TraceCost":
        return one_line(
            f"TraceCost of {trace.name or 'unnamed'}",
            [],
            [
                f"{format_duration(trace._cost_func_calls_duration)} compute",
                f"{format_iec_bytes(trace._cost_saved_activation_memory)} saved",
            ],
        )
    facts = []
    for key in _GROUP_FIELDS.get(name, ()):
        value = getattr(trace, key, None)
        if isinstance(value, (str, int, float, bool)):
            facts.append(f"{key.split('_', 2)[-1]} {value}")
    return one_line(f"{name} of {trace.name or 'unnamed'}", facts)


_EVENT_LANES = (
    ("op_events", "ops"),
    ("aten_events", "aten calls"),
    ("module_enter_events", "module enters"),
    ("module_exit_events", "module exits"),
    ("backward_events", "backward events"),
)


def capture_events_repr(events: Any) -> str:
    """Journal lane sizes only; never traverse an event or retained payload."""

    return fail_open(
        lambda: one_line(
            "CaptureEvents",
            [f"{format_count(len(events.op_events))} ops"],
            [
                f"{format_count(len(getattr(events, field)))} {noun}"
                for field, noun in _EVENT_LANES[1:]
            ],
        ),
        lambda _error: "CaptureEvents: metadata unavailable",
    )


def capture_events_str(events: Any) -> str:
    return fail_open(
        lambda: render_card(
            capture_events_repr(events),
            [(noun, str(len(getattr(events, field)))) for field, noun in _EVENT_LANES],
            more=(".op_events", ".aten_events", ".backward_events", ".live_index"),
        ),
        lambda _error: capture_events_repr(events),
    )


def ir_record_repr(record: Any) -> str:
    if type(record).__name__ == "CaptureEvents":
        return capture_events_repr(record)
    if type(record).__name__ == "_AtenTensorFact":
        return fail_open(
            lambda: one_line(
                "AtenTensorFact", [], [tensor_text(record.dtype, record.shape, record.device)]
            ),
            lambda _error: "AtenTensorFact: metadata unavailable",
        )
    return short_dataclass_repr(record)
