"""Bounded cards for backward records, source frames and Trace concern views."""

from __future__ import annotations

import os
from typing import Any

from ._display_bounded import short_dataclass_repr
from ._display_card import REPR_COLS, inline_names, one_line, render_card
from ._display_numbers import format_duration, format_iec_bytes, plural
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
            [record.class_name, plural(len(record.calls), "call")],
        )
    if kind == "GradFnCall":
        return one_line(
            f"GradFnCall {record.label}", [f"backward pass {record.backward_pass_number}"]
        )
    return one_line(
        f"BackwardPass {record.pass_number}",
        [record.trigger, record.status, plural(len(record.grad_fn_calls), "call")],
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
            ("implicit", "implicit backward" if record.implicit else "explicit backward"),
            ("roots", str(len(record.root_grad_fn_ids))),
        ]
        more = (".grad_fn_calls", ".root_meta", ".engine_flags")
    return render_card(backward_repr(record), rows, more=more)


def frame_repr(frame: Any) -> str:
    return fail_open(
        lambda: one_line("FuncCallLocation", [_frame_location(frame)]),
        lambda _error: "FuncCallLocation: metadata unavailable",
    )


def _frame_location(frame: Any) -> str:
    path = _source_path(str(frame.file))
    suffix = f":{frame.line_number} in {frame.func_name}"
    location = f"{path}{suffix}"
    budget = REPR_COLS - len("FuncCallLocation: ")
    if len(location) > budget:
        location = f"{'/'.join(path.rsplit('/', 2)[-2:])}{suffix}"
    return location if len(location) <= budget else "..." + location[-(budget - 3) :]


def _source_path(file: str) -> str:
    """Show installed files from their package root, local files relative to cwd."""

    normalized = file.replace("\\", "/")
    for marker in ("/site-packages/", "/dist-packages/"):
        if marker in normalized:
            return normalized.rsplit(marker, 1)[1]
    try:
        relative = os.path.relpath(file)
    except ValueError:
        return file
    return file if relative.startswith("..") else relative


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


def group_repr(group: Any, summary: Any) -> str:
    """Identity and stored scalar leaves only; group properties may run analysis."""

    return fail_open(
        lambda: _group_identity(group, summary),
        lambda _error: f"{type(group).__name__}: metadata unavailable",
    )


def _group_identity(group: Any, summary: Any) -> str:
    trace = group._trace
    name = type(group).__name__
    if name == "TraceCost":
        return one_line(
            f"TraceCost of {trace.name or 'unnamed'}",
            [],
            [
                f"{format_duration(float(trace._cost_func_calls_duration))} compute",
                f"{format_iec_bytes(trace._cost_saved_activation_memory)} saved",
            ],
        )
    facts = _group_facts(name, trace, summary)
    return one_line(f"{name} of {trace.name or 'unnamed'}", facts)


def _group_facts(name: str, trace: Any, summary: Any) -> list[str]:
    """Reuse Trace wording through the record's existing report door."""

    # Keep the concern dispatch cohesive so every group shares the same
    # stored-fact policy; its essential branches exceed the usual CC target.
    if name == "TraceGraph":
        return [summary._graph_text(trace)]
    if name == "TraceHealth":
        return [summary._health_text(trace)]
    if name == "TraceAutograd":
        return ["backward-ready" if trace._options_backward_ready else "not backward-ready"]
    if name == "TraceSource":
        return [trace._source_class_name or "model unknown"]
    facts: list[str] = []
    for key in _GROUP_FIELDS.get(name, ()):
        value = getattr(trace, key, None)
        if isinstance(value, bool):
            facts.append("verified" if value else "unverified")
            continue
        if isinstance(value, (str, int, float, bool)):
            label = key.split("_", 2)[-1]
            facts.append(str(value) if label in ("mode", "backend") else f"{label} {value}")
    return facts


def group_str(group: Any, summary: Any) -> str:
    """One headline row and the existing concern attributes, without running analysis."""

    return fail_open(lambda: _group_card(group, summary), lambda _error: group_repr(group, summary))


def _group_card(group: Any, summary: Any) -> str:
    name, trace = type(group).__name__, group._trace
    if name == "TraceCost":
        rows = [
            ("compute", format_duration(float(trace._cost_func_calls_duration))),
            ("saved", format_iec_bytes(trace._cost_saved_activation_memory)),
        ]
    else:
        rows = [("status", ", ".join(_group_facts(name, trace, summary)))]
    attributes = {
        "TraceCost": (".forward_duration", ".saved_activation_memory"),
        "TraceGraph": (".edges", ".is_recurrent", ".has_conditional_branching"),
        "TraceHealth": (".completeness_verified", ".escape_detector_verified"),
        "TraceAutograd": (".has_backward_pass", ".backward_passes"),
        "TraceSource": (".class_name", ".class_location"),
        "TraceSettings": (".mode", ".seed"),
        "TraceProvenance": (".backend", ".verified"),
    }
    return render_card(group_repr(group, summary), rows, more=attributes.get(name, ()))


_EVENT_LANES = (
    ("op_events", "op event", "ops"),
    ("aten_events", "aten event", "aten"),
    ("module_enter_events", "module enter", "enters"),
    ("module_exit_events", "module exit", "exits"),
    ("backward_events", "backward event", "backward"),
)


def capture_events_repr(events: Any) -> str:
    """Journal lane sizes only; never traverse an event or retained payload."""

    return fail_open(
        lambda: one_line(
            "CaptureEvents",
            [plural(len(events.op_events), "op event")],
            [plural(len(getattr(events, field)), noun) for field, noun, _ in _EVENT_LANES[1:]],
        ),
        lambda _error: "CaptureEvents: metadata unavailable",
    )


def capture_events_str(events: Any) -> str:
    return fail_open(
        lambda: render_card(
            capture_events_repr(events),
            [
                (label, plural(len(getattr(events, field)), noun))
                for field, noun, label in _EVENT_LANES
            ],
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
