"""Trace summary, HTML repr and first-nonfinite functions (record rebuild destination).

Bodies of the ``Trace`` summary verbs live here; ``Trace`` keeps one thin method
per verb that forwards to these functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from .._capture.structure_only import claim_status_for
from .._data_classes._nonfinite import coverage_gap_note, first_nonfinite_layer
from .._data_classes.func_call_location import display_frame
from .._errors import CaptureNotActiveError, InvalidArgumentError
from .._source_links import file_line_text, terminal_file_line_link, vscode_file_line_link
from .._utils._display_card import TRACE_CARD_MAX_ROWS, one_line, render_card
from .._utils._display_numbers import (
    format_count,
    format_duration,
    format_exact_count,
    format_iec_bytes,
)
from .._utils.fail_open import fail_open
from ._factcore import capture_fingerprint
from ._health import _parse_persisted
from ._notebook.cards import trace_repr_html
from ._summary_config import resolve_config
from ._summary_result import build_rebuilt_summary

if TYPE_CHECKING:
    from .._data_classes.trace import Trace


def first_nonfinite(trace: Trace, link_format: Literal["terminal", "html", "text"] = "text") -> str:
    """Body of ``Trace.first_nonfinite``; the method docstring is the contract."""

    # ``kind="saved"`` skips ops that retained no payload. The raising
    # ``kind="trace"`` gate contradicted this method's own contract ("the first
    # SAVED non-finite out"): on any selective-save capture it hit an unsaved op
    # and raised ValueError, which took ``print(trace)``, ``_repr_html_``, and
    # ``report.explain`` down with it -- in exactly the mode the performance guide
    # recommends for large models. Skipping is honest only because the clean
    # answer below names how many ops could not be examined.
    layer = first_nonfinite_layer(trace, kind="saved")
    if layer is not None:
        frame = display_frame(getattr(layer, "code_context", None))
        location = "source unavailable"
        if frame is not None:
            file_path = str(getattr(frame, "file", "unknown"))
            line_number = getattr(frame, "line_number", "unknown")
            if link_format == "terminal":
                # link_format="terminal" is the EXPLICIT escape opt-in
                # (pinned register contract); the tty auto-gate governs
                # only default/auto call sites.
                location = terminal_file_line_link(file_path, line_number, enable_links=True)
            elif link_format == "html":
                location = vscode_file_line_link(file_path, line_number)
            elif link_format == "text":
                location = file_line_text(file_path, line_number)
            else:
                raise InvalidArgumentError(
                    f"link_format must be 'terminal', 'html', or 'text'; received {link_format!r}",
                    code="link_format_invalid",
                    remedy="pass link_format='terminal', 'html', or 'text'",
                    argument="link_format",
                )
        parents = ", ".join(getattr(layer, "parents", None) or []) or "none"
        module = getattr(layer, "module", None) or "no module"
        return (
            f"First non-finite saved out is in layer {layer.layer_label} "
            f"(op {getattr(layer, 'func_name', 'unknown')}, module {module}), "
            f"shape={getattr(layer, 'shape', None)}, "
            f"dtype={getattr(layer, 'dtype', None)}, parents={parents}, "
            f"source={location}."
        )
    # A scoped clean answer must not read like a whole-capture one: ops that
    # retained no payload, and payloads whose dtype has no runnable ``isfinite``
    # (quantized, sparse), are both named. fp8 is NOT in that class -- it is
    # widened exactly and really is checked.
    return (
        f"No non-finite tensor values found in saved outs{coverage_gap_note(trace, kind='saved')}."
    )


def repr_html(trace: Trace) -> str:
    """Body of ``Trace._repr_html_``; the method docstring is the contract."""

    return trace_repr_html(trace)


def _refuse_one_call_only(**one_call_only: Any) -> None:
    """Refuse the one-call execution options on a summary of an existing capture.

    Parameters
    ----------
    **one_call_only
        ``input_shape``, ``execution_mode`` and ``grad_mode`` as passed.

    Raises
    ------
    InvalidArgumentError
        ``summary_one_call_only`` when any of them is set.
    """

    offending = sorted(name for name, value in one_call_only.items() if value is not None)
    if offending:
        raise InvalidArgumentError(
            f"trace.summary() reports an EXISTING capture; {', '.join(offending)} "
            "only make sense on the one-call door that runs the forward.",
            code="summary_one_call_only",
            remedy="use tl.summary(model, x, ...) for one-call execution options",
        )


def summary(  # noqa: PLR0913 - the ratified public grammar: every axis is a named keyword
    trace: Trace,
    level: str | None = None,
    *,
    # Rebuilt grammar (ratified summary spellings). Spellings
    # DOCUMENTED-UNSTABLE pending naming-session ratification.
    view: str | None = None,
    depth: Any = None,
    columns: list[str] | None = None,
    where: Any = None,
    buffers: str | None = None,
    fold_repeats: Any = None,
    max_rows: int | None = None,
    flop_convention: str | None = None,
    units: str | None = None,
    style: str | None = None,
    max_scan_elements: int | None = None,
    # One-call-only spellings (typed errors here, never no-ops).
    input_shape: Any = None,
    execution_mode: str | None = None,
    grad_mode: str | None = None,
    # Internal plumbing from the one-call door (not public grammar).
    _execution_note: str | None = None,
    _input_synthesis: str | None = None,
    # Anything else refuses typed (removed legacy spellings name their
    # successor; unknown names get the nearest grammar option).
    **unsupported: Any,
) -> str:
    """Body of ``Trace.summary``; the method docstring is the contract."""
    _refuse_one_call_only(
        input_shape=input_shape, execution_mode=execution_mode, grad_mode=grad_mode
    )
    new_kwargs: dict[str, Any] = {}
    for name, value in (
        ("view", view),
        ("depth", depth),
        ("where", where),
        ("buffers", buffers),
        ("fold_repeats", fold_repeats),
        ("flop_convention", flop_convention),
        ("units", units),
        ("style", style),
        ("max_scan_elements", max_scan_elements),
    ):
        if value is not None:
            new_kwargs[name] = value

    config = resolve_config(
        level=level if level is not None else "auto",
        columns=columns,
        **new_kwargs,
        **({"max_rows": max_rows} if max_rows is not None else {}),
        **unsupported,
    )
    if not trace._tracing_finished:  # a cleaned trace raises trace_cleaned_up here
        raise CaptureNotActiveError(
            "trace.summary() cannot run before the forward pass is complete; "
            "a partial capture has no summary (repr(trace) shows the live op count).",
            code="trace_not_finished",
            remedy="call summary() after trace(...) has returned; use repr(trace) mid-forward",
            operation="summary",
        )
    return build_rebuilt_summary(
        trace,
        config,
        execution_note=_execution_note,
        input_synthesis=_input_synthesis,
    )


__all__ = ["first_nonfinite", "repr_html", "summary"]


_MORE = (".summary()", ".modules", ".layers", ".ops", ".params")


def _stored_verdict(trace: Any) -> str:
    """Read a previously settled verdict; never derive or persist a health record."""

    payload = getattr(trace, "annotations", {}).get("health_facts")
    if payload is None:
        return "not_checked"
    revision = capture_fingerprint(trace)
    facts = _parse_persisted(payload, revision)
    return facts.verdict if facts.capture_revision == revision else "not_checked"


def _health_text(trace: Any) -> str:
    verdict = fail_open(lambda: _stored_verdict(trace), lambda _error: "not_checked")
    return {
        "checked_and_clean": "no NaN/Inf (exact)",
        "found": "NaN/Inf FOUND (.nonfinite_ops)",
    }.get(verdict, "NaN/Inf not checked")


def _state_tokens(trace: Any) -> list[str]:
    """Non-default states remain visible even when optional counts do not fit."""

    tokens: list[str] = []
    outcome = getattr(trace, "outcome", None)
    status = getattr(getattr(outcome, "status", None), "value", None)
    if status and status != "complete":
        boundary = getattr(outcome, "boundary_label", None)
        detail = f": stopped at {boundary}" if boundary else ""
        tokens.append(f"({status}{detail})")
    state = getattr(getattr(trace, "state", None), "name", "PRISTINE")
    if trace.is_pristine is False:
        tokens.append("(edited)")
    if state != "PRISTINE":
        tokens.append(f"({state.lower()})")
    if getattr(trace, "_options_structure_only", False):
        # Existing claim property is metadata-only; never execute a probe.
        tokens.append(f"(structure_only={claim_status_for(trace).value.upper()})")
    if getattr(trace, "_provenance_verified", None) is False:
        tokens.append("(capture unverified; counts are lower bounds)")
    armed = fail_open(trace._armed_summary_line, lambda _error: None)
    if armed:
        tokens.append(f"({armed})")
    return tokens


def _trace_honesty_suffix(trace: Any) -> str:
    return " ".join((*_state_tokens(trace), _health_text(trace)))


def _trace_repr(trace: Any) -> str:
    """One-line identity and cached health; safe on partial and cleaned traces."""

    return fail_open(lambda: _identity(trace), lambda _error: "Trace (metadata unavailable)")


def _identity(trace: Any) -> str:
    if trace.__dict__.get("_tl_cleaned_up", False):
        return f"Trace {getattr(trace, 'name', None) or 'unnamed'}: (freed)"
    name = getattr(trace, "name", None) or "unnamed"
    model = getattr(trace, "_source_class_name", None) or "model unknown"
    layers = len(getattr(trace, "_layer_logs", {}) or {})
    ops = getattr(trace, "_num_ops", 0)
    states = " ".join(_state_tokens(trace))
    identity = f"Trace {name}"
    health = _health_text(trace)
    progress = "(capture in progress)" if not getattr(trace, "_tracing_finished", False) else ""
    # Drop optional metadata in order, keeping every honesty marker visible.
    required = [progress, states, health]
    counts = [f"{format_count(layers)} layers", f"{format_count(ops)} ops"]
    for optional in ([model, *counts], counts, []):
        line = one_line(identity, [*optional, *required])
        if len(line) <= 100 or not optional:
            return line
    return identity


def _trace_rows(trace: Any) -> list[tuple[str, str]]:
    """The shared text/HTML rows; all reads are metadata-only."""

    if trace.__dict__.get("_tl_cleaned_up", False):
        return []
    exact = format_exact_count
    modules = len(getattr(trace, "_module_logs", {}) or {})
    params = exact(getattr(trace, "num_params", 0))
    param_ops = exact(getattr(trace, "_num_layers_with_params", 0))
    duration = float(getattr(trace, "_provenance_duration", 0))
    compute = float(getattr(trace, "_cost_func_calls_duration", 0))
    saved = sum(bool(getattr(op, "has_saved_activation", False)) for op in trace._layer_list)
    rows = [
        (
            "model",
            f"{trace._source_class_name}, {exact(modules)} modules, {params} params in {param_ops} param ops",
        ),
        (
            "captured",
            f"{format_duration(duration)} ({format_duration(duration - compute)} of it logging), seed {trace._options_seed}",
        ),
        ("graph", _graph_text(trace)),
        (
            "saved",
            f"{exact(saved)} of {exact(len(trace._layer_list))} layer outputs, {format_iec_bytes(trace._cost_saved_activation_memory)}",
        ),
        ("outputs", f"{exact(len(trace.output_layers))} output sites"),
        ("flops", "MACs reported as FLOPs // 2"),
    ]
    if getattr(trace, "_provenance_verified", None) is False:
        rows.insert(
            0,
            (
                "capture",
                f"unverified: {getattr(trace, '_provenance_verification_reason', 'reason unavailable')}; counts are lower bounds",
            ),
        )
    return rows


def _graph_text(trace: Any) -> str:
    graph = trace.graph
    topology = "recurrent" if graph.is_recurrent else "feedforward"
    if getattr(trace, "_provenance_verified", None) is False:
        topology = "structure unverified"
    branching = "with branching" if graph.is_branching else "no branching"
    conditional = (
        "with conditional branching"
        if graph.has_conditional_branching
        else "no conditional branching"
    )
    return f"{topology}, {branching}, {conditional}"


def _trace_text(trace: Any) -> str:
    return fail_open(
        lambda: render_card(
            _trace_repr(trace), _trace_rows(trace), more=_MORE, max_rows=TRACE_CARD_MAX_ROWS
        ),
        lambda _error: _trace_repr(trace),
    )
