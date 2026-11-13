"""Trace summary, HTML repr and first-nonfinite functions (record rebuild destination).

Bodies of the ``Trace`` summary verbs live here; ``Trace`` keeps one thin method
per verb that forwards to these functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from .._capture.structure_only import claim_status_for
from .._data_classes._nonfinite import (
    _unmaterialized_disk_payload_count,
    coverage_gap_note,
    first_nonfinite_layer,
)
from .._data_classes._value_state import LAZY_DISK
from .._data_classes.func_call_location import display_frame
from .._errors import CaptureNotActiveError, InvalidArgumentError
from .._source_links import (
    file_line_text,
    terminal_file_line_link,
    vscode_file_line_link,
)
from .._utils._display_card import REPR_COLS, TRACE_CARD_MAX_ROWS, one_line, render_card
from .._utils._display_numbers import (
    format_count,
    format_duration,
    format_exact_count,
    format_iec_bytes,
    plural,
)
from .._utils._record_group_table import record_member
from .._utils.fail_open import fail_open
from ._factcore import _FACTCORE_CACHE, capture_fingerprint
from ._health import _health_text as _verdict_text, _parse_persisted
from ._notebook._axis_labels import sdpa_attention_hint
from ._notebook.cards import trace_repr_html
from ._summary_config import resolve_config
from ._summary_result import build_rebuilt_summary
from ._topology_text import topology_text

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
        frame = display_frame(record_member(layer, "code_context", None))
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


_MORE = (".summary()", ".modules", ".layers", ".ops", ".params", ".output_layers")


def _stored_facts(trace: Any) -> Any:
    """Read a previously settled health record; never derive or persist one."""

    payload = getattr(trace, "annotations", {}).get("health_facts")
    if payload is None:
        return None
    revision = capture_fingerprint(trace)
    facts = _parse_persisted(payload, revision)
    return facts if facts.capture_revision == revision else None


def _stored_verdict(trace: Any) -> str:
    """Read a previously settled verdict; never derive or persist a health record."""

    facts = _stored_facts(trace)
    return "not_checked" if facts is None else facts.verdict


def _coverage_text(trace: Any) -> str:
    """After a settled partial scan: what it examined and why the rest was not.

    The gap note is the one ``explain()`` and ``first_nonfinite()`` print, counted
    from payload metadata; no payload is read and nothing is scanned here.
    """

    facts = fail_open(lambda: _stored_facts(trace), lambda _error: None)
    if facts is None or facts.verdict != "not_checked" or facts.basis == "persisted_invalid":
        return ""
    note = coverage_gap_note(trace, kind="saved").strip()
    if not note:
        return ""
    checked = facts.num_checked_ops
    examined = f"{plural(checked, 'op')} examined, none non-finite" if checked else "no op examined"
    return f"{examined} {note}"


def _record_kind(op: Any) -> str:
    """Which population of the card an op record counts in."""

    if op.is_input:
        return "inputs"
    if op.is_output:
        return "outputs"
    return "buffers" if op.graph.is_buffer else "ops"


def _population(trace: Any) -> dict[str, int]:
    """One count per kind of record: ops, the layers they group into, buffers, inputs, outputs.

    ``ops`` are the computing ops (inputs, outputs and buffer reads excluded), ``layers``
    the layers those ops form, ``passes`` the most passes of any of them and ``saved`` how
    many of their outputs are saved.
    """

    counts = dict.fromkeys(("ops", "passes", "saved", "buffers", "inputs", "outputs"), 0)
    layers: set[str] = set()
    for op in trace._layer_list:
        kind = _record_kind(op)
        counts[kind] += 1
        if kind == "ops":
            layers.add(str(op.layer_label))
            counts["passes"] = max(counts["passes"], int(op.num_passes or 1))
            counts["saved"] += bool(getattr(op, "has_saved_activation", False))
    counts["layers"] = len(layers)
    return counts


def _ops_in_layers_text(counts: dict[str, int], *, compact: bool) -> str:
    """``69 ops in 61 layers (up to 2 passes)``: the layer/op split in one clause."""

    text = (
        f"{_count_text(counts['ops'], 'op', compact=compact)} in "
        f"{_count_text(counts['layers'], 'layer', compact=compact)}"
    )
    if counts["passes"] > 1:
        text += f" (up to {format_exact_count(counts['passes'])} passes)"
    return text


def _saved_text(trace: Any) -> str:
    """Saved op outputs and their bytes, then the other records; disk payloads keep their token."""

    counts = _population(trace)
    others = ", ".join(
        _count_text(counts[key], noun) for key, noun in _OTHER_RECORDS if counts[key]
    )
    text = (
        f"{format_exact_count(counts['saved'])} of {format_exact_count(counts['ops'])} op outputs, "
        f"{format_iec_bytes(trace._cost_saved_activation_memory)}"
    )
    if others:
        text += f"; {others}"
    disk = fail_open(
        lambda: _unmaterialized_disk_payload_count(trace, kind="saved"), lambda _error: 0
    )
    return f"{text} {LAZY_DISK}" if disk else text


def _health_text(trace: Any) -> str:
    """NaN/Inf health in words from the stored verdict, never running a check."""

    verdict = fail_open(lambda: _stored_verdict(trace), lambda _error: "not_checked")
    return _verdict_text(verdict)


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
    if trace._options_structure_only:
        # Existing claim property is metadata-only; never execute a probe.
        tokens.append(f"(structure_only={claim_status_for(trace).value.upper()})")
    if trace._provenance_verified is False:
        tokens.append("(capture unverified; counts are lower bounds)")
    armed = fail_open(trace._armed_summary_line, lambda _error: None)
    if armed:
        tokens.append(f"({armed})")
    return tokens


def _trace_honesty_suffix(trace: Any) -> str:
    """The non-default state tokens followed by the health text."""

    return " ".join((*_state_tokens(trace), _health_text(trace)))


def _trace_repr(trace: Any) -> str:
    """One-line identity and cached health; safe on partial and cleaned traces."""

    return fail_open(lambda: _identity(trace), lambda _error: "Trace (metadata unavailable)")


def _identity(trace: Any) -> str:
    """Build the Trace repr line, dropping model and counts before any state or health marker."""

    if trace.__dict__.get("_tl_cleaned_up", False):
        return f"Trace {getattr(trace, 'name', None) or 'unnamed'}: (freed)"
    name = getattr(trace, "name", None) or "unnamed"
    model = trace._source_class_name or "model unknown"
    states = " ".join(_state_tokens(trace))
    identity = f"Trace {name}"
    health = _health_text(trace)
    progress = "(capture in progress)" if not trace._tracing_finished else ""
    # Drop optional metadata in order, keeping every honesty marker visible.
    counts = _count_facts(trace)
    for optional in ([model, *counts], counts, []):
        suffix = " ".join((progress, states)).strip()
        details = " ".join((", ".join(optional), suffix)).strip()
        line = one_line(identity, [details, health])
        if len(line) <= REPR_COLS or not optional:
            return line
    return identity


def _count_facts(trace: Any) -> list[str]:
    """Op and layer counts for a finished trace, or the op event count while capture runs."""

    if not trace._tracing_finished:
        return [plural(len(trace.event_stream.op_events), "op event")]
    return [_ops_in_layers_text(_population(trace), compact=True)]


#: The records the saved row counts after the op outputs, in order.
_OTHER_RECORDS = (("buffers", "buffer"), ("inputs", "input"), ("outputs", "output"))


def _count_text(count: int, noun: str, *, compact: bool = False) -> str:
    """Use the shared count precision and plural rules for a named count."""

    number = format_count(count) if compact else format_exact_count(count)
    return f"{number} {plural(count, noun).partition(' ')[2]}"


def _trace_rows(trace: Any) -> list[tuple[str, str]]:
    """The shared text/HTML rows; all reads are metadata-only."""

    if trace.__dict__.get("_tl_cleaned_up", False):
        return []
    if not trace._tracing_finished:
        return [("captured", f"{plural(len(trace.event_stream.op_events), 'op event')} so far")]
    modules = max(0, len(trace._module_logs) - 1)
    params = _count_text(getattr(trace, "num_params", 0), "param")
    param_tensors = _count_text(trace._num_param_tensors, "tensor")
    duration = float(trace._provenance_duration)
    compute = float(trace._cost_func_calls_duration)
    rows = [
        (
            "model",
            f"{_count_text(modules, 'submodule')}, {params} in {param_tensors}",
        ),
        (
            "captured",
            f"{_captured_text(duration, compute)}, seed {trace._options_seed}",
        ),
        ("graph", _graph_text(trace)),
        ("saved", _saved_text(trace)),
    ]
    rows.extend(_coverage_rows(trace))
    cached = _FACTCORE_CACHE.get(id(trace))
    if cached is not None and cached[0]() is trace:
        aggregation = cached[1].compute
        rows.append(
            ("flops", f"{aggregation.partition_total!s} forward, {aggregation.macs_total!s}")
        )
    if sdpa_attention_hint(trace):
        rows.append(
            (
                "note",
                "attention weights not materialized (fused attention); "
                "re-capture with attn_implementation='eager'",
            )
        )
    if trace._provenance_verified is False:
        rows.insert(
            0,
            (
                "capture",
                f"unverified: {trace._provenance_verification_reason or 'reason unavailable'}; counts are lower bounds",
            ),
        )
        rows.insert(1, ("bounds", _lower_bounds(trace)))
    return rows


def _lower_bounds(trace: Any) -> str:
    """Unverified capture counts remain explicit lower bounds in the card."""

    return (
        f"at least {plural(len(trace), 'total tensor')}, "
        f"at least {plural(trace._num_layers_with_params, 'parameter operation')}, "
        f"at least {plural(trace.num_params, 'param')} total"
    )


def _coverage_rows(trace: Any) -> list[tuple[str, str]]:
    """Settled scan coverage without reading payloads; the saved row carries disk storage."""

    text = _coverage_text(trace)
    if not text:
        facts = fail_open(lambda: _stored_facts(trace), lambda _error: None)
        unexamined = getattr(facts, "num_unexamined_ops", 0) if facts is not None else 0
        text = f"{plural(unexamined, 'output')} could not be examined" if unexamined else ""
    return [("coverage", text)] if text else []


def _captured_text(duration: float, compute: float) -> str:
    """Describe capture overhead using the same precision as the duration."""

    total = format_duration(duration)
    logging = format_duration(max(0.0, duration - compute))
    if duration > 0 and logging == total:
        return f"{total}, almost all logging"
    return f"{total} ({logging} of it logging)"


def _graph_text(trace: Any) -> str:
    """Graph topology in words: recurrence or shared modules, tied weights, branching."""

    graph = trace.graph
    topology = topology_text(trace)
    if trace._provenance_verified is False:
        topology = "structure unverified"
    branching = "with branching" if graph.is_branching else "no branching"
    conditional = (
        "with conditional branching"
        if graph.has_conditional_branching
        else "no conditional branching"
    )
    return f"{topology}, {branching}, {conditional}"


def _trace_text(trace: Any) -> str:
    """Trace card: the repr line over the shared rows; the repr replaces it on failure."""

    return fail_open(
        lambda: render_card(
            _trace_repr(trace), _trace_rows(trace), more=_MORE, max_rows=TRACE_CARD_MAX_ROWS
        ),
        lambda _error: _trace_repr(trace),
    )
