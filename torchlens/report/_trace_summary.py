"""Trace summary, HTML repr and first-nonfinite functions (record rebuild destination).

Bodies of the ``Trace`` summary verbs live here; ``Trace`` keeps one thin method
per verb that forwards to these functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from .._data_classes._nonfinite import coverage_gap_note, first_nonfinite_layer
from .._errors import CaptureNotActiveError, InvalidArgumentError
from .._source_links import file_line_text, terminal_file_line_link, vscode_file_line_link
from ._notebook.cards import BRIDGE_SENTINEL_HTML, trace_repr_html
from ._summary_config import resolve_config
from ._summary_report import SummaryReport
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
        stack = getattr(layer, "code_context", None) or []
        location = "source unavailable"
        if stack:
            frame = stack[0]
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

    card = trace_repr_html(trace)
    if card == BRIDGE_SENTINEL_HTML:
        return card
    if not trace._tracing_finished:
        return card
    try:
        report = trace.summary()
        summary_html = report._repr_html_() if isinstance(report, SummaryReport) else ""
    except Exception:  # noqa: BLE001 - a repr must never raise; the card still renders
        summary_html = ""
    return card + summary_html


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
