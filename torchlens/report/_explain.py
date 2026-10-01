"""Plain-language reports for completed TorchLens logs."""

from __future__ import annotations

import traceback
from collections import Counter
from collections.abc import Mapping
from typing import Any, Literal

import torch

from .._capture_honesty import (
    capture_advisories,
    capture_verification,
    episode_facts,
    poison_facts,
    refuse_presenter_subject,
)

Audience = Literal["researcher", "practitioner", "auto"]
ExplainFormat = Literal["text", "json"]


def explain(
    log: Any,
    audience: Audience = "auto",
    format: ExplainFormat = "text",
    *,
    max_tokens: int | None = None,
) -> str | dict[str, Any]:
    """Explain a TorchLens log in plain language.

    Parameters
    ----------
    log:
        Completed ``Trace``-like object.
    audience:
        Report style. ``"researcher"`` includes graph-pattern detail,
        ``"practitioner"`` emphasizes operational status, and ``"auto"``
        selects a balanced report.
    format:
        ``"text"`` for the existing prose report or ``"json"`` for a
        structured dictionary.
    max_tokens:
        Optional token budget for the text report (DOCUMENTED-UNSTABLE
        spelling, naming ratification pending). Whole sections are dropped in
        a fixed low-value-first order until the estimate (~4 characters per
        token) fits, and every drop is disclosed in a trailing ``Truncation``
        section. The ``Capture status`` honesty section is never dropped: a
        budget below that floor returns the floor plus a disclosure rather
        than a misleading fragment. Not supported with ``format="json"``
        (that schema is fixed-shape and already minimal).

    Returns
    -------
    str | dict[str, Any]
        Multi-section prose, or the stable flat schema documented below.

    Raises
    ------
    ValueError
        If ``audience``, ``format``, or ``max_tokens`` is not supported.

    Notes
    -----
    The JSON schema is identified by ``schema="torchlens.explain.v1"`` and
    contains these stable snake-case keys: ``schema``, ``audience``,
    ``capture_status``, ``capture_verified``, ``capture_verification_reason``,
    ``rescue_rerun``, ``model_class``, ``layer_count``, ``operation_count``,
    ``saved_tensor_count``, ``total_tensor_count``, ``has_backward_pass``,
    ``exception_type``, ``exception_message``, ``last_completed_op_label``,
    ``last_completed_op_shape``, ``last_completed_op_dtype``,
    ``last_completed_op_device``, ``failing_boundary``, and
    ``first_nonfinite``. Evidence unavailable from the supplied log is reported
    as ``"unknown"`` rather than inferred. ``capture_status`` is the log's
    settled ``CaptureOutcome`` status value (``"partial"`` for a failed
    partial capture), never assumed complete; ``capture_verified`` is the
    tri-state stored fact (``None`` = no ceiling recorded).
    """

    refuse_presenter_subject(log, "tl.report.explain")
    if audience not in {"researcher", "practitioner", "auto"}:
        raise ValueError("audience must be 'researcher', 'practitioner', or 'auto'.")
    if format not in {"text", "json"}:
        raise ValueError("format must be 'text' or 'json'.")
    if max_tokens is not None:
        if format == "json":
            raise ValueError(
                "max_tokens applies to format='text' only: the "
                "torchlens.explain.v1 JSON schema is fixed-shape and already "
                "minimal. Drop max_tokens, or use format='text'."
            )
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError(
                "max_tokens must be a positive integer token budget for the "
                "text report; omit it for the full report."
            )

    if _is_partial_trace(log):
        diagnosis = _partial_diagnosis(log)
        if format == "json":
            return _partial_json(log, audience, diagnosis)
        return _budgeted_report(_partial_sections(diagnosis), max_tokens, drop_order=())

    if format == "json":
        return _full_json(log, audience)

    sections = [
        ("Capture status", _capture_status_lines(log)),
        ("Model summary", _model_summary_lines(log)),
        ("Capture summary", _capture_summary_lines(log)),
        ("Backward summary", _backward_summary_lines(log)),
        ("Anomalies", _anomaly_lines(log)),
        ("Interventions", _intervention_lines(log)),
        ("Notable patterns", _pattern_lines(log, audience=audience)),
    ]
    logged_value_lines = _logged_value_lines(log)
    if logged_value_lines:
        sections.append(("Logged values", logged_value_lines))
    return _budgeted_report(sections, max_tokens, drop_order=_SECTION_DROP_ORDER)


#: Low-value-first order in which ``max_tokens`` drops report sections.
#: ``Capture status`` is deliberately absent: the honesty facts never drop.
_SECTION_DROP_ORDER: tuple[str, ...] = (
    "Logged values",
    "Notable patterns",
    "Interventions",
    "Backward summary",
    "Capture summary",
    "Model summary",
    "Anomalies",
)


def _estimate_tokens(text: str) -> int:
    """Estimate the token count of a report at ~4 characters per token.

    Parameters
    ----------
    text:
        Rendered report text.

    Returns
    -------
    int
        Conservative whole-token estimate (always at least 1).
    """

    return max(1, (len(text) + 3) // 4)


def _partial_sections(diagnosis: dict[str, Any]) -> list[tuple[str, list[str]]]:
    """Return the section structure of a partial-capture report.

    Parameters
    ----------
    diagnosis:
        Evidence fields from :func:`_partial_diagnosis`.

    Returns
    -------
    list[tuple[str, list[str]]]
        Ordered (title, bullet lines) sections; all are budget-undroppable
        because every line is failure evidence.
    """

    return [
        (
            "Capture status",
            [
                "- This is a partial capture; only operations completed"
                " before the failure are known.",
            ],
        ),
        (
            "Failure diagnosis",
            [
                (
                    "- Last completed op: "
                    f"{diagnosis['last_completed_op_label']} "
                    f"(shape={diagnosis['last_completed_op_shape']}, "
                    f"dtype={diagnosis['last_completed_op_dtype']}, "
                    f"device={diagnosis['last_completed_op_device']})."
                ),
                f"- Failing boundary: {diagnosis['failing_boundary']}.",
                (
                    "- Captured exception: "
                    f"{diagnosis['exception_type']}: {diagnosis['exception_message']}"
                ),
                f"- First non-finite evidence: {diagnosis['first_nonfinite']}",
            ],
        ),
    ]


def _render_report(
    sections: list[tuple[str, list[str]]],
    dropped: list[str],
    max_tokens: int | None,
    *,
    below_floor: bool,
) -> str:
    """Render report sections, appending a truncation disclosure if needed.

    Parameters
    ----------
    sections:
        Ordered (title, bullet lines) sections to render.
    dropped:
        Titles of sections dropped to honor the budget.
    max_tokens:
        Requested token budget, for the disclosure line.
    below_floor:
        Whether the undroppable floor still exceeds the budget.

    Returns
    -------
    str
        Rendered report text.
    """

    lines = ["TorchLens report"]
    for title, body in sections:
        lines.extend(["", title, *body])
    if dropped or below_floor:
        lines.extend(["", "Truncation"])
        if dropped:
            lines.append(
                f"- Sections dropped to fit max_tokens={max_tokens} "
                f"(~4 characters/token estimate): {', '.join(dropped)}."
            )
        if below_floor:
            lines.append(
                "- The budget is below the undroppable floor; the capture-"
                "status and failure-evidence facts above are never dropped."
            )
    return "\n".join(lines)


def _budgeted_report(
    sections: list[tuple[str, list[str]]],
    max_tokens: int | None,
    *,
    drop_order: tuple[str, ...],
) -> str:
    """Render a report, dropping whole sections to honor a token budget.

    Parameters
    ----------
    sections:
        Ordered (title, bullet lines) sections.
    max_tokens:
        Token budget, or ``None`` for the full report (byte-identical to the
        historical unbudgeted rendering).
    drop_order:
        Section titles eligible for dropping, lowest-value first.

    Returns
    -------
    str
        Rendered report; any dropped content is disclosed in a trailing
        ``Truncation`` section, never silently omitted.
    """

    if max_tokens is None:
        return _render_report(sections, [], None, below_floor=False)
    dropped: list[str] = []
    while True:
        kept = [(title, body) for title, body in sections if title not in dropped]
        text = _render_report(kept, dropped, max_tokens, below_floor=False)
        if _estimate_tokens(text) <= max_tokens:
            return text
        remaining = [title for title in drop_order if title not in dropped]
        if not remaining:
            return _render_report(kept, dropped, max_tokens, below_floor=True)
        dropped.append(remaining[0])


def _is_partial_trace(log: Any) -> bool:
    """Return whether ``log`` is a :class:`PartialTrace` instance.

    Parameters
    ----------
    log:
        Candidate capture object.

    Returns
    -------
    bool
        Whether the object is a failed partial capture wrapper.
    """

    from ..partial import PartialTrace

    return isinstance(log, PartialTrace)


def _last_partial_op(log: Any) -> Any | None:
    """Return the last recorded completed operation in a partial capture.

    Parameters
    ----------
    log:
        Partial capture wrapper.

    Returns
    -------
    Any | None
        Last raw operation, or ``None`` when none completed.
    """

    raw_layers = tuple(getattr(log, "raw_layers", ()))
    return raw_layers[-1] if raw_layers else None


def _partial_boundary(log: Any) -> str:
    """Return the nearest evidence-backed failure boundary description.

    Parameters
    ----------
    log:
        Partial capture wrapper.

    Returns
    -------
    str
        Recorded exception field or traceback location, otherwise ``"unknown"``.
    """

    exception = log.original_exception
    fields = getattr(exception, "fields", {})
    if isinstance(fields, dict) and (fields.get("layer") or fields.get("op")):
        return f"layer={fields.get('layer', 'unknown')}, op={fields.get('op', 'unknown')}"
    extracted = traceback.extract_tb(exception.__traceback__)
    forward_frames = [frame for frame in extracted if frame.name == "forward"]
    if forward_frames:
        frame = forward_frames[-1]
        return f"forward at {frame.filename}:{frame.lineno}"
    if extracted:
        frame = extracted[-1]
        return f"{frame.name} at {frame.filename}:{frame.lineno}"
    return "unknown"


def _partial_diagnosis(log: Any) -> dict[str, Any]:
    """Collect evidence-backed fields for a failed partial capture.

    Parameters
    ----------
    log:
        Partial capture wrapper.

    Returns
    -------
    dict[str, Any]
        Last-op, boundary, exception, and non-finite evidence.
    """

    op = _last_partial_op(log)
    output = getattr(op, "out", None) if op is not None else None
    label = (
        str(getattr(op, "_label_raw", getattr(op, "_layer_label_raw", "unknown")))
        if op is not None
        else "unknown"
    )
    shape = getattr(op, "shape", None) if op is not None else None
    dtype = getattr(op, "dtype", None) if op is not None else None
    device = getattr(op, "device", None) if op is not None else None
    if isinstance(output, torch.Tensor):
        shape = tuple(output.shape) if shape is None else shape
        dtype = output.dtype if dtype is None else dtype
        device = output.device if device is None else device
    exception = log.original_exception
    return {
        "last_completed_op_label": label,
        "last_completed_op_shape": tuple(shape) if shape is not None else "unknown",
        "last_completed_op_dtype": str(dtype) if dtype is not None else "unknown",
        "last_completed_op_device": str(device) if device is not None else "unknown",
        "failing_boundary": _partial_boundary(log),
        "exception_type": type(exception).__name__,
        "exception_message": str(exception),
        "first_nonfinite": str(log.first_nonfinite()),
    }


def _capture_verification(log: Any) -> dict[str, Any]:
    """Return the capture's verification/outcome facts, never inferred.

    The report layer's honesty contract (report/AGENTS.md) requires a rescued
    or ceilinged capture (``capture_verified=False``) to stay visible in every
    report surface. ``capture_status`` used to be HARDCODED ``"complete"`` for
    every non-partial log (round-7 R67/R88 HIGH), so a HALTED or rescued
    capture explained as clean.

    Parameters
    ----------
    log:
        Capture object.

    Returns
    -------
    dict[str, Any]
        ``capture_status`` (the settled ``CaptureOutcome`` status value, or
        ``"unknown"`` when the log carries none), tri-state
        ``capture_verified`` (``None`` = no ceiling recorded),
        ``capture_verification_reason``, and ``rescue_rerun``.
    """

    # Delegates to the ONE shared source (torchlens._capture_honesty) so
    # explain, to_agent_json, and every exporter preamble read identical facts.
    return capture_verification(log)


def _episode_status_lines(episode: Mapping[str, Any]) -> list[str]:
    """Return the episode-capture disclosure bullets (F40b/F40c families).

    Parameters
    ----------
    episode:
        The ``episode_facts`` mapping (non-``None``).

    Returns
    -------
    list[str]
        Bullet lines for the episode declaration, step-join claim, and
        forced-tokens disclosure.
    """

    basis = episode.get("fidelity_basis")
    lines = [
        "- Episode capture: "
        f"{episode.get('n_steps_declared')} declared step(s), "
        f"token_feed={episode.get('token_feed')}, "
        f"fidelity_basis={basis}; per-step ledger at "
        "trace.annotations['episode']."
    ]
    step_join = str(episode.get("step_join"))
    if step_join == "unmeasured":
        lines.append(
            "- Cross-step continuity is DECLARED, not measured: "
            "token_feed reflects the declaration only; this artifact "
            "carries no step_join envelope."
        )
    elif step_join.startswith("broken_at_step_") or step_join.startswith(
        "declared_crossing_at_step_"
    ):
        lines.append(
            f"- Cross-step join BREAK measured ({step_join}): the trace "
            "is intact but chain-shaped at that join; step-series reads, "
            "whole-episode replay, and the blessing fold refuse across "
            "it, per-segment reads are unaffected."
        )
    elif step_join == "measured_with_unchecked_joins":
        lines.append(
            "- Cross-step joins measured with UNCHECKED gaps (reasons in "
            "the step_join envelope): series claims refuse across the "
            "unchecked joins."
        )
    if basis == "forced":
        lines.append(
            "- Forced-tokens episode: a disclosed NON-VERIFYING mode; "
            "emitted tokens were supplied, not generated."
        )
    return lines


def _capture_status_lines(log: Any) -> list[str]:
    """Return capture outcome/verification lines for the text report.

    Parameters
    ----------
    log:
        Completed trace-like object.

    Returns
    -------
    list[str]
        Bullet lines for the capture-status section.
    """

    facts = _capture_verification(log)
    lines = [f"- Capture outcome: {facts['capture_status']}."]
    # L7a G3 render honesty (memo sec 3.4).
    if bool(getattr(log, "structure_only", False)):
        lines.append(
            "- Structure-only capture: shapes/dtypes are HYPOTHESES, not "
            "measurements; discharge against a real capture to corroborate."
        )
    if facts["capture_verified"] is False:
        reason = facts["capture_verification_reason"] or "unrecorded reason"
        lines.append(
            f"- Capture verification: UNVERIFIED ({reason}); parts of this "
            "forward may be missing or unattributed -- treat every summary "
            "below as a lower bound on what ran."
        )
    elif facts["capture_verified"] is True:
        lines.append("- Capture verification: verified.")
    else:
        lines.append("- Capture verification: no ceiling recorded.")
    if facts["rescue_rerun"]:
        lines.append(
            "- This result came from the disclosed rescue re-run "
            "(mode_rescue_rerun), not the primary capture."
        )
    poison = poison_facts(log)
    if poison["poisoned"]:
        lines.append(
            "- POISONED sparse run (path_faithfulness="
            f"{poison.get('path_faithfulness', 'unknown')}): this trace was "
            "returned via return_diverged=True and its values are NOT "
            "model-faithful; inspect the divergence, do not publish the numbers."
        )
    episode = episode_facts(log)
    if episode is not None:
        lines.extend(_episode_status_lines(episode))
    for advisory in capture_advisories(log):
        lines.append(
            f"- Capture advisory: {advisory.get('kind')} "
            f"x{advisory.get('count')} (first at "
            f"{advisory.get('first_location') or 'unknown location'})."
        )
    return lines


def _base_json(log: Any, audience: Audience) -> dict[str, Any]:
    """Return fields common to complete and partial JSON reports.

    Parameters
    ----------
    log:
        Capture object.
    audience:
        Requested audience label.

    Returns
    -------
    dict[str, Any]
        Common stable-schema fields.
    """

    # ONE machine core (sumfam D17/item 14): the count fields below read
    # the same FactCore sections agent_json reads, so the two projections
    # cannot drift apart; the parity gate pins every overlapping field.
    try:
        from ._factcore import factcore

        core = factcore(log)
        layer_count = core.counts.layers
        operation_count = core.counts.tracked_tensor_rows
        saved_count = core.memory.at_capture_saved_ops
    except Exception:  # noqa: BLE001 -- partial/foreign logs keep the raw fallback
        core = None
        layer_count = _safe_len(getattr(log, "layer_list", None))
        operation_count = int(getattr(log, "num_ops", 0) or 0)
        saved_count = int(getattr(log, "num_saved_ops", 0) or 0)
    return {
        "schema": "torchlens.explain.v1",
        "audience": audience,
        **_capture_verification(log),
        "model_class": getattr(log, "model_class_name", type(log).__name__),
        "layer_count": layer_count,
        "operation_count": operation_count,
        "saved_tensor_count": saved_count,
        "total_tensor_count": int(getattr(log, "num_tensors", 0) or 0),
        "has_backward_pass": bool(getattr(log, "has_backward_pass", False)),
        "exception_type": "unknown",
        "exception_message": "unknown",
        "last_completed_op_label": "unknown",
        "last_completed_op_shape": "unknown",
        "last_completed_op_dtype": "unknown",
        "last_completed_op_device": "unknown",
        "failing_boundary": "unknown",
        "first_nonfinite": _first_nonfinite_summary(log),
    }


def _full_json(log: Any, audience: Audience) -> dict[str, Any]:
    """Build the stable JSON schema for a completed trace.

    Parameters
    ----------
    log:
        Completed trace.
    audience:
        Requested audience label.

    Returns
    -------
    dict[str, Any]
        Stable full-trace report dictionary.
    """

    return _base_json(log, audience)


def _partial_json(
    log: Any,
    audience: Audience,
    diagnosis: dict[str, Any],
) -> dict[str, Any]:
    """Build the stable JSON schema for a partial trace.

    Parameters
    ----------
    log:
        Partial capture wrapper.
    audience:
        Requested audience label.
    diagnosis:
        Evidence fields from :func:`_partial_diagnosis`.

    Returns
    -------
    dict[str, Any]
        Stable failed-capture report dictionary.
    """

    result = _base_json(log, audience)
    result.update(diagnosis)
    result.update(
        {
            "capture_status": "partial",
            "model_class": getattr(log.trace, "model_class_name", type(log.trace).__name__),
            "layer_count": len(log.raw_layers),
            "operation_count": len(log.raw_layers),
            "saved_tensor_count": sum(
                bool(getattr(op, "has_saved_activation", False)) for op in log.raw_layers
            ),
            "total_tensor_count": len(log.raw_layers),
        }
    )
    return result


def _first_nonfinite_summary(log: Any) -> str:
    """Return a saved-output non-finite summary without speculation.

    Parameters
    ----------
    log:
        Completed trace-like object.

    Returns
    -------
    str
        First recorded non-finite detail, or a scoped clean statement.
    """

    from ._health import health_facts

    # D4/D14 (F09): explain never scans -- this summary serves the basis in
    # hand (capture record, prior scan memo, persisted artifact record) and
    # says NOT-CHECKED otherwise, with the spelling that would scan.
    facts = health_facts(log, allow_scan=False)
    if facts.verdict == "found":
        labels = facts.nonfinite_labels or facts.alias_nonfinite_labels
        return _first_nonfinite_detail(log, str(labels[0]).rsplit(":", 1)[0])
    if facts.verdict == "checked_and_clean":
        return (
            f"No non-finite values found ({facts.checked} output(s) examined; basis: "
            f"{facts.source_basis or facts.basis})."
        )
    if (facts.source_basis or facts.basis) not in ("unscanned", "underivable"):
        # A basis exists but coverage has gaps: disclose exactly what the
        # scan could not or did not examine (disk-backed payloads,
        # unsaved outputs) instead of a bare NOT-CHECKED. The shared
        # gap-note wording (one source, never drifting) is safe here: a
        # basis is in hand, so no new scan runs.
        from ..data_classes._nonfinite import coverage_gap_note

        return (
            "No non-finite values found in the examined outputs"
            f"{coverage_gap_note(log, kind='saved')} -- NOT-CHECKED overall "
            f"(basis: {facts.source_basis or facts.basis})."
        )
    return (
        "Non-finite health NOT-CHECKED on this object; scan retained payloads with "
        "tl.report.health_facts(trace) or trace.nonfinite_ops."
    )


def _first_nonfinite_detail(log: Any, saved_label: str) -> str:
    """Return the log's own non-finite detail, or a scoped fallback.

    ``log.first_nonfinite()`` re-asks the trace under its own scan contract and can
    itself raise ``ValueError`` on a selective-save trace (it reads unsaved ``.out``
    payloads, saved or not, and revalidating a memo reads them too). explain() must
    not propagate that: fall back to the scoped saved-output statement so the
    report always honors its ``unknown``/clean contract.

    Parameters
    ----------
    log:
        Completed trace-like object.
    saved_label:
        Label of the saved output already found non-finite.

    Returns
    -------
    str
        First recorded non-finite detail, or a scoped fallback statement.
    """

    scoped = f"saved output {saved_label} is non-finite"
    if not hasattr(log, "first_nonfinite"):
        return scoped
    try:
        # Explicit link_format: report text is publishable data, so the source
        # location must stay plain ``path:line`` -- never an OSC 8 escape or a
        # resolved-absolute-path URI. Duck-typed logs without the kwarg fall
        # into the TypeError arm and keep the scoped statement.
        return str(log.first_nonfinite(link_format="text"))
    except (ValueError, RuntimeError, TypeError):
        return scoped


def _model_summary_lines(log: Any) -> list[str]:
    """Return model architecture and cost summary lines.

    Parameters
    ----------
    log:
        Model log to summarize.

    Returns
    -------
    list[str]
        Bullet lines for the model section.
    """

    from ._factcore import factcore

    model_class_name = getattr(log, "model_class_name", type(log).__name__)
    # Single-source law (sumfam D1/item 11): headline facts come from
    # FactCore, never raw trace attributes. A husked/foreign log that cannot
    # serve the substrate degrades to the legacy zero-shape (explain is the
    # one member that never refuses).
    try:
        core = factcore(log)
    except Exception:  # noqa: BLE001 -- cleaned/foreign logs keep the degraded report
        num_params = int(getattr(log, "num_params", 0) or 0)
        return [
            f"- Architecture: {model_class_name}.",
            f"- Parameters: {_format_count(num_params)} unique.",
            "- Forward FLOPs: unavailable on this object.",
            f"- Modules represented: {_safe_len(getattr(log, 'modules', None))}.",
        ]
    num_params = core.params.total or 0
    trainable_params = core.params.trainable or 0
    frozen_params = core.params.frozen or 0
    total_flops = int(core.compute.partition_total)
    lines = [
        f"- Architecture: {model_class_name}.",
        (
            f"- Parameters: {_format_count(num_params)} unique "
            f"({_format_count(trainable_params)} trainable, {_format_count(frozen_params)} frozen)."
        ),
    ]
    for group in core.params.tied_groups:
        lines.append(f"- Tied parameters: {' == '.join(group)}.")
    flops_line = f"- Forward FLOPs (actual-path analytic): {_format_count(total_flops)}."
    if core.compute.coverage.unknown:
        flops_line += (
            f" LOWER BOUND: {core.compute.coverage.unknown} op(s) carry no cost rule "
            "(see trace.unknown_flop_ops)."
        )
    lines.append(flops_line)
    lines.append(f"- Modules represented: {_format_count(core.counts.modules)}.")
    return lines


def _capture_summary_lines(log: Any) -> list[str]:
    """Return layer, operation, pass, and tensor capture summary lines.

    Parameters
    ----------
    log:
        Model log to summarize.

    Returns
    -------
    list[str]
        Bullet lines for the capture section.
    """

    from ._factcore import factcore

    # Counts come from the explicit grain vocabulary (sumfam D3): the bare
    # word "operations" cannot denote both the compute ops and the tracked
    # tensor rows, so both grains print under their names. Husked/foreign
    # logs degrade to the legacy zero-shape.
    try:
        core = factcore(log)
    except Exception:  # noqa: BLE001 -- cleaned/foreign logs keep the degraded report
        return [
            f"- Layers logged: {_safe_len(getattr(log, 'layer_list', None))}.",
            f"- Operations logged: {int(getattr(log, 'num_ops', 0) or 0)}.",
        ]
    pass_counts = [
        int(value)
        for value in (getattr(log, "layer_num_calls", {}) or {}).values()
        if isinstance(value, int)
    ]
    max_ops = max(pass_counts, default=1)
    return [
        f"- Layers logged: {_format_count(core.counts.layers)}.",
        (
            f"- Compute ops logged: {_format_count(core.counts.compute_ops)} "
            f"({_format_count(core.counts.tracked_tensor_rows)} tracked tensor rows incl. "
            f"{_format_count(core.counts.alias_rows)} boundary/buffer rows)."
        ),
        (
            f"- Tensors saved at capture: {_format_count(core.memory.at_capture_saved_ops)} "
            f"of {_format_count(core.counts.tracked_tensor_rows)}."
        ),
        f"- Maximum observed ops for one layer: {_format_count(max_ops)}.",
    ]


def _backward_summary_lines(log: Any) -> list[str]:
    """Return backward-pass capture summary lines.

    Parameters
    ----------
    log:
        Model log to summarize.

    Returns
    -------
    list[str]
        Bullet lines for the backward section.
    """

    if not bool(getattr(log, "has_backward_pass", False)):
        return ["- No backward passes are recorded on this log."]
    pass_count = int(getattr(log, "num_backward_passes", 0) or 0)
    grad_fn_count = int(getattr(log, "num_grad_fns", 0) or 0)
    grad_fn_call_count = int(getattr(log, "num_grad_fn_calls", 0) or 0)
    saved_grad_count = _safe_len(getattr(log, "saved_grad_ops", None))
    total_backward_memory = getattr(log, "total_backward_memory", None)
    memory_line = (
        f"- Unique backward payload memory: {total_backward_memory}."
        if total_backward_memory is not None
        else "- Unique backward payload memory: unavailable."
    )
    return [
        f"- Backward passes: {_format_count(pass_count)}.",
        (
            f"- GradFn records: {_format_count(grad_fn_count)} nodes, "
            f"{_format_count(grad_fn_call_count)} calls."
        ),
        f"- Op gradient records saved: {_format_count(saved_grad_count)}.",
        memory_line,
    ]


def _anomaly_lines(log: Any) -> list[str]:
    """Return NaN/Inf anomaly lines.

    Parameters
    ----------
    log:
        Model log to inspect.

    Returns
    -------
    list[str]
        Bullet lines describing non-finite outs.
    """

    from ._health import health_facts

    # D14 (sumfam item 11): bare explain renders the THREE health states
    # from whatever basis is in hand and NEVER scans -- and it says "not
    # audited" with the spelling. The false negative this kills: a
    # payload-stripped all-NaN artifact reading as clean.
    facts = health_facts(log, allow_scan=False)
    verdict = facts.verdict
    basis = facts.source_basis or facts.basis
    if verdict == "found":
        labels = facts.nonfinite_labels or facts.alias_nonfinite_labels
        first = labels[0]
        lines = [
            (
                f"- FOUND: {len(facts.nonfinite_labels)} op output(s) contain NaN or Inf "
                f"values (basis: {basis})."
            ),
            f"- First affected op: {first}.",
        ]
        if facts.alias_nonfinite_labels:
            lines.append(
                f"- Alias-view hits (boundary/buffer rows): {len(facts.alias_nonfinite_labels)}."
            )
        detail = _first_nonfinite_detail(log, str(first).rsplit(":", 1)[0])
        lines.append(f"- Detail: {detail}")
        lines.append("- Not audited: judgments (severity, follow-ups) come from tl.audit(trace).")
        return lines
    if verdict == "checked_and_clean":
        return [
            (
                f"- CHECKED-AND-CLEAN: {facts.checked} op output(s) examined, none "
                f"non-finite (basis: {basis})."
            ),
            "- Not audited: judgments (severity, follow-ups) come from tl.audit(trace).",
        ]
    if basis not in ("unscanned", "underivable"):
        # A basis exists but coverage has gaps (disk-backed, inference-mode,
        # uncheckable dtypes, unsaved outputs): disclose the exact gaps
        # through the shared wording -- safe, no new scan runs.
        from ..data_classes._nonfinite import coverage_gap_note

        return [
            (
                "- NOT-CHECKED overall: no non-finite values in the examined outputs"
                f"{coverage_gap_note(log, kind='saved')} (basis: {basis})."
            ),
            "- Not audited: judgments (severity, follow-ups) come from tl.audit(trace).",
        ]
    return [
        (
            f"- NOT-CHECKED: non-finite health has not been established on this object "
            f"({facts.unexamined} saved output(s) unexamined; basis: {basis})."
        ),
        (
            "- To scan retained payloads (payload_scan cost): "
            "tl.report.health_facts(trace) or trace.nonfinite_ops."
        ),
    ]


def _intervention_lines(log: Any) -> list[str]:
    """Return intervention summary lines.

    Parameters
    ----------
    log:
        Model log to inspect.

    Returns
    -------
    list[str]
        Bullet lines describing applied intervention recipes.
    """

    spec = getattr(log, "_intervention_spec", None)
    target_specs = tuple(getattr(spec, "target_value_specs", ()) or ())
    hook_specs = tuple(getattr(spec, "hook_specs", ()) or ())
    history = list(getattr(log, "state_history", []) or [])
    if not target_specs and not hook_specs and not history:
        return ["- No interventions are recorded on this log."]
    return [
        f"- Target-value edits: {_format_count(len(target_specs))}.",
        f"- Hook edits: {_format_count(len(hook_specs))}.",
        f"- Recorded intervention operations: {_format_count(len(history))}.",
    ]


#: Rendering bounds for the Logged values section: user values are arbitrary
#: objects, and report text must stay bounded no matter what was logged.
_LOGGED_VALUE_MAX_ENTRIES = 10
_LOGGED_VALUE_MAX_REPR = 80


def _logged_value_lines(log: Any) -> list[str]:
    """Return capture-time ``log_value`` read-back lines, bounded.

    Parameters
    ----------
    log:
        Model log to inspect.

    Returns
    -------
    list[str]
        Bullet lines for recorded values (empty when none were recorded);
        entries beyond the cap are disclosed by count, never silently dropped.
    """

    values = (getattr(log, "annotations", {}) or {}).get("logged_values", {})
    if not isinstance(values, dict) or not values:
        return []
    lines = []
    for name, value in list(values.items())[:_LOGGED_VALUE_MAX_ENTRIES]:
        rendered = repr(value)
        if len(rendered) > _LOGGED_VALUE_MAX_REPR:
            rendered = rendered[: _LOGGED_VALUE_MAX_REPR - 3] + "..."
        lines.append(f"- {name} = {rendered}")
    omitted = len(values) - _LOGGED_VALUE_MAX_ENTRIES
    if omitted > 0:
        lines.append(f"- ... and {omitted} more (read them all via trace.logged_values).")
    return lines


def _pattern_lines(log: Any, *, audience: Audience) -> list[str]:
    """Return notable graph-pattern lines.

    Parameters
    ----------
    log:
        Model log to inspect.
    audience:
        Requested report audience.

    Returns
    -------
    list[str]
        Bullet lines for graph patterns.
    """

    lines = [
        _shared_parameter_line(log),
        _recurrent_loop_line(log),
        _dynamic_control_flow_line(log),
    ]
    if audience in {"researcher", "auto"}:
        lines.append(_operation_mix_line(log))
    if audience == "practitioner":
        lines.append(_operational_status_line(log))
    return lines


def _shared_parameter_line(log: Any) -> str:
    """Return a shared-parameter pattern line.

    A parameter is *shared* (weight-tied across modules, or reused across
    recurrent passes) exactly when the same parameter object participates in
    more than one operation. The ground-truth signal is therefore the parameter's
    usage count -- ``num_calls`` / distinct ``used_by_ops`` -- NOT
    ``co_parent_params``, which lists the sibling params of the SAME op
    (weight+bias of one Linear) and so is both a false positive for every plain
    multi-param op and a false negative for genuine weight tying (a tied weight is
    the sole param of each op, so ``co_parent_params`` is empty). ``used_by_layers``
    also cannot detect tying: two tied modules roll into one equivalent layer
    label, so only the op-level usage count separates sharing from independence.

    Parameters
    ----------
    log:
        Model log to inspect.

    Returns
    -------
    str
        Shared-parameter summary.
    """

    shared = 0
    for param_log in getattr(log, "param_logs", []) or []:
        used_by_ops = set(getattr(param_log, "used_by_ops", ()) or ())
        try:
            num_calls = int(getattr(param_log, "num_calls", 0) or 0)
        except (TypeError, ValueError):
            num_calls = 0
        if len(used_by_ops) > 1 or num_calls > 1:
            shared += 1
    if shared:
        return (
            f"- Shared parameters: {_format_count(shared)} parameter object(s) "
            "are used by more than one operation (weight tying or recurrent reuse)."
        )
    return "- Shared parameters: none reported."


def _recurrent_loop_line(log: Any) -> str:
    """Return a recurrent-loop pattern line.

    Parameters
    ----------
    log:
        Model log to inspect.

    Returns
    -------
    str
        Recurrent-loop summary.
    """

    max_loops = int(getattr(log, "max_layer_op_count", 1) or 1)
    if max_loops > 1:
        return f"- Recurrent loops: at least one layer was observed across {max_loops} ops."
    return "- Recurrent loops: no repeated layer ops were detected."


def _dynamic_control_flow_line(log: Any) -> str:
    """Return a dynamic-control-flow pattern line.

    Parameters
    ----------
    log:
        Model log to inspect.

    Returns
    -------
    str
        Dynamic-control-flow summary.
    """

    event_count = _safe_len(getattr(log, "conditional_records", None))
    has_branching = bool(getattr(log, "has_conditional_branching", False))
    if event_count or has_branching:
        return (
            f"- Dynamic control flow: {_format_count(event_count)} conditional event(s) recorded."
        )
    return "- Dynamic control flow: no conditional events were recorded for this input."


def _operation_mix_line(log: Any) -> str:
    """Return a compact operation mix line.

    Parameters
    ----------
    log:
        Model log to inspect.

    Returns
    -------
    str
        Most common operation types.
    """

    names = [
        str(getattr(layer, "func_name", "unknown"))
        for layer in getattr(log, "layer_list", []) or []
        if str(getattr(layer, "func_name", "none")) != "none"
    ]
    if not names:
        return "- Operation mix: no operation names were available."
    common = ", ".join(f"{name} x{count}" for name, count in Counter(names).most_common(5))
    return f"- Operation mix: {common}."


def _operational_status_line(log: Any) -> str:
    """Return a practitioner-oriented operational status line.

    Parameters
    ----------
    log:
        Model log to inspect.

    Returns
    -------
    str
        Operational status summary.
    """

    cache_hit = bool(getattr(log, "capture_cache_hit", False))
    streamed = sum(
        1
        for layer in getattr(log, "layer_list", []) or []
        if getattr(layer, "out_ref", None) is not None
        or getattr(layer, "grad_ref", None) is not None
    )
    return f"- Operational status: cache_hit={cache_hit}, streamed_ops={streamed}."


def _safe_len(value: Any) -> int:
    """Return ``len(value)`` or zero when unavailable.

    Parameters
    ----------
    value:
        Object to measure.

    Returns
    -------
    int
        Length or zero.
    """

    if value is None:
        return 0
    try:
        return len(value)
    except TypeError:
        return 0


def _format_count(value: int) -> str:
    """Return an integer with comma separators.

    Parameters
    ----------
    value:
        Integer value to format.

    Returns
    -------
    str
        Formatted value.
    """

    return f"{value:,}"
