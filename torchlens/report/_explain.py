"""Plain-language reports for completed TorchLens logs."""

from __future__ import annotations

from collections import Counter
from typing import Any, Literal

import torch

Audience = Literal["researcher", "practitioner", "auto"]


def explain(log: Any, audience: Audience = "auto") -> str:
    """Explain a TorchLens log in plain language.

    Parameters
    ----------
    log:
        Completed ``ModelLog``-like object.
    audience:
        Report style. ``"researcher"`` includes graph-pattern detail,
        ``"practitioner"`` emphasizes operational status, and ``"auto"``
        selects a balanced report.

    Returns
    -------
    str
        Multi-section plain-language report.

    Raises
    ------
    ValueError
        If ``audience`` is not supported.
    """

    if audience not in {"researcher", "practitioner", "auto"}:
        raise ValueError("audience must be 'researcher', 'practitioner', or 'auto'.")

    lines = [
        "TorchLens report",
        "",
        "Model summary",
        *_model_summary_lines(log),
        "",
        "Capture summary",
        *_capture_summary_lines(log),
        "",
        "Anomalies",
        *_anomaly_lines(log),
        "",
        "Interventions",
        *_intervention_lines(log),
        "",
        "Notable patterns",
        *_pattern_lines(log, audience=audience),
    ]
    return "\n".join(lines)


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

    model_name = getattr(log, "model_name", type(log).__name__)
    total_params = int(getattr(log, "total_params", 0) or 0)
    trainable_params = int(getattr(log, "total_params_trainable", 0) or 0)
    frozen_params = int(getattr(log, "total_params_frozen", 0) or 0)
    total_flops = int(getattr(log, "total_flops_forward", getattr(log, "total_flops", 0)) or 0)
    module_count = _safe_len(getattr(log, "modules", None))
    return [
        f"- Architecture: {model_name}.",
        (
            f"- Parameters: {_format_count(total_params)} total "
            f"({_format_count(trainable_params)} trainable, {_format_count(frozen_params)} frozen)."
        ),
        f"- Forward FLOPs: {_format_count(total_flops)}.",
        f"- Modules represented: {_format_count(module_count)}.",
    ]


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

    layer_count = _safe_len(getattr(log, "layer_list", None))
    operation_count = int(getattr(log, "num_operations", 0) or 0)
    tensor_total = int(getattr(log, "num_tensors_total", 0) or 0)
    tensor_saved = int(getattr(log, "num_tensors_saved", 0) or 0)
    pass_counts = [
        int(value)
        for value in (getattr(log, "layer_num_passes", {}) or {}).values()
        if isinstance(value, int)
    ]
    max_passes = max(pass_counts, default=1)
    return [
        f"- Layers logged: {_format_count(layer_count)}.",
        f"- Operations logged: {_format_count(operation_count)}.",
        f"- Tensors saved: {_format_count(tensor_saved)} of {_format_count(tensor_total)}.",
        f"- Maximum observed passes for one layer: {_format_count(max_passes)}.",
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
        Bullet lines describing non-finite activations.
    """

    nonfinite_labels: list[str] = []
    for layer in getattr(log, "layer_list", []) or []:
        activation = getattr(layer, "activation", None)
        if not isinstance(activation, torch.Tensor) or activation.numel() == 0:
            continue
        try:
            has_nonfinite = bool((~torch.isfinite(activation.detach())).any().item())
        except (RuntimeError, TypeError):
            continue
        if has_nonfinite:
            nonfinite_labels.append(str(getattr(layer, "layer_label", "unknown")))

    if not nonfinite_labels:
        return ["- No NaN or Inf values were found in saved activations."]
    first = nonfinite_labels[0]
    return [
        f"- {len(nonfinite_labels)} saved activation(s) contain NaN or Inf values.",
        f"- First affected layer: {first}.",
        f"- Detail: {log.first_nonfinite() if hasattr(log, 'first_nonfinite') else 'unavailable'}",
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
    history = list(getattr(log, "operation_history", []) or [])
    if not target_specs and not hook_specs and not history:
        return ["- No interventions are recorded on this log."]
    return [
        f"- Target-value edits: {_format_count(len(target_specs))}.",
        f"- Hook edits: {_format_count(len(hook_specs))}.",
        f"- Recorded intervention operations: {_format_count(len(history))}.",
    ]


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
        linked = getattr(param_log, "linked_params", ()) or ()
        if linked:
            shared += 1
    if shared:
        return (
            f"- Shared parameters: {_format_count(shared)} parameter entries report linked params."
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

    max_loops = int(getattr(log, "max_recurrent_loops", 1) or 1)
    if max_loops > 1:
        return f"- Recurrent loops: at least one layer was observed across {max_loops} passes."
    return "- Recurrent loops: no repeated layer passes were detected."


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

    event_count = _safe_len(getattr(log, "conditional_events", None))
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
    streamed = int(getattr(log, "num_streamed_passes", 1) or 1)
    return f"- Operational status: cache_hit={cache_hit}, streamed_passes={streamed}."


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
