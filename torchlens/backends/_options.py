"""Shared backend trace-option defaulting and rejection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._deprecations import MISSING
from .registry import BackendUnsupportedError


@dataclass(frozen=True)
class ExtraKwargPolicy:
    """Declarative policy for backend-extra public trace kwargs.

    Parameters
    ----------
    runtime_option_names:
        Option names that use the runtime-mutation rejection message when present.
    runtime_message:
        Message template used for runtime-mutation kwargs. It receives ``names``.
    fallback_message:
        Message template used for non-runtime extras. It receives ``names``.
    always_runtime:
        Whether every rejected extra should use ``runtime_message``.
    """

    runtime_option_names: frozenset[str]
    runtime_message: str
    fallback_message: str
    always_runtime: bool = False


@dataclass(frozen=True)
class PreviewTraceOptionPolicy:
    """Declarative unsupported-option policy for preview capture backends.

    Parameters
    ----------
    backend_name:
        Backend display name used only for internal diagnostics.
    input_kwargs_message:
        Error message for keyword forward inputs, or ``None`` if allowed.
    full_save_message:
        Error message when ``layers_to_save`` is not full-save compatible.
    rejected_truthy_messages:
        Option-specific messages for unsupported truthy options.
    output_device_message:
        Error message when ``output_device`` is not ``"same"``.
    output_device_error:
        Exception type raised for unsupported ``output_device``.
    save_raw_activations_false_message:
        Error message when ``save_raw_activations`` is false, or ``None`` if allowed.
    save_window_message:
        Error message for non-default lookback settings, or ``None`` if allowed.
    """

    backend_name: str
    input_kwargs_message: str | None = None
    full_save_message: str | None = None
    rejected_truthy_messages: dict[str, str] | None = None
    output_device_message: str | None = None
    output_device_error: type[Exception] = BackendUnsupportedError
    save_raw_activations_false_message: str | None = None
    save_window_message: str | None = None


JAX_EXTRA_KWARG_POLICY = ExtraKwargPolicy(
    runtime_option_names=frozenset(
        {
            "halt",
            "intervene",
            "recipes",
            "save",
            "stop_after",
            "storage",
            "streaming",
        }
    ),
    runtime_message=(
        "JAX backend preview does not support runtime-mutation or stop-early "
        "options: {names}. Static-label save= selectors are supported as "
        "post-finalization payload filters, but trace(intervene=...) and "
        "trace(halt=...) need predicate-time concrete values and mutation/partial "
        "replay semantics that jaxpr tracing does not expose through TorchLens' "
        "current public labels. Use an unfiltered tl.trace(..., backend='jax') "
        "call, static-label save= selectors, or the PyTorch backend for "
        "intervention, halt, streaming, and value-dependent predicates."
    ),
    fallback_message=(
        "JAX backend preview does not support: {names}. "
        "Use full-save JAX trace capture or the PyTorch backend for this surface."
    ),
)
"""Extra public-kwarg rejection policy for the JAX preview backend."""


TINYGRAD_EXTRA_KWARG_POLICY = ExtraKwargPolicy(
    runtime_option_names=frozenset(),
    runtime_message=(
        "tinygrad backend preview does not support runtime-mutation or stop-early "
        "options: {names}. Static-label save= selectors are supported as "
        "post-finalization payload filters, but trace(intervene=...) and "
        "trace(halt=...) need predicate-time concrete values and a way to replace or "
        "truncate lazy UOp descendants before realize(), which tinygrad does not expose "
        "through a stable TorchLens surface. Use an unfiltered tl.trace(..., "
        "backend='tinygrad') call, static-label save= selectors, or the PyTorch backend "
        "for intervention, halt, streaming, and value-dependent predicates."
    ),
    fallback_message="",
    always_runtime=True,
)
"""Extra public-kwarg rejection policy for the tinygrad preview backend."""


PADDLE_EXTRA_KWARG_POLICY = ExtraKwargPolicy(
    runtime_option_names=frozenset(),
    runtime_message=(
        "paddle backend preview does not support runtime-mutation or stop-early "
        "options: {names}. Static-label save= selectors are supported as "
        "post-finalization payload filters, but trace(intervene=...) and "
        "trace(halt=...) need predicate-time concrete values and a way to replace or "
        "truncate Paddle dygraph descendants before execution completes, which Paddle "
        "does not expose through a stable TorchLens surface. Use an unfiltered "
        "tl.trace(..., backend='paddle') call, static-label save= selectors, or the "
        "PyTorch backend for intervention, halt, streaming, and value-dependent predicates."
    ),
    fallback_message="",
    always_runtime=True,
)
"""Extra public-kwarg rejection policy for the Paddle preview backend."""


JAX_PREVIEW_TRACE_OPTION_POLICY = PreviewTraceOptionPolicy(
    backend_name="JAX",
    input_kwargs_message=(
        "JAX backend preview supports positional args only. Pass keyword values as "
        "explicit params/input leaves or declared static positional args."
    ),
    full_save_message="JAX backend preview is full-save only; save shaping is unsupported.",
    rejected_truthy_messages={
        "activation_transform": (
            "JAX backend preview does not support activation_transform; full-save forward "
            "capture only. Use full-save JAX trace capture or the PyTorch backend."
        ),
        "detach_saved_activations": (
            "JAX backend preview does not support detach_saved_activations; full-save forward "
            "capture only. Use full-save JAX trace capture or the PyTorch backend."
        ),
        "save_grads": (
            "JAX backend preview does not support save_grads; full-save forward capture only. "
            "Use tl.backends.jax.GradOptions for derived gradients."
        ),
        "save_arg_values": (
            "JAX backend preview does not support save_arg_values; full-save forward capture "
            "only. Use full-save JAX trace capture or the PyTorch backend."
        ),
        "save_code_context": (
            "JAX backend preview does not support save_code_context; full-save forward capture "
            "only. Use full-save JAX trace capture or the PyTorch backend."
        ),
        "backward_ready": (
            "JAX backend preview does not support backward_ready; full-save forward capture "
            "only. Use tl.backends.jax.GradOptions for derived gradients."
        ),
        "module_filter": (
            "JAX backend preview does not support module_filter; full-save forward capture "
            "only. Use full-save JAX trace capture or the PyTorch backend."
        ),
        "transform": (
            "JAX backend preview does not support transform; full-save forward capture only. "
            "Use full-save JAX trace capture or the PyTorch backend."
        ),
        "layer_visualizers": (
            "JAX backend preview does not support layer_visualizers; full-save forward capture "
            "only. Use full-save JAX trace capture or the PyTorch backend."
        ),
        "save_visualizations": (
            "JAX backend preview does not support save_visualizations; full-save forward "
            "capture only. Use full-save JAX trace capture or the PyTorch backend."
        ),
    },
    output_device_message="JAX backend preview only supports output_device='same'.",
    save_raw_activations_false_message=(
        "JAX backend preview is full-save only; save_raw_activations=False is unsupported."
    ),
    save_window_message=(
        "JAX backend preview is full-save only; save-window shaping is unsupported."
    ),
)
"""Unsupported public trace-option policy for the JAX preview backend."""


TINYGRAD_PREVIEW_TRACE_OPTION_POLICY = PreviewTraceOptionPolicy(
    backend_name="tinygrad",
    input_kwargs_message="tinygrad backend preview supports positional args only.",
    full_save_message="tinygrad backend preview is full-save only; save shaping is unsupported.",
    rejected_truthy_messages={
        name: (f"tinygrad backend preview does not support {name}; full-save forward capture only.")
        for name in (
            "activation_transform",
            "detach_saved_activations",
            "save_grads",
            "save_arg_values",
            "save_code_context",
            "save_rng_states",
            "backward_ready",
            "module_filter",
            "transform",
            "layer_visualizers",
            "save_visualizations",
        )
    },
    output_device_message="tinygrad backend preview only supports output_device='same'.",
    save_raw_activations_false_message=(
        "tinygrad backend preview is full-save only; save_raw_activations=False is unsupported."
    ),
    save_window_message=(
        "tinygrad backend preview is full-save only; save-window shaping is unsupported."
    ),
)
"""Unsupported public trace-option policy for the tinygrad preview backend."""


PADDLE_PREVIEW_TRACE_OPTION_POLICY = PreviewTraceOptionPolicy(
    backend_name="paddle",
    input_kwargs_message="paddle backend preview supports positional args only.",
    full_save_message="paddle backend preview is full-save only; save shaping is unsupported.",
    rejected_truthy_messages={
        name: (f"paddle backend preview does not support {name}; full-save forward capture only.")
        for name in (
            "activation_transform",
            "detach_saved_activations",
            "save_grads",
            "save_arg_values",
            "save_code_context",
            "save_rng_states",
            "backward_ready",
            "module_filter",
            "transform",
            "layer_visualizers",
            "save_visualizations",
        )
    },
    output_device_message="paddle backend preview only supports output_device='same'.",
    save_raw_activations_false_message=(
        "paddle backend preview is full-save only; save_raw_activations=False is unsupported."
    ),
    save_window_message=(
        "paddle backend preview is full-save only; save-window shaping is unsupported."
    ),
)
"""Unsupported public trace-option policy for the Paddle preview backend."""


MLX_PREVIEW_TRACE_OPTION_POLICY = PreviewTraceOptionPolicy(
    backend_name="MLX",
    rejected_truthy_messages={
        "save_grads": "backward capture is not supported on the mlx backend",
    },
    output_device_message="MLX backend only supports output_device='same' in technical preview.",
    output_device_error=ValueError,
)
"""Unsupported public trace-option policy for the MLX backend object entry."""


def is_missing(value: object) -> bool:
    """Return whether ``value`` is the public missing sentinel.

    Parameters
    ----------
    value:
        Candidate value.

    Returns
    -------
    bool
        ``True`` when ``value`` is ``MISSING``.
    """

    return value is MISSING


def default_if_missing(value: Any, default: Any) -> Any:
    """Return ``default`` when ``value`` is the public missing sentinel.

    Parameters
    ----------
    value:
        Candidate value.
    default:
        Replacement returned for ``MISSING``.

    Returns
    -------
    Any
        ``default`` or ``value``.
    """

    return default if is_missing(value) else value


def reject_extra_trace_kwargs(kwargs: dict[str, Any], policy: ExtraKwargPolicy) -> None:
    """Reject non-default extra public trace kwargs for a backend.

    Parameters
    ----------
    kwargs:
        Extra keyword arguments that reached the backend object entry.
    policy:
        Declarative backend rejection policy.

    Returns
    -------
    None
        Returns when no non-default extras are present.
    """

    rejected = {
        key: value for key, value in kwargs.items() if value is not None and not is_missing(value)
    }
    if not rejected:
        return
    names = ", ".join(sorted(rejected))
    if policy.always_runtime or policy.runtime_option_names & set(rejected):
        raise BackendUnsupportedError(policy.runtime_message.format(names=names))
    raise BackendUnsupportedError(policy.fallback_message.format(names=names))


def reject_unsupported_trace_options(
    options: dict[str, Any],
    policy: PreviewTraceOptionPolicy,
) -> None:
    """Reject unsupported normalized public trace options.

    Parameters
    ----------
    options:
        Normalized public trace options keyed by option name.
    policy:
        Declarative backend rejection policy.

    Returns
    -------
    None
        Returns when all configured options are supported.
    """

    if policy.input_kwargs_message is not None and options.get("input_kwargs"):
        raise BackendUnsupportedError(policy.input_kwargs_message)
    if policy.full_save_message is not None and options.get("layers_to_save") not in ("all", None):
        raise BackendUnsupportedError(policy.full_save_message)
    for option_name, message in (policy.rejected_truthy_messages or {}).items():
        if options.get(option_name):
            raise BackendUnsupportedError(message)
    if policy.output_device_message is not None and options.get("output_device") != "same":
        raise policy.output_device_error(policy.output_device_message)
    if policy.save_raw_activations_false_message is not None and not options.get(
        "save_raw_activations"
    ):
        raise BackendUnsupportedError(policy.save_raw_activations_false_message)
    if policy.save_window_message is not None and (
        options.get("lookback") != 0 or options.get("lookback_payload_policy") != "metadata_only"
    ):
        raise BackendUnsupportedError(policy.save_window_message)
