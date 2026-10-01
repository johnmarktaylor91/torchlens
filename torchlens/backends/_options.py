"""Shared backend trace-option defaulting and rejection helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .._deprecations import MISSING
from .registry import (
    BackendCapabilityConformanceError,
    BackendSpec,
    BackendUnsupportedError,
    require_capability_implementation,
)

TRACE_OPTION_CAPABILITY_GATES: dict[str, str] = {
    "intervene": "interventions",
    "halt": "interventions",
    "recipes": "interventions",
    "storage": "streaming",
    "streaming": "streaming",
    "save_grads": "backward_capture",
    "backward_ready": "backward_capture",
    "save_rng_states": "rng_replay",
    "random_seed": "rng_replay",
    "structure_only": "structure_only_capture",
}
"""Public trace options whose support is owned by a ``BackendCapabilities`` flag.

The rejection helpers below consult this map so the registered capability table
is the load-bearing authority: an option listed here is rejected for a backend
exactly when the named capability flag is ``False``. The ``True`` direction is
also fail-closed: a backend whose declarative policy rejects an option has a
capture path that provably never dispatches it, so a ``True`` flag on such a
backend is a self-contradictory registration and raises
``BackendCapabilityConformanceError`` — whether or not an implementation
factory is bound. A binding the capture path never consumes must not admit the
option (it would be silently ignored); the only way a gated option is admitted
is through a backend whose capture path actually consumes it, i.e. whose
policy does not reject it."""


def _undispatched_capability_error(
    spec: BackendSpec, gate: str, option_name: str
) -> BackendCapabilityConformanceError:
    """Build the refusal for a True flag whose option the policy still rejects.

    Parameters
    ----------
    spec:
        Backend spec whose registration is self-contradictory.
    gate:
        Gated capability flag name.
    option_name:
        Public trace option owned by ``gate``.

    Returns
    -------
    BackendCapabilityConformanceError
        Typed refusal explaining that the bound implementation is never
        dispatched by this backend's capture path.
    """

    return BackendCapabilityConformanceError(
        f"Backend {spec.name!r} declares capability {gate!r} as True, but its "
        f"registered capture path rejects option {option_name!r} and never "
        "dispatches the bound implementation. A binding the backend does not "
        "consume must not admit the option; keep the flag False or implement "
        "real dispatch in the capture path."
    )


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
    inert_values:
        Backend-specific explicit values that remain equivalent to an omitted
        public option.
    """

    runtime_option_names: frozenset[str]
    runtime_message: str
    fallback_message: str
    always_runtime: bool = False
    inert_values: Mapping[str, Any] | None = None


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
        "paddle backend preview does not support these options: {names}. "
        "Live trace(intervene=...), trace(halt=...), recipes=, and static-label "
        "save= selectors are supported on the eager Paddle preview; streaming, "
        "storage=, and the remaining torch-only runtime options are not. Use "
        "the PyTorch backend for streaming and stop-early capture shaping."
    ),
    fallback_message="",
    always_runtime=True,
    inert_values={
        "lookback": 0,
        "lookback_payload_policy": "metadata_only",
        "capture": None,
        "storage": None,
        "streaming": None,
        "inference_only": False,
        "cache": False,
        "stop_after": None,
        "raise_on_nan": False,
        "profile": False,
        "payload_policy": None,
        "save_preview": None,
        "chunk_size": None,
        "chunk_paths": None,
    },
)
"""Extra public-kwarg rejection policy for the Paddle preview backend.

``intervene``, ``halt``, and ``recipes`` are absent because the Paddle
capture path pops and dispatches them before extra-kwarg rejection runs; the
``interventions`` capability flag owns all three.
"""


MLX_EXTRA_KWARG_POLICY = ExtraKwargPolicy(
    runtime_option_names=frozenset(),
    runtime_message=(
        "MLX backend preview does not support: {names}. Static-label save= "
        "selectors, static-label trace(intervene=tl.when(...)), and static-label "
        "trace(halt=...) are supported; streaming/storage and value-dependent "
        "predicates need predicate-time concrete values or replay semantics that "
        "MLX lazy evaluation does not expose through a stable TorchLens surface. "
        "Use an unfiltered tl.trace(..., backend='mlx') call or the PyTorch "
        "backend for this surface."
    ),
    fallback_message="",
    always_runtime=True,
    inert_values={
        "lookback": 0,
        "lookback_payload_policy": "metadata_only",
        "capture": None,
        "storage": None,
        "streaming": None,
        "inference_only": False,
        "cache": False,
        "stop_after": None,
        "raise_on_nan": False,
        "profile": False,
        "recipes": None,
        "payload_policy": None,
        "save_preview": None,
        "chunk_size": None,
        "chunk_paths": None,
        "save_outs_to": None,
        "keep_outs_in_memory": True,
        "out_sink": None,
        "cache_dir": None,
        "save_mode": "copy",
        "capture_tensor_grad_hooks": True,
        "save_raw_gradients": True,
        "source_context_lines": 7,
        "unwrap_when_done": False,
        "reconstruction_ready": False,
    },
)
"""Extra public-kwarg rejection policy for the MLX preview backend."""


TF_EXTRA_KWARG_POLICY = ExtraKwargPolicy(
    runtime_option_names=frozenset(
        {
            "stop_after",
            "storage",
            "streaming",
        }
    ),
    runtime_message=(
        "tf backend preview does not support runtime-mutation or stop-early options: {names}."
    ),
    fallback_message="tf backend preview does not support: {names}.",
    inert_values={
        "lookback": 0,
        "lookback_payload_policy": "metadata_only",
    },
)
"""Extra public-kwarg rejection policy for the TensorFlow preview backend.

``intervene``/``halt``/``recipes`` left this table when tf lifted the
``interventions`` capability: ``intervene=`` dispatches for real, and the two
unimplemented spellings refuse typed inside the tf capture path (a declarative
rejection here would classify the True flag as a self-contradictory
registration)."""


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
    full_save_message="MLX backend preview does not support layers_to_save; use static save= selectors.",
    rejected_truthy_messages={
        "save_grads": "MLX backend preview does not support save_grads; backward capture is unavailable.",
    }
    | {
        name: f"MLX backend preview does not support {name}; forward capture only."
        for name in (
            "activation_transform",
            "detach_saved_activations",
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
    output_device_message="MLX backend only supports output_device='same' in technical preview.",
    # DECLARED CAPABILITY (not an omission): MLX honors
    # ``save_raw_activations=False`` at its capture-policy seam (payloads
    # dropped, shape/dtype metadata preserved), so it carries no
    # ``save_raw_activations_false_message`` while the four full-save-only
    # previews refuse. Pinned by tests/test_mlx_option_policy.py.
    save_raw_activations_false_message=None,
)
"""Unsupported public trace-option policy for the MLX backend object entry."""


TF_PREVIEW_TRACE_OPTION_POLICY = PreviewTraceOptionPolicy(
    backend_name="tf",
    full_save_message="tf backend preview is full-save only.",
    rejected_truthy_messages={
        name: f"tf backend preview does not support {name}; full-save forward capture only."
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
    output_device_message="tf backend preview only supports output_device='same'.",
    save_raw_activations_false_message="tf backend preview is full-save only.",
)
"""Unsupported public trace-option policy for the TensorFlow preview backend."""


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


def reject_extra_trace_kwargs(
    kwargs: dict[str, Any],
    policy: ExtraKwargPolicy,
    *,
    spec: BackendSpec | None = None,
) -> None:
    """Reject non-default extra public trace kwargs for a backend.

    Parameters
    ----------
    kwargs:
        Extra keyword arguments that reached the backend object entry.
    policy:
        Declarative backend rejection policy.
    spec:
        Registered backend spec. When provided, options in
        ``TRACE_OPTION_CAPABILITY_GATES`` refuse typed in BOTH flag states:
        flag ``False`` uses the policy message, and flag ``True`` raises
        ``BackendCapabilityConformanceError`` because this policy's rejection
        list proves the backend's capture path never dispatches the option —
        a bound implementation the backend does not consume must not admit it.

    Returns
    -------
    None
        Returns when no non-default extras are present.
    """

    inert_values = policy.inert_values or {}
    rejected = {}
    for key, value in kwargs.items():
        if is_missing(value) or value is None:
            continue
        if key in inert_values and inert_values[key] == value:
            continue
        if spec is not None:
            gate = TRACE_OPTION_CAPABILITY_GATES.get(key)
            if gate is not None and getattr(spec.capabilities, gate):
                require_capability_implementation(spec, gate)
                raise _undispatched_capability_error(spec, gate, key)
        rejected[key] = value
    if not rejected:
        return
    names = ", ".join(sorted(rejected))
    if policy.always_runtime or policy.runtime_option_names & set(rejected):
        raise BackendUnsupportedError(policy.runtime_message.format(names=names))
    raise BackendUnsupportedError(policy.fallback_message.format(names=names))


def reject_unsupported_trace_options(
    options: dict[str, Any],
    policy: PreviewTraceOptionPolicy,
    *,
    spec: BackendSpec | None = None,
) -> None:
    """Reject unsupported normalized public trace options.

    Parameters
    ----------
    options:
        Normalized public trace options keyed by option name.
    policy:
        Declarative backend rejection policy.
    spec:
        Registered backend spec. When provided, options in
        ``TRACE_OPTION_CAPABILITY_GATES`` refuse typed in BOTH flag states:
        flag ``False`` uses the policy message, and flag ``True`` raises
        ``BackendCapabilityConformanceError`` because this policy's rejection
        list proves the backend's capture path never dispatches the option —
        a bound implementation the backend does not consume must not admit it.

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
            if spec is not None:
                gate = TRACE_OPTION_CAPABILITY_GATES.get(option_name)
                if gate is not None and getattr(spec.capabilities, gate):
                    require_capability_implementation(spec, gate)
                    raise _undispatched_capability_error(spec, gate, option_name)
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
