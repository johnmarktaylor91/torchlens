"""Shared backend trace-option defaulting and rejection helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .._deprecations import MISSING
from ._backend_protocol import refusal_code_for_capability
from .registry import (
    _GATED_TRACE_OPTION_FLAGS,
    BackendCapabilityConformanceError,
    BackendRegistryError,
    BackendSpec,
    BackendUnsupportedError,
    require_capability_implementation,
)

TRACE_OPTION_CAPABILITY_GATES: dict[str, str] = dict(_GATED_TRACE_OPTION_FLAGS)
"""Public trace options whose support is owned by a ``BackendCapabilities`` flag.

Derived from the registry's gated capability records, in their table order.
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


def _option_refusal_code(option_name: str) -> str:
    """The contracted refusal code for one trace option a backend refuses (SPEC 1.3).

    An option a capability flag owns (``TRACE_OPTION_CAPABILITY_GATES``) refuses
    with that flag's code; every other option refuses with
    ``trace_option_unsupported``.

    Parameters
    ----------
    option_name:
        Public trace option or extra keyword the backend refuses.

    Returns
    -------
    str
        A member of ``REFUSAL_CODES``.
    """

    gate = TRACE_OPTION_CAPABILITY_GATES.get(option_name)
    if gate is None:
        return "trace_option_unsupported"
    return refusal_code_for_capability(gate)


def _options_refusal_code(option_names: Iterable[str]) -> str:
    """One code for a refusal naming several options: theirs if shared, else the option code."""

    codes = {_option_refusal_code(name) for name in option_names}
    return codes.pop() if len(codes) == 1 else "trace_option_unsupported"


def _output_device_refusal(policy: PreviewTraceOptionPolicy) -> Exception:
    """The policy's ``output_device`` refusal, coded when its error type takes a code."""

    message = str(policy.output_device_message)
    if issubclass(policy.output_device_error, BackendRegistryError):
        return policy.output_device_error(
            message,
            code=_option_refusal_code("output_device"),
            backend=policy.backend_name,
            option="output_device",
        )
    return policy.output_device_error(message)


@dataclass(frozen=True)
class OptionCapabilityRow:
    """Operation-grain capability truth for ONE (backend, trace option) pair.

    Compo wave 0 (row 0.7): the composition ledger references these stable
    row ids so the future flag split (a backend supporting ``intervene=`` but
    not ``halt=``, inexpressible under the five coarse flags) is an AUTHORITY
    change in :func:`operation_grain_capability_rows`, never a ledger
    rewrite. DOCUMENTED-UNSTABLE spellings pending naming ratification.

    Parameters
    ----------
    row_id:
        Stable ledger id (``trace_option:<option>:<backend>``).
    backend:
        Registered backend name.
    option_name:
        Public gated trace option.
    coarse_flag:
        The ``BackendCapabilities`` flag that owns the option TODAY (the
        registry's gated capability records, projected onto options by
        ``TRACE_OPTION_CAPABILITY_GATES``).
    supported:
        Operation-grain support truth. Currently projected from the coarse
        flag; the split refines this value without changing row ids.
    """

    row_id: str
    backend: str
    option_name: str
    coarse_flag: str
    supported: bool


def operation_grain_capability_rows(spec: BackendSpec) -> tuple[OptionCapabilityRow, ...]:
    """Project one backend's coarse capability flags onto per-option rows.

    Parameters
    ----------
    spec:
        Registered backend spec.

    Returns
    -------
    tuple[OptionCapabilityRow, ...]
        One row per gated public trace option, in gate-table order.
    """

    rows = []
    for option_name, coarse_flag in TRACE_OPTION_CAPABILITY_GATES.items():
        rows.append(
            OptionCapabilityRow(
                row_id=f"trace_option:{option_name}:{spec.name}",
                backend=spec.name,
                option_name=option_name,
                coarse_flag=coarse_flag,
                supported=bool(getattr(spec.capabilities, coarse_flag)),
            )
        )
    return tuple(rows)


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
        Error message when ``save`` is not the full-save ``"all"``.
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
        "trace(stop_after=...) need predicate-time concrete values and mutation/partial "
        "replay semantics that jaxpr tracing does not expose through TorchLens' "
        "current public labels. Use an unfiltered tl.trace(..., backend='jax') "
        "call, static-label save= selectors, or the PyTorch backend for "
        "intervention, stop_after, storage, and value-dependent predicates."
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
        "tinygrad backend does not support these options: {names}. Static-selector "
        "trace(intervene=tl.when(...)) edits and static-label save= selectors are "
        "supported; trace(stop_after=...), storage, and value-dependent predicates are "
        "not, because they need predicate-time concrete values or a way to truncate lazy "
        "UOp descendants before realize(). Use the PyTorch backend for them."
    ),
    fallback_message="",
    always_runtime=True,
)
"""Extra public-kwarg rejection policy for the tinygrad preview backend."""


PADDLE_EXTRA_KWARG_POLICY = ExtraKwargPolicy(
    runtime_option_names=frozenset(),
    runtime_message=(
        "paddle backend preview does not support these options: {names}. "
        "Live trace(intervene=...), trace(stop_after=...), facets=, and static-label "
        "save= selectors are supported on the eager Paddle preview; "
        "storage=, and the remaining torch-only runtime options are not. Use "
        "the PyTorch backend for storage= and stop-early capture shaping."
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
        "on_nonfinite": "ignore",
        "profile": False,
        "payload_policy": None,
        "chunk_size": None,
        "chunk_paths": None,
    },
)
"""Extra public-kwarg rejection policy for the Paddle preview backend.

``intervene`` and ``halt`` are absent because the Paddle capture path pops
and dispatches them before extra-kwarg rejection runs (the ``interventions``
capability flag owns both); ``facets`` is popped and snapshotted the same way
under the backend-neutral ``facet_recipes`` record.
"""


MLX_EXTRA_KWARG_POLICY = ExtraKwargPolicy(
    runtime_option_names=frozenset(),
    runtime_message=(
        "MLX backend preview does not support: {names}. Static-label save= "
        "selectors, static-label trace(intervene=tl.when(...)), and static-label "
        "trace(stop_after=...) are supported; storage= and value-dependent "
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
        "on_nonfinite": "ignore",
        "profile": False,
        "facets": None,
        "payload_policy": None,
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
            "storage",
            "streaming",
        }
    ),
    runtime_message=("tf backend preview does not support runtime-mutation options: {names}."),
    fallback_message="tf backend preview does not support: {names}.",
    inert_values={
        "lookback": 0,
        "lookback_payload_policy": "metadata_only",
    },
)
"""Extra public-kwarg rejection policy for the TensorFlow preview backend.

``intervene=`` and ``stop_after=`` (``halt``) are not in this table: both run live
in the tf eager capture path and are popped before this policy runs, as is
``facets`` (``facet_recipes`` record)."""


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
        "save_args": (
            "JAX backend preview does not support save_args; full-save forward capture "
            "only. Use full-save JAX trace capture or the PyTorch backend."
        ),
        "save_source_context": (
            "JAX backend preview does not support save_source_context; full-save forward capture "
            "only. Use full-save JAX trace capture or the PyTorch backend."
        ),
        "backward_ready": (
            "JAX backend preview does not support backward_ready; full-save forward capture "
            "only. Use tl.backends.jax.GradOptions for derived gradients."
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
    full_save_message="tinygrad backend preview is full-save only; save shaping is unsupported.",
    rejected_truthy_messages={
        name: (f"tinygrad backend preview does not support {name}; full-save forward capture only.")
        for name in (
            "activation_transform",
            "detach_saved_activations",
            "save_grads",
            "save_args",
            "save_rng",
            "backward_ready",
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
    full_save_message="paddle backend preview is full-save only; save shaping is unsupported.",
    rejected_truthy_messages={
        name: (f"paddle backend preview does not support {name}; full-save forward capture only.")
        for name in (
            "activation_transform",
            "detach_saved_activations",
            "save_grads",
            "save_args",
            "save_rng",
            "backward_ready",
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
    full_save_message=(
        "MLX backend preview capture_trace takes only save='all'; pass static save= "
        "selectors through tl.trace instead."
    ),
    rejected_truthy_messages={
        "save_grads": "MLX backend preview does not support save_grads; backward capture is unavailable.",
    }
    | {
        name: f"MLX backend preview does not support {name}; forward capture only."
        for name in (
            "activation_transform",
            "detach_saved_activations",
            "save_args",
            "save_rng",
            "backward_ready",
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
            "save_args",
            "save_rng",
            "backward_ready",
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


def resolve_optional_capture_field(capture: Any, field_name: str, flat_value: Any) -> Any:
    """Resolve one ``capture=`` field whose backend-local flat kwarg defaults to ``None``.

    Some preview backend ``capture_trace`` parameters predate ``CaptureOptions``
    and still spell a handful of optional fields (e.g. ``save_grads``,
    ``transform``) with a concrete ``None`` default instead
    of the ``MissingType`` sentinel, so ``is_missing()`` cannot tell "the
    caller left this unset" apart from an explicit ``capture=`` value the way
    it can for every ``MISSING``-defaulted field. ``flat_value is not None``
    is treated as an explicit direct/legacy flat-kwarg call and wins; otherwise
    an explicitly-set ``capture.field_name`` wins; otherwise the result stays
    ``None`` (the flat default), matching the field's pre-``capture=`` behavior.

    Parameters
    ----------
    capture:
        Grouped ``CaptureOptions`` supplied to ``trace()``, or ``None``.
    field_name:
        ``CaptureOptions`` attribute name to read when the flat value is unset.
    flat_value:
        Backend-local flat kwarg's current value (``None`` when unset).

    Returns
    -------
    Any
        The effective value for this capture knob.
    """

    if flat_value is not None:
        return flat_value
    if capture is not None and capture._is_field_explicit(field_name):
        return getattr(capture, field_name)
    return None


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
    code = _options_refusal_code(rejected)
    options = sorted(rejected)
    if policy.always_runtime or policy.runtime_option_names & set(rejected):
        raise BackendUnsupportedError(
            policy.runtime_message.format(names=names), code=code, options=options
        )
    raise BackendUnsupportedError(
        policy.fallback_message.format(names=names), code=code, options=options
    )


def _reject_truthy_trace_options(
    options: dict[str, Any],
    policy: PreviewTraceOptionPolicy,
    spec: BackendSpec | None,
) -> None:
    """Refuse every truthy option the policy rejects, typed by its capability gate."""

    for option_name, message in (policy.rejected_truthy_messages or {}).items():
        if options.get(option_name):
            if spec is not None:
                gate = TRACE_OPTION_CAPABILITY_GATES.get(option_name)
                if gate is not None and getattr(spec.capabilities, gate):
                    require_capability_implementation(spec, gate)
                    raise _undispatched_capability_error(spec, gate, option_name)
            raise BackendUnsupportedError(
                message,
                code=_option_refusal_code(option_name),
                backend=policy.backend_name,
                option=option_name,
            )


def _reject_save_trace_options(options: dict[str, Any], policy: PreviewTraceOptionPolicy) -> None:
    """Refuse raw-activation and save-window settings the policy does not support."""

    if policy.save_raw_activations_false_message is not None and not options.get(
        "save_raw_activations"
    ):
        raise BackendUnsupportedError(
            policy.save_raw_activations_false_message,
            code=_option_refusal_code("save_raw_activations"),
            backend=policy.backend_name,
            option="save_raw_activations",
        )
    if policy.save_window_message is not None and (
        options.get("lookback") != 0 or options.get("lookback_payload_policy") != "metadata_only"
    ):
        raise BackendUnsupportedError(
            policy.save_window_message,
            code=_option_refusal_code("lookback"),
            backend=policy.backend_name,
            option="lookback",
        )


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
        raise BackendUnsupportedError(
            policy.input_kwargs_message,
            code=_option_refusal_code("input_kwargs"),
            backend=policy.backend_name,
            option="input_kwargs",
        )
    if policy.full_save_message is not None and options.get("save") not in ("all", None):
        raise BackendUnsupportedError(
            policy.full_save_message,
            code=_option_refusal_code("save"),
            backend=policy.backend_name,
            option="save",
        )
    _reject_truthy_trace_options(options, policy, spec)
    if policy.output_device_message is not None and options.get("output_device") != "same":
        raise _output_device_refusal(policy)
    _reject_save_trace_options(options, policy)


def pop_stop_directive(options: dict[str, Any]) -> Any:
    """Pop ``halt`` and ``stop_after``; return the selector a preview backend halts on.

    ``stop_after=`` is the one public stop directive and the trace door passes
    ``halt=None``, so a preview backend that halts on a selector receives
    ``stop_after`` in that role. ``MISSING`` reads as absent.
    """

    halt = options.pop("halt", None)
    stop_after = options.pop("stop_after", None)
    chosen = halt if stop_after is None or stop_after is MISSING else stop_after
    return None if chosen is MISSING else chosen
