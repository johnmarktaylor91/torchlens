"""Public backend registry for TorchLens capture and validation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Final, Literal, TypeAlias, cast

from ..errors._base import ConfigurationError
from ._protocol import CaptureBackend

BackendName: TypeAlias = Literal["torch", "mlx", "jax", "tinygrad", "paddle", "tf", "fake"] | str
"""Backend name accepted by public APIs.

The ``"fake"`` literal is reserved for tests and downstream conformance fixtures
that register process-local specs; TorchLens intentionally does not ship a
default fake backend.
"""

TORCH_BACKEND_NAME: Final[BackendName] = "torch"
"""Canonical registry name of the default torch backend.

Public (non-``backends``) code that branches on backend identity must compare
against this constant, never a hard-coded literal — the backend-literal gate
in ``tests/test_backend_registry.py`` enforces it.
"""

JAX_BACKEND_NAME: Final[BackendName] = "jax"
"""Canonical registry name of the JAX preview backend."""

TINYGRAD_BACKEND_NAME: Final[BackendName] = "tinygrad"
"""Canonical registry name of the tinygrad preview backend."""

CanHandleFn: TypeAlias = Callable[[object, object, dict[Any, Any] | None], bool]
CaptureTraceFn: TypeAlias = Callable[..., Any]
ValidateEntryFn: TypeAlias = Callable[..., bool]
ValidateTraceFn: TypeAlias = Callable[..., Any]
CaptureBackendFactory: TypeAlias = Callable[[], CaptureBackend]


def _restore_backend_error(
    error_type: type[BaseException],
    args: tuple[object, ...],
    state: dict[str, object],
) -> BaseException:
    """Rebuild a backend error without appending its remedy a second time.

    Parameters
    ----------
    error_type:
        Concrete backend error class stored by pickle.
    args:
        Already-formatted ``BaseException.args`` tuple.
    state:
        Instance dictionary containing structured diagnostic fields.

    Returns
    -------
    BaseException
        Restored backend error with its exact message and fields.
    """

    error = error_type.__new__(error_type)
    BaseException.__init__(error, *args)
    error.__dict__.update(state)
    return error


TRACE_OPTION_CAPABILITY_EPOCHS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "epoch1_module_modes_control_flow",
        (
            "jax_control_flow",
            "jax_max_control_flow_unroll",
            "module_identity_mode",
        ),
    ),
    ("epoch2_tinygrad_t1", ()),
    ("epoch3_codec_materialization", ("payload_policy", "save_preview")),
    ("epoch4_intermediate_derived_grads", ()),
    ("epoch5_container_structure", ("capture_container_structure",)),
    ("epoch6_inference_only", ("inference_only",)),
    ("epoch7_semantic_output_decode", ("output_style", "output_head")),
    ("epoch8_forward_chunking", ("chunk_size", "chunk_paths")),
)
"""Ordered public trace-option capability epochs.

Each epoch must update registry capabilities, ``CaptureOptions``, cache-key
coverage, docs, and tests in one patch.
"""

PUBLIC_OPTION_SPINE_TRACE_OPTIONS: tuple[str, ...] = tuple(
    option for _epoch_name, options in TRACE_OPTION_CAPABILITY_EPOCHS for option in options
)
"""Trace options declared by the public-option API spine."""

TORCH_TRACE_OPTIONS: tuple[str, ...] = PUBLIC_OPTION_SPINE_TRACE_OPTIONS
"""Public trace-option names accepted by torch.

Torch implements container-structure capture, inference-only capture,
semantic output decode, and forward chunking. Backend-preview-only controls
such as JAX control-flow selection, non-default module identity modes,
payload-policy overrides, and save-preview mode are value-validated by the
torch entrypoint rather than silently treated as operational options.
"""

JAX_TRACE_OPTIONS: tuple[str, ...] = (
    "jax_static_argnums",
    "grad_options",
    "jax_control_flow",
    "jax_max_control_flow_unroll",
    "module_identity_mode",
)
"""Trace options implemented by the JAX preview backend."""

MLX_TRACE_OPTIONS: tuple[str, ...] = ("module_identity_mode", "grad_options")
"""Trace options implemented by the MLX preview backend."""

TINYGRAD_TRACE_OPTIONS: tuple[str, ...] = ("module_identity_mode", "grad_options")
"""Trace options implemented by the tinygrad preview backend."""

PADDLE_TRACE_OPTIONS: tuple[str, ...] = ("module_identity_mode", "grad_options")
"""Trace options implemented by the Paddle preview backend."""

TF_TRACE_OPTIONS: tuple[str, ...] = ("module_identity_mode", "grad_options")
"""Trace options implemented by the TensorFlow preview backend."""


class BackendRegistryError(ConfigurationError, ValueError):
    """Base class for backend registry failures."""

    code: str = "backend_error"
    default_remedy: str = "pass an explicitly registered backend compatible with the operation"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        remedy: str | None = None,
        **context: object,
    ) -> None:
        """Initialize a backend refusal with its stable code and remedy.

        Parameters
        ----------
        message:
            Description of the rejected backend request and its cause.
        code:
            Stable refusal code. The class code is used when omitted; a raise
            site passes it to declare its contracted code where it raises.
        remedy:
            Concrete caller action. The class-specific default is used when omitted.
        **context:
            Structured, non-authoritative diagnostic context.
        """

        resolved_remedy = remedy or type(self).default_remedy
        message_text = message.rstrip()
        if not message_text.endswith((".", "!", "?", ":", ";")):
            message_text = f"{message_text}."
        super().__init__(
            f"{message_text} Remedy: {resolved_remedy.rstrip().rstrip('.')}.",
            code=code or type(self).code,
            remedy=resolved_remedy,
            **cast(dict[str, Any], context),
        )

    def __reduce__(
        self,
    ) -> tuple[
        object,
        tuple[type[BaseException], tuple[object, ...], dict[str, object]],
    ]:
        """Return a pickle reconstruction recipe preserving structured fields."""

        return (
            _restore_backend_error,
            (type(self), self.args, dict(self.__dict__)),
        )


class UnknownBackendError(BackendRegistryError):
    """Raised when an explicit backend name is not registered."""

    code = "unknown_backend"
    default_remedy = "pass one of the backend names listed by torchlens.backends"


class BackendMismatchError(BackendRegistryError):
    """Raised when an explicit backend cannot handle the supplied model/input."""

    code = "backend_mismatch"
    default_remedy = "select the backend that owns the supplied model and tensor inputs"


class BackendAmbiguityError(BackendRegistryError):
    """Raised when backend auto-resolution has multiple equal-priority matches."""

    code = "backend_ambiguity"
    default_remedy = "pass backend= explicitly to select one matching backend"


class BackendUnsupportedError(BackendRegistryError, NotImplementedError):
    """Raised when a backend lacks a requested capability."""

    code = "backend_unsupported"
    default_remedy = "omit the unsupported option or use a backend that implements it"


class BackendPayloadUnsupportedError(BackendUnsupportedError):
    """Raised when an audit-only backend payload cannot materialize."""

    code = "backend_payload_unsupported"
    default_remedy = "save metadata only or use a backend with a supported payload codec"


class BackendRuntimeCompatibilityError(BackendRegistryError):
    """Raised when a backend runtime is incompatible with serialized metadata."""

    code = "backend_runtime_compatibility"
    default_remedy = "install a compatible backend runtime or load the artifact for analysis only"


class BackendCapabilityConformanceError(BackendUnsupportedError):
    """Raised when a ``True`` capability flag has no registered implementation.

    The capability table is a public promise: ``True`` means supported. A
    flag flipped to ``True`` on a spec that registers no implementing surface
    for that capability must refuse typed instead of silently admitting the
    option and dropping the behavior.
    """

    code = "backend_capability_conformance"
    default_remedy = "disable the advertised capability or register its implementing surface"


GATED_CAPABILITY_FLAGS: frozenset[str] = frozenset(
    {
        "backward_capture",
        "fastlog",
        "interventions",
        "rng_replay",
        "streaming",
        "structure_only_capture",
    }
)
"""Capability flags that open behavior gates and therefore require a bound
implementing surface (see ``BackendSpec.capability_implementations``)."""


@dataclass(frozen=True)
class BackendCapabilities:
    """Consolidated capability flags for a registered backend.

    Parameters
    ----------
    backward_capture:
        Whether true backward graph capture is supported.
    validation_replay:
        Whether the backend can perform replay validation.
    fastlog:
        Whether sparse ``tl.record`` capture is supported.
    interventions:
        Whether live intervention capture is supported.
    rng_replay:
        Whether operation-level RNG replay is supported.
    payload_materialization:
        Whether loaded payloads can materialize as runtime arrays.
    streaming:
        Whether streaming save is supported.
    structure_only_capture:
        Whether structure-only capture (``structure_only=True``) is
        supported. DOCUMENTED-UNSTABLE surface pending naming-session/S2
        ratification.
    intermediate_derived_grads:
        Whether the backend can derive exact op-level gradients outside true
        backward capture.
    input_container_structure:
        Public input-container structure level. ``"none"`` means no stable
        paths/specs, ``"paths_only"`` means path metadata may exist but is not
        generally reconstructable, and ``"full_spec"`` means full portable
        ``ContainerSpec`` metadata is supported.
    output_container_structure:
        Public output-container structure level. ``"none"`` means no stable
        paths/specs, ``"paths_only"`` means path metadata may exist but is not
        generally reconstructable, and ``"full_spec"`` means full portable
        ``ContainerSpec`` metadata is supported.
    module_identity_modes:
        Supported module identity modes.
    save_levels:
        Supported portable save levels.
    trace_options:
        Backend-specific public ``trace`` keyword options accepted by this
        backend in addition to the backend-neutral options.
    """

    backward_capture: bool
    validation_replay: bool
    fastlog: bool
    interventions: bool
    rng_replay: bool
    payload_materialization: bool
    streaming: bool
    structure_only_capture: bool = False
    intermediate_derived_grads: bool = False
    input_container_structure: Literal["none", "paths_only", "full_spec"] = "none"
    output_container_structure: Literal["none", "paths_only", "full_spec"] = "none"
    module_identity_modes: tuple[str, ...] = ("torch_module",)
    save_levels: tuple[str, ...] = ("audit", "executable_with_callables", "portable")
    trace_options: tuple[str, ...] = ()


@dataclass(frozen=True)
class SerializationPolicy:
    """Backend-owned serialization policy placeholder.

    Parameters
    ----------
    payload_policy:
        Public payload policy literal.
    body_format:
        Manifest body format literal.
    manifest_schema_versions:
        Manifest schema versions this backend can load.
    runtime_name:
        Runtime fingerprint name written in backend-aware manifests.
    """

    payload_policy: str = "full"
    body_format: str = "safetensors"
    manifest_schema_versions: tuple[int, ...] = (1, 2)
    runtime_name: str | None = None


@dataclass(frozen=True)
class BackendSpec:
    """Public backend registry unit.

    Parameters
    ----------
    name:
        Backend identifier used by ``backend=`` and ``Trace.backend``.
    can_handle:
        Side-effect-light detector used only for ``backend=None`` or mismatch checks.
    capture_trace:
        Public trace entry for this backend.
    validate_entry:
        Public validation entry for model/input validation.
    validate_trace:
        Validation entry for an already-built trace.
    capabilities:
        Consolidated capability flags.
    capture_backend:
        Optional factory for the lower-level Protocol adapter used by shared
        capture orchestration.
    serialization_policy:
        Backend-owned serialization policy.
    priority:
        Auto-resolution priority. Equal-priority matches are ambiguous.
    coercible:
        Whether explicit resolution may accept inputs ``can_handle`` returns false for.
    aliases:
        Alternate explicit names.
    capability_implementations:
        Lazy factories for the implementing surface of each ``True`` gated
        capability flag (``GATED_CAPABILITY_FLAGS``). Gates open only through
        :func:`require_capability_implementation`, never through the boolean
        alone, so a flag flip without a registered implementation refuses
        typed instead of silently admitting unimplemented behavior.
    """

    name: BackendName
    can_handle: CanHandleFn
    capture_trace: CaptureTraceFn
    validate_entry: ValidateEntryFn
    validate_trace: ValidateTraceFn
    capabilities: BackendCapabilities
    capture_backend: CaptureBackendFactory | None = None
    serialization_policy: SerializationPolicy = field(default_factory=SerializationPolicy)
    priority: int = 0
    coercible: bool = False
    aliases: tuple[str, ...] = ()
    # compare=False keeps the frozen spec hashable (dicts are not) and
    # registry replacement uses identity, not binding equality.
    capability_implementations: dict[str, Callable[[], object]] | None = field(
        default=None, compare=False
    )


_REGISTRY: dict[str, BackendSpec] = {}

_CAPTURE_BACKEND_REQUIRED_ATTRIBUTES: tuple[str, ...] = (
    "active_logging",
    "apply_live_hooks",
    "build_record_context",
    "cleanup_failed_forward_session",
    "cleanup_forward_memory",
    "cleanup_halted_forward_session",
    "cleanup_model_session",
    "copy_replacement_metadata",
    "detect_backend_semantics",
    "emit_function_outputs",
    "extract_and_mark_outputs",
    "fetch_label_move_input_tensors",
    "finalize_forward_session",
    "inference_context",
    "is_parameter",
    "is_tensor",
    "is_wrapped",
    "log_source_tensor",
    "name",
    "pause_logging",
    "pop_module_frame",
    "prepare_model",
    "prepare_model_once",
    "prepare_model_session",
    "push_existing_module_frame",
    "restore_rng",
    "safe_copy",
    "set_tensor_label",
    "setup_inputs_and_device",
    "seed_rng",
    "set_capture_producer_policy",
    "snapshot_rng",
    "start_session",
    "supports_backward_capture",
    "tensor_ref",
    "unwrap",
    "wrap",
)
"""Runtime attributes required when a spec exposes a shared capture adapter."""


def _validate_capture_backend_factory(spec: BackendSpec) -> None:
    """Validate a supplied shared capture backend factory.

    Parameters
    ----------
    spec:
        Backend spec being registered.

    Returns
    -------
    None
        Returns when no factory is supplied or all required attributes exist.
    """

    if spec.capture_backend is None:
        return
    try:
        backend = spec.capture_backend()
    except ImportError as exc:
        if str(spec.name) == "torch" and "partially initialized module" in str(exc):
            return
        raise
    missing = [name for name in _CAPTURE_BACKEND_REQUIRED_ATTRIBUTES if not hasattr(backend, name)]
    if missing:
        names = ", ".join(sorted(missing))
        raise TypeError(
            f"Backend {spec.name!r} capture_backend is missing CaptureBackend "
            f"attribute(s): {names}."
        )


def _validate_capability_implementations(spec: BackendSpec) -> None:
    """Require an implementation factory for every ``True`` gated flag.

    Parameters
    ----------
    spec:
        Backend spec being registered.

    Returns
    -------
    None
        Returns when every ``True`` gated capability flag has a factory.

    Raises
    ------
    BackendCapabilityConformanceError
        If a gated flag is ``True`` without a registered implementation
        factory. Factories are not called here so registration stays
        import-light; :func:`require_capability_implementation` resolves them
        at gate time.
    """

    implementations = spec.capability_implementations or {}
    missing = [
        flag
        for flag in sorted(GATED_CAPABILITY_FLAGS)
        if getattr(spec.capabilities, flag) and implementations.get(flag) is None
    ]
    if missing:
        names = ", ".join(missing)
        raise BackendCapabilityConformanceError(
            f"Backend {spec.name!r} declares capability flag(s) {names} as True "
            "without binding an implementing surface in "
            "capability_implementations. A boolean flip alone must not admit "
            "unimplemented behavior; register the implementation factory or "
            "keep the flag False."
        )


def require_capability_implementation(spec: BackendSpec, flag: str) -> object:
    """Resolve the implementing surface behind a ``True`` gated capability flag.

    Parameters
    ----------
    spec:
        Backend spec whose gate is being opened.
    flag:
        Gated capability flag name from ``GATED_CAPABILITY_FLAGS``.

    Returns
    -------
    object
        The non-``None`` implementing surface the factory resolves to.

    Raises
    ------
    BackendCapabilityConformanceError
        If the flag has no bound factory, the factory raises, or it resolves
        to ``None`` — i.e. the flag promises support the backend does not
        actually register.
    """

    implementations = spec.capability_implementations or {}
    factory = implementations.get(flag)
    if factory is None:
        raise BackendCapabilityConformanceError(
            f"Backend {spec.name!r} reports capability {flag!r} as True but "
            "registers no implementing surface for it; refusing instead of "
            "silently ignoring the requested behavior."
        )
    try:
        implementation = factory()
    except Exception as exc:
        raise BackendCapabilityConformanceError(
            f"Backend {spec.name!r} capability {flag!r} implementation factory "
            f"failed to resolve: {exc}"
        ) from exc
    if implementation is None:
        raise BackendCapabilityConformanceError(
            f"Backend {spec.name!r} capability {flag!r} implementation factory "
            "resolved to None; the capability is not actually implemented."
        )
    return implementation


def register_backend_spec(spec: BackendSpec, *, replace: bool = False) -> None:
    """Register a backend spec.

    Parameters
    ----------
    spec:
        Backend spec to register.
    replace:
        Whether to replace an existing spec with the same name or alias.

    Returns
    -------
    None
        The process-local backend registry is updated.
    """

    names = (str(spec.name), *spec.aliases)
    for name in names:
        if not replace and name in _REGISTRY:
            raise ValueError(f"Backend {name!r} is already registered.")
    _validate_capture_backend_factory(spec)
    _validate_capability_implementations(spec)
    if replace:
        replaced_specs = {
            existing_spec for name in names if (existing_spec := _REGISTRY.get(name)) is not None
        }
        for existing_spec in replaced_specs:
            for registered_name, registered_spec in list(_REGISTRY.items()):
                if registered_spec is existing_spec:
                    del _REGISTRY[registered_name]
    for name in names:
        _REGISTRY[name] = spec


def unregister_backend_spec(name: str) -> None:
    """Remove a backend spec from the process-local registry.

    Parameters
    ----------
    name:
        Registered backend name or alias.

    Returns
    -------
    None
        The matching spec entries are removed.
    """

    spec = _REGISTRY.pop(name, None)
    if spec is None:
        return
    for alias in (str(spec.name), *spec.aliases):
        if _REGISTRY.get(alias) is spec:
            del _REGISTRY[alias]


def get_backend_spec(name: str) -> BackendSpec:
    """Return a registered backend spec by name.

    Parameters
    ----------
    name:
        Registered backend name or alias.

    Returns
    -------
    BackendSpec
        Matching backend spec.
    """

    try:
        return _REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise UnknownBackendError(
            f"Unknown backend {name!r}. Registered backends: {known}."
        ) from exc


def registered_backend_specs() -> tuple[BackendSpec, ...]:
    """Return unique registered backend specs.

    Returns
    -------
    tuple[BackendSpec, ...]
        Registered specs in first-registration order.
    """

    seen: set[int] = set()
    specs: list[BackendSpec] = []
    for spec in _REGISTRY.values():
        if id(spec) in seen:
            continue
        seen.add(id(spec))
        specs.append(spec)
    return tuple(specs)


def resolve_backend_spec(
    backend: BackendName | None,
    model: object,
    input_args: object,
    input_kwargs: dict[Any, Any] | None = None,
) -> BackendSpec:
    """Resolve the backend spec for a public call.

    Parameters
    ----------
    backend:
        Explicit backend name, or ``None`` for detector-based resolution.
    model:
        Candidate model or callable.
    input_args:
        Positional input object supplied by the user.
    input_kwargs:
        Keyword input mapping supplied by the user.

    Returns
    -------
    BackendSpec
        Resolved backend spec.
    """

    if backend is not None:
        spec = get_backend_spec(str(backend))
        if not spec.coercible and not spec.can_handle(model, input_args, input_kwargs):
            raise BackendMismatchError(
                f"backend={backend!r} cannot handle model type {type(model).__qualname__}."
            )
        return spec

    matches = [
        spec
        for spec in registered_backend_specs()
        if spec.can_handle(model, input_args, input_kwargs)
    ]
    if not matches:
        return get_backend_spec("torch")
    max_priority = max(spec.priority for spec in matches)
    winners = [spec for spec in matches if spec.priority == max_priority]
    if len(winners) > 1:
        names = ", ".join(sorted(str(spec.name) for spec in winners))
        raise BackendAmbiguityError(f"backend=None is ambiguous for: {names}.")
    return winners[0]
