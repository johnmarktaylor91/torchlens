"""Public backend registry for TorchLens capture and validation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Final, Literal, TypeAlias

from ._backend_names import TORCH_BACKEND_NAME as TORCH_BACKEND_NAME
from ._backend_protocol import _install_spec_lookup, protocol_problems
from ._entry_refusals import refuse_backend_registered
from ._errors import (
    AmbiguousBackendError as AmbiguousBackendError,
    BackendCapabilityConformanceError as BackendCapabilityConformanceError,
    BackendMismatchError as BackendMismatchError,
    BackendPayloadUnsupportedError as BackendPayloadUnsupportedError,
    BackendRegistryError as BackendRegistryError,
    BackendRuntimeCompatibilityError as BackendRuntimeCompatibilityError,
    BackendUnsupportedError as BackendUnsupportedError,
    UnknownBackendError as UnknownBackendError,
    _restore_backend_error as _restore_backend_error,
)
from ._protocol import CaptureBackend
from ._trace_options import (
    _BACKEND_CAPABILITIES as _BACKEND_CAPABILITIES,
    _GATED_TRACE_OPTION_FLAGS as _GATED_TRACE_OPTION_FLAGS,
    GATED_CAPABILITY_FLAGS as GATED_CAPABILITY_FLAGS,
    JAX_TRACE_OPTIONS as JAX_TRACE_OPTIONS,
    MLX_TRACE_OPTIONS as MLX_TRACE_OPTIONS,
    PADDLE_TRACE_OPTIONS as PADDLE_TRACE_OPTIONS,
    PUBLIC_OPTION_SPINE_TRACE_OPTIONS as PUBLIC_OPTION_SPINE_TRACE_OPTIONS,
    TF_TRACE_OPTIONS as TF_TRACE_OPTIONS,
    TINYGRAD_TRACE_OPTIONS as TINYGRAD_TRACE_OPTIONS,
    TORCH_TRACE_OPTIONS as TORCH_TRACE_OPTIONS,
    _BackendCapability as _BackendCapability,
    _trace_option_names as _trace_option_names,
)

BackendName: TypeAlias = Literal["torch", "mlx", "jax", "tinygrad", "paddle", "tf", "fake"] | str
"""Backend name accepted by public APIs.

The ``"fake"`` literal is reserved for tests and downstream conformance fixtures
that register process-local specs; TorchLens intentionally does not ship a
default fake backend.
"""

JAX_BACKEND_NAME: Final[BackendName] = "jax"
"""Canonical registry name of the JAX backend."""

TINYGRAD_BACKEND_NAME: Final[BackendName] = "tinygrad"
"""Canonical registry name of the tinygrad backend."""

CanHandleFn: TypeAlias = Callable[[object, object, dict[Any, Any] | None], bool]
CaptureTraceFn: TypeAlias = Callable[..., Any]
ValidateEntryFn: TypeAlias = Callable[..., bool]
ValidateTraceFn: TypeAlias = Callable[..., Any]
CaptureBackendFactory: TypeAlias = Callable[[], CaptureBackend]


@dataclass(frozen=True)
class BackendCapabilities:
    """Consolidated capability flags for a registered backend.

    Parameters
    ----------
    backward_capture:
        Whether true backward graph capture is supported.
    has_validation_replay:
        Whether the backend can perform replay validation.
    fastlog:
        Whether sparse ``tl.record`` capture is supported.
    can_intervene:
        Whether live intervention capture is supported.
    rng_replay:
        Whether operation-level RNG replay is supported.
    payload_materialization:
        Whether loaded payloads can materialize as runtime arrays.
    has_streaming:
        Whether streaming save is supported.
    structure_only_capture:
        Whether structure-only capture (``structure_only=True``) is
        supported. DOCUMENTED-UNSTABLE surface; its spelling may still change.
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

    Examples
    --------
    >>> caps = BackendCapabilities(False, False, False, False, False, False, False)
    >>> (caps.fastlog, caps.save_levels[0])
    (False, 'audit')
    >>> tl.backends.spec("torch").capabilities.backward_capture
    True
    """

    backward_capture: bool
    has_validation_replay: bool
    fastlog: bool
    can_intervene: bool
    rng_replay: bool
    payload_materialization: bool
    has_streaming: bool
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
        Backend identifier used by ``backend=`` and ``Trace.provenance.backend``.
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
    protocol:
        Lazy factory for this backend's ``BackendProtocol`` object (the one
        protocol every backend fills; ``_backend_protocol.py``). Validated at
        registration like ``capture_backend``.

    Examples
    --------
    >>> demo = BackendSpec(name="demo", can_handle=lambda m, a, k: False,
    ...     capture_trace=lambda *a, **k: None, validate_entry=lambda *a, **k: None,
    ...     validate_trace=lambda *a, **k: None,
    ...     capabilities=tl.backends.BackendCapabilities(*[False] * 7))
    >>> (demo.name, demo.priority)
    ('demo', 0)
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
    protocol: Callable[[], Any] | None = None


_REGISTRY: dict[str, BackendSpec] = {}

_CAPTURE_BACKEND_REQUIRED_ATTRIBUTES: tuple[str, ...] = (
    "active_logging",
    "apply_live_hooks",
    "build_record_context",
    "cleanup_failed_forward_session",
    "cleanup_forward_memory",
    "cleanup_stopped_forward_session",
    "cleanup_model_session",
    "detect_backend_semantics",
    "emit_function_outputs",
    "extract_and_mark_outputs",
    "_fetch_label_move_input_tensors",
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


def _validate_protocol_factory(spec: BackendSpec) -> None:
    """Validate a supplied ``BackendProtocol`` factory.

    Parameters
    ----------
    spec:
        Backend spec being registered.

    Returns
    -------
    None
        Returns when the protocol object (if any) satisfies every member of
        the protocol.

    Raises
    ------
    TypeError
        When the protocol object misses members or contradicts the spec.
    """

    if spec.protocol is None:
        return
    try:
        candidate = spec.protocol()
    except ImportError as exc:
        if str(spec.name) == "torch" and "partially initialized module" in str(exc):
            return
        raise
    problems = protocol_problems(candidate, expected_name=str(spec.name))
    if problems:
        raise TypeError(
            f"Backend {spec.name!r} protocol does not satisfy BackendProtocol: "
            + "; ".join(problems)
            + "."
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

    Examples
    --------
    >>> torch_spec = tl.backends.spec("torch")
    >>> require_capability_implementation(torch_spec, "fastlog") is not None
    True
    >>> import dataclasses
    >>> bare = dataclasses.replace(torch_spec, capability_implementations={})
    >>> try:
    ...     require_capability_implementation(bare, "fastlog")
    ... except tl.errors.BackendCapabilityConformanceError as err:
    ...     print(err.code)
    backend_capability_conformance
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

    Examples
    --------
    >>> demo = tl.backends.BackendSpec(name="demo", can_handle=lambda m, a, k: False,
    ...     capture_trace=lambda *a, **k: None, validate_entry=lambda *a, **k: None,
    ...     validate_trace=lambda *a, **k: None,
    ...     capabilities=tl.backends.BackendCapabilities(*[False] * 7))
    >>> register(demo)
    >>> "demo" in [s.name for s in tl.backends.specs()]
    True
    >>> tl.backends.unregister("demo")
    """

    names = (str(spec.name), *spec.aliases)
    for name in names:
        if not replace and name in _REGISTRY:
            refuse_backend_registered(name)
    _validate_capture_backend_factory(spec)
    _validate_protocol_factory(spec)
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

    Examples
    --------
    >>> demo = tl.backends.BackendSpec(name="demo", can_handle=lambda m, a, k: False,
    ...     capture_trace=lambda *a, **k: None, validate_entry=lambda *a, **k: None,
    ...     validate_trace=lambda *a, **k: None,
    ...     capabilities=tl.backends.BackendCapabilities(*[False] * 7))
    >>> tl.backends.register(demo)
    >>> unregister("demo")
    >>> "demo" in [s.name for s in tl.backends.specs()]
    False
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

    Examples
    --------
    >>> spec("torch").name
    'torch'
    >>> try:
    ...     spec("no_such_backend")
    ... except tl.errors.UnknownBackendError as err:
    ...     print(err.code)
    unknown_backend
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

    Examples
    --------
    >>> names = [s.name for s in specs()]
    >>> "torch" in names
    True
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
    inputs: object,
    input_kwargs: dict[Any, Any] | None = None,
) -> BackendSpec:
    """Resolve the backend spec for a public call.

    Parameters
    ----------
    backend:
        Explicit backend name, or ``None`` for detector-based resolution.
    model:
        Candidate model or callable.
    inputs:
        Positional input object supplied by the user.
    input_kwargs:
        Keyword input mapping supplied by the user.

    Returns
    -------
    BackendSpec
        Resolved backend spec.

    Examples
    --------
    >>> resolve(None, model, (x,)).name
    'torch'
    >>> resolve("torch", model, (x,)).name
    'torch'
    """

    if backend is not None:
        spec = get_backend_spec(str(backend))
        if not spec.coercible and not spec.can_handle(model, inputs, input_kwargs):
            raise BackendMismatchError(
                f"backend={backend!r} cannot handle model type {type(model).__qualname__}."
            )
        return spec

    matches = [
        spec for spec in registered_backend_specs() if spec.can_handle(model, inputs, input_kwargs)
    ]
    if not matches:
        return get_backend_spec("torch")
    max_priority = max(spec.priority for spec in matches)
    winners = [spec for spec in matches if spec.priority == max_priority]
    if len(winners) > 1:
        names = ", ".join(sorted(str(spec.name) for spec in winners))
        raise AmbiguousBackendError(f"backend=None is ambiguous for: {names}.")
    return winners[0]


_install_spec_lookup(get_backend_spec)
