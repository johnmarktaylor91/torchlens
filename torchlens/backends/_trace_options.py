"""Per-backend trace-option vocabularies (leaf module: no registry import).

``torchlens.backends.registry`` re-exports every name here; the protocol shells
read their option list from this leaf so constructing a shell never imports
the registry that is constructing it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class _BackendCapability:
    """One backend capability, stated as data.

    A backend admits a capability in one of two ways, and the record says
    which. A gated capability is a ``BackendCapabilities`` boolean named
    ``name``: ``True`` opens its ``trace_options`` and must bind an
    implementing surface (``GATED_CAPABILITY_FLAGS``). A trace-option
    capability is admitted by listing its ``trace_options`` in
    ``BackendCapabilities.trace_options``. When ``public_spine`` is ``True``
    the public-option spine declares the options for every backend, so the
    dispatcher strips an omitted option and refuses an explicit one on a
    backend that does not list it; when ``False`` the options are
    backend-extra keywords only their owning backends spell.

    Parameters
    ----------
    name:
        Stable capability id. For a gated capability it is the
        ``BackendCapabilities`` flag name.
    trace_options:
        Public ``trace`` keywords the capability admits, in declaration order.
    gated:
        Whether ``name`` is a ``BackendCapabilities`` boolean gate.
    public_spine:
        Whether the public-option spine declares ``trace_options`` for every
        backend.
    """

    name: str
    trace_options: tuple[str, ...] = ()
    gated: bool = False
    public_spine: bool = False


_INTERVENTIONS = _BackendCapability(
    "can_intervene", ("intervene", "halt", "stop_after", "facets"), gated=True
)
_STREAMING = _BackendCapability("has_streaming", ("storage", "streaming"), gated=True)
_BACKWARD_CAPTURE = _BackendCapability(
    "backward_capture", ("save_grads", "backward_ready"), gated=True
)
_RNG_REPLAY = _BackendCapability("rng_replay", ("save_rng", "seed"), gated=True)
_STRUCTURE_ONLY = _BackendCapability("structure_only_capture", ("structure_only",), gated=True)
_FASTLOG = _BackendCapability("fastlog", gated=True)
_CONTROL_FLOW = _BackendCapability(
    "control_flow", ("jax_control_flow", "jax_max_control_flow_unroll"), public_spine=True
)
_MODULE_IDENTITY = _BackendCapability(
    "module_identity_mode", ("module_identity_mode",), public_spine=True
)
_CODEC_MATERIALIZATION = _BackendCapability(
    "codec_materialization", ("payload_policy",), public_spine=True
)
_CONTAINER_STRUCTURE = _BackendCapability(
    "container_structure", ("capture_container_structure",), public_spine=True
)
_INFERENCE_ONLY = _BackendCapability("inference_only", ("inference_only",), public_spine=True)
_FORWARD_CHUNKING = _BackendCapability(
    "forward_chunking", ("chunk_size", "chunk_paths"), public_spine=True
)
_STATIC_ARGNUMS = _BackendCapability("static_argnums", ("jax_static_argnums",))
_GRAD_OPTIONS = _BackendCapability("grad_options", ("grad_options",))

_BACKEND_CAPABILITIES: tuple[_BackendCapability, ...] = (
    _INTERVENTIONS,
    _STREAMING,
    _BACKWARD_CAPTURE,
    _RNG_REPLAY,
    _STRUCTURE_ONLY,
    _FASTLOG,
    _CONTROL_FLOW,
    _MODULE_IDENTITY,
    _CODEC_MATERIALIZATION,
    _CONTAINER_STRUCTURE,
    _INFERENCE_ONLY,
    _FORWARD_CHUNKING,
    _STATIC_ARGNUMS,
    _GRAD_OPTIONS,
)
"""Every backend capability, in table order.

The order is load-bearing: it fixes the order of the derived gate table, the
public-option spine and the composition ledger rows. Adding a capability
updates registry capabilities, ``CaptureOptions``, cache-key coverage, docs
and tests in one patch.
"""


def _trace_option_names(capabilities: Iterable[_BackendCapability]) -> tuple[str, ...]:
    """Flatten capability records into their public trace-option names, in order.

    Parameters
    ----------
    capabilities:
        Capability records in declaration order.

    Returns
    -------
    tuple[str, ...]
        The records' ``trace_options`` concatenated.
    """

    return tuple(option for capability in capabilities for option in capability.trace_options)


GATED_CAPABILITY_FLAGS: frozenset[str] = frozenset(
    capability.name for capability in _BACKEND_CAPABILITIES if capability.gated
)
"""Capability flags that open behavior gates and therefore require a bound
implementing surface (see ``BackendSpec.capability_implementations``)."""

_GATED_TRACE_OPTION_FLAGS: dict[str, str] = {
    option: capability.name
    for capability in _BACKEND_CAPABILITIES
    if capability.gated
    for option in capability.trace_options
}
"""Public trace option to the ``BackendCapabilities`` flag that owns it, in table order."""

PUBLIC_OPTION_SPINE_TRACE_OPTIONS: tuple[str, ...] = _trace_option_names(
    capability for capability in _BACKEND_CAPABILITIES if capability.public_spine
)
"""Trace options declared by the public-option API spine."""

_TORCH_TRACE_OPTION_CAPABILITIES: tuple[_BackendCapability, ...] = tuple(
    capability for capability in _BACKEND_CAPABILITIES if capability.public_spine
)
"""Torch admits the whole public-option spine.

Torch implements container-structure capture, inference-only capture,
semantic output decode, and forward chunking. Backend-preview-only controls
such as JAX control-flow selection, non-default module identity modes,
payload-policy overrides, and save-preview mode are value-validated by the
torch entrypoint rather than silently treated as operational options.
"""

TORCH_TRACE_OPTIONS: tuple[str, ...] = _trace_option_names(_TORCH_TRACE_OPTION_CAPABILITIES)
"""Public trace-option names accepted by torch."""

_JAX_TRACE_OPTION_CAPABILITIES: tuple[_BackendCapability, ...] = (
    _STATIC_ARGNUMS,
    _GRAD_OPTIONS,
    _CONTROL_FLOW,
    _MODULE_IDENTITY,
)
"""Trace-option capabilities the JAX preview backend implements."""

JAX_TRACE_OPTIONS: tuple[str, ...] = _trace_option_names(_JAX_TRACE_OPTION_CAPABILITIES)
"""Trace options implemented by the JAX preview backend."""

_IDENTITY_AND_GRADS: tuple[_BackendCapability, ...] = (_MODULE_IDENTITY, _GRAD_OPTIONS)
"""The two trace-option capabilities every eager preview backend implements."""

_MLX_TRACE_OPTION_CAPABILITIES = _IDENTITY_AND_GRADS
_TINYGRAD_TRACE_OPTION_CAPABILITIES = _IDENTITY_AND_GRADS
_PADDLE_TRACE_OPTION_CAPABILITIES = _IDENTITY_AND_GRADS
_TF_TRACE_OPTION_CAPABILITIES = _IDENTITY_AND_GRADS

MLX_TRACE_OPTIONS: tuple[str, ...] = _trace_option_names(_MLX_TRACE_OPTION_CAPABILITIES)
"""Trace options implemented by the MLX preview backend."""

TINYGRAD_TRACE_OPTIONS: tuple[str, ...] = _trace_option_names(_TINYGRAD_TRACE_OPTION_CAPABILITIES)
"""Trace options implemented by the tinygrad preview backend."""

PADDLE_TRACE_OPTIONS: tuple[str, ...] = _trace_option_names(_PADDLE_TRACE_OPTION_CAPABILITIES)
"""Trace options implemented by the Paddle preview backend."""

TF_TRACE_OPTIONS: tuple[str, ...] = _trace_option_names(_TF_TRACE_OPTION_CAPABILITIES)
"""Trace options implemented by the TensorFlow preview backend."""
