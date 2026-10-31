"""Registry-backed capability exports for the TensorFlow backend preview."""

from __future__ import annotations

from ..registry import get_backend_spec

_SPEC = get_backend_spec("tf")
_CAPABILITIES = _SPEC.capabilities

supports_backward_capture = _CAPABILITIES.backward_capture
supports_fastlog = _CAPABILITIES.fastlog
supports_intervention = _CAPABILITIES.interventions
supports_intermediate_derived_grads = _CAPABILITIES.intermediate_derived_grads
supports_payload_materialization = _CAPABILITIES.payload_materialization
supports_rng_replay = _CAPABILITIES.rng_replay
supports_streaming = _CAPABILITIES.streaming
supports_validation_replay = _CAPABILITIES.validation_replay
input_container_structure = _CAPABILITIES.input_container_structure
output_container_structure = _CAPABILITIES.output_container_structure
module_identity_modes = _CAPABILITIES.module_identity_modes
payload_policy = _SPEC.serialization_policy.payload_policy
trace_options = _CAPABILITIES.trace_options

__all__ = [
    "input_container_structure",
    "module_identity_modes",
    "output_container_structure",
    "payload_policy",
    "supports_backward_capture",
    "supports_fastlog",
    "supports_intermediate_derived_grads",
    "supports_intervention",
    "supports_payload_materialization",
    "supports_rng_replay",
    "supports_streaming",
    "supports_validation_replay",
    "trace_options",
]
