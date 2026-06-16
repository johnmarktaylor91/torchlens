"""Capability flags for the TensorFlow backend preview."""

from __future__ import annotations

from ..registry import TF_TRACE_OPTIONS

supports_backward_capture = False
supports_fastlog = False
supports_intervention = False
supports_intermediate_derived_grads = False
supports_payload_materialization = True
supports_rng_replay = False
supports_streaming = False
supports_validation_replay = True
input_container_structure = "paths_only"
output_container_structure = "paths_only"
module_identity_modes = ("function_root", "object_module")
payload_policy = "array_payloads"
trace_options = TF_TRACE_OPTIONS

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
