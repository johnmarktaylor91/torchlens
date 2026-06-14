"""Capability flags for the technical-preview MLX backend."""

from __future__ import annotations

supports_backward_capture = False
supports_fastlog = False
supports_intervention = False
supports_intermediate_derived_grads = False
supports_payload_materialization = True
supports_rng_replay = False
supports_compile_capture = False
payload_policy = "array_payloads"
module_identity_modes = ("function_root", "object_module")

__all__ = [
    "supports_backward_capture",
    "supports_compile_capture",
    "supports_fastlog",
    "supports_intervention",
    "supports_intermediate_derived_grads",
    "supports_payload_materialization",
    "supports_rng_replay",
    "module_identity_modes",
    "payload_policy",
]
