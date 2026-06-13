"""Capability flags for the tinygrad backend preview."""

from __future__ import annotations

supports_backward_capture = False
supports_fastlog = False
supports_intervention = False
supports_payload_materialization = False
supports_rng_replay = False
supports_validation_replay = True
module_identity_modes = ("function_root",)
payload_policy = "audit_only"
live_payload_policy = "dev_python_realized_copy"

__all__ = [
    "live_payload_policy",
    "module_identity_modes",
    "payload_policy",
    "supports_backward_capture",
    "supports_fastlog",
    "supports_intervention",
    "supports_payload_materialization",
    "supports_rng_replay",
    "supports_validation_replay",
]
