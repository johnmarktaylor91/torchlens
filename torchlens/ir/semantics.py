"""Capture policy and backend semantics records."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BackendSemantics:
    """Backend-specific operation facts normalized into portable scalars."""

    backend_grad_handle: object | None
    grad_fn_class_name: str | None
    autograd_memory: int | None
    num_autograd_tensors: int | None
    mutates_inputs: tuple[int, ...]
    bytes_delta_at_call: int | None
    bytes_peak_at_call: int | None


@dataclass(frozen=True, slots=True)
class CapturePolicy:
    """Resolved per-output capture policy."""

    must_keep_topology: bool
    save_payload: bool
    requires_isolation: bool
    save_args: bool
    save_code: bool
    save_rng: bool
    save_grad: bool
    stream: bool
