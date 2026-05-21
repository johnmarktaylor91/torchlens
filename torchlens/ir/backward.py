"""Backward-capture sidecar records for backend-neutral Trace projection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BackwardSidecar:
    """Runtime-only backward capture summary attached to a finalized Trace."""

    backend_name: str
    has_backward_pass: bool
    grad_fn_logs: object
    grad_fn_order: tuple[int, ...]
    backward_root_grad_fn_ids: int | None
    backward_num_calls: int
    backward_peak_memory: int | None
    backward_memory_backend: str | None
