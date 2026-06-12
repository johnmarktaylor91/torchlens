"""Capture policy and backend semantics records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SaveMode = Literal["copy", "reference", "view", "cpu_async"]


@dataclass(frozen=True, slots=True)
class BackendSemantics:
    """Backend-specific operation facts normalized into portable scalars."""

    backend_grad_handle: object | None
    grad_fn_class_name: str | None
    autograd_memory: int | None
    num_autograd_tensors: int | None
    mutated_input_positions: tuple[object, ...]
    aliased_output_inputs: tuple[object, ...]
    unknown_aliasing: bool
    bytes_delta_at_call: int | None
    bytes_peak_at_call: int | None

    @property
    def mutates_inputs(self) -> tuple[object, ...]:
        """Return legacy mutation positions for compatibility.

        Returns
        -------
        tuple[object, ...]
            Input positions known to be mutated by the backend operation.
        """

        return self.mutated_input_positions


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
    save_mode: SaveMode = "copy"
