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
    """Resolved per-output capture policy.

    R47-2: only the three CONSUMED facts remain. The six former fields
    (must_keep_topology, requires_isolation, save_args, save_code, save_rng,
    stream) were declaration-only -- populated at every constructor site,
    four of them computed per-op on the torch hot path, then never read
    anywhere (stream was hardcoded False everywhere).
    """

    save_payload: bool
    save_grad: bool
    save_mode: SaveMode = "copy"
