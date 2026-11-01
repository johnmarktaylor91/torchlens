"""Torch's ``BackendProtocol`` object: the reference, by delegation.

Delegates to the registered torch spec and the public validation entry. It
changes nothing about torch capture; the capture-pipeline rebuild replaces
this adapter with a native implementation and the conformance suite's torch
cells are that rebuild's acceptance test.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .._backend_protocol import FrameworkRequirement, SpecBackedProtocol

TORCH_FRAMEWORK = FrameworkRequirement(
    distributions=("torch",),
    minimum="2.1",
    below=None,
    tested_through="2.13.0",
    pinned="2.13.0",
    extra="",
)


class TorchProtocol(SpecBackedProtocol):
    """The reference backend's protocol object (delegating adapter)."""

    def __init__(self) -> None:
        from ..registry import TORCH_TRACE_OPTIONS

        super().__init__(
            name="torch",
            maturity="stable",
            framework=TORCH_FRAMEWORK,
            materialization="eager",
            trace_options=TORCH_TRACE_OPTIONS,
            refusal_codes=(
                "trace_option_unsupported",
                "value_dependent_branch_unsupported",
                "followed_by_unsupported",
                "torchscript_boundary",
                "scripted_function_not_captured",
                "foreign_kernel_boundary",
                "unsupported_tensor_variant",
                "output_unsupported_tensor_variant",
                "subclass_ctor_under_dispatch_mode_unsupported",
                "predicate_return_invalid",
                "structure_only_substrate_mismatch",
                "lookback_payload_policy_conflict",
                "legacy_constructor_uncaptured",
                "input_rung_requires_module_root",
                "held_tensor_scan_truncated",
                "meta_kernel_unavailable",
                "emit_nvtx_unavailable",
                "dlpack_call_not_captured",
                "output_attribution_failed",
                "wrappers_removed_before_capture",
                "unwrap_during_active_capture",
                "release_during_active_capture",
            ),
        )

    def runtime_available(self) -> str | None:
        """Torch is a base dependency; only the range check applies."""

        return super().runtime_available()

    def reference_output(self, model: Any, args: Any, kwargs: dict[str, Any] | None) -> Any:
        """Run the model on pristine torch: wrappers removed for the call."""

        from ...validation._pristine import pristine_torch_oracle

        with pristine_torch_oracle():
            return super().reference_output(model, args, kwargs)

    def is_tensor(self, value: Any) -> bool:
        """Whether ``value`` is a ``torch.Tensor``."""

        import torch

        return isinstance(value, torch.Tensor)

    def host_copy(self, value: Any) -> Any:
        """Detached CPU NumPy copy; bfloat16 widens to float32 (NumPy has no bfloat16)."""

        import torch

        tensor = value.detach().cpu()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        return tensor.numpy().copy()

    def realize_once(self, values: Sequence[Any]) -> None:
        """Eager framework: no evaluation step."""

        return None


def protocol() -> TorchProtocol:
    """Factory bound to ``BackendSpec.protocol`` for torch."""

    return TorchProtocol()
