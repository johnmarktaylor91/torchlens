"""Monotonic torch callable moves for sparse runnable reattachment (a data-only leaf).

The resolver in ``torchlens._utils._torch_compat`` (``resolve_runnable_torch_alias``)
reads this table; it lives apart so the compat chokepoint stays within its size
row. This module imports nothing from torch or TorchLens.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["RunnableTorchAlias", "_RUNNABLE_TORCH_ALIASES"]


@dataclass(frozen=True, slots=True)
class RunnableTorchAlias:
    """One monotonic callable move used by sparse runnable reattachment.

    The table is capability-bounded rather than selected by parsing
    ``torch.__version__``. Source absence is established by the caller's
    allowlisted exact-resolution pass, which runs first and only reaches the
    alias stage when the recorded path does not resolve through a safe public
    namespace; the private/internal recorded paths in this table are therefore
    redirected to their public equivalents (a present internal binding such as
    ``torch._C._nn.linear`` is intentionally routed to its public wrapper for
    replay safety). This resolver additionally probes that an entry's *target*
    exists in the current runtime (:func:`_runtime_alias_target_exists`) and
    never hands back an alias whose target is absent -- so an unbounded/legacy
    version key can never manufacture a dead alias. A ``None`` maximum leaves
    a monotonic alias open above its minimum; target existence remains the
    capability gate in that case.
    """

    source: str
    target_namespace: str
    target_qualname: str | None
    provenance: str
    recorded_min_version: tuple[int, int]
    recorded_max_version: tuple[int, int] | None
    strip_target_prefix: str = ""


_RUNNABLE_TORCH_ALIASES: tuple[RunnableTorchAlias, ...] = (
    RunnableTorchAlias(
        "torch._C._nn.linear",
        "torch.nn.functional",
        "linear",
        "private_to_public:_C._nn.linear->torch.nn.functional.linear",
        (2, 1),
        (2, 14),
    ),
    RunnableTorchAlias(
        "_C._nn.linear",
        "torch.nn.functional",
        "linear",
        "private_to_public:_C._nn.linear->torch.nn.functional.linear",
        (2, 1),
        (2, 14),
    ),
    RunnableTorchAlias(
        "torch._VF.linear",
        "torch.nn.functional",
        "linear",
        "private_to_public:_VF.linear->torch.nn.functional.linear",
        (2, 1),
        (2, 12),
    ),
    RunnableTorchAlias(
        "_VF.linear",
        "torch.nn.functional",
        "linear",
        "private_to_public:_VF.linear->torch.nn.functional.linear",
        (2, 1),
        (2, 12),
    ),
    RunnableTorchAlias(
        "torch._C._nn.gelu",
        "torch.nn.functional",
        "gelu",
        "private_to_public:_C._nn.gelu->torch.nn.functional.gelu",
        (2, 1),
        (2, 12),
    ),
    RunnableTorchAlias(
        "torch._C._nn.softplus",
        "torch.nn.functional",
        "softplus",
        "private_to_public:_C._nn.softplus->torch.nn.functional.softplus",
        (2, 1),
        (2, 12),
    ),
    RunnableTorchAlias(
        "torch._C._nn.cross_entropy_loss",
        "torch.nn.functional",
        "cross_entropy",
        "private_to_public:_C._nn.cross_entropy_loss->torch.nn.functional.cross_entropy",
        (2, 1),
        (2, 12),
    ),
    RunnableTorchAlias(
        "torch._C._special.*",
        "torch.special",
        None,
        "private_to_public:_C._special.special_*->torch.special.*",
        (2, 1),
        (2, 12),
        "special_",
    ),
    RunnableTorchAlias(
        "torch._C._fft.*",
        "torch.fft",
        None,
        "private_to_public:_C._fft.fft_*->torch.fft.*",
        (2, 1),
        (2, 12),
        "fft_",
    ),
    RunnableTorchAlias(
        "torch._C._linalg.*",
        "torch.linalg",
        None,
        "private_to_public:_C._linalg.linalg_*->torch.linalg.*",
        (2, 1),
        (2, 14),
        "linalg_",
    ),
    RunnableTorchAlias(
        "torch._C._TensorBase.*",
        "torch.Tensor",
        None,
        "tensor_base_drift:torch._C._TensorBase->torch.Tensor",
        (2, 1),
        (2, 12),
    ),
    RunnableTorchAlias(
        "torch._C.TensorBase.*",
        "torch.Tensor",
        None,
        "tensor_base_drift:torch._C.TensorBase->torch.Tensor",
        (2, 1),
        (2, 12),
    ),
    RunnableTorchAlias(
        "torch.TensorBase.*",
        "torch.Tensor",
        None,
        "tensor_base_drift:torch.TensorBase->torch.Tensor",
        (2, 1),
        (2, 12),
    ),
    RunnableTorchAlias(
        "torch.Tensor.*",
        "torch.Tensor",
        None,
        "tensor_prefix_drift:torch.Tensor->torch.Tensor",
        (2, 1),
        (2, 12),
    ),
    RunnableTorchAlias(
        "_TensorBase.*",
        "torch.Tensor",
        None,
        "tensor_base_drift:_TensorBase->torch.Tensor",
        (2, 1),
        (2, 12),
    ),
    RunnableTorchAlias(
        "TensorBase.*",
        "torch.Tensor",
        None,
        "tensor_base_drift:TensorBase->torch.Tensor",
        (2, 1),
        (2, 12),
    ),
    RunnableTorchAlias(
        "Tensor.*",
        "torch.Tensor",
        None,
        "tensor_prefix_drift:Tensor->torch.Tensor",
        (2, 1),
        (2, 12),
    ),
    RunnableTorchAlias(
        "torch._C._VariableFunctions.*",
        "torch",
        None,
        "private_to_public:torch._C._VariableFunctions->torch",
        (2, 1),
        (2, 12),
    ),
    RunnableTorchAlias(
        "torch._C._VariableFunctionsClass.*",
        "torch",
        None,
        "private_to_public:torch._C._VariableFunctionsClass->torch",
        (2, 1),
        None,
    ),
    RunnableTorchAlias(
        "torch._VariableFunctionsClass.*",
        "torch",
        None,
        "private_to_public:torch._VariableFunctionsClass->torch",
        (2, 1),
        None,
    ),
    RunnableTorchAlias(
        "_VariableFunctionsClass.*",
        "torch",
        None,
        "private_to_public:_VariableFunctionsClass->torch",
        (2, 1),
        None,
    ),
)
