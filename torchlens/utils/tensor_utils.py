"""Tensor utilities: NaN-aware comparison, memory calculation, safe_copy, safe device transfer.

Many functions in this module use ``pause_logging()`` to temporarily disable
the torchlens logging toggle before calling tensor methods.  This is
necessary because tensor methods like ``.clone()``, ``.to()``,
``.nelement()``, and ``.element_size()`` are all decorated at import time
(see ``decoration/torch_funcs.py``).  Without pausing, these internal calls
would be logged as user operations, creating spurious entries and, in the
case of ``safe_copy`` called from *inside* the logging pipeline, infinite
recursion.

The ``_clean_*`` function imports (e.g. ``_clean_clone``) MUST be resolved
before decoration runs, since after decoration the module-level names point
to wrapped versions.
"""

import copy
from typing import Any, Optional, cast

import numpy as np
import torch

# Maximum absolute tolerance for floating-point comparison in tensor_nanequal.
# Used by validation replay to allow tiny numerical differences caused by
# non-deterministic GPU reductions or float16 rounding.  Set conservatively
# tight to catch genuine mismatches while tolerating hardware noise.
MAX_FLOATING_POINT_TOLERANCE = 1e-5

# Maximum relative tolerance for floating-point comparison in tensor_nanequal.
# Deep convolution replays can differ by a few ULPs above the absolute floor
# while still matching the saved operation numerically.
REL_FLOATING_POINT_TOLERANCE = 1e-4

_DTYPE_FLOAT_TOLERANCES: dict[torch.dtype, tuple[float, float]] = {
    torch.float16: (1e-3, 1e-3),
    torch.bfloat16: (1e-2, 1e-2),
    torch.float32: (REL_FLOATING_POINT_TOLERANCE, MAX_FLOATING_POINT_TOLERANCE),
    torch.float64: (REL_FLOATING_POINT_TOLERANCE, MAX_FLOATING_POINT_TOLERANCE),
}

# Cached result of torch.cuda.is_available().  Evaluated once per process
# because CUDA availability cannot change at runtime.  Avoids repeated
# calls into the CUDA runtime (which involve driver queries).
_cuda_available: Optional[bool] = None


def _is_cuda_available() -> bool:
    """Return True if CUDA is available (cached after first call).

    The result is cached in a module-level global because CUDA availability
    is fixed for the lifetime of the process, and ``torch.cuda.is_available()``
    involves a non-trivial driver query.
    """
    global _cuda_available
    if _cuda_available is None:
        _cuda_available = torch.cuda.is_available()
    return _cuda_available


def _tolerances_for_dtype(dtype: torch.dtype) -> tuple[float, float]:
    """Return replay comparison tolerances for ``dtype``.

    Parameters
    ----------
    dtype:
        Tensor dtype being compared.

    Returns
    -------
    tuple[float, float]
        ``(rtol, atol)`` pair for ``torch.allclose``.
    """

    return _DTYPE_FLOAT_TOLERANCES.get(
        dtype,
        (REL_FLOATING_POINT_TOLERANCE, MAX_FLOATING_POINT_TOLERANCE),
    )


def tensor_all_nan(tensor: torch.Tensor) -> bool:
    """Return True if every element in the tensor is NaN."""
    if torch.isnan(tensor).int().sum() == tensor.numel():
        return True
    else:
        return False


def _quantized_tensor_equal(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> bool:
    """Return exact equality for quantized tensors without floating ops.

    Parameters
    ----------
    tensor_a:
        First quantized tensor.
    tensor_b:
        Second quantized tensor.

    Returns
    -------
    bool
        True if quantization metadata and integer payloads match.
    """

    if not (tensor_a.is_quantized and tensor_b.is_quantized):
        return False
    if tensor_a.qscheme() != tensor_b.qscheme():
        return False
    if not torch.equal(tensor_a.int_repr(), tensor_b.int_repr()):
        return False
    if tensor_a.qscheme() in (torch.per_tensor_affine, torch.per_tensor_symmetric):
        return tensor_a.q_scale() == tensor_b.q_scale() and (
            tensor_a.q_zero_point() == tensor_b.q_zero_point()
        )
    return (
        tensor_a.q_per_channel_axis() == tensor_b.q_per_channel_axis()
        and torch.equal(tensor_a.q_per_channel_scales(), tensor_b.q_per_channel_scales())
        and torch.equal(
            tensor_a.q_per_channel_zero_points(),
            tensor_b.q_per_channel_zero_points(),
        )
    )


def tensor_nanequal(
    tensor_a: torch.Tensor, tensor_b: torch.Tensor, allow_tolerance: bool = False
) -> bool:
    """NaN-aware tensor equality check, used by validation replay.

    NaN positions are treated as equal (NaN == NaN is True here), which
    differs from IEEE 754 semantics.  This is intentional: validation
    needs to confirm that the replay produced the same NaN pattern, not
    that NaN != NaN.

    ``pause_logging()`` is required because this function is called during
    active logging (from ``_tag_tensor_and_track_variations``) and uses
    decorated tensor methods like ``.resolve_conj()``, ``.isinf()``, etc.
    Without pausing, these calls re-enter the logging pipeline and cause
    infinite recursion.

    Args:
        tensor_a: First tensor.
        tensor_b: Second tensor.
        allow_tolerance: If True, allow element-wise differences up to
            :data:`MAX_FLOATING_POINT_TOLERANCE` (for floating-point
            non-determinism on GPU).

    Returns:
        True if the tensors are considered equal.
    """
    from .._state import pause_logging

    if tensor_a.shape != tensor_b.shape:
        return False

    if tensor_a.dtype != tensor_b.dtype:
        return False

    with pause_logging():
        if tensor_a.is_quantized or tensor_b.is_quantized:
            return _quantized_tensor_equal(tensor_a, tensor_b)

        # Inf positions must match exactly (inf != -inf).
        if not torch.equal(tensor_a.isinf(), tensor_b.isinf()):
            return False

        # Replace NaNs with a sentinel value so torch.equal treats NaN positions
        # as equal.  The sentinel (0.7234691827346) is arbitrary but unlikely to
        # appear in real data.  Complex tensors need view_as_real/view_as_complex
        # because torch.nan_to_num doesn't support complex dtypes directly.
        if tensor_a.is_complex():
            tensor_a_nonan = torch.view_as_complex(
                torch.nan_to_num(torch.view_as_real(tensor_a.resolve_conj()), 0.7234691827346)
            )
            tensor_b_nonan = torch.view_as_complex(
                torch.nan_to_num(torch.view_as_real(tensor_b.resolve_conj()), 0.7234691827346)
            )
        else:
            tensor_a_nonan = torch.nan_to_num(tensor_a, 0.7234691827346)
            tensor_b_nonan = torch.nan_to_num(tensor_b, 0.7234691827346)

        if torch.equal(tensor_a_nonan, tensor_b_nonan):
            return True

        # Tolerance path: allow small floating-point differences (e.g. from
        # convolution replay order, non-deterministic GPU reductions, or
        # mixed-precision rounding).
        if (
            allow_tolerance
            and (tensor_a_nonan.dtype != torch.bool)
            and (tensor_b_nonan.dtype != torch.bool)
        ):
            rtol, atol = _tolerances_for_dtype(tensor_a_nonan.dtype)
            if torch.allclose(tensor_a_nonan, tensor_b_nonan, rtol=rtol, atol=atol):
                return True

    return False


def safe_to(obj: Any, device: str) -> Any:
    """Move a tensor to ``device`` without triggering torchlens logging.

    Non-tensor objects are returned unchanged.  ``pause_logging()`` is
    required because ``.to()`` is a decorated tensor method — calling it
    while logging is active would create a spurious log entry.

    Args:
        obj: A tensor or arbitrary object.
        device: Target device string (e.g. ``"cpu"``, ``"cuda:0"``).

    Returns:
        The tensor on the target device, or the original object if not a tensor.
    """
    from .._state import pause_logging

    if isinstance(obj, torch.Tensor):
        with pause_logging():
            return obj.to(device)
    else:
        return obj


def get_tensor_memory_amount(t: torch.Tensor) -> int:
    """Return the memory footprint of a tensor in bytes.

    ``pause_logging()`` is required because ``.nelement()`` and
    ``.element_size()`` are decorated tensor methods.  Without pausing,
    calling them during active logging would trigger the logging pipeline
    recursively (infinite loop).

    Meta tensors have no storage and return 0.  Sparse tensors report only
    the size of their non-zero values.

    Args:
        t: Tensor to measure.

    Returns:
        Size in bytes, or 0 on failure / meta tensors.
    """
    from .._state import pause_logging

    try:
        with pause_logging():
            if t.device.type == "meta":
                return 0
            if t.is_sparse:
                # Sparse tensors: only the values storage counts.
                return t._values().nelement() * t._values().element_size()
            return t.nelement() * t.element_size()
    except Exception:
        return 0


def concatenate_batch_tensors(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Concatenate two tensors along the leading batch dimension.

    Parameters
    ----------
    left:
        Existing accumulated tensor.
    right:
        New chunk tensor.

    Returns
    -------
    torch.Tensor
        Tensor containing ``left`` followed by ``right`` on dimension 0.
    """

    from .._state import pause_logging

    with pause_logging():
        return torch.cat([left, right], dim=0)


def _safe_get_memory_format(t: torch.Tensor) -> torch.memory_format:
    """Best-effort memory format probe — returns ``preserve_format`` on any error.

    ``is_contiguous(memory_format=...)`` is the recommended query; it is
    undefined for some exotic layouts (sparse, meta), so we wrap in a
    try/except and fall back to ``preserve_format`` (clone's default).
    """
    try:
        if t.is_contiguous(memory_format=torch.channels_last):
            return torch.channels_last
        if t.is_contiguous(memory_format=torch.channels_last_3d):
            return torch.channels_last_3d
    except (RuntimeError, TypeError, AttributeError):
        pass
    return torch.preserve_format


def safe_copy(x: Any, detach_tensor: bool = False) -> Any:
    """Copy a tensor (or parameter) without triggering torchlens logging.

    Uses ``pause_logging()`` so that ``.clone()``, ``.detach()``,
    ``.cpu()`` etc. don't get logged — these are all decorated tensor
    methods, and calling them during active logging would create spurious
    entries or infinite recursion.

    For non-tensor inputs, falls back to ``copy.copy()`` (shallow copy),
    which is safe because non-tensor objects don't have circular-reference
    issues the way tensor wrappers do (see :func:`_safe_copy_arg` for the
    deeper discussion on why ``deepcopy`` is avoided).

    Args:
        x: Input value (tensor, parameter, or arbitrary object).
        detach_tensor: If True, detach the clone from the autograd graph.
            This is used when saving activations to avoid retaining the
            full computational graph in memory.

    Returns:
        A copy with the same values and dtype but independent storage.
    """
    from .._state import pause_logging

    if isinstance(x, (torch.Tensor, torch.nn.Parameter)):
        with pause_logging():
            # Preserve memory_format (channels_last, channels_last_3d, etc.) so
            # downstream layout-sensitive ops see the same layout they would
            # see without TorchLens. Falls back to ``preserve_format`` which is
            # ``clone()``'s default but spelled explicitly for clarity.
            mem_fmt = _safe_get_memory_format(x)
            if not detach_tensor:
                try:
                    return x.clone(memory_format=mem_fmt)
                except (TypeError, RuntimeError):
                    # Some tensor variants (sparse, some subclasses) reject the
                    # memory_format kwarg; fall back to a plain clone.
                    return x.clone()
            # Detach path: use pure-torch ops — no numpy round-trip.
            # This avoids crashes on sparse, quantized, complex32, meta, float8, etc.
            try:
                vals_tensor = x.detach().clone(memory_format=mem_fmt)
            except (TypeError, RuntimeError):
                try:
                    vals_tensor = x.detach().clone()
                except Exception:
                    try:
                        vals_tensor = x.data.cpu().clone()
                    except Exception:
                        # Last resort: return shape-preserving zero tensor
                        vals_tensor = torch.zeros(x.shape, dtype=torch.float32)
            # Preserve the raw label so postprocessing can map this tensor
            # back to its ModelLog entry.
            if hasattr(x, "tl_tensor_label_raw"):
                setattr(vals_tensor, "tl_tensor_label_raw", getattr(x, "tl_tensor_label_raw"))
            if isinstance(x, torch.nn.Parameter):
                return torch.nn.Parameter(vals_tensor)
            return vals_tensor
    else:
        # Non-tensor: shallow copy is sufficient and avoids deepcopy's
        # circular-reference pitfalls.
        return copy.copy(x)


def print_override(t: torch.Tensor, func_name: str) -> str:
    """Safe ``__str__``/``__repr__`` for tensors during active logging.

    The default ``Tensor.__repr__`` calls decorated methods internally,
    which would re-enter the logging pipeline and cause infinite recursion.
    This override pauses logging, converts to a numpy array for formatting,
    and appends autograd metadata (``grad_fn`` / ``requires_grad``) to
    match the standard PyTorch repr style.

    Falls back to a shape/dtype summary for tensors that can't be converted
    to numpy (sparse, quantized, meta, float8, etc.).

    Args:
        t: Tensor to format.
        func_name: Either ``"__str__"`` or ``"__repr__"``.

    Returns:
        Human-readable string representation of the tensor.
    """
    from .._state import pause_logging

    try:
        with pause_logging():
            cpu_data = t.data.cpu()
            # numpy() doesn't support bfloat16 — upcast first.
            if cpu_data.dtype == torch.bfloat16:
                cpu_data = cpu_data.to(torch.float32)
        n = cpu_data.detach().numpy()
        np_str = getattr(n, func_name)()
        # Cosmetic: replace "array" with "tensor" to match PyTorch style.
        np_str = np_str.replace("array", "tensor")
        np_str = np_str.replace("\n", "\n ")
    except Exception:
        # Fallback for sparse, quantized, meta, float8, etc.
        np_str = f"tensor(shape={list(t.shape)}, dtype={t.dtype})"
    # Append autograd info to mimic standard PyTorch repr.
    if t.grad_fn is not None:
        grad_fn_str = f", grad_fn={type(t.grad_fn).__name__})"
        np_str = np_str[0:-1] + grad_fn_str
    elif t.requires_grad:
        np_str = np_str[0:-1] + ", requires_grad=True)"
    return cast(str, np_str)
