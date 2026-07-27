"""Tensor utilities: NaN-aware comparison, memory calculation, safe_copy, safe device transfer.

Many functions in this module use ``pause_logging()`` to temporarily disable
the torchlens logging toggle before calling tensor custom_methods.  This is
necessary because tensor custom_methods like ``.clone()``, ``.to()``,
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
from math import prod
from typing import Any, Callable, Literal, Optional, cast

import torch

from ._torch_compat import get_functorch_wrapped_tensor_checker

from ..backends.torch._tl import get_tensor_label, set_tensor_label

SaveMode = Literal["copy", "reference", "view", "cpu_async"]

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

_TensorSizeMethod = Callable[[torch.Tensor], int]


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


def is_functorch_wrapped_tensor(value: Any) -> bool:
    """Return whether ``value`` is a functorch wrapper tensor.

    Parameters
    ----------
    value:
        Object to inspect.

    Returns
    -------
    bool
        True when PyTorch reports a functorch wrapper tensor.
    """

    if not isinstance(value, torch.Tensor):
        return False
    checker = get_functorch_wrapped_tensor_checker()
    if checker is None:
        return False
    try:
        return bool(checker(value))
    except RuntimeError:
        return False


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
    decorated tensor custom_methods like ``.resolve_conj()``, ``.isinf()``, etc.
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

    if is_functorch_wrapped_tensor(tensor_a) or is_functorch_wrapped_tensor(tensor_b):
        return False

    if tensor_a.shape != tensor_b.shape:
        return False

    if tensor_a.dtype != tensor_b.dtype:
        return False

    # Meta tensors carry no data: with shape and dtype already matched there
    # is nothing left to compare, and any content op (torch.equal, .isinf())
    # raises "Cannot copy out of meta tensor" on them.
    if tensor_a.is_meta or tensor_b.is_meta:
        return tensor_a.is_meta and tensor_b.is_meta

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


def _unwrapped_tensor_size_method(method: _TensorSizeMethod) -> _TensorSizeMethod:
    """Return the undecorated implementation for a Tensor size method.

    Parameters
    ----------
    method:
        Tensor method descriptor or TorchLens wrapper to resolve.

    Returns
    -------
    _TensorSizeMethod
        Original method when TorchLens has decorated it, otherwise ``method``.
    """

    from .. import _state

    return cast(_TensorSizeMethod, _state._decorated_to_orig.get(id(method), method))


def _dense_tensor_memory_amount(t: torch.Tensor) -> int:
    """Return dense tensor bytes without entering the logging toggle machinery.

    Parameters
    ----------
    t:
        Dense tensor to measure.

    Returns
    -------
    int
        Number of bytes represented by ``t.nelement() * t.element_size()``.
    """

    nelement = _unwrapped_tensor_size_method(torch.Tensor.nelement)
    element_size = _unwrapped_tensor_size_method(torch.Tensor.element_size)
    return int(nelement(t) * element_size(t))


def get_memory_amount(t: torch.Tensor) -> int:
    """Return the memory footprint of a tensor in bytes.

    Tensor size methods are called through their unwrapped implementations when
    TorchLens has decorated them, avoiding logging recursion without toggling
    global logging state for each tensor.

    Meta tensors have no storage and return 0.  Sparse tensors report only
    the size of their non-zero values.

    Args:
        t: Tensor to measure.

    Returns:
        Size in bytes, or 0 on failure / meta tensors.
    """

    try:
        if t.device.type == "meta":
            return 0
        if t.is_sparse:
            # Sparse tensors: only the values storage counts.
            return _dense_tensor_memory_amount(t._values())
        return _dense_tensor_memory_amount(t)
    except Exception:
        return 0


def get_memory_amount_from_metadata(
    t: torch.Tensor,
    shape: tuple[int, ...] | torch.Size,
    dtype: torch.dtype,
) -> int:
    """Return tensor memory bytes using already-captured dense metadata.

    Parameters
    ----------
    t:
        Tensor being measured.
    shape:
        Already-captured tensor shape.
    dtype:
        Already-captured tensor dtype.

    Returns
    -------
    int
        Size in bytes, or the guarded tensor-method fallback for layouts whose
        storage size is not represented by ``shape * dtype.itemsize``.
    """

    try:
        if t.device.type == "meta":
            return 0
        if t.is_sparse:
            return get_memory_amount(t)
        return int(prod(shape) * dtype.itemsize)
    except Exception:
        return get_memory_amount(t)


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

    Standard (torch.contiguous_format) is checked FIRST and wins on ties.
    For tensors with a size-1 dimension (most commonly ``C=1``, e.g. a mono
    spectrogram or single-channel image), ``is_contiguous(memory_format=
    torch.channels_last)`` is degenerately also ``True`` even though the
    tensor is genuinely NCHW-contiguous — the collapsed size-1 axis makes
    both stride orderings equally valid descriptions of the same bytes.
    Checking ``channels_last`` first (the prior behavior) would then force
    ``.clone(memory_format=torch.channels_last)`` on an already-standard
    tensor, physically rewriting its strides to the channels-last layout.
    That silently corrupts any downstream ``.view()`` call the traced model
    makes under the (correct, for its real input) assumption of standard
    contiguity — a real capture bug, not a model bug. See
    ``torchlens/utils/tensor_utils.py`` history / BC-ResNet capture repro.
    """
    try:
        if t.is_contiguous(memory_format=torch.contiguous_format):
            return torch.contiguous_format
        if t.is_contiguous(memory_format=torch.channels_last):
            return torch.channels_last
        if t.is_contiguous(memory_format=torch.channels_last_3d):
            return torch.channels_last_3d
    except (RuntimeError, TypeError, AttributeError):
        pass
    return torch.preserve_format


def _copy_tensor_payload(
    x: torch.Tensor | torch.nn.Parameter,
    *,
    detach_tensor: bool,
    save_mode: SaveMode,
) -> torch.Tensor:
    """Return a tensor payload according to the requested save mode.

    Parameters
    ----------
    x:
        Tensor or parameter to materialize.
    detach_tensor:
        Whether the saved payload should be detached from autograd.
    save_mode:
        Payload retention mode. ``"copy"`` safely clones; ``"reference"`` safely
        preserves the original value by relying on capture-time in-place handling;
        ``"view"`` stores a live alias that downstream in-place operations can mutate;
        and ``"cpu_async"`` clones to CPU with ``non_blocking=True``.

    Returns
    -------
    torch.Tensor
        Tensor payload for storage.
    """

    if save_mode == "reference":
        return x.detach() if detach_tensor else x
    if save_mode == "view":
        return x
    if save_mode == "cpu_async":
        payload = x.detach() if detach_tensor else x
        try:
            if payload.device.type != "cpu":
                cpu_payload = torch.empty_like(
                    payload,
                    device="cpu",
                    memory_format=_safe_get_memory_format(payload),
                    pin_memory=True,
                )
                return cpu_payload.copy_(payload, non_blocking=True)
        except (TypeError, RuntimeError):
            pass
        return payload.to(device="cpu", non_blocking=True, copy=True)

    mem_fmt = _safe_get_memory_format(x)
    if not detach_tensor:
        try:
            return x.clone(memory_format=mem_fmt)
        except (TypeError, RuntimeError):
            return x.clone()
    try:
        return x.detach().clone(memory_format=mem_fmt)
    except (TypeError, RuntimeError):
        try:
            return x.detach().clone()
        except Exception:
            try:
                return x.data.cpu().clone()
            except Exception:
                return torch.zeros(x.shape, dtype=torch.float32)


def _clone_tensor_payload(
    x: torch.Tensor | torch.nn.Parameter,
    *,
    detach_tensor: bool,
    save_mode: SaveMode,
) -> torch.Tensor | torch.nn.Parameter:
    """Clone or retain one tensor payload without triggering TorchLens logging.

    Parameters
    ----------
    x
        Tensor or parameter to clone or retain.
    detach_tensor
        Whether to detach copied payloads from the autograd graph.
    save_mode
        Tensor retention mode. ``"copy"`` preserves historical clone behavior,
        ``"reference"`` stores the source tensor, ``"view"`` stores the
        graph-connected source tensor, and ``"cpu_async"`` copies to CPU.

    Returns
    -------
    torch.Tensor | torch.nn.Parameter
        Tensor payload with TorchLens raw label preserved, or a rewrapped
        parameter payload for parameter inputs.
    """
    from .._state import pause_logging

    with pause_logging():
        if save_mode not in {"copy", "reference", "view", "cpu_async"}:
            raise ValueError("save_mode must be one of 'copy', 'reference', 'view', or 'cpu_async'")
        vals_tensor = _copy_tensor_payload(
            x,
            detach_tensor=detach_tensor,
            save_mode=save_mode,
        )
        label = None if isinstance(x, torch.nn.Parameter) else get_tensor_label(x)
        if label is not None:
            set_tensor_label(vals_tensor, label)
        if isinstance(x, torch.nn.Parameter):
            return torch.nn.Parameter(vals_tensor)
        return vals_tensor


def copy_tensor_payload(
    x: Any,
    *,
    save_mode: SaveMode = "copy",
    detach_tensor: bool = False,
) -> Any:
    """Copy an output payload with tensor-clone and shallow non-tensor semantics.

    Uses ``pause_logging()`` so that ``.clone()``, ``.detach()``,
    ``.cpu()`` etc. don't get logged — these are all decorated tensor
    custom_methods, and calling them during active logging would create spurious
    entries or infinite recursion.

    For non-tensor inputs, falls back to ``copy.copy()`` (shallow copy),
    which is safe because non-tensor objects don't have circular-reference
    issues the way tensor wrappers do (see :func:`_safe_copy_arg` for the
    deeper discussion on why ``deepcopy`` is avoided).

    Parameters
    ----------
    x
        Input value, tensor, parameter, or arbitrary object.
    save_mode
        Tensor retention mode. ``"copy"`` preserves historical clone behavior.
        ``"reference"`` stores the detached source tensor. ``"view"`` stores
        the graph-connected source tensor. ``"cpu_async"`` copies to CPU using
        ``non_blocking=True``.
    detach_tensor
        If True, detach the saved payload from the autograd graph. This is used
        when saving outs to avoid retaining the full computational graph in
        memory.

    Returns
    -------
    Any
        Tensor payload copy/retention result, or a shallow copy for non-tensors.
    """

    if isinstance(x, (torch.Tensor, torch.nn.Parameter)):
        return _clone_tensor_payload(x, detach_tensor=detach_tensor, save_mode=save_mode)
    else:
        # Non-tensor: shallow copy is sufficient and avoids deepcopy's
        # circular-reference pitfalls.
        return copy.copy(x)


def safe_copy(x: Any, detach_tensor: bool = False, save_mode: SaveMode = "copy") -> Any:
    """Compatibility alias for :func:`copy_tensor_payload`.

    Parameters
    ----------
    x
        Input value, tensor, parameter, or arbitrary object.
    detach_tensor
        Whether tensor payloads should detach from autograd.
    save_mode
        Tensor retention mode.

    Returns
    -------
    Any
        Output-payload copy result.
    """

    return copy_tensor_payload(x, save_mode=save_mode, detach_tensor=detach_tensor)


def print_override(t: torch.Tensor, func_name: str) -> str:
    """Safe ``__str__``/``__repr__`` for tensors during active logging.

    The default ``Tensor.__repr__`` calls decorated custom_methods internally,
    which would re-enter the logging pipeline and cause infinite recursion.
    This override pauses logging, converts to a numpy array for formatting,
    and appends autograd metadata (``grad_fn_handle`` / ``requires_grad``) to
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
            # ``.detach()`` is a decorated torch method like any other; calling
            # it outside pause_logging (while a trace is actively logging)
            # would log a real "detach" op and consume a raw-op-counter slot,
            # leaving a graph orphan and staling any raw labels recorded just
            # before this repr fired. Keep it inside the paused block.
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
        grad_fn_str = f", grad_fn_handle={type(t.grad_fn).__name__})"
        np_str = np_str[0:-1] + grad_fn_str
    elif t.requires_grad:
        np_str = np_str[0:-1] + ", requires_grad=True)"
    return cast(str, np_str)


# ======================================================================================
# r37 INV-2 -- THE one absolute-byte three-valued alias/overlap engine.
#
# Every disjointness / overlap / identity / containment proof over tensor memory in the
# runnable witness/execution surface routes through these helpers. Local pointer-equality
# shortcuts are FORBIDDEN (hon1_1: ``torch.from_numpy(arr[:6])`` vs ``arr[2:8]`` own
# DISTINCT torch storages with distinct base pointers over genuinely overlapping host
# memory, so ``data_ptr() != data_ptr()`` is never a disjointness proof). All coordinates
# are ABSOLUTE, device-scoped byte addresses; the relation vocabulary is exactly
# ``overlap | disjoint | unknown`` and anything unproven is ``unknown`` (fail closed).
# ======================================================================================

AliasRelation = Literal["overlap", "disjoint", "unknown"]
"""Three-valued alias-proof vocabulary (INV-2). ``unknown`` is a first-class verdict."""

ALIAS_ENUMERATION_ELEMENT_CAP = 65536
"""Exact-enumeration bound (inclusive, per view) for the alias proof engine."""


class TensorByteFootprint:
    """Absolute, device-scoped byte footprint of one strided tensor view.

    ``start_byte``/``end_byte`` bound the touched span on ABSOLUTE addresses
    (``storage.data_ptr()`` + offset + min/max stride contributions; negative and
    zero strides sound). ``origin_byte`` is the absolute address of the
    ``storage_offset`` element (the grid origin for residue/enumeration proofs).
    """

    __slots__ = (
        "device_key",
        "start_byte",
        "end_byte",
        "origin_byte",
        "element_size",
        "shape",
        "strides",
        "numel",
    )

    def __init__(
        self,
        device_key: tuple[str, Optional[int]],
        start_byte: int,
        end_byte: int,
        origin_byte: int,
        element_size: int,
        shape: tuple[int, ...],
        strides: tuple[int, ...],
        numel: int,
    ) -> None:
        self.device_key = device_key
        self.start_byte = start_byte
        self.end_byte = end_byte
        self.origin_byte = origin_byte
        self.element_size = element_size
        self.shape = shape
        self.strides = strides
        self.numel = numel


def tensor_byte_footprint(value: torch.Tensor) -> Optional[TensorByteFootprint]:
    """Compute a tensor's absolute byte footprint, or ``None`` when unprovable.

    ``None`` (the caller must treat the relation as ``unknown``) covers exotic
    layouts that refuse geometry reads AND any tensor whose storage base pointer is
    ``0`` with nonzero elements -- every meta tensor reports ``data_ptr() == 0``, so
    absolute-address math on it would collide unrelated tensors (pre-closed r38
    adjacent: meta ``data_ptr==0``).
    """

    try:
        storage_ptr = int(value.untyped_storage().data_ptr())
        element_size = int(value.element_size())
        numel = int(value.numel())
        if storage_ptr == 0 and numel > 0:
            return None
        device = value.device
        device_key = (str(device.type), device.index)
        origin = storage_ptr + int(value.storage_offset()) * element_size
        shape = tuple(int(dim) for dim in value.shape)
        strides = tuple(int(stride) for stride in value.stride())
        if numel == 0:
            return TensorByteFootprint(
                device_key, origin, origin, origin, element_size, shape, strides, 0
            )
        low = 0
        high = 0
        for size, stride in zip(shape, strides):
            contribution = (size - 1) * stride
            if contribution < 0:
                low += contribution
            else:
                high += contribution
        return TensorByteFootprint(
            device_key,
            origin + low * element_size,
            origin + high * element_size + element_size,
            origin,
            element_size,
            shape,
            strides,
            numel,
        )
    except (RuntimeError, AttributeError, TypeError, ValueError, NotImplementedError):
        return None


def _footprint_is_dense_interval(footprint: TensorByteFootprint) -> bool:
    """Return whether a footprint's element starts cover ONE canonical dense byte interval (r39).

    Pure-integer proof (corr2_6): the touched element addresses form a contiguous no-hole,
    no-overlap grid -- so the WHOLE ``[start_byte, end_byte)`` byte span is fully covered -- iff,
    after dropping singleton dims and sorting the rest by absolute element stride, the smallest
    absolute stride is ``1`` and each next equals the running product of the preceding dimension
    sizes (then multiply by that size). This is the canonical row-major recurrence up to a
    dimension permutation and independent per-dim sign, so it proves contiguous, transposed/
    permuted-dense, and mathematically-valid negative-stride layouts alike -- and NOTHING else.

    It is deliberately NOT ``numel * element_size == end_byte - start_byte``: duplicate element
    addresses plus holes can satisfy that count/span equality without dense coverage. A zero
    stride on any non-singleton dim (an expanded view) repeats addresses -> not dense -> ``False``.
    Numel-independent: no enumeration, sound above the enumeration cap.
    """

    if footprint.numel == 0:
        return False
    dims = [
        (abs(stride), size) for size, stride in zip(footprint.shape, footprint.strides) if size > 1
    ]
    if not dims:
        # All dims singleton: the footprint touches exactly one element -> a trivially dense
        # (single-element) interval of ``element_size`` bytes.
        return True
    if any(abs_stride == 0 for abs_stride, _size in dims):
        # An expanded (zero-stride) non-singleton dim repeats addresses -> not dense.
        return False
    dims.sort(key=lambda item: item[0])
    expected = 1
    for abs_stride, size in dims:
        if abs_stride != expected:
            return False
        expected *= size
    return True


def _footprint_stride_gcd(footprint: TensorByteFootprint) -> int:
    """gcd of nonzero element strides over nonsingleton dims (``0`` == one element)."""

    from math import gcd

    result = 0
    for size, stride in zip(footprint.shape, footprint.strides):
        if size > 1 and stride != 0:
            result = gcd(result, abs(stride))
    return result


def footprint_touched_element_addresses(footprint: TensorByteFootprint) -> set[int]:
    """Enumerate the ABSOLUTE byte address of every element start a view touches.

    Pure Python integer arithmetic ONLY (r37 corr2-2): no torch factory may appear
    here, so the proof is identical under an implicit CPU default, a process-global
    meta default device, and nested ``torch.device(...)`` modes. Bounded by
    :data:`ALIAS_ENUMERATION_ELEMENT_CAP` at the call site.
    """

    if footprint.numel == 0:
        return set()
    esize = footprint.element_size
    addresses = {footprint.origin_byte}
    for size, stride in zip(footprint.shape, footprint.strides):
        if size <= 1:
            continue
        step = stride * esize
        addresses = {address + index * step for address in addresses for index in range(size)}
    return addresses


def touched_bytes_relation(left: torch.Tensor, right: torch.Tensor) -> AliasRelation:
    """Three-valued exact touched-byte relation on absolute, device-scoped addresses.

    Proof layers, in order (INV-2): repeated object identity proves ``overlap``;
    unprovable footprints are ``unknown``; empty views, distinct device address
    spaces, and disjoint absolute byte intervals prove ``disjoint``; identical
    absolute geometry proves ``overlap``; an element-grid residue/GCD argument on
    absolute coordinates (equal element sizes, byte starts congruent on the shared
    element grid) proves ONLY disjointness; bounded pure-integer enumeration of
    absolute touched addresses proves either; everything else is ``unknown``. No
    bounding-interval overlap alone is an overlap proof, no complexity cap is a
    disjointness proof, and storage-pointer (in)equality NEVER decides anything --
    distinct storage objects can overlay one host allocation (hon1_1).
    """

    left_footprint = tensor_byte_footprint(left)
    right_footprint = tensor_byte_footprint(right)
    if left_footprint is None or right_footprint is None:
        return "unknown"
    if left_footprint.numel == 0 or right_footprint.numel == 0:
        return "disjoint"
    if left is right:
        return "overlap"
    if left_footprint.device_key != right_footprint.device_key:
        # Distinct device address spaces cannot share bytes. Same device TYPE with
        # one concrete and one None index is conservatively comparable only when
        # equal; treat a None-vs-concrete mismatch as unknown (unprovable).
        left_type, left_index = left_footprint.device_key
        right_type, right_index = right_footprint.device_key
        if left_type != right_type:
            return "disjoint"
        if left_index is None or right_index is None:
            return "unknown"
        return "disjoint"
    if (
        left_footprint.end_byte <= right_footprint.start_byte
        or right_footprint.end_byte <= left_footprint.start_byte
    ):
        return "disjoint"
    if (
        left_footprint.element_size == right_footprint.element_size
        and left_footprint.origin_byte == right_footprint.origin_byte
        and left_footprint.shape == right_footprint.shape
        and left_footprint.strides == right_footprint.strides
    ):
        return "overlap"
    if left_footprint.element_size == right_footprint.element_size:
        esize = left_footprint.element_size
        delta_bytes = left_footprint.origin_byte - right_footprint.origin_byte
        if delta_bytes % esize == 0:
            # Shared element grid: every touched element address of a view is
            # congruent to its origin modulo gcd(strides)*esize, so an origin-residue
            # disagreement modulo the combined gcd proves disjointness. A congruence
            # NEVER proves overlap.
            from math import gcd

            combined = gcd(
                _footprint_stride_gcd(left_footprint), _footprint_stride_gcd(right_footprint)
            )
            if combined == 0:
                # Both views touch exactly one element inside overlapping bounds.
                return (
                    "overlap"
                    if left_footprint.origin_byte == right_footprint.origin_byte
                    else "disjoint"
                )
            if (delta_bytes // esize) % combined != 0:
                return "disjoint"
    # r39 corr2_6 (the sole relaxation, sequenced after all fail-closed work): when BOTH
    # footprints are proven canonical dense byte intervals, each fully covers its own
    # ``[start_byte, end_byte)`` span, so their device-scoped byte intervals already passed the
    # disjointness check above => the overlapping region is touched by both => ``overlap``,
    # exactly and numel-independently (no enumeration, sound above the cap). This never converts
    # an ``unknown`` into a false ``overlap``: it fires ONLY on the provable dense geometry
    # (contiguous, permuted/transposed, signed-stride), keeping genuinely sparse/expanded
    # over-cap layouts ``unknown``. Element sizes need not match -- both byte intervals are
    # individually proved full.
    if _footprint_is_dense_interval(left_footprint) and _footprint_is_dense_interval(
        right_footprint
    ):
        return "overlap"
    if (
        left_footprint.numel <= ALIAS_ENUMERATION_ELEMENT_CAP
        and right_footprint.numel <= ALIAS_ENUMERATION_ELEMENT_CAP
    ):
        left_addresses = footprint_touched_element_addresses(left_footprint)
        right_addresses = footprint_touched_element_addresses(right_footprint)
        if (
            left_footprint.element_size == right_footprint.element_size
            and (left_footprint.origin_byte - right_footprint.origin_byte)
            % left_footprint.element_size
            == 0
        ):
            return "overlap" if left_addresses & right_addresses else "disjoint"
        left_bytes = {
            address + byte
            for address in left_addresses
            for byte in range(left_footprint.element_size)
        }
        right_bytes = {
            address + byte
            for address in right_addresses
            for byte in range(right_footprint.element_size)
        }
        return "overlap" if left_bytes & right_bytes else "disjoint"
    return "unknown"


def footprints_overlap_possible(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Conservative Boolean adapter: ``True`` unless PROVEN disjoint.

    For callers that need a can-touch pre-check (write-back sampling, TOCTOU
    machinery): ``overlap`` and ``unknown`` both return ``True`` -- an unproven
    relation must never be treated as disjoint (INV-2).
    """

    return touched_bytes_relation(left, right) != "disjoint"
