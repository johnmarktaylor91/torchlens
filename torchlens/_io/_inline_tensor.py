"""Inline metadata tensors: a bounded, TorchLens-owned pickle reduction.

A plain ``torch.Tensor`` in portable metadata (a value logged during the forward
under ``annotations``, a small ``raw_input`` / ``raw_output``) used to pickle
through torch's default reduction, whose storage leg is
``torch.storage._load_from_bytes``: a nested ``torch.load`` over bytes read from
the file. The restricted loader (``_safe_unpickle``) refuses that leg by name, so
the canonical metadata pickler emits this reduction for plain tensors instead:
the tensor's raw bytes, its dtype name, shape, ``requires_grad`` flag and device
string. Reconstruction allocates exactly the bytes present in the stream through
a buffer-backed, non-resizable storage, so no later opcode can grow it, and it
constructs nothing but a tensor.

Tensor kinds and what happens to each at save time:

* dense strided ``torch.Tensor`` (any dtype, any device, contiguous or not,
  scalar or zero-size, conj / neg bits set): stored as a contiguous copy with the
  shape, dtype, ``requires_grad`` and device kept; values round-trip exactly.
* ``torch.nn.Parameter``: torch's own ``_rebuild_parameter`` reduction wraps the
  plain ``.data`` tensor, which takes this path, so a Parameter loads back as a
  Parameter with its ``requires_grad``.
* sparse, nested and meta tensors: torch's own reductions carry plain dense
  tensors (or no storage at all); those inner tensors take this path.
* quantized tensors and other ``torch.Tensor`` subclasses: outside this path;
  the write-side portability preflight refuses the save typed, naming the key.
"""

from __future__ import annotations

import math
import pickle
import re
import warnings
from functools import cache
from typing import Any

import torch

from .._utils._torch_symbols import torch_attr

__all__ = [
    "INLINE_TENSOR_GLOBAL",
    "rebuild_inline_tensor",
    "reduce_inline_tensor",
]

#: The ``(module, name)`` pair the restricted loader admits for this reduction.
INLINE_TENSOR_GLOBAL: tuple[str, str] = ("torchlens._io._inline_tensor", "rebuild_inline_tensor")


#: What ``torch.empty`` raises for a dtype it cannot allocate plainly: c10 errors
#: surface as ``RuntimeError`` (``NotImplementedError`` is one), argument-parser
#: rejections as ``TypeError`` or ``ValueError``.
_NOT_ALLOCATABLE_ERRORS: tuple[type[Exception], ...] = (RuntimeError, TypeError, ValueError)


@cache
def _dtypes_by_name() -> dict[str, tuple[torch.dtype, int]]:
    """Map canonical dtype names to ``(dtype, itemsize)`` for plain dense dtypes only.

    Built on the first inline-tensor load, not at import, and cached; lookups are
    exact-string on ``str(dtype)``. Quantized dtypes (and any dtype torch cannot
    allocate plainly) are left out, so a stream naming one is refused as an unknown
    dtype before any allocation.
    """

    table: dict[str, tuple[torch.dtype, int]] = {}
    for name in dir(torch):
        value = torch_attr(name)
        if not isinstance(value, torch.dtype):
            continue
        try:
            # torch warns when some dtypes are allocated (complex32, the quantized dtypes);
            # this probe is ours, so its warnings must not surface in the user's session.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                probe = torch.empty((), dtype=value)
        except _NOT_ALLOCATABLE_ERRORS:
            continue
        if probe.is_quantized:
            continue
        table[str(value)] = (value, int(probe.element_size()))
    return table


_DEVICE_PATTERN = re.compile(r"^[a-z_]+(:\d{1,3})?$")
_MAX_DIMS = 64


def reduce_inline_tensor(tensor: torch.Tensor) -> tuple[Any, tuple[Any, ...]] | None:
    """Return the inline reduction for a plain dense tensor, else ``None``.

    Only an exact ``torch.Tensor`` with ``torch.strided`` layout that is neither
    quantized nor nested takes this path; every other kind returns ``None`` so
    the pickler falls through to torch's own reduction (which the loader judges).

    Parameters
    ----------
    tensor:
        Tensor the metadata pickler is about to serialize.

    Returns
    -------
    tuple | None
        ``(rebuild_inline_tensor, (data, dtype_name, shape, requires_grad, device))``
        or ``None`` when the tensor is not a plain dense tensor.
    """

    if type(tensor) is not torch.Tensor:
        return None
    if tensor.layout is not torch.strided or tensor.is_quantized or tensor.is_nested:
        return None
    if tensor.device.type == "meta":
        return None
    with torch.no_grad():
        flat = tensor.detach().cpu().resolve_conj().resolve_neg().contiguous().reshape(-1)
        data = flat.view(torch.uint8).numpy().tobytes() if flat.numel() else b""
    return (
        rebuild_inline_tensor,
        (
            data,
            str(tensor.dtype),
            tuple(tensor.shape),
            bool(tensor.requires_grad),
            str(tensor.device),
        ),
    )


def rebuild_inline_tensor(
    data: Any,
    dtype_name: Any,
    shape: Any,
    requires_grad: Any = False,
    device: Any = "cpu",
) -> torch.Tensor:
    """Rebuild a plain tensor from its inline bytes; refuse anything else typed.

    Every argument is validated by exact type before any allocation, and the byte
    length must equal ``numel * itemsize`` exactly: the result can never be larger
    than the bytes present in the stream. The storage is buffer-backed and not
    resizable, so a following ``_rebuild_tensor_v2`` cannot grow it.

    Parameters
    ----------
    data:
        Raw little-endian element bytes (``bytes``).
    dtype_name:
        Canonical dtype name such as ``"torch.float32"``.
    shape:
        Tuple of non-negative ``int`` dimensions.
    requires_grad:
        Whether the loaded tensor requires grad (``bool``).
    device:
        Device string the tensor lived on when saved.

    Returns
    -------
    torch.Tensor
        The reconstructed tensor.

    Raises
    ------
    pickle.UnpicklingError
        If any argument is malformed, the byte length does not match, or the saved
        device is unavailable on this machine.
    """

    dtype, numel = _checked_layout(data, dtype_name, shape, requires_grad, device)
    if numel == 0:
        tensor = torch.empty(shape, dtype=dtype)
    else:
        # ``bytearray`` copies the stream bytes into a writable buffer; the storage
        # ``torch.frombuffer`` wraps around it is not resizable.
        tensor = torch.frombuffer(bytearray(data), dtype=torch.uint8).view(dtype).reshape(shape)
    return _placed(tensor, dtype_name, requires_grad, device)


def _malformed_inline_tensor(detail: str) -> pickle.UnpicklingError:
    """Return the typed refusal for a malformed inline tensor, naming ``detail``."""

    return pickle.UnpicklingError(
        "Blocked malformed inline tensor during bundle metadata unpickle: " + detail
    )


def _checked_layout(
    data: Any, dtype_name: Any, shape: Any, requires_grad: Any, device: Any
) -> tuple[torch.dtype, int]:
    """Validate every inline-tensor argument by exact type; return ``(dtype, numel)``.

    Raises
    ------
    pickle.UnpicklingError
        If any argument is malformed or the byte length is not ``numel * itemsize``.
    """

    if type(data) is not bytes:
        raise _malformed_inline_tensor(f"payload must be bytes, got {type(data).__name__}.")
    if type(dtype_name) is not str or dtype_name not in _dtypes_by_name():
        raise _malformed_inline_tensor(f"unknown dtype {dtype_name!r}.")
    if (
        type(shape) is not tuple
        or len(shape) > _MAX_DIMS
        or not all(type(dim) is int and dim >= 0 for dim in shape)
    ):
        raise _malformed_inline_tensor("shape must be a tuple of non-negative ints.")
    if type(requires_grad) is not bool:
        raise _malformed_inline_tensor("requires_grad must be a bool.")
    if type(device) is not str or len(device) > 32 or not _DEVICE_PATTERN.match(device):
        raise _malformed_inline_tensor(f"device must be a short device string, got {device!r}.")
    dtype, itemsize = _dtypes_by_name()[dtype_name]
    numel = math.prod(shape)
    if numel * itemsize != len(data):
        raise _malformed_inline_tensor(
            f"byte length {len(data)} does not match shape {shape} of dtype {dtype_name} "
            f"({numel} x {itemsize} bytes)."
        )
    return dtype, numel


def _placed(
    tensor: torch.Tensor, dtype_name: str, requires_grad: bool, device: str
) -> torch.Tensor:
    """Move ``tensor`` to its saved device and restore ``requires_grad``.

    Raises
    ------
    pickle.UnpicklingError
        If the saved device is unavailable here or the dtype cannot require grad.
    """

    if device != "cpu":
        try:
            tensor = tensor.to(torch.device(device))
        except Exception as exc:  # noqa: BLE001 -- unavailable or unknown device
            raise _malformed_inline_tensor(
                f"the tensor was saved on device {device!r}, which is unavailable here; "
                "re-save the trace with the tensor moved to the CPU."
            ) from exc
    if requires_grad:
        try:
            tensor.requires_grad_(True)
        except RuntimeError as exc:
            raise _malformed_inline_tensor(f"dtype {dtype_name} cannot require grad.") from exc
    return tensor
