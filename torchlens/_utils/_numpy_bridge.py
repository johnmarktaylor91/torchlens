"""Bridge-aware tensor/NumPy conversion (the one authority).

Product code turns a tensor into a NumPy array, or an array into a tensor,
through :func:`tensor_to_numpy`, :func:`tensor_from_numpy` and :func:`host_array`;
byte copies go through ``tensor_leaves.copy_host_bytes``. On a working
torch/NumPy bridge each is exactly the historical call. When
``HAS_NUMPY_BRIDGE`` is False (a torch build compiled against NumPy 1.x, torch
2.1 and 2.2, running under NumPy 2, where every bridge call raises "Numpy is
not available") each falls back to a NumPy-free byte copy, so digests,
exports and plots keep working instead of failing deep inside a call.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .. import _state
from . import _torch_compat
from ._torch_compat import get_numpy_bridge_support
from ._torch_symbols import torch_dtype_named
from .tensor_leaves import copy_host_bytes, flat_byte_view, tensor_payload_bytes

# Dtype names torch's own bridge converts (``Tensor.numpy`` and ``torch.from_numpy``).
# The fallbacks check this table rather than asking NumPy whether it knows the name:
# importing ``ml_dtypes`` (onnx, jax and tensorflow do) registers ``bfloat16`` and the
# ``float8_*`` names with NumPy, and the fallback must still refuse what torch refuses.
_BRIDGE_DTYPE_NAMES = frozenset(
    {
        "bool",
        "uint8",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
        "complex64",
        "complex128",
    }
)


def _bridge_works() -> bool:
    """Return :func:`get_numpy_bridge_support`, probing with capture logging paused.

    The first call runs a real ``torch.from_numpy`` / ``Tensor.numpy`` probe; a
    conversion inside an active capture (the RNG digest) would otherwise record
    those probe ops into the user's trace. Once probed, the cached flag is read
    directly so a conversion costs no extra Python calls.
    """

    if _torch_compat._NUMPY_BRIDGE_PROBED:
        return _torch_compat.HAS_NUMPY_BRIDGE

    with _state.pause_logging():
        return get_numpy_bridge_support()


def tensor_to_numpy(tensor: torch.Tensor, *, force: bool = False) -> Any:
    """Return ``tensor.cpu().numpy()``, also when torch's NumPy bridge is broken.

    With a working bridge this is exactly ``tensor.cpu().numpy()`` (same
    strides, same refusals). When ``HAS_NUMPY_BRIDGE`` is False the array is
    rebuilt by NumPy itself from a byte copy of the CPU-contiguous logical
    payload, so export, plot and analysis paths keep working instead of
    raising "Numpy is not available".

    Parameters
    ----------
    tensor:
        Dense strided tensor whose dtype NumPy can represent.
    force:
        On a working bridge, call ``tensor.numpy(force=True)`` (detach, move to
        CPU, resolve conj/neg bits) instead of ``tensor.cpu().numpy()``. The
        fallback always does all of that.

    Returns
    -------
    numpy.ndarray
        Host array with the tensor's shape, dtype and values.

    Raises
    ------
    TypeError
        When NumPy has no matching dtype (bf16, fp8), as ``Tensor.numpy`` does.
    """

    if _bridge_works():
        return tensor.numpy(force=True) if force else tensor.cpu().numpy()
    dtype_name = str(tensor.dtype).removeprefix("torch.")
    if dtype_name not in _BRIDGE_DTYPE_NAMES:
        raise TypeError(f"Got unsupported ScalarType {dtype_name}")
    numpy_dtype = np.dtype(dtype_name)
    host = tensor.detach().resolve_conj().resolve_neg().cpu().contiguous()
    payload = bytearray(copy_host_bytes(flat_byte_view(host)))
    return np.frombuffer(payload, dtype=numpy_dtype).reshape(tuple(tensor.shape))


def tensor_numpy_bytes(tensor: torch.Tensor) -> bytes:
    """Return ``tensor.cpu().numpy().tobytes()``, also when torch's NumPy bridge is broken.

    Tries the NumPy read first and probes the bridge only after it raises, so a
    working bridge makes exactly the historical calls and never runs the probe
    (the RNG digest reads a generator's state this way inside a capture).

    Parameters
    ----------
    tensor:
        Dense strided tensor whose dtype NumPy can represent.

    Returns
    -------
    bytes
        The tensor's contiguous payload bytes.
    """

    host = tensor.cpu()
    try:
        return host.numpy().tobytes()
    except RuntimeError:
        if _bridge_works():
            raise
        return tensor_payload_bytes(host)


def tensor_from_numpy(array: Any) -> torch.Tensor:
    """Return ``torch.from_numpy(array)``, also when torch's NumPy bridge is broken.

    With a working bridge this is exactly ``torch.from_numpy`` (zero-copy, the
    tensor shares the array's memory). When ``HAS_NUMPY_BRIDGE`` is False the
    tensor is built from a byte copy through the buffer protocol
    (``torch.frombuffer``), so it does NOT share memory with ``array``.

    Parameters
    ----------
    array:
        A ``numpy.ndarray`` whose dtype torch can represent.

    Returns
    -------
    torch.Tensor
        CPU tensor with the array's shape, dtype and values.

    Raises
    ------
    TypeError
        When torch has no matching dtype, as ``torch.from_numpy`` does.
    """

    if _bridge_works():
        return torch.from_numpy(array)
    # ``np.ascontiguousarray`` promotes 0-d input to shape (1,): keep the
    # caller's shape explicitly.
    shape = tuple(np.shape(array))
    contiguous = np.ascontiguousarray(array)
    torch_dtype = (
        torch_dtype_named(contiguous.dtype.name)
        if contiguous.dtype.name in _BRIDGE_DTYPE_NAMES
        else None
    )
    if torch_dtype is None:
        raise TypeError(
            f"can't convert np.ndarray of type numpy.{contiguous.dtype.name}; torch has "
            "no matching dtype"
        )
    if contiguous.size == 0:
        return torch.empty(shape, dtype=torch_dtype)
    flat = torch.frombuffer(bytearray(contiguous.tobytes()), dtype=torch_dtype)
    return flat.reshape(shape)


def host_array(value: Any) -> Any:
    """Return ``value`` as a host NumPy array; tensors go through :func:`tensor_to_numpy`.

    Parameters
    ----------
    value:
        NumPy-like object, or a tensor-like with ``detach``/``cpu``/``numpy`` methods.

    Returns
    -------
    numpy.ndarray
        NumPy view or copy of the input.
    """

    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        return tensor_to_numpy(value.detach())
    return np.asarray(value)
