"""Tensor utilities: NaN-aware comparison, memory calculation, safe_copy, safe device transfer."""

import copy
from typing import Any, Optional

import numpy as np
import torch

MAX_FLOATING_POINT_TOLERANCE = 3e-6

_cuda_available: Optional[bool] = None


def _is_cuda_available() -> bool:
    """Return True if CUDA is available on this machine (cached after first call)."""
    global _cuda_available
    if _cuda_available is None:
        _cuda_available = torch.cuda.is_available()
    return _cuda_available


def tensor_all_nan(tensor: torch.Tensor) -> bool:
    """Returns True if tensor is all nans, False otherwise."""
    if torch.isnan(tensor).int().sum() == tensor.numel():
        return True
    else:
        return False


def tensor_nanequal(tensor_a: torch.Tensor, tensor_b: torch.Tensor, allow_tolerance=False) -> bool:
    """Returns True if the two tensors are equal, allowing for nans."""
    if tensor_a.shape != tensor_b.shape:
        return False

    if tensor_a.dtype != tensor_b.dtype:
        return False

    if not torch.equal(tensor_a.isinf(), tensor_b.isinf()):
        return False

    if tensor_a.is_complex():
        tensor_a_nonan = torch.view_as_complex(
            torch.nan_to_num(torch.view_as_real(tensor_a), 0.7234691827346)
        )
        tensor_b_nonan = torch.view_as_complex(
            torch.nan_to_num(torch.view_as_real(tensor_b), 0.7234691827346)
        )
    else:
        tensor_a_nonan = torch.nan_to_num(tensor_a, 0.7234691827346)
        tensor_b_nonan = torch.nan_to_num(tensor_b, 0.7234691827346)

    if torch.equal(tensor_a_nonan, tensor_b_nonan):
        return True

    if (
        allow_tolerance
        and (tensor_a_nonan.dtype != torch.bool)
        and (tensor_b_nonan.dtype != torch.bool)
        and ((tensor_a_nonan - tensor_b_nonan).abs().max() <= MAX_FLOATING_POINT_TOLERANCE)
    ):
        return True

    return False


def safe_to(obj: Any, device: str) -> Any:
    """Moves object to device if it's a tensor, does nothing otherwise.

    Args:
        obj: The object.
        device: which device to move to

    Returns:
        Object either moved to device if a tensor, same object if otherwise.
    """
    from .._state import pause_logging

    if isinstance(obj, torch.Tensor):
        with pause_logging():
            return obj.to(device)
    else:
        return obj


def get_tensor_memory_amount(t: torch.Tensor) -> int:
    """Returns the size of a tensor in bytes.

    Args:
        t: Tensor.

    Returns:
        Size of tensor in bytes.
    """
    from .._state import pause_logging

    try:
        with pause_logging():
            if t.device.type == "meta":
                return 0
            if t.is_sparse:
                return t._values().nelement() * t._values().element_size()
            return t.nelement() * t.element_size()
    except Exception:
        return 0


def safe_copy(x, detach_tensor: bool = False):
    """Utility function to make a copy of a tensor or parameter, or just copy
    the thing if it's not a tensor.  Uses ``pause_logging()`` so that
    clone / cpu / to calls don't get logged.

    Args:
        x: Input
        detach_tensor: Whether to detach the cloned tensor from the computational graph or not.

    Returns:
        Safely copied variant of the input with same values and same class, but different memory
    """
    from .._state import pause_logging

    if isinstance(x, (torch.Tensor, torch.nn.Parameter)):
        with pause_logging():
            if not detach_tensor:
                return x.clone()
            # Detach path: use pure-torch ops — no numpy round-trip.
            # This avoids crashes on sparse, quantized, complex32, meta, float8, etc.
            try:
                vals_tensor = x.detach().clone()
            except Exception:
                # Fallback for exotic dtypes that can't clone directly
                try:
                    vals_tensor = x.data.cpu().clone()
                except Exception:
                    # Last resort: return shape-preserving zero tensor
                    vals_tensor = torch.zeros(x.shape, dtype=torch.float32)
            if hasattr(x, "tl_tensor_label_raw"):
                setattr(vals_tensor, "tl_tensor_label_raw", getattr(x, "tl_tensor_label_raw"))
            if isinstance(x, torch.nn.Parameter):
                return torch.nn.Parameter(vals_tensor)
            return vals_tensor
    else:
        return copy.copy(x)


def print_override(t: torch.Tensor, func_name: str):
    """Overrides the __str__ and __repr__ methods of Tensor so as not to lead to any infinite recursion.

    Args:
        t: Tensor
        func_name: Either "__str__" or "__repr__"

    Returns:
        The string representation of the tensor.
    """
    from .._state import pause_logging

    try:
        with pause_logging():
            cpu_data = t.data.cpu()
            if cpu_data.dtype == torch.bfloat16:
                cpu_data = cpu_data.to(torch.float32)
        n = np.array(cpu_data)
        np_str = getattr(n, func_name)()
        np_str = np_str.replace("array", "tensor")
        np_str = np_str.replace("\n", "\n ")
    except Exception:
        # Fallback for sparse, quantized, meta, float8, etc.
        np_str = f"tensor(shape={list(t.shape)}, dtype={t.dtype})"
    if t.grad_fn is not None:
        grad_fn_str = f", grad_fn={type(t.grad_fn).__name__})"
        np_str = np_str[0:-1] + grad_fn_str
    elif t.requires_grad:
        np_str = np_str[0:-1] + ", requires_grad=True)"
    return np_str
