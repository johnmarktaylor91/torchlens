"""Numeric NumPy values carried by the existing array payload transport."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from .tensor_policy import FailReason, Ok, TensorPolicyDecision


class NumpyPayloadCodec:
    """Preserve raw numeric arrays without importing a model backend runtime."""

    backend_name = "numpy"
    codec_name = "numpy_safetensors_v1"

    def __init__(
        self, encoded_array: Callable[..., Any], manifest_builder: Callable[..., dict[str, Any]]
    ) -> None:
        self._encoded_array = encoded_array
        self._manifest_builder = manifest_builder

    def can_encode(self, value: Any) -> bool:
        """Recognize plain numeric arrays; object and structured arrays remain unencodable."""

        return (
            isinstance(value, np.ndarray)
            and value.dtype.kind in "biufc"
            and not value.dtype.hasobject
        )

    def validate_for_save(self, value: Any, *, strict: bool = True) -> TensorPolicyDecision:
        """Require the same numeric-array boundary regardless of the save strictness knob."""

        return Ok() if self.can_encode(value) else FailReason("raw array has a nonnumeric dtype")

    def to_numpy(self, value: Any) -> Any:
        """Own a contiguous copy before the asynchronous chunk writer can observe it."""

        return self._encoded_array(np.array(value, copy=True, order="C"), str(value.dtype), "cpu")

    def from_numpy(self, array: np.ndarray, entry: Any, **kwargs: Any) -> np.ndarray:
        """Restore an independent NumPy value; NumPy arrays are always host values."""

        return np.array(array, copy=True, order="C")

    def manifest_fields(self, value: Any, encoded: Any) -> dict[str, Any]:
        """Use the shared array metadata builder so the record reference retains its codec."""

        return self._manifest_builder(
            logical_backend=self.backend_name, codec=self.codec_name, value=value, encoded=encoded
        )


def numpy_to_transport_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert an encoded NumPy array to the CPU torch transport tensor."""

    _raise_for_unsupported_array_dtype(array, backend_name="transport")
    # np.array(copy=True, order="C") already yields a C-contiguous buffer, so
    # from_numpy's zero-copy wrap is contiguous by construction.
    return torch.from_numpy(np.array(array, copy=True, order="C"))


def _unsupported_array_dtype_reason(array: np.ndarray) -> str | None:
    """Return an unsupported dtype reason for a host array, if any."""

    if array.dtype.kind in {"O", "S", "U", "V"}:
        return f"dtype {array.dtype} is not supported by the payload codec"
    return None


def _dtype_string_is_object_like(dtype: str) -> bool:
    """Return whether a dtype string describes object or string data."""

    lowered = dtype.lower()
    return any(token in lowered for token in ("object", "str", "string", "bytes", "void"))


def _raise_for_unsupported_array_dtype(array: np.ndarray, *, backend_name: str) -> None:
    """Raise ``TypeError`` if ``array`` has a non-transportable dtype."""

    reason = _unsupported_array_dtype_reason(array)
    if reason is not None:
        raise TypeError(f"{backend_name} payload {reason}.")
