"""Backend-neutral payload codecs for portable bundle writes.

This module keeps logical array extraction separate from the physical
``safetensors`` transport used by bundle files. The torch codec preserves the
legacy write path; preview backend codecs expose NumPy arrays for direct unit
tests and future materialization-enabled saves.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from typing import Any, ClassVar, Protocol

import numpy as np
import torch

from . import JaxPayloadLoadHint, PayloadLoadHints
from .tensor_policy import FailReason, Ok, SkipReason, TensorPolicyDecision, is_supported_for_save


@dataclass(frozen=True)
class EncodedArray:
    """A logical backend payload normalized to an array boundary.

    Parameters
    ----------
    array:
        Host NumPy array carrying the payload values.
    logical_dtype:
        Backend-native dtype string before transport conversion.
    logical_device:
        Backend-native device string before transport conversion.
    codec_metadata:
        Optional JSON-ready metadata needed by future load-side codecs.
    """

    array: np.ndarray
    logical_dtype: str
    logical_device: str
    codec_metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class _JaxHintResult:
    """Internal result for JAX hint application."""

    applied: bool
    value: Any


class PayloadCodec(Protocol):
    """Protocol implemented by backend payload codecs."""

    backend_name: str
    codec_name: str

    def can_encode(self, value: Any) -> bool:
        """Return whether this codec recognizes ``value`` as a logical payload."""

    def validate_for_save(self, value: Any, *, strict: bool = True) -> TensorPolicyDecision:
        """Validate whether ``value`` can be written as a portable payload."""

    def to_numpy(self, value: Any) -> EncodedArray:
        """Convert ``value`` to a host NumPy representation."""

    def from_numpy(
        self,
        array: np.ndarray,
        entry: Any,
        *,
        map_location: Any,
        payload_hints: PayloadLoadHints | Mapping[str, Any] | None = None,
        strict_runtime: bool = True,
    ) -> Any:
        """Rebuild a logical backend payload from a NumPy array."""

    def manifest_fields(self, value: Any, encoded: EncodedArray) -> dict[str, Any]:
        """Return optional v2 manifest fields for ``value`` and ``encoded``."""


class TorchPayloadCodec:
    """Payload codec for the existing PyTorch bundle behavior."""

    backend_name = "torch"
    codec_name = "torch_safetensors_v1"

    def can_encode(self, value: Any) -> bool:
        """Return whether ``value`` is a plain PyTorch tensor payload."""

        return isinstance(value, torch.Tensor)

    def validate_for_save(self, value: Any, *, strict: bool = True) -> TensorPolicyDecision:
        """Delegate PyTorch payload validation to the existing tensor policy."""

        if not isinstance(value, torch.Tensor):
            reason = f"expected torch.Tensor, got {type(value).__name__}"
            return FailReason(reason) if strict else SkipReason(reason)
        return is_supported_for_save(value, strict=strict)

    def to_numpy(self, value: Any) -> EncodedArray:
        """Convert a PyTorch tensor to a detached CPU NumPy array."""

        if not isinstance(value, torch.Tensor):
            raise TypeError(f"torch codec expected torch.Tensor, got {type(value).__name__}.")
        array = value.detach().cpu().numpy()
        return EncodedArray(
            array=array,
            logical_dtype=str(value.dtype).replace("torch.", ""),
            logical_device=str(value.device),
        )

    def from_numpy(
        self,
        array: np.ndarray,
        entry: Any,
        *,
        map_location: Any,
        payload_hints: PayloadLoadHints | Mapping[str, Any] | None = None,
        strict_runtime: bool = True,
    ) -> Any:
        """Rebuild a PyTorch tensor from a NumPy array."""

        del payload_hints
        tensor = torch.from_numpy(np.ascontiguousarray(array))
        if map_location is None:
            return tensor
        return tensor.to(map_location)

    def manifest_fields(self, value: Any, encoded: EncodedArray) -> dict[str, Any]:
        """Return no optional fields so torch manifests stay byte-compatible."""

        return {}


class JaxPayloadCodec:
    """Payload codec for addressable dense JAX arrays."""

    backend_name = "jax"
    codec_name = "numpy_safetensors_v1"
    _PRNG_DTAG_TO_IMPL: ClassVar[dict[str, str]] = {"fry": "threefry2x32"}

    def can_encode(self, value: Any) -> bool:
        """Return whether ``value`` looks like a JAX array without importing JAX."""

        value_type = type(value)
        module_name = value_type.__module__
        return module_name.startswith(("jax", "jaxlib")) and hasattr(value, "dtype")

    def validate_for_save(self, value: Any, *, strict: bool = True) -> TensorPolicyDecision:
        """Validate that a JAX value is dense, addressable, and host-copyable."""

        reason = self._unsupported_reason(value)
        if reason is None:
            return Ok()
        if strict:
            return FailReason(reason)
        return SkipReason(reason)

    def to_numpy(self, value: Any) -> EncodedArray:
        """Copy a JAX array to host NumPy storage."""

        import jax

        if not self.can_encode(value):
            raise TypeError(f"jax codec cannot encode {type(value).__name__}.")
        unaddressable_reason = _jax_unaddressable_reason(value)
        if unaddressable_reason is not None:
            raise TypeError(unaddressable_reason)
        logical_dtype = str(getattr(value, "dtype", "unknown"))
        sharding_metadata = _jax_sharding_metadata(value)
        if _is_jax_typed_prng_dtype(logical_dtype):
            key_data = jax.device_get(jax.random.key_data(value))
            array = np.asarray(key_data, dtype=np.uint32)
            dtag = _jax_prng_dtag_from_dtype(logical_dtype)
            impl = self._jax_prng_impl(value, logical_dtype)
            _raise_for_unsupported_array_dtype(array, backend_name=self.backend_name)
            return EncodedArray(
                array=np.ascontiguousarray(array),
                logical_dtype=logical_dtype,
                logical_device=_device_string(value),
                codec_metadata=_json_ready_mapping(
                    {
                        "logical_shape": [int(dim) for dim in getattr(value, "shape", ())],
                        "weak_type": getattr(value, "weak_type", None),
                        "committed": getattr(value, "committed", None),
                        "jax_prng_key_typed": True,
                        "jax_prng_impl": impl,
                        "jax_prng_dtag": dtag,
                        **sharding_metadata,
                    }
                ),
            )
        host_value = jax.device_get(value)
        array = np.asarray(host_value)
        _raise_for_unsupported_array_dtype(array, backend_name=self.backend_name)
        return EncodedArray(
            array=np.ascontiguousarray(array),
            logical_dtype=logical_dtype,
            logical_device=_device_string(value),
            codec_metadata=_json_ready_mapping(
                {
                    "logical_shape": [int(dim) for dim in array.shape],
                    "weak_type": getattr(value, "weak_type", None),
                    "committed": getattr(value, "committed", None),
                    **sharding_metadata,
                }
            ),
        )

    def from_numpy(
        self,
        array: np.ndarray,
        entry: Any,
        *,
        map_location: Any,
        payload_hints: PayloadLoadHints | Mapping[str, Any] | None = None,
        strict_runtime: bool = True,
    ) -> Any:
        """Rebuild a JAX array from host NumPy storage."""

        from ..backends.registry import BackendRuntimeCompatibilityError

        try:
            import jax
            import jax.numpy as jnp
        except ImportError as exc:
            raise BackendRuntimeCompatibilityError(
                "Portable JAX payload materialization requires the jax runtime."
            ) from exc

        logical_dtype = str(_entry_field(entry, "logical_dtype") or array.dtype)
        if logical_dtype.startswith("key<"):
            impl = self._jax_prng_impl_from_entry(entry, logical_dtype)
            try:
                value = jax.random.wrap_key_data(jnp.asarray(array, dtype=jnp.uint32), impl=impl)
            except (TypeError, ValueError, RuntimeError) as exc:
                raise BackendRuntimeCompatibilityError(
                    f"Portable JAX PRNG key impl {impl!r} is not supported by the installed "
                    "jax runtime."
                ) from exc
            hint_result = _apply_jax_payload_hints(
                jax,
                value,
                entry,
                map_location=map_location,
                payload_hints=payload_hints,
            )
            return hint_result.value

        dtype = getattr(jnp, logical_dtype, None)
        value = jnp.asarray(array, dtype=dtype) if dtype is not None else jnp.asarray(array)
        hint_result = _apply_jax_payload_hints(
            jax,
            value,
            entry,
            map_location=map_location,
            payload_hints=payload_hints,
        )
        if hint_result.applied:
            return hint_result.value
        if map_location is None:
            return value
        target_device = _resolve_jax_device(jax, map_location)
        if target_device is None:
            return value
        try:
            return jax.device_put(value, target_device)
        except (TypeError, ValueError, RuntimeError):
            return value

    def manifest_fields(self, value: Any, encoded: EncodedArray) -> dict[str, Any]:
        """Return v2 manifest vocabulary for a JAX payload."""

        return _manifest_fields(
            logical_backend=self.backend_name,
            codec=self.codec_name,
            value=value,
            encoded=encoded,
        )

    def _unsupported_reason(self, value: Any) -> str | None:
        """Return the first JAX codec incompatibility reason, if any."""

        if not self.can_encode(value):
            return f"jax codec cannot encode {type(value).__name__}"
        unaddressable_reason = _jax_unaddressable_reason(value)
        if unaddressable_reason is not None:
            return unaddressable_reason
        dtype = getattr(value, "dtype", None)
        if dtype is not None and _is_jax_typed_prng_dtype(str(dtype)):
            return None
        if dtype is not None and _dtype_string_is_object_like(str(dtype)):
            return f"object/string dtype {dtype} is not supported by the payload codec"
        try:
            array = np.asarray(value)
        except (TypeError, ValueError, RuntimeError) as exc:
            return f"JAX array could not be copied to host: {exc}"
        return _unsupported_array_dtype_reason(array)

    def _jax_prng_impl(self, value: Any, logical_dtype: str) -> str:
        """Return the canonical PRNG implementation name for a typed key."""

        try:
            import jax

            return str(jax.random.key_impl(value))
        except (AttributeError, TypeError, ValueError, RuntimeError):
            dtag = _maybe_string(getattr(value, "_impl", None))
            if dtag is None:
                dtag = _jax_prng_dtag_from_dtype(logical_dtype)
            return self._jax_prng_impl_from_dtag(dtag)

    def _jax_prng_impl_from_entry(self, entry: Any, logical_dtype: str) -> str:
        """Resolve a canonical PRNG implementation name from manifest metadata."""

        from ..backends.registry import BackendRuntimeCompatibilityError

        codec_metadata = _entry_field(entry, "codec_metadata")
        impl = None
        dtag = None
        if isinstance(codec_metadata, Mapping):
            impl = codec_metadata.get("jax_prng_impl")
            dtag = codec_metadata.get("jax_prng_dtag")
        if isinstance(impl, str) and impl:
            return impl
        if not isinstance(dtag, str) or not dtag:
            dtag = _jax_prng_dtag_from_dtype(logical_dtype)
        try:
            return self._jax_prng_impl_from_dtag(dtag)
        except ValueError as exc:
            raise BackendRuntimeCompatibilityError(
                f"Portable JAX PRNG key impl/tag {dtag!r} is not supported."
            ) from exc

    def _jax_prng_impl_from_dtag(self, dtag: str | None) -> str:
        """Map a JAX PRNG dtype display tag to a canonical implementation name."""

        if dtag is None:
            raise ValueError("missing JAX PRNG key dtype tag")
        impl = self._PRNG_DTAG_TO_IMPL.get(dtag)
        if impl is None:
            raise ValueError(f"unsupported JAX PRNG key dtype tag {dtag!r}")
        return impl


class TinygradPayloadCodec:
    """Payload codec for realized dense tinygrad tensors."""

    backend_name = "tinygrad"
    codec_name = "numpy_safetensors_v1"

    def can_encode(self, value: Any) -> bool:
        """Return whether ``value`` looks like a tinygrad tensor."""

        value_type = type(value)
        return value_type.__module__.startswith("tinygrad") and hasattr(value, "dtype")

    def validate_for_save(self, value: Any, *, strict: bool = True) -> TensorPolicyDecision:
        """Validate that a tinygrad tensor can be copied to a dense host array."""

        reason = self._unsupported_reason(value)
        if reason is None:
            return Ok()
        if strict:
            return FailReason(reason)
        return SkipReason(reason)

    def to_numpy(self, value: Any) -> EncodedArray:
        """Copy a tinygrad tensor to host NumPy storage."""

        if not self.can_encode(value):
            raise TypeError(f"tinygrad codec cannot encode {type(value).__name__}.")
        if hasattr(value, "numpy"):
            array = np.asarray(value.numpy())
        else:
            array = np.asarray(value.tolist())
        _raise_for_unsupported_array_dtype(array, backend_name=self.backend_name)
        return EncodedArray(
            array=np.ascontiguousarray(array),
            logical_dtype=str(getattr(value, "dtype", array.dtype)),
            logical_device=str(getattr(value, "device", "unknown")),
            codec_metadata=_json_ready_mapping(
                {
                    "logical_shape": [int(dim) for dim in array.shape],
                    "tinygrad_device": str(getattr(value, "device", "unknown")),
                    "source_type": type(value).__name__,
                }
            ),
        )

    def from_numpy(
        self,
        array: np.ndarray,
        entry: Any,
        *,
        map_location: Any,
        payload_hints: PayloadLoadHints | Mapping[str, Any] | None = None,
        strict_runtime: bool = True,
    ) -> Any:
        """Rebuild a tinygrad tensor from host NumPy storage."""

        del payload_hints
        from ..backends.registry import BackendRuntimeCompatibilityError

        try:
            from tinygrad import Tensor, dtypes
        except ImportError as exc:
            raise BackendRuntimeCompatibilityError(
                "Portable tinygrad payload materialization requires the tinygrad runtime."
            ) from exc

        logical_dtype = str(_entry_field(entry, "logical_dtype") or array.dtype)
        dtype = _resolve_tinygrad_dtype(dtypes, logical_dtype)
        if dtype is None:
            raise BackendRuntimeCompatibilityError(
                f"Portable tinygrad payload dtype {logical_dtype!r} is not supported by "
                "the installed tinygrad runtime."
            )

        device = map_location
        if device is None:
            device = _entry_field(entry, "logical_device") or _entry_field(entry, "device_at_save")
        kwargs: dict[str, Any] = {"dtype": dtype}
        if device not in (None, "", "unknown"):
            kwargs["device"] = device
        return Tensor(array, **kwargs).realize()

    def manifest_fields(self, value: Any, encoded: EncodedArray) -> dict[str, Any]:
        """Return v2 manifest vocabulary for a tinygrad payload."""

        return _manifest_fields(
            logical_backend=self.backend_name,
            codec=self.codec_name,
            value=value,
            encoded=encoded,
        )

    def _unsupported_reason(self, value: Any) -> str | None:
        """Return the first tinygrad codec incompatibility reason, if any."""

        if not self.can_encode(value):
            return f"tinygrad codec cannot encode {type(value).__name__}"
        try:
            if hasattr(value, "numpy"):
                array = np.asarray(value.numpy())
            else:
                array = np.asarray(value.tolist())
        except (TypeError, ValueError, RuntimeError) as exc:
            return f"tinygrad tensor could not be copied to host: {exc}"
        return _unsupported_array_dtype_reason(array)


class MlxPayloadCodec:
    """Payload codec for dense MLX arrays."""

    backend_name = "mlx"
    codec_name = "numpy_safetensors_v1"

    def can_encode(self, value: Any) -> bool:
        """Return whether ``value`` looks like an MLX array."""

        value_type = type(value)
        return (
            value_type.__module__.startswith("mlx.core")
            and value_type.__name__ == "array"
            and hasattr(value, "dtype")
        )

    def validate_for_save(self, value: Any, *, strict: bool = True) -> TensorPolicyDecision:
        """Validate that an MLX array can be copied to a dense host array."""

        reason = self._unsupported_reason(value)
        if reason is None:
            return Ok()
        if strict:
            return FailReason(reason)
        return SkipReason(reason)

    def to_numpy(self, value: Any) -> EncodedArray:
        """Copy an MLX array to host NumPy storage."""

        if not self.can_encode(value):
            raise TypeError(f"mlx codec cannot encode {type(value).__name__}.")
        mx = _import_mlx_core()
        try:
            mx.eval(value)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise TypeError(f"MLX array could not be realized before host copy: {exc}") from exc
        array = np.asarray(value)
        _raise_for_unsupported_array_dtype(array, backend_name=self.backend_name)
        return EncodedArray(
            array=np.ascontiguousarray(array),
            logical_dtype=str(getattr(value, "dtype", array.dtype)),
            logical_device=_device_string(value),
            codec_metadata=_json_ready_mapping(
                {
                    "logical_shape": [int(dim) for dim in array.shape],
                    "source_type": type(value).__name__,
                }
            ),
        )

    def from_numpy(
        self,
        array: np.ndarray,
        entry: Any,
        *,
        map_location: Any,
        payload_hints: PayloadLoadHints | Mapping[str, Any] | None = None,
        strict_runtime: bool = True,
    ) -> Any:
        """Rebuild an MLX array from host NumPy storage."""

        del payload_hints, strict_runtime
        from ..backends.registry import BackendRuntimeCompatibilityError

        mx = _import_mlx_core()
        logical_dtype = str(_entry_field(entry, "logical_dtype") or array.dtype)
        dtype = _resolve_mlx_dtype(mx, logical_dtype)
        if dtype is None:
            raise BackendRuntimeCompatibilityError(
                f"Portable MLX payload dtype {logical_dtype!r} is not supported by "
                "the installed MLX runtime."
            )

        target_device = _resolve_mlx_device(mx, map_location)
        if target_device is None:
            return mx.array(array, dtype=dtype)

        previous_device = mx.default_device()
        try:
            mx.set_default_device(target_device)
            return mx.array(array, dtype=dtype)
        finally:
            mx.set_default_device(previous_device)

    def manifest_fields(self, value: Any, encoded: EncodedArray) -> dict[str, Any]:
        """Return v2 manifest vocabulary for an MLX payload."""

        return _manifest_fields(
            logical_backend=self.backend_name,
            codec=self.codec_name,
            value=value,
            encoded=encoded,
        )

    def _unsupported_reason(self, value: Any) -> str | None:
        """Return the first MLX codec incompatibility reason, if any."""

        if not self.can_encode(value):
            return f"mlx codec cannot encode {type(value).__name__}"
        dtype = getattr(value, "dtype", None)
        if dtype is not None and _dtype_string_is_object_like(str(dtype)):
            return f"object/string dtype {dtype} is not supported by the payload codec"
        try:
            array = np.asarray(value)
        except (TypeError, ValueError, RuntimeError) as exc:
            return f"MLX array could not be copied to host: {exc}"
        return _unsupported_array_dtype_reason(array)


class PaddlePayloadCodec:
    """Payload codec for dense Paddle tensors."""

    backend_name = "paddle"
    codec_name = "numpy_safetensors_v1"

    def can_encode(self, value: Any) -> bool:
        """Return whether ``value`` looks like a Paddle tensor."""

        value_type = type(value)
        return value_type.__module__.startswith("paddle") and hasattr(value, "dtype")

    def validate_for_save(self, value: Any, *, strict: bool = True) -> TensorPolicyDecision:
        """Validate that a Paddle tensor can be copied to dense host storage."""

        reason = self._unsupported_reason(value)
        if reason is None:
            return Ok()
        if strict:
            return FailReason(reason)
        return SkipReason(reason)

    def to_numpy(self, value: Any) -> EncodedArray:
        """Copy a Paddle tensor to host NumPy storage."""

        if not self.can_encode(value):
            raise TypeError(f"paddle codec cannot encode {type(value).__name__}.")
        logical_dtype = str(getattr(value, "dtype", "unknown"))
        logical_device = str(getattr(value, "place", "unknown"))
        try:
            array = np.asarray(value.numpy())
        except (TypeError, ValueError, RuntimeError) as exc:
            raise TypeError(f"Paddle tensor could not be copied to host: {exc}") from exc
        _raise_for_unsupported_array_dtype(array, backend_name=self.backend_name)
        return EncodedArray(
            array=np.ascontiguousarray(array),
            logical_dtype=logical_dtype,
            logical_device=logical_device,
            codec_metadata=_json_ready_mapping(
                {
                    "logical_shape": [int(dim) for dim in array.shape],
                    "source_type": type(value).__name__,
                    "paddle_logical_dtype": logical_dtype,
                    "paddle_logical_device": logical_device,
                    "paddle_bfloat16_transport_bits": logical_dtype == "paddle.bfloat16",
                }
            ),
        )

    def from_numpy(
        self,
        array: np.ndarray,
        entry: Any,
        *,
        map_location: Any,
        payload_hints: PayloadLoadHints | Mapping[str, Any] | None = None,
        strict_runtime: bool = True,
    ) -> Any:
        """Rebuild a Paddle tensor from host NumPy storage."""

        del payload_hints, strict_runtime
        from ..backends.registry import BackendRuntimeCompatibilityError

        try:
            import paddle
        except ImportError as exc:
            raise BackendRuntimeCompatibilityError(
                "Portable Paddle payload materialization requires the paddle runtime."
            ) from exc

        logical_dtype = str(_entry_field(entry, "logical_dtype") or array.dtype)
        dtype = _resolve_paddle_dtype(paddle, logical_dtype)
        if dtype is None:
            raise BackendRuntimeCompatibilityError(
                f"Portable Paddle payload dtype {logical_dtype!r} is not supported by "
                "the installed paddle runtime."
            )

        place = _resolve_paddle_place(paddle, map_location)
        if place is None:
            place = _resolve_paddle_place(paddle, _entry_field(entry, "logical_device"))
        try:
            if logical_dtype == "paddle.bfloat16":
                value = paddle.to_tensor(np.ascontiguousarray(array), place=place)
                return value if str(value.dtype) == logical_dtype else value.view(dtype)
            return paddle.to_tensor(np.ascontiguousarray(array), dtype=dtype, place=place)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise BackendRuntimeCompatibilityError(
                f"Portable Paddle payload could not be materialized: {exc}"
            ) from exc

    def manifest_fields(self, value: Any, encoded: EncodedArray) -> dict[str, Any]:
        """Return v2 manifest vocabulary for a Paddle payload."""

        return _manifest_fields(
            logical_backend=self.backend_name,
            codec=self.codec_name,
            value=value,
            encoded=encoded,
        )

    def _unsupported_reason(self, value: Any) -> str | None:
        """Return the first Paddle codec incompatibility reason, if any."""

        if not self.can_encode(value):
            return f"paddle codec cannot encode {type(value).__name__}"
        dtype = getattr(value, "dtype", None)
        if dtype is not None and _dtype_string_is_object_like(str(dtype)):
            return f"object/string dtype {dtype} is not supported by the payload codec"
        try:
            array = np.asarray(value.numpy())
        except (TypeError, ValueError, RuntimeError) as exc:
            return f"Paddle tensor could not be copied to host: {exc}"
        return _unsupported_array_dtype_reason(array)


class NullPayloadCodec:
    """Payload codec for backends without materialization support this round."""

    backend_name = "unknown"
    codec_name = "none"

    def can_encode(self, value: Any) -> bool:
        """Return False for all values."""

        return False

    def validate_for_save(self, value: Any, *, strict: bool = True) -> TensorPolicyDecision:
        """Reject all payload values."""

        reason = f"no payload codec is registered for {self.backend_name!r}"
        return FailReason(reason) if strict else SkipReason(reason)

    def to_numpy(self, value: Any) -> EncodedArray:
        """Raise because no payload conversion is available."""

        raise TypeError(f"no payload codec is registered for {self.backend_name!r}.")

    def from_numpy(
        self,
        array: np.ndarray,
        entry: Any,
        *,
        map_location: Any,
        payload_hints: PayloadLoadHints | Mapping[str, Any] | None = None,
        strict_runtime: bool = True,
    ) -> Any:
        """Raise because load-side conversion is unavailable."""

        del payload_hints
        raise TypeError(f"no payload codec is registered for {self.backend_name!r}.")

    def manifest_fields(self, value: Any, encoded: EncodedArray) -> dict[str, Any]:
        """Return no optional manifest fields."""

        return {}


_CODECS: dict[str, PayloadCodec] = {
    "torch": TorchPayloadCodec(),
    "jax": JaxPayloadCodec(),
    "mlx": MlxPayloadCodec(),
    "paddle": PaddlePayloadCodec(),
    "tinygrad": TinygradPayloadCodec(),
}
_TORCH_BACKEND_NAME = "torch"


def get_payload_codec(backend_name: str) -> PayloadCodec:
    """Return the registered payload codec for ``backend_name``.

    Parameters
    ----------
    backend_name:
        Registered TorchLens backend name.

    Returns
    -------
    PayloadCodec
        Backend codec, or a rejecting null codec for unsupported backends.
    """

    codec = _CODECS.get(backend_name)
    if codec is not None:
        return codec
    null_codec = NullPayloadCodec()
    null_codec.backend_name = backend_name
    return null_codec


def register_payload_codec(backend_name: str, codec: PayloadCodec) -> None:
    """Register or replace a payload codec.

    Parameters
    ----------
    backend_name:
        Registry key.
    codec:
        Payload codec implementation.
    """

    _CODECS[backend_name] = codec


def numpy_to_transport_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert an encoded NumPy array to the CPU torch transport tensor."""

    _raise_for_unsupported_array_dtype(array, backend_name="transport")
    return torch.from_numpy(np.array(array, copy=True, order="C")).contiguous()


def materialize_transport_tensor(
    tensor: torch.Tensor,
    entry: Any,
    *,
    map_location: Any,
    payload_hints: PayloadLoadHints | Mapping[str, Any] | None = None,
) -> Any:
    """Decode a safetensors transport tensor into its logical backend payload.

    Parameters
    ----------
    tensor:
        Tensor decoded from the physical safetensors blob.
    entry:
        Manifest tensor entry or public body-index mapping for the payload.
    map_location:
        Optional target device passed to the logical backend codec.
    payload_hints:
        Optional backend-specific materialization hints.

    Returns
    -------
    Any
        Materialized logical payload for the entry's backend.

    Raises
    ------
    BackendPayloadUnsupportedError
        If the manifest declares an unknown or mismatched codec.
    BackendRuntimeCompatibilityError
        If the logical backend runtime is unavailable or incompatible.
    """

    from ..backends.registry import (
        BackendPayloadUnsupportedError,
        BackendRuntimeCompatibilityError,
    )

    logical_backend = str(_entry_field(entry, "logical_backend") or "torch")
    codec_name = _entry_field(entry, "codec")
    if logical_backend == _TORCH_BACKEND_NAME:
        if map_location is None:
            return tensor
        return tensor.to(map_location)

    codec = get_payload_codec(logical_backend)
    if codec.codec_name == "none" or codec_name != codec.codec_name:
        raise BackendPayloadUnsupportedError(
            f"Manifest declares unsupported payload codec {codec_name!r} for "
            f"backend {logical_backend!r}."
        )

    try:
        array = _transport_tensor_to_numpy(tensor, entry)
        codec_map_location = map_location
        if isinstance(map_location, str) and map_location == "cpu":
            codec_map_location = None
        if isinstance(map_location, torch.device) and map_location.type == "cpu":
            codec_map_location = None
        return codec.from_numpy(
            array,
            entry,
            map_location=codec_map_location,
            payload_hints=payload_hints,
            strict_runtime=True,
        )
    except BackendRuntimeCompatibilityError:
        raise
    except ImportError as exc:
        raise BackendRuntimeCompatibilityError(
            f"Portable {logical_backend} payload materialization requires the "
            f"{logical_backend} runtime."
        ) from exc


def _manifest_fields(
    *,
    logical_backend: str,
    codec: str,
    value: Any,
    encoded: EncodedArray,
) -> dict[str, Any]:
    """Build optional v2 tensor manifest fields for a non-torch payload."""

    transport_tensor = numpy_to_transport_tensor(encoded.array)
    return {
        "logical_backend": logical_backend,
        "codec": codec,
        "logical_dtype": encoded.logical_dtype,
        "logical_device": encoded.logical_device,
        "transport_backend": "safetensors.torch",
        "transport_dtype": str(transport_tensor.dtype).replace("torch.", ""),
        "codec_metadata": encoded.codec_metadata,
    }


def _entry_field(entry: Any, field_name: str) -> Any:
    """Read a field from a dataclass-like object or mapping."""

    if isinstance(entry, Mapping):
        return entry.get(field_name)
    return getattr(entry, field_name, None)


def _transport_tensor_to_numpy(tensor: torch.Tensor, entry: Any) -> np.ndarray:
    """Convert a torch transport tensor to host NumPy storage for a codec."""

    transport = tensor.detach().cpu().contiguous()
    logical_dtype = str(_entry_field(entry, "logical_dtype") or "")
    if logical_dtype in {"bfloat16", "jax.numpy.bfloat16"}:
        array = transport.to(torch.float32).numpy()
    else:
        array = transport.numpy()
    if _entry_has_typed_jax_prng_key(entry):
        return array
    logical_shape = _logical_shape_from_metadata(_entry_field(entry, "codec_metadata"))
    if logical_shape is not None:
        return array.reshape(logical_shape)
    return array


def _entry_has_typed_jax_prng_key(entry: Any) -> bool:
    """Return whether an entry stores raw JAX typed PRNG key data."""

    codec_metadata = _entry_field(entry, "codec_metadata")
    if not isinstance(codec_metadata, Mapping):
        return False
    flag = codec_metadata.get("jax_prng_key_typed")
    if isinstance(flag, str):
        return flag.lower() == "true"
    return flag is True


def _logical_shape_from_metadata(codec_metadata: Any) -> tuple[int, ...] | None:
    """Return logical shape metadata from a codec metadata mapping.

    Parameters
    ----------
    codec_metadata:
        Manifest codec metadata mapping.

    Returns
    -------
    tuple[int, ...] or None
        Logical shape when the metadata contains a valid list of integer
        dimensions.
    """

    if not isinstance(codec_metadata, Mapping):
        return None
    logical_shape = codec_metadata.get("logical_shape")
    if isinstance(logical_shape, str):
        try:
            logical_shape = json.loads(logical_shape)
        except json.JSONDecodeError:
            return None
    if not isinstance(logical_shape, list):
        return None
    if not all(isinstance(dim, int) and not isinstance(dim, bool) for dim in logical_shape):
        return None
    return tuple(logical_shape)


def _apply_jax_payload_hints(
    jax_module: Any,
    value: Any,
    entry: Any,
    *,
    map_location: Any,
    payload_hints: PayloadLoadHints | Mapping[str, Any] | None,
) -> _JaxHintResult:
    """Apply explicit JAX payload hints to a decoded value.

    Parameters
    ----------
    jax_module:
        Imported ``jax`` module.
    value:
        Decoded JAX value before optional device placement.
    entry:
        Manifest entry carrying codec metadata.
    map_location:
        Public load ``map_location`` value, used only to preserve unchanged
        single-device behavior outside explicit hints.
    payload_hints:
        Optional backend payload hints.

    Returns
    -------
    _JaxHintResult
        Whether a hint was applied and the resulting value.
    """

    del map_location
    hint = _coerce_jax_payload_hint(payload_hints)
    if hint is None:
        return _JaxHintResult(applied=False, value=value)
    if hint.sharding is not None:
        return _JaxHintResult(
            applied=True,
            value=_jax_device_put_or_raise(jax_module, value, hint.sharding),
        )
    if not hint.reconstruct_sharding:
        return _JaxHintResult(applied=False, value=value)
    sharding = _reconstruct_jax_named_sharding(jax_module, entry, hint)
    return _JaxHintResult(
        applied=True,
        value=_jax_device_put_or_raise(jax_module, value, sharding),
    )


def _coerce_jax_payload_hint(
    payload_hints: PayloadLoadHints | Mapping[str, Any] | None,
) -> JaxPayloadLoadHint | None:
    """Return a JAX payload hint from public dataclasses or mappings."""

    if payload_hints is None:
        return None
    if isinstance(payload_hints, PayloadLoadHints):
        return payload_hints.jax
    if not isinstance(payload_hints, Mapping):
        return None
    jax_hint = payload_hints.get("jax")
    if jax_hint is None or isinstance(jax_hint, JaxPayloadLoadHint):
        return jax_hint
    if isinstance(jax_hint, Mapping):
        axis_names = jax_hint.get("mesh_axis_names")
        if axis_names is not None:
            axis_names = tuple(str(axis_name) for axis_name in axis_names)
        return JaxPayloadLoadHint(
            sharding=jax_hint.get("sharding"),
            reconstruct_sharding=bool(jax_hint.get("reconstruct_sharding", False)),
            mesh_axis_names=axis_names,
            platform=jax_hint.get("platform"),
            devices=jax_hint.get("devices"),
        )
    return None


def _jax_device_put_or_raise(jax_module: Any, value: Any, sharding: Any) -> Any:
    """Place a JAX value on an explicit sharding or fail closed."""

    from ..backends.registry import BackendRuntimeCompatibilityError

    try:
        return jax_module.device_put(value, sharding)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise BackendRuntimeCompatibilityError(
            f"Requested JAX payload sharding could not be applied: {exc}"
        ) from exc


def _reconstruct_jax_named_sharding(
    jax_module: Any,
    entry: Any,
    hint: JaxPayloadLoadHint,
) -> Any:
    """Reconstruct a JAX ``NamedSharding`` from codec metadata."""

    from ..backends.registry import BackendRuntimeCompatibilityError

    codec_metadata = _entry_field(entry, "codec_metadata")
    if not isinstance(codec_metadata, Mapping):
        raise BackendRuntimeCompatibilityError(
            "Requested JAX sharding reconstruction, but payload codec metadata is missing."
        )
    if codec_metadata.get("jax_sharding_schema_version") != 1:
        raise BackendRuntimeCompatibilityError(
            "Requested JAX sharding reconstruction, but the saved sharding schema is unsupported."
        )
    if codec_metadata.get("jax_sharding_reconstructible") is not True:
        raise BackendRuntimeCompatibilityError(
            "Requested JAX sharding reconstruction, but the saved sharding is not reconstructible."
        )
    if codec_metadata.get("jax_sharding_is_fully_addressable") is not True:
        raise BackendRuntimeCompatibilityError(
            "Requested JAX sharding reconstruction for a multi-host or unaddressable payload; "
            "only fully addressable single-host JAX arrays are supported."
        )

    named = codec_metadata.get("jax_named_sharding")
    if not isinstance(named, Mapping):
        raise BackendRuntimeCompatibilityError(
            "Requested JAX sharding reconstruction, but jax_named_sharding metadata is missing."
        )
    mesh_metadata = named.get("mesh")
    if not isinstance(mesh_metadata, Mapping):
        raise BackendRuntimeCompatibilityError(
            "Requested JAX sharding reconstruction, but mesh metadata is missing."
        )
    axis_names = _decode_jax_mesh_axis_names(mesh_metadata)
    if hint.mesh_axis_names is not None and hint.mesh_axis_names != axis_names:
        raise BackendRuntimeCompatibilityError(
            "Requested JAX sharding reconstruction with mesh axis names "
            f"{hint.mesh_axis_names!r}, but saved axis names are {axis_names!r}."
        )
    mesh_shape = _decode_jax_mesh_shape(mesh_metadata, axis_names)
    partition_spec = _decode_jax_partition_spec(named.get("partition_spec"))
    devices = _resolve_jax_mesh_devices(jax_module, mesh_shape, hint)

    try:
        from jax.sharding import Mesh, NamedSharding

        device_array = np.asarray(devices, dtype=object).reshape(
            tuple(mesh_shape[axis_name] for axis_name in axis_names)
        )
        return NamedSharding(Mesh(device_array, axis_names), partition_spec)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise BackendRuntimeCompatibilityError(
            f"Saved JAX NamedSharding metadata could not be reconstructed: {exc}"
        ) from exc


def _decode_jax_mesh_axis_names(mesh_metadata: Mapping[str, Any]) -> tuple[str, ...]:
    """Decode mesh axis names from a JAX sharding metadata block."""

    from ..backends.registry import BackendRuntimeCompatibilityError

    axis_names = mesh_metadata.get("axis_names")
    if not isinstance(axis_names, list) or not all(isinstance(name, str) for name in axis_names):
        raise BackendRuntimeCompatibilityError(
            "Saved JAX sharding metadata must contain mesh.axis_names as strings."
        )
    return tuple(axis_names)


def _decode_jax_mesh_shape(
    mesh_metadata: Mapping[str, Any],
    axis_names: tuple[str, ...],
) -> dict[str, int]:
    """Decode and validate mesh shape metadata."""

    from ..backends.registry import BackendRuntimeCompatibilityError

    shape = mesh_metadata.get("shape")
    if not isinstance(shape, Mapping):
        raise BackendRuntimeCompatibilityError(
            "Saved JAX sharding metadata must contain mesh.shape as an object."
        )
    decoded: dict[str, int] = {}
    for axis_name in axis_names:
        axis_size = shape.get(axis_name)
        if not isinstance(axis_size, int) or isinstance(axis_size, bool) or axis_size <= 0:
            raise BackendRuntimeCompatibilityError(
                f"Saved JAX sharding metadata has invalid mesh size for axis {axis_name!r}."
            )
        decoded[axis_name] = axis_size
    if set(decoded) != {str(key) for key in shape.keys()}:
        raise BackendRuntimeCompatibilityError(
            "Saved JAX sharding metadata mesh.shape keys must match mesh.axis_names."
        )
    return decoded


def _decode_jax_partition_spec(partition_spec: Any) -> Any:
    """Decode a JSON primitive partition spec into JAX ``PartitionSpec``."""

    from ..backends.registry import BackendRuntimeCompatibilityError

    if not isinstance(partition_spec, list):
        raise BackendRuntimeCompatibilityError(
            "Saved JAX sharding metadata must contain partition_spec as a list."
        )
    decoded_parts: list[Any] = []
    for part in partition_spec:
        if part is None or isinstance(part, str):
            decoded_parts.append(part)
        elif isinstance(part, list) and all(isinstance(axis_name, str) for axis_name in part):
            decoded_parts.append(tuple(part))
        else:
            raise BackendRuntimeCompatibilityError(
                "Saved JAX sharding metadata contains an unsupported PartitionSpec entry."
            )
    try:
        from jax.sharding import PartitionSpec

        return PartitionSpec(*decoded_parts)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise BackendRuntimeCompatibilityError(
            f"Saved JAX PartitionSpec metadata could not be reconstructed: {exc}"
        ) from exc


def _resolve_jax_mesh_devices(
    jax_module: Any,
    mesh_shape: Mapping[str, int],
    hint: JaxPayloadLoadHint,
) -> list[Any]:
    """Resolve local devices for a reconstructed JAX mesh."""

    from ..backends.registry import BackendRuntimeCompatibilityError

    required = math.prod(mesh_shape.values())
    if hint.devices is not None:
        devices = list(hint.devices)
    else:
        try:
            devices = list(
                jax_module.local_devices(backend=hint.platform)
                if hint.platform is not None
                else jax_module.local_devices()
            )
        except RuntimeError as exc:
            raise BackendRuntimeCompatibilityError(
                f"Requested JAX mesh devices are not available: {exc}"
            ) from exc
    if len(devices) != required:
        platform_text = f" {hint.platform}" if hint.platform is not None else ""
        raise BackendRuntimeCompatibilityError(
            f"Saved JAX sharding requires {required} local{platform_text} devices; "
            f"found {len(devices)}."
        )
    return devices


def _resolve_jax_device(jax_module: Any, map_location: Any) -> Any | None:
    """Resolve a JAX device from a user map-location value when possible."""

    if not isinstance(map_location, str):
        return None
    requested = map_location.lower()
    try:
        devices = list(jax_module.devices())
    except RuntimeError:
        return None
    for device in devices:
        device_text = str(device).lower()
        platform = str(getattr(device, "platform", "")).lower()
        if requested in {device_text, platform} or requested in device_text:
            return device
    return None


def _resolve_tinygrad_dtype(dtypes_module: Any, dtype_name: str) -> Any | None:
    """Resolve a tinygrad dtype object from a manifest dtype string."""

    candidate_names = [dtype_name]
    if dtype_name.startswith("dtypes."):
        candidate_names.append(dtype_name.split(".", maxsplit=1)[1])
    if dtype_name.startswith("tinygrad.dtype."):
        candidate_names.append(dtype_name.rsplit(".", maxsplit=1)[-1])
    if dtype_name == "float":
        candidate_names.append("float32")
    for candidate_name in candidate_names:
        dtype = getattr(dtypes_module, candidate_name, None)
        if dtype is not None:
            return dtype
    return None


def _import_mlx_core() -> Any:
    """Import ``mlx.core`` or raise a targeted backend compatibility error."""

    from ..backends.registry import BackendRuntimeCompatibilityError

    try:
        import mlx.core as mx
    except ImportError as exc:
        raise BackendRuntimeCompatibilityError(
            "mlx-runtime-missing: Portable MLX payload materialization requires the mlx runtime."
        ) from exc
    return mx


def _resolve_mlx_dtype(mx_module: Any, dtype_name: str) -> Any | None:
    """Resolve an MLX dtype object from a manifest dtype string."""

    candidate_names = [dtype_name]
    if dtype_name.startswith("mlx.core."):
        candidate_names.append(dtype_name.rsplit(".", maxsplit=1)[-1])
    if dtype_name == "bool":
        candidate_names.append("bool_")
    if dtype_name == "mlx.core.bool":
        candidate_names.append("bool_")
    for candidate_name in candidate_names:
        dtype = getattr(mx_module, candidate_name, None)
        if dtype is not None:
            return dtype
    return None


def _resolve_mlx_device(mx_module: Any, map_location: Any) -> Any | None:
    """Resolve an MLX default device override from ``map_location`` when possible."""

    if map_location is None:
        return None
    device_type = getattr(mx_module, "DeviceType", None)
    device = getattr(mx_module, "Device", None)
    if device is not None and isinstance(map_location, device):
        return map_location
    if device_type is not None and isinstance(map_location, device_type):
        return mx_module.Device(map_location)
    if not isinstance(map_location, str):
        return None
    requested = map_location.lower()
    if requested.startswith("device(") and "cpu" in requested:
        requested = "cpu"
    if requested.startswith("device(") and "gpu" in requested:
        requested = "gpu"
    if requested in {"cpu", "gpu"}:
        device_type_value = getattr(mx_module, requested, None)
        if device_type_value is None or device is None:
            return None
        return mx_module.Device(device_type_value)
    return None


def _resolve_paddle_dtype(paddle_module: Any, dtype_name: str) -> Any | None:
    """Resolve a Paddle dtype object from a manifest dtype string."""

    candidate_names = [dtype_name]
    if dtype_name.startswith("paddle."):
        candidate_names.append(dtype_name.rsplit(".", maxsplit=1)[-1])
    if dtype_name == "bool":
        candidate_names.append("bool")
    for candidate_name in candidate_names:
        dtype = getattr(paddle_module, candidate_name, None)
        if dtype is not None:
            return dtype
    return None


def _resolve_paddle_place(paddle_module: Any, place_name: Any) -> Any | None:
    """Resolve a Paddle place object from manifest or map-location text."""

    if place_name is None:
        return None
    base_place = getattr(paddle_module, "Place", None)
    if base_place is not None and isinstance(place_name, base_place):
        return place_name
    if not isinstance(place_name, str):
        return None
    requested = place_name.lower()
    if requested in {"", "unknown"}:
        return None
    if requested in {"cpu", "place(cpu)"}:
        return paddle_module.CPUPlace()
    if requested in {"gpu", "cuda", "place(gpu)", "place(cuda)"}:
        cuda_place = getattr(paddle_module, "CUDAPlace", None)
        if cuda_place is not None:
            return cuda_place(0)
    for prefix in ("gpu:", "cuda:"):
        if requested.startswith(prefix):
            cuda_place = getattr(paddle_module, "CUDAPlace", None)
            if cuda_place is None:
                return None
            try:
                return cuda_place(int(requested.split(":", maxsplit=1)[1]))
            except ValueError:
                return None
    return None


def _device_string(value: Any) -> str:
    """Return a stable best-effort logical device string."""

    devices = getattr(value, "devices", None)
    if callable(devices):
        try:
            return ",".join(sorted(str(device) for device in devices()))
        except (TypeError, RuntimeError):
            return "unknown"
    device = getattr(value, "device", None)
    if callable(device):
        try:
            return str(device())
        except (TypeError, RuntimeError):
            return "unknown"
    if device is not None:
        return str(device)
    return "unknown"


def _maybe_string(value: Any) -> str | None:
    """Return ``str(value)`` unless ``value`` is ``None``."""

    if value is None:
        return None
    return str(value)


def _is_jax_typed_prng_dtype(dtype: str) -> bool:
    """Return whether a dtype string names a JAX typed PRNG key."""

    return dtype.startswith("key<")


def _jax_prng_dtag_from_dtype(dtype: str) -> str | None:
    """Return the display tag from a JAX typed PRNG key dtype string."""

    if not _is_jax_typed_prng_dtype(dtype) or not dtype.endswith(">"):
        return None
    return dtype.removeprefix("key<").removesuffix(">")


def _json_ready_mapping(values: dict[str, Any]) -> dict[str, Any]:
    """Return a mapping with only JSON-friendly values."""

    cleaned: dict[str, Any] = {}
    for key, value in values.items():
        if value is None:
            continue
        cleaned[key] = _json_ready_value(value)
    return cleaned


def _json_ready_value(value: Any) -> Any:
    """Convert one value to JSON-friendly primitives."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_ready_value(item) for key, item in value.items() if item is not None
        }
    if isinstance(value, (list, tuple)):
        return [_json_ready_value(item) for item in value]
    return str(value)


def _jax_unaddressable_reason(value: Any) -> str | None:
    """Return the strict-fail reason for JAX arrays that cannot be host-assembled."""

    if getattr(value, "is_fully_addressable", True) is False:
        return (
            "JAX array payloads must be fully addressable to assemble host values; "
            "multi-host or unaddressable sharded arrays are not supported by the payload codec"
        )
    return None


def _jax_sharding_metadata(value: Any) -> dict[str, Any]:
    """Return JSON primitive metadata for a JAX array sharding."""

    sharding = getattr(value, "sharding", None)
    if sharding is None:
        return {}
    named_sharding = _jax_named_sharding_contract(sharding)
    metadata: dict[str, Any] = {
        "jax_sharding_schema_version": 1,
        "jax_sharding_kind": type(sharding).__name__,
        "jax_sharding_reconstructible": named_sharding is not None,
        "jax_sharding_is_fully_addressable": getattr(value, "is_fully_addressable", None),
        "jax_sharding_is_fully_replicated": getattr(value, "is_fully_replicated", None),
        "jax_sharding_device_count": _jax_sharding_device_count(value),
        "jax_sharding_addressable_device_count": _jax_sharding_addressable_device_count(sharding),
        "jax_sharding_platforms": _jax_sharding_platforms(sharding),
    }
    mesh_metadata = _jax_named_sharding_mesh_metadata(sharding)
    if mesh_metadata:
        metadata.update(mesh_metadata)
    partition_spec = _jax_sharding_partition_spec(sharding)
    if partition_spec is not None:
        metadata["jax_sharding_partition_spec"] = partition_spec
    if named_sharding is not None:
        metadata["jax_named_sharding"] = named_sharding
    return _json_ready_mapping(metadata)


def _jax_named_sharding_mesh_metadata(sharding: Any) -> dict[str, Any]:
    """Return flat audit metadata for a JAX ``NamedSharding`` mesh."""

    mesh = getattr(sharding, "mesh", None)
    if mesh is None:
        return {}
    axis_names = getattr(mesh, "axis_names", None)
    shape = getattr(mesh, "shape", None)
    metadata: dict[str, Any] = {}
    if axis_names is not None:
        metadata["jax_sharding_mesh_axis_names"] = [str(axis_name) for axis_name in axis_names]
    if isinstance(shape, Mapping):
        metadata["jax_sharding_mesh_shape"] = {str(key): int(value) for key, value in shape.items()}
    return metadata


def _jax_named_sharding_contract(sharding: Any) -> dict[str, Any] | None:
    """Return the reconstructible JAX ``NamedSharding`` metadata block."""

    if type(sharding).__name__ != "NamedSharding":
        return None
    mesh = getattr(sharding, "mesh", None)
    if mesh is None:
        return None
    axis_names = getattr(mesh, "axis_names", None)
    shape = getattr(mesh, "shape", None)
    partition_spec = _jax_sharding_partition_spec(sharding)
    if axis_names is None or not isinstance(shape, Mapping) or partition_spec is None:
        return None
    axis_names_list = [str(axis_name) for axis_name in axis_names]
    mesh_shape = {str(key): int(value) for key, value in shape.items()}
    if set(axis_names_list) != set(mesh_shape):
        return None
    return {
        "mesh": {
            "axis_names": axis_names_list,
            "shape": mesh_shape,
        },
        "partition_spec": partition_spec,
    }


def _jax_sharding_partition_spec(sharding: Any) -> list[Any] | None:
    """Return the JAX partition spec as JSON primitives."""

    spec = getattr(sharding, "spec", None)
    if spec is None:
        return None
    parts = getattr(spec, "partitions", None)
    if parts is None:
        try:
            parts = tuple(spec)
        except TypeError:
            return None
    encoded_parts: list[Any] = []
    for part in parts:
        encoded_part = _jax_partition_spec_part(part)
        if encoded_part is _UNSUPPORTED_JAX_PARTITION_SPEC:
            return None
        encoded_parts.append(encoded_part)
    return encoded_parts


_UNSUPPORTED_JAX_PARTITION_SPEC = object()


def _jax_partition_spec_part(part: Any) -> Any:
    """Encode one JAX ``PartitionSpec`` part as a JSON primitive."""

    if part is None or isinstance(part, str):
        return part
    if isinstance(part, tuple) and all(isinstance(axis_name, str) for axis_name in part):
        return list(part)
    if isinstance(part, list) and all(isinstance(axis_name, str) for axis_name in part):
        return list(part)
    return _UNSUPPORTED_JAX_PARTITION_SPEC


def _jax_sharding_platforms(sharding: Any) -> list[str]:
    """Return sorted device platforms represented by a JAX sharding."""

    devices = getattr(sharding, "device_set", None)
    if devices is None:
        devices = getattr(sharding, "addressable_devices", None)
    if devices is None:
        return []
    try:
        return sorted({str(getattr(device, "platform", "unknown")) for device in devices})
    except TypeError:
        return []


def _jax_sharding_device_count(value: Any) -> int:
    """Return a best-effort JAX sharding device count."""

    sharding = getattr(value, "sharding", None)
    if sharding is None:
        return 0
    devices = getattr(sharding, "device_set", None)
    if devices is not None:
        try:
            return len(devices)
        except TypeError:
            return 0
    addressable_devices = getattr(sharding, "addressable_devices", None)
    if addressable_devices is not None:
        try:
            return len(addressable_devices)
        except TypeError:
            return 0
    return 0


def _jax_sharding_addressable_device_count(sharding: Any) -> int:
    """Return a best-effort JAX sharding addressable device count."""

    addressable_devices = getattr(sharding, "addressable_devices", None)
    if addressable_devices is None:
        return 0
    try:
        return len(addressable_devices)
    except TypeError:
        return 0


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
