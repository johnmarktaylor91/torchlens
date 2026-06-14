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
from typing import Any, ClassVar, Protocol

import numpy as np
import torch

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
        strict_runtime: bool,
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
        strict_runtime: bool,
    ) -> Any:
        """Rebuild a PyTorch tensor from a NumPy array."""

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
        logical_dtype = str(getattr(value, "dtype", "unknown"))
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
                        "sharding": _maybe_string(getattr(value, "sharding", None)),
                        "committed": getattr(value, "committed", None),
                        "jax_prng_key_typed": True,
                        "jax_prng_impl": impl,
                        "jax_prng_dtag": dtag,
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
                    "sharding": _maybe_string(getattr(value, "sharding", None)),
                    "committed": getattr(value, "committed", None),
                }
            ),
        )

    def from_numpy(
        self,
        array: np.ndarray,
        entry: Any,
        *,
        map_location: Any,
        strict_runtime: bool,
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
                return jax.random.wrap_key_data(jnp.asarray(array, dtype=jnp.uint32), impl=impl)
            except (TypeError, ValueError, RuntimeError) as exc:
                raise BackendRuntimeCompatibilityError(
                    f"Portable JAX PRNG key impl {impl!r} is not supported by the installed "
                    "jax runtime."
                ) from exc

        dtype = getattr(jnp, logical_dtype, None)
        value = jnp.asarray(array, dtype=dtype) if dtype is not None else jnp.asarray(array)
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
        if getattr(value, "is_fully_addressable", True) is False:
            return "unaddressable JAX arrays are not supported by the payload codec"
        if _jax_sharding_device_count(value) > 1:
            return "sharded JAX arrays are not supported by the payload codec"
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
        strict_runtime: bool,
    ) -> Any:
        """Rebuild a tinygrad tensor from host NumPy storage."""

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
        strict_runtime: bool,
    ) -> Any:
        """Raise because load-side conversion is unavailable."""

        raise TypeError(f"no payload codec is registered for {self.backend_name!r}.")

    def manifest_fields(self, value: Any, encoded: EncodedArray) -> dict[str, Any]:
        """Return no optional manifest fields."""

        return {}


_CODECS: dict[str, PayloadCodec] = {
    "torch": TorchPayloadCodec(),
    "jax": JaxPayloadCodec(),
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


def _resolve_jax_device(jax_module: Any, map_location: Any) -> Any | None:
    """Resolve a JAX device from a user map-location value when possible."""

    if not isinstance(map_location, str):
        return map_location
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
    """Return a mapping with only simple JSON-friendly values."""

    cleaned: dict[str, Any] = {}
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        elif isinstance(value, list) and all(
            isinstance(item, (str, int, float, bool)) for item in value
        ):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned


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
