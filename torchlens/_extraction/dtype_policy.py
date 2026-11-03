"""Save-dtype routing: explicit, routed, disclosed (extract D11 + item 9).

``dtype=`` is global or per-key; ``None`` preserves the captured dtype. The
cast runs AFTER pool and postprocess transforms, BEFORE the CPU-contiguous
snapshot. Captured, requested, stored, and conversion facts are manifested.

bf16 and both fp8 dtypes (``float8_e4m3fn``, ``float8_e5m2``) round-trip
BYTE-EXACT through safetensors on the pinned stack, while NumPy raises
``TypeError`` for all three — so portable exports widen to fp32 with exact
disclosure or refuse on request (the exporter's rule; this module supplies
the facts). Storage support is checked through the ONE
``_io/tensor_policy`` door, never re-derived here.

Every spelling here is DOCUMENTED-UNSTABLE pending the naming sprint.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from .._errors import InvalidArgumentError

__tl_layer__ = "L5"

__all__ = ["cast_for_store", "resolve_dtype_policy"]

#: Closed name -> dtype table for string spellings (the reverse of
#: ``str(dtype)`` without the ``torch.`` prefix).
_DTYPE_NAMES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
    "int64": torch.int64,
    "int32": torch.int32,
    "int16": torch.int16,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "bool": torch.bool,
}


def _coerce_dtype(value: Any) -> torch.dtype:
    """Coerce one dtype spelling to a ``torch.dtype``.

    Parameters
    ----------
    value:
        ``torch.dtype`` or a closed-vocabulary name string.

    Returns
    -------
    torch.dtype
        The resolved dtype.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``extraction_dtype_invalid`` outside the closed vocabulary.
    """

    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str) and value in _DTYPE_NAMES:
        return _DTYPE_NAMES[value]
    raise InvalidArgumentError(
        f"dtype= value {value!r} is neither a torch.dtype nor one of the "
        f"accepted names {sorted(_DTYPE_NAMES)}.",
        code="extraction_dtype_invalid",
        remedy="pass a torch.dtype (e.g. torch.float16) or its name string",
        value=repr(value),
    )


def resolve_dtype_policy(dtype: Any) -> tuple[dict[str, Any] | None, Any]:
    """Resolve the ``dtype=`` kwarg into per-key casts plus the signature record.

    Parameters
    ----------
    dtype:
        ``None`` (preserve) | dtype/name (global) | per-key
        ``Mapping[str, dtype/name]``.

    Returns
    -------
    tuple[dict | None, Any]
        ``(policy, record)``: policy is ``{"global": torch.dtype}`` or
        ``{"per_key": {key: torch.dtype}}``; record is the JSON-portable
        signature form using name strings.
    """

    if dtype is None:
        return None, None
    if isinstance(dtype, Mapping):
        per_key = {str(key): _coerce_dtype(value) for key, value in dtype.items()}
        record = {
            "per_key": {key: str(value).removeprefix("torch.") for key, value in per_key.items()}
        }
        return {"per_key": per_key}, record
    resolved = _coerce_dtype(dtype)
    return {"global": resolved}, {"global": str(resolved).removeprefix("torch.")}


def _target_for_key(policy: Mapping[str, Any] | None, key: str) -> torch.dtype | None:
    """Look up the target dtype applying to one output key.

    Parameters
    ----------
    policy:
        Resolved dtype policy.
    key:
        Output key.

    Returns
    -------
    torch.dtype | None
        The cast target, or ``None`` to preserve.
    """

    if policy is None:
        return None
    if "global" in policy:
        return policy["global"]
    return policy.get("per_key", {}).get(key)


def cast_for_store(
    key: str,
    tensor: torch.Tensor,
    policy: Mapping[str, Any] | None,
    shard_format: str,
) -> tuple[torch.Tensor, dict[str, Any] | None]:
    """Cast one stored tensor per the dtype policy and disclose the facts.

    Parameters
    ----------
    key:
        Output key.
    tensor:
        Post-transform tensor about to be snapshotted.
    policy:
        Resolved dtype policy.
    shard_format:
        Active shard format (``"safetensors"`` / ``"pt"``); storage support
        for the target dtype is checked through ``_io/tensor_policy``.

    Returns
    -------
    tuple[torch.Tensor, dict | None]
        The (possibly cast) tensor and the JSON-portable conversion fact
        (``None`` when no cast applied).

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``extraction_dtype_unsupported`` when the cast kernel or the shard
        format cannot carry the target dtype.
    """

    target = _target_for_key(policy, key)
    if target is None or tensor.dtype == target:
        return tensor, None
    try:
        cast = tensor.to(target)
    except (RuntimeError, TypeError) as exc:
        raise InvalidArgumentError(
            f"Casting output key {key!r} from {tensor.dtype} to {target} "
            f"failed ({exc}); this cast kernel is unsupported on the "
            "capture device.",
            code="extraction_dtype_unsupported",
            remedy="choose a supported save dtype for this key (e.g. float16)",
            key=key,
            captured_dtype=str(tensor.dtype),
            requested_dtype=str(target),
        ) from exc
    if shard_format == "safetensors" and cast.dtype not in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        # fp8 is measured byte-exact through the shard codec on the pinned
        # stack (extract D11) but excluded from the tlspec bundle policy;
        # every OTHER dtype routes through the ONE _io/tensor_policy door.
        from .._io import tensor_policy

        verdict = tensor_policy.is_supported_for_save(cast)
        if not isinstance(verdict, tensor_policy.Ok):
            raise InvalidArgumentError(
                f"Output key {key!r} cast to {target} cannot be stored by "
                f"the safetensors shard codec: "
                f"{getattr(verdict, 'text', 'unsupported tensor')}.",
                code="extraction_dtype_unsupported",
                remedy="choose a storable save dtype, or shard_format='pt'",
                key=key,
                requested_dtype=str(target),
            )
    captured_bits = tensor.element_size() * 8
    stored_bits = cast.element_size() * 8
    same_class = tensor.is_floating_point() == cast.is_floating_point()
    exact_widening = same_class and stored_bits >= captured_bits and tensor.is_floating_point()
    return cast, {
        "captured_dtype": str(tensor.dtype),
        "stored_dtype": str(cast.dtype),
        "conversion": f"{str(tensor.dtype).removeprefix('torch.')}->"
        f"{str(cast.dtype).removeprefix('torch.')}",
        "potentially_lossy": not exact_widening,
    }
