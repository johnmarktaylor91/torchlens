"""Shard codecs: safetensors default behind the manifest field (item 11).

The shard format is a MANIFEST FIELD (extract D1), so a format change is a
value change, never a layout break: ``safetensors`` is the v2 default
(keyed row slicing, third-party readable, bf16/fp8 byte-exact), ``pt``
stays readable forever and writable for one deprecation cycle via the
explicit ``shard_format="pt"`` opt-out.

Dense keys store one batch-major tensor per key. Trimmed keys (D4) store
the Arrow triplet under reserved subkeys — ``<key>/values``,
``<key>/offsets``, ``<key>/shapes`` — a collision-free namespace because
user keys refuse path separators at call time (item 4).

Every spelling here is DOCUMENTED-UNSTABLE pending the naming sprint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .._errors import InvalidArgumentError
from .ragged import RaggedBatch

__tl_layer__ = "L5"

__all__ = [
    "SHARD_FORMATS",
    "read_shard",
    "shard_extension",
    "validate_shard_format",
    "write_shard",
]

#: Closed shard-format vocabulary. The manifest records the value; readers
#: dispatch on the record, never the filename.
SHARD_FORMATS: tuple[str, ...] = ("safetensors", "pt")

#: Reserved trimmed-carrier subkeys (user keys cannot contain "/").
_RAGGED_SUBKEYS = ("values", "offsets", "shapes")


def validate_shard_format(shard_format: Any) -> str:
    """Validate a ``shard_format=`` request against the closed vocabulary.

    Parameters
    ----------
    shard_format:
        Requested format.

    Returns
    -------
    str
        The validated format.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``extraction_shard_format_invalid`` outside the vocabulary.
    """

    if shard_format in SHARD_FORMATS:
        return str(shard_format)
    raise InvalidArgumentError(
        f"shard_format= value {shard_format!r} is not in the closed vocabulary {SHARD_FORMATS}.",
        code="extraction_shard_format_invalid",
        remedy="pass 'safetensors' (default) or 'pt'",
        value=repr(shard_format),
    )


def shard_extension(shard_format: str) -> str:
    """Return the filename extension for one shard format.

    Parameters
    ----------
    shard_format:
        Validated shard format.

    Returns
    -------
    str
        ``".safetensors"`` or ``".pt"``.
    """

    return ".safetensors" if shard_format == "safetensors" else ".pt"


def _shapes_tensor(row_shapes: tuple[tuple[int, ...], ...]) -> torch.Tensor:
    """Encode per-row shapes as one dense int64 tensor.

    Rows of a single carrier share their feature rank; only the leading
    extent varies, so the shape table is rectangular by construction.

    Parameters
    ----------
    row_shapes:
        Per-row logical shapes.

    Returns
    -------
    torch.Tensor
        ``[rows, rank]`` int64 tensor (``[0, 0]``-shaped when empty).
    """

    if not row_shapes:
        return torch.zeros(0, 0, dtype=torch.int64)
    return torch.tensor([list(shape) for shape in row_shapes], dtype=torch.int64)


def write_shard(
    path: Path, payload: dict[str, torch.Tensor | RaggedBatch], shard_format: str
) -> None:
    """Write one shard payload in the artifact's format.

    Parameters
    ----------
    path:
        Destination TEMP path (the commit protocol renames it).
    payload:
        Per-key dense tensors and/or trimmed carriers.
    shard_format:
        Validated shard format.
    """

    if shard_format == "pt":
        flat_pt: dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, RaggedBatch):
                flat_pt[key] = {
                    "values": value.values,
                    "offsets": value.offsets,
                    "shapes": _shapes_tensor(value.row_shapes),
                }
            else:
                flat_pt[key] = value
        torch.save(flat_pt, path)
        return
    from safetensors.torch import save_file

    flat: dict[str, torch.Tensor] = {}
    for key, value in payload.items():
        if isinstance(value, RaggedBatch):
            flat[f"{key}/values"] = value.values.contiguous()
            flat[f"{key}/offsets"] = value.offsets.contiguous()
            flat[f"{key}/shapes"] = _shapes_tensor(value.row_shapes)
        else:
            flat[key] = value.contiguous()
    save_file(flat, str(path))


def _ragged_from_parts(
    values: torch.Tensor, offsets: torch.Tensor, shapes: torch.Tensor
) -> RaggedBatch:
    """Rebuild a trimmed carrier from its stored triplet.

    Parameters
    ----------
    values:
        Packed values.
    offsets:
        Row offsets.
    shapes:
        Dense per-row shape table.

    Returns
    -------
    RaggedBatch
        The carrier.
    """

    row_shapes = tuple(tuple(int(x) for x in row) for row in shapes.tolist())
    return RaggedBatch(values=values, offsets=offsets, row_shapes=row_shapes)


def read_shard(
    path: Path, shard_format: str, keys: list[str] | None = None
) -> dict[str, torch.Tensor | RaggedBatch]:
    """Read one shard payload eagerly (readers dispatch on the manifest).

    Parameters
    ----------
    path:
        Shard file path.
    shard_format:
        The artifact's recorded shard format.
    keys:
        Optional output-key subset.

    Returns
    -------
    dict[str, torch.Tensor | RaggedBatch]
        Per-key payloads with trimmed carriers rebuilt.
    """

    if shard_format == "pt":
        return _read_pt_shard(path, keys)
    return _read_safetensors_shard(path, keys)


def _read_pt_shard(path: Path, keys: list[str] | None) -> dict[str, torch.Tensor | RaggedBatch]:
    """Read one ``.pt`` shard (mmap + weights_only; the 13.4x retrofit).

    Parameters
    ----------
    path:
        Shard file path.
    keys:
        Optional output-key subset.

    Returns
    -------
    dict[str, torch.Tensor | RaggedBatch]
        Per-key payloads.
    """

    raw = torch.load(path, weights_only=True, mmap=True)
    out: dict[str, torch.Tensor | RaggedBatch] = {}
    for key, value in raw.items():
        if keys is not None and key not in keys:
            continue
        if isinstance(value, dict) and set(value) == set(_RAGGED_SUBKEYS):
            out[key] = _ragged_from_parts(value["values"], value["offsets"], value["shapes"])
        else:
            out[key] = value
    return out


def _read_safetensors_shard(
    path: Path, keys: list[str] | None
) -> dict[str, torch.Tensor | RaggedBatch]:
    """Read one safetensors shard, rebuilding trimmed carriers.

    Parameters
    ----------
    path:
        Shard file path.
    keys:
        Optional output-key subset.

    Returns
    -------
    dict[str, torch.Tensor | RaggedBatch]
        Per-key payloads.
    """

    from safetensors.torch import load_file

    flat = load_file(str(path))
    out: dict[str, torch.Tensor | RaggedBatch] = {}
    ragged_parts: dict[str, dict[str, torch.Tensor]] = {}
    for flat_key, tensor in flat.items():
        base, sep, subkey = flat_key.rpartition("/")
        if sep and subkey in _RAGGED_SUBKEYS:
            ragged_parts.setdefault(base, {})[subkey] = tensor
            continue
        if keys is None or flat_key in keys:
            out[flat_key] = tensor
    for base, parts in ragged_parts.items():
        if keys is not None and base not in keys:
            continue
        if set(parts) == set(_RAGGED_SUBKEYS):
            out[base] = _ragged_from_parts(parts["values"], parts["offsets"], parts["shapes"])
    return out
