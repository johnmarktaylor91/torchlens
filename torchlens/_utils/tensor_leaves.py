"""Tensor-leaf helpers: walk a nest for tensors and read a tensor's raw payload.

Torch-only and dependency-free within the package, so digest, identity and
mutation code can import these without pulling the full tensor utilities.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any, cast

import torch


def tensor_payload_bytes(tensor: torch.Tensor) -> bytes:
    """Return a tensor's raw contiguous payload bytes, for ANY dense dtype.

    ``tensor.numpy()`` refuses dtypes NumPy lacks (bf16, the fp8 family)
    with a raw ``TypeError`` -- the crash bf16 stimuli hit in the stimulus
    digest, the envelope input digest, and the callable-identity fold.
    Reinterpreting the flat buffer as ``uint8`` never routes
    the element type through NumPy.

    Parameters
    ----------
    tensor:
        Any dense tensor (moved to CPU here; 0-dim and empty safe).

    Returns
    -------
    bytes
        The flattened contiguous payload bytes.
    """

    flat = tensor.detach().reshape(-1).cpu().contiguous()
    if flat.numel() == 0:
        return b""
    return flat.view(torch.uint8).numpy().tobytes()


def iter_tensors(
    value: Any,
    *,
    sort_keys: bool = False,
    mapping_types: tuple[type, ...] = (dict,),
    max_depth: int | None = None,
) -> Iterator[torch.Tensor]:
    """Yield every tensor in a nest of lists, tuples and mappings.

    The one container walker for tensor leaves. Lists and tuples (namedtuples
    included) are walked in order; mappings in insertion order, or by
    ``repr`` of their keys with ``sort_keys=True`` for digests that must not
    depend on insertion order. Any other value, a dataclass included, is a
    leaf and yields nothing unless it is a tensor.

    Parameters
    ----------
    value:
        Root of the nest.
    sort_keys:
        Visit mapping values in ``repr``-sorted key order.
    mapping_types:
        Types walked as mappings (``isinstance``); ``(dict,)`` by default.
    max_depth:
        Containers at this depth or deeper are not entered (a tensor at any
        depth is still yielded); ``None`` for no bound.

    Yields
    ------
    torch.Tensor
        Tensor leaves in traversal order.
    """

    yield from _iter_tensors_at(value, 0, sort_keys, mapping_types, max_depth)


def _iter_tensors_at(
    value: Any,
    depth: int,
    sort_keys: bool,
    mapping_types: tuple[type, ...],
    max_depth: int | None,
) -> Iterator[torch.Tensor]:
    """Recursive step of :func:`iter_tensors` (positional for the hot path)."""

    if isinstance(value, torch.Tensor):
        yield value
        return
    if max_depth is not None and depth >= max_depth:
        return
    if isinstance(value, (list, tuple)):
        children: Iterable[Any] = value
    elif isinstance(value, mapping_types):
        mapping = cast("Mapping[Any, Any]", value)
        keys = sorted(mapping, key=repr) if sort_keys else list(mapping)
        children = [mapping[key] for key in keys]
    else:
        return
    for child in children:
        yield from _iter_tensors_at(child, depth + 1, sort_keys, mapping_types, max_depth)
