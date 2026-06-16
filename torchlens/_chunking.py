"""Input chunking helpers for forward-only chunked capture."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from .intervention.errors import BatchChunkInputAmbiguityError, ChunkedForwardConfigError


ChunkPath = tuple[Any, ...]


@dataclass(frozen=True)
class ChunkPlan:
    """Validated plan for splitting positional inputs along batch dimension 0.

    Parameters
    ----------
    chunk_size:
        Requested maximum chunk size.
    total_size:
        Leading batch size shared by all chunked leaves.
    paths:
        Canonical leaf paths selected for splitting.
    """

    chunk_size: int
    total_size: int
    paths: tuple[ChunkPath, ...]


def normalize_chunk_size(chunk_size: int | None) -> int | None:
    """Validate and normalize a public ``chunk_size`` value.

    Parameters
    ----------
    chunk_size:
        User-supplied chunk size.

    Returns
    -------
    int | None
        Normalized positive integer, or ``None`` when chunking is disabled.

    Raises
    ------
    ChunkedForwardConfigError
        If ``chunk_size`` is not a positive integer.
    """

    if chunk_size is None:
        return None
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ChunkedForwardConfigError("chunk_size must be a positive integer.")
    return chunk_size


def normalize_chunk_paths(chunk_paths: Iterable[Any] | None) -> tuple[str, ...] | None:
    """Return a stable public representation for explicit chunk paths.

    Parameters
    ----------
    chunk_paths:
        User-supplied explicit path iterable.

    Returns
    -------
    tuple[str, ...] | None
        String paths in caller order, or ``None`` when auto mode is active.
    """

    if chunk_paths is None:
        return None
    return tuple(str(path) for path in chunk_paths)


def plan_chunks(
    input_args: Any,
    *,
    chunk_size: int,
    chunk_paths: Iterable[Any] | None = None,
) -> ChunkPlan:
    """Build a chunking plan for normalized positional inputs.

    Parameters
    ----------
    input_args:
        Model-ready positional input payload.
    chunk_size:
        Positive maximum chunk size.
    chunk_paths:
        Optional explicit paths naming tensor leaves to split.

    Returns
    -------
    ChunkPlan
        Validated chunking plan.

    Raises
    ------
    BatchChunkInputAmbiguityError
        If auto mode finds multiple batched tensor leaves.
    ChunkedForwardConfigError
        If no splittable tensor is found, or explicit paths are invalid.
    """

    normalized_size = normalize_chunk_size(chunk_size)
    if normalized_size is None:
        raise ChunkedForwardConfigError("chunk_size must be set before planning chunks.")
    roots = _positional_roots(input_args)
    public_paths = normalize_chunk_paths(chunk_paths)
    if public_paths is None:
        tensor_leaves = [
            (path, leaf) for path, leaf in _iter_standard_leaves(roots) if _is_batched_tensor(leaf)
        ]
        if len(tensor_leaves) == 1:
            path, tensor = tensor_leaves[0]
            return ChunkPlan(normalized_size, int(tensor.shape[0]), (path,))
        if len(tensor_leaves) > 1:
            details = ", ".join(
                f"{format_chunk_path(path)} shape={tuple(tensor.shape)}"
                for path, tensor in tensor_leaves
            )
            raise BatchChunkInputAmbiguityError(
                "chunk_size found multiple ndim>0 tensor leaves; pass chunk_paths=[...] "
                f"to select batched inputs explicitly. Candidates: {details}."
            )
        raise ChunkedForwardConfigError(
            "chunk_size requires at least one ndim>0 tensor input leaf."
        )

    parsed_paths = tuple(parse_chunk_path(path) for path in public_paths)
    if not parsed_paths:
        raise ChunkedForwardConfigError("chunk_paths must name at least one tensor leaf.")
    batch_sizes: list[int] = []
    for path in parsed_paths:
        value = _get_path(roots, path)
        if not isinstance(value, torch.Tensor) or value.ndim == 0:
            raise ChunkedForwardConfigError(
                f"chunk path {format_chunk_path(path)} must reference an ndim>0 tensor leaf."
            )
        batch_sizes.append(int(value.shape[0]))
    first_size = batch_sizes[0]
    if any(size != first_size for size in batch_sizes):
        details = ", ".join(
            f"{format_chunk_path(path)} batch={size}"
            for path, size in zip(parsed_paths, batch_sizes, strict=True)
        )
        raise ChunkedForwardConfigError(
            f"all chunk_paths must have identical leading batch size; got {details}."
        )
    return ChunkPlan(normalized_size, first_size, parsed_paths)


def iter_chunked_inputs(input_args: Any, plan: ChunkPlan) -> list[Any]:
    """Return per-chunk input trees for a validated plan.

    Parameters
    ----------
    input_args:
        Model-ready positional input payload.
    plan:
        Validated chunking plan.

    Returns
    -------
    list[Any]
        Chunked input payloads. Containers are rebuilt, and tensor slices are
        views into the original model-ready tensors.
    """

    chunks: list[Any] = []
    for start in range(0, plan.total_size, plan.chunk_size):
        end = min(start + plan.chunk_size, plan.total_size)
        chunks.append(_replace_paths(input_args, plan.paths, start, end))
    return chunks


def format_chunk_path(path: ChunkPath) -> str:
    """Format a canonical chunk path for user-facing messages.

    Parameters
    ----------
    path:
        Canonical path tuple.

    Returns
    -------
    str
        Stable public path string.
    """

    rendered = str(path[0])
    for part in path[1:]:
        if isinstance(part, int):
            rendered += f".{part}"
        else:
            escaped = str(part).replace('"', '\\"')
            rendered += f'["{escaped}"]'
    return rendered


def parse_chunk_path(path: Any) -> ChunkPath:
    """Parse a public chunk path string into a canonical path tuple.

    Parameters
    ----------
    path:
        Public path. Integer values and digit strings address positional roots.
        Nested components may use ``.name``, ``.0``, or ``["key"]`` notation.

    Returns
    -------
    ChunkPath
        Parsed path.

    Raises
    ------
    ChunkedForwardConfigError
        If the path syntax is unsupported.
    """

    if isinstance(path, int) and not isinstance(path, bool):
        return (path,)
    text = str(path)
    match = re.match(r"^\d+", text)
    if match is None:
        raise ChunkedForwardConfigError(
            f"chunk path {text!r} must start with a positional root index like '0'."
        )
    parts: list[Any] = [int(match.group(0))]
    index = match.end()
    while index < len(text):
        if text[index] == ".":
            index += 1
            component_match = re.match(r"[A-Za-z_][A-Za-z0-9_]*|\d+", text[index:])
            if component_match is None:
                raise ChunkedForwardConfigError(f"invalid chunk path syntax: {text!r}.")
            token = component_match.group(0)
            parts.append(int(token) if token.isdigit() else token)
            index += len(token)
            continue
        if text[index] == "[":
            bracket_match = re.match(r"\[(?:'([^']*)'|\"([^\"]*)\"|(\d+))\]", text[index:])
            if bracket_match is None:
                raise ChunkedForwardConfigError(f"invalid chunk path syntax: {text!r}.")
            token = next(group for group in bracket_match.groups() if group is not None)
            parts.append(int(token) if token.isdigit() else token)
            index += len(bracket_match.group(0))
            continue
        raise ChunkedForwardConfigError(f"invalid chunk path syntax: {text!r}.")
    return tuple(parts)


def _is_batched_tensor(value: Any) -> bool:
    """Return whether ``value`` is a tensor leaf with a batch axis."""

    return isinstance(value, torch.Tensor) and value.ndim > 0


def _positional_roots(input_args: Any) -> tuple[Any, ...]:
    """Return positional roots for path lookup.

    Parameters
    ----------
    input_args:
        Model-ready positional input payload.

    Returns
    -------
    tuple[Any, ...]
        Tuple of positional root values.
    """

    if isinstance(input_args, tuple):
        return input_args
    if isinstance(input_args, list):
        return tuple(input_args)
    return (input_args,)


def _iter_standard_leaves(value: Any, prefix: ChunkPath = ()) -> list[tuple[ChunkPath, Any]]:
    """List leaves reachable through standard containers only."""

    if isinstance(value, Mapping):
        leaves: list[tuple[ChunkPath, Any]] = []
        for key, item in value.items():
            leaves.extend(_iter_standard_leaves(item, (*prefix, key)))
        return leaves
    if _is_namedtuple(value):
        leaves = []
        for field_name in value._fields:
            leaves.extend(_iter_standard_leaves(getattr(value, field_name), (*prefix, field_name)))
        return leaves
    if isinstance(value, (list, tuple)):
        leaves = []
        for index, item in enumerate(value):
            leaves.extend(_iter_standard_leaves(item, (*prefix, index)))
        return leaves
    return [(prefix, value)]


def _get_path(roots: tuple[Any, ...], path: ChunkPath) -> Any:
    """Return the value at ``path`` from positional roots."""

    if not path:
        raise ChunkedForwardConfigError("chunk path cannot be empty.")
    root_index = path[0]
    if not isinstance(root_index, int) or root_index < 0 or root_index >= len(roots):
        raise ChunkedForwardConfigError(f"chunk path {format_chunk_path(path)} has invalid root.")
    value = roots[root_index]
    for part in path[1:]:
        value = _get_child(value, part, path)
    return value


def _get_child(value: Any, part: Any, full_path: ChunkPath) -> Any:
    """Return one child from a standard container."""

    try:
        if isinstance(value, Mapping):
            return value[part]
        if _is_namedtuple(value):
            if not isinstance(part, str):
                raise KeyError(part)
            return getattr(value, part)
        if isinstance(value, (list, tuple)):
            if not isinstance(part, int):
                raise KeyError(part)
            return value[part]
    except (KeyError, IndexError, AttributeError) as exc:
        raise ChunkedForwardConfigError(
            f"chunk path {format_chunk_path(full_path)} does not exist."
        ) from exc
    raise ChunkedForwardConfigError(
        f"chunk path {format_chunk_path(full_path)} descends into unsupported object "
        f"{type(value).__name__}; only list, tuple, dict, and namedtuple are supported."
    )


def _replace_paths(input_args: Any, paths: tuple[ChunkPath, ...], start: int, end: int) -> Any:
    """Return ``input_args`` with each selected tensor leaf sliced."""

    if isinstance(input_args, tuple):
        return _replace_in_container(input_args, (), paths, start, end)
    if isinstance(input_args, list):
        return _replace_in_container(input_args, (), paths, start, end)
    replacements = tuple(path[1:] for path in paths if path[0] == 0)
    if len(replacements) != len(paths):
        raise ChunkedForwardConfigError("bare input chunk paths must use positional root 0.")
    return _replace_in_value(input_args, (), replacements, start, end)


def _replace_in_container(
    value: Any,
    prefix: ChunkPath,
    paths: tuple[ChunkPath, ...],
    start: int,
    end: int,
) -> Any:
    """Rebuild a positional container with selected leaves sliced."""

    return _replace_in_value(value, prefix, paths, start, end)


def _replace_in_value(
    value: Any,
    prefix: ChunkPath,
    paths: tuple[ChunkPath, ...],
    start: int,
    end: int,
) -> Any:
    """Return ``value`` with selected descendants sliced."""

    if prefix in paths:
        if not isinstance(value, torch.Tensor):
            raise ChunkedForwardConfigError(
                f"chunk path {format_chunk_path(prefix)} is not a tensor."
            )
        return value[start:end]
    child_paths = tuple(
        path for path in paths if len(path) > len(prefix) and path[: len(prefix)] == prefix
    )
    if not child_paths:
        return value
    if isinstance(value, Mapping):
        rebuilt = {
            key: _replace_in_value(item, (*prefix, key), child_paths, start, end)
            for key, item in value.items()
        }
        return _rebuild_mapping(value, rebuilt)
    if _is_namedtuple(value):
        rebuilt_items = [
            _replace_in_value(getattr(value, field), (*prefix, field), child_paths, start, end)
            for field in value._fields
        ]
        return type(value)(*rebuilt_items)
    if isinstance(value, list):
        return [
            _replace_in_value(item, (*prefix, index), child_paths, start, end)
            for index, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        return tuple(
            _replace_in_value(item, (*prefix, index), child_paths, start, end)
            for index, item in enumerate(value)
        )
    return value


def _is_namedtuple(value: Any) -> bool:
    """Return whether ``value`` is a namedtuple instance."""

    return isinstance(value, tuple) and hasattr(value, "_fields")


def _rebuild_mapping(original: Mapping[Any, Any], values: dict[Any, Any]) -> Mapping[Any, Any]:
    """Rebuild a mapping after replacing selected children.

    Parameters
    ----------
    original:
        Original mapping object.
    values:
        Replacement key-value pairs.

    Returns
    -------
    Mapping[Any, Any]
        Plain dict for plain dict inputs; otherwise the original mapping type
        constructed from the replacement pairs.
    """

    if type(original) is dict:
        return dict(values)
    return type(original)(values)  # type: ignore[call-arg]
