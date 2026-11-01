"""Chunked capture fan-out helpers for the PREPARE stage.

Moved verbatim from ``torchlens._chunked_capture_helpers and torchlens._chunking`` by the capture seam.
"""

from __future__ import annotations

import collections.abc
import re
import time
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, NoReturn

import torch

from ..._data_classes.trace import Trace
from ..._deprecations import MISSING
from ..._trace_state import TraceState
from ...errors._base import TorchLensError
from ...intervention.errors import (
    AmbiguousBatchInputError,
    AppendMismatchError,
    ChunkedForwardConfigError,
)
from ...intervention.predicates import InterventionPredicate
from ...options import StorageOptions
from ...recording.options import HaltPredicateFn


def _should_store_auto_coerced_raw_input(original: Any, coerced: Any) -> bool:
    """Return whether auto-coercion should preserve the original raw input.

    Parameters
    ----------
    original:
        User-provided input before duck-typed coercion.
    coerced:
        Model-ready input returned by ``_coerce_input_args``.

    Returns
    -------
    bool
        ``True`` when a supported non-tensor ergonomic input was converted to a
        different object and should be available for display/save.
    """

    return coerced is not original and _contains_auto_coercible_raw_input(original)


def _refuse_chunk_option_conflict(message: str, alternative: str) -> NoReturn:
    """Raise the typed refusal for an option that cannot compose with ``chunk_size``."""

    raise ChunkedForwardConfigError(
        message,
        code="trace_option_unsupported",
        remedy=f"{alternative}, or run without chunk_size",
    )


def _validate_chunked_forward_capture(
    *,
    input_kwargs: dict[Any, Any] | None,
    backward_ready: bool,
    save_grads: bool,
    hooks: Any | None,
    intervene: InterventionPredicate | None,
    halt: HaltPredicateFn | None,
    streaming: StorageOptions,
) -> None:
    """Reject unsupported ``trace(chunk_size=...)`` option combinations.

    Parameters
    ----------
    input_kwargs:
        Keyword inputs supplied to ``trace`` after preprocessing.
    backward_ready:
        Resolved training-compatible capture flag.
    save_grads:
        Whether gradient capture is enabled.
    hooks:
        Public live hook plan supplied to this capture.
    intervene:
        Public predicate intervention supplied to this capture.
    halt:
        Public halt predicate supplied to this capture.
    streaming:
        Resolved streaming options.

    Raises
    ------
    ChunkedForwardConfigError
        If chunked forward capture cannot compose with the requested options.
    """
    if input_kwargs:
        _refuse_chunk_option_conflict(
            "chunk_size capture is positional-input only in v1; keyword inputs are unsupported.",
            "pass the inputs positionally",
        )
    if backward_ready:
        _refuse_chunk_option_conflict(
            "chunk_size cannot combine with backward_ready=True.", "drop it"
        )
    if save_grads:
        _refuse_chunk_option_conflict(
            "chunk_size cannot be combined with save_grads.", "drop save_grads"
        )
    if hooks is not None:
        _refuse_chunk_option_conflict("chunk_size cannot be combined with hooks=.", "drop hooks=")
    if intervene is not None:
        _refuse_chunk_option_conflict(
            "chunk_size cannot be combined with intervene=.", "drop intervene="
        )
    if halt is not None:
        _refuse_chunk_option_conflict("chunk_size cannot be combined with halt=.", "drop halt=")
    if streaming.path is not None or streaming.on_save is not None:
        _refuse_chunk_option_conflict(
            "chunk_size is in-memory only in v1 and cannot be combined with streaming storage.",
            "drop storage=tl.to_disk(...)",
        )


@contextmanager
def _typed_chunk_forward(chunk_index: int, chunk_size: int, plan: Any) -> Iterator[int]:
    """Run one chunk's capture, yielding its row count; a failure refuses typed.

    ``AppendMismatchError`` from chunk assembly means an activation is not row-wise over the
    batch; any other error is the forward failing on a partial batch. Both become
    ``ChunkedForwardConfigError`` (``trace_option_unsupported``) pointing at a plain capture.
    """

    total_rows = int(plan.total_size)
    chunk_rows = min(chunk_size, total_rows - chunk_index * chunk_size)
    where = f"chunk {chunk_index} ({chunk_rows} of {total_rows} rows)"
    try:
        yield chunk_rows
    except AppendMismatchError as exc:
        raise ChunkedForwardConfigError(
            f"chunk_size: assembling {where} found an activation that is not row-wise "
            f"over the batch ({exc}); this forward is batch-coupled, so a chunked capture "
            "cannot stand for the whole batch.",
            code="trace_option_unsupported",
            remedy="run without chunk_size",
            chunk_index=chunk_index,
        ) from exc
    except TorchLensError:
        raise
    except Exception as exc:  # noqa: BLE001 - the model's own failure on a partial batch
        raise ChunkedForwardConfigError(
            f"chunk_size: the forward failed on {where} with {type(exc).__name__}: {exc}; "
            "a forward that needs the whole batch cannot run in chunks.",
            code="trace_option_unsupported",
            remedy=(
                "run without chunk_size (a forward that also fails on the whole batch "
                "is failing on its own)"
            ),
            chunk_index=chunk_index,
        ) from exc


def _validate_chunk_append_candidate(old_trace: Trace, new_trace: Trace) -> None:
    """Validate chunk append compatibility without reading unsaved payloads.

    Parameters
    ----------
    old_trace:
        Existing accumulated trace.
    new_trace:
        Freshly captured chunk trace.

    Returns
    -------
    None
        Raises when graph metadata is incompatible.
    """

    old_hash = getattr(old_trace, "graph_shape_hash", None)
    new_hash = getattr(new_trace, "graph_shape_hash", None)
    if old_hash != new_hash:
        raise AppendMismatchError("graph shape changed")

    old_labels = tuple(layer._layer_label_raw for layer in old_trace.layer_list)
    new_labels = tuple(layer._layer_label_raw for layer in new_trace.layer_list)
    if old_labels != new_labels:
        raise AppendMismatchError("topology or site labels changed")


@dataclass(frozen=True)
class _ChunkAppend:
    """Bookkeeping for one appended chunk, recorded on the trace's run history.

    Attributes
    ----------
    chunk_size:
        Leading batch size for the appended chunk.
    total_batch_size:
        Total batch size across all chunks.
    append_sequence_id:
        One-based append sequence identifier.
    chunk_paths:
        Normalized public chunk paths.
    """

    chunk_size: int
    total_batch_size: int
    append_sequence_id: int
    chunk_paths: tuple[str, ...] | None


def _append_chunk_trace_state(trace: Trace, new_trace: Trace, chunk: _ChunkAppend) -> None:
    """Append a chunk trace while preserving predicate-save public behavior.

    Parameters
    ----------
    trace:
        Existing accumulated trace to mutate.
    new_trace:
        Freshly captured chunk trace to append.
    chunk:
        The appended chunk's sizes, sequence identifier and chunk paths.

    Returns
    -------
    None
        ``trace`` is updated in place.
    """

    _validate_chunk_append_candidate(trace, new_trace)
    started_at = time.monotonic()
    old_hash = getattr(trace, "graph_shape_hash", None)
    new_hash = getattr(new_trace, "graph_shape_hash", None)
    old_predicate_options = getattr(trace, "_predicate_save_options", MISSING)
    new_predicate_options = getattr(new_trace, "_predicate_save_options", MISSING)
    try:
        trace._predicate_save_options = None
        new_trace._predicate_save_options = None
        trace._append_state_from(new_trace)
    finally:
        if old_predicate_options is MISSING:
            trace.__dict__.pop("_predicate_save_options", None)
        else:
            trace._predicate_save_options = old_predicate_options
        if new_predicate_options is MISSING:
            new_trace.__dict__.pop("_predicate_save_options", None)
        else:
            new_trace._predicate_save_options = new_predicate_options

    trace.is_appended = True
    trace._append_sequence_id = chunk.append_sequence_id
    trace.state = TraceState.APPENDED
    trace._has_direct_writes = False
    trace._out_recipe_revision = getattr(trace, "_spec_revision", 0)
    duration_s = time.monotonic() - started_at
    trace.last_run = {
        "engine": "append",
        "timestamp": time.monotonic(),
        "started_at": started_at,
        "duration_s": duration_s,
        "spec_revision": getattr(trace, "_spec_revision", 0),
        "append": True,
        "strict": False,
        "hooks": 0,
        "chunk_size": chunk.chunk_size,
        "total_batch_size": chunk.total_batch_size,
        "append_sequence_id": chunk.append_sequence_id,
        "old_graph_shape_hash": old_hash,
        "new_graph_shape_hash": new_hash,
        "chunk_paths": chunk.chunk_paths,
    }
    trace.append_history.append(dict(trace.last_run))
    trace._record_operation(
        "append",
        engine="append",
        started_at=started_at,
        duration_s=duration_s,
        hook_count=0,
        chunk_size=chunk.chunk_size,
        total_batch_size=chunk.total_batch_size,
        append_sequence_id=chunk.append_sequence_id,
        old_graph_shape_hash=old_hash,
        new_graph_shape_hash=new_hash,
    )


def _contains_auto_coercible_raw_input(value: Any) -> bool:
    """Return whether ``value`` contains a raw input handled by auto-coercion.

    Parameters
    ----------
    value:
        Candidate user input or nested positional container.

    Returns
    -------
    bool
        ``True`` for text, PIL images, NumPy arrays, or containers containing
        one of those values.
    """

    if isinstance(value, str) or _is_pil_image_value(value) or _is_numpy_array_value(value):
        return True
    if isinstance(value, collections.abc.Mapping):
        return any(_contains_auto_coercible_raw_input(item) for item in value.values())
    if isinstance(value, collections.abc.Sequence) and not isinstance(value, bytes | bytearray):
        return any(_contains_auto_coercible_raw_input(item) for item in value)
    return False


def _is_pil_image_value(value: Any) -> bool:
    """Return whether ``value`` is a PIL image without requiring PIL at import.

    Parameters
    ----------
    value:
        Candidate object.

    Returns
    -------
    bool
        ``True`` when Pillow is installed and ``value`` is a PIL image.
    """

    try:
        from PIL.Image import Image as PILImage
    except ImportError:
        return False
    return isinstance(value, PILImage)


def _is_numpy_array_value(value: Any) -> bool:
    """Return whether ``value`` is a NumPy array without requiring NumPy.

    Parameters
    ----------
    value:
        Candidate object.

    Returns
    -------
    bool
        ``True`` when NumPy is installed and ``value`` is an ndarray.
    """

    try:
        import numpy as np
    except ImportError:
        return False
    return isinstance(value, np.ndarray)


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
    AmbiguousBatchInputError
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
            raise AmbiguousBatchInputError(
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
