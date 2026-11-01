"""Internal chunked-forward helper functions for public trace capture."""

from __future__ import annotations

import collections.abc
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, NoReturn

from ._data_classes.trace import Trace
from ._deprecations import MISSING
from ._trace_state import TraceState
from .errors._base import TorchLensError
from .intervention.errors import AppendMismatchError, ChunkedForwardConfigError
from .intervention.predicates import InterventionPredicate
from .options import StorageOptions
from .recording.options import HaltPredicateFn


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
