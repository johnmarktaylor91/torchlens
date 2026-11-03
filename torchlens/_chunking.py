"""Input chunking helpers for forward-only chunked capture."""

from __future__ import annotations

from torchlens._capture.prepare.chunked import (
    ChunkPath as ChunkPath,
    ChunkPlan as ChunkPlan,
    _get_child as _get_child,
    _get_path as _get_path,
    _is_batched_tensor as _is_batched_tensor,
    _is_namedtuple as _is_namedtuple,
    _iter_standard_leaves as _iter_standard_leaves,
    _positional_roots as _positional_roots,
    _rebuild_mapping as _rebuild_mapping,
    _replace_in_container as _replace_in_container,
    _replace_in_value as _replace_in_value,
    _replace_paths as _replace_paths,
    format_chunk_path as format_chunk_path,
    iter_chunked_inputs as iter_chunked_inputs,
    normalize_chunk_paths as normalize_chunk_paths,
    normalize_chunk_size as normalize_chunk_size,
    parse_chunk_path as parse_chunk_path,
    plan_chunks as plan_chunks,
)
