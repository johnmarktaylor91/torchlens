"""Internal chunked-forward helper functions for public trace capture."""

from __future__ import annotations

from torchlens._capture.prepare.chunked import (
    _append_chunk_trace_state as _append_chunk_trace_state,
    _ChunkAppend as _ChunkAppend,
    _contains_auto_coercible_raw_input as _contains_auto_coercible_raw_input,
    _is_numpy_array_value as _is_numpy_array_value,
    _is_pil_image_value as _is_pil_image_value,
    _should_store_auto_coerced_raw_input as _should_store_auto_coerced_raw_input,
    _validate_chunk_append_candidate as _validate_chunk_append_candidate,
    _validate_chunked_forward_capture as _validate_chunked_forward_capture,
)
