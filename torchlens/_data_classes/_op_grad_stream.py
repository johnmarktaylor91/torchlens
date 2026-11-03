"""Grad-blob streaming for ``Op._log_tensor_grad``.

Split out of ``op.py`` (the ``_op_transforms`` precedent): both grad-save
paths of ``Op._log_tensor_grad`` stream the same pair of tensor fields when the
trace defers its streaming-bundle finalization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .op import Op


def stream_grad_blobs(op: Op, trace: Any, writer: Any) -> None:
    """Stream ``op``'s grad and transformed-grad blobs when ``trace`` defers finalization.

    Parameters
    ----------
    op:
        Op whose grad slot was just written.
    trace:
        Trace owning ``op``.
    writer:
        Streaming bundle writer, or ``None`` when the trace does not stream.
    """

    if writer is not None and getattr(trace, "_defer_streaming_bundle_finalization", False):
        op._stream_tensor_blob(
            writer,
            tensor_field="grad",
            pending_field="_pending_grad_blob_id",
            kind="grad",
        )
        op._stream_tensor_blob(
            writer,
            tensor_field="transformed_grad",
            pending_field="_pending_transformed_grad_blob_id",
            kind="transformed_grad",
        )
