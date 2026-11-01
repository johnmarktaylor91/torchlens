"""Forward-time disk streaming of predicate-selected op payloads."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace


def _stream_predicate_payloads(
    trace: Trace,
    fields_dict: dict[str, Any],
    *,
    disk_payload: torch.Tensor | None,
    transformed_disk_payload: torch.Tensor | None,
) -> None:
    """Write predicate-selected disk payloads during forward capture.

    Parameters
    ----------
    trace:
        Active trace whose writer receives payload blobs.
    fields_dict:
        Mutable operation fields receiving pending blob ids.
    disk_payload:
        Detached raw payload for disk storage.
    transformed_disk_payload:
        Detached transformed payload for disk storage.
    """

    writer = getattr(trace, "_out_writer", None)
    if writer is None:
        return
    label = fields_dict["_label_raw"]
    for payload, pending_field, kind in (
        (disk_payload, "_pending_blob_id", "out"),
        (transformed_disk_payload, "_pending_transformed_out_blob_id", "transformed_out"),
    ):
        if payload is None:
            continue
        blob_id = writer.next_blob_id()
        fields_dict[pending_field] = blob_id
        writer.submit_blob(blob_id, payload, kind=kind, label=label)
