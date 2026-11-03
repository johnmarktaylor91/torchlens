"""Non-strict save: null the refs of blobs the payload codec skipped, and fix the saved counts.

Split out of ``bundle.save`` (which collects the skipped blob ids); imports only
the format contract and the state adapter.
"""

from __future__ import annotations

from typing import Any

from .._data_classes._state_adapter import state_items
from .._utils._record_group_table import record_member
from .format_contract import BlobRef


def _apply_skipped_blobs_to_scrubbed_state(
    scrubbed_state: dict[str, Any],
    skipped_blob_ids: set[str],
) -> None:
    """Replace skipped blob refs with ``None`` in scrubbed metadata.

    Parameters
    ----------
    scrubbed_state:
        Scrubbed metadata dict produced by ``scrub_for_save``.
    skipped_blob_ids:
        Blob ids skipped under ``strict=False``.
    """

    for field_name, field_value in list(scrubbed_state.items()):
        scrubbed_state[field_name] = _replace_skipped_blob_refs(field_value, skipped_blob_ids)

    layer_list = scrubbed_state.get("_layer_list")
    if isinstance(layer_list, list):
        grad_labels = [
            layer.layer_label for layer in layer_list if bool(getattr(layer, "has_grad", False))
        ]
        scrubbed_state["_num_saved_ops"] = sum(
            1 for layer in layer_list if bool(getattr(layer, "has_saved_activation", False))
        )
        scrubbed_state["_cost_saved_activation_memory"] = sum(
            int(record_member(layer, "activation_memory", 0) or 0)
            for layer in layer_list
            if bool(getattr(layer, "has_saved_activation", False))
        )
        scrubbed_state["_autograd_has_grads"] = bool(grad_labels)


def _replace_skipped_blob_refs(value: Any, skipped_blob_ids: set[str]) -> Any:
    """Walk a scrubbed object graph and null out selected blob refs.

    Parameters
    ----------
    value:
        Scrubbed value to inspect.
    skipped_blob_ids:
        Blob ids skipped under ``strict=False``.

    Returns
    -------
    Any
        Value with matching ``BlobRef`` instances replaced by ``None``.
    """

    if isinstance(value, BlobRef):
        if value.blob_id in skipped_blob_ids:
            return None
        return value
    if isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = _replace_skipped_blob_refs(item, skipped_blob_ids)
        return value
    if isinstance(value, tuple):
        return tuple(_replace_skipped_blob_refs(item, skipped_blob_ids) for item in value)
    if isinstance(value, dict):
        for key, item in list(value.items()):
            value[key] = _replace_skipped_blob_refs(item, skipped_blob_ids)
        return value

    spec = getattr(type(value), "_PORTABLE_STATE_SPEC", None)
    if spec is None:
        return value

    for field_name, field_value in list(state_items(value)):
        replaced_value = _replace_skipped_blob_refs(field_value, skipped_blob_ids)
        setattr(value, field_name, replaced_value)
        if (
            field_name == "out"
            and replaced_value is None
            and hasattr(value, "has_saved_activation")
        ):
            value.has_saved_activation = False
        if field_name == "grad" and replaced_value is None and hasattr(value, "has_grad"):
            value.has_grad = False
    return value
