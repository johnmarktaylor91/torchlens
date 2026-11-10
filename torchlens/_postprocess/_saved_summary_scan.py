"""Raw row scan behind the saved-output summary refresh.

``_refresh_fast_saved_summary`` finds the saved ops, sums the physical bytes
of what they retain (each storage once), counts the saved module calls and
walks the autograd graph their payloads keep alive. Through the ``Op`` facade
that costs dozens of counted calls per op. This scan reads the cells of the
building op store directly and asks torch only for what it must: each plain
tensor's storage, its size and its address.

It gives exactly the facade's results. A payload that is not a plain strided
``torch.Tensor``, an op whose cells are not plain, or a trace whose reads of
``Op.out`` carry a side effect (the synthesized-input warning, the
reference-mutation check outside postprocessing) goes through the facade
functions instead.
"""

from __future__ import annotations

from typing import Any

import torch

from .._data_classes._op_column_scan import _building_rows
from .._data_classes._op_out_read import _synthesized_out_read_due
from .._quantities import Bytes
from .._save_budget import _retained_storage_identities, retained_activation_bytes
from .._state import pause_logging
from .._trace_core.cell_states import LazyCell
from .._utils._torch_symbols import tensor_read

__all__ = ("refresh_saved_summary_by_rows",)

_TENSOR = torch.Tensor
_STRIDED = torch.strided
#: Op attributes read here, by the descriptor names the facade functions use.
_ATTRS = ("has_saved_activation", "_graph_is_orphan", "out", "_payload_transformed_out")


def _field_ids(record: Any, layout: Any) -> tuple[int, ...] | None:
    """Return the store field id behind each of ``_ATTRS``, or ``None``."""

    cls = record.__class__
    fids = tuple(getattr(getattr(cls, name, None), "_fid", None) for name in _ATTRS)
    if [fid for fid in fids if fid.__class__ is not int]:
        return None
    member = layout.fid_by_name
    if member.get("is_orphan") != fids[1] or member.get("transformed_out") != fids[3]:
        return None
    return fids


def _scan_target(trace: Any) -> tuple[Any, Any, list[int], tuple[int, ...]] | None:
    """Return ``(owner, store, row keys, field ids)`` when the raw scan applies, else ``None``."""

    scan = _building_rows(trace)
    if scan is None or not trace.__dict__.get("_postprocessing_active", False):
        return None
    if _synthesized_out_read_due(trace):
        return None
    owner, store, keys = scan
    if not keys:
        return owner, store, keys, ()
    if [key for key in keys if key < 0]:
        return None
    fids = _field_ids(owner.resolve(keys[0]), store.layout)
    return None if fids is None else (owner, store, keys, fids)


def _plain_cells(rows: list[Any], fid: int, keys: list[int]) -> list[Any]:
    """Return one column's cells at ``keys``, an immutable default marker read as its value."""

    return [
        cell.source if cell.__class__ is LazyCell and not cell.make else cell
        for cell in [rows[key][fid] for key in keys]
    ]


def _saved_rows(trace: Any) -> tuple[Any, list[int], list[tuple[Any, Any]]] | None:
    """Return ``(owner, saved row keys, (out, transformed_out) per saved op)``, or ``None``.

    ``None`` when the raw scan does not apply and the facade path must run.
    """

    target = _scan_target(trace)
    if target is None or not target[2]:
        return None if target is None else (target[0], [], [])
    owner, store, keys, (f_saved, f_orphan, f_out, f_transformed) = target
    rows = store._rows
    saved = [rows[key][f_saved] for key in keys]
    orphans = _plain_cells(rows, f_orphan, keys)
    if [1 for flag in saved + orphans if flag.__class__ is not bool]:
        return None
    saved_keys = [key for key, flag, orphan in zip(keys, saved, orphans) if flag and not orphan]
    outs = [rows[key][f_out] for key in saved_keys]
    transformed = _plain_cells(rows, f_transformed, saved_keys)
    unusual = [1 for cell in outs + transformed if cell.__class__ is LazyCell]
    if unusual or [1 for cell in outs if cell is None]:
        return None
    refs = [rows[key][store.layout.fid_by_name["_source_trace_ref"]] for key in saved_keys]
    if refs and ([ref for ref in refs if ref is not refs[0]] or refs[0]() is not trace):
        return None
    return owner, saved_keys, list(zip(outs, transformed))


def _other_entries(payload: Any) -> list[tuple[tuple[Any, ...], int]] | None:
    """Return the byte model's storage entries for a payload that is not a plain tensor.

    ``None`` when ``retained_activation_bytes`` skips the payload (not a tensor, or meta).
    """

    if not isinstance(payload, _TENSOR) or payload.is_meta:
        return None
    return _retained_storage_identities(payload)


def _retained_bytes(pairs: list[tuple[Any, Any]], keys: list[int], owner: Any) -> int:
    """Return ``retained_activation_bytes`` of the saved ops at ``keys`` (payloads ``pairs``)."""
    # One flat loop on purpose (complexity): a helper per payload is a counted call per op.
    total, seen, devices, read = 0, set(), {}, None
    with pause_logging():
        for pair in pairs:
            counted = False
            for payload in pair:
                if payload is None:
                    continue
                if (
                    payload.__class__ is not _TENSOR
                    or payload.layout is not _STRIDED
                    or payload.is_meta
                ):
                    entries = _other_entries(payload)
                    if entries is None:
                        continue
                else:
                    read = read or tensor_read(payload, "untyped_storage")
                    try:
                        storage = read(payload)
                        num_bytes, address = storage.nbytes(), storage.data_ptr()
                    except Exception:  # the byte model's own fallback, for this payload alone
                        entries = _retained_storage_identities(payload)
                    else:
                        try:
                            device = devices[payload.device]
                        except KeyError:
                            device = devices[payload.device] = str(payload.device)
                        entries = [((device, address, num_bytes), num_bytes)]
                counted = True
                for identity, size in entries:
                    if identity not in seen:
                        seen.add(identity)
                        total += size if size.__class__ is int else int(size)
            if not counted:
                return retained_activation_bytes(owner.records(keys))
    return total


def _pinned_payloads(pairs: list[tuple[Any, Any]]) -> list[Any]:
    """Return the saved payloads, minus those ``pinned_autograd_bytes`` provably skips."""

    return [
        payload
        for pair in pairs
        for payload in pair
        if payload is not None and (payload.__class__ is not _TENSOR or payload.grad_fn is not None)
    ]


def _count_saved_module_calls(trace: Any, store: Any, keys: list[int], saved: list[int]) -> int:
    """Return ``count_saved_module_calls_from_output_ops`` from raw label cells."""

    rows, fid = store._rows, store.layout.fid_by_name
    f_label, f_layer = fid["label"], fid["layer_label"]
    saved_labels = {rows[key][f_label] for key in saved}
    # ``op_label_aliases`` from raw cells, without a call per op.
    passes: dict[Any, int] = {}
    first: dict[Any, Any] = {}
    for key in keys:
        layer = rows[key][f_layer]
        if layer in passes:
            passes[layer] += 1
        else:
            passes[layer] = 1
            first[layer] = rows[key][f_label]
    aliases = {layer: label for layer, label in first.items() if passes[layer] == 1}
    return sum(
        1
        for module_call in getattr(trace, "module_calls", ())
        if any(
            aliases.get(label, label) in saved_labels
            for label in getattr(module_call, "output_ops", ())
        )
    )


def _plain_labels(store: Any, keys: list[int]) -> bool:
    """Return whether every ``label`` and ``layer_label`` cell at ``keys`` is a ``str``."""

    rows, fid = store._rows, store.layout.fid_by_name
    f_label, f_layer = fid["label"], fid["layer_label"]
    cells = [rows[key][f_label] for key in keys] + [rows[key][f_layer] for key in keys]
    return not [cell for cell in cells if cell.__class__ is not str]


def refresh_saved_summary_by_rows(
    trace: Any, pinned_autograd_bytes: Any, torch_backend: str
) -> bool:
    """Refresh the saved-output summary from raw rows; ``False`` keeps the facade path.

    Parameters
    ----------
    trace:
        Trace whose retained ops are finalized.
    pinned_autograd_bytes:
        The torch backend's autograd walk (passed in: this module stays out
        of the backend package's import cycle).
    torch_backend:
        The torch backend's name.

    Returns
    -------
    bool
        ``True`` when ``_num_saved_ops``, the retained byte total, the saved
        module-call count and the pinned autograd bytes were all refreshed
        here, exactly as ``_refresh_fast_saved_summary`` refreshes them.
    """

    found = _saved_rows(trace)
    if found is None:
        return False
    owner, saved, pairs = found
    store = owner._store
    keys = _building_rows(trace)[2]
    if not _plain_labels(store, keys):
        return False
    trace._num_saved_ops = len(saved)
    total = _retained_bytes(pairs, saved, owner)
    trace._cost_saved_activation_memory = Bytes(total)
    _count_saved_module_calls(trace, store, keys, saved)
    backend_name = str(trace._provenance_backend or torch_backend)
    if backend_name != torch_backend:
        trace._cost_pinned_autograd_memory = None
    else:
        payloads = _pinned_payloads(pairs)
        trace._cost_pinned_autograd_memory = Bytes(pinned_autograd_bytes(payloads))
    return True
