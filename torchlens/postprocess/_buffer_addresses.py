"""Buffer-address resolution for postprocess step 0.

Split out of ``_materialize.py`` to keep it under the file-size cap: maps
each buffer op label to the registered buffer address it reads, from the
recorded capture-time address first and value/shape matching against the
model's registered buffers only as a unique fallback, plus the initial
buffer snapshots the join keys on.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from torchlens.ir.events import OpEvent

from ..utils import safe_copy


def _buffer_addresses_by_label(
    buffer_initial_values: Mapping[str, Any],
    buffer_write_events: tuple[Any, ...],
    op_events: list[OpEvent],
) -> dict[str, str]:
    """Join initial registered-buffer addresses to source buffer events.

    Parameters
    ----------
    buffer_initial_values
        The model's declared registered-buffer state universe.
    buffer_write_events
        The journal's buffer-write lane.
    op_events
        Ordered operation events for the capture.

    Returns
    -------
    dict[str, str]
        Buffer addresses keyed by raw buffer source label.
    """

    unmatched_addresses = list((buffer_initial_values or {}).items())
    by_label: dict[str, str] = {}
    source_buffer_events = [
        event for event in op_events if event.kind == "source" and event.layer_type == "buffer"
    ]
    source_buffer_labels = {event.label_raw for event in source_buffer_events}
    op_events_by_label = {event.label_raw: event for event in op_events}
    for event in op_events:
        if event.function.func_name not in {"batch_norm", "batchnorm"}:
            continue
        module_address = event.modules[-1][0] if event.modules else None
        if module_address is None:
            continue
        args_positions = event.parent_arg_positions.get("args", {})
        for arg_position, buffer_name in ((3, "running_mean"), (4, "running_var")):
            label_raw = args_positions.get(arg_position)
            if isinstance(label_raw, str) and label_raw in source_buffer_labels:
                by_label[label_raw] = f"{module_address}.{buffer_name}"

    for write_event in buffer_write_events:
        producer_label_raw = getattr(write_event, "producer_label_raw", None)
        if not isinstance(producer_label_raw, str):
            continue
        producer_event = op_events_by_label.get(producer_label_raw)
        if producer_event is None:
            continue
        address = getattr(write_event, "address", None)
        if not isinstance(address, str) or not address.endswith(".num_batches_tracked"):
            continue
        for edge in producer_event.parents:
            parent_event = op_events_by_label.get(edge.parent_label_raw)
            if (
                parent_event is not None
                and parent_event.kind == "source"
                and parent_event.layer_type == "buffer"
            ):
                by_label.setdefault(edge.parent_label_raw, address)

    used_addresses = set(by_label.values())
    unmatched_addresses = [
        (address, value)
        for address, value in unmatched_addresses
        if address not in used_addresses and not address.endswith(".num_batches_tracked")
    ]
    unmatched_address_names = {address for address, _value in unmatched_addresses}
    for event in source_buffer_events:
        if event.label_raw in by_label:
            continue
        # r83 C2: the address the CAPTURE recorded for this event is
        # AUTHORITATIVE. The value/shape ladder below exists only to fill a
        # genuinely MISSING (anonymous) address; it must never OVERRIDE a
        # recorded one.
        recorded = _recorded_buffer_address(event)
        if recorded is not None:
            # A recorded address that names a REGISTERED buffer is assigned
            # directly (this also fixes the predecessor's suffixed-vs-bare
            # comparison, which never fired for a nested buffer). A recorded
            # address that is NOT a registered buffer is a plain-attribute or
            # list-element constant: the value/shape heuristics must NOT
            # override it with a DIFFERENT registered buffer's address -- that
            # was free's silent wrong-bind (a plain `mean` given `std`'s
            # address by shape, replayed WRONG under VERIFIED). Leaving it
            # unassigned keeps it out of this registered-address map -- so its
            # `backend_address` stays None, as a non-registered buffer's should,
            # and its display address is recovered from the equivalence class
            # downstream (`postprocess.control_flow`), reaching the honest
            # `UNSUPPORTED_TENSOR_CONSTANT` refusal on runnable save. Skipping
            # the ladder here is what both blocks the theft AND preserves the
            # recorded-vs-heuristic `address`/`backend_address` distinction.
            if recorded not in unmatched_address_names:
                continue
            address = recorded
        else:
            address = _unique_buffer_address_by_value(
                event.output.tensor.payload, unmatched_addresses
            )
            if address is None:
                address = _unique_buffer_address_by_shape(
                    event.output.tensor.shape, unmatched_addresses
                )
        if address is not None:
            by_label[event.label_raw] = address
            unmatched_addresses = [
                (candidate_address, value)
                for candidate_address, value in unmatched_addresses
                if candidate_address != address
            ]
            unmatched_address_names.discard(address)
    return by_label


def _recorded_buffer_address(event: OpEvent) -> str | None:
    """Return the buffer address the CAPTURE recorded for a source event (r83 C2).

    Source-buffer events are built with ``equivalence_class =
    f"buffer_{address}"`` plus the canonical module-stack suffix appended by
    ``_append_module_suffix_to_equivalence_class`` (``"_".join`` of the module
    addresses). The suffix is reconstructed exactly from ``event.modules``, so
    the recorded address is recovered without ambiguity and without a schema
    change -- the live tensor's own ``TensorMeta.address`` stamp is NOT usable
    here, because session cleanup strips it before postprocess runs.

    The predecessor of this helper additionally required the decoded address to
    be an unassigned REGISTERED buffer, which silently discarded the recorded
    address for every plain-attribute or list-element buffer source -- and, in
    practice, for every nested buffer too, since it compared the suffixed string
    against bare addresses. Both fell through to the value/shape heuristics.

    Parameters
    ----------
    event
        Source-buffer operation event being resolved.

    Returns
    -------
    str | None
        The recorded address, or ``None`` when the event records none.
    """

    equivalence_class = event.equivalence_class
    if equivalence_class is None or not equivalence_class.startswith("buffer_"):
        return None
    candidate = equivalence_class.removeprefix("buffer_")
    suffix = "_".join(module_pass[0] for module_pass in event.modules)
    if suffix:
        if not candidate.endswith(suffix):
            return None
        candidate = candidate[: -len(suffix)]
    # ``extra_addr=None`` stringifies into the class; that is a missing address.
    if not candidate or candidate == "None":
        return None
    return candidate


def _buffer_alias_snapshots_by_address(
    buffer_write_events: tuple[Any, ...],
    source_model_ref: Any,
) -> dict[str, torch.Tensor]:
    """Return refreshed snapshots for indirectly updated aliased buffers.

    Parameters
    ----------
    buffer_write_events
        The journal's buffer-write lane.
    source_model_ref
        Weakref to the source model (declared input).

    Returns
    -------
    dict[str, torch.Tensor]
        Final snapshots for buffer addresses that were updated only through an alias.
    """

    directly_written = {getattr(event, "address", None) for event in buffer_write_events}
    model = None if source_model_ref is None else source_model_ref()
    if model is None or not hasattr(model, "named_buffers"):
        return {}
    snapshots: dict[str, torch.Tensor] = {}
    for address, tensor in model.named_buffers():
        if address in directly_written or not isinstance(tensor, torch.Tensor):
            continue
        snapshots[address] = safe_copy(tensor, detach_tensor=True)
    return snapshots


def _unique_buffer_address_by_value(
    payload: object,
    candidates: list[tuple[str, object]],
) -> str | None:
    """Return the unique candidate address whose value matches the payload.

    Parameters
    ----------
    payload
        Event payload value.
    candidates
        Candidate buffer address/value pairs.

    Returns
    -------
    str | None
        Unique matching address, otherwise ``None``.
    """

    matches = [address for address, value in candidates if _tensor_values_match(payload, value)]
    return matches[0] if len(matches) == 1 else None


def _unique_buffer_address_by_shape(
    shape: tuple[int, ...] | None,
    candidates: list[tuple[str, object]],
) -> str | None:
    """Return the unique candidate address whose tensor shape matches.

    Parameters
    ----------
    shape
        Event tensor shape.
    candidates
        Candidate buffer address/value pairs.

    Returns
    -------
    str | None
        Unique matching address, otherwise ``None``.
    """

    if shape is None:
        return None
    matches = [
        address
        for address, value in candidates
        if isinstance(value, torch.Tensor) and tuple(value.shape) == shape
    ]
    return matches[0] if len(matches) == 1 else None


def _tensor_values_match(left: object, right: object) -> bool:
    """Return whether two tensor-like values have identical contents.

    Parameters
    ----------
    left
        First value.
    right
        Second value.

    Returns
    -------
    bool
        True when both values are tensors with equal shape, dtype, and values.
    """

    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        return False
    if left.is_meta or right.is_meta:
        # W1-FAB (weightsfree memo D6): value equality is unobservable on the
        # meta substrate (aten::equal has no meta kernel); an UNKNOWN verdict
        # never claims a match — the address ladder simply does not fill.
        return False
    return bool(
        left.shape == right.shape and left.dtype == right.dtype and torch.equal(left, right)
    )
