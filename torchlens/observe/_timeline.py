"""The categorized memory timeline v2 artifact (observe item 11, the flagship).

A typed, categorized, module-contained data artifact over TorchLens's OWN
records -- the product torch's deprecated memory timeline never had. Three
products, kept apart and never conflated: (1) logical bytes produced/recorded
per event by CLOSED category; (2) the persistent logical baseline; (3)
cumulative produced bytes, named as such and NEVER called live memory. What
cannot be observed is named ABSENT with a reason, never zeroed or guessed.

The saved-for-backward band is NOT a stackable category (the round-3
parameter phantom: most of the bytes autograd retains on real models are
PARAMETER storages already counted in the parameter band). Every op event
carries the gross band BESIDE the storage-class decomposition and the
first-save counters; the one honestly stackable saved series is
``newly_saved_activation``, and ``saved_parameter`` renders as an ANNOTATION
on the parameter band. Every spelling here is DOCUMENTED-UNSTABLE pending
naming-session ratification.
"""

from __future__ import annotations

from typing import Any

__all__ = ["SCHEMA_ID", "memory_timeline_v2", "module_rollup"]

SCHEMA_ID = "torchlens.memory_timeline.v2"

#: Closed category vocabulary. ``other`` is never dropped or guessed.
CATEGORIES = (
    "parameter",
    "buffer",
    "input",
    "activation",
    "autograd_saved",
    "op_gradient",
    "parameter_gradient",
    "other",
)

#: Named-absent facts: what this artifact structurally cannot observe, with
#: the reason (never a fabricated zero).
ABSENT = {
    "optimizer_state": (
        "no optimizer was an input to this capture; the training-monitor tier "
        "(R20) populates this category when it lands"
    ),
    "allocator_reserved_bytes": (
        "allocator scope, a different metric kind; CaptureOptions("
        "track_device_memory=True) samples allocator counters separately"
    ),
    "native_workspaces": "cuDNN/cuBLAS workspace bytes are invisible to the wrapper layer",
    "driver_and_nccl": "driver and communicator residency is invisible to the wrapper layer",
    "free_and_destruction_times": (
        "logical records carry production and last-recorded-consumer facts, "
        "never allocator free events; the liveness view is an ESTIMATE"
    ),
}


def _require_trace(trace: Any) -> None:
    """Refuse non-Trace capture products with the metadata-only remedy."""

    from .._errors import InvalidArgumentError

    if type(trace).__name__ == "Recording" or hasattr(trace, "to_trace"):
        raise InvalidArgumentError(
            "the memory timeline needs a full-structure Trace; sparse "
            "Recording/fastlog products do not carry the per-op metadata rows",
            code="timeline_requires_trace",
            remedy=(
                "cook the recording with Recording.to_trace() (a metadata-only "
                "trace suffices: activation_memory is metadata-derived)"
            ),
        )


def _op_rows(trace: Any) -> list[Any]:
    """Return per-pass op records in execution order (multi-pass safe)."""

    ops: list[Any] = []
    for layer in trace.layer_list:
        num_passes = int(getattr(layer, "num_passes", 1) or 1)
        if num_passes > 1:
            inner = getattr(layer, "ops", None)
            if inner is not None and hasattr(inner, "values"):
                ops.extend(inner.values())
                continue
        ops.append(layer)
    ops.sort(
        key=lambda op: (
            int(getattr(op, "step_index", 0) or 0),
            str(getattr(op, "layer_label", "")),
        )
    )
    return ops


def _public_label(op: Any) -> str:
    """Canonical public label through the item-1 convention."""

    from ..capture._nonfinite_prefix import canonical_public_label

    return canonical_public_label(op)


def _owner_and_stack(record: Any) -> tuple[str, tuple[str, ...]]:
    """Return (innermost exclusive owner, ordered module-call stack)."""

    stack = tuple(str(frame) for frame in getattr(record, "module_call_stack", ()) or ())
    if stack:
        return stack[-1], stack
    module_address = getattr(record, "module_address", None) or getattr(
        record, "atomic_module_address", None
    )
    if isinstance(module_address, str) and module_address:
        return module_address, (module_address,)
    return "", ()


def _saved_band_for(trace: Any, op: Any) -> dict[str, Any] | None:
    """Return the op's saved-band decomposition (items 7-8), or None."""

    bands = trace.__dict__.get("_autograd_saved_bands")
    if not isinstance(bands, dict):
        return None
    raw_label = getattr(op, "raw_label", None)
    if raw_label is None:
        return None
    band = bands.get(f"{raw_label}_raw", bands.get(str(raw_label)))
    return dict(band) if isinstance(band, dict) else None


#: The closed event-row column table with its defaults (order = row order).
_EVENT_DEFAULTS: dict[str, Any] = {
    "ordinal": None,
    "phase": "",
    "kind": "",
    "category": "",
    "label": "",
    "bytes": 0,
    "tensor_count": 1,
    "dtype": None,
    "device": None,
    "owner": "",
    "module_stack": (),
    "site": None,
    "alias_of": None,
    "availability": "metadata",
    "producer_ordinal": None,
    "last_consumer_ordinal": None,
}


def _event(**fields: Any) -> dict[str, Any]:
    """Build one timeline event row over the closed column table.

    ``bytes_value`` normalizes to the integer ``bytes`` column and
    ``module_stack`` to a list; every other key must be a declared column
    (typos fail loudly). ``extra`` merges disclosure columns (e.g. the
    saved-band basis). The reserved lifetime slots are always present: a
    lifetime is never presented as an observed allocation/free event.
    """

    extra = fields.pop("extra", None)
    fields["bytes"] = int(fields.pop("bytes_value"))
    fields["module_stack"] = list(fields.pop("module_stack", ()))
    unknown = set(fields) - set(_EVENT_DEFAULTS)
    if unknown:
        # Internal invariant (a typo'd column would silently mint a wrong
        # row); builtin error class, deliberately outside the S-17 census.
        raise KeyError(f"unknown timeline event columns: {sorted(unknown)}")
    row = dict(_EVENT_DEFAULTS)
    row["module_stack"] = []
    row.update(fields)
    row["lifetime_start"] = None
    row["lifetime_end"] = None
    row["lifetime_evidence"] = None
    if extra:
        row.update(extra)
    return row


def _tied_groups(trace: Any) -> tuple[tuple[str, ...], ...]:
    """Return the FactCore tied-parameter groups (C02 identity authority).

    An unavailable FactCore degrades to no tying knowledge, never a crash.
    """

    try:
        from ..report._factcore import factcore

        return tuple(
            tuple(group) for group in factcore(trace).params.tied_groups if len(group) >= 2
        )
    except Exception:  # noqa: BLE001 - identity degradation, never a crash.
        return ()


def _tied_alias_index(trace: Any) -> dict[str, str]:
    """Map each tied parameter address to its group's counted address."""

    alias_of: dict[str, str] = {}
    for group in _tied_groups(trace):
        counted = group[0]
        for member in group[1:]:
            alias_of[member] = counted
    return alias_of


def _capture_fingerprint(trace: Any) -> str | None:
    """Return the FactCore capture fingerprint when derivable."""

    try:
        from ..report._factcore import capture_fingerprint

        return capture_fingerprint(trace)
    except Exception:  # noqa: BLE001 - identity degradation, never a crash.
        return None


def _last_consumer_index(ops: list[Any]) -> dict[str, int]:
    """Map each op label to the max execution ordinal of its recorded consumers."""

    ordinals: dict[str, int] = {}
    for op in ops:
        label = str(getattr(op, "layer_label", "") or "")
        if label:
            ordinals[label] = int(getattr(op, "step_index", 0) or 0)
        full = _public_label(op)
        if full:
            ordinals[full] = int(getattr(op, "step_index", 0) or 0)
    last: dict[str, int] = {}
    for op in ops:
        consumer_ordinal = int(getattr(op, "step_index", 0) or 0)
        for parent in getattr(op, "parents", ()) or ():
            parent_label = str(parent)
            if parent_label in ordinals:
                last[parent_label] = max(last.get(parent_label, -1), consumer_ordinal)
    return last


def _persistent_events(trace: Any) -> tuple[list[dict[str, Any]], int]:
    """Build the persistent-baseline rows (parameters incl. tied aliases, buffers).

    Returns
    -------
    tuple[list[dict[str, Any]], int]
        ``(events, persistent_bytes)`` with each unique parameter storage
        counted once; tied addresses surface as zero-byte alias rows so
        weight tying is visible instead of silently absent.
    """

    alias_of = _tied_alias_index(trace)
    events: list[dict[str, Any]] = []
    persistent_bytes = 0
    emitted_param_addresses: set[str] = set()
    for param in getattr(trace, "params", ()) or ():
        address = str(getattr(param, "address", "") or "")
        emitted_param_addresses.add(address)
        bytes_value = int(getattr(param, "param_memory", 0) or 0)
        counted_alias = alias_of.get(address)
        if counted_alias is not None and counted_alias in emitted_param_addresses:
            bytes_value = 0
        else:
            counted_alias = None
            persistent_bytes += bytes_value
        events.append(
            _event(
                ordinal=None,
                phase="persistent",
                kind="parameter",
                category="parameter",
                label=address,
                bytes_value=bytes_value,
                dtype=str(getattr(param, "dtype", None)),
                owner=str(getattr(param, "module_address", "") or ""),
                module_stack=(str(getattr(param, "module_address", "") or ""),),
                alias_of=counted_alias,
                availability="measured",
            )
        )
    for group in _tied_groups(trace):
        counted = next((member for member in group if member in emitted_param_addresses), group[0])
        for member in group:
            if member == counted or member in emitted_param_addresses:
                continue
            events.append(
                _event(
                    ordinal=None,
                    phase="persistent",
                    kind="parameter",
                    category="parameter",
                    label=member,
                    bytes_value=0,
                    owner=".".join(member.split(".")[:-1]),
                    module_stack=(".".join(member.split(".")[:-1]),),
                    alias_of=counted,
                    availability="measured",
                )
            )
    for buffer_record in getattr(trace, "buffers", ()) or ():
        events.append(_buffer_event(buffer_record))
        persistent_bytes += events[-1]["bytes"]
    return events, persistent_bytes


def _buffer_event(buffer_record: Any) -> dict[str, Any]:
    """Build one persistent buffer row from a Buffer record."""

    address = str(
        getattr(buffer_record, "buffer_address", None)
        or getattr(buffer_record, "address", "")
        or ""
    )
    shape = tuple(getattr(buffer_record, "shape", ()) or ())
    element_size = int(getattr(buffer_record, "element_size", 0) or 0)
    numel = 1
    for dimension in shape:
        numel *= int(dimension)
    bytes_value = int(
        getattr(buffer_record, "buffer_memory", None)
        or getattr(buffer_record, "activation_memory", None)
        or (numel * element_size)
        or 0
    )
    return _event(
        ordinal=None,
        phase="persistent",
        kind="buffer",
        category="buffer",
        label=address,
        bytes_value=bytes_value,
        dtype=str(getattr(buffer_record, "dtype", None)),
        owner=str(getattr(buffer_record, "module_address", "") or ""),
        module_stack=(str(getattr(buffer_record, "module_address", "") or ""),),
        availability="measured",
    )


def _forward_events(
    trace: Any, ops: list[Any], last_consumer: dict[str, int]
) -> tuple[list[dict[str, Any]], int]:
    """Build forward-phase op rows + the non-stackable saved-band rows.

    Output pseudo-rows are EXCLUDED (they re-point at their producers and
    double-count); buffer pseudo-rows ride the persistent buffer band.

    Returns
    -------
    tuple[list[dict[str, Any]], int]
        ``(events, max_forward_ordinal)``.
    """

    events: list[dict[str, Any]] = []
    max_forward_ordinal = 0
    for op in ops:
        layer_type = str(getattr(op, "layer_type", "") or "")
        if layer_type in ("output", "buffer"):
            continue
        ordinal = int(getattr(op, "step_index", 0) or 0)
        max_forward_ordinal = max(max_forward_ordinal, ordinal)
        label = _public_label(op)
        owner, stack = _owner_and_stack(op)
        try:
            site = getattr(op, "site_key", None)
        except Exception:  # noqa: BLE001 - site keys are optional identity.
            site = None
        events.append(
            _event(
                ordinal=ordinal,
                phase="forward",
                kind="op",
                category="input" if layer_type == "input" else "activation",
                label=label,
                bytes_value=int(getattr(op, "activation_memory", 0) or 0),
                dtype=str(getattr(op, "dtype", None)),
                device=str(getattr(op, "device_ref", None) or "") or None,
                owner=owner,
                module_stack=stack,
                site=site if isinstance(site, str) else None,
                availability="metadata",
                producer_ordinal=ordinal,
                last_consumer_ordinal=last_consumer.get(str(getattr(op, "layer_label", "") or "")),
            )
        )
        gross_saved = int(getattr(op, "autograd_memory", 0) or 0)
        if gross_saved:
            band = _saved_band_for(trace, op)
            events.append(
                _event(
                    ordinal=ordinal,
                    phase="forward",
                    kind="op_saved_band",
                    category="autograd_saved",
                    label=label,
                    bytes_value=gross_saved,
                    tensor_count=int(getattr(op, "num_autograd_tensors", 0) or 0),
                    owner=owner,
                    module_stack=stack,
                    availability="measured" if band is not None else "gross_only",
                    extra={
                        "saved_band_basis": (
                            "gross call attribution (per-op dedup only; views "
                            "count in full); NOT a stackable category"
                        ),
                        "decomposition": band,
                    },
                )
            )
    return events, max_forward_ordinal


def _gradient_events(trace: Any, start_ordinal: int) -> tuple[list[dict[str, Any]], str | None]:
    """Build backward-phase gradient rows, partitioned by backward pass.

    Returns
    -------
    tuple[list[dict[str, Any]], str | None]
        ``(events, parameter_gradient_note)`` -- the note names the honest
        absence when no live parameter gradient was readable.
    """

    try:
        backward_passes = list(getattr(trace, "backward_passes", ()) or ())
    except (ValueError, AttributeError):
        backward_passes = []
    if not backward_passes:
        return [], "no backward pass was captured"
    events: list[dict[str, Any]] = []
    gradient_ordinal = start_ordinal
    for backward_pass in backward_passes:
        pass_index = int(getattr(backward_pass, "pass_index", 0) or 0)
        for op in getattr(trace, "saved_grad_ops", ()) or ():
            try:
                grad = op.grad_for(bwd=pass_index)
            except (KeyError, ValueError):
                continue
            if grad is None or not hasattr(grad, "numel"):
                continue
            gradient_ordinal += 1
            owner, stack = _owner_and_stack(op)
            events.append(
                _event(
                    ordinal=gradient_ordinal,
                    phase=f"backward:{pass_index}",
                    kind="op_gradient",
                    category="op_gradient",
                    label=_public_label(op),
                    bytes_value=int(grad.numel() * grad.element_size()),
                    dtype=str(grad.dtype),
                    device=str(grad.device),
                    owner=owner,
                    module_stack=stack,
                    availability="measured",
                )
            )
    param_rows = _parameter_gradient_events(trace, gradient_ordinal, len(backward_passes))
    events.extend(param_rows)
    if not param_rows:
        return events, (
            "no live parameter gradients were readable (Param records hold "
            "live refs; loaded traces carry none)"
        )
    return events, None


def _parameter_gradient_events(
    trace: Any, start_ordinal: int, final_pass: int
) -> list[dict[str, Any]]:
    """Build parameter-gradient rows from readable live Param refs."""

    events: list[dict[str, Any]] = []
    gradient_ordinal = start_ordinal
    for param in getattr(trace, "params", ()) or ():
        try:
            grad = param.grad
        except Exception:  # noqa: BLE001 - live-ref resolution is best-effort.
            grad = None
        if grad is None or not hasattr(grad, "numel"):
            continue
        gradient_ordinal += 1
        events.append(
            _event(
                ordinal=gradient_ordinal,
                phase=f"backward:{final_pass}",
                kind="parameter_gradient",
                category="parameter_gradient",
                label=str(getattr(param, "address", "") or ""),
                bytes_value=int(grad.numel() * grad.element_size()),
                dtype=str(grad.dtype),
                device=str(grad.device),
                owner=str(getattr(param, "module_address", "") or ""),
                availability="measured",
            )
        )
    return events


def _saved_band_summary(
    events: list[dict[str, Any]], *, gross_total: int, available: bool
) -> dict[str, Any]:
    """Summarize the saved band: gross beside decomposition, never stacked."""

    newly_saved_activation = 0
    newly_saved_total = 0
    saved_parameter_total = 0
    for row in events:
        band = row.get("decomposition")
        if isinstance(band, dict):
            newly_saved_activation += int(band.get("newly_saved_activation", 0))
            newly_saved_total += int(band.get("newly_saved_bytes", 0))
            saved_parameter_total += int(band.get("saved_parameter", 0))
    return {
        "basis": "gross per-op call attribution; not a stackable category",
        "gross_total": gross_total,
        "decomposition_available": available,
        "decomposition_absent_reason": (
            None
            if available
            else (
                "saved-band decomposition is live-capture bookkeeping; this "
                "trace (loaded, or captured without an autograd walk) "
                "carries none"
            )
        ),
        "saved_parameter_total": saved_parameter_total if available else None,
        "newly_saved_total": newly_saved_total if available else None,
        "stackable_newly_saved_activation": (newly_saved_activation if available else None),
        "saved_parameter_presentation": (
            "annotation on the parameter band -- real bytes autograd "
            "re-holds, but no new resident bytes; stacking them double-plots"
        ),
    }


def memory_timeline_v2(trace: Any) -> dict[str, Any]:
    """Build the ``torchlens.memory_timeline.v2`` artifact for one trace.

    Works on metadata-only and selective-save traces (``activation_memory``
    is metadata-derived). The full JSON stays unbinned and O(records);
    binning is the RENDERER's disclosed concern.

    Parameters
    ----------
    trace:
        Completed TorchLens trace (loaded artifacts welcome; live-only facts
        such as the saved-band decomposition disclose honest absence).

    Returns
    -------
    dict[str, Any]
        JSON-serializable artifact: identity header, closed categories,
        named-absent facts, per-event rows, and the three kept-apart
        products (per-event logical bytes, the persistent baseline,
        cumulative produced bytes).

    Raises
    ------
    InvalidArgumentError
        For sparse ``Recording`` inputs (code ``timeline_requires_trace``).
    """

    from .._capture_honesty import capture_honesty_facts

    _require_trace(trace)
    ops = _op_rows(trace)
    saved_band_available = isinstance(trace.__dict__.get("_autograd_saved_bands"), dict)
    events, persistent_bytes = _persistent_events(trace)
    forward_rows, max_forward_ordinal = _forward_events(trace, ops, _last_consumer_index(ops))
    events.extend(forward_rows)
    gradient_rows, parameter_gradient_note = _gradient_events(trace, max_forward_ordinal)
    events.extend(gradient_rows)

    category_totals = dict.fromkeys(CATEGORIES, 0)
    for row in events:
        category_totals[row["category"]] += row["bytes"]

    return {
        "schema": SCHEMA_ID,
        "scope": "logical",
        "disclaimer": "logical recorded tensor bytes; not allocator residency",
        "capture_fingerprint": _capture_fingerprint(trace),
        "trace_id": getattr(trace, "trace_id", None),
        "step": None,
        "torchlens_capture_honesty": capture_honesty_facts(trace),
        "categories": list(CATEGORIES),
        "absent": dict(ABSENT),
        "persistent_baseline_bytes": persistent_bytes,
        "cumulative_produced_bytes": sum(
            row["bytes"] for row in events if row["category"] in ("input", "activation")
        ),
        "category_totals": category_totals,
        "saved_band": _saved_band_summary(
            events,
            gross_total=category_totals["autograd_saved"],
            available=saved_band_available,
        ),
        "parameter_gradient_note": parameter_gradient_note,
        "events": events,
    }


def module_rollup(artifact: dict[str, Any], *, depth: int = 1) -> dict[str, Any]:
    """Roll timeline events up to module owners at one declared depth.

    Ownership is EXCLUSIVE at the displayed depth (each event charges exactly
    one owner: its innermost recorded module truncated to ``depth``); parent
    totals are therefore INCLUSIVE views by construction and the root total
    equals the sum of the exclusive rows -- the conservation law the tests
    pin.

    Parameters
    ----------
    artifact:
        A ``torchlens.memory_timeline.v2`` artifact.
    depth:
        Module-address depth (dot components) to display.

    Returns
    -------
    dict[str, Any]
        ``{"depth", "rows": {owner: {category: bytes}}, "total_bytes"}``.
    """

    rows: dict[str, dict[str, int]] = {}
    total = 0
    for row in artifact["events"]:
        owner = str(row.get("owner", "") or "")
        # Module-call frames spell pass-qualified calls ("block:2"); the
        # rollup aggregates by module ADDRESS so a reused module's calls land
        # in one row (containment stays per-call in the event rows).
        components = [component.split(":", 1)[0] for component in owner.split(".") if component]
        truncated = ".".join(components[:depth]) if components else ""
        bucket = rows.setdefault(truncated, dict.fromkeys(CATEGORIES, 0))
        bucket[row["category"]] += row["bytes"]
        total += row["bytes"]
    return {"depth": depth, "rows": rows, "total_bytes": total}
