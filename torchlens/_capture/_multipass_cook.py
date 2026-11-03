"""Fold the sealed cores of a multi-pass Recording into one cook-ready stream.

Every ``Recorder`` pass is sealed as its own run core with pass-local raw
labels (``linear_1_2_raw`` in each pass), module call indexes restarting at
1 and freshly minted parameter barcodes. ``tl.trace`` of a model that calls
the same submodule once per input numbers those calls 1, 2, 3 in one label
space and loop grouping then folds them into one layer with per-pass ops.
This module renumbers every later pass into that single label space, so
``Recording.to_trace()`` on a multi-pass Recording yields the Trace
``tl.trace`` gives for the same sequence of calls: ``linear_1_1:1``,
``linear_1_1:2``, ... with module passes ``lin:1``, ``lin:2``, ...

The first pass is used as it was sealed; only later passes are rewritten.
A pass whose records cannot be renumbered faithfully is refused with the
typed ``recording_multipass_not_convertible`` error naming the shape.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from .._errors import CaptureNotActiveError
from .._ir.op_record import OpRecord
from .._ir.predicate import ModuleStackFrame, RecordContext

if TYPE_CHECKING:
    from .session import CapturedRunCore

__all__ = ("FoldedPasses", "fold_pass_cores")

_BOUNDARY_TYPES = frozenset({"module_enter", "module_exit"})
_SOURCE_TYPES = frozenset({"input", "buffer"})
_CODE = "recording_multipass_not_convertible"


@dataclass(frozen=True, slots=True)
class _PassOffsets:
    """What a later pass adds to each of its pass-local counters."""

    raw_index: int
    step_index: int
    seq: int
    func_call_id: int
    type_index: Mapping[str, int]
    module_calls: Mapping[str, int]


@dataclass(slots=True)
class FoldedPasses:
    """One label space over every pass of a multi-pass Recording."""

    events: list[Any] = field(default_factory=list)
    call_site_classes: dict[str, str] = field(default_factory=dict)
    output_tensors: list[Any] = field(default_factory=list)
    output_tensor_addresses: list[str] = field(default_factory=list)
    output_labels: list[str] = field(default_factory=list)
    layer_counter: int = 0


def _unconvertible(shape: str, *, code: str = _CODE) -> CaptureNotActiveError:
    """The typed refusal for a multi-pass shape the fold cannot convert."""

    return CaptureNotActiveError(
        f"Recording.to_trace() cannot convert this multi-pass Recording: {shape}",
        code=code,
        remedy="record one pass per Recording before converting",
    )


def fold_pass_cores(
    cores: Sequence[CapturedRunCore],
    cloned_events: Sequence[Sequence[Any]],
) -> FoldedPasses:
    """Renumber every pass after the first into the first pass's label space.

    Parameters
    ----------
    cores
        Sealed cores in Recorder pass order.
    cloned_events
        Per-core replay clones of ``core.events`` (independent mutable dicts).

    Returns
    -------
    FoldedPasses
        The relabeled event stream and the merged run facts a cook installs.
    """

    folded = FoldedPasses()
    canonical_barcodes = _canonical_barcodes(cores[-1])
    totals = _Totals()
    for pass_number, (core, events) in enumerate(zip(cores, cloned_events, strict=True), 1):
        _check_pass_convertible(core, pass_number)
        if pass_number == 1:
            relabeled = list(events)
            label_map = {event.label_raw: event.label_raw for event in events}
        else:
            offsets = totals.offsets()
            label_map = _label_map(events, offsets, pass_number)
            barcode_map = _barcode_map(events, canonical_barcodes)
            relabeled = [
                _relabel_record(event, offsets, label_map, barcode_map) for event in events
            ]
        folded.events.extend(relabeled)
        _merge_pass_facts(folded, core, label_map, pass_number)
        totals.absorb(relabeled, core)
    folded.layer_counter = max(totals.layer_counter, totals.raw_index)
    return folded


class _Totals:
    """Running maxima of the pass-local counters already placed in the label space."""

    def __init__(self) -> None:
        self.raw_index = 0
        self.step_index = 0
        self.seq = 0
        self.func_call_id = 0
        self.layer_counter = 0
        self.type_index: dict[str, int] = {}
        self.module_calls: dict[str, int] = {}

    def offsets(self) -> _PassOffsets:
        return _PassOffsets(
            raw_index=self.raw_index,
            step_index=self.step_index,
            seq=self.seq,
            func_call_id=self.func_call_id,
            type_index=dict(self.type_index),
            module_calls=dict(self.module_calls),
        )

    def absorb(self, events: Iterable[Any], core: CapturedRunCore) -> None:
        self.layer_counter += int(core.projection_facts.get("layer_counter", 0) or 0)
        for event in events:
            self.raw_index = max(self.raw_index, int(event.raw_index or 0))
            self.step_index = max(self.step_index, int(event.step_index or 0))
            self.seq = max(self.seq, int(getattr(event, "seq", 0) or 0))
            if event.func_call_id is not None:
                self.func_call_id = max(self.func_call_id, int(event.func_call_id))
            self._absorb_counts(event)

    def _absorb_counts(self, event: Any) -> None:
        layer_type = event.layer_type
        if layer_type in _BOUNDARY_TYPES:
            address, _kind, call_index = _split_boundary_label(event.label_raw)
            self.module_calls[address] = max(self.module_calls.get(address, 0), call_index)
            return
        self.type_index[layer_type] = max(
            self.type_index.get(layer_type, 0), int(event.type_index or 0)
        )
        for address, call_index in event.modules:
            self.module_calls[address] = max(self.module_calls.get(address, 0), int(call_index))


def _check_pass_convertible(core: CapturedRunCore, pass_number: int) -> None:
    """Refuse a pass whose post-seal knowledge the fold cannot carry."""

    journal = core.projection_facts.get("capture_events")
    watermark = core.amendment_watermark
    if journal is None or pass_number == 1 or watermark is None:
        return
    late = [amendment for amendment in journal.op_amendments if amendment.seq > watermark]
    if late:
        raise _unconvertible(
            f"pass {pass_number} carries {len(late)} post-seal amendment(s) "
            f"({sorted({amendment.family for amendment in late})}) the fold cannot renumber",
            code=_CODE,
        )


def _canonical_barcodes(last_core: CapturedRunCore) -> dict[str, str]:
    """Return ``parameter address -> barcode`` of the pass whose Param logs the cook installs."""

    param_logs = last_core.projection_facts.get("param_logs") or {}
    canonical: dict[str, str] = {}
    for address, param_log in param_logs.items():
        barcode = getattr(param_log, "barcode", None)
        if isinstance(barcode, str):
            canonical[str(address)] = barcode
    return canonical


def _barcode_map(events: Iterable[Any], canonical: Mapping[str, str]) -> dict[str, str]:
    """Map this pass's parameter barcodes onto the canonical ones by address."""

    mapping: dict[str, str] = {}
    for event in events:
        for ref in event.params:
            target = canonical.get(ref.address)
            if target is not None and target != ref.barcode:
                mapping[ref.barcode] = target
    return mapping


def _split_boundary_label(label: str) -> tuple[str, str, int]:
    """Split ``<address>:enter:<n>`` into its parts, refusing any other spelling."""

    parts = label.rsplit(":", 2)
    if len(parts) != 3 or parts[1] not in ("enter", "exit") or not parts[2].isdigit():
        raise _unconvertible(
            f"module boundary label {label!r} is not <address>:enter|exit:<n>", code=_CODE
        )
    return parts[0], parts[1], int(parts[2])


def _relabeled(event: Any, offsets: _PassOffsets) -> tuple[str, int, int, int]:
    """Return the new ``(label_raw, raw_index, type_index, step_index)`` of one record."""

    layer_type = event.layer_type
    if layer_type in _BOUNDARY_TYPES:
        address, kind, call_index = _split_boundary_label(event.label_raw)
        new_call = call_index + offsets.module_calls.get(address, 0)
        return f"{address}:{kind}:{new_call}", event.raw_index, event.type_index, event.step_index
    type_offset = offsets.type_index.get(layer_type, 0)
    new_type_index = int(event.type_index) + type_offset
    new_raw_index = int(event.raw_index) + offsets.raw_index
    new_step_index = int(event.step_index or 0) + offsets.step_index
    if layer_type in _SOURCE_TYPES or event.kind == "source":
        expected = f"{layer_type}_{event.type_index}_raw"
        new_label = f"{layer_type}_{new_type_index}_raw"
    else:
        expected = f"{layer_type}_{event.type_index}_{event.raw_index}_raw"
        new_label = f"{layer_type}_{new_type_index}_{new_raw_index}_raw"
    if event.label_raw != expected:
        raise _unconvertible(
            f"record {event.label_raw!r} does not follow the {layer_type} label spelling "
            f"{expected!r}, so its pass cannot be renumbered",
            code=_CODE,
        )
    return new_label, new_raw_index, new_type_index, new_step_index


def _label_map(events: Sequence[Any], offsets: _PassOffsets, pass_number: int) -> dict[str, str]:
    """Return ``old raw label -> new raw label`` for every record of one pass."""

    mapping: dict[str, str] = {}
    for event in events:
        new_label, _raw, _type, _step = _relabeled(event, offsets)
        if event.label_raw in mapping:
            raise _unconvertible(
                f"pass {pass_number} records the raw label {event.label_raw!r} twice", code=_CODE
            )
        mapping[event.label_raw] = new_label
    return mapping


def _mapped_label(label: str, label_map: Mapping[str, str]) -> str:
    """Translate a within-pass reference, refusing one that names no record of the pass."""

    try:
        return label_map[label]
    except KeyError:
        raise _unconvertible(
            f"a record refers to {label!r}, which no record of its pass carries", code=_CODE
        ) from None


def _relabel_record(
    event: Any,
    offsets: _PassOffsets,
    label_map: Mapping[str, str],
    barcode_map: Mapping[str, str],
) -> Any:
    """Return one later-pass record rewritten into the first pass's label space."""

    if not isinstance(event, OpRecord):
        raise _unconvertible(
            f"record {event.label_raw!r} is a {type(event).__name__}, not an OpRecord", code=_CODE
        )
    new_label, raw_index, type_index, step_index = _relabeled(event, offsets)
    modules = tuple(
        (address, int(call_index) + offsets.module_calls.get(address, 0))
        for address, call_index in event.modules
    )
    core = replace(
        event.core,
        seq=int(event.core.seq or 0) + offsets.seq,
        label_raw=new_label,
        layer_label_raw=_mapped_label(event.core.layer_label_raw, label_map),
        raw_index=raw_index,
        type_index=type_index,
        step_index=step_index,
        pass_index=1,
        parents=tuple(
            replace(edge, parent_label_raw=_mapped_label(edge.parent_label_raw, label_map))
            for edge in event.core.parents
        ),
        func_call_id=_shifted(event.core.func_call_id, offsets.func_call_id),
    )
    return replace(
        event,
        core=core,
        function=_relabeled_function(event.function, offsets),
        graph=_relabeled_graph(event.graph, label_map, barcode_map),
        modules_facet=_relabeled_modules(event.modules_facet, modules, offsets),
        params_facet=_relabeled_params(event.params_facet, barcode_map),
        recording=_relabeled_recording(event.recording, offsets, label_map),
    )


def _shifted(value: int | None, offset: int) -> int | None:
    return None if value is None else int(value) + offset


def _relabeled_function(function: Any, offsets: _PassOffsets) -> Any:
    if function is None or function.func_call_id is None:
        return function
    return replace(function, func_call_id=int(function.func_call_id) + offsets.func_call_id)


def _relabeled_graph(
    graph: Any, label_map: Mapping[str, str], barcode_map: Mapping[str, str]
) -> Any:
    if graph is None:
        return None
    positions = {
        _mapped_label(parent, label_map): dict(slots)
        for parent, slots in graph.parent_arg_positions.items()
    }
    return replace(
        graph,
        parent_arg_positions=positions,
        equivalence_class=_canonical_class(graph.equivalence_class, barcode_map),
    )


def _canonical_class(equivalence_class: str | None, barcode_map: Mapping[str, str]) -> str | None:
    """Rewrite a parameter fingerprint onto the canonical barcodes of its parameters."""

    if equivalence_class is None:
        return None
    rewritten = equivalence_class
    for old, new in barcode_map.items():
        if old in rewritten:
            rewritten = rewritten.replace(old, new)
    return rewritten


def _relabeled_modules(
    facet: Any, modules: tuple[tuple[str, int], ...], offsets: _PassOffsets
) -> Any:
    if facet is None:
        return None
    stack = tuple(
        frame
        if frame.call_index is None
        else replace(
            frame, call_index=int(frame.call_index) + offsets.module_calls.get(frame.address, 0)
        )
        for frame in facet.module_stack
    )
    return replace(facet, module_stack=stack, modules=modules)


def _relabeled_params(facet: Any, barcode_map: Mapping[str, str]) -> Any:
    if facet is None or not barcode_map:
        return facet
    params = tuple(
        replace(ref, barcode=barcode_map.get(ref.barcode, ref.barcode)) for ref in facet.params
    )
    return replace(facet, params=params)


def _relabeled_recording(facet: Any, offsets: _PassOffsets, label_map: Mapping[str, str]) -> Any:
    if facet is None or not isinstance(facet.record_context, RecordContext):
        return facet
    return replace(
        facet, record_context=_relabeled_context(facet.record_context, offsets, label_map)
    )


def _relabeled_context(
    ctx: RecordContext, offsets: _PassOffsets, label_map: Mapping[str, str]
) -> RecordContext:
    """Rewrite one predicate record context (and its lookback) into the merged label space.

    ``pass_number`` keeps the Recorder pass it was recorded in: that is the
    context's own provenance within the multi-pass Recording.
    """

    raw_label = ctx.raw_label if ctx.raw_label is not None else ctx.label_raw or None
    new_raw = label_map.get(raw_label) if raw_label else None
    if new_raw is None:
        # A lookback context from outside this pass's stream keeps its own spelling.
        return ctx
    label, prefix = _relabeled_context_labels(ctx, new_raw, offsets)
    innermost = _innermost_address(ctx)
    return replace(
        ctx,
        label=label,
        raw_label=new_raw if ctx.raw_label is not None else ctx.raw_label,
        label_raw=new_raw if ctx.label_raw else ctx.label_raw,
        label_prefix=prefix,
        step_index=_shifted(ctx.step_index, offsets.step_index),
        type_index=_shifted_type_index(ctx, offsets),
        raw_index=_shifted(ctx.raw_index, offsets.raw_index),
        func_call_id=_shifted(ctx.func_call_id, offsets.func_call_id),
        module_pass_number=_shifted(ctx.module_pass_number, offsets.module_calls.get(innermost, 0)),
        module_stack=tuple(_relabeled_frame(frame, offsets) for frame in ctx.module_stack),
        parent_labels=tuple(label_map.get(parent, parent) for parent in ctx.parent_labels),
        parent_labels_raw=tuple(label_map.get(parent, parent) for parent in ctx.parent_labels_raw),
        recent_events=tuple(
            _relabeled_context(recent, offsets, label_map) for recent in ctx.recent_events
        ),
        recent_ops=tuple(
            _relabeled_context(recent, offsets, label_map) for recent in ctx.recent_ops
        ),
    )


def _relabeled_context_labels(
    ctx: RecordContext, new_raw: str, offsets: _PassOffsets
) -> tuple[str, str]:
    """Return the renumbered in-flight label and prefix alias of one record context."""

    label = ctx.label
    if label == ctx.raw_label or label == ctx.label_raw:
        label = new_raw
    elif label == ctx.label_prefix and ctx.type_index is not None and ctx.layer_type:
        type_index = int(ctx.type_index) + offsets.type_index.get(ctx.layer_type, 0)
        label = f"{ctx.layer_type}_{type_index}"
    prefix = label.rsplit("_", 2)[0] if ctx.label_prefix else ctx.label_prefix
    return label, prefix


def _shifted_type_index(ctx: RecordContext, offsets: _PassOffsets) -> int | None:
    if ctx.type_index is None or ctx.layer_type is None:
        return ctx.type_index
    return int(ctx.type_index) + offsets.type_index.get(ctx.layer_type, 0)


def _innermost_address(ctx: RecordContext) -> str:
    for frame in reversed(ctx.module_stack):
        if isinstance(frame, ModuleStackFrame) and frame.address:
            return frame.address
    return ctx.address or ""


def _relabeled_frame(frame: Any, offsets: _PassOffsets) -> Any:
    if not isinstance(frame, ModuleStackFrame):
        return frame
    key = frame.address if frame.address else "root"
    return replace(frame, pass_number=frame.pass_number + offsets.module_calls.get(key, 0))


def _merge_pass_facts(
    folded: FoldedPasses,
    core: CapturedRunCore,
    label_map: Mapping[str, str],
    pass_number: int,
) -> None:
    """Add one pass's call-site classes and outputs, translated, to the fold."""

    facts = core.projection_facts
    classes = facts.get("call_site_classes") or {}
    for old_label, new_label in label_map.items():
        site_class = classes.get(old_label)
        if site_class is not None:
            folded.call_site_classes[new_label] = site_class
    tensors = tuple(facts.get("output_tensors", ()))
    addresses = tuple(facts.get("output_tensor_addresses", ()))
    labels = tuple(facts.get("output_labels", ()))
    if len(labels) != len(tensors):
        raise _unconvertible(
            f"pass {pass_number} sealed {len(tensors)} outputs but {len(labels)} labels", code=_CODE
        )
    for index, (tensor, label) in enumerate(zip(tensors, labels, strict=True)):
        if label is None:
            raise _unconvertible(
                f"an output tensor of pass {pass_number} carries no capture label", code=_CODE
            )
        folded.output_tensors.append(tensor)
        folded.output_tensor_addresses.append(addresses[index] if index < len(addresses) else "")
        folded.output_labels.append(_mapped_label(label, label_map))
