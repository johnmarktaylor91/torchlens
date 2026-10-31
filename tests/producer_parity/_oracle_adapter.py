"""Test-side inverse oracle adapter (design-of-record 6.3).

``op_event_from_record`` feeds the UNCHANGED capture-oracle characterizer: a
compat ``OpEvent`` in, the same object out (identity); a decomposed
``OpRecord`` in (P3+), a reconstructed genuine ``OpEvent`` out, so the
characterizer and its goldens are provably insensitive to the record-model
migration. The adapter dies in S15 with ``OpEvent``.

``grad_fn_handle`` never lives on an ``OpRecord`` (single ownership: the
journal's ``grad_fn_handles_by_label_raw`` side index), so callers that need
handle-bearing reconstructions pass the index value through the keyword.
"""

from __future__ import annotations

from typing import Any

from torchlens.ir.events import OpEvent
from torchlens.ir.op_record import OpRecord


def op_event_from_record(record: Any, *, grad_fn_handle: Any = None) -> OpEvent:
    """Return a genuine compat ``OpEvent`` for one journal record."""

    if isinstance(record, OpEvent):
        return record
    if not isinstance(record, OpRecord):
        raise TypeError(f"not a journal op record: {type(record).__qualname__}")

    # Reconstruct the legacy smuggled transform_config channels: the clean
    # user mapping plus fn_code_location (facet field) plus the UNCONDITIONAL
    # `_tl_annotations` stamp both legacy producers write on every op.
    transform_config: dict[str, object] = dict(record.transform_config)
    transform_facet = record.transform
    fn_code_location = None if transform_facet is None else transform_facet.fn_code_location
    if fn_code_location is not None:
        transform_config["fn_code_location"] = fn_code_location
    annotations_facet = record.annotations_facet
    transform_config["_tl_annotations"] = (
        dict(annotations_facet.annotations) if annotations_facet is not None else {}
    )

    return OpEvent(
        kind=record.kind,
        label_raw=record.label_raw,
        layer_label_raw=record.layer_label_raw,
        layer_type=record.layer_type,
        raw_index=record.raw_index,
        type_index=record.type_index,
        step_index=record.step_index,
        source_trace=None,
        source_trace_id=None,
        tracing_finished=record.tracing_finished,
        construction_done=record.construction_done,
        function=record.function,
        output=record.output,
        templates=record.templates,
        parents=record.parents,
        parent_arg_positions=record.parent_arg_positions,
        _edge_uses=tuple(record._edge_uses),
        params=tuple(record.params),
        parent_params=tuple(record.parent_params),
        module_stack=tuple(record.module_stack),
        modules=tuple(record.modules),
        backend_semantics=record.backend_semantics,
        policy=record.policy,
        predicate_matched=record.predicate_matched,
        pass_index=record.pass_index,
        grad_fn_class_qualname=record.grad_fn_class_qualname,
        grad_fn_handle=grad_fn_handle,
        equivalence_class=record.equivalence_class,
        is_transform=record.is_transform,
        transform_kind=record.transform_kind,
        transform_chain=tuple(record.transform_chain),
        transform_config=transform_config,
        transform_fn_name=record.transform_fn_name,
        transform_fn_qualname=record.transform_fn_qualname,
        transform_fn_source=record.transform_fn_source,
        is_output_parent=record.is_output_parent,
        has_internal_source_ancestor=record.has_internal_source_ancestor,
        internal_source_ancestors=record.internal_source_ancestors,
        input_ancestors=record.input_ancestors,
        root_ancestors=record.root_ancestors,
        func_call_id=record.func_call_id,
        is_bottom_level=record.is_bottom_level,
        is_scalar_bool=record.is_scalar_bool,
        bool_value=record.bool_value,
        intervention_fired=record.intervention_fired,
        intervention_replaced=record.intervention_replaced,
        fire_results=tuple(record.fire_results),
        intervention_template_ref=None,
        record_context=record.record_context,
        capture_spec=record.capture_spec,
        unattributed_tensor_args=tuple(record.unattributed_tensor_args),
        dropped_edge_tensor_args=tuple(record.dropped_edge_tensor_args),
        input_was_parameter=record.input_was_parameter,
        seq=record.seq,
    )
