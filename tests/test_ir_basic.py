"""Basic schema tests for the backend-neutral capture IR."""

from __future__ import annotations

import importlib
import sys
from dataclasses import FrozenInstanceError, fields, is_dataclass
from unittest.mock import patch

import pytest

from torchlens.ir import (
    ArgTemplateRef,
    BackendSemantics,
    BackwardSidecar,
    BlobRef,
    BufferEvent,
    CaptureEvents,
    CapturePolicy,
    ConditionalEvent,
    ContainerSpec,
    DeferredRef,
    FireResult,
    FunctionCallRef,
    FunctionEventInput,
    InterventionState,
    InterventionTemplateRef,
    ModuleEvent,
    ModuleFrame,
    OpEvent,
    OutputRef,
    ParamRef,
    ParentEdge,
    RecordContext,
    ReservedLabel,
    TensorRef,
    TraceBuildState,
)


def _assert_frozen_and_slotted(instance: object) -> None:
    """Assert a frozen slots dataclass rejects mutation and undeclared fields."""
    assert is_dataclass(instance)
    field_name = fields(instance)[0].name
    with pytest.raises(FrozenInstanceError):
        setattr(instance, field_name, "changed")
    with pytest.raises(AttributeError):
        object.__setattr__(instance, "undeclared_slot", "changed")


def _build_ir_instances() -> dict[str, object]:
    """Build one sensible instance of every exported IR dataclass."""
    blob_ref = BlobRef(
        uri="bundle://activation/0",
        format="raw_bytes",
        dtype="torch.float32",
        shape=(1, 2),
        byte_length=8,
        sha256="0" * 64,
    )
    deferred_ref = DeferredRef(
        backend="torch",
        handle_id="handle-0",
        blob_ref=blob_ref,
        inferred_shape=(1, 2),
        inferred_dtype="torch.float32",
        materialize_fn=None,
    )
    tensor_ref = TensorRef(
        label_raw="linear_1_1_raw",
        shape=(1, 2),
        dtype="torch.float32",
        device="cpu",
        requires_grad=False,
        memory=8,
        payload=deferred_ref,
        blob_ref=blob_ref,
        backend_handle_id="tensor-0",
    )
    container_spec = ContainerSpec(
        kind="tuple",
        type_name="tuple",
        fields=(),
        length=1,
        metadata=(("source", "test"),),
    )
    output_ref = OutputRef(
        tensor=tensor_ref,
        transformed_tensor=None,
        has_saved_outs=True,
        output_device="cpu",
        out_postfunc=None,
        detach_saved_activations=True,
        visualizer_path=None,
        multi_output_index=None,
        is_part_of_iterable_output=False,
        container_path=(),
        container_spec=container_spec,
        child_versions=(),
    )
    function_ref = FunctionCallRef(
        func=None,
        func_name="linear",
        func_qualname="torch.nn.functional.linear",
        func_call_id=1,
        code_context=(),
        func_duration=0.1,
        flops_forward=None,
        flops_backward=None,
        func_rng_states=None,
        func_autocast_state=None,
        arg_names=("input",),
        num_args_total=1,
        num_pos_args=1,
        num_kwargs=0,
        non_tensor_pos_args=(),
        non_tensor_kwargs=(),
        func_non_tensor_args=(),
        is_inplace=False,
        func_config=(),
    )
    arg_template_ref = ArgTemplateRef(
        saved_args=None,
        saved_kwargs=None,
        args_template=None,
        kwargs_template=None,
        has_saved_args=False,
    )
    parent_edge = ParentEdge(
        parent_label_raw="input_1_0_raw",
        arg_position=0,
        edge_use="arg",
    )
    param_ref = ParamRef(
        barcode="param-0",
        address="layer.weight",
        shape=(2, 2),
        dtype="torch.float32",
        trainable=True,
        module_address="layer",
    )
    module_frame = ModuleFrame(
        address="layer",
        address_normalized="layer",
        module_type="Linear",
        call_index=1,
        fx_qualpath=None,
        entry_argnames=("input",),
    )
    buffer_event = BufferEvent(
        buffer_address="layer.running_mean",
        name="running_mean",
        address="layer",
        buffer_pass=1,
        parent_label_raw=None,
        shape=(2,),
        dtype="torch.float32",
        memory=8,
        module_stack=(module_frame,),
    )
    module_event = ModuleEvent(
        address="layer",
        all_addresses=("layer",),
        call_index=1,
        call_label="layer:1",
        layers_raw=("linear_1_1_raw",),
        input_layers_raw=("input_1_0_raw",),
        output_layers_raw=("linear_1_1_raw",),
        forward_args_summary=(),
        forward_kwargs_summary=(),
        forward_args=None,
        forward_kwargs=None,
        call_parent=None,
        call_children=(),
    )
    backend_semantics = BackendSemantics(
        grad_fn_id=None,
        grad_fn_name=None,
        autograd_saved_memory=None,
        num_autograd_saved_tensors=None,
        mutates_inputs=(),
        bytes_delta_at_call=None,
        bytes_peak_at_call=None,
    )
    capture_policy = CapturePolicy(
        must_keep_topology=True,
        save_payload=True,
        requires_isolation=False,
        save_args=False,
        save_code=False,
        save_rng=False,
        save_grad=False,
        stream=False,
    )
    fire_result = FireResult(
        plan_id="plan-0",
        site_label="linear_1_1_raw",
        fired_at_capture_index=1,
        pre_hook_shape=(1, 2),
        post_hook_shape=(1, 2),
        pre_hook_dtype="torch.float32",
        post_hook_dtype="torch.float32",
        replaced=False,
        fire_record=None,
    )
    intervention_template_ref = InterventionTemplateRef(
        template_id="template-0",
        spec_revision=1,
        template_kind="live",
        template_args=(),
    )
    op_event = OpEvent(
        kind="op",
        label_raw="linear_1_1_raw",
        layer_label_raw="linear_1_1_raw",
        layer_type="linear",
        capture_index=1,
        type_index=1,
        compute_index=1,
        source_trace_id=None,
        tracing_finished=False,
        construction_done=False,
        function=function_ref,
        output=output_ref,
        templates=arg_template_ref,
        parents=(parent_edge,),
        params=(param_ref,),
        module_stack=(module_frame,),
        backend_semantics=backend_semantics,
        policy=capture_policy,
        predicate_matched=True,
        is_bottom_level=True,
        is_scalar_bool=False,
        bool_value=None,
        intervention_fired=True,
        intervention_replaced=False,
        fire_results=(fire_result,),
        intervention_template_ref=intervention_template_ref,
    )
    conditional_event = ConditionalEvent(
        conditional_id=1,
        record={"label": "bool_1_1_raw"},
        arm_entry_edges=(("entry", "then"),),
        edge_call_indices=(("entry", "then", 1, "bool_1_1_raw", 1),),
    )
    reserved_label = ReservedLabel(
        label="linear_1_1_raw",
        label_raw="linear_1_1_raw",
        capture_index=1,
        type_index=1,
        layer_type="linear",
        site=("linear_1_1_raw", "linear", 0),
    )
    function_event_input = FunctionEventInput(
        func=object(),
        func_name="linear",
        func_qualname="torch.nn.functional.linear",
        args=(object(),),
        kwargs={"bias": object()},
        raw_output=None,
        arg_copies=None,
        kwarg_copies=None,
        module_stack=(module_frame,),
        is_bottom_level_func=True,
        func_call_id=1,
        expected_output_count=1,
    )
    record_context = RecordContext(
        kind="op",
        label="linear_1_1",
        raw_label="linear_1_1_raw",
        pass_index=1,
        event_index=1,
        compute_index=1,
        layer_type="linear",
        type_index=1,
        capture_index=1,
        func_name="linear",
        address="layer",
        module_type="Linear",
        module_pass_index=1,
        module_stack=(module_frame,),
        recent_events=(),
        recent_ops=(),
        parent_labels=("input_1_0",),
        input_output_address=None,
        shape=(1, 2),
        dtype="torch.float32",
        tensor_device="cpu",
        tensor_requires_grad=False,
        output_index=0,
        is_bottom_level_func=True,
        time_since_pass_start=0.0,
        sample_id=None,
        label_raw="linear_1_1_raw",
        label_prefix="linear",
        func_call_id=1,
        parent_labels_raw=("input_1_0_raw",),
        is_output_parent=False,
        backend_requires_isolation=False,
        is_scalar_bool=False,
        bool_value=None,
    )
    intervention_state = InterventionState(
        has_direct_writes=False,
        spec_revision=1,
        out_recipe_revision=1,
        append_sequence_id=0,
        warned_direct_write=False,
        warned_mutate_in_place=False,
        last_run_ctx=None,
    )
    trace_build_state = TraceBuildState(
        raw_layer_dict={"linear_1_1_raw": object()},
        raw_layer_labels_list=["linear_1_1_raw"],
        output_container_specs_by_raw_label={"linear_1_1_raw": container_spec},
        output_container_specs=(container_spec,),
        module_events=[module_event],
        conditional_events=[conditional_event],
    )
    backward_sidecar = BackwardSidecar(
        backend_name="torch",
        has_backward_pass=False,
        grad_fn_logs={},
        grad_fn_order=(),
        backward_root_grad_fn_id=None,
        backward_num_calls=0,
        backward_peak_memory=None,
        backward_memory_backend=None,
    )

    return {
        "blob_ref": blob_ref,
        "deferred_ref": deferred_ref,
        "tensor_ref": tensor_ref,
        "container_spec": container_spec,
        "output_ref": output_ref,
        "function_ref": function_ref,
        "arg_template_ref": arg_template_ref,
        "parent_edge": parent_edge,
        "param_ref": param_ref,
        "module_frame": module_frame,
        "buffer_event": buffer_event,
        "module_event": module_event,
        "backend_semantics": backend_semantics,
        "capture_policy": capture_policy,
        "fire_result": fire_result,
        "intervention_template_ref": intervention_template_ref,
        "op_event": op_event,
        "conditional_event": conditional_event,
        "reserved_label": reserved_label,
        "function_event_input": function_event_input,
        "record_context": record_context,
        "intervention_state": intervention_state,
        "trace_build_state": trace_build_state,
        "backward_sidecar": backward_sidecar,
    }


def test_imports_every_public_ir_type() -> None:
    """Import every IR type from the package root."""
    instances = _build_ir_instances()

    assert set(instances) == {
        "blob_ref",
        "deferred_ref",
        "tensor_ref",
        "container_spec",
        "output_ref",
        "function_ref",
        "arg_template_ref",
        "parent_edge",
        "param_ref",
        "module_frame",
        "buffer_event",
        "module_event",
        "backend_semantics",
        "capture_policy",
        "fire_result",
        "intervention_template_ref",
        "op_event",
        "conditional_event",
        "reserved_label",
        "function_event_input",
        "record_context",
        "intervention_state",
        "trace_build_state",
        "backward_sidecar",
    }


def test_frozen_dataclasses_are_frozen_and_slotted() -> None:
    """Assert frozen IR dataclasses enforce immutability and slots."""
    instances = _build_ir_instances()
    frozen_names = {
        "blob_ref",
        "deferred_ref",
        "tensor_ref",
        "container_spec",
        "output_ref",
        "function_ref",
        "arg_template_ref",
        "parent_edge",
        "param_ref",
        "module_frame",
        "buffer_event",
        "module_event",
        "backend_semantics",
        "capture_policy",
        "fire_result",
        "intervention_template_ref",
        "op_event",
        "conditional_event",
        "reserved_label",
        "function_event_input",
        "record_context",
        "backward_sidecar",
    }

    for name in frozen_names:
        _assert_frozen_and_slotted(instances[name])


def test_mutable_slotted_state_dataclasses_reject_undeclared_fields() -> None:
    """Assert mutable state dataclasses are slotted but not frozen."""
    instances = _build_ir_instances()
    intervention_state = instances["intervention_state"]
    trace_build_state = instances["trace_build_state"]

    assert is_dataclass(intervention_state)
    assert is_dataclass(trace_build_state)
    setattr(intervention_state, "has_direct_writes", True)
    setattr(trace_build_state, "raw_layer_labels_list", [])
    with pytest.raises(AttributeError):
        setattr(intervention_state, "undeclared_slot", True)
    with pytest.raises(AttributeError):
        setattr(trace_build_state, "undeclared_slot", True)


def test_capture_events_mutation_and_label_reservation() -> None:
    """Assert CaptureEvents is mutable and reserves labels atomically."""
    op_event = _build_ir_instances()["op_event"]
    assert isinstance(op_event, OpEvent)
    events = CaptureEvents()

    events.append(op_event)
    events.extend([op_event])
    assert events.op_events == [op_event, op_event]

    no_labels = events.reserve_label_block("linear", 0)
    assert no_labels == ()
    assert events.raw_layer_counter == 0
    assert events.raw_layer_type_counter == {}

    labels = events.reserve_label_block("linear", 2)
    assert [label.label_raw for label in labels] == [
        "linear_1_1_raw",
        "linear_2_2_raw",
    ]
    assert [label.capture_index for label in labels] == [1, 2]
    assert [label.type_index for label in labels] == [1, 2]

    next_label = events.reserve_label("relu")
    assert next_label.label_raw == "relu_1_3_raw"
    assert events.raw_layer_counter == 3
    assert events.raw_layer_type_counter == {"linear": 2, "relu": 1}


def test_ir_imports_without_torch_module() -> None:
    """Assert importing IR succeeds when torch is absent from sys.modules."""
    with patch.dict(sys.modules, {"torch": None}):
        module = importlib.import_module("torchlens.ir")

    assert module.TensorRef is TensorRef
