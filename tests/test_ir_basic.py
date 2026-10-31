"""Basic schema tests for the backend-neutral capture IR."""

from __future__ import annotations

import importlib
import subprocess
import sys
from dataclasses import FrozenInstanceError, fields, is_dataclass
from typing import get_type_hints
from unittest.mock import patch

import pytest

from torchlens.ir import (
    ArgTemplateRef,
    BackendSemantics,
    BlobRef,
    CaptureEvents,
    CapturePolicy,
    ConditionalEvent,
    ContainerSpec,
    DeferredRef,
    DeviceRef,
    DtypeRef,
    FireResult,
    FunctionCallRef,
    FunctionEventInput,
    InterventionState,
    InterventionTemplateRef,
    ModuleCaptureWorkspace,
    ModuleEnterEvent,
    ModuleExitEvent,
    ModuleFrame,
    ModulePrepEvent,
    OpEvent,
    OutputRef,
    ParamRef,
    ParentEdge,
    PreHookProvenanceEvent,
    RawGraphWorkspace,
    RecordContext,
    ReservedLabel,
    TensorRef,
    WrapperRuntimeWorkspace,
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
        length=1,
    )
    output_ref = OutputRef(
        tensor=tensor_ref,
        transformed_tensor=None,
        has_saved_activation=True,
        output_device="cpu",
        activation_transform=None,
        detach_saved_activations=True,
        visualizer_path=None,
        multi_output_index=None,
        in_multi_output=False,
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
    module_prep_event = ModulePrepEvent(
        address="layer",
        all_addresses=("layer",),
        module_type_str="linear",
        cls_qualname="torch.nn.modules.linear.Linear",
        class_name="Linear",
        address_children=(),
        class_source_file=None,
        class_source_line=None,
        init_source_file=None,
        init_source_line=None,
        forward_source_file=None,
        forward_source_line=None,
        class_docstring=None,
        init_signature=None,
        init_docstring=None,
        forward_signature=None,
        forward_docstring=None,
        forward_pre_hooks=(),
        forward_hooks=(),
        backward_pre_hooks=(),
        backward_hooks=(),
        full_backward_pre_hooks=(),
        full_backward_hooks=(),
        training_at_prep=True,
        custom_attributes=(),
        custom_methods=(),
    )
    module_enter_event = ModuleEnterEvent(
        address="layer",
        call_index=1,
        call_label="layer:1",
        training=True,
        code_context=(),
        call_stack=(),
        forward_start_time=0.0,
        forward_args=None,
        forward_kwargs=None,
        forward_args_template=None,
        forward_kwargs_template=None,
        layer_argnames=(("input_1_0_raw", 0),),
    )
    module_exit_event = ModuleExitEvent(
        address="layer",
        call_index=1,
        call_label="layer:1",
        forward_duration=0.1,
        output_structure=container_spec,
        output_tensor_labels_raw=("linear_1_1_raw",),
        per_output_atomic=(("linear_1_1_raw", (module_frame,), True, ("layer", 1)),),
    )
    backend_semantics = BackendSemantics(
        backend_grad_handle=None,
        grad_fn_class_name=None,
        autograd_memory=None,
        num_autograd_tensors=None,
        mutated_input_positions=(),
        aliased_output_inputs=(),
        unknown_aliasing=False,
        bytes_delta_at_call=None,
        bytes_peak_at_call=None,
    )
    # R47-2 (eca53c86): CapturePolicy keeps only its three consumed facts.
    capture_policy = CapturePolicy(
        save_payload=True,
        save_grad=False,
        save_mode="copy",
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
        raw_index=1,
        type_index=1,
        step_index=1,
        source_trace=None,
        source_trace_id=None,
        tracing_finished=False,
        construction_done=False,
        function=function_ref,
        output=output_ref,
        templates=arg_template_ref,
        parents=(parent_edge,),
        parent_arg_positions={"args": {0: "input_1_0_raw"}, "kwargs": {}},
        _edge_uses=(),
        params=(param_ref,),
        parent_params=(),
        module_stack=(module_frame,),
        modules=(("layer", 1),),
        backend_semantics=backend_semantics,
        policy=capture_policy,
        predicate_matched=True,
        pass_index=1,
        grad_fn_class_qualname=None,
        grad_fn_handle=None,
        equivalence_class="linear_hash_modulelayer",
        is_transform=False,
        transform_kind=None,
        transform_chain=(),
        transform_config={},
        transform_fn_name=None,
        transform_fn_qualname=None,
        transform_fn_source=None,
        is_output_parent=False,
        has_internal_source_ancestor=False,
        internal_source_ancestors=frozenset(),
        input_ancestors=frozenset({"input_1_0_raw"}),
        root_ancestors=frozenset({"input_1_0_raw"}),
        func_call_id=1,
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
        raw_index=1,
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
        step_index=1,
        layer_type="linear",
        type_index=1,
        raw_index=1,
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
        dtype=DtypeRef.from_value("torch.float32"),
        tensor_device=DeviceRef.from_value("cpu"),
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
        last_run=None,
    )
    raw_graph_workspace = RawGraphWorkspace(
        raw_layer_dict={"linear_1_1_raw": object()},
        raw_layer_labels_list=["linear_1_1_raw"],
    )
    module_capture_workspace = ModuleCaptureWorkspace()
    wrapper_runtime_workspace = WrapperRuntimeWorkspace()
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
        "module_prep_event": module_prep_event,
        "module_enter_event": module_enter_event,
        "module_exit_event": module_exit_event,
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
        "raw_graph_workspace": raw_graph_workspace,
        "module_capture_workspace": module_capture_workspace,
        "wrapper_runtime_workspace": wrapper_runtime_workspace,
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
        "module_prep_event",
        "module_enter_event",
        "module_exit_event",
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
        "raw_graph_workspace",
        "module_capture_workspace",
        "wrapper_runtime_workspace",
    }


def test_frozen_dataclasses_are_frozen_and_slotted() -> None:
    """Assert frozen IR dataclasses enforce immutability and slots."""
    instances = _build_ir_instances()
    frozen_names = {
        "blob_ref",
        "deferred_ref",
        "tensor_ref",
        "output_ref",
        "function_ref",
        "arg_template_ref",
        "parent_edge",
        "param_ref",
        "module_frame",
        "backend_semantics",
        "capture_policy",
        "fire_result",
        "intervention_template_ref",
        "op_event",
        "conditional_event",
        "reserved_label",
        "function_event_input",
        "record_context",
    }

    for name in frozen_names:
        _assert_frozen_and_slotted(instances[name])


def test_container_spec_is_rich_leaf_type() -> None:
    """Assert IR output metadata uses the leaf recursive ContainerSpec."""

    from torchlens.ir.container import ContainerSpec as LeafContainerSpec

    instances = _build_ir_instances()
    events_module = importlib.import_module("torchlens.ir.events")
    hints = get_type_hints(
        OutputRef,
        globalns={**vars(events_module), "TensorRef": TensorRef},
    )
    assert ContainerSpec is LeafContainerSpec
    assert isinstance(instances["container_spec"], LeafContainerSpec)
    assert hints["container_spec"] == LeafContainerSpec | None


def test_container_leaf_import_order_has_no_cycle() -> None:
    """Import the container leaf before higher-level intervention modules."""

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import torchlens.ir.container; "
                "import torchlens.intervention.types; "
                "print('IMPORT_OK')"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "IMPORT_OK"


def test_mutable_slotted_state_dataclasses_reject_undeclared_fields() -> None:
    """Assert mutable state dataclasses are slotted but not frozen."""
    instances = _build_ir_instances()
    intervention_state = instances["intervention_state"]
    raw_graph_workspace = instances["raw_graph_workspace"]

    assert is_dataclass(intervention_state)
    assert is_dataclass(raw_graph_workspace)
    intervention_state.has_direct_writes = True
    raw_graph_workspace.raw_layer_labels_list = []
    with pytest.raises(AttributeError):
        intervention_state.undeclared_slot = True
    with pytest.raises(AttributeError):
        raw_graph_workspace.undeclared_slot = True


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
    assert [label.raw_index for label in labels] == [1, 2]
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


def test_capture_events_concat_follows_declared_merge_law() -> None:
    """journal.concat enforces the declared per-lane merge policies."""
    from dataclasses import fields as dataclass_fields

    from torchlens.ir.capture_events import LANE_MERGE_POLICIES, CaptureEvents
    from torchlens.ir.events import ModulePrepEvent, PreHookProvenanceEvent

    # The merge law is total: every journal lane on CaptureEvents has a
    # policy. Lanes are derived structurally (the list-typed journal buffers,
    # i.e. ``field(default_factory=list)``), not by name: the amendment lane
    # (``op_amendments``) is a journal lane whose rows are not events, and a
    # name-suffix heuristic would silently miss any such lane.
    lane_fields = {f.name for f in dataclass_fields(CaptureEvents) if f.default_factory is list}
    assert lane_fields == set(LANE_MERGE_POLICIES)

    def _prep(address: str) -> ModulePrepEvent:
        return ModulePrepEvent(
            address=address,
            all_addresses=(address,),
            module_type_str="Linear",
            cls_qualname="torch.nn.Linear",
            class_name="Linear",
            address_children=(),
            class_source_file=None,
            class_source_line=None,
            init_source_file=None,
            init_source_line=None,
            forward_source_file=None,
            forward_source_line=None,
            class_docstring=None,
            init_signature=None,
            init_docstring=None,
            forward_signature=None,
            forward_docstring=None,
            forward_pre_hooks=None,
            forward_hooks=None,
            backward_pre_hooks=None,
            backward_hooks=None,
            full_backward_pre_hooks=None,
            full_backward_hooks=None,
            training_at_prep=False,
            custom_attributes=(),
            custom_methods=(),
        )

    def _pre_hook(address: str) -> PreHookProvenanceEvent:
        return PreHookProvenanceEvent(
            address=address,
            call_index=1,
            inputs_before_pre_hooks=None,
            inputs_after_pre_hooks=None,
            effects=(),
            capture_complete=True,
            incomplete_reasons=(),
        )

    target = CaptureEvents()
    run_one = CaptureEvents()
    run_one.append_module_prep(_prep("linear"))
    run_one.append_pre_hook(_pre_hook("linear"))
    target.concat(run_one)
    assert [event.address for event in target.module_prep_events] == ["linear"]
    assert len(target.pre_hook_events) == 1

    run_two = CaptureEvents()
    run_two.append_module_prep(_prep("other"))
    run_two.append_pre_hook(_pre_hook("other"))
    target.concat(run_two)
    # first_run_only: the second run's identical module structure is skipped.
    assert [event.address for event in target.module_prep_events] == ["linear"]
    # append_restamp: pre-hook facts accumulate with re-stamped unique seqs.
    assert [event.address for event in target.pre_hook_events] == ["linear", "other"]
    all_seqs = [event.seq for event in (*target.module_prep_events, *target.pre_hook_events)]
    assert len(all_seqs) == len(set(all_seqs))
    assert all(seq >= 1 for seq in all_seqs)
    assert max(all_seqs) <= target.event_seq

    # run_local lanes never merge, and self-concat is a no-op.
    before = list(target.pre_hook_events)
    target.concat(target)
    assert target.pre_hook_events == before


def _merge_prep_event(address: str) -> ModulePrepEvent:
    """Build a minimal module-prep event for merge-law tests."""

    return ModulePrepEvent(
        address=address,
        all_addresses=(address,),
        module_type_str="Linear",
        cls_qualname="torch.nn.Linear",
        class_name="Linear",
        address_children=(),
        class_source_file=None,
        class_source_line=None,
        init_source_file=None,
        init_source_line=None,
        forward_source_file=None,
        forward_source_line=None,
        class_docstring=None,
        init_signature=None,
        init_docstring=None,
        forward_signature=None,
        forward_docstring=None,
        forward_pre_hooks=None,
        forward_hooks=None,
        backward_pre_hooks=None,
        backward_hooks=None,
        full_backward_pre_hooks=None,
        full_backward_hooks=None,
        training_at_prep=False,
        custom_attributes=(),
        custom_methods=(),
    )


def _merge_pre_hook_event(address: str) -> PreHookProvenanceEvent:
    """Build a minimal pre-hook provenance event for merge-law tests."""

    return PreHookProvenanceEvent(
        address=address,
        call_index=1,
        inputs_before_pre_hooks=None,
        inputs_after_pre_hooks=None,
        effects=(),
        capture_complete=True,
        incomplete_reasons=(),
    )


def test_concat_merges_intervention_events_lane() -> None:
    """A populated intervention-edit lane merges under append_restamp."""

    from torchlens.ir.capture_events import CaptureEvents
    from torchlens.ir.events import InterventionAppliedEvent

    source = CaptureEvents()
    source.append_intervention(
        InterventionAppliedEvent(
            label_raw="x_raw", kind="replaced", origin="raw_forward_hook", timestamp=1.0
        )
    )
    target = CaptureEvents()
    target.concat(source)
    assert [event.label_raw for event in target.intervention_events] == ["x_raw"]
    assert target.intervention_events[0].seq == 1
    assert target.intervention_events[0].seq <= target.event_seq


def test_lane_appenders_cover_every_merging_policy() -> None:
    """Every non-run_local lane policy has a registered single-writer appender.

    A lane declared as merging but left unwired must be impossible to ship:
    this assertion is the totality half, and ``concat`` itself fail-closes at
    runtime (see ``test_concat_fails_closed_on_declared_unwired_lane``).
    """

    from torchlens.ir.capture_events import _LANE_APPENDERS, LANE_MERGE_POLICIES, CaptureEvents

    merging_lanes = {lane for lane, policy in LANE_MERGE_POLICIES.items() if policy != "run_local"}
    assert merging_lanes <= set(_LANE_APPENDERS)
    for appender_name in _LANE_APPENDERS.values():
        assert callable(getattr(CaptureEvents(), appender_name))


def test_concat_fails_closed_on_declared_unwired_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    """concat refuses (never silently skips) a merging lane with no appender."""

    from torchlens.ir import capture_events as capture_events_module
    from torchlens.ir.capture_events import CaptureEvents

    monkeypatch.setitem(
        capture_events_module.LANE_MERGE_POLICIES, "backward_events", "append_restamp"
    )
    target = CaptureEvents()
    with pytest.raises(capture_events_module.LaneMergePolicyError, match="backward_events"):
        target.concat(CaptureEvents())


def test_concat_preserves_source_chronology_across_lanes() -> None:
    """Merged events keep the source stream's cross-lane chronological order."""

    from torchlens.ir.capture_events import CaptureEvents
    from torchlens.ir.events import InterventionAppliedEvent

    source = CaptureEvents()
    source.append_module_prep(_merge_prep_event("a"))
    source.append_pre_hook(_merge_pre_hook_event("x"))
    source.append_module_prep(_merge_prep_event("b"))
    source.append_intervention(
        InterventionAppliedEvent(
            label_raw="edit_raw", kind="replaced", origin="raw_forward_hook", timestamp=1.0
        )
    )
    source.append_pre_hook(_merge_pre_hook_event("y"))

    def _flatten(events: CaptureEvents) -> list[tuple[int, str]]:
        rows = [(event.seq, f"prep:{event.address}") for event in events.module_prep_events]
        rows += [(event.seq, f"pre_hook:{event.address}") for event in events.pre_hook_events]
        rows += [(event.seq, f"edit:{event.label_raw}") for event in events.intervention_events]
        return sorted(rows)

    source_order = [identity for _seq, identity in _flatten(source)]
    target = CaptureEvents()
    target.concat(source)
    target_order = [identity for _seq, identity in _flatten(target)]
    assert (
        target_order
        == source_order
        == [
            "prep:a",
            "pre_hook:x",
            "prep:b",
            "edit:edit_raw",
            "pre_hook:y",
        ]
    )


def test_concat_clones_events_and_never_mutates_the_source_stream() -> None:
    """Merging restamps clones; the sealed source stream's seqs stay intact."""

    from torchlens.ir.capture_events import CaptureEvents

    source = CaptureEvents()
    source.append_module_prep(_merge_prep_event("a"))
    source.append_pre_hook(_merge_pre_hook_event("x"))
    source_events = [*source.module_prep_events, *source.pre_hook_events]
    source_seqs_before = [event.seq for event in source_events]

    target = CaptureEvents()
    target.append_pre_hook(_merge_pre_hook_event("existing"))
    target.concat(source)

    assert [event.seq for event in source_events] == source_seqs_before
    merged = [*target.module_prep_events, *target.pre_hook_events[1:]]
    assert all(
        merged_event is not source_event
        for merged_event, source_event in zip(merged, source_events)
    )
    all_target_seqs = [event.seq for event in (*target.module_prep_events, *target.pre_hook_events)]
    assert len(all_target_seqs) == len(set(all_target_seqs))


def test_concat_rejects_invalid_source_sequencing() -> None:
    """FAIL-AFTER-WHERE-PASSED-BEFORE: an invalid source seq domain refuses to merge.

    Review be2-closure probe regression: a source whose cross-lane seq domain is
    invalid (a duplicate, unstamped, or counter-bypassing stamp -- exactly what
    the journal seq invariants reject on a standalone stream) used to be
    silently sorted with dict-lane-order tie-breaking and re-stamped into a
    green target journal, laundering the producer defect. ``concat`` must
    reject the source typed and fail-closed, BEFORE any event moves.
    """

    from torchlens.ir.capture_events import CaptureEvents, SourceSequencingError
    from torchlens.ir.events import InterventionAppliedEvent

    def _duplicate_cross_lane_source() -> CaptureEvents:
        source = CaptureEvents()
        source.append_intervention(
            InterventionAppliedEvent(
                label_raw="edited_raw", kind="replaced", origin="raw_forward_hook", timestamp=1.0
            )
        )
        source.append_pre_hook(_merge_pre_hook_event("mod"))
        # Simulate a producer/writer regression: cross-lane duplicate stamps.
        object.__setattr__(source.pre_hook_events[0], "seq", source.intervention_events[0].seq)
        return source

    target = CaptureEvents()
    with pytest.raises(SourceSequencingError, match="appears in both"):
        target.concat(_duplicate_cross_lane_source())
    # Fail-closed atomicity: nothing merged and the target journal is untouched.
    assert not target.intervention_events
    assert not target.pre_hook_events
    assert target.event_seq == 0

    unstamped = CaptureEvents()
    unstamped.pre_hook_events.append(_merge_pre_hook_event("hand_built"))
    with pytest.raises(SourceSequencingError, match="unstamped"):
        CaptureEvents().concat(unstamped)

    counter_bypass = CaptureEvents()
    counter_bypass.append_pre_hook(_merge_pre_hook_event("mod"))
    object.__setattr__(counter_bypass.pre_hook_events[0], "seq", 7)
    with pytest.raises(SourceSequencingError, match="writer counter"):
        CaptureEvents().concat(counter_bypass)

    non_monotone = CaptureEvents()
    non_monotone.append_pre_hook(_merge_pre_hook_event("a"))
    non_monotone.append_pre_hook(_merge_pre_hook_event("b"))
    events = list(non_monotone.pre_hook_events)
    non_monotone.pre_hook_events[:] = [events[1], events[0]]
    with pytest.raises(SourceSequencingError, match="does not increase"):
        CaptureEvents().concat(non_monotone)
