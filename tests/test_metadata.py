"""Comprehensive metadata field testing for Trace and Op.

Uses small/fast models from example_models.py to verify that all key metadata
fields are populated correctly across different model types.
"""

import copy
import linecache
import pickle

import example_models
import pytest
import torch
import torch.nn as nn

import torchlens
import torchlens as tl
from torchlens import trace as trace_fn
from torchlens.capture.flops import (
    compute_backward_flops,
    compute_forward_flops,
)
from torchlens.data_classes import FuncCallLocation


class _SharedMultiOutputModel(nn.Module):
    """Small DAG whose shared node reaches two outputs."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return two children of one shared operation.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Additive and subtractive branches from one shared value.
        """

        shared = x * 2
        return shared + 1, shared - 1


# =============================================================================
# Trace fields
# =============================================================================


@pytest.mark.smoke
def test_general_info_fields(small_input):
    model = example_models.SimpleFF()
    mh = trace_fn(model, small_input)
    assert isinstance(mh.model_class_name, str)
    assert len(mh.model_class_name) > 0
    assert mh._tracing_finished is True
    assert isinstance(mh.num_ops, int)
    assert mh.num_ops > 0


@pytest.mark.smoke
def test_model_structure_non_recurrent(small_input: torch.Tensor) -> None:
    model = example_models.SimpleFF()
    mh = trace_fn(model, small_input)
    assert mh.is_recurrent is False
    assert mh.is_branching is False
    assert list(mh.recurrent_layers) == []


def test_model_structure_branching(small_input: torch.Tensor) -> None:
    model = example_models.SimpleBranching()
    mh = trace_fn(model, small_input)
    assert mh.is_branching is True


def test_model_structure_recurrent(input_2d: torch.Tensor) -> None:
    model = example_models.RecurrentParamsSimple()
    mh = trace_fn(model, input_2d)
    assert mh.is_recurrent is True
    assert len(mh.recurrent_layers) > 0
    assert all(layer.num_passes > 1 for layer in mh.recurrent_layers)
    assert [layer.layer_label for layer in mh.recurrent_layers] == [
        layer.layer_label for layer in mh.layers if layer.num_passes > 1
    ]
    first_recurrent_layer = mh.recurrent_layers[0]
    assert mh.recurrent_layers[first_recurrent_layer.layer_label] is first_recurrent_layer
    assert mh.recurrent_layers[f"{first_recurrent_layer.layer_label}:1"] is first_recurrent_layer


def test_recurrent_layers_accessor_rejects_non_recurrent_layer(input_2d: torch.Tensor) -> None:
    """Recurrent-layer lookup should not fall through to all layer records."""

    model = example_models.RecurrentParamsSimple()
    mh = trace_fn(model, input_2d)
    non_recurrent_layer = next(layer for layer in mh.layers if layer.num_passes == 1)
    with pytest.raises(KeyError):
        mh.recurrent_layers[non_recurrent_layer.layer_label]


def test_model_structure_conditional() -> None:
    model = example_models.ConditionalBranching()
    model_input = -torch.ones(6, 3, 224, 224)
    mh = trace_fn(model, model_input)
    assert mh.has_conditional_branching is True


def test_layer_tracking_fields(small_input):
    model = example_models.SimpleFF()
    mh = trace_fn(model, small_input)
    assert isinstance(mh.layer_list, list)
    assert len(mh.layer_list) > 0
    assert isinstance(mh.layer_labels, list)
    assert len(mh.layer_labels) > 0
    assert isinstance(mh.layer_dict_main_keys, dict)
    assert isinstance(mh.layer_dict_all_keys, dict)


@pytest.mark.smoke
def test_input_output_layers(small_input):
    model = example_models.SimpleFF()
    mh = trace_fn(model, small_input)
    assert isinstance(mh.input_layers, list)
    assert len(mh.input_layers) >= 1
    assert isinstance(mh.output_layers, list)
    assert len(mh.output_layers) >= 1


def test_buffer_layers():
    model = example_models.BufferModel()
    model_input = torch.rand(12, 12)
    mh = trace_fn(model, model_input)
    assert isinstance(mh.buffer_layers, list)
    assert len(mh.buffer_layers) > 0


def test_tensor_info_fields(small_input):
    model = example_models.SimpleFF()
    mh = trace_fn(model, small_input)
    assert isinstance(mh.num_tensors, int)
    assert mh.num_tensors > 0
    assert isinstance(mh.total_activation_memory, (int, float))
    assert mh.total_activation_memory > 0
    assert isinstance(mh.total_activation_memory, torchlens.Bytes)
    assert len(str(mh.total_activation_memory)) > 0


def test_forward_peak_memory_is_populated(small_input):
    """forward_peak_memory is measured and labeled on the default capture path.

    Regression: forward_peak_memory was declared and serialized but never written,
    so it was always 0 while backward_peak_memory was measured. The forward pass
    is now bracketed by a peak-memory probe (CUDA device peak; CPU/MPS host
    resident-set-size or MPS allocator delta).

    The default path deliberately does NOT run a tracemalloc probe -- that
    allocator hook costs 1.7x-2.5x total capture time -- so the coarse host delta
    may legitimately round to 0 for a model this small. The measurement is
    present and labeled; positive Python-allocation peaks for small models are
    the opt-in behavior asserted by
    ``test_measure_python_peak_memory_opt_in_reports_positive_peak``.
    """

    model = example_models.SimpleFF()
    mh = trace_fn(model, small_input)
    assert isinstance(mh.forward_peak_memory, torchlens.Bytes)
    assert int(mh.forward_peak_memory) >= 0
    assert mh.forward_memory_backend in {"cpu", "cuda", "mps"}
    assert mh.measure_python_peak_memory is False


def test_measure_python_peak_memory_opt_in_reports_positive_peak(small_input):
    """The opt-in tracemalloc probe reads positive even for a tiny model.

    ``CaptureOptions(measure_python_peak_memory=True)`` folds the stdlib
    tracemalloc Python-allocation peak into ``forward_peak_memory`` via ``max()``,
    which stays reliably positive where the host RSS delta rounds to zero. Only
    the measurement changes: the captured graph must be identical to the default
    path.
    """

    model = example_models.SimpleFF()
    baseline = trace_fn(model, small_input)
    measured = trace_fn(
        model,
        small_input,
        capture=torchlens.options.CaptureOptions(measure_python_peak_memory=True),
    )
    assert measured.measure_python_peak_memory is True
    assert isinstance(measured.forward_peak_memory, torchlens.Bytes)
    assert int(measured.forward_peak_memory) > 0
    assert measured.forward_memory_backend in {"cpu", "cuda", "mps"}
    assert [op.layer_label for op in measured.ops] == [op.layer_label for op in baseline.ops]


def test_param_info_fields(small_input):
    model = example_models.BatchNormModel()
    mh = trace_fn(model, small_input)
    assert isinstance(mh.num_param_tensors, int)
    assert isinstance(mh.num_params, int)
    assert mh.num_params > 0


def test_module_info_fields(small_input):
    model = example_models.NestedModules()
    mh = trace_fn(model, small_input)
    # Module info is now accessed via structured Module objects
    assert len(mh.modules) > 1
    root = mh.modules["self"]
    assert root.address == "self"
    assert len(root.layers) > 0
    # Submodules should have valid addresses and class names
    for ml in mh.modules:
        if ml.address != "self":
            assert isinstance(ml.class_name, str)
            assert ml.address_parent in mh.modules


def test_time_fields(small_input):
    model = example_models.SimpleFF()
    mh = trace_fn(model, small_input)
    assert isinstance(mh.capture_duration, float)
    assert mh.capture_duration > 0
    assert isinstance(mh.setup_duration, float)
    assert isinstance(mh.forward_duration, float)
    assert isinstance(mh.cleanup_duration, float)
    assert isinstance(mh.overhead_duration, float)


def test_multi_input_layers():
    model = example_models.MultiInputs()
    inputs = [
        torch.rand(6, 3, 224, 224),
        torch.rand(6, 3, 224, 224),
        torch.rand(6, 3, 224, 224),
    ]
    mh = trace_fn(model, inputs)
    assert len(mh.input_layers) == 3


def test_multi_output_layers(small_input):
    model = example_models.MultiOutputs()
    mh = trace_fn(model, small_input)
    assert len(mh.output_layers) >= 2


def test_internal_source_ops(small_input):
    model = example_models.SimpleInternallyGenerated()
    mh = trace_fn(model, small_input)
    assert isinstance(mh.internal_source_ops, list)
    assert len(mh.internal_source_ops) > 0


def test_equivalent_ops(input_2d):
    model = example_models.RecurrentParamsSimple()
    mh = trace_fn(model, input_2d)
    assert isinstance(mh.op_equivalence_classes, dict)
    assert len(mh.op_equivalence_classes) > 0


# =============================================================================
# Op fields
# =============================================================================


def test_label_fields(small_input):
    model = example_models.SimpleFF()
    mh = trace_fn(model, small_input)
    entry = mh[0]
    assert isinstance(entry.layer_label, str)
    assert isinstance(entry.layer_type, str)
    assert isinstance(entry.pass_index, int)
    assert isinstance(entry.lookup_keys, list)
    assert len(entry.lookup_keys) > 0


def test_input_layer_properties(small_input):
    model = example_models.SimpleFF()
    mh = trace_fn(model, small_input)
    input_entry = None
    for label in mh.layer_labels:
        e = mh[label]
        if e.is_input:
            input_entry = e
            break
    assert input_entry is not None
    assert input_entry.is_input is True
    assert input_entry.is_output is False
    assert input_entry.has_parents is False
    assert input_entry.has_children is True


def test_output_layer_properties(small_input):
    model = example_models.SimpleFF()
    mh = trace_fn(model, small_input)
    output_entry = None
    for label in mh.layer_labels:
        e = mh[label]
        if e.is_output:
            output_entry = e
            break
    assert output_entry is not None
    assert output_entry.is_output is True
    assert output_entry.has_children is False


def test_entry_tensor_info_fields(small_input):
    model = example_models.SimpleFF()
    mh = trace_fn(model, small_input)
    entry = mh[0]
    assert isinstance(entry.shape, (tuple, torch.Size))
    assert isinstance(entry.dtype, torch.dtype)
    assert isinstance(entry.activation_memory, (int, float))
    assert entry.out is not None


def test_function_call_fields(small_input):
    model = example_models.SimpleFF()
    mh = trace_fn(model, small_input)
    non_input = None
    for label in mh.layer_labels:
        e = mh[label]
        if not e.is_input:
            non_input = e
            break
    assert non_input is not None
    assert isinstance(non_input.func_name, str)
    assert len(non_input.func_name) > 0
    assert isinstance(non_input.func_duration, float)


def test_graph_relationships(small_input):
    model = example_models.SimpleFF()
    mh = trace_fn(model, small_input)
    for label in mh.layer_labels:
        entry = mh[label]
        assert isinstance(entry.parents, tuple)
        assert isinstance(entry.children, tuple)
        for parent_label in entry.parents:
            parent = mh[parent_label]
            assert label in parent.children
        for child_label in entry.children:
            child = mh[child_label]
            assert label in child.parents


def test_inplace_function_flag(small_input):
    model = example_models.InPlaceFuncs()
    mh = trace_fn(model, small_input)
    for label in mh.layer_labels:
        entry = mh[label]
        assert isinstance(entry.is_inplace, bool)
    found_relu_ = False
    for label in mh.layer_labels:
        entry = mh[label]
        if entry.func_name and entry.func_name.endswith("_"):
            found_relu_ = True
            break
    assert found_relu_, "InPlaceFuncs should have relu_ (inplace function name)"


def test_param_fields_with_params(small_input):
    model = example_models.BatchNormModel()
    mh = trace_fn(model, small_input)
    found = False
    for label in mh.layer_labels:
        entry = mh[label]
        if entry.uses_params:
            assert entry.num_params > 0
            found = True
            break
    assert found, "BatchNormModel should have layers computed with params"


def test_param_fields_without_params(small_input):
    model = example_models.SimpleFF()
    mh = trace_fn(model, small_input)
    for label in mh.layer_labels:
        entry = mh[label]
        assert entry.uses_params is False


def test_module_fields(small_input):
    model = example_models.NestedModules()
    mh = trace_fn(model, small_input)
    found = False
    for label in mh.layer_labels:
        entry = mh[label]
        if entry.in_submodule:
            assert isinstance(entry.module, str)
            assert isinstance(entry.module_call_depth, int)
            assert entry.module_call_depth > 0
            found = True
            break
    assert found, "NestedModules should have layers inside submodules"


def test_buffer_layer_fields():
    model = example_models.BufferModel()
    model_input = torch.rand(12, 12)
    mh = trace_fn(model, model_input)
    found = False
    for label in mh.layer_labels:
        entry = mh[label]
        if entry.is_buffer:
            assert isinstance(entry.address, str)
            found = True
            break
    assert found, "BufferModel should have buffer layers"


def test_internally_initialized_fields(small_input):
    model = example_models.SimpleInternallyGenerated()
    mh = trace_fn(model, small_input)
    found = False
    for label in mh.layer_labels:
        entry = mh[label]
        if entry.is_internal_source:
            found = True
            break
    assert found, "SimpleInternallyGenerated should have internally init layers"


def test_sibling_spouse_fields(small_input):
    model = example_models.SimpleFF()
    mh = trace_fn(model, small_input)
    for label in mh.layer_labels:
        entry = mh[label]
        assert isinstance(entry.siblings, list)
        assert isinstance(entry.co_parents, list)


def test_layer_siblings_tolerate_orphan_relation_labels():
    """Layer aggregates mirror the op-level orphan tolerance (r3 R05-N1).

    ``Op.siblings``/``Op.co_parents`` deliberately resolve relation labels
    through the ``orphans`` fallback on ``keep_orphans=True`` traces and may
    APPEND orphan labels to their result; the Layer aggregates (``siblings``,
    ``co_parents``, ``children``, ``parents``, and the per-pass dict views)
    previously did a bare ``trace[label]`` lookup on those same labels and
    raised from public read-only properties — the ``children`` instance even
    fired INSIDE ``Op.siblings`` whenever a relation label resolved to a
    Layer, outside its own fallback. No public capture constructs the state
    today (orphans are component-disjoint by the step-3 bidirectional flood),
    so the state is injected surgically: a mainline parent's relation list
    gains an orphan child label, exactly the shape the op-level fallback
    exists for.
    """

    class _WithOrphans(nn.Module):
        def forward(self, x):
            internal = torch.ones(3)
            _a = internal * 2
            _b = internal + 3
            return x * 5

    mh = trace_fn(
        model=_WithOrphans(),
        input_args=torch.randn(2, 3),
        capture=torchlens.options.CaptureOptions(keep_orphans=True),
    )
    orphan_labels = list(mh.orphans.keys())
    assert orphan_labels, "probe model must produce orphans"
    orphan_sibling = next(k for k in orphan_labels if mh.orphans[k].siblings)

    mainline = next(
        mh[label] for label in mh.layer_labels if not mh[label].is_input and mh[label].parents
    )
    parent_op = list(mh[mainline.parents[0]].ops.values())[0]
    # Direct assignment is the supported mutation spelling on Op records; the
    # list normalizes to the immutable view type.
    parent_op.children = list(parent_op.children) + [orphan_sibling]

    op0 = list(mainline.ops.values())[0]
    assert orphan_sibling in op0.siblings  # op level tolerates and appends
    aggregated = mainline.siblings  # RED before the fix: uncaught lookup error
    assert mh.orphans[orphan_sibling].layer_label in aggregated
    assert isinstance(mainline.co_parents, list)

    # b3 R05-N2: the 62aba742 tolerance stopped at label aggregates -- the
    # OBJECT-resolving surfaces of the same relation family crashed on the
    # fix's own output (`parent_layer.children` succeeded while
    # `parent_layer.get_children()` raised "not found"). They now resolve
    # orphan labels through the same fallback.
    parent_layer = mh[mainline.parents[0]]
    layer_children = parent_layer.get_children()
    assert orphan_sibling in {getattr(c, "layer_label", None) for c in layer_children} or any(
        getattr(c, "label", None) == orphan_sibling for c in layer_children
    )
    op_children = parent_op.get_children()
    assert any(
        getattr(c, "label", "").startswith(orphan_sibling.split(":")[0]) for c in op_children
    )
    assert isinstance(parent_op.get_parents(), list)
    assert isinstance(parent_layer.get_parents(), list)


def test_conditional_fields():
    model = example_models.ConditionalBranching()
    model_input = -torch.ones(6, 3, 224, 224)
    mh = trace_fn(model, model_input)
    found = False
    for label in mh.layer_labels:
        entry = mh[label]
        if entry.is_in_conditional_body:
            found = True
            break
    assert found, "ConditionalBranching should have layers in cond branches"


def test_distances_with_flag(small_input):
    model = example_models.SimpleFF()
    mh = trace_fn(
        model,
        small_input,
        capture=tl.options.CaptureOptions(compute_input_output_distances=True),
    )
    for label in mh.layer_labels:
        entry = mh[label]
        assert entry.min_distance_from_input is not None
        assert entry.max_distance_from_input is not None
        assert entry.min_distance_to_output is not None
        assert entry.max_distance_to_output is not None
        assert isinstance(entry.min_distance_from_input, int)


# =============================================================================
# Recurrent metadata
# =============================================================================


def test_recurrent_pass_indexbers(input_2d):
    model = example_models.RecurrentParamsSimple()
    mh = trace_fn(model, input_2d)
    found_multi_pass = any(
        op.pass_index > 1 for layer in mh.layer_logs.values() for op in layer.ops.values()
    )
    assert found_multi_pass, "Recurrent model should have layers with pass_index > 1"


def test_recurrent_layer_ops_total(input_2d):
    model = example_models.RecurrentParamsSimple()
    mh = trace_fn(model, input_2d)
    found = False
    for label in mh.layer_labels:
        entry = mh[label]
        if entry.num_passes > 1:
            found = True
            break
    assert found, "Recurrent model should have layers with ops_total > 1"


def test_recurrent_same_layer_operations(input_2d):
    model = example_models.RecurrentParamsSimple()
    mh = trace_fn(model, input_2d)
    found = False
    for label in mh.layer_labels:
        entry = mh[label]
        if entry.recurrent_ops and len(entry.recurrent_ops) > 0:
            found = True
            break
    assert found, "Recurrent model should have recurrent_ops"


def test_layer_logs_fewer_than_layer_list(input_2d):
    model = example_models.RecurrentParamsSimple()
    mh = trace_fn(model, input_2d)
    assert isinstance(mh.layer_logs, dict)
    assert len(mh.layer_logs) < len(mh.layer_list)


# =============================================================================
# Trace access patterns
# =============================================================================


def test_trace_len(small_input):
    model = example_models.SimpleFF()
    mh = trace_fn(model, small_input)
    assert len(mh) == len(mh.layer_list)


def test_getitem_by_positive_index(small_input):
    model = example_models.SimpleFF()
    mh = trace_fn(model, small_input)
    entry = mh[0]
    assert entry.layer_label == mh.layer_labels[0]


def test_getitem_by_negative_index(small_input):
    model = example_models.SimpleFF()
    mh = trace_fn(model, small_input)
    entry = mh[-1]
    assert entry.layer_label == mh.layer_labels[-1]


def test_getitem_by_label(small_input):
    model = example_models.SimpleFF()
    mh = trace_fn(model, small_input)
    label = mh.layer_labels[0]
    entry = mh[label]
    assert entry.layer_label == label


def test_trace_iter(small_input):
    model = example_models.SimpleFF()
    mh = trace_fn(model, small_input)
    items = list(mh)
    assert len(items) == len(mh.layer_list)


def test_layer_labels_properties(small_input):
    model = example_models.SimpleFF()
    mh = trace_fn(model, small_input)
    assert isinstance(mh.layer_labels, list)
    assert all(isinstance(lbl, str) for lbl in mh.layer_labels)
    assert isinstance(mh.layer_labels, list)
    assert all(isinstance(lbl, str) for lbl in mh.layer_labels)
    assert isinstance(mh.op_labels, list)
    assert all(isinstance(lbl, str) for lbl in mh.op_labels)


def test_output_descendants_complete_when_distances_disabled() -> None:
    """Disabling distance fields must not degrade serialized output ancestry."""

    trace = trace_fn(
        _SharedMultiOutputModel(),
        torch.ones(1),
        capture=tl.options.CaptureOptions(
            compute_input_output_distances=False, layers_to_save="all"
        ),
    )
    expected_outputs = set(trace.output_layers)
    shared = next(op for op in trace.ops if op.func_name == "__mul__")

    assert trace.input_ops[0].output_descendants == expected_outputs
    assert shared.output_descendants == expected_outputs


def test_exhaustive_saved_layer_count_uses_finalized_layer_list() -> None:
    """Exhaustive capture must report its saved unique-layer count after Step 11."""

    trace = trace_fn(
        nn.Sequential(nn.Linear(2, 2), nn.ReLU()),
        torch.ones(1, 2),
        capture=tl.options.CaptureOptions(layers_to_save="all"),
    )
    expected = len(
        {
            op.layer_label
            for op in trace.ops
            if op.has_saved_activation and not getattr(op, "is_orphan", False)
        }
    )

    assert trace.num_saved_layers == expected
    assert trace.num_saved_layers == trace.num_saved_ops


# =============================================================================
# Function args saving
# =============================================================================


def test_saved_args_populated(small_input):
    model = example_models.SimpleFF()
    mh = trace_fn(model, small_input, capture=tl.options.CaptureOptions(save_arg_values=True))
    assert mh.save_arg_values is True
    found = False
    for label in mh.layer_labels:
        entry = mh[label]
        if not entry.is_input and entry.saved_args is not None:
            found = True
            break
    assert found, "save_arg_values=True should populate saved_args"


def test_saved_args_not_populated(small_input):
    model = example_models.SimpleFF()
    mh = trace_fn(model, small_input)
    assert mh.save_arg_values is False


# =============================================================================
# Activation transform
# =============================================================================


def test_transform(small_input):
    model = example_models.SimpleFF()
    mh = trace_fn(model, small_input, save=tl.options.SaveOptions(activation_transform=torch.mean))
    for label in mh.layer_labels:
        entry = mh[label]
        if entry.transformed_out is not None:
            assert entry.transformed_out.dim() == 0, (
                f"Layer {label} should be scalar after torch.mean transform"
            )


# =============================================================================
# FLOPs: forward formula correctness
# =============================================================================


def test_flops_zero_cost_ops():
    """Zero-cost ops return 0."""
    shape = (2, 3, 4)
    for op in ["view", "reshape", "transpose", "contiguous", "clone", "unsqueeze"]:
        result = compute_forward_flops(op, shape, [], (), {})
        assert result == 0, f"{op} should return 0 FLOPs, got {result}"


def test_flops_elementwise_add():
    """add: 1 FLOP per element."""
    shape = (2, 3, 4)  # 24 elements
    result = compute_forward_flops("add", shape, [], (), {})
    assert result == 24


def test_flops_elementwise_sigmoid():
    """sigmoid: 4 FLOPs per element."""
    shape = (2, 3)  # 6 elements
    result = compute_forward_flops("sigmoid", shape, [], (), {})
    assert result == 24  # 6 * 4


def test_flops_elementwise_exp():
    """exp: 8 FLOPs per element."""
    shape = (10,)
    result = compute_forward_flops("exp", shape, [], (), {})
    assert result == 80  # 10 * 8


def test_flops_elementwise_gelu():
    """gelu: 14 FLOPs per element."""
    shape = (5,)
    result = compute_forward_flops("gelu", shape, [], (), {})
    assert result == 70  # 5 * 14


def test_flops_linear():
    """Linear: 2 * batch * in * out (+ bias)."""
    output_shape = (4, 10)
    param_shapes = [(10, 5)]  # weight only, no bias
    result = compute_forward_flops("linear", output_shape, param_shapes, (), {})
    assert result == 2 * 4 * 5 * 10  # 400


def test_flops_linear_with_bias():
    """Linear with bias adds output_numel."""
    output_shape = (4, 10)
    param_shapes = [(10, 5), (10,)]  # weight + bias
    result = compute_forward_flops("linear", output_shape, param_shapes, (), {})
    assert result == 2 * 4 * 5 * 10 + 40  # 440


def test_flops_conv2d():
    """Conv2d: 2 * output_numel * in_channels_per_group * kernel_size."""
    output_shape = (1, 16, 32, 32)
    param_shapes = [(16, 3, 3, 3)]
    result = compute_forward_flops("conv2d", output_shape, param_shapes, (), {})
    expected = 2 * (1 * 16 * 32 * 32) * 3 * (3 * 3)
    assert result == expected


def test_flops_matmul():
    """matmul: 2*M*K*N."""
    a = torch.randn(3, 4)
    b = torch.randn(4, 5)
    output_shape = (3, 5)
    result = compute_forward_flops("matmul", output_shape, [], (a, b), {})
    assert result == 2 * 3 * 4 * 5  # 120


@pytest.mark.parametrize(
    "a_shape,b_shape,output_shape,expected",
    [
        ((5, 3), (3,), (5,), 30),
        ((2, 4, 5), (5,), (2, 4), 80),
        ((3,), (3, 5), (5,), 30),
    ],
)
def test_flops_matmul_vector_operands(
    a_shape: tuple[int, ...],
    b_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    expected: int,
) -> None:
    """Matmul accounts for unit axes removed from vector-operand outputs."""

    a = torch.randn(a_shape)
    b = torch.randn(b_shape)

    result = compute_forward_flops("matmul", output_shape, [], (a, b), {})

    assert result == expected


def test_flops_bmm():
    """bmm: 2*batch*M*K*N."""
    a = torch.randn(2, 3, 4)
    b = torch.randn(2, 4, 5)
    output_shape = (2, 3, 5)
    result = compute_forward_flops("bmm", output_shape, [], (a, b), {})
    assert result == 2 * 2 * 3 * 4 * 5  # 240


def test_flops_batchnorm():
    """BatchNorm: 5 FLOPs per element."""
    output_shape = (2, 16, 8, 8)  # 2048 elements
    result = compute_forward_flops("batch_norm", output_shape, [], (), {})
    assert result == 5 * 2048


def test_flops_softmax():
    """Softmax: 5 FLOPs per element."""
    output_shape = (4, 100)  # 400 elements
    result = compute_forward_flops("softmax", output_shape, [], (), {})
    assert result == 5 * 400


def test_flops_reduction_sum():
    """Sum: input_numel FLOPs."""
    input_tensor = torch.randn(3, 4, 5)
    output_shape = ()
    result = compute_forward_flops("sum", output_shape, [], (input_tensor,), {})
    assert result == 60  # 3*4*5


def test_flops_dropout():
    """Dropout: 2 FLOPs per element."""
    output_shape = (10, 20)
    result = compute_forward_flops("dropout", output_shape, [], (), {})
    assert result == 2 * 200


def test_flops_embedding_zero():
    """Embedding: lookup only, 0 FLOPs."""
    output_shape = (4, 128)
    result = compute_forward_flops("embedding", output_shape, [], (), {})
    assert result == 0


def test_flops_unknown_op_returns_none():
    result = compute_forward_flops("totally_made_up_op", (3, 4), [], (), {})
    assert result is None


def test_flops_none_func_name():
    result = compute_forward_flops(None, (3, 4), [], (), {})
    assert result is None


def test_flops_none_output_shape_elementwise():
    result = compute_forward_flops("add", None, [], (), {})
    assert result is None


def test_flops_scalar_elementwise():
    """Scalar tensor: 1 element."""
    result = compute_forward_flops("add", (), [], (), {})
    assert result == 1


def test_flops_empty_tensor():
    """Empty tensor: 0 elements."""
    result = compute_forward_flops("add", (0,), [], (), {})
    assert result == 0


# =============================================================================
# FLOPs: backward estimation
# =============================================================================


def test_backward_flops_conv2d():
    """Conv backward = 2.0x forward."""
    result = compute_backward_flops("conv2d", 1000)
    assert result == 2000


def test_backward_flops_relu():
    """ReLU backward = 1.0x forward."""
    result = compute_backward_flops("relu", 500)
    assert result == 500


def test_backward_flops_sigmoid():
    """Sigmoid backward = 1.5x forward."""
    result = compute_backward_flops("sigmoid", 400)
    assert result == 600


def test_backward_flops_none_forward():
    """None forward -> None backward."""
    result = compute_backward_flops("conv2d", None)
    assert result is None


def test_backward_flops_unknown_op():
    """Unknown op gets default 1.0x multiplier."""
    result = compute_backward_flops("some_unknown_op", 100)
    assert result == 100


# =============================================================================
# FLOPs: integration tests
# =============================================================================


def test_flops_simple_linear_model():
    """FLOPs populated for a simple linear model."""
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
    x = torch.randn(4, 10)
    mh = trace_fn(model, x)
    flops_values = [entry.flops_forward for entry in mh.layer_list]
    non_none = [f for f in flops_values if f is not None]
    assert len(non_none) > 0, "No layers have FLOPs computed"


def test_flops_total_positive():
    """total_flops_forward should be positive for a model with compute."""
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU())
    x = torch.randn(4, 10)
    mh = trace_fn(model, x)
    assert mh.total_flops_forward > 0
    assert mh.total_flops_backward > 0
    assert mh.total_flops == mh.total_flops_forward + mh.total_flops_backward


def test_flops_by_type():
    """flops_by_op_type returns a dict with expected keys."""
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU())
    x = torch.randn(4, 10)
    mh = trace_fn(model, x)
    fbt = mh.flops_by_op_type()
    assert isinstance(fbt, dict)
    assert len(fbt) > 0
    for _layer_type, info in fbt.items():
        assert "forward" in info
        assert "backward" in info
        assert "count" in info


def test_flops_conv_model():
    """Conv model has FLOPs in expected range."""
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
    )
    x = torch.randn(1, 3, 32, 32)
    mh = trace_fn(model, x)
    # Conv2d(3, 16, 3): 2 * (1*16*32*32) * 3 * 9 = 884736
    assert mh.total_flops_forward > 800000


def test_flops_conv_transpose_uses_input_work_basis() -> None:
    """Grouped transposed convolution counts input scatter MACs exactly."""

    model = nn.ConvTranspose2d(
        6,
        4,
        kernel_size=3,
        stride=2,
        padding=1,
        groups=2,
        bias=False,
    )
    trace = trace_fn(model, torch.randn(1, 6, 8, 8))
    operation = next(entry for entry in trace.layer_list if entry.func_name == "conv_transpose2d")

    assert operation.shape == (1, 4, 15, 15)
    assert operation.flops_forward == 2 * 6 * 8 * 8 * (4 // 2) * 3 * 3


def test_flops_coverage_on_model():
    """At least 50% of non-input layers should have non-None FLOPs."""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.Sigmoid(),
    )
    x = torch.randn(4, 10)
    mh = trace_fn(model, x)
    total = len(mh.layer_list)
    with_flops = sum(1 for e in mh.layer_list if e.flops_forward is not None)
    coverage = with_flops / total if total > 0 else 0
    assert coverage >= 0.5, f"FLOPs coverage too low: {coverage:.1%}"


# =============================================================================
# Module training mode tracking (issue #52)
# =============================================================================


def test_module_training_modes_populated(small_input: torch.Tensor) -> None:
    """Module.training should capture the training flag."""
    model = example_models.SimpleFF()
    model.train()
    mh = trace_fn(model, small_input)
    assert mh.root_module.training is True
    assert mh.modules["self"].training is True


def test_module_training_modes_train_vs_eval():
    """Training mode should be captured correctly for each submodule."""
    model = nn.Sequential(nn.Linear(5, 5), nn.ReLU(), nn.Linear(5, 3))
    x = torch.rand(2, 5)

    model.train()
    mh_train = trace_fn(model, x)
    for ml in mh_train.modules:
        if ml.address != "self":
            assert ml.training is True, f"Module {ml.address} should be training=True"

    model.eval()
    mh_eval = trace_fn(model, x)
    for ml in mh_eval.modules:
        if ml.address != "self":
            assert ml.training is False, f"Module {ml.address} should be training=False"


@pytest.mark.slow
def test_flops_resnet18_range():
    """ResNet-18 ~ 1.8 GFLOPs. Check we're in the right ballpark."""
    torchvision = pytest.importorskip("torchvision")
    model = torchvision.models.resnet18(weights=None)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    mh = trace_fn(model, x)
    gflops = mh.total_flops_forward / 1e9
    assert 1.0 < gflops < 5.0, f"ResNet-18 FLOPs = {gflops:.2f}G, expected ~1.8G"


def test_flops_addbmm_batch():
    """addbmm should account for batch dimension."""
    bias = torch.randn(3, 5)
    batch1 = torch.randn(2, 3, 4)
    batch2 = torch.randn(2, 4, 5)
    output_shape = (3, 5)
    result = compute_forward_flops("addbmm", output_shape, [], (bias, batch1, batch2), {})
    # 2 * batch * m * k * n + output_numel = 2*2*3*4*5 + 15 = 255
    assert result == 2 * 2 * 3 * 4 * 5 + 15


def test_flops_baddbmm_batch():
    """baddbmm should account for batch dimension."""
    bias = torch.randn(2, 3, 5)
    batch1 = torch.randn(2, 3, 4)
    batch2 = torch.randn(2, 4, 5)
    output_shape = (2, 3, 5)
    result = compute_forward_flops("baddbmm", output_shape, [], (bias, batch1, batch2), {})
    # 2 * batch * m * k * n + output_numel = 2*2*3*4*5 + 30 = 270
    assert result == 2 * 2 * 3 * 4 * 5 + 30


def test_flops_einsum_matmul():
    """einsum with matmul-like subscripts should compute FLOPs."""
    a = torch.randn(3, 4)
    b = torch.randn(4, 5)
    output_shape = (3, 5)
    result = compute_forward_flops("einsum", output_shape, [], ("ij,jk->ik", a, b), {})
    assert result == 2 * 3 * 4 * 5  # 120


def test_flops_einsum_attention_contraction() -> None:
    """Attention einsum contracts the shared feature axis, not a matrix tail axis."""

    class _AttentionEinsum(nn.Module):
        """Two-operand attention-score einsum."""

        def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
            """Contract query and key over their shared feature axis."""

            return torch.einsum("bid,bjd->bij", query, key)

    q = torch.randn(2, 5, 4)
    k = torch.randn(2, 7, 4)
    output_shape = (2, 5, 7)

    direct = compute_forward_flops("einsum", output_shape, [], ("bid,bjd->bij", q, k), {})
    nested = compute_forward_flops("einsum", output_shape, [], ("bid,bjd->bij", (q, k)), {})

    assert direct == 2 * 2 * 5 * 7 * 4
    assert nested == 2 * 2 * 5 * 7 * 4
    trace = trace_fn(_AttentionEinsum(), (q, k))
    operation = next(entry for entry in trace.layer_list if entry.func_name == "einsum")
    assert operation.flops_forward == 2 * 2 * 5 * 7 * 4


def test_unknown_flops_survive_trace_materialization() -> None:
    """An unregistered operation remains unknown instead of becoming zero FLOPs."""

    class _PadModel(nn.Module):
        """Model containing an intentionally unregistered pad operation."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Pad the final dimension."""

            return torch.nn.functional.pad(x, (1, 1))

    trace = trace_fn(_PadModel(), torch.randn(2, 3))
    operation = next(entry for entry in trace.layer_list if entry.func_name == "pad")

    assert operation.flops_forward is None
    assert operation.flops_backward is None
    assert trace.total_flops_forward == 0


def test_rerun_refreshes_shape_derived_flops_everywhere() -> None:
    """A changed batch size refreshes every public route to one operation's FLOPs."""

    class _LinearModel(nn.Module):
        """One biased linear operation with batch-dependent FLOPs."""

        def __init__(self) -> None:
            """Initialize a four-to-eight feature projection."""

            super().__init__()
            self.linear = nn.Linear(4, 8)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Apply the projection."""

            return self.linear(x)

    model = _LinearModel()
    trace = trace_fn(model, torch.randn(2, 4))
    captured = next(entry for entry in trace.layer_list if entry.func_name == "linear")
    assert captured.flops_forward == 144

    with pytest.warns(UserWarning, match="Tensor shape changed"):
        result = trace.run(inputs=torch.randn(8, 4))
    refreshed = next(entry for entry in result.trace.layer_list if entry.func_name == "linear")

    assert refreshed.shape == (8, 8)
    assert refreshed.flops_forward == 576
    assert result.trace[refreshed.layer_label].flops_forward == 576
    assert result.trace.total_flops_forward == sum(
        entry.flops_forward for entry in result.trace.layer_list if entry.flops_forward is not None
    )


def test_refresh_shape_change_emits_single_aggregated_warning() -> None:
    """A multi-layer shape-change refresh emits ONE aggregated warning (B8-36).

    Per-layer warnings with the label interpolated into the message defeated
    Python's warning dedup: hundreds of warnings per refresh on a real CNN, and
    ``-W error`` aborted the refresh at layer one. The refresh now aggregates to
    one warning naming the changed-layer count.
    """

    import re
    import warnings as warnings_module

    class _TwoStage(nn.Module):
        """Two linear stages so several layers change shape at once."""

        def __init__(self) -> None:
            """Initialize two chained projections."""

            super().__init__()
            self.linear1 = nn.Linear(4, 8)
            self.linear2 = nn.Linear(8, 8)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Apply linear -> relu -> linear."""

            return self.linear2(torch.relu(self.linear1(x)))

    model = _TwoStage()
    trace = trace_fn(model, torch.randn(2, 4))

    with warnings_module.catch_warnings(record=True) as caught:
        warnings_module.simplefilter("always")
        trace.run(inputs=torch.randn(8, 4))

    shape_warnings = [w for w in caught if "Tensor shape changed" in str(w.message)]
    assert len(shape_warnings) == 1
    message = str(shape_warnings[0].message)
    match = re.search(r"Tensor shape changed for (\d+) layer", message)
    assert match is not None
    assert int(match.group(1)) >= 2


def test_flops_pool_with_kernel():
    """Pooling should account for kernel_size."""
    input_tensor = torch.randn(1, 16, 32, 32)
    output_shape = (1, 16, 16, 16)
    out_numel = 1 * 16 * 16 * 16  # 4096
    # kernel_size=2 means 2*2=4 comparisons per output element
    result = compute_forward_flops("max_pool2d", output_shape, [], (input_tensor, (2, 2)), {})
    assert result == out_numel * 4

    # kernel_size as int
    result_int = compute_forward_flops("max_pool2d", output_shape, [], (input_tensor, 3), {})
    assert result_int == out_numel * 3


def test_flops_sdpa():
    """scaled_dot_product_attention has its own correct handler."""
    q = torch.randn(2, 8, 10, 64)
    k = torch.randn(2, 8, 10, 64)
    v = torch.randn(2, 8, 10, 64)
    output_shape = (2, 8, 10, 64)
    result = compute_forward_flops("scaled_dot_product_attention", output_shape, [], (q, k, v), {})
    assert result is not None
    assert result > 0


# =============================================================================
# FuncCallLocation tests
# =============================================================================


def _get_code_context_with_flag(small_input, save_code_context: bool):
    """Helper: return a non-input layer's code_context for either source-loading mode."""
    model = example_models.SimpleFF()
    mh = trace_fn(
        model,
        small_input,
        capture=tl.options.CaptureOptions(save_code_context=save_code_context),
    )
    for label in mh.layer_labels:
        entry = mh[label]
        if not entry.is_input:
            return entry.code_context, mh
    raise RuntimeError("No non-input layer found")


def _get_code_context(small_input):
    """Helper: run a model and return the code_context from a non-input layer."""
    return _get_code_context_with_flag(small_input, save_code_context=True)


# --- Class structure ---


def test_code_context_returns_list_of_func_call_locations(small_input):
    stack, _ = _get_code_context(small_input)
    assert isinstance(stack, list)
    assert len(stack) > 0
    for loc in stack:
        assert isinstance(loc, FuncCallLocation)


def test_func_call_location_fields_populated(small_input):
    stack, _ = _get_code_context(small_input)
    loc = stack[0]
    assert isinstance(loc.file, str)
    assert isinstance(loc.line_number, int)
    assert isinstance(loc.func_name, str)
    assert isinstance(loc.call_line, str)
    assert isinstance(loc.code_context, (list, type(None)))
    assert isinstance(loc.source_context, str)
    assert isinstance(loc.code_context_labeled, str)
    assert isinstance(loc.num_context_lines, int)


def test_optional_fields_are_str_or_none(small_input):
    stack, _ = _get_code_context(small_input)
    for loc in stack:
        assert loc.func_signature is None or isinstance(loc.func_signature, str)
        assert loc.func_docstring is None or isinstance(loc.func_docstring, str)


# --- Content correctness ---


def test_forward_frame_present(small_input):
    stack, _ = _get_code_context(small_input)
    func_names = [loc.func_name for loc in stack]
    assert "forward" in func_names


def test_no_torchlens_internals_in_stack(small_input):
    import os

    torchlens_pkg_dir = os.path.dirname(os.path.abspath(torchlens.__file__))
    stack, _ = _get_code_context(small_input)
    for loc in stack:
        assert not loc.file.startswith(torchlens_pkg_dir), (
            f"Internal file {loc.file} should not appear in stack"
        )


def test_call_line_is_stripped(small_input):
    stack, _ = _get_code_context(small_input)
    for loc in stack:
        if loc.call_line:
            assert loc.call_line == loc.call_line.strip()


# --- Dunders ---


def test_repr_contains_key_info(small_input):
    stack, _ = _get_code_context(small_input)
    # Find a frame with code context
    loc = None
    for entry in stack:
        if entry.code_context is not None:
            loc = entry
            break
    assert loc is not None
    r = repr(loc)
    assert loc.file in r
    assert str(loc.line_number) in r
    assert loc.func_name in r
    assert "--->" in r


def test_repr_source_unavailable():
    loc = FuncCallLocation(
        file="test.py",
        line_number=1,
        func_name="test",
        func_signature=None,
        func_docstring=None,
        call_line="",
        code_context=None,
        source_context="None",
        code_context_labeled="",
        num_context_lines=0,
    )
    r = repr(loc)
    assert "source unavailable" in r


def test_func_call_location_no_source_state_with_save_code_context_off(small_input, monkeypatch):
    stack, mh = _get_code_context_with_flag(small_input, save_code_context=False)
    assert isinstance(stack, list)
    assert len(stack) > 0
    captured_entries = [
        entry for entry in mh.layer_list if not entry.is_input and entry.func_name != "none"
    ]
    assert all(len(entry.code_context) > 0 for entry in captured_entries)
    for entry in captured_entries:
        for frame in entry.code_context:
            assert frame.file is not None
            assert frame.line_number is not None
            assert frame.code_firstlineno is not None
            assert frame.source_loading_enabled is False

    loc = stack[0]
    assert loc.file is not None
    assert loc.line_number is not None
    assert loc.code_firstlineno is not None
    assert loc.source_loading_enabled is False
    assert loc._source_loaded is True
    assert loc._frame_func_obj is None

    accessed_files = []
    original_getlines = linecache.getlines

    def _tracking_getlines(*args, **kwargs):
        accessed_files.append(args[0])
        return original_getlines(*args, **kwargs)

    with monkeypatch.context() as local_patch:
        local_patch.setattr(linecache, "getlines", _tracking_getlines)

        assert loc.source_context == "None"
        assert loc.code_context is None
        assert loc.code_context_labeled == ""
        assert loc.call_line == ""
        assert loc.num_context_lines == 0
        assert loc.func_signature is None
        assert loc.func_docstring is None
        assert len(loc) == 0
        with pytest.raises(IndexError):
            _ = loc[0]
        assert repr(loc).endswith("code: source unavailable")
        assert accessed_files == []

    mh_roundtrip = pickle.loads(pickle.dumps(mh))
    roundtrip_stack = next(
        entry.code_context for entry in mh_roundtrip.layer_list if not entry.is_input
    )
    assert len(roundtrip_stack) > 0
    assert roundtrip_stack[0].source_loading_enabled is False
    assert roundtrip_stack[0].source_context == "None"


def test_getitem_returns_context_line(small_input):
    stack, _ = _get_code_context(small_input)
    loc = None
    for entry in stack:
        if entry.code_context is not None and len(entry.code_context) > 0:
            loc = entry
            break
    assert loc is not None
    assert loc[0] == loc.code_context[0]


def test_getitem_slice(small_input):
    stack, _ = _get_code_context(small_input)
    loc = None
    for entry in stack:
        if entry.code_context is not None and len(entry.code_context) >= 3:
            loc = entry
            break
    assert loc is not None
    sliced = loc[1:3]
    assert isinstance(sliced, list)
    assert len(sliced) == 2
    assert sliced == loc.code_context[1:3]


def test_len_matches_code_context(small_input):
    stack, _ = _get_code_context(small_input)
    for loc in stack:
        if loc.code_context is not None:
            assert len(loc) == len(loc.code_context) == loc.num_context_lines


# --- Context lines parameter ---


def test_default_num_context_lines(small_input):
    stack, _ = _get_code_context(small_input)
    for loc in stack:
        if loc.code_context is not None:
            assert loc.num_context_lines == 15  # 7 + 1 + 7
            break


def test_custom_num_context_lines(small_input):
    model = example_models.SimpleFF()
    mh = trace_fn(
        model,
        small_input,
        capture=tl.options.CaptureOptions(source_context_lines=3, save_code_context=True),
    )
    for label in mh.layer_labels:
        entry = mh[label]
        if not entry.is_input and entry.code_context:
            for loc in entry.code_context:
                if loc.code_context is not None:
                    assert loc.num_context_lines == 7  # 3 + 1 + 3
                    return
    pytest.fail(
        "No non-input layer with code context found. SimpleFF traced with "
        "save_code_context=True is fully under this test's control, so an empty "
        "code-context surface is a capture regression, not an environment limit "
        "(hardened from a silent fall-through skip, R79 skip audit 2026-08-15)."
    )


def test_num_context_lines_stored_on_trace(small_input):
    model = example_models.SimpleFF()
    mh = trace_fn(model, small_input, capture=tl.options.CaptureOptions(source_context_lines=5))
    assert mh.num_context_lines == 5


# --- Labeled code context ---


def test_code_context_labeled_has_arrow(small_input):
    stack, _ = _get_code_context(small_input)
    for loc in stack:
        if loc.code_context is not None:
            assert loc.code_context_labeled.count("  --->  ") == 1
            break


def test_code_context_labeled_arrow_points_to_call_line(small_input):
    stack, _ = _get_code_context(small_input)
    for loc in stack:
        if loc.code_context is not None and loc.call_line:
            for line in loc.code_context_labeled.split("\n"):
                if "--->" in line:
                    assert loc.call_line in line
                    break
            break


# =============================================================================
# Meta-validation: corrupt outs and verify detection
# =============================================================================


@pytest.fixture
def valid_mh_and_ground_truth():
    """Return a valid (Trace, ground_truth_tensors) pair for SimpleFF."""
    model = example_models.SimpleFF()
    x = torch.rand(2, 3, 32, 32)
    mh = trace_fn(
        model,
        x,
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
    )
    ground_truth = [mh[label].out.clone() for label in mh.output_layers]
    return mh, ground_truth


def test_uncorrupted_ops(valid_mh_and_ground_truth):
    """Sanity check: validation ops when nothing is corrupted."""
    mh, ground_truth = valid_mh_and_ground_truth
    assert mh.validate_forward_pass(ground_truth) is True


def test_corrupt_output_outs(valid_mh_and_ground_truth):
    """Replacing the output layer's out with random data should fail."""
    mh, ground_truth = valid_mh_and_ground_truth
    output_label = mh.output_layers[0]
    output_op = mh.layers[output_label].ops[0]
    original = output_op.out
    output_op.out = torch.randn_like(original)
    assert mh.validate_forward_pass(ground_truth) is False


def test_corrupt_intermediate_outs(valid_mh_and_ground_truth):
    """Replacing a non-output, non-input layer's out should fail."""
    mh, ground_truth = valid_mh_and_ground_truth
    # Find an intermediate layer (not input, not output)
    intermediate = [
        label
        for label in mh.layer_labels
        if label not in mh.input_layers and label not in mh.output_layers
    ]
    assert len(intermediate) > 0, "No intermediate layers found"
    target = intermediate[0]
    target_op = mh.layers[target].ops[0]
    original = target_op.out
    target_op.out = torch.randn_like(original)
    assert mh.validate_forward_pass(ground_truth) is False


def test_swap_two_layers_outs(valid_mh_and_ground_truth):
    """Swapping out between two non-output layers should fail."""
    mh, ground_truth = valid_mh_and_ground_truth
    non_output = [label for label in mh.layer_labels if label not in mh.output_layers]
    assert len(non_output) >= 2, "Need at least 2 non-output layers to swap"
    a, b = non_output[0], non_output[1]
    op_a = mh.layers[a].ops[0]
    op_b = mh.layers[b].ops[0]
    ta = op_a.out.clone()
    tb = op_b.out.clone()
    # Only swap if they're different shapes or values — otherwise the swap is a no-op
    if ta.shape == tb.shape and torch.equal(ta, tb):
        pytest.skip("Layers have identical tensors; swap is invisible")
    op_a.out = tb
    op_b.out = ta
    assert mh.validate_forward_pass(ground_truth) is False


def test_zero_out_outs(valid_mh_and_ground_truth):
    """Zeroing out a layer's out should fail."""
    mh, ground_truth = valid_mh_and_ground_truth
    non_output = [label for label in mh.layer_labels if label not in mh.output_layers]
    assert len(non_output) > 0
    target = non_output[0]
    target_op = mh.layers[target].ops[0]
    target_op.out = torch.zeros_like(target_op.out)
    assert mh.validate_forward_pass(ground_truth) is False


def test_add_noise_to_outs(valid_mh_and_ground_truth):
    """Adding gaussian noise to a layer's out should fail."""
    mh, ground_truth = valid_mh_and_ground_truth
    non_output = [label for label in mh.layer_labels if label not in mh.output_layers]
    assert len(non_output) > 0
    target = non_output[0]
    target_op = mh.layers[target].ops[0]
    original = target_op.out
    target_op.out = original + torch.randn_like(original) * 0.1
    assert mh.validate_forward_pass(ground_truth) is False


def test_scale_outs(valid_mh_and_ground_truth):
    """Scaling a layer's out by a large scalar should fail."""
    mh, ground_truth = valid_mh_and_ground_truth
    non_output = [label for label in mh.layer_labels if label not in mh.output_layers]
    assert len(non_output) > 0
    target = non_output[0]
    target_op = mh.layers[target].ops[0]
    target_op.out = target_op.out * 100.0
    assert mh.validate_forward_pass(ground_truth) is False


def test_wrong_shape_outs(valid_mh_and_ground_truth):
    """Replacing out with a wrong-shaped tensor should fail."""
    mh, ground_truth = valid_mh_and_ground_truth
    output_label = mh.output_layers[0]
    mh.layers[output_label].ops[0].out = torch.randn(1, 1)
    assert mh.validate_forward_pass(ground_truth) is False


def test_corrupt_saved_args(valid_mh_and_ground_truth):
    """Modifying saved function arguments (saved_args) should fail."""
    mh, ground_truth = valid_mh_and_ground_truth
    # Find a non-input layer that has saved_args with tensors
    for label in mh.layer_labels:
        entry = mh.layers[label].ops[0]
        if entry.is_input:
            continue
        if entry.saved_args and any(isinstance(a, torch.Tensor) for a in entry.saved_args):
            for i, arg in enumerate(entry.saved_args):
                if isinstance(arg, torch.Tensor):
                    corrupted_args = list(entry.saved_args)
                    corrupted_args[i] = torch.randn_like(arg)
                    entry.saved_args = tuple(corrupted_args)
                    assert mh.validate_forward_pass(ground_truth) is False
                    return
    pytest.fail(
        "No layer with tensor saved_args found. The fixture traces with "
        "save_arg_values enabled on a model with tensor-consuming ops, so an "
        "empty saved_args surface is a capture regression, not an environment "
        "limit (hardened from a silent fall-through skip, R79 skip audit "
        "2026-08-15)."
    )


# =============================================================================
# Conditional Branch Detection (Bug #88 fix + THEN labeling)
# =============================================================================


class TestConditionalBranchDetection:
    """Tests for conditional branch detection: backward-only IF flood + THEN detection."""

    # --- Shared helpers ---

    @staticmethod
    def _log(model, x, save_code_context=False):
        return trace_fn(
            model, x, capture=tl.options.CaptureOptions(save_code_context=save_code_context)
        )

    @staticmethod
    def _cond_input():
        """Negative input so ConditionalBranching takes the else branch."""
        return -torch.ones(2, 3, 32, 32)

    @staticmethod
    def _pos_input():
        """Positive input so ConditionalBranching takes the if branch."""
        return torch.ones(2, 3, 32, 32)

    # --- Core detection tests ---

    def test_if_branch_detected(self):
        """ConditionalBranching has is_in_conditional_body=True nodes."""
        model = example_models.ConditionalBranching()
        mh = self._log(model, self._cond_input())
        found = any(mh[label].is_in_conditional_body for label in mh.layer_labels)
        assert found, "Should detect conditional branch nodes"

    def test_then_branch_detected(self):
        """ConditionalBranching with save_code_context has conditional_then_children.

        Uses positive input so the if-body (THEN branch) actually executes.
        """
        model = example_models.ConditionalBranching()
        mh = self._log(model, self._pos_input(), save_code_context=True)
        found = any(len(mh[label].conditional_then_children) > 0 for label in mh.layer_labels)
        assert found, "Should detect THEN branch children with save_code_context"

    def test_branch_start_has_both_if_and_then(self):
        """Branch start has both IF and THEN children when if-body executes."""
        model = example_models.ConditionalBranching()
        mh = self._log(model, self._pos_input(), save_code_context=True)
        for label in mh.layer_labels:
            entry = mh[label]
            if entry.conditional_entry_children:
                assert len(entry.conditional_then_children) > 0, (
                    f"Branch start {label} has IF children but no THEN children"
                )

    def test_if_and_then_children_disjoint(self):
        """No overlap between IF and THEN children sets."""
        model = example_models.ConditionalBranching()
        mh = self._log(model, self._pos_input(), save_code_context=True)
        for label in mh.layer_labels:
            entry = mh[label]
            if_set = set(entry.conditional_entry_children)
            then_set = set(entry.conditional_then_children)
            overlap = if_set & then_set
            assert len(overlap) == 0, f"IF and THEN overlap at {label}: {overlap}"

    def test_terminal_bool_exists(self):
        """At least one is_terminal_bool=True node."""
        model = example_models.ConditionalBranching()
        mh = self._log(model, self._cond_input())
        found = any(mh[label].is_terminal_bool for label in mh.layer_labels)
        assert found, "Should have at least one terminal bool layer"

    # --- False positive tests (Bug #88) ---

    def test_no_false_cond_branch_without_condition(self):
        """Model with no conditions has zero is_in_conditional_body nodes."""
        model = example_models.SimpleFF()
        x = torch.rand(2, 3, 32, 32)
        mh = self._log(model, x)
        cond_nodes = [label for label in mh.layer_labels if mh[label].is_in_conditional_body]
        assert len(cond_nodes) == 0, f"SimpleFF should have no cond nodes: {cond_nodes}"

    def test_non_conditional_layers_not_marked(self):
        """In ConditionalBranching, output-ancestor non-branch layers NOT marked."""
        model = example_models.ConditionalBranching()
        mh = self._log(model, self._cond_input())
        for label in mh.layer_labels:
            entry = mh[label]
            if not entry.conditional_role_stacks and not entry.conditional_entry_children:
                assert entry.is_in_conditional_body is False, (
                    f"Output ancestor {label} falsely marked is_in_conditional_body"
                )

    def test_condition_chain_no_spill(self):
        """ConditionalChainedBools doesn't spill markings to main graph."""
        model = example_models.ConditionalChainedBools()
        x = torch.ones(2, 3, 32, 32)
        mh = self._log(model, x)
        for label in mh.layer_labels:
            entry = mh[label]
            # Output-ancestor nodes that are NOT branch-starts should not be marked
            if not entry.conditional_role_stacks and not entry.conditional_entry_children:
                assert entry.is_in_conditional_body is False, (
                    f"Node {label} falsely marked is_in_conditional_body (Bug #88)"
                )

    # --- Edge/model log tests ---

    def test_conditional_branch_edges_populated(self):
        """trace.conditional_branch_edges is non-empty."""
        model = example_models.ConditionalBranching()
        mh = self._log(model, self._cond_input())
        assert len(mh.conditional_branch_edges) > 0

    def test_conditional_then_arm_edges_populated(self):
        """Canonical THEN arm edges non-empty with save_code_context."""
        model = example_models.ConditionalBranching()
        mh = self._log(model, self._pos_input(), save_code_context=True)
        then_edges = [
            edge
            for (_, branch_kind), edges in mh.conditional_arm_entry_edges.items()
            if branch_kind == "then"
            for edge in edges
        ]
        assert len(then_edges) > 0

    def test_edges_reference_valid_labels(self):
        """All labels in edge tuples exist in trace.layer_labels."""
        model = example_models.ConditionalBranching()
        mh = self._log(model, self._pos_input(), save_code_context=True)
        all_labels = set(mh.layer_labels)
        for parent, child in mh.conditional_branch_edges:
            assert parent in all_labels, f"IF edge parent {parent} not in layer_labels"
            assert child in all_labels, f"IF edge child {child} not in layer_labels"
        for edges in mh.conditional_arm_entry_edges.values():
            for parent, child in edges:
                assert parent in all_labels, f"arm edge parent {parent} not in layer_labels"
                assert child in all_labels, f"arm edge child {child} not in layer_labels"

    # --- Post-validation tests ---

    def test_no_branch_bool_unused(self):
        """ConditionalNoBranch: post-validation clears false IF markings."""
        model = example_models.ConditionalNoBranch()
        x = torch.rand(2, 3, 32, 32)
        mh = self._log(model, x, save_code_context=True)
        # After post-validation, if no ast.If found, IF markings cleared
        branch_starts = [
            label for label in mh.layer_labels if len(mh[label].conditional_entry_children) > 0
        ]
        assert len(branch_starts) == 0, (
            f"ConditionalNoBranch should have no branch starts after post-validation: {branch_starts}"
        )

    def test_always_true_has_then_branch(self):
        """ConditionalAlwaysTrue still detects THEN branch."""
        model = example_models.ConditionalAlwaysTrue()
        x = torch.rand(2, 3, 32, 32)
        mh = self._log(model, x, save_code_context=True)
        found = any(len(mh[label].conditional_then_children) > 0 for label in mh.layer_labels)
        assert found, "ConditionalAlwaysTrue should detect THEN branch"

    # --- Complex scenario tests ---

    def test_nested_conditions(self):
        """ConditionalNested has conditional branching at both levels."""
        model = example_models.ConditionalNested()
        x = torch.rand(2, 3, 32, 32)
        mh = self._log(model, x)
        assert mh.has_conditional_branching is True

    def test_multiple_branches_independent(self):
        """ConditionalMultipleBranches has 2 distinct branch starts."""
        model = example_models.ConditionalMultipleBranches()
        x = torch.ones(2, 3, 32, 32)
        mh = self._log(model, x)
        branch_starts = [
            label for label in mh.layer_labels if len(mh[label].conditional_entry_children) > 0
        ]
        assert len(branch_starts) >= 2, (
            f"Expected >= 2 branch starts, got {len(branch_starts)}: {branch_starts}"
        )

    def test_conditional_with_modules(self):
        """ConditionalWithModules correctly labels module-based branches."""
        model = example_models.ConditionalWithModules()
        x = torch.rand(2, 5)
        mh = self._log(model, x)
        assert mh.has_conditional_branching is True

    # --- Visualization integration tests ---

    def test_if_label_in_visualization(self):
        """Rendered graph contains 'IF' edge label."""
        import os
        import tempfile

        from torchlens.visualization import show_model_graph

        model = example_models.ConditionalBranching()
        x = self._cond_input()
        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = os.path.join(tmpdir, "cond_if_test")
            show_model_graph(
                model,
                x,
                view="unrolled",
                visualization=tl.options.VisualizationOptions(
                    save_only=True, container_path=outpath, file_format="dot"
                ),
            )
            dot_file = outpath + ".dot"
            if os.path.exists(dot_file):
                with open(dot_file) as f:
                    dot_content = f.read()
                assert "IF" in dot_content, "Graph should contain IF edge label"

    def test_then_label_in_visualization(self):
        """Rendered graph contains 'THEN' edge label with save_code_context."""
        import os
        import tempfile

        model = example_models.ConditionalBranching()
        x = self._pos_input()
        mh = trace_fn(model, x, capture=tl.options.CaptureOptions(save_code_context=True))
        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = os.path.join(tmpdir, "cond_then_test")
            mh.draw(
                vis_mode="unrolled",
                vis_outpath=outpath,
                vis_save_only=True,
                vis_fileformat="dot",
            )
            dot_file = outpath + ".dot"
            if os.path.exists(dot_file):
                with open(dot_file) as f:
                    dot_content = f.read()
                assert "THEN" in dot_content, "Graph should contain THEN edge label"

    # --- Rolled graph tests ---

    def test_rolled_graph_conditional_edges(self):
        """Rolled view preserves IF/THEN labels."""
        import os
        import tempfile

        from torchlens.visualization import show_model_graph

        model = example_models.ConditionalBranching()
        x = self._cond_input()
        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = os.path.join(tmpdir, "cond_rolled_test")
            show_model_graph(
                model,
                x,
                view="rolled",
                visualization=tl.options.VisualizationOptions(
                    save_only=True, container_path=outpath, file_format="dot"
                ),
            )
            dot_file = outpath + ".dot"
            if os.path.exists(dot_file):
                with open(dot_file) as f:
                    dot_content = f.read()
                assert "IF" in dot_content, "Rolled graph should contain IF edge label"

    def test_cond_fields_survive_deepcopy(self):
        """Fields persist through Trace deepcopy cycle."""
        model = example_models.ConditionalBranching()
        mh = self._log(model, self._pos_input(), save_code_context=True)
        mh2 = copy.deepcopy(mh)
        assert mh2.conditional_branch_edges == mh.conditional_branch_edges
        assert mh2.conditional_arm_entry_edges == mh.conditional_arm_entry_edges
        for label in mh.layer_labels:
            assert mh2[label].conditional_entry_children == mh[label].conditional_entry_children
            assert mh2[label].conditional_then_children == mh[label].conditional_then_children

    # --- Fallback tests (without source context) ---

    def test_if_detection_works_without_source_context(self):
        """Bug #88 fix works without save_code_context."""
        model = example_models.ConditionalBranching()
        mh = self._log(model, self._cond_input(), save_code_context=False)
        found = any(mh[label].is_in_conditional_body for label in mh.layer_labels)
        assert found, "IF detection should work without source context"

    def test_then_empty_without_source_context(self):
        """conditional_then_children stays empty when save_code_context=False."""
        model = example_models.ConditionalBranching()
        mh = self._log(model, self._cond_input(), save_code_context=False)
        for label in mh.layer_labels:
            assert len(mh[label].conditional_then_children) == 0, (
                f"THEN children should be empty without source context: {label}"
            )


def test_transpose_positional_args_expose_salient_dims():
    """Positional torch.transpose calls expose dim0/dim1 via the arg-name fallback."""
    import torch
    from torch import nn

    import torchlens as tl

    class _TransposeModel(nn.Module):
        def forward(self, x):
            return torch.transpose(x, 0, 1).contiguous()

    trace = tl.trace(
        _TransposeModel(),
        torch.randn(2, 3),
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
    )
    op = next(o for o in trace.layer_list if "transpose" in o.func_name)
    assert op.func_config.get("dim0") == 0
    assert op.func_config.get("dim1") == 1
