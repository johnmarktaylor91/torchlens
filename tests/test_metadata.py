"""Comprehensive metadata field testing for ModelHistory and TensorLogEntry.

Uses small/fast models from example_models.py to verify that all key metadata
fields are populated correctly across different model types.
"""

import pytest
import torch
import torch.nn as nn

import example_models
from torchlens import log_forward_pass
from torchlens.flops import (
    BACKWARD_MULTIPLIERS,
    ELEMENTWISE_FLOPS,
    SPECIALTY_HANDLERS,
    ZERO_FLOPS_OPS,
    compute_backward_flops,
    compute_forward_flops,
)


# =============================================================================
# ModelHistory fields
# =============================================================================


def test_general_info_fields(small_input):
    model = example_models.SimpleFF()
    mh = log_forward_pass(model, small_input)
    assert isinstance(mh.model_name, str)
    assert len(mh.model_name) > 0
    assert mh._pass_finished is True
    assert isinstance(mh.num_operations, int)
    assert mh.num_operations > 0


def test_model_structure_non_recurrent(small_input):
    model = example_models.SimpleFF()
    mh = log_forward_pass(model, small_input)
    assert mh.model_is_recurrent is False
    assert mh.model_is_branching is False


def test_model_structure_branching(small_input):
    model = example_models.SimpleBranching()
    mh = log_forward_pass(model, small_input)
    assert mh.model_is_branching is True


def test_model_structure_recurrent(input_2d):
    model = example_models.RecurrentParamsSimple()
    mh = log_forward_pass(model, input_2d)
    assert mh.model_is_recurrent is True


def test_model_structure_conditional():
    model = example_models.ConditionalBranching()
    model_input = -torch.ones(6, 3, 224, 224)
    mh = log_forward_pass(model, model_input)
    assert mh.model_has_conditional_branching is True


def test_layer_tracking_fields(small_input):
    model = example_models.SimpleFF()
    mh = log_forward_pass(model, small_input)
    assert isinstance(mh.layer_list, list)
    assert len(mh.layer_list) > 0
    assert isinstance(mh.layer_labels, list)
    assert len(mh.layer_labels) > 0
    assert isinstance(mh.layer_dict_main_keys, dict)
    assert isinstance(mh.layer_dict_all_keys, dict)


def test_input_output_layers(small_input):
    model = example_models.SimpleFF()
    mh = log_forward_pass(model, small_input)
    assert isinstance(mh.input_layers, list)
    assert len(mh.input_layers) >= 1
    assert isinstance(mh.output_layers, list)
    assert len(mh.output_layers) >= 1


def test_buffer_layers():
    model = example_models.BufferModel()
    model_input = torch.rand(12, 12)
    mh = log_forward_pass(model, model_input)
    assert isinstance(mh.buffer_layers, list)
    assert len(mh.buffer_layers) > 0


def test_tensor_info_fields(small_input):
    model = example_models.SimpleFF()
    mh = log_forward_pass(model, small_input)
    assert isinstance(mh.num_tensors_total, int)
    assert mh.num_tensors_total > 0
    assert isinstance(mh.tensor_fsize_total, (int, float))
    assert mh.tensor_fsize_total > 0
    assert isinstance(mh.tensor_fsize_total_nice, str)
    assert len(mh.tensor_fsize_total_nice) > 0


def test_param_info_fields(small_input):
    model = example_models.BatchNormModel()
    mh = log_forward_pass(model, small_input)
    assert isinstance(mh.total_param_tensors, int)
    assert isinstance(mh.total_params, int)
    assert mh.total_params > 0


def test_module_info_fields(small_input):
    model = example_models.NestedModules()
    mh = log_forward_pass(model, small_input)
    assert isinstance(mh.module_addresses, list)
    assert len(mh.module_addresses) > 0
    assert isinstance(mh.module_types, dict)
    assert isinstance(mh.module_passes, list)
    assert isinstance(mh.module_children, dict)


def test_time_fields(small_input):
    model = example_models.SimpleFF()
    mh = log_forward_pass(model, small_input)
    assert isinstance(mh.elapsed_time_total, float)
    assert mh.elapsed_time_total > 0
    assert isinstance(mh.elapsed_time_setup, float)
    assert isinstance(mh.elapsed_time_forward_pass, float)
    assert isinstance(mh.elapsed_time_cleanup, float)
    assert isinstance(mh.elapsed_time_torchlens_logging, float)


def test_multi_input_layers():
    model = example_models.MultiInputs()
    inputs = [
        torch.rand(6, 3, 224, 224),
        torch.rand(6, 3, 224, 224),
        torch.rand(6, 3, 224, 224),
    ]
    mh = log_forward_pass(model, inputs)
    assert len(mh.input_layers) == 3


def test_multi_output_layers(small_input):
    model = example_models.MultiOutputs()
    mh = log_forward_pass(model, small_input)
    assert len(mh.output_layers) >= 2


def test_internally_initialized_layers(small_input):
    model = example_models.SimpleInternallyGenerated()
    mh = log_forward_pass(model, small_input)
    assert isinstance(mh.internally_initialized_layers, list)
    assert len(mh.internally_initialized_layers) > 0


def test_equivalent_operations(input_2d):
    model = example_models.RecurrentParamsSimple()
    mh = log_forward_pass(model, input_2d)
    assert isinstance(mh.equivalent_operations, dict)
    assert len(mh.equivalent_operations) > 0


# =============================================================================
# TensorLogEntry fields
# =============================================================================


def test_label_fields(small_input):
    model = example_models.SimpleFF()
    mh = log_forward_pass(model, small_input)
    entry = mh[0]
    assert isinstance(entry.layer_label, str)
    assert isinstance(entry.layer_type, str)
    assert isinstance(entry.pass_num, int)
    assert isinstance(entry.lookup_keys, list)
    assert len(entry.lookup_keys) > 0


def test_input_layer_properties(small_input):
    model = example_models.SimpleFF()
    mh = log_forward_pass(model, small_input)
    input_entry = None
    for label in mh.layer_labels:
        e = mh[label]
        if e.is_input_layer:
            input_entry = e
            break
    assert input_entry is not None
    assert input_entry.is_input_layer is True
    assert input_entry.is_output_layer is False
    assert input_entry.has_parents is False
    assert input_entry.has_children is True


def test_output_layer_properties(small_input):
    model = example_models.SimpleFF()
    mh = log_forward_pass(model, small_input)
    output_entry = None
    for label in mh.layer_labels:
        e = mh[label]
        if e.is_output_layer:
            output_entry = e
            break
    assert output_entry is not None
    assert output_entry.is_output_layer is True
    assert output_entry.has_children is False


def test_entry_tensor_info_fields(small_input):
    model = example_models.SimpleFF()
    mh = log_forward_pass(model, small_input)
    entry = mh[0]
    assert isinstance(entry.tensor_shape, (tuple, torch.Size))
    assert isinstance(entry.tensor_dtype, torch.dtype)
    assert isinstance(entry.tensor_fsize, (int, float))
    assert entry.tensor_contents is not None


def test_function_call_fields(small_input):
    model = example_models.SimpleFF()
    mh = log_forward_pass(model, small_input)
    non_input = None
    for label in mh.layer_labels:
        e = mh[label]
        if not e.is_input_layer:
            non_input = e
            break
    assert non_input is not None
    assert isinstance(non_input.func_applied_name, str)
    assert len(non_input.func_applied_name) > 0
    assert isinstance(non_input.func_time_elapsed, float)


def test_graph_relationships(small_input):
    model = example_models.SimpleFF()
    mh = log_forward_pass(model, small_input)
    for label in mh.layer_labels:
        entry = mh[label]
        assert isinstance(entry.parent_layers, list)
        assert isinstance(entry.child_layers, list)
        for parent_label in entry.parent_layers:
            parent = mh[parent_label]
            assert label in parent.child_layers
        for child_label in entry.child_layers:
            child = mh[child_label]
            assert label in child.parent_layers


def test_inplace_function_flag(small_input):
    model = example_models.InPlaceFuncs()
    mh = log_forward_pass(model, small_input)
    for label in mh.layer_labels:
        entry = mh[label]
        assert isinstance(entry.function_is_inplace, bool)
    found_relu_ = False
    for label in mh.layer_labels:
        entry = mh[label]
        if entry.func_applied_name and entry.func_applied_name.endswith("_"):
            found_relu_ = True
            break
    assert found_relu_, "InPlaceFuncs should have relu_ (inplace function name)"


def test_param_fields_with_params(small_input):
    model = example_models.BatchNormModel()
    mh = log_forward_pass(model, small_input)
    found = False
    for label in mh.layer_labels:
        entry = mh[label]
        if entry.computed_with_params:
            assert entry.num_params_total > 0
            found = True
            break
    assert found, "BatchNormModel should have layers computed with params"


def test_param_fields_without_params(small_input):
    model = example_models.SimpleFF()
    mh = log_forward_pass(model, small_input)
    for label in mh.layer_labels:
        entry = mh[label]
        assert entry.computed_with_params is False


def test_module_fields(small_input):
    model = example_models.NestedModules()
    mh = log_forward_pass(model, small_input)
    found = False
    for label in mh.layer_labels:
        entry = mh[label]
        if entry.is_computed_inside_submodule:
            assert isinstance(entry.containing_module_origin, str)
            assert isinstance(entry.module_nesting_depth, int)
            assert entry.module_nesting_depth > 0
            found = True
            break
    assert found, "NestedModules should have layers inside submodules"


def test_buffer_layer_fields():
    model = example_models.BufferModel()
    model_input = torch.rand(12, 12)
    mh = log_forward_pass(model, model_input)
    found = False
    for label in mh.layer_labels:
        entry = mh[label]
        if entry.is_buffer_layer:
            assert isinstance(entry.buffer_address, str)
            found = True
            break
    assert found, "BufferModel should have buffer layers"


def test_internally_initialized_fields(small_input):
    model = example_models.SimpleInternallyGenerated()
    mh = log_forward_pass(model, small_input)
    found = False
    for label in mh.layer_labels:
        entry = mh[label]
        if entry.initialized_inside_model:
            found = True
            break
    assert found, "SimpleInternallyGenerated should have internally init layers"


def test_sibling_spouse_fields(small_input):
    model = example_models.SimpleFF()
    mh = log_forward_pass(model, small_input)
    for label in mh.layer_labels:
        entry = mh[label]
        assert isinstance(entry.sibling_layers, list)
        assert isinstance(entry.spouse_layers, list)


def test_conditional_fields():
    model = example_models.ConditionalBranching()
    model_input = -torch.ones(6, 3, 224, 224)
    mh = log_forward_pass(model, model_input)
    found = False
    for label in mh.layer_labels:
        entry = mh[label]
        if entry.in_cond_branch:
            found = True
            break
    assert found, "ConditionalBranching should have layers in cond branches"


def test_distances_with_flag(small_input):
    model = example_models.SimpleFF()
    mh = log_forward_pass(model, small_input, mark_input_output_distances=True)
    for label in mh.layer_labels:
        entry = mh[label]
        assert entry.min_distance_from_input is not None
        assert entry.max_distance_from_input is not None
        assert entry.min_distance_from_output is not None
        assert entry.max_distance_from_output is not None
        assert isinstance(entry.min_distance_from_input, int)


# =============================================================================
# Recurrent metadata
# =============================================================================


def test_recurrent_pass_numbers(input_2d):
    model = example_models.RecurrentParamsSimple()
    mh = log_forward_pass(model, input_2d)
    found_multi_pass = False
    for label in mh.layer_labels:
        entry = mh[label]
        if entry.pass_num > 1:
            found_multi_pass = True
            break
    assert found_multi_pass, "Recurrent model should have layers with pass_num > 1"


def test_recurrent_layer_passes_total(input_2d):
    model = example_models.RecurrentParamsSimple()
    mh = log_forward_pass(model, input_2d)
    found = False
    for label in mh.layer_labels:
        entry = mh[label]
        if entry.layer_passes_total > 1:
            found = True
            break
    assert found, "Recurrent model should have layers with passes_total > 1"


def test_recurrent_same_layer_operations(input_2d):
    model = example_models.RecurrentParamsSimple()
    mh = log_forward_pass(model, input_2d)
    found = False
    for label in mh.layer_labels:
        entry = mh[label]
        if entry.same_layer_operations and len(entry.same_layer_operations) > 0:
            found = True
            break
    assert found, "Recurrent model should have same_layer_operations"


def test_rolled_layer_list(input_2d):
    model = example_models.RecurrentParamsSimple()
    mh = log_forward_pass(model, input_2d)
    assert isinstance(mh.layer_list_rolled, list)
    assert len(mh.layer_list_rolled) < len(mh.layer_list)


# =============================================================================
# ModelHistory access patterns
# =============================================================================


def test_model_history_len(small_input):
    model = example_models.SimpleFF()
    mh = log_forward_pass(model, small_input)
    assert len(mh) == len(mh.layer_labels)


def test_getitem_by_positive_index(small_input):
    model = example_models.SimpleFF()
    mh = log_forward_pass(model, small_input)
    entry = mh[0]
    assert entry.layer_label == mh.layer_labels[0]


def test_getitem_by_negative_index(small_input):
    model = example_models.SimpleFF()
    mh = log_forward_pass(model, small_input)
    entry = mh[-1]
    assert entry.layer_label == mh.layer_labels[-1]


def test_getitem_by_label(small_input):
    model = example_models.SimpleFF()
    mh = log_forward_pass(model, small_input)
    label = mh.layer_labels[0]
    entry = mh[label]
    assert entry.layer_label == label


def test_model_history_iter(small_input):
    model = example_models.SimpleFF()
    mh = log_forward_pass(model, small_input)
    items = list(mh)
    assert len(items) == len(mh.layer_labels)


def test_layer_labels_properties(small_input):
    model = example_models.SimpleFF()
    mh = log_forward_pass(model, small_input)
    assert isinstance(mh.layer_labels, list)
    assert all(isinstance(lbl, str) for lbl in mh.layer_labels)
    assert isinstance(mh.layer_labels_no_pass, list)
    assert all(isinstance(lbl, str) for lbl in mh.layer_labels_no_pass)
    assert isinstance(mh.layer_labels_w_pass, list)
    assert all(isinstance(lbl, str) for lbl in mh.layer_labels_w_pass)


# =============================================================================
# Function args saving
# =============================================================================


def test_creation_args_populated(small_input):
    model = example_models.SimpleFF()
    mh = log_forward_pass(model, small_input, save_function_args=True)
    assert mh.save_function_args is True
    found = False
    for label in mh.layer_labels:
        entry = mh[label]
        if not entry.is_input_layer and entry.creation_args is not None:
            found = True
            break
    assert found, "save_function_args=True should populate creation_args"


def test_creation_args_not_populated(small_input):
    model = example_models.SimpleFF()
    mh = log_forward_pass(model, small_input)
    assert mh.save_function_args is False


# =============================================================================
# Activation postfunc
# =============================================================================


def test_postfunc_applied(small_input):
    model = example_models.SimpleFF()
    mh = log_forward_pass(model, small_input, activation_postfunc=torch.mean)
    for label in mh.layer_labels:
        entry = mh[label]
        if entry.tensor_contents is not None:
            assert entry.tensor_contents.dim() == 0, (
                f"Layer {label} should be scalar after torch.mean postfunc"
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
    mh = log_forward_pass(model, x)
    flops_values = [entry.flops_forward for entry in mh.layer_list]
    non_none = [f for f in flops_values if f is not None]
    assert len(non_none) > 0, "No layers have FLOPs computed"


def test_flops_total_positive():
    """total_flops_forward should be positive for a model with compute."""
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU())
    x = torch.randn(4, 10)
    mh = log_forward_pass(model, x)
    assert mh.total_flops_forward > 0
    assert mh.total_flops_backward > 0
    assert mh.total_flops == mh.total_flops_forward + mh.total_flops_backward


def test_flops_by_type():
    """flops_by_type returns a dict with expected keys."""
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU())
    x = torch.randn(4, 10)
    mh = log_forward_pass(model, x)
    fbt = mh.flops_by_type()
    assert isinstance(fbt, dict)
    assert len(fbt) > 0
    for layer_type, info in fbt.items():
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
    mh = log_forward_pass(model, x)
    # Conv2d(3, 16, 3): 2 * (1*16*32*32) * 3 * 9 = 884736
    assert mh.total_flops_forward > 800000


def test_flops_coverage_on_model():
    """At least 50% of non-input layers should have non-None FLOPs."""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.Sigmoid(),
    )
    x = torch.randn(4, 10)
    mh = log_forward_pass(model, x)
    total = len(mh.layer_list)
    with_flops = sum(1 for e in mh.layer_list if e.flops_forward is not None)
    coverage = with_flops / total if total > 0 else 0
    assert coverage >= 0.5, f"FLOPs coverage too low: {coverage:.1%}"


# =============================================================================
# Module training mode tracking (issue #52)
# =============================================================================


def test_module_training_modes_populated(small_input):
    """module_training_modes should map module addresses to their training flag."""
    model = example_models.SimpleFF()
    model.train()
    mh = log_forward_pass(model, small_input)
    # SimpleFF has no submodules, so dict should be empty
    assert isinstance(mh.module_training_modes, dict)


def test_module_training_modes_train_vs_eval():
    """Training mode should be captured correctly for each submodule."""
    model = nn.Sequential(nn.Linear(5, 5), nn.ReLU(), nn.Linear(5, 3))
    x = torch.rand(2, 5)

    model.train()
    mh_train = log_forward_pass(model, x)
    for addr, mode in mh_train.module_training_modes.items():
        assert mode is True, f"Module {addr} should be training=True"

    model.eval()
    mh_eval = log_forward_pass(model, x)
    for addr, mode in mh_eval.module_training_modes.items():
        assert mode is False, f"Module {addr} should be training=False"


@pytest.mark.slow
def test_flops_resnet18_range():
    """ResNet-18 ~ 1.8 GFLOPs. Check we're in the right ballpark."""
    torchvision = pytest.importorskip("torchvision")
    model = torchvision.models.resnet18(weights=None)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    mh = log_forward_pass(model, x)
    gflops = mh.total_flops_forward / 1e9
    assert 1.0 < gflops < 5.0, f"ResNet-18 FLOPs = {gflops:.2f}G, expected ~1.8G"
