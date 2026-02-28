"""Tests for toy models from example_models.py.

All existing validation + visualization tests migrated from test_validation_and_visuals.py,
plus new API coverage tests.
"""

from os.path import join as opj

import torch

from conftest import VIS_OUTPUT_DIR

import example_models
from torchlens import (
    log_forward_pass,
    get_model_metadata,
    show_model_graph,
    validate_saved_activations,
)


# =============================================================================
# Simple operations
# =============================================================================


def test_model_simple_ff(default_input1):
    model = example_models.SimpleFF()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "simple_ff"),
    )


def test_model_inplace_funcs(default_input1):
    model = example_models.InPlaceFuncs()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "inplace_funcs"),
    )


def test_model_simple_internally_generated(default_input1):
    model = example_models.SimpleInternallyGenerated()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "simple_internally_generated"),
    )


def test_model_new_tensor_inside(default_input1):
    model = example_models.NewTensorInside()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "new_tensor_inside"),
    )


def test_model_new_tensor_from_numpy(default_input1):
    model = example_models.TensorFromNumpy()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "new_tensor_from_numpy"),
    )


def test_model_simple_random(default_input1):
    model = example_models.SimpleRandom()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "simple_random"),
    )


# =============================================================================
# Dropout
# =============================================================================


def test_dropout_model_real_train(default_input1):
    model = example_models.DropoutModelReal()
    model.train()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "dropout_real_train"),
    )


def test_dropout_model_real_eval(default_input1):
    model = example_models.DropoutModelReal()
    model.eval()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "dropout_real_eval"),
    )


def test_dropout_model_dummy_zero_train(default_input1):
    model = example_models.DropoutModelDummyZero()
    model.train()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "dropout_dummyzero_train"),
    )


def test_dropout_model_dummy_zero_eval(default_input1):
    model = example_models.DropoutModelDummyZero()
    model.eval()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "dropout_dummyzero_eval"),
    )


# =============================================================================
# BatchNorm
# =============================================================================


def test_batchnorm_train(default_input1):
    model = example_models.BatchNormModel()
    model.train()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_buffer_layers=True,
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "batchnorm_train_showbuffer"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_buffer_layers=False,
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "batchnorm_train_invisbuffer"),
    )


def test_batchnorm_eval(default_input1):
    model = example_models.BatchNormModel()
    model.eval()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "batchnorm_eval"),
    )


# =============================================================================
# Tensor operations
# =============================================================================


def test_concat_tensors(default_input1):
    model = example_models.ConcatTensors()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "concat_tensors"),
    )


def test_split_tensor(default_input1):
    model = example_models.SplitTensor()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "split_tensors"),
    )


def test_identity_model(default_input1):
    model = example_models.IdentityModule()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "identity_model"),
    )


def test_assign_tensor(input_2d):
    model = example_models.AssignTensor()
    assert validate_saved_activations(model, input_2d)
    show_model_graph(
        model,
        input_2d,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "assigntensor"),
    )


def test_get_and_set_item(default_input1):
    model = example_models.GetAndSetItem()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "get_set_item"),
    )


def test_getitem_tracking(input_2d):
    model = example_models.GetItemTracking()
    assert validate_saved_activations(model, input_2d)
    show_model_graph(
        model,
        input_2d,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "getitem_tracking"),
    )


def test_inplace_zero_tensor(default_input1):
    model = example_models.InPlaceZeroTensor()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "inplace_zerotensor"),
    )


def test_slice_operations(default_input1):
    model = example_models.SliceOperations()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "slice_operations"),
    )


def test_dummy_operations(default_input1):
    model = example_models.DummyOperations()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "dummy_operations"),
    )


def test_sametensor_arg(default_input1):
    model = example_models.SameTensorArg()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "same_tensor_arg"),
    )


# =============================================================================
# Multiple inputs / outputs
# =============================================================================


def test_multiple_inputs_arg(default_input1, default_input2, default_input3):
    model = example_models.MultiInputs()
    assert validate_saved_activations(model, [default_input1, default_input2, default_input3])
    show_model_graph(
        model,
        [default_input1, default_input2, default_input3],
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "multiple_inputs_arg"),
    )


def test_multiple_inputs_kwarg(default_input1, default_input2, default_input3):
    model = example_models.MultiInputs()
    assert validate_saved_activations(
        model, [], {"x": default_input1, "y": default_input2, "z": default_input3}
    )
    show_model_graph(
        model,
        [],
        {"x": default_input1, "y": default_input2, "z": default_input3},
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "multiple_inputs_kwarg"),
    )


def test_multiple_inputs_arg_kwarg_mix(default_input1, default_input2, default_input3):
    model = example_models.MultiInputs()
    assert validate_saved_activations(
        model, [default_input1], {"y": default_input2, "z": default_input3}
    )
    show_model_graph(
        model,
        [],
        {"x": default_input1, "y": default_input2, "z": default_input3},
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "multiple_inputs_arg_kwarg_mix"),
    )


def test_list_input(default_input1, default_input2, default_input3):
    model = example_models.ListInput()
    assert validate_saved_activations(model, [[default_input1, default_input2, default_input3]])
    show_model_graph(
        model,
        [[default_input1, default_input2, default_input3]],
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "list_inputs"),
    )


def test_dict_input(default_input1, default_input2, default_input3):
    model = example_models.DictInput()
    assert validate_saved_activations(
        model, [{"x": default_input1, "y": default_input2, "z": default_input3}]
    )
    show_model_graph(
        model,
        [{"x": default_input1, "y": default_input2, "z": default_input3}],
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "dict_inputs"),
    )


def test_nested_input(default_input1, default_input2, default_input3, default_input4):
    model = example_models.NestedInput()
    assert validate_saved_activations(
        model,
        [
            {
                "list1": [default_input1, default_input2],
                "list2": [default_input3, default_input4],
            }
        ],
    )
    show_model_graph(
        model,
        [
            {
                "list1": [default_input1, default_input2],
                "list2": [default_input3, default_input4],
            }
        ],
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "nested_inputs"),
    )


def test_multi_outputs(default_input1):
    model = example_models.MultiOutputs()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "multi_outputs"),
    )


def test_list_output(default_input1):
    model = example_models.ListOutput()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "list_output"),
    )


def test_dict_output(default_input1):
    model = example_models.DictOutput()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "dict_output"),
    )


def test_nested_output(default_input1):
    model = example_models.NestedOutput()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "nested_output"),
    )


# =============================================================================
# Buffers
# =============================================================================


def test_buffer_model():
    model = example_models.BufferModel()
    model_input = torch.rand(12, 12)
    assert validate_saved_activations(model, model_input)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "buffer_visible"),
        vis_buffer_layers=True,
    )
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "buffer_invisible"),
        vis_buffer_layers=False,
    )


def test_buffer_rewrite_model():
    model = example_models.BufferRewriteModel()
    model_input = torch.rand(12, 12)
    assert validate_saved_activations(model, model_input)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_nesting_depth=1,
        vis_outpath=opj(
            VIS_OUTPUT_DIR,
            "toy-networks",
            "buffer_rewrite_model_visible_unnested_unrolled",
        ),
        vis_buffer_layers=True,
    )
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_nesting_depth=1,
        vis_outpath=opj(
            VIS_OUTPUT_DIR,
            "toy-networks",
            "buffer_rewrite_model_invisible_unnested_unrolled",
        ),
        vis_buffer_layers=False,
    )
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            VIS_OUTPUT_DIR,
            "toy-networks",
            "buffer_rewrite_model_visible_nested_unrolled",
        ),
        vis_buffer_layers=True,
    )
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            VIS_OUTPUT_DIR,
            "toy-networks",
            "buffer_rewrite_model_invisible_nested_unrolled",
        ),
        vis_buffer_layers=False,
    )
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="rolled",
        vis_nesting_depth=1,
        vis_outpath=opj(
            VIS_OUTPUT_DIR,
            "toy-networks",
            "buffer_rewrite_model_visible_unnested_rolled",
        ),
        vis_buffer_layers=True,
    )
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="rolled",
        vis_nesting_depth=1,
        vis_outpath=opj(
            VIS_OUTPUT_DIR,
            "toy-networks",
            "buffer_rewrite_model_invisible_unnested_rolled",
        ),
        vis_buffer_layers=False,
    )
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(
            VIS_OUTPUT_DIR,
            "toy-networks",
            "buffer_rewrite_model_visible_nested_rolled",
        ),
        vis_buffer_layers=True,
    )
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(
            VIS_OUTPUT_DIR,
            "toy-networks",
            "buffer_rewrite_model_invisible_nested_rolled",
        ),
        vis_buffer_layers=False,
    )


# =============================================================================
# Branching
# =============================================================================


def test_simple_branching(default_input1):
    model = example_models.SimpleBranching()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "simple_branching"),
    )


def test_conditional_branching(zeros_input, ones_input):
    model = example_models.ConditionalBranching()
    assert validate_saved_activations(model, -ones_input)
    assert validate_saved_activations(model, ones_input)
    show_model_graph(
        model,
        -ones_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "conditional_branching_negative"),
    )
    show_model_graph(
        model,
        ones_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "conditional_branching_positive"),
    )


# =============================================================================
# Nesting
# =============================================================================


def test_repeated_module(default_input1):
    model = example_models.RepeatedModule()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "repeated_module"),
    )


def test_nested_modules(default_input1):
    model = example_models.NestedModules()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "nested_modules_fulldepth"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_nesting_depth=1,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "nested_modules_depth1"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_nesting_depth=2,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "nested_modules_depth2"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_nesting_depth=3,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "nested_modules_depth3"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_nesting_depth=4,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "nested_modules_depth4"),
    )


# =============================================================================
# Loops
# =============================================================================


def test_orphan_tensors(default_input1):
    model = example_models.OrphanTensors()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "orphan_tensors"),
    )


def test_simple_loop_no_param(default_input1):
    model = example_models.SimpleLoopNoParam()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "simple_loop_no_param_unrolled"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "simple_loop_no_param_rolled"),
    )


def test_same_op_repeat(vector_input):
    model = example_models.SameOpRepeat()
    assert validate_saved_activations(model, vector_input)
    show_model_graph(
        model,
        vector_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "same_op_repeat_unrolled"),
    )
    show_model_graph(
        model,
        vector_input,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "same_op_repeat_rolled"),
    )


def test_repeated_op_type_in_loop(default_input1):
    model = example_models.RepeatedOpTypeInLoop()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "same_op_type_in_loop_unrolled"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "same_op_type_in_loop_rolled"),
    )


def test_varying_loop_noparam1(default_input1):
    model = example_models.VaryingLoopNoParam1()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "varying_loop_noparam1_unrolled"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "varying_loop_noparam1_rolled"),
    )


def test_varying_loop_noparam2(default_input1):
    model = example_models.VaryingLoopNoParam2()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "varying_loop_noparam2_unrolled"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "varying_loop_noparam2_rolled"),
    )


def test_varying_loop_withparam(vector_input):
    model = example_models.VaryingLoopWithParam()
    assert validate_saved_activations(model, vector_input)
    show_model_graph(
        model,
        vector_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "varying_loop_withparam_unrolled"),
    )
    show_model_graph(
        model,
        vector_input,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "varying_loop_withparam_rolled"),
    )


def test_looping_internal_funcs(default_input1):
    model = example_models.LoopingInternalFuncs()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "looping_internal_funcs_unrolled"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "looping_internal_funcs_rolled"),
    )


def test_looping_from_inputs1(default_input1, default_input2, default_input3):
    model = example_models.LoopingFromInputs1()
    assert validate_saved_activations(model, [default_input1, default_input2, default_input3])
    show_model_graph(
        model,
        [default_input1, default_input2, default_input3],
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "looping_from_inputs1_unrolled"),
    )
    show_model_graph(
        model,
        [default_input1, default_input2, default_input3],
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "looping_from_inputs1_rolled"),
    )


def test_looping_from_inputs2(default_input1, default_input2, default_input3):
    model = example_models.LoopingFromInputs2()
    assert validate_saved_activations(model, [[default_input1, default_input2, default_input3]])
    show_model_graph(
        model,
        [[default_input1, default_input2, default_input3]],
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "looping_from_inputs2_unrolled"),
    )
    show_model_graph(
        model,
        [[default_input1, default_input2, default_input3]],
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "looping_from_inputs2_rolled"),
    )


def test_looping_inputs_and_outputs(default_input1, default_input2, default_input3):
    model = example_models.LoopingInputsAndOutputs()
    assert validate_saved_activations(model, [[default_input1, default_input2, default_input3]])
    show_model_graph(
        model,
        [[default_input1, default_input2, default_input3]],
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            VIS_OUTPUT_DIR,
            "toy-networks",
            "looping_inputs_and_outputs_unrolled",
        ),
    )
    show_model_graph(
        model,
        [[default_input1, default_input2, default_input3]],
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "looping_inputs_and_outputs_rolled"),
    )


def test_stochastic_loop():
    model = example_models.StochasticLoop()
    model_input = torch.zeros(2, 2)
    assert validate_saved_activations(model, model_input)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "stochastic_loop_unrolled1"),
    )
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "stochastic_loop_rolled1"),
    )
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "stochastic_loop_unrolled2"),
    )
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "stochastic_loop_rolled2"),
    )
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "stochastic_loop_unrolled3"),
    )
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "stochastic_loop_rolled3"),
    )


# =============================================================================
# Recurrent
# =============================================================================


def test_recurrent_params_simple(input_2d):
    model = example_models.RecurrentParamsSimple()
    assert validate_saved_activations(model, input_2d)
    show_model_graph(
        model,
        input_2d,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "recurrent_params_simple_unrolled"),
    )
    show_model_graph(
        model,
        input_2d,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "recurrent_params_simple_rolled"),
    )


def test_recurrent_params_complex(input_2d):
    model = example_models.RecurrentParamsComplex()
    assert validate_saved_activations(model, input_2d)
    show_model_graph(
        model,
        input_2d,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "recurrent_params_complex_unrolled"),
    )
    show_model_graph(
        model,
        input_2d,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "recurrent_params_complex_rolled"),
    )


def test_looping_params_doublenested(input_2d):
    model = example_models.LoopingParamsDoubleNested()
    assert validate_saved_activations(model, input_2d)
    show_model_graph(
        model,
        input_2d,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            VIS_OUTPUT_DIR,
            "toy-networks",
            "looping_params_doublenested_unrolled",
        ),
    )
    show_model_graph(
        model,
        input_2d,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(
            VIS_OUTPUT_DIR,
            "toy-networks",
            "looping_params_doublenested_rolled",
        ),
    )


# =============================================================================
# Module clashes
# =============================================================================


def test_module_looping_clash1(default_input1):
    model = example_models.ModuleLoopingClash1()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "module_looping_clash1_unrolled"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "module_looping_clash1_rolled"),
    )


def test_module_looping_clash2(default_input1):
    model = example_models.ModuleLoopingClash2()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "module_looping_clash2_unrolled"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "module_looping_clash2_rolled"),
    )


def test_module_looping_clash3(default_input1):
    model = example_models.ModuleLoopingClash3()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "module_looping_clash3_unrolled"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "module_looping_clash3_rolled"),
    )


def test_nested_param_free_loops(default_input1):
    """Tests nested loop topology where inner ops have the same equivalence type
    across levels but surrounding ops differ per level.

    4 outer iterations x 3 inner levels = 12 sin operations, all with the same
    equivalence type. They should be ONE group of 12 passes.
    """
    model = example_models.NestedParamFreeLoops()
    assert validate_saved_activations(model, default_input1)
    mh = log_forward_pass(model, default_input1)

    # All sin ops should be in one layer group with 12 passes
    sin_pass_counts = set()
    for label in mh.layer_labels:
        entry = mh[label]
        if entry.func_applied_name == "sin":
            sin_pass_counts.add(entry.layer_passes_total)
    assert sin_pass_counts == {12}, (
        f"sin ops fragmented into groups with pass counts {sin_pass_counts}, expected {{12}}"
    )

    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "nested_param_free_loops_unrolled"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "nested_param_free_loops_rolled"),
    )


def test_parallel_loops(default_input1):
    model = example_models.ParallelLoops()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "parallel_loops_unrolled"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "parallel_loops_rolled"),
    )


def test_shared_param_loop_external(input_2d):
    model = example_models.SharedParamLoopExternal()
    assert validate_saved_activations(model, input_2d)
    show_model_graph(
        model,
        input_2d,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "shared_param_loop_external_unrolled"),
    )
    show_model_graph(
        model,
        input_2d,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "shared_param_loop_external_rolled"),
    )


def test_interleaved_shared_param_loops(input_2d):
    model = example_models.InterleavedSharedParamLoops()
    assert validate_saved_activations(model, input_2d)
    show_model_graph(
        model,
        input_2d,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "interleaved_shared_param_loops_unrolled"),
    )
    show_model_graph(
        model,
        input_2d,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "interleaved_shared_param_loops_rolled"),
    )


def test_nested_loops_independent_params(input_2d):
    model = example_models.NestedLoopsIndependentParams()
    assert validate_saved_activations(model, input_2d)
    show_model_graph(
        model,
        input_2d,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "nested_loops_independent_params_unrolled"),
    )
    show_model_graph(
        model,
        input_2d,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "nested_loops_independent_params_rolled"),
    )


def test_self_feeding_no_param(default_input1):
    model = example_models.SelfFeedingNoParam()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "self_feeding_no_param_unrolled"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "self_feeding_no_param_rolled"),
    )


def test_diamond_loop(input_2d):
    model = example_models.DiamondLoop()
    assert validate_saved_activations(model, input_2d)
    show_model_graph(
        model,
        input_2d,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "diamond_loop_unrolled"),
    )
    show_model_graph(
        model,
        input_2d,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "diamond_loop_rolled"),
    )


def test_accumulator_loop(input_2d):
    model = example_models.AccumulatorLoop()
    assert validate_saved_activations(model, input_2d)
    show_model_graph(
        model,
        input_2d,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "accumulator_loop_unrolled"),
    )
    show_model_graph(
        model,
        input_2d,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "accumulator_loop_rolled"),
    )


def test_single_iteration_loop(input_2d):
    model = example_models.SingleIterationLoop()
    assert validate_saved_activations(model, input_2d)
    show_model_graph(
        model,
        input_2d,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "single_iteration_loop_unrolled"),
    )
    show_model_graph(
        model,
        input_2d,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "single_iteration_loop_rolled"),
    )


def test_long_loop(input_2d):
    model = example_models.LongLoop()
    assert validate_saved_activations(model, input_2d)
    show_model_graph(
        model,
        input_2d,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "long_loop_unrolled"),
    )
    show_model_graph(
        model,
        input_2d,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "long_loop_rolled"),
    )


def test_data_dependent_branch_loop(input_2d):
    model = example_models.DataDependentBranchLoop()
    assert validate_saved_activations(model, input_2d)
    show_model_graph(
        model,
        input_2d,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "data_dependent_branch_loop_unrolled"),
    )
    show_model_graph(
        model,
        input_2d,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "data_dependent_branch_loop_rolled"),
    )


def test_sequential_param_free_loops(default_input1):
    model = example_models.SequentialParamFreeLoops()
    assert validate_saved_activations(model, default_input1)
    mh = log_forward_pass(model, default_input1)
    # Verify the two sequential loops produce SEPARATE groups (not merged)
    sin_groups = set()
    for label in mh.layer_labels:
        entry = mh[label]
        if entry.func_applied_name == "sin":
            sin_groups.add(entry.layer_passes_total)
    assert sin_groups == {3}, f"Sequential sin loops should each have 3 passes, got {sin_groups}"
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "sequential_param_free_loops_unrolled"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "sequential_param_free_loops_rolled"),
    )


# =============================================================================
# Complex models
# =============================================================================


def test_propertymodel(input_complex):
    model = example_models.PropertyModel()
    assert validate_saved_activations(model, input_complex)
    show_model_graph(
        model,
        input_complex,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "propertymodel"),
    )


def test_ubermodel1(input_2d):
    model = example_models.UberModel1()
    assert validate_saved_activations(model, [[input_2d, input_2d * 2, input_2d * 3]])
    show_model_graph(
        model,
        [[input_2d, input_2d * 2, input_2d * 3]],
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "ubermodel1"),
    )


def test_ubermodel2():
    model = example_models.UberModel2()
    model_input = torch.rand(2, 1, 3, 3)
    assert validate_saved_activations(model, model_input)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "ubermodel2"),
    )


def test_ubermodel3(input_2d):
    model = example_models.UberModel3()
    assert validate_saved_activations(model, input_2d)
    show_model_graph(
        model,
        input_2d,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "ubermodel3_unrolled"),
    )
    show_model_graph(
        model,
        input_2d,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "ubermodel3_rolled"),
    )


def test_ubermodel4(input_2d):
    model = example_models.UberModel4()
    assert validate_saved_activations(model, input_2d)
    show_model_graph(
        model,
        input_2d,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "ubermodel4_unrolled"),
    )
    show_model_graph(
        model,
        input_2d,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "ubermodel4_rolled"),
    )


def test_ubermodel5():
    model = example_models.UberModel5()
    model_input = torch.rand(1, 1, 3, 3)
    assert validate_saved_activations(model, model_input)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "ubermodel5"),
    )


def test_ubermodel6(default_input1):
    model = example_models.UberModel6()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "ubermodel6_unrolled"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "ubermodel6_rolled"),
    )


def test_ubermodel7(input_2d):
    model = example_models.UberModel7()
    assert validate_saved_activations(model, input_2d)
    show_model_graph(
        model,
        input_2d,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "ubermodel7_unrolled"),
    )
    show_model_graph(
        model,
        input_2d,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "ubermodel7_rolled"),
    )


def test_ubermodel8():
    model = example_models.UberModel8()
    model_input = torch.rand(2, 1, 3, 3)
    assert validate_saved_activations(model, model_input)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "ubermodel8"),
    )


def test_ubermodel9():
    model = example_models.UberModel9()
    model_input = torch.rand(1, 1, 3, 3)
    assert validate_saved_activations(model, model_input)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "ubermodel9"),
    )


# =============================================================================
# NEW: GeluModel (exists in example_models but had no test)
# =============================================================================


def test_gelu_model(default_input1):
    model = example_models.GeluModel()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "gelu_model"),
    )


# =============================================================================
# NEW: API coverage tests
# =============================================================================


def test_log_forward_pass_layers_to_save(default_input1):
    """Test layers_to_save parameter selectively saves activations."""
    model = example_models.SimpleFF()
    mh = log_forward_pass(model, default_input1, layers_to_save="all")
    all_labels = mh.layer_labels
    assert len(all_labels) > 0

    # Save only the first layer
    mh_partial = log_forward_pass(model, default_input1, layers_to_save=[all_labels[0]])
    # The layer list should still track all layers
    assert len(mh_partial.layer_labels) > 0
    # But only the requested layer should have saved activations
    saved_entry = mh_partial[all_labels[0]]
    assert saved_entry.tensor_contents is not None


def test_log_forward_pass_save_function_args(default_input1):
    """Test save_function_args=True populates creation_args on entries."""
    model = example_models.SimpleFF()
    mh = log_forward_pass(model, default_input1, save_function_args=True)
    assert mh.save_function_args is True
    # At least one non-input layer should have creation_args
    found = False
    for label in mh.layer_labels:
        entry = mh[label]
        if not entry.is_input_layer and entry.creation_args is not None:
            found = True
            break
    assert found, "No non-input layer had creation_args populated"


def test_log_forward_pass_activation_postfunc(default_input1):
    """Test activation_postfunc applies to saved tensors."""
    model = example_models.SimpleFF()
    mh = log_forward_pass(model, default_input1, activation_postfunc=torch.mean)
    # All saved tensors should be scalar (mean reduces to scalar)
    for label in mh.layer_labels:
        entry = mh[label]
        if entry.tensor_contents is not None:
            assert entry.tensor_contents.dim() == 0, (
                f"Layer {label} tensor should be scalar after torch.mean postfunc"
            )


def test_log_forward_pass_mark_distances(default_input1):
    """Test mark_input_output_distances=True populates distance fields."""
    model = example_models.SimpleFF()
    mh = log_forward_pass(model, default_input1, mark_input_output_distances=True)
    for label in mh.layer_labels:
        entry = mh[label]
        assert entry.min_distance_from_input is not None
        assert entry.max_distance_from_input is not None
        assert entry.min_distance_from_output is not None
        assert entry.max_distance_from_output is not None


def test_get_model_metadata(default_input1):
    """Test get_model_metadata returns ModelHistory without saving activations."""
    model = example_models.SimpleFF()
    mh = get_model_metadata(model, default_input1)
    assert len(mh.layer_labels) > 0
    assert mh.num_tensors_total > 0


def test_model_history_getitem_by_index(default_input1):
    """Test ModelHistory supports integer indexing."""
    model = example_models.SimpleFF()
    mh = log_forward_pass(model, default_input1)
    first = mh[0]
    last = mh[-1]
    assert first.layer_label == mh.layer_labels[0]
    assert last.layer_label == mh.layer_labels[-1]


def test_model_history_getitem_by_label(default_input1):
    """Test ModelHistory supports string label access."""
    model = example_models.SimpleFF()
    mh = log_forward_pass(model, default_input1)
    label = mh.layer_labels[0]
    entry = mh[label]
    assert entry.layer_label == label


def test_model_history_len(default_input1):
    """Test len(ModelHistory) matches layer count."""
    model = example_models.SimpleFF()
    mh = log_forward_pass(model, default_input1)
    assert len(mh) == len(mh.layer_labels)


def test_model_history_iter(default_input1):
    """Test iteration over ModelHistory yields entries."""
    model = example_models.SimpleFF()
    mh = log_forward_pass(model, default_input1)
    items = list(mh)
    assert len(items) == len(mh.layer_labels)


def test_model_history_layer_labels(default_input1):
    """Test layer_labels, layer_labels_no_pass, layer_labels_w_pass properties."""
    model = example_models.SimpleFF()
    mh = log_forward_pass(model, default_input1)
    assert isinstance(mh.layer_labels, list)
    assert all(isinstance(lbl, str) for lbl in mh.layer_labels)
    assert isinstance(mh.layer_labels_no_pass, list)
    assert all(isinstance(lbl, str) for lbl in mh.layer_labels_no_pass)
    assert isinstance(mh.layer_labels_w_pass, list)
    assert all(isinstance(lbl, str) for lbl in mh.layer_labels_w_pass)


def test_rolled_vs_unrolled_visualization(input_2d):
    """Test both rolled and unrolled modes produce output on recurrent model."""
    import os

    model = example_models.RecurrentParamsSimple()
    unrolled_path = opj(VIS_OUTPUT_DIR, "toy-networks", "test_rolled_vs_unrolled_unrolled")
    rolled_path = opj(VIS_OUTPUT_DIR, "toy-networks", "test_rolled_vs_unrolled_rolled")
    show_model_graph(
        model,
        input_2d,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=unrolled_path,
    )
    show_model_graph(
        model,
        input_2d,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=rolled_path,
    )
    # Both should produce output files
    assert os.path.exists(unrolled_path + ".pdf") or os.path.exists(unrolled_path)
    assert os.path.exists(rolled_path + ".pdf") or os.path.exists(rolled_path)


# =============================================================================
# View mutation / child tensor variation tests
# =============================================================================


def test_view_mutation_unsqueeze(input_2d):
    """Mutation through unsqueeze view should be logged without error."""
    model = example_models.ViewMutationUnsqueeze()
    assert validate_saved_activations(model, input_2d)
    mh = log_forward_pass(model, input_2d, save_function_args=True)
    assert mh is not None
    assert len(mh.layer_labels) > 0


def test_view_mutation_reshape(input_2d):
    """Mutation through reshape view should be logged without error."""
    model = example_models.ViewMutationReshape()
    assert validate_saved_activations(model, input_2d)
    mh = log_forward_pass(model, input_2d, save_function_args=True)
    assert mh is not None
    assert len(mh.layer_labels) > 0


def test_view_mutation_transpose(input_2d):
    """Mutation through transpose view should be logged without error."""
    model = example_models.ViewMutationTranspose()
    assert validate_saved_activations(model, input_2d)
    mh = log_forward_pass(model, input_2d, save_function_args=True)
    assert mh is not None
    assert len(mh.layer_labels) > 0


def test_multiple_view_mutations(input_2d):
    """Multiple views mutated independently should be logged without error."""
    model = example_models.MultipleViewMutations()
    assert validate_saved_activations(model, input_2d)
    mh = log_forward_pass(model, input_2d, save_function_args=True)
    assert mh is not None
    assert len(mh.layer_labels) > 0


def test_chained_view_mutation(input_2d):
    """Mutation through chained views should be logged without error."""
    model = example_models.ChainedViewMutation()
    assert validate_saved_activations(model, input_2d)
    mh = log_forward_pass(model, input_2d, save_function_args=True)
    assert mh is not None
    assert len(mh.layer_labels) > 0


def test_output_matches_parent_no_false_positive(input_2d):
    """No mutation model: verify no false-positive child tensor variations."""
    model = example_models.OutputMatchesParent()
    assert validate_saved_activations(model, input_2d)
    mh = log_forward_pass(model, input_2d, save_function_args=True)
    assert mh is not None
    assert len(mh.layer_labels) > 0
    # No layer should have child tensor variations since nothing is mutated
    for label in mh.layer_labels:
        entry = mh[label]
        assert not entry.has_child_tensor_variations, (
            f"Layer {label} should not have child tensor variations in a "
            f"mutation-free model, but has_child_tensor_variations={entry.has_child_tensor_variations}"
        )
