# This is for validating all the different model types. Have it both validate and spit out the visual for checking.
# Let the default input size be 3x3x224x224.

import numpy as np
import os
from os.path import join as opj

# import cornet
import pytest
import requests
import timm.models
import torch
import torch.nn as nn
import torchaudio.models
import torchvision
import visualpriors
from PIL import Image

from transformers import (
    BertForNextSentencePrediction,
    BertTokenizer,
    GPT2Model,
    GPT2Tokenizer,
)
from torch_geometric.nn import DimeNet

import wbreaks_example_models
from torchlens.user_funcs import show_model_graph, validate_saved_activations

# Define inputs
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.use_deterministic_algorithms(True)

sub_dirs = [
    "cornet",
    "graph-neural-networks",
    "language-models",
    "multimodal-models",
    "taskonomy",
    "timm",
    "torchaudio",
    "torchvision-main",
    "torchvision-detection",
    "torchvision-opticflow",
    "torchvision-segmentation",
    "torchvision-video",
    "torchvision-quantize",
    "taskonomy",
]

for sub_dir in sub_dirs:
    os.makedirs(opj("visualization_outputs", sub_dir), exist_ok=True)


@pytest.fixture
def default_input1():
    return torch.rand(6, 3, 224, 224)


@pytest.fixture
def default_input2():
    return torch.rand(6, 3, 224, 224)


@pytest.fixture
def default_input3():
    return torch.rand(6, 3, 224, 224)


@pytest.fixture
def default_input4():
    return torch.rand(6, 3, 224, 224)


@pytest.fixture
def zeros_input():
    return torch.zeros(6, 3, 224, 224)


@pytest.fixture
def ones_input():
    return torch.ones(6, 3, 224, 224)


@pytest.fixture
def vector_input():
    return torch.rand(5)


@pytest.fixture
def input_2d():
    return torch.rand(5, 5)


@pytest.fixture
def input_complex():
    return torch.complex(torch.rand(3, 3), torch.rand(3, 3)),


# Test different operations

def test_model_simple_ff(default_input1):
    print("Testing SimpleFF model")
    model = wbreaks_example_models.SimpleFF()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "toy-networks", "simple_ff"),
        vis_fileformat="svg",
        vis_graph_with_dynamo_explain=True
    )

'''
def test_model_inplace_funcs(default_input1):
    model = example_models.InPlaceFuncs()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "toy-networks", "inplace_funcs"),
        vis_graph_with_dynamo_explain=True
    )


def test_model_simple_internally_generated(default_input1):
    model = example_models.SimpleInternallyGenerated()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "simple_internally_generated"
        ),
        vis_graph_with_dynamo_explain=True
    )


def test_model_new_tensor_inside(default_input1):
    model = example_models.NewTensorInside()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "toy-networks", "new_tensor_inside"),
        vis_graph_with_dynamo_explain=True
    )


def test_model_new_tensor_from_numpy(default_input1):
    model = example_models.TensorFromNumpy()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "new_tensor_from_numpy"
        ),
        vis_graph_with_dynamo_explain=True
    )


def test_model_simple_random(default_input1):
    model = example_models.SimpleRandom()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "toy-networks", "simple_random"),
        vis_graph_with_dynamo_explain=True
    )


def test_dropout_model_real_train(default_input1):
    model = example_models.DropoutModelReal()
    model.train()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "toy-networks", "dropout_real_train"),
        vis_graph_with_dynamo_explain=True
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
        vis_outpath=opj("visualization_outputs", "toy-networks", "dropout_real_eval"),
        vis_graph_with_dynamo_explain=True
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
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "dropout_dummyzero_train"
        ),
        vis_graph_with_dynamo_explain=True
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
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "dropout_dummyzero_eval"
        ),
        vis_graph_with_dynamo_explain=True
    )


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
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "batchnorm_train_showbuffer"
        ),
        vis_graph_with_dynamo_explain=True
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_buffer_layers=False,
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "batchnorm_train_invisbuffer"
        ),
        vis_graph_with_dynamo_explain=True
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
        vis_outpath=opj("visualization_outputs", "toy-networks", "batchnorm_eval"),
        vis_graph_with_dynamo_explain=True
    )


def test_concat_tensors(default_input1):
    model = example_models.ConcatTensors()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "toy-networks", "concat_tensors"),
        vis_graph_with_dynamo_explain=True
    )


def test_split_tensor(default_input1):
    model = example_models.SplitTensor()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "toy-networks", "split_tensors"),
        vis_graph_with_dynamo_explain=True
    )


def test_identity_model(default_input1):
    model = example_models.IdentityModule()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "toy-networks", "identity_model"),
        vis_graph_with_dynamo_explain=True
    )


def test_assign_tensor(input_2d):
    model = example_models.AssignTensor()
    assert validate_saved_activations(model, input_2d)
    show_model_graph(
        model,
        input_2d,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "toy-networks", "assigntensor"),
        vis_graph_with_dynamo_explain=True
    )


def test_get_and_set_item(default_input1):
    model = example_models.GetAndSetItem()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "toy-networks", "get_set_item"),
        vis_graph_with_dynamo_explain=True
    )


def test_getitem_tracking(input_2d):
    model = example_models.GetItemTracking()
    assert validate_saved_activations(model, input_2d)
    show_model_graph(
        model,
        input_2d,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "toy-networks", "getitem_tracking"),
        vis_graph_with_dynamo_explain=True
    )


def test_inplace_zero_tensor(default_input1):
    model = example_models.InPlaceZeroTensor()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "toy-networks", "inplace_zerotensor"),
        vis_graph_with_dynamo_explain=True
    )


def test_slice_operations(default_input1):
    model = example_models.SliceOperations()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "toy-networks", "slice_operations"),
        vis_graph_with_dynamo_explain=True
    )


def test_dummy_operations(default_input1):
    model = example_models.DummyOperations()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "toy-networks", "dummy_operations"),
        vis_graph_with_dynamo_explain=True
    )


def test_sametensor_arg(default_input1):
    model = example_models.SameTensorArg()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "toy-networks", "same_tensor_arg"),
        vis_graph_with_dynamo_explain=True
    )


def test_multiple_inputs_arg(default_input1, default_input2, default_input3):
    model = example_models.MultiInputs()
    assert validate_saved_activations(
        model, [default_input1, default_input2, default_input3]
    )
    show_model_graph(
        model,
        [default_input1, default_input2, default_input3],
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "toy-networks", "multiple_inputs_arg"),
        vis_graph_with_dynamo_explain=True
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
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "multiple_inputs_kwarg"
        ),
        vis_graph_with_dynamo_explain=True
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
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "multiple_inputs_arg_kwarg_mix"
        ),
        vis_graph_with_dynamo_explain=True
    )


def test_list_input(default_input1, default_input2, default_input3):
    model = example_models.ListInput()
    assert validate_saved_activations(
        model, [[default_input1, default_input2, default_input3]]
    )
    show_model_graph(
        model,
        [[default_input1, default_input2, default_input3]],
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "toy-networks", "list_inputs"),
        vis_graph_with_dynamo_explain=True
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
        vis_outpath=opj("visualization_outputs", "toy-networks", "dict_inputs"),
        vis_graph_with_dynamo_explain=True
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
        vis_outpath=opj("visualization_outputs", "toy-networks", "nested_inputs"),
        vis_graph_with_dynamo_explain=True
    )


def test_multi_outputs(default_input1):
    model = example_models.MultiOutputs()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "toy-networks", "multi_outputs"),
        vis_graph_with_dynamo_explain=True
    )


def test_list_output(default_input1):
    model = example_models.ListOutput()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "toy-networks", "list_output"),
        vis_graph_with_dynamo_explain=True
    )


def test_dict_output(default_input1):
    model = example_models.DictOutput()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "toy-networks", "dict_output"),
        vis_graph_with_dynamo_explain=True
    )


def test_nested_output(default_input1):
    model = example_models.NestedOutput()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "toy-networks", "nested_output"),
        vis_graph_with_dynamo_explain=True
    )


def test_buffer_model():
    model = example_models.BufferModel()
    model_input = torch.rand(12, 12)
    assert validate_saved_activations(model, model_input)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "toy-networks", "buffer_visible"),
        vis_buffer_layers=True,
        vis_graph_with_dynamo_explain=True
    )
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "toy-networks", "buffer_invisible"),
        vis_buffer_layers=False,
        vis_graph_with_dynamo_explain=True
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
        vis_outpath=opj("visualization_outputs", "toy-networks", "buffer_rewrite_model_visible_unnested_unrolled"),
        vis_buffer_layers=True,
        vis_graph_with_dynamo_explain=True
    )
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_nesting_depth=1,
        vis_outpath=opj("visualization_outputs", "toy-networks", "buffer_rewrite_model_invisible_unnested_unrolled"),
        vis_buffer_layers=False,
        vis_graph_with_dynamo_explain=True
    )
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "toy-networks", "buffer_rewrite_model_visible_nested_unrolled"),
        vis_buffer_layers=True,
    )
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "toy-networks", "buffer_rewrite_model_invisible_nested_unrolled"),
        vis_buffer_layers=False,
        vis_graph_with_dynamo_explain=True
    )
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="rolled",
        vis_nesting_depth=1,
        vis_outpath=opj("visualization_outputs", "toy-networks", "buffer_rewrite_model_visible_unnested_rolled"),
        vis_buffer_layers=True,
        vis_graph_with_dynamo_explain=True
    )
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="rolled",
        vis_nesting_depth=1,
        vis_outpath=opj("visualization_outputs", "toy-networks", "buffer_rewrite_model_invisible_unnested_rolled"),
        vis_buffer_layers=False,
        vis_graph_with_dynamo_explain=True
    )
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj("visualization_outputs", "toy-networks", "buffer_rewrite_model_visible_nested_rolled"),
        vis_buffer_layers=True,
        vis_graph_with_dynamo_explain=True
    )
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj("visualization_outputs", "toy-networks", "buffer_rewrite_model_invisible_nested_rolled"),
        vis_buffer_layers=False,
        vis_graph_with_dynamo_explain=True
    )


def test_simple_branching(default_input1):
    model = example_models.SimpleBranching()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "toy-networks", "simple_branching"),
        vis_graph_with_dynamo_explain=True
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
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "conditional_branching_negative"
        ),
        vis_graph_with_dynamo_explain=True
    )
    show_model_graph(
        model,
        ones_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "conditional_branching_positive"
        ),
        vis_graph_with_dynamo_explain=True
    )


def test_repeated_module(default_input1):
    model = example_models.RepeatedModule()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "toy-networks", "repeated_module"),
        vis_graph_with_dynamo_explain=True
    )


def test_nested_modules(default_input1):
    model = example_models.NestedModules()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "nested_modules_fulldepth"
        ),
        vis_graph_with_dynamo_explain=True
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_nesting_depth=1,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "nested_modules_depth1"
        ),
        vis_graph_with_dynamo_explain=True
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_nesting_depth=2,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "nested_modules_depth2"
        ),
        vis_graph_with_dynamo_explain=True
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_nesting_depth=3,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "nested_modules_depth3"
        ),
        vis_graph_with_dynamo_explain=True
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_nesting_depth=4,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "nested_modules_depth4"
        ),
        vis_graph_with_dynamo_explain=True
    )


def test_orphan_tensors(default_input1):
    model = example_models.OrphanTensors()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "toy-networks", "orphan_tensors"),
        vis_graph_with_dynamo_explain=True
    )


def test_simple_loop_no_param(default_input1):
    model = example_models.SimpleLoopNoParam()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "simple_loop_no_param_unrolled"
        ),
        vis_graph_with_dynamo_explain=True
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "simple_loop_no_param_rolled"
        ),
        vis_graph_with_dynamo_explain=True
    )


def test_same_op_repeat(vector_input):
    model = example_models.SameOpRepeat()
    assert validate_saved_activations(model, vector_input)
    show_model_graph(
        model,
        vector_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "same_op_repeat_unrolled"
        ),
        vis_graph_with_dynamo_explain=True
    )
    show_model_graph(
        model,
        vector_input,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "same_op_repeat_rolled"
        ),
        vis_graph_with_dynamo_explain=True
    )


def test_repeated_op_type_in_loop(default_input1):
    model = example_models.RepeatedOpTypeInLoop()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "same_op_type_in_loop_unrolled"
        ),
        vis_graph_with_dynamo_explain=True
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "same_op_type_in_loop_rolled"
        ),
        vis_graph_with_dynamo_explain=True
    )


def test_varying_loop_noparam1(default_input1):
    model = example_models.VaryingLoopNoParam1()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "varying_loop_noparam1_unrolled"
        ),
        vis_graph_with_dynamo_explain=True
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "varying_loop_noparam1_rolled"
        ),
        vis_graph_with_dynamo_explain=True
    )


@pytest.mark.xfail
def test_varying_loop_noparam2(default_input1):
    model = example_models.VaryingLoopNoParam2()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "varying_loop_noparam2_unrolled"
        ),
        vis_graph_with_dynamo_explain=True
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "varying_loop_noparam2_rolled"
        ),
        vis_graph_with_dynamo_explain=True
    )


def test_varying_loop_withparam(vector_input):
    model = example_models.VaryingLoopWithParam()
    assert validate_saved_activations(model, vector_input)
    show_model_graph(
        model,
        vector_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "varying_loop_withparam_unrolled"
        ),
        vis_graph_with_dynamo_explain=True
    )
    show_model_graph(
        model,
        vector_input,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "varying_loop_withparam_rolled"
        ),
        vis_graph_with_dynamo_explain=True
    )


def test_looping_internal_funcs(default_input1):
    model = example_models.LoopingInternalFuncs()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "looping_internal_funcs_unrolled"
        ),
        vis_graph_with_dynamo_explain=True
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "looping_internal_funcs_rolled"
        ),
        vis_graph_with_dynamo_explain=True
    )


def test_looping_from_inputs1(default_input1, default_input2, default_input3):
    model = example_models.LoopingFromInputs1()
    assert validate_saved_activations(
        model, [default_input1, default_input2, default_input3]
    )
    show_model_graph(
        model,
        [default_input1, default_input2, default_input3],
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "looping_from_inputs1_unrolled"
        ),
        vis_graph_with_dynamo_explain=True
    )
    show_model_graph(
        model,
        [default_input1, default_input2, default_input3],
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "looping_from_inputs1_rolled"
        ),
        vis_graph_with_dynamo_explain=True
    )


def test_looping_from_inputs2(default_input1, default_input2, default_input3):
    model = example_models.LoopingFromInputs2()
    assert validate_saved_activations(
        model, [[default_input1, default_input2, default_input3]]
    )
    show_model_graph(
        model,
        [[default_input1, default_input2, default_input3]],
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "looping_from_inputs2_unrolled"
        ),
        vis_graph_with_dynamo_explain=True
    )
    show_model_graph(
        model,
        [[default_input1, default_input2, default_input3]],
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "looping_from_inputs2_rolled"
        ),
        vis_graph_with_dynamo_explain=True
    )


def test_looping_inputs_and_outputs(default_input1, default_input2, default_input3):
    model = example_models.LoopingInputsAndOutputs()
    assert validate_saved_activations(
        model, [[default_input1, default_input2, default_input3]]
    )
    show_model_graph(
        model,
        [[default_input1, default_input2, default_input3]],
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs",
            "toy-networks",
            "looping_inputs_and_outputs_unrolled",
        ),
        vis_graph_with_dynamo_explain=True
    )
    show_model_graph(
        model,
        [[default_input1, default_input2, default_input3]],
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "looping_inputs_and_outputs_rolled"
        ),
        vis_graph_with_dynamo_explain=True
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
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "stochastic_loop_unrolled1"
        ),
        vis_graph_with_dynamo_explain=True
    )
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "stochastic_loop_rolled1"
        ),
    )
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "stochastic_loop_unrolled2"
        ),
    )
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "stochastic_loop_rolled2"
        ),
    )
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "stochastic_loop_unrolled3"
        ),
    )
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "stochastic_loop_rolled3"
        ),
    )

'''

def test_recurrent_params_simple(input_2d):
    print("Running test_recurrent_params_simple")
    model = wbreaks_example_models.RecurrentParamsSimple()
    assert validate_saved_activations(model, input_2d)
    show_model_graph(
        model,
        input_2d,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "recurrent_params_simple_unrolled"
        ),
        vis_fileformat="svg",
        vis_graph_with_dynamo_explain=True
    )
    show_model_graph(
        model,
        input_2d,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "recurrent_params_simple_rolled"
        ),
        vis_fileformat="svg",
        vis_graph_with_dynamo_explain=True
    )


def test_recurrent_params_complex(input_2d):
    print("Running test_recurrent_params_complex")
    model = wbreaks_example_models.RecurrentParamsComplex()
    assert validate_saved_activations(model, input_2d)
    show_model_graph(
        model,
        input_2d,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "recurrent_params_complex_unrolled"
        ),
        vis_fileformat="svg",
        vis_graph_with_dynamo_explain=True
    )
    show_model_graph(
        model,
        input_2d,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "recurrent_params_complex_rolled"
        ),
        vis_fileformat="svg",
        vis_graph_with_dynamo_explain=True
    )

'''
def test_looping_params_doublenested(input_2d):
    print("Running test_looping_params_doublenested")
    model = example_models.LoopingParamsDoubleNested()
    assert validate_saved_activations(model, input_2d)
    show_model_graph(
        model,
        input_2d,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs",
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
            "visualization_outputs",
            "toy-networks",
            "looping_params_doublenested_rolled",
        ),
    )

def test_module_looping_clash1(default_input1):
    print("Running test_module_looping_clash1")
    model = example_models.ModuleLoopingClash1()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "module_looping_clash1_unrolled"
        ),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "module_looping_clash1_rolled"
        ),
    )


def test_module_looping_clash2(default_input1):
    print("Running test_module_looping_clash2")
    model = example_models.ModuleLoopingClash2()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "module_looping_clash2_unrolled"
        ),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "module_looping_clash2_rolled"
        ),
    )


def test_module_looping_clash3(default_input1):
    print("Running test_module_looping_clash3")
    model = example_models.ModuleLoopingClash3()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "module_looping_clash3_unrolled"
        ),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(
            "visualization_outputs", "toy-networks", "module_looping_clash3_rolled"
        ),
    )

def test_propertymodel(input_complex):
    model = example_models.PropertyModel()
    assert validate_saved_activations(model, input_complex)
    show_model_graph(
        model,
        input_complex,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "toy-networks", "propertymodel"),
    )
'''

def test_ubermodel1(input_2d):
    print("Running test_ubermodel1")
    model = wbreaks_example_models.UberModel1()
    inputs = (input_2d, input_2d * 2, input_2d * 3)
    assert validate_saved_activations(model, inputs)
    show_model_graph(
        model,
        inputs,  # Pass inputs as a tuple, otherwise it won't work as a list
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "toy-networks", "ubermodel1"),
        vis_fileformat="svg",
        vis_graph_with_dynamo_explain=True
    )

'''
def test_ubermodel2():
    print("Running test_ubermodel2")
    model = example_models.UberModel2()
    model_input = torch.rand(2, 1, 3, 3)
    assert validate_saved_activations(model, model_input)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "toy-networks", "ubermodel2"),
    )

def test_ubermodel3(input_2d):
    model = example_models.UberModel3()
    assert validate_saved_activations(model, input_2d)
    show_model_graph(
        model,
        input_2d,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "toy-networks", "ubermodel3_unrolled"),
    )
    show_model_graph(
        model,
        input_2d,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj("visualization_outputs", "toy-networks", "ubermodel3_rolled"),
    )


def test_ubermodel4(input_2d):
    model = example_models.UberModel4()
    assert validate_saved_activations(model, input_2d)
    show_model_graph(
        model,
        input_2d,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "toy-networks", "ubermodel4_unrolled"),
    )
    show_model_graph(
        model,
        input_2d,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj("visualization_outputs", "toy-networks", "ubermodel4_rolled"),
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
        vis_outpath=opj("visualization_outputs", "toy-networks", "ubermodel5"),
    )


def test_ubermodel6(default_input1):
    model = example_models.UberModel6()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "toy-networks", "ubermodel6_unrolled"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj("visualization_outputs", "toy-networks", "ubermodel6_rolled"),
    )


def test_ubermodel7(input_2d):
    model = example_models.UberModel7()
    assert validate_saved_activations(model, input_2d)
    show_model_graph(
        model,
        input_2d,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "toy-networks", "ubermodel7_unrolled"),
    )
    show_model_graph(
        model,
        input_2d,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj("visualization_outputs", "toy-networks", "ubermodel7_rolled"),
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
        vis_outpath=opj("visualization_outputs", "toy-networks", "ubermodel8"),
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
        vis_outpath=opj("visualization_outputs", "toy-networks", "ubermodel9"),
    )
'''


'''
# Torchvision Main Models


def test_alexnet(default_input1):
    print("Running test_alexnet")
    model = torchvision.models.AlexNet()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "torchvision-main", "alexnet"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_nesting_depth=1,
        vis_outpath=opj("visualization_outputs", "torchvision-main", "alexnet_depth1"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_nesting_depth=2,
        vis_outpath=opj("visualization_outputs", "torchvision-main", "alexnet_depth2"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_nesting_depth=3,
        vis_outpath=opj("visualization_outputs", "torchvision-main", "alexnet_depth3"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_nesting_depth=4,
        vis_outpath=opj("visualization_outputs", "torchvision-main", "alexnet_depth4"),
    )

def test_googlenet(default_input1):
    print("Running test_googlenet")
    model = torchvision.models.GoogLeNet()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_buffer_layers=True,
        vis_outpath=opj(
            "visualization_outputs", "torchvision-main", "googlenet_showbuffer"
        ),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_buffer_layers=False,
        vis_outpath=opj(
            "visualization_outputs", "torchvision-main", "googlenet_nobuffer"
        ),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_buffer_layers=False,
        vis_outpath=opj(
            "visualization_outputs", "torchvision-main", "googlenet_nobuffer_rolled"
        ),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_buffer_layers=True,
        vis_nesting_depth=1,
        vis_outpath=opj(
            "visualization_outputs", "torchvision-main", "googlenet_showbuffer_depth1"
        ),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_buffer_layers=False,
        vis_nesting_depth=1,
        vis_outpath=opj(
            "visualization_outputs", "torchvision-main", "googlenet_nobuffer_depth1"
        ),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_buffer_layers=True,
        vis_nesting_depth=2,
        vis_outpath=opj(
            "visualization_outputs", "torchvision-main", "googlenet_showbuffer_depth2"
        ),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_buffer_layers=False,
        vis_nesting_depth=2,
        vis_outpath=opj(
            "visualization_outputs", "torchvision-main", "googlenet_nobuffer_depth2"
        ),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_buffer_layers=True,
        vis_nesting_depth=3,
        vis_outpath=opj(
            "visualization_outputs", "torchvision-main", "googlenet_showbuffer_depth3"
        ),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_buffer_layers=False,
        vis_nesting_depth=3,
        vis_outpath=opj(
            "visualization_outputs", "torchvision-main", "googlenet_nobuffer_depth3"
        ),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_buffer_layers=True,
        vis_nesting_depth=4,
        vis_outpath=opj(
            "visualization_outputs", "torchvision-main", "googlenet_showbuffer_depth4"
        ),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_buffer_layers=False,
        vis_nesting_depth=4,
        vis_outpath=opj(
            "visualization_outputs", "torchvision-main", "googlenet_nobuffer_depth4"
        ),
    )
'''

'''
def test_vgg16(default_input1):
    model = torchvision.models.vgg16()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "torchvision-main", "vgg16"),
    )


def test_vgg19(default_input1):
    model = torchvision.models.vgg19()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "torchvision-main", "vgg19"),
    )


def test_resnet50(default_input1):
    model = torchvision.models.resnet50()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "torchvision-main", "resnet50"),
    )


def test_resnet101(default_input1):
    model = torchvision.models.resnet101()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "torchvision-main", "resnet101"),
    )


def test_resnet152(default_input1):
    model = torchvision.models.resnet152()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "torchvision-main", "resnet152"),
    )


def test_convnext_large(default_input1):
    model = torchvision.models.convnext_large()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "torchvision-main", "convnext_large"),
    )


def test_densenet121(default_input1):
    model = torchvision.models.densenet121()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "torchvision-main", "densenet121"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_nesting_depth=1,
        vis_outpath=opj(
            "visualization_outputs", "torchvision-main", "densenet121_depth1"
        ),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_nesting_depth=2,
        vis_outpath=opj(
            "visualization_outputs", "torchvision-main", "densenet121_depth2"
        ),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_nesting_depth=3,
        vis_outpath=opj(
            "visualization_outputs", "torchvision-main", "densenet121_depth3"
        ),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_nesting_depth=4,
        vis_outpath=opj(
            "visualization_outputs", "torchvision-main", "densenet121_depth4"
        ),
    )


def test_efficientnet_b6(default_input1):
    model = torchvision.models.efficientnet_b6()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "torchvision-main", "efficientnet_b6"),
    )


def test_squeezenet(default_input1):
    model = torchvision.models.squeezenet1_1()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "torchvision-main", "squeezenet"),
    )


def test_mobilenet(default_input1):
    model = torchvision.models.mobilenet_v3_large()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs", "torchvision-main", "mobilenet_vg_large"
        ),
    )


def test_wide_resnet(default_input1):
    model = torchvision.models.wide_resnet101_2()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs", "torchvision-main", "wide_resnet101_2"
        ),
    )


def test_mnasnet(default_input1):
    model = torchvision.models.mnasnet1_3()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "torchvision-main", "mnasnet1_3"),
    )


def test_shufflenet(default_input1):
    model = torchvision.models.shufflenet_v2_x1_5()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs", "torchvision-main", "shufflenet_v2_x1_5"
        ),
    )


def test_resnext(default_input1):
    model = torchvision.models.resnext101_64x4d()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs", "torchvision-main", "resnext101_64x4d"
        ),
    )


def test_regnet(default_input1):
    model = torchvision.models.regnet_x_32gf()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "torchvision-main", "regnet_x_32gf"),
    )


def test_swin_v2b(default_input1):
    model = torchvision.models.swin_v2_b()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "torchvision-main", "swin_v2b"),
    )


def test_vit(default_input1):
    model = torchvision.models.vit_l_16()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "torchvision-main", "vit_l_16"),
    )


def test_maxvit(default_input1):
    model = torchvision.models.maxvit_t()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "torchvision-main", "max_vit_t"),
    )


def test_inception_v3():
    model = torchvision.models.inception_v3()
    model_input = torch.randn(2, 3, 299, 299)
    assert validate_saved_activations(model, model_input)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "torchvision-main", "inception_v3"),
    )



# Cornet Models

def test_cornet_s(default_input1):
    model = cornet.cornet_s()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "cornet", "cornet_s_unrolled"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj("visualization_outputs", "cornet", "cornet_s_rolled"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_nesting_depth=1,
        vis_outpath=opj("visualization_outputs", "cornet", "cornet_s_unrolled_depth1"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_nesting_depth=1,
        vis_outpath=opj("visualization_outputs", "cornet", "cornet_s_rolled_depth1"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_nesting_depth=2,
        vis_outpath=opj("visualization_outputs", "cornet", "cornet_s_unrolled_depth2"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_nesting_depth=2,
        vis_outpath=opj("visualization_outputs", "cornet", "cornet_s_rolled_depth2"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_nesting_depth=3,
        vis_outpath=opj("visualization_outputs", "cornet", "cornet_s_unrolled_depth3"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_nesting_depth=3,
        vis_outpath=opj("visualization_outputs", "cornet", "cornet_s_rolled_depth3"),
    )


def test_cornet_r(default_input1):
    model = cornet.cornet_r()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "cornet", "cornet_r_unrolled"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj("visualization_outputs", "cornet", "cornet_r_rolled"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_nesting_depth=1,
        vis_outpath=opj("visualization_outputs", "cornet", "cornet_r_unrolled_depth1"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_nesting_depth=1,
        vis_outpath=opj("visualization_outputs", "cornet", "cornet_r_rolled_depth1"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_nesting_depth=2,
        vis_outpath=opj("visualization_outputs", "cornet", "cornet_r_unrolled_depth2"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_nesting_depth=2,
        vis_outpath=opj("visualization_outputs", "cornet", "cornet_r_rolled_depth2"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_nesting_depth=3,
        vis_outpath=opj("visualization_outputs", "cornet", "cornet_r_unrolled_depth3"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_nesting_depth=3,
        vis_outpath=opj("visualization_outputs", "cornet", "cornet_r_rolled_depth3"),
    )


def test_cornet_rt():
    model = cornet.cornet_rt()
    model_input = torch.rand((6, 3, 224, 224)).to("cuda")
    assert validate_saved_activations(model, model_input)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "cornet", "cornet_rt_unrolled"),
    )
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj("visualization_outputs", "cornet", "cornet_rt_rolled"),
    )
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_nesting_depth=1,
        vis_outpath=opj("visualization_outputs", "cornet", "cornet_rt_unrolled_depth1"),
    )
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="rolled",
        vis_nesting_depth=1,
        vis_outpath=opj("visualization_outputs", "cornet", "cornet_rt_rolled_depth1"),
    )
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_nesting_depth=2,
        vis_outpath=opj("visualization_outputs", "cornet", "cornet_rt_unrolled_depth2"),
    )
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="rolled",
        vis_nesting_depth=2,
        vis_outpath=opj("visualization_outputs", "cornet", "cornet_rt_rolled_depth2"),
    )
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_nesting_depth=3,
        vis_outpath=opj("visualization_outputs", "cornet", "cornet_rt_unrolled_depth3"),
    )
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="rolled",
        vis_nesting_depth=3,
        vis_outpath=opj("visualization_outputs", "cornet", "cornet_rt_rolled_depth3"),
    )


def test_cornet_z(default_input1):
    model = cornet.cornet_z()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "cornet", "cornet_z_unrolled"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj("visualization_outputs", "cornet", "cornet_z_rolled"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_nesting_depth=1,
        vis_outpath=opj("visualization_outputs", "cornet", "cornet_z_unrolled_depth1"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_nesting_depth=1,
        vis_outpath=opj("visualization_outputs", "cornet", "cornet_z_rolled_depth1"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_nesting_depth=2,
        vis_outpath=opj("visualization_outputs", "cornet", "cornet_z_unrolled_depth2"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_nesting_depth=2,
        vis_outpath=opj("visualization_outputs", "cornet", "cornet_z_rolled_depth2"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_nesting_depth=3,
        vis_outpath=opj("visualization_outputs", "cornet", "cornet_z_unrolled_depth3"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_nesting_depth=3,
        vis_outpath=opj("visualization_outputs", "cornet", "cornet_z_rolled_depth3"),
    )

# Torchvision Segmentation Models
def test_segment_deeplab_v3_resnet50(default_input1):
    print("Running test_segment_deeplab_v3_resnet50")
    model = torchvision.models.segmentation.deeplabv3_resnet50()
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs",
            "torchvision-segmentation",
            "segment_deeplabv3_resnet50",
        ),
    )
    assert validate_saved_activations(model, default_input1)

def test_segment_deeplabv3_resnet101(default_input1):
    model = torchvision.models.segmentation.deeplabv3_resnet101()
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs",
            "torchvision-segmentation",
            "segment_deeplabv3_resnet101",
        ),
    )
    assert validate_saved_activations(model, default_input1)


def test_segment_deeplabv3_mobilenet(default_input1):
    model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large()
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs",
            "torchvision-segmentation",
            "segment_deeplabv3_mobilenet_v3_large",
        ),
    )
    assert validate_saved_activations(model, default_input1)

def test_segment_lraspp_mobilenet(default_input1):
    model = torchvision.models.segmentation.lraspp_mobilenet_v3_large()
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs",
            "torchvision-segmentation",
            "segment_lraspp_mobilenet_v3_large",
        ),
    )
    assert validate_saved_activations(model, default_input1)


def test_segment_fcn_resnet50(default_input1):
    model = torchvision.models.segmentation.fcn_resnet50()
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs", "torchvision-segmentation", "segment_fcn_resnet50"
        ),
    )
    assert validate_saved_activations(model, default_input1)


def test_segment_fcn_resnet101(default_input1):
    model = torchvision.models.segmentation.fcn_resnet101()
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs", "torchvision-segmentation", "segment_fcn_resnet101"
        ),
    )
    assert validate_saved_activations(model, default_input1)
'''

'''

# Torchvision Detection Models


def test_fasterrcnn_mobilenet_train(default_input1, default_input2):
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn()
    input_tensors = [default_input1[0], default_input2[0]]
    targets = [
        {
            "boxes": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
            "labels": torch.tensor([1, 2]),
        },
        {
            "boxes": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
            "labels": torch.tensor([1, 2]),
        },
    ]
    model_inputs = (input_tensors, targets)
    show_model_graph(
        model,
        model_inputs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs",
            "torchvision-detection",
            "detect_fasterrcnn_mobilenet_v3_large_320_fpn_train",
        ),
    )
    assert validate_saved_activations(model, model_inputs)

def test_fasterrcnn_mobilenet_eval(default_input1, default_input2):
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn()
    input_tensors = [default_input1[0], default_input2[0]]
    model = model.eval()
    show_model_graph(
        model,
        [input_tensors],
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs",
            "torchvision-detection",
            "detect_fasterrcnn_mobilenet_v3_large_320_fpn_eval",
        ),
    )
    assert validate_saved_activations(model, [input_tensors])


def test_fcos_resnet50_train(default_input1, default_input2):
    model = torchvision.models.detection.fcos_resnet50_fpn()
    input_tensors = [default_input1[0], default_input2[0]]
    targets = [
        {
            "boxes": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
            "labels": torch.tensor([1, 2]),
        },
        {
            "boxes": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
            "labels": torch.tensor([1, 2]),
        },
    ]
    model_inputs = (input_tensors, targets)
    show_model_graph(
        model,
        model_inputs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs",
            "torchvision-detection",
            "detect_fcos_resnet50_fpn_train",
        ),
        random_seed=1,
    )
    assert validate_saved_activations(model, model_inputs, random_seed=1)


def test_fcos_resnet50_eval(default_input1, default_input2):
    model = torchvision.models.detection.fcos_resnet50_fpn()
    input_tensors = [default_input1[0], default_input2[0]]
    model = model.eval()
    show_model_graph(
        model,
        [input_tensors],
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs",
            "torchvision-detection",
            "detect_fcos_resnet50_fpn_eval",
        ),
    )
    assert validate_saved_activations(model, [input_tensors])


def test_retinanet_resnet50_train(default_input1, default_input2):
    model = torchvision.models.detection.retinanet_resnet50_fpn()
    input_tensors = [default_input1[0], default_input2[0]]
    targets = [
        {
            "boxes": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
            "labels": torch.tensor([1, 2]),
        },
        {
            "boxes": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
            "labels": torch.tensor([1, 2]),
        },
    ]
    model_inputs = (input_tensors, targets)
    show_model_graph(
        model,
        model_inputs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs",
            "torchvision-detection",
            "detect_retinanet_resnet50_fpn_train",
        ),
    )
    assert validate_saved_activations(model, model_inputs)


def test_retinanet_resnet50_eval(default_input1, default_input2):
    model = torchvision.models.detection.retinanet_resnet50_fpn()
    input_tensors = [default_input1[0], default_input2[0]]
    model = model.eval()
    show_model_graph(
        model,
        [input_tensors],
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs",
            "torchvision-detection",
            "detect_retinanet_resnet50_fpn_eval",
        ),
    )
    assert validate_saved_activations(model, [input_tensors])


def test_ssd300_vgg16_train(default_input1, default_input2):
    model = torchvision.models.detection.ssd300_vgg16()
    input_tensors = [default_input1[0], default_input2[0]]
    targets = [
        {
            "boxes": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
            "labels": torch.tensor([1, 2]),
        },
        {
            "boxes": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
            "labels": torch.tensor([1, 2]),
        },
    ]
    model_inputs = (input_tensors, targets)
    show_model_graph(
        model,
        model_inputs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs",
            "torchvision-detection",
            "detect_ssd300_vgg16_train",
        ),
    )
    assert validate_saved_activations(model, model_inputs)


def test_ssd300_vgg16_eval(default_input1, default_input2):
    model = torchvision.models.detection.ssd300_vgg16()
    input_tensors = [default_input1[0], default_input2[0]]
    model = model.eval()
    show_model_graph(
        model,
        [input_tensors],
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs", "torchvision-detection", "detect_ssd300_vgg16_eval"
        ),
    )
    assert validate_saved_activations(model, [input_tensors])

def test_quantize_resnet50(default_input1):
    model = torchvision.models.quantization.resnet50()
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs", "torchvision-quantize", "quantize_resnet50"
        ),
    )
    assert validate_saved_activations(model, default_input1)

def test_quantize_resnext101_64x4d(default_input1):
    model = torchvision.models.quantization.resnext101_64x4d()
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs", "torchvision-quantize", "quantize_resnetx101_64x4d"
        ),
    )
    assert validate_saved_activations(model, default_input1)


def test_quantize_shufflenet_v2_1x(default_input1):
    model = torchvision.models.quantization.shufflenet_v2_x1_5()
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs",
            "torchvision-quantize",
            "quantize_shufflenet_v2_x1_5",
        ),
    )
    assert validate_saved_activations(model, default_input1)


def test_quantize_mobilenet_v3_large(default_input1):
    model = torchvision.models.quantization.mobilenet_v3_large()
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs",
            "torchvision-quantize",
            "quantize_mobilenet_v3_large",
        ),
    )
    assert validate_saved_activations(model, default_input1)


def test_quantize_googlenet(default_input1):
    model = torchvision.models.quantization.googlenet()
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs", "torchvision-quantize", "quantize_googlenet"
        ),
    )
    assert validate_saved_activations(model, default_input1)


def test_quantize_inception_v3():
    model = torchvision.models.quantization.inception_v3()
    model_input = torch.randn(2, 3, 299, 299)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs", "torchvision-quantize", "quantize_inception_v3"
        ),
    )
    assert validate_saved_activations(model, model_input)

def test_taskonomy(default_input1):
    model = visualpriors.taskonomy_network.TaskonomyNetwork()
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "taskonomy", "taskonomy"),
    )
    assert validate_saved_activations(model, default_input1)


def test_video_mc3_18():
    model = torchvision.models.video.mc3_18()
    model_input = torch.randn(6, 3, 16, 112, 112)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "torchvision-video", "video_mc3_18"),
    )
    assert validate_saved_activations(model, model_input)

def test_video_mvit_v2_s():
    model = torchvision.models.video.mvit_v2_s()
    model_input = torch.randn(16, 3, 448, 896)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs", "torchvision-video", "video_mvit_v2_s"
        ),
    )


def test_video_r2plus1_18():
    model = torchvision.models.video.r2plus1d_18()
    model_input = torch.randn(16, 3, 16, 112, 112)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs", "torchvision-video", "video_r2plus1d_18"
        ),
    )
    assert validate_saved_activations(model, model_input)


def test_video_r3d_18():
    model = torchvision.models.video.r3d_18()
    model_input = torch.randn(16, 3, 16, 112, 112)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "torchvision-video", "video_r3d_18"),
    )
    assert validate_saved_activations(model, model_input)


def test_video_s3d():
    model = torchvision.models.video.s3d()
    model_input = torch.randn(16, 3, 16, 224, 224)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "torchvision-video", "video_s3d"),
    )
    assert validate_saved_activations(model, model_input)
'''


'''
def test_opticflow_raftlarge():  # TODO: the loop-finding prccedure messes up on this somehow.
    model = torchvision.models.optical_flow.raft_large()
    model_input = [torch.rand(6, 3, 224, 224), torch.rand(6, 3, 224, 224)]
    show_model_graph(
        model,
        model_input,
        save_only=True,
        random_seed=1,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs", "torchvision-opticflow", "opticflow_raftlarge"
        ),
    )
    assert validate_saved_activations(model, model_input, random_seed=1)

# TIMM models


def test_timm_adv_inception_v3(default_input1):
    model = timm.models.adv_inception_v3(pretrained=True)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "timm", "timm_adv_inception_v3"),
    )
    assert validate_saved_activations(model, default_input1)

def test_timm_beit_base_patch16_224(default_input1):
    model = timm.models.beit_base_patch16_224()
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "timm", "timm_beit_base_patch16_224"),
    )
    assert validate_saved_activations(model, default_input1)


def test_timm_cait_s24_224(default_input1):
    model = timm.models.cait_s24_224()
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "timm", "timm_cait_s24_224"),
    )
    assert validate_saved_activations(model, default_input1)


def test_timm_coat_mini(default_input1):
    model = timm.models.coat_mini()
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "timm", "timm_coat_mini"),
    )
    assert validate_saved_activations(model, default_input1)


def test_timm_convit_base(default_input1):
    model = timm.create_model("convit_base", pretrained=True)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "timm", "timm_convit_base"),
    )
    assert validate_saved_activations(model, default_input1)


def test_timm_darknet21():
    model = timm.create_model("darknet21", pretrained=False)
    model_input = torch.randn(1, 3, 224, 224)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "timm", "timm_darknet21"),
    )
    assert validate_saved_activations(model, model_input)


def test_timm_ghostnet_100():
    model = timm.create_model("ghostnet_100", pretrained=True)
    model_input = torch.randn(1, 3, 224, 224)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "timm", "timm_ghostnet_100"),
    )
    assert validate_saved_activations(model, model_input)


def test_timm_mixnet_m():
    model = timm.create_model("mixnet_m", pretrained=False)
    model_input = torch.randn(1, 3, 224, 224)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "timm", "timm_mixnet_m"),
    )
    assert validate_saved_activations(model, model_input)


def test_timm_poolformer_s24():
    model = timm.create_model("poolformer_s24", pretrained=False)
    model_input = torch.randn(1, 3, 224, 224)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "timm", "timm_poolformer_s24"),
    )
    assert validate_saved_activations(model, model_input)


def test_timm_resnest14d():
    model = timm.create_model("resnest14d", pretrained=False)
    model_input = torch.randn(6, 3, 224, 224)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "timm", "timm_resnest14d"),
    )
    assert validate_saved_activations(model, model_input)


def test_timm_edgenext_small():
    model = timm.create_model("edgenext_small", pretrained=False)
    model_input = torch.randn(1, 3, 224, 224)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "timm", "timm_edgenext_small"),
    )
    assert validate_saved_activations(model, model_input)


def test_timm_gluon_resnext101_32x4d():
    model = timm.create_model("gluon_resnext101_32x4d", pretrained=False)
    model_input = torch.randn(1, 3, 224, 224)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "timm", "gluon_resnext101_32x4d"),
    )
    assert validate_saved_activations(model, model_input)


def test_timm_hardcorenas_f():
    model = timm.create_model("hardcorenas_f", pretrained=False)
    model_input = torch.randn(1, 3, 224, 224)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "timm", "hardcorenas_f"),
    )
    assert validate_saved_activations(model, model_input)


def test_timm_semnasnet_100():
    model = timm.create_model("semnasnet_100", pretrained=False)
    model_input = torch.randn(1, 3, 224, 224)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "timm", "semnasnet_100"),
    )
    assert validate_saved_activations(model, model_input)


def test_timm_xcit_tiny_24_p8_224():
    model = timm.create_model("xcit_tiny_24_p8_224", pretrained=False)
    model_input = torch.randn(1, 3, 224, 224)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "timm", "xcit_tiny_24_p8_224"),
    )
    assert validate_saved_activations(model, model_input)


def test_timm_seresnet152():
    model = timm.create_model("seresnet152", pretrained=False)
    model_input = torch.randn(1, 3, 224, 224)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "timm", "seresnet152"),
    )
    assert validate_saved_activations(model, model_input)


def test_timm_ecaresnet101d():
    model = timm.create_model("ecaresnet101d", pretrained=False)
    model_input = torch.randn(1, 3, 224, 224)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "timm", "ecaresnet101d"),
    )
    assert validate_saved_activations(model, model_input)

# Torchaudio


def test_audio_conv_tasnet_base():
    model = torchaudio.models.conv_tasnet_base()
    model_input = torch.randn(6, 1, 3)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs", "torchaudio", "audio_conv_tasnet_base"
        ),
    )
    assert validate_saved_activations(model, model_input)

def test_audio_hubert_base():
    model = torchaudio.models.hubert_base()
    model_input = torch.randn(12, 2000)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "torchaudio", "audio_hubert_base"),
        random_seed=1,
    )
    assert validate_saved_activations(model, model_input, random_seed=1)


def test_audio_hubert_large():
    model = torchaudio.models.hubert_large()
    model_input = torch.randn(12, 2000)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "torchaudio", "audio_hubert_large"),
    )
    assert validate_saved_activations(model, model_input)


def test_audio_hubert_xlarge():
    model = torchaudio.models.hubert_xlarge()
    model_input = torch.randn(12, 2000)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "torchaudio", "audio_hubert_xlarge"),
    )
    assert validate_saved_activations(model, model_input)


def test_audio_wav2vec2_base():
    model = torchaudio.models.wav2vec2_base()
    model_input = torch.randn(12, 2000)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "torchaudio", "audio_wave2vec2_base"),
    )
    assert validate_saved_activations(model, model_input)


def test_audio_wav2vec2_large():
    model = torchaudio.models.wav2vec2_large()
    model_input = torch.randn(12, 2000)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "torchaudio", "audio_wave2vec2_large"),
    )
    assert validate_saved_activations(model, model_input)


def test_audio_wav2vec2_large_lv60k():
    model = torchaudio.models.wav2vec2_large_lv60k()
    model_input = torch.randn(12, 2000)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs", "torchaudio", "audio_wave2vec2_large_lv60k"
        ),
    )
    assert validate_saved_activations(model, model_input)


def test_audio_wav2letter():
    model = torchaudio.models.Wav2Letter()
    model_input = torch.randn(1, 1, 2000)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "torchaudio", "audio_wav2letter"),
    )
    assert validate_saved_activations(model, model_input)


def test_deepspeech():
    model = torchaudio.models.DeepSpeech(3)
    model_input = torch.randn(12, 2000, 3)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "torchaudio", "audio_deepspeech"),
    )
    assert validate_saved_activations(model, model_input)

# Language models

def test_lstm():
    model = example_models.LSTMModel()
    model_input = torch.rand(5, 5, 5)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "language-models", "language_lstm_unrolled"),
    )
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj("visualization_outputs", "language-models", "language_lstm_rolled"),
    )
    assert validate_saved_activations(model, model_input)

def test_rnn():
    model = example_models.RNNModel()
    model_input = torch.rand(5, 5, 5)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "language-models", "language_rnn_unrolled"),
    )
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj("visualization_outputs", "language-models", "language_rnn_rolled"),
    )
    assert validate_saved_activations(model, model_input)


def test_gpt2():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2")
    model_inputs = ["to be or not to be", "that is the question"]
    model_inputs = tokenizer(*model_inputs, return_tensors="pt")
    show_model_graph(
        model,
        [],
        model_inputs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "language-models", "gpt2"),
    )
    assert validate_saved_activations(model, [], model_inputs)


def test_bert():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")
    model_inputs = ("to be or not to be", "that is the question")
    model_inputs = tokenizer(*model_inputs, return_tensors="pt")
    model_inputs["labels"] = torch.LongTensor([1])
    show_model_graph(
        model,
        [],
        model_inputs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "language-models", "bert"),
    )
    assert validate_saved_activations(model, [], model_inputs)

# Multimodal


def test_clip():  # for some reason CLIP breaks the PyCharm debugger
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    image = Image.fromarray(np.random.random((640, 480, 3)).astype(np.uint8))
    model_inputs = processor(
        text=["a photo of a cat", "a photo of a dog"],
        images=image,
        return_tensors="pt",
        padding=True,
    )
    show_model_graph(
        model,
        [],
        model_inputs,
        random_seed=1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "multimodal-models", "clip"),
    )
    assert validate_saved_activations(model, [], model_inputs, random_seed=1)

def test_stable_diffusion():
    try:
        import UNet
    except ModuleNotFoundError:
        pytest.skip()

    model = UNet(3, 16, 10)
    model_inputs = (torch.rand(6, 3, 224, 224), torch.tensor([1]), torch.tensor([1.]), torch.tensor([3.]))
    show_model_graph(
        model,
        model_inputs,
        random_seed=1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "multimodal-models", "stable_diffusion"),
    )
    assert validate_saved_activations(model, model_inputs, random_seed=1)


# Text to speech

def test_styletts():
    try:
        from StyleTTS.models import TextEncoder
    except ModuleNotFoundError:
        pytest.skip()

    model = TextEncoder(3, 3, 3, 100)
    tokens = torch.tensor([[3, 0, 1, 2, 0, 2, 2, 3, 1, 4]])
    input_lengths = torch.ones(1, dtype=torch.long) * 10
    m = torch.ones(1, 10)
    model_inputs = (tokens, input_lengths, m)
    show_model_graph(
        model,
        model_inputs,
        random_seed=1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "text-to-speech", "styletts_text_encoder"),
    )
    assert validate_saved_activations(model, model_inputs, random_seed=1)

# Graph neural networks


def test_dimenet():
    model = DimeNet(6, 3, 4, 2, 6, 3)
    z = torch.tensor([6, 1, 1, 1, 1])
    pose = torch.tensor(
        [
            [-1.2700e-02, 1.0858e00, 8.0000e-03],
            [2.2000e-03, -6.0000e-03, 2.0000e-03],
            [1.0117e00, 1.4638e00, 3.0000e-04],
            [-5.4080e-01, 1.4475e00, -8.7660e-01],
            [-5.2380e-01, 1.4379e00, 9.0640e-01],
        ]
    )
    batch = None
    model_inputs = (z, pose, batch)
    show_model_graph(
        model,
        model_inputs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "graph-neural-networks", "dimenet"),
    )
    assert validate_saved_activations(model, model_inputs)

# Quantum machine-learning model

def test_qml():
    import pennylane as qml
    n_qubits = 2
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, diff_method="backprop")
    def qnode(inputs, weights):
        # print(inputs)
        qml.RX(inputs[0][0], wires=0)
        qml.RY(weights, wires=0)
        return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]

    weight_shapes = {"weights": 1}
    qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

    clayer_1 = torch.nn.Linear(2, 2)
    clayer_2 = torch.nn.Linear(2, 2)
    softmax = torch.nn.Softmax(dim=1)
    layers = [clayer_1, qlayer, clayer_2, softmax]
    model = torch.nn.Sequential(*layers)
    model_inputs = torch.rand(1, 2, requires_grad=False)

    show_model_graph(
        model,
        model_inputs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "quantum", "qml"),
    )
    assert validate_saved_activations(model, model_inputs)


# Lightning modules

def test_lightning():
    try:
        import lightning as L
    except ModuleNotFoundError:
        pytest.skip()

    class OneHotAutoEncoder(L.LightningModule):

        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(16, 4),
                nn.ReLU(),
                nn.Linear(4, 16),
            )

        def forward(self, x):
            x_hat = self.model(x)
            return nn.functional.mse_loss(x_hat, x)

    model = OneHotAutoEncoder()
    model_inputs = [torch.randn(2, 16)]

    show_model_graph(
        model,
        model_inputs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "lightning", "one-hot-autoencoder"),
    )
    assert validate_saved_activations(model, model_inputs)
'''