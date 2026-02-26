"""Tests for real-world models.

All optional-dependency imports are local (inside test functions) using
pytest.importorskip() for non-torchvision packages. Tests with missing
packages show as SKIPPED, never ERROR.

Only torch, torchvision, pytest, and torchlens are imported at the top level.
"""

from os.path import join as opj

import numpy as np
import pytest
import torch
import torchvision

import example_models
from torchlens import show_model_graph, validate_saved_activations


# =============================================================================
# TorchVision Classification Models
# =============================================================================


def test_alexnet(default_input1):
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
    model = torchvision.models.GoogLeNet()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_buffer_layers=True,
        vis_outpath=opj("visualization_outputs", "torchvision-main", "googlenet_showbuffer"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_buffer_layers=False,
        vis_outpath=opj("visualization_outputs", "torchvision-main", "googlenet_nobuffer"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_buffer_layers=False,
        vis_outpath=opj("visualization_outputs", "torchvision-main", "googlenet_nobuffer_rolled"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_buffer_layers=True,
        vis_nesting_depth=1,
        vis_outpath=opj("visualization_outputs", "torchvision-main", "googlenet_showbuffer_depth1"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_buffer_layers=False,
        vis_nesting_depth=1,
        vis_outpath=opj("visualization_outputs", "torchvision-main", "googlenet_nobuffer_depth1"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_buffer_layers=True,
        vis_nesting_depth=2,
        vis_outpath=opj("visualization_outputs", "torchvision-main", "googlenet_showbuffer_depth2"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_buffer_layers=False,
        vis_nesting_depth=2,
        vis_outpath=opj("visualization_outputs", "torchvision-main", "googlenet_nobuffer_depth2"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_buffer_layers=True,
        vis_nesting_depth=3,
        vis_outpath=opj("visualization_outputs", "torchvision-main", "googlenet_showbuffer_depth3"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_buffer_layers=False,
        vis_nesting_depth=3,
        vis_outpath=opj("visualization_outputs", "torchvision-main", "googlenet_nobuffer_depth3"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_buffer_layers=True,
        vis_nesting_depth=4,
        vis_outpath=opj("visualization_outputs", "torchvision-main", "googlenet_showbuffer_depth4"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_buffer_layers=False,
        vis_nesting_depth=4,
        vis_outpath=opj("visualization_outputs", "torchvision-main", "googlenet_nobuffer_depth4"),
    )


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
        vis_outpath=opj("visualization_outputs", "torchvision-main", "densenet121_depth1"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_nesting_depth=2,
        vis_outpath=opj("visualization_outputs", "torchvision-main", "densenet121_depth2"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_nesting_depth=3,
        vis_outpath=opj("visualization_outputs", "torchvision-main", "densenet121_depth3"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_nesting_depth=4,
        vis_outpath=opj("visualization_outputs", "torchvision-main", "densenet121_depth4"),
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
        vis_outpath=opj("visualization_outputs", "torchvision-main", "mobilenet_vg_large"),
    )


def test_wide_resnet(default_input1):
    model = torchvision.models.wide_resnet101_2()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "torchvision-main", "wide_resnet101_2"),
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
        vis_outpath=opj("visualization_outputs", "torchvision-main", "shufflenet_v2_x1_5"),
    )


def test_resnext(default_input1):
    model = torchvision.models.resnext101_64x4d()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "torchvision-main", "resnext101_64x4d"),
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


# =============================================================================
# CORNet Models (requires cornet package)
# =============================================================================


def test_cornet_s(default_input1):
    cornet = pytest.importorskip("cornet")
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
    cornet = pytest.importorskip("cornet")
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
    cornet = pytest.importorskip("cornet")
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
    cornet = pytest.importorskip("cornet")
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


# =============================================================================
# Torchvision Segmentation Models
# =============================================================================


def test_segment_deeplab_v3_resnet50(default_input1):
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


# =============================================================================
# Torchvision Detection Models
# =============================================================================


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


# =============================================================================
# Torchvision Quantized Models (one representative)
# =============================================================================


def test_quantize_resnet50(default_input1):
    model = torchvision.models.quantization.resnet50()
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "torchvision-quantize", "quantize_resnet50"),
    )
    assert validate_saved_activations(model, default_input1)


# =============================================================================
# Taskonomy (requires visualpriors)
# =============================================================================


def test_taskonomy(default_input1):
    visualpriors = pytest.importorskip("visualpriors")
    model = visualpriors.taskonomy_network.TaskonomyNetwork()
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "taskonomy", "taskonomy"),
    )
    assert validate_saved_activations(model, default_input1)


# =============================================================================
# Torchvision Video Models
# =============================================================================


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
        vis_outpath=opj("visualization_outputs", "torchvision-video", "video_mvit_v2_s"),
    )


def test_video_r2plus1_18():
    model = torchvision.models.video.r2plus1d_18()
    model_input = torch.randn(16, 3, 16, 112, 112)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "torchvision-video", "video_r2plus1d_18"),
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
    model_input = torch.randn(1, 3, 16, 224, 224)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "torchvision-video", "video_s3d"),
    )
    assert validate_saved_activations(model, model_input)


# =============================================================================
# Optical Flow
# =============================================================================


def test_opticflow_raftsmall():
    model = torchvision.models.optical_flow.raft_small()
    model_input = [torch.rand(6, 3, 224, 224), torch.rand(6, 3, 224, 224)]
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "torchvision-opticflow", "opticflow_raftsmall"),
    )
    assert validate_saved_activations(model, model_input)


def test_opticflow_raftlarge():
    # NOTE: the loop-finding procedure messes up on this somehow.
    model = torchvision.models.optical_flow.raft_large()
    model_input = [torch.rand(6, 3, 224, 224), torch.rand(6, 3, 224, 224)]
    show_model_graph(
        model,
        model_input,
        save_only=True,
        random_seed=1,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "torchvision-opticflow", "opticflow_raftlarge"),
    )
    assert validate_saved_activations(model, model_input, random_seed=1)


# =============================================================================
# TIMM Models (requires timm)
# =============================================================================


def test_timm_adv_inception_v3(default_input1):
    timm = pytest.importorskip("timm")
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
    timm = pytest.importorskip("timm")
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
    timm = pytest.importorskip("timm")
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
    timm = pytest.importorskip("timm")
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
    timm = pytest.importorskip("timm")
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
    timm = pytest.importorskip("timm")
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
    timm = pytest.importorskip("timm")
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
    timm = pytest.importorskip("timm")
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
    timm = pytest.importorskip("timm")
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
    timm = pytest.importorskip("timm")
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
    timm = pytest.importorskip("timm")
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
    timm = pytest.importorskip("timm")
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
    timm = pytest.importorskip("timm")
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
    timm = pytest.importorskip("timm")
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
    timm = pytest.importorskip("timm")
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
    timm = pytest.importorskip("timm")
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
    timm = pytest.importorskip("timm")
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


# =============================================================================
# Torchaudio Models (requires torchaudio)
# =============================================================================


def test_audio_conv_tasnet_base():
    torchaudio = pytest.importorskip("torchaudio")
    model = torchaudio.models.conv_tasnet_base()
    model_input = torch.randn(6, 1, 3)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "torchaudio", "audio_conv_tasnet_base"),
    )
    assert validate_saved_activations(model, model_input)


def test_audio_hubert_base():
    torchaudio = pytest.importorskip("torchaudio")
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


def test_audio_wav2vec2_base():
    torchaudio = pytest.importorskip("torchaudio")
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


def test_audio_wav2letter():
    torchaudio = pytest.importorskip("torchaudio")
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
    torchaudio = pytest.importorskip("torchaudio")
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


# =============================================================================
# Language Models
# =============================================================================


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
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
    model = transformers.GPT2Model.from_pretrained("gpt2")
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
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    model = transformers.BertForNextSentencePrediction.from_pretrained("bert-base-uncased")
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


# =============================================================================
# Multimodal / Special Models
# =============================================================================


def test_clip():
    transformers = pytest.importorskip("transformers")
    from PIL import Image

    model = transformers.CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = transformers.CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
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
        pytest.skip("UNet not available")

    model = UNet(3, 16, 10)
    model_inputs = (
        torch.rand(6, 3, 224, 224),
        torch.tensor([1]),
        torch.tensor([1.0]),
        torch.tensor([3.0]),
    )
    show_model_graph(
        model,
        model_inputs,
        random_seed=1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj("visualization_outputs", "multimodal-models", "stable_diffusion"),
    )
    assert validate_saved_activations(model, model_inputs, random_seed=1)


def test_styletts():
    try:
        from StyleTTS.models import TextEncoder
    except ModuleNotFoundError:
        pytest.skip("StyleTTS not available")

    model = TextEncoder(3, 3, 3, 100)
    model.eval()
    tokens = torch.tensor([[3, 0, 1, 2, 0, 2, 2, 3, 1, 4]])
    input_lengths = torch.ones(1, dtype=torch.long) * 10
    m = torch.ones(1, 10, dtype=torch.bool)
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


def test_dimenet():
    torch_geometric_nn = pytest.importorskip("torch_geometric.nn")
    DimeNet = torch_geometric_nn.DimeNet
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


def test_qml():
    qml = pytest.importorskip("pennylane")
    n_qubits = 2
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, diff_method="backprop")
    def qnode(inputs, weights):
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


def test_lightning():
    L = pytest.importorskip("lightning")
    import torch.nn as nn

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
