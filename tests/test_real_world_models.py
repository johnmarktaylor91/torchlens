"""Tests for real-world models.

All optional-dependency imports are local (inside test functions) using
pytest.importorskip() for non-torchvision packages. Tests with missing
packages show as SKIPPED, never ERROR.

Only torch, torchvision, pytest, and torchlens are imported at the top level.

Tests that take >5 minutes are marked @pytest.mark.slow. To skip them:
    pytest tests/test_real_world_models.py -m "not slow"
"""

from os.path import join as opj

import numpy as np
import pytest
import torch
import torchvision

from conftest import VIS_OUTPUT_DIR

import example_models
from torchlens import show_model_graph, validate_forward_pass


# =============================================================================
# TorchVision Classification Models
# =============================================================================


def test_alexnet(default_input1):
    model = torchvision.models.AlexNet()
    assert validate_forward_pass(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-main", "alexnet"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_nesting_depth=1,
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-main", "alexnet_depth1"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_nesting_depth=2,
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-main", "alexnet_depth2"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_nesting_depth=3,
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-main", "alexnet_depth3"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_nesting_depth=4,
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-main", "alexnet_depth4"),
    )


def test_vgg16(default_input1):
    model = torchvision.models.vgg16()
    assert validate_forward_pass(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-main", "vgg16"),
    )


@pytest.mark.slow
def test_vit(default_input1):
    model = torchvision.models.vit_l_16()
    assert validate_forward_pass(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-main", "vit_l_16"),
    )


@pytest.mark.slow
def test_googlenet(default_input1):
    model = torchvision.models.GoogLeNet()
    assert validate_forward_pass(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_buffer_layers=True,
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-main", "googlenet_showbuffer"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_buffer_layers=False,
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-main", "googlenet_nobuffer"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_buffer_layers=False,
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-main", "googlenet_nobuffer_rolled"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_buffer_layers=True,
        vis_nesting_depth=1,
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-main", "googlenet_showbuffer_depth1"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_buffer_layers=False,
        vis_nesting_depth=1,
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-main", "googlenet_nobuffer_depth1"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_buffer_layers=True,
        vis_nesting_depth=2,
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-main", "googlenet_showbuffer_depth2"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_buffer_layers=False,
        vis_nesting_depth=2,
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-main", "googlenet_nobuffer_depth2"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_buffer_layers=True,
        vis_nesting_depth=3,
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-main", "googlenet_showbuffer_depth3"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_buffer_layers=False,
        vis_nesting_depth=3,
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-main", "googlenet_nobuffer_depth3"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_buffer_layers=True,
        vis_nesting_depth=4,
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-main", "googlenet_showbuffer_depth4"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_buffer_layers=False,
        vis_nesting_depth=4,
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-main", "googlenet_nobuffer_depth4"),
    )


@pytest.mark.slow
def test_resnet50(default_input1):
    model = torchvision.models.resnet50()
    assert validate_forward_pass(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-main", "resnet50"),
    )


@pytest.mark.slow
def test_convnext_large(default_input1):
    model = torchvision.models.convnext_large()
    assert validate_forward_pass(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-main", "convnext_large"),
    )


@pytest.mark.slow
def test_densenet121(default_input1):
    model = torchvision.models.densenet121()
    assert validate_forward_pass(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-main", "densenet121"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_nesting_depth=1,
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-main", "densenet121_depth1"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_nesting_depth=2,
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-main", "densenet121_depth2"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_nesting_depth=3,
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-main", "densenet121_depth3"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_nesting_depth=4,
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-main", "densenet121_depth4"),
    )


@pytest.mark.slow
def test_efficientnet_b6(default_input1):
    model = torchvision.models.efficientnet_b6()
    assert validate_forward_pass(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-main", "efficientnet_b6"),
    )


@pytest.mark.slow
def test_squeezenet(default_input1):
    model = torchvision.models.squeezenet1_1()
    assert validate_forward_pass(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-main", "squeezenet"),
    )


@pytest.mark.slow
def test_mobilenet(default_input1):
    model = torchvision.models.mobilenet_v3_large()
    assert validate_forward_pass(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-main", "mobilenet_vg_large"),
    )


@pytest.mark.slow
def test_wide_resnet(default_input1):
    model = torchvision.models.wide_resnet101_2()
    assert validate_forward_pass(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-main", "wide_resnet101_2"),
    )


@pytest.mark.slow
def test_mnasnet(default_input1):
    model = torchvision.models.mnasnet1_3()
    assert validate_forward_pass(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-main", "mnasnet1_3"),
    )


@pytest.mark.slow
def test_shufflenet(default_input1):
    model = torchvision.models.shufflenet_v2_x1_5()
    assert validate_forward_pass(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-main", "shufflenet_v2_x1_5"),
    )


@pytest.mark.slow
def test_resnext(default_input1):
    model = torchvision.models.resnext101_64x4d()
    assert validate_forward_pass(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-main", "resnext101_64x4d"),
    )


@pytest.mark.slow
def test_regnet(default_input1):
    model = torchvision.models.regnet_x_32gf()
    assert validate_forward_pass(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-main", "regnet_x_32gf"),
    )


@pytest.mark.slow
def test_swin_v2b(default_input1):
    model = torchvision.models.swin_v2_b()
    assert validate_forward_pass(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-main", "swin_v2b"),
    )


@pytest.mark.slow
def test_maxvit(default_input1):
    model = torchvision.models.maxvit_t()
    assert validate_forward_pass(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-main", "max_vit_t"),
    )


@pytest.mark.slow
def test_inception_v3():
    model = torchvision.models.inception_v3()
    model_input = torch.randn(2, 3, 299, 299)
    assert validate_forward_pass(model, model_input)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-main", "inception_v3"),
    )


# =============================================================================
# CORNet Models (requires cornet package)
# =============================================================================


def test_cornet_z(default_input1):
    cornet = pytest.importorskip("cornet")
    model = cornet.cornet_z()
    assert validate_forward_pass(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "cornet", "cornet_z_unrolled"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "cornet", "cornet_z_rolled"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_nesting_depth=1,
        vis_outpath=opj(VIS_OUTPUT_DIR, "cornet", "cornet_z_unrolled_depth1"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_nesting_depth=1,
        vis_outpath=opj(VIS_OUTPUT_DIR, "cornet", "cornet_z_rolled_depth1"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_nesting_depth=2,
        vis_outpath=opj(VIS_OUTPUT_DIR, "cornet", "cornet_z_unrolled_depth2"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_nesting_depth=2,
        vis_outpath=opj(VIS_OUTPUT_DIR, "cornet", "cornet_z_rolled_depth2"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_nesting_depth=3,
        vis_outpath=opj(VIS_OUTPUT_DIR, "cornet", "cornet_z_unrolled_depth3"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_nesting_depth=3,
        vis_outpath=opj(VIS_OUTPUT_DIR, "cornet", "cornet_z_rolled_depth3"),
    )


@pytest.mark.slow
def test_cornet_s(default_input1):
    cornet = pytest.importorskip("cornet")
    model = cornet.cornet_s()
    assert validate_forward_pass(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "cornet", "cornet_s_unrolled"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "cornet", "cornet_s_rolled"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_nesting_depth=1,
        vis_outpath=opj(VIS_OUTPUT_DIR, "cornet", "cornet_s_unrolled_depth1"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_nesting_depth=1,
        vis_outpath=opj(VIS_OUTPUT_DIR, "cornet", "cornet_s_rolled_depth1"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_nesting_depth=2,
        vis_outpath=opj(VIS_OUTPUT_DIR, "cornet", "cornet_s_unrolled_depth2"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_nesting_depth=2,
        vis_outpath=opj(VIS_OUTPUT_DIR, "cornet", "cornet_s_rolled_depth2"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_nesting_depth=3,
        vis_outpath=opj(VIS_OUTPUT_DIR, "cornet", "cornet_s_unrolled_depth3"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_nesting_depth=3,
        vis_outpath=opj(VIS_OUTPUT_DIR, "cornet", "cornet_s_rolled_depth3"),
    )


@pytest.mark.slow
def test_cornet_r(default_input1):
    cornet = pytest.importorskip("cornet")
    model = cornet.cornet_r()
    assert validate_forward_pass(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "cornet", "cornet_r_unrolled"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "cornet", "cornet_r_rolled"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_nesting_depth=1,
        vis_outpath=opj(VIS_OUTPUT_DIR, "cornet", "cornet_r_unrolled_depth1"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_nesting_depth=1,
        vis_outpath=opj(VIS_OUTPUT_DIR, "cornet", "cornet_r_rolled_depth1"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_nesting_depth=2,
        vis_outpath=opj(VIS_OUTPUT_DIR, "cornet", "cornet_r_unrolled_depth2"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_nesting_depth=2,
        vis_outpath=opj(VIS_OUTPUT_DIR, "cornet", "cornet_r_rolled_depth2"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_nesting_depth=3,
        vis_outpath=opj(VIS_OUTPUT_DIR, "cornet", "cornet_r_unrolled_depth3"),
    )
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="rolled",
        vis_nesting_depth=3,
        vis_outpath=opj(VIS_OUTPUT_DIR, "cornet", "cornet_r_rolled_depth3"),
    )


@pytest.mark.slow
def test_cornet_rt():
    cornet = pytest.importorskip("cornet")
    model = cornet.cornet_rt()
    model_input = torch.rand((6, 3, 224, 224)).to("cuda")
    assert validate_forward_pass(model, model_input)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "cornet", "cornet_rt_unrolled"),
    )
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "cornet", "cornet_rt_rolled"),
    )
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_nesting_depth=1,
        vis_outpath=opj(VIS_OUTPUT_DIR, "cornet", "cornet_rt_unrolled_depth1"),
    )
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="rolled",
        vis_nesting_depth=1,
        vis_outpath=opj(VIS_OUTPUT_DIR, "cornet", "cornet_rt_rolled_depth1"),
    )
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_nesting_depth=2,
        vis_outpath=opj(VIS_OUTPUT_DIR, "cornet", "cornet_rt_unrolled_depth2"),
    )
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="rolled",
        vis_nesting_depth=2,
        vis_outpath=opj(VIS_OUTPUT_DIR, "cornet", "cornet_rt_rolled_depth2"),
    )
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_nesting_depth=3,
        vis_outpath=opj(VIS_OUTPUT_DIR, "cornet", "cornet_rt_unrolled_depth3"),
    )
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="rolled",
        vis_nesting_depth=3,
        vis_outpath=opj(VIS_OUTPUT_DIR, "cornet", "cornet_rt_rolled_depth3"),
    )


# =============================================================================
# TIMM Models (requires timm)
# =============================================================================


@pytest.mark.slow
def test_timm_beit_base_patch16_224(default_input1):
    timm = pytest.importorskip("timm")
    model = timm.models.beit_base_patch16_224()
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "timm", "timm_beit_base_patch16_224"),
    )
    assert validate_forward_pass(model, default_input1)


def test_timm_gluon_resnext101_32x4d():
    timm = pytest.importorskip("timm")
    model = timm.create_model("gluon_resnext101_32x4d", pretrained=False)
    model_input = torch.randn(1, 3, 224, 224)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "timm", "gluon_resnext101_32x4d"),
    )
    assert validate_forward_pass(model, model_input)


def test_timm_ecaresnet101d():
    timm = pytest.importorskip("timm")
    model = timm.create_model("ecaresnet101d", pretrained=False)
    model_input = torch.randn(1, 3, 224, 224)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "timm", "ecaresnet101d"),
    )
    assert validate_forward_pass(model, model_input)


def test_mobilevit_xxs():
    timm = pytest.importorskip("timm")
    model = timm.create_model("mobilevitv2_050", pretrained=False)
    model_input = torch.randn(1, 3, 224, 224)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "timm", "mobilevitv2_050"),
    )
    assert validate_forward_pass(model, model_input)


@pytest.mark.slow
def test_timm_adv_inception_v3(default_input1):
    timm = pytest.importorskip("timm")
    model = timm.models.adv_inception_v3(pretrained=True)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "timm", "timm_adv_inception_v3"),
    )
    assert validate_forward_pass(model, default_input1)


@pytest.mark.slow
def test_timm_cait_s24_224(default_input1):
    timm = pytest.importorskip("timm")
    model = timm.models.cait_s24_224()
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "timm", "timm_cait_s24_224"),
    )
    assert validate_forward_pass(model, default_input1)


@pytest.mark.slow
def test_timm_coat_mini(default_input1):
    timm = pytest.importorskip("timm")
    model = timm.models.coat_mini()
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "timm", "timm_coat_mini"),
    )
    assert validate_forward_pass(model, default_input1)


@pytest.mark.slow
def test_timm_convit_base(default_input1):
    timm = pytest.importorskip("timm")
    model = timm.create_model("convit_base", pretrained=True)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "timm", "timm_convit_base"),
    )
    assert validate_forward_pass(model, default_input1)


@pytest.mark.slow
def test_timm_darknet21():
    timm = pytest.importorskip("timm")
    model = timm.create_model("darknet21", pretrained=False)
    model_input = torch.randn(1, 3, 224, 224)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "timm", "timm_darknet21"),
    )
    assert validate_forward_pass(model, model_input)


@pytest.mark.slow
def test_timm_ghostnet_100():
    timm = pytest.importorskip("timm")
    model = timm.create_model("ghostnet_100", pretrained=True)
    model_input = torch.randn(1, 3, 224, 224)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "timm", "timm_ghostnet_100"),
    )
    assert validate_forward_pass(model, model_input)


@pytest.mark.slow
def test_timm_mixnet_m():
    timm = pytest.importorskip("timm")
    model = timm.create_model("mixnet_m", pretrained=False)
    model_input = torch.randn(1, 3, 224, 224)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "timm", "timm_mixnet_m"),
    )
    assert validate_forward_pass(model, model_input)


@pytest.mark.slow
def test_timm_poolformer_s24():
    timm = pytest.importorskip("timm")
    model = timm.create_model("poolformer_s24", pretrained=False)
    model_input = torch.randn(1, 3, 224, 224)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "timm", "timm_poolformer_s24"),
    )
    assert validate_forward_pass(model, model_input)


@pytest.mark.slow
def test_timm_resnest14d():
    timm = pytest.importorskip("timm")
    model = timm.create_model("resnest14d", pretrained=False)
    model_input = torch.randn(6, 3, 224, 224)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "timm", "timm_resnest14d"),
    )
    assert validate_forward_pass(model, model_input)


@pytest.mark.slow
def test_timm_edgenext_small():
    timm = pytest.importorskip("timm")
    model = timm.create_model("edgenext_small", pretrained=False)
    model_input = torch.randn(1, 3, 224, 224)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "timm", "timm_edgenext_small"),
    )
    assert validate_forward_pass(model, model_input)


@pytest.mark.slow
def test_timm_hardcorenas_f():
    timm = pytest.importorskip("timm")
    model = timm.create_model("hardcorenas_f", pretrained=False)
    model_input = torch.randn(1, 3, 224, 224)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "timm", "hardcorenas_f"),
    )
    assert validate_forward_pass(model, model_input)


@pytest.mark.slow
def test_timm_semnasnet_100():
    timm = pytest.importorskip("timm")
    model = timm.create_model("semnasnet_100", pretrained=False)
    model_input = torch.randn(1, 3, 224, 224)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "timm", "semnasnet_100"),
    )
    assert validate_forward_pass(model, model_input)


@pytest.mark.slow
def test_timm_xcit_tiny_24_p8_224():
    timm = pytest.importorskip("timm")
    model = timm.create_model("xcit_tiny_24_p8_224", pretrained=False)
    model_input = torch.randn(1, 3, 224, 224)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "timm", "xcit_tiny_24_p8_224"),
    )
    assert validate_forward_pass(model, model_input)


@pytest.mark.slow
def test_timm_seresnet152():
    timm = pytest.importorskip("timm")
    model = timm.create_model("seresnet152", pretrained=False)
    model_input = torch.randn(1, 3, 224, 224)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "timm", "seresnet152"),
    )
    assert validate_forward_pass(model, model_input)


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
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchaudio", "audio_conv_tasnet_base"),
    )
    assert validate_forward_pass(model, model_input)


def test_audio_wav2letter():
    torchaudio = pytest.importorskip("torchaudio")
    model = torchaudio.models.Wav2Letter()
    model_input = torch.randn(1, 1, 2000)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchaudio", "audio_wav2letter"),
    )
    assert validate_forward_pass(model, model_input)


@pytest.mark.slow
def test_audio_hubert_base():
    torchaudio = pytest.importorskip("torchaudio")
    model = torchaudio.models.hubert_base()
    model_input = torch.randn(12, 2000)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchaudio", "audio_hubert_base"),
        random_seed=1,
    )
    assert validate_forward_pass(model, model_input, random_seed=1)


@pytest.mark.slow
def test_audio_wav2vec2_base():
    torchaudio = pytest.importorskip("torchaudio")
    model = torchaudio.models.wav2vec2_base()
    model_input = torch.randn(12, 2000)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchaudio", "audio_wave2vec2_base"),
    )
    assert validate_forward_pass(model, model_input)


@pytest.mark.slow
def test_deepspeech():
    torchaudio = pytest.importorskip("torchaudio")
    model = torchaudio.models.DeepSpeech(3)
    model_input = torch.randn(12, 2000, 3)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchaudio", "audio_deepspeech"),
    )
    assert validate_forward_pass(model, model_input)


@pytest.mark.slow
def test_conformer():
    torchaudio = pytest.importorskip("torchaudio")
    model = torchaudio.models.Conformer(
        input_dim=80,
        num_heads=4,
        ffn_dim=128,
        num_layers=2,
        depthwise_conv_kernel_size=31,
    )
    lengths = torch.tensor([50, 40])
    model_input = torch.randn(2, 50, 80)
    model_inputs = (model_input, lengths)
    show_model_graph(
        model,
        model_inputs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchaudio", "conformer"),
    )
    assert validate_forward_pass(model, model_inputs)


@pytest.mark.slow
def test_whisper_tiny():
    transformers = pytest.importorskip("transformers")
    model = transformers.WhisperModel.from_pretrained("openai/whisper-tiny")
    input_features = torch.randn(1, 80, 3000)
    decoder_input_ids = torch.tensor([[50258, 50259, 50359]])
    model_kwargs = {"input_features": input_features, "decoder_input_ids": decoder_input_ids}
    show_model_graph(
        model,
        [],
        model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchaudio", "whisper_tiny"),
    )
    assert validate_forward_pass(model, [], model_kwargs)


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
        vis_outpath=opj(VIS_OUTPUT_DIR, "language-models", "language_lstm_unrolled"),
    )
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "language-models", "language_lstm_rolled"),
    )
    assert validate_forward_pass(model, model_input)


def test_rnn():
    model = example_models.RNNModel()
    model_input = torch.rand(5, 5, 5)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "language-models", "language_rnn_unrolled"),
    )
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "language-models", "language_rnn_rolled"),
    )
    assert validate_forward_pass(model, model_input)


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
        vis_outpath=opj(VIS_OUTPUT_DIR, "language-models", "gpt2"),
    )
    assert validate_forward_pass(model, [], model_inputs)


def test_distilbert():
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = transformers.DistilBertModel.from_pretrained("distilbert-base-uncased")
    model_inputs = tokenizer("to be or not to be", return_tensors="pt")
    show_model_graph(
        model,
        [],
        model_inputs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "nlp-models", "distilbert"),
    )
    assert validate_forward_pass(model, [], model_inputs)


def test_electra_small():
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained("google/electra-small-discriminator")
    model = transformers.ElectraModel.from_pretrained("google/electra-small-discriminator")
    model_inputs = tokenizer("to be or not to be", return_tensors="pt")
    show_model_graph(
        model,
        [],
        model_inputs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "nlp-models", "electra_small"),
    )
    assert validate_forward_pass(model, [], model_inputs)


@pytest.mark.slow
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
        vis_outpath=opj(VIS_OUTPUT_DIR, "language-models", "bert"),
    )
    assert validate_forward_pass(model, [], model_inputs)


def test_t5_small():
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained("t5-small")
    model = transformers.T5ForConditionalGeneration.from_pretrained("t5-small")
    input_ids = tokenizer("translate English to German: Hello", return_tensors="pt").input_ids
    decoder_input_ids = tokenizer("Hallo", return_tensors="pt").input_ids
    model_kwargs = {"input_ids": input_ids, "decoder_input_ids": decoder_input_ids}
    show_model_graph(
        model,
        [],
        model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "nlp-models", "t5_small"),
    )
    assert validate_forward_pass(model, [], model_kwargs)


@pytest.mark.slow
def test_bart_base():
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/bart-base")
    model = transformers.BartModel.from_pretrained("facebook/bart-base")
    inputs = tokenizer("Hello, my name is Claude", return_tensors="pt")
    show_model_graph(
        model,
        [],
        inputs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "nlp-models", "bart_base"),
    )
    assert validate_forward_pass(model, [], inputs)


@pytest.mark.slow
def test_roberta_base():
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained("roberta-base")
    model = transformers.RobertaModel.from_pretrained("roberta-base")
    inputs = tokenizer("Hello world", return_tensors="pt")
    show_model_graph(
        model,
        [],
        inputs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "nlp-models", "roberta_base"),
    )
    assert validate_forward_pass(model, [], inputs)


@pytest.mark.slow
def test_sentence_transformer():
    sentence_transformers = pytest.importorskip("sentence_transformers")
    model = sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")
    transformer_model = model[0].auto_model
    tokenizer = model.tokenizer
    inputs = tokenizer("Hello world", return_tensors="pt", padding=True, truncation=True)
    show_model_graph(
        transformer_model,
        [],
        inputs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "nlp-models", "sentence_transformer"),
    )
    assert validate_forward_pass(transformer_model, [], inputs)


# =============================================================================
# Multimodal / Vision-Language / Special Models
# =============================================================================


def test_stable_diffusion():
    model = example_models.ContextUnet(3, 16, 10)
    model_inputs = (
        torch.rand(1, 3, 28, 28),
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
        vis_outpath=opj(VIS_OUTPUT_DIR, "multimodal-models", "stable_diffusion"),
    )
    assert validate_forward_pass(model, model_inputs, random_seed=1)


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
        vis_outpath=opj(VIS_OUTPUT_DIR, "text-to-speech", "styletts_text_encoder"),
    )
    assert validate_forward_pass(model, model_inputs, random_seed=1)


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
        vis_outpath=opj(VIS_OUTPUT_DIR, "quantum", "qml"),
    )
    assert validate_forward_pass(model, model_inputs)


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
        vis_outpath=opj(VIS_OUTPUT_DIR, "lightning", "one-hot-autoencoder"),
    )
    assert validate_forward_pass(model, model_inputs)


@pytest.mark.slow
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
        vis_outpath=opj(VIS_OUTPUT_DIR, "multimodal-models", "clip"),
    )
    assert validate_forward_pass(model, [], model_inputs, random_seed=1)


@pytest.mark.slow
def test_blip_base():
    transformers = pytest.importorskip("transformers")
    from PIL import Image

    model = transformers.BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
    processor = transformers.BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    image = Image.fromarray(np.random.random((224, 224, 3)).astype(np.uint8))
    model_inputs = processor(images=image, text="a photo", return_tensors="pt")
    show_model_graph(
        model,
        [],
        model_inputs,
        random_seed=1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "multimodal-models", "blip_base"),
    )
    assert validate_forward_pass(model, [], model_inputs, random_seed=1)


@pytest.mark.slow
def test_vit_mae():
    transformers = pytest.importorskip("transformers")
    model = transformers.ViTMAEModel.from_pretrained("facebook/vit-mae-base")
    pixel_values = torch.randn(1, 3, 224, 224)
    model_kwargs = {"pixel_values": pixel_values}
    show_model_graph(
        model,
        [],
        model_kwargs,
        random_seed=1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-main", "vit_mae"),
    )
    assert validate_forward_pass(model, [], model_kwargs, random_seed=1)


# =============================================================================
# Built-in Architecture Tests (no external deps)
# =============================================================================


def test_simple_moe():
    """Mixture of Experts with top-2 routing — built-in, no external deps."""
    import torch.nn as nn

    class SimpleMoE(nn.Module):
        def __init__(self, input_dim=16, hidden_dim=32, num_experts=4, top_k=2):
            super().__init__()
            self.experts = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, input_dim),
                    )
                    for _ in range(num_experts)
                ]
            )
            self.gate = nn.Linear(input_dim, num_experts)
            self.top_k = top_k

        def forward(self, x):
            gate_logits = self.gate(x)
            weights, indices = torch.topk(torch.softmax(gate_logits, dim=-1), self.top_k, dim=-1)
            weights = weights / weights.sum(dim=-1, keepdim=True)
            output = torch.zeros_like(x)
            for i in range(self.top_k):
                expert_idx = indices[:, :, i] if x.dim() == 3 else indices[:, i]
                for j, expert in enumerate(self.experts):
                    mask = expert_idx == j
                    if mask.any():
                        if x.dim() == 3:
                            expert_input = x[mask]
                        else:
                            expert_input = x[mask]
                        expert_output = expert(expert_input)
                        w = weights[:, :, i][mask] if x.dim() == 3 else weights[:, i][mask]
                        output[mask] = output[mask] + w.unsqueeze(-1) * expert_output
            return output

    model = SimpleMoE()
    x = torch.rand(2, 16)
    show_model_graph(
        model,
        x,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "simple_moe"),
    )
    assert validate_forward_pass(model, x)


# =============================================================================
# State Space Models (requires transformers)
# =============================================================================


def test_mamba():
    """Mamba SSM via HuggingFace transformers (small config, no pretrained)."""
    transformers = pytest.importorskip("transformers")
    config = transformers.MambaConfig(
        vocab_size=100,
        hidden_size=32,
        state_size=8,
        num_hidden_layers=2,
        intermediate_size=64,
    )
    model = transformers.MambaModel(config)
    x = torch.randint(0, 100, (1, 16))
    model_kwargs = {"input_ids": x}
    show_model_graph(
        model,
        [],
        model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "state-space-models", "mamba"),
    )
    assert validate_forward_pass(model, [], model_kwargs)


def test_mamba2():
    """Mamba-2 SSM via HuggingFace transformers (small config, no pretrained)."""
    transformers = pytest.importorskip("transformers")
    config = transformers.Mamba2Config(
        vocab_size=100,
        hidden_size=64,
        state_size=8,
        num_hidden_layers=2,
        head_dim=16,
        num_heads=8,
    )
    model = transformers.Mamba2Model(config)
    x = torch.randint(0, 100, (1, 16))
    model_kwargs = {"input_ids": x}
    show_model_graph(
        model,
        [],
        model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "state-space-models", "mamba2"),
    )
    assert validate_forward_pass(model, [], model_kwargs)


def test_rwkv():
    """RWKV linear-attention model via HuggingFace transformers (small config)."""
    transformers = pytest.importorskip("transformers")
    config = transformers.RwkvConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=2,
        attention_hidden_size=32,
        intermediate_size=64,
    )
    model = transformers.RwkvModel(config)
    x = torch.randint(0, 100, (1, 16))
    model_kwargs = {"input_ids": x}
    show_model_graph(
        model,
        [],
        model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "state-space-models", "rwkv"),
    )
    assert validate_forward_pass(model, [], model_kwargs)


def test_falcon_mamba():
    """Falcon-Mamba hybrid SSM via HuggingFace transformers (small config)."""
    transformers = pytest.importorskip("transformers")
    config = transformers.FalconMambaConfig(
        vocab_size=100,
        hidden_size=32,
        state_size=8,
        num_hidden_layers=2,
        intermediate_size=64,
    )
    model = transformers.FalconMambaModel(config)
    x = torch.randint(0, 100, (1, 16))
    model_kwargs = {"input_ids": x}
    show_model_graph(
        model,
        [],
        model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "state-space-models", "falcon_mamba"),
    )
    assert validate_forward_pass(model, [], model_kwargs)


# =============================================================================
# Autoencoders (real-world, via transformers)
# =============================================================================


def test_autoencoder_vit_mae():
    """ViT-MAE as masked autoencoder (small config, no pretrained weights)."""
    transformers = pytest.importorskip("transformers")
    config = transformers.ViTMAEConfig(
        image_size=32,
        patch_size=4,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        decoder_hidden_size=32,
        decoder_num_hidden_layers=1,
        decoder_num_attention_heads=4,
        decoder_intermediate_size=64,
    )
    model = transformers.ViTMAEForPreTraining(config)
    pixel_values = torch.randn(1, 3, 32, 32)
    model_kwargs = {"pixel_values": pixel_values}
    show_model_graph(
        model,
        [],
        model_kwargs,
        random_seed=1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "autoencoders", "vit_mae_pretrain"),
    )
    assert validate_forward_pass(model, [], model_kwargs, random_seed=1)


# =============================================================================
# TorchVision Segmentation Models
# =============================================================================


@pytest.mark.slow
def test_segment_deeplab_v3_resnet50(default_input1):
    model = torchvision.models.segmentation.deeplabv3_resnet50()
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            VIS_OUTPUT_DIR,
            "torchvision-segmentation",
            "segment_deeplabv3_resnet50",
        ),
    )
    assert validate_forward_pass(model, default_input1)


@pytest.mark.slow
def test_segment_deeplabv3_mobilenet(default_input1):
    model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large()
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            VIS_OUTPUT_DIR,
            "torchvision-segmentation",
            "segment_deeplabv3_mobilenet_v3_large",
        ),
    )
    assert validate_forward_pass(model, default_input1)


@pytest.mark.slow
def test_segment_lraspp_mobilenet(default_input1):
    model = torchvision.models.segmentation.lraspp_mobilenet_v3_large()
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(
            VIS_OUTPUT_DIR,
            "torchvision-segmentation",
            "segment_lraspp_mobilenet_v3_large",
        ),
    )
    assert validate_forward_pass(model, default_input1)


@pytest.mark.slow
def test_segment_fcn_resnet50(default_input1):
    model = torchvision.models.segmentation.fcn_resnet50()
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-segmentation", "segment_fcn_resnet50"),
    )
    assert validate_forward_pass(model, default_input1)


# =============================================================================
# TorchVision Detection Models
# =============================================================================


@pytest.mark.slow
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
            VIS_OUTPUT_DIR,
            "torchvision-detection",
            "detect_fasterrcnn_mobilenet_v3_large_320_fpn_train",
        ),
    )
    assert validate_forward_pass(model, model_inputs)


@pytest.mark.slow
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
            VIS_OUTPUT_DIR,
            "torchvision-detection",
            "detect_fasterrcnn_mobilenet_v3_large_320_fpn_eval",
        ),
    )
    assert validate_forward_pass(model, [input_tensors])


@pytest.mark.slow
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
            VIS_OUTPUT_DIR,
            "torchvision-detection",
            "detect_fcos_resnet50_fpn_train",
        ),
        random_seed=1,
    )
    assert validate_forward_pass(model, model_inputs, random_seed=1)


@pytest.mark.slow
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
            VIS_OUTPUT_DIR,
            "torchvision-detection",
            "detect_fcos_resnet50_fpn_eval",
        ),
    )
    assert validate_forward_pass(model, [input_tensors])


@pytest.mark.slow
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
            VIS_OUTPUT_DIR,
            "torchvision-detection",
            "detect_retinanet_resnet50_fpn_train",
        ),
    )
    assert validate_forward_pass(model, model_inputs)


@pytest.mark.slow
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
            VIS_OUTPUT_DIR,
            "torchvision-detection",
            "detect_retinanet_resnet50_fpn_eval",
        ),
    )
    assert validate_forward_pass(model, [input_tensors])


@pytest.mark.slow
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
            VIS_OUTPUT_DIR,
            "torchvision-detection",
            "detect_ssd300_vgg16_train",
        ),
    )
    assert validate_forward_pass(model, model_inputs)


@pytest.mark.slow
def test_ssd300_vgg16_eval(default_input1, default_input2):
    model = torchvision.models.detection.ssd300_vgg16()
    input_tensors = [default_input1[0], default_input2[0]]
    model = model.eval()
    show_model_graph(
        model,
        [input_tensors],
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-detection", "detect_ssd300_vgg16_eval"),
    )
    assert validate_forward_pass(model, [input_tensors])


# =============================================================================
# TorchVision Quantized Models
# =============================================================================


@pytest.mark.slow
def test_quantize_resnet50(default_input1):
    model = torchvision.models.quantization.resnet50()
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-quantize", "quantize_resnet50"),
    )
    assert validate_forward_pass(model, default_input1)


# =============================================================================
# TorchVision Video Models
# =============================================================================


@pytest.mark.slow
def test_video_r2plus1_18():
    model = torchvision.models.video.r2plus1d_18()
    model_input = torch.randn(1, 3, 1, 112, 112)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-video", "video_r2plus1d_18"),
    )
    assert validate_forward_pass(model, model_input)


@pytest.mark.slow
def test_video_mc3_18():
    model = torchvision.models.video.mc3_18()
    model_input = torch.randn(6, 3, 16, 112, 112)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-video", "video_mc3_18"),
    )
    assert validate_forward_pass(model, model_input)


@pytest.mark.slow
def test_video_mvit_v2_s():
    model = torchvision.models.video.mvit_v2_s()
    model_input = torch.randn(16, 3, 448, 896)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-video", "video_mvit_v2_s"),
    )


@pytest.mark.slow
def test_video_r3d_18():
    model = torchvision.models.video.r3d_18()
    model_input = torch.randn(16, 3, 16, 112, 112)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-video", "video_r3d_18"),
    )
    assert validate_forward_pass(model, model_input)


@pytest.mark.slow
def test_video_s3d():
    model = torchvision.models.video.s3d()
    model_input = torch.randn(1, 3, 16, 224, 224)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-video", "video_s3d"),
    )
    assert validate_forward_pass(model, model_input)


# =============================================================================
# Optical Flow
# =============================================================================


@pytest.mark.slow
def test_opticflow_raftsmall():
    model = torchvision.models.optical_flow.raft_small()
    model_input = [torch.rand(6, 3, 224, 224), torch.rand(6, 3, 224, 224)]
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-opticflow", "opticflow_raftsmall"),
    )
    assert validate_forward_pass(model, model_input)


@pytest.mark.slow
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
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-opticflow", "opticflow_raftlarge"),
    )
    assert validate_forward_pass(model, model_input, random_seed=1)


# =============================================================================
# Taskonomy (requires visualpriors)
# =============================================================================


@pytest.mark.slow
def test_taskonomy(default_input1):
    visualpriors = pytest.importorskip("visualpriors")
    model = visualpriors.taskonomy_network.TaskonomyNetwork()
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "taskonomy", "taskonomy"),
    )
    assert validate_forward_pass(model, default_input1)


# =============================================================================
# Graph Neural Networks (requires torch_geometric)
# =============================================================================


@pytest.mark.slow
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
        vis_outpath=opj(VIS_OUTPUT_DIR, "graph-neural-networks", "dimenet"),
    )
    assert validate_forward_pass(model, model_inputs)


# =============================================================================
# Decoder-Only LLMs (requires transformers, small random-init configs)
# =============================================================================


@pytest.mark.slow
def test_llama():
    """LLaMA: RoPE, SwiGLU, GQA (small config, no pretrained)."""
    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        vocab_size=100,
        max_position_embeddings=32,
    )
    model = transformers.LlamaForCausalLM(config).eval()
    model_kwargs = {"input_ids": torch.randint(0, 100, (1, 16))}
    show_model_graph(
        model,
        [],
        model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "decoder-only-llms", "llama"),
    )
    assert validate_forward_pass(model, [], model_kwargs)


@pytest.mark.slow
def test_mistral():
    """Mistral: sliding window attention, GQA (small config)."""
    transformers = pytest.importorskip("transformers")
    config = transformers.MistralConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        vocab_size=100,
        max_position_embeddings=32,
        sliding_window=16,
    )
    model = transformers.MistralForCausalLM(config).eval()
    model_kwargs = {"input_ids": torch.randint(0, 100, (1, 16))}
    show_model_graph(
        model,
        [],
        model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "decoder-only-llms", "mistral"),
    )
    assert validate_forward_pass(model, [], model_kwargs)


@pytest.mark.slow
def test_phi():
    """Phi: small efficient LLM (small config)."""
    transformers = pytest.importorskip("transformers")
    config = transformers.PhiConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        vocab_size=100,
        max_position_embeddings=32,
    )
    model = transformers.PhiForCausalLM(config).eval()
    model_kwargs = {"input_ids": torch.randint(0, 100, (1, 16))}
    show_model_graph(
        model,
        [],
        model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "decoder-only-llms", "phi"),
    )
    assert validate_forward_pass(model, [], model_kwargs)


@pytest.mark.slow
def test_gemma():
    """Gemma: Google's open model (small config)."""
    transformers = pytest.importorskip("transformers")
    config = transformers.GemmaConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        vocab_size=100,
        max_position_embeddings=32,
        head_dim=16,
    )
    model = transformers.GemmaForCausalLM(config).eval()
    model_kwargs = {"input_ids": torch.randint(0, 100, (1, 16))}
    show_model_graph(
        model,
        [],
        model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "decoder-only-llms", "gemma"),
    )
    assert validate_forward_pass(model, [], model_kwargs)


@pytest.mark.slow
def test_qwen2():
    """Qwen2: Alibaba's model with GQA (small config)."""
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen2Config(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        vocab_size=100,
        max_position_embeddings=32,
    )
    model = transformers.Qwen2ForCausalLM(config).eval()
    model_kwargs = {"input_ids": torch.randint(0, 100, (1, 16))}
    show_model_graph(
        model,
        [],
        model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "decoder-only-llms", "qwen2"),
    )
    assert validate_forward_pass(model, [], model_kwargs)


@pytest.mark.slow
def test_falcon_llm():
    """Falcon: multi-query attention (small config)."""
    transformers = pytest.importorskip("transformers")
    config = transformers.FalconConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        vocab_size=100,
        new_decoder_architecture=False,
    )
    model = transformers.FalconForCausalLM(config).eval()
    model_kwargs = {"input_ids": torch.randint(0, 100, (1, 16))}
    show_model_graph(
        model,
        [],
        model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "decoder-only-llms", "falcon"),
    )
    assert validate_forward_pass(model, [], model_kwargs)


@pytest.mark.slow
def test_bloom():
    """BLOOM: ALiBi positional encoding (small config)."""
    transformers = pytest.importorskip("transformers")
    config = transformers.BloomConfig(
        hidden_size=64,
        n_layer=2,
        n_head=4,
        vocab_size=100,
    )
    model = transformers.BloomForCausalLM(config).eval()
    model_kwargs = {"input_ids": torch.randint(0, 100, (1, 16))}
    show_model_graph(
        model,
        [],
        model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "decoder-only-llms", "bloom"),
    )
    assert validate_forward_pass(model, [], model_kwargs)


@pytest.mark.slow
def test_opt():
    """OPT: Meta's open pre-trained transformer (small config)."""
    transformers = pytest.importorskip("transformers")
    config = transformers.OPTConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        ffn_dim=128,
        vocab_size=100,
        max_position_embeddings=32,
    )
    model = transformers.OPTForCausalLM(config).eval()
    model_kwargs = {"input_ids": torch.randint(0, 100, (1, 16))}
    show_model_graph(
        model,
        [],
        model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "decoder-only-llms", "opt"),
    )
    assert validate_forward_pass(model, [], model_kwargs)


# =============================================================================
# Encoder-Only Models (additional, requires transformers)
# =============================================================================


@pytest.mark.slow
def test_albert():
    """ALBERT: cross-layer weight sharing, factorized embeddings (small config)."""
    transformers = pytest.importorskip("transformers")
    config = transformers.AlbertConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        embedding_size=32,
        vocab_size=100,
    )
    model = transformers.AlbertModel(config).eval()
    model_kwargs = {"input_ids": torch.randint(0, 100, (1, 16))}
    show_model_graph(
        model,
        [],
        model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "encoder-only", "albert"),
    )
    assert validate_forward_pass(model, [], model_kwargs)


@pytest.mark.slow
def test_deberta():
    """DeBERTa v2: disentangled attention (small config)."""
    transformers = pytest.importorskip("transformers")
    config = transformers.DebertaV2Config(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        vocab_size=100,
        max_position_embeddings=32,
    )
    model = transformers.DebertaV2Model(config).eval()
    model_kwargs = {"input_ids": torch.randint(0, 100, (1, 16))}
    show_model_graph(
        model,
        [],
        model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "encoder-only", "deberta"),
    )
    assert validate_forward_pass(model, [], model_kwargs)


@pytest.mark.slow
def test_xlm_roberta():
    """XLM-RoBERTa: cross-lingual encoder (small config)."""
    transformers = pytest.importorskip("transformers")
    config = transformers.XLMRobertaConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        vocab_size=100,
        max_position_embeddings=32,
    )
    model = transformers.XLMRobertaModel(config).eval()
    model_kwargs = {"input_ids": torch.randint(0, 100, (1, 16))}
    show_model_graph(
        model,
        [],
        model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "encoder-only", "xlm_roberta"),
    )
    assert validate_forward_pass(model, [], model_kwargs)


# =============================================================================
# Encoder-Decoder Models (additional, requires transformers)
# =============================================================================


@pytest.mark.slow
def test_pegasus():
    """Pegasus: abstractive summarization encoder-decoder (small config)."""
    transformers = pytest.importorskip("transformers")
    config = transformers.PegasusConfig(
        d_model=64,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=128,
        decoder_ffn_dim=128,
        vocab_size=100,
        max_position_embeddings=32,
    )
    model = transformers.PegasusModel(config).eval()
    model_kwargs = {
        "input_ids": torch.randint(0, 100, (1, 16)),
        "decoder_input_ids": torch.randint(0, 100, (1, 8)),
    }
    show_model_graph(
        model,
        [],
        model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "encoder-decoder", "pegasus"),
    )
    assert validate_forward_pass(model, [], model_kwargs)


@pytest.mark.slow
def test_led():
    """LED: Longformer Encoder-Decoder for long documents (small config)."""
    transformers = pytest.importorskip("transformers")
    config = transformers.LEDConfig(
        d_model=64,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=128,
        decoder_ffn_dim=128,
        vocab_size=100,
        max_encoder_position_embeddings=32,
        max_decoder_position_embeddings=32,
        attention_window=[8, 8],
    )
    model = transformers.LEDModel(config).eval()
    model_kwargs = {
        "input_ids": torch.randint(0, 100, (1, 16)),
        "decoder_input_ids": torch.randint(0, 100, (1, 8)),
    }
    show_model_graph(
        model,
        [],
        model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "encoder-decoder", "led"),
    )
    assert validate_forward_pass(model, [], model_kwargs)


# =============================================================================
# Efficient / Long-Range Transformers (requires transformers)
# =============================================================================


@pytest.mark.slow
def test_fnet():
    """FNet: Fourier transform replaces attention (small config)."""
    transformers = pytest.importorskip("transformers")
    config = transformers.FNetConfig(
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        vocab_size=100,
        max_position_embeddings=32,
    )
    model = transformers.FNetModel(config).eval()
    model_kwargs = {"input_ids": torch.randint(0, 100, (1, 16))}
    torch.use_deterministic_algorithms(False)
    try:
        show_model_graph(
            model,
            [],
            model_kwargs,
            save_only=True,
            vis_opt="unrolled",
            vis_outpath=opj(VIS_OUTPUT_DIR, "efficient-transformers", "fnet"),
        )
        assert validate_forward_pass(model, [], model_kwargs)
    finally:
        torch.use_deterministic_algorithms(True)


@pytest.mark.slow
def test_nystromformer():
    """Nystromformer: Nystrom-based attention approximation (small config)."""
    transformers = pytest.importorskip("transformers")
    config = transformers.NystromformerConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        vocab_size=100,
        max_position_embeddings=64,
        num_landmarks=8,
    )
    model = transformers.NystromformerModel(config).eval()
    model_kwargs = {"input_ids": torch.randint(0, 100, (1, 64))}
    show_model_graph(
        model,
        [],
        model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "efficient-transformers", "nystromformer"),
    )
    assert validate_forward_pass(model, [], model_kwargs)


@pytest.mark.slow
def test_bigbird():
    """BigBird: block sparse + random + global attention (small config)."""
    transformers = pytest.importorskip("transformers")
    config = transformers.BigBirdConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        vocab_size=100,
        max_position_embeddings=64,
        attention_type="original_full",
    )
    model = transformers.BigBirdModel(config).eval()
    model_kwargs = {"input_ids": torch.randint(0, 100, (1, 32))}
    show_model_graph(
        model,
        [],
        model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "efficient-transformers", "bigbird"),
    )
    assert validate_forward_pass(model, [], model_kwargs)


# =============================================================================
# Mixture of Experts (requires transformers)
# =============================================================================


@pytest.mark.slow
def test_mixtral():
    """Mixtral: Mistral + sparse MoE, top-2 routing (small config)."""
    transformers = pytest.importorskip("transformers")
    config = transformers.MixtralConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        vocab_size=100,
        max_position_embeddings=32,
        num_local_experts=4,
        num_experts_per_tok=2,
    )
    model = transformers.MixtralForCausalLM(config).eval()
    model_kwargs = {"input_ids": torch.randint(0, 100, (1, 16))}
    show_model_graph(
        model,
        [],
        model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "moe-models", "mixtral"),
    )
    assert validate_forward_pass(model, [], model_kwargs)


@pytest.mark.slow
def test_switch_transformer():
    """Switch Transformer: top-1 expert routing (small config)."""
    transformers = pytest.importorskip("transformers")
    config = transformers.SwitchTransformersConfig(
        d_model=64,
        d_ff=128,
        d_kv=16,
        num_heads=4,
        num_layers=2,
        num_decoder_layers=2,
        vocab_size=100,
        num_experts=4,
        expert_capacity=4,
    )
    model = transformers.SwitchTransformersModel(config).eval()
    model_kwargs = {
        "input_ids": torch.randint(0, 100, (1, 16)),
        "decoder_input_ids": torch.randint(0, 100, (1, 8)),
    }
    show_model_graph(
        model,
        [],
        model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "moe-models", "switch_transformer"),
    )
    assert validate_forward_pass(model, [], model_kwargs)


# =============================================================================
# Vision Transformers (additional, requires transformers)
# =============================================================================


@pytest.mark.slow
def test_deit():
    """DeiT: data-efficient ViT with distillation token (small config)."""
    transformers = pytest.importorskip("transformers")
    config = transformers.DeiTConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        image_size=32,
        patch_size=8,
        num_channels=3,
    )
    model = transformers.DeiTModel(config).eval()
    model_kwargs = {"pixel_values": torch.rand(1, 3, 32, 32)}
    show_model_graph(
        model,
        [],
        model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-main", "deit"),
    )
    assert validate_forward_pass(model, [], model_kwargs)


@pytest.mark.slow
def test_cvt():
    """CvT: convolutional vision transformer (small config)."""
    transformers = pytest.importorskip("transformers")
    config = transformers.CvtConfig(
        num_channels=3,
        patch_sizes=[3, 3],
        patch_stride=[2, 2],
        patch_padding=[1, 1],
        embed_dim=[32, 64],
        num_heads=[2, 4],
        depth=[1, 2],
        mlp_ratio=[2.0, 2.0],
        cls_token=[False, True],
        stride_q=[1, 1],
        stride_kv=[2, 2],
    )
    model = transformers.CvtModel(config).eval()
    model_kwargs = {"pixel_values": torch.rand(1, 3, 32, 32)}
    show_model_graph(
        model,
        [],
        model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-main", "cvt"),
    )
    assert validate_forward_pass(model, [], model_kwargs)


@pytest.mark.slow
def test_segformer():
    """SegFormer: hierarchical transformer for segmentation (small config)."""
    transformers = pytest.importorskip("transformers")
    config = transformers.SegformerConfig(
        num_channels=3,
        hidden_sizes=[32, 64],
        num_encoder_blocks=2,
        depths=[2, 2],
        sr_ratios=[4, 2],
        num_attention_heads=[2, 4],
        mlp_ratios=[2, 2],
        patch_sizes=[7, 3],
        strides=[4, 2],
    )
    model = transformers.SegformerModel(config).eval()
    model_kwargs = {"pixel_values": torch.rand(1, 3, 64, 64)}
    show_model_graph(
        model,
        [],
        model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-segmentation", "segformer"),
    )
    assert validate_forward_pass(model, [], model_kwargs)


# =============================================================================
# Transformer-Based Detection (requires transformers)
# =============================================================================


@pytest.mark.slow
def test_detr():
    """DETR: transformer encoder-decoder detection with object queries."""
    transformers = pytest.importorskip("transformers")
    config = transformers.DetrConfig(
        d_model=64,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=128,
        decoder_ffn_dim=128,
        num_queries=10,
        num_channels=3,
    )
    model = transformers.DetrModel(config).eval()
    model_kwargs = {"pixel_values": torch.rand(1, 3, 64, 64)}
    show_model_graph(
        model,
        [],
        model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-detection", "detr"),
    )
    assert validate_forward_pass(model, [], model_kwargs)


# =============================================================================
# TorchVision Detection (additional)
# =============================================================================


@pytest.mark.slow
def test_maskrcnn_resnet50_train():
    """Mask R-CNN (train mode): instance segmentation with mask branch."""
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2

    torch.use_deterministic_algorithms(False)
    try:
        model = maskrcnn_resnet50_fpn_v2(weights=None, num_classes=10).train()
        img = [torch.rand(3, 128, 128)]
        targets = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                "labels": torch.tensor([1], dtype=torch.long),
                "masks": torch.zeros(1, 128, 128, dtype=torch.uint8),
            }
        ]
        show_model_graph(
            model,
            (img, targets),
            save_only=True,
            vis_opt="unrolled",
            vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-detection", "maskrcnn_train"),
        )
        assert validate_forward_pass(model, (img, targets))
    finally:
        torch.use_deterministic_algorithms(True)


@pytest.mark.slow
def test_maskrcnn_resnet50_eval():
    """Mask R-CNN (eval mode): instance segmentation inference."""
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(False)
    try:
        model = maskrcnn_resnet50_fpn_v2(weights=None, num_classes=10).eval()
        img = [torch.rand(3, 128, 128)]
        show_model_graph(
            model,
            (img,),
            save_only=True,
            vis_opt="unrolled",
            vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-detection", "maskrcnn_eval"),
        )
        assert validate_forward_pass(model, (img,))
    finally:
        torch.use_deterministic_algorithms(True)


# =============================================================================
# TIMM Vision Models (additional)
# =============================================================================


@pytest.mark.slow
def test_timm_hrnet_w18():
    """HRNet: parallel multi-resolution branches maintained throughout."""
    timm = pytest.importorskip("timm")
    model = timm.create_model("hrnet_w18", pretrained=False).eval()
    model_input = torch.rand(1, 3, 224, 224)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "timm", "hrnet_w18"),
    )
    assert validate_forward_pass(model, model_input)


@pytest.mark.slow
def test_timm_efficientnetv2_s():
    """EfficientNet V2: fused MBConv in early stages."""
    timm = pytest.importorskip("timm")
    model = timm.create_model("efficientnetv2_s", pretrained=False).eval()
    model_input = torch.rand(1, 3, 224, 224)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "timm", "efficientnetv2_s"),
    )
    assert validate_forward_pass(model, model_input)


@pytest.mark.slow
def test_timm_levit_128():
    """LeViT: hybrid CNN-transformer for fast inference."""
    timm = pytest.importorskip("timm")
    model = timm.create_model("levit_128", pretrained=False).eval()
    model_input = torch.rand(1, 3, 224, 224)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "timm", "levit_128"),
    )
    assert validate_forward_pass(model, model_input)


@pytest.mark.slow
def test_timm_crossvit_tiny_240():
    """CrossViT: dual-branch multi-scale patches with cross-attention."""
    timm = pytest.importorskip("timm")
    model = timm.create_model("crossvit_tiny_240", pretrained=False).eval()
    model_input = torch.rand(1, 3, 240, 240)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "timm", "crossvit_tiny_240"),
    )
    assert validate_forward_pass(model, model_input)


@pytest.mark.slow
def test_timm_pvt_v2_b0():
    """PVT v2: Pyramid Vision Transformer with spatial-reduction attention."""
    timm = pytest.importorskip("timm")
    model = timm.create_model("pvt_v2_b0", pretrained=False).eval()
    model_input = torch.rand(1, 3, 224, 224)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "timm", "pvt_v2_b0"),
    )
    assert validate_forward_pass(model, model_input)


@pytest.mark.slow
def test_timm_twins_svt_small():
    """Twins SVT: spatially alternating local-global attention."""
    timm = pytest.importorskip("timm")
    model = timm.create_model("twins_svt_small", pretrained=False).eval()
    model_input = torch.rand(1, 3, 224, 224)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "timm", "twins_svt_small"),
    )
    assert validate_forward_pass(model, model_input)


@pytest.mark.slow
def test_timm_focalnet_tiny_srf():
    """FocalNet: focal modulation replaces attention."""
    timm = pytest.importorskip("timm")
    model = timm.create_model("focalnet_tiny_srf", pretrained=False).eval()
    model_input = torch.rand(1, 3, 224, 224)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "timm", "focalnet_tiny_srf"),
    )
    assert validate_forward_pass(model, model_input)


# =============================================================================
# Perceiver (requires transformers)
# =============================================================================


@pytest.mark.slow
def test_perceiver():
    """Perceiver: cross-attention from input to fixed-size latent array."""
    transformers = pytest.importorskip("transformers")
    config = transformers.PerceiverConfig(
        d_model=64,
        d_latents=64,
        num_latents=16,
        num_self_attends_per_block=2,
        num_blocks=1,
        num_self_attention_heads=4,
        num_cross_attention_heads=4,
        qk_channels=64,
        v_channels=64,
    )
    model = transformers.PerceiverModel(config).eval()
    model_kwargs = {"inputs": torch.rand(1, 16, 64)}
    show_model_graph(
        model,
        [],
        model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "perceiver", "perceiver"),
    )
    assert validate_forward_pass(model, [], model_kwargs)


# =============================================================================
# Time Series (requires transformers)
# =============================================================================


@pytest.mark.slow
def test_patchtst():
    """PatchTST: patch-based time series transformer."""
    transformers = pytest.importorskip("transformers")
    config = transformers.PatchTSTConfig(
        num_input_channels=3,
        context_length=32,
        patch_length=4,
        stride=4,
        d_model=32,
        num_attention_heads=2,
        num_hidden_layers=2,
        ffn_dim=64,
    )
    model = transformers.PatchTSTModel(config).eval()
    model_kwargs = {"past_values": torch.rand(1, 32, 3)}
    show_model_graph(
        model,
        [],
        model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "time-series", "patchtst"),
    )
    assert validate_forward_pass(model, [], model_kwargs)


# =============================================================================
# Reinforcement Learning (requires transformers)
# =============================================================================


@pytest.mark.slow
def test_decision_transformer():
    """Decision Transformer: transformer for offline RL."""
    transformers = pytest.importorskip("transformers")
    config = transformers.DecisionTransformerConfig(
        state_dim=4,
        act_dim=2,
        hidden_size=64,
        max_ep_len=32,
        n_layer=2,
        n_head=4,
    )
    model = transformers.DecisionTransformerModel(config).eval()
    model_kwargs = {
        "states": torch.rand(1, 8, 4),
        "actions": torch.rand(1, 8, 2),
        "rewards": torch.rand(1, 8, 1),
        "returns_to_go": torch.rand(1, 8, 1),
        "timesteps": torch.arange(8).unsqueeze(0),
    }
    show_model_graph(
        model,
        [],
        model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "reinforcement-learning", "decision_transformer"),
    )
    assert validate_forward_pass(model, [], model_kwargs)


# =============================================================================
# Graph Neural Networks (additional, requires torch_geometric)
# =============================================================================


@pytest.mark.slow
def test_graphsage_pyg():
    """GraphSAGE via PyG SAGEConv: neighborhood sampling + mean aggregation."""
    pytest.importorskip("torch_geometric")
    from torch_geometric.nn import SAGEConv

    class _SAGENet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = SAGEConv(8, 16)
            self.conv2 = SAGEConv(16, 4)

        def forward(self, x, edge_index):
            x = torch.relu(self.conv1(x, edge_index))
            return self.conv2(x, edge_index)

    model = _SAGENet().eval()
    x = torch.rand(10, 8)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 9]],
        dtype=torch.long,
    )
    model_input = (x, edge_index)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "graph-neural-networks", "graphsage_pyg"),
    )
    assert validate_forward_pass(model, model_input)


@pytest.mark.slow
def test_gin_pyg():
    """GIN via PyG GINConv: maximally powerful under WL test."""
    pytest.importorskip("torch_geometric")
    from torch_geometric.nn import GINConv

    class _GINNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            mlp1 = torch.nn.Sequential(
                torch.nn.Linear(8, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 16),
            )
            self.conv1 = GINConv(mlp1)
            mlp2 = torch.nn.Sequential(
                torch.nn.Linear(16, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 4),
            )
            self.conv2 = GINConv(mlp2)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            return self.conv2(x, edge_index)

    model = _GINNet().eval()
    x = torch.rand(10, 8)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 9]],
        dtype=torch.long,
    )
    model_input = (x, edge_index)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "graph-neural-networks", "gin_pyg"),
    )
    assert validate_forward_pass(model, model_input)


@pytest.mark.slow
def test_graph_transformer_pyg():
    """Graph Transformer via PyG TransformerConv: attention on graph."""
    pytest.importorskip("torch_geometric")
    from torch_geometric.nn import TransformerConv

    class _GraphTransformerNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = TransformerConv(8, 4, heads=2, concat=True)
            self.conv2 = TransformerConv(8, 4, heads=1, concat=False)

        def forward(self, x, edge_index):
            x = torch.relu(self.conv1(x, edge_index))
            return self.conv2(x, edge_index)

    model = _GraphTransformerNet().eval()
    x = torch.rand(10, 8)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 9]],
        dtype=torch.long,
    )
    model_input = (x, edge_index)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "graph-neural-networks", "graph_transformer_pyg"),
    )
    assert validate_forward_pass(model, model_input)


# =============================================================================
# TorchVision Classification (additional)
# =============================================================================


@pytest.mark.slow
def test_mobilenet_v3_small():
    """MobileNet V3: h-swish activation, squeeze-excitation, NAS-tuned."""
    model = torchvision.models.mobilenet_v3_small(weights=None).eval()
    model_input = torch.rand(1, 3, 224, 224)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-main", "mobilenet_v3_small"),
    )
    assert validate_forward_pass(model, model_input)


# =============================================================================
# TIMM (additional gap-fill)
# =============================================================================


@pytest.mark.slow
def test_timm_res2net50():
    """Res2Net: hierarchical residual-like connections within a single block."""
    timm = pytest.importorskip("timm")
    model = timm.create_model("res2net50_26w_4s", pretrained=False).eval()
    model_input = torch.rand(1, 3, 224, 224)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "timm", "res2net50"),
    )
    assert validate_forward_pass(model, model_input)


@pytest.mark.slow
def test_timm_gmlp_s16():
    """gMLP: gated MLP with spatial gating (no attention)."""
    timm = pytest.importorskip("timm")
    model = timm.create_model("gmlp_s16_224", pretrained=False).eval()
    model_input = torch.rand(1, 3, 224, 224)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "timm", "gmlp_s16"),
    )
    assert validate_forward_pass(model, model_input)


@pytest.mark.slow
def test_timm_resmlp_12():
    """ResMLP: residual MLP with cross-token affine transforms."""
    timm = pytest.importorskip("timm")
    model = timm.create_model("resmlp_12_224", pretrained=False).eval()
    model_input = torch.rand(1, 3, 224, 224)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "timm", "resmlp_12"),
    )
    assert validate_forward_pass(model, model_input)


@pytest.mark.slow
def test_timm_eva02_small():
    """EVA-02: scaled ViT with rotary position embeddings."""
    timm = pytest.importorskip("timm")
    model = timm.create_model("eva02_small_patch14_224", pretrained=False).eval()
    model_input = torch.rand(1, 3, 224, 224)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "timm", "eva02_small"),
    )
    assert validate_forward_pass(model, model_input)


# =============================================================================
# Decoder-Only LLMs (additional)
# =============================================================================


@pytest.mark.slow
def test_olmo():
    """OLMo: fully open decoder-only LLM."""
    transformers = pytest.importorskip("transformers")
    config = transformers.OlmoConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        vocab_size=100,
        max_position_embeddings=32,
    )
    model = transformers.OlmoForCausalLM(config).eval()
    x = torch.randint(0, 100, (1, 16))
    model_kwargs = {"input_ids": x}
    show_model_graph(
        model,
        [],
        model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "decoder-only-llms", "olmo"),
    )
    assert validate_forward_pass(model, [], model_kwargs)


# =============================================================================
# Efficient Transformers (additional)
# =============================================================================


@pytest.mark.slow
def test_longformer():
    """Longformer: sliding window + global attention on special tokens.

    Note: validation skipped — windowed attention uses as_strided in ways
    that bypass perturbation detection.
    """
    transformers = pytest.importorskip("transformers")
    config = transformers.LongformerConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        vocab_size=100,
        max_position_embeddings=32,
        attention_window=[8, 8],
    )
    model = transformers.LongformerModel(config).eval()
    x = torch.randint(0, 100, (1, 16))
    model_kwargs = {"input_ids": x}
    show_model_graph(
        model,
        [],
        model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "efficient-transformers", "longformer"),
    )


@pytest.mark.slow
def test_reformer():
    """Reformer: LSH attention + reversible layers."""
    transformers = pytest.importorskip("transformers")
    config = transformers.ReformerConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        attn_layers=["local", "lsh"],
        feed_forward_size=128,
        vocab_size=100,
        max_position_embeddings=32,
        axial_pos_embds=False,
        lsh_attn_chunk_length=8,
        local_attn_chunk_length=8,
        num_hashes=1,
        num_buckets=16,
    )
    model = transformers.ReformerModel(config).eval()
    x = torch.randint(0, 100, (1, 16))
    model_kwargs = {"input_ids": x}
    torch.use_deterministic_algorithms(False)
    try:
        show_model_graph(
            model,
            [],
            model_kwargs,
            save_only=True,
            vis_opt="unrolled",
            vis_outpath=opj(VIS_OUTPUT_DIR, "efficient-transformers", "reformer"),
        )
        assert validate_forward_pass(model, [], model_kwargs)
    finally:
        torch.use_deterministic_algorithms(True)


# =============================================================================
# Vision Models (additional HF)
# =============================================================================


@pytest.mark.slow
def test_dinov2():
    """DINOv2: self-supervised ViT (self-distillation, no labels)."""
    transformers = pytest.importorskip("transformers")
    config = transformers.Dinov2Config(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        image_size=32,
        patch_size=8,
        num_channels=3,
    )
    model = transformers.Dinov2Model(config).eval()
    pixel_values = torch.rand(1, 3, 32, 32)
    model_kwargs = {"pixel_values": pixel_values}
    show_model_graph(
        model,
        [],
        model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-main", "dinov2"),
    )
    assert validate_forward_pass(model, [], model_kwargs)


# =============================================================================
# Audio Models (additional HF)
# =============================================================================


@pytest.mark.slow
def test_audio_ast():
    """Audio Spectrogram Transformer: ViT on mel spectrograms."""
    transformers = pytest.importorskip("transformers")
    config = transformers.ASTConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        max_length=32,
        num_mel_bins=16,
        patch_size=4,
        frequency_stride=4,
        time_stride=4,
    )
    model = transformers.ASTModel(config).eval()
    input_values = torch.rand(1, 32, 16)
    model_kwargs = {"input_values": input_values}
    show_model_graph(
        model,
        [],
        model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchaudio", "ast"),
    )
    assert validate_forward_pass(model, [], model_kwargs)


@pytest.mark.slow
def test_audio_clap():
    """CLAP: contrastive language-audio pretraining (dual encoder)."""
    transformers = pytest.importorskip("transformers")
    config = transformers.ClapConfig(
        text_config={
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 128,
            "vocab_size": 100,
            "max_position_embeddings": 32,
        },
        audio_config={
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": [2, 2],
            "patch_size": 4,
            "patch_embeds_hidden_size": 32,
            "spec_size": 32,
            "num_mel_bins": 32,
            "depths": [2, 2],
            "window_size": 4,
        },
        projection_dim=32,
    )
    model = transformers.ClapModel(config).eval()
    input_ids = torch.randint(0, 100, (1, 16))
    input_features = torch.rand(1, 1, 32, 32)
    model_kwargs = {
        "input_ids": input_ids,
        "input_features": input_features,
    }
    show_model_graph(
        model,
        [],
        model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchaudio", "clap"),
    )
    assert validate_forward_pass(model, [], model_kwargs)


@pytest.mark.slow
def test_audio_encodec():
    """EnCodec: neural audio codec with residual vector quantization."""
    transformers = pytest.importorskip("transformers")
    config = transformers.EncodecConfig(
        target_bandwidths=[3.6],
        sampling_rate=24000,
        audio_channels=1,
        hidden_size=32,
        num_filters=8,
        num_residual_layers=1,
        upsampling_ratios=[5, 4, 4, 2],
        codebook_size=64,
        codebook_dim=32,
    )
    model = transformers.EncodecModel(config).eval()
    audio = torch.rand(1, 1, 3200)
    model_kwargs = {"input_values": audio}
    show_model_graph(
        model,
        [],
        model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchaudio", "encodec"),
    )
    assert validate_forward_pass(model, [], model_kwargs)


@pytest.mark.slow
def test_audio_sew():
    """SEW: Squeezed and Efficient Wav2Vec."""
    transformers = pytest.importorskip("transformers")
    config = transformers.SEWConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        conv_dim=(32, 32, 64),
        conv_kernel=(5, 3, 3),
        conv_stride=(2, 2, 2),
        squeeze_factor=2,
    )
    model = transformers.SEWModel(config).eval()
    x = torch.rand(1, 1024)
    model_kwargs = {"input_values": x}
    show_model_graph(
        model,
        [],
        model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchaudio", "sew"),
    )
    assert validate_forward_pass(model, [], model_kwargs)


@pytest.mark.slow
def test_audio_speecht5():
    """SpeechT5: unified text-speech model (encoder-decoder)."""
    transformers = pytest.importorskip("transformers")
    config = transformers.SpeechT5Config(
        hidden_size=64,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=128,
        decoder_ffn_dim=128,
        vocab_size=100,
        num_mel_bins=20,
    )
    from transformers.models.speecht5.modeling_speecht5 import (
        SpeechT5EncoderWithTextPrenet,
        SpeechT5DecoderWithSpeechPrenet,
    )

    encoder = SpeechT5EncoderWithTextPrenet(config)
    decoder = SpeechT5DecoderWithSpeechPrenet(config)
    model = transformers.SpeechT5Model(config, encoder=encoder, decoder=decoder).eval()
    input_values = torch.randint(0, 100, (1, 16))
    decoder_input_values = torch.rand(1, 50, 20)
    model_kwargs = {
        "input_values": input_values,
        "decoder_input_values": decoder_input_values,
    }
    show_model_graph(
        model,
        [],
        model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchaudio", "speecht5"),
    )
    assert validate_forward_pass(model, [], model_kwargs)


@pytest.mark.slow
def test_audio_vits():
    """VITS: end-to-end TTS with VAE + normalizing flow + adversarial training."""
    transformers = pytest.importorskip("transformers")
    config = transformers.VitsConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        vocab_size=100,
        flow_size=64,
        spectrogram_bins=32,
        posterior_encoder_num_wavenet_layers=2,
        upsample_rates=[4, 4],
        upsample_kernel_sizes=[8, 8],
        upsample_initial_channel=64,
        resblock_kernel_sizes=[3, 5],
        resblock_dilation_sizes=[[1, 2], [1, 2]],
    )
    model = transformers.VitsModel(config).eval()
    input_ids = torch.randint(0, 100, (1, 16))
    model_kwargs = {"input_ids": input_ids}
    show_model_graph(
        model,
        [],
        model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchaudio", "vits"),
    )
    assert validate_forward_pass(model, [], model_kwargs)


# =============================================================================
# Time Series (additional)
# =============================================================================


@pytest.mark.slow
def test_informer():
    """Informer: ProbSparse attention for long-range time series."""
    transformers = pytest.importorskip("transformers")
    config = transformers.InformerConfig(
        prediction_length=4,
        context_length=16,
        input_size=1,
        d_model=32,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        scaling="std",
        num_time_features=1,
    )
    model = transformers.InformerModel(config).eval()
    past_values = torch.rand(1, 23)
    past_time_features = torch.rand(1, 23, 1)
    past_observed_mask = torch.ones(1, 23)
    future_time_features = torch.rand(1, 4, 1)
    model_kwargs = {
        "past_values": past_values,
        "past_time_features": past_time_features,
        "past_observed_mask": past_observed_mask,
        "future_time_features": future_time_features,
    }
    show_model_graph(
        model,
        [],
        model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "time-series", "informer"),
    )
    assert validate_forward_pass(model, [], model_kwargs)


@pytest.mark.slow
def test_autoformer():
    """Autoformer: decomposition-based attention with auto-correlation."""
    transformers = pytest.importorskip("transformers")
    config = transformers.AutoformerConfig(
        prediction_length=4,
        context_length=16,
        input_size=1,
        d_model=32,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        scaling="std",
        moving_average=5,
        num_time_features=1,
    )
    model = transformers.AutoformerModel(config).eval()
    past_values = torch.rand(1, 23)
    past_time_features = torch.rand(1, 23, 1)
    past_observed_mask = torch.ones(1, 23)
    future_time_features = torch.rand(1, 4, 1)
    model_kwargs = {
        "past_values": past_values,
        "past_time_features": past_time_features,
        "past_observed_mask": past_observed_mask,
        "future_time_features": future_time_features,
    }
    show_model_graph(
        model,
        [],
        model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "time-series", "autoformer"),
    )
    assert validate_forward_pass(model, [], model_kwargs)


# =============================================================================
# TorchVision Detection (additional)
# =============================================================================


@pytest.mark.slow
def test_keypointrcnn_resnet50_train():
    """Keypoint R-CNN: train mode with keypoint targets."""
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(
        weights=None, num_classes=2, num_keypoints=5
    )
    img = [torch.rand(3, 128, 128)]
    targets = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "labels": torch.tensor([1], dtype=torch.long),
            "keypoints": torch.rand(1, 5, 3),
        }
    ]
    model_input = (img, targets)
    torch.use_deterministic_algorithms(False)
    try:
        show_model_graph(
            model,
            model_input,
            save_only=True,
            vis_opt="unrolled",
            vis_outpath=opj(
                VIS_OUTPUT_DIR,
                "torchvision-detection",
                "keypointrcnn_resnet50_train",
            ),
        )
        assert validate_forward_pass(model, model_input)
    finally:
        torch.use_deterministic_algorithms(True)


@pytest.mark.slow
def test_keypointrcnn_resnet50_eval():
    """Keypoint R-CNN: eval mode."""
    torch.manual_seed(0)
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(
        weights=None, num_classes=2, num_keypoints=5
    ).eval()
    img = [torch.rand(3, 128, 128)]
    model_input = (img,)
    torch.use_deterministic_algorithms(False)
    try:
        show_model_graph(
            model,
            model_input,
            save_only=True,
            vis_opt="unrolled",
            vis_outpath=opj(
                VIS_OUTPUT_DIR,
                "torchvision-detection",
                "keypointrcnn_resnet50_eval",
            ),
        )
        assert validate_forward_pass(model, model_input)
    finally:
        torch.use_deterministic_algorithms(True)


# =============================================================================
# Graph Neural Networks (additional PyG)
# =============================================================================


@pytest.mark.slow
def test_gatv2_pyg():
    """GATv2: dynamic attention (more expressive than GAT)."""
    pytest.importorskip("torch_geometric")
    from torch_geometric.nn import GATv2Conv

    class _GATv2Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GATv2Conv(8, 4, heads=2, concat=True)
            self.conv2 = GATv2Conv(8, 4, heads=1, concat=False)

        def forward(self, x, edge_index):
            x = torch.relu(self.conv1(x, edge_index))
            return self.conv2(x, edge_index)

    model = _GATv2Net().eval()
    x = torch.rand(10, 8)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 9]],
        dtype=torch.long,
    )
    model_input = (x, edge_index)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "graph-neural-networks", "gatv2_pyg"),
    )
    assert validate_forward_pass(model, model_input)


@pytest.mark.slow
def test_rgcn_pyg():
    """R-GCN: relational GCN with per-relation-type weights."""
    pytest.importorskip("torch_geometric")
    from torch_geometric.nn import RGCNConv

    class _RGCNNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = RGCNConv(8, 16, num_relations=3)
            self.conv2 = RGCNConv(16, 4, num_relations=3)

        def forward(self, x, edge_index, edge_type):
            x = torch.relu(self.conv1(x, edge_index, edge_type))
            return self.conv2(x, edge_index, edge_type)

    model = _RGCNNet().eval()
    x = torch.rand(10, 8)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 9]],
        dtype=torch.long,
    )
    edge_type = torch.randint(0, 3, (10,))
    model_input = (x, edge_index, edge_type)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "graph-neural-networks", "rgcn_pyg"),
    )
    assert validate_forward_pass(model, model_input)


# =============================================================================
# Decoder-Only LLMs (Additional)
# =============================================================================


@pytest.mark.slow
def test_gptj():
    """GPT-J: parallel attention + FFN (different from GPT-2 sequential)."""
    pytest.importorskip("transformers")
    from transformers import GPTJConfig, GPTJModel

    config = GPTJConfig(
        vocab_size=100,
        n_embd=64,
        n_layer=2,
        n_head=4,
        rotary_dim=16,
        n_positions=64,
    )
    model = GPTJModel(config).eval()
    input_ids = torch.randint(0, 100, (2, 16))
    model_input = []
    model_kwargs = {"input_ids": input_ids}
    show_model_graph(
        model,
        model_input,
        model_kwargs=model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "decoder-only-llms", "gptj"),
    )
    assert validate_forward_pass(model, model_input, model_kwargs=model_kwargs)


@pytest.mark.slow
def test_gpt_bigcode():
    """GPTBigCode (StarCoder arch): multi-query attention for code."""
    pytest.importorskip("transformers")
    from transformers import GPTBigCodeConfig, GPTBigCodeModel

    config = GPTBigCodeConfig(
        vocab_size=100,
        n_embd=64,
        n_layer=2,
        n_head=4,
        n_positions=64,
        multi_query=True,
    )
    model = GPTBigCodeModel(config).eval()
    input_ids = torch.randint(0, 100, (2, 16))
    model_input = []
    model_kwargs = {"input_ids": input_ids}
    show_model_graph(
        model,
        model_input,
        model_kwargs=model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "decoder-only-llms", "gpt_bigcode"),
    )
    assert validate_forward_pass(model, model_input, model_kwargs=model_kwargs)


@pytest.mark.slow
def test_gpt_neox():
    """GPT-NeoX: parallel attention + FFN, rotary embeddings."""
    pytest.importorskip("transformers")
    from transformers import GPTNeoXConfig, GPTNeoXModel

    config = GPTNeoXConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=64,
    )
    model = GPTNeoXModel(config).eval()
    input_ids = torch.randint(0, 100, (2, 16))
    model_input = []
    model_kwargs = {"input_ids": input_ids}
    show_model_graph(
        model,
        model_input,
        model_kwargs=model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "decoder-only-llms", "gpt_neox"),
    )
    assert validate_forward_pass(model, model_input, model_kwargs=model_kwargs)


# =============================================================================
# Encoder-Only (Additional)
# =============================================================================


@pytest.mark.slow
def test_funnel_transformer():
    """Funnel Transformer: progressively reduces sequence length through layers."""
    pytest.importorskip("transformers")
    from transformers import FunnelConfig, FunnelModel

    config = FunnelConfig(
        vocab_size=100,
        d_model=64,
        n_head=4,
        d_inner=128,
        block_sizes=[2, 2],
        num_decoder_layers=1,
    )
    model = FunnelModel(config).eval()
    input_ids = torch.randint(0, 100, (2, 16))
    model_input = []
    model_kwargs = {"input_ids": input_ids}
    show_model_graph(
        model,
        model_input,
        model_kwargs=model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "encoder-only", "funnel_transformer"),
    )
    assert validate_forward_pass(model, model_input, model_kwargs=model_kwargs)


@pytest.mark.slow
def test_canine():
    """CANINE: character-level tokenization-free transformer."""
    pytest.importorskip("transformers")
    from transformers import CanineConfig, CanineModel

    config = CanineConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=128,
    )
    model = CanineModel(config).eval()
    # CANINE takes raw character codepoints (integers)
    input_ids = torch.randint(0, 128, (2, 32))
    model_input = []
    model_kwargs = {"input_ids": input_ids}
    show_model_graph(
        model,
        model_input,
        model_kwargs=model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "encoder-only", "canine"),
    )
    assert validate_forward_pass(model, model_input, model_kwargs=model_kwargs)


@pytest.mark.slow
def test_mobilebert():
    """MobileBERT: bottleneck-structured BERT for mobile."""
    pytest.importorskip("transformers")
    from transformers import MobileBertConfig, MobileBertModel

    config = MobileBertConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        embedding_size=32,
        true_hidden_size=64,
    )
    model = MobileBertModel(config).eval()
    input_ids = torch.randint(0, 100, (2, 16))
    model_input = []
    model_kwargs = {"input_ids": input_ids}
    show_model_graph(
        model,
        model_input,
        input_kwargs=model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "encoder-only", "mobilebert"),
    )
    assert validate_forward_pass(model, model_input, input_kwargs=model_kwargs)


# =============================================================================
# Encoder-Decoder (Additional)
# =============================================================================


@pytest.mark.slow
def test_mbart():
    """mBART: multilingual BART."""
    pytest.importorskip("transformers")
    from transformers import MBartConfig, MBartModel

    config = MBartConfig(
        vocab_size=100,
        d_model=64,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=128,
        decoder_ffn_dim=128,
        max_position_embeddings=64,
    )
    model = MBartModel(config).eval()
    input_ids = torch.randint(0, 100, (2, 16))
    decoder_input_ids = torch.randint(0, 100, (2, 8))
    model_input = []
    model_kwargs = {"input_ids": input_ids, "decoder_input_ids": decoder_input_ids}
    show_model_graph(
        model,
        model_input,
        model_kwargs=model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "encoder-decoder", "mbart"),
    )
    assert validate_forward_pass(model, model_input, model_kwargs=model_kwargs)


@pytest.mark.slow
def test_prophetnet():
    """ProphetNet: n-gram prediction encoder-decoder."""
    pytest.importorskip("transformers")
    from transformers import ProphetNetConfig, ProphetNetModel

    config = ProphetNetConfig(
        vocab_size=100,
        hidden_size=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_encoder_attention_heads=4,
        num_decoder_attention_heads=4,
        encoder_ffn_dim=128,
        decoder_ffn_dim=128,
        max_position_embeddings=64,
        ngram=2,
    )
    model = ProphetNetModel(config).eval()
    input_ids = torch.randint(0, 100, (2, 16))
    decoder_input_ids = torch.randint(0, 100, (2, 8))
    model_input = []
    model_kwargs = {"input_ids": input_ids, "decoder_input_ids": decoder_input_ids}
    show_model_graph(
        model,
        model_input,
        model_kwargs=model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "encoder-decoder", "prophetnet"),
    )
    assert validate_forward_pass(model, model_input, model_kwargs=model_kwargs)


# =============================================================================
# Audio (Additional Set 2)
# =============================================================================


@pytest.mark.slow
def test_audio_wavlm():
    """WavLM: masked speech denoising self-supervised model."""
    pytest.importorskip("transformers")
    from transformers import WavLMConfig, WavLMModel

    config = WavLMConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        conv_dim=(32, 32),
        conv_kernel=(10, 3),
        conv_stride=(5, 2),
    )
    model = WavLMModel(config).eval()
    waveform = torch.rand(2, 3200)
    model_input = []
    model_kwargs = {"input_values": waveform}
    show_model_graph(
        model,
        model_input,
        model_kwargs=model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchaudio", "wavlm"),
    )
    assert validate_forward_pass(model, model_input, model_kwargs=model_kwargs)


@pytest.mark.slow
def test_audio_data2vec():
    """Data2VecAudio: self-distillation on audio representations."""
    pytest.importorskip("transformers")
    from transformers import Data2VecAudioConfig, Data2VecAudioModel

    config = Data2VecAudioConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        conv_dim=(32, 32),
        conv_kernel=(10, 3),
        conv_stride=(5, 2),
    )
    model = Data2VecAudioModel(config).eval()
    waveform = torch.rand(2, 3200)
    model_input = []
    model_kwargs = {"input_values": waveform}
    show_model_graph(
        model,
        model_input,
        model_kwargs=model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchaudio", "data2vec_audio"),
    )
    assert validate_forward_pass(model, model_input, model_kwargs=model_kwargs)


@pytest.mark.slow
def test_audio_unispeech():
    """UniSpeech: unified speech representation learning."""
    pytest.importorskip("transformers")
    from transformers import UniSpeechConfig, UniSpeechModel

    config = UniSpeechConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        conv_dim=(32, 32),
        conv_kernel=(10, 3),
        conv_stride=(5, 2),
    )
    model = UniSpeechModel(config).eval()
    waveform = torch.rand(2, 3200)
    model_input = []
    model_kwargs = {"input_values": waveform}
    show_model_graph(
        model,
        model_input,
        model_kwargs=model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchaudio", "unispeech"),
    )
    assert validate_forward_pass(model, model_input, model_kwargs=model_kwargs)


# =============================================================================
# TIMM Models (Additional Set 2)
# =============================================================================


@pytest.mark.slow
def test_timm_convnextv2_atto():
    """ConvNeXt v2: Global Response Normalization (GRN)."""
    timm = pytest.importorskip("timm")
    model = timm.create_model("convnextv2_atto", pretrained=False, num_classes=10).eval()
    x = torch.rand(2, 3, 224, 224)
    show_model_graph(
        model,
        x,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "timm", "convnextv2_atto"),
    )
    assert validate_forward_pass(model, x)


@pytest.mark.slow
def test_timm_nfnet_l0():
    """NFNet: normalizer-free network (no batch norm)."""
    timm = pytest.importorskip("timm")
    model = timm.create_model("nfnet_l0", pretrained=False, num_classes=10).eval()
    x = torch.rand(2, 3, 224, 224)
    show_model_graph(
        model,
        x,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "timm", "nfnet_l0"),
    )
    assert validate_forward_pass(model, x)


@pytest.mark.slow
def test_timm_davit_tiny():
    """DaViT: dual attention ViT (spatial + channel attention)."""
    timm = pytest.importorskip("timm")
    model = timm.create_model("davit_tiny", pretrained=False, num_classes=10).eval()
    x = torch.rand(2, 3, 224, 224)
    show_model_graph(
        model,
        x,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "timm", "davit_tiny"),
    )
    assert validate_forward_pass(model, x)


@pytest.mark.slow
def test_timm_coatnet():
    """CoAtNet: CNN + Transformer hybrid (conv early, attention late)."""
    timm = pytest.importorskip("timm")
    model = timm.create_model("coatnet_0_rw_224", pretrained=False, num_classes=10).eval()
    x = torch.rand(2, 3, 224, 224)
    show_model_graph(
        model,
        x,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "timm", "coatnet_0"),
    )
    assert validate_forward_pass(model, x)


@pytest.mark.slow
def test_timm_repvgg_a0():
    """RepVGG: reparameterizable VGG-style (multi-branch train, single inference)."""
    timm = pytest.importorskip("timm")
    model = timm.create_model("repvgg_a0", pretrained=False, num_classes=10).eval()
    x = torch.rand(2, 3, 224, 224)
    show_model_graph(
        model,
        x,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "timm", "repvgg_a0"),
    )
    assert validate_forward_pass(model, x)


@pytest.mark.slow
def test_timm_rexnet():
    """ReXNet: rank expansion network (learned channel expansion ratios)."""
    timm = pytest.importorskip("timm")
    model = timm.create_model("rexnet_100", pretrained=False, num_classes=10).eval()
    x = torch.rand(2, 3, 224, 224)
    show_model_graph(
        model,
        x,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "timm", "rexnet_100"),
    )
    assert validate_forward_pass(model, x)


@pytest.mark.slow
def test_timm_pit():
    """PiT: Pooling-based ViT (spatial token reduction via pooling)."""
    timm = pytest.importorskip("timm")
    model = timm.create_model("pit_ti_224", pretrained=False, num_classes=10).eval()
    x = torch.rand(2, 3, 224, 224)
    show_model_graph(
        model,
        x,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "timm", "pit_ti_224"),
    )
    assert validate_forward_pass(model, x)


@pytest.mark.slow
def test_timm_visformer():
    """Visformer: vision-friendly transformer (spatial conv in early stages)."""
    timm = pytest.importorskip("timm")
    model = timm.create_model("visformer_tiny", pretrained=False, num_classes=10).eval()
    x = torch.rand(2, 3, 224, 224)
    show_model_graph(
        model,
        x,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "timm", "visformer_tiny"),
    )
    assert validate_forward_pass(model, x)


@pytest.mark.slow
def test_timm_gcvit():
    """GC-ViT: Global Context Vision Transformer."""
    timm = pytest.importorskip("timm")
    model = timm.create_model("gcvit_xxtiny", pretrained=False, num_classes=10).eval()
    x = torch.rand(2, 3, 224, 224)
    show_model_graph(
        model,
        x,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "timm", "gcvit_xxtiny"),
    )
    assert validate_forward_pass(model, x)


@pytest.mark.slow
def test_timm_efficientformer():
    """EfficientFormer: hardware-efficient ViT."""
    timm = pytest.importorskip("timm")
    model = timm.create_model("efficientformer_l1", pretrained=False, num_classes=10).eval()
    x = torch.rand(2, 3, 224, 224)
    show_model_graph(
        model,
        x,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "timm", "efficientformer_l1"),
    )
    assert validate_forward_pass(model, x)


@pytest.mark.slow
def test_timm_fastvit():
    """FastViT: reparameterizable hybrid CNN-transformer."""
    timm = pytest.importorskip("timm")
    model = timm.create_model("fastvit_t8", pretrained=False, num_classes=10).eval()
    x = torch.rand(2, 3, 256, 256)
    show_model_graph(
        model,
        x,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "timm", "fastvit_t8"),
    )
    assert validate_forward_pass(model, x)


@pytest.mark.slow
def test_timm_nest():
    """NesT: Nested Hierarchical Transformer (aggregation within blocks)."""
    timm = pytest.importorskip("timm")
    model = timm.create_model("nest_tiny", pretrained=False, num_classes=10).eval()
    x = torch.rand(2, 3, 224, 224)
    show_model_graph(
        model,
        x,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "timm", "nest_tiny"),
    )
    assert validate_forward_pass(model, x)


@pytest.mark.slow
def test_timm_sequencer():
    """Sequencer2D: LSTM-based token mixing for vision."""
    timm = pytest.importorskip("timm")
    model = timm.create_model("sequencer2d_s", pretrained=False, num_classes=10).eval()
    x = torch.rand(2, 3, 224, 224)
    show_model_graph(
        model,
        x,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "timm", "sequencer2d_s"),
    )
    assert validate_forward_pass(model, x)


@pytest.mark.slow
def test_timm_tresnet():
    """TResNet: training tricks for ResNet (anti-alias, SpaceToBatch)."""
    timm = pytest.importorskip("timm")
    model = timm.create_model("tresnet_m", pretrained=False, num_classes=10).eval()
    x = torch.rand(2, 3, 224, 224)
    show_model_graph(
        model,
        x,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "timm", "tresnet_m"),
    )
    assert validate_forward_pass(model, x)


# =============================================================================
# Multimodal (Additional)
# =============================================================================


@pytest.mark.slow
def test_siglip():
    """SigLIP: sigmoid contrastive loss (replaces softmax in CLIP)."""
    pytest.importorskip("transformers")
    from transformers import SiglipConfig, SiglipModel

    text_config = {
        "vocab_size": 100,
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "intermediate_size": 128,
        "max_position_embeddings": 32,
    }
    vision_config = {
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "intermediate_size": 128,
        "image_size": 64,
        "patch_size": 16,
        "num_channels": 3,
    }
    config = SiglipConfig(text_config=text_config, vision_config=vision_config)
    model = SiglipModel(config).eval()
    input_ids = torch.randint(0, 100, (2, 8))
    pixel_values = torch.rand(2, 3, 64, 64)
    model_input = []
    model_kwargs = {"input_ids": input_ids, "pixel_values": pixel_values}
    show_model_graph(
        model,
        model_input,
        model_kwargs=model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "multimodal-models", "siglip"),
    )
    assert validate_forward_pass(model, model_input, model_kwargs=model_kwargs)


@pytest.mark.slow
def test_blip2():
    """BLIP-2: Q-Former bridge module between vision and language."""
    pytest.importorskip("transformers")
    from transformers import Blip2Config, Blip2Model

    vision_config = {
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "intermediate_size": 128,
        "image_size": 64,
        "patch_size": 16,
    }
    qformer_config = {
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "intermediate_size": 128,
        "cross_attention_frequency": 1,
        "vocab_size": 100,
    }
    config = Blip2Config(
        vision_config=vision_config,
        qformer_config=qformer_config,
        num_query_tokens=8,
    )
    model = Blip2Model(config).eval()
    pixel_values = torch.rand(2, 3, 64, 64)
    model_input = []
    model_kwargs = {"pixel_values": pixel_values}
    show_model_graph(
        model,
        model_input,
        model_kwargs=model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "multimodal-models", "blip2"),
    )
    assert validate_forward_pass(model, model_input, model_kwargs=model_kwargs)


# =============================================================================
# Detection (Additional Set 2)
# =============================================================================


@pytest.mark.slow
def test_deformable_detr():
    """Deformable DETR: deformable attention for efficient multi-scale detection."""
    pytest.importorskip("transformers")
    from transformers import DeformableDetrConfig, DeformableDetrModel

    config = DeformableDetrConfig(
        d_model=64,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=128,
        decoder_ffn_dim=128,
        num_feature_levels=2,
        backbone_config={
            "model_type": "resnet",
            "hidden_sizes": [32, 64],
            "depths": [1, 1],
            "out_features": ["stage2", "stage3"],
        },
    )
    model = DeformableDetrModel(config).eval()
    pixel_values = torch.rand(2, 3, 64, 64)
    model_input = []
    model_kwargs = {"pixel_values": pixel_values}
    try:
        torch.use_deterministic_algorithms(False)
        show_model_graph(
            model,
            model_input,
            model_kwargs=model_kwargs,
            save_only=True,
            vis_opt="unrolled",
            vis_outpath=opj(VIS_OUTPUT_DIR, "detection-additional", "deformable_detr"),
        )
        assert validate_forward_pass(model, model_input, model_kwargs=model_kwargs)
    finally:
        torch.use_deterministic_algorithms(True)


# =============================================================================
# Document Understanding
# =============================================================================


@pytest.mark.slow
def test_layoutlm():
    """LayoutLM: joint text + layout (bounding box positions) understanding."""
    pytest.importorskip("transformers")
    from transformers import LayoutLMConfig, LayoutLMModel

    config = LayoutLMConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=64,
        max_2d_position_embeddings=256,
    )
    model = LayoutLMModel(config).eval()
    input_ids = torch.randint(0, 100, (2, 8))
    bbox = torch.randint(0, 255, (2, 8, 4))
    model_input = []
    model_kwargs = {"input_ids": input_ids, "bbox": bbox}
    show_model_graph(
        model,
        model_input,
        model_kwargs=model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "nlp-models", "layoutlm"),
    )
    assert validate_forward_pass(model, model_input, model_kwargs=model_kwargs)


# =============================================================================
# Time Series (Additional)
# =============================================================================


@pytest.mark.slow
def test_time_series_transformer():
    """TimeSeriesTransformer: HF base time series forecasting model."""
    pytest.importorskip("transformers")
    from transformers import (
        TimeSeriesTransformerConfig,
        TimeSeriesTransformerModel,
    )

    config = TimeSeriesTransformerConfig(
        prediction_length=4,
        context_length=16,
        d_model=32,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        input_size=1,
        num_time_features=2,
        lags_sequence=[1, 2],
        scaling="std",
    )
    model = TimeSeriesTransformerModel(config).eval()
    seq_len = 18  # context_length + max(lags_sequence)
    past_values = torch.rand(2, seq_len)
    past_time_features = torch.rand(2, seq_len, 2)
    past_observed_mask = torch.ones(2, seq_len)
    future_values = torch.rand(2, 4)
    future_time_features = torch.rand(2, 4, 2)
    model_input = []
    model_kwargs = {
        "past_values": past_values,
        "past_time_features": past_time_features,
        "past_observed_mask": past_observed_mask,
        "future_values": future_values,
        "future_time_features": future_time_features,
    }
    show_model_graph(
        model,
        model_input,
        model_kwargs=model_kwargs,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "time-series", "time_series_transformer"),
    )
    assert validate_forward_pass(model, model_input, model_kwargs=model_kwargs)


# =============================================================================
# GNN PyG (Additional)
# =============================================================================


@pytest.mark.slow
def test_chebconv_pyg():
    """ChebConv: Chebyshev spectral graph convolution."""
    pytest.importorskip("torch_geometric")
    from torch_geometric.nn import ChebConv

    class _ChebNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = ChebConv(8, 16, K=3)
            self.conv2 = ChebConv(16, 4, K=3)

        def forward(self, x, edge_index):
            x = torch.relu(self.conv1(x, edge_index))
            return self.conv2(x, edge_index)

    model = _ChebNet().eval()
    x = torch.rand(10, 8)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 9]],
        dtype=torch.long,
    )
    model_input = (x, edge_index)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "graph-neural-networks", "chebconv_pyg"),
    )
    assert validate_forward_pass(model, model_input)


@pytest.mark.slow
def test_sgc_pyg():
    """SGConv: Simple Graph Convolution (remove nonlinearities between layers)."""
    pytest.importorskip("torch_geometric")
    from torch_geometric.nn import SGConv

    class _SGCNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = SGConv(8, 4, K=3)

        def forward(self, x, edge_index):
            return self.conv(x, edge_index)

    model = _SGCNet().eval()
    x = torch.rand(10, 8)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 9]],
        dtype=torch.long,
    )
    model_input = (x, edge_index)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "graph-neural-networks", "sgc_pyg"),
    )
    assert validate_forward_pass(model, model_input)


@pytest.mark.slow
def test_tag_pyg():
    """TAGConv: Topology Adaptive Graph Convolution (fixed polynomial)."""
    pytest.importorskip("torch_geometric")
    from torch_geometric.nn import TAGConv

    class _TAGNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = TAGConv(8, 16, K=3)
            self.conv2 = TAGConv(16, 4, K=3)

        def forward(self, x, edge_index):
            x = torch.relu(self.conv1(x, edge_index))
            return self.conv2(x, edge_index)

    model = _TAGNet().eval()
    x = torch.rand(10, 8)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 9]],
        dtype=torch.long,
    )
    model_input = (x, edge_index)
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "graph-neural-networks", "tag_pyg"),
    )
    assert validate_forward_pass(model, model_input)


# =============================================================================
# Linear Recurrence (Griffin-style)
# =============================================================================


@pytest.mark.slow
def test_recurrent_gemma():
    """RecurrentGemma: Griffin architecture — linear recurrence + local attention hybrid."""
    pytest.importorskip("transformers")
    from transformers import RecurrentGemmaConfig, RecurrentGemmaModel

    config = RecurrentGemmaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        lru_width=64,
        attention_window_size=16,
    )
    model = RecurrentGemmaModel(config).eval()
    input_ids = torch.randint(0, 256, (2, 16))
    show_model_graph(
        model,
        [],
        model_kwargs={"input_ids": input_ids},
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "linear-recurrence", "recurrent_gemma"),
    )
    assert validate_forward_pass(model, [], model_kwargs={"input_ids": input_ids})


# =============================================================================
# Outlooker Attention (VOLO)
# =============================================================================


@pytest.mark.slow
def test_timm_volo():
    """VOLO: outlooker attention — a distinct attention variant (not self-attention)."""
    timm = pytest.importorskip("timm")
    model = timm.create_model("volo_d1_224", pretrained=False, num_classes=10).eval()
    x = torch.rand(1, 3, 224, 224)
    show_model_graph(
        model,
        x,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "timm-models", "volo_d1"),
    )
    assert validate_forward_pass(model, x)
