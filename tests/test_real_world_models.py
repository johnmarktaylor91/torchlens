"""Tests for real-world models.

All optional-dependency imports are local (inside test functions) using
pytest.importorskip() for non-torchvision packages. Tests with missing
packages show as SKIPPED, never ERROR.

Only torch, torchvision, pytest, and torchlens are imported at the top level.
"""

from os.path import join as opj

import pytest
import torch
import torchvision

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
