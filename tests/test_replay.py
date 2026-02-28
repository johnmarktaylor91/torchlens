"""Tests for layer-by-layer replay (fast forward computation).

Validates that replaying a logged forward pass through individual layers
reproduces the same output as the original model.
"""

import pytest
import torch
import torchvision

import torchlens as tl


# ---------------------------------------------------------------------------
# Replay helpers (extracted from debug_torchlens.py / test_models.py)
# ---------------------------------------------------------------------------


def prepare_replay_graph(net_list):
    """Prepare all layers for fast replay and build label2idx mapping.

    Args:
        net_list: List of TensorLogEntry from model_history.layer_list

    Returns:
        label2idx: Dict mapping layer_label to index
    """
    label2idx = {}
    for i, layer in enumerate(net_list):
        label2idx[layer.layer_label] = i
        if layer.func_applied_name != "none":
            layer.prepare_replay()
    return label2idx


def model_log_forward_fast(net_list, x, label2idx=None):
    """Optimized layer-by-layer forward using replay_fast.

    Args:
        net_list: List of TensorLogEntry from model_history.layer_list
        x: Input tensor
        label2idx: Optional precomputed label to index mapping.

    Returns:
        Output tensor from the forward pass.
    """
    if label2idx is None:
        label2idx = {layer.layer_label: i for i, layer in enumerate(net_list)}

    for layer in net_list:
        func_name = layer.func_applied_name
        layer_type = layer.layer_type

        if func_name == "none":
            layer.tensor_contents = (
                x
                if layer_type == "input"
                else (
                    layer.tensor_contents if layer_type == "buffer" else None
                )
            )
            continue

        x_in, buffer_in = [], []
        op_num = layer.operation_num
        for plabel in layer.parent_layers:
            p = net_list[label2idx[plabel]]
            if p.layer_type == "buffer":
                buffer_in.append(p.tensor_contents)
            else:
                tc = p.tensor_contents
                x_in.append(
                    x if tc is None and p.operation_num == op_num - 1 else tc
                )
                if (
                    p.layer_type not in ("input", "buffer")
                    and label2idx[p.child_layers[-1]]
                    <= label2idx[layer.layer_label]
                ):
                    p.tensor_contents = None

        x = layer.replay_fast(x_in, buffer_in)
        layer.tensor_contents = (
            None
            if layer.child_layers
            and net_list[label2idx[layer.child_layers[-1]]].operation_num
            <= op_num + 1
            else x
        )
    return x


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _replay_and_compare(model, x, atol=1e-5):
    """Log, replay, and assert output matches the original model."""
    model.eval()
    with torch.no_grad():
        expected = model(x)

    model_history = tl.log_forward_pass(
        model, x, vis_opt="none", save_function_args=True
    )
    layer_list = model_history.layer_list
    label2idx = prepare_replay_graph(layer_list)
    replayed = model_log_forward_fast(layer_list, x, label2idx)

    assert torch.allclose(replayed, expected, atol=atol), (
        f"Replay mismatch: max diff = {(replayed - expected).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# Tests — TorchVision classification models
# ---------------------------------------------------------------------------


class TestReplayTorchVisionClassification:
    """Replay validation for common torchvision classifiers."""

    @pytest.fixture
    def img_input(self):
        return torch.rand(1, 3, 224, 224)

    def test_vgg11_replay(self, img_input):
        model = torchvision.models.vgg11(weights=None)
        _replay_and_compare(model, img_input)

    def test_resnet18_replay(self, img_input):
        model = torchvision.models.resnet18(weights=None)
        _replay_and_compare(model, img_input)

    def test_resnet50_replay(self, img_input):
        model = torchvision.models.resnet50(weights=None)
        _replay_and_compare(model, img_input)

    def test_resnet101_replay(self, img_input):
        model = torchvision.models.resnet101(weights=None)
        _replay_and_compare(model, img_input)

    def test_vit_b_32_replay(self, img_input):
        model = torchvision.models.vit_b_32(weights=None)
        _replay_and_compare(model, img_input)

    def test_vit_b_16_replay(self, img_input):
        model = torchvision.models.vit_b_16(weights=None)
        _replay_and_compare(model, img_input)

    def test_densenet121_replay(self, img_input):
        model = torchvision.models.densenet121(weights=None)
        _replay_and_compare(model, img_input)

    def test_mobilenet_v3_small_replay(self, img_input):
        model = torchvision.models.mobilenet_v3_small(weights=None)
        _replay_and_compare(model, img_input)

    def test_swin_t_replay(self, img_input):
        model = torchvision.models.swin_t(weights=None)
        _replay_and_compare(model, img_input)
