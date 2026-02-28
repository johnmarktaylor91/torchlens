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

from conftest import VIS_OUTPUT_DIR

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
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-main", "vgg16"),
    )


def test_vit(default_input1):
    model = torchvision.models.vit_l_16()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model,
        default_input1,
        save_only=True,
        vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchvision-main", "vit_l_16"),
    )


# =============================================================================
# CORNet Models (requires cornet package)
# =============================================================================


def test_cornet_z(default_input1):
    cornet = pytest.importorskip("cornet")
    model = cornet.cornet_z()
    assert validate_saved_activations(model, default_input1)
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


# =============================================================================
# TIMM Models (requires timm)
# =============================================================================


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
    assert validate_saved_activations(model, default_input1)


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
        vis_outpath=opj(VIS_OUTPUT_DIR, "timm", "ecaresnet101d"),
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
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchaudio", "audio_conv_tasnet_base"),
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
        vis_outpath=opj(VIS_OUTPUT_DIR, "torchaudio", "audio_wav2letter"),
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
        vis_outpath=opj(VIS_OUTPUT_DIR, "language-models", "language_lstm_unrolled"),
    )
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "language-models", "language_lstm_rolled"),
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
        vis_outpath=opj(VIS_OUTPUT_DIR, "language-models", "language_rnn_unrolled"),
    )
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "language-models", "language_rnn_rolled"),
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
        vis_outpath=opj(VIS_OUTPUT_DIR, "language-models", "gpt2"),
    )
    assert validate_saved_activations(model, [], model_inputs)


# =============================================================================
# Multimodal / Special Models
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
        vis_outpath=opj(VIS_OUTPUT_DIR, "text-to-speech", "styletts_text_encoder"),
    )
    assert validate_saved_activations(model, model_inputs, random_seed=1)


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
        vis_outpath=opj(VIS_OUTPUT_DIR, "lightning", "one-hot-autoencoder"),
    )
    assert validate_saved_activations(model, model_inputs)
