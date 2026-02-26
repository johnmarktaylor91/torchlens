"""Slow/complex real-world model tests.

These tests exercise models that are slower to run (detection, video, optical
flow, audio, NLP, generative, graph-neural-network, quantum-ML, etc.) or
require heavyweight optional dependencies.  They are separated from
test_real_world_models.py so the fast feedforward / classification /
segmentation / quantization / TIMM tests can be iterated on quickly without
waiting for the full suite.

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
    model_input = torch.randn(16, 3, 16, 224, 224)
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
        vis_outpath=opj(
            "visualization_outputs", "torchvision-opticflow", "opticflow_raftsmall_unrolled"
        ),
    )
    show_model_graph(
        model,
        model_input,
        save_only=True,
        vis_opt="rolled",
        vis_outpath=opj(
            "visualization_outputs", "torchvision-opticflow", "opticflow_raftsmall_rolled"
        ),
    )
    assert validate_saved_activations(model, model_input)


@pytest.mark.xfail
def test_opticflow_raftlarge():
    model = torchvision.models.optical_flow.raft_large()
    model_input = [torch.rand(6, 3, 224, 224), torch.rand(6, 3, 224, 224)]
    show_model_graph(
        model,
        model_input,
        save_only=True,
        random_seed=1,
        vis_opt="unrolled",
        vis_outpath=opj(
            "visualization_outputs", "torchvision-opticflow", "opticflow_raftlarge_unrolled"
        ),
    )
    show_model_graph(
        model,
        model_input,
        save_only=True,
        random_seed=1,
        vis_opt="rolled",
        vis_outpath=opj(
            "visualization_outputs", "torchvision-opticflow", "opticflow_raftlarge_rolled"
        ),
    )
    assert validate_saved_activations(model, model_input, random_seed=1)


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
