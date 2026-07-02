# FAITHFUL PORT of cialab/DeepFocus @ master (original framework: TFLearn / TensorFlow 1.x)
# https://raw.githubusercontent.com/cialab/deepfocus/master/classificationModel3.py
# https://raw.githubusercontent.com/cialab/deepfocus/master/hyperparameterModel.py
#
# Senaras, Niazi, Sahiner, Pennell, Tozbikian, Lozanski, Gurcan 2018 (PLOS ONE)
# "DeepFocus: Detection of out-of-focus regions in whole slide digital images using deep
# learning" (doi:10.1371/journal.pone.0205387). Original repo is TFLearn/TensorFlow 1.x
# (`tflearn.layers.core`/`conv`/`normalization`) with a saved TF1 checkpoint (`ver5.*`); no
# PyTorch anywhere and TFLearn/TF1 are not reasonably installable alongside our torch>=2.1
# environment -- transcribed faithfully into torch here.
#
# `classificationModel3.createModel` builds, using the default `hyperparameterModel` values
# (CNN1: 5x5x32, CNN2: 3x3x32, CNN3: 3x3x64 + maxpool, CNN4: 3x3x128 + maxpool (applied TWICE
# in the real code -- see the back-to-back `conv_2d`+`max_pool_2d` block under `if
# parameters.CNN4Size > 0:`), FullyConn1: 128, FullyConn2: 64), a CNN classifier over 64x64x3
# input patches producing a 2-way (in-focus / out-of-focus) softmax. Every conv/FC layer, its
# activation ('relu'), its batch-normalization placement (immediately after each conv/FC, before
# any pooling/dropout), the two max-pool-2 layers gating on CNN3/CNN4, and the 0.2 dropout after
# each of the two FC layers are transcribed 1:1 from the real `createModel` function. The
# image-augmentation (`ImageAugmentation`/`ImagePreprocessing`) and `tflearn.regression` training
# wrapper are training-time-only scaffolding (data augmentation + optimizer config), not part of
# the network's forward computation graph, and are intentionally omitted -- the traceable "model"
# is the `network`/`g` computation graph from input to the final softmax classifier layer.

import torch
import torch.nn as nn


class DeepFocusHyperparams:
    """Transcribed verbatim from hyperparameterModel.py's default __init__ values."""

    def __init__(self):
        self.CNN1Size = 5
        self.CNN1FeatureSize = 32
        self.CNN2Size = 3
        self.CNN2FeatureSize = 32
        self.CNN3Size = 3
        self.CNN3FeatureSize = 64
        self.CNN4Size = 3
        self.CNN4FeatureSize = 128
        self.FullyConn1Size = 128
        self.FullyConn2Size = 64


class DeepFocus(nn.Module):
    """Real DeepFocus CNN classifier from classificationModel3.py:createModel, transcribed
    faithfully into torch (same conv/pool/batchnorm/FC/dropout stack and ordering)."""

    def __init__(self, params: DeepFocusHyperparams | None = None, num_classes: int = 2):
        super().__init__()
        if params is None:
            params = DeepFocusHyperparams()
        self.params = params

        in_ch = 3
        layers: list[nn.Module] = []

        # 1. CNN1 (always present): conv 'same' padding, relu, batchnorm
        layers.append(nn.Conv2d(in_ch, params.CNN1FeatureSize, params.CNN1Size, padding="same"))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm2d(params.CNN1FeatureSize))
        in_ch = params.CNN1FeatureSize

        # 2. CNN2 (gated): conv, relu, batchnorm -- no pool (real code: CNN2 has no max_pool_2d)
        if params.CNN2Size > 0:
            layers.append(nn.Conv2d(in_ch, params.CNN2FeatureSize, params.CNN2Size, padding="same"))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(params.CNN2FeatureSize))
            in_ch = params.CNN2FeatureSize

        # 3. CNN3 (gated): conv, relu, batchnorm, then max_pool_2d(2)
        if params.CNN3Size > 0:
            layers.append(nn.Conv2d(in_ch, params.CNN3FeatureSize, params.CNN3Size, padding="same"))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(params.CNN3FeatureSize))
            layers.append(nn.MaxPool2d(2))
            in_ch = params.CNN3FeatureSize

        # 4. CNN4 (gated): TWO conv+relu+batchnorm+maxpool blocks back-to-back, exactly as in
        #    the real `if parameters.CNN4Size > 0:` branch (the second conv reuses the same
        #    CNN4FeatureSize/CNN4Size hyperparameters, in->out channels both CNN4FeatureSize).
        if params.CNN4Size > 0:
            layers.append(nn.Conv2d(in_ch, params.CNN4FeatureSize, params.CNN4Size, padding="same"))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(params.CNN4FeatureSize))
            layers.append(nn.MaxPool2d(2))
            in_ch = params.CNN4FeatureSize

            layers.append(nn.Conv2d(in_ch, params.CNN4FeatureSize, params.CNN4Size, padding="same"))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(params.CNN4FeatureSize))
            layers.append(nn.MaxPool2d(2))
            in_ch = params.CNN4FeatureSize

        self.conv_stack = nn.Sequential(*layers)
        self._conv_out_ch = in_ch

        # FullyConn1 / FullyConn2 (both gated on FullyConn2Size in the real code -- transcribed
        # verbatim, including the real repo's `if parameters.FullyConn2Size > 0:` guard reused
        # for the FIRST fully-connected block too):
        self.fc1: nn.Module | None = None
        self.fc2: nn.Module | None = None
        if params.FullyConn2Size > 0:
            # lazily sized on first forward since flattened spatial size depends on input res
            self._fc1_out = params.FullyConn1Size
            self._fc2_out = params.FullyConn2Size

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.out = nn.LazyLinear(num_classes)
        self._built_fc = False

    def _build_fc(self, flat_dim: int, device: torch.device) -> None:
        params = self.params
        in_dim = flat_dim
        fc1_out = params.FullyConn1Size
        fc2_out = params.FullyConn2Size
        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, fc1_out), nn.ReLU(), nn.BatchNorm1d(fc1_out), nn.Dropout(0.2)
        ).to(device)
        self.fc2 = nn.Sequential(
            nn.Linear(fc1_out, fc2_out), nn.ReLU(), nn.BatchNorm1d(fc2_out), nn.Dropout(0.2)
        ).to(device)
        self.fc1.train(self.training)
        self.fc2.train(self.training)
        self._built_fc = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        network = self.conv_stack(x)
        flat = network.flatten(1)

        if self.params.FullyConn2Size > 0:
            if not self._built_fc:
                self._build_fc(flat.shape[1], flat.device)
            flat = self.fc1(flat)
            flat = self.fc2(flat)

        logits = self.out(flat)
        return torch.softmax(logits, dim=-1)


def build_deepfocus():
    torch.manual_seed(0)
    model = DeepFocus()
    model.eval()
    # Lazy FC blocks (self.fc1/self.fc2) need one warmup pass to materialize shapes before
    # tracing; running the warmup in eval mode means BatchNorm1d uses running stats and
    # tolerates batch_size=1.
    with torch.no_grad():
        model(torch.zeros(1, 3, 64, 64))
    return model


def example_input_deepfocus():
    torch.manual_seed(0)
    return torch.rand(2, 3, 64, 64)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("DeepFocus", "build_deepfocus", "example_input_deepfocus", 2018, "ported"),
]
