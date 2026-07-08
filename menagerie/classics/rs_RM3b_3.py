# SOURCE: vendored from Advanced-Vision-and-Learning-Lab/HLTDNN @ master
# ("Histogram Layer Time Delay Neural Networks For Passive Sonar Classification",
# Ritu, Barnes, Martell, Van Dine, Peeples, WASPAA 2023). Vendored files:
#   Utils/RBFHistogramPooling.py  (HistogramLayer)
#   Utils/Generate_Spatial_Dims.py (generate_spatial_dimensions)
#   Utils/TDNN.py                  (TDNN acoustic backbone)
#   Utils/Histogram_Model.py       (HistRes: backbone + parallel histogram branch)
#   Utils/Network_functions.py     (initialize_model's TDNN+histogram factory logic,
#                                    replicated in build_acoustic_histogram below)
# The real classes are copied verbatim; only relative imports were collapsed into this
# single file, and TDNN's leftover ``pdb.set_trace()`` debug breakpoint (not part of
# the architecture -- a harness artifact left in the original forward()) was removed
# so the vendored model can run non-interactively.
from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor, nn
from torchvision import models

MENAGERIE_ZOO = "vendored-pytorch"


def generate_spatial_dimensions(n: int) -> tuple[int, int]:
    """Find a (near-)square (h, w) factor pair of ``n`` for adaptive-pool sizing.

    Verbatim from Utils/Generate_Spatial_Dims.py.
    """
    if n < 0:
        raise ValueError("Input must be a non-negative integer")

    root = math.sqrt(n)

    if (int(root + 0.5) ** 2) == n:
        return (int(root), int(root))

    items = []
    end_range = int(n**0.5) + 1
    for i in range(1, end_range):
        if n % i == 0:
            j = n // i
            items.append((j, i))

    return items[-1]


class HistogramLayer(nn.Module):
    """Learnable radial-basis-function histogram pooling layer.

    Verbatim from Utils/RBFHistogramPooling.py (2D branch used here).
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size,
        dim: int = 2,
        num_bins: int = 4,
        stride: int = 1,
        padding: int = 0,
        normalize_count: bool = True,
        normalize_bins: bool = True,
        count_include_pad: bool = False,
        ceil_mode: bool = False,
    ) -> None:
        """Initialize the learnable bin-center/bin-width convolutions and pooling."""
        super().__init__()
        self.in_channels = in_channels
        self.numBins = num_bins
        self.stride = stride
        self.kernel_size = kernel_size
        self.dim = dim
        self.padding = padding
        self.normalize_count = normalize_count
        self.normalize_bins = normalize_bins
        self.count_include_pad = count_include_pad
        self.ceil_mode = ceil_mode

        if self.dim == 2:
            self.bin_centers_conv = nn.Conv2d(
                self.in_channels,
                self.numBins * self.in_channels,
                1,
                groups=self.in_channels,
                bias=True,
            )
            self.bin_centers_conv.weight.data.fill_(1)
            self.bin_centers_conv.weight.requires_grad = False
            self.bin_widths_conv = nn.Conv2d(
                self.numBins * self.in_channels,
                self.numBins * self.in_channels,
                1,
                groups=self.numBins * self.in_channels,
                bias=False,
            )
            self.hist_pool = nn.AvgPool2d(
                self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                ceil_mode=self.ceil_mode,
                count_include_pad=self.count_include_pad,
            )
            self.centers = self.bin_centers_conv.bias
            self.widths = self.bin_widths_conv.weight
        else:
            raise RuntimeError("This vendored entry only wires up the 2D histogram branch")

    def forward(self, xx: Tensor) -> Tensor:
        """Compute a learnable localized RBF histogram over the input feature map."""
        xx = self.bin_centers_conv(xx)
        xx = self.bin_widths_conv(xx)
        xx = torch.exp(-(xx**2))

        if self.normalize_bins:
            xx = self.constrain_bins(xx)

        if self.normalize_count:
            xx = self.hist_pool(xx)
        else:
            xx = np.prod(np.asarray(self.hist_pool.kernel_size)) * self.hist_pool(xx)
        return xx

    def constrain_bins(self, xx: Tensor) -> Tensor:
        """Enforce the sum-to-one constraint across bins (2D branch)."""
        n, c, h, w = xx.size()
        xx_sum = xx.reshape(n, c // self.numBins, self.numBins, h, w).sum(2) + torch.tensor(10e-6)
        xx_sum = torch.repeat_interleave(xx_sum, self.numBins, dim=1)
        return xx / xx_sum


class TDNN(nn.Module):
    """Baseline acoustic-feature CNN backbone ("TDNN" in the original repo).

    Verbatim from Utils/TDNN.py, minus the leftover ``pdb.set_trace()`` at the top
    of ``forward`` (a debug breakpoint artifact, not part of the architecture).
    """

    def __init__(
        self,
        in_channels: int,
        stride: int = 1,
        dilation: int = 1,
        batch_norm: bool = True,
        num_class: int = 4,
        output_len: int = 1,
        drop_p: float | None = 0.1,
    ) -> None:
        """Initialize the TDNN convolution stack and classifier head."""
        super().__init__()
        self.in_channels = in_channels
        self.stride = stride
        self.dilation = dilation
        self.batch_norm = batch_norm
        self.output_len = output_len
        self.drop_p = drop_p

        self.conv1 = nn.Conv2d(
            self.in_channels, 16, kernel_size=(11, 11), padding="same", bias=True
        )
        self.conv2 = nn.Conv2d(16, 16, kernel_size=(3, 3), padding="same", bias=True)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=(3, 3), padding="same", bias=True)
        self.conv4 = nn.Conv2d(16, 4, kernel_size=(3, 3), padding="same", bias=True)
        self.conv5 = nn.Conv1d(4, 256, kernel_size=1, padding="same", bias=True)

        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 1))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 1))
        self.maxpool3 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 2))
        self.maxpool4 = nn.MaxPool2d(kernel_size=(1, 8), stride=(1, 4))

        self.nonlinearity = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.avgpool = nn.AdaptiveAvgPool1d(self.output_len)

        if drop_p is not None:
            self.dropout: nn.Module = nn.Dropout(p=self.drop_p)
        else:
            self.dropout = nn.Sequential()

        self.fc = nn.Linear(self.conv5.out_channels * self.output_len, num_class)

    def forward(self, x: Tensor) -> Tensor:
        """Classify a 2D acoustic feature map (e.g. a spectrogram).

        input: size (batch, channels, audio_feature_x, audio_feature_y)
        output: size (batch, num_class)
        """
        x = self.conv1(x)
        x = self.nonlinearity(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.nonlinearity(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.nonlinearity(x)
        x = self.maxpool3(x)

        x = self.conv4(x)
        x = self.nonlinearity(x)
        x = self.maxpool4(x)

        x = torch.flatten(x, start_dim=-2)

        x = self.conv5(x)
        x = self.sigmoid(x)
        x = self.avgpool(x).flatten(start_dim=1)

        x = self.dropout(x)

        return self.fc(x)


class HistRes(nn.Module):
    """Backbone + parallel learnable-histogram classification head.

    Verbatim from Utils/Histogram_Model.py (only the TDNN backbone branch is
    exercised here; the resnet/densenet/efficientnet/regnet branches are kept
    intact as in the real source but are not constructed by ``build_*`` below).
    """

    def __init__(
        self,
        histogram_layer: nn.Module,
        parallel: bool = True,
        model_name: str = "resnet18",
        add_bn: bool = True,
        scale: int = 5,
        pretrained: bool = True,
        TDNN_feats: int = 1,
    ) -> None:
        """Initialize the backbone, histogram branch, and pooled-feature fusion."""
        super().__init__()
        self.parallel = parallel
        self.add_bn = add_bn
        self.scale = scale
        self.model_name = model_name
        self.bn_norm = None
        self.fc = None
        self.dropout = None

        if model_name == "resnet18":
            self.backbone = models.resnet18(weights="DEFAULT" if pretrained else None)
            num_ftrs = self.backbone.fc.in_features
        elif model_name == "resnet50":
            self.backbone = models.resnet50(weights="DEFAULT" if pretrained else None)
            num_ftrs = self.backbone.fc.in_features
        elif model_name == "resnet50_wide":
            self.backbone = models.wide_resnet50_2(weights="DEFAULT" if pretrained else None)
            num_ftrs = self.backbone.fc.in_features
        elif model_name == "resnet50_next":
            self.backbone = models.resnext50_32x4d(weights="DEFAULT" if pretrained else None)
            num_ftrs = self.backbone.fc.in_features
        elif model_name == "densenet121":
            self.backbone = models.densenet121(
                weights="DEFAULT" if pretrained else None, memory_efficient=True
            )
            self.bn_norm = self.backbone.features.norm5
            self.backbone.features.norm5 = nn.Sequential()
            self.backbone.avgpool = nn.Sequential()
            num_ftrs = self.backbone.classifier.in_features
            self.fc = self.backbone.classifier
            self.backbone.classifier = torch.nn.Sequential()
        elif model_name == "efficientnet":
            self.backbone = models.efficientnet_b0(weights="DEFAULT" if pretrained else None)
            num_ftrs = self.backbone.classifier[-1].in_features
            self.fc = self.backbone.classifier[-1]
            self.backbone.classifier[-1] = torch.nn.Sequential()
        elif model_name == "regnet":
            self.backbone = models.regnet_x_400mf(weights="DEFAULT" if pretrained else None)
            num_ftrs = self.backbone.fc.in_features
        elif model_name == "TDNN":
            self.backbone = TDNN(in_channels=TDNN_feats)
            num_ftrs = self.backbone.fc.in_features
            self.dropout = self.backbone.dropout
        else:
            raise RuntimeError(f"{model_name} not implemented")

        if self.add_bn:
            if self.bn_norm is None:
                self.bn_norm = nn.BatchNorm2d(num_ftrs)

        if self.dropout is None:
            self.dropout = nn.Sequential()

        self.histogram_layer = histogram_layer

        output_size = int(num_ftrs / histogram_layer.bin_widths_conv.out_channels)
        output_size = generate_spatial_dimensions(output_size)
        histogram_layer.hist_pool = nn.AdaptiveAvgPool2d(output_size)

        if self.fc is None:
            self.fc = self.backbone.fc
            self.backbone.fc = torch.nn.Sequential()

    def forward(self, x: Tensor) -> Tensor:
        """Run the backbone, then fuse pooled backbone features with histogram features."""
        if self.model_name == "densenet121":
            x = self.backbone(x).unsqueeze(2).unsqueeze(3)
        elif self.model_name == "efficientnet":
            x = self.backbone.features(x)
        elif self.model_name == "regnet":
            x = self.backbone.stem(x)
            x = self.backbone.trunk_output(x)
        elif self.model_name == "TDNN":
            x = self.backbone.conv1(x)
            x = self.backbone.nonlinearity(x)
            x = self.backbone.maxpool1(x)

            x = self.backbone.conv2(x)
            x = self.backbone.nonlinearity(x)
            x = self.backbone.maxpool2(x)

            x = self.backbone.conv3(x)
            x = self.backbone.nonlinearity(x)
            x = self.backbone.maxpool3(x)

            x = self.backbone.conv4(x)
            x = self.backbone.nonlinearity(x)
            x = self.backbone.maxpool4(x)
        else:
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)

            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)

        if self.parallel:
            if self.add_bn:
                if self.model_name == "TDNN":
                    x_pool = torch.flatten(x, start_dim=-2)
                    x_pool = self.backbone.conv5(x_pool)
                    x_pool = self.backbone.sigmoid(x_pool)
                    x_pool = self.backbone.avgpool(x_pool)
                    x_pool = torch.flatten(self.bn_norm(x_pool.unsqueeze(-1)), start_dim=1)
                else:
                    x_pool = torch.flatten(self.bn_norm(self.backbone.avgpool(x)), start_dim=1)
            else:
                if self.model_name == "TDNN":
                    x_pool = torch.flatten(x, start_dim=-2)
                    x_pool = self.backbone.conv5(x_pool)
                    x_pool = self.backbone.sigmoid(x_pool)
                    x_pool = self.backbone.avgpool(x_pool)
                    x_pool = torch.flatten(x_pool, start_dim=1)
                else:
                    x_pool = torch.flatten(self.backbone.avgpool(x), start_dim=1)

            x_hist = torch.flatten(self.histogram_layer(x), start_dim=1)
            x_combine = torch.cat((x_pool, x_hist), dim=1)
            x_combine = self.dropout(x_combine)
            output = self.fc(x_combine)
        else:
            x = torch.flatten(self.histogram_layer(x), start_dim=1)
            x = self.dropout(x)
            output = self.fc(x)

        return output


def build_acoustic_histogram() -> HistRes:
    """Build the HistRes(TDNN) acoustic-histogram classifier at the paper's scale=5 config.

    Replicates Utils/Network_functions.py's ``initialize_model`` factory for
    ``model_name="TDNN", histogram=True`` on the DeepShip underwater-acoustic
    dataset config (out_channels['TDNN']=256, feat_map_size=4, numBins=4,
    kernel_size['TDNN']=[4, 4], stride=[2, 2], in_channels['TDNN']=4, num_classes=4).
    """
    num_feature_maps = 256
    feat_map_size = 4
    num_bins = 4
    kernel_size = [4, 4]
    stride = [2, 2]
    raw_in_channels = 4
    num_classes = 4

    hist_in_channels = int(num_feature_maps / (feat_map_size * num_bins))
    histogram_layer = HistogramLayer(
        hist_in_channels,
        kernel_size,
        dim=2,
        num_bins=num_bins,
        stride=stride,
        normalize_count=True,
        normalize_bins=True,
    )

    model = HistRes(
        histogram_layer,
        parallel=True,
        model_name="TDNN",
        add_bn=True,
        scale=5,
        pretrained=False,
        TDNN_feats=1,
    )

    reduced_dim = int((num_feature_maps / feat_map_size) / num_bins)
    if raw_in_channels == reduced_dim:
        model.histogram_layer = histogram_layer
    else:
        conv_reduce = nn.Conv2d(raw_in_channels, reduced_dim, (1, 1))
        model.histogram_layer = nn.Sequential(conv_reduce, histogram_layer)

    num_ftrs = model.fc.in_features * 2
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def example_input_acoustic_histogram() -> Tensor:
    """Return a single-channel 2D acoustic feature map (e.g. STFT) example."""
    return torch.randn(2, 1, 8, 48)


MENAGERIE_ENTRIES = [
    (
        "Acoustic Histogram (HistRes-TDNN)",
        build_acoustic_histogram,
        example_input_acoustic_histogram,
        2023,
        "RM3b-acoustic-histogram",
    ),
]
