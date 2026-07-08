# SOURCE: vendored from Advanced-Vision-and-Learning-Lab/HLTDNN @ master (b43dae3)
# ("Histogram Layer Time Delay Neural Networks For Passive Sonar Classification";
#  same Histogram-Layer family/lineage as GatorSense/Histogram_Layer, applied to
#  acoustic 2-D feature maps -- the "2-D histogram of acoustic features for scene
#  classification" family). Files combined: Utils/RBFHistogramPooling.py,
#  Utils/TDNN.py, Utils/Generate_Spatial_Dims.py, Utils/Histogram_Model.py.
# Minimal fixes applied: dropped a leftover `pdb.set_trace()` debug breakpoint
# (and its now-unused `import pdb`) from TDNN.forward so the real model actually
# runs; inlined the local `Utils.*` imports into this single file.

import math

import torch
import torch.nn as nn
from torchvision import models


def generate_spatial_dimensions(n):
    """
    Find a (near-)square (rows, cols) factor pair of ``n`` for reshaping a
    flat histogram-pooled feature count back into a 2-D spatial grid.

    Parameters
    ----------
    n : int
        Desired value for spatial resolution (rows * cols).

    Returns
    -------
    tuple[int, int]
        (x dimension, y dimension).
    """

    if n < 0:
        raise ValueError("Input must be a non-negative integer")

    root = math.sqrt(n)

    if (int(root + 0.5) ** 2) == n:
        return (int(root), int(root))
    else:
        items = []
        end_range = int(n**0.5) + 1
        for i in range(1, end_range):
            if n % i == 0:
                j = n // i
                items.append((j, i))

        return items[-1]


class HistogramLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        kernel_size,
        dim=2,
        num_bins=4,
        stride=1,
        padding=0,
        normalize_count=True,
        normalize_bins=True,
        count_include_pad=False,
        ceil_mode=False,
    ):
        # inherit nn.module
        super().__init__()

        # define layer properties
        # histogram bin data
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

        # For each data type, apply two 1x1 convolutions, 1) to learn bin center (bias)
        # and 2) to learn bin width
        # Time series/ signal Data
        if self.dim == 1:
            self.bin_centers_conv = nn.Conv1d(
                self.in_channels,
                self.numBins * self.in_channels,
                1,
                groups=self.in_channels,
                bias=True,
            )
            self.bin_centers_conv.weight.data.fill_(1)
            self.bin_centers_conv.weight.requires_grad = False
            self.bin_widths_conv = nn.Conv1d(
                self.numBins * self.in_channels,
                self.numBins * self.in_channels,
                1,
                groups=self.numBins * self.in_channels,
                bias=False,
            )
            self.hist_pool = nn.AvgPool1d(
                self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                ceil_mode=self.ceil_mode,
                count_include_pad=self.count_include_pad,
            )
            self.centers = self.bin_centers_conv.bias
            self.widths = self.bin_widths_conv.weight

        # Image Data
        elif self.dim == 2:
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

        # Spatial/Temporal or Volumetric Data
        elif self.dim == 3:
            self.bin_centers_conv = nn.Conv3d(
                self.in_channels,
                self.numBins * self.in_channels,
                1,
                groups=self.in_channels,
                bias=True,
            )
            self.bin_centers_conv.weight.data.fill_(1)
            self.bin_centers_conv.weight.requires_grad = False
            self.bin_widths_conv = nn.Conv3d(
                self.numBins * self.in_channels,
                self.numBins * self.in_channels,
                1,
                groups=self.numBins * self.in_channels,
                bias=False,
            )
            self.hist_pool = nn.AvgPool3d(
                self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                ceil_mode=self.ceil_mode,
                count_include_pad=self.count_include_pad,
            )
            self.centers = self.bin_centers_conv.bias
            self.widths = self.bin_widths_conv.weight

        else:
            raise RuntimeError("Invalid dimension for histogram layer")

    def forward(self, xx):
        # xx is the input and is a torch.tensor
        # each element of output is the frequency for the bin for that window

        # Pass through first convolution to learn bin centers
        xx = self.bin_centers_conv(xx)

        # Pass through second convolution to learn bin widths
        xx = self.bin_widths_conv(xx)

        # Pass through radial basis function
        xx = torch.exp(-(xx**2))

        # Enforce sum to one constraint
        # Add small positive constant in case sum is zero
        if self.normalize_bins:
            xx = self.constrain_bins(xx)

        # Get localized histogram output, if normalize, average count
        if self.normalize_count:
            xx = self.hist_pool(xx)
        else:
            import numpy as np

            xx = np.prod(np.asarray(self.hist_pool.kernel_size)) * self.hist_pool(xx)

        return xx

    def constrain_bins(self, xx):
        # Enforce sum to one constraint across bins
        # Time series/ signal Data
        if self.dim == 1:
            n, c, size_l = xx.size()
            xx_sum = xx.reshape(n, c // self.numBins, self.numBins, size_l).sum(2) + torch.tensor(
                10e-6
            )
            xx_sum = torch.repeat_interleave(xx_sum, self.numBins, dim=1)
            xx = xx / xx_sum

        # Image Data
        elif self.dim == 2:
            n, c, h, w = xx.size()
            xx_sum = xx.reshape(n, c // self.numBins, self.numBins, h, w).sum(2) + torch.tensor(
                10e-6
            )
            xx_sum = torch.repeat_interleave(xx_sum, self.numBins, dim=1)
            xx = xx / xx_sum

        # Spatial/Temporal or Volumetric Data
        elif self.dim == 3:
            n, c, d, h, w = xx.size()
            xx_sum = xx.reshape(n, c // self.numBins, self.numBins, d, h, w).sum(2) + torch.tensor(
                10e-6
            )
            xx_sum = torch.repeat_interleave(xx_sum, self.numBins, dim=1)
            xx = xx / xx_sum

        else:
            raise RuntimeError("Invalid dimension for histogram layer")

        return xx


class TDNN(nn.Module):
    """Baseline TDNN model (real acoustic-feature-map backbone used by HistRes)."""

    def __init__(
        self,
        in_channels,
        stride=1,
        dilation=1,
        batch_norm=True,
        num_class=4,
        output_len=1,
        drop_p=0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.stride = stride
        self.dilation = dilation
        self.batch_norm = batch_norm
        self.output_len = output_len
        self.drop_p = drop_p

        # Define convolution layers
        self.conv1 = nn.Conv2d(
            self.in_channels, 16, kernel_size=(11, 11), padding="same", bias=True
        )
        self.conv2 = nn.Conv2d(16, 16, kernel_size=(3, 3), padding="same", bias=True)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=(3, 3), padding="same", bias=True)
        self.conv4 = nn.Conv2d(16, 4, kernel_size=(3, 3), padding="same", bias=True)
        self.conv5 = nn.Conv1d(4, 256, kernel_size=(1), padding="same", bias=True)

        # Define max pooling layers
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 1))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 1))
        self.maxpool3 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 2))
        self.maxpool4 = nn.MaxPool2d(kernel_size=(1, 8), stride=(1, 4))

        # Define nonlinearity
        self.nonlinearity = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # Define average pooling layer for desired length of signal
        self.avgpool = nn.AdaptiveAvgPool1d(self.output_len)

        # Add dropout if needed
        if drop_p is not None:
            self.dropout = nn.Dropout(p=self.drop_p)
        else:
            self.dropout = nn.Sequential()

        # Define classifier (fully connected layer)
        # Do not apply sigmoid, cross-entropy takes raw logits
        self.fc = nn.Linear(self.conv5.out_channels * self.output_len, num_class)

    def forward(self, x):
        """
        input: size (batch, channels, audio_feature_x, audio_feature_y)
        output: size (batch, num_class)
        """

        # Pass through feature extraction layers
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

        # Reshape to be N x C x (MxN)
        x = torch.flatten(x, start_dim=-2)

        # Apply last convolution filter, sigmoid, and average pool to desired length
        x = self.conv5(x)
        x = self.sigmoid(x)
        x = self.avgpool(x).flatten(start_dim=1)

        # Add dropout
        x = self.dropout(x)

        # Get classifier outputs for classes
        x = self.fc(x)

        return x


class HistRes(nn.Module):
    def __init__(
        self,
        histogram_layer,
        parallel=True,
        model_name="resnet18",
        add_bn=True,
        scale=5,
        pretrained=True,
        TDNN_feats=1,
    ):
        # inherit nn.module
        super().__init__()
        self.parallel = parallel
        self.add_bn = add_bn
        self.scale = scale
        self.model_name = model_name
        self.bn_norm = None
        self.fc = None
        self.dropout = None

        # Default to use resnet18, otherwise use Resnet50
        # Defines feature extraction backbone model and redefines linear layer
        if model_name == "resnet18":
            self.backbone = models.resnet18(weights=None if not pretrained else "DEFAULT")
            num_ftrs = self.backbone.fc.in_features

        elif model_name == "resnet50":
            self.backbone = models.resnet50(weights=None if not pretrained else "DEFAULT")
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
            else:
                pass

        # Add dropout if needed for TDNN models only
        if self.dropout is None:
            self.dropout = nn.Sequential()

        # Define histogram layer and fc
        self.histogram_layer = histogram_layer

        # Change histogram layer pooling to adapt to feature constraint:
        # number of histogram layer features = number of convolutional features
        output_size = int(num_ftrs / histogram_layer.bin_widths_conv.out_channels)
        output_size = generate_spatial_dimensions(output_size)
        histogram_layer.hist_pool = nn.AdaptiveAvgPool2d(output_size)

        if self.fc is None:
            self.fc = self.backbone.fc
            self.backbone.fc = torch.nn.Sequential()

    def forward(self, x):
        if self.model_name == "TDNN":
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

        # All ResNet models
        else:
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)

            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)

        # Pass through histogram layer and pooling layer
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


def build_acoustic_histogram_tdnn() -> HistRes:
    """
    Build a tiny "Acoustic Histogram" model: a learnable 2-D RBF histogram
    pooling layer running in parallel with a TDNN acoustic backbone (the real,
    unmodified ``HistRes`` + ``TDNN`` + ``HistogramLayer`` classes), matching
    the real `initialize_model(...)` wiring from
    ``Utils/Network_functions.py`` for the histogram+TDNN configuration.

    Returns
    -------
    HistRes
        Tiny HistRes(model_name="TDNN") instance with fc rewired for the
        concatenated pool+histogram feature vector, as the real repo's
        `initialize_model` does.
    """

    num_classes = 4
    # TDNN's conv4 emits 4 channels -> histogram_layer in_channels must match.
    histogram_layer = HistogramLayer(
        in_channels=4,
        kernel_size=2,
        dim=2,
        num_bins=4,
        stride=1,
        padding=0,
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

    # Real repo's `initialize_model` rewires fc to the doubled (parallel) feature width.
    num_ftrs = model.fc.in_features * 2
    model.fc = nn.Linear(num_ftrs, num_classes)
    # eval() so BatchNorm2d tolerates a single-sample batch during tracing.
    return model.eval()


def example_input_acoustic_histogram_tdnn() -> torch.Tensor:
    """
    Create an example acoustic feature-map input (matching the real repo's
    documented "MFCC that is 16 x 48 (TDNN models)" feature shape).

    Returns
    -------
    torch.Tensor
        Example input tensor with shape ``(1, 1, 16, 48)`` (batch, channel,
        frequency, time).
    """

    return torch.randn(1, 1, 16, 48)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "Acoustic Histogram (Histogram Layer TDNN)",
        "build_acoustic_histogram_tdnn",
        "example_input_acoustic_histogram_tdnn",
        "2023",
        "RM4b_2",
    )
]
