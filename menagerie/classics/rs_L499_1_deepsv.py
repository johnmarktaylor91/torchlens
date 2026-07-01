# FAITHFUL PORT of CSuperlei/DeepSV @ master (Typical_Model/model.py) (original framework: Keras/TensorFlow)
#
# DeepSV (Cai et al., BMC Bioinformatics 2019) classifies candidate genomic deletions
# from BAM-derived image encodings ("deletion images") using a small 2D CNN: five
# repeated blocks of [Conv2d -> BatchNorm -> LeakyReLU], four of them followed by
# 2x2 max-pooling, then Flatten -> Dense(256, relu) -> Dropout. The real repo's
# `cnn_model()` builds this with Keras `channels_first` ordering and a
# (3, 255, 255) input (3-channel deletion image). Ported layer-for-layer: same
# filter counts (96 throughout), same kernel/pool sizes, same LeakyReLU alpha
# (0.1), same dropout rate (0.3), same BatchNorm placement (before the
# activation, over the channel axis) and the same terminal Dense(256)+Dropout
# "embedding" head (the real script does not add a final classification layer
# either -- `cnn_model()` returns the 256-d Dropout output as `model`, with
# downstream fitting scripts appending a classifier separately).
import torch
from torch import nn

MENAGERIE_ZOO = "ported-pytorch"


class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm2d -> LeakyReLU, mirroring convlution_block() in model.py."""

    def __init__(
        self,
        in_channels,
        n_filters,
        batch_normalization=True,
        kernel_size=(3, 3),
        padding=1,
        stride=(1, 1),
        alpha=0.1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, n_filters, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.bn = nn.BatchNorm2d(n_filters) if batch_normalization else None
        self.act = nn.LeakyReLU(negative_slope=alpha)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return self.act(x)


class DeepSVCNN(nn.Module):
    """Faithful port of cnn_model() from Typical_Model/model.py."""

    def __init__(
        self,
        in_channels=3,
        n_base_filters=96,
        pool_size=(2, 2),
        dropout_rate=0.3,
        batch_normalization=True,
        flatten_dim=None,
    ):
        super().__init__()
        self.conv1a = ConvBlock(in_channels, n_base_filters, batch_normalization)
        self.conv1b = ConvBlock(n_base_filters, n_base_filters, batch_normalization)
        self.pool1 = nn.MaxPool2d(pool_size, stride=pool_size, padding=0)

        self.conv2a = ConvBlock(n_base_filters, n_base_filters, batch_normalization)
        self.conv2b = ConvBlock(n_base_filters, n_base_filters, batch_normalization)
        self.pool2 = nn.MaxPool2d(pool_size)

        self.conv3a = ConvBlock(n_base_filters, n_base_filters, batch_normalization)
        self.conv3b = ConvBlock(n_base_filters, n_base_filters, batch_normalization)
        self.pool3 = nn.MaxPool2d(pool_size)

        self.conv4a = ConvBlock(n_base_filters, n_base_filters, batch_normalization)
        self.conv4b = ConvBlock(n_base_filters, n_base_filters, batch_normalization)
        self.pool4 = nn.MaxPool2d(pool_size)

        self.conv5 = ConvBlock(n_base_filters, n_base_filters, batch_normalization)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.flatten = nn.Flatten()
        # NOTE: the 256-unit Dense head's input width is shape-derived in the
        # real Keras graph (Flatten() output size depends on input
        # resolution); flatten_dim must be pre-computed for the chosen input
        # size (n_base_filters * H_out * W_out after 4 stride-2 max-pools).
        self.fc = nn.Linear(flatten_dim, 256)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.conv1a(x)
        x = self.conv1b(x)
        x = self.pool1(x)

        x = self.conv2a(x)
        x = self.conv2b(x)
        x = self.pool2(x)

        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.pool3(x)

        x = self.conv4a(x)
        x = self.conv4b(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.dropout1(x)

        x = self.flatten(x)
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout2(x)
        return x


def build_deepsv():
    # 64x64 input, 4 stride-2 max-pools -> 4x4 spatial; 8 base filters ->
    # flatten_dim = 8 * 4 * 4 = 128.
    return DeepSVCNN(in_channels=3, n_base_filters=8, flatten_dim=128)


def example_input_deepsv():
    # Real deletion images are (3, 255, 255); shrink spatial size for a tiny
    # trace while keeping the 3-channel structure and enough resolution to
    # survive four 2x2 max-pools without collapsing to zero.
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    ("DeepSV", build_deepsv, example_input_deepsv, 2019, "ported-pytorch"),
]
