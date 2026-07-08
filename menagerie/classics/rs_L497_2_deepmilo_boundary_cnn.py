# FAITHFUL PORT of tuantrieu/DeepMILO @ master (source/predict_boundary_sep_cnn.py)
# (original framework: Keras (standalone `keras`, pre-Keras-3 `keras.layers.advanced_activations`))
#
# DeepMILO (Trieu, Cheng & Bui, Genome Biology 2020, "DeepMILO: a deep learning
# approach to predict the impact of non-coding sequence variants on 3D chromatin
# structure") predicts CTCF/cohesin insulator-loop boundaries from a one-hot
# DNA window. The "separate CNN" boundary model (source/predict_boundary_sep_cnn.py)
# is a standalone architecture (not stitched from other saved .h5 sub-models,
# unlike the sep_cnn+lstm composite), so it is the faithful-port target: an
# initial Conv2D+BN+LeakyReLU+Dropout stem over a (segment_size, 5, 1)
# one-hot-with-N sequence window, fanned out into THREE parallel dilated-conv
# branches (dilation_rate 1/3/7, each Conv2D->BN->LeakyReLU->MaxPool->Dropout->
# Flatten), concatenated, then a 2-layer Dense+BN+LeakyReLU+Dropout head down
# to a single sigmoid boundary-probability output. Every layer/hyperparameter
# (kernel sizes, dilation rates, LeakyReLU slope, dropout rate) below matches
# the real Keras code; only the Keras -> torch layer translation and the
# CLI/training/data-generator glue are new.
import torch
from torch import nn

MENAGERIE_ZOO = "ported-pytorch"

SEGMENT_SIZE = 4000  # real: 4000bp boundary window
N_LETTERS = 5  # real: len('ACGTN')
LEAKY = 0.2
DROPOUT = 0.15


class DilatedConvBranch(nn.Module):
    """Faithful port of `get_dilated_convnet(input, dilation_rate)`:
    Conv2D(512, (5,1), dilation_rate) -> BN -> LeakyReLU -> MaxPool((L-16,1)) -> Dropout -> Flatten.
    """

    def __init__(self, in_channels, seq_len, dilation_rate):
        super().__init__()
        # Keras Conv2D 'same' padding along the height (sequence) axis with a
        # dilated (5,1) kernel: pad = dilation_rate * (kernel_size - 1) // 2 per side.
        pad_h = dilation_rate * (5 - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            512,
            kernel_size=(5, 1),
            stride=(1, 1),
            dilation=(dilation_rate, 1),
            padding=(pad_h, 0),
        )
        self.bn = nn.BatchNorm2d(512)
        self.act = nn.LeakyReLU(LEAKY)
        # Keras: MaxPooling2D(pool_size=(segment_size - 16, 1), strides=(segment_size - 16, 1), padding='valid')
        pool_h = seq_len - 16
        self.pool = nn.MaxPool2d(kernel_size=(pool_h, 1), stride=(pool_h, 1))
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.drop(x)
        return torch.flatten(x, start_dim=1)


class DeepMILOBoundaryCNN(nn.Module):
    """Faithful port of the sep_cnn boundary-prediction Keras graph."""

    def __init__(self, segment_size=SEGMENT_SIZE, n_letters=N_LETTERS):
        super().__init__()
        self.segment_size = segment_size
        # Keras: Conv2D(256, (17,5), strides=(1,1), padding='valid') over
        # input shape (segment_size, n_letters, 1) [channels-last NHWC].
        # In NCHW torch terms: in_channels=1, kernel=(17, n_letters), valid
        # padding, producing height segment_size-16 and width 1.
        self.stem_conv = nn.Conv2d(1, 256, kernel_size=(17, n_letters), stride=(1, 1), padding=0)
        self.stem_bn = nn.BatchNorm2d(256)
        self.stem_act = nn.LeakyReLU(LEAKY)
        self.stem_drop = nn.Dropout(DROPOUT)

        stem_len = segment_size - 16  # 'valid' conv shrinks height by kernel_h-1
        self.branch1 = DilatedConvBranch(256, stem_len, dilation_rate=1)
        self.branch2 = DilatedConvBranch(256, stem_len, dilation_rate=3)
        self.branch3 = DilatedConvBranch(256, stem_len, dilation_rate=7)

        self.fc1 = nn.Linear(512 * 3, 256)
        self.fc1_bn = nn.BatchNorm1d(256)
        self.fc1_act = nn.LeakyReLU(LEAKY)
        self.fc1_drop = nn.Dropout(DROPOUT)

        self.fc2 = nn.Linear(256, 128)
        self.fc2_bn = nn.BatchNorm1d(128)
        self.fc2_act = nn.LeakyReLU(LEAKY)
        self.fc2_drop = nn.Dropout(DROPOUT)

        self.fc_out = nn.Linear(128, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, seq_input):
        # seq_input: (N, 1, segment_size, n_letters) NCHW one-hot-with-N window.
        x = self.stem_conv(seq_input)
        x = self.stem_bn(x)
        x = self.stem_act(x)
        x = self.stem_drop(x)

        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x = torch.cat([x1, x2, x3], dim=1)

        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = self.fc1_act(x)
        x = self.fc1_drop(x)

        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = self.fc2_act(x)
        x = self.fc2_drop(x)

        x = self.fc_out(x)
        return self.out_act(x)


# ---------------------------------------------------------------------------
# menagerie staging entry point
# ---------------------------------------------------------------------------
# Use a small synthetic segment_size for trace speed (real model uses 4000bp);
# the architecture (stem + 3 dilated branches + dense head) is identical.
_TINY_SEGMENT = 64


def build_deepmilo_boundary_cnn():
    return DeepMILOBoundaryCNN(segment_size=_TINY_SEGMENT, n_letters=N_LETTERS)


def example_input_deepmilo_boundary_cnn():
    return torch.randn(2, 1, _TINY_SEGMENT, N_LETTERS)


MENAGERIE_ENTRIES = [
    (
        "DeepMILO-BoundaryCNN",
        build_deepmilo_boundary_cnn,
        example_input_deepmilo_boundary_cnn,
        2020,
        "SOURCE_AVAILABLE",
    ),
]
