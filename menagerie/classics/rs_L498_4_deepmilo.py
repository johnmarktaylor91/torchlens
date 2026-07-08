# FAITHFUL PORT of tuantrieu/DeepMILO @ master (original framework: Keras 1.x/TF1.x, via
# source/predict_boundary_sep_cnn.py)
#
# DeepMILO boundary-sequence CNN: predicts CTCF/cohesin chromatin-loop insulator boundaries
# from one-hot DNA sequence (Trieu, Martinez-Fundichely & Khurana, PLOS Comp Bio 2020). A
# stem Conv2D(256, kernel=(17,5)) -> BatchNorm -> LeakyReLU -> Dropout collapses the
# 5-letter (ACGTN) one-hot axis, feeding THREE parallel dilated-Conv2D branches
# (`get_dilated_convnet`, dilation_rate in {1, 3, 7}; each: Conv2D(512, kernel=(5,1),
# padding=same, dilated) -> BatchNorm -> LeakyReLU -> MaxPool2D(collapsing the full
# remaining sequence length) -> Dropout -> Flatten), concatenated and read out through
# Dense(256)->BN->LeakyReLU->Dropout -> Dense(128)->BN->LeakyReLU->Dropout -> Dense(1)->sigmoid.
# The original repo ships only in early-Keras submodule paths
# (`keras.layers.advanced_activations.LeakyReLU`, `keras.utils.multi_gpu_model`) tied to a
# TF1.x-era Keras that cannot be installed cleanly alongside the modern torch/tf stack in this
# environment (and the sibling data_generator.py/.mat data pipeline is dataset-specific, not
# needed for the architecture itself), so this is a faithful architectural transcription of
# the `get_dilated_convnet` helper + the full model-assembly block in
# `predict_boundary_sep_cnn.py` into base-env torch: every layer (dilated Conv2D, BatchNorm,
# LeakyReLU(0.2), the 3-branch concat, the two dense+BN+LeakyReLU+dropout blocks, the final
# sigmoid head) is preserved. Keras's channels-last NHWC convention is handled by permuting to
# NCHW for torch's Conv2d/BatchNorm2d calls.
import torch
from torch import nn

MENAGERIE_ZOO = "ported-pytorch"

LEAKY = 0.2
DROPOUT = 0.15


class DilatedConvBranch(nn.Module):
    """Port of `get_dilated_convnet(input, dilation_rate)`."""

    def __init__(self, in_channels, seq_len_after_stem, dilation_rate):
        super().__init__()
        # Conv2D(filters=512, kernel_size=(5,1), strides=(1,1), dilation_rate=d, padding='same')
        pad_h = (5 - 1) * dilation_rate // 2
        self.conv = nn.Conv2d(
            in_channels,
            512,
            kernel_size=(5, 1),
            stride=(1, 1),
            dilation=(dilation_rate, 1),
            padding=(pad_h, 0),
        )
        self.bn = nn.BatchNorm2d(512)
        self.leaky_relu = nn.LeakyReLU(LEAKY)
        pool_h = seq_len_after_stem - 16
        self.pool = nn.MaxPool2d(kernel_size=(pool_h, 1), stride=(pool_h, 1))
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leaky_relu(x)
        x = self.pool(x)
        x = self.drop(x)
        return torch.flatten(x, start_dim=1)


class DeepMILOBoundaryCNN(nn.Module):
    """Faithful port of DeepMILO's `predict_boundary_sep_cnn.py` model-assembly block."""

    def __init__(self, segment_size, nbr_feature=5):
        super().__init__()
        self.segment_size = segment_size

        # x = Conv2D(filters=256, kernel_size=(17,5), padding='valid')(seq_input)
        self.stem_conv = nn.Conv2d(1, 256, kernel_size=(17, nbr_feature), stride=(1, 1), padding=0)
        self.stem_bn = nn.BatchNorm2d(256)
        self.stem_leaky = nn.LeakyReLU(LEAKY)
        self.stem_drop = nn.Dropout(DROPOUT)

        seq_len_after_stem = segment_size - 17 + 1  # 'valid' conv along the sequence axis

        self.branch1 = DilatedConvBranch(256, seq_len_after_stem, dilation_rate=1)
        self.branch2 = DilatedConvBranch(256, seq_len_after_stem, dilation_rate=3)
        self.branch3 = DilatedConvBranch(256, seq_len_after_stem, dilation_rate=7)

        merged_dim = 512 * 3

        self.dense1 = nn.Linear(merged_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dense2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dense3 = nn.Linear(128, 1)
        self.leaky_relu = nn.LeakyReLU(LEAKY)
        self.dropout = nn.Dropout(DROPOUT)
        self.sigmoid = nn.Sigmoid()

    def forward(self, seq_input):
        # seq_input: (batch, segment_size, nbr_feature, 1) Keras NHWC -> torch NCHW.
        x = seq_input.permute(0, 3, 1, 2)

        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = self.stem_leaky(x)
        x = self.stem_drop(x)

        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        merged = torch.cat([x1, x2, x3], dim=1)

        h = self.dense1(merged)
        h = self.bn1(h)
        h = self.leaky_relu(h)
        h = self.dropout(h)

        h = self.dense2(h)
        h = self.bn2(h)
        h = self.leaky_relu(h)
        h = self.dropout(h)

        h = self.dense3(h)
        return self.sigmoid(h)


def build_deepmilo():
    # Original segment_size=4000 (chromatin-boundary sequence length); shrunk to 64 for a
    # fast trace-sized build (kept large enough that the dilated conv + pooling arithmetic
    # stays valid: seq_len_after_stem=64-17+1=48 > 16).
    return DeepMILOBoundaryCNN(segment_size=64, nbr_feature=5)


def example_input_deepmilo():
    # (batch, segment_size, nbr_feature=len('ACGTN'), 1) one-hot DNA sequence window.
    return torch.rand(2, 64, 5, 1)


MENAGERIE_ENTRIES = [
    ("DeepMILO", build_deepmilo, example_input_deepmilo, 2020, "SOURCE_AVAILABLE"),
]
