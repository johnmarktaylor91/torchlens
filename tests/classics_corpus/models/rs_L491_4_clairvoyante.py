# FAITHFUL PORT of aquaskyline/Clairvoyante @ master (original framework: TensorFlow 1.x,
# `clairvoyante/clairvoyante_v3.py`)
#
# Clairvoyante (Nat. Commun. 2019) is the direct predecessor to Clair3 (see the
# `HKU-BAL/Clair3` entry in this same batch, which vendors Clair3's real torch code):
# a small variant caller that reads a pileup tensor and jointly predicts base change,
# zygosity, variant type, and indel length via a shared 3-layer SELU CNN trunk feeding
# 5 parallel dense heads. The repo is TF1.x graph-mode (`tf.Graph`/`tf.placeholder`/
# `tf.Session`, `tf.contrib`), which cannot run in a modern base-torch environment, so
# `_buildGraph()` in `clairvoyante/clairvoyante_v3.py` (the `v3`/non-slim model, the most
# current non-deprecated variant per the repo's own module list) was transcribed layer for
# layer:
#
#   input: (batch, 33, 4, matrixNum=4)             [2*flankingBaseNum(16)+1 = 33 positions]
#   conv1 = Conv2d(4, 16, kernel=(1,4), pad=same, act=selu)
#   pool1 = MaxPool2d(kernel=(5,1), stride=1)         [pollSize1]
#   conv2 = Conv2d(16, 32, kernel=(2,4), pad=same, act=selu)
#   pool2 = MaxPool2d(kernel=(4,1), stride=1)         [pollSize2]
#   conv3 = Conv2d(32, 48, kernel=(3,4), pad=same, act=selu)
#   pool3 = MaxPool2d(kernel=(3,1), stride=1)         [pollSize3]
#   flatten -> fc4 = Linear(flat_size, 336, act=selu) -> dropout_selu(0.5)
#   fc5 = Linear(336, 168, act=selu) -> dropout_selu(0.0)
#   YBaseChangeSigmoid  = Linear(336(fc4 output), 4, act=sigmoid)   [reads dropout4, not fc5]
#   YZygosityFC         = Linear(168, 2, act=selu) -> +eps -> softmax
#   YVarTypeFC          = Linear(168, 4, act=selu) -> +eps -> softmax
#   YIndelLengthFC      = Linear(168, 6, act=selu) -> +eps -> softmax
#
# `selu.selu` is the standard SELU nonlinearity (alpha=1.6732632423543772848170429916717,
# scale=1.0507009873554804934193349852946), i.e. `torch.nn.SELU()`/`F.selu` with its
# defaults. `dropout_selu` is SELU-safe alpha-dropout (mean/variance preserving), i.e.
# `torch.nn.AlphaDropout`, so it is used in place of standard `nn.Dropout` here (functional
# match to `selu.py`'s `dropout_selu_impl`, not a bitwise transcription of that helper's
# TF-specific noise machinery). `flat_size` is computed exactly as in the original
# `_buildGraph` (`inputShape[i] - sum(pollSize[i]-1 for each pool)` per spatial axis, times
# `numFeature3`), and `YBaseChangeSigmoid` reads `dropout4` (fc4's dropout output) not fc5
# -- exactly as in the source (`tf.layers.dense(inputs=dropout4, ...)` on line 125 of
# clairvoyante_v3.py), which looks asymmetric versus the other 3 heads but is what upstream
# ships.
import torch
from torch import nn

MENAGERIE_ZOO = "ported-pytorch"

FLANKING_BASE_NUM = 16
NO_OF_POSITIONS = 2 * FLANKING_BASE_NUM + 1  # 33
MATRIX_NUM = 4
INPUT_WIDTH = 4  # second spatial axis of the (33, 4, matrixNum) pileup tensor

POOL_SIZE_1 = (5, 1)
POOL_SIZE_2 = (4, 1)
POOL_SIZE_3 = (3, 1)

NUM_FEATURE_1 = 16
NUM_FEATURE_2 = 32
NUM_FEATURE_3 = 48

HIDDEN_UNITS_4 = 336
HIDDEN_UNITS_5 = 168

OUTPUT_BASE_CHANGE = 4
OUTPUT_ZYGOSITY = 2
OUTPUT_VAR_TYPE = 4
OUTPUT_INDEL_LENGTH = 6

DROPOUT_RATE_FC4 = 0.5
DROPOUT_RATE_FC5 = 0.0


class Clairvoyante(nn.Module):
    """Faithful port of Clairvoyante v3's `_buildGraph` (clairvoyante_v3.py)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(MATRIX_NUM, NUM_FEATURE_1, kernel_size=(1, 4), padding="same")
        self.pool1 = nn.MaxPool2d(kernel_size=POOL_SIZE_1, stride=1)
        self.conv2 = nn.Conv2d(NUM_FEATURE_1, NUM_FEATURE_2, kernel_size=(2, 4), padding="same")
        self.pool2 = nn.MaxPool2d(kernel_size=POOL_SIZE_2, stride=1)
        self.conv3 = nn.Conv2d(NUM_FEATURE_2, NUM_FEATURE_3, kernel_size=(3, 4), padding="same")
        self.pool3 = nn.MaxPool2d(kernel_size=POOL_SIZE_3, stride=1)

        flat_h = (
            NO_OF_POSITIONS - (POOL_SIZE_1[0] - 1) - (POOL_SIZE_2[0] - 1) - (POOL_SIZE_3[0] - 1)
        )
        flat_w = INPUT_WIDTH - (POOL_SIZE_1[1] - 1) - (POOL_SIZE_2[1] - 1) - (POOL_SIZE_3[1] - 1)
        flat_size = flat_h * flat_w * NUM_FEATURE_3
        self.flat_h = flat_h
        self.flat_w = flat_w

        self.fc4 = nn.Linear(flat_size, HIDDEN_UNITS_4)
        self.dropout4 = nn.AlphaDropout(p=DROPOUT_RATE_FC4)
        self.fc5 = nn.Linear(HIDDEN_UNITS_4, HIDDEN_UNITS_5)
        self.dropout5 = nn.AlphaDropout(p=DROPOUT_RATE_FC5)

        self.y_base_change = nn.Linear(HIDDEN_UNITS_4, OUTPUT_BASE_CHANGE)
        self.y_zygosity_fc = nn.Linear(HIDDEN_UNITS_5, OUTPUT_ZYGOSITY)
        self.y_var_type_fc = nn.Linear(HIDDEN_UNITS_5, OUTPUT_VAR_TYPE)
        self.y_indel_length_fc = nn.Linear(HIDDEN_UNITS_5, OUTPUT_INDEL_LENGTH)

        self.selu = nn.SELU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.epsilon = 1e-10

    def forward(self, x):
        # x: (batch, matrixNum=4, 33, 4) NCHW; upstream is NHWC (batch, 33, 4, matrixNum).
        x = self.selu(self.conv1(x))
        x = self.pool1(x)
        x = self.selu(self.conv2(x))
        x = self.pool2(x)
        x = self.selu(self.conv3(x))
        x = self.pool3(x)

        x = torch.flatten(x, start_dim=1)
        fc4 = self.selu(self.fc4(x))
        dropout4 = self.dropout4(fc4)
        fc5 = self.selu(self.fc5(dropout4))
        dropout5 = self.dropout5(fc5)

        y_base_change_sigmoid = self.sigmoid(self.y_base_change(dropout4))

        y_zygosity_logits = self.selu(self.y_zygosity_fc(dropout5)) + self.epsilon
        y_zygosity_softmax = self.softmax(y_zygosity_logits)

        y_var_type_logits = self.selu(self.y_var_type_fc(dropout5)) + self.epsilon
        y_var_type_softmax = self.softmax(y_var_type_logits)

        y_indel_length_logits = self.selu(self.y_indel_length_fc(dropout5)) + self.epsilon
        y_indel_length_softmax = self.softmax(y_indel_length_logits)

        return torch.cat(
            [y_base_change_sigmoid, y_zygosity_softmax, y_var_type_softmax, y_indel_length_softmax],
            dim=-1,
        )


def build_clairvoyante():
    return Clairvoyante()


def example_input_clairvoyante():
    # (batch, matrixNum=4, 33, 4) NCHW pileup tensor.
    return torch.randn(2, MATRIX_NUM, NO_OF_POSITIONS, INPUT_WIDTH)


MENAGERIE_ENTRIES = [
    ("Clairvoyante", build_clairvoyante, example_input_clairvoyante, 2019, "ported-pytorch"),
]
