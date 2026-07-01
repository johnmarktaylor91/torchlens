# FAITHFUL PORT of Hughes-Genome-Group/deepC @ master (original framework: TensorFlow 1.x
# `tf.compat.v1`, tensorflow2.1plus_compatibility_version/deepCregr.py::inference +
# dilated_layer + convolutional_layer, with default hyperparameters from
# run_training_deepCregr.py's `tf.app.flags` defaults)
#
# DeepC predicts long-range chromatin interactions (Hi-C-derived contact classes) from raw
# one-hot DNA sequence using a dilated-convolution architecture explicitly acknowledging
# the WaveNet dilated-conv design (see file header in the original: "adapted from
# ibab/tensorflow-wavenet"). The real code is `tf.compat.v1`-only (manual `tf.Variable`
# creation, `tf.name_scope`, no Keras `Model`/`Layer` objects) and cannot run against a
# torch-only base env, so this module transcribes the `inference()` graph-building
# function faithfully into self-contained torch: a strided 1D-conv stack (conv_layers=3,
# hidden_units_scheme=[300,600,900], kernel_width_scheme=[20,8,8], max_pool_scheme=
# [5,5,5], each conv+ReLU+MaxPool+Dropout) followed by a WaveNet-style dilated-conv stack
# (dilation_scheme=[1,2,4,8] -- the original always prepends an initial dilation=1 layer
# before the configured `dilation_scheme`; dilation_units=20, dilation_width=3; each layer
# is `tanh(dilated_conv) * sigmoid(gated_conv)`, non-residual by default since
# `dilation_residual=False`), then a final dense projection to `num_classes` regression
# outputs (`tf.matmul` + bias over the flattened dilated-stack output, exactly as
# `final_dense` in the original).
import torch
import torch.nn.functional as f
from torch import nn

MENAGERIE_ZOO = "ported-pytorch"

# Architecture defaults taken from run_training_deepCregr.py flags.DEFINE_* calls.
CONV_LAYERS = 3
HIDDEN_UNITS_SCHEME = [300, 600, 900]  # first `conv_layers` entries of '300,600,900,20'
KERNEL_WIDTH_SCHEME = [20, 8, 8]  # first `conv_layers` entries of '20,8,8,1'
MAX_POOL_SCHEME = [5, 5, 5]  # first `conv_layers` entries of '5,5,5,1'
DILATION_SCHEME = [2, 4, 8]  # inference() always prepends an extra dilation=1 layer first
DILATION_UNITS = 20
DILATION_WIDTH = 3
NUM_CLASSES = 50  # flags.DEFINE_integer('num_classes', 50, ...)
INPUT_DEPTH = 4  # one-hot ACGT
BP_CONTEXT = 4000  # original default is 1,000,000 (full Hi-C window); shrunk for tracing,
# kept divisible by prod(MAX_POOL_SCHEME) = 125 so the pooled-length math matches exactly.


class ConvolutionalLayer(nn.Module):
    """Port of `convolutional_layer`: Conv1D('SAME') -> ReLU -> MaxPool -> Dropout."""

    def __init__(
        self, in_channels: int, units: int, kernel_width: int, pool_width: int, keep_prob: float
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, units, kernel_size=kernel_width, padding="same")
        self.pool = nn.MaxPool1d(kernel_size=pool_width, stride=pool_width, ceil_mode=True)
        self.dropout = nn.Dropout(p=1.0 - keep_prob)

    def forward(self, x):
        # x: (batch, channels, length)
        out = f.relu(self.conv(x))
        out = self.pool(out)
        out = self.dropout(out)
        return out


class DilatedLayer(nn.Module):
    """Port of `dilated_layer`: gated dilated conv, `tanh(dilated) * sigmoid(gated)`."""

    def __init__(self, in_channels: int, dilation_units: int, dilation_width: int, dilation: int):
        super().__init__()
        self.dilation = dilation
        pad = (dilation_width - 1) * dilation // 2
        self.dilated_conv = nn.Conv1d(
            in_channels, dilation_units, kernel_size=dilation_width, dilation=dilation, padding=pad
        )
        self.gate_conv = nn.Conv1d(
            in_channels, dilation_units, kernel_size=dilation_width, dilation=dilation, padding=pad
        )

    def forward(self, x):
        dilated = self.dilated_conv(x)
        gated = self.gate_conv(x)
        out = torch.tanh(dilated) * torch.sigmoid(gated)
        # original `dilated_conv` slices back to the input width when the "SAME"-style
        # padding introduces an off-by-one; enforce identical output length here too.
        if out.shape[-1] != x.shape[-1]:
            out = out[..., : x.shape[-1]]
        return out


class DeepC(nn.Module):
    """Faithful port of `inference()` in Hughes-Genome-Group/deepC's deepCregr.py."""

    def __init__(
        self,
        input_depth: int = INPUT_DEPTH,
        conv_layers: int = CONV_LAYERS,
        hidden_units_scheme=None,
        kernel_width_scheme=None,
        max_pool_scheme=None,
        dilation_scheme=None,
        dilation_units: int = DILATION_UNITS,
        dilation_width: int = DILATION_WIDTH,
        num_classes: int = NUM_CLASSES,
        keep_prob_inner: float = 0.8,
        bp_context: int = BP_CONTEXT,
    ):
        super().__init__()
        hidden_units_scheme = hidden_units_scheme or HIDDEN_UNITS_SCHEME
        kernel_width_scheme = kernel_width_scheme or KERNEL_WIDTH_SCHEME
        max_pool_scheme = max_pool_scheme or MAX_POOL_SCHEME
        dilation_scheme = dilation_scheme or DILATION_SCHEME

        # Convolutional_stack
        conv_blocks = []
        in_channels = input_depth
        for i in range(conv_layers):
            conv_blocks.append(
                ConvolutionalLayer(
                    in_channels,
                    hidden_units_scheme[i],
                    kernel_width_scheme[i],
                    max_pool_scheme[i],
                    keep_prob_inner,
                )
            )
            in_channels = hidden_units_scheme[i]
        self.conv_stack = nn.ModuleList(conv_blocks)

        # dilated_stack: an initial dilation=1 layer, then the configured dilation_scheme.
        dilated_blocks = [DilatedLayer(in_channels, dilation_units, dilation_width, dilation=1)]
        in_channels = dilation_units
        for dilation in dilation_scheme:
            dilated_blocks.append(
                DilatedLayer(in_channels, dilation_units, dilation_width, dilation)
            )
        self.dilated_stack = nn.ModuleList(dilated_blocks)

        pooled_len = bp_context
        for pool_width in max_pool_scheme[:conv_layers]:
            pooled_len = -(
                -pooled_len // pool_width
            )  # ceil division, matches MaxPool1d(ceil_mode=True)
        self.fully_connected_width = pooled_len * dilation_units

        # final_dense: tf.matmul(current_layer, weights) + biases
        self.final_dense = nn.Linear(self.fully_connected_width, num_classes)

    def forward(self, seqs):
        # seqs: (batch, bp_context, input_depth) one-hot DNA, matches the original
        # `seqs_placeholder` layout; transpose to torch's (batch, channels, length).
        x = seqs.transpose(1, 2)
        for block in self.conv_stack:
            x = block(x)
        for block in self.dilated_stack:
            x = block(x)
        flat = x.reshape(x.shape[0], -1)
        return self.final_dense(flat)


def build_deepc():
    return DeepC()


def example_input_deepc():
    return torch.randn(1, BP_CONTEXT, INPUT_DEPTH)


MENAGERIE_ENTRIES = [
    ("DeepC", build_deepc, example_input_deepc, 2019, "REIMPLEMENT"),
]
