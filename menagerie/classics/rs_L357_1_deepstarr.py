# FAITHFUL PORT of bernardo-de-almeida/DeepSTARR @ b02e460c7581934bb6c8910e53be04da10688781
# (original framework: Keras/TensorFlow)
#
# Source: DeepSTARR/DeepSTARR_training.ipynb, `DeepSTARR(params=params)` builder function.
# de Almeida et al. 2022, "DeepSTARR predicts enhancer activity from DNA sequence and
# enables the de novo design of synthetic enhancers", Nature Genetics.
#
# The original repository ships only a Keras/TensorFlow training notebook
# (DeepSTARR/DeepSTARR_training.ipynb) with no PyTorch original to vendor directly, so the
# architecture is transcribed faithfully here from the notebook's exact layer stack and
# published hyperparameters: 4x [Conv1d -> BatchNorm -> ReLU -> MaxPool(2)] with kernel
# sizes (7, 3, 5, 3) and filter counts (256, 60, 60, 120), followed by 2x
# [Linear -> BatchNorm -> ReLU -> Dropout(0.4)] with widths (256, 256), and two linear
# scalar output heads (developmental and housekeeping enhancer activity), exactly matching
# the `tasks = ['Dev', 'Hk']` dual-head structure in the notebook.
import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class DeepSTARR(nn.Module):
    """DeepSTARR dual-head enhancer-activity CNN (de Almeida et al. 2022).

    Faithful port of the Keras `DeepSTARR()` builder: a 4-layer 1D-conv tower
    (Conv1d -> BatchNorm -> ReLU -> MaxPool) followed by a 2-layer dense tower
    (Linear -> BatchNorm -> ReLU -> Dropout), ending in two linear regression
    heads for developmental and housekeeping enhancer activity.
    """

    def __init__(
        self,
        seq_len: int = 249,
        num_filters=(256, 60, 60, 120),
        kernel_sizes=(7, 3, 5, 3),
        dense_neurons=(256, 256),
        dropout_prob: float = 0.4,
    ):
        super().__init__()
        self.seq_len = seq_len

        conv_layers = []
        in_channels = 4
        for num_filter, kernel_size in zip(num_filters, kernel_sizes):
            conv_layers.append(
                nn.Conv1d(in_channels, num_filter, kernel_size=kernel_size, padding="same")
            )
            conv_layers.append(nn.BatchNorm1d(num_filter))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool1d(2))
            in_channels = num_filter
        self.conv_body = nn.Sequential(*conv_layers)

        flat_len = seq_len
        for _ in num_filters:
            flat_len = flat_len // 2
        flat_dim = flat_len * num_filters[-1]

        dense_layers = []
        in_dim = flat_dim
        for dense_dim in dense_neurons:
            dense_layers.append(nn.Linear(in_dim, dense_dim))
            dense_layers.append(nn.BatchNorm1d(dense_dim))
            dense_layers.append(nn.ReLU())
            dense_layers.append(nn.Dropout(dropout_prob))
            in_dim = dense_dim
        self.dense_body = nn.Sequential(*dense_layers)

        self.head_dev = nn.Linear(in_dim, 1)
        self.head_hk = nn.Linear(in_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, 4) one-hot DNA -> conv1d wants channels-first
        x = x.transpose(1, 2)
        x = self.conv_body(x)
        x = x.flatten(1)
        x = self.dense_body(x)
        return self.head_dev(x), self.head_hk(x)


def build_deepstarr():
    return DeepSTARR(seq_len=64)


def example_input_deepstarr():
    return torch.randn(2, 64, 4)


MENAGERIE_ENTRIES = [
    ("DeepSTARR", build_deepstarr, example_input_deepstarr, 2022, "ported-pytorch"),
]
