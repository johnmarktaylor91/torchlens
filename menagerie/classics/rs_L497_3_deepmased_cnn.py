# FAITHFUL PORT of leylabmpi/DeepMAsED @ master (DeepMAsED/Models.py `deepmased`)
# (original framework: tensorflow.keras)
#
# DeepMAsED (Mineeva, Rojas-Carulla, Ley, Baaijens & Krause, Bioinformatics 2020,
# "DeepMAsED: evaluating the quality of metagenomic assemblies") is a 1D-over-
# contig-positions CNN misassembly detector. The real `deepmased.__init__`
# builds a `keras.Sequential`: an initial Conv2D(filters, kernel=(2,n_features),
# 'valid') + BatchNorm over a (max_len, n_features, 1) contig-feature-matrix
# input, then `n_conv - 1` further Conv2D(2**i * filters, kernel=(2,1),
# strides=2) + BatchNorm layers (each halving the length while doubling
# channels), an AveragePooling2D((pool_window, 1)), Flatten, `n_fc - 1`
# Dense(n_hid, relu)+Dropout blocks, and a final Dense(1, sigmoid)+Dropout head.
# Real training defaults (DeepMAsED/Commands/Train.py argparse + Train.py
# n_features=11): filters=8, n_hid=50, n_conv=5, n_fc=3, max_len=10000,
# dropout=0.5, pool_window=50, n_features=11. Every layer/hyperparameter below
# matches the real Keras Sequential; only the Keras -> torch layer translation
# and CLI/data-generator/training glue are new.
import torch
from torch import nn

MENAGERIE_ZOO = "ported-pytorch"


class DeepMAsEDCNN(nn.Module):
    def __init__(
        self,
        max_len=10000,
        filters=8,
        n_conv=5,
        n_features=11,
        pool_window=50,
        dropout=0.5,
        n_fc=3,
        n_hid=50,
    ):
        super().__init__()
        self.max_len = max_len
        self.filters = filters
        self.n_conv = n_conv
        self.n_features = n_features
        self.pool_window = pool_window
        self.dropout = dropout
        self.n_fc = n_fc
        self.n_hid = n_hid

        layers = []
        # keras: Conv2D(filters, kernel_size=(2, n_features), input_shape=(max_len, n_features, 1),
        #               activation='relu', padding='valid')
        layers.append(nn.Conv2d(1, filters, kernel_size=(2, n_features), padding=0))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm2d(filters))
        cur_len = max_len - 1  # 'valid' conv with kernel_h=2 shrinks length by 1
        cur_channels = filters

        for i in range(1, n_conv):
            out_channels = 2**i * filters
            # keras: Conv2D(2**i*filters, kernel_size=(2,1), strides=2, activation='relu')
            # (default keras Conv2D padding is 'valid')
            layers.append(
                nn.Conv2d(cur_channels, out_channels, kernel_size=(2, 1), stride=(2, 1), padding=0)
            )
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(out_channels))
            cur_len = (cur_len - 2) // 2 + 1  # 'valid' conv output length formula, stride 2
            cur_channels = out_channels

        # keras: AveragePooling2D((pool_window, 1))
        pool_window = min(pool_window, cur_len) if cur_len > 0 else 1
        layers.append(nn.AvgPool2d(kernel_size=(pool_window, 1)))
        self.conv_stack = nn.Sequential(*layers)

        pooled_len = cur_len // pool_window if pool_window > 0 else cur_len
        flat_dim = (
            max(pooled_len, 1) * cur_channels * 1
        )  # width dim collapses to 1 after the first conv

        fc_layers = []
        in_dim = flat_dim
        for _ in range(n_fc - 1):
            fc_layers.append(nn.Linear(in_dim, n_hid))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(p=dropout))
            in_dim = n_hid
        fc_layers.append(nn.Linear(in_dim, 1))
        fc_layers.append(nn.Sigmoid())
        fc_layers.append(nn.Dropout(p=dropout))
        self.fc_stack = nn.Sequential(*fc_layers)

    def forward(self, x):
        # x: (N, 1, max_len, n_features) NCHW contig-feature matrix.
        x = self.conv_stack(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc_stack(x)


# ---------------------------------------------------------------------------
# menagerie staging entry point
# ---------------------------------------------------------------------------
# Use a small synthetic max_len for trace speed (real model uses max_len=10000);
# the layer stack (n_conv strided-conv stages + pooling + n_fc dense head) is
# identical, with n_conv/pool_window scaled down so the tiny sequence survives
# all n_conv-1 stride-2 halvings.
_TINY_MAX_LEN = 64
_N_FEATURES = 11
_N_CONV = 3
_POOL_WINDOW = 4


def build_deepmased_cnn():
    return DeepMAsEDCNN(
        max_len=_TINY_MAX_LEN,
        filters=8,
        n_conv=_N_CONV,
        n_features=_N_FEATURES,
        pool_window=_POOL_WINDOW,
        dropout=0.5,
        n_fc=3,
        n_hid=50,
    )


def example_input_deepmased_cnn():
    return torch.randn(2, 1, _TINY_MAX_LEN, _N_FEATURES)


MENAGERIE_ENTRIES = [
    ("DeepMAsED-CNN", build_deepmased_cnn, example_input_deepmased_cnn, 2020, "SOURCE_AVAILABLE"),
]
