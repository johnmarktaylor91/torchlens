# FAITHFUL PORT of sidhomj/DeepTCR @ master (DeepTCR/functions/Layers.py, DeepTCR/DeepTCR.py)
# (original framework: TensorFlow 1.x, graph-mode via tf.compat.v1)
#
# Sidhom et al. 2021, "DeepTCR is a deep learning framework for revealing sequence
# concepts within T-cell repertoires", Nature Communications.
#
# The real DeepTCR repo is TF1.x graph-mode code (tf.compat.v1.placeholder /
# tf.compat.v1.get_variable / tf.compat.v1.layers.*) entangled with a large stateful
# `DeepTCR_*` class (DeepTCR/DeepTCR.py, ~280KB) that dynamically builds one of many
# possible graphs depending on config flags (alpha/beta chain use, V/D/J gene features,
# HLA features, trainable AA embedding, VAE vs. supervised heads, multi-sample dropout,
# etc.). It cannot run in a base torch env, so this ports the well-defined core
# "supervised sequence classifier" configuration faithfully from the actual functions:
#
#   - `Convolutional_Features` / `Conv_Model` (Layers.py:91-243), 'medium' net size
#     (units = [32, 64, 128]), `use_only_seq=True`, single (beta) chain, no gene/HLA
#     features, `trainable_embedding=False` (raw one-hot AA channels feed the conv
#     tower directly, matching the `else` branch at Layers.py:167-171).
#   - First conv1d: kernel=5 (paper default `kernel=5`), stride 1, 'same' padding,
#     then leaky_relu + dropout (Layers.py:104-108).
#   - Subsequent convs: kernel=3, stride=3 (downsampling), 'same' padding, then
#     leaky_relu + dropout (Layers.py:110-115) -- exactly `Convolutional_Features`'s
#     `else` branch, which hardcodes kernel=3 for every layer after the first.
#   - Feature pooling: `tf.reduce_max` over the sequence axis then flatten
#     (Layers.py:117-118, the `net != 'ae'` / classifier branch of `Convolutional_Features`).
#   - Classification head: `num_fc_layers=0` (paper default), so `GO.Features` feeds
#     directly into a single `tf.compat.v1.layers.dense(GO.Features, self.Y.shape[1])`
#     logits layer (DeepTCR.py:3360), i.e. one Linear layer to `n_classes`.
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"

# Amino-acid one-hot depth used throughout DeepTCR (`tf.one_hot(X_Seq, depth=21)`).
_AA_DEPTH = 21


class DeepTCRConvFeatures(nn.Module):
    """Sequence-CNN feature extractor, faithful port of `Convolutional_Features`
    (medium net size: units = [32, 64, 128]) in DeepTCR/functions/Layers.py.
    """

    def __init__(
        self,
        in_channels: int = _AA_DEPTH,
        units=(32, 64, 128),
        kernel: int = 5,
        dropout: float = 0.0,
    ):
        super().__init__()
        convs = []
        in_ch = in_channels
        for i, num_units in enumerate(units):
            if i == 0:
                convs.append(
                    nn.Conv1d(in_ch, num_units, kernel_size=kernel, stride=1, padding="same")
                )
            else:
                convs.append(nn.Conv1d(in_ch, num_units, kernel_size=3, stride=3, padding=1))
            in_ch = num_units
        self.convs = nn.ModuleList(convs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, in_channels, seq_len)
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x)
            x = self.dropout(x)
        # tf.reduce_max over the sequence axis, then flatten (Layers.py:117-118)
        x = torch.amax(x, dim=2)
        return x.flatten(1)


class DeepTCRSequenceClassifier(nn.Module):
    """DeepTCR supervised sequence classifier (`use_only_seq=True`, single chain,
    'medium' conv net, `num_fc_layers=0`): AA one-hot -> conv feature tower ->
    global max-pool -> linear classification head.
    """

    def __init__(
        self,
        seq_len: int = 40,
        n_classes: int = 2,
        kernel: int = 5,
        units=(32, 64, 128),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.conv_features = DeepTCRConvFeatures(
            in_channels=_AA_DEPTH, units=units, kernel=kernel, dropout=dropout
        )
        self.classifier = nn.Linear(units[-1], n_classes)

    def forward(self, x_onehot):
        # x_onehot: (batch, seq_len, 21) one-hot amino-acid sequence
        x = x_onehot.transpose(1, 2)
        feats = self.conv_features(x)
        return self.classifier(feats)


def build_deeptcr():
    return DeepTCRSequenceClassifier(seq_len=40, n_classes=2)


def example_input_deeptcr():
    return torch.randn(2, 40, 21)


MENAGERIE_ENTRIES = [
    ("DeepTCR", build_deeptcr, example_input_deeptcr, 2021, "ported-pytorch"),
]
