# SOURCE: vendored from CSUBioGroup/DeepPPISP @ master
# https://raw.githubusercontent.com/CSUBioGroup/DeepPPISP/master/models/deep_ppi.py
# https://raw.githubusercontent.com/CSUBioGroup/DeepPPISP/master/models/BasicModule.py
# https://raw.githubusercontent.com/CSUBioGroup/DeepPPISP/master/utils/config.py
#
# Zeng, Guo, Ma, Peng, Chen 2020 (Bioinformatics) "DeepPPISP: prediction of protein-protein
# interaction sites using deep learning". Sequence-, PSSM-, and DSSP-derived local + global
# features fused through a multi-window Conv2d "TextCNN-style" branch (`ConvsLayer`) and a
# global sequence-embedding branch, concatenated with hand-built local sliding-window
# features, then passed through a 3-layer MLP head to a per-residue interaction-site score.
#
# `ConvsLayer` and `DeepPPI` are copied verbatim from `models/deep_ppi.py` (the real
# `BasicModule` base class is just `nn.Module` + save/load helpers, inlined here so the file
# is self-contained). `DefaultConfig` values (`max_sequence_length`, `seq_dim`, `dssp_dim`,
# `pssm_dim`, `kernels`, `cnn_chanel`) are copied from `utils/config.py` / the `ratio`-derived
# assignment in `DeepPPI.__init__`, matching the real `demo()` call in `train.py`:
# `DeepPPI(class_nums=1, window_size=3, ratio=(2, 1))`. Only edits: module-level mutable
# `configs` singleton reduced to a plain namespace object (no import-time dependency on
# `utils/config.py`), and `max_sequence_length` shrunk from the real 500 to 40 purely to keep
# the traced example tiny -- no architectural change (the network is fully convolutional over
# the sequence-length axis via the kernel/padding scheme, so any length works).

import torch as t
from torch import nn


class _Config:
    """Inlined from utils/config.py (DefaultConfig), mutable fields only."""

    max_sequence_length = 40  # real default is 500; shrunk for a tiny traced example
    seq_dim = 20
    dssp_dim = 9
    pssm_dim = 20
    kernels = [13, 15, 17]
    dropout = 0.2
    cnn_chanel = 0  # set by DeepPPI.__init__ from `ratio`, mirroring the real code


configs = _Config()


class BasicModule(nn.Module):
    """Inlined from models/BasicModule.py (save/load helpers dropped; unused here)."""

    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))


class ConvsLayer(BasicModule):
    def __init__(self):
        super(ConvsLayer, self).__init__()

        self.kernels = configs.kernels
        hidden_channels = configs.cnn_chanel
        in_channel = 1
        features_L = configs.max_sequence_length
        seq_dim = configs.seq_dim
        dssp_dim = configs.dssp_dim
        pssm_dim = configs.pssm_dim
        W_size = seq_dim + dssp_dim + pssm_dim

        padding1 = (self.kernels[0] - 1) // 2
        padding2 = (self.kernels[1] - 1) // 2
        padding3 = (self.kernels[2] - 1) // 2
        self.conv1 = nn.Sequential()
        self.conv1.add_module(
            "conv1",
            nn.Conv2d(
                in_channel,
                hidden_channels,
                padding=(padding1, 0),
                kernel_size=(self.kernels[0], W_size),
            ),
        )
        self.conv1.add_module("ReLU", nn.PReLU())
        self.conv1.add_module("pooling1", nn.MaxPool2d(kernel_size=(features_L, 1), stride=1))

        self.conv2 = nn.Sequential()
        self.conv2.add_module(
            "conv2",
            nn.Conv2d(
                in_channel,
                hidden_channels,
                padding=(padding2, 0),
                kernel_size=(self.kernels[1], W_size),
            ),
        )
        self.conv2.add_module("ReLU", nn.ReLU())
        self.conv2.add_module("pooling2", nn.MaxPool2d(kernel_size=(features_L, 1), stride=1))

        self.conv3 = nn.Sequential()
        self.conv3.add_module(
            "conv3",
            nn.Conv2d(
                in_channel,
                hidden_channels,
                padding=(padding3, 0),
                kernel_size=(self.kernels[2], W_size),
            ),
        )
        self.conv3.add_module("ReLU", nn.ReLU())
        self.conv3.add_module("pooling3", nn.MaxPool2d(kernel_size=(features_L, 1), stride=1))

    def forward(self, x):
        features1 = self.conv1(x)
        features2 = self.conv2(x)
        features3 = self.conv3(x)
        features = t.cat((features1, features2, features3), 1)
        shapes = features.data.shape
        features = features.view(shapes[0], shapes[1] * shapes[2] * shapes[3])

        return features


class DeepPPI(BasicModule):
    def __init__(self, class_nums, window_size, ratio=None):
        super(DeepPPI, self).__init__()
        global configs
        configs.kernels = [13, 15, 17]
        self.dropout = configs.dropout = 0.2

        seq_dim = configs.seq_dim * configs.max_sequence_length

        self.seq_layers = nn.Sequential()
        self.seq_layers.add_module("seq_embedding_layer", nn.Linear(seq_dim, seq_dim))
        self.seq_layers.add_module("seq_embedding_ReLU", nn.ReLU())

        seq_dim = configs.seq_dim
        dssp_dim = configs.dssp_dim
        pssm_dim = configs.pssm_dim
        local_dim = (window_size * 2 + 1) * (pssm_dim + dssp_dim + seq_dim)
        if ratio:
            configs.cnn_chanel = (local_dim * int(ratio[0])) // (int(ratio[1]) * 3)
        input_dim = configs.cnn_chanel * 3 + local_dim

        self.multi_CNN = nn.Sequential()
        self.multi_CNN.add_module("layer_convs", ConvsLayer())

        self.DNN1 = nn.Sequential()
        self.DNN1.add_module("DNN_layer1", nn.Linear(input_dim, 1024))
        self.DNN1.add_module("ReLU1", nn.ReLU())
        self.DNN2 = nn.Sequential()
        self.DNN2.add_module("DNN_layer2", nn.Linear(1024, 256))
        self.DNN2.add_module("ReLU2", nn.ReLU())

        self.outLayer = nn.Sequential(nn.Linear(256, class_nums), nn.Sigmoid())

    def forward(self, seq, dssp, pssm, local_features):
        shapes = seq.data.shape
        features = seq.view(shapes[0], shapes[1] * shapes[2] * shapes[3])
        features = self.seq_layers(features)
        features = features.view(shapes[0], shapes[1], shapes[2], shapes[3])

        features = t.cat((features, dssp, pssm), 3)
        features = self.multi_CNN(features)
        features = t.cat((features, local_features), 1)
        features = self.DNN1(features)
        features = self.DNN2(features)
        features = self.outLayer(features)

        return features


MENAGERIE_ZOO = "vendored-pytorch"


def build_deepppisp():
    # matches the real demo() call: DeepPPI(class_nums=1, window_size=3, ratio=(2, 1))
    model = DeepPPI(class_nums=1, window_size=3, ratio=(2, 1))
    model.eval()
    return model


def example_input_deepppisp():
    batch = 1
    L = configs.max_sequence_length
    window_size = 3
    local_dim = (window_size * 2 + 1) * (configs.pssm_dim + configs.dssp_dim + configs.seq_dim)
    seq = t.randn(batch, 1, L, configs.seq_dim)
    dssp = t.randn(batch, 1, L, configs.dssp_dim)
    pssm = t.randn(batch, 1, L, configs.pssm_dim)
    local_features = t.randn(batch, local_dim)
    return seq, dssp, pssm, local_features


MENAGERIE_ENTRIES = [
    (
        "DeepPPISP",
        build_deepppisp,
        example_input_deepppisp,
        2020,
        MENAGERIE_ZOO,
    ),
]
