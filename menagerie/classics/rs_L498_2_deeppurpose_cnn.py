# SOURCE: vendored from kexinhuang12345/DeepPurpose @ master
# (DeepPurpose/encoders.py, class CNN)
#
# DeepPurpose CNN drug/protein encoder: a stack of 1D convolutions over a one-hot
# SMILES/amino-acid encoding, followed by adaptive max-pool and a linear projection to a
# fixed-size embedding (the DeepDTA-style encoder DeepPurpose uses inside its DTI/CompoundPred
# pipelines). Copied verbatim from DeepPurpose/encoders.py aside from: (1) dropping the
# `DeepPurpose.utils`/`DeepPurpose.model_helper` package imports (encoders.py imports
# `from DeepPurpose.utils import *` at module scope purely for unrelated helper functions used
# by *other* classes in that file -- e.g. rdkit/descriptastorus-backed featurizers -- none of
# which `CNN` itself calls; those third-party deps (rdkit, descriptastorus, subword-nmt) are
# not installed base libs and are not needed to run this class), and (2) inlining a module-level
# `device` constant instead of importing it. The `CNN` class body (conv stack, adaptive pool,
# fc1, `_get_conv_output`/`_forward_features`/`forward`) is untouched.
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

MENAGERIE_ZOO = "vendored-pytorch"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN(nn.Sequential):
    def __init__(self, encoding, **config):
        super(CNN, self).__init__()
        if encoding == "drug":
            in_ch = [63] + config["cnn_drug_filters"]
            kernels = config["cnn_drug_kernels"]
            layer_size = len(config["cnn_drug_filters"])
            self.conv = nn.ModuleList(
                [
                    nn.Conv1d(
                        in_channels=in_ch[i], out_channels=in_ch[i + 1], kernel_size=kernels[i]
                    )
                    for i in range(layer_size)
                ]
            )
            self.conv = self.conv.double()
            n_size_d = self._get_conv_output((63, 100))
            self.fc1 = nn.Linear(n_size_d, config["hidden_dim_drug"])

        if encoding == "protein":
            in_ch = [26] + config["cnn_target_filters"]
            kernels = config["cnn_target_kernels"]
            layer_size = len(config["cnn_target_filters"])
            self.conv = nn.ModuleList(
                [
                    nn.Conv1d(
                        in_channels=in_ch[i], out_channels=in_ch[i + 1], kernel_size=kernels[i]
                    )
                    for i in range(layer_size)
                ]
            )
            self.conv = self.conv.double()
            n_size_p = self._get_conv_output((26, 1000))

            self.fc1 = nn.Linear(n_size_p, config["hidden_dim_protein"])

    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input.double())
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        for layer in self.conv:
            x = F.relu(layer(x))
        x = F.adaptive_max_pool1d(x, output_size=1)
        return x

    def forward(self, v):
        v = self._forward_features(v.double())
        v = v.view(v.size(0), -1)
        v = self.fc1(v.float())
        return v


def build_deeppurpose_cnn_drug():
    # Config values match the DeepPurpose default CNN_drug hyperparameters from
    # DeepPurpose/utils.py's `generate_config` (cnn_drug_filters=[32,64,96],
    # cnn_drug_kernels=[4,6,8], hidden_dim_drug=256), shrunk for a fast trace-sized build.
    config = {
        "cnn_drug_filters": [8, 16],
        "cnn_drug_kernels": [4, 6],
        "hidden_dim_drug": 32,
    }
    return CNN("drug", **config)


def example_input_deeppurpose_cnn_drug():
    # (batch, 63, 100): one-hot SMILES-alphabet encoding (63 characters), length-100 sequence.
    return torch.rand(2, 63, 100).double()


MENAGERIE_ENTRIES = [
    (
        "DeepPurpose-CNN",
        build_deeppurpose_cnn_drug,
        example_input_deeppurpose_cnn_drug,
        2020,
        "SOURCE_AVAILABLE",
    ),
]
