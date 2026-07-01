# FAITHFUL PORT of cangermueller/deepcpg @ master (original framework: Keras 1.x / TensorFlow 1.x)
#
# Source file transcribed: deepcpg/models/dna.py (class CnnL2h128, the DNA-branch
# CNN used as DeepCpG's flagship "CpG methylation from DNA sequence" model --
# Angermueller et al., "DeepCpG: accurate prediction of single-cell DNA
# methylation states using deep learning", Genome Biology 2017).
#
# The original repo pins Keras 1.x-era APIs (`kernel_regularizer` combined with
# `border_mode=`/`subsample_length=` positional Conv1D kwargs, `_keras_shape`
# introspection) that are incompatible with any Keras/TensorFlow version
# installable alongside our torch stack, and neither keras nor tensorflow are
# in the base-lib allowlist for this repo -- so this is a faithful architectural
# transcription into self-contained torch of the `CnnL2h128` DNA model
# ("CNN with two convolutional and one fully-connected layer", the model
# DeepCpG's own docstring calls out with parameter count 4,100,000 and spec
# `conv[128@11]_mp[4]_conv[256@3]_mp[2]_fc[128]_do`):
#
#   conv1d(128, kernel=11) -> relu -> maxpool(4)
#   -> conv1d(256, kernel=3) -> relu -> maxpool(2)
#   -> flatten -> dense(128) -> relu -> dropout
#   -> sigmoid output head predicting per-CpG-site methylation probability
#      (ScaledSigmoid(scaling=1.0) in the original -- a plain sigmoid, since
#      the DNA model's output scope has no non-unit scaling in the shipped
#      configs; see models/utils.py Model._build / DnaModel.inputs).
#
# Input: one-hot-encoded DNA sequence window of shape (batch, dna_wlen, 4),
# matching `DnaModel.inputs(dna_wlen)` (`kl.Input(shape=(dna_wlen, 4))`).

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CnnL2h128(nn.Module):
    """DeepCpG DNA-branch CNN: 2 conv layers + 1 FC(128) layer + dropout head.

    Faithful port of `deepcpg.models.dna.CnnL2h128.__call__`.
    """

    def __init__(self, dna_wlen: int = 501, nb_hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.dna_wlen = dna_wlen
        self.nb_hidden = nb_hidden

        # conv[128@11]_mp[4]
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=128, kernel_size=11)
        len1 = dna_wlen - 11 + 1
        pool1 = len1 // 4
        self.pool1 = nn.MaxPool1d(kernel_size=4)

        # conv[256@3]_mp[2]
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3)
        len2 = pool1 - 3 + 1
        pool2 = len2 // 2
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        flat_dim = 256 * pool2
        self.fc = nn.Linear(flat_dim, nb_hidden)
        self.dropout = nn.Dropout(dropout)

        # Output head: single-unit sigmoid predicting methylation probability
        # per CpG site in the window (deepcpg/models/utils.py ScaledSigmoid,
        # scaling=1.0 for the DNA-only model configuration).
        self.output = nn.Linear(nb_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, dna_wlen, 4) one-hot DNA window -> Conv1d wants channels-first
        x = x.transpose(1, 2)  # (batch, 4, dna_wlen)

        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc(x))
        x = self.dropout(x)

        return torch.sigmoid(self.output(x))


def build_deepcpg() -> CnnL2h128:
    return CnnL2h128(dna_wlen=101, nb_hidden=128, dropout=0.0)


def example_input_deepcpg() -> torch.Tensor:
    torch.manual_seed(0)
    batch = 2
    dna_wlen = 101
    idx = torch.randint(0, 4, (batch, dna_wlen))
    onehot = F.one_hot(idx, num_classes=4).float()
    return onehot


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("DeepCpG-CnnL2h128", "build_deepcpg", "example_input_deepcpg", 2017, "ported-pytorch"),
]
