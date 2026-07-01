# SOURCE: vendored from DENGARDEN/DeepCpf1_torch_public @ master
#
# Vendored PyTorch re-implementation of Seq-DeepCpf1 (Kim et al., "Deep learning
# improves prediction of CRISPR-Cpf1 guide RNA activity", Nat. Biotechnol. 2018;
# original R/Keras code at lje00006/DeepCpf1). This module is the ACTUAL
# `SeqDeepCpf1Net` class from DENGARDEN/DeepCpf1_torch_public/deepcpf1_network.py,
# copied verbatim (imports/unused training-script utilities trimmed; the
# nn.Module architecture itself is untouched) plus a thin build()/example_input()
# staging wrapper for TorchLens tracing.
#
# Architecture (unmodified from source): a single Conv2d over the one-hot
# (seq_len, 4) encoded 34bp target sequence, adaptive average pool, then a
# 4-layer MLP head (80 -> 80 -> 40 -> 40 -> 1) with dropout between every
# layer, predicting indel-frequency (CRISPR-Cpf1 guide RNA on-target activity).

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeqDeepCpf1Net(nn.Module):
    def __init__(self, args):
        super().__init__()

        # Adopting DeepCpf1 NN structure
        self.Seq_deepCpf1_C1 = nn.Conv2d(
            in_channels=1, out_channels=80, kernel_size=(args.kernel_size, 4)
        )
        torch.nn.init.xavier_uniform_(self.Seq_deepCpf1_C1.weight)
        self.Seq_deepCpf1_C1.bias.data.fill_(0.0)
        w_ = (args.sequence_length - args.kernel_size + 0) // 1 + 1  # After CONV
        w_ = (w_ - args.pool_size) // args.pool_size + 1  # After POOL
        self.Seq_deepCpf1_P1 = torch.nn.AdaptiveAvgPool2d((w_, 1))
        self.Seq_deepCpf1_DO1 = nn.Dropout(0.3)
        # 4 * 14 * 80 (flattened)

        dim = w_ * 80
        self.Seq_deepCpf1_D1 = nn.Linear(in_features=dim, out_features=80)
        self.Seq_deepCpf1_DO2 = nn.Dropout(0.3)
        self.Seq_deepCpf1_D2 = nn.Linear(in_features=80, out_features=40)
        self.Seq_deepCpf1_DO3 = nn.Dropout(0.3)
        self.Seq_deepCpf1_D3 = nn.Linear(in_features=40, out_features=40)
        self.Seq_deepCpf1_DO4 = nn.Dropout(0.3)
        self.Seq_deepCpf1_Output = nn.Linear(in_features=40, out_features=1)

    def forward(self, x, _):
        # Input matrix.dim == 34 * 4 one-hot encoding matrix
        x = F.relu(self.Seq_deepCpf1_C1(x))
        x = self.Seq_deepCpf1_P1(x)
        x = torch.flatten(x, start_dim=1)
        x = self.Seq_deepCpf1_DO1(x)
        x = F.relu(self.Seq_deepCpf1_D1(x))
        x = self.Seq_deepCpf1_DO2(x)
        x = F.relu(self.Seq_deepCpf1_D2(x))
        x = self.Seq_deepCpf1_DO3(x)
        x = F.relu(self.Seq_deepCpf1_D3(x))
        x = self.Seq_deepCpf1_DO4(x)

        return self.Seq_deepCpf1_Output(x)


class _Args:
    """Minimal stand-in for the argparse.Namespace the original script builds."""

    def __init__(self, sequence_length: int = 34, kernel_size: int = 5, pool_size: int = 2):
        self.sequence_length = sequence_length
        self.kernel_size = kernel_size
        self.pool_size = pool_size


def build_deepcpf1() -> SeqDeepCpf1Net:
    args = _Args(sequence_length=34, kernel_size=5, pool_size=2)
    return SeqDeepCpf1Net(args)


def example_input_deepcpf1() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    batch = 2
    seq_len = 34
    idx = torch.randint(0, 4, (batch, seq_len))
    onehot = F.one_hot(idx, num_classes=4).float()
    x = onehot.unsqueeze(1)  # (batch, 1, seq_len, 4) for Conv2d
    ca = torch.zeros(batch)  # unused chromatin-accessibility placeholder (forward ignores it)
    return (x, ca)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("Seq-DeepCpf1", "build_deepcpf1", "example_input_deepcpf1", 2018, "vendored-pytorch"),
]
