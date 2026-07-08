# SOURCE: vendored from QijinYin/DeepHistone @ master
# https://raw.githubusercontent.com/QijinYin/DeepHistone/master/model.py
#
# DeepHistone (2018) -- dual-branch DenseNet-style CNN that jointly encodes a
# one-hot DNA sequence window and a chromatin-accessibility (DNase) track
# through two independent `ModuleDense` towers (each: a stem conv + two dense
# blocks with 1x9 growth convolutions + BN-ReLU-conv transition/pool stages),
# concatenates the flattened features, and predicts 7 histone-modification
# marks with a small MLP head. `NetDeepHistone` (the real nn.Module) is
# reproduced verbatim below; only the thin `DeepHistone` training-harness
# wrapper class (optimizer/criterion/train_on_batch/etc., not an nn.Module)
# is dropped since it adds no architecture.

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# ============================================================================
# model.py (verbatim, training-harness wrapper class omitted)
# ============================================================================


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_planes,
        grow_rate,
    ):
        super(BasicBlock, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.ReLU(),
            nn.Conv2d(in_planes, grow_rate, (1, 9), 1, (0, 4)),
            # nn.Dropout2d(0.2)
        )

    def forward(self, x):
        out = self.block(x)
        return torch.cat([x, out], 1)


class DenseBlock(nn.Module):
    def __init__(
        self,
        nb_layers,
        in_planes,
        grow_rate,
    ):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(nb_layers):
            layers.append(
                BasicBlock(
                    in_planes + i * grow_rate,
                    grow_rate,
                )
            )
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class ModuleDense(nn.Module):
    def __init__(
        self,
        SeqOrDnase="seq",
    ):
        super(ModuleDense, self).__init__()
        self.SeqOrDnase = SeqOrDnase
        if self.SeqOrDnase == "seq":
            self.conv1 = nn.Sequential(
                nn.Conv2d(1, 128, (4, 9), 1, (0, 4)),
                # nn.Dropout2d(0.2),
            )
        elif self.SeqOrDnase == "dnase":
            self.conv1 = nn.Sequential(
                nn.Conv2d(1, 128, (1, 9), 1, (0, 4)),
                # nn.Dropout2d(0.2),
            )
        self.block1 = DenseBlock(3, 128, 128)
        self.trans1 = nn.Sequential(
            nn.BatchNorm2d(128 + 3 * 128),
            nn.ReLU(),
            nn.Conv2d(128 + 3 * 128, 256, (1, 1), 1),
            # nn.Dropout2d(0.2),
            nn.MaxPool2d((1, 4)),
        )
        self.block2 = DenseBlock(3, 256, 256)
        self.trans2 = nn.Sequential(
            nn.BatchNorm2d(256 + 3 * 256),
            nn.ReLU(),
            nn.Conv2d(256 + 3 * 256, 512, (1, 1), 1),
            # nn.Dropout2d(0.2),
            nn.MaxPool2d((1, 4)),
        )
        self.out_size = 1000 // 4 // 4 * 512

    def forward(self, seq):
        n, h, w = seq.size()
        if self.SeqOrDnase == "seq":
            seq = seq.view(n, 1, 4, w)
        elif self.SeqOrDnase == "dnase":
            seq = seq.view(n, 1, 1, w)
        out = self.conv1(seq)
        out = self.block1(out)
        out = self.trans1(out)
        out = self.block2(out)
        out = self.trans2(out)
        n, c, h, w = out.size()
        out = out.view(n, c * h * w)
        return out


class NetDeepHistone(nn.Module):
    def __init__(
        self,
    ):
        super(NetDeepHistone, self).__init__()
        self.seq_map = ModuleDense(
            SeqOrDnase="seq",
        )
        self.seq_len = self.seq_map.out_size
        self.dns_map = ModuleDense(
            SeqOrDnase="dnase",
        )
        self.dns_len = self.dns_map.out_size
        combined_len = self.dns_len + self.seq_len
        self.linear_map = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(int(combined_len), 925),
            nn.BatchNorm1d(925),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(925, 7),
            nn.Sigmoid(),
        )

    def forward(self, seq, dns):
        flat_seq = self.seq_map(seq)
        n, h, w = dns.size()
        dns = self.dns_map(dns)
        flat_dns = dns.view(n, -1)
        combined = torch.cat([flat_seq, flat_dns], 1)
        out = self.linear_map(combined)
        return out


# ============================================================================
# build_/example_input_ harness
# ============================================================================


def build_deephistone():
    model = NetDeepHistone()
    model.eval()
    return model


def example_input_deephistone():
    torch.manual_seed(0)
    batch, length = 2, 1000
    # seq: one-hot DNA (4 rows: A/C/G/T) x 1000bp window, real repo reshapes
    # (n, h, w) -> (n, 1, 4, w) inside ModuleDense.forward.
    seq = torch.rand(batch, 4, length)
    # dns: single-row chromatin-accessibility (DNase) track, same window.
    dns = torch.rand(batch, 1, length)
    return (seq, dns)


MENAGERIE_ENTRIES = [
    ("DeepHistone", build_deephistone, example_input_deephistone, 2018, "vendored-pytorch"),
]
