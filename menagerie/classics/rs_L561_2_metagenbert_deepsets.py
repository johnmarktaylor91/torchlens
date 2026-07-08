# SOURCE: vendored from CorvusVaine/MetagenBERT @ main (DeepSets.py)
"""MetagenBERT: attention-pooled DeepSets multiple-instance-learning (MIL)
classifier over per-read DNA embeddings, used to aggregate metagenomic
shotgun-sequencing reads (embedded upstream by a separately pretrained
DNABERT-2-style encoder, see embedding-dna2-ordered.py in the same repo)
into a per-sample phenotype prediction (e.g. cirrhosis / T2D classification).

``DeepSets`` (permutation-invariant set encoder: per-instance ``Phi`` MLP ->
pooling (sum/max/mean/attention MIL) -> set-level ``Rho`` MLP) is the actual
distinctive architecture this repo contributes; the DNA-embedding upstream
model is a separately pretrained/loaded HF checkpoint (not part of this
repo's own architecture code), so it is intentionally out of scope here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# DeepSets.py (verbatim: DeepSets / Phi / Rho)
# ---------------------------------------------------------------------------
class DeepSets(nn.Module):
    def __init__(self, phi, rho, mil_layer, device):
        super(DeepSets, self).__init__()
        self.phi = phi
        self.rho = rho
        self.mil_layer = mil_layer
        self.device = device
        if mil_layer == "attention":
            self.attention = nn.Sequential(
                nn.Linear(self.phi.last_hidden_size, self.phi.last_hidden_size // 3),
                nn.Tanh(),
                nn.Linear(self.phi.last_hidden_size // 3, 1),
            ).to(self.device)
        self.criterion = (
            nn.BCEWithLogitsLoss() if self.rho.output_size <= 2 else nn.CrossEntropyLoss()
        )

    def forward(self, x):
        # compute the representation for each data point
        x = self.phi.forward(x)
        A = None
        # sum up the representations
        if self.mil_layer == "sum":
            x = torch.sum(x, dim=1, keepdim=True)
        if self.mil_layer == "max":
            x = torch.max(x, dim=1, keepdim=True)[0]
        if self.mil_layer == "mean":
            x = torch.mean(x, dim=1, keepdim=True)
        if self.mil_layer == "attention":
            A = self.attention(x)
            A = F.softmax(A, dim=1)
            x = torch.bmm(torch.transpose(A, 2, 1), x)
        # compute the output
        out = self.rho.forward(x)
        return out, A


class Phi(nn.Module):
    def __init__(self, embed_size, hidden_init=200, n_layer=1, dropout=0.2):
        super(Phi, self).__init__()
        layer_size = [embed_size, hidden_init]
        n_layer -= 1
        for i in range(n_layer):
            hidden_init = hidden_init // 2
            layer_size.append(hidden_init)
        self.layers = []
        for i in range(len(layer_size) - 1):
            self.layers.append(nn.Linear(layer_size[i], layer_size[i + 1]))
            self.layers.append(nn.LeakyReLU())
            self.layers.append(nn.Dropout(dropout))
        self.nets = nn.Sequential(*self.layers[:-1])  # Remove the last drop out
        self.last_hidden_size = layer_size[-1]

    def forward(self, x):
        return self.nets(x)


class Rho(nn.Module):
    def __init__(self, phi_hidden_size, hidden_init=100, n_layer=1, dropout=0.2, output_size=1):
        super(Rho, self).__init__()
        self.output_size = output_size
        layer_size = [phi_hidden_size, hidden_init]
        n_layer -= 1
        for i in range(n_layer):
            hidden_init = hidden_init // 2
            layer_size.append(hidden_init)
        self.layers = []
        for i in range(len(layer_size) - 1):
            self.layers.append(nn.Linear(layer_size[i], layer_size[i + 1]))
            self.layers.append(nn.LeakyReLU())
            self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.Linear(layer_size[-1], output_size))
        self.nets = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.nets(x)


# ---------------------------------------------------------------------------
# torchlens returns (out, A) as a tuple; wrap so forward() returns a plain
# tensor for a clean single-output capture (real repo's forward also
# returns this tuple, unchanged here beyond unpacking for capture).
# ---------------------------------------------------------------------------
class DeepSetsTraceWrapper(nn.Module):
    def __init__(self, model: DeepSets):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _attn = self.model(x)
        return out


_EMBED_SIZE = 16
_N_READS = 6
_BATCH = 2


def build_deepsets() -> DeepSetsTraceWrapper:
    phi = Phi(embed_size=_EMBED_SIZE, hidden_init=32, n_layer=2, dropout=0.0)
    rho = Rho(phi.last_hidden_size, hidden_init=16, n_layer=1, dropout=0.0, output_size=1)
    model = DeepSets(phi, rho, mil_layer="attention", device="cpu")
    model.eval()
    return DeepSetsTraceWrapper(model)


def example_input_deepsets() -> torch.Tensor:
    return torch.randn(_BATCH, _N_READS, _EMBED_SIZE)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "MetagenBERT-DeepSets",
        "build_deepsets",
        "example_input_deepsets",
        2024,
        "vendored-pytorch",
    ),
]
