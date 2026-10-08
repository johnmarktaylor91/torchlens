"""Small CPU architectures for state-leak probes."""

from __future__ import annotations

import math

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))


class ConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(8, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.bn(self.conv1(x)))
        h = torch.relu(self.conv2(h))
        return self.head(self.pool(h).flatten(1))


class Block(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.d = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(d)
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(mask, float("-inf")).softmax(-1)
        x = x + self.proj(att @ v)
        return x + self.mlp(self.ln2(x))


class TinyDecoder(nn.Module):
    def __init__(self, vocab: int = 32, d: int = 16, n_layers: int = 2) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d) for _ in range(n_layers)])
        self.lnf = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.emb(ids)
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(self.lnf(x))


def build(kind: str, seed: int = 0) -> tuple[nn.Module, torch.Tensor, torch.Tensor, str, int]:
    """Return (model, x, x2, site_module_address, site_feature_dim)."""

    torch.manual_seed(seed)
    if kind == "mlp":
        model = MLP().eval()
        x, x2 = torch.randn(3, 8), torch.randn(3, 8)
        return model, x, x2, "fc2", 16
    if kind == "conv":
        model = ConvNet().eval()
        x, x2 = torch.randn(2, 3, 6, 6), torch.randn(2, 3, 6, 6)
        return model, x, x2, "conv2", 8
    if kind == "decoder":
        model = TinyDecoder().eval()
        g = torch.Generator().manual_seed(seed + 1)
        x = torch.randint(0, 32, (2, 5), generator=g)
        x2 = torch.randint(0, 32, (2, 5), generator=g)
        return model, x, x2, "blocks.0", 16
    raise ValueError(kind)
