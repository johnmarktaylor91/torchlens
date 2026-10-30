"""Seeded byte decoder with alternating causal convolution and attention."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

WIDTH = 64
VOCAB = 256
MAX_POSITIONS = 512


def parameter(rng, shape, scale):
    values = rng.standard_normal(shape).astype(np.float32) * np.float32(scale)
    return nn.Parameter(torch.from_numpy(np.asarray(values)))


class RMSNorm(nn.Module):
    def __init__(self, rng, spread=0.15):
        super().__init__()
        self.weight = nn.Parameter(torch.from_numpy(rng.lognormal(0.0, spread, WIDTH).astype(np.float32)))

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5) * self.weight


class ConvBlock(nn.Module):
    def __init__(self, rng):
        super().__init__()
        self.norm = RMSNorm(rng)
        self.depthwise = parameter(rng, (WIDTH, 1, 3), 3 ** -0.5)
        self.projection = parameter(rng, (WIDTH, WIDTH), WIDTH ** -0.5)
        self.gate = parameter(rng, (), 0.5)

    def forward(self, x):
        normalized = self.norm(x).T.unsqueeze(0)
        mixed = F.conv1d(F.pad(normalized, (2, 0)), self.depthwise, groups=WIDTH)
        branch = F.linear(F.silu(mixed.squeeze(0).T), self.projection)
        return x + self.gate.sigmoid() * branch


class AttentionBlock(nn.Module):
    def __init__(self, rng):
        super().__init__()
        self.norm = RMSNorm(rng)
        self.qkv = parameter(rng, (3 * WIDTH, WIDTH), WIDTH ** -0.5)
        self.projection = parameter(rng, (WIDTH, WIDTH), WIDTH ** -0.5)
        self.gate = parameter(rng, (), 0.5)

    def forward(self, x):
        length = x.shape[0]
        q, k, v = F.linear(self.norm(x), self.qkv).chunk(3, -1)
        q, k, v = [t.reshape(length, 4, 16).transpose(0, 1) for t in (q, k, v)]
        scores = q @ k.transpose(-1, -2) / 4.0
        future = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
        weights = scores.masked_fill(future, torch.finfo(x.dtype).min).softmax(-1)
        joined = (weights @ v).transpose(0, 1).reshape(length, WIDTH)
        return x + self.gate.sigmoid() * F.linear(joined, self.projection)


class ByteDecoder(nn.Module):
    def __init__(self, seed):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.token_embedding = parameter(rng, (VOCAB, WIDTH), 0.2)
        self.position_embedding = parameter(rng, (MAX_POSITIONS, WIDTH), 0.1)
        self.blocks = nn.ModuleList([ConvBlock(rng) if i % 2 == 0 else AttentionBlock(rng)
                                     for i in range(6)])
        self.final_norm = RMSNorm(rng, spread=0.7)
        self.readout = parameter(rng, (VOCAB, WIDTH), WIDTH ** -0.5)

    def embed(self, ids):
        if ids.ndim != 1 or not 0 < len(ids) <= MAX_POSITIONS:
            raise ValueError("expected a nonempty byte sequence of at most 512 positions")
        return F.embedding(ids, self.token_embedding) + self.position_embedding[:len(ids)]

    def states(self, ids):
        state, recorded = self.embed(ids), []
        for block in self.blocks:
            state = block(state)
            recorded.append(state)
        return recorded

    def unembed(self, state):
        return F.linear(state, self.readout)

    def forward(self, ids):
        return self.unembed(self.final_norm(self.states(ids)[-1]))


def build_model(weight_seed):
    return ByteDecoder(weight_seed).eval()
