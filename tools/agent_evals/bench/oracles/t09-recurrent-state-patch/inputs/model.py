"""Seeded two-layer recurrent language model and ordered sequence pairs."""
import numpy as np
import torch
from torch import nn


class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(64, 32)
        self.gru = nn.GRU(32, 48, num_layers=2, batch_first=True)
        self.readout = nn.Linear(48, 64)

    def forward(self, tokens, state=None):
        residual, state = self.gru(self.embedding(tokens), state)
        return self.readout(residual), state


def build_model(seed):
    rng = np.random.default_rng(seed)
    model = LanguageModel()
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            scale = 0.08 if 'bias' in name else 1 / np.sqrt(param.shape[-1])
            param.copy_(torch.from_numpy((rng.standard_normal(param.shape) * scale).astype(np.float32)))
    return model.eval()


def make_pairs(seed, n_pairs, length=24):
    rng = np.random.default_rng(seed)
    clean = rng.integers(0, 64, (n_pairs, length), dtype=np.int64)
    corrupt = (clean + rng.integers(1, 64, clean.shape, dtype=np.int64)) % 64
    return clean, corrupt
