"""Shared fixture builders for the W051-AGENT regression suites (no tests here).

Every builder is DETERMINISTIC (explicit seeds, eval mode). ``MiniTransformer``
is the REALISTIC fixture AUD-CODE 4.9 asked for: the registry/dump/query
fixtures were all <= 9 ops, so paging, backstop, and fold behaviour past a
handful of rows was never exercised. Four hand-rolled attention blocks
(no fused ``nn.MultiheadAttention`` fast path) capture as ~108 ops in ~2 s.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

import torchlens as tl


class Block(nn.Module):
    """One pre-norm attention + MLP block, written out op by op."""

    def __init__(self, d: int, h: int) -> None:
        """Build the block's linears and norms."""

        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)
        self.h = h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attention then MLP, both residual."""

        b, t, d = x.shape
        y = self.ln1(x)
        q = self.q(y).view(b, t, self.h, d // self.h).transpose(1, 2)
        k = self.k(y).view(b, t, self.h, d // self.h).transpose(1, 2)
        v = self.v(y).view(b, t, self.h, d // self.h).transpose(1, 2)
        att = torch.softmax(q @ k.transpose(-2, -1) / (d // self.h) ** 0.5, dim=-1)
        y = (att @ v).transpose(1, 2).reshape(b, t, d)
        x = x + self.o(y)
        return x + self.fc2(torch.nn.functional.gelu(self.fc1(self.ln2(x))))


class MiniTransformer(nn.Module):
    """Four repeated blocks + head; returns logits AND a scalar (rank-0) loss."""

    def __init__(self, n_blocks: int = 4, d: int = 16, h: int = 2, vocab: int = 32) -> None:
        """Build the deterministic stack."""

        super().__init__()
        torch.manual_seed(0)
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(6, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(n_blocks)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed, run the blocks, project; the second output is a scalar."""

        x = self.tok(ids) + self.pos(torch.arange(ids.shape[1]))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        return logits, logits.float().logsumexp(-1).mean()


class Loop(nn.Module):
    """One module called three times: a multi-pass (recurrence-grouped) trace."""

    def __init__(self) -> None:
        """Build the shared cell."""

        super().__init__()
        torch.manual_seed(0)
        self.cell = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Three passes through the same cell; output shape equals input shape."""

        for _ in range(3):
            x = torch.tanh(self.cell(x))
        return x


def mini_ids() -> torch.Tensor:
    """The one token-id input every realistic artifact shares."""

    return torch.arange(6).unsqueeze(0)


def save_realistic_artifact(directory: Path, *, save_all: bool = False) -> Path:
    """Capture and save the ~108-op MiniTransformer fixture artifact.

    Parameters
    ----------
    directory:
        Directory to write into.
    save_all:
        Retain every payload (for stats over heterogeneous ranks) instead of
        the softmax/gelu subset.

    Returns
    -------
    Path
        The saved ``.tlspec`` path.
    """

    model = MiniTransformer().eval()
    if save_all:
        log = tl.trace(model, mini_ids(), capture=tl.options.CaptureOptions(layers_to_save="all"))
    else:
        log = tl.trace(model, mini_ids(), save=tl.func("softmax") | tl.func("gelu"))
    path = directory / ("mini_all.tlspec" if save_all else "mini.tlspec")
    tl.save(log, str(path))
    return path


def save_loop_artifact(directory: Path) -> Path:
    """Capture and save the multi-pass Loop fixture artifact."""

    generator = torch.Generator().manual_seed(1234)
    log = tl.trace(Loop().eval(), torch.randn(2, 8, generator=generator))
    path = directory / "loop.tlspec"
    tl.save(log, str(path))
    return path
