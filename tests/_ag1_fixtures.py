"""Tiny transformer fixtures for the attribution graph, with answer keys by plain algebra.

The model is a pre-norm transformer: token embedding plus a position
embedding, then blocks of ``x + attn(norm1(x))`` and ``x + mlp(norm2(x))``,
then a final norm and an unembedding. The attention pattern and every
normalisation scale are their own modules, so a graph can hold them at
their captured values by name.

The answer key never calls the code under test. It recomputes the forward in
plain tensor algebra from the weights, keeps the attention patterns and
normalisation scales of that forward, and then evaluates the resulting
linear map: a source's edge into a target is the target's linear read-out
(an encoder column at an MLP input, or a logit direction at the last
position) of the map applied to the source's vector alone, with every bias
set to zero. Constant terms are the same map applied to the biases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

D_MODEL = 4
N_POS = 3
VOCAB = 5
N_FEATURES = 3
EPS = 1e-6
TOKENS = (1, 3, 2)


class Scale(nn.Module):
    """The normalisation scale ``1 / rms(x)``, its own module so it can be held."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + EPS)


class RMSNorm(nn.Module):
    """``x * scale(x) * weight``."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = Scale()
        self.weight = nn.Parameter(torch.ones(D_MODEL))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale(x) * self.weight


class Pattern(nn.Module):
    """Causal softmax over attention scores."""

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        n = scores.shape[-1]
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        return torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)


class Attention(nn.Module):
    """One-head attention with a nameable pattern module."""

    def __init__(self) -> None:
        super().__init__()
        self.q = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.k = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.v = nn.Linear(D_MODEL, D_MODEL)
        self.o = nn.Linear(D_MODEL, D_MODEL)
        self.pattern = Pattern()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.q(x) @ self.k(x).transpose(-1, -2) / math.sqrt(D_MODEL)
        return self.o(self.pattern(scores) @ self.v(x))


class MLP(nn.Module):
    """``relu(x W1 + b1) W2 + b2``."""

    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Linear(D_MODEL, N_FEATURES)
        self.down = nn.Linear(N_FEATURES, D_MODEL)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(torch.relu(self.up(x)))


class Block(nn.Module):
    """Pre-norm residual block."""

    def __init__(self) -> None:
        super().__init__()
        self.norm1 = RMSNorm()
        self.attn = Attention()
        self.norm2 = RMSNorm()
        self.mlp = MLP()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class TinyTransformer(nn.Module):
    """Embedding, position embedding, blocks, final norm, unembedding."""

    def __init__(self, n_layers: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        self.pos = nn.Parameter(torch.zeros(N_POS, D_MODEL))
        self.blocks = nn.ModuleList([Block() for _ in range(n_layers)])
        self.norm = RMSNorm()
        self.unembed = nn.Linear(D_MODEL, VOCAB)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens) + self.pos
        for block in self.blocks:
            x = block(x)
        return self.unembed(self.norm(x))


def _unit_rows(gen: torch.Generator, rows: int, cols: int, norm: float) -> torch.Tensor:
    """Rows of length ``norm`` in fixed random directions."""

    value = torch.randn(rows, cols, generator=gen, dtype=torch.float64)
    return value / value.norm(dim=1, keepdim=True) * norm


def build_model(n_layers: int, dtype: torch.dtype = torch.float64) -> TinyTransformer:
    """Return the fixture model with its fixed weights.

    Every MLP-up row has norm 0.9 and every MLP input has norm at most 2
    (an RMS-normalised width-4 vector with unit gain), so a pre-activation
    moves by at most 1.8 from its bias. The biases are +3, -30, +3: features
    0 and 2 are active at every position and feature 1 never is, so each
    layer has exactly 2 * N_POS active features.
    """

    gen = torch.Generator().manual_seed(20261016)
    model = TinyTransformer(n_layers).double()
    with torch.no_grad():
        model.embed.weight.copy_(torch.randn(VOCAB, D_MODEL, generator=gen, dtype=torch.float64))
        model.pos.copy_(0.3 * torch.randn(N_POS, D_MODEL, generator=gen, dtype=torch.float64))
        for block in model.blocks:
            attn = block.attn
            for linear in (attn.q, attn.k, attn.v, attn.o):
                linear.weight.copy_(
                    0.5 * torch.randn(D_MODEL, D_MODEL, generator=gen, dtype=torch.float64)
                )
            attn.v.bias.copy_(0.1 * torch.randn(D_MODEL, generator=gen, dtype=torch.float64))
            attn.o.bias.copy_(0.1 * torch.randn(D_MODEL, generator=gen, dtype=torch.float64))
            block.mlp.up.weight.copy_(_unit_rows(gen, N_FEATURES, D_MODEL, 0.9))
            block.mlp.up.bias.copy_(torch.tensor([3.0, -30.0, 3.0], dtype=torch.float64))
            block.mlp.down.weight.copy_(
                0.5 * torch.randn(D_MODEL, N_FEATURES, generator=gen, dtype=torch.float64)
            )
            block.mlp.down.bias.copy_(
                0.1 * torch.randn(D_MODEL, generator=gen, dtype=torch.float64)
            )
        model.unembed.weight.copy_(torch.randn(VOCAB, D_MODEL, generator=gen, dtype=torch.float64))
        model.unembed.bias.copy_(0.1 * torch.randn(VOCAB, generator=gen, dtype=torch.float64))
    return model.to(dtype).eval()


def frozen_names(n_layers: int) -> list[str]:
    """Module names of every attention pattern and normalisation scale."""

    names = []
    for layer in range(n_layers):
        names += [
            f"blocks.{layer}.norm1.scale",
            f"blocks.{layer}.attn.pattern",
            f"blocks.{layer}.norm2.scale",
        ]
    return [*names, "norm.scale"]


class Transcoder:
    """A transcoder in the SAE Lens orientation; ``W_dec``/``b_dec`` may be per MLP."""

    def __init__(
        self,
        W_enc: torch.Tensor,
        b_enc: torch.Tensor,
        W_dec: torch.Tensor | dict[str, torch.Tensor],
        b_dec: torch.Tensor | dict[str, torch.Tensor],
    ) -> None:
        self.W_enc = W_enc
        self.b_enc = b_enc
        self.W_dec = W_dec
        self.b_dec = b_dec

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x @ self.W_enc + self.b_enc)


def exact_transcoders(model: TinyTransformer) -> dict[str, Transcoder]:
    """F1: per-layer transcoders equal to each MLP (zero reconstruction error)."""

    return {
        f"blocks.{layer}.mlp": Transcoder(
            block.mlp.up.weight.detach().T.clone(),
            block.mlp.up.bias.detach().clone(),
            block.mlp.down.weight.detach().T.clone(),
            block.mlp.down.bias.detach().clone(),
        )
        for layer, block in enumerate(model.blocks)
    }


def lossy_transcoders(model: TinyTransformer) -> dict[str, Transcoder]:
    """F2: per-layer transcoders that reconstruct imperfectly (scaled decoder, shifted bias)."""

    return {
        name: Transcoder(tc.W_enc, tc.b_enc, 0.8 * tc.W_dec, tc.b_dec + 0.1)  # type: ignore[operator]
        for name, tc in exact_transcoders(model).items()
    }


def cross_layer_transcoders(model: TinyTransformer) -> dict[str, Transcoder]:
    """F3: a cross-layer transcoder; the layer-l reader writes every MLP output from l on."""

    gen = torch.Generator().manual_seed(44)
    n_layers = len(model.blocks)
    dtype = model.embed.weight.dtype
    out: dict[str, Transcoder] = {}
    for layer, block in enumerate(model.blocks):
        writes = [f"blocks.{w}.mlp" for w in range(layer, n_layers)]
        out[f"blocks.{layer}.mlp"] = Transcoder(
            block.mlp.up.weight.detach().T.clone(),
            block.mlp.up.bias.detach().clone(),
            {
                w: (0.4 * torch.randn(N_FEATURES, D_MODEL, generator=gen, dtype=torch.float64)).to(
                    dtype
                )
                for w in writes
            },
            {
                w: (0.05 * torch.randn(D_MODEL, generator=gen, dtype=torch.float64)).to(dtype)
                for w in writes
            },
        )
    return out


@dataclass
class Frozen:
    """The plain-algebra forward and the quantities the frozen graph holds."""

    resid0: torch.Tensor
    pos: torch.Tensor
    patterns: list[torch.Tensor]
    scales1: list[torch.Tensor]
    scales2: list[torch.Tensor]
    scale_final: torch.Tensor
    mlp_in: list[torch.Tensor]
    mlp_out: list[torch.Tensor]
    logits: torch.Tensor


def _w(linear: nn.Linear) -> torch.Tensor:
    return linear.weight.detach().double()


def _b(linear: nn.Linear) -> torch.Tensor:
    assert linear.bias is not None
    return linear.bias.detach().double()


def plain_forward(model: TinyTransformer, tokens: tuple[int, ...] = TOKENS) -> Frozen:
    """The model's forward in plain float64 algebra, keeping what is frozen."""

    emb = model.embed.weight.detach().double()[list(tokens)]
    pos = model.pos.detach().double()
    resid = emb + pos
    patterns, s1s, s2s, ins, outs = [], [], [], [], []
    n = len(tokens)
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    for block in model.blocks:
        s1 = torch.rsqrt((resid * resid).mean(-1, keepdim=True) + EPS)
        h = resid * s1 * block.norm1.weight.detach().double()
        a = block.attn
        scores = (h @ _w(a.q).T) @ (h @ _w(a.k).T).T / math.sqrt(D_MODEL)
        pattern = torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)
        resid = resid + (pattern @ (h @ _w(a.v).T + _b(a.v))) @ _w(a.o).T + _b(a.o)
        s2 = torch.rsqrt((resid * resid).mean(-1, keepdim=True) + EPS)
        x = resid * s2 * block.norm2.weight.detach().double()
        out = torch.relu(x @ _w(block.mlp.up).T + _b(block.mlp.up)) @ _w(block.mlp.down).T
        out = out + _b(block.mlp.down)
        resid = resid + out
        patterns.append(pattern)
        s1s.append(s1)
        s2s.append(s2)
        ins.append(x)
        outs.append(out)
    sf = torch.rsqrt((resid * resid).mean(-1, keepdim=True) + EPS)
    logits = (resid * sf * model.norm.weight.detach().double()) @ _w(model.unembed).T
    logits = logits + _b(model.unembed)
    return Frozen(emb, pos, patterns, s1s, s2s, sf, ins, outs, logits)


def linear_map(
    model: TinyTransformer,
    frozen: Frozen,
    resid0: torch.Tensor,
    mlp_writes: dict[int, torch.Tensor],
    *,
    biases: bool,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """The frozen linear graph: MLP inputs per layer and logits from injected vectors.

    ``resid0`` enters the residual stream before the first block and
    ``mlp_writes[l]`` replaces the output of MLP ``l``; patterns and scales
    are the captured ones. With ``biases=False`` every model bias and the
    position embedding are zero, so the map is linear in its inputs.
    """

    on = 1.0 if biases else 0.0
    resid = resid0 + on * frozen.pos
    ins = []
    for layer, block in enumerate(model.blocks):
        h = resid * frozen.scales1[layer] * block.norm1.weight.detach().double()
        a = block.attn
        values = h @ _w(a.v).T + on * _b(a.v)
        resid = resid + (frozen.patterns[layer] @ values) @ _w(a.o).T + on * _b(a.o)
        ins.append(resid * frozen.scales2[layer] * block.norm2.weight.detach().double())
        resid = resid + mlp_writes.get(layer, torch.zeros_like(resid))
    h = resid * frozen.scale_final * model.norm.weight.detach().double()
    return ins, h @ _w(model.unembed).T + on * _b(model.unembed)
