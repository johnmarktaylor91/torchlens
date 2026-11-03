"""The deterministic nonlinear edge-attribution fixture, shared by the EAP and audit tests.

The edge-attribution bars fixture as an ``nn.Module``: token 0 embeds
``e_clean`` and token 1 ``e_corrupt`` (the ``input`` node); the senders ``h1``
and ``h2`` are two bias-free one-layer MLPs that both read the embedding
through their own identity pre-norm (graph names ``m0`` and ``m1``); the
receiver ``R`` is the state entering the identity final norm before the
unembedding ``V`` (graph name ``logits``); the metric is ``M = sum_j w_j
tanh((V r)_j)`` at the last position. The identity norms carry an RMSNorm
class name so the graph builder classifies them as the pre-norms they stand
in for. Float64 throughout.
"""

from __future__ import annotations

import torch
from torch import nn


class FixtureEmbedding(nn.Embedding):
    """Token 0 embeds ``e_clean``, token 1 ``e_corrupt``."""


class FixtureRMSNorm(nn.Module):
    """An identity pre-norm (the fixture reads the residual state unnormalized)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 1.0


class FixtureMLP(nn.Module):
    """A bias-free linear sender ``u = A e``."""

    def __init__(self, a: list[list[float]]) -> None:
        super().__init__()
        self.proj = nn.Linear(2, 2, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.proj.weight.copy_(torch.tensor(a, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Fixture(nn.Module):
    """``r = e + A1 e + A2 e``; logits ``V r`` (the bar's fixture as a module)."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = FixtureEmbedding(2, 2, dtype=torch.float64)
        with torch.no_grad():
            self.embed.weight.copy_(torch.tensor([[1.0, 0.5], [-1.0, 0.25]]))
        self.norm1, self.norm2, self.norm_f = FixtureRMSNorm(), FixtureRMSNorm(), FixtureRMSNorm()
        self.h1 = FixtureMLP([[1.5, 0.0], [0.0, 0.0]])
        self.h2 = FixtureMLP([[0.0, 0.0], [0.5, 2.0]])
        self.lm_head = nn.Linear(2, 2, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.tensor([[1.0, 0.5], [-0.5, 1.0]]))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        e = self.embed(ids)
        r = e + self.h1(self.norm1(e)) + self.h2(self.norm2(e))
        return self.lm_head(self.norm_f(r))


def fixture_metric(logits: torch.Tensor) -> torch.Tensor:
    """``M = 1.0 tanh(l_0) + 2.0 tanh(l_1)`` at the last position, per example."""

    return 1.0 * torch.tanh(logits[:, -1, 0]) + 2.0 * torch.tanh(logits[:, -1, 1])


CLEAN, CORRUPT = torch.tensor([[0]]), torch.tensor([[1]])
