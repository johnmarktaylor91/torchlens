"""Shared fixtures/helpers for the weightsfree (F33 / D8) test suite.

No test functions live here. Every parity fixture constructs BOTH twins
(real + meta) back-to-back BEFORE either is captured, per the W1-ORD
discharge recipe: honest twins constructed on opposite sides of a wrap
flip refute each other on saved-reference bindings alone (weightsfree
memo L4/D8).

All spellings exercised here are DOCUMENTED-UNSTABLE pending
naming-session/S2 ratification.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torchlens as tl
from torchlens.options import CaptureOptions


def structure_options() -> CaptureOptions:
    """The one structure-only capture option state the suite uses."""

    return CaptureOptions(structure_only=True)


def weightsfree_trace(model: nn.Module, *inputs: object, **kwargs: object):
    """Capture ``model`` structure-only (the D8 power path, face 1)."""

    return tl.trace(model, *inputs, capture=structure_options(), **kwargs)


def gate_metric(real_trace, meta_trace) -> tuple[bool, object]:
    """The panel's gate assertion, using only shipped authorities.

    Returns ``(digest_equal, discharge)`` where the gate passes iff
    ``digest_equal`` is True AND ``discharge.verdict`` is CORROBORATED.
    """

    digest_equal = tl.hash.trace(real_trace) == tl.hash.trace(meta_trace)
    discharge = meta_trace.discharge_against(real_trace)
    return digest_equal, discharge


def assert_gate(real_trace, meta_trace, name: str) -> object:
    """Assert the full parity gate for one fixture; return the discharge."""

    digest_equal, discharge = gate_metric(real_trace, meta_trace)
    assert digest_equal, (
        f"[{name}] address-free graph-shape digests differ: "
        f"real={tl.hash.trace(real_trace)} meta={tl.hash.trace(meta_trace)}"
    )
    assert discharge.verdict.value == "corroborated", (
        f"[{name}] discharge verdict {discharge.verdict.value!r}; "
        f"first contradiction: {discharge.first_contradiction}"
    )
    return discharge


class FunctionalChain(nn.Module):
    """Fixture 8.1-1a: the elementwise-ref decomposition path."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.matmul(x, self.weight)
        y = F.relu(y)
        y = y + 1.0
        y = torch.softmax(y, dim=-1)
        y = F.layer_norm(y, (8,))
        return y.mean(dim=-1)


class LinearReluLinear(nn.Module):
    """Fixture 8.1-1b: 30 params, 120 fp32 geometry bytes exact."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class ConvBnPool(nn.Module):
    """Fixture 8.1-1c: buffer-holding CNN (BatchNorm running stats)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(F.relu(self.bn(self.conv(x))))
        return self.fc(torch.flatten(y, 1))


class FactoryToy(nn.Module):
    """Fixture 8.1-1d: bare factory ops + dunders (pins W1-CTX)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc(x)
        y = y + torch.ones(1, 4)
        z = torch.zeros(1, 4)
        y = 2.0 * y + z
        return F.relu(y)


class SavedRefBoundary(nn.Module):
    """Fixture 8.1-1e (module-boundary variant): pre-wrap saved reference."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 4)
        self.act = F.gelu  # saved reference bound at construction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class KwargForward(nn.Module):
    """Keyword call form (the axis that hid L3 for a round)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(6, 3)

    def forward(self, x: torch.Tensor | None = None, scale: float = 1.0) -> torch.Tensor:
        return F.relu(self.fc(x)) * scale


def build_twins(factory: type[nn.Module]) -> tuple[nn.Module, nn.Module]:
    """Construct (real, meta) twins back-to-back, both eval, per W1-ORD."""

    torch.manual_seed(0)
    real = factory()
    real.eval()
    with torch.device("meta"):
        meta = factory()
    meta.eval()
    return real, meta


def meta_like(tensor: torch.Tensor) -> torch.Tensor:
    """A meta twin of an input tensor (same shape/dtype, no storage)."""

    return torch.empty_like(tensor, device="meta")
