"""The autograd.grad boundary replay is faithful to writes through views.

An in-place op on a view (``y[:, 5:6].add_(v)``) advances the label of every
storage alias it wrote through, so a later reader of the base ``y`` records the
in-place op as its parent while it read the whole post-write base. The boundary's
subgraph replay used to hand that reader the op's view-shaped output and to write
the in-place op into a clone, so the replayed root never matched its saved value
and the replay raised. The replay now writes in place as capture did and gives the
reader the replayed alias it read.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import pytest
import torch
from torch import nn

import torchlens as tl


def _base_read(y: torch.Tensor) -> torch.Tensor:
    """Write a column view in place, then read only the base."""

    y[:, 5:6].add_(0.25)
    return y


def _both_read(y: torch.Tensor) -> torch.Tensor:
    """Read the written view and the base."""

    col = y[:, 5:6].add_(0.25)
    return y * 3.0 + col


def _setitem_view(y: torch.Tensor) -> torch.Tensor:
    """Assign into a view, then read the view and the base."""

    col = y[:, 5:6]
    col[:] = col * 0.5
    return y + col


def _read_before_and_after(y: torch.Tensor) -> torch.Tensor:
    """Read the base before the write and again after it."""

    early = y * 3.0
    y[:, 5:6].add_(0.25)
    return y * early


def _intermediate_view(y: torch.Tensor) -> torch.Tensor:
    """Write a view of a view, then read the intermediate (strided) view."""

    inner = y[:, 2:8]
    inner[:, 3:4].mul_(1.5)
    return inner


def _sibling_view(y: torch.Tensor) -> torch.Tensor:
    """Read a sibling view, taken before the write, that overlaps the written view."""

    sibling = y[:, 4:7]
    y[:, 5:6].add_(0.25)
    return sibling


_WRITES: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "base_read": _base_read,
    "both_read": _both_read,
    "setitem_view": _setitem_view,
    "read_before_and_after": _read_before_and_after,
    "intermediate_view": _intermediate_view,
    "sibling_view": _sibling_view,
}


class _GradRegion(nn.Module):
    """Write through a view of an intermediate inside an ``autograd.grad`` region."""

    def __init__(self, write: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.lin = nn.Linear(8, 8)
        self.write = write

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return an energy and its gradient with respect to ``x``."""

        x = x.detach().clone().requires_grad_(True)
        read = self.write(self.lin(x) * 2.0)
        energy = (read**2).sum()
        (grad,) = torch.autograd.grad(energy, x)
        return energy, grad


def _input() -> torch.Tensor:
    return torch.randn(3, 8, generator=torch.Generator().manual_seed(0))


@pytest.mark.parametrize("write", sorted(_WRITES))
def test_view_write_then_alias_read_validates(write: str) -> None:
    """Full validation (metadata included) passes when the region reads a written alias."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert tl.validate(_GradRegion(_WRITES[write]), _input(), scope="forward")


@pytest.mark.parametrize("write", sorted(_WRITES))
def test_grad_region_gradient_matches_eager(write: str) -> None:
    """The control: the model's own gradient equals a plain eager run's."""

    model = _GradRegion(_WRITES[write])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(model, _input())
    eager_energy, eager_grad = model(_input())
    logged = [trace[label].out for label in trace.output_layers]
    assert torch.allclose(logged[0], eager_energy.detach())
    assert torch.allclose(logged[1], eager_grad)
