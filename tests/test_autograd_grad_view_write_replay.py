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
from torchlens.validation._replay_storage_context import replayed_storage_alias


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
    return y * 3.0 + col


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
def test_view_write_then_alias_read_replays(write: str) -> None:
    """The boundary replay reproduces the region's gradient when it reads a written alias."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert tl.validate(
            _GradRegion(_WRITES[write]), _input(), scope="forward", validate_metadata=False
        )


@pytest.mark.parametrize("write", ["both_read", "setitem_view"])
def test_view_write_then_alias_read_validates(write: str) -> None:
    """Full validation (metadata included) passes when the written view is read too."""

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


def test_alias_search_prefers_the_receiver_lineage() -> None:
    """A lineage match wins over an equal-geometry tensor elsewhere in the replay."""

    base = torch.zeros(3, 8)
    written = base[:, 5:6]
    other = base.view(3, 8)
    found = replayed_storage_alias(written, ((3, 8), None), [written, base], [other])
    assert found is base


def test_alias_search_refuses_an_ambiguous_read() -> None:
    """Two contiguous aliases of one shape at different offsets cannot be told apart."""

    flat = torch.zeros(16, requires_grad=True).clone()
    written = flat[5:7]
    first, second = flat[0:8], flat[4:12]
    with pytest.raises(ValueError, match="ambiguous"):
        replayed_storage_alias(written, ((8,), None), [written], [first, second])


def test_alias_search_ignores_other_storage_and_geometry() -> None:
    """Only a tensor on the written storage with exactly the read geometry matches."""

    base = torch.zeros(3, 8)
    written = base[:, 5:6]
    foreign = torch.zeros(3, 8)
    strided = base[:, 4:7]
    assert replayed_storage_alias(written, ((3, 8), None), [written], [foreign]) is None
    assert replayed_storage_alias(written, ((3, 3), ((3, 3), (8, 1), 2)), [], [strided]) is None
    assert replayed_storage_alias(written, ((3, 3), ((3, 3), (8, 1), 4)), [], [strided]) is strided
