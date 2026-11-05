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
from torchlens.validation import _autograd_grad_replay as grad_replay
from torchlens.validation._core import _op_for_validation_label
from torchlens.validation._replay_storage_context import (
    alias_provenance_holds,
    alias_value_holds,
    replayed_storage_alias,
    writes_through_view,
)


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


def _whole_write(y: torch.Tensor) -> torch.Tensor:
    """Write the whole (non-view) intermediate in place: replayed into a clone, as before."""

    early = y * 3.0
    y.add_(0.25)
    return y + early


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
    "whole_write": _whole_write,
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
    """Return the fixed ``(3, 8)`` input every region in this file is traced on."""

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


def _two_bases(y: torch.Tensor) -> torch.Tensor:
    """Write views of two same-shape bases; the second base's reader adds nothing to the root."""

    z = y * 1.5
    z[:, 1:2].add_(0.5)
    y[:, 5:6].add_(0.25)
    return z + y * 0.0


def _narrow_base(y: torch.Tensor) -> torch.Tensor:
    """Write a view of a narrower base; the full base's reader adds nothing to the root."""

    w = y[:, :4] * 1.5
    w[:, 0:1].add_(0.5)
    y[:, 5:6].add_(0.25)
    return w.sum() + (y * 0.0).sum()


def _plant_wrong_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the grad replay read the zero-weighted reader's parent as the FIRST view write.

    A corrupted record: the reader really read the base the second write reached.
    """

    original = grad_replay.record_member

    def planted(obj: object, name: str, *default: object) -> object:
        """Return the recorded member, with the planted parent for the zero-weighted reader."""

        value = original(obj, name, *default)
        if name != "parent_arg_positions" or "mul" not in str(getattr(obj, "func_name", "")):
            return value
        trace = obj._source_trace  # type: ignore[attr-defined]
        first, last = [op for op in trace.ops if op.is_inplace][-2:]
        receiver = (value or {}).get("args", {}).get(0)
        if receiver is None or _op_for_validation_label(trace, receiver).label != last.label:
            return value
        return {**value, "args": {**value["args"], 0: first.label}}

    monkeypatch.setattr(grad_replay, "record_member", planted)


@pytest.mark.parametrize("write", ["two_bases", "narrow_base"])
def test_uncorrupted_control_validates(write: str) -> None:
    """The control for the planted records: the honest record of each region validates."""

    region = _GradRegion({"two_bases": _two_bases, "narrow_base": _narrow_base}[write])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert tl.validate(region, _input(), scope="forward", validate_metadata=False)


@pytest.mark.parametrize("write", ["two_bases", "narrow_base"])
def test_planted_wrong_parent_fails_validation(write: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A reader recorded under the wrong in-place parent fails even when the root hides it.

    ``two_bases``: the wrong parent's lineage holds a same-shape alias whose value
    is not what the reader read. ``narrow_base``: no alias has the shape read.
    Both used to validate, since the misread reader adds nothing to the root.
    """

    _plant_wrong_parent(monkeypatch)
    region = _GradRegion({"two_bases": _two_bases, "narrow_base": _narrow_base}[write])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert not tl.validate(region, _input(), scope="forward", validate_metadata=False)


def test_alias_value_check_compares_the_saved_read() -> None:
    """The hand-off is checked against the slot's saved value, NaN-aware, shape first."""

    base = torch.tensor([[1.0, float("nan")], [3.0, 4.0]])
    assert alias_value_holds(base, base.clone())
    assert not alias_value_holds(base, base + 1.0)
    assert not alias_value_holds(base, base[:1].clone())
    assert not alias_value_holds(base, None)


def test_alias_provenance_needs_the_written_storage_and_recorded_layout() -> None:
    """A perturbed replay accepts only an alias on the written storage at the recorded layout."""

    base = torch.zeros(3, 8)
    written = base[:, 5:6]
    strided = base[:, 4:7]
    assert alias_provenance_holds(base, written, ((3, 8), None))
    assert not alias_provenance_holds(torch.zeros(3, 8), written, ((3, 8), None))
    assert not alias_provenance_holds(strided, written, ((3, 3), None))
    assert alias_provenance_holds(strided, written, ((3, 3), ((3, 3), (8, 1), 4)))
    assert not alias_provenance_holds(strided, written, ((3, 3), ((3, 3), (8, 1), 2)))


def test_only_a_view_receiver_is_written_in_place() -> None:
    """Writes through a view are replayed in place; every other in-place slot keeps a clone."""

    leaf = torch.zeros(3, 8, requires_grad=True)
    base = leaf * 2.0
    view = base[:, 5:6]
    assert writes_through_view(("args", 0), view, frozenset())
    assert not writes_through_view(("args", 0), base, frozenset())
    assert not writes_through_view(("args", 1), view, frozenset())
    assert not writes_through_view(("kwargs", "other"), view, frozenset())
    assert not writes_through_view(("args", 0), leaf, frozenset())
    enclosing = frozenset({(base.device, base.untyped_storage().data_ptr())})
    assert not writes_through_view(("args", 0), view, enclosing)


class _NestedGradRegion(nn.Module):
    """An inner ``autograd.grad`` region that writes through a view, inside an outer one."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.lin = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return an energy built on an inner gradient and its gradient with respect to ``x``."""

        x = x.detach().clone().requires_grad_(True)
        h = self.lin(x) * 2.0
        inner = _base_read(h * 1.5)
        (g,) = torch.autograd.grad((inner**2).sum(), h, create_graph=True)
        energy = (g**2).sum() + (h**2).sum()
        (grad,) = torch.autograd.grad(energy, x)
        return energy, grad


def test_nested_view_write_replays() -> None:
    """A view write inside a nested replay replays, and its control matches eager."""

    model = _NestedGradRegion()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(model, _input())
        assert tl.validate(model, _input(), scope="forward", validate_metadata=False)
    eager_energy, eager_grad = model(_input())
    logged = [trace[label].out for label in trace.output_layers]
    assert torch.allclose(logged[0], eager_energy.detach())
    assert torch.allclose(logged[1], eager_grad)


def test_nested_replay_never_writes_into_the_enclosing_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every storage an in-place write reaches in a nested replay is the nested replay's own."""

    seen: list[tuple[bool, bool]] = []
    original = grad_replay.writes_through_view

    def spy(slot: tuple[str, object], receiver: torch.Tensor, enclosing: frozenset) -> bool:
        """Record whether the receiver is enclosing-owned and whether it was written in place."""

        decision = original(slot, receiver, enclosing)
        key = (receiver.device, receiver.untyped_storage().data_ptr())
        seen.append((bool(enclosing), decision and key in enclosing))
        return decision

    monkeypatch.setattr(grad_replay, "writes_through_view", spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert tl.validate(_NestedGradRegion(), _input(), scope="forward", validate_metadata=False)
    assert any(nested for nested, _ in seen)
    assert not any(into_enclosing for _, into_enclosing in seen)
