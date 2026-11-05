"""The ``autograd.grad`` boundary replay rebuilds its subgraph in the recorded order.

Autograd's engine runs ready backward nodes newest first, and the gradient
contributions reaching one node are summed in the order its consumers run. A replay
that built the subgraph in another order summed the same terms in another order, so a
gradient made of cancelling terms (forces on a symmetric lattice) missed the saved one
by a rounding step that is large relative to the near-zero result, and validation
failed. Each replayed parent is also written in the memory layout the captured call
saw, as forward replay does for ordinary ops. Validation compares exactly as before;
the replay order and layout are what changed.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens.validation.core as validation_core
from torchlens.backends.torch._autograd_grad_marker import is_autograd_grad_recorder
from torchlens.validation._autograd_grad_replay import _recorded_order, _select_output


class _CancellingBranches(nn.Module):
    """Energy of three branches; the first two contribute exactly opposite gradients."""

    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(5)
        self.w1 = nn.Parameter(torch.randn(6, 6, generator=generator))
        self.w3 = nn.Parameter(torch.randn(6, 6, generator=generator) * 1e-3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the energy and its gradient with respect to ``x``."""

        x = x.detach().clone().requires_grad_(True)
        energy = (torch.tanh(x @ self.w1) * 1000).sum()
        energy = energy + (torch.tanh(x @ self.w1) * -1000).sum()
        energy = energy + torch.sin(x @ self.w3).sum()
        (grad,) = torch.autograd.grad(energy, x)
        return energy, grad


class _ThreeBranches(nn.Module):
    """Energy of three independent branches of one input."""

    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.w1 = nn.Parameter(torch.randn(6, 6, generator=generator))
        self.w2 = nn.Parameter(torch.randn(6, 6, generator=generator))
        self.w3 = nn.Parameter(torch.randn(6, 6, generator=generator) * 1e-3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the energy and its gradient with respect to ``x``."""

        x = x.detach().clone().requires_grad_(True)
        energy = (torch.tanh(x @ self.w1) * 1000).sum()
        energy = energy + (torch.tanh(x @ self.w2) * -999.9).sum()
        energy = energy + torch.sin(x @ self.w3).sum()
        (grad,) = torch.autograd.grad(energy, x)
        return energy, grad


class _ColumnInput(nn.Module):
    """The differentiated input is a strided column that a ``Linear(1, k)`` reads."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.emb = nn.Linear(1, 192)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return an energy of nearly cancelling terms and its gradient on the column."""

        column = (x.detach().clone().requires_grad_(True) * 2.0)[:, :, 5:6]
        energy = (self.emb(column) ** 2).sum() - (self.emb(column.contiguous()) ** 2).sum()
        energy = energy + column.sum() * 1e-3
        (grad,) = torch.autograd.grad(energy, column)
        return energy, grad


def _validate(model: nn.Module, x: torch.Tensor) -> bool:
    """Validate forward, silencing the capture's own warnings."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bool(tl.validate(model, x, scope="forward"))


def test_cancelling_gradient_terms_validate() -> None:
    x = torch.randn(4, 6, generator=torch.Generator().manual_seed(6))
    assert _validate(_CancellingBranches(), x)


def test_strided_column_input_replays_in_its_captured_layout() -> None:
    # The replay leaf for the column is a compact copy; fed to Linear as is, it
    # takes the fused addmm path instead of the strided matmul the call ran.
    x = torch.randn(2, 65, 6, generator=torch.Generator().manual_seed(3))
    assert _validate(_ColumnInput(), x)


def test_plain_replay_reproduces_saved_gradient_bitwise(monkeypatch: pytest.MonkeyPatch) -> None:
    replayed: list[tuple[torch.Tensor, torch.Tensor]] = []
    original = validation_core.execute_replay_func

    def recording(layer: Any, input_args: dict[str, Any], layers_to_perturb: list[str]) -> Any:
        output = original(layer, input_args, layers_to_perturb)
        if is_autograd_grad_recorder(layer.func) and not layers_to_perturb:
            replayed.append((_select_output(layer, output).detach(), layer.out.detach()))
        return output

    monkeypatch.setattr(validation_core, "execute_replay_func", recording)
    x = torch.randn(4, 6, generator=torch.Generator().manual_seed(3))
    assert _validate(_ThreeBranches(), x)
    assert replayed
    for got, saved in replayed:
        assert torch.equal(got, saved)


def _fake_op(label: str, parents: tuple[str, ...], raw_index: Any) -> SimpleNamespace:
    """Stand-in op carrying only what the ordering reads."""

    return SimpleNamespace(label=label, parents=parents, raw_index=raw_index)


def test_recorded_order_follows_capture_index() -> None:
    ops = {
        "x": _fake_op("x", (), 1),
        "c": _fake_op("c", ("x",), 4),
        "a": _fake_op("a", ("x",), 2),
        "b": _fake_op("b", ("x",), 3),
        "sum": _fake_op("sum", ("c", "a", "b"), 5),
    }
    ordered = _recorded_order(ops, ops.__getitem__)  # type: ignore[arg-type]
    assert [op.label for op in ordered] == ["x", "a", "b", "c", "sum"]


def test_recorded_order_keeps_parents_first_without_an_index() -> None:
    ops = {
        "x": _fake_op("x", (), 7),
        "y": _fake_op("y", ("x",), None),
        "z": _fake_op("z", ("y",), 2),
    }
    ordered = _recorded_order(ops, ops.__getitem__)  # type: ignore[arg-type]
    assert [op.label for op in ordered] == ["x", "y", "z"]
