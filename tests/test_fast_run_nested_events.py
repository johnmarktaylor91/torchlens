"""Guarded fast live run: nested wrapped calls are counted the way capture counts them.

Capture records the innermost wrapped call that produced a tensor, so
``F.relu(x, inplace=True)`` is one ``relu_`` op and the outer ``F.relu`` call
adds nothing. The fast live run must drop that outer call too; read as a
``relu`` event, a later plain ``relu`` plan claimed it and the run refused with
a false arm divergence.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import torchlens as tl
from torchlens._fast_live_events import _NestedEventTracker, returns_nested_output
from torchlens.errors import PathDivergenceError

Activation = Callable[..., torch.Tensor]

_ACTIVATIONS: dict[str, Activation] = {
    "relu": F.relu,
    "leaky_relu": F.leaky_relu,
    "hardtanh": F.hardtanh,
    "relu6": F.relu6,
    "elu": F.elu,
}


class InplaceMLP(nn.Module):
    """Linear, layer norm, then an in-place activation as the module's output."""

    def __init__(self, act: Activation, inplace: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)
        self.norm = nn.LayerNorm(4)
        self.act = act
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the activation in place on the normalized features."""

        return self.act(self.norm(self.linear(x)), inplace=self.inplace)


class MaskedBlock(nn.Module):
    """The in-place module, a masked fill, then the same activation called plainly."""

    def __init__(self, act: Activation) -> None:
        super().__init__()
        self.mlp = InplaceMLP(act)
        self.act = act

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mask the module output, then apply the plain activation."""

        masked = self.mlp(x).masked_fill(mask[..., None] > 0, 0.0)
        return self.act(masked[..., :2] + 1)


def _inputs(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return seeded features and an integer mask."""

    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(2, 3, 4, generator=generator)
    mask = torch.randint(0, 2, (2, 3), generator=generator)
    return x, mask


def _build(act: Activation) -> MaskedBlock:
    """Build a seeded, eval-mode block."""

    torch.manual_seed(0)
    return MaskedBlock(act).eval()


def _saved_payloads(trace: Any) -> dict[str, torch.Tensor]:
    """Return every saved tensor payload by op label."""

    return {
        op.label: op.out.detach()
        for op in trace._layer_list
        if op.has_saved_activation and isinstance(op.out, torch.Tensor)
    }


@pytest.mark.parametrize("act_name", sorted(_ACTIVATIONS))
def test_fast_run_drops_the_outer_call_of_an_inplace_activation(act_name: str) -> None:
    """An in-place activation followed by a plain one reruns fast and equals a full rerun."""

    act = _ACTIVATIONS[act_name]
    model = _build(act)
    trace = tl.trace(model, _inputs(0))
    if act_name == "relu":
        assert {"relu_", "relu"} <= {str(op.func_name) for op in trace._layer_list}
    fresh = _inputs(1)
    fast = trace.run(inputs=fresh, fast=True)
    full = tl.trace(model, _inputs(0)).run(inputs=fresh)
    assert torch.equal(full.output, fast.output)
    want, have = _saved_payloads(full.trace), _saved_payloads(fast.trace)
    assert want and want.keys() == have.keys()
    for label, value in want.items():
        assert torch.equal(value, have[label]), label


@pytest.mark.parametrize("captured_inplace", [True, False])
def test_fast_run_still_refuses_when_the_inplace_spelling_changes(
    captured_inplace: bool,
) -> None:
    """Switching between the in-place and plain spelling after capture is a typed divergence."""

    model = _build(F.relu)
    model.mlp.inplace = captured_inplace
    trace = tl.trace(model, _inputs(0))
    model.mlp.inplace = not captured_inplace
    with pytest.raises(PathDivergenceError):
        trace.run(inputs=_inputs(1), fast=True)


class _Host(_NestedEventTracker):
    """Record which events the tracker hands to collection."""

    def __init__(self) -> None:
        self.active = True
        self.collection_paused = False
        self.owner_thread_id = threading.get_ident()
        self.seen: list[str] = []
        self.reset_event_nesting()

    def capture_function(
        self,
        func_name: str,
        output: Any,
        args: tuple[Any, ...] = (),
        property_accessor: str | None = None,
    ) -> None:
        self.seen.append(func_name)


def _event(host: _Host, name: str, output: Any, *, transparent: bool = False) -> None:
    """Run one bottom-level event."""

    token = host.enter_event(transparent)
    assert token is not None
    host.exit_event(token, name, output, (), None)


def test_tracker_drops_only_an_outer_call_returning_its_nested_output() -> None:
    """Nested same-object returns are dropped; sequential ones and new tensors are not."""

    x, y = torch.ones(2), torch.zeros(2)
    host = _Host()
    _event(host, "relu_", x)
    _event(host, "mul_", x)
    assert host.seen == ["relu_", "mul_"]

    host = _Host()
    outer = host.enter_event(False)
    assert outer is not None
    _event(host, "relu_", x)
    host.exit_event(outer, "relu", x, (x,), None)
    assert host.seen == ["relu_"]

    host = _Host()
    outer = host.enter_event(False)
    assert outer is not None
    _event(host, "relu_", x)
    host.exit_event(outer, "relu", y, (x,), None)
    assert host.seen == ["relu_", "relu"]


def test_tracker_keeps_the_enclosing_call_bottom_level_across_a_transparent_call() -> None:
    """A transparent nested call (``as_subclass``) restores the enclosing marker, as capture does."""

    x = torch.ones(2)
    host = _Host()
    outer = host.enter_event(False)
    assert outer is not None
    _event(host, "as_subclass", x, transparent=True)
    host.exit_event(outer, "tanh", x, (x,), None)
    assert host.seen == ["as_subclass", "tanh"]


def test_tracker_ignores_paused_and_foreign_thread_calls() -> None:
    """Payload-save calls and other threads never open an event."""

    host = _Host()
    host.collection_paused = True
    assert host.enter_event(False) is None
    host.collection_paused = False
    host.owner_thread_id = -1
    assert host.enter_event(False) is None


def test_returns_nested_output_needs_tensor_identity() -> None:
    """Tuple leaves count; equal values, empty outputs and missing nested outputs do not."""

    values, indices = torch.ones(2), torch.zeros(2, dtype=torch.long)
    assert returns_nested_output((values, indices), values)
    assert returns_nested_output((values, indices), (values, indices))
    assert not returns_nested_output(values, values.clone())
    assert not returns_nested_output(values, ())
    assert not returns_nested_output(None, values)
