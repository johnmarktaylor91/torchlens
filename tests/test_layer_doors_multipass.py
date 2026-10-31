"""``Layer.do``, ``Layer.set`` and ``Layer.attach_hooks`` address every pass of a multi-pass layer.

A bare layer label addresses only a single-pass layer, so the three Layer doors
must not forward it for a layer that ran several passes: a module reused in a
loop would otherwise refuse with ``SiteAmbiguityError`` on every door. A Layer
spans all its passes, so each door discloses the two-site fan-out. Each edit
here is checked against the same edit written in plain torch.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.errors import MultiMatchWarning


class _SharedThenFresh(nn.Module):
    """One Linear reused for two passes, then a fresh single-pass Linear."""

    def __init__(self) -> None:
        """Initialize the shared and the fresh projection."""

        super().__init__()
        self.shared = nn.Linear(3, 3)
        self.fresh = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the shared projection twice, then the fresh one."""

        for _call in range(2):
            x = torch.relu(self.shared(x))
        return self.fresh(x)


def _oracle(
    model: _SharedThenFresh, x: torch.Tensor, *, shared_edit=None, fresh_edit=None
) -> torch.Tensor:
    """Run the model in plain torch with an edit on each shared pass and the fresh layer."""

    with torch.no_grad():
        for _call in range(2):
            y = model.shared(x)
            x = torch.relu(y if shared_edit is None else shared_edit(y))
        y = model.fresh(x)
        return y if fresh_edit is None else fresh_edit(y)


@pytest.fixture
def captured() -> Iterator[tuple[_SharedThenFresh, torch.Tensor, tl.Trace]]:
    """Capture the model ready for interventions."""

    torch.manual_seed(0)
    model = _SharedThenFresh().eval()
    x = torch.randn(2, 3)
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    try:
        yield model, x, trace
    finally:
        trace.cleanup()


def _linear_labels(trace: tl.Trace) -> tuple[str, str]:
    """Return the multi-pass shared and the single-pass fresh linear layer labels."""

    linears = [layer for layer in trace.layer_logs.values() if layer.layer_type == "linear"]
    shared = [layer.layer_label for layer in linears if layer.num_passes == 2]
    fresh = [layer.layer_label for layer in linears if layer.num_passes == 1]
    assert len(shared) == 1 and len(fresh) == 1
    return shared[0], fresh[0]


def _double(t: torch.Tensor) -> torch.Tensor:
    """Plain-torch twin of ``tl.scale(2.0)``."""

    return t * 2.0


def _zero(t: torch.Tensor) -> torch.Tensor:
    """Plain-torch twin of setting the site to zeros."""

    return torch.zeros_like(t)


@pytest.mark.parametrize("door", ("do", "set", "attach_hooks"))
def test_layer_door_on_multi_pass_layer_edits_every_pass(captured, door: str) -> None:
    """Each Layer door on a two-pass layer edits both passes, matching plain torch."""

    model, x, trace = captured
    shared, _fresh = _linear_labels(trace)
    fork = trace.fork()
    layer = fork[shared]
    with pytest.warns(MultiMatchWarning, match="matched 2 sites"):
        if door == "do":
            layer.do(tl.scale(2.0))
        elif door == "set":
            layer.set(torch.zeros(2, 3), confirm_mutation=True)
        else:
            layer.attach_hooks(tl.scale(2.0), confirm_mutation=True)
        # The recipe doors apply on push, which re-resolves the same two sites.
        if door != "do":
            fork.push()
    want = _oracle(model, x, shared_edit=_zero if door == "set" else _double)
    assert torch.allclose(fork[fork.output_layers[0]].out, want, atol=1e-6)
    for pass_index in (1, 2):
        assert fork[f"{shared}:{pass_index}"].interventions


def test_layer_door_on_single_pass_layer_edits_only_that_layer(captured) -> None:
    """A single-pass Layer door keeps addressing just its own layer."""

    model, x, trace = captured
    shared, fresh = _linear_labels(trace)
    fork = trace.fork()
    fork[fresh].do(tl.scale(2.0))
    assert torch.allclose(
        fork[fork.output_layers[0]].out, _oracle(model, x, fresh_edit=_double), atol=1e-6
    )
    assert not any(fork[f"{shared}:{pass_index}"].interventions for pass_index in (1, 2))
