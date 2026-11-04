"""``transformed_output`` never describes a forward the trace no longer holds.

``output_transform`` runs on the model's return value. A rerun re-runs the
model and re-applies it; a replay recomputes ops from captured values and has
no model return to apply it to. After a replay rewrites an output op's values,
the inherited transform result describes the parent's forward, so reading it
refuses typed, the same way the fork's ``trace.output`` does. A replay that
leaves every output op's values unchanged keeps the inherited result.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import PayloadUnavailableError


class _TinyClassifier(nn.Module):
    """Two linear layers; zero-ablating ``hidden`` leaves only the head bias."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(2, 3, bias=False)
        self.head = nn.Linear(3, 3)
        with torch.no_grad():
            self.hidden.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 1.0], [0.0, 0.0]]))
            self.head.weight.copy_(torch.eye(3) * 2.0)
            self.head.bias.copy_(torch.tensor([0.0, 0.5, 1.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.head(torch.relu(self.hidden(x)))


def _times_ten(output: torch.Tensor) -> torch.Tensor:
    return output * 10.0


def _capture() -> tuple[nn.Module, torch.Tensor, Any]:
    model = _TinyClassifier().eval()
    x = torch.ones(1, 2)
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(output_transform=_times_ten))
    return model, x, trace


def _fork_do(trace: Any, model: nn.Module, x: torch.Tensor, edit: Any, engine: str) -> Any:
    options = tl.options.InterventionOptions(engine=engine)
    return trace.fork().do(tl.module("hidden"), edit, model=model, inputs=x, intervention=options)


def test_replay_that_changes_an_output_op_refuses_transformed_output() -> None:
    """The parent's transform result is not served as the edited fork's."""

    model, x, trace = _capture()
    out = trace.output_layers[0]
    edited = _fork_do(trace, model, x, tl.zero_ablate(), "replay")

    assert not torch.allclose(edited[out].out, trace[out].out)
    with pytest.raises(PayloadUnavailableError) as caught:
        _ = edited.transformed_output
    assert caught.value.code == "output_not_stored"
    # The parent keeps its own, still-true transform result.
    assert torch.allclose(trace.transformed_output, trace[out].out * 10.0)


def test_replay_that_changes_no_output_op_keeps_transformed_output() -> None:
    """An edit whose replay leaves the outputs unchanged keeps the inherited result."""

    model, x, trace = _capture()
    out = trace.output_layers[0]
    edited = _fork_do(trace, model, x, tl.scale(1.0), "replay")

    assert torch.equal(edited[out].out, trace[out].out)
    assert torch.equal(edited.transformed_output, trace.transformed_output)


def test_rerun_recomputes_transformed_output_from_the_edited_forward() -> None:
    """The rerun engine re-applies ``output_transform`` to the edited return value."""

    model, x, trace = _capture()
    out = trace.output_layers[0]
    edited = _fork_do(trace, model, x, tl.zero_ablate(), "rerun")

    assert not torch.allclose(edited[out].out, trace[out].out)
    assert torch.allclose(edited.transformed_output, edited[out].out * 10.0)
    assert torch.allclose(trace.transformed_output, trace[out].out * 10.0)
