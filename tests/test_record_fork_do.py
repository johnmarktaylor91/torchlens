"""Recording.fork and Recording.do delegate to the cooked Trace (RC-293)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.recording.exceptions import RecordingConfigError


class _ReluNet(nn.Module):
    """Two linear layers around a relu, the site every test edits."""

    def __init__(self) -> None:
        """Build the layers."""

        super().__init__()
        self.first = nn.Linear(3, 4)
        self.second = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run first, relu, second."""

        return self.second(torch.relu(self.first(x)))


@pytest.fixture
def recorded() -> tuple[tl.Recording, nn.Module, torch.Tensor]:
    """A sparse recording with its model and input."""

    torch.manual_seed(0)
    model = _ReluNet()
    x = torch.randn(2, 3)
    return tl.record(model, x, save=tl.func("relu")), model, x


def test_recording_fork_returns_a_fresh_trace(
    recorded: tuple[tl.Recording, nn.Module, torch.Tensor],
) -> None:
    """Recording.fork is Trace.fork of the cooked trace: a new Trace, not the recording."""

    recording, _model, _x = recorded
    fork = recording.fork()
    assert isinstance(fork, tl.Trace)
    cooked = recording.to_trace()
    assert fork is not cooked
    assert len(fork.ops) == len(cooked.ops)
    named = recording.fork("edited")
    assert isinstance(named, tl.Trace)


def test_recording_do_edits_and_replays(
    recorded: tuple[tl.Recording, nn.Module, torch.Tensor],
) -> None:
    """Recording.do runs Trace.do on the cooked trace and returns the edited Trace."""

    recording, model, x = recorded
    relu_label = next(label for label in recording.to_trace().layer_labels if "relu" in label)
    edited = recording.do(relu_label, tl.zero_ablate(), model=model, inputs=x)
    assert isinstance(edited, tl.Trace)
    assert len(edited.ops) == len(recording.to_trace().ops)


def test_record_episode_keyword_refuses_typed() -> None:
    """tl.record(..., episode=) refuses typed and teaches tl.trace (RC-270)."""

    torch.manual_seed(0)
    model = _ReluNet()
    x = torch.randn(2, 3)
    with pytest.raises(RecordingConfigError) as info:
        tl.record(model, x, save=tl.func("relu"), episode=object())
    assert info.value.fields["code"] == "episode_unsupported_on_record"
    assert "tl.trace(model, x, episode=" in str(info.value)
    with pytest.raises(RecordingConfigError) as info:
        tl.recording.Recorder(model, save=tl.func("relu"), episode=object())
    assert info.value.fields["code"] == "episode_unsupported_on_record"
