"""Recording.fork and Recording.do delegate to the cooked Trace (RC-293)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._utils.hashing import compute_graph_shape_hash
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


def test_cooked_trace_carries_the_recorder_identity_facts(
    recorded: tuple[tl.Recording, nn.Module, torch.Tensor],
) -> None:
    """The cooked Trace names the module the Recorder ran, so ``do`` can gate on it."""

    recording, model, _x = recorded
    cooked = recording.to_trace()
    qualname = f"{type(model).__module__}.{type(model).__qualname__}"
    assert cooked._source_class_qualname == qualname
    assert cooked._provenance_root_entry_point == f"module_call:{qualname}.forward"
    assert cooked._provenance_model_object_id == id(model)
    # Ops outside a container output carry no slot, as on the exhaustive path.
    assert all(op.multi_output_index is None for op in cooked.ops if not op.in_multi_output)


def test_cooked_trace_graph_shape_matches_exhaustive_capture(
    recorded: tuple[tl.Recording, nn.Module, torch.Tensor],
) -> None:
    """The rerun divergence detector sees the same graph for a cooked and a traced model."""

    recording, model, x = recorded
    cooked = recording.to_trace()
    exhaustive = tl.trace(model, x)
    # Root-level ops have no containing module on either path, and the input
    # source carries the exhaustive func_name spelling.
    cooked_relu = next(op for op in cooked.ops if op.type == "relu")
    exhaustive_relu = next(op for op in exhaustive.ops if op.type == "relu")
    assert cooked_relu.module is None and exhaustive_relu.module is None
    assert list(cooked_relu.modules) == list(exhaustive_relu.modules) == []
    assert cooked["input_1"].func_name == exhaustive["input_1"].func_name == "none"
    # The raw-event hash is session-only; a cooked journal is shaped differently,
    # so the cooked Trace carries none and rerun falls back to the graph hash.
    assert cooked._raw_event_shape_hash is None
    assert compute_graph_shape_hash(cooked, rerun_signature=True) == compute_graph_shape_hash(
        exhaustive, rerun_signature=True
    )
